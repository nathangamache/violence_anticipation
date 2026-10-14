"""
Real-Time Violence Anticipation Pipeline
RTSP ingestion + Qwen2.5-VL embeddings (GPU0) + LSTM inference (GPU1)

Usage:
    python rtsp_inference.py --rtsp rtsp://username:password@192.168.1.x:554/stream \
                             --vlm_checkpoint /path/to/qwen2.5-vl \
                             --lstm_checkpoint /path/to/lstm_checkpoint.pth \
                             --fps 1.0
"""

import argparse
import time
import queue
import threading
import logging
import signal
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
log = logging.getLogger("ViolenceAnticipation")


# ─────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────
@dataclass
class Frame:
    """A captured frame with metadata."""
    image: np.ndarray          # BGR from OpenCV
    timestamp: float           # time.time()
    frame_idx: int


@dataclass
class EmbeddedFrame:
    """A frame with its VLM embedding attached."""
    embedding: torch.Tensor    # shape: (3584,) on CPU
    timestamp: float
    frame_idx: int


@dataclass
class Prediction:
    """Model output for a sequence."""
    score: float               # violence probability 0–1
    is_alert: bool             # threshold exceeded
    timestamp: float
    frame_idx: int
    lead_time_budget: float    # seconds until assumed event if sustained alert


# ─────────────────────────────────────────────
# Model definition — matches checkpoint exactly
#
# Confirmed architecture from inspect_checkpoint.py:
#   frame_encoder:  Linear(3584→256) + LN + ReLU + Dropout
#                   Linear(256→256)  + LN + ReLU + Dropout   (2-block encoder)
#   lstm:           Bidirectional LSTM, hidden=128, 2 layers  (output dim = 256)
#   attention:      Linear(256→64) + Tanh + Linear(64→1) → softmax weights
#   classifier:     Linear(256→64) + ReLU + Dropout + Linear(64→1) + Sigmoid
# ─────────────────────────────────────────────
class ViolenceLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 3584,
        encoder_dim: int = 256,
        lstm_hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        sequence_length: int = 32,
    ):
        super().__init__()
        self.sequence_length = sequence_length

        # 2-block frame encoder  (matches frame_encoder.0/1/4/5 in checkpoint)
        self.frame_encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_dim),   # .0
            nn.LayerNorm(encoder_dim),            # .1
            nn.ReLU(),                            # .2
            nn.Dropout(dropout),                  # .3
            nn.Linear(encoder_dim, encoder_dim),  # .4
            nn.LayerNorm(encoder_dim),            # .5
            nn.ReLU(),                            # .6
            nn.Dropout(dropout),                  # .7
        )

        # Bidirectional LSTM  (output dim = lstm_hidden * 2 = 256)
        self.lstm = nn.LSTM(
            input_size=encoder_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * 2  # 256

        # Attention over time steps  (matches attention.0/2 in checkpoint)
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),  # .0
            nn.Tanh(),                    # .1
            nn.Linear(64, 1),            # .2
        )

        # Classifier  (matches classifier.0/3 in checkpoint)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),  # .0
            nn.ReLU(),                    # .1
            nn.Dropout(dropout),          # .2
            nn.Linear(64, 1),            # .3
            nn.Sigmoid(),                 # .4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        returns: (batch, 1) violence probability
        """
        B, T, D = x.shape

        # Encode each frame independently
        x_flat = x.reshape(B * T, D)
        encoded = self.frame_encoder(x_flat)       # (B*T, encoder_dim)
        encoded = encoded.reshape(B, T, -1)        # (B, T, encoder_dim)

        # Bidirectional LSTM over sequence
        lstm_out, _ = self.lstm(encoded)           # (B, T, lstm_hidden*2)

        # Attention pooling
        attn_scores = self.attention(lstm_out)     # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T, 1)
        context = (attn_weights * lstm_out).sum(dim=1)    # (B, lstm_hidden*2)

        return self.classifier(context)            # (B, 1)


# ─────────────────────────────────────────────
# Thread 1: RTSP Frame Capture
# ─────────────────────────────────────────────
class RTSPCapture(threading.Thread):
    """
    Continuously reads from an RTSP stream, dropping frames to maintain
    the target sample rate. Puts Frame objects onto a queue.
    """

    def __init__(
        self,
        rtsp_url: str,
        frame_queue: queue.Queue,
        target_fps: float = 1.0,
        reconnect_delay: float = 5.0,
        max_queue_size: int = 30,
    ):
        super().__init__(daemon=True, name="RTSPCapture")
        self.rtsp_url = rtsp_url
        self.frame_queue = frame_queue
        self.target_fps = target_fps
        self.sample_interval = 1.0 / target_fps
        self.reconnect_delay = reconnect_delay
        self.max_queue_size = max_queue_size
        self._stop_event = threading.Event()
        self.frame_idx = 0
        self.frames_captured = 0
        self.frames_dropped = 0

    def stop(self):
        self._stop_event.set()

    def _open_stream(self) -> Optional[cv2.VideoCapture]:
        log.info(f"Connecting to RTSP stream: {self.rtsp_url}")
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        # Reduce internal buffer to minimise latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            log.info(f"Stream opened. Native FPS: {native_fps:.1f}")
            return cap
        log.warning("Failed to open stream.")
        return None

    def run(self):
        cap = None
        last_sample_time = 0.0

        while not self._stop_event.is_set():
            if cap is None or not cap.isOpened():
                cap = self._open_stream()
                if cap is None:
                    log.warning(f"Retrying in {self.reconnect_delay}s...")
                    time.sleep(self.reconnect_delay)
                    continue

            ret, frame_bgr = cap.read()
            if not ret:
                log.warning("Lost stream. Attempting reconnect...")
                cap.release()
                cap = None
                time.sleep(self.reconnect_delay)
                continue

            self.frame_idx += 1
            now = time.time()

            # Only pass frames that meet sample interval
            if (now - last_sample_time) < self.sample_interval:
                self.frames_dropped += 1
                continue

            last_sample_time = now

            # Drop oldest if queue is full (prefer freshness)
            if self.frame_queue.qsize() >= self.max_queue_size:
                try:
                    self.frame_queue.get_nowait()
                    self.frames_dropped += 1
                except queue.Empty:
                    pass

            self.frame_queue.put(Frame(
                image=frame_bgr.copy(),
                timestamp=now,
                frame_idx=self.frame_idx,
            ))
            self.frames_captured += 1

            if self.frames_captured % 60 == 0:
                log.info(
                    f"[Capture] {self.frames_captured} sampled, "
                    f"{self.frames_dropped} dropped, "
                    f"queue={self.frame_queue.qsize()}"
                )

        if cap:
            cap.release()
        log.info("[Capture] Thread stopped.")


# ─────────────────────────────────────────────
# Thread 2: VLM Embedding Extraction (GPU 0)
# ─────────────────────────────────────────────
class EmbeddingExtractor(threading.Thread):
    """
    Loads Qwen2.5-VL on GPU 0, consumes Frame objects, produces EmbeddedFrame objects.
    """

    def __init__(
        self,
        vlm_checkpoint: str,
        frame_queue: queue.Queue,
        embed_queue: queue.Queue,
        device: str = "cuda:0",
        target_size: tuple = (448, 448),
    ):
        super().__init__(daemon=True, name="EmbeddingExtractor")
        self.vlm_checkpoint = vlm_checkpoint
        self.frame_queue = frame_queue
        self.embed_queue = embed_queue
        self.device = device
        self.target_size = target_size
        self._stop_event = threading.Event()
        self.frames_processed = 0
        self.model = None
        self.processor = None

    def stop(self):
        self._stop_event.set()

    def _load_model(self):
        log.info(f"Loading Qwen2.5-VL on {self.device}...")
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(
                self.vlm_checkpoint,
                trust_remote_code=True,
                use_fast=True,
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.vlm_checkpoint,
                dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
            )
            self.model.eval()
            log.info("Qwen2.5-VL loaded successfully.")
        except ImportError:
            raise RuntimeError(
                "transformers library not found or Qwen2.5-VL not available. "
                "Install with: pip install transformers>=4.37.0"
            )

    @torch.no_grad()
    def _extract_embedding(self, frame: Frame) -> torch.Tensor:
        """
        Extract VLM visual embedding for a single frame.
        Returns a (3584,) tensor on CPU.

        Uses the correct Qwen2.5-VL two-step processor API:
          1. apply_chat_template() → prompt string (no image, no tensors)
          2. processor(text=..., images=...) → proper input dict
        """
        # Convert BGR → RGB PIL
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize(self.target_size, Image.LANCZOS)

        # Step 1: build prompt string — image placeholder, NO PIL object here
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe the scene."},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Step 2: process text + image → proper input dict
        inputs = self.processor(
            text=prompt,
            images=pil_img,
            return_tensors="pt",
        )

        # Step 3: move all tensors to GPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Step 4: forward pass with hidden states (no generation needed)
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
        )

        # Mean-pool last hidden state over sequence → (hidden_dim,)
        last_hidden = outputs.hidden_states[-1]   # (1, seq_len, hidden_dim)
        embedding = last_hidden[0].mean(dim=0)    # (hidden_dim,)

        if embedding.shape[0] != 3584:
            log.warning(
                f"Embedding dim mismatch: got {embedding.shape[0]}, expected 3584. "
                "Check your model variant."
            )

        return embedding.cpu().float()

    def run(self):
        self._load_model()

        while not self._stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            t0 = time.time()
            try:
                embedding = self._extract_embedding(frame)
                elapsed = time.time() - t0
                self.frames_processed += 1

                self.embed_queue.put(EmbeddedFrame(
                    embedding=embedding,
                    timestamp=frame.timestamp,
                    frame_idx=frame.frame_idx,
                ))

                if self.frames_processed % 10 == 0:
                    log.info(
                        f"[Embedder] Processed {self.frames_processed} frames, "
                        f"last took {elapsed:.2f}s, "
                        f"embed_queue={self.embed_queue.qsize()}"
                    )
            except Exception as e:
                log.error(f"[Embedder] Error on frame {frame.frame_idx}: {e}", exc_info=True)

        log.info("[Embedder] Thread stopped.")


# ─────────────────────────────────────────────
# Thread 3: LSTM Inference + Alert (GPU 1)
# ─────────────────────────────────────────────
class LSTMInference(threading.Thread):
    """
    Loads the LSTM student model on GPU 1.
    Maintains a rolling sequence buffer and runs inference when full.
    Emits Prediction objects and triggers alert callbacks.
    """

    def __init__(
        self,
        lstm_checkpoint: str,
        embed_queue: queue.Queue,
        alert_callback,
        device: str = "cuda:1",
        sequence_length: int = 32,
        violence_threshold: float = 0.5,
        alert_cooldown: float = 30.0,
        model_kwargs: dict = None,
    ):
        super().__init__(daemon=True, name="LSTMInference")
        self.lstm_checkpoint = lstm_checkpoint
        self.embed_queue = embed_queue
        self.alert_callback = alert_callback
        self.device = device
        self.sequence_length = sequence_length
        self.violence_threshold = violence_threshold
        self.alert_cooldown = alert_cooldown
        self.model_kwargs = model_kwargs or {}
        self._stop_event = threading.Event()
        self.model = None
        self.buffer: deque = deque(maxlen=sequence_length)
        self.last_alert_time = 0.0
        self.inferences_run = 0
        self.alerts_fired = 0

    def stop(self):
        self._stop_event.set()

    def _load_model(self):
        log.info(f"Loading LSTM model on {self.device}...")
        checkpoint = torch.load(self.lstm_checkpoint, map_location="cpu", weights_only=False)

        # Detect checkpoint format
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            saved_kwargs = checkpoint.get("model_config", {})
            self.model_kwargs = {**saved_kwargs, **self.model_kwargs}
        elif isinstance(checkpoint, dict) and all(
            isinstance(v, torch.Tensor) for v in checkpoint.values()
        ):
            state_dict = checkpoint
        else:
            state_dict = checkpoint

        model = ViolenceLSTM(
            sequence_length=self.sequence_length,
            **self.model_kwargs,
        )
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        self.model = model
        log.info("LSTM model loaded successfully.")

    @torch.no_grad()
    def _run_inference(self) -> float:
        """Run LSTM on current buffer, return violence probability."""
        # Stack buffer → (1, seq_len, 3584)
        seq = torch.stack(list(self.buffer), dim=0).unsqueeze(0).to(self.device)
        prob = self.model(seq)           # (1, 1)
        return prob.item()

    def run(self):
        self._load_model()

        while not self._stop_event.is_set():
            try:
                ef = self.embed_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            self.buffer.append(ef.embedding)

            # Only infer when buffer is full (we have a complete sequence)
            if len(self.buffer) < self.sequence_length:
                log.debug(
                    f"[LSTM] Buffer filling: {len(self.buffer)}/{self.sequence_length}"
                )
                continue

            t0 = time.time()
            score = self._run_inference()
            inference_ms = (time.time() - t0) * 1000
            self.inferences_run += 1

            is_alert = score >= self.violence_threshold
            now = ef.timestamp

            pred = Prediction(
                score=score,
                is_alert=is_alert,
                timestamp=now,
                frame_idx=ef.frame_idx,
                lead_time_budget=self.alert_cooldown,
            )

            log.info(
                f"[LSTM] frame={ef.frame_idx:06d} | "
                f"score={score:.4f} | "
                f"alert={'YES' if is_alert else 'no '} | "
                f"inference={inference_ms:.1f}ms"
            )

            # Fire alert with cooldown to avoid alert flooding
            if is_alert and (now - self.last_alert_time) > self.alert_cooldown:
                self.last_alert_time = now
                self.alerts_fired += 1
                try:
                    self.alert_callback(pred)
                except Exception as e:
                    log.error(f"[LSTM] Alert callback error: {e}", exc_info=True)

        log.info(
            f"[LSTM] Thread stopped. "
            f"Inferences: {self.inferences_run}, Alerts: {self.alerts_fired}"
        )


# ─────────────────────────────────────────────
# Alert Handler
# ─────────────────────────────────────────────
class AlertHandler:
    """
    Handles violence alerts. Extend this for email, SMS, webhook, etc.
    Currently: console log + optional frame snapshot save.
    """

    def __init__(self, snapshot_dir: Optional[str] = None):
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        if self.snapshot_dir:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.alert_log = []

    def on_alert(self, pred: Prediction, frame: Optional[np.ndarray] = None):
        ts = datetime.fromtimestamp(pred.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        log.warning(
            f"\n{'='*60}\n"
            f"  WARNING: VIOLENCE ANTICIPATED\n"
            f"  Score:      {pred.score:.4f}\n"
            f"  Time:       {ts}\n"
            f"  Frame:      {pred.frame_idx}\n"
            f"{'='*60}"
        )
        self.alert_log.append(
            {"timestamp": ts, "score": pred.score, "frame_idx": pred.frame_idx}
        )

        if frame is not None and self.snapshot_dir:
            fname = self.snapshot_dir / f"alert_{pred.frame_idx}_{int(pred.timestamp)}.jpg"
            cv2.imwrite(str(fname), frame)
            log.info(f"Snapshot saved: {fname}")

    def summary(self):
        log.info(f"Total alerts fired: {len(self.alert_log)}")
        for a in self.alert_log:
            log.info(f"  {a['timestamp']} | score={a['score']:.4f} | frame={a['frame_idx']}")


# ─────────────────────────────────────────────
# Pipeline Orchestrator
# ─────────────────────────────────────────────
class ViolenceAnticipationPipeline:
    """
    Wires together all threads and manages lifecycle.
    """

    def __init__(self, args):
        self.args = args
        self.alert_handler = AlertHandler(snapshot_dir=args.snapshot_dir)

        # Queues connecting the threads
        self.frame_queue = queue.Queue(maxsize=60)
        self.embed_queue = queue.Queue(maxsize=120)

        # Thread 1: RTSP capture
        self.capture = RTSPCapture(
            rtsp_url=args.rtsp,
            frame_queue=self.frame_queue,
            target_fps=args.fps,
        )

        # Thread 2: VLM embeddings on GPU 0
        self.embedder = EmbeddingExtractor(
            vlm_checkpoint=args.vlm_checkpoint,
            frame_queue=self.frame_queue,
            embed_queue=self.embed_queue,
            device="cuda:0",
            target_size=(args.frame_size, args.frame_size),
        )

        # Thread 3: LSTM inference on GPU 1
        self.inferencer = LSTMInference(
            lstm_checkpoint=args.lstm_checkpoint,
            embed_queue=self.embed_queue,
            alert_callback=self.alert_handler.on_alert,
            device="cuda:1",
            sequence_length=args.sequence_length,
            violence_threshold=args.threshold,
            alert_cooldown=args.alert_cooldown,
        )

        self._running = False

    def start(self):
        log.info("Starting Violence Anticipation Pipeline...")
        log.info(f"  RTSP URL:        {self.args.rtsp}")
        log.info(f"  Sample FPS:      {self.args.fps}")
        log.info(f"  Sequence length: {self.args.sequence_length} frames")
        log.info(f"  Alert threshold: {self.args.threshold}")
        log.info(f"  Alert cooldown:  {self.args.alert_cooldown}s")

        self.inferencer.start()   # Load models first (takes time)
        self.embedder.start()
        time.sleep(2)             # Give model loading a head start
        self.capture.start()
        self._running = True
        log.info("Pipeline running. Press Ctrl+C to stop.")

    def stop(self):
        log.info("Shutting down pipeline...")
        self.capture.stop()
        self.embedder.stop()
        self.inferencer.stop()
        self.capture.join(timeout=5)
        self.embedder.join(timeout=10)
        self.inferencer.join(timeout=5)
        self.alert_handler.summary()
        log.info("Pipeline stopped.")

    def run_forever(self):
        self.start()

        # Handle Ctrl+C gracefully
        def _sig_handler(sig, frame):
            log.info("Interrupt received, stopping...")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)

        # Keep main thread alive, print periodic stats
        try:
            while True:
                time.sleep(30)
                log.info(
                    f"[Stats] "
                    f"frame_q={self.frame_queue.qsize()} | "
                    f"embed_q={self.embed_queue.qsize()} | "
                    f"captured={self.capture.frames_captured} | "
                    f"embedded={self.embedder.frames_processed} | "
                    f"inferences={self.inferencer.inferences_run} | "
                    f"alerts={self.inferencer.alerts_fired}"
                )
        except SystemExit:
            pass


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Real-Time Violence Anticipation Pipeline")
    p.add_argument(
        "--rtsp", required=True,
        help="RTSP stream URL, e.g. rtsp://user:pass@192.168.1.x:554/stream1"
    )
    p.add_argument(
        "--vlm_checkpoint", required=True,
        help="Path to Qwen2.5-VL model directory (HuggingFace format)"
    )
    p.add_argument(
        "--lstm_checkpoint", required=True,
        help="Path to trained LSTM .pth checkpoint"
    )
    p.add_argument(
        "--fps", type=float, default=1.0,
        help="Frames per second to sample from the stream (default: 1.0)"
    )
    p.add_argument(
        "--sequence_length", type=int, default=32,
        help="Number of frames per LSTM input sequence (match training, default: 32)"
    )
    p.add_argument(
        "--threshold", type=float, default=0.5,
        help="Violence probability threshold for alert (default: 0.5)"
    )
    p.add_argument(
        "--alert_cooldown", type=float, default=30.0,
        help="Minimum seconds between consecutive alerts (default: 30.0)"
    )
    p.add_argument(
        "--frame_size", type=int, default=448,
        help="Resize frames to this square size before VLM (default: 448)"
    )
    p.add_argument(
        "--snapshot_dir", type=str, default="snapshots",
        help="Directory to save alert frame snapshots (default: snapshots/)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = ViolenceAnticipationPipeline(args)
    pipeline.run_forever()