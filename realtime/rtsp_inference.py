"""Live violence anticipation: RTSP capture, Qwen2.5-VL embeddings (GPU0), BiLSTM inference (GPU1).

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
log = logging.getLogger("ViolenceAnticipation")


@dataclass
class Frame:
    image: np.ndarray          # BGR from OpenCV
    timestamp: float           # time.time()
    frame_idx: int


@dataclass
class EmbeddedFrame:
    embedding: torch.Tensor    # shape: (3584,) on CPU
    timestamp: float
    frame_idx: int


@dataclass
class Prediction:
    score: float               # violence probability 0–1
    is_alert: bool
    timestamp: float
    frame_idx: int
    lead_time_budget: float


class ViolenceLSTM(nn.Module):
    """BiLSTM + attention head; module indices must line up with the temporal-model checkpoint keys."""

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

        self.frame_encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=encoder_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * 2

        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, input_dim) -> (B, 1) violence probability."""
        B, T, D = x.shape

        x_flat = x.reshape(B * T, D)
        encoded = self.frame_encoder(x_flat)       # (B*T, encoder_dim)
        encoded = encoded.reshape(B, T, -1)        # (B, T, encoder_dim)

        lstm_out, _ = self.lstm(encoded)           # (B, T, lstm_hidden*2)

        attn_scores = self.attention(lstm_out)     # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T, 1)
        context = (attn_weights * lstm_out).sum(dim=1)    # (B, lstm_hidden*2)

        return self.classifier(context)            # (B, 1)


class RTSPCapture(threading.Thread):
    """Reads an RTSP stream and queues one Frame per 1/target_fps seconds, reconnecting on loss."""

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

            if (now - last_sample_time) < self.sample_interval:
                self.frames_dropped += 1
                continue

            last_sample_time = now

            # Prefer freshness: drop the oldest frame when the queue is full.
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


class EmbeddingExtractor(threading.Thread):
    """Runs Qwen2.5-VL on GPU 0, turning queued Frames into EmbeddedFrames."""

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
        """Return the mean-pooled last hidden state for one frame as a (3584,) CPU tensor."""
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize(self.target_size, Image.LANCZOS)

        # Image placeholder only: the PIL image goes to processor(), not apply_chat_template().
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

        inputs = self.processor(
            text=prompt,
            images=pil_img,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
        )

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


class LSTMInference(threading.Thread):
    """Runs ViolenceLSTM on GPU 1 over a rolling embedding buffer and fires alert callbacks."""

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


class AlertHandler:
    """Logs alerts and, when given a frame, saves a snapshot."""

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


class ViolenceAnticipationPipeline:
    """Wires the capture, embedding and inference threads together and manages their lifecycle."""

    def __init__(self, args):
        self.args = args
        self.alert_handler = AlertHandler(snapshot_dir=args.snapshot_dir)

        self.frame_queue = queue.Queue(maxsize=60)
        self.embed_queue = queue.Queue(maxsize=120)

        self.capture = RTSPCapture(
            rtsp_url=args.rtsp,
            frame_queue=self.frame_queue,
            target_fps=args.fps,
        )

        self.embedder = EmbeddingExtractor(
            vlm_checkpoint=args.vlm_checkpoint,
            frame_queue=self.frame_queue,
            embed_queue=self.embed_queue,
            device="cuda:0",
            target_size=(args.frame_size, args.frame_size),
        )

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

        self.inferencer.start()   # model loading is slow, start consumers first
        self.embedder.start()
        time.sleep(2)
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

        def _sig_handler(sig, frame):
            log.info("Interrupt received, stopping...")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)

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