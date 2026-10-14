#!/usr/bin/env python3
"""Fast curation analysis: Qwen embeddings -> LSTM scores -> stage timeline written to candidates.json.

Uses infer_embedding_only (~0.5 s/frame) instead of text-generation analysis (~6 s/frame).

Usage:
    python experiments/fast_analyze_curation.py --output_dir data/anticipation_curation \
        --model_path outputs/outputs_temporal/best_temporal_model.pt --vlm_device cuda:0 --lstm_device cuda:1 --fps 0.5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make the vlm package importable regardless of cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


# Must match train_temporal_model.py exactly so its checkpoints load.
class TemporalViolenceModel(nn.Module):
    def __init__(self, input_dim=3584, hidden_dim=256, lstm_hidden=128,
                 num_lstm_layers=2, dropout=0.3, bidirectional=True, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.frame_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=lstm_hidden,
                            num_layers=num_lstm_layers, batch_first=True,
                            dropout=dropout if num_lstm_layers > 1 else 0,
                            bidirectional=bidirectional)
        lstm_output_dim = lstm_hidden * 2 if bidirectional else lstm_hidden
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 64), nn.GELU(),
            nn.Dropout(dropout / 2), nn.Linear(64, 1))

    def forward(self, x, lengths=None):
        B, T, D = x.shape
        encoded = self.frame_encoder(x.view(B * T, D)).view(B, T, -1)
        lstm_out, _ = self.lstm(encoded)
        if self.use_attention:
            attn_weights = F.softmax(self.attention(lstm_out), dim=1)
            context = (lstm_out * attn_weights).sum(dim=1)
        else:
            context = lstm_out[:, -1, :]
        return self.classifier(context).squeeze(-1)


def extract_frames(video_path: str, fps: float = 0.5,
                   max_frames: int = 600) -> List[Tuple[int, float, Image.Image]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps <= 0:
        cap.release()
        return []

    frame_interval = max(1, int(video_fps / fps))

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append((frame_idx, timestamp, pil_image))

            if len(frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return frames


def extract_embeddings_from_video(video_path: str, teacher, fps: float = 0.5,
                                   verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """VLM embeddings [N, 3584] and timestamps [N] in seconds via infer_embedding_only."""
    frames = extract_frames(video_path, fps=fps)
    if not frames:
        return np.array([]), np.array([])

    if verbose:
        print(f"  Extracted {len(frames)} frames, extracting embeddings...")

    embeddings = []
    timestamps = []

    iterator = frames
    if verbose:
        iterator = tqdm(frames, desc="  Embeddings", leave=False)

    for frame_idx, timestamp, image in iterator:
        emb = teacher.infer_embedding_only(image)
        if emb is not None:
            embeddings.append(emb)
            timestamps.append(timestamp)

    if not embeddings:
        return np.array([]), np.array([])

    return np.stack(embeddings), np.array(timestamps)


@torch.no_grad()
def score_with_lstm(model, embeddings, seq_len, device, batch_size=128):
    """Sliding-window LSTM scores; the first seq_len-1 frames reuse the first window's score."""
    n = len(embeddings)
    if n < seq_len:
        padded = np.zeros((seq_len, embeddings.shape[1]), dtype=np.float32)
        padded[-n:] = embeddings
        x = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(device)
        score = torch.sigmoid(model(x)).cpu().item()
        return np.full(n, score)

    n_seq = n - seq_len + 1
    all_scores = []
    for start in range(0, n_seq, batch_size):
        end = min(start + batch_size, n_seq)
        batch = np.stack([embeddings[i:i + seq_len] for i in range(start, end)])
        x = torch.tensor(batch, dtype=torch.float32).to(device)
        scores = torch.sigmoid(model(x)).cpu().numpy()
        all_scores.extend(scores)

    padded_scores = np.concatenate([
        np.full(seq_len - 1, all_scores[0]),
        np.array(all_scores)
    ])
    return padded_scores


def classify_stages(scores: np.ndarray, timestamps: np.ndarray,
                    violence_threshold: float = 0.75,
                    tension_threshold: float = 0.3,
                    smoothing_window: int = 3,
                    ramp_up_frames: int = 30) -> Dict:
    """Heuristic per-frame stages (calm/tension/escalation/violence/aftermath) from smoothed LSTM scores.

    The first ramp_up_frames are labelled "ramp_up" and ignored: they have no full LSTM window yet.
    """
    n = len(scores)
    if n == 0:
        return {"error": "No scores"}

    if smoothing_window > 1 and n >= smoothing_window:
        kernel = np.ones(smoothing_window) / smoothing_window
        smoothed = np.convolve(scores, kernel, mode='same')
    else:
        smoothed = scores.copy()

    if n > 1:
        derivative = np.gradient(smoothed)
    else:
        derivative = np.zeros(n)

    stages = []
    peak_reached = False
    peak_score = 0.0

    for i in range(n):
        if i < ramp_up_frames:
            stages.append("ramp_up")
            continue

        s = smoothed[i]
        d = derivative[i]

        if s >= violence_threshold:
            stage = "violence"
            peak_reached = True
            peak_score = max(peak_score, s)
        elif peak_reached and s < violence_threshold:
            stage = "aftermath"
        elif s >= tension_threshold:
            if d > 0.01:
                stage = "escalation"
            else:
                stage = "tension"
        else:
            stage = "calm"

        stages.append(stage)

    stage_timeline = []
    for i in range(n):
        stage_timeline.append({
            "frame": i,
            "timestamp": float(timestamps[i]),
            "stage": stages[i],
            "score": float(smoothed[i]),
            "raw_score": float(scores[i]),
        })

    violence_start = None
    escalation_start = None
    tension_start = None

    for i, stage in enumerate(stages):
        if stage == "ramp_up":
            continue
        if stage == "violence" and violence_start is None:
            violence_start = float(timestamps[i])
        if stage in ("tension", "escalation") and tension_start is None:
            tension_start = float(timestamps[i])
        if stage == "escalation" and escalation_start is None:
            escalation_start = float(timestamps[i])

    if escalation_start is None and tension_start is not None:
        escalation_start = tension_start

    anticipation_window = None
    if violence_start is not None and escalation_start is not None:
        if escalation_start < violence_start:
            anticipation_window = violence_start - escalation_start

    has_violence = violence_start is not None

    post_ramp_mask = np.array([s != "ramp_up" for s in stages])
    if post_ramp_mask.any():
        post_ramp_scores = smoothed[post_ramp_mask]
        max_score = float(post_ramp_scores.max())
        mean_score = float(post_ramp_scores.mean())
        score_std = float(post_ramp_scores.std())
    else:
        max_score = float(smoothed.max())
        mean_score = float(smoothed.mean())
        score_std = float(smoothed.std())

    return {
        "stage_timeline": stage_timeline,
        "violence_start": violence_start,
        "escalation_start": escalation_start,
        "anticipation_window": anticipation_window,
        "total_frames_analyzed": n,
        "frames_after_ramp_up": int(post_ramp_mask.sum()),
        "has_violence": has_violence,
        "has_escalation": escalation_start is not None,
        "max_score": max_score,
        "mean_score": mean_score,
        "score_std": score_std,
    }


def main():
    parser = argparse.ArgumentParser(description="Fast video analysis for anticipation curation")
    parser.add_argument('--output_dir', type=str, default='data/anticipation_curation',
                        help='Curation output directory (with candidates.json)')
    parser.add_argument('--model_path', type=str, default='outputs/outputs_temporal/best_temporal_model.pt',
                        help='Trained LSTM model checkpoint')
    parser.add_argument('--vlm_device', type=str, default='cuda:0',
                        help='Device for VLM embedding extraction')
    parser.add_argument('--lstm_device', type=str, default='cuda:1',
                        help='Device for LSTM scoring')
    parser.add_argument('--fps', type=float, default=0.5,
                        help='Frames per second to analyze (0.2 = 1 frame every 5s, 0.5 = 1 every 2s)')
    parser.add_argument('--violence_threshold', type=float, default=0.75,
                        help='LSTM score threshold for violence stage (default 0.75)')
    parser.add_argument('--tension_threshold', type=float, default=0.3,
                        help='LSTM score threshold for tension stage (default 0.3)')
    parser.add_argument('--ramp_up_frames', type=int, default=30,
                        help='Number of initial LSTM frames to skip (ramp-up artifact, default 30)')
    parser.add_argument('--max_analyze', type=int, default=None,
                        help='Maximum number of videos to analyze')
    parser.add_argument('--reanalyze', action='store_true',
                        help='Re-analyze already analyzed videos (e.g. after changing thresholds). '
                             'Will reuse saved embeddings if available.')
    parser.add_argument('--save_embeddings', action='store_true',
                        help='Also save extracted embeddings to disk for later training')
    parser.add_argument('--embeddings_dir', type=str, default='data/processed/vlm_emb/curated',
                        help='Where to save embeddings if --save_embeddings is set')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    candidates_file = output_dir / "candidates.json"

    if not candidates_file.exists():
        print("No candidates.json found. Run search and download first.")
        return

    with open(candidates_file) as f:
        data = json.load(f)
    candidates = data.get("candidates", [])

    to_analyze = [
        c for c in candidates
        if c.get("download_status") == "downloaded"
        and c.get("local_path")
        and os.path.exists(c.get("local_path", ""))
        and (c.get("analysis_status") == "pending" or args.reanalyze)
    ]

    if args.max_analyze:
        to_analyze = to_analyze[:args.max_analyze]

    if not to_analyze:
        print("No videos to analyze (all already analyzed or no downloads).")
        print("Use --reanalyze to re-analyze with new thresholds.")
        return

    can_skip_vlm = False
    if args.reanalyze and args.save_embeddings:
        emb_dir = Path(args.embeddings_dir)
        existing_embs = sum(1 for c in to_analyze 
                          if (emb_dir / f"{c['video_id']}.npz").exists())
        if existing_embs == len(to_analyze):
            can_skip_vlm = True
            print(f"All {existing_embs} embeddings already exist, skipping VLM loading")
        else:
            print(f"Embeddings found for {existing_embs}/{len(to_analyze)} videos")

    print(f"Videos to analyze: {len(to_analyze)}")
    print(f"FPS: {args.fps} ({1/args.fps:.1f}s between frames)")
    print(f"Thresholds: violence={args.violence_threshold}, tension={args.tension_threshold}")
    print(f"Ramp-up skip: {args.ramp_up_frames} frames")

    avg_duration = np.mean([c.get('duration_seconds', 300) for c in to_analyze])
    avg_frames = avg_duration * args.fps
    est_per_frame = 0.6  # ~0.5-0.7s per frame for embedding extraction
    est_total = len(to_analyze) * avg_frames * est_per_frame
    print(f"Estimated time: {est_total/60:.0f} minutes ({est_total/3600:.1f} hours)")

    teacher = None
    if not can_skip_vlm:
        print(f"\nLoading VLM on {args.vlm_device}...")
        try:
            from vlm.qwen_teacher import QwenTeacher, QwenTeacherConfig
        except Exception as e:
            print(f"Could not import QwenTeacher: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return

        teacher = QwenTeacher(QwenTeacherConfig(
            device=args.vlm_device,
            prompt_mode="violence_embedding",
            max_new_tokens=1,
            context_mode="none",
        ))
        print("  VLM loaded")
    else:
        print("\nSkipping VLM load (all embeddings cached)")

    print(f"Loading LSTM model on {args.lstm_device}...")
    lstm_device = torch.device(args.lstm_device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.model_path, map_location=lstm_device, weights_only=False)
    model_args = checkpoint.get('args', {})

    lstm_model = TemporalViolenceModel(
        input_dim=model_args.get('input_dim', 3584),
        hidden_dim=model_args.get('hidden_dim', 256),
        lstm_hidden=model_args.get('lstm_hidden', 128),
        num_lstm_layers=model_args.get('num_lstm_layers', 2),
        dropout=model_args.get('dropout', 0.3),
    )
    lstm_model.load_state_dict(checkpoint['model_state_dict'])
    lstm_model.to(lstm_device).eval()
    seq_len = model_args.get('sequence_length', 30)
    print(f"  LSTM loaded (seq_len={seq_len})")

    if args.save_embeddings:
        emb_dir = Path(args.embeddings_dir)
        emb_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Saving embeddings to {emb_dir}")

    print(f"\n{'='*60}")
    print("ANALYZING VIDEOS")
    print(f"{'='*60}\n")

    total_time = 0
    analyzed_count = 0
    violence_count = 0
    anticipation_count = 0

    for idx, candidate in enumerate(to_analyze):
        vid = candidate["video_id"]
        video_path = candidate["local_path"]
        title = candidate.get("title", "Unknown")[:60]

        print(f"[{idx+1}/{len(to_analyze)}] {vid}")
        print(f"  Title: {title}...")

        start_time = time.time()

        try:
            emb_cache_path = Path(args.embeddings_dir) / f"{vid}.npz" if args.save_embeddings else None
            loaded_from_cache = False

            if emb_cache_path and emb_cache_path.exists() and args.reanalyze:
                cached = np.load(emb_cache_path)
                embeddings = cached['emb'].astype(np.float32)
                timestamps = cached['timestamps']
                loaded_from_cache = True
                print(f"  Loaded {len(embeddings)} cached embeddings")
            else:
                embeddings, timestamps = extract_embeddings_from_video(
                    video_path, teacher, fps=args.fps, verbose=args.verbose
                )

            if len(embeddings) == 0:
                print(f"  FAILED: No embeddings extracted")
                candidate["analysis_status"] = "failed"
                candidate["notes"] = "No embeddings extracted"
                continue

            if not loaded_from_cache:
                elapsed_emb = time.time() - start_time
                print(f"  Embeddings: {len(embeddings)} frames in {elapsed_emb:.1f}s "
                      f"({elapsed_emb/len(embeddings):.2f}s/frame)")

            scores = score_with_lstm(lstm_model, embeddings, seq_len, lstm_device)

            result = classify_stages(
                scores, timestamps,
                violence_threshold=args.violence_threshold,
                tension_threshold=args.tension_threshold,
                ramp_up_frames=args.ramp_up_frames,
            )

            elapsed_total = time.time() - start_time
            total_time += elapsed_total

            if "error" in result:
                print(f"  FAILED: {result['error']}")
                candidate["analysis_status"] = "failed"
                candidate["notes"] = result["error"]
                continue

            candidate["analysis_status"] = "analyzed"
            candidate["has_violence"] = result["has_violence"]
            candidate["violence_start_second"] = result["violence_start"]
            candidate["escalation_start_second"] = result["escalation_start"]
            candidate["anticipation_window_seconds"] = result["anticipation_window"]
            candidate["stage_timeline"] = result["stage_timeline"]
            candidate["notes"] = (
                f"max_score={result['max_score']:.3f}, "
                f"mean_score={result['mean_score']:.3f}, "
                f"analyzed_at_fps={args.fps}"
            )

            analyzed_count += 1

            if result["has_violence"]:
                violence_count += 1
                print(f"  VIOLENCE at {result['violence_start']:.1f}s "
                      f"(max_score={result['max_score']:.3f})")
                if result["anticipation_window"]:
                    anticipation_count += 1
                    print(f"  Anticipation window: {result['anticipation_window']:.1f}s "
                          f"(escalation at {result['escalation_start']:.1f}s)")
            else:
                print(f"  No violence (max_score={result['max_score']:.3f}, "
                      f"mean={result['mean_score']:.3f})")

            print(f"  Done in {elapsed_total:.1f}s")

            if args.save_embeddings:
                emb_path = Path(args.embeddings_dir) / f"{vid}.npz"
                t_indices = np.arange(len(embeddings))
                raw_frame_indices = np.array([f[0] for f in
                    extract_frames(video_path, fps=args.fps)[:len(embeddings)]]
                ) if False else t_indices  # disabled: raw_frame_idx is saved as arange(N), not real frame indices
                np.savez_compressed(
                    emb_path,
                    emb=embeddings.astype(np.float16),
                    t=t_indices,
                    raw_frame_idx=t_indices,
                    timestamps=timestamps,
                    scores=scores,
                )

        except Exception as e:
            print(f"  ERROR: {e}")
            candidate["analysis_status"] = "failed"
            candidate["notes"] = str(e)
            import traceback
            traceback.print_exc()

        # Save progress after each video (in case of crash)
        with open(candidates_file, "w") as f:
            json.dump(data, f, indent=2)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"  Analyzed: {analyzed_count}/{len(to_analyze)}")
    print(f"  With violence: {violence_count}")
    print(f"  With 10+ sec anticipation: {anticipation_count}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    if analyzed_count > 0:
        print(f"  Avg per video: {total_time/analyzed_count:.1f}s")

    with open(candidates_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {candidates_file}")
    print(f"\nNext steps:")
    print(f"  python curate_anticipation_dataset.py --output_dir {args.output_dir} filter --min_anticipation 10")
    print(f"  python curate_anticipation_dataset.py --output_dir {args.output_dir} export --source_dir {args.output_dir}")


if __name__ == "__main__":
    main()