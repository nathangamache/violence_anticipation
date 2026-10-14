#!/usr/bin/env python3
"""
Experiment 1: Person Detection Filtering

Theory: If we force the model to predict "no violence" whenever there are no
people in the frame, we should reduce the false positive rate by eliminating
any above-threshold predictions on empty/people-free frames.

Pipeline:
1. Run YOLOv8 person detection on raw video frames (matching VLM sample rate)
2. Create a person-presence mask for each video
3. Score each video with the temporal LSTM model (baseline)
4. Apply person mask: score=0 when no people detected (filtered)
5. Compare FA rates, precision, recall between baseline and filtered

Usage:
    pip install ultralytics

    python experiments/exp1_person_detection.py \
        --model_path outputs/outputs_temporal/best_temporal_model.pt \
        --embeddings_dir data/processed/vlm_emb \
        --labels_dir data/processed/labels_multih \
        --raw_videos_dir data/raw_videos \
        --test_index test_index.json \
        --device cuda:1 \
        --output_dir experiments/output/exp1_person_detection
"""

import argparse
import json
import os
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from tqdm import tqdm
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, average_precision_score, roc_curve
)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: ultralytics not installed. Run: pip install ultralytics")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================================================================
# Model Definition (must match train_temporal_model.py exactly)
# ==============================================================================
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


# ==============================================================================
# Data Utilities
# ==============================================================================
def classify_video_source(video_id):
    if '_x264' in video_id:
        return 'ucf_crime'
    if video_id.startswith('v=') or '__#' in video_id:
        return 'xd_violence'
    return 'unknown'


def find_video_file(video_name, raw_videos_dir):
    """Find the actual video file for a given video name."""
    raw_dir = Path(raw_videos_dir)
    source = classify_video_source(video_name)
    search_dirs = []
    if source == 'ucf_crime':
        search_dirs = [raw_dir / 'ucf_crime']
    elif source == 'xd_violence':
        search_dirs = [raw_dir / 'xd_violence']
    search_dirs.append(raw_dir)

    for sd in search_dirs:
        if not sd.exists():
            continue
        for ext in ['.mp4', '.avi', '.mkv']:
            p = sd / f"{video_name}{ext}"
            if p.exists():
                return p
        for p in sd.rglob(f"{video_name}.*"):
            if p.suffix.lower() in ['.mp4', '.avi', '.mkv']:
                return p
        if video_name.endswith('_x264'):
            base = video_name[:-5]
            for p in sd.rglob(f"{base}*"):
                if p.suffix.lower() in ['.mp4', '.avi', '.mkv']:
                    return p
    return None


def find_embedding_file(video_name, embeddings_dir):
    embeddings_dir = Path(embeddings_dir)
    source = classify_video_source(video_name)
    for subdir in [source, 'ucf_crime', 'xd_violence', '']:
        for ext in ['.npz', '.npy']:
            path = (embeddings_dir / subdir / f"{video_name}{ext}") if subdir else (embeddings_dir / f"{video_name}{ext}")
            if path.exists():
                return path
    return None


def find_label_file(video_name, labels_dir):
    labels_dir = Path(labels_dir)
    for ext in ['.npy', '.npz']:
        p = labels_dir / f"{video_name}{ext}"
        if p.exists():
            return p
    for sub in ['ucf_crime', 'xd_violence']:
        for ext in ['.npy', '.npz']:
            p = labels_dir / sub / f"{video_name}{ext}"
            if p.exists():
                return p
    return None


def load_embedding(path):
    path = Path(path)
    try:
        if path.suffix == '.npz':
            data = np.load(path)
            for key in ['emb', 'embedding', 'embeddings', 'features', 'arr_0']:
                if key in data.keys():
                    arr = data[key]
                    return arr.astype(np.float32) if arr.dtype == np.float16 else arr
        else:
            arr = np.load(path)
            return arr.astype(np.float32) if arr.dtype == np.float16 else arr
    except Exception:
        pass
    return None


def load_labels_from_file(path):
    path = Path(path)
    try:
        if path.suffix == '.npz':
            data = np.load(path)
            for key in ['labels', 'label', 'arr_0']:
                if key in data.keys():
                    return data[key]
        else:
            return np.load(path)
    except Exception:
        pass
    return None


# ==============================================================================
# Person Detection
# ==============================================================================
def detect_people_in_video(video_path, yolo_model, sample_every_t=5, total_T=None,
                           confidence_threshold=0.3):
    """
    Run YOLO person detection on sampled frames from a video.

    Returns:
        person_mask: np.ndarray of shape [T] - 1=person detected, 0=no person
        person_counts: np.ndarray of shape [T] - person count per frame
        person_pct: float - percentage of frames with at least one person
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None, 0.0

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_T is None:
        total_T = total_video_frames

    t_steps = list(range(0, total_T, max(1, sample_every_t)))
    person_mask = np.zeros(len(t_steps), dtype=np.int32)
    person_counts = np.zeros(len(t_steps), dtype=np.int32)

    for idx, t in enumerate(t_steps):
        raw_frame_idx = int(round(((t + 0.5) / float(total_T)) * float(total_video_frames)))
        raw_frame_idx = max(0, min(total_video_frames - 1, raw_frame_idx))

        cap.set(cv2.CAP_PROP_POS_FRAMES, raw_frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        results = yolo_model(frame, verbose=False, conf=confidence_threshold)
        for r in results:
            boxes = r.boxes
            person_boxes = boxes[boxes.cls == 0]  # Class 0 = person in COCO
            n_people = len(person_boxes)
            person_mask[idx] = 1 if n_people > 0 else 0
            person_counts[idx] = n_people

    cap.release()
    person_pct = person_mask.mean() * 100
    return person_mask, person_counts, person_pct


# ==============================================================================
# Model Scoring
# ==============================================================================
@torch.no_grad()
def score_embeddings(model, embeddings, seq_len, device, batch_size=256):
    """Score embeddings using sliding window."""
    n = len(embeddings)
    if n < seq_len:
        return np.full(n, 0.5)
    n_seq = n - seq_len + 1
    all_scores = []
    for start in range(0, n_seq, batch_size):
        end = min(start + batch_size, n_seq)
        batch = np.stack([embeddings[i:i + seq_len] for i in range(start, end)])
        x = torch.tensor(batch, dtype=torch.float32).to(device)
        scores = torch.sigmoid(model(x)).cpu().numpy()
        all_scores.extend(scores)
    return np.concatenate([np.full(seq_len - 1, all_scores[0]), np.array(all_scores)])


# ==============================================================================
# Metrics
# ==============================================================================
def compute_metrics(scores, labels, threshold):
    """Compute classification and operational metrics."""
    preds = (scores >= threshold).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    roc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.5
    pr_auc = average_precision_score(labels, scores) if len(set(labels)) > 1 else 0

    # Count FA events (contiguous FP regions)
    in_fa = False
    n_fa = 0
    for p, l in zip(preds, labels):
        if p == 1 and l == 0 and not in_fa:
            in_fa = True
            n_fa += 1
        elif p == 0 or l == 1:
            in_fa = False
    total_hours = len(labels) / 3600  # assuming 1 fps

    return {
        'roc_auc': roc, 'pr_auc': pr_auc,
        'precision': prec, 'recall': rec, 'f1': f1,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'n_fa_events': n_fa,
        'fa_per_hour': n_fa / total_hours if total_hours > 0 else 0,
        'fpr': fpr,
    }


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--embeddings_dir', type=str, default='data/processed/vlm_emb')
    parser.add_argument('--labels_dir', type=str, default='data/processed/labels_multih')
    parser.add_argument('--raw_videos_dir', type=str, default='data/raw_videos')
    parser.add_argument('--test_index', type=str, default='test_index.json')
    parser.add_argument('--source_filter', type=str, default=None,
                        help='Filter to xd_violence or ucf_crime, or None for both')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Detection threshold (auto-detected from baseline if not set)')
    parser.add_argument('--yolo_model', type=str, default='yolov8n.pt',
                        help='YOLO model to use')
    parser.add_argument('--yolo_conf', type=float, default=0.3,
                        help='YOLO confidence threshold for person detection')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--output_dir', type=str, default='experiments/output/exp1_person_detection')
    parser.add_argument('--sample_every_t', type=int, default=5,
                        help='VLM sampling rate (every_t from vlm_label_dataset.py)')
    parser.add_argument('--person_holdover', type=int, default=5,
                        help='After a person is detected, keep person_mask=1 for this many '
                             'additional frames. Prevents YOLO single-frame misses from '
                             'fragmenting alarm events. Set to 0 to disable smoothing.')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # --- Load violence anticipation model ---
    print("Loading violence anticipation model...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model_args = checkpoint.get('args', {})
    model = TemporalViolenceModel(
        input_dim=model_args.get('input_dim', 3584),
        hidden_dim=model_args.get('hidden_dim', 256),
        lstm_hidden=model_args.get('lstm_hidden', 128),
        num_lstm_layers=model_args.get('num_lstm_layers', 2),
        dropout=model_args.get('dropout', 0.3),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    seq_len = model_args.get('sequence_length', 30)
    print(f"  Loaded model (seq_len={seq_len})")

    # --- Load YOLO ---
    print("Loading YOLO person detector...")
    if not YOLO_AVAILABLE:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        return
    yolo = YOLO(args.yolo_model)
    print(f"  Loaded {args.yolo_model}")

    # --- Load metadata for total_frames ---
    metadata = {}
    for meta_file in ['data/metadata/xd_violence.json', 'data/metadata/ucf_crime.json']:
        if os.path.exists(meta_file):
            with open(meta_file) as f:
                metadata.update(json.load(f))

    # --- Load test videos ---
    print(f"\nLoading test videos from {args.test_index}...")
    with open(args.test_index) as f:
        items = json.load(f)

    results_per_video = []
    all_scores_baseline = []
    all_scores_filtered = []
    all_labels = []
    all_person_masks = []

    for item in tqdm(items, desc="Processing videos"):
        video_name = Path(item.get('graph_path', item.get('label_path', ''))).stem
        source = classify_video_source(video_name)

        if args.source_filter and source != args.source_filter:
            continue

        # Load embeddings and labels
        emb_path = find_embedding_file(video_name, args.embeddings_dir)
        lbl_path = find_label_file(video_name, args.labels_dir)
        if emb_path is None or lbl_path is None:
            continue

        emb = load_embedding(emb_path)
        lbl = load_labels_from_file(lbl_path)
        if emb is None or lbl is None:
            continue
        if lbl.ndim > 1:
            lbl = lbl.max(axis=1)
        lbl = (lbl > 0).astype(np.float32)

        min_len = min(len(emb), len(lbl))
        emb, lbl = emb[:min_len], lbl[:min_len]

        # Find raw video file
        video_path = find_video_file(video_name, args.raw_videos_dir)
        if video_path is None:
            continue

        # --- Run person detection ---
        person_mask, person_counts, person_pct = detect_people_in_video(
            video_path, yolo,
            sample_every_t=args.sample_every_t,
            total_T=len(emb) * args.sample_every_t,
            confidence_threshold=args.yolo_conf,
        )

        if person_mask is None:
            continue

        # Align person_mask with embeddings (nearest-neighbor resize)
        if len(person_mask) != len(emb):
            indices = np.round(np.linspace(0, len(person_mask) - 1, len(emb))).astype(int)
            person_mask = person_mask[indices]
            person_counts = person_counts[indices]

        # --- Temporal smoothing of person mask ---
        # If YOLO saw a person within the last N frames, assume they're still there.
        # This prevents single-frame YOLO misses from fragmenting continuous alarms
        # into many short ones.
        raw_person_pct = person_mask.mean() * 100  # before smoothing
        if args.person_holdover > 0:
            smoothed_mask = np.zeros_like(person_mask)
            frames_since_person = args.person_holdover + 1  # start as "no person seen"
            for i in range(len(person_mask)):
                if person_mask[i] == 1:
                    frames_since_person = 0
                    smoothed_mask[i] = 1
                else:
                    frames_since_person += 1
                    if frames_since_person <= args.person_holdover:
                        smoothed_mask[i] = 1  # holdover: person was recently here
                    else:
                        smoothed_mask[i] = 0  # genuinely no person for a while
            person_mask = smoothed_mask
        smoothed_person_pct = person_mask.mean() * 100

        # --- Score with model (baseline) ---
        scores = score_embeddings(model, emb, seq_len, device)

        # --- Apply person mask ---
        # No people in frame -> force score to 0 -> model cannot predict violence
        # People in frame -> keep original model score
        scores_filtered = scores.copy()
        scores_filtered[person_mask == 0] = 0.0

        # Track diagnostic stats
        n_no_person = int((person_mask == 0).sum())
        n_no_person_but_high_score = int(((person_mask == 0) & (scores > 0.3)).sum())

        results_per_video.append({
            'video_id': video_name,
            'source': source,
            'has_violence': bool(lbl.max() > 0),
            'n_frames': len(emb),
            'person_pct_raw': float(raw_person_pct),
            'person_pct_smoothed': float(smoothed_person_pct),
            'avg_person_count': float(person_counts.mean()),
            'max_person_count': int(person_counts.max()),
            'n_no_person_frames': n_no_person,
            'n_no_person_high_score': n_no_person_but_high_score,
        })

        all_scores_baseline.extend(scores)
        all_scores_filtered.extend(scores_filtered)
        all_labels.extend(lbl)
        all_person_masks.extend(person_mask)

    # --- Aggregate results ---
    all_scores_baseline = np.array(all_scores_baseline)
    all_scores_filtered = np.array(all_scores_filtered)
    all_labels = np.array(all_labels)
    all_person_masks = np.array(all_person_masks)

    print(f"\nProcessed {len(results_per_video)} videos")
    print(f"Total frames: {len(all_labels):,}")

    # --- Find threshold if not provided ---
    threshold = args.threshold
    if threshold is None:
        prec, rec, threshs = precision_recall_curve(all_labels, all_scores_baseline)
        f1s = 2 * prec * rec / (prec + rec + 1e-8)
        best_idx = np.argmax(f1s[:-1])
        threshold = threshs[best_idx]
        print(f"Auto-detected threshold: {threshold:.4f}")

    # --- Compare baseline vs filtered ---
    baseline_metrics = compute_metrics(all_scores_baseline, all_labels, threshold)
    filtered_metrics = compute_metrics(all_scores_filtered, all_labels, threshold)

    print(f"\n{'='*70}")
    print("RESULTS: Person Detection Filtering")
    print(f"{'='*70}")
    print(f"Threshold: {threshold:.4f}")
    print(f"\n{'Metric':<25} {'Baseline':>12} {'Person-Filtered':>15} {'Change':>10}")
    print("-" * 65)
    for metric in ['roc_auc', 'pr_auc', 'precision', 'recall', 'f1', 'fp', 'fn', 'n_fa_events', 'fa_per_hour']:
        b = baseline_metrics[metric]
        f = filtered_metrics[metric]
        if isinstance(b, float):
            change = f - b
            print(f"  {metric:<23} {b:>12.4f} {f:>15.4f} {change:>+10.4f}")
        else:
            change = f - b
            print(f"  {metric:<23} {b:>12} {f:>15} {change:>+10}")

    # --- Key insight stats ---
    raw_pcts = [r['person_pct_raw'] for r in results_per_video]
    smoothed_pcts = [r['person_pct_smoothed'] for r in results_per_video]
    violent_pcts = [r['person_pct_smoothed'] for r in results_per_video if r['has_violence']]
    nonviolent_pcts = [r['person_pct_smoothed'] for r in results_per_video if not r['has_violence']]
    total_no_person = sum(r.get('n_no_person_frames', 0) for r in results_per_video)
    total_no_person_high = sum(r.get('n_no_person_high_score', 0) for r in results_per_video)

    print(f"\nPerson Detection Stats:")
    print(f"  Raw YOLO:      {np.mean(raw_pcts):.1f}% frames with people (median={np.median(raw_pcts):.1f}%)")
    print(f"  After holdover({args.person_holdover}): {np.mean(smoothed_pcts):.1f}% frames with people (median={np.median(smoothed_pcts):.1f}%)")
    if violent_pcts:
        print(f"  Violent vids:  {np.mean(violent_pcts):.1f}% frames with people (smoothed)")
    if nonviolent_pcts:
        print(f"  Non-viol vids: {np.mean(nonviolent_pcts):.1f}% frames with people (smoothed)")

    print(f"\n  KEY INSIGHT:")
    print(f"    Frames with NO people: {total_no_person:,} / {len(all_labels):,} ({100*total_no_person/len(all_labels):.1f}%)")
    print(f"    Of those, model scored >0.3: {total_no_person_high:,} (these become suppressed FPs)")
    fp_reduction = baseline_metrics['fp'] - filtered_metrics['fp']
    print(f"    FP frames eliminated: {fp_reduction:,} ({100*fp_reduction/max(baseline_metrics['fp'],1):.1f}% of baseline FPs)")
    fa_reduction = baseline_metrics['n_fa_events'] - filtered_metrics['n_fa_events']
    print(f"    FA events eliminated: {fa_reduction}")

    # --- Check if person filtering hurts recall ---
    if filtered_metrics['fn'] > baseline_metrics['fn']:
        fn_increase = filtered_metrics['fn'] - baseline_metrics['fn']
        print(f"\n  WARNING: Person filtering increased FN by {fn_increase} frames")
        print(f"    This means YOLO missed people during {fn_increase} violent frames.")
        print(f"    Recall dropped: {baseline_metrics['recall']:.4f} -> {filtered_metrics['recall']:.4f}")

    # --- Save results ---
    output = {
        'threshold': threshold,
        'person_holdover': args.person_holdover,
        'baseline_metrics': baseline_metrics,
        'filtered_metrics': filtered_metrics,
        'person_stats': {
            'mean_person_pct_raw': float(np.mean(raw_pcts)),
            'mean_person_pct_smoothed': float(np.mean(smoothed_pcts)),
            'median_person_pct_smoothed': float(np.median(smoothed_pcts)),
            'violent_mean_pct': float(np.mean(violent_pcts)) if violent_pcts else 0,
            'nonviolent_mean_pct': float(np.mean(nonviolent_pcts)) if nonviolent_pcts else 0,
            'total_no_person_frames': total_no_person,
            'total_no_person_high_score': total_no_person_high,
            'fp_reduction': fp_reduction,
            'fa_event_reduction': fa_reduction,
        },
        'per_video': results_per_video,
    }
    with open(out_dir / 'exp1_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    # --- Plot: Person % histogram + metric comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Person detection rate per video
    axes[0].hist(smoothed_pcts, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('% Frames with People (smoothed)')
    axes[0].set_ylabel('Videos')
    axes[0].set_title(f'Person Detection Rate per Video\n(holdover={args.person_holdover} frames)')
    axes[0].axvline(x=np.mean(smoothed_pcts), color='red', linestyle='--',
                    label=f'Mean={np.mean(smoothed_pcts):.1f}%')
    axes[0].legend()

    # Metric comparison bars
    metrics_compare = ['precision', 'recall', 'f1', 'roc_auc']
    x = np.arange(len(metrics_compare))
    b_vals = [baseline_metrics[m] for m in metrics_compare]
    f_vals = [filtered_metrics[m] for m in metrics_compare]
    width = 0.35
    axes[1].bar(x - width / 2, b_vals, width, label='Baseline', color='steelblue', alpha=0.7)
    axes[1].bar(x + width / 2, f_vals, width, label='Person-Filtered', color='coral', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics_compare)
    axes[1].set_ylabel('Score')
    axes[1].set_title('Baseline vs Person-Filtered')
    axes[1].legend()
    axes[1].set_ylim([0, 1.05])

    # FA comparison
    axes[2].bar(['Baseline', 'Person-Filtered'],
                [baseline_metrics['fa_per_hour'], filtered_metrics['fa_per_hour']],
                color=['steelblue', 'coral'], alpha=0.7, edgecolor='black')
    axes[2].set_ylabel('False Alarms per Hour')
    axes[2].set_title('FA Rate Comparison')
    for i, v in enumerate([baseline_metrics['fa_per_hour'], filtered_metrics['fa_per_hour']]):
        axes[2].text(i, v + 0.1, f'{v:.1f}', ha='center', fontsize=12)

    plt.suptitle(f'Experiment 1: Person Detection Filtering (threshold={threshold:.4f})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'exp1_person_detection_results.png', dpi=150)
    plt.close()

    # --- Plot: Score distributions with/without people ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    no_person_scores = all_scores_baseline[all_person_masks == 0]
    has_person_scores = all_scores_baseline[all_person_masks == 1]

    if len(no_person_scores) > 0:
        axes[0].hist(no_person_scores, bins=50, alpha=0.6, color='gray',
                     density=True, label=f'No people (n={len(no_person_scores):,})')
    if len(has_person_scores) > 0:
        axes[0].hist(has_person_scores, bins=50, alpha=0.6, color='steelblue',
                     density=True, label=f'People present (n={len(has_person_scores):,})')
    axes[0].axvline(x=threshold, color='red', linestyle='--', label=f'Threshold={threshold:.3f}')
    axes[0].set_xlabel('Model Score')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Score Distribution by Person Presence')
    axes[0].legend()

    # Breakdown: no-person frames that are FP vs TN
    if len(no_person_scores) > 0:
        no_person_labels = all_labels[all_person_masks == 0]
        no_person_fp = no_person_scores[(no_person_scores >= threshold) & (no_person_labels == 0)]
        no_person_tn = no_person_scores[(no_person_scores < threshold) & (no_person_labels == 0)]
        no_person_tp = no_person_scores[(no_person_scores >= threshold) & (no_person_labels == 1)]
        no_person_fn = no_person_scores[(no_person_scores < threshold) & (no_person_labels == 1)]

        counts = [len(no_person_tn), len(no_person_fp), len(no_person_fn), len(no_person_tp)]
        labels_bar = ['TN\n(correct reject)', 'FP\n(eliminated!)', 'FN\n(was missed)', 'TP\n(lost recall)']
        colors = ['green', 'red', 'orange', 'blue']
        bars = axes[1].bar(labels_bar, counts, color=colors, alpha=0.7, edgecolor='black')
        axes[1].set_ylabel('Frame Count')
        axes[1].set_title('No-Person Frames Breakdown')
        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                             str(cnt), ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(out_dir / 'exp1_score_by_person_presence.png', dpi=150)
    plt.close()

    print(f"\nResults saved to {out_dir}/")


if __name__ == '__main__':
    main()