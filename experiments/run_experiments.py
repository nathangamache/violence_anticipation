#!/usr/bin/env python3
"""
Comprehensive Experiment Runner for Violence Anticipation
=========================================================

Runs 4 experiments:

1. PERSON DETECTION FILTERING
   - Run YOLOv8 person detection on raw video frames
   - Compute % frames with people per video
   - Mask model scores to only frames with people
   - Measure FA rate reduction

2. DATASET-SPLIT EVALUATION
   a. Score all non-violent clips → expected: all scores low
   b. Score all violent clips    → expected: scores rise near violence
   c. Build unified eval dataset from both, generate metrics

3. CROSS-DATASET TRAINING
   - Train on XD-Violence only → test on UCF-Crime and XD-Violence
   - Train on UCF-Crime only   → test on XD-Violence and UCF-Crime
   - Train on both combined    → test on XD-Violence and UCF-Crime
   - Compare generalization

4. UCF-CRIME EVALUATION WITH GRAPHS
   - Build UCF-Crime simulation
   - Run detection + anticipation evaluation
   - Generate ROC, PR, timeline, lead-time, FA distribution graphs

Usage:
    # Run all experiments
    python run_experiments.py --all \
        --embeddings_dir data/processed/vlm_emb \
        --labels_dir data/processed/labels_multih \
        --model_path outputs/outputs_temporal/best_temporal_model.pt \
        --raw_videos_dir data/raw_videos \
        --output_dir experiments_output

    # Run individual experiments
    python run_experiments.py --exp1 ...
    python run_experiments.py --exp2 ...
    python run_experiments.py --exp3 ...
    python run_experiments.py --exp4 ...
"""

import argparse
import json
import os
import re
import time
import copy
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix, f1_score
)
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


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
        self.lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=lstm_hidden,
            num_layers=num_lstm_layers, batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        lstm_output_dim = lstm_hidden * 2 if bidirectional else lstm_hidden
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_dim, 64), nn.Tanh(), nn.Linear(64, 1),
            )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 64), nn.GELU(),
            nn.Dropout(dropout / 2), nn.Linear(64, 1),
        )

    def forward(self, x, lengths=None):
        B, T, D = x.shape
        encoded = self.frame_encoder(x.view(B * T, D)).view(B, T, -1)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(encoded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(encoded)
        if self.use_attention:
            attn_weights = F.softmax(self.attention(lstm_out), dim=1)
            context = (lstm_out * attn_weights).sum(dim=1)
        else:
            context = lstm_out[:, -1, :]
        return self.classifier(context).squeeze(-1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * ce).mean()


# ==============================================================================
# Data Loading Utilities
# ==============================================================================
def classify_video_source(video_id: str) -> str:
    if '_x264' in video_id:
        return 'ucf_crime'
    if video_id.startswith('v=') or re.search(r'\d{4}__#', video_id) or '__#' in video_id:
        return 'xd_violence'
    return 'unknown'


def extract_video_name(item) -> str:
    if isinstance(item, dict):
        path = item.get('graph_path') or item.get('label_path', '')
        return Path(path).stem
    return str(item)


def find_embedding_file(video_name, embeddings_dir):
    embeddings_dir = Path(embeddings_dir)
    source = classify_video_source(video_name)
    for subdir in [source, 'ucf_crime', 'xd_violence', '']:
        for ext in ['.npz', '.npy']:
            path = embeddings_dir / subdir / f"{video_name}{ext}" if subdir else embeddings_dir / f"{video_name}{ext}"
            if path.exists():
                return path
    return None


def find_label_file(video_name, labels_dir):
    labels_dir = Path(labels_dir)
    for ext in ['.npy', '.npz']:
        path = labels_dir / f"{video_name}{ext}"
        if path.exists():
            return path
    for subdir in ['ucf_crime', 'xd_violence']:
        for ext in ['.npy', '.npz']:
            path = labels_dir / subdir / f"{video_name}{ext}"
            if path.exists():
                return path
    return None


def load_embedding(path):
    try:
        path = Path(path)
        if path.suffix == '.npz':
            data = np.load(path)
            for key in ['emb', 'embedding', 'embeddings', 'features', 'arr_0']:
                if key in data.keys():
                    arr = data[key]
                    return arr.astype(np.float32) if arr.dtype == np.float16 else arr
        else:
            arr = np.load(path)
            return arr.astype(np.float32) if arr.dtype == np.float16 else arr
    except Exception as e:
        pass
    return None


def load_labels_from_file(path):
    try:
        path = Path(path)
        if path.suffix == '.npz':
            data = np.load(path)
            for key in ['labels', 'label', 'arr_0']:
                if key in data.keys():
                    return data[key]
            if data.keys():
                return data[list(data.keys())[0]]
        else:
            return np.load(path)
    except:
        pass
    return None


def load_video_data(index_path, embeddings_dir, labels_dir, source_filter=None):
    """Load all videos matching source filter, return list of dicts."""
    with open(index_path) as f:
        items = json.load(f)

    videos = []
    missing_emb, missing_lbl = 0, 0
    for item in items:
        video_name = extract_video_name(item)
        if source_filter and classify_video_source(video_name) != source_filter:
            continue
        emb_path = find_embedding_file(video_name, embeddings_dir)
        lbl_path = find_label_file(video_name, labels_dir)
        if emb_path is None:
            missing_emb += 1
            continue
        if lbl_path is None:
            missing_lbl += 1
            continue
        emb = load_embedding(emb_path)
        lbl = load_labels_from_file(lbl_path)
        if emb is None or lbl is None:
            continue
        if lbl.ndim > 1:
            lbl = lbl.max(axis=1)
        lbl = (lbl > 0).astype(np.float32)
        min_len = min(len(emb), len(lbl))
        videos.append({
            'embeddings': emb[:min_len],
            'labels': lbl[:min_len],
            'video_id': video_name,
            'source': classify_video_source(video_name),
        })
    if missing_emb or missing_lbl:
        print(f"  [load] Missing: {missing_emb} embeddings, {missing_lbl} labels")
    return videos


def load_all_videos_both_sources(index_paths, embeddings_dir, labels_dir):
    """Load videos from all index files, both sources."""
    all_videos = []
    seen = set()
    for idx_path in index_paths:
        vids = load_video_data(idx_path, embeddings_dir, labels_dir, source_filter=None)
        for v in vids:
            if v['video_id'] not in seen:
                seen.add(v['video_id'])
                all_videos.append(v)
    return all_videos


# ==============================================================================
# Sequence Dataset (for training / evaluation)
# ==============================================================================
class TemporalViolenceDataset(Dataset):
    def __init__(self, video_data, sequence_length=30, horizon_sec=30.0, fps=1.0, stride=5, augment=False):
        self.sequence_length = sequence_length
        self.horizon_frames = int(horizon_sec * fps)
        self.stride = stride
        self.augment = augment
        self.sequences = []
        self._build(video_data)

    def _build(self, video_data):
        for video in video_data:
            emb, lbl = video['embeddings'], video['labels']
            for start in range(0, len(emb) - self.sequence_length + 1, self.stride):
                end = start + self.sequence_length
                seq_emb = emb[start:end]
                horizon_end = min(end + self.horizon_frames, len(lbl))
                future_viol = lbl[end:horizon_end].max() if horizon_end > end else 0
                current_viol = lbl[start:end].max()
                label = max(current_viol, future_viol)
                self.sequences.append({'embeddings': seq_emb, 'label': float(label)})
        pos_rate = np.mean([s['label'] for s in self.sequences]) if self.sequences else 0
        print(f"  Built {len(self.sequences)} sequences ({pos_rate:.1%} positive)")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        s = self.sequences[idx]
        emb = torch.tensor(s['embeddings'], dtype=torch.float32)
        if self.augment and np.random.random() < 0.15:
            emb = emb + torch.randn_like(emb) * 0.01
        return {'embeddings': emb, 'label': torch.tensor(s['label'], dtype=torch.float32)}


# ==============================================================================
# Model Helpers
# ==============================================================================
def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    args = checkpoint.get('args', {})
    model = TemporalViolenceModel(
        input_dim=args.get('input_dim', 3584),
        hidden_dim=args.get('hidden_dim', 256),
        lstm_hidden=args.get('lstm_hidden', 128),
        num_lstm_layers=args.get('num_lstm_layers', 2),
        dropout=args.get('dropout', 0.3),
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()
    seq_len = args.get('sequence_length', 30)
    return model, seq_len, args


@torch.no_grad()
def score_embeddings(model, embeddings, seq_len, device, batch_size=256):
    """Score a sequence of embeddings frame-by-frame using sliding window."""
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


def train_model(train_videos, val_videos, args_dict, device, output_dir):
    """Train a TemporalViolenceModel and return the best checkpoint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_len = args_dict.get('sequence_length', 30)
    horizon = args_dict.get('horizon_sec', 30.0)
    fps = args_dict.get('fps', 1.0)
    stride = args_dict.get('stride', 5)
    epochs = args_dict.get('epochs', 100)
    patience = args_dict.get('patience', 15)
    batch_size = args_dict.get('batch_size', 64)
    lr = args_dict.get('lr', 1e-4)

    train_ds = TemporalViolenceDataset(train_videos, seq_len, horizon, fps, stride, augment=True)
    val_ds = TemporalViolenceDataset(val_videos, seq_len, horizon, fps, stride, augment=False)

    if len(train_ds) == 0 or len(val_ds) == 0:
        print(f"  [SKIP] Not enough data: train={len(train_ds)}, val={len(val_ds)}")
        return None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = TemporalViolenceModel(
        input_dim=args_dict.get('input_dim', 3584),
        hidden_dim=args_dict.get('hidden_dim', 256),
        lstm_hidden=args_dict.get('lstm_hidden', 128),
        num_lstm_layers=args_dict.get('num_lstm_layers', 2),
        dropout=args_dict.get('dropout', 0.3),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)

    best_val_auc = 0
    best_state = None
    no_improve = 0
    history = {'train_loss': [], 'train_auc': [], 'val_auc': [], 'val_pr_auc': []}

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        all_labels, all_scores = [], []
        for batch in train_loader:
            emb = batch['embeddings'].to(device)
            lbl = batch['label'].to(device)
            optimizer.zero_grad()
            logits = model(emb)
            loss = criterion(logits, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            all_labels.extend(lbl.cpu().numpy())
            all_scores.extend(torch.sigmoid(logits).detach().cpu().numpy())
        scheduler.step()

        train_auc = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0.5
        history['train_loss'].append(total_loss / len(train_loader))
        history['train_auc'].append(train_auc)

        # Val
        model.eval()
        val_labels, val_scores = [], []
        with torch.no_grad():
            for batch in val_loader:
                emb = batch['embeddings'].to(device)
                logits = model(emb)
                val_labels.extend(batch['label'].numpy())
                val_scores.extend(torch.sigmoid(logits).cpu().numpy())

        val_auc = roc_auc_score(val_labels, val_scores) if len(set(val_labels)) > 1 else 0.5
        val_pr = average_precision_score(val_labels, val_scores) if len(set(val_labels)) > 1 else 0
        history['val_auc'].append(val_auc)
        history['val_pr_auc'].append(val_pr)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, val_pr={val_pr:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    if best_state is None:
        return None

    model.load_state_dict(best_state)
    save_path = output_dir / 'best_temporal_model.pt'
    torch.save({
        'model_state_dict': best_state,
        'args': args_dict,
        'best_val_auc': best_val_auc,
        'history': history,
    }, save_path)
    print(f"    Saved model to {save_path} (val_auc={best_val_auc:.4f})")
    return model, history


# ==============================================================================
# Simulation Builder (for Exp 2, 4)
# ==============================================================================
def build_simulation(videos, fps=1.0, horizon_sec=30.0, duration_hours=8.0,
                     violence_pct=1.0, seed=42):
    """Build long-form simulation from video list."""
    np.random.seed(seed)
    violent = [v for v in videos if v['labels'].max() > 0]
    nonviolent = [v for v in videos if v['labels'].max() == 0]

    if not violent:
        print("  WARNING: No violent videos for simulation")
        # Use only non-violent
        all_emb = np.concatenate([v['embeddings'] for v in nonviolent])
        all_lbl = np.zeros(len(all_emb))
        return {'embeddings': all_emb, 'anticipation_labels': all_lbl,
                'frame_labels': all_lbl, 'fps': fps}

    target_frames = int(duration_hours * 3600 * fps)
    np.random.shuffle(violent)
    np.random.shuffle(nonviolent)

    stream_emb, stream_lbl = [], []
    avg_pad = max(1, (target_frames // max(len(violent), 1)))
    nv_idx = 0

    for v_vid in violent:
        # Add non-violent padding
        pad_frames = int(avg_pad * np.random.uniform(0.5, 1.5))
        added = 0
        while added < pad_frames and nonviolent:
            nv_vid = nonviolent[nv_idx % len(nonviolent)]
            stream_emb.append(nv_vid['embeddings'])
            stream_lbl.append(np.zeros(len(nv_vid['embeddings'])))
            added += len(nv_vid['embeddings'])
            nv_idx += 1
        # Add violent video
        stream_emb.append(v_vid['embeddings'])
        stream_lbl.append(v_vid['labels'])

    all_emb = np.concatenate(stream_emb)
    all_lbl = np.concatenate(stream_lbl)

    # Build anticipation labels
    horizon_frames = int(horizon_sec * fps)
    antic_labels = np.zeros(len(all_lbl))
    for t in range(len(all_lbl)):
        future_end = min(t + horizon_frames, len(all_lbl))
        antic_labels[t] = all_lbl[t:future_end].max()

    return {
        'embeddings': all_emb, 'frame_labels': all_lbl,
        'anticipation_labels': antic_labels, 'fps': fps,
    }


# ==============================================================================
# Evaluation & Event Analysis
# ==============================================================================
def evaluate_simulation(model, sim_data, seq_len, device, threshold=None):
    """Full evaluation on simulation data: ROC, PR, events, lead times, FA."""
    embeddings = sim_data['embeddings']
    labels = sim_data['anticipation_labels']
    fps = float(sim_data.get('fps', 1.0))

    scores = score_embeddings(model, embeddings, seq_len, device)
    timestamps = np.arange(len(embeddings)) / fps

    roc_auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.5
    pr_auc = average_precision_score(labels, scores) if len(set(labels)) > 1 else 0

    # Find best threshold if not given
    if threshold is None:
        prec, rec, threshs = precision_recall_curve(labels, scores)
        f1s = 2 * prec * rec / (prec + rec + 1e-8)
        best_idx = np.argmax(f1s[:-1])
        threshold = threshs[best_idx]

    preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tp, fp, fn, tn = cm[1, 1], cm[0, 1], cm[1, 0], cm[0, 0]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Event analysis
    events, lead_times = _analyze_events(labels, preds, scores, timestamps, fps)
    fa_times = _find_false_alarms(labels, preds, timestamps)

    n_viol = len([e for e in events if e['is_violence']])
    n_det = len([e for e in events if e['is_violence'] and e.get('detected', False)])
    total_sec = len(embeddings) / fps

    fpr_arr, tpr_arr, roc_threshs = roc_curve(labels, scores)
    pr_prec, pr_rec, pr_threshs = precision_recall_curve(labels, scores)

    return {
        'scores': scores, 'labels': labels, 'predictions': preds,
        'timestamps': timestamps, 'threshold': threshold,
        'roc_auc': roc_auc, 'pr_auc': pr_auc,
        'precision': precision, 'recall': recall, 'f1': f1,
        'confusion_matrix': cm,
        'fpr': fpr_arr, 'tpr': tpr_arr, 'roc_thresholds': roc_threshs,
        'pr_precision': pr_prec, 'pr_recall': pr_rec, 'pr_thresholds': pr_threshs,
        'events': events, 'lead_times': lead_times,
        'n_violence_events': n_viol, 'n_detected': n_det,
        'n_false_alarms': len(fa_times),
        'detection_rate': n_det / n_viol if n_viol > 0 else 0,
        'fa_per_hour': len(fa_times) / (total_sec / 3600),
        'avg_lead_time': np.mean(lead_times) if lead_times else 0,
        'total_duration_sec': total_sec,
    }


def _analyze_events(labels, preds, scores, timestamps, fps):
    events, lead_times = [], []
    in_event, event_start = False, 0
    for i, label in enumerate(labels):
        if label == 1 and not in_event:
            in_event, event_start = True, i
        elif label == 0 and in_event:
            in_event = False
            events.append({'is_violence': True, 'start_frame': event_start, 'end_frame': i - 1,
                           'start_time': timestamps[event_start], 'end_time': timestamps[i - 1],
                           'duration_sec': (i - 1 - event_start) / fps})
    if in_event:
        events.append({'is_violence': True, 'start_frame': event_start, 'end_frame': len(labels) - 1,
                       'start_time': timestamps[event_start], 'end_time': timestamps[-1],
                       'duration_sec': (len(labels) - 1 - event_start) / fps})

    horizon_frames = int(30 * fps)
    for event in events:
        if not event['is_violence']:
            continue
        start = event['start_frame']
        check_start = max(0, start - horizon_frames)
        detected, lead_time = False, None
        for i in range(check_start, event['end_frame'] + 1):
            if preds[i] == 1 and not detected:
                detected = True
                lead_time = max(0, (start - i) / fps)
        event['detected'] = detected
        event['lead_time_sec'] = lead_time
        event['max_score'] = float(scores[event['start_frame']:event['end_frame'] + 1].max())
        if detected and lead_time is not None:
            lead_times.append(lead_time)
    return events, lead_times


def _find_false_alarms(labels, preds, timestamps):
    fa_times, in_fa = [], False
    for i, (l, p) in enumerate(zip(labels, preds)):
        if p == 1 and l == 0 and not in_fa:
            in_fa = True
            fa_times.append(timestamps[i])
        elif p == 0 or l == 1:
            in_fa = False
    return fa_times


# ==============================================================================
# Plotting Functions
# ==============================================================================
def plot_roc(results, path, title="ROC Curve"):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(results['fpr'], results['tpr'], 'b-', lw=2.5, label=f'ROC (AUC={results["roc_auc"]:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.fill_between(results['fpr'], results['tpr'], alpha=0.15)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title(title); ax.legend(loc='lower right')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1]); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def plot_pr(results, path, title="PR Curve"):
    fig, ax = plt.subplots(figsize=(8, 8))
    baseline = results['labels'].mean()
    ax.plot(results['pr_recall'], results['pr_precision'], 'b-', lw=2.5, label=f'PR (AP={results["pr_auc"]:.4f})')
    ax.axhline(y=baseline, color='k', linestyle='--', lw=1, label=f'Baseline ({baseline:.3f})')
    ax.fill_between(results['pr_recall'], results['pr_precision'], alpha=0.15)
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title(title); ax.legend(loc='upper right')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1]); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def plot_score_dist(results, path, title="Score Distribution"):
    fig, ax = plt.subplots(figsize=(10, 6))
    neg = results['scores'][results['labels'] == 0]
    pos = results['scores'][results['labels'] == 1]
    ax.hist(neg, bins=50, alpha=0.6, label=f'Non-violent (n={len(neg):,})', color='blue', density=True)
    ax.hist(pos, bins=50, alpha=0.6, label=f'Violent (n={len(pos):,})', color='red', density=True)
    ax.axvline(x=results['threshold'], color='green', linestyle='--', lw=2, label=f'Threshold ({results["threshold"]:.3f})')
    ax.set_xlabel('Score'); ax.set_ylabel('Density'); ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def plot_timeline(results, path, title="Timeline", max_sec=3600):
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    max_idx = min(int(max_sec), len(results['timestamps']))
    t = results['timestamps'][:max_idx]
    axes[0].fill_between(t, 0, results['labels'][:max_idx], alpha=0.7, color='red', label='Violence')
    axes[0].set_ylabel('Ground Truth'); axes[0].legend(loc='upper right')
    axes[1].plot(t, results['scores'][:max_idx], 'b-', lw=0.5, alpha=0.8)
    axes[1].axhline(y=results['threshold'], color='green', linestyle='--', lw=2)
    axes[1].set_ylabel('Score'); axes[1].set_ylim([0, 1]); axes[1].legend(['Score', f'Threshold={results["threshold"]:.3f}'], loc='upper right')
    axes[2].fill_between(t, 0, results['predictions'][:max_idx], alpha=0.7, color='orange', label='Alert')
    axes[2].set_ylabel('Predictions'); axes[2].set_xlabel('Time (s)'); axes[2].legend(loc='upper right')
    fig.suptitle(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def plot_confusion(results, path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Pred -', 'Pred +'], yticklabels=['True -', 'True +'], annot_kws={'size': 14})
    ax.set_title(f'{title}\n(threshold={results["threshold"]:.3f})')
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def plot_lead_times(results, path, title="Lead Time Distribution"):
    fig, ax = plt.subplots(figsize=(10, 6))
    if results['lead_times']:
        ax.hist(results['lead_times'], bins=15, color='green', alpha=0.7, edgecolor='black')
        ax.axvline(x=results['avg_lead_time'], color='red', linestyle='--', lw=2, label=f'Avg: {results["avg_lead_time"]:.1f}s')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No detections', ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_xlabel('Lead Time (s)'); ax.set_ylabel('Count'); ax.set_title(title); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def plot_fa_distribution(results, path, title="FA Distribution"):
    fig, ax = plt.subplots(figsize=(12, 5))
    fa_times = _find_false_alarms(results['labels'], results['predictions'], results['timestamps'])
    if fa_times:
        ax.hist([t / 3600 for t in fa_times], bins=20, color='orange', alpha=0.7, edgecolor='black')
    else:
        ax.text(0.5, 0.5, 'No false alarms', ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('FA Count')
    ax.set_title(f'{title} (Total: {len(fa_times)}, Rate: {results["fa_per_hour"]:.1f}/hr)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def generate_all_plots(results, out_dir, prefix=""):
    """Generate full set of evaluation plots."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = f"{prefix}_" if prefix else ""
    plot_roc(results, out_dir / f'{p}roc_curve.png', f'{prefix} ROC Curve'.strip())
    plot_pr(results, out_dir / f'{p}pr_curve.png', f'{prefix} PR Curve'.strip())
    plot_score_dist(results, out_dir / f'{p}score_distribution.png', f'{prefix} Score Distribution'.strip())
    plot_timeline(results, out_dir / f'{p}timeline.png', f'{prefix} Timeline'.strip())
    plot_confusion(results, out_dir / f'{p}confusion_matrix.png', f'{prefix} Confusion Matrix'.strip())
    plot_lead_times(results, out_dir / f'{p}lead_times.png', f'{prefix} Lead Times'.strip())
    plot_fa_distribution(results, out_dir / f'{p}fa_distribution.png', f'{prefix} FA Distribution'.strip())
    print(f"  Plots saved to {out_dir}/")


def save_results_json(results, path):
    """Save numeric results to JSON (no arrays)."""
    out = {}
    for k, v in results.items():
        if isinstance(v, (int, float, str, bool)):
            out[k] = v
        elif isinstance(v, np.floating):
            out[k] = float(v)
        elif isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
            out[k] = v
        elif isinstance(v, np.ndarray) and v.ndim == 2:
            out[k] = v.tolist()
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)


# ==============================================================================
# EXPERIMENT 1: Person Detection Filtering
# ==============================================================================
def run_experiment_1(args, model, seq_len, device):
    """
    Experiment 1 runs as a standalone script: experiments/exp1_person_detection.py
    
    It requires YOLOv8 + raw video access, so it must be run separately.
    This function prints the run command.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Person Detection Filtering")
    print("=" * 70)
    print("\nExperiment 1 is a standalone script: experiments/exp1_person_detection.py")
    print("\nRun it with:")
    print(f"  python experiments/exp1_person_detection.py \\")
    print(f"    --model_path {args.model_path} \\")
    print(f"    --embeddings_dir {args.embeddings_dir} \\")
    print(f"    --labels_dir {args.labels_dir} \\")
    print(f"    --raw_videos_dir {args.raw_videos_dir} \\")
    print(f"    --test_index {args.test_index} \\")
    print(f"    --device {args.device} \\")
    print(f"    --output_dir experiments/output/exp1_person_detection")



# ==============================================================================
# EXPERIMENT 2: Non-Violent vs Violent Clip Analysis + Unified Eval
# ==============================================================================
def run_experiment_2(args, model, seq_len, device):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Non-Violent vs Violent Clip Analysis")
    print("=" * 70)

    out_dir = Path(args.output_dir) / 'exp2_clip_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load ALL test videos (both sources)
    print("\nLoading all test videos...")
    all_videos = load_video_data(args.test_index, args.embeddings_dir, args.labels_dir, source_filter=None)
    print(f"  Loaded {len(all_videos)} test videos")

    violent_videos = [v for v in all_videos if v['labels'].max() > 0]
    nonviolent_videos = [v for v in all_videos if v['labels'].max() == 0]
    print(f"  Violent: {len(violent_videos)}, Non-violent: {len(nonviolent_videos)}")

    # 2a: Score non-violent clips
    print("\n--- 2a: Scoring non-violent clips ---")
    nv_scores_all = []
    nv_per_video = []
    for v in tqdm(nonviolent_videos, desc="Non-violent"):
        scores = score_embeddings(model, v['embeddings'], seq_len, device)
        nv_scores_all.extend(scores)
        nv_per_video.append({
            'video_id': v['video_id'], 'source': v['source'],
            'mean_score': float(np.mean(scores)), 'max_score': float(np.max(scores)),
            'std_score': float(np.std(scores)), 'n_frames': len(scores),
        })

    nv_scores_all = np.array(nv_scores_all)
    print(f"  Non-violent frames: {len(nv_scores_all):,}")
    print(f"  Score stats: mean={nv_scores_all.mean():.4f}, std={nv_scores_all.std():.4f}, "
          f"max={nv_scores_all.max():.4f}, >0.5: {(nv_scores_all > 0.5).mean():.2%}")

    # 2b: Score violent clips
    print("\n--- 2b: Scoring violent clips ---")
    v_scores_all, v_labels_all = [], []
    v_per_video = []
    for v in tqdm(violent_videos, desc="Violent"):
        scores = score_embeddings(model, v['embeddings'], seq_len, device)
        v_scores_all.extend(scores)
        v_labels_all.extend(v['labels'])
        viol_frames = v['labels'] > 0
        v_per_video.append({
            'video_id': v['video_id'], 'source': v['source'],
            'mean_score': float(np.mean(scores)), 'max_score': float(np.max(scores)),
            'mean_score_during_violence': float(np.mean(scores[viol_frames])) if viol_frames.any() else 0,
            'mean_score_before_violence': float(np.mean(scores[~viol_frames])) if (~viol_frames).any() else 0,
            'n_frames': len(scores), 'n_violent_frames': int(viol_frames.sum()),
        })

    v_scores_all = np.array(v_scores_all)
    v_labels_all = np.array(v_labels_all)
    print(f"  Violent clip frames: {len(v_scores_all):,}")
    print(f"  Score during violence: mean={v_scores_all[v_labels_all > 0].mean():.4f}")
    print(f"  Score during non-violence: mean={v_scores_all[v_labels_all == 0].mean():.4f}")

    # 2c: Build unified evaluation dataset
    print("\n--- 2c: Unified Evaluation ---")
    sim_data = build_simulation(all_videos, duration_hours=8.0, seed=42)
    results = evaluate_simulation(model, sim_data, seq_len, device)

    print(f"\n  Unified Eval Results:")
    print(f"    ROC-AUC: {results['roc_auc']:.4f}")
    print(f"    PR-AUC:  {results['pr_auc']:.4f}")
    print(f"    Detection Rate: {results['detection_rate']:.1%}")
    print(f"    FA/Hour: {results['fa_per_hour']:.1f}")
    print(f"    Avg Lead Time: {results['avg_lead_time']:.1f}s")

    # Generate plots
    generate_all_plots(results, out_dir, prefix="unified")

    # Plot: non-violent vs violent score distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(nv_scores_all, bins=50, alpha=0.7, color='blue', density=True)
    axes[0].set_xlabel('Score'); axes[0].set_ylabel('Density')
    axes[0].set_title(f'Non-Violent Clips (n={len(nv_scores_all):,})\nmean={nv_scores_all.mean():.4f}')
    axes[0].axvline(x=results['threshold'], color='red', linestyle='--')

    axes[1].hist(v_scores_all[v_labels_all == 0], bins=50, alpha=0.6, color='blue', density=True, label='Non-viol frames')
    axes[1].hist(v_scores_all[v_labels_all > 0], bins=50, alpha=0.6, color='red', density=True, label='Violent frames')
    axes[1].set_xlabel('Score'); axes[1].set_title(f'Violent Clips (n={len(v_scores_all):,})')
    axes[1].axvline(x=results['threshold'], color='green', linestyle='--')
    axes[1].legend()

    # Per-source breakdown
    sources = sorted(set(v['source'] for v in all_videos))
    per_source_means = {}
    for src in sources:
        src_scores_nv = [v['mean_score'] for v in nv_per_video if v['source'] == src]
        src_scores_v = [v['mean_score'] for v in v_per_video if v['source'] == src]
        per_source_means[src] = {
            'nonviolent_mean': np.mean(src_scores_nv) if src_scores_nv else 0,
            'violent_mean': np.mean(src_scores_v) if src_scores_v else 0,
        }

    x = np.arange(len(sources))
    width = 0.35
    axes[2].bar(x - width/2, [per_source_means[s]['nonviolent_mean'] for s in sources], width,
                label='Non-violent', color='blue', alpha=0.7)
    axes[2].bar(x + width/2, [per_source_means[s]['violent_mean'] for s in sources], width,
                label='Violent', color='red', alpha=0.7)
    axes[2].set_xticks(x); axes[2].set_xticklabels(sources, rotation=15)
    axes[2].set_ylabel('Mean Score'); axes[2].set_title('Mean Score by Source & Violence')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(out_dir / 'exp2_clip_analysis.png', dpi=150)
    plt.close()

    # Save results
    save_results_json(results, out_dir / 'exp2_unified_metrics.json')
    with open(out_dir / 'exp2_per_video_nonviolent.json', 'w') as f:
        json.dump(nv_per_video, f, indent=2)
    with open(out_dir / 'exp2_per_video_violent.json', 'w') as f:
        json.dump(v_per_video, f, indent=2)

    print(f"\nResults saved to {out_dir}/")
    return results


# ==============================================================================
# EXPERIMENT 3: Cross-Dataset Training
# ==============================================================================
def run_experiment_3(args, device):
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Cross-Dataset Training")
    print("=" * 70)

    out_dir = Path(args.output_dir) / 'exp3_cross_dataset'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all videos split by source
    print("\nLoading all video data...")
    train_xd = load_video_data(args.train_index, args.embeddings_dir, args.labels_dir, source_filter='xd_violence')
    val_xd = load_video_data(args.val_index, args.embeddings_dir, args.labels_dir, source_filter='xd_violence')
    test_xd = load_video_data(args.test_index, args.embeddings_dir, args.labels_dir, source_filter='xd_violence')

    train_ucf = load_video_data(args.train_index, args.embeddings_dir, args.labels_dir, source_filter='ucf_crime')
    val_ucf = load_video_data(args.val_index, args.embeddings_dir, args.labels_dir, source_filter='ucf_crime')
    test_ucf = load_video_data(args.test_index, args.embeddings_dir, args.labels_dir, source_filter='ucf_crime')

    print(f"  XD-Violence: {len(train_xd)} train, {len(val_xd)} val, {len(test_xd)} test")
    print(f"  UCF-Crime:   {len(train_ucf)} train, {len(val_ucf)} val, {len(test_ucf)} test")

    # Base training args
    base_args = {
        'input_dim': 3584, 'hidden_dim': 256, 'lstm_hidden': 128,
        'num_lstm_layers': 2, 'dropout': 0.3, 'sequence_length': 30,
        'stride': 5, 'epochs': 100, 'patience': 15, 'batch_size': 64,
        'lr': 1e-4, 'horizon_sec': 30.0, 'fps': 1.0,
    }

    # Define training configurations
    configs = [
        ('xd_only', train_xd, val_xd),
        ('ucf_only', train_ucf, val_ucf),
        ('combined', train_xd + train_ucf, val_xd + val_ucf),
    ]

    all_results = {}

    for config_name, train_data, val_data in configs:
        print(f"\n{'='*50}")
        print(f"Training: {config_name}")
        print(f"  Train videos: {len(train_data)}, Val videos: {len(val_data)}")
        print(f"{'='*50}")

        model_dir = out_dir / config_name
        result = train_model(train_data, val_data, base_args, device, model_dir)

        if result is None:
            print(f"  FAILED to train {config_name}")
            all_results[config_name] = {'error': 'training_failed'}
            continue

        trained_model, history = result

        # Test on each dataset
        config_results = {'history': history}
        for test_name, test_data in [('xd_violence', test_xd), ('ucf_crime', test_ucf), ('both', test_xd + test_ucf)]:
            if not test_data:
                continue
            print(f"\n  Testing {config_name} on {test_name}...")
            sim = build_simulation(test_data, duration_hours=4.0, seed=42)
            res = evaluate_simulation(trained_model, sim, base_args['sequence_length'], device)
            config_results[f'test_{test_name}'] = {
                'roc_auc': res['roc_auc'], 'pr_auc': res['pr_auc'],
                'detection_rate': res['detection_rate'], 'fa_per_hour': res['fa_per_hour'],
                'avg_lead_time': res['avg_lead_time'], 'f1': res['f1'],
                'threshold': res['threshold'],
            }
            print(f"    ROC-AUC={res['roc_auc']:.4f}, PR-AUC={res['pr_auc']:.4f}, "
                  f"Det={res['detection_rate']:.1%}, FA/hr={res['fa_per_hour']:.1f}")

            # Generate plots for each test
            generate_all_plots(res, model_dir / f'test_{test_name}', prefix=f"{config_name}_on_{test_name}")

        all_results[config_name] = config_results

    # Summary comparison plot
    print("\n\nGenerating cross-dataset comparison...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    train_configs = [c[0] for c in configs]
    test_datasets = ['xd_violence', 'ucf_crime', 'both']
    colors = {'xd_violence': 'steelblue', 'ucf_crime': 'coral', 'both': 'green'}

    for metric_idx, metric in enumerate(['roc_auc', 'pr_auc', 'detection_rate', 'fa_per_hour']):
        ax = axes[metric_idx // 2][metric_idx % 2]
        x = np.arange(len(train_configs))
        width = 0.25

        for i, test_name in enumerate(test_datasets):
            vals = []
            for config_name in train_configs:
                r = all_results.get(config_name, {})
                test_r = r.get(f'test_{test_name}', {})
                vals.append(test_r.get(metric, 0))

            ax.bar(x + i * width, vals, width, label=f'Test: {test_name}',
                   color=colors[test_name], alpha=0.7)

        ax.set_xticks(x + width)
        ax.set_xticklabels(train_configs)
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} by Training Config')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Cross-Dataset Training Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'exp3_cross_dataset_comparison.png', dpi=150)
    plt.close()

    # Save all results
    with open(out_dir / 'exp3_all_results.json', 'w') as f:
        # Convert any numpy types
        def convert(obj):
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        json.dump(convert(all_results), f, indent=2)

    print(f"\nAll results saved to {out_dir}/")
    return all_results


# ==============================================================================
# EXPERIMENT 4: UCF-Crime Evaluation with Graphs
# ==============================================================================
def run_experiment_4(args, model, seq_len, device):
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: UCF-Crime Detection & Anticipation Evaluation")
    print("=" * 70)

    out_dir = Path(args.output_dir) / 'exp4_ucf_crime_eval'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load UCF-Crime test videos
    print("\nLoading UCF-Crime test videos...")
    ucf_test = load_video_data(args.test_index, args.embeddings_dir, args.labels_dir, source_filter='ucf_crime')
    print(f"  Loaded {len(ucf_test)} UCF-Crime test videos")

    if not ucf_test:
        print("  No UCF-Crime test videos found!")
        # Try loading from all indices
        print("  Trying all indices...")
        for idx_path in [args.train_index, args.val_index, args.test_index]:
            vids = load_video_data(idx_path, args.embeddings_dir, args.labels_dir, source_filter='ucf_crime')
            ucf_test.extend(vids)
        # Deduplicate
        seen = set()
        unique = []
        for v in ucf_test:
            if v['video_id'] not in seen:
                seen.add(v['video_id'])
                unique.append(v)
        ucf_test = unique
        print(f"  Found {len(ucf_test)} total UCF-Crime videos")

    if not ucf_test:
        print("  ERROR: No UCF-Crime videos available!")
        return

    violent = [v for v in ucf_test if v['labels'].max() > 0]
    nonviolent = [v for v in ucf_test if v['labels'].max() == 0]
    print(f"  Violent: {len(violent)}, Non-violent: {len(nonviolent)}")

    # Build UCF-Crime simulation
    print("\nBuilding UCF-Crime simulation...")
    sim_data = build_simulation(ucf_test, duration_hours=8.0, seed=42)
    print(f"  Simulation: {len(sim_data['embeddings']):,} frames, "
          f"{sim_data['anticipation_labels'].mean()*100:.2f}% positive")

    # Evaluate
    print("\nEvaluating model on UCF-Crime...")
    results = evaluate_simulation(model, sim_data, seq_len, device)

    print(f"\n{'='*50}")
    print(f"UCF-CRIME EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  ROC-AUC:        {results['roc_auc']:.4f}")
    print(f"  PR-AUC:         {results['pr_auc']:.4f}")
    print(f"  F1:             {results['f1']:.4f}")
    print(f"  Detection Rate: {results['detection_rate']:.1%}")
    print(f"  FA/Hour:        {results['fa_per_hour']:.1f}")
    print(f"  Avg Lead Time:  {results['avg_lead_time']:.1f}s")
    print(f"  Threshold:      {results['threshold']:.4f}")

    # Also evaluate on XD-Violence for comparison
    print("\nLoading XD-Violence test for comparison...")
    xd_test = load_video_data(args.test_index, args.embeddings_dir, args.labels_dir, source_filter='xd_violence')
    xd_results = None
    if xd_test:
        xd_sim = build_simulation(xd_test, duration_hours=8.0, seed=42)
        xd_results = evaluate_simulation(model, xd_sim, seq_len, device)
        print(f"\n  XD-Violence comparison:")
        print(f"    ROC-AUC: {xd_results['roc_auc']:.4f} vs UCF: {results['roc_auc']:.4f}")
        print(f"    PR-AUC:  {xd_results['pr_auc']:.4f} vs UCF: {results['pr_auc']:.4f}")

    # Generate all plots for UCF-Crime
    print("\nGenerating UCF-Crime evaluation plots...")
    generate_all_plots(results, out_dir, prefix="ucf_crime")

    # If we have XD-Violence comparison, create overlay plots
    if xd_results:
        # Overlay ROC
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        axes[0].plot(results['fpr'], results['tpr'], 'r-', lw=2.5, label=f'UCF-Crime (AUC={results["roc_auc"]:.4f})')
        axes[0].plot(xd_results['fpr'], xd_results['tpr'], 'b-', lw=2.5, label=f'XD-Violence (AUC={xd_results["roc_auc"]:.4f})')
        axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
        axes[0].set_title('ROC Comparison: UCF-Crime vs XD-Violence')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        # Overlay PR
        axes[1].plot(results['pr_recall'], results['pr_precision'], 'r-', lw=2.5, label=f'UCF-Crime (AP={results["pr_auc"]:.4f})')
        axes[1].plot(xd_results['pr_recall'], xd_results['pr_precision'], 'b-', lw=2.5, label=f'XD-Violence (AP={xd_results["pr_auc"]:.4f})')
        axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
        axes[1].set_title('PR Comparison: UCF-Crime vs XD-Violence')
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / 'exp4_ucf_vs_xd_comparison.png', dpi=150)
        plt.close()

        # Summary bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        metrics_to_compare = ['roc_auc', 'pr_auc', 'f1', 'detection_rate']
        x = np.arange(len(metrics_to_compare))
        width = 0.35
        ucf_vals = [results[m] for m in metrics_to_compare]
        xd_vals = [xd_results[m] for m in metrics_to_compare]

        ax.bar(x - width/2, xd_vals, width, label='XD-Violence (training domain)', color='steelblue', alpha=0.7)
        ax.bar(x + width/2, ucf_vals, width, label='UCF-Crime (cross-domain)', color='coral', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_compare])
        ax.set_ylabel('Score')
        ax.set_title('Model Performance: In-Domain vs Cross-Domain')
        ax.legend()
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, (xd_v, ucf_v) in enumerate(zip(xd_vals, ucf_vals)):
            ax.text(i - width/2, xd_v + 0.02, f'{xd_v:.3f}', ha='center', fontsize=9)
            ax.text(i + width/2, ucf_v + 0.02, f'{ucf_v:.3f}', ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(out_dir / 'exp4_domain_comparison.png', dpi=150)
        plt.close()

    # Save results
    save_results_json(results, out_dir / 'exp4_ucf_crime_metrics.json')
    if xd_results:
        save_results_json(xd_results, out_dir / 'exp4_xd_violence_metrics.json')

    print(f"\nAll plots and metrics saved to {out_dir}/")
    return results


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run Violence Anticipation Experiments")
    parser.add_argument('--embeddings_dir', type=str, default='data/processed/vlm_emb')
    parser.add_argument('--labels_dir', type=str, default='data/processed/labels_multih')
    parser.add_argument('--model_path', type=str, default='outputs/outputs_temporal/best_temporal_model.pt')
    parser.add_argument('--raw_videos_dir', type=str, default='data/raw_videos')
    parser.add_argument('--train_index', type=str, default='train_index.json')
    parser.add_argument('--val_index', type=str, default='val_index.json')
    parser.add_argument('--test_index', type=str, default='test_index.json')
    parser.add_argument('--output_dir', type=str, default='experiments/output')
    parser.add_argument('--device', type=str, default='cuda:1')

    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--exp1', action='store_true', help='Experiment 1: Person detection filtering')
    parser.add_argument('--exp2', action='store_true', help='Experiment 2: Clip analysis + unified eval')
    parser.add_argument('--exp3', action='store_true', help='Experiment 3: Cross-dataset training')
    parser.add_argument('--exp4', action='store_true', help='Experiment 4: UCF-Crime evaluation')

    args = parser.parse_args()

    if not (args.all or args.exp1 or args.exp2 or args.exp3 or args.exp4):
        parser.print_help()
        print("\nPlease specify --all or individual experiments (--exp1, --exp2, etc.)")
        return

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load base model (for exp 1, 2, 4)
    model, seq_len, model_args = None, 30, {}
    if args.all or args.exp1 or args.exp2 or args.exp4:
        print(f"\nLoading model from {args.model_path}...")
        model, seq_len, model_args = load_model(args.model_path, device)
        print(f"  Loaded (seq_len={seq_len}, params={sum(p.numel() for p in model.parameters()):,})")

    # Run experiments
    if args.all or args.exp1:
        run_experiment_1(args, model, seq_len, device)

    if args.all or args.exp2:
        run_experiment_2(args, model, seq_len, device)

    if args.all or args.exp3:
        run_experiment_3(args, device)

    if args.all or args.exp4:
        run_experiment_4(args, model, seq_len, device)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to: {args.output_dir}/")
    print("=" * 70)


if __name__ == '__main__':
    main()