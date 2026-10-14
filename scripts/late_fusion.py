#!/usr/bin/env python3
"""
Late Fusion for Violence Anticipation

Trains separate models for each modality (VLM, Audio, Pose) and combines
their predictions at inference time using various fusion strategies:
- max: Take maximum probability across modalities
- mean: Average probabilities
- weighted: Weighted average (learnable or fixed)
- learned: Train a small fusion network on top

This avoids the issue of one modality's noise drowning out another in early fusion.

Usage:
    # Step 1: Train individual models (if not already done)
    python scripts/late_fusion.py --mode train_individual \
        --vlm_dir data/processed/vlm_emb \
        --audio_dir data/processed/audio_features \
        --labels_dir data/processed/labels_multih \
        --output_dir outputs/late_fusion

    # Step 2: Evaluate with late fusion
    python scripts/late_fusion.py --mode evaluate \
        --vlm_model outputs/late_fusion/vlm_model/best_temporal_model.pt \
        --audio_model outputs/late_fusion/audio_model/best_temporal_model.pt \
        --vlm_sim simulation_data/xd_violence_8hr.npz \
        --audio_sim simulation_data/xd_violence_audio_only.npz \
        --fusion_strategy weighted \
        --output_dir final/late_fusion_eval

    # Step 3: Grid search for optimal weights
    python scripts/late_fusion.py --mode optimize_weights \
        --vlm_model outputs/late_fusion/vlm_model/best_temporal_model.pt \
        --audio_model outputs/late_fusion/audio_model/best_temporal_model.pt \
        --vlm_sim simulation_data/xd_violence_8hr.npz \
        --audio_sim simulation_data/xd_violence_audio_only.npz \
        --output_dir final/late_fusion_optimization
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn


# ==============================================================================
# Model Loading
# ==============================================================================

class TemporalLSTMWithEncoder(nn.Module):
    """
    Temporal LSTM model with frame encoder and attention.
    This matches the architecture used in train_temporal_model.py
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3, encoder_dim: int = 256):
        super().__init__()
        
        # Frame encoder to reduce dimensionality
        self.frame_encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.ReLU(),
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            encoder_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Encode each frame
        x = x.view(-1, x.shape[-1])
        x = self.frame_encoder(x)
        x = x.view(batch_size, seq_len, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        return self.classifier(context)


class SimpleLSTM(nn.Module):
    """Simple LSTM without frame encoder (for smaller input dims like audio)."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.classifier(last_hidden)


def load_model(model_path: Path, device: str = 'cuda:0') -> Tuple[nn.Module, dict]:
    """Load a trained model from checkpoint, auto-detecting architecture."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get model config
    config = checkpoint.get('config', {})
    state_dict = checkpoint['model_state_dict']
    
    # Detect architecture from state dict keys
    has_frame_encoder = any('frame_encoder' in k for k in state_dict.keys())
    
    # Get dimensions from state dict
    if has_frame_encoder:
        input_dim = state_dict['frame_encoder.0.weight'].shape[1]
        encoder_dim = state_dict['frame_encoder.0.weight'].shape[0]
        hidden_dim = state_dict['lstm.weight_hh_l0'].shape[1]
    else:
        input_dim = state_dict['lstm.weight_ih_l0'].shape[1]
        encoder_dim = 256  # default
        hidden_dim = state_dict['lstm.weight_hh_l0'].shape[1]
    
    # Count LSTM layers
    num_layers = sum(1 for k in state_dict.keys() if 'lstm.weight_ih_l' in k and 'reverse' not in k)
    
    config['input_dim'] = input_dim
    config['hidden_dim'] = hidden_dim
    config['num_layers'] = num_layers
    config['has_frame_encoder'] = has_frame_encoder
    config['encoder_dim'] = encoder_dim
    
    print(f"  Detected architecture: {'TemporalLSTMWithEncoder' if has_frame_encoder else 'SimpleLSTM'}")
    print(f"  Input dim: {input_dim}, Encoder dim: {encoder_dim}, Hidden dim: {hidden_dim}, Layers: {num_layers}")
    
    # Create model with correct architecture
    if has_frame_encoder:
        model = TemporalLSTMWithEncoder(input_dim, hidden_dim, num_layers, encoder_dim=encoder_dim)
    else:
        model = SimpleLSTM(input_dim, hidden_dim, num_layers)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, config


def load_simulation(sim_path: Path) -> Dict:
    """Load a prebuilt simulation file."""
    data = np.load(sim_path, allow_pickle=True)
    
    # Print available keys for debugging
    print(f"  Simulation keys: {list(data.keys())}")
    
    # Handle different key names
    embeddings = data.get('embeddings', data.get('emb', data.get('arr_0', None)))
    frame_labels = data.get('frame_labels', data.get('labels', None))
    anticipation_labels = data.get('anticipation_labels', data.get('ant_labels', None))
    video_boundaries = data.get('video_boundaries', data.get('boundaries', None))
    video_names = data.get('video_names', None)
    fps = float(data.get('fps', 1.0))
    
    # If no anticipation labels, create them from frame labels
    if anticipation_labels is None and frame_labels is not None:
        print("  Creating anticipation labels from frame labels...")
        horizon_frames = int(30 * fps)  # 30 second horizon
        anticipation_labels = np.zeros_like(frame_labels)
        for t in range(len(frame_labels)):
            future_end = min(t + horizon_frames, len(frame_labels))
            anticipation_labels[t] = frame_labels[t:future_end].max()
    
    # If no video boundaries, create a single boundary for the whole simulation
    if video_boundaries is None:
        print("  No video boundaries found, treating as single video...")
        video_boundaries = np.array([[0, len(embeddings)]])
    
    return {
        'embeddings': embeddings,
        'frame_labels': frame_labels,
        'anticipation_labels': anticipation_labels,
        'video_boundaries': video_boundaries,
        'video_names': video_names,
        'fps': fps,
    }


# ==============================================================================
# Inference
# ==============================================================================

def run_inference(
    model: nn.Module,
    embeddings: np.ndarray,
    sequence_length: int = 30,
    device: str = 'cuda:0',
    batch_size: int = 256,
) -> np.ndarray:
    """Run inference on embeddings and return frame-level predictions."""
    model.eval()
    n_frames = len(embeddings)
    predictions = np.zeros(n_frames, dtype=np.float32)
    
    # Pad beginning
    pad_size = sequence_length - 1
    padded = np.vstack([
        np.zeros((pad_size, embeddings.shape[1]), dtype=np.float32),
        embeddings
    ])
    
    # Create sequences
    sequences = []
    for i in range(n_frames):
        seq = padded[i:i + sequence_length]
        sequences.append(seq)
    
    sequences = np.array(sequences)
    
    # Batch inference
    with torch.no_grad():
        for i in range(0, n_frames, batch_size):
            batch = torch.tensor(sequences[i:i + batch_size], dtype=torch.float32, device=device)
            preds = model(batch).squeeze(-1).cpu().numpy()
            predictions[i:i + len(preds)] = preds
    
    return predictions


# ==============================================================================
# Late Fusion Strategies
# ==============================================================================

def fuse_predictions(
    predictions: Dict[str, np.ndarray],
    strategy: str = 'max',
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Fuse predictions from multiple modalities.
    
    Args:
        predictions: Dict mapping modality name to prediction array
        strategy: Fusion strategy ('max', 'mean', 'weighted', 'product')
        weights: Optional weights for weighted fusion
        
    Returns:
        Fused predictions array
    """
    # Check lengths and align if needed
    lengths = {k: len(v) for k, v in predictions.items()}
    print(f"  Prediction lengths: {lengths}")
    
    if len(set(lengths.values())) > 1:
        # Different lengths - truncate to shortest
        min_len = min(lengths.values())
        print(f"  Truncating to shortest length: {min_len}")
        predictions = {k: v[:min_len] for k, v in predictions.items()}
    
    pred_arrays = [predictions[k] for k in sorted(predictions.keys())]
    pred_stack = np.stack(pred_arrays, axis=0)  # Shape: [n_modalities, n_frames]
    
    if strategy == 'max':
        return np.max(pred_stack, axis=0)
    
    elif strategy == 'mean':
        return np.mean(pred_stack, axis=0)
    
    elif strategy == 'weighted':
        if weights is None:
            weights = {k: 1.0 / len(predictions) for k in predictions.keys()}
        
        fused = np.zeros(pred_stack.shape[1])
        total_weight = sum(weights.values())
        
        for i, name in enumerate(sorted(predictions.keys())):
            w = weights.get(name, 1.0) / total_weight
            fused += w * pred_stack[i]
        
        return fused
    
    elif strategy == 'product':
        # Geometric mean - good for independent classifiers
        return np.power(np.prod(pred_stack, axis=0), 1.0 / len(pred_arrays))
    
    elif strategy == 'noisy_or':
        # P(violence) = 1 - product(1 - P_i)
        return 1 - np.prod(1 - pred_stack, axis=0)
    
    else:
        raise ValueError(f"Unknown fusion strategy: {strategy}")


# ==============================================================================
# Evaluation
# ==============================================================================

def resample_to_timeline(
    values: np.ndarray,
    src_fps: float,
    dst_len: int,
    dst_fps: float,
) -> np.ndarray:
    """
    Resample a 1D prediction array from src_fps to dst_fps with length dst_len,
    using time-based interpolation.
    """
    values = np.asarray(values, dtype=np.float32).reshape(-1)

    if dst_len <= 0:
        return np.asarray([], dtype=np.float32)

    # Fast path
    if len(values) == dst_len and abs(float(src_fps) - float(dst_fps)) < 1e-9:
        return values

    # If fps is invalid or empty stream, fall back to truncate/pad
    if float(src_fps) <= 0 or float(dst_fps) <= 0 or len(values) == 0:
        out = np.zeros(dst_len, dtype=np.float32)
        n = min(dst_len, len(values))
        if n > 0:
            out[:n] = values[:n]
            if n < dst_len:
                out[n:] = values[-1]
        return out

    src_fps = float(src_fps)
    dst_fps = float(dst_fps)

    # time axis (seconds)
    src_t = np.arange(len(values), dtype=np.float64) / src_fps
    dst_t = np.arange(dst_len, dtype=np.float64) / dst_fps

    # Only resample over overlap, clamp to last available src time
    last_t = float(src_t[-1]) if len(src_t) > 0 else 0.0
    if last_t <= 0.0:
        # degenerate stream (length 1)
        return np.full(dst_len, float(values[-1]) if len(values) else 0.0, dtype=np.float32)

    dst_t = np.clip(dst_t, 0.0, last_t)

    return np.interp(dst_t, src_t, values).astype(np.float32)


def convert_boundaries_to_ref(
    boundaries: Optional[np.ndarray],
    src_fps: float,
    dst_fps: float,
    dst_len: int,
) -> np.ndarray:
    """
    Convert [start,end] boundaries expressed in src index units (src_fps)
    into dst index units (dst_fps), clipped to [0,dst_len].
    """
    if dst_len <= 0:
        return np.array([[0, 0]], dtype=np.int64)

    if boundaries is None:
        return np.array([[0, dst_len]], dtype=np.int64)

    b = np.asarray(boundaries)
    if b.ndim != 2 or b.shape[1] != 2:
        return np.array([[0, dst_len]], dtype=np.int64)

    src_fps = float(src_fps) if src_fps is not None else 0.0
    dst_fps = float(dst_fps) if dst_fps is not None else 0.0

    out = []
    for s, e in b:
        s = float(s)
        e = float(e)

        # Convert indices -> seconds
        if src_fps > 0:
            s_t = s / src_fps
            e_t = e / src_fps
        else:
            # If fps unknown, treat as already in "seconds-ish" and map proportionally
            s_t = s
            e_t = e

        # seconds -> dst indices
        if dst_fps > 0:
            s_ref = int(round(s_t * dst_fps))
            e_ref = int(round(e_t * dst_fps))
        else:
            s_ref = int(round(s_t))
            e_ref = int(round(e_t))

        if s_ref < dst_len and e_ref > 0:
            out.append([max(0, s_ref), min(dst_len, e_ref)])

    if not out:
        return np.array([[0, dst_len]], dtype=np.int64)

    cleaned = []
    for s, e in out:
        if e > s:
            cleaned.append([s, e])

    return np.array(cleaned if cleaned else [[0, dst_len]], dtype=np.int64)

def evaluate_predictions(
    predictions: np.ndarray,
    frame_labels: np.ndarray,
    anticipation_labels: np.ndarray,
    video_boundaries: np.ndarray,
    threshold: float = 0.5,
    fps: float = 1.0,
    horizon_sec: float = 30.0,
) -> Dict:
    """Evaluate predictions with operational metrics (robust to minor length mismatch)."""
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

    predictions = np.asarray(predictions, dtype=np.float32).reshape(-1)
    frame_labels = np.asarray(frame_labels).reshape(-1)
    anticipation_labels = np.asarray(anticipation_labels).reshape(-1)

    # Hard safety: align lengths
    min_len = min(len(predictions), len(frame_labels), len(anticipation_labels))
    if min_len <= 0:
        return {
            'roc_auc': 0.5,
            'pr_auc': 0.0,
            'detection_rate': 0.0,
            'fa_per_hour': 0.0,
            'avg_lead_time_sec': 0.0,
            'n_events': 0,
            'n_detected': 0,
            'n_false_alarms': 0,
            'total_hours': 0.0,
            'threshold': threshold,
        }

    if len(predictions) != min_len:
        predictions = predictions[:min_len]
    if len(frame_labels) != min_len:
        frame_labels = frame_labels[:min_len]
    if len(anticipation_labels) != min_len:
        anticipation_labels = anticipation_labels[:min_len]

    fps = float(fps) if fps is not None else 1.0
    if fps <= 0:
        fps = 1.0

    # Clip/normalize boundaries to current length
    video_boundaries = convert_boundaries_to_ref(video_boundaries, fps, fps, min_len)

    # Classification metrics
    try:
        roc_auc = roc_auc_score(anticipation_labels, predictions)
    except Exception:
        roc_auc = 0.5

    try:
        precision, recall, _ = precision_recall_curve(anticipation_labels, predictions)
        pr_auc = auc(recall, precision)
    except Exception:
        pr_auc = 0.0

    # Operational metrics
    total_duration = (len(predictions) / fps) / 3600.0  # hours

    # Find violence events (segments) using frame_labels within each video boundary
    events = []
    for start, end in video_boundaries:
        start = int(max(0, start))
        end = int(min(min_len, end))
        if end <= start:
            continue

        video_labels = frame_labels[start:end]

        in_event = False
        event_start = 0

        for t in range(len(video_labels)):
            if video_labels[t] > 0 and not in_event:
                in_event = True
                event_start = t
            elif video_labels[t] == 0 and in_event:
                events.append({
                    'global_start': start + event_start,
                    'global_end': start + t,
                })
                in_event = False

        if in_event:
            events.append({
                'global_start': start + event_start,
                'global_end': end,
            })

    # Check detections within anticipation window
    horizon_frames = int(float(horizon_sec) * fps)
    detected = 0
    lead_times = []

    for ev in events:
        anticipation_start = max(0, ev['global_start'] - horizon_frames)
        anticipation_end = min(min_len, ev['global_end'])

        window_preds = predictions[anticipation_start:anticipation_end]
        if window_preds.size == 0:
            continue

        hits = np.where(window_preds >= threshold)[0]
        if hits.size > 0:
            detected += 1
            first_det_idx = int(hits[0])
            first_det_global = anticipation_start + first_det_idx
            lead_time_frames = max(0, ev['global_start'] - first_det_global)
            lead_times.append(lead_time_frames / fps)

    # False alarms: rising edges in non-anticipation regions
    false_alarms = 0
    alert_active = False

    for t in range(min_len):
        if predictions[t] >= threshold:
            if (not alert_active) and (anticipation_labels[t] == 0):
                false_alarms += 1
            alert_active = True
        else:
            alert_active = False

    detection_rate = detected / len(events) if events else 0.0
    fa_per_hour = false_alarms / total_duration if total_duration > 0 else 0.0
    avg_lead_time = float(np.mean(lead_times)) if lead_times else 0.0

    return {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'detection_rate': float(detection_rate),
        'fa_per_hour': float(fa_per_hour),
        'avg_lead_time_sec': float(avg_lead_time),
        'n_events': int(len(events)),
        'n_detected': int(detected),
        'n_false_alarms': int(false_alarms),
        'total_hours': float(total_duration),
        'threshold': float(threshold),
    }


def threshold_sweep(
    predictions: np.ndarray,
    frame_labels: np.ndarray,
    anticipation_labels: np.ndarray,
    video_boundaries: np.ndarray,
    fps: float = 1.0,
    thresholds: List[float] = None,
) -> List[Dict]:
    """Sweep thresholds and return metrics for each."""
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = []
    for thresh in thresholds:
        metrics = evaluate_predictions(
            predictions, frame_labels, anticipation_labels,
            video_boundaries, thresh, fps
        )
        results.append(metrics)
    
    return results


# ==============================================================================
# Weight Optimization
# ==============================================================================

def optimize_weights(
    predictions: Dict[str, np.ndarray],
    frame_labels: np.ndarray,
    anticipation_labels: np.ndarray,
    video_boundaries: np.ndarray,
    fps: float = 1.0,
    target_metric: str = 'detection_rate',
    constraint_metric: str = 'fa_per_hour',
    constraint_max: float = 10.0,
) -> Tuple[Dict[str, float], Dict]:
    """
    Grid search for optimal fusion weights.
    
    Args:
        predictions: Dict of modality predictions
        target_metric: Metric to maximize
        constraint_metric: Metric to constrain
        constraint_max: Maximum value for constraint
        
    Returns:
        best_weights, best_metrics
    """
    modalities = list(predictions.keys())
    
    if len(modalities) == 2:
        # Grid search for 2 modalities
        best_score = -float('inf')
        best_weights = None
        best_metrics = None
        best_threshold = 0.5
        
        for w1 in np.arange(0.0, 1.05, 0.1):
            w2 = 1.0 - w1
            weights = {modalities[0]: w1, modalities[1]: w2}
            
            fused = fuse_predictions(predictions, 'weighted', weights)
            
            # Try multiple thresholds
            for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
                metrics = evaluate_predictions(
                    fused, frame_labels, anticipation_labels,
                    video_boundaries, thresh, fps
                )
                
                # Check constraint
                if metrics[constraint_metric] > constraint_max:
                    continue
                
                score = metrics[target_metric]
                if score > best_score:
                    best_score = score
                    best_weights = weights.copy()
                    best_metrics = metrics.copy()
                    best_threshold = thresh
        
        if best_weights:
            best_metrics['optimal_threshold'] = best_threshold
        
        return best_weights, best_metrics
    
    else:
        # For more modalities, use random search or Bayesian optimization
        # For now, just use equal weights
        weights = {m: 1.0 / len(modalities) for m in modalities}
        fused = fuse_predictions(predictions, 'weighted', weights)
        metrics = evaluate_predictions(
            fused, frame_labels, anticipation_labels,
            video_boundaries, 0.5, fps
        )
        return weights, metrics


# ==============================================================================
# Main Functions
# ==============================================================================

def train_individual_models(
    vlm_dir: Optional[Path],
    audio_dir: Optional[Path],
    pose_dir: Optional[Path],
    labels_dir: Path,
    output_dir: Path,
    train_index: str,
    val_index: str,
    test_index: str,
    source_filter: str,
    device: str = 'cuda:0',
):
    """Train individual models for each modality."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Feature dimensions
    dims = {'vlm': 3584, 'audio': 91, 'pose': 540}
    
    modalities = []
    if vlm_dir:
        modalities.append(('vlm', vlm_dir))
    if audio_dir:
        modalities.append(('audio', audio_dir))
    if pose_dir:
        modalities.append(('pose', pose_dir))
    
    for name, emb_dir in modalities:
        print(f"\n{'='*60}")
        print(f"Training {name.upper()} model")
        print('='*60)
        
        model_dir = output_dir / f'{name}_model'
        
        cmd = [
            'python', 'training/train_temporal_model.py',
            '--embeddings_dir', str(emb_dir),
            '--labels_dir', str(labels_dir),
            '--train_index', train_index,
            '--val_index', val_index,
            '--test_index', test_index,
            '--source_filter', source_filter,
            '--input_dim', str(dims[name]),
            '--sequence_length', '30',
            '--epochs', '100',
            '--patience', '15',
            '--batch_size', '64',
            '--lr', '1e-4',
            '--output_dir', str(model_dir),
            '--device', device,
        ]
        
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


def evaluate_late_fusion(
    vlm_model_path: Optional[Path],
    audio_model_path: Optional[Path],
    pose_model_path: Optional[Path],
    vlm_sim_path: Optional[Path],
    audio_sim_path: Optional[Path],
    pose_sim_path: Optional[Path],
    fusion_strategy: str,
    weights: Optional[Dict[str, float]],
    output_dir: Path,
    device: str = 'cuda:0',
):
    """Evaluate late fusion of multiple models (time-align modalities before eval)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions: Dict[str, np.ndarray] = {}
    per_mod_fps: Dict[str, float] = {}
    per_mod_boundaries: Dict[str, Optional[np.ndarray]] = {}

    frame_labels = None
    anticipation_labels = None
    video_boundaries = None
    fps = 1.0
    label_source = None

    modalities = [
        ('vlm', vlm_model_path, vlm_sim_path),
        ('audio', audio_model_path, audio_sim_path),
        ('pose', pose_model_path, pose_sim_path),
    ]

    for name, model_path, sim_path in modalities:
        if model_path is None or sim_path is None:
            continue

        if not Path(model_path).exists() or not Path(sim_path).exists():
            print(f"Skipping {name}: model or simulation not found")
            continue

        print(f"Loading {name} model and running inference...")

        model, config = load_model(model_path, device)
        sim_data = load_simulation(sim_path)

        seq_len = int(config.get('sequence_length', 30))
        preds = run_inference(
            model,
            sim_data['embeddings'],
            sequence_length=seq_len,
            device=device,
        )

        predictions[name] = preds
        per_mod_fps[name] = float(sim_data.get('fps', 1.0))
        per_mod_boundaries[name] = sim_data.get('video_boundaries', None)

        # Pick labels from the first modality that actually has them
        if frame_labels is None and sim_data.get('frame_labels', None) is not None:
            frame_labels = sim_data['frame_labels']
            anticipation_labels = sim_data['anticipation_labels']
            video_boundaries = sim_data['video_boundaries']
            fps = float(sim_data.get('fps', 1.0))
            label_source = name

    if len(predictions) < 2:
        print("Error: Need at least 2 modalities for late fusion")
        return

    if frame_labels is None or anticipation_labels is None:
        print("Error: Missing labels in simulation files (frame_labels / anticipation_labels).")
        return

    frame_labels = np.asarray(frame_labels).reshape(-1)
    anticipation_labels = np.asarray(anticipation_labels).reshape(-1)

    ref_fps = float(fps) if fps is not None else 1.0
    if ref_fps <= 0:
        ref_fps = 1.0

    ref_len = int(len(frame_labels))

    # Use overlap-in-time across modalities so we don't invent extra time at the end
    end_times = [ref_len / max(ref_fps, 1e-9)]
    for m in predictions.keys():
        m_fps = float(per_mod_fps.get(m, ref_fps))
        if m_fps <= 0:
            m_fps = ref_fps
        end_times.append(len(predictions[m]) / max(m_fps, 1e-9))

    common_end = float(min(end_times))
    ref_len = max(1, int(np.floor(common_end * ref_fps)))

    # Truncate labels to reference length
    frame_labels = frame_labels[:ref_len]
    anticipation_labels = anticipation_labels[:ref_len]

    # Choose best boundaries:
    # - If the label source only had a dummy single boundary, try to steal real boundaries from another modality (audio usually).
    vb = video_boundaries
    vb_is_dummy = (
        vb is None or
        (np.asarray(vb).ndim == 2 and len(vb) == 1 and int(vb[0][0]) == 0 and int(vb[0][1]) >= int(len(frame_labels)))
    )

    if vb_is_dummy:
        for m in predictions.keys():
            b = per_mod_boundaries.get(m)
            if b is not None and np.asarray(b).ndim == 2 and len(b) > 1:
                vb = convert_boundaries_to_ref(b, float(per_mod_fps[m]), ref_fps, ref_len)
                break

    video_boundaries = convert_boundaries_to_ref(vb, ref_fps, ref_fps, ref_len)

    # Align every modality's predictions onto the reference timeline
    for m in list(predictions.keys()):
        m_fps = float(per_mod_fps.get(m, ref_fps))
        if m_fps <= 0:
            m_fps = ref_fps
        predictions[m] = resample_to_timeline(predictions[m], m_fps, ref_len, ref_fps)

    fps = ref_fps  # use aligned fps going forward

    print(f"\nFusing {list(predictions.keys())} with strategy: {fusion_strategy}")

    # Fuse predictions (they should now all match ref_len)
    fused = fuse_predictions(predictions, fusion_strategy, weights)

    # Evaluate
    results = {'individual': {}, 'fused': {}, 'fusion_strategy': fusion_strategy, 'weights': weights}

    for name, preds in predictions.items():
        print(f"\n{name.upper()} individual results:")
        sweep = threshold_sweep(preds, frame_labels, anticipation_labels, video_boundaries, fps)

        best = max(sweep, key=lambda x: x['detection_rate'] - 0.1 * x['fa_per_hour'])
        results['individual'][name] = best

        print(f"  Detection rate: {best['detection_rate']*100:.1f}%")
        print(f"  FA/hour: {best['fa_per_hour']:.1f}")
        print(f"  Lead time: {best['avg_lead_time_sec']:.1f}s")

    print(f"\nFUSED ({fusion_strategy}) results:")
    sweep = threshold_sweep(fused, frame_labels, anticipation_labels, video_boundaries, fps)
    best = max(sweep, key=lambda x: x['detection_rate'] - 0.1 * x['fa_per_hour'])

    results['fused'] = best
    results['threshold_sweep'] = sweep
    results['label_source'] = label_source
    results['aligned_ref_fps'] = float(fps)
    results['aligned_ref_len'] = int(ref_len)

    print(f"  Detection rate: {best['detection_rate']*100:.1f}%")
    print(f"  FA/hour: {best['fa_per_hour']:.1f}")
    print(f"  Lead time: {best['avg_lead_time_sec']:.1f}s")
    print(f"  ROC-AUC: {best['roc_auc']:.4f}")

    # Save results
    with open(output_dir / 'late_fusion_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save predictions (aligned)
    np.savez_compressed(
        output_dir / 'predictions.npz',
        **{f'{k}_preds': v for k, v in predictions.items()},
        fused_preds=fused,
        frame_labels=frame_labels,
        anticipation_labels=anticipation_labels,
        video_boundaries=video_boundaries,
        fps=fps,
    )

    print(f"\nResults saved to {output_dir}")
    return results


def run_weight_optimization(
    vlm_model_path: Path,
    audio_model_path: Path,
    vlm_sim_path: Path,
    audio_sim_path: Path,
    output_dir: Path,
    device: str = 'cuda:0',
):
    """Optimize fusion weights (time-align audio to VLM timeline before searching)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions: Dict[str, np.ndarray] = {}

    # --- Load VLM (reference labels/timeline) ---
    print("Loading VLM model...")
    vlm_model, vlm_config = load_model(vlm_model_path, device)
    vlm_sim = load_simulation(vlm_sim_path)

    vlm_seq = int(vlm_config.get('sequence_length', 30))
    vlm_preds = run_inference(vlm_model, vlm_sim['embeddings'], sequence_length=vlm_seq, device=device)

    frame_labels = np.asarray(vlm_sim['frame_labels']).reshape(-1)
    anticipation_labels = np.asarray(vlm_sim['anticipation_labels']).reshape(-1)
    video_boundaries = vlm_sim.get('video_boundaries', None)
    ref_fps = float(vlm_sim.get('fps', 1.0))
    if ref_fps <= 0:
        ref_fps = 1.0

    # --- Load Audio ---
    print("Loading Audio model...")
    audio_model, audio_config = load_model(audio_model_path, device)
    audio_sim = load_simulation(audio_sim_path)

    audio_seq = int(audio_config.get('sequence_length', 30))
    audio_preds = run_inference(audio_model, audio_sim['embeddings'], sequence_length=audio_seq, device=device)

    audio_fps = float(audio_sim.get('fps', ref_fps))
    if audio_fps <= 0:
        audio_fps = ref_fps

    # --- Overlap-in-time alignment ---
    ref_len = int(len(frame_labels))
    end_ref = ref_len / max(ref_fps, 1e-9)
    end_audio = len(audio_preds) / max(audio_fps, 1e-9)
    common_end = float(min(end_ref, end_audio))
    ref_len = max(1, int(np.floor(common_end * ref_fps)))

    frame_labels = frame_labels[:ref_len]
    anticipation_labels = anticipation_labels[:ref_len]

    # Prefer audio boundaries if they exist and are more informative
    audio_boundaries = audio_sim.get('video_boundaries', None)
    if audio_boundaries is not None and np.asarray(audio_boundaries).ndim == 2 and len(audio_boundaries) > 1:
        video_boundaries = convert_boundaries_to_ref(audio_boundaries, audio_fps, ref_fps, ref_len)
    else:
        video_boundaries = convert_boundaries_to_ref(video_boundaries, ref_fps, ref_fps, ref_len)

    # Align predictions
    predictions['vlm'] = np.asarray(vlm_preds, dtype=np.float32).reshape(-1)[:ref_len]
    predictions['audio'] = resample_to_timeline(audio_preds, audio_fps, ref_len, ref_fps)

    fps = ref_fps

    print("\nOptimizing weights...")

    results = []
    for max_fa in [5.0, 10.0, 15.0, 20.0]:
        print(f"\n--- Constraint: FA/hour <= {max_fa} ---")

        best_weights, best_metrics = optimize_weights(
            predictions,
            frame_labels,
            anticipation_labels,
            video_boundaries,
            fps,
            target_metric='detection_rate',
            constraint_metric='fa_per_hour',
            constraint_max=max_fa,
        )

        if best_weights:
            print(f"Best weights: VLM={best_weights['vlm']:.2f}, Audio={best_weights['audio']:.2f}")
            print(f"Detection rate: {best_metrics['detection_rate']*100:.1f}%")
            print(f"FA/hour: {best_metrics['fa_per_hour']:.1f}")
            print(f"Threshold: {best_metrics.get('optimal_threshold', 0.5)}")

            results.append({
                'constraint_fa_max': max_fa,
                'weights': best_weights,
                'metrics': best_metrics,
            })

    print("\n--- Comparing fusion strategies ---")
    for strategy in ['max', 'mean', 'noisy_or', 'product']:
        fused = fuse_predictions(predictions, strategy)
        metrics = evaluate_predictions(fused, frame_labels, anticipation_labels, video_boundaries, 0.5, fps)
        print(f"{strategy}: Det={metrics['detection_rate']*100:.1f}%, FA={metrics['fa_per_hour']:.1f}/hr")

    with open(output_dir / 'weight_optimization.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Late fusion for violence anticipation")
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train_individual', 'evaluate', 'optimize_weights'])
    
    # Directories
    parser.add_argument('--vlm_dir', type=str, default=None)
    parser.add_argument('--audio_dir', type=str, default=None)
    parser.add_argument('--pose_dir', type=str, default=None)
    parser.add_argument('--labels_dir', type=str, default='data/processed/labels_multih')
    parser.add_argument('--output_dir', type=str, default='outputs/late_fusion')
    
    # Model paths
    parser.add_argument('--vlm_model', type=str, default=None)
    parser.add_argument('--audio_model', type=str, default=None)
    parser.add_argument('--pose_model', type=str, default=None)
    
    # Simulation paths
    parser.add_argument('--vlm_sim', type=str, default=None)
    parser.add_argument('--audio_sim', type=str, default=None)
    parser.add_argument('--pose_sim', type=str, default=None)
    
    # Training params
    parser.add_argument('--train_index', type=str, default='train_index.json')
    parser.add_argument('--val_index', type=str, default='val_index.json')
    parser.add_argument('--test_index', type=str, default='test_index.json')
    parser.add_argument('--source_filter', type=str, default='xd_violence')
    
    # Fusion params
    parser.add_argument('--fusion_strategy', type=str, default='max',
                        choices=['max', 'mean', 'weighted', 'product', 'noisy_or'])
    parser.add_argument('--vlm_weight', type=float, default=0.7)
    parser.add_argument('--audio_weight', type=float, default=0.3)
    
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()
    
    if args.mode == 'train_individual':
        train_individual_models(
            vlm_dir=Path(args.vlm_dir) if args.vlm_dir else None,
            audio_dir=Path(args.audio_dir) if args.audio_dir else None,
            pose_dir=Path(args.pose_dir) if args.pose_dir else None,
            labels_dir=Path(args.labels_dir),
            output_dir=Path(args.output_dir),
            train_index=args.train_index,
            val_index=args.val_index,
            test_index=args.test_index,
            source_filter=args.source_filter,
            device=args.device,
        )
    
    elif args.mode == 'evaluate':
        weights = None
        if args.fusion_strategy == 'weighted':
            weights = {'vlm': args.vlm_weight, 'audio': args.audio_weight}
        
        evaluate_late_fusion(
            vlm_model_path=Path(args.vlm_model) if args.vlm_model else None,
            audio_model_path=Path(args.audio_model) if args.audio_model else None,
            pose_model_path=Path(args.pose_model) if args.pose_model else None,
            vlm_sim_path=Path(args.vlm_sim) if args.vlm_sim else None,
            audio_sim_path=Path(args.audio_sim) if args.audio_sim else None,
            pose_sim_path=Path(args.pose_sim) if args.pose_sim else None,
            fusion_strategy=args.fusion_strategy,
            weights=weights,
            output_dir=Path(args.output_dir),
            device=args.device,
        )
    
    elif args.mode == 'optimize_weights':
        run_weight_optimization(
            vlm_model_path=Path(args.vlm_model),
            audio_model_path=Path(args.audio_model),
            vlm_sim_path=Path(args.vlm_sim),
            audio_sim_path=Path(args.audio_sim),
            output_dir=Path(args.output_dir),
            device=args.device,
        )


if __name__ == '__main__':
    main()