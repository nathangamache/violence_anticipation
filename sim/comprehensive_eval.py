#!/usr/bin/env python3
"""
Comprehensive Evaluation & Visualization for Violence Anticipation Models

Supports multiple model architectures:
1. TemporalViolenceModel (LSTM-based temporal model from train_temporal_model.py)
2. VLMStudentModel (improved student model from train_student.py)
3. StudentMLP (original student model from train_student_original.py)

Usage:
    # Temporal LSTM model (auto-detected)
    python comprehensive_eval.py \
        --model_path outputs_temporal/best_temporal_model.pt \
        --prebuilt_sim simulation_data/xd_violence_8hr_sim.npz \
        --threshold 0.388 \
        --output_dir ./eval_temporal

    # VLMStudentModel 
    python comprehensive_eval.py \
        --model_path runs/outputs_xd/best_model.pt \
        --prebuilt_sim simulation_data/xd_violence_8hr_sim.npz \
        --threshold 0.6 \
        --model_type vlm_student \
        --output_dir ./eval_student

    # StudentMLP (original)
    python comprehensive_eval.py \
        --model_path outputs_original/best.pt \
        --prebuilt_sim simulation_data/xd_violence_8hr_sim.npz \
        --threshold 0.5 \
        --model_type student_mlp \
        --output_dir ./eval_mlp
"""

import argparse
import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# ==============================================================================
# Model Definitions
# ==============================================================================

class TemporalViolenceModel(nn.Module):
    """Temporal LSTM model from train_temporal_model.py."""
    
    def __init__(self, input_dim=3584, hidden_dim=256, lstm_hidden=128,
                 num_lstm_layers=2, dropout=0.3, bidirectional=True, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        
        self.frame_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=lstm_hidden, num_layers=num_lstm_layers,
            batch_first=True, dropout=dropout if num_lstm_layers > 1 else 0,
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
    
    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(B * T, D)
        encoded = self.frame_encoder(x_flat).view(B, T, -1)
        lstm_out, _ = self.lstm(encoded)
        
        if self.use_attention:
            attn_weights = F.softmax(self.attention(lstm_out), dim=1)
            context = (lstm_out * attn_weights).sum(dim=1)
        else:
            context = lstm_out[:, -1, :]
        
        return self.classifier(context).squeeze(-1)


class VLMStudentModel(nn.Module):
    """Improved student model from train_student.py."""
    
    def __init__(self, input_dim=3584, hidden_dims=[512, 256, 128], dropout=0.3,
                 use_temporal=False, temporal_hidden=128):
        super().__init__()
        self.use_temporal = use_temporal
        
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hdim), nn.BatchNorm1d(hdim),
                nn.GELU(), nn.Dropout(dropout),
            ])
            prev_dim = hdim
        self.feature_extractor = nn.Sequential(*layers)
        
        if use_temporal:
            self.lstm = nn.LSTM(hidden_dims[-1], temporal_hidden, 1, batch_first=True, bidirectional=True)
            final_dim = temporal_hidden * 2
        else:
            final_dim = hidden_dims[-1]
        
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, 64), nn.GELU(),
            nn.Dropout(dropout / 2), nn.Linear(64, 1),
        )
    
    def forward(self, x, return_features=False):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        if x.dim() == 3 and self.use_temporal:
            B, T, D = x.shape
            features = self.feature_extractor(x.view(B * T, D)).view(B, T, -1)
            lstm_out, _ = self.lstm(features)
            features = lstm_out[:, -1, :]
        else:
            features = self.feature_extractor(x)
        
        logits = self.classifier(features).squeeze(-1)
        return (logits, features) if return_features else logits


class StudentMLP(nn.Module):
    """Original student model from train_student_original.py."""
    
    def __init__(self, in_dim, out_dim=1, hidden=256, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ==============================================================================
# Model Loading
# ==============================================================================

def detect_model_type(checkpoint):
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', {}))
    if any('lstm' in k for k in state_dict.keys()):
        return 'temporal'
    if any('feature_extractor' in k for k in state_dict.keys()):
        return 'vlm_student'
    if any(k.startswith('net.') for k in state_dict.keys()):
        return 'student_mlp'
    return 'temporal'


def load_model(model_path, device, model_type='auto'):
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        checkpoint = {'args': {}}
    
    args = checkpoint.get('args', {})
    
    if model_type == 'auto':
        model_type = detect_model_type(checkpoint)
        print(f"  Auto-detected: {model_type}")
    
    if model_type == 'temporal':
        model = TemporalViolenceModel(
            input_dim=args.get('input_dim', 3584),
            hidden_dim=args.get('hidden_dim', 256),
            lstm_hidden=args.get('lstm_hidden', 128),
            num_lstm_layers=args.get('num_lstm_layers', 2),
            dropout=args.get('dropout', 0.3),
        )
        seq_len = args.get('sequence_length', 30)
    elif model_type == 'vlm_student':
        model = VLMStudentModel(
            input_dim=args.get('input_dim', 3584),
            hidden_dims=args.get('hidden_dims', [512, 256, 128]),
            dropout=args.get('dropout', 0.3),
            use_temporal=args.get('use_temporal', False),
        )
        seq_len = 1
    elif model_type == 'student_mlp':
        in_dim = checkpoint.get('in_dim', args.get('in_dim', 3584))
        model = StudentMLP(in_dim=in_dim, out_dim=args.get('out_dim', 1),
                          hidden=args.get('hidden', 256), drop=args.get('dropout', 0.2))
        seq_len = 1
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"  Type: {model_type}, Seq len: {seq_len}, Params: {sum(p.numel() for p in model.parameters()):,}")
    return model, model_type, seq_len, args


# ==============================================================================
# Evaluation
# ==============================================================================

@dataclass
class EvaluationResults:
    scores: np.ndarray
    predictions: np.ndarray
    labels: np.ndarray
    timestamps: np.ndarray
    threshold: float
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    fpr: np.ndarray
    tpr: np.ndarray
    roc_thresholds: np.ndarray
    pr_precision: np.ndarray
    pr_recall: np.ndarray
    pr_thresholds: np.ndarray
    confusion_mat: np.ndarray
    n_violence_events: int
    n_detected_events: int
    n_false_alarms: int
    lead_times: List[float]
    avg_lead_time: float
    total_duration_sec: float
    fa_per_hour: float
    detection_rate: float
    model_type: str = 'temporal'
    events: List[Dict] = field(default_factory=list)
    false_alarm_times: List[float] = field(default_factory=list)


class ComprehensiveEvaluator:
    def __init__(self, model_path, device='cuda:1', model_type='auto'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model, self.model_type, self.sequence_length, self.model_args = load_model(
            model_path, self.device, model_type
        )
    
    @torch.no_grad()
    def score_sequences(self, embeddings, batch_size=256):
        n_frames = len(embeddings)
        
        if self.model_type == 'temporal':
            n_seq = n_frames - self.sequence_length + 1
            all_scores = []
            for start in tqdm(range(0, n_seq, batch_size), desc="Scoring"):
                end = min(start + batch_size, n_seq)
                batch = np.stack([embeddings[i:i+self.sequence_length] for i in range(start, end)])
                x = torch.tensor(batch, dtype=torch.float32).to(self.device)
                scores = torch.sigmoid(self.model(x)).cpu().numpy()
                all_scores.extend(scores)
            return np.concatenate([np.zeros(self.sequence_length - 1), np.array(all_scores)])
        else:
            all_scores = []
            for start in tqdm(range(0, n_frames, batch_size), desc="Scoring"):
                end = min(start + batch_size, n_frames)
                x = torch.tensor(embeddings[start:end], dtype=torch.float32).to(self.device)
                logits = self.model(x)
                if logits.dim() > 1 and logits.shape[-1] > 1:
                    logits = logits[:, 0]
                all_scores.extend(torch.sigmoid(logits).cpu().numpy())
            return np.array(all_scores)
    
    def evaluate(self, sim_path, threshold=0.388, smoothing_window=0, 
                 optical_flow_sim=None, expected_input_dim=None):
        print("\n" + "="*70 + "\nCOMPREHENSIVE EVALUATION\n" + "="*70)
        
        data = np.load(sim_path)
        embeddings, labels = data['embeddings'], data['anticipation_labels']
        fps = float(data.get('fps', 1.0))
        n_frames = len(embeddings)
        
        # Check if we need to fuse optical flow features
        current_dim = embeddings.shape[1] if embeddings.ndim > 1 else embeddings.shape[0]
        
        if expected_input_dim is not None and current_dim != expected_input_dim:
            print(f"Dimension mismatch: model expects {expected_input_dim}, data has {current_dim}")
            
            if optical_flow_sim is not None:
                print(f"Loading optical flow features from {optical_flow_sim}...")
                flow_data = np.load(optical_flow_sim)
                flow_features = flow_data.get('embeddings', flow_data.get('features', flow_data.get('arr_0')))
                
                # Align lengths
                min_len = min(len(embeddings), len(flow_features))
                embeddings = embeddings[:min_len]
                flow_features = flow_features[:min_len]
                labels = labels[:min_len]
                n_frames = min_len
                
                # Normalize flow features
                flow_mean = flow_features.mean(axis=0, keepdims=True)
                flow_std = flow_features.std(axis=0, keepdims=True) + 1e-8
                flow_features = (flow_features - flow_mean) / flow_std
                
                # Fuse
                embeddings = np.concatenate([embeddings, flow_features], axis=1)
                print(f"Fused features: {embeddings.shape[1]} dimensions")
            else:
                print("ERROR: Model expects fused features but --optical_flow_sim not provided")
                print("Please provide optical flow simulation with --optical_flow_sim")
                raise ValueError(f"Dimension mismatch: expected {expected_input_dim}, got {current_dim}")
        
        print(f"Frames: {n_frames:,}, Duration: {n_frames/fps/3600:.2f}h, Threshold: {threshold}")
        
        scores = self.score_sequences(embeddings)
        if smoothing_window > 1:
            scores = np.convolve(scores, np.ones(smoothing_window)/smoothing_window, mode='same')
        
        timestamps = np.arange(n_frames) / fps
        predictions = (scores >= threshold).astype(int)
        
        roc_auc = roc_auc_score(labels, scores)
        pr_auc = average_precision_score(labels, scores)
        cm = confusion_matrix(labels, predictions, labels=[0, 1])
        tp, fp, fn, tn = cm[1,1], cm[0,1], cm[1,0], cm[0,0]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        fpr, tpr, roc_thresholds = roc_curve(labels, scores)
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(labels, scores)
        
        events, lead_times = self._analyze_events(labels, predictions, scores, timestamps, fps)
        false_alarm_times = self._find_false_alarm_times(labels, predictions, timestamps)
        
        n_violence = len([e for e in events if e['is_violence']])
        n_detected = len([e for e in events if e['is_violence'] and e['detected']])
        n_fa = len(false_alarm_times)
        total_sec = n_frames / fps
        
        return EvaluationResults(
            scores=scores, predictions=predictions, labels=labels, timestamps=timestamps,
            threshold=threshold, roc_auc=roc_auc, pr_auc=pr_auc, precision=precision,
            recall=recall, f1=f1, accuracy=accuracy, fpr=fpr, tpr=tpr,
            roc_thresholds=roc_thresholds, pr_precision=pr_precision, pr_recall=pr_recall,
            pr_thresholds=pr_thresholds, confusion_mat=cm, n_violence_events=n_violence,
            n_detected_events=n_detected, n_false_alarms=n_fa, lead_times=lead_times,
            avg_lead_time=np.mean(lead_times) if lead_times else 0,
            total_duration_sec=total_sec, fa_per_hour=n_fa/(total_sec/3600),
            detection_rate=n_detected/n_violence if n_violence > 0 else 0,
            model_type=self.model_type, events=events, false_alarm_times=false_alarm_times,
        )
    
    def _analyze_events(self, labels, predictions, scores, timestamps, fps):
        events, lead_times = [], []
        in_event, event_start = False, 0
        
        for i, label in enumerate(labels):
            if label == 1 and not in_event:
                in_event, event_start = True, i
            elif label == 0 and in_event:
                in_event = False
                events.append({'is_violence': True, 'start_frame': event_start, 'end_frame': i-1,
                              'start_time': timestamps[event_start], 'end_time': timestamps[i-1],
                              'duration_sec': (i-1-event_start)/fps})
        
        if in_event:
            events.append({'is_violence': True, 'start_frame': event_start, 'end_frame': len(labels)-1,
                          'start_time': timestamps[event_start], 'end_time': timestamps[-1],
                          'duration_sec': (len(labels)-1-event_start)/fps})
        
        horizon_frames = int(30 * fps)
        for event in events:
            if not event['is_violence']:
                continue
            start, end = event['start_frame'], event['end_frame']
            check_start = max(0, start - horizon_frames)
            
            detected, max_score, lead_time = False, 0, None
            for i in range(check_start, end + 1):
                max_score = max(max_score, scores[i])
                if predictions[i] == 1 and not detected:
                    detected, lead_time = True, max(0, (start - i) / fps)
            
            event.update({'detected': detected, 'lead_time_sec': lead_time, 'max_score': float(max_score)})
            if detected and lead_time is not None:
                lead_times.append(lead_time)
        
        return events, lead_times
    
    def _find_false_alarm_times(self, labels, predictions, timestamps):
        fa_times, in_fa = [], False
        for i, (label, pred) in enumerate(zip(labels, predictions)):
            if pred == 1 and label == 0 and not in_fa:
                in_fa = True
                fa_times.append(timestamps[i])
            elif pred == 0 or label == 1:
                in_fa = False
        return fa_times
    
    def threshold_sweep(self, sim_path, thresholds=None, optical_flow_sim=None, expected_input_dim=None):
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        data = np.load(sim_path)
        embeddings, labels = data['embeddings'], data['anticipation_labels']
        fps = float(data.get('fps', 1.0))
        
        # Check if we need to fuse optical flow features
        current_dim = embeddings.shape[1] if embeddings.ndim > 1 else embeddings.shape[0]
        
        if expected_input_dim is not None and current_dim != expected_input_dim and optical_flow_sim is not None:
            flow_data = np.load(optical_flow_sim)
            flow_features = flow_data.get('embeddings', flow_data.get('features', flow_data.get('arr_0')))
            
            min_len = min(len(embeddings), len(flow_features))
            embeddings = embeddings[:min_len]
            flow_features = flow_features[:min_len]
            labels = labels[:min_len]
            
            flow_mean = flow_features.mean(axis=0, keepdims=True)
            flow_std = flow_features.std(axis=0, keepdims=True) + 1e-8
            flow_features = (flow_features - flow_mean) / flow_std
            
            embeddings = np.concatenate([embeddings, flow_features], axis=1)
        
        scores = self.score_sequences(embeddings)
        timestamps = np.arange(len(embeddings)) / fps
        
        results = []
        print("\nThreshold sweep...")
        for thresh in thresholds:
            predictions = (scores >= thresh).astype(int)
            events, lead_times = self._analyze_events(labels, predictions, scores, timestamps, fps)
            fa_times = self._find_false_alarm_times(labels, predictions, timestamps)
            
            n_violence = len([e for e in events if e['is_violence']])
            n_detected = len([e for e in events if e['is_violence'] and e['detected']])
            total_sec = len(embeddings) / fps
            
            r = {'threshold': thresh, 'detection_rate': n_detected/n_violence if n_violence > 0 else 0,
                 'fa_per_hour': len(fa_times)/(total_sec/3600),
                 'avg_lead_time': np.mean(lead_times) if lead_times else 0,
                 'n_detected': n_detected, 'n_violence': n_violence, 'n_false_alarms': len(fa_times)}
            results.append(r)
            print(f"  θ={thresh:.2f}: det={r['detection_rate']:.1%}, fa/hr={r['fa_per_hour']:.1f}, lead={r['avg_lead_time']:.1f}s")
        
        return results


# ==============================================================================
# Visualization
# ==============================================================================

def plot_roc_curve(results, output_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(results.fpr, results.tpr, 'b-', lw=2.5, label=f'ROC (AUC={results.roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    idx = np.argmin(np.abs(results.roc_thresholds - results.threshold))
    ax.scatter([results.fpr[idx]], [results.tpr[idx]], c='red', s=150, zorder=5, label=f'θ={results.threshold:.3f}')
    ax.fill_between(results.fpr, results.tpr, alpha=0.2)
    ax.set_xlabel('FPR', fontsize=12); ax.set_ylabel('TPR', fontsize=12)
    ax.set_title(f'ROC Curve - {results.model_type.upper()}', fontsize=14)
    ax.legend(loc='lower right'); ax.set_xlim([0, 1]); ax.set_ylim([0, 1]); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()
    print(f"  Saved: {output_path}")


def plot_pr_curve(results, output_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    baseline = results.labels.mean()
    ax.plot(results.pr_recall, results.pr_precision, 'b-', lw=2.5, label=f'PR (AP={results.pr_auc:.4f})')
    ax.axhline(y=baseline, color='k', linestyle='--', lw=1, label=f'Baseline ({baseline:.3f})')
    ax.scatter([results.recall], [results.precision], c='red', s=150, zorder=5, label=f'θ={results.threshold:.3f}')
    ax.fill_between(results.pr_recall, results.pr_precision, alpha=0.2)
    ax.set_xlabel('Recall', fontsize=12); ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'PR Curve - {results.model_type.upper()}', fontsize=14)
    ax.legend(loc='upper right'); ax.set_xlim([0, 1]); ax.set_ylim([0, 1]); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()
    print(f"  Saved: {output_path}")


def plot_score_distribution(results, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    violent = results.scores[results.labels == 1]
    nonviolent = results.scores[results.labels == 0]
    ax.hist(nonviolent, bins=50, alpha=0.6, label=f'Non-violent (n={len(nonviolent):,})', color='blue', density=True)
    ax.hist(violent, bins=50, alpha=0.6, label=f'Violent (n={len(violent):,})', color='red', density=True)
    ax.axvline(x=results.threshold, color='green', linestyle='--', lw=2, label=f'Threshold ({results.threshold:.3f})')
    ax.set_xlabel('Score', fontsize=12); ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Score Distribution - {results.model_type.upper()}', fontsize=14)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()
    print(f"  Saved: {output_path}")


def plot_confusion_matrix(results, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = results.confusion_mat
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Pred -', 'Pred +'], yticklabels=['True -', 'True +'], annot_kws={'size': 14})
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.7, f'({cm_norm[i,j]*100:.1f}%)', ha='center', va='center', fontsize=10)
    ax.set_title(f'Confusion Matrix - {results.model_type.upper()}\n(θ={results.threshold:.3f})', fontsize=14)
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()
    print(f"  Saved: {output_path}")


def plot_timeline(results, output_path, max_duration=3600):
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    max_idx = min(int(max_duration), len(results.timestamps))
    times = results.timestamps[:max_idx]
    
    axes[0].fill_between(times, 0, results.labels[:max_idx], alpha=0.7, color='red', label='Violence')
    axes[0].set_ylabel('Ground Truth'); axes[0].set_ylim([0, 1.2]); axes[0].legend(loc='upper right')
    
    axes[1].plot(times, results.scores[:max_idx], 'b-', lw=0.5, alpha=0.8)
    axes[1].axhline(y=results.threshold, color='green', linestyle='--', lw=2, label=f'θ={results.threshold:.3f}')
    axes[1].fill_between(times, 0, results.scores[:max_idx], alpha=0.3)
    axes[1].set_ylabel('Score'); axes[1].set_ylim([0, 1]); axes[1].legend(loc='upper right')
    
    axes[2].fill_between(times, 0, results.predictions[:max_idx], alpha=0.7, color='orange', label='Alert')
    axes[2].set_ylabel('Predictions'); axes[2].set_ylim([0, 1.2]); axes[2].legend(loc='upper right')
    axes[2].set_xlabel('Time (s)')
    
    fig.suptitle(f'Timeline - {results.model_type.upper()} (First {max_duration/60:.0f} min)', fontsize=14)
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()
    print(f"  Saved: {output_path}")


def plot_lead_time_histogram(results, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    if results.lead_times:
        ax.hist(results.lead_times, bins=15, color='green', alpha=0.7, edgecolor='black')
        ax.axvline(x=results.avg_lead_time, color='red', linestyle='--', lw=2, label=f'Avg: {results.avg_lead_time:.1f}s')
        ax.axvline(x=np.median(results.lead_times), color='blue', linestyle=':', lw=2, label=f'Med: {np.median(results.lead_times):.1f}s')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No detections', ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_xlabel('Lead Time (s)'); ax.set_ylabel('Count')
    ax.set_title(f'Lead Time Distribution - {results.model_type.upper()}', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()
    print(f"  Saved: {output_path}")


def plot_threshold_tradeoff(sweep_results, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    thresholds = [r['threshold'] for r in sweep_results]
    det_rates = [r['detection_rate'] * 100 for r in sweep_results]
    fa_rates = [r['fa_per_hour'] for r in sweep_results]
    lead_times = [r['avg_lead_time'] for r in sweep_results]
    
    ax.plot(fa_rates, det_rates, 'b-o', lw=2, markersize=8)
    for i, t in enumerate(thresholds):
        ax.annotate(f'θ={t:.1f}\n({lead_times[i]:.0f}s)', (fa_rates[i], det_rates[i]),
                    textcoords='offset points', xytext=(5, 5), fontsize=8)
    ax.set_xlabel('FA/Hour'); ax.set_ylabel('Detection Rate (%)')
    ax.set_title('Detection Rate vs FA Rate Tradeoff', fontsize=14)
    ax.grid(True, alpha=0.3); ax.set_ylim([0, 105]); ax.set_xlim(left=0)
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()
    print(f"  Saved: {output_path}")


def plot_event_analysis(results, output_path):
    violence_events = [e for e in results.events if e['is_violence']]
    if not violence_events:
        print(f"  Skipped: {output_path} (no events)")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    detected = sum(1 for e in violence_events if e['detected'])
    missed = len(violence_events) - detected
    
    bars = axes[0].bar(['Detected', 'Missed'], [detected, missed], color=['green', 'red'], alpha=0.7)
    axes[0].set_ylabel('Events'); axes[0].set_title(f'Detection Status ({detected}/{len(violence_events)} = {results.detection_rate:.1%})')
    for bar, cnt in zip(bars, [detected, missed]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, str(cnt), ha='center', fontsize=12)
    
    max_scores = [e['max_score'] for e in violence_events]
    colors = ['green' if e['detected'] else 'red' for e in violence_events]
    axes[1].bar(range(len(violence_events)), max_scores, color=colors, alpha=0.7)
    axes[1].axhline(y=results.threshold, color='blue', linestyle='--', lw=2, label=f'θ={results.threshold:.3f}')
    axes[1].set_xlabel('Event Index'); axes[1].set_ylabel('Max Score'); axes[1].set_title('Max Score per Event'); axes[1].legend()
    
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()
    print(f"  Saved: {output_path}")


def plot_false_alarm_distribution(results, output_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    if results.false_alarm_times:
        ax.hist([t/3600 for t in results.false_alarm_times], bins=20, color='orange', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Time (hours)'); ax.set_ylabel('FA Count')
        ax.set_title(f'FA Distribution - {results.model_type.upper()} (Total: {len(results.false_alarm_times)}, Rate: {results.fa_per_hour:.1f}/hr)')
    else:
        ax.text(0.5, 0.5, 'No false alarms', ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()
    print(f"  Saved: {output_path}")


def create_summary_figure(results, output_path):
    fig = plt.figure(figsize=(12, 8))
    model_name = {'temporal': 'Temporal LSTM', 'vlm_student': 'VLM Student', 'student_mlp': 'Student MLP'}.get(results.model_type, results.model_type.upper())
    fig.suptitle(f'Violence Anticipation - {model_name} Summary', fontsize=16, fontweight='bold')
    
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    ax_text = fig.add_subplot(gs[0, :]); ax_text.axis('off')
    metrics_text = f"""
    ┌────────────────────────────────────────────────────────────────────────────┐
    │  Detection Rate: {results.detection_rate:.1%}    │    FA/Hour: {results.fa_per_hour:.1f}    │    Avg Lead: {results.avg_lead_time:.1f}s  │
    │  ROC-AUC: {results.roc_auc:.4f}    │    PR-AUC: {results.pr_auc:.4f}    │    F1: {results.f1:.4f}    │    θ: {results.threshold:.3f}  │
    └────────────────────────────────────────────────────────────────────────────┘"""
    ax_text.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=10, family='monospace', transform=ax_text.transAxes)
    
    ax_roc = fig.add_subplot(gs[1, 0])
    ax_roc.plot(results.fpr, results.tpr, 'b-', lw=2); ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlabel('FPR'); ax_roc.set_ylabel('TPR'); ax_roc.set_title(f'ROC (AUC={results.roc_auc:.3f})')
    ax_roc.set_xlim([0, 1]); ax_roc.set_ylim([0, 1])
    
    ax_pr = fig.add_subplot(gs[1, 1])
    ax_pr.plot(results.pr_recall, results.pr_precision, 'b-', lw=2)
    ax_pr.set_xlabel('Recall'); ax_pr.set_ylabel('Precision'); ax_pr.set_title(f'PR (AP={results.pr_auc:.3f})')
    ax_pr.set_xlim([0, 1]); ax_pr.set_ylim([0, 1])
    
    ax_cm = fig.add_subplot(gs[1, 2])
    sns.heatmap(results.confusion_mat, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                xticklabels=['Pred -', 'Pred +'], yticklabels=['True -', 'True +'], cbar=False)
    ax_cm.set_title('Confusion Matrix')
    
    ax_dist = fig.add_subplot(gs[2, 0])
    ax_dist.hist(results.scores[results.labels == 0], bins=30, alpha=0.6, label='Non-violent', density=True)
    ax_dist.hist(results.scores[results.labels == 1], bins=30, alpha=0.6, label='Violent', density=True)
    ax_dist.axvline(x=results.threshold, color='r', linestyle='--')
    ax_dist.set_xlabel('Score'); ax_dist.set_title('Score Dist'); ax_dist.legend(fontsize=8)
    
    ax_lead = fig.add_subplot(gs[2, 1])
    if results.lead_times:
        ax_lead.hist(results.lead_times, bins=10, color='green', alpha=0.7)
        ax_lead.axvline(x=results.avg_lead_time, color='r', linestyle='--')
    ax_lead.set_xlabel('Lead Time (s)'); ax_lead.set_title(f'Lead Times (avg={results.avg_lead_time:.1f}s)')
    
    ax_events = fig.add_subplot(gs[2, 2])
    violence_events = [e for e in results.events if e['is_violence']]
    if violence_events:
        detected = sum(1 for e in violence_events if e['detected'])
        ax_events.bar(['Detected', 'Missed'], [detected, len(violence_events) - detected], color=['green', 'red'])
        ax_events.set_title(f'Events ({detected}/{len(violence_events)})')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {output_path}")


def save_metrics_json(results, sweep_results, output_path):
    model_name = {'temporal': 'Temporal LSTM', 'vlm_student': 'VLM Student (BatchNorm+GELU)', 'student_mlp': 'Student MLP (LayerNorm)'}.get(results.model_type, results.model_type)
    metrics = {
        'model_config': {'threshold': float(results.threshold), 'model_type': results.model_type, 'model_name': model_name},
        'classification_metrics': {'roc_auc': float(results.roc_auc), 'pr_auc': float(results.pr_auc),
                                   'precision': float(results.precision), 'recall': float(results.recall),
                                   'f1': float(results.f1), 'accuracy': float(results.accuracy)},
        'operational_metrics': {'detection_rate': float(results.detection_rate), 'fa_per_hour': float(results.fa_per_hour),
                                'avg_lead_time_sec': float(results.avg_lead_time), 'total_violence_events': int(results.n_violence_events),
                                'detected_events': int(results.n_detected_events), 'total_false_alarms': int(results.n_false_alarms),
                                'total_duration_hours': float(results.total_duration_sec / 3600)},
        'confusion_matrix': {'tn': int(results.confusion_mat[0, 0]), 'fp': int(results.confusion_mat[0, 1]),
                            'fn': int(results.confusion_mat[1, 0]), 'tp': int(results.confusion_mat[1, 1])},
        'lead_times': [float(lt) for lt in results.lead_times],
        'event_details': [{'event_id': i+1, 'start_time_sec': float(e['start_time']), 'duration_sec': float(e['duration_sec']),
                          'detected': bool(e['detected']), 'lead_time_sec': float(e.get('lead_time_sec')) if e.get('lead_time_sec') else None,
                          'max_score': float(e['max_score'])} for i, e in enumerate(results.events) if e['is_violence']],
        'threshold_sweep': sweep_results,
    }
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {output_path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Model Evaluation")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--prebuilt_sim', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=0.388)
    parser.add_argument('--model_type', type=str, default='auto', choices=['auto', 'temporal', 'vlm_student', 'student_mlp'])
    parser.add_argument('--smoothing_window', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./final_evaluation')
    parser.add_argument('--device', type=str, default='cuda:1')
    
    # Optical flow fusion support
    parser.add_argument('--optical_flow_sim', type=str, default=None,
                        help='Path to optical flow simulation NPZ (for fused models)')
    parser.add_argument('--input_dim', type=int, default=None,
                        help='Expected input dimension (auto-detected from model if not specified)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator = ComprehensiveEvaluator(args.model_path, args.device, args.model_type)
    
    # Check if model expects fused features
    expected_dim = args.input_dim
    if expected_dim is None and hasattr(evaluator, 'model_args') and evaluator.model_args:
        expected_dim = evaluator.model_args.get('input_dim', 3584)
    
    results = evaluator.evaluate(
        args.prebuilt_sim, 
        args.threshold, 
        args.smoothing_window,
        optical_flow_sim=args.optical_flow_sim,
        expected_input_dim=expected_dim,
    )
    
    print("\n" + "="*70 + "\nRESULTS SUMMARY\n" + "="*70)
    print(f"Model Type: {results.model_type}")
    print(f"\nOperational: Det={results.detection_rate:.1%}, FA/hr={results.fa_per_hour:.1f}, Lead={results.avg_lead_time:.1f}s")
    print(f"Classification: ROC-AUC={results.roc_auc:.4f}, PR-AUC={results.pr_auc:.4f}, F1={results.f1:.4f}")
    
    sweep_results = evaluator.threshold_sweep(
        args.prebuilt_sim,
        optical_flow_sim=args.optical_flow_sim,
        expected_input_dim=expected_dim,
    )
    
    print("\nGenerating visualizations...")
    plot_roc_curve(results, output_dir / 'roc_curve.png')
    plot_pr_curve(results, output_dir / 'pr_curve.png')
    plot_score_distribution(results, output_dir / 'score_distribution.png')
    plot_confusion_matrix(results, output_dir / 'confusion_matrix.png')
    plot_timeline(results, output_dir / 'timeline.png')
    plot_lead_time_histogram(results, output_dir / 'lead_time_histogram.png')
    plot_threshold_tradeoff(sweep_results, output_dir / 'threshold_tradeoff.png')
    plot_event_analysis(results, output_dir / 'event_analysis.png')
    plot_false_alarm_distribution(results, output_dir / 'false_alarm_distribution.png')
    create_summary_figure(results, output_dir / 'summary.png')
    save_metrics_json(results, sweep_results, output_dir / 'metrics.json')
    
    print("\n" + "="*70 + f"\nAll outputs saved to: {output_dir}\n" + "="*70)


if __name__ == '__main__':
    main()