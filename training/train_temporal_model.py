#!/usr/bin/env python3
"""
Temporal Violence Anticipation Model

This version uses temporal context (sequence of frames) instead of single-frame
classification. This should significantly reduce false alarms by:
1. Requiring consistent predictions across multiple frames
2. Learning temporal patterns that precede violence
3. Better distinguishing brief anomalies from real threats

Key improvements:
- LSTM/GRU to model temporal dependencies
- Sequence-level training (not just frame-level)
- Multi-scale temporal attention
- Confidence calibration
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    f1_score, roc_curve, confusion_matrix
)
from tqdm import tqdm

# Matplotlib setup for non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================================================================
# Temporal Model Architecture
# ==============================================================================
class TemporalViolenceModel(nn.Module):
    """
    Temporal model that processes sequences of embeddings.
    
    Architecture:
    1. Frame encoder: Projects each frame embedding
    2. Temporal encoder: LSTM/GRU to capture temporal patterns
    3. Attention: Weighs important frames in the sequence
    4. Classifier: Predicts violence probability
    """
    
    def __init__(
        self,
        input_dim: int = 3584,
        hidden_dim: int = 256,
        lstm_hidden: int = 128,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        
        # Frame encoder
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
        
        # Temporal encoder (LSTM)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        lstm_output_dim = lstm_hidden * 2 if bidirectional else lstm_hidden
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] - Batch of sequences
            lengths: [B] - Actual lengths (for variable length sequences)
        
        Returns:
            logits: [B] - Violence prediction for each sequence
        """
        B, T, D = x.shape
        
        # Encode each frame
        x_flat = x.view(B * T, D)
        encoded = self.frame_encoder(x_flat)
        encoded = encoded.view(B, T, -1)
        
        # Temporal encoding
        if lengths is not None:
            # Pack for variable length sequences
            packed = nn.utils.rnn.pack_padded_sequence(
                encoded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(encoded)
        
        # Aggregate temporal features
        if self.use_attention:
            # Attention-weighted pooling
            attn_weights = self.attention(lstm_out)  # [B, T, 1]
            attn_weights = F.softmax(attn_weights, dim=1)
            context = (lstm_out * attn_weights).sum(dim=1)  # [B, hidden]
        else:
            # Just use last hidden state
            context = lstm_out[:, -1, :]
        
        # Classify
        logits = self.classifier(context).squeeze(-1)
        
        return logits
    
    def predict_with_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return predictions and attention weights for interpretability."""
        B, T, D = x.shape
        
        x_flat = x.view(B * T, D)
        encoded = self.frame_encoder(x_flat)
        encoded = encoded.view(B, T, -1)
        
        lstm_out, _ = self.lstm(encoded)
        
        attn_weights = self.attention(lstm_out)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        context = (lstm_out * attn_weights).sum(dim=1)
        logits = self.classifier(context).squeeze(-1)
        
        return logits, attn_weights.squeeze(-1)


# ==============================================================================
# Sequence Dataset
# ==============================================================================
class TemporalViolenceDataset(Dataset):
    """
    Dataset that returns sequences of frames for temporal modeling.
    
    Each sample is a window of consecutive frames with a label indicating
    whether violence occurs within the prediction horizon.
    """
    
    def __init__(
        self,
        video_data: List[Dict],  # List of {embeddings: [T, D], labels: [T]}
        sequence_length: int = 30,
        horizon_sec: float = 30.0,
        fps: float = 1.0,
        stride: int = 5,  # How many frames to skip between sequences
        augment: bool = False,
    ):
        self.sequence_length = sequence_length
        self.horizon_frames = int(horizon_sec * fps)
        self.stride = stride
        self.augment = augment
        
        self.sequences = []
        self._build_sequences(video_data)
    
    def _build_sequences(self, video_data: List[Dict]):
        """Build sequence samples from video data."""
        
        for video in video_data:
            embeddings = video['embeddings']
            labels = video['labels']
            
            # Slide window across video
            for start in range(0, len(embeddings) - self.sequence_length + 1, self.stride):
                end = start + self.sequence_length
                
                # Get sequence
                seq_emb = embeddings[start:end]
                seq_labels = labels[start:end]
                
                # Label: is there violence in the horizon after the sequence?
                horizon_end = min(end + self.horizon_frames, len(labels))
                future_violence = labels[end:horizon_end].max() if horizon_end > end else 0
                
                # Also consider violence during the sequence
                current_violence = seq_labels.max()
                
                # Positive if violence in current sequence or upcoming horizon
                label = max(current_violence, future_violence)
                
                self.sequences.append({
                    'embeddings': seq_emb,
                    'label': float(label),
                    'video_id': video.get('video_id', 'unknown'),
                })
        
        print(f"  Built {len(self.sequences)} sequences")
        pos_rate = np.mean([s['label'] for s in self.sequences])
        print(f"  Positive rate: {pos_rate:.2%}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        embeddings = torch.tensor(seq['embeddings'], dtype=torch.float32)
        label = torch.tensor(seq['label'], dtype=torch.float32)
        
        # Data augmentation
        if self.augment and np.random.random() < 0.5:
            # Random temporal jittering (small shifts)
            if np.random.random() < 0.3:
                noise = torch.randn_like(embeddings) * 0.01
                embeddings = embeddings + noise
        
        return {
            'embeddings': embeddings,
            'label': label,
        }


# ==============================================================================
# Data Loading Helpers
# ==============================================================================
def classify_video_source(video_id: str) -> str:
    if '_x264' in video_id:
        return 'ucf_crime'
    if video_id.startswith('v='):
        return 'xd_violence'
    if re.search(r'\d{4}__#', video_id) or '__#' in video_id:
        return 'xd_violence'
    return 'unknown'


def find_embedding_file(video_name: str, embeddings_dir: Path) -> Optional[Path]:
    source = classify_video_source(video_name)
    for subdir in [source, 'ucf_crime', 'xd_violence', '']:
        for ext in ['.npz', '.npy']:
            if subdir:
                path = embeddings_dir / subdir / f"{video_name}{ext}"
            else:
                path = embeddings_dir / f"{video_name}{ext}"
            if path.exists():
                return path
    return None


def find_label_file(video_name: str, labels_dir: Path) -> Optional[Path]:
    for ext in ['.npy', '.npz']:
        path = labels_dir / f"{video_name}{ext}"
        if path.exists():
            return path
    return None


def load_embedding(path: Path) -> Optional[np.ndarray]:
    try:
        if path.suffix == '.npz':
            data = np.load(path)
            for key in ['emb', 'embedding', 'embeddings', 'features', 'arr_0']:
                if key in data.keys():
                    arr = data[key]
                    if arr.dtype == np.float16:
                        arr = arr.astype(np.float32)
                    return arr
        else:
            arr = np.load(path)
            if arr.dtype == np.float16:
                arr = arr.astype(np.float32)
            return arr
    except Exception as e:
        print(f"Error loading {path}: {e}")
    return None


def load_labels_from_file(path: Path) -> Optional[np.ndarray]:
    try:
        if path.suffix == '.npz':
            data = np.load(path)
            for key in ['labels', 'label', 'arr_0']:
                if key in data.keys():
                    return data[key]
        else:
            return np.load(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")
    return None


def extract_video_name(item) -> str:
    if isinstance(item, dict):
        path = item.get('graph_path') or item.get('label_path', '')
        return Path(path).stem
    return str(item)


def load_video_data(
    index_path: str,
    embeddings_dir: Path,
    labels_dir: Path,
    source_filter: Optional[str] = None,
) -> List[Dict]:
    """Load all videos and return list of {embeddings, labels, video_id}."""
    
    with open(index_path) as f:
        items = json.load(f)
    
    video_data = []
    
    for item in tqdm(items, desc="Loading videos"):
        video_name = extract_video_name(item)
        
        if source_filter and classify_video_source(video_name) != source_filter:
            continue
        
        emb_path = find_embedding_file(video_name, embeddings_dir)
        label_path = find_label_file(video_name, labels_dir)
        
        if emb_path is None or label_path is None:
            continue
        
        embeddings = load_embedding(emb_path)
        labels = load_labels_from_file(label_path)
        
        if embeddings is None or labels is None:
            continue
        
        if labels.ndim > 1:
            labels = labels.max(axis=1)
        labels = (labels > 0).astype(np.float32)
        
        min_len = min(len(embeddings), len(labels))
        embeddings = embeddings[:min_len]
        labels = labels[:min_len]
        
        video_data.append({
            'embeddings': embeddings,
            'labels': labels,
            'video_id': video_name,
        })
    
    return video_data


# ==============================================================================
# Training
# ==============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        return (focal_weight * ce_loss).mean()


def train_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    all_labels = []
    all_scores = []
    
    for batch in tqdm(loader, desc="Training", leave=False):
        embeddings = batch['embeddings'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        logits = model(embeddings)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(torch.sigmoid(logits).detach().cpu().numpy())
    
    metrics = {
        'loss': total_loss / len(loader),
        'roc_auc': roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0.5,
        'pr_auc': average_precision_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0,
    }
    
    return metrics


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels = []
    all_scores = []
    
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        embeddings = batch['embeddings'].to(device)
        labels = batch['label']
        
        logits = model(embeddings)
        scores = torch.sigmoid(logits).cpu().numpy()
        
        all_labels.extend(labels.numpy())
        all_scores.extend(scores)
    
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    metrics = {
        'roc_auc': roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0.5,
        'pr_auc': average_precision_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0,
        'labels': all_labels,
        'scores': all_scores,
    }
    
    # Find best threshold
    precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])
    
    metrics['best_threshold'] = thresholds[best_idx]
    metrics['best_f1'] = f1_scores[best_idx]
    metrics['best_precision'] = precision[best_idx]
    metrics['best_recall'] = recall[best_idx]
    
    return metrics


# ==============================================================================
# Visualization Functions
# ==============================================================================

def plot_training_curves(history: Dict, output_dir: Path):
    """Plot training and validation loss/metrics over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ROC-AUC curves
    ax = axes[1]
    ax.plot(epochs, history['train_roc_auc'], 'b-', label='Train ROC-AUC', linewidth=2)
    ax.plot(epochs, history['val_roc_auc'], 'r-', label='Val ROC-AUC', linewidth=2)
    
    # Mark best epoch
    best_epoch = np.argmax(history['val_roc_auc']) + 1
    best_val_roc = max(history['val_roc_auc'])
    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax.scatter([best_epoch], [best_val_roc], color='g', s=100, zorder=5)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ROC-AUC')
    ax.set_title('ROC-AUC Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()


def plot_roc_curve(labels: np.ndarray, scores: np.ndarray, output_path: Path, title: str = "ROC Curve"):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = roc_auc_score(labels, scores)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.fill_between(fpr, tpr, alpha=0.3)
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_pr_curve(labels: np.ndarray, scores: np.ndarray, output_path: Path, title: str = "Precision-Recall Curve"):
    """Plot Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    pr_auc = average_precision_score(labels, scores)
    baseline = labels.mean()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(recall, precision, 'b-', linewidth=2, label=f'PR Curve (AUC = {pr_auc:.4f})')
    ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1, label=f'Baseline ({baseline:.3f})')
    ax.fill_between(recall, precision, alpha=0.3)
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_score_distribution(labels: np.ndarray, scores: np.ndarray, output_path: Path, title: str = "Score Distribution"):
    """Plot score distributions for positive and negative classes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    neg_scores = scores[labels == 0]
    pos_scores = scores[labels == 1]
    
    ax.hist(neg_scores, bins=50, alpha=0.6, label=f'Negative (n={len(neg_scores)})', color='blue', density=True)
    ax.hist(pos_scores, bins=50, alpha=0.6, label=f'Positive (n={len(pos_scores)})', color='red', density=True)
    
    ax.set_xlabel('Prediction Score')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix(labels: np.ndarray, scores: np.ndarray, threshold: float, output_path: Path, title: str = "Confusion Matrix"):
    """Plot confusion matrix at given threshold."""
    preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    ax.set_title(f'{title}\n(threshold={threshold:.3f})')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_threshold_sweep(labels: np.ndarray, scores: np.ndarray, output_path: Path, title: str = "Threshold Sweep"):
    """Plot precision, recall, F1 across thresholds."""
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    f1s = []
    
    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
    ax.plot(thresholds, recalls, 'r-', label='Recall', linewidth=2)
    ax.plot(thresholds, f1s, 'g-', label='F1', linewidth=2)
    
    # Mark best F1
    best_idx = np.argmax(f1s)
    best_thresh = thresholds[best_idx]
    best_f1 = f1s[best_idx]
    ax.axvline(x=best_thresh, color='g', linestyle='--', alpha=0.7)
    ax.scatter([best_thresh], [best_f1], color='g', s=100, zorder=5, label=f'Best F1={best_f1:.3f} @ {best_thresh:.3f}')
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_learning_rate(lr_history: List[float], output_path: Path):
    """Plot learning rate schedule."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(lr_history) + 1), lr_history, 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_summary_figure(
    history: Dict,
    val_labels: np.ndarray,
    val_scores: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    val_metrics: Dict,
    test_metrics: Dict,
    output_path: Path,
):
    """Create a single summary figure with all key plots."""
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Training curves (top left)
    ax1 = fig.add_subplot(3, 3, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_roc_auc'], 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, history['val_roc_auc'], 'r-', label='Val', linewidth=2)
    best_epoch = np.argmax(history['val_roc_auc']) + 1
    ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('ROC-AUC')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation ROC (top middle)
    ax2 = fig.add_subplot(3, 3, 2)
    fpr, tpr, _ = roc_curve(val_labels, val_scores)
    ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC={val_metrics["roc_auc"]:.4f}')
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel('FPR')
    ax2.set_ylabel('TPR')
    ax2.set_title('Validation ROC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Test ROC (top right)
    ax3 = fig.add_subplot(3, 3, 3)
    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    ax3.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC={test_metrics["roc_auc"]:.4f}')
    ax3.plot([0, 1], [0, 1], 'k--')
    ax3.set_xlabel('FPR')
    ax3.set_ylabel('TPR')
    ax3.set_title('Test ROC')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Validation PR (middle left)
    ax4 = fig.add_subplot(3, 3, 4)
    precision, recall, _ = precision_recall_curve(val_labels, val_scores)
    ax4.plot(recall, precision, 'b-', linewidth=2, label=f'AUC={val_metrics["pr_auc"]:.4f}')
    ax4.axhline(y=val_labels.mean(), color='k', linestyle='--')
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.set_title('Validation PR')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Test PR (middle middle)
    ax5 = fig.add_subplot(3, 3, 5)
    precision, recall, _ = precision_recall_curve(test_labels, test_scores)
    ax5.plot(recall, precision, 'b-', linewidth=2, label=f'AUC={test_metrics["pr_auc"]:.4f}')
    ax5.axhline(y=test_labels.mean(), color='k', linestyle='--')
    ax5.set_xlabel('Recall')
    ax5.set_ylabel('Precision')
    ax5.set_title('Test PR')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Score distribution (middle right)
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.hist(test_scores[test_labels == 0], bins=30, alpha=0.6, label='Negative', color='blue', density=True)
    ax6.hist(test_scores[test_labels == 1], bins=30, alpha=0.6, label='Positive', color='red', density=True)
    ax6.axvline(x=test_metrics['best_threshold'], color='g', linestyle='--', label=f'Thresh={test_metrics["best_threshold"]:.3f}')
    ax6.set_xlabel('Score')
    ax6.set_ylabel('Density')
    ax6.set_title('Test Score Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Confusion matrix (bottom left)
    ax7 = fig.add_subplot(3, 3, 7)
    preds = (test_scores >= test_metrics['best_threshold']).astype(int)
    cm = confusion_matrix(test_labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax7)
    ax7.set_xlabel('Predicted')
    ax7.set_ylabel('Actual')
    ax7.set_title(f'Test Confusion Matrix @ {test_metrics["best_threshold"]:.3f}')
    
    # 8. Threshold sweep (bottom middle)
    ax8 = fig.add_subplot(3, 3, 8)
    thresholds = np.linspace(0, 1, 50)
    f1s = []
    for t in thresholds:
        preds = (test_scores >= t).astype(int)
        tp = ((preds == 1) & (test_labels == 1)).sum()
        fp = ((preds == 1) & (test_labels == 0)).sum()
        fn = ((preds == 0) & (test_labels == 1)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f1s.append(f1)
    ax8.plot(thresholds, f1s, 'g-', linewidth=2)
    best_idx = np.argmax(f1s)
    ax8.scatter([thresholds[best_idx]], [f1s[best_idx]], color='r', s=100, zorder=5)
    ax8.set_xlabel('Threshold')
    ax8.set_ylabel('F1 Score')
    ax8.set_title('Test F1 vs Threshold')
    ax8.grid(True, alpha=0.3)
    
    # 9. Summary text (bottom right)
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = f"""
TRAINING SUMMARY
================

Best Epoch: {best_epoch}
Total Epochs: {len(history['train_loss'])}

VALIDATION METRICS
------------------
ROC-AUC: {val_metrics['roc_auc']:.4f}
PR-AUC:  {val_metrics['pr_auc']:.4f}
Best F1: {val_metrics['best_f1']:.4f} @ thresh={val_metrics['best_threshold']:.3f}

TEST METRICS
------------
ROC-AUC: {test_metrics['roc_auc']:.4f}
PR-AUC:  {test_metrics['pr_auc']:.4f}
Best F1: {test_metrics['best_f1']:.4f} @ thresh={test_metrics['best_threshold']:.3f}

DATA STATISTICS
---------------
Val samples: {len(val_labels)} ({val_labels.mean():.1%} positive)
Test samples: {len(test_labels)} ({test_labels.mean():.1%} positive)
"""
    ax9.text(0.1, 0.5, summary_text, transform=ax9.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_all_visualizations(
    history: Dict,
    val_metrics: Dict,
    test_metrics: Dict,
    lr_history: List[float],
    output_dir: Path,
):
    """Generate all visualization plots."""
    graphs_dir = output_dir / 'graphs'
    graphs_dir.mkdir(exist_ok=True)
    
    val_labels = val_metrics['labels']
    val_scores = val_metrics['scores']
    test_labels = test_metrics['labels']
    test_scores = test_metrics['scores']
    
    print("Generating visualizations...")
    
    # Training curves
    print("  - Training curves...")
    plot_training_curves(history, graphs_dir)
    
    # ROC curves
    print("  - ROC curves...")
    plot_roc_curve(val_labels, val_scores, graphs_dir / 'roc_curve_val.png', 'Validation ROC Curve')
    plot_roc_curve(test_labels, test_scores, graphs_dir / 'roc_curve_test.png', 'Test ROC Curve')
    
    # PR curves
    print("  - PR curves...")
    plot_pr_curve(val_labels, val_scores, graphs_dir / 'pr_curve_val.png', 'Validation PR Curve')
    plot_pr_curve(test_labels, test_scores, graphs_dir / 'pr_curve_test.png', 'Test PR Curve')
    
    # Score distributions
    print("  - Score distributions...")
    plot_score_distribution(val_labels, val_scores, graphs_dir / 'score_distribution_val.png', 'Validation Score Distribution')
    plot_score_distribution(test_labels, test_scores, graphs_dir / 'score_distribution_test.png', 'Test Score Distribution')
    
    # Confusion matrices
    print("  - Confusion matrices...")
    plot_confusion_matrix(val_labels, val_scores, val_metrics['best_threshold'], graphs_dir / 'confusion_matrix_val.png', 'Validation Confusion Matrix')
    plot_confusion_matrix(test_labels, test_scores, test_metrics['best_threshold'], graphs_dir / 'confusion_matrix_test.png', 'Test Confusion Matrix')
    
    # Threshold sweeps
    print("  - Threshold sweeps...")
    plot_threshold_sweep(val_labels, val_scores, graphs_dir / 'threshold_sweep_val.png', 'Validation Threshold Sweep')
    plot_threshold_sweep(test_labels, test_scores, graphs_dir / 'threshold_sweep_test.png', 'Test Threshold Sweep')
    
    # Learning rate
    print("  - Learning rate schedule...")
    if lr_history:
        plot_learning_rate(lr_history, graphs_dir / 'learning_rate.png')
    
    # Summary figure
    print("  - Summary figure...")
    create_summary_figure(
        history, val_labels, val_scores, test_labels, test_scores,
        val_metrics, test_metrics, graphs_dir / 'summary.png'
    )
    
    print(f"Visualizations saved to {graphs_dir}/")


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train Temporal Violence Model")
    
    # Data
    parser.add_argument('--embeddings_dir', type=str, default='data/processed/vlm_emb')
    parser.add_argument('--labels_dir', type=str, default='data/processed/labels_multih')
    parser.add_argument('--train_index', type=str, default='train_index.json')
    parser.add_argument('--val_index', type=str, default='val_index.json')
    parser.add_argument('--test_index', type=str, default='test_index.json')
    parser.add_argument('--source_filter', type=str, default='xd_violence')
    
    # Model
    parser.add_argument('--input_dim', type=int, default=3584)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--lstm_hidden', type=int, default=128)
    parser.add_argument('--num_lstm_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--sequence_length', type=int, default=30)
    parser.add_argument('--stride', type=int, default=5)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--focal_alpha', type=float, default=0.75)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    
    # Task
    parser.add_argument('--horizon_sec', type=float, default=30.0)
    parser.add_argument('--fps', type=float, default=1.0)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs_temporal')
    parser.add_argument('--device', type=str, default='cuda:1')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = Path(args.output_dir)
    
    # Save args
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Device
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print(f"Warning: {args.device} not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading video data...")
    train_videos = load_video_data(
        args.train_index, Path(args.embeddings_dir), Path(args.labels_dir), args.source_filter
    )
    val_videos = load_video_data(
        args.val_index, Path(args.embeddings_dir), Path(args.labels_dir), args.source_filter
    )
    test_videos = load_video_data(
        args.test_index, Path(args.embeddings_dir), Path(args.labels_dir), args.source_filter
    )
    
    print(f"Loaded: {len(train_videos)} train, {len(val_videos)} val, {len(test_videos)} test videos")
    
    # Create datasets
    print("\nBuilding sequence datasets...")
    train_dataset = TemporalViolenceDataset(
        train_videos, 
        sequence_length=args.sequence_length,
        horizon_sec=args.horizon_sec,
        fps=args.fps,
        stride=args.stride,
        augment=True,
    )
    val_dataset = TemporalViolenceDataset(
        val_videos,
        sequence_length=args.sequence_length,
        horizon_sec=args.horizon_sec,
        fps=args.fps,
        stride=args.stride,
        augment=False,
    )
    test_dataset = TemporalViolenceDataset(
        test_videos,
        sequence_length=args.sequence_length,
        horizon_sec=args.horizon_sec,
        fps=args.fps,
        stride=args.stride,
        augment=False,
    )
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model
    model = TemporalViolenceModel(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        lstm_hidden=args.lstm_hidden,
        num_lstm_layers=args.num_lstm_layers,
        dropout=args.dropout,
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    
    # Training loop with history tracking
    best_val_roc = 0
    best_model_state = None
    epochs_without_improvement = 0
    
    history = {
        'train_loss': [],
        'train_roc_auc': [],
        'train_pr_auc': [],
        'val_roc_auc': [],
        'val_pr_auc': [],
        'val_best_f1': [],
        'val_best_threshold': [],
    }
    lr_history = []
    
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Record learning rate
        lr_history.append(optimizer.param_groups[0]['lr'])
        
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['train_roc_auc'].append(train_metrics['roc_auc'])
        history['train_pr_auc'].append(train_metrics['pr_auc'])
        history['val_roc_auc'].append(val_metrics['roc_auc'])
        history['val_pr_auc'].append(val_metrics['pr_auc'])
        history['val_best_f1'].append(val_metrics['best_f1'])
        history['val_best_threshold'].append(val_metrics['best_threshold'])
        
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, ROC-AUC: {train_metrics['roc_auc']:.4f}")
        print(f"  Val   - ROC-AUC: {val_metrics['roc_auc']:.4f}, PR-AUC: {val_metrics['pr_auc']:.4f}")
        print(f"          Best F1: {val_metrics['best_f1']:.4f} @ thresh={val_metrics['best_threshold']:.3f}")
        
        if val_metrics['roc_auc'] > best_val_roc:
            best_val_roc = val_metrics['roc_auc']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            print(f"          *** New best! ***")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"\nEarly stopping after {args.patience} epochs without improvement")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"\nValidation:")
    print(f"  ROC-AUC: {val_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC: {val_metrics['pr_auc']:.4f}")
    print(f"  Best F1: {val_metrics['best_f1']:.4f} @ thresh={val_metrics['best_threshold']:.3f}")
    
    print(f"\nTest:")
    print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"  Best F1: {test_metrics['best_f1']:.4f} @ thresh={test_metrics['best_threshold']:.3f}")
    
    # Save model
    save_path = output_dir / 'best_temporal_model.pt'
    torch.save({
        'model_state_dict': best_model_state,
        'args': vars(args),
        'val_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                        for k, v in val_metrics.items() if k not in ['labels', 'scores']},
        'test_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                         for k, v in test_metrics.items() if k not in ['labels', 'scores']},
    }, save_path)
    print(f"\nModel saved to {save_path}")
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    generate_all_visualizations(history, val_metrics, test_metrics, lr_history, output_dir)
    
    # Save final metrics as JSON
    final_metrics = {
        'validation': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                       for k, v in val_metrics.items() if k not in ['labels', 'scores']},
        'test': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                 for k, v in test_metrics.items() if k not in ['labels', 'scores']},
        'training_history': {k: [float(x) for x in v] for k, v in history.items()},
    }
    
    with open(output_dir / 'final_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - best_temporal_model.pt")
    print(f"  - args.json")
    print(f"  - final_metrics.json")
    print(f"  - graphs/")


if __name__ == '__main__':
    main()