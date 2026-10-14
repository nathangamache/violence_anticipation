#!/usr/bin/env python3
"""Frame-level MLP student on VLM embeddings: focal loss, threshold strategies, per-source diagnostics."""

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
    f1_score, roc_curve, matthews_corrcoef
)
from tqdm import tqdm


class FocalLoss(nn.Module):
    """Binary focal loss on logits: -alpha_t * (1 - p_t)^gamma * log(p_t)."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt = p if y=1, 1-p if y=0
        
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_weight = (1 - pt) ** self.gamma
        
        loss = alpha_t * focal_weight * BCE_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class VLMStudentModel(nn.Module):
    """Linear-BN-GELU-Dropout MLP over VLM embeddings, with an optional BiLSTM head (use_temporal)."""
    def __init__(
        self,
        input_dim: int = 3584,  # Qwen2.5-VL embedding dim
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.3,
        use_temporal: bool = False,
        temporal_hidden: int = 128,
    ):
        super().__init__()
        
        self.use_temporal = use_temporal
        
        layers = []
        prev_dim = input_dim
        
        for i, hdim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.BatchNorm1d(hdim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hdim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        if use_temporal:
            self.lstm = nn.LSTM(
                input_size=hidden_dims[-1],
                hidden_size=temporal_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            final_dim = temporal_hidden * 2
        else:
            final_dim = hidden_dims[-1]
        
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, 64),
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
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """x: [B, input_dim], or [B, T, input_dim] when use_temporal."""
        # BatchNorm1d needs a batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        if x.dim() == 3 and self.use_temporal:
            B, T, D = x.shape
            x_flat = x.view(B * T, D)
            features = self.feature_extractor(x_flat)
            features = features.view(B, T, -1)
            
            lstm_out, _ = self.lstm(features)
            features = lstm_out[:, -1, :]
        else:
            features = self.feature_extractor(x)
        
        logits = self.classifier(features).squeeze(-1)
        
        if return_features:
            return logits, features
        return logits


def classify_video_source(video_id: str) -> str:
    if '_x264' in video_id:
        return 'ucf_crime'
    if video_id.startswith('v='):
        return 'xd_violence'
    if re.search(r'\d{4}__#', video_id) or '__#' in video_id:
        return 'xd_violence'  # Movies are in XD-Violence
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
    
    source = classify_video_source(video_name)
    for subdir in [source, 'ucf_crime', 'xd_violence']:
        for ext in ['.npy', '.npz']:
            path = labels_dir / subdir / f"{video_name}{ext}"
            if path.exists():
                return path
    return None


def load_embedding(path: Path) -> Optional[np.ndarray]:
    try:
        if path.suffix == '.npz':
            data = np.load(path)
            keys = list(data.keys())
            for key in ['emb', 'embedding', 'embeddings', 'features', 'arr_0']:
                if key in keys:
                    arr = data[key]
                    if arr.dtype == np.float16:
                        arr = arr.astype(np.float32)
                    return arr
            if keys:
                arr = data[keys[0]]
                if arr.dtype == np.float16:
                    arr = arr.astype(np.float32)
                return arr
        else:
            arr = np.load(path)
            if arr.dtype == np.float16:
                arr = arr.astype(np.float32)
            return arr
    except Exception as e:
        print(f"Error loading embedding {path}: {e}")
    return None


def load_labels_from_file(path: Path) -> Optional[np.ndarray]:
    try:
        if path.suffix == '.npz':
            data = np.load(path)
            keys = list(data.keys())
            for key in ['labels', 'label', 'arr_0']:
                if key in keys:
                    return data[key]
            if keys:
                return data[keys[0]]
        else:
            return np.load(path)
    except Exception as e:
        print(f"Error loading labels {path}: {e}")
    return None


def extract_video_name(item) -> str:
    if isinstance(item, dict):
        path = item.get('graph_path') or item.get('label_path', '')
        return Path(path).stem
    elif isinstance(item, str):
        return Path(item).stem
    return str(item)


class ViolenceAnticipationDataset(Dataset):
    """One sample per embedding row; label = any violence in rows [t, t + horizon_sec * fps)."""
    
    def __init__(
        self,
        video_items: List,
        embeddings_dir: Path,
        labels_dir: Path,
        horizon_sec: float = 30.0,
        fps: float = 1.0,
        min_frames: int = 5,
        source_filter: Optional[str] = None,
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.labels_dir = Path(labels_dir)
        self.horizon_sec = horizon_sec
        self.fps = fps
        self.min_frames = min_frames
        self.source_filter = source_filter.lower() if source_filter else None
        
        self.video_ids = [extract_video_name(item) for item in video_items]
        
        if self.source_filter:
            original_count = len(self.video_ids)
            self.video_ids = [
                vid for vid in self.video_ids 
                if classify_video_source(vid) == self.source_filter
            ]
            print(f"  Source filter '{self.source_filter}': {original_count} -> {len(self.video_ids)} videos")
        
        self.samples = []
        self._build_samples()
    
    def _build_samples(self):
        missing_emb = 0
        missing_label = 0
        load_errors = 0
        
        for vid in tqdm(self.video_ids, desc="Building dataset"):
            emb_path = find_embedding_file(vid, self.embeddings_dir)
            label_path = find_label_file(vid, self.labels_dir)
            
            if emb_path is None:
                missing_emb += 1
                continue
            if label_path is None:
                missing_label += 1
                continue
            
            try:
                embeddings = load_embedding(emb_path)
                labels = load_labels_from_file(label_path)
                
                if embeddings is None or labels is None:
                    load_errors += 1
                    continue
                
                if labels.ndim > 1:
                    labels = labels.max(axis=1) if labels.shape[1] > 1 else labels.squeeze()
                
                labels = (labels > 0).astype(float)
                
                if len(embeddings) < self.min_frames:
                    continue
                
                min_len = min(len(embeddings), len(labels))
                embeddings = embeddings[:min_len]
                labels = labels[:min_len]
                
                horizon_frames = int(self.horizon_sec * self.fps)
                
                for t in range(len(embeddings)):
                    future_start = t
                    future_end = min(t + horizon_frames, len(labels))
                    
                    if future_end <= future_start:
                        continue
                    
                    anticipation_label = float(labels[future_start:future_end].max())
                    
                    self.samples.append({
                        'video_id': vid,
                        'frame_idx': t,
                        'embedding': embeddings[t],
                        'label': anticipation_label,
                        'original_label': float(labels[t]),
                    })
            except Exception as e:
                print(f"Error loading {vid}: {e}")
                load_errors += 1
                continue
        
        if missing_emb > 0 or missing_label > 0 or load_errors > 0:
            print(f"  Missing: {missing_emb} embeddings, {missing_label} labels, {load_errors} load errors")
        print(f"  Created {len(self.samples)} samples from {len(self.video_ids) - missing_emb - missing_label - load_errors} videos")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'embedding': torch.tensor(sample['embedding'], dtype=torch.float32),
            'label': torch.tensor(sample['label'], dtype=torch.float32),
            'video_id': sample['video_id'],
            'frame_idx': sample['frame_idx'],
        }


class SourceBalancedSampler(torch.utils.data.Sampler):
    """Samples each source in proportion to its weight (default: inverse source frequency)."""
    def __init__(
        self, 
        dataset,
        samples_per_epoch: Optional[int] = None,
        source_weights: Optional[Dict[str, float]] = None,
    ):
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch or len(dataset)
        
        self.source_indices = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            source = classify_video_source(sample['video_id'])
            self.source_indices[source].append(idx)
        
        if source_weights is None:
            total = len(dataset)
            self.source_weights = {
                src: total / len(indices) 
                for src, indices in self.source_indices.items()
            }
        else:
            self.source_weights = source_weights
        
        weight_sum = sum(self.source_weights.values())
        self.source_weights = {k: v / weight_sum for k, v in self.source_weights.items()}
        
        print(f"  Source-balanced sampling:")
        for src, indices in sorted(self.source_indices.items()):
            weight = self.source_weights.get(src, 0)
            print(f"    {src}: {len(indices)} samples, weight={weight:.3f}")
    
    def __iter__(self):
        indices = []
        
        for src, weight in self.source_weights.items():
            n_samples = int(self.samples_per_epoch * weight)
            src_indices = self.source_indices[src]
            
            if len(src_indices) == 0:
                continue
            
            if n_samples > len(src_indices):
                sampled = np.random.choice(src_indices, size=n_samples, replace=True)
            else:
                sampled = np.random.choice(src_indices, size=n_samples, replace=False)
            
            indices.extend(sampled.tolist())
        
        np.random.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        return self.samples_per_epoch


def find_threshold_f1(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])  # last PR point has no threshold
    return thresholds[best_idx], f1_scores[best_idx]


def find_threshold_youden(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx], j_scores[best_idx]


def find_threshold_precision_floor(
    y_true: np.ndarray, 
    y_scores: np.ndarray,
    min_precision: float = 0.3
) -> Tuple[float, float]:
    """Find threshold that achieves minimum precision while maximizing recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    valid_mask = precision[:-1] >= min_precision
    
    if not valid_mask.any():
        return find_threshold_f1(y_true, y_scores)
    
    valid_recalls = recall[:-1][valid_mask]
    valid_thresholds = thresholds[valid_mask]
    best_idx = np.argmax(valid_recalls)
    
    return valid_thresholds[best_idx], valid_recalls[best_idx]


def find_threshold_mcc(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(0, 1, 100)
    best_mcc = -1
    best_thresh = 0.5
    
    for thresh in thresholds:
        preds = (y_scores >= thresh).astype(int)
        if len(np.unique(preds)) < 2:
            continue
        mcc = matthews_corrcoef(y_true, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = thresh
    
    return best_thresh, best_mcc


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        use_focal_loss: bool = True,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        class_weight: Optional[float] = None,
        grad_clip: float = 1.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.grad_clip = grad_clip
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            pos_weight = torch.tensor([class_weight]) if class_weight else None
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        self.best_val_roc = 0.0
        self.best_model_state = None
    
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        all_labels = []
        all_scores = []
        
        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            emb = batch['embedding'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(emb)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item() * len(labels)
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(torch.sigmoid(logits).detach().cpu().numpy())
        
        self.scheduler.step()
        
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)
        
        metrics = {
            'loss': total_loss / len(all_labels),
            'roc_auc': roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.5,
            'pr_auc': average_precision_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0,
        }
        
        return metrics
    
    @torch.no_grad()
    def evaluate(
        self, 
        loader: DataLoader, 
        return_predictions: bool = False
    ) -> Dict[str, any]:
        self.model.eval()
        
        all_labels = []
        all_scores = []
        all_video_ids = []
        all_frame_idxs = []
        
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            emb = batch['embedding'].to(self.device)
            labels = batch['label']
            
            logits = self.model(emb)
            scores = torch.sigmoid(logits).cpu().numpy()
            
            all_labels.extend(labels.numpy())
            all_scores.extend(scores)
            all_video_ids.extend(batch['video_id'])
            all_frame_idxs.extend(batch['frame_idx'].numpy())
        
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)
        
        pos_rate = all_labels.mean()
        
        if len(np.unique(all_labels)) < 2:
            metrics = {
                'roc_auc': 0.5,
                'pr_auc': pos_rate,
                'pos_rate': pos_rate,
                'n_samples': len(all_labels),
            }
        else:
            metrics = {
                'roc_auc': roc_auc_score(all_labels, all_scores),
                'pr_auc': average_precision_score(all_labels, all_scores),
                'pos_rate': pos_rate,
                'n_samples': len(all_labels),
            }
        
        threshold_methods = {
            'f1': find_threshold_f1,
            'youden': find_threshold_youden,
            'precision_floor': lambda y, s: find_threshold_precision_floor(y, s, min_precision=0.3),
            'mcc': find_threshold_mcc,
        }
        
        for name, method in threshold_methods.items():
            try:
                thresh, score = method(all_labels, all_scores)
                preds = (all_scores >= thresh).astype(int)
                
                tp = ((preds == 1) & (all_labels == 1)).sum()
                fp = ((preds == 1) & (all_labels == 0)).sum()
                fn = ((preds == 0) & (all_labels == 1)).sum()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics[f'{name}_threshold'] = thresh
                metrics[f'{name}_precision'] = precision
                metrics[f'{name}_recall'] = recall
                metrics[f'{name}_f1'] = f1
            except Exception as e:
                print(f"Error computing {name} threshold: {e}")
        
        if return_predictions:
            metrics['labels'] = all_labels
            metrics['scores'] = all_scores
            metrics['video_ids'] = all_video_ids
            metrics['frame_idxs'] = all_frame_idxs
        
        return metrics
    
    def train(self, n_epochs: int, patience: int = 10) -> Dict[str, List[float]]:
        history = defaultdict(list)
        epochs_without_improvement = 0
        
        for epoch in range(n_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{n_epochs}")
            print(f"{'='*60}")
            
            train_metrics = self.train_epoch()
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"ROC-AUC: {train_metrics['roc_auc']:.4f}, "
                  f"PR-AUC: {train_metrics['pr_auc']:.4f}")
            
            val_metrics = self.evaluate(self.val_loader)
            print(f"Val   - ROC-AUC: {val_metrics['roc_auc']:.4f}, "
                  f"PR-AUC: {val_metrics['pr_auc']:.4f}")
            print(f"        F1@thresh: {val_metrics.get('f1_f1', 0):.4f} "
                  f"(thresh={val_metrics.get('f1_threshold', 0):.3f})")
            
            if val_metrics['roc_auc'] > self.best_val_roc:
                self.best_val_roc = val_metrics['roc_auc']
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
                print(f"        *** New best validation ROC-AUC: {self.best_val_roc:.4f} ***")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"\n        Early stopping triggered after {patience} epochs without improvement")
                    break
            
            for k, v in train_metrics.items():
                history[f'train_{k}'].append(v)
            for k, v in val_metrics.items():
                if isinstance(v, (int, float)):
                    history[f'val_{k}'].append(v)
        
        print(f"\nTraining complete. Best val ROC-AUC: {self.best_val_roc:.4f}")
        return dict(history)
    
    def load_best_model(self):
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
    
    def final_evaluation(self) -> Dict[str, Dict]:
        self.load_best_model()
        
        results = {}
        
        for name, loader in [('val', self.val_loader), ('test', self.test_loader)]:
            print(f"\n{'='*60}")
            print(f"Final {name.upper()} Evaluation")
            print(f"{'='*60}")
            
            metrics = self.evaluate(loader, return_predictions=True)
            
            print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"PR-AUC:  {metrics['pr_auc']:.4f}")
            print(f"Positive Rate: {metrics['pos_rate']:.4f}")
            print(f"\nThreshold Strategies:")
            
            for method in ['f1', 'youden', 'precision_floor', 'mcc']:
                print(f"  {method:15s}: thresh={metrics.get(f'{method}_threshold', 0):.3f}, "
                      f"P={metrics.get(f'{method}_precision', 0):.3f}, "
                      f"R={metrics.get(f'{method}_recall', 0):.3f}, "
                      f"F1={metrics.get(f'{method}_f1', 0):.3f}")
            
            results[name] = metrics
        
        print(f"\n{'='*60}")
        print("GENERALIZATION GAP ANALYSIS")
        print(f"{'='*60}")
        
        roc_gap = results['val']['roc_auc'] - results['test']['roc_auc']
        pr_gap = results['val']['pr_auc'] - results['test']['pr_auc']
        
        print(f"ROC-AUC Gap (val - test): {roc_gap:.4f}")
        print(f"PR-AUC Gap (val - test):  {pr_gap:.4f}")
        
        if roc_gap > 0.15:
            print("WARNING: Large generalization gap detected!")
            print("   Consider: more regularization, data augmentation, or checking for domain shift")
        
        return results


def per_source_analysis(results: Dict, split_name: str = 'test'):
    metrics = results[split_name]
    
    if 'video_ids' not in metrics:
        print("No per-video predictions available")
        return
    
    video_ids = metrics['video_ids']
    labels = metrics['labels']
    scores = metrics['scores']
    
    source_data = defaultdict(lambda: {'labels': [], 'scores': []})
    
    for vid, label, score in zip(video_ids, labels, scores):
        source = classify_video_source(vid)
        source_data[source]['labels'].append(label)
        source_data[source]['scores'].append(score)
    
    print(f"\n{'='*60}")
    print(f"PER-SOURCE ANALYSIS ({split_name.upper()})")
    print(f"{'='*60}")
    
    for source, data in sorted(source_data.items()):
        y_true = np.array(data['labels'])
        y_scores = np.array(data['scores'])
        
        n_samples = len(y_true)
        pos_rate = y_true.mean()
        
        if len(np.unique(y_true)) < 2:
            print(f"\n{source} ({n_samples} samples, {pos_rate:.1%} positive):")
            print(f"  WARNING: Only one class present, cannot compute ROC-AUC")
            continue
        
        roc_auc = roc_auc_score(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
        
        print(f"\n{source} ({n_samples} samples, {pos_rate:.1%} positive):")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  PR-AUC:  {pr_auc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train violence anticipation student model")
    
    parser.add_argument('--embeddings_dir', type=str, required=True,
                        help='Directory containing VLM embeddings (.npy files)')
    parser.add_argument('--labels_dir', type=str, required=True,
                        help='Directory containing frame-level labels (.npy files)')
    parser.add_argument('--train_index', type=str, required=True,
                        help='JSON file with train video IDs')
    parser.add_argument('--val_index', type=str, required=True,
                        help='JSON file with val video IDs')
    parser.add_argument('--test_index', type=str, required=True,
                        help='JSON file with test video IDs')
    
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256, 128],
                        help='Hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--input_dim', type=int, default=3584,
                        help='VLM embedding dimension')
    
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm')
    
    parser.add_argument('--use_focal_loss', action='store_true', default=True,
                        help='Use focal loss instead of BCE')
    parser.add_argument('--focal_alpha', type=float, default=0.75,
                        help='Focal loss alpha (weight for positive class)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma (focusing parameter)')
    
    parser.add_argument('--horizon_sec', type=float, default=30.0,
                        help='Anticipation horizon in seconds')
    parser.add_argument('--fps', type=float, default=1.0,
                        help='Frame rate of embeddings')
    parser.add_argument('--source_balanced', action='store_true',
                        help='Use source-balanced sampling to upweight underrepresented sources')
    parser.add_argument('--ucf_weight', type=float, default=None,
                        help='Manual weight for UCF-Crime source (e.g., 2.0 for 2x sampling)')
    parser.add_argument('--source_filter', type=str, default=None,
                        help='Only use videos from this source (e.g., ucf_crime, xd_violence)')
    
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for model and results')
    parser.add_argument('--save_model', action='store_true',
                        help='Save best model checkpoint')
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='Device to use for training (e.g., cuda:0, cuda:1, cpu)')
    
    args = parser.parse_args()
    
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print(f"Warning: {args.device} requested but CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.train_index) as f:
        train_items = json.load(f)
    with open(args.val_index) as f:
        val_items = json.load(f)
    with open(args.test_index) as f:
        test_items = json.load(f)
    
    print(f"Train: {len(train_items)} items")
    print(f"Val:   {len(val_items)} items")
    print(f"Test:  {len(test_items)} items")
    
    if train_items:
        print(f"Sample item format: {type(train_items[0])}")
        if isinstance(train_items[0], dict):
            print(f"  Keys: {list(train_items[0].keys())}")
            sample_name = extract_video_name(train_items[0])
            print(f"  Extracted name: {sample_name}")
    
    print("\nBuilding datasets...")
    train_dataset = ViolenceAnticipationDataset(
        train_items, args.embeddings_dir, args.labels_dir,
        horizon_sec=args.horizon_sec, fps=args.fps,
        source_filter=args.source_filter
    )
    val_dataset = ViolenceAnticipationDataset(
        val_items, args.embeddings_dir, args.labels_dir,
        horizon_sec=args.horizon_sec, fps=args.fps,
        source_filter=args.source_filter
    )
    test_dataset = ViolenceAnticipationDataset(
        test_items, args.embeddings_dir, args.labels_dir,
        horizon_sec=args.horizon_sec, fps=args.fps,
        source_filter=args.source_filter
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")
    
    train_labels = [s['label'] for s in train_dataset.samples]
    pos_rate = np.mean(train_labels)
    class_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
    print(f"\nTraining positive rate: {pos_rate:.4f}")
    print(f"Class weight (neg/pos): {class_weight:.2f}")
    
    if args.source_balanced:
        source_weights = None
        if args.ucf_weight is not None:
            source_weights = {
                'ucf_crime': args.ucf_weight,
                'xd_violence': 1.0,
                'Movies': 1.0,  # never matches: classify_video_source has no 'Movies' source
                'unknown': 0.5,
            }
        
        train_sampler = SourceBalancedSampler(
            train_dataset, 
            samples_per_epoch=len(train_dataset),
            source_weights=source_weights,
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            sampler=train_sampler,
            num_workers=4
        )
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = VLMStudentModel(
        input_dim=args.input_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_focal_loss=args.use_focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        class_weight=class_weight,
        grad_clip=args.grad_clip,
    )
    
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    history = trainer.train(args.epochs, patience=args.patience)
    
    results = trainer.final_evaluation()
    
    per_source_analysis(results, 'test')
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    results_to_save = {}
    for split, metrics in results.items():
        results_to_save[split] = {}
        for k, v in metrics.items():
            if k in ['labels', 'scores', 'video_ids', 'frame_idxs']:
                continue
            results_to_save[split][k] = convert_to_serializable(v)
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    if args.save_model:
        trainer.load_best_model()
        torch.save({
            'model_state_dict': model.state_dict(),
            'args': vars(args),
            'results': results_to_save,
        }, os.path.join(args.output_dir, 'best_model.pt'))
        print(f"\nModel saved to {args.output_dir}/best_model.pt")
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)


if __name__ == '__main__':
    main()