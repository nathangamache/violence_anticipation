#!/usr/bin/env python3
"""
Contrastive projection V3: classification-first MLP projector over VLM embeddings with auxiliary SupCon.

Usage:
    python tuning/contrastive_projection.py --embeddings_dir data/processed/vlm_emb \
        --labels_dir data/processed/labels_multih --output_dir ./contrastive_v3 \
        --model_size large --epochs 100 --device cuda:1
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class ProjectionNetwork(nn.Module):
    """MLP encoder with a contrastive projection head and a classification head."""
    
    MODEL_CONFIGS = {
        'small': {'hidden_dims': [1024, 512, 256], 'dropout': 0.2},
        'medium': {'hidden_dims': [1536, 768, 384], 'dropout': 0.25},
        'large': {'hidden_dims': [2048, 1024, 512, 256], 'dropout': 0.3},
    }
    
    def __init__(
        self,
        input_dim: int = 3584,
        model_size: str = 'large',
        projection_dim: int = 128,
    ):
        super().__init__()
        
        config = self.MODEL_CONFIGS[model_size]
        hidden_dims = config['hidden_dims']
        dropout = config['dropout']
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        self.projector = nn.Sequential(
            nn.Linear(hidden_dims[-1], projection_dim),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1),
        )
        
        self._init_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Model '{model_size}': {total_params:,} parameters")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor):
        """Returns (projection, logits, features)"""
        features = self.encoder(x)
        
        proj = self.projector(features)
        proj = F.normalize(proj, p=2, dim=-1)
        
        logits = self.classifier(features).squeeze(-1)
        
        return proj, logits, features
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get encoder features for export."""
        return self.encoder(x)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class SimpleContrastiveLoss(nn.Module):
    """Supervised contrastive loss without hard-negative mining."""
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]
        
        if batch_size < 4:
            return torch.tensor(0.0, device=device)
        
        sim = torch.matmul(features, features.T) / self.temperature
        sim = torch.clamp(sim, -20, 20)  # Stability
        
        labels = labels.view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float()
        mask_self = torch.eye(batch_size, device=device)
        mask_pos = mask_pos - mask_self
        
        if mask_pos.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()
        
        exp_sim = torch.exp(sim) * (1 - mask_self)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        pos_count = mask_pos.sum(dim=1).clamp(min=1)
        loss = -(mask_pos * log_prob).sum(dim=1) / pos_count
        
        has_pos = mask_pos.sum(dim=1) > 0
        if has_pos.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        return loss[has_pos].mean()


def classify_video_source(video_id: str) -> str:
    if '_x264' in video_id:
        return 'ucf_crime'
    if video_id.startswith('v=') or '__#' in video_id:
        return 'xd_violence'
    return 'unknown'


def find_file(video_name: str, base_dir: Path, extensions: List[str]):
    source = classify_video_source(video_name)
    for subdir in [source, 'ucf_crime', 'xd_violence', '']:
        for ext in extensions:
            path = base_dir / subdir / f"{video_name}{ext}" if subdir else base_dir / f"{video_name}{ext}"
            if path.exists():
                return path
    return None


def load_embedding(path: Path):
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
    except:
        pass
    return None


def load_labels(path: Path):
    try:
        if path.suffix == '.npz':
            data = np.load(path)
            for key in ['labels', 'label', 'arr_0']:
                if key in data.keys():
                    return data[key]
        return np.load(path)
    except:
        pass
    return None


class SimpleDataset(Dataset):
    """Frame-level (VLM embedding, anticipation label) samples for the clips in an index JSON."""
    
    def __init__(
        self,
        index_path: str,
        embeddings_dir: Path,
        labels_dir: Path,
        source_filter: str = None,
        horizon_sec: float = 30.0,
        fps: float = 1.0,
        max_samples: int = None,
    ):
        self.samples = []
        
        with open(index_path) as f:
            items = json.load(f)
        
        horizon_frames = int(horizon_sec * fps)
        
        for item in tqdm(items, desc="Loading"):
            if isinstance(item, dict):
                video_name = Path(item.get('graph_path', item.get('label_path', ''))).stem
            else:
                video_name = str(item)
            
            if source_filter and classify_video_source(video_name) != source_filter:
                continue
            
            emb_path = find_file(video_name, embeddings_dir, ['.npz', '.npy'])
            label_path = find_file(video_name, labels_dir, ['.npy', '.npz'])
            
            if emb_path is None or label_path is None:
                continue
            
            embeddings = load_embedding(emb_path)
            labels_raw = load_labels(label_path)
            
            if embeddings is None or labels_raw is None:
                continue
            
            if labels_raw.ndim > 1:
                labels_raw = labels_raw.max(axis=1)
            labels_raw = (labels_raw > 0).astype(np.float32)
            
            min_len = min(len(embeddings), len(labels_raw))
            embeddings = embeddings[:min_len]
            labels_raw = labels_raw[:min_len]
            
            antic_labels = np.zeros_like(labels_raw)
            for t in range(len(labels_raw)):
                future_end = min(t + horizon_frames, len(labels_raw))
                if labels_raw[t:future_end].max() > 0:
                    antic_labels[t] = 1.0
            
            for t in range(len(embeddings)):
                self.samples.append({
                    'embedding': embeddings[t],
                    'label': antic_labels[t],
                    'video_id': video_name,
                })
        
        np.random.shuffle(self.samples)
        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]
        
        n_pos = sum(1 for s in self.samples if s['label'] > 0.5)
        n_neg = len(self.samples) - n_pos
        print(f"Dataset: {len(self.samples)} samples ({n_pos} pos, {n_neg} neg, rate={n_pos/len(self.samples):.1%})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'embedding': torch.tensor(s['embedding'], dtype=torch.float32),
            'label': torch.tensor(s['label'], dtype=torch.float32),
        }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 15,
    contrastive_weight: float = 0.1,
) -> Tuple[Dict, float]:
    
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)
    
    focal_loss = FocalLoss(alpha=0.75, gamma=2.0)
    contrastive_loss = SimpleContrastiveLoss(temperature=0.1)
    
    best_val_auc = 0.0
    best_state = None
    no_improve = 0
    
    history = defaultdict(list)
    
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0
        train_cls_sum = 0
        train_con_sum = 0
        train_labels_all = []
        train_scores_all = []
        n_batches = 0
        
        for batch in train_loader:
            emb = batch['embedding'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            proj, logits, _ = model(emb)
            
            loss_cls = focal_loss(logits, labels)
            
            loss_con = contrastive_loss(proj, labels)
            
            loss = loss_cls + contrastive_weight * loss_con
            
            if torch.isnan(loss) or loss.item() > 100:
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_cls_sum += loss_cls.item()
            train_con_sum += loss_con.item() if not torch.isnan(loss_con) else 0
            n_batches += 1
            
            with torch.no_grad():
                scores = torch.sigmoid(logits).cpu().numpy()
                train_labels_all.extend(labels.cpu().numpy())
                train_scores_all.extend(scores)
        
        scheduler.step()
        
        if n_batches == 0:
            print(f"Epoch {epoch+1}: No valid batches!")
            continue
        
        train_loss = train_loss_sum / n_batches
        train_cls = train_cls_sum / n_batches
        train_con = train_con_sum / n_batches
        
        train_labels_all = np.array(train_labels_all)
        train_scores_all = np.array(train_scores_all)
        
        if len(np.unique(train_labels_all)) > 1:
            train_auc = roc_auc_score(train_labels_all, train_scores_all)
            train_ap = average_precision_score(train_labels_all, train_scores_all)
        else:
            train_auc = 0.5
            train_ap = train_labels_all.mean()
        
        model.eval()
        val_labels_all = []
        val_scores_all = []
        
        with torch.no_grad():
            for batch in val_loader:
                emb = batch['embedding'].to(device)
                labels = batch['label']
                
                _, logits, _ = model(emb)
                scores = torch.sigmoid(logits).cpu().numpy()
                
                val_labels_all.extend(labels.numpy())
                val_scores_all.extend(scores)
        
        val_labels_all = np.array(val_labels_all)
        val_scores_all = np.array(val_scores_all)
        
        if len(np.unique(val_labels_all)) > 1:
            val_auc = roc_auc_score(val_labels_all, val_scores_all)
            val_ap = average_precision_score(val_labels_all, val_scores_all)
            precision, recall, thresholds = precision_recall_curve(val_labels_all, val_scores_all)
            f1s = 2 * precision * recall / (precision + recall + 1e-8)
            val_f1 = f1s[:-1].max()
            val_thresh = thresholds[f1s[:-1].argmax()]
        else:
            val_auc = 0.5
            val_ap = val_labels_all.mean()
            val_f1 = 0
            val_thresh = 0.5
        
        history['train_loss'].append(float(train_loss))
        history['train_cls'].append(float(train_cls))
        history['train_con'].append(float(train_con))
        history['train_auc'].append(float(train_auc))
        history['train_ap'].append(float(train_ap))
        history['val_auc'].append(float(val_auc))
        history['val_ap'].append(float(val_ap))
        history['val_f1'].append(float(val_f1))
        history['val_thresh'].append(float(val_thresh))
        history['lr'].append(float(optimizer.param_groups[0]['lr']))
        
        gap = train_auc - val_auc
        
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"loss={train_loss:.4f} (cls={train_cls:.4f}, con={train_con:.4f}) | "
              f"train_auc={train_auc:.4f} val_auc={val_auc:.4f} (gap={gap:+.4f}) | "
              f"val_ap={val_ap:.4f} val_f1={val_f1:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            print(f"  *** New best: {best_val_auc:.4f} ***")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return dict(history), best_val_auc


def export_embeddings(
    model: nn.Module,
    embeddings_dir: Path,
    output_dir: Path,
    device: torch.device,
    source_filter: str = None,
):
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(embeddings_dir.rglob("*.npz")) + list(embeddings_dir.rglob("*.npy"))
    print(f"Exporting {len(files)} files...")
    
    for path in tqdm(files):
        video_name = path.stem
        
        if source_filter and classify_video_source(video_name) != source_filter:
            continue
        
        embeddings = load_embedding(path)
        if embeddings is None:
            continue
        
        projected = []
        batch_size = 256
        
        with torch.no_grad():
            for i in range(0, len(embeddings), batch_size):
                batch = torch.tensor(embeddings[i:i+batch_size], dtype=torch.float32).to(device)
                features = model.get_features(batch).cpu().numpy()
                projected.append(features)
        
        projected = np.concatenate(projected, axis=0)
        
        rel_path = path.relative_to(embeddings_dir)
        out_path = output_dir / rel_path.parent / f"{video_name}.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(out_path, emb=projected.astype(np.float16))
    
    print(f"Exported to {output_dir}")


def plot_history(history: Dict, output_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Total')
    axes[0, 0].plot(epochs, history['train_cls'], 'g--', label='Classification')
    axes[0, 0].plot(epochs, history['train_con'], 'r--', label='Contrastive')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, history['train_auc'], 'b-', label='Train')
    axes[0, 1].plot(epochs, history['val_auc'], 'r-', label='Val')
    axes[0, 1].fill_between(epochs, history['train_auc'], history['val_auc'], alpha=0.2, color='gray')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('ROC-AUC')
    axes[0, 1].set_title('ROC-AUC (shaded = gap)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0.4, 1.0])
    
    axes[1, 0].plot(epochs, history['train_ap'], 'b-', label='Train')
    axes[1, 0].plot(epochs, history['val_ap'], 'r-', label='Val')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('PR-AUC')
    axes[1, 0].set_title('PR-AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    ax1 = axes[1, 1]
    ax1.plot(epochs, history['val_f1'], 'g-', label='Val F1')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('F1', color='g')
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.set_ylim([0, 1])
    
    ax2 = ax1.twinx()
    ax2.plot(epochs, history['val_thresh'], 'b--', alpha=0.5, label='Threshold')
    ax2.set_ylabel('Threshold', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.set_ylim([0, 1])
    
    ax1.set_title('Validation F1 and Threshold')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--embeddings_dir', default='data/processed/vlm_emb')
    parser.add_argument('--labels_dir', default='data/processed/labels_multih')
    parser.add_argument('--train_index', default='train_index.json')
    parser.add_argument('--val_index', default='val_index.json')
    parser.add_argument('--test_index', default='test_index.json')
    parser.add_argument('--source_filter', default='xd_violence')
    
    parser.add_argument('--model_size', default='large', choices=['small', 'medium', 'large'])
    parser.add_argument('--input_dim', type=int, default=3584)
    parser.add_argument('--projection_dim', type=int, default=128)
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--contrastive_weight', type=float, default=0.1)
    
    parser.add_argument('--horizon_sec', type=float, default=30.0)
    parser.add_argument('--max_train_samples', type=int, default=None)
    
    parser.add_argument('--export_embeddings', action='store_true')
    parser.add_argument('--output_dir', default='./contrastive_v3')
    parser.add_argument('--device', default='cuda:1')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("\nLoading data...")
    train_data = SimpleDataset(
        args.train_index, Path(args.embeddings_dir), Path(args.labels_dir),
        source_filter=args.source_filter, horizon_sec=args.horizon_sec,
        max_samples=args.max_train_samples,
    )
    val_data = SimpleDataset(
        args.val_index, Path(args.embeddings_dir), Path(args.labels_dir),
        source_filter=args.source_filter, horizon_sec=args.horizon_sec,
    )
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = ProjectionNetwork(
        input_dim=args.input_dim,
        model_size=args.model_size,
        projection_dim=args.projection_dim,
    )
    
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    history, best_val_auc = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        contrastive_weight=args.contrastive_weight,
    )
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'best_val_auc': best_val_auc,
    }, output_dir / 'model.pt')
    
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    plot_history(history, output_dir / 'training_curves.png')
    
    print("\nEvaluating on test...")
    test_data = SimpleDataset(
        args.test_index, Path(args.embeddings_dir), Path(args.labels_dir),
        source_filter=args.source_filter, horizon_sec=args.horizon_sec,
    )
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model.eval()
    test_labels, test_scores = [], []
    with torch.no_grad():
        for batch in test_loader:
            emb = batch['embedding'].to(device)
            _, logits, _ = model(emb)
            test_labels.extend(batch['label'].numpy())
            test_scores.extend(torch.sigmoid(logits).cpu().numpy())
    
    test_labels = np.array(test_labels)
    test_scores = np.array(test_scores)
    
    test_auc = roc_auc_score(test_labels, test_scores)
    test_ap = average_precision_score(test_labels, test_scores)
    precision, recall, thresholds = precision_recall_curve(test_labels, test_scores)
    f1s = 2 * precision * recall / (precision + recall + 1e-8)
    test_f1 = f1s[:-1].max()
    
    print(f"\nTest Results:")
    print(f"  ROC-AUC: {test_auc:.4f}")
    print(f"  PR-AUC:  {test_ap:.4f}")
    print(f"  Best F1: {test_f1:.4f}")
    
    metrics = {
        'best_val_auc': float(best_val_auc),
        'test_auc': float(test_auc),
        'test_ap': float(test_ap),
        'test_f1': float(test_f1),
    }
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    if args.export_embeddings:
        print("\nExporting embeddings...")
        export_dir = output_dir / 'projected_emb'
        
        with torch.no_grad():
            sample = torch.randn(1, args.input_dim).to(device)
            features = model.get_features(sample)
            feature_dim = features.shape[1]
        
        export_embeddings(model, Path(args.embeddings_dir), export_dir, device, args.source_filter)
        print(f"\nProjected dim: {feature_dim}")
        print(f"Use: --embeddings_dir {export_dir} --input_dim {feature_dim}")
    
    print(f"\nOutputs: {output_dir}")


if __name__ == '__main__':
    main()