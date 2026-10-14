#!/usr/bin/env python3
"""SupCon + focal projector fine-tuned on frozen VLM embeddings.

Usage:
    python contrastive_finetune.py --embeddings_dir data/processed/vlm_emb \
        --labels_dir data/processed/labels_multih --train_index train_index.json \
        --output_dir outputs_contrastive --device cuda:1
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
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm


class SupervisedContrastiveLoss(nn.Module):
    """Supervised contrastive (SupCon) loss."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]
        
        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        labels = labels.view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float().to(device)
        mask_self = torch.eye(batch_size, device=device)
        mask_pos = mask_pos - mask_self
        
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        
        exp_logits = torch.exp(logits) * (1 - mask_self)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        mask_pos_sum = torch.clamp(mask_pos.sum(dim=1), min=1)
        mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / mask_pos_sum
        
        return -mean_log_prob_pos.mean()


class CombinedContrastiveLoss(nn.Module):
    """Weighted SupCon + focal (gamma=2) classification loss."""
    
    def __init__(self, temperature: float = 0.07, contrastive_weight: float = 0.5):
        super().__init__()
        self.supcon = SupervisedContrastiveLoss(temperature)
        self.contrastive_weight = contrastive_weight
        self.classification_weight = 1 - contrastive_weight
    
    def forward(self, features, logits, labels):
        loss_con = self.supcon(features, labels)
        
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        p_t = probs * labels + (1 - probs) * (1 - labels)
        focal_weight = (1 - p_t) ** 2
        loss_cls = (focal_weight * ce_loss).mean()
        
        total = self.contrastive_weight * loss_con + self.classification_weight * loss_cls
        
        return total, {'contrastive': loss_con.item(), 'classification': loss_cls.item(), 'total': total.item()}


class ContrastiveProjector(nn.Module):
    """Learns a projection from VLM embeddings to a contrastive space."""
    
    def __init__(self, input_dim=3584, projection_dim=256, hidden_dim=512, dropout=0.2):
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        projected = self.projector(x)
        projected = F.normalize(projected, dim=1)
        logits = self.classifier(projected).squeeze(-1)
        return projected, logits


class ContrastiveDataset(Dataset):
    def __init__(self, video_data, horizon_sec=30.0, fps=1.0, balance_classes=True):
        self.horizon_frames = int(horizon_sec * fps)
        self.samples = []
        
        for video in video_data:
            embeddings = video['embeddings']
            labels = video['labels']
            min_len = min(len(embeddings), len(labels))
            
            for t in range(min_len):
                future_end = min(t + self.horizon_frames, min_len)
                ant_label = float(labels[t:future_end].max()) if future_end > t else 0
                self.samples.append({'embedding': embeddings[t], 'label': ant_label})
        
        if balance_classes:
            pos = [s for s in self.samples if s['label'] > 0]
            neg = [s for s in self.samples if s['label'] == 0]
            if len(pos) < len(neg):
                pos = pos * (len(neg) // len(pos))
            self.samples = pos + neg
            np.random.shuffle(self.samples)
        
        print(f"  Samples: {len(self.samples)}, Pos rate: {np.mean([s['label'] for s in self.samples]):.2%}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'embedding': torch.tensor(s['embedding'], dtype=torch.float32),
            'label': torch.tensor(s['label'], dtype=torch.float32),
        }


def classify_video_source(video_id):
    if '_x264' in video_id: return 'ucf_crime'
    if video_id.startswith('v='): return 'xd_violence'
    if re.search(r'\d{4}__#', video_id) or '__#' in video_id: return 'xd_violence'
    return 'unknown'

def find_embedding_file(video_name, embeddings_dir):
    source = classify_video_source(video_name)
    for subdir in [source, 'ucf_crime', 'xd_violence', '']:
        for ext in ['.npz', '.npy']:
            path = embeddings_dir / subdir / f"{video_name}{ext}" if subdir else embeddings_dir / f"{video_name}{ext}"
            if path.exists(): return path
    return None

def find_label_file(video_name, labels_dir):
    for ext in ['.npy', '.npz']:
        path = labels_dir / f"{video_name}{ext}"
        if path.exists(): return path
    return None

def load_embedding(path):
    try:
        if path.suffix == '.npz':
            data = np.load(path)
            for key in ['emb', 'embedding', 'embeddings', 'arr_0']:
                if key in data.keys():
                    arr = data[key]
                    return arr.astype(np.float32) if arr.dtype == np.float16 else arr
        return np.load(path).astype(np.float32)
    except: return None

def load_labels_file(path):
    try:
        if path.suffix == '.npz':
            data = np.load(path)
            for key in ['labels', 'label', 'arr_0']:
                if key in data.keys(): return data[key]
        return np.load(path)
    except: return None

def extract_video_name(item):
    if isinstance(item, dict):
        path = item.get('graph_path') or item.get('label_path', '')
        return Path(path).stem
    return str(item)

def load_video_data(index_path, embeddings_dir, labels_dir, source_filter=None):
    with open(index_path) as f:
        items = json.load(f)
    
    video_data = []
    for item in tqdm(items, desc="Loading"):
        video_name = extract_video_name(item)
        if source_filter and classify_video_source(video_name) != source_filter: continue
        
        emb_path = find_embedding_file(video_name, embeddings_dir)
        label_path = find_label_file(video_name, labels_dir)
        if not emb_path or not label_path: continue
        
        embeddings = load_embedding(emb_path)
        labels = load_labels_file(label_path)
        if embeddings is None or labels is None: continue
        
        if labels.ndim > 1: labels = labels.max(axis=1)
        labels = (labels > 0).astype(np.float32)
        
        video_data.append({'embeddings': embeddings, 'labels': labels, 'video_id': video_name})
    
    return video_data


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_losses = defaultdict(float)
    all_labels, all_scores = [], []
    
    for batch in tqdm(loader, desc="Training", leave=False):
        emb = batch['embedding'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        projected, logits = model(emb)
        loss, loss_dict = criterion(projected, logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        for k, v in loss_dict.items(): total_losses[k] += v
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(torch.sigmoid(logits).detach().cpu().numpy())
    
    metrics = {k: v / len(loader) for k, v in total_losses.items()}
    if len(set(all_labels)) > 1:
        metrics['roc_auc'] = roc_auc_score(all_labels, all_scores)
    return metrics


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_scores, all_embeddings = [], [], []
    
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        emb = batch['embedding'].to(device)
        projected, logits = model(emb)
        
        all_labels.extend(batch['label'].numpy())
        all_scores.extend(torch.sigmoid(logits).cpu().numpy())
        all_embeddings.append(projected.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    all_embeddings = np.vstack(all_embeddings)
    
    metrics = {}
    if len(set(all_labels)) > 1:
        metrics['roc_auc'] = roc_auc_score(all_labels, all_scores)
        metrics['pr_auc'] = average_precision_score(all_labels, all_scores)
    
    pos_emb = all_embeddings[all_labels == 1]
    neg_emb = all_embeddings[all_labels == 0]
    if len(pos_emb) > 0 and len(neg_emb) > 0:
        metrics['class_separation'] = np.linalg.norm(pos_emb.mean(0) - neg_emb.mean(0))
    
    metrics['labels'] = all_labels
    metrics['scores'] = all_scores
    metrics['embeddings'] = all_embeddings
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_dir', default='data/processed/vlm_emb')
    parser.add_argument('--labels_dir', default='data/processed/labels_multih')
    parser.add_argument('--train_index', default='train_index.json')
    parser.add_argument('--val_index', default='val_index.json')
    parser.add_argument('--test_index', default='test_index.json')
    parser.add_argument('--source_filter', default='xd_violence')
    parser.add_argument('--input_dim', type=int, default=3584)
    parser.add_argument('--projection_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--contrastive_weight', type=float, default=0.5)
    parser.add_argument('--horizon_sec', type=float, default=30.0)
    parser.add_argument('--output_dir', default='./outputs_contrastive')
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("\nLoading data...")
    train_videos = load_video_data(args.train_index, Path(args.embeddings_dir), Path(args.labels_dir), args.source_filter)
    val_videos = load_video_data(args.val_index, Path(args.embeddings_dir), Path(args.labels_dir), args.source_filter)
    test_videos = load_video_data(args.test_index, Path(args.embeddings_dir), Path(args.labels_dir), args.source_filter)
    print(f"Loaded: {len(train_videos)} train, {len(val_videos)} val, {len(test_videos)} test")
    
    print("\nBuilding datasets...")
    train_dataset = ContrastiveDataset(train_videos, args.horizon_sec, balance_classes=True)
    val_dataset = ContrastiveDataset(val_videos, args.horizon_sec, balance_classes=False)
    test_dataset = ContrastiveDataset(test_videos, args.horizon_sec, balance_classes=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = ContrastiveProjector(args.input_dim, args.projection_dim, args.hidden_dim, args.dropout).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    criterion = CombinedContrastiveLoss(args.temperature, args.contrastive_weight)
    
    best_val_roc = 0
    best_state = None
    patience_counter = 0
    
    print("\n" + "="*60)
    print("CONTRASTIVE TRAINING")
    print("="*60)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_m = train_epoch(model, train_loader, optimizer, criterion, device)
        val_m = evaluate(model, val_loader, device)
        scheduler.step()
        
        print(f"  Train - Loss: {train_m.get('total',0):.4f}, ROC: {train_m.get('roc_auc',0):.4f}")
        print(f"  Val   - ROC: {val_m.get('roc_auc',0):.4f}, PR: {val_m.get('pr_auc',0):.4f}, Sep: {val_m.get('class_separation',0):.4f}")
        
        if val_m.get('roc_auc', 0) > best_val_roc:
            best_val_roc = val_m['roc_auc']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print("  *** New best! ***")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping")
                break
    
    model.load_state_dict(best_state)
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    val_m = evaluate(model, val_loader, device)
    test_m = evaluate(model, test_loader, device)
    
    print(f"\nVal  - ROC: {val_m.get('roc_auc',0):.4f}, PR: {val_m.get('pr_auc',0):.4f}, Sep: {val_m.get('class_separation',0):.4f}")
    print(f"Test - ROC: {test_m.get('roc_auc',0):.4f}, PR: {test_m.get('pr_auc',0):.4f}, Sep: {test_m.get('class_separation',0):.4f}")
    
    torch.save({'model_state_dict': best_state, 'args': vars(args)}, Path(args.output_dir) / 'contrastive_model.pt')
    np.savez_compressed(Path(args.output_dir) / 'test_projected.npz', 
                        embeddings=test_m['embeddings'], labels=test_m['labels'], scores=test_m['scores'])
    print(f"\nSaved to {args.output_dir}")


if __name__ == '__main__':
    main()