#!/usr/bin/env python3
"""Run the V1 contrastive projector over the 8-hour stitched simulation."""

import argparse
import json
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score


class ContrastiveProjector(nn.Module):
    """Must match training architecture."""
    
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
    
    def forward(self, x):
        projected = self.projector(x)
        projected = F.normalize(projected, dim=1)
        logits = self.classifier(projected).squeeze(-1)
        return projected, logits


@dataclass
class SimulationResults:
    threshold: float
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    confusion_mat: np.ndarray
    n_violence_events: int
    n_detected_events: int
    n_false_alarms: int
    avg_lead_time_sec: float
    total_duration_sec: float
    
    def print_summary(self):
        print("\n" + "="*70)
        print("CONTRASTIVE MODEL SIMULATION RESULTS")
        print("="*70)
        print(f"\nThreshold: {self.threshold:.3f}")
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                 No    Yes")
        print(f"  Actual No    {self.confusion_mat[0,0]:5d}  {self.confusion_mat[0,1]:5d}")
        print(f"  Actual Yes   {self.confusion_mat[1,0]:5d}  {self.confusion_mat[1,1]:5d}")
        print(f"\nPerformance:")
        print(f"  ROC-AUC:   {self.roc_auc:.4f}")
        print(f"  PR-AUC:    {self.pr_auc:.4f}")
        print(f"  Precision: {self.precision:.4f}")
        print(f"  Recall:    {self.recall:.4f}")
        print(f"  F1:        {self.f1:.4f}")
        print(f"\nEvent-Level:")
        print(f"  Violence events: {self.n_violence_events}")
        print(f"  Detected: {self.n_detected_events}")
        print(f"  False alarms: {self.n_false_alarms}")
        if self.avg_lead_time_sec > 0:
            print(f"  Avg lead time: {self.avg_lead_time_sec:.1f}s")
        print(f"\nOperational:")
        fa_per_hour = self.n_false_alarms / (self.total_duration_sec / 3600)
        det_rate = self.n_detected_events / self.n_violence_events if self.n_violence_events > 0 else 0
        print(f"  False alarms/hour: {fa_per_hour:.1f}")
        print(f"  Detection rate: {det_rate:.1%}")


class ContrastiveSimulator:
    def __init__(self, model_path: str, device: str = 'cuda:1', threshold: float = 0.5):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        
        print(f"Loading contrastive model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        args = checkpoint.get('args', {})
        self.model = ContrastiveProjector(
            input_dim=args.get('input_dim', 3584),
            projection_dim=args.get('projection_dim', 256),
            hidden_dim=args.get('hidden_dim', 512),
            dropout=args.get('dropout', 0.2),
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded. Device: {self.device}")
    
    @torch.no_grad()
    def score_batch(self, embeddings: np.ndarray) -> np.ndarray:
        x = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        _, logits = self.model(x)
        return torch.sigmoid(logits).cpu().numpy()
    
    def run_simulation(
        self,
        sim_path: str,
        smoothing_window: int = 1,
        batch_size: int = 1024,
    ) -> SimulationResults:
        print("\n" + "="*70)
        print("RUNNING CONTRASTIVE MODEL SIMULATION")
        print("="*70)
        
        print(f"Loading simulation from {sim_path}...")
        data = np.load(sim_path)
        embeddings = data['embeddings']
        labels = data['anticipation_labels']
        fps = float(data.get('fps', 1.0))
        
        n_frames = len(embeddings)
        print(f"  Frames: {n_frames:,}")
        print(f"  Duration: {n_frames / fps / 3600:.2f} hours")
        print(f"  Threshold: {self.threshold}")
        print(f"  Smoothing window: {smoothing_window}")
        
        print("\nScoring frames...")
        start_time = time.time()
        
        all_scores = []
        for i in range(0, n_frames, batch_size):
            batch = embeddings[i:i+batch_size]
            scores = self.score_batch(batch)
            all_scores.extend(scores)
            if (i + batch_size) % 10000 < batch_size:
                print(f"  Processed {min(i+batch_size, n_frames):,}/{n_frames:,}...")
        
        all_scores = np.array(all_scores)
        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.1f}s ({n_frames/elapsed:.0f} fps)")
        
        predictions = []
        recent_scores = []
        
        for score in all_scores:
            recent_scores.append(score)
            if len(recent_scores) > smoothing_window:
                recent_scores.pop(0)
            
            if len(recent_scores) >= smoothing_window:
                smoothed_above = all(s >= self.threshold for s in recent_scores)
            else:
                smoothed_above = False
            
            predictions.append(1 if smoothed_above else 0)
        
        predictions = np.array(predictions)
        
        return self._compute_metrics(all_scores, predictions, labels, fps)
    
    def _compute_metrics(self, scores, predictions, labels, fps) -> SimulationResults:
        cm = confusion_matrix(labels, predictions, labels=[0, 1])
        
        roc_auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.5
        pr_auc = average_precision_score(labels, scores) if len(np.unique(labels)) > 1 else 0
        
        tp, fp, fn = cm[1,1], cm[0,1], cm[1,0]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        violence_events = []
        in_event = False
        event_start = 0
        
        for i, label in enumerate(labels):
            if label == 1 and not in_event:
                in_event = True
                event_start = i
            elif label == 0 and in_event:
                in_event = False
                violence_events.append((event_start, i - 1))
        if in_event:
            violence_events.append((event_start, len(labels) - 1))
        
        detected = 0
        lead_times = []
        horizon_frames = int(30 * fps)
        
        for start, end in violence_events:
            check_start = max(0, start - horizon_frames)
            event_preds = predictions[check_start:end + 1]
            
            if any(p == 1 for p in event_preds):
                detected += 1
                for i, p in enumerate(event_preds):
                    if p == 1:
                        lead_time = (start - check_start - i) / fps
                        if lead_time > 0:
                            lead_times.append(lead_time)
                        break
        
        false_alarms = 0
        in_alarm = False
        for pred, label in zip(predictions, labels):
            if pred == 1 and label == 0 and not in_alarm:
                false_alarms += 1
                in_alarm = True
            elif pred == 0:
                in_alarm = False
        
        return SimulationResults(
            threshold=self.threshold,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            precision=precision,
            recall=recall,
            f1=f1,
            confusion_mat=cm,
            n_violence_events=len(violence_events),
            n_detected_events=detected,
            n_false_alarms=false_alarms,
            avg_lead_time_sec=np.mean(lead_times) if lead_times else 0,
            total_duration_sec=len(labels) / fps,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--prebuilt_sim', required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--smoothing_window', type=int, default=3)
    parser.add_argument('--output_dir', default='./simulation_results_contrastive')
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    simulator = ContrastiveSimulator(
        model_path=args.model_path,
        device=args.device,
        threshold=args.threshold,
    )
    
    results = simulator.run_simulation(
        args.prebuilt_sim,
        smoothing_window=args.smoothing_window,
    )
    
    results.print_summary()
    
    with open(Path(args.output_dir) / 'results.json', 'w') as f:
        json.dump({
            'threshold': results.threshold,
            'roc_auc': results.roc_auc,
            'pr_auc': results.pr_auc,
            'precision': results.precision,
            'recall': results.recall,
            'f1': results.f1,
            'n_violence_events': results.n_violence_events,
            'n_detected_events': results.n_detected_events,
            'n_false_alarms': results.n_false_alarms,
            'avg_lead_time_sec': results.avg_lead_time_sec,
        }, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()