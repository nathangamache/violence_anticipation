#!/usr/bin/env python3
"""Ensemble of the simple student MLP and the temporal LSTM on the long simulation.

Strategies: and (both above threshold), or (either), average, weighted.
"""

import argparse
import json
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from tqdm import tqdm


class SimpleStudentModel(nn.Module):
    """Per-frame student MLP; layer layout must match the saved checkpoint."""
    
    def __init__(self, input_dim=3584, hidden_dims=[512, 256, 128], dropout=0.3):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[2], 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        features = self.feature_extractor(x)
        return self.classifier(features).squeeze(-1)


class TemporalViolenceModel(nn.Module):
    """Copy of the temporal LSTM; must match the saved checkpoint."""
    
    def __init__(
        self,
        input_dim=3584,
        hidden_dim=256,
        lstm_hidden=128,
        num_lstm_layers=2,
        dropout=0.3,
        bidirectional=True,
        use_attention=True,
    ):
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
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        lstm_output_dim = lstm_hidden * 2 if bidirectional else lstm_hidden
        
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )
    
    def forward(self, x):
        B, T, D = x.shape
        
        x_flat = x.view(B * T, D)
        encoded = self.frame_encoder(x_flat)
        encoded = encoded.view(B, T, -1)
        
        lstm_out, _ = self.lstm(encoded)
        
        if self.use_attention:
            attn_weights = self.attention(lstm_out)
            attn_weights = F.softmax(attn_weights, dim=1)
            context = (lstm_out * attn_weights).sum(dim=1)
        else:
            context = lstm_out[:, -1, :]
        
        logits = self.classifier(context).squeeze(-1)
        return logits


@dataclass
class EnsembleResults:
    strategy: str
    simple_threshold: float
    temporal_threshold: float
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
        print("ENSEMBLE MODEL SIMULATION RESULTS")
        print("="*70)
        print(f"\nStrategy: {self.strategy}")
        print(f"Simple threshold: {self.simple_threshold:.3f}")
        print(f"Temporal threshold: {self.temporal_threshold:.3f}")
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


class EnsembleSimulator:
    def __init__(
        self,
        simple_model_path: str,
        temporal_model_path: str,
        device: str = 'cuda:1',
        simple_threshold: float = 0.6,
        temporal_threshold: float = 0.388,
        strategy: str = 'and',
        temporal_weight: float = 0.5,
        sequence_length: int = 30,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.simple_threshold = simple_threshold
        self.temporal_threshold = temporal_threshold
        self.strategy = strategy
        self.temporal_weight = temporal_weight
        self.sequence_length = sequence_length
        
        print(f"Loading simple model from {simple_model_path}...")
        simple_ckpt = torch.load(simple_model_path, map_location=self.device, weights_only=False)
        
        state_dict = simple_ckpt.get('model_state_dict', simple_ckpt)
        hidden_dim_0 = state_dict['feature_extractor.0.weight'].shape[0]
        hidden_dim_1 = state_dict['feature_extractor.4.weight'].shape[0]
        hidden_dim_2 = state_dict['feature_extractor.8.weight'].shape[0]
        input_dim = state_dict['feature_extractor.0.weight'].shape[1]
        
        self.simple_model = SimpleStudentModel(
            input_dim=input_dim,
            hidden_dims=[hidden_dim_0, hidden_dim_1, hidden_dim_2],
            dropout=0.3,
        )
        
        self.simple_model.load_state_dict(state_dict)
        
        self.simple_model.to(self.device)
        self.simple_model.eval()
        
        print(f"Loading temporal model from {temporal_model_path}...")
        temporal_ckpt = torch.load(temporal_model_path, map_location=self.device, weights_only=False)
        
        temporal_args = temporal_ckpt.get('args', {})
        self.temporal_model = TemporalViolenceModel(
            input_dim=temporal_args.get('input_dim', 3584),
            hidden_dim=temporal_args.get('hidden_dim', 256),
            lstm_hidden=temporal_args.get('lstm_hidden', 128),
            num_lstm_layers=temporal_args.get('num_lstm_layers', 2),
            dropout=temporal_args.get('dropout', 0.3),
        )
        self.temporal_model.load_state_dict(temporal_ckpt['model_state_dict'])
        self.temporal_model.to(self.device)
        self.temporal_model.eval()
        
        print(f"Models loaded. Strategy: {strategy}")
    
    @torch.no_grad()
    def score_frame_simple(self, embedding: np.ndarray) -> float:
        x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
        logit = self.simple_model(x)
        return torch.sigmoid(logit).cpu().item()
    
    @torch.no_grad()
    def score_sequence_temporal(self, embeddings: np.ndarray) -> float:
        x = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0).to(self.device)
        logit = self.temporal_model(x)
        return torch.sigmoid(logit).cpu().item()
    
    def combine_predictions(
        self,
        simple_score: float,
        temporal_score: float,
    ) -> Tuple[int, float]:
        """Return (prediction, combined_score) for the configured strategy."""
        if self.strategy == 'and':
            pred = 1 if (simple_score >= self.simple_threshold and 
                        temporal_score >= self.temporal_threshold) else 0
            combined_score = min(simple_score, temporal_score)
        
        elif self.strategy == 'or':
            pred = 1 if (simple_score >= self.simple_threshold or 
                        temporal_score >= self.temporal_threshold) else 0
            combined_score = max(simple_score, temporal_score)
        
        elif self.strategy == 'average':
            combined_score = (simple_score + temporal_score) / 2
            avg_threshold = (self.simple_threshold + self.temporal_threshold) / 2
            pred = 1 if combined_score >= avg_threshold else 0
        
        elif self.strategy == 'weighted':
            combined_score = (
                (1 - self.temporal_weight) * simple_score +
                self.temporal_weight * temporal_score
            )
            weighted_threshold = (
                (1 - self.temporal_weight) * self.simple_threshold +
                self.temporal_weight * self.temporal_threshold
            )
            pred = 1 if combined_score >= weighted_threshold else 0
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return pred, combined_score
    
    def run_simulation(
        self,
        sim_path: str,
        smoothing_window: int = 3,
    ) -> EnsembleResults:
        print("\n" + "="*70)
        print("RUNNING ENSEMBLE SIMULATION")
        print("="*70)
        
        print(f"Loading simulation from {sim_path}...")
        data = np.load(sim_path)
        embeddings = data['embeddings']
        labels = data['anticipation_labels']
        fps = float(data.get('fps', 1.0))
        
        n_frames = len(embeddings)
        print(f"  Frames: {n_frames:,}")
        print(f"  Duration: {n_frames / fps / 3600:.2f} hours")
        print(f"  Strategy: {self.strategy}")
        print(f"  Simple threshold: {self.simple_threshold}")
        print(f"  Temporal threshold: {self.temporal_threshold}")
        print(f"  Smoothing window: {smoothing_window}")
        
        print("\nProcessing frames...")
        start_time = time.time()
        
        all_scores = []
        all_predictions = []
        
        recent_combined_scores = []
        
        for i in tqdm(range(n_frames), desc="Scoring"):
            simple_score = self.score_frame_simple(embeddings[i])
            
            if i >= self.sequence_length - 1:
                seq_start = i - self.sequence_length + 1
                seq = embeddings[seq_start:i+1]
                temporal_score = self.score_sequence_temporal(seq)
            else:
                temporal_score = simple_score
            
            pred, combined_score = self.combine_predictions(simple_score, temporal_score)
            
            recent_combined_scores.append(combined_score)
            if len(recent_combined_scores) > smoothing_window:
                recent_combined_scores.pop(0)
            
            if len(recent_combined_scores) >= smoothing_window:
                if self.strategy == 'and':
                    avg_threshold = (self.simple_threshold + self.temporal_threshold) / 2
                    smoothed_pred = 1 if all(s >= avg_threshold for s in recent_combined_scores) else 0
                else:
                    smoothed_pred = pred
            else:
                smoothed_pred = 0
            
            all_scores.append(combined_score)
            all_predictions.append(smoothed_pred)
        
        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.1f}s ({n_frames/elapsed:.0f} fps)")
        
        return self._compute_metrics(
            np.array(all_scores),
            np.array(all_predictions),
            labels,
            fps,
        )
    
    def _compute_metrics(self, scores, predictions, labels, fps) -> EnsembleResults:
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
        
        return EnsembleResults(
            strategy=self.strategy,
            simple_threshold=self.simple_threshold,
            temporal_threshold=self.temporal_threshold,
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
    parser.add_argument('--simple_model_path', required=True)
    parser.add_argument('--temporal_model_path', required=True)
    parser.add_argument('--prebuilt_sim', required=True)
    parser.add_argument('--simple_threshold', type=float, default=0.6)
    parser.add_argument('--temporal_threshold', type=float, default=0.388)
    parser.add_argument('--strategy', choices=['and', 'or', 'average', 'weighted'], default='and')
    parser.add_argument('--temporal_weight', type=float, default=0.5)
    parser.add_argument('--smoothing_window', type=int, default=3)
    parser.add_argument('--output_dir', default='./simulation_results_ensemble')
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    simulator = EnsembleSimulator(
        simple_model_path=args.simple_model_path,
        temporal_model_path=args.temporal_model_path,
        device=args.device,
        simple_threshold=args.simple_threshold,
        temporal_threshold=args.temporal_threshold,
        strategy=args.strategy,
        temporal_weight=args.temporal_weight,
    )
    
    results = simulator.run_simulation(
        args.prebuilt_sim,
        smoothing_window=args.smoothing_window,
    )
    
    results.print_summary()
    
    with open(Path(args.output_dir) / 'results.json', 'w') as f:
        json.dump({
            'strategy': results.strategy,
            'simple_threshold': results.simple_threshold,
            'temporal_threshold': results.temporal_threshold,
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