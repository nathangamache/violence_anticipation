#!/usr/bin/env python3
"""
Live Simulator for Temporal Violence Model

This simulator processes the pre-built simulation using the temporal (LSTM) model
which looks at sequences of frames instead of single frames.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score,
    precision_recall_curve
)


# ==============================================================================
# Temporal Model (must match training)
# ==============================================================================
class TemporalViolenceModel(nn.Module):
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
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
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


# ==============================================================================
# Simulation Results
# ==============================================================================
@dataclass 
class SimulationResults:
    timestamps: List[float]
    scores: List[float]
    predictions: List[int]
    ground_truth: List[int]
    threshold: float
    confusion_mat: np.ndarray
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    total_duration_sec: float
    fps_processed: float
    n_violence_events: int
    n_detected_events: int
    n_false_alarms: int
    avg_lead_time_sec: float
    
    def print_summary(self):
        print("\n" + "="*70)
        print("LIVE SIMULATION RESULTS (TEMPORAL MODEL)")
        print("="*70)
        
        print(f"\nStream Statistics:")
        print(f"  Total duration: {self.total_duration_sec/60:.1f} minutes")
        print(f"  Frames processed: {len(self.timestamps)}")
        print(f"  Processing FPS: {self.fps_processed:.1f}")
        
        n_pos = sum(self.ground_truth)
        n_neg = len(self.ground_truth) - n_pos
        print(f"\nClass Distribution:")
        print(f"  Non-violent frames: {n_neg} ({100*n_neg/len(self.ground_truth):.1f}%)")
        print(f"  Violent frames: {n_pos} ({100*n_pos/len(self.ground_truth):.1f}%)")
        
        print(f"\nThreshold: {self.threshold:.3f}")
        
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                 No    Yes")
        print(f"  Actual No    {self.confusion_mat[0,0]:5d}  {self.confusion_mat[0,1]:5d}")
        print(f"  Actual Yes   {self.confusion_mat[1,0]:5d}  {self.confusion_mat[1,1]:5d}")
        
        print(f"\nPerformance Metrics:")
        print(f"  ROC-AUC:   {self.roc_auc:.4f}")
        print(f"  PR-AUC:    {self.pr_auc:.4f}")
        print(f"  Precision: {self.precision:.4f}")
        print(f"  Recall:    {self.recall:.4f}")
        print(f"  F1 Score:  {self.f1:.4f}")
        
        print(f"\nEvent-Level Analysis:")
        print(f"  Violence events in stream: {self.n_violence_events}")
        print(f"  Events detected: {self.n_detected_events}")
        print(f"  False alarms: {self.n_false_alarms}")
        if self.avg_lead_time_sec > 0:
            print(f"  Average lead time: {self.avg_lead_time_sec:.1f} seconds")
        
        if len(self.timestamps) > 0:
            false_alarm_rate = self.n_false_alarms / (self.total_duration_sec / 3600)
            print(f"\nOperational Metrics:")
            print(f"  False alarms per hour: {false_alarm_rate:.1f}")
            if self.n_violence_events > 0:
                detection_rate = self.n_detected_events / self.n_violence_events
                print(f"  Event detection rate: {detection_rate:.1%}")


# ==============================================================================
# Temporal Simulator
# ==============================================================================
class TemporalSimulator:
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda:1',
        threshold: float = 0.5,
        sequence_length: int = 30,
        fps: float = 1.0,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.sequence_length = sequence_length
        self.fps = fps
        
        # Load model
        print(f"Loading temporal model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        model_args = checkpoint.get('args', {})
        
        self.model = TemporalViolenceModel(
            input_dim=model_args.get('input_dim', 3584),
            hidden_dim=model_args.get('hidden_dim', 256),
            lstm_hidden=model_args.get('lstm_hidden', 128),
            num_lstm_layers=model_args.get('num_lstm_layers', 2),
            dropout=model_args.get('dropout', 0.3),
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded. Device: {self.device}")
        print(f"Sequence length: {self.sequence_length}")
    
    @torch.no_grad()
    def process_sequence(self, embeddings: np.ndarray) -> float:
        """Process a sequence of embeddings and return violence probability."""
        x = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0).to(self.device)
        logit = self.model(x)
        score = torch.sigmoid(logit).cpu().item()
        return score
    
    def run_simulation(self, sim_path: str, stride: int = 1) -> SimulationResults:
        """
        Run simulation on pre-built data using sliding window.
        
        Args:
            sim_path: Path to NPZ file with embeddings and labels
            stride: How many frames to advance between predictions
        """
        print("\n" + "="*70)
        print("RUNNING TEMPORAL MODEL SIMULATION")
        print("="*70)
        
        print(f"Loading simulation from {sim_path}...")
        data = np.load(sim_path)
        
        embeddings = data['embeddings']
        anticipation_labels = data['anticipation_labels']
        fps = float(data.get('fps', 1.0))
        
        n_frames = len(embeddings)
        print(f"  Frames: {n_frames:,}")
        print(f"  Duration: {n_frames / fps / 3600:.2f} hours")
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  Stride: {stride}")
        
        # Process with sliding window
        print("\nProcessing sequences...")
        start_time = time.time()
        
        timestamps = []
        scores = []
        predictions = []
        ground_truth = []
        
        # We'll make a prediction for each position where we have a full sequence
        n_sequences = (n_frames - self.sequence_length) // stride + 1
        
        for i, start_idx in enumerate(range(0, n_frames - self.sequence_length + 1, stride)):
            end_idx = start_idx + self.sequence_length
            
            # Get sequence
            seq = embeddings[start_idx:end_idx]
            
            # Predict
            score = self.process_sequence(seq)
            pred = 1 if score >= self.threshold else 0
            
            # The label for this sequence: is there violence at the end or upcoming?
            # Use the label at the last frame of the sequence
            label = int(anticipation_labels[end_idx - 1])
            
            timestamps.append(end_idx / fps)
            scores.append(score)
            predictions.append(pred)
            ground_truth.append(label)
            
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1:,}/{n_sequences:,} sequences...")
        
        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.1f}s ({n_sequences/elapsed:.0f} seq/s)")
        
        return self._compute_results(timestamps, scores, predictions, ground_truth, elapsed, fps)
    
    def _compute_results(
        self,
        timestamps: List[float],
        scores: List[float],
        predictions: List[int],
        ground_truth: List[int],
        elapsed_time: float,
        fps: float,
    ) -> SimulationResults:
        
        y_true = np.array(ground_truth)
        y_scores = np.array(scores)
        y_pred = np.array(predictions)
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        if len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, y_scores)
            pr_auc = average_precision_score(y_true, y_scores)
        else:
            roc_auc = 0.5
            pr_auc = y_true.mean()
        
        tp = cm[1, 1]
        fp = cm[0, 1]
        fn = cm[1, 0]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Event-level analysis
        violence_events = []
        in_event = False
        event_start = 0
        
        for i, label in enumerate(ground_truth):
            if label == 1 and not in_event:
                in_event = True
                event_start = i
            elif label == 0 and in_event:
                in_event = False
                violence_events.append((event_start, i - 1))
        
        if in_event:
            violence_events.append((event_start, len(ground_truth) - 1))
        
        # Count detected events
        detected_events = 0
        lead_times = []
        horizon_frames = int(30 * fps)  # 30 second horizon
        
        for start, end in violence_events:
            check_start = max(0, start - horizon_frames)
            event_preds = predictions[check_start:end + 1]
            
            if any(p == 1 for p in event_preds):
                detected_events += 1
                for i, p in enumerate(event_preds):
                    if p == 1:
                        lead_time = (start - check_start - i) / fps
                        if lead_time > 0:
                            lead_times.append(lead_time)
                        break
        
        # Count false alarms
        false_alarms = 0
        in_alarm = False
        
        for pred, label in zip(predictions, ground_truth):
            if pred == 1 and label == 0 and not in_alarm:
                false_alarms += 1
                in_alarm = True
            elif pred == 0:
                in_alarm = False
        
        return SimulationResults(
            timestamps=timestamps,
            scores=scores,
            predictions=predictions,
            ground_truth=ground_truth,
            threshold=self.threshold,
            confusion_mat=cm,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            precision=precision,
            recall=recall,
            f1=f1,
            total_duration_sec=timestamps[-1] if timestamps else 0,
            fps_processed=len(timestamps) / elapsed_time if elapsed_time > 0 else 0,
            n_violence_events=len(violence_events),
            n_detected_events=detected_events,
            n_false_alarms=false_alarms,
            avg_lead_time_sec=np.mean(lead_times) if lead_times else 0,
        )
    
    def find_optimal_threshold(self, sim_path: str, target_fa_per_hour: float = 0.5) -> float:
        """Find threshold that achieves target false alarm rate."""
        print(f"\nFinding threshold for {target_fa_per_hour} FA/hr...")
        
        data = np.load(sim_path)
        embeddings = data['embeddings']
        anticipation_labels = data['anticipation_labels']
        fps = float(data.get('fps', 1.0))
        
        n_frames = len(embeddings)
        duration_hours = n_frames / fps / 3600
        
        # Collect all scores
        all_scores = []
        all_labels = []
        
        for start_idx in range(0, n_frames - self.sequence_length + 1, 5):  # stride=5 for speed
            end_idx = start_idx + self.sequence_length
            seq = embeddings[start_idx:end_idx]
            score = self.process_sequence(seq)
            label = int(anticipation_labels[end_idx - 1])
            all_scores.append(score)
            all_labels.append(label)
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        # Find threshold that gives target FA rate
        n_neg = (all_labels == 0).sum()
        target_fp = int(target_fa_per_hour * duration_hours)
        
        # Sort scores and find threshold
        neg_scores = all_scores[all_labels == 0]
        neg_scores_sorted = np.sort(neg_scores)[::-1]  # Descending
        
        if target_fp < len(neg_scores_sorted):
            threshold = neg_scores_sorted[target_fp]
        else:
            threshold = neg_scores_sorted[-1] if len(neg_scores_sorted) > 0 else 0.5
        
        print(f"  Selected threshold: {threshold:.4f}")
        return threshold


def main():
    parser = argparse.ArgumentParser(description="Temporal Model Live Simulator")
    
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--prebuilt_sim', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--target_fa_per_hour', type=float, default=0.5,
                        help='Target false alarms per hour for auto threshold')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for sliding window (1=every frame)')
    parser.add_argument('--output_dir', type=str, default='./simulation_results_temporal')
    parser.add_argument('--device', type=str, default='cuda:1')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize simulator
    simulator = TemporalSimulator(
        model_path=args.model_path,
        device=args.device,
        threshold=args.threshold or 0.5,
    )
    
    # Auto-select threshold if not provided
    if args.threshold is None:
        threshold = simulator.find_optimal_threshold(
            args.prebuilt_sim, 
            target_fa_per_hour=args.target_fa_per_hour
        )
        simulator.threshold = threshold
    
    # Run simulation
    results = simulator.run_simulation(args.prebuilt_sim, stride=args.stride)
    
    # Print results
    results.print_summary()
    
    # Save results
    results_dict = {
        'threshold': results.threshold,
        'roc_auc': results.roc_auc,
        'pr_auc': results.pr_auc,
        'precision': results.precision,
        'recall': results.recall,
        'f1': results.f1,
        'confusion_matrix': results.confusion_mat.tolist(),
        'total_duration_sec': results.total_duration_sec,
        'n_violence_events': results.n_violence_events,
        'n_detected_events': results.n_detected_events,
        'n_false_alarms': results.n_false_alarms,
        'avg_lead_time_sec': results.avg_lead_time_sec,
    }
    
    with open(Path(args.output_dir) / 'results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()