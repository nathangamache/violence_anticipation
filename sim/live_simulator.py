#!/usr/bin/env python3
"""Replay a simulated stream through the single-frame VLMStudentModel and report alarm metrics.

Usage:
    python sim/live_simulator.py --model_path runs/outputs_xd/best_model.pt \
        --prebuilt_sim simulation_data/xd_violence_8hr_sim.npz --threshold 0.6 --output_dir simulation_results/final
    python sim/live_simulator.py --model_path runs/outputs_xd/best_model.pt --synthetic --test_index test_index.json
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping plots")

from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)


# Must match training/train_student.py::VLMStudentModel.
class VLMStudentModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 3584,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.3,
        use_temporal: bool = False,
        temporal_hidden: int = 128,
    ):
        super().__init__()
        self.use_temporal = use_temporal
        
        layers = []
        prev_dim = input_dim
        
        for hdim in hidden_dims:
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return logits


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


def load_labels(path: Path) -> Optional[np.ndarray]:
    try:
        if path.suffix == '.npz':
            data = np.load(path)
            for key in ['labels', 'label', 'arr_0']:
                if key in data.keys():
                    return data[key]
            if data.keys():
                return data[list(data.keys())[0]]
        else:
            return np.load(path)
    except Exception as e:
        print(f"Error loading labels {path}: {e}")
    return None


def extract_video_name(item) -> str:
    if isinstance(item, dict):
        path = item.get('graph_path') or item.get('label_path', '')
        return Path(path).stem
    return str(item)


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
        print("LIVE SIMULATION RESULTS")
        print("="*70)
        
        print(f"\nStream Statistics:")
        print(f"  Total duration: {self.total_duration_sec/60:.1f} minutes")
        print(f"  Frames processed: {len(self.timestamps)}")
        print(f"  Processing FPS: {self.fps_processed:.1f}")
        
        print(f"\nClass Distribution:")
        print(f"  Non-violent frames: {sum(1 for g in self.ground_truth if g == 0)} ({100*sum(1 for g in self.ground_truth if g == 0)/len(self.ground_truth):.1f}%)")
        print(f"  Violent frames: {sum(1 for g in self.ground_truth if g == 1)} ({100*sum(1 for g in self.ground_truth if g == 1)/len(self.ground_truth):.1f}%)")
        
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


class LiveSimulator:
    """Frame-by-frame replay of a single-frame VLMStudentModel over a simulated stream."""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda:1',
        threshold: float = 0.5,
        horizon_sec: float = 30.0,
        fps: float = 1.0,
        smoothing_window: int = 1,
        alert_cooldown: int = 0,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.horizon_sec = horizon_sec
        self.fps = fps
        self.smoothing_window = smoothing_window
        self.alert_cooldown = alert_cooldown
        
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model_args = checkpoint.get('args', {})
        input_dim = model_args.get('input_dim', 3584)
        hidden_dims = model_args.get('hidden_dims', [512, 256, 128])
        dropout = model_args.get('dropout', 0.3)
        
        self.model = VLMStudentModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded. Device: {self.device}")
    
    def process_frame(self, embedding: np.ndarray) -> Tuple[float, int]:
        with torch.no_grad():
            x = torch.tensor(embedding, dtype=torch.float32).to(self.device)
            logit = self.model(x)
            score = torch.sigmoid(logit).cpu().item()
            prediction = 1 if score >= self.threshold else 0
        return score, prediction
    
    def run_synthetic_simulation(
        self,
        embeddings_dir: Path,
        labels_dir: Path,
        test_index_path: str,
        source_filter: Optional[str] = 'xd_violence',
        dilute_ratio: float = 10.0,
    ) -> SimulationResults:
        """Build a stream from per-video test embeddings, diluted with whole non-violent videos, and replay it."""
        print("\n" + "="*70)
        print("SYNTHETIC LIVE SIMULATION")
        print("="*70)
        
        with open(test_index_path) as f:
            test_items = json.load(f)
        
        print(f"Loaded {len(test_items)} test items")
        
        all_frames = []
        
        for item in test_items:
            video_name = extract_video_name(item)
            
            if source_filter and classify_video_source(video_name) != source_filter:
                continue
            
            emb_path = find_embedding_file(video_name, embeddings_dir)
            label_path = find_label_file(video_name, labels_dir)
            
            if emb_path is None or label_path is None:
                continue
            
            embeddings = load_embedding(emb_path)
            labels = load_labels(label_path)
            
            if embeddings is None or labels is None:
                continue
            
            if labels.ndim > 1:
                labels = labels.max(axis=1)
            labels = (labels > 0).astype(int)
            
            horizon_frames = int(self.horizon_sec * self.fps)
            min_len = min(len(embeddings), len(labels))
            
            for t in range(min_len):
                future_end = min(t + horizon_frames, min_len)
                ant_label = int(labels[t:future_end].max()) if future_end > t else 0
                
                all_frames.append({
                    'embedding': embeddings[t],
                    'label': ant_label,
                    'video_id': video_name,
                    'frame_idx': t,
                })
        
        print(f"Collected {len(all_frames)} frames")
        
        violent_frames = [f for f in all_frames if f['label'] == 1]
        nonviolent_frames = [f for f in all_frames if f['label'] == 0]
        
        print(f"  Violent: {len(violent_frames)}")
        print(f"  Non-violent: {len(nonviolent_frames)}")
        
        if dilute_ratio > 0:
            video_frames = defaultdict(list)
            for f in all_frames:
                video_frames[f['video_id']].append(f)
            
            violent_videos = []
            nonviolent_videos = []
            
            for vid, frames in video_frames.items():
                has_violence = any(f['label'] == 1 for f in frames)
                if has_violence:
                    violent_videos.append((vid, frames))
                else:
                    nonviolent_videos.append((vid, frames))
            
            violent_frame_count = sum(len(frames) for _, frames in violent_videos)
            target_nonviolent = int(violent_frame_count * dilute_ratio)
            
            selected_nonviolent = []
            current_count = 0
            np.random.shuffle(nonviolent_videos)
            
            for vid, frames in nonviolent_videos:
                if current_count >= target_nonviolent:
                    break
                selected_nonviolent.append((vid, frames))
                current_count += len(frames)
            
            while current_count < target_nonviolent and nonviolent_videos:
                vid, frames = nonviolent_videos[np.random.randint(len(nonviolent_videos))]
                selected_nonviolent.append((vid, frames))
                current_count += len(frames)
            
            # Shuffle whole videos so frame order within each video is preserved.
            all_videos = violent_videos + selected_nonviolent
            np.random.shuffle(all_videos)
            
            simulation_frames = []
            for vid, frames in all_videos:
                frames_sorted = sorted(frames, key=lambda x: x['frame_idx'])
                simulation_frames.extend(frames_sorted)
            
            n_violent = sum(1 for f in simulation_frames if f['label'] == 1)
            n_nonviolent = len(simulation_frames) - n_violent
            print(f"  After dilution: {len(simulation_frames)} frames ({n_violent} violent, {n_nonviolent} non-violent)")
            print(f"  Preserved temporal order within {len(all_videos)} videos")
        else:
            simulation_frames = all_frames
        
        print("\nRunning simulation...")
        if self.smoothing_window > 1:
            print(f"  Temporal smoothing: {self.smoothing_window} consecutive frames required")
        if self.alert_cooldown > 0:
            print(f"  Alert cooldown: {self.alert_cooldown} frames")
        
        start_time = time.time()
        
        timestamps = []
        scores = []
        predictions = []
        ground_truth = []
        
        recent_scores = []
        frames_since_alert = self.alert_cooldown  # Allow immediate first alert
        
        for i, frame in enumerate(simulation_frames):
            t = i / self.fps
            score, _ = self.process_frame(frame['embedding'])
            
            recent_scores.append(score)
            if len(recent_scores) > self.smoothing_window:
                recent_scores.pop(0)
            
            if len(recent_scores) >= self.smoothing_window:
                smoothed_above = all(s >= self.threshold for s in recent_scores)
            else:
                smoothed_above = False
            
            if self.alert_cooldown > 0:
                if smoothed_above and frames_since_alert >= self.alert_cooldown:
                    pred = 1
                    frames_since_alert = 0
                else:
                    pred = 0
                    frames_since_alert += 1
            else:
                pred = 1 if smoothed_above else 0
            
            timestamps.append(t)
            scores.append(score)
            predictions.append(pred)
            ground_truth.append(frame['label'])
            
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1}/{len(simulation_frames)} frames...")
        
        elapsed = time.time() - start_time
        
        return self._compute_results(
            timestamps, scores, predictions, ground_truth, elapsed
        )
    
    def _compute_results(
        self,
        timestamps: List[float],
        scores: List[float],
        predictions: List[int],
        ground_truth: List[int],
        elapsed_time: float,
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
        
        detected_events = 0
        lead_times = []
        
        for start, end in violence_events:
            check_start = max(0, start - int(self.horizon_sec * self.fps))
            event_predictions = predictions[check_start:end + 1]
            
            if any(p == 1 for p in event_predictions):
                detected_events += 1
                for i, p in enumerate(event_predictions):
                    if p == 1:
                        lead_time = (start - check_start - i) / self.fps
                        if lead_time > 0:
                            lead_times.append(lead_time)
                        break
        
        false_alarms = 0
        in_alarm = False
        
        for i, (pred, label) in enumerate(zip(predictions, ground_truth)):
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
    
    def run_prebuilt_simulation(self, sim_path: str) -> SimulationResults:
        """Replay a pre-built NPZ holding `embeddings` [N, D] and `anticipation_labels` [N]."""
        print("\n" + "="*70)
        print("RUNNING PRE-BUILT SIMULATION")
        print("="*70)
        
        print(f"Loading simulation from {sim_path}...")
        data = np.load(sim_path)
        
        embeddings = data['embeddings']
        anticipation_labels = data['anticipation_labels']
        fps = float(data.get('fps', 1.0))
        
        print(f"  Frames: {len(embeddings):,}")
        print(f"  Duration: {len(embeddings) / fps / 3600:.2f} hours")
        print(f"  Violence rate: {100 * anticipation_labels.mean():.2f}%")
        
        if self.smoothing_window > 1:
            print(f"  Temporal smoothing: {self.smoothing_window} consecutive frames required")
        if self.alert_cooldown > 0:
            print(f"  Alert cooldown: {self.alert_cooldown} frames")
        
        print("\nProcessing frames...")
        start_time = time.time()
        
        timestamps = []
        scores = []
        predictions = []
        ground_truth = []
        
        recent_scores = []
        frames_since_alert = self.alert_cooldown
        
        for i in range(len(embeddings)):
            t = i / fps
            score, _ = self.process_frame(embeddings[i])
            
            recent_scores.append(score)
            if len(recent_scores) > self.smoothing_window:
                recent_scores.pop(0)
            
            if len(recent_scores) >= self.smoothing_window:
                smoothed_above = all(s >= self.threshold for s in recent_scores)
            else:
                smoothed_above = False
            
            if self.alert_cooldown > 0:
                if smoothed_above and frames_since_alert >= self.alert_cooldown:
                    pred = 1
                    frames_since_alert = 0
                else:
                    pred = 0
                    frames_since_alert += 1
            else:
                pred = 1 if smoothed_above else 0
            
            timestamps.append(t)
            scores.append(score)
            predictions.append(pred)
            ground_truth.append(int(anticipation_labels[i]))
            
            if (i + 1) % 5000 == 0:
                print(f"  Processed {i+1:,}/{len(embeddings):,} frames...")
        
        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.1f}s ({len(embeddings)/elapsed:.0f} fps)")
        
        return self._compute_results(
            timestamps, scores, predictions, ground_truth, elapsed
        )
    
    def find_optimal_threshold(
        self,
        embeddings_dir: Path,
        labels_dir: Path,
        test_index_path: str,
        source_filter: Optional[str] = 'xd_violence',
        target_precision: float = 0.5,
    ) -> float:
        """Return the max-recall threshold with precision >= target_precision, else the best-F1 threshold."""
        with open(test_index_path) as f:
            test_items = json.load(f)
        
        all_scores = []
        all_labels = []
        
        for item in test_items:
            video_name = extract_video_name(item)
            
            if source_filter and classify_video_source(video_name) != source_filter:
                continue
            
            emb_path = find_embedding_file(video_name, embeddings_dir)
            label_path = find_label_file(video_name, labels_dir)
            
            if emb_path is None or label_path is None:
                continue
            
            embeddings = load_embedding(emb_path)
            labels = load_labels(label_path)
            
            if embeddings is None or labels is None:
                continue
            
            if labels.ndim > 1:
                labels = labels.max(axis=1)
            labels = (labels > 0).astype(int)
            
            horizon_frames = int(self.horizon_sec * self.fps)
            min_len = min(len(embeddings), len(labels))
            
            for t in range(min_len):
                future_end = min(t + horizon_frames, min_len)
                ant_label = int(labels[t:future_end].max()) if future_end > t else 0
                
                score, _ = self.process_frame(embeddings[t])
                all_scores.append(score)
                all_labels.append(ant_label)
        
        precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
        
        valid_idx = np.where(precision[:-1] >= target_precision)[0]
        if len(valid_idx) > 0:
            best_idx = valid_idx[np.argmax(recall[:-1][valid_idx])]
            return thresholds[best_idx]
        else:
            f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
            return thresholds[np.argmax(f1_scores[:-1])]


def plot_results(results: SimulationResults, output_dir: Path):
    if not HAS_MATPLOTLIB:
        print("Skipping plots (matplotlib not available)")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    scores_neg = [s for s, g in zip(results.scores, results.ground_truth) if g == 0]
    scores_pos = [s for s, g in zip(results.scores, results.ground_truth) if g == 1]
    
    ax.hist(scores_neg, bins=50, alpha=0.5, label='Non-violent', density=True)
    ax.hist(scores_pos, bins=50, alpha=0.5, label='Violent', density=True)
    ax.axvline(results.threshold, color='r', linestyle='--', label=f'Threshold={results.threshold:.2f}')
    ax.set_xlabel('Score')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution')
    ax.legend()
    
    ax = axes[0, 1]
    im = ax.imshow(results.confusion_mat, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Non-violent', 'Violent'])
    ax.set_yticklabels(['Non-violent', 'Violent'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{results.confusion_mat[i, j]}', 
                   ha='center', va='center', fontsize=14)
    
    ax = axes[1, 0]
    sample_size = min(1000, len(results.timestamps))
    indices = np.linspace(0, len(results.timestamps)-1, sample_size, dtype=int)
    
    times = [results.timestamps[i] / 60 for i in indices]
    scores_sample = [results.scores[i] for i in indices]
    gt_sample = [results.ground_truth[i] for i in indices]
    
    ax.plot(times, scores_sample, 'b-', alpha=0.7, label='Score')
    ax.fill_between(times, 0, 1, where=[g == 1 for g in gt_sample], 
                    alpha=0.3, color='red', label='Violence')
    ax.axhline(results.threshold, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Score')
    ax.set_title('Score Timeline (sampled)')
    ax.legend()
    ax.set_ylim(0, 1)
    
    ax = axes[1, 1]
    ax.text(0.5, 0.5, f"ROC-AUC: {results.roc_auc:.3f}\nPR-AUC: {results.pr_auc:.3f}\n\n"
            f"Precision: {results.precision:.3f}\nRecall: {results.recall:.3f}\nF1: {results.f1:.3f}",
            ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_title('Performance Summary')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'simulation_results.png', dpi=150)
    plt.close()
    print(f"\nPlot saved to {output_dir / 'simulation_results.png'}")


def main():
    parser = argparse.ArgumentParser(description="Live Video Simulator")
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='Device to use')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Classification threshold (if None, auto-select)')
    parser.add_argument('--target_precision', type=float, default=0.5,
                        help='Target precision for auto threshold selection')
    parser.add_argument('--horizon_sec', type=float, default=30.0,
                        help='Anticipation horizon in seconds')
    parser.add_argument('--fps', type=float, default=1.0,
                        help='Frame rate')
    
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic simulation with existing embeddings')
    parser.add_argument('--prebuilt_sim', type=str, default=None,
                        help='Path to pre-built simulation NPZ file (from build_long_simulation.py)')
    parser.add_argument('--embeddings_dir', type=str, default='data/processed/vlm_emb',
                        help='Directory containing embeddings')
    parser.add_argument('--labels_dir', type=str, default='data/processed/labels_multih',
                        help='Directory containing labels')
    parser.add_argument('--test_index', type=str, default='test_index.json',
                        help='Test index file')
    parser.add_argument('--source_filter', type=str, default='xd_violence',
                        help='Filter to specific source')
    parser.add_argument('--dilute_ratio', type=float, default=10.0,
                        help='Ratio of non-violent to violent frames (0 to disable)')
    
    parser.add_argument('--output_dir', type=str, default='./simulation_results',
                        help='Output directory')
    parser.add_argument('--smoothing_window', type=int, default=1,
                        help='Require N consecutive frames above threshold to trigger alert (1=disabled)')
    parser.add_argument('--alert_cooldown', type=int, default=0,
                        help='Minimum frames between alerts (0=disabled)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = Path(args.output_dir)
    
    simulator = LiveSimulator(
        model_path=args.model_path,
        device=args.device,
        threshold=args.threshold or 0.5,
        horizon_sec=args.horizon_sec,
        fps=args.fps,
        smoothing_window=args.smoothing_window,
        alert_cooldown=args.alert_cooldown,
    )
    
    if args.threshold is None and args.synthetic:
        print(f"\nAuto-selecting threshold for {args.target_precision:.0%} precision...")
        threshold = simulator.find_optimal_threshold(
            embeddings_dir=Path(args.embeddings_dir),
            labels_dir=Path(args.labels_dir),
            test_index_path=args.test_index,
            source_filter=args.source_filter,
            target_precision=args.target_precision,
        )
        print(f"Selected threshold: {threshold:.3f}")
        simulator.threshold = threshold
    
    if args.prebuilt_sim:
        results = simulator.run_prebuilt_simulation(args.prebuilt_sim)
    elif args.synthetic:
        results = simulator.run_synthetic_simulation(
            embeddings_dir=Path(args.embeddings_dir),
            labels_dir=Path(args.labels_dir),
            test_index_path=args.test_index,
            source_filter=args.source_filter,
            dilute_ratio=args.dilute_ratio,
        )
    else:
        print("Use --prebuilt_sim or --synthetic mode")
        return
    
    results.print_summary()
    
    plot_results(results, output_dir)
    
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
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()