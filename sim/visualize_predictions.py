#!/usr/bin/env python3
"""
Video Visualization for Violence Anticipation Model

Creates a video with overlaid model predictions showing:
- Real-time violence probability score (0-100%)
- Alert status (green/yellow/red)
- Ground truth comparison
- Lead time when alert is triggered
- Score history graph

Usage:
    python visualize_predictions.py \
        --model_path outputs_temporal_violence/best_temporal_model.pt \
        --video_path data/raw_videos/xd_violence/some_video.mp4 \
        --embeddings_path data/processed/vlm_emb_violence/xd_violence/some_video.npz \
        --labels_path data/processed/labels_multih/some_video.npy \
        --output_path visualizations/some_video_predictions.mp4 \
        --threshold 0.388 \
        --device cuda:0
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from tqdm import tqdm


# ==============================================================================
# Model Definition (must match training)
# ==============================================================================

class TemporalViolenceModel(nn.Module):
    """Temporal LSTM model."""
    
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


# ==============================================================================
# Visualization Class
# ==============================================================================

@dataclass
class FramePrediction:
    frame_idx: int
    timestamp_sec: float
    score: float
    is_alert: bool
    ground_truth: int  # 0=normal, 1=anticipation window
    is_violence_active: int  # 0=no, 1=yes (actual violence happening)


class VideoVisualizer:
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda:0',
        threshold: float = 0.388,
        sequence_length: int = 30,
        horizon_sec: float = 30.0,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.sequence_length = sequence_length
        self.horizon_sec = horizon_sec
        
        # Load model
        print(f"Loading model from {model_path}...")
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
    
    @torch.no_grad()
    def predict_sequence(self, embeddings: np.ndarray) -> float:
        """Predict violence probability for a sequence of embeddings."""
        x = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0).to(self.device)
        logit = self.model(x)
        return torch.sigmoid(logit).cpu().item()
    
    def load_data(
        self,
        embeddings_path: str,
        labels_path: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load embeddings and labels."""
        # Load embeddings
        emb_data = np.load(embeddings_path)
        embeddings = emb_data['emb']
        emb_timesteps = emb_data['t']
        
        # Convert float16 to float32 if needed
        if embeddings.dtype == np.float16:
            embeddings = embeddings.astype(np.float32)
        
        # Load labels
        labels = np.load(labels_path)
        if labels.ndim > 1:
            labels = labels.max(axis=1)
        labels = (labels > 0).astype(np.float32)
        
        return embeddings, emb_timesteps, labels
    
    def compute_predictions(
        self,
        embeddings: np.ndarray,
        emb_timesteps: np.ndarray,
        labels: np.ndarray,
        video_fps: float,
        emb_fps: float = 1.0,
    ) -> List[FramePrediction]:
        """Compute predictions for all frames."""
        predictions = []
        n_emb = len(embeddings)
        
        # Score history for smoothing visualization
        score_history = []
        
        print("Computing predictions...")
        for i in tqdm(range(n_emb)):
            # Get sequence ending at current frame
            if i >= self.sequence_length - 1:
                seq_start = i - self.sequence_length + 1
                seq = embeddings[seq_start:i+1]
                score = self.predict_sequence(seq)
            else:
                # Not enough frames yet
                score = 0.0
            
            score_history.append(score)
            
            # Determine alert status
            is_alert = score >= self.threshold
            
            # Get ground truth for this timestep
            t = emb_timesteps[i]
            
            # Compute anticipation label (is violence coming in next horizon_sec?)
            horizon_frames = int(self.horizon_sec * emb_fps)
            future_end = min(t + horizon_frames, len(labels))
            anticipation_label = int(labels[t:future_end].max()) if future_end > t else 0
            
            # Is violence currently active?
            violence_active = int(labels[t]) if t < len(labels) else 0
            
            # Timestamp
            timestamp = t / emb_fps
            
            predictions.append(FramePrediction(
                frame_idx=i,
                timestamp_sec=timestamp,
                score=score,
                is_alert=is_alert,
                ground_truth=anticipation_label,
                is_violence_active=violence_active,
            ))
        
        return predictions
    
    def create_visualization(
        self,
        video_path: str,
        embeddings_path: str,
        labels_path: str,
        output_path: str,
        show_graph: bool = True,
        graph_history: int = 60,  # seconds of history to show
    ):
        """Create visualization video with overlaid predictions."""
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {video_width}x{video_height} @ {video_fps:.1f}fps, {total_frames} frames")
        
        # Load data and compute predictions
        embeddings, emb_timesteps, labels = self.load_data(embeddings_path, labels_path)
        
        # Embedding FPS (typically 1 fps based on every_t=5 and video processing)
        if len(emb_timesteps) > 1:
            emb_fps = 1.0 / np.mean(np.diff(emb_timesteps))
        else:
            emb_fps = 1.0
        
        predictions = self.compute_predictions(
            embeddings, emb_timesteps, labels, video_fps, emb_fps
        )
        
        # Create prediction lookup by timestep
        pred_by_t = {emb_timesteps[p.frame_idx]: p for p in predictions}
        
        # Setup output video
        # Add space for graph at bottom
        graph_height = 150 if show_graph else 0
        output_height = video_height + graph_height
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, video_fps, (video_width, output_height))
        
        print(f"Creating visualization: {output_path}")
        
        # Process frames
        score_history = []
        gt_history = []
        time_history = []
        
        for frame_idx in tqdm(range(total_frames), desc="Rendering"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Current time in seconds
            current_time = frame_idx / video_fps
            
            # Find closest embedding timestep
            closest_t = min(emb_timesteps, key=lambda t: abs(t - current_time))
            pred = pred_by_t.get(closest_t)
            
            if pred is not None:
                score = pred.score
                is_alert = pred.is_alert
                ground_truth = pred.ground_truth
                violence_active = pred.is_violence_active
            else:
                score = 0.0
                is_alert = False
                ground_truth = 0
                violence_active = 0
            
            # Update history
            score_history.append(score)
            gt_history.append(ground_truth)
            time_history.append(current_time)
            
            # Trim history
            max_history = int(graph_history * video_fps)
            if len(score_history) > max_history:
                score_history = score_history[-max_history:]
                gt_history = gt_history[-max_history:]
                time_history = time_history[-max_history:]
            
            # Draw overlays on frame
            frame = self._draw_overlays(
                frame, score, is_alert, ground_truth, violence_active, current_time
            )
            
            # Add graph if enabled
            if show_graph:
                graph = self._draw_graph(
                    video_width, graph_height,
                    score_history, gt_history, time_history
                )
                frame = np.vstack([frame, graph])
            
            out.write(frame)
        
        cap.release()
        out.release()
        
        print(f"Visualization saved to: {output_path}")
        
        # Also save predictions as JSON
        json_path = output_path.replace('.mp4', '_predictions.json')
        self._save_predictions_json(predictions, json_path)
    
    def _draw_overlays(
        self,
        frame: np.ndarray,
        score: float,
        is_alert: bool,
        ground_truth: int,
        violence_active: int,
        current_time: float,
    ) -> np.ndarray:
        """Draw prediction overlays on frame."""
        h, w = frame.shape[:2]
        
        # Colors
        if violence_active:
            border_color = (0, 0, 255)  # Red - violence happening
            status_text = "VIOLENCE ACTIVE"
        elif is_alert:
            border_color = (0, 165, 255)  # Orange - alert
            status_text = "ALERT: Violence Predicted"
        elif ground_truth:
            border_color = (0, 255, 255)  # Yellow - in anticipation window
            status_text = "Anticipation Window"
        else:
            border_color = (0, 255, 0)  # Green - normal
            status_text = "Normal"
        
        # Draw border
        thickness = 8
        cv2.rectangle(frame, (0, 0), (w-1, h-1), border_color, thickness)
        
        # Semi-transparent overlay for text background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 140), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Score bar
        bar_x, bar_y = 20, 30
        bar_width, bar_height = 200, 25
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Score fill
        fill_width = int(bar_width * score)
        if score >= self.threshold:
            fill_color = (0, 0, 255)  # Red
        elif score >= self.threshold * 0.7:
            fill_color = (0, 165, 255)  # Orange
        else:
            fill_color = (0, 255, 0)  # Green
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), fill_color, -1)
        
        # Score text
        cv2.putText(frame, f"Violence Score: {score*100:.1f}%", (bar_x + bar_width + 10, bar_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Threshold line
        thresh_x = bar_x + int(bar_width * self.threshold)
        cv2.line(frame, (thresh_x, bar_y - 5), (thresh_x, bar_y + bar_height + 5), (255, 255, 255), 2)
        
        # Status text
        cv2.putText(frame, status_text, (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, border_color, 2)
        
        # Time
        minutes = int(current_time // 60)
        seconds = current_time % 60
        cv2.putText(frame, f"Time: {minutes:02d}:{seconds:05.2f}", (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Threshold indicator
        cv2.putText(frame, f"Threshold: {self.threshold*100:.1f}%", (230, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Ground truth indicator (top right)
        gt_text = "GT: Violence" if ground_truth else "GT: Normal"
        gt_color = (0, 0, 255) if ground_truth else (0, 255, 0)
        cv2.putText(frame, gt_text, (w - 180, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, gt_color, 2)
        
        return frame
    
    def _draw_graph(
        self,
        width: int,
        height: int,
        score_history: List[float],
        gt_history: List[int],
        time_history: List[float],
    ) -> np.ndarray:
        """Draw score history graph."""
        graph = np.zeros((height, width, 3), dtype=np.uint8)
        graph[:] = (30, 30, 30)  # Dark gray background
        
        if len(score_history) < 2:
            return graph
        
        # Margins
        margin_left = 60
        margin_right = 20
        margin_top = 20
        margin_bottom = 30
        
        graph_width = width - margin_left - margin_right
        graph_height = height - margin_top - margin_bottom
        
        # Draw grid
        for i in range(5):
            y = margin_top + int(graph_height * i / 4)
            cv2.line(graph, (margin_left, y), (width - margin_right, y), (60, 60, 60), 1)
            label = f"{100 - i*25}%"
            cv2.putText(graph, label, (5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Draw threshold line
        thresh_y = margin_top + int(graph_height * (1 - self.threshold))
        cv2.line(graph, (margin_left, thresh_y), (width - margin_right, thresh_y), (0, 0, 255), 2)
        cv2.putText(graph, "Threshold", (margin_left + 5, thresh_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Draw ground truth regions
        n_points = len(score_history)
        for i in range(n_points - 1):
            if gt_history[i]:
                x1 = margin_left + int(graph_width * i / n_points)
                x2 = margin_left + int(graph_width * (i + 1) / n_points)
                cv2.rectangle(graph, (x1, margin_top), (x2, margin_top + graph_height),
                             (40, 40, 80), -1)  # Dark blue for GT regions
        
        # Draw score line
        points = []
        for i, score in enumerate(score_history):
            x = margin_left + int(graph_width * i / n_points)
            y = margin_top + int(graph_height * (1 - score))
            points.append((x, y))
        
        for i in range(len(points) - 1):
            color = (0, 255, 0) if score_history[i] < self.threshold else (0, 0, 255)
            cv2.line(graph, points[i], points[i+1], color, 2)
        
        # Labels
        cv2.putText(graph, "Score History", (margin_left, margin_top - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Time labels
        if time_history:
            start_time = time_history[0]
            end_time = time_history[-1]
            cv2.putText(graph, f"{start_time:.0f}s", (margin_left, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            cv2.putText(graph, f"{end_time:.0f}s", (width - margin_right - 30, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Legend
        cv2.rectangle(graph, (width - 150, margin_top), (width - margin_right, margin_top + 50), (50, 50, 50), -1)
        cv2.putText(graph, "GT Window", (width - 145, margin_top + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 180), 1)
        cv2.putText(graph, "Score", (width - 145, margin_top + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        cv2.putText(graph, "Alert", (width - 145, margin_top + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        
        return graph
    
    def _save_predictions_json(self, predictions: List[FramePrediction], path: str):
        """Save predictions to JSON file."""
        data = {
            'threshold': self.threshold,
            'sequence_length': self.sequence_length,
            'predictions': [
                {
                    'frame_idx': p.frame_idx,
                    'timestamp_sec': p.timestamp_sec,
                    'score': p.score,
                    'is_alert': p.is_alert,
                    'ground_truth': p.ground_truth,
                    'is_violence_active': p.is_violence_active,
                }
                for p in predictions
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Predictions saved to: {path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize Violence Anticipation Predictions")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained temporal model')
    parser.add_argument('--video_path', type=str, required=True,
                        help='Path to input video')
    parser.add_argument('--embeddings_path', type=str, required=True,
                        help='Path to embeddings .npz file')
    parser.add_argument('--labels_path', type=str, required=True,
                        help='Path to labels .npy file')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path for output video')
    parser.add_argument('--threshold', type=float, default=0.388,
                        help='Alert threshold')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--no_graph', action='store_true',
                        help='Disable score history graph')
    parser.add_argument('--graph_history', type=int, default=60,
                        help='Seconds of history to show in graph')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    
    # Create visualizer
    visualizer = VideoVisualizer(
        model_path=args.model_path,
        device=args.device,
        threshold=args.threshold,
    )
    
    # Create visualization
    visualizer.create_visualization(
        video_path=args.video_path,
        embeddings_path=args.embeddings_path,
        labels_path=args.labels_path,
        output_path=args.output_path,
        show_graph=not args.no_graph,
        graph_history=args.graph_history,
    )


if __name__ == '__main__':
    main()