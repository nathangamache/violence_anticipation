#!/usr/bin/env python3
"""Evaluate the temporal LSTM on a simulation stream with consecutive-frame smoothing and alert cooldown.

Usage (add --sweep to grid over threshold, window and cooldown):
    python scripts/evaluate_with_smoothing.py --model_path outputs/outputs_temporal/best_temporal_model.pt \
        --sim_path simulation_data/xd_violence_vlm_matched.npz --threshold 0.3 --smoothing_window 3 \
        --alert_cooldown 30 --output_dir ./final/smoothing_test --device cuda:0
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class TemporalViolenceModel(nn.Module):
    """Temporal LSTM model matching train_temporal_model.py architecture."""
    
    def __init__(
        self,
        input_dim: int = 3584,
        hidden_dim: int = 256,
        lstm_hidden: int = 128,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.frame_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True,
        )
        
        lstm_output_dim = lstm_hidden * 2
        
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        
        x_flat = x.view(B * T, D)
        encoded = self.frame_encoder(x_flat)
        encoded = encoded.view(B, T, -1)
        
        lstm_out, _ = self.lstm(encoded)
        
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)
        
        logits = self.classifier(context).squeeze(-1)
        return logits


def load_model(model_path: str, device: str) -> Tuple[nn.Module, dict]:
    """Load model from checkpoint, auto-detecting architecture."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    input_dim = state_dict['frame_encoder.0.weight'].shape[1]
    hidden_dim = state_dict['frame_encoder.0.weight'].shape[0]
    lstm_hidden = state_dict['lstm.weight_hh_l0'].shape[1]
    num_layers = sum(1 for k in state_dict.keys() if 'lstm.weight_ih_l' in k and 'reverse' not in k)
    
    model = TemporalViolenceModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        lstm_hidden=lstm_hidden,
        num_lstm_layers=num_layers,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    config = {
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'lstm_hidden': lstm_hidden,
        'num_layers': num_layers,
        'sequence_length': checkpoint.get('args', {}).get('sequence_length', 30),
    }
    
    return model, config


def run_inference(
    model: nn.Module,
    embeddings: np.ndarray,
    sequence_length: int = 30,
    device: str = 'cuda:0',
    batch_size: int = 256,
) -> np.ndarray:
    """Sigmoid score per frame from its trailing sequence_length window (zero-padded at the start)."""
    model.eval()
    n_frames = len(embeddings)
    scores = np.zeros(n_frames, dtype=np.float32)
    
    pad_size = sequence_length - 1
    padded = np.vstack([
        np.zeros((pad_size, embeddings.shape[1]), dtype=np.float32),
        embeddings
    ])
    
    with torch.no_grad():
        for i in tqdm(range(0, n_frames, batch_size), desc="Scoring"):
            batch_end = min(i + batch_size, n_frames)
            batch_seqs = []
            
            for j in range(i, batch_end):
                seq = padded[j:j + sequence_length]
                batch_seqs.append(seq)
            
            batch = torch.tensor(np.array(batch_seqs), dtype=torch.float32, device=device)
            logits = model(batch)
            batch_scores = torch.sigmoid(logits).cpu().numpy()
            scores[i:batch_end] = batch_scores
    
    return scores


def apply_smoothing(scores: np.ndarray, threshold: float, window_size: int) -> np.ndarray:
    """Binary predictions: 1 once window_size consecutive scores reach threshold."""
    n = len(scores)
    predictions = np.zeros(n, dtype=np.int32)
    
    consecutive = 0
    
    for i in range(n):
        if scores[i] >= threshold:
            consecutive += 1
            if consecutive >= window_size:
                predictions[i] = 1
        else:
            consecutive = 0
    
    return predictions


def apply_cooldown(predictions: np.ndarray, cooldown_frames: int) -> np.ndarray:
    """After an alert, suppress further alerts for cooldown_frames."""
    result = np.zeros_like(predictions)
    last_alert = -cooldown_frames - 1
    
    for i in range(len(predictions)):
        if predictions[i] > 0 and (i - last_alert) > cooldown_frames:
            result[i] = 1
            last_alert = i
    
    return result


def find_violence_events(frame_labels: np.ndarray) -> List[Tuple[int, int]]:
    """Inclusive (start, end) runs of positive frame_labels."""
    events = []
    in_event = False
    event_start = 0
    
    for i, label in enumerate(frame_labels):
        if label > 0 and not in_event:
            in_event = True
            event_start = i
        elif label == 0 and in_event:
            in_event = False
            events.append((event_start, i - 1))
    
    if in_event:
        events.append((event_start, len(frame_labels) - 1))
    
    return events


def evaluate(
    scores: np.ndarray,
    predictions: np.ndarray,
    frame_labels: np.ndarray,
    anticipation_labels: np.ndarray,
    fps: float = 1.0,
    horizon_sec: float = 30.0,
) -> Dict:
    from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
    
    try:
        roc_auc = roc_auc_score(anticipation_labels, scores)
    except:
        roc_auc = 0.5
    
    try:
        pr_auc = average_precision_score(anticipation_labels, scores)
    except:
        pr_auc = 0.0
    
    cm = confusion_matrix(anticipation_labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    events = find_violence_events(frame_labels)
    horizon_frames = int(horizon_sec * fps)
    
    detected_events = 0
    lead_times = []
    
    for start, end in events:
        check_start = max(0, start - horizon_frames)
        window_preds = predictions[check_start:end + 1]
        
        if window_preds.max() > 0:
            detected_events += 1
            first_det = np.argmax(window_preds > 0)
            lead_time = max(0, start - (check_start + first_det)) / fps
            lead_times.append(lead_time)
    
    # false alarm = rising edge of predictions outside anticipation_labels
    false_alarms = 0
    alert_active = False
    
    for i in range(len(predictions)):
        if predictions[i] > 0:
            if not alert_active and anticipation_labels[i] == 0:
                false_alarms += 1
            alert_active = True
        else:
            alert_active = False
    
    total_hours = len(predictions) / fps / 3600
    
    return {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'n_events': len(events),
        'detected_events': detected_events,
        'detection_rate': detected_events / len(events) if events else 0,
        'false_alarms': false_alarms,
        'fa_per_hour': false_alarms / total_hours if total_hours > 0 else 0,
        'avg_lead_time_sec': float(np.mean(lead_times)) if lead_times else 0,
        'lead_times': lead_times,
        'total_hours': total_hours,
    }


def sweep_parameters(
    scores: np.ndarray,
    frame_labels: np.ndarray,
    anticipation_labels: np.ndarray,
    fps: float = 1.0,
    thresholds: List[float] = None,
    smoothing_windows: List[int] = None,
    cooldowns: List[int] = None,
) -> List[Dict]:
    """Sweep threshold, smoothing, and cooldown parameters."""
    
    if thresholds is None:
        thresholds = [0.2, 0.3, 0.4, 0.5]
    if smoothing_windows is None:
        smoothing_windows = [1, 3, 5, 10]
    if cooldowns is None:
        cooldowns = [0, 30, 60]
    
    results = []
    
    for thresh in thresholds:
        for window in smoothing_windows:
            for cooldown in cooldowns:
                preds = apply_smoothing(scores, thresh, window)
                
                if cooldown > 0:
                    preds = apply_cooldown(preds, cooldown)
                
                metrics = evaluate(scores, preds, frame_labels, anticipation_labels, fps)
                
                results.append({
                    'threshold': thresh,
                    'smoothing_window': window,
                    'cooldown': cooldown,
                    **metrics,
                })
                
                print(f"θ={thresh:.1f}, w={window}, cd={cooldown}: "
                      f"det={metrics['detection_rate']*100:.1f}%, "
                      f"fa/hr={metrics['fa_per_hour']:.1f}, "
                      f"lead={metrics['avg_lead_time_sec']:.1f}s")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate with smoothing")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--sim_path', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--smoothing_window', type=int, default=1)
    parser.add_argument('--alert_cooldown', type=int, default=0)
    parser.add_argument('--sweep', action='store_true', help='Sweep all parameter combinations')
    parser.add_argument('--output_dir', type=str, default='./smoothing_results')
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {args.model_path}...")
    model, config = load_model(args.model_path, args.device)
    print(f"  Input dim: {config['input_dim']}, LSTM hidden: {config['lstm_hidden']}")
    
    print(f"Loading simulation from {args.sim_path}...")
    sim = np.load(args.sim_path, allow_pickle=True)
    embeddings = sim['embeddings']
    frame_labels = sim['frame_labels']
    anticipation_labels = sim['anticipation_labels']
    fps = float(sim.get('fps', 1.0))
    
    print(f"  Frames: {len(embeddings):,}, Duration: {len(embeddings)/fps/3600:.2f}h")
    print(f"  Violence frames: {(frame_labels > 0).sum():,}")
    
    print("\nRunning inference...")
    scores = run_inference(
        model, embeddings,
        sequence_length=config.get('sequence_length', 30),
        device=args.device,
    )
    
    np.save(output_dir / 'scores.npy', scores)
    
    if args.sweep:
        print("\nSweeping parameters...")
        results = sweep_parameters(scores, frame_labels, anticipation_labels, fps)
        
        with open(output_dir / 'sweep_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*70)
        print("BEST CONFIGURATIONS")
        print("="*70)
        
        valid = [r for r in results if r['fa_per_hour'] < 5]
        if valid:
            best = max(valid, key=lambda x: x['detection_rate'])
            print(f"\nBest detection (FA < 5/hr):")
            print(f"  θ={best['threshold']}, w={best['smoothing_window']}, cd={best['cooldown']}")
            print(f"  Detection: {best['detection_rate']*100:.1f}%, FA/hr: {best['fa_per_hour']:.1f}, Lead: {best['avg_lead_time_sec']:.1f}s")
        
        valid = [r for r in results if r['fa_per_hour'] < 1]
        if valid:
            best = max(valid, key=lambda x: x['detection_rate'])
            print(f"\nBest detection (FA < 1/hr):")
            print(f"  θ={best['threshold']}, w={best['smoothing_window']}, cd={best['cooldown']}")
            print(f"  Detection: {best['detection_rate']*100:.1f}%, FA/hr: {best['fa_per_hour']:.1f}, Lead: {best['avg_lead_time_sec']:.1f}s")
        
        valid = [r for r in results if r['detection_rate'] > 0.9]
        if valid:
            best = min(valid, key=lambda x: x['fa_per_hour'])
            print(f"\nLowest FA (detection > 90%):")
            print(f"  θ={best['threshold']}, w={best['smoothing_window']}, cd={best['cooldown']}")
            print(f"  Detection: {best['detection_rate']*100:.1f}%, FA/hr: {best['fa_per_hour']:.1f}, Lead: {best['avg_lead_time_sec']:.1f}s")
    
    else:
        print(f"\nApplying smoothing (window={args.smoothing_window}, cooldown={args.alert_cooldown})...")
        
        predictions = apply_smoothing(scores, args.threshold, args.smoothing_window)
        if args.alert_cooldown > 0:
            predictions = apply_cooldown(predictions, args.alert_cooldown)
        
        metrics = evaluate(scores, predictions, frame_labels, anticipation_labels, fps)
        
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"Threshold: {args.threshold}")
        print(f"Smoothing window: {args.smoothing_window}")
        print(f"Alert cooldown: {args.alert_cooldown}")
        print(f"\nDetection rate: {metrics['detection_rate']*100:.1f}%")
        print(f"False alarms/hour: {metrics['fa_per_hour']:.1f}")
        print(f"Avg lead time: {metrics['avg_lead_time_sec']:.1f}s")
        print(f"Events: {metrics['detected_events']}/{metrics['n_events']}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
        results = {
            'config': {
                'threshold': args.threshold,
                'smoothing_window': args.smoothing_window,
                'alert_cooldown': args.alert_cooldown,
            },
            'metrics': metrics,
        }
        
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()