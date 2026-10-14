#!/usr/bin/env python3
"""
Optical Flow Feature Extraction for Violence Detection

Extracts motion-based features that correlate with violence:
1. Flow magnitude (how much movement)
2. Flow variance (chaotic vs uniform motion)
3. Sudden motion changes (acceleration)
4. Directional entropy (random vs coordinated movement)
5. Region-based statistics (where is motion concentrated)

These features capture temporal dynamics that static VLM embeddings miss.

Usage:
    # Process videos matching embeddings (single process)
    python optical_flow_features.py \
        --videos_dir data/videos \
        --embeddings_dir data/processed/vlm_emb \
        --output_dir data/processed/optical_flow \
        --device cuda:0

    # PARALLEL PROCESSING: Run multiple workers
    # Terminal 1:
    python optical_flow_features.py \
        --videos_dir data/videos \
        --embeddings_dir data/processed/vlm_emb \
        --output_dir data/processed/optical_flow \
        --worker_id 0 --num_workers 4

    # Terminal 2:
    python optical_flow_features.py \
        --videos_dir data/videos \
        --embeddings_dir data/processed/vlm_emb \
        --output_dir data/processed/optical_flow \
        --worker_id 1 --num_workers 4

    # ... etc for workers 2 and 3

    # Or use the convenience script to launch all workers:
    python optical_flow_features.py \
        --videos_dir data/videos \
        --embeddings_dir data/processed/vlm_emb \
        --output_dir data/processed/optical_flow \
        --num_workers 4 \
        --launch_parallel
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from tqdm import tqdm
from scipy import ndimage
from collections import deque
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ==============================================================================
# Optical Flow Computation
# ==============================================================================

class OpticalFlowExtractor:
    """
    Extracts optical flow and computes violence-relevant motion features.
    """
    
    def __init__(
        self,
        method: str = 'farneback',  # 'farneback', 'lucas_kanade', 'raft'
        resize_height: int = 256,   # Resize for faster processing
        device: str = 'cuda:0',
    ):
        self.method = method
        self.resize_height = resize_height
        self.device = device
        
        # For temporal smoothing
        self.flow_history = deque(maxlen=5)
        self.magnitude_history = deque(maxlen=10)
        
        # RAFT model (if using deep optical flow)
        self.raft_model = None
        if method == 'raft' and TORCH_AVAILABLE:
            self._load_raft()
    
    def _load_raft(self):
        """Load RAFT optical flow model."""
        try:
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
            self.raft_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
            self.raft_model = self.raft_model.to(self.device).eval()
            print("Loaded RAFT optical flow model")
        except Exception as e:
            print(f"Could not load RAFT: {e}, falling back to Farneback")
            self.method = 'farneback'
    
    def compute_flow_farneback(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
    ) -> np.ndarray:
        """
        Compute dense optical flow using Farneback method.
        
        Returns:
            flow: [H, W, 2] array with (dx, dy) per pixel
        """
        # Convert to grayscale
        if len(prev_frame.shape) == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame
            curr_gray = curr_frame
        
        # Compute flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        return flow
    
    def compute_flow_raft(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
    ) -> np.ndarray:
        """Compute optical flow using RAFT (deep learning method)."""
        import torch
        import torchvision.transforms.functional as F
        
        # Preprocess
        prev_tensor = torch.from_numpy(prev_frame).permute(2, 0, 1).float() / 255.0
        curr_tensor = torch.from_numpy(curr_frame).permute(2, 0, 1).float() / 255.0
        
        prev_tensor = prev_tensor.unsqueeze(0).to(self.device)
        curr_tensor = curr_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            flow = self.raft_model(prev_tensor, curr_tensor)[-1]
        
        # Convert to numpy [H, W, 2]
        flow = flow[0].permute(1, 2, 0).cpu().numpy()
        
        return flow
    
    def compute_flow(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
    ) -> np.ndarray:
        """Compute optical flow using configured method."""
        if self.method == 'raft' and self.raft_model is not None:
            return self.compute_flow_raft(prev_frame, curr_frame)
        else:
            return self.compute_flow_farneback(prev_frame, curr_frame)
    
    def extract_features(self, flow: np.ndarray) -> Dict[str, float]:
        """
        Extract violence-relevant features from optical flow.
        
        Args:
            flow: [H, W, 2] optical flow
            
        Returns:
            Dictionary of feature values
        """
        # Flow components
        dx = flow[:, :, 0]
        dy = flow[:, :, 1]
        
        # Magnitude and angle
        magnitude = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        
        # Store for temporal analysis
        self.magnitude_history.append(magnitude.mean())
        self.flow_history.append(flow.copy())
        
        features = {}
        
        # === Global Motion Statistics ===
        features['magnitude_mean'] = float(magnitude.mean())
        features['magnitude_std'] = float(magnitude.std())
        features['magnitude_max'] = float(magnitude.max())
        features['magnitude_p90'] = float(np.percentile(magnitude, 90))
        features['magnitude_p99'] = float(np.percentile(magnitude, 99))
        
        # Motion intensity (normalized)
        features['motion_intensity'] = float(np.clip(magnitude.mean() / 10.0, 0, 1))
        
        # === Motion Variance (Chaos Indicator) ===
        # High variance = chaotic motion (fighting)
        # Low variance = uniform motion (walking, sports)
        features['flow_variance_x'] = float(dx.var())
        features['flow_variance_y'] = float(dy.var())
        features['flow_variance_total'] = float(dx.var() + dy.var())
        
        # === Directional Analysis ===
        # Entropy of flow directions (high = chaotic, low = coordinated)
        angle_hist, _ = np.histogram(angle.flatten(), bins=8, range=(-np.pi, np.pi))
        angle_hist = angle_hist / (angle_hist.sum() + 1e-8)
        directional_entropy = -np.sum(angle_hist * np.log(angle_hist + 1e-8))
        features['directional_entropy'] = float(directional_entropy)
        
        # Dominant direction strength
        features['dominant_direction_strength'] = float(angle_hist.max())
        
        # === Spatial Distribution ===
        # Divide frame into grid and analyze motion distribution
        h, w = magnitude.shape
        grid_h, grid_w = 3, 3
        cell_h, cell_w = h // grid_h, w // grid_w
        
        grid_magnitudes = []
        for i in range(grid_h):
            for j in range(grid_w):
                cell = magnitude[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                grid_magnitudes.append(cell.mean())
        
        grid_magnitudes = np.array(grid_magnitudes)
        
        # Motion concentration (is motion localized or spread out?)
        features['motion_concentration'] = float(grid_magnitudes.max() / (grid_magnitudes.mean() + 1e-8))
        features['motion_spread'] = float(grid_magnitudes.std())
        
        # Center vs periphery motion
        center_motion = grid_magnitudes[4]  # Center cell
        edge_motion = np.mean([grid_magnitudes[i] for i in [0, 1, 2, 3, 5, 6, 7, 8]])
        features['center_motion_ratio'] = float(center_motion / (edge_motion + 1e-8))
        
        # === Temporal Features (if history available) ===
        if len(self.magnitude_history) >= 2:
            # Acceleration (change in motion)
            mag_diff = self.magnitude_history[-1] - self.magnitude_history[-2]
            features['motion_acceleration'] = float(mag_diff)
        else:
            features['motion_acceleration'] = 0.0
        
        # Motion consistency (always include)
        if len(self.magnitude_history) >= 5:
            recent_mags = list(self.magnitude_history)[-5:]
            features['motion_consistency'] = float(np.std(recent_mags))
        else:
            features['motion_consistency'] = 0.0
        
        # Flow change (if previous flow available)
        if len(self.flow_history) >= 2:
            prev_flow = self.flow_history[-2]
            flow_diff = np.abs(flow - prev_flow)
            features['flow_change_mean'] = float(flow_diff.mean())
            features['flow_change_max'] = float(flow_diff.max())
        else:
            features['flow_change_mean'] = 0.0
            features['flow_change_max'] = 0.0
        
        # === High-Motion Region Analysis ===
        # Find regions with very high motion
        high_motion_mask = magnitude > np.percentile(magnitude, 90)
        features['high_motion_area_ratio'] = float(high_motion_mask.mean())
        
        # Contiguity of high-motion regions
        labeled, num_features = ndimage.label(high_motion_mask)
        features['num_high_motion_regions'] = float(num_features)
        
        # === Violence-Specific Composite Scores ===
        # Sudden chaotic motion (fighting signature)
        features['chaos_score'] = float(
            features['directional_entropy'] * 
            features['motion_intensity'] * 
            (1 + features['motion_acceleration'])
        )
        
        # Impact score (sudden localized high motion)
        features['impact_score'] = float(
            features['magnitude_p99'] * 
            features['motion_concentration'] * 
            (1 + abs(features['motion_acceleration']))
        )
        
        return features
    
    def reset_history(self):
        """Reset temporal history (call between videos)."""
        self.flow_history.clear()
        self.magnitude_history.clear()


def get_feature_names() -> List[str]:
    """Return list of all optical flow feature names."""
    return [
        'magnitude_mean', 'magnitude_std', 'magnitude_max', 
        'magnitude_p90', 'magnitude_p99', 'motion_intensity',
        'flow_variance_x', 'flow_variance_y', 'flow_variance_total',
        'directional_entropy', 'dominant_direction_strength',
        'motion_concentration', 'motion_spread', 'center_motion_ratio',
        'motion_acceleration', 'motion_consistency',
        'flow_change_mean', 'flow_change_max',
        'high_motion_area_ratio', 'num_high_motion_regions',
        'chaos_score', 'impact_score',
    ]


# ==============================================================================
# Video Processing
# ==============================================================================

def find_video_file(video_name: str, videos_dir: Path) -> Optional[Path]:
    """Find video file for a given video name."""
    extensions = ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.MP4', '.AVI']
    
    for ext in extensions:
        path = videos_dir / f"{video_name}{ext}"
        if path.exists():
            return path
    
    for subdir in ['', 'xd_violence', 'ucf_crime', 'XD_Violence', 'UCF_Crime']:
        base = videos_dir / subdir if subdir else videos_dir
        if not base.exists():
            continue
        for ext in extensions:
            path = base / f"{video_name}{ext}"
            if path.exists():
                return path
        matches = list(base.glob(f"**/{video_name}.*"))
        for m in matches:
            if m.suffix.lower() in [e.lower() for e in extensions]:
                return m
    return None


def process_single_video(
    video_path: Path,
    output_path: Path,
    extractor: OpticalFlowExtractor,
    sample_rate: int = 1,
    resize_height: int = 256,
    show_progress: bool = True,
) -> Optional[Dict]:
    """
    Process a single video and extract optical flow features.
    
    Args:
        video_path: Path to video file
        output_path: Path to save output .npz
        extractor: OpticalFlowExtractor instance
        sample_rate: Process every Nth frame
        resize_height: Resize frames for faster processing
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Compute resize dimensions
    scale = resize_height / height
    new_width = int(width * scale)
    new_height = resize_height
    
    print(f"Processing {video_path.name}: {total_frames} frames @ {fps:.1f} fps")
    
    extractor.reset_history()
    
    all_features = []
    frame_indices = []
    
    prev_frame = None
    frame_idx = 0
    
    if show_progress:
        pbar = tqdm(total=total_frames // sample_rate, desc=video_path.stem[:30])
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_rate == 0:
            # Resize
            frame_resized = cv2.resize(frame, (new_width, new_height))
            
            if prev_frame is not None:
                # Compute flow
                flow = extractor.compute_flow(prev_frame, frame_resized)
                
                # Extract features
                features = extractor.extract_features(flow)
                feature_vector = [features[name] for name in get_feature_names()]
                
                all_features.append(feature_vector)
                frame_indices.append(frame_idx)
            
            prev_frame = frame_resized
            if show_progress:
                pbar.update(1)
        
        frame_idx += 1
    
    if show_progress:
        pbar.close()
    cap.release()
    
    if not all_features:
        print(f"No features extracted from {video_path}")
        return None
    
    # Stack and save
    all_features = np.array(all_features, dtype=np.float32)
    frame_indices = np.array(frame_indices)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=all_features,
        frame_indices=frame_indices,
        feature_names=get_feature_names(),
        video_info={
            'total_frames': total_frames,
            'fps': fps,
            'width': width,
            'height': height,
            'sample_rate': sample_rate,
        }
    )
    
    print(f"Saved {len(all_features)} frames to {output_path}")
    
    return {
        'n_frames': len(all_features),
        'feature_dim': all_features.shape[1],
    }


def process_videos_from_embeddings(
    videos_dir: Path,
    embeddings_dir: Path,
    output_dir: Path,
    extractor: OpticalFlowExtractor,
    sample_rate: int = 1,
    source_filter: str = None,
    worker_id: int = 0,
    num_workers: int = 1,
):
    """Process all videos that have embeddings with worker support."""
    
    emb_files = sorted(list(embeddings_dir.rglob("*.npz")))
    
    # Filter by source first
    if source_filter:
        filtered = []
        for emb_path in emb_files:
            video_name = emb_path.stem
            is_xd = '__#' in video_name or video_name.startswith('v=')
            is_ucf = '_x264' in video_name
            if source_filter == 'xd_violence' and is_xd:
                filtered.append(emb_path)
            elif source_filter == 'ucf_crime' and is_ucf:
                filtered.append(emb_path)
        emb_files = filtered
    
    print(f"Found {len(emb_files)} embedding files (after filter)")
    
    # Split work among workers
    if num_workers > 1:
        emb_files = [f for i, f in enumerate(emb_files) if i % num_workers == worker_id]
        print(f"Worker {worker_id}/{num_workers}: processing {len(emb_files)} videos")
    
    processed = 0
    skipped = 0
    not_found = 0
    
    for emb_path in tqdm(emb_files, desc=f"Worker {worker_id}" if num_workers > 1 else "Processing"):
        video_name = emb_path.stem
        
        # Output path
        rel_path = emb_path.relative_to(embeddings_dir)
        output_path = output_dir / rel_path.parent / f"{video_name}.npz"
        
        if output_path.exists():
            skipped += 1
            continue
        
        # Find video
        video_path = find_video_file(video_name, videos_dir)
        if video_path is None:
            not_found += 1
            if not_found <= 3:
                print(f"Video not found: {video_name}")
            continue
        
        # Process
        result = process_single_video(
            video_path, output_path, extractor, sample_rate
        )
        
        if result:
            processed += 1
    
    print(f"\nWorker {worker_id} complete: processed={processed}, skipped={skipped}, not_found={not_found}")


def process_video_task(args):
    """Process a single video (for multiprocessing pool)."""
    video_path, output_path, sample_rate, resize_height = args
    
    try:
        extractor = OpticalFlowExtractor(method='farneback', resize_height=resize_height)
        result = process_single_video(
            Path(video_path), 
            Path(output_path), 
            extractor, 
            sample_rate,
            resize_height,
            show_progress=False,
        )
        return {'status': 'success', 'video': video_path, 'result': result}
    except Exception as e:
        return {'status': 'error', 'video': video_path, 'error': str(e)}


def process_videos_parallel_pool(
    videos_dir: Path,
    embeddings_dir: Path,
    output_dir: Path,
    sample_rate: int = 1,
    source_filter: str = None,
    num_processes: int = 4,
    resize_height: int = 256,
):
    """Process videos using multiprocessing pool (single machine, multiple cores)."""
    
    emb_files = sorted(list(embeddings_dir.rglob("*.npz")))
    
    # Filter and prepare tasks
    tasks = []
    
    for emb_path in emb_files:
        video_name = emb_path.stem
        
        # Source filter
        if source_filter:
            is_xd = '__#' in video_name or video_name.startswith('v=')
            is_ucf = '_x264' in video_name
            if source_filter == 'xd_violence' and not is_xd:
                continue
            if source_filter == 'ucf_crime' and not is_ucf:
                continue
        
        # Output path
        rel_path = emb_path.relative_to(embeddings_dir)
        output_path = output_dir / rel_path.parent / f"{video_name}.npz"
        
        if output_path.exists():
            continue
        
        # Find video
        video_path = find_video_file(video_name, videos_dir)
        if video_path is None:
            continue
        
        tasks.append((str(video_path), str(output_path), sample_rate, resize_height))
    
    print(f"Processing {len(tasks)} videos with {num_processes} processes")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    completed = 0
    errors = 0
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {executor.submit(process_video_task, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing"):
            result = future.result()
            if result['status'] == 'success':
                completed += 1
            else:
                errors += 1
                if errors <= 5:
                    print(f"Error: {result['video']}: {result['error']}")
    
    print(f"\nComplete: {completed} processed, {errors} errors")


def launch_parallel_workers(
    args,
    num_workers: int,
):
    """Launch multiple worker processes."""
    
    processes = []
    
    for worker_id in range(num_workers):
        cmd = [
            sys.executable,
            __file__,
            '--videos_dir', str(args.videos_dir),
            '--embeddings_dir', str(args.embeddings_dir),
            '--output_dir', str(args.output_dir),
            '--sample_rate', str(args.sample_rate),
            '--resize_height', str(args.resize_height),
            '--worker_id', str(worker_id),
            '--num_workers', str(num_workers),
        ]
        
        if args.source_filter:
            cmd.extend(['--source_filter', args.source_filter])
        
        print(f"Launching worker {worker_id}: {' '.join(cmd)}")
        
        # Launch in background
        p = subprocess.Popen(cmd)
        processes.append(p)
    
    print(f"\nLaunched {num_workers} workers. Waiting for completion...")
    
    # Wait for all to complete
    for p in processes:
        p.wait()
    
    print("All workers complete!")


# ==============================================================================
# Feature Analysis
# ==============================================================================

def analyze_features(
    features_dir: Path,
    labels_dir: Path,
    output_path: Path,
):
    """Analyze discriminative power of optical flow features."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    
    all_features = []
    all_labels = []
    
    for feat_path in tqdm(list(features_dir.rglob("*.npz")), desc="Loading"):
        video_name = feat_path.stem
        
        label_path = labels_dir / f"{video_name}.npy"
        if not label_path.exists():
            label_path = labels_dir / f"{video_name}.npz"
        if not label_path.exists():
            continue
        
        # Load features
        feat_data = np.load(feat_path, allow_pickle=True)
        features = feat_data['features']
        frame_indices = feat_data['frame_indices']
        
        # Load labels
        if label_path.suffix == '.npz':
            labels = np.load(label_path)
            labels = labels.get('labels', labels.get('arr_0'))
        else:
            labels = np.load(label_path)
        
        if labels.ndim > 1:
            labels = labels.max(axis=1)
        labels = (labels > 0).astype(float)
        
        # Align (features are at frame_indices positions)
        # This is approximate - depends on sample_rate alignment
        for i, feat in enumerate(features):
            # Map frame index to label index (assuming labels at ~1 FPS)
            # This needs adjustment based on your actual label rate
            label_idx = min(i, len(labels) - 1)
            all_features.append(feat)
            all_labels.append(labels[label_idx])
    
    if not all_features:
        print("No features loaded!")
        return
    
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    
    print(f"Loaded {len(all_features)} samples")
    print(f"Violence rate: {all_labels.mean():.1%}")
    
    feature_names = get_feature_names()
    
    # Per-feature AUC
    print("\nPer-Feature ROC-AUC:")
    feature_aucs = []
    for i, name in enumerate(feature_names):
        try:
            auc = roc_auc_score(all_labels, all_features[:, i])
        except:
            auc = 0.5
        feature_aucs.append(auc)
        print(f"  {name:<30} {auc:.3f}")
    
    # Logistic regression
    print("\nLogistic Regression:")
    n = len(all_labels)
    idx = np.random.permutation(n)
    train_idx, test_idx = idx[:int(0.7*n)], idx[int(0.7*n):]
    
    # Normalize features
    mean = all_features[train_idx].mean(axis=0)
    std = all_features[train_idx].std(axis=0) + 1e-8
    X_train = (all_features[train_idx] - mean) / std
    X_test = (all_features[test_idx] - mean) / std
    
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, all_labels[train_idx])
    
    train_auc = roc_auc_score(all_labels[train_idx], clf.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(all_labels[test_idx], clf.predict_proba(X_test)[:, 1])
    
    print(f"  Train AUC: {train_auc:.4f}")
    print(f"  Test AUC:  {test_auc:.4f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    sorted_idx = np.argsort(feature_aucs)[::-1]
    colors = ['green' if a > 0.55 else 'gray' for a in np.array(feature_aucs)[sorted_idx]]
    ax.barh(range(len(feature_aucs)), np.array(feature_aucs)[sorted_idx], color=colors)
    ax.set_yticks(range(len(feature_aucs)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.axvline(x=0.5, color='red', linestyle='--')
    ax.set_xlabel('ROC-AUC')
    ax.set_title(f'Optical Flow Feature Discriminative Power\n(Logistic Test AUC: {test_auc:.3f})')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"\nSaved analysis to {output_path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Optical Flow Feature Extraction")
    
    parser.add_argument('--video_path', type=str, default=None,
                        help='Single video to process')
    parser.add_argument('--videos_dir', type=str, default=None,
                        help='Directory of videos')
    parser.add_argument('--embeddings_dir', type=str, default=None,
                        help='Embeddings directory (to get video names)')
    
    parser.add_argument('--output_dir', type=str, default='data/processed/optical_flow')
    
    parser.add_argument('--method', type=str, default='farneback',
                        choices=['farneback', 'raft'],
                        help='Optical flow method')
    parser.add_argument('--sample_rate', type=int, default=1,
                        help='Process every Nth frame')
    parser.add_argument('--resize_height', type=int, default=256,
                        help='Resize height for processing')
    parser.add_argument('--source_filter', type=str, default=None,
                        choices=['xd_violence', 'ucf_crime'])
    parser.add_argument('--device', type=str, default='cuda:0')
    
    # Parallel processing options
    parser.add_argument('--worker_id', type=int, default=0,
                        help='Worker ID (0 to num_workers-1)')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Total number of workers')
    parser.add_argument('--launch_parallel', action='store_true',
                        help='Launch multiple worker processes automatically')
    parser.add_argument('--num_processes', type=int, default=4,
                        help='Number of processes for multiprocessing pool')
    parser.add_argument('--use_pool', action='store_true',
                        help='Use multiprocessing pool (single machine)')
    
    # Analysis
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze existing features')
    parser.add_argument('--features_dir', type=str, default=None)
    parser.add_argument('--labels_dir', type=str, default=None)
    
    args = parser.parse_args()
    
    # Analysis mode
    if args.analyze:
        if not args.features_dir or not args.labels_dir:
            print("Need --features_dir and --labels_dir for analysis")
            return
        analyze_features(
            Path(args.features_dir),
            Path(args.labels_dir),
            Path(args.output_dir) / 'flow_feature_analysis.png'
        )
        return
    
    # Launch parallel workers mode
    if args.launch_parallel:
        if not args.videos_dir or not args.embeddings_dir:
            print("Need --videos_dir and --embeddings_dir for parallel launch")
            return
        launch_parallel_workers(args, args.num_workers)
        return
    
    # Multiprocessing pool mode
    if args.use_pool:
        if not args.videos_dir or not args.embeddings_dir:
            print("Need --videos_dir and --embeddings_dir for pool mode")
            return
        process_videos_parallel_pool(
            Path(args.videos_dir),
            Path(args.embeddings_dir),
            Path(args.output_dir),
            args.sample_rate,
            args.source_filter,
            args.num_processes,
            args.resize_height,
        )
        return
    
    if not args.video_path and not args.videos_dir:
        print("Specify --video_path or --videos_dir")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = OpticalFlowExtractor(
        method=args.method,
        resize_height=args.resize_height,
        device=args.device,
    )
    
    if args.video_path:
        video_path = Path(args.video_path)
        output_path = output_dir / f"{video_path.stem}.npz"
        process_single_video(video_path, output_path, extractor, args.sample_rate, args.resize_height)
    
    elif args.videos_dir and args.embeddings_dir:
        process_videos_from_embeddings(
            Path(args.videos_dir),
            Path(args.embeddings_dir),
            output_dir,
            extractor,
            args.sample_rate,
            args.source_filter,
            args.worker_id,
            args.num_workers,
        )
    
    elif args.videos_dir:
        # Process all videos in directory
        videos_dir = Path(args.videos_dir)
        for ext in ['*.mp4', '*.avi', '*.mkv']:
            for video_path in videos_dir.glob(ext):
                output_path = output_dir / f"{video_path.stem}.npz"
                if not output_path.exists():
                    process_single_video(video_path, output_path, extractor, args.sample_rate, args.resize_height)
    
    print(f"\nFeatures saved to: {output_dir}")
    print(f"Feature dimension: {len(get_feature_names())}")


if __name__ == '__main__':
    main()