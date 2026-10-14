#!/usr/bin/env python3
"""
Fuse VLM Embeddings with Pose Features for Violence Anticipation

This script combines:
1. VLM embeddings (3584 dims) - semantic scene understanding
2. Pose features (~350 dims) - body language and motion

Fusion strategies:
- concat: Simple concatenation [vlm; pose]
- concat_norm: Normalize each modality before concatenation
- weighted: Weighted concatenation with learned or fixed weights

Usage:
    # Build fused embeddings directory (for training)
    python scripts/build_fused_features.py \
        --mode build_embeddings \
        --vlm_dir data/processed/vlm_emb \
        --pose_dir data/processed/pose_features \
        --output_dir data/processed/fused_features \
        --source_filter xd_violence

    # Build fused simulation file (for evaluation)
    python scripts/build_fused_features.py \
        --mode build_simulation \
        --vlm_sim simulation_data/xd_violence_8hr_sim.npz \
        --pose_dir data/processed/pose_features \
        --output simulation_data/xd_violence_8hr_fused.npz

    # Check separability of fused features
    python scripts/build_fused_features.py \
        --mode check_separability \
        --vlm_dir data/processed/vlm_emb \
        --pose_dir data/processed/pose_features \
        --labels_dir data/processed/labels_multih \
        --source_filter xd_violence
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


def classify_video_source(video_id: str) -> str:
    """Classify video source from ID."""
    if '_x264' in video_id:
        return 'ucf_crime'
    if video_id.startswith('v='):
        return 'xd_violence'
    if '__#' in video_id:
        return 'xd_violence'
    return 'unknown'


def load_vlm_embedding(vlm_path: Path) -> Optional[np.ndarray]:
    """Load VLM embedding from file."""
    try:
        if vlm_path.suffix == '.npz':
            data = np.load(vlm_path)
            for key in ['emb', 'embedding', 'embeddings', 'arr_0']:
                if key in data.keys():
                    arr = data[key]
                    if arr.dtype == np.float16:
                        arr = arr.astype(np.float32)
                    return arr
        else:
            arr = np.load(vlm_path)
            if arr.dtype == np.float16:
                arr = arr.astype(np.float32)
            return arr
    except Exception as e:
        print(f"Error loading VLM embedding {vlm_path}: {e}")
    return None


def load_pose_features(pose_path: Path) -> Optional[np.ndarray]:
    """Load pose features from file."""
    try:
        data = np.load(pose_path)
        features = data.get('features', data.get('arr_0'))
        if features is not None:
            return features.astype(np.float32)
    except Exception as e:
        print(f"Error loading pose features {pose_path}: {e}")
    return None


def find_file_in_subdirs(base_dir: Path, video_name: str, extensions: List[str]) -> Optional[Path]:
    """Find a file in base_dir or its subdirectories."""
    source = classify_video_source(video_name)
    
    # Try source subdirectory first
    for subdir in [source, 'xd_violence', 'ucf_crime', '']:
        for ext in extensions:
            if subdir:
                path = base_dir / subdir / f"{video_name}{ext}"
            else:
                path = base_dir / f"{video_name}{ext}"
            if path.exists():
                return path
    return None


def normalize_features(features: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features to zero mean and unit variance.
    
    Returns:
        normalized: Normalized features
        mean: Mean used for normalization
        std: Std used for normalization
    """
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + eps
    normalized = (features - mean) / std
    return normalized, mean.squeeze(), std.squeeze()


def align_features(
    vlm_features: np.ndarray,
    pose_features: np.ndarray,
    vlm_times: Optional[np.ndarray] = None,
    pose_times: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align VLM and pose features by length.
    
    If they have different lengths, truncate to the shorter one.
    More sophisticated alignment could use timestamps if available.
    """
    min_len = min(len(vlm_features), len(pose_features))
    return vlm_features[:min_len], pose_features[:min_len]


def fuse_features(
    vlm_features: np.ndarray,
    pose_features: np.ndarray,
    strategy: str = 'concat_norm',
    vlm_weight: float = 1.0,
    pose_weight: float = 1.0,
) -> np.ndarray:
    """
    Fuse VLM embeddings with pose features.
    
    Args:
        vlm_features: [T, vlm_dim] VLM embeddings
        pose_features: [T, pose_dim] Pose features
        strategy: Fusion strategy ('concat', 'concat_norm', 'weighted')
        vlm_weight: Weight for VLM features (for 'weighted' strategy)
        pose_weight: Weight for pose features (for 'weighted' strategy)
    
    Returns:
        fused: [T, vlm_dim + pose_dim] Fused features
    """
    # Align lengths
    vlm_features, pose_features = align_features(vlm_features, pose_features)
    
    if strategy == 'concat':
        # Simple concatenation
        return np.concatenate([vlm_features, pose_features], axis=1)
    
    elif strategy == 'concat_norm':
        # Normalize each modality before concatenation
        vlm_norm, _, _ = normalize_features(vlm_features)
        pose_norm, _, _ = normalize_features(pose_features)
        return np.concatenate([vlm_norm, pose_norm], axis=1)
    
    elif strategy == 'weighted':
        # Weighted concatenation (scale each modality)
        vlm_norm, _, _ = normalize_features(vlm_features)
        pose_norm, _, _ = normalize_features(pose_features)
        return np.concatenate([vlm_norm * vlm_weight, pose_norm * pose_weight], axis=1)
    
    else:
        raise ValueError(f"Unknown fusion strategy: {strategy}")


def build_fused_embeddings(
    vlm_dir: Path,
    pose_dir: Path,
    output_dir: Path,
    source_filter: Optional[str] = None,
    strategy: str = 'concat_norm',
    skip_existing: bool = True,
):
    """
    Build fused embeddings for all videos.
    
    Creates files matching the structure expected by train_temporal_model.py
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all VLM embedding files
    vlm_files = []
    for ext in ['*.npz', '*.npy']:
        vlm_files.extend(vlm_dir.rglob(ext))
    
    print(f"Found {len(vlm_files)} VLM embedding files")
    
    # Track statistics
    processed = 0
    skipped = 0
    errors = 0
    no_pose = 0
    
    # Compute global pose statistics for normalization (optional)
    # For now, we normalize per-video
    
    for vlm_path in tqdm(vlm_files, desc="Building fused embeddings"):
        video_name = vlm_path.stem
        source = classify_video_source(video_name)
        
        if source_filter and source != source_filter:
            continue
        
        # Output path
        out_subdir = output_dir / source
        out_subdir.mkdir(exist_ok=True)
        out_path = out_subdir / f"{video_name}.npz"
        
        if skip_existing and out_path.exists():
            skipped += 1
            continue
        
        # Load VLM embedding
        vlm_features = load_vlm_embedding(vlm_path)
        if vlm_features is None:
            errors += 1
            continue
        
        # Find and load pose features
        pose_path = find_file_in_subdirs(pose_dir, video_name, ['.npz', '.npy'])
        if pose_path is None:
            no_pose += 1
            # Option: Save VLM-only with zero-padded pose features
            # For now, skip
            continue
        
        pose_features = load_pose_features(pose_path)
        if pose_features is None:
            errors += 1
            continue
        
        # Fuse
        try:
            fused = fuse_features(vlm_features, pose_features, strategy=strategy)
            
            # Save
            np.savez_compressed(
                out_path,
                emb=fused.astype(np.float16),  # Save as float16 to save space
                vlm_dim=vlm_features.shape[1],
                pose_dim=pose_features.shape[1],
                strategy=strategy,
            )
            processed += 1
            
        except Exception as e:
            print(f"Error fusing {video_name}: {e}")
            errors += 1
    
    print(f"\nDone!")
    print(f"  Processed: {processed}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  No pose features: {no_pose}")
    print(f"  Errors: {errors}")
    
    # Save metadata
    metadata = {
        'strategy': strategy,
        'vlm_dir': str(vlm_dir),
        'pose_dir': str(pose_dir),
        'processed': processed,
        'source_filter': source_filter,
    }
    with open(output_dir / 'fusion_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nFused embeddings saved to: {output_dir}")


def build_fused_simulation(
    vlm_sim_path: Path,
    pose_dir: Path,
    output_path: Path,
    strategy: str = 'concat_norm',
):
    """
    Build a fused simulation file for evaluation.
    
    The simulation file contains concatenated embeddings from multiple videos.
    We need to identify which videos are in the simulation and fuse their features.
    """
    print(f"Loading VLM simulation from {vlm_sim_path}...")
    vlm_sim = np.load(vlm_sim_path)
    
    vlm_embeddings = vlm_sim['embeddings']
    frame_labels = vlm_sim['frame_labels']
    anticipation_labels = vlm_sim['anticipation_labels']
    fps = float(vlm_sim.get('fps', 1.0))
    horizon_sec = float(vlm_sim.get('horizon_sec', 30.0))
    
    print(f"VLM simulation shape: {vlm_embeddings.shape}")
    print(f"Labels shape: {anticipation_labels.shape}")
    
    # Check if simulation has video boundaries
    if 'video_boundaries' in vlm_sim.keys():
        video_boundaries = vlm_sim['video_boundaries']
        video_names = vlm_sim.get('video_names', None)
        print(f"Found {len(video_boundaries)} video boundaries")
    else:
        # Need to infer or handle differently
        print("Warning: No video boundaries in simulation file")
        print("Will attempt to match by loading pose features for same videos")
        video_boundaries = None
        video_names = None
    
    # If we have video boundaries and names, fuse per-video
    if video_boundaries is not None and video_names is not None:
        fused_embeddings = []
        current_pos = 0
        
        for i, (vid_name, boundary) in enumerate(zip(video_names, video_boundaries)):
            vid_name = str(vid_name)
            start, end = boundary
            
            # Get VLM embeddings for this video
            vlm_chunk = vlm_embeddings[start:end]
            
            # Find pose features
            pose_path = find_file_in_subdirs(pose_dir, vid_name, ['.npz', '.npy'])
            
            if pose_path is not None:
                pose_features = load_pose_features(pose_path)
                if pose_features is not None:
                    # Fuse
                    fused_chunk = fuse_features(vlm_chunk, pose_features, strategy=strategy)
                    fused_embeddings.append(fused_chunk)
                    continue
            
            # No pose features - pad with zeros
            pose_dim = 350  # Approximate, will be fixed later
            pose_zeros = np.zeros((len(vlm_chunk), pose_dim), dtype=np.float32)
            fused_chunk = fuse_features(vlm_chunk, pose_zeros, strategy='concat')
            fused_embeddings.append(fused_chunk)
        
        fused_embeddings = np.vstack(fused_embeddings)
    
    else:
        # No boundaries - try to match overall length
        # This is a fallback and may not work well
        print("Attempting direct pose feature loading...")
        
        # Try to find pose features that match the video names in metadata
        # For now, just create zero-padded version
        pose_dim = 350
        pose_zeros = np.zeros((len(vlm_embeddings), pose_dim), dtype=np.float32)
        
        vlm_norm, _, _ = normalize_features(vlm_embeddings)
        fused_embeddings = np.concatenate([vlm_norm, pose_zeros], axis=1)
        
        print("Warning: Using zero-padded pose features (no video boundaries)")
    
    # Ensure labels match
    min_len = min(len(fused_embeddings), len(anticipation_labels))
    fused_embeddings = fused_embeddings[:min_len]
    frame_labels = frame_labels[:min_len]
    anticipation_labels = anticipation_labels[:min_len]
    
    print(f"Fused embeddings shape: {fused_embeddings.shape}")
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=fused_embeddings.astype(np.float32),
        frame_labels=frame_labels,
        anticipation_labels=anticipation_labels,
        fps=fps,
        horizon_sec=horizon_sec,
        vlm_dim=vlm_embeddings.shape[1],
        pose_dim=fused_embeddings.shape[1] - vlm_embeddings.shape[1],
        strategy=strategy,
    )
    
    print(f"Fused simulation saved to: {output_path}")


def build_fused_simulation_from_videos(
    vlm_dir: Path,
    pose_dir: Path,
    labels_dir: Path,
    output_path: Path,
    video_list: Optional[List[str]] = None,
    source_filter: Optional[str] = None,
    strategy: str = 'concat_norm',
    fps: float = 1.0,
    horizon_sec: float = 30.0,
    max_videos: Optional[int] = None,
):
    """
    Build a fused simulation from individual video files.
    
    This is more flexible than build_fused_simulation as it doesn't require
    an existing simulation file.
    """
    # Find all videos
    if video_list:
        video_names = video_list
    else:
        # Find from VLM directory
        vlm_files = list(vlm_dir.rglob('*.npz')) + list(vlm_dir.rglob('*.npy'))
        video_names = [f.stem for f in vlm_files]
    
    if source_filter:
        video_names = [v for v in video_names if classify_video_source(v) == source_filter]
    
    if max_videos:
        video_names = video_names[:max_videos]
    
    print(f"Building simulation from {len(video_names)} videos...")
    
    all_embeddings = []
    all_frame_labels = []
    all_anticipation_labels = []
    video_boundaries = []
    valid_video_names = []
    
    current_pos = 0
    horizon_frames = int(horizon_sec * fps)
    
    for vid_name in tqdm(video_names, desc="Loading videos"):
        # Load VLM
        vlm_path = find_file_in_subdirs(vlm_dir, vid_name, ['.npz', '.npy'])
        if vlm_path is None:
            continue
        
        vlm_features = load_vlm_embedding(vlm_path)
        if vlm_features is None:
            continue
        
        # Load pose
        pose_path = find_file_in_subdirs(pose_dir, vid_name, ['.npz', '.npy'])
        if pose_path is None:
            continue
        
        pose_features = load_pose_features(pose_path)
        if pose_features is None:
            continue
        
        # Load labels
        label_path = find_file_in_subdirs(labels_dir, vid_name, ['.npy', '.npz'])
        if label_path is None:
            continue
        
        try:
            if label_path.suffix == '.npz':
                labels_data = np.load(label_path)
                labels = labels_data.get('labels', labels_data.get('arr_0'))
            else:
                labels = np.load(label_path)
            
            if labels.ndim > 1:
                labels = labels.max(axis=1)
            labels = (labels > 0).astype(np.float32)
        except:
            continue
        
        # Align all three
        min_len = min(len(vlm_features), len(pose_features), len(labels))
        vlm_features = vlm_features[:min_len]
        pose_features = pose_features[:min_len]
        labels = labels[:min_len]
        
        if min_len == 0:
            continue
        
        # Fuse
        fused = fuse_features(vlm_features, pose_features, strategy=strategy)
        
        # Create anticipation labels
        anticipation_labels = np.zeros_like(labels)
        for t in range(len(labels)):
            future_end = min(t + horizon_frames, len(labels))
            anticipation_labels[t] = labels[t:future_end].max()
        
        # Track boundaries
        start = current_pos
        end = current_pos + len(fused)
        video_boundaries.append([start, end])
        current_pos = end
        
        all_embeddings.append(fused)
        all_frame_labels.append(labels)
        all_anticipation_labels.append(anticipation_labels)
        valid_video_names.append(vid_name)
    
    if not all_embeddings:
        raise RuntimeError("No valid videos found!")
    
    # Concatenate
    embeddings = np.vstack(all_embeddings)
    frame_labels = np.concatenate(all_frame_labels)
    anticipation_labels = np.concatenate(all_anticipation_labels)
    
    print(f"\nSimulation stats:")
    print(f"  Videos: {len(valid_video_names)}")
    print(f"  Total frames: {len(embeddings)}")
    print(f"  Duration: {len(embeddings) / fps / 3600:.2f} hours")
    print(f"  Embedding dim: {embeddings.shape[1]}")
    print(f"  Violence frames: {frame_labels.sum():.0f} ({frame_labels.mean()*100:.2f}%)")
    print(f"  Anticipation frames: {anticipation_labels.sum():.0f} ({anticipation_labels.mean()*100:.2f}%)")
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings.astype(np.float32),
        frame_labels=frame_labels,
        anticipation_labels=anticipation_labels,
        fps=fps,
        horizon_sec=horizon_sec,
        video_boundaries=np.array(video_boundaries),
        video_names=np.array(valid_video_names, dtype=object),
        strategy=strategy,
    )
    
    print(f"\nSaved to: {output_path}")


def check_separability(
    vlm_dir: Path,
    pose_dir: Path,
    labels_dir: Path,
    source_filter: Optional[str] = None,
    max_videos: int = 100,
    strategy: str = 'concat_norm',
):
    """
    Check feature separability for VLM, pose, and fused features.
    """
    print("Checking feature separability...")
    
    # Find videos
    vlm_files = list(vlm_dir.rglob('*.npz')) + list(vlm_dir.rglob('*.npy'))
    video_names = [f.stem for f in vlm_files]
    
    if source_filter:
        video_names = [v for v in video_names if classify_video_source(v) == source_filter]
    
    video_names = video_names[:max_videos]
    
    vlm_violent, vlm_nonviolent = [], []
    pose_violent, pose_nonviolent = [], []
    fused_violent, fused_nonviolent = [], []
    
    for vid_name in tqdm(video_names, desc="Loading"):
        # Load VLM
        vlm_path = find_file_in_subdirs(vlm_dir, vid_name, ['.npz', '.npy'])
        if vlm_path is None:
            continue
        vlm_features = load_vlm_embedding(vlm_path)
        if vlm_features is None:
            continue
        
        # Load pose
        pose_path = find_file_in_subdirs(pose_dir, vid_name, ['.npz', '.npy'])
        if pose_path is None:
            continue
        pose_features = load_pose_features(pose_path)
        if pose_features is None:
            continue
        
        # Load labels
        label_path = find_file_in_subdirs(labels_dir, vid_name, ['.npy', '.npz'])
        if label_path is None:
            continue
        
        try:
            if label_path.suffix == '.npz':
                labels_data = np.load(label_path)
                labels = labels_data.get('labels', labels_data.get('arr_0'))
            else:
                labels = np.load(label_path)
            
            if labels.ndim > 1:
                labels = labels.max(axis=1)
            labels = (labels > 0).astype(int)
        except:
            continue
        
        # Align
        min_len = min(len(vlm_features), len(pose_features), len(labels))
        vlm_features = vlm_features[:min_len]
        pose_features = pose_features[:min_len]
        labels = labels[:min_len]
        
        # Fuse
        fused = fuse_features(vlm_features, pose_features, strategy=strategy)
        
        # Split by label
        violent_mask = labels > 0
        
        if violent_mask.sum() > 0:
            vlm_violent.append(vlm_features[violent_mask])
            pose_violent.append(pose_features[violent_mask])
            fused_violent.append(fused[violent_mask])
        
        if (~violent_mask).sum() > 0:
            vlm_nonviolent.append(vlm_features[~violent_mask])
            pose_nonviolent.append(pose_features[~violent_mask])
            fused_nonviolent.append(fused[~violent_mask])
    
    # Concatenate
    def compute_metrics(violent, nonviolent, name):
        if not violent or not nonviolent:
            print(f"\n{name}: Insufficient data")
            return
        
        v = np.vstack(violent)
        nv = np.vstack(nonviolent)
        
        v_mean = v.mean(axis=0)
        nv_mean = nv.mean(axis=0)
        
        diff = np.abs(v_mean - nv_mean)
        cos_sim = np.dot(v_mean, nv_mean) / (np.linalg.norm(v_mean) * np.linalg.norm(nv_mean) + 1e-8)
        
        print(f"\n{name}:")
        print(f"  Violent frames: {len(v)}")
        print(f"  Non-violent frames: {len(nv)}")
        print(f"  Feature dim: {v.shape[1]}")
        print(f"  Mean abs difference: {diff.mean():.6f}")
        print(f"  Max difference: {diff.max():.6f}")
        print(f"  Dims with >0.1 diff: {(diff > 0.1).sum()}")
        print(f"  Cosine similarity: {cos_sim:.6f}")
        print(f"  --> Separability score: {1 - cos_sim:.6f} (higher = better)")
    
    compute_metrics(vlm_violent, vlm_nonviolent, "VLM Embeddings")
    compute_metrics(pose_violent, pose_nonviolent, "Pose Features")
    compute_metrics(fused_violent, fused_nonviolent, "Fused Features")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                        choices=['build_embeddings', 'build_simulation', 'build_simulation_from_videos', 'check_separability'])
    
    # Paths
    parser.add_argument('--vlm_dir', type=str, default='data/processed/vlm_emb')
    parser.add_argument('--vlm_sim', type=str, default=None)
    parser.add_argument('--pose_dir', type=str, default='data/processed/pose_features')
    parser.add_argument('--labels_dir', type=str, default='data/processed/labels_multih')
    parser.add_argument('--output_dir', type=str, default='data/processed/fused_features')
    parser.add_argument('--output', type=str, default=None)
    
    # Options
    parser.add_argument('--source_filter', type=str, default=None,
                        choices=['xd_violence', 'ucf_crime'])
    parser.add_argument('--strategy', type=str, default='concat_norm',
                        choices=['concat', 'concat_norm', 'weighted'])
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--max_videos', type=int, default=None)
    parser.add_argument('--fps', type=float, default=1.0)
    parser.add_argument('--horizon_sec', type=float, default=30.0)
    
    args = parser.parse_args()
    
    if args.mode == 'build_embeddings':
        build_fused_embeddings(
            vlm_dir=Path(args.vlm_dir),
            pose_dir=Path(args.pose_dir),
            output_dir=Path(args.output_dir),
            source_filter=args.source_filter,
            strategy=args.strategy,
            skip_existing=args.skip_existing,
        )
    
    elif args.mode == 'build_simulation':
        if args.vlm_sim is None:
            raise ValueError("--vlm_sim required for build_simulation mode")
        if args.output is None:
            raise ValueError("--output required for build_simulation mode")
        
        build_fused_simulation(
            vlm_sim_path=Path(args.vlm_sim),
            pose_dir=Path(args.pose_dir),
            output_path=Path(args.output),
            strategy=args.strategy,
        )
    
    elif args.mode == 'build_simulation_from_videos':
        if args.output is None:
            raise ValueError("--output required for build_simulation_from_videos mode")
        
        build_fused_simulation_from_videos(
            vlm_dir=Path(args.vlm_dir),
            pose_dir=Path(args.pose_dir),
            labels_dir=Path(args.labels_dir),
            output_path=Path(args.output),
            source_filter=args.source_filter,
            strategy=args.strategy,
            fps=args.fps,
            horizon_sec=args.horizon_sec,
            max_videos=args.max_videos,
        )
    
    elif args.mode == 'check_separability':
        check_separability(
            vlm_dir=Path(args.vlm_dir),
            pose_dir=Path(args.pose_dir),
            labels_dir=Path(args.labels_dir),
            source_filter=args.source_filter,
            max_videos=args.max_videos or 100,
            strategy=args.strategy,
        )


if __name__ == '__main__':
    main()