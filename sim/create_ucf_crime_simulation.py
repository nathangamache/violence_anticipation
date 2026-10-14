#!/usr/bin/env python3
"""
Create a simulation NPZ from UCF-Crime embeddings for evaluation.

This script builds a simulation file that can be used with comprehensive_eval.py
to generate evaluation graphs (ROC, PR, confusion matrix, etc.) for UCF-Crime.

Usage:
    python create_ucf_crime_simulation.py \
        --vlm_emb_dir data/processed/vlm_emb/ucf_crime \
        --labels_dir data/processed/labels_multih/ucf_crime \
        --output simulation_data/ucf_crime_8hr_sim.npz \
        --duration_hours 8.0
"""

import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import json


def load_video_data(
    vlm_emb_dir: str,
    labels_dir: str,
    source_filter: str = None,
) -> List[Dict]:
    """Load embeddings and labels for all videos."""
    vlm_dir = Path(vlm_emb_dir)
    lab_dir = Path(labels_dir)
    
    video_data = []
    
    for emb_file in sorted(vlm_dir.glob('*.npz')):
        vid_name = emb_file.stem
        
        # Skip if source filter doesn't match
        if source_filter:
            if source_filter == 'ucf' and '_x264' not in vid_name:
                continue
            if source_filter == 'xd' and '_x264' in vid_name:
                continue
        
        # Find label file
        label_file = lab_dir / f'{vid_name}.npy'
        if not label_file.exists():
            # Try subdirectories
            for subdir in lab_dir.iterdir():
                if subdir.is_dir():
                    lf = subdir / f'{vid_name}.npy'
                    if lf.exists():
                        label_file = lf
                        break
        
        if not label_file.exists():
            continue
        
        try:
            # Load embeddings
            z = np.load(emb_file)
            emb = z.get('emb', z.get('embeddings', z.get('arr_0')))
            if emb is None:
                continue
            
            # Load labels
            labels = np.load(label_file)
            if labels.ndim > 1:
                labels = labels.max(axis=1)
            
            # Align lengths
            min_len = min(len(emb), len(labels))
            emb = emb[:min_len].astype(np.float32)
            labels = (labels[:min_len] > 0).astype(np.int32)
            
            video_data.append({
                'name': vid_name,
                'emb': emb,
                'labels': labels,
                'has_violence': labels.sum() > 0,
            })
            
        except Exception as e:
            print(f"Warning: Failed to load {vid_name}: {e}")
            continue
    
    return video_data


def create_simulation(
    video_data: List[Dict],
    output_path: str,
    duration_hours: float = 8.0,
    fps: float = 1.0,
    seed: int = 42,
):
    """Create simulation NPZ from video data."""
    np.random.seed(seed)
    
    target_frames = int(duration_hours * 3600 * fps)
    
    # Separate violent and non-violent videos
    violent_videos = [v for v in video_data if v['has_violence']]
    normal_videos = [v for v in video_data if not v['has_violence']]
    
    print(f"Loaded {len(video_data)} videos: {len(violent_videos)} with violence, {len(normal_videos)} normal")
    
    # Count total frames
    total_violent_frames = sum(v['labels'].sum() for v in violent_videos)
    total_normal_frames = sum(len(v['labels']) - v['labels'].sum() for v in violent_videos)
    total_normal_frames += sum(len(v['labels']) for v in normal_videos)
    
    print(f"Total frames available: {total_violent_frames} violent, {total_normal_frames} normal")
    
    # Build simulation by interleaving videos
    all_emb = []
    all_labels = []
    
    # Shuffle videos
    np.random.shuffle(violent_videos)
    np.random.shuffle(normal_videos)
    
    # Interleave: add some normal videos between violent ones
    video_order = []
    v_idx = 0
    n_idx = 0
    
    while v_idx < len(violent_videos) or n_idx < len(normal_videos):
        # Add 2-3 normal videos
        for _ in range(np.random.randint(2, 4)):
            if n_idx < len(normal_videos):
                video_order.append(normal_videos[n_idx])
                n_idx += 1
        
        # Add 1 violent video
        if v_idx < len(violent_videos):
            video_order.append(violent_videos[v_idx])
            v_idx += 1
    
    # Concatenate
    for video in video_order:
        all_emb.append(video['emb'])
        all_labels.append(video['labels'])
    
    all_emb = np.vstack(all_emb)
    all_labels = np.concatenate(all_labels)
    
    # Trim or pad to target duration
    if len(all_labels) > target_frames:
        all_emb = all_emb[:target_frames]
        all_labels = all_labels[:target_frames]
    elif len(all_labels) < target_frames:
        # Repeat with offset
        repeats = target_frames // len(all_labels) + 1
        all_emb = np.tile(all_emb, (repeats, 1))[:target_frames]
        all_labels = np.tile(all_labels, repeats)[:target_frames]
    
    # Stats
    n_violent = all_labels.sum()
    anomaly_pct = 100.0 * n_violent / len(all_labels)
    actual_hours = len(all_labels) / fps / 3600
    
    print(f"\nSimulation created:")
    print(f"  Frames: {len(all_labels)}")
    print(f"  Duration: {actual_hours:.2f} hours")
    print(f"  Violent frames: {n_violent} ({anomaly_pct:.1f}%)")
    print(f"  Videos included: {len(video_order)}")
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        embeddings=all_emb.astype(np.float32),
        anticipation_labels=all_labels.astype(np.int32),
        fps=np.array([fps]),
        metadata=json.dumps({
            'n_videos': len(video_order),
            'n_violent_videos': len(violent_videos),
            'n_normal_videos': len(normal_videos),
            'anomaly_pct': anomaly_pct,
            'duration_hours': actual_hours,
        })
    )
    print(f"\nSaved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create UCF-Crime simulation for evaluation")
    parser.add_argument('--vlm_emb_dir', type=str, required=True,
                       help='Directory with VLM embeddings')
    parser.add_argument('--labels_dir', type=str, required=True,
                       help='Directory with labels')
    parser.add_argument('--output', type=str, required=True,
                       help='Output NPZ path')
    parser.add_argument('--duration_hours', type=float, default=8.0,
                       help='Target duration in hours')
    parser.add_argument('--fps', type=float, default=1.0,
                       help='Frames per second')
    parser.add_argument('--source_filter', type=str, default=None,
                       choices=['ucf', 'xd', None],
                       help='Filter videos by source')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    video_data = load_video_data(
        args.vlm_emb_dir,
        args.labels_dir,
        args.source_filter,
    )
    
    if not video_data:
        print("Error: No videos loaded!")
        return
    
    create_simulation(
        video_data,
        args.output,
        args.duration_hours,
        args.fps,
        args.seed,
    )


if __name__ == '__main__':
    main()