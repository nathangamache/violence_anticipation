#!/usr/bin/env python3
"""
Diagnostic script - v3 for Nathan's nested directory structure

Structure:
- vlm_emb/xd_violence/*.npz
- vlm_emb/ucf_crime/*.npz
- labels_multih/*.npy (flat structure)

Index files contain: [{"graph_path": "...", "label_path": "..."}, ...]
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def extract_video_name(item) -> str:
    """Extract video name from index item."""
    if isinstance(item, dict):
        path = item.get('graph_path') or item.get('label_path', '')
        return Path(path).stem
    elif isinstance(item, str):
        return Path(item).stem
    return str(item)


def classify_video_source(video_id: str) -> str:
    """Classify video source from its ID."""
    if '_x264' in video_id:
        return 'ucf_crime'
    if video_id.startswith('v='):
        return 'xd_violence'
    if re.search(r'\d{4}__#', video_id) or '__#' in video_id:
        return 'xd_violence'  # Movies are in XD-Violence
    return 'unknown'


def find_embedding_file(video_name: str, embeddings_dir: Path) -> Optional[Path]:
    """Find embedding file in nested directory structure."""
    # Determine which subdirectory based on video name
    source = classify_video_source(video_name)
    
    # Try direct path first
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
    """Find label file (supports flat and nested structure)."""
    # Try flat structure first (labels_multih/*.npy)
    for ext in ['.npy', '.npz']:
        path = labels_dir / f"{video_name}{ext}"
        if path.exists():
            return path
    
    # Try nested structure
    source = classify_video_source(video_name)
    for subdir in [source, 'ucf_crime', 'xd_violence']:
        for ext in ['.npy', '.npz', '.jsonl']:
            path = labels_dir / subdir / f"{video_name}{ext}"
            if path.exists():
                return path
    
    return None


def load_embedding(path: Path) -> Optional[np.ndarray]:
    """Load embedding from .npy or .npz file."""
    try:
        if path.suffix == '.npz':
            data = np.load(path)
            # NPZ files have named arrays - find the embedding
            keys = list(data.keys())
            if 'embedding' in keys:
                return data['embedding']
            elif 'embeddings' in keys:
                return data['embeddings']
            elif 'arr_0' in keys:
                return data['arr_0']
            elif len(keys) == 1:
                return data[keys[0]]
            else:
                # Return first array that looks like embeddings
                for k in keys:
                    arr = data[k]
                    if arr.ndim >= 1:
                        return arr
        else:
            return np.load(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")
    return None


def load_labels(path: Path) -> Optional[np.ndarray]:
    """Load labels from .npy, .npz, or .jsonl file."""
    try:
        if path.suffix == '.jsonl':
            # Parse JSONL - each line is a JSON object
            labels = []
            with open(path) as f:
                for line in f:
                    obj = json.loads(line)
                    # Assuming each line has a 'label' or 'violent' field
                    if 'label' in obj:
                        labels.append(obj['label'])
                    elif 'violent' in obj:
                        labels.append(1 if obj['violent'] else 0)
            return np.array(labels)
        elif path.suffix == '.npz':
            data = np.load(path)
            keys = list(data.keys())
            if 'labels' in keys:
                return data['labels']
            elif 'label' in keys:
                return data['label']
            elif 'arr_0' in keys:
                return data['arr_0']
            elif len(keys) == 1:
                return data[keys[0]]
        else:
            return np.load(path)
    except Exception as e:
        print(f"Error loading labels {path}: {e}")
    return None


def scan_directory_structure(base_dir: Path):
    """Scan and report directory structure."""
    print(f"\nScanning {base_dir}:")
    
    if not base_dir.exists():
        print(f"  ERROR: Directory does not exist!")
        return
    
    # Check for subdirectories
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if subdirs:
        print(f"  Subdirectories: {[d.name for d in subdirs]}")
        for subdir in subdirs:
            files = list(subdir.glob('*'))
            extensions = defaultdict(int)
            for f in files:
                if f.is_file():
                    extensions[f.suffix] += 1
            print(f"    {subdir.name}/: {dict(extensions)}")
            if files:
                print(f"      Sample files: {[f.name for f in files[:3]]}")
    
    # Check for files in root
    root_files = [f for f in base_dir.iterdir() if f.is_file()]
    if root_files:
        extensions = defaultdict(int)
        for f in root_files:
            extensions[f.suffix] += 1
        print(f"  Root files: {dict(extensions)}")
        print(f"    Sample: {[f.name for f in root_files[:3]]}")


def load_dataset_stats(
    video_items: List,
    embeddings_dir: Path,
    labels_dir: Path,
    horizon_sec: float = 30.0,
    fps: float = 1.0,
) -> Dict:
    """Load and compute statistics for a dataset split."""
    
    stats = {
        'n_videos': 0,
        'n_frames': 0,
        'n_positive_frames': 0,
        'source_dist': defaultdict(int),
        'videos_with_violence': 0,
        'per_video': [],
        'missing_emb': 0,
        'missing_label': 0,
        'load_errors': 0,
    }
    
    horizon_frames = int(horizon_sec * fps)
    video_names = [extract_video_name(item) for item in video_items]
    
    # Debug first few
    print(f"\n  Processing {len(video_names)} videos...")
    if video_names:
        print(f"  Sample names: {video_names[:3]}")
    
    for i, vid in enumerate(video_names):
        emb_path = find_embedding_file(vid, embeddings_dir)
        label_path = find_label_file(vid, labels_dir)
        
        if emb_path is None:
            stats['missing_emb'] += 1
            if i < 3:  # Debug first few missing
                print(f"    Missing embedding for: {vid}")
            continue
        
        if label_path is None:
            stats['missing_label'] += 1
            continue
        
        embeddings = load_embedding(emb_path)
        labels = load_labels(label_path)
        
        if embeddings is None or labels is None:
            stats['load_errors'] += 1
            continue
        
        try:
            # Handle multi-dimensional labels
            if labels.ndim > 1:
                labels = labels.max(axis=1) if labels.shape[1] > 1 else labels.squeeze()
            
            labels = (labels > 0).astype(float)
            
            # Align lengths
            min_len = min(len(embeddings), len(labels))
            labels = labels[:min_len]
            
            # Compute anticipation labels
            anticipation_labels = []
            for t in range(min_len):
                future_end = min(t + horizon_frames, min_len)
                ant_label = float(labels[t:future_end].max()) if future_end > t else 0
                anticipation_labels.append(ant_label)
            
            anticipation_labels = np.array(anticipation_labels)
            
            source = classify_video_source(vid)
            
            stats['n_videos'] += 1
            stats['n_frames'] += min_len
            stats['n_positive_frames'] += int(anticipation_labels.sum())
            stats['source_dist'][source] += 1
            
            if labels.max() > 0:
                stats['videos_with_violence'] += 1
            
            stats['per_video'].append({
                'video_id': vid,
                'source': source,
                'n_frames': min_len,
                'emb_shape': embeddings.shape,
                'pos_rate': float(anticipation_labels.mean()),
            })
            
        except Exception as e:
            print(f"  Error processing {vid}: {e}")
            stats['load_errors'] += 1
    
    stats['pos_rate'] = stats['n_positive_frames'] / stats['n_frames'] if stats['n_frames'] > 0 else 0
    
    print(f"  Loaded: {stats['n_videos']} videos, {stats['n_frames']:,} frames")
    print(f"  Missing: {stats['missing_emb']} emb, {stats['missing_label']} labels, {stats['load_errors']} errors")
    
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_dir', type=str, required=True)
    parser.add_argument('--labels_dir', type=str, required=True)
    parser.add_argument('--train_index', type=str, required=True)
    parser.add_argument('--val_index', type=str, required=True)
    parser.add_argument('--test_index', type=str, required=True)
    parser.add_argument('--horizon_sec', type=float, default=30.0)
    parser.add_argument('--fps', type=float, default=1.0)
    parser.add_argument('--output_dir', type=str, default='./diagnostics')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load indices
    with open(args.train_index) as f:
        train_items = json.load(f)
    with open(args.val_index) as f:
        val_items = json.load(f)
    with open(args.test_index) as f:
        test_items = json.load(f)
    
    print(f"Loaded indices: {len(train_items)} train, {len(val_items)} val, {len(test_items)} test")
    
    # Scan directory structures
    embeddings_dir = Path(args.embeddings_dir)
    labels_dir = Path(args.labels_dir)
    
    scan_directory_structure(embeddings_dir)
    scan_directory_structure(labels_dir)
    
    # Process each split
    print("\n" + "="*70)
    print("TRAIN SPLIT")
    train_stats = load_dataset_stats(train_items, embeddings_dir, labels_dir, args.horizon_sec, args.fps)
    
    print("\n" + "="*70)
    print("VAL SPLIT")
    val_stats = load_dataset_stats(val_items, embeddings_dir, labels_dir, args.horizon_sec, args.fps)
    
    print("\n" + "="*70)
    print("TEST SPLIT")
    test_stats = load_dataset_stats(test_items, embeddings_dir, labels_dir, args.horizon_sec, args.fps)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for name, stats in [('Train', train_stats), ('Val', val_stats), ('Test', test_stats)]:
        print(f"\n{name}:")
        print(f"  Videos: {stats['n_videos']}, Frames: {stats['n_frames']:,}, Pos rate: {stats['pos_rate']:.2%}")
        print(f"  Source distribution: {dict(stats['source_dist'])}")
    
    # Check embedding dimensions
    if train_stats['per_video']:
        shapes = [v['emb_shape'] for v in train_stats['per_video'][:5]]
        print(f"\nEmbedding shapes (sample): {shapes}")
    
    # Recommendations
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    
    total = train_stats['n_videos'] + val_stats['n_videos'] + test_stats['n_videos']
    if total == 0:
        print("\n❌ No videos loaded! Check:")
        print("   1. File extensions (.npz vs .npy)")
        print("   2. Directory structure matches classifier")
        print("   3. Video names in index match filenames")
    else:
        print(f"\n✓ Loaded {total} videos successfully!")
        print("\nYou can now run training with:")
        print(f"  python train_student_v2.py --embeddings_dir {args.embeddings_dir} --labels_dir {args.labels_dir} ...")


if __name__ == '__main__':
    main()