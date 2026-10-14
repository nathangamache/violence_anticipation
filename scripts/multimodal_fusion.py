#!/usr/bin/env python3
"""Early fusion of VLM, audio and pose features (concat, concat_norm or weighted).

Usage:
    python scripts/multimodal_fusion.py --mode {check_separability,build_embeddings,build_simulation} \
        --vlm_dir data/processed/vlm_emb --audio_dir data/processed/audio_features \
        --pose_dir data/processed/pose_features --labels_dir data/processed/labels_multih \
        --modalities vlm audio [pose] --source_filter xd_violence \
        [--output_dir data/processed/multimodal_features | --output simulation_data/<name>.npz]
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

import numpy as np
from tqdm import tqdm


def classify_video_source(video_id: str) -> str:
    if '_x264' in video_id:
        return 'ucf_crime'
    if video_id.startswith('v='):
        return 'xd_violence'
    if '__#' in video_id:
        return 'xd_violence'
    return 'unknown'


def find_file_in_subdirs(base_dir: Path, video_name: str, extensions: List[str]) -> Optional[Path]:
    """Find a file in base_dir or its subdirectories."""
    source = classify_video_source(video_name)
    
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
    """Normalize features to zero mean and unit variance."""
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + eps
    normalized = (features - mean) / std
    return normalized, mean.squeeze(), std.squeeze()


def load_vlm_features(path: Path) -> Optional[np.ndarray]:
    try:
        if path.suffix == '.npz':
            data = np.load(path)
            for key in ['emb', 'embedding', 'embeddings', 'arr_0']:
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
        pass
    return None


def load_audio_features(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.load(path)
        features = data.get('features', data.get('arr_0'))
        if features is not None:
            return features.astype(np.float32)
    except Exception as e:
        pass
    return None


def load_pose_features(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.load(path)
        features = data.get('features', data.get('arr_0'))
        if features is not None:
            return features.astype(np.float32)
    except Exception as e:
        pass
    return None


def load_labels(path: Path) -> Optional[np.ndarray]:
    try:
        if path.suffix == '.npz':
            data = np.load(path)
            labels = data.get('labels', data.get('arr_0'))
        else:
            labels = np.load(path)
        
        if labels is not None:
            if labels.ndim > 1:
                labels = labels.max(axis=1)
            return (labels > 0).astype(np.float32)
    except Exception as e:
        pass
    return None


class MultiModalFuser:
    """Handles fusion of multiple modalities."""
    
    def __init__(
        self,
        vlm_dir: Optional[Path] = None,
        audio_dir: Optional[Path] = None,
        pose_dir: Optional[Path] = None,
        labels_dir: Optional[Path] = None,
        modalities: List[str] = ['vlm', 'audio'],
        strategy: str = 'concat_norm',
    ):
        self.vlm_dir = vlm_dir
        self.audio_dir = audio_dir
        self.pose_dir = pose_dir
        self.labels_dir = labels_dir
        self.modalities = modalities
        self.strategy = strategy
        
        self.dims = {}
    
    def load_video_features(self, video_name: str) -> Dict[str, Optional[np.ndarray]]:
        """Load all available features for a video."""
        features = {}
        
        if 'vlm' in self.modalities and self.vlm_dir:
            path = find_file_in_subdirs(self.vlm_dir, video_name, ['.npz', '.npy'])
            features['vlm'] = load_vlm_features(path) if path else None
            if features['vlm'] is not None:
                self.dims['vlm'] = features['vlm'].shape[-1]
        
        if 'audio' in self.modalities and self.audio_dir:
            path = find_file_in_subdirs(self.audio_dir, video_name, ['.npz', '.npy'])
            features['audio'] = load_audio_features(path) if path else None
            if features['audio'] is not None:
                self.dims['audio'] = features['audio'].shape[-1]
        
        if 'pose' in self.modalities and self.pose_dir:
            path = find_file_in_subdirs(self.pose_dir, video_name, ['.npz', '.npy'])
            features['pose'] = load_pose_features(path) if path else None
            if features['pose'] is not None:
                self.dims['pose'] = features['pose'].shape[-1]
        
        return features
    
    def load_video_labels(self, video_name: str) -> Optional[np.ndarray]:
        if self.labels_dir is None:
            return None
        path = find_file_in_subdirs(self.labels_dir, video_name, ['.npy', '.npz'])
        return load_labels(path) if path else None
    
    def align_features(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Align features to the same length."""
        lengths = [len(f) for f in features.values() if f is not None]
        if not lengths:
            return features
        
        min_len = min(lengths)
        
        aligned = {}
        for key, feat in features.items():
            if feat is not None:
                aligned[key] = feat[:min_len]
            else:
                aligned[key] = None
        
        return aligned
    
    def fuse_features(self, features: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Fuse multiple modalities into a single feature vector."""
        available = {k: v for k, v in features.items() if v is not None}
        
        if not available:
            return None
        
        available = self.align_features(available)
        
        if self.strategy == 'concat':
            return np.concatenate(list(available.values()), axis=1)
        
        elif self.strategy == 'concat_norm':
            normalized = []
            for key in self.modalities:
                if key in available and available[key] is not None:
                    norm, _, _ = normalize_features(available[key])
                    normalized.append(norm)
            
            if not normalized:
                return None
            
            return np.concatenate(normalized, axis=1)
        
        elif self.strategy == 'weighted':
            weights = {'vlm': 1.0, 'audio': 1.5, 'pose': 1.0}
            normalized = []
            for key in self.modalities:
                if key in available and available[key] is not None:
                    norm, _, _ = normalize_features(available[key])
                    normalized.append(norm * weights.get(key, 1.0))
            
            if not normalized:
                return None
            
            return np.concatenate(normalized, axis=1)
        
        else:
            raise ValueError(f"Unknown fusion strategy: {self.strategy}")
    
    def get_total_dim(self) -> int:
        total = 0
        for mod in self.modalities:
            if mod in self.dims:
                total += self.dims[mod]
        return total


def check_separability(
    vlm_dir: Optional[Path],
    audio_dir: Optional[Path],
    pose_dir: Optional[Path],
    labels_dir: Path,
    source_filter: Optional[str] = None,
    max_videos: int = 200,
    modalities: List[str] = ['vlm', 'audio', 'pose'],
    strategy: str = 'concat_norm',
):
    """Check feature separability for each modality and fused features."""
    print("Checking feature separability...")
    print(f"Modalities: {modalities}")
    print(f"Strategy: {strategy}")
    
    fuser = MultiModalFuser(
        vlm_dir=vlm_dir,
        audio_dir=audio_dir,
        pose_dir=pose_dir,
        labels_dir=labels_dir,
        modalities=modalities,
        strategy=strategy,
    )
    
    label_files = list(labels_dir.rglob('*.npy')) + list(labels_dir.rglob('*.npz'))
    video_names = [f.stem for f in label_files]
    
    if source_filter:
        video_names = [v for v in video_names if classify_video_source(v) == source_filter]
    
    video_names = video_names[:max_videos]
    print(f"Analyzing {len(video_names)} videos...")
    
    modality_violent = defaultdict(list)
    modality_nonviolent = defaultdict(list)
    fused_violent = []
    fused_nonviolent = []
    
    for video_name in tqdm(video_names, desc="Loading features"):
        features = fuser.load_video_features(video_name)
        labels = fuser.load_video_labels(video_name)
        
        if labels is None:
            continue
        
        all_features = {k: v for k, v in features.items() if v is not None}
        if not all_features:
            continue
        
        min_len = min(len(labels), min(len(f) for f in all_features.values()))
        labels = labels[:min_len]
        
        violent_mask = labels > 0
        
        for mod, feat in all_features.items():
            feat = feat[:min_len]
            if violent_mask.sum() > 0:
                modality_violent[mod].append(feat[violent_mask])
            if (~violent_mask).sum() > 0:
                modality_nonviolent[mod].append(feat[~violent_mask])
        
        fused = fuser.fuse_features({k: v[:min_len] for k, v in all_features.items()})
        if fused is not None:
            if violent_mask.sum() > 0:
                fused_violent.append(fused[violent_mask])
            if (~violent_mask).sum() > 0:
                fused_nonviolent.append(fused[~violent_mask])
    
    def compute_metrics(violent_list, nonviolent_list, name):
        if not violent_list or not nonviolent_list:
            print(f"\n{name}: Insufficient data")
            return None
        
        v = np.vstack(violent_list)
        nv = np.vstack(nonviolent_list)
        
        v_mean = v.mean(axis=0)
        nv_mean = nv.mean(axis=0)
        
        diff = np.abs(v_mean - nv_mean)
        
        v_norm = np.linalg.norm(v_mean)
        nv_norm = np.linalg.norm(nv_mean)
        
        if v_norm < 1e-8 or nv_norm < 1e-8:
            cos_sim = 0.0
        else:
            cos_sim = np.dot(v_mean, nv_mean) / (v_norm * nv_norm)
        
        print(f"\n{name}:")
        print(f"  Violent frames: {len(v)}")
        print(f"  Non-violent frames: {len(nv)}")
        print(f"  Feature dim: {v.shape[1]}")
        print(f"  Mean abs difference: {diff.mean():.6f}")
        print(f"  Max difference: {diff.max():.6f}")
        print(f"  Dims with >0.1 diff: {(diff > 0.1).sum()}")
        print(f"  Cosine similarity: {cos_sim:.6f}")
        print(f"  --> Separability score: {1 - cos_sim:.6f} (higher = better)")
        
        return {
            'violent_frames': len(v),
            'nonviolent_frames': len(nv),
            'feature_dim': v.shape[1],
            'mean_diff': float(diff.mean()),
            'max_diff': float(diff.max()),
            'dims_gt_0.1': int((diff > 0.1).sum()),
            'cosine_similarity': float(cos_sim),
            'separability_score': float(1 - cos_sim),
        }
    
    results = {}
    
    for mod in modalities:
        if mod in modality_violent and mod in modality_nonviolent:
            results[mod] = compute_metrics(
                modality_violent[mod], 
                modality_nonviolent[mod], 
                f"{mod.upper()} Features"
            )
    
    if fused_violent and fused_nonviolent:
        results['fused'] = compute_metrics(
            fused_violent, 
            fused_nonviolent, 
            f"FUSED Features ({'+'.join(modalities)})"
        )
    
    return results


def build_fused_embeddings(
    vlm_dir: Optional[Path],
    audio_dir: Optional[Path],
    pose_dir: Optional[Path],
    output_dir: Path,
    modalities: List[str],
    source_filter: Optional[str] = None,
    strategy: str = 'concat_norm',
    skip_existing: bool = True,
):
    """Build fused embeddings for all videos."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fuser = MultiModalFuser(
        vlm_dir=vlm_dir,
        audio_dir=audio_dir,
        pose_dir=pose_dir,
        modalities=modalities,
        strategy=strategy,
    )
    
    primary_dir = vlm_dir or audio_dir or pose_dir
    if primary_dir is None:
        raise ValueError("At least one modality directory must be specified")
    
    feature_files = list(primary_dir.rglob('*.npz')) + list(primary_dir.rglob('*.npy'))
    video_names = list(set(f.stem for f in feature_files))
    
    if source_filter:
        video_names = [v for v in video_names if classify_video_source(v) == source_filter]
    
    print(f"Processing {len(video_names)} videos...")
    print(f"Modalities: {modalities}")
    print(f"Strategy: {strategy}")
    
    processed = 0
    skipped = 0
    errors = 0
    missing_modality = 0
    
    for video_name in tqdm(video_names, desc="Building fused embeddings"):
        source = classify_video_source(video_name)
        out_subdir = output_dir / source
        out_subdir.mkdir(exist_ok=True)
        
        out_path = out_subdir / f"{video_name}.npz"
        
        if skip_existing and out_path.exists():
            skipped += 1
            continue
        
        features = fuser.load_video_features(video_name)
        
        available = [k for k, v in features.items() if v is not None]
        if len(available) < len(modalities):
            missing_modality += 1
            continue
        
        try:
            fused = fuser.fuse_features(features)
            
            if fused is None or len(fused) == 0:
                errors += 1
                continue
            
            np.savez_compressed(
                out_path,
                emb=fused.astype(np.float16),
                modalities=modalities,
                strategy=strategy,
                dims={k: v for k, v in fuser.dims.items()},
            )
            processed += 1
            
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
            errors += 1
    
    print(f"\nDone!")
    print(f"  Processed: {processed}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Missing modality: {missing_modality}")
    print(f"  Errors: {errors}")
    print(f"  Total feature dim: {fuser.get_total_dim()}")
    
    metadata = {
        'modalities': modalities,
        'strategy': strategy,
        'dims': fuser.dims,
        'total_dim': fuser.get_total_dim(),
        'processed': processed,
        'source_filter': source_filter,
    }
    with open(output_dir / 'fusion_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nFused embeddings saved to: {output_dir}")
    return metadata


def build_fused_simulation(
    vlm_dir: Optional[Path],
    audio_dir: Optional[Path],
    pose_dir: Optional[Path],
    labels_dir: Path,
    output_path: Path,
    modalities: List[str],
    source_filter: Optional[str] = None,
    strategy: str = 'concat_norm',
    fps: float = 1.0,
    horizon_sec: float = 30.0,
    max_videos: Optional[int] = None,
):
    """Build a fused simulation file for evaluation."""
    print(f"Building fused simulation...")
    print(f"Modalities: {modalities}")
    print(f"Strategy: {strategy}")
    
    fuser = MultiModalFuser(
        vlm_dir=vlm_dir,
        audio_dir=audio_dir,
        pose_dir=pose_dir,
        labels_dir=labels_dir,
        modalities=modalities,
        strategy=strategy,
    )
    
    label_files = list(labels_dir.rglob('*.npy')) + list(labels_dir.rglob('*.npz'))
    video_names = list(set(f.stem for f in label_files))
    
    if source_filter:
        video_names = [v for v in video_names if classify_video_source(v) == source_filter]
    
    if max_videos:
        video_names = video_names[:max_videos]
    
    print(f"Processing {len(video_names)} videos...")
    
    all_embeddings = []
    all_frame_labels = []
    all_anticipation_labels = []
    video_boundaries = []
    valid_video_names = []
    
    current_pos = 0
    horizon_frames = int(horizon_sec * fps)
    
    for video_name in tqdm(video_names, desc="Loading videos"):
        features = fuser.load_video_features(video_name)
        labels = fuser.load_video_labels(video_name)
        
        if labels is None:
            continue
        
        available = {k: v for k, v in features.items() if v is not None}
        if len(available) < len(modalities):
            continue
        
        fused = fuser.fuse_features(features)
        if fused is None:
            continue
        
        min_len = min(len(fused), len(labels))
        fused = fused[:min_len]
        labels = labels[:min_len]
        
        if min_len == 0:
            continue
        
        anticipation_labels = np.zeros_like(labels)
        for t in range(len(labels)):
            future_end = min(t + horizon_frames, len(labels))
            anticipation_labels[t] = labels[t:future_end].max()
        
        start = current_pos
        end = current_pos + len(fused)
        video_boundaries.append([start, end])
        current_pos = end
        
        all_embeddings.append(fused)
        all_frame_labels.append(labels)
        all_anticipation_labels.append(anticipation_labels)
        valid_video_names.append(video_name)
    
    if not all_embeddings:
        raise RuntimeError("No valid videos found!")
    
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
        modalities=modalities,
        strategy=strategy,
        dims=fuser.dims,
    )
    
    print(f"\nSaved to: {output_path}")
    
    return {
        'n_videos': len(valid_video_names),
        'n_frames': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'modalities': modalities,
        'dims': fuser.dims,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-modal feature fusion for violence anticipation")
    parser.add_argument('--mode', type=str, required=True,
                        choices=['check_separability', 'build_embeddings', 'build_simulation'])
    
    parser.add_argument('--vlm_dir', type=str, default=None,
                        help='VLM embeddings directory')
    parser.add_argument('--audio_dir', type=str, default=None,
                        help='Audio features directory')
    parser.add_argument('--pose_dir', type=str, default=None,
                        help='Pose features directory')
    parser.add_argument('--labels_dir', type=str, default='data/processed/labels_multih',
                        help='Labels directory')
    parser.add_argument('--output_dir', type=str, default='data/processed/multimodal_features',
                        help='Output directory for fused embeddings')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path for simulation')
    
    parser.add_argument('--modalities', type=str, nargs='+', default=['vlm', 'audio'],
                        choices=['vlm', 'audio', 'pose'],
                        help='Modalities to fuse')
    parser.add_argument('--strategy', type=str, default='concat_norm',
                        choices=['concat', 'concat_norm', 'weighted'],
                        help='Fusion strategy')
    parser.add_argument('--source_filter', type=str, default=None,
                        choices=['xd_violence', 'ucf_crime'],
                        help='Filter by video source')
    
    parser.add_argument('--max_videos', type=int, default=None,
                        help='Maximum videos to process')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip existing files')
    parser.add_argument('--fps', type=float, default=1.0,
                        help='Frames per second')
    parser.add_argument('--horizon_sec', type=float, default=30.0,
                        help='Anticipation horizon in seconds')
    
    args = parser.parse_args()
    
    vlm_dir = Path(args.vlm_dir) if args.vlm_dir else None
    audio_dir = Path(args.audio_dir) if args.audio_dir else None
    pose_dir = Path(args.pose_dir) if args.pose_dir else None
    labels_dir = Path(args.labels_dir) if args.labels_dir else None
    
    if args.mode == 'check_separability':
        check_separability(
            vlm_dir=vlm_dir,
            audio_dir=audio_dir,
            pose_dir=pose_dir,
            labels_dir=labels_dir,
            source_filter=args.source_filter,
            max_videos=args.max_videos or 200,
            modalities=args.modalities,
            strategy=args.strategy,
        )
    
    elif args.mode == 'build_embeddings':
        build_fused_embeddings(
            vlm_dir=vlm_dir,
            audio_dir=audio_dir,
            pose_dir=pose_dir,
            output_dir=Path(args.output_dir),
            modalities=args.modalities,
            source_filter=args.source_filter,
            strategy=args.strategy,
            skip_existing=args.skip_existing,
        )
    
    elif args.mode == 'build_simulation':
        if args.output is None:
            raise ValueError("--output required for build_simulation mode")
        
        build_fused_simulation(
            vlm_dir=vlm_dir,
            audio_dir=audio_dir,
            pose_dir=pose_dir,
            labels_dir=labels_dir,
            output_path=Path(args.output),
            modalities=args.modalities,
            source_filter=args.source_filter,
            strategy=args.strategy,
            fps=args.fps,
            horizon_sec=args.horizon_sec,
            max_videos=args.max_videos,
        )


if __name__ == '__main__':
    main()