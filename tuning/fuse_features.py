#!/usr/bin/env python3
"""Early fusion of VLM embeddings, optical-flow features and optional VLM-QA features.

Usage:
    python fuse_features.py --vlm_dir data/processed/vlm_emb --flow_dir data/processed/optical_flow \
        --output_dir data/processed/fused_features [--qa_dir data/processed/vlm_qa_features]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


def load_features(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.load(path, allow_pickle=True)
        for key in ['emb', 'embedding', 'embeddings', 'features', 'arr_0']:
            if key in data.keys():
                arr = data[key]
                return arr.astype(np.float32) if arr.dtype == np.float16 else arr
        return None
    except:
        return None


def align_features(
    features_list: List[np.ndarray],
    names: List[str],
) -> List[np.ndarray]:
    """Subsample longer streams with linspace to the shortest stream's length."""
    lengths = [len(f) for f in features_list]
    min_len = min(lengths)
    
    aligned = []
    for feat, name, orig_len in zip(features_list, names, lengths):
        if len(feat) > min_len:
            indices = np.linspace(0, len(feat) - 1, min_len, dtype=int)
            feat = feat[indices]
        aligned.append(feat)
    
    return aligned


def normalize_features(
    features: np.ndarray,
    method: str = 'standard',
) -> np.ndarray:
    if method == 'standard':
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-8
        return (features - mean) / std
    elif method == 'minmax':
        min_val = features.min(axis=0, keepdims=True)
        max_val = features.max(axis=0, keepdims=True)
        return (features - min_val) / (max_val - min_val + 1e-8)
    elif method == 'l2':
        norm = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        return features / norm
    else:
        return features


def fuse_concat(feature_list: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(feature_list, axis=1)


def fuse_weighted(
    feature_list: List[np.ndarray],
    weights: List[float],
) -> np.ndarray:
    weighted = []
    for feat, w in zip(feature_list, weights):
        weighted.append(feat * w)
    return np.concatenate(weighted, axis=1)


def fuse_features_for_video(
    video_name: str,
    vlm_dir: Optional[Path],
    flow_dir: Optional[Path],
    qa_dir: Optional[Path],
    fusion_method: str = 'concat',
    normalize: str = 'standard',
    vlm_weight: float = 1.0,
    flow_weight: float = 1.0,
    qa_weight: float = 1.0,
) -> Optional[np.ndarray]:
    features_list = []
    names = []
    weights = []
    
    if vlm_dir:
        vlm_path = vlm_dir / f"{video_name}.npz"
        if not vlm_path.exists():
            for subdir in ['xd_violence', 'ucf_crime', '']:
                p = vlm_dir / subdir / f"{video_name}.npz" if subdir else vlm_path
                if p.exists():
                    vlm_path = p
                    break
        
        if vlm_path.exists():
            vlm_feat = load_features(vlm_path)
            if vlm_feat is not None:
                features_list.append(vlm_feat)
                names.append('vlm')
                weights.append(vlm_weight)
    
    if flow_dir:
        flow_path = flow_dir / f"{video_name}.npz"
        if not flow_path.exists():
            for subdir in ['xd_violence', 'ucf_crime', '']:
                p = flow_dir / subdir / f"{video_name}.npz" if subdir else flow_path
                if p.exists():
                    flow_path = p
                    break
        
        if flow_path.exists():
            flow_feat = load_features(flow_path)
            if flow_feat is not None:
                features_list.append(flow_feat)
                names.append('flow')
                weights.append(flow_weight)
    
    if qa_dir:
        qa_path = qa_dir / f"{video_name}.npz"
        if not qa_path.exists():
            for subdir in ['xd_violence', 'ucf_crime', '']:
                p = qa_dir / subdir / f"{video_name}.npz" if subdir else qa_path
                if p.exists():
                    qa_path = p
                    break
        
        if qa_path.exists():
            qa_feat = load_features(qa_path)
            if qa_feat is not None:
                features_list.append(qa_feat)
                names.append('qa')
                weights.append(qa_weight)
    
    if not features_list:
        return None
    
    features_list = align_features(features_list, names)
    
    if normalize:
        features_list = [normalize_features(f, normalize) for f in features_list]
    
    if fusion_method == 'concat':
        fused = fuse_concat(features_list)
    elif fusion_method == 'weighted':
        fused = fuse_weighted(features_list, weights)
    else:
        fused = fuse_concat(features_list)
    
    return fused


def process_all_videos(
    vlm_dir: Optional[Path],
    flow_dir: Optional[Path],
    qa_dir: Optional[Path],
    output_dir: Path,
    fusion_method: str = 'concat',
    normalize: str = 'standard',
    vlm_weight: float = 1.0,
    flow_weight: float = 1.0,
    qa_weight: float = 1.0,
    source_filter: str = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_names = set()
    
    for dir_path in [vlm_dir, flow_dir, qa_dir]:
        if dir_path and dir_path.exists():
            for npz_path in dir_path.rglob("*.npz"):
                video_names.add(npz_path.stem)
    
    print(f"Found {len(video_names)} unique videos")
    
    processed = 0
    skipped = 0
    
    dims_info = {}
    
    for video_name in tqdm(video_names, desc="Fusing features"):
        if source_filter:
            is_xd = '__#' in video_name or video_name.startswith('v=')
            is_ucf = '_x264' in video_name
            if source_filter == 'xd_violence' and not is_xd:
                continue
            if source_filter == 'ucf_crime' and not is_ucf:
                continue
        
        out_path = output_dir / f"{video_name}.npz"
        if out_path.exists():
            skipped += 1
            continue
        
        fused = fuse_features_for_video(
            video_name,
            vlm_dir,
            flow_dir,
            qa_dir,
            fusion_method,
            normalize,
            vlm_weight,
            flow_weight,
            qa_weight,
        )
        
        if fused is None:
            continue
        
        np.savez_compressed(out_path, emb=fused.astype(np.float16))
        processed += 1
        
        if 'dim' not in dims_info:
            dims_info['dim'] = fused.shape[1]
            dims_info['sample_video'] = video_name
            dims_info['n_frames'] = fused.shape[0]
    
    print(f"\nProcessed: {processed}, Skipped: {skipped}")
    
    if dims_info:
        print(f"Output dimension: {dims_info['dim']}")
    
    config = {
        'vlm_dir': str(vlm_dir) if vlm_dir else None,
        'flow_dir': str(flow_dir) if flow_dir else None,
        'qa_dir': str(qa_dir) if qa_dir else None,
        'fusion_method': fusion_method,
        'normalize': normalize,
        'weights': {
            'vlm': vlm_weight,
            'flow': flow_weight,
            'qa': qa_weight,
        },
        'output_dim': dims_info.get('dim'),
        'n_videos': processed,
    }
    
    with open(output_dir / 'fusion_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nOutputs saved to {output_dir}")
    print(f"Use with: --embeddings_dir {output_dir} --input_dim {dims_info.get('dim', 'unknown')}")


def main():
    parser = argparse.ArgumentParser(description="Multimodal Feature Fusion")
    
    parser.add_argument('--vlm_dir', type=str, default=None,
                        help='Directory with VLM embeddings')
    parser.add_argument('--flow_dir', type=str, default=None,
                        help='Directory with optical flow features')
    parser.add_argument('--qa_dir', type=str, default=None,
                        help='Directory with QA features')
    
    parser.add_argument('--output_dir', type=str, required=True)
    
    parser.add_argument('--fusion_method', type=str, default='concat',
                        choices=['concat', 'weighted'],
                        help='Fusion method')
    parser.add_argument('--normalize', type=str, default='standard',
                        choices=['standard', 'minmax', 'l2', 'none'],
                        help='Normalization method')
    
    parser.add_argument('--vlm_weight', type=float, default=1.0)
    parser.add_argument('--flow_weight', type=float, default=1.0)
    parser.add_argument('--qa_weight', type=float, default=1.0)
    
    parser.add_argument('--source_filter', type=str, default=None,
                        choices=['xd_violence', 'ucf_crime'])
    
    args = parser.parse_args()
    
    if not args.vlm_dir and not args.flow_dir and not args.qa_dir:
        print("Need at least one of: --vlm_dir, --flow_dir, --qa_dir")
        return
    
    process_all_videos(
        Path(args.vlm_dir) if args.vlm_dir else None,
        Path(args.flow_dir) if args.flow_dir else None,
        Path(args.qa_dir) if args.qa_dir else None,
        Path(args.output_dir),
        args.fusion_method,
        args.normalize if args.normalize != 'none' else None,
        args.vlm_weight,
        args.flow_weight,
        args.qa_weight,
        args.source_filter,
    )


if __name__ == '__main__':
    main()