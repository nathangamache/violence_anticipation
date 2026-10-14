#!/usr/bin/env python3
"""Create a diluted simulation (lower violent fraction) by adding more non-violent footage.

Usage:
    python sim/create_diluted_simulation.py \
        --violence_sim simulation_data/xd_violence_vlm_clean_t02.npz \
        --output simulation_data/xd_violence_diluted_5pct.npz \
        --target_anomaly_pct 5.0 --total_duration_hours 8.0 \
        [--normal_emb_dir data/processed/vlm_emb/normal_surveillance]
"""

import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple
import json


def load_simulation(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        'embeddings': data['embeddings'],
        'anticipation_labels': data.get('anticipation_labels', data.get('labels', data.get('frame_labels'))),
        'fps': float(data.get('fps', 1.0)),
    }


def load_normal_embeddings(emb_dir: str, max_files: int = None) -> List[np.ndarray]:
    emb_dir = Path(emb_dir)
    embeddings = []
    
    files = sorted(emb_dir.glob('*.npz'))
    if max_files:
        files = files[:max_files]
    
    for f in files:
        try:
            z = np.load(f)
            emb = z.get('emb', z.get('embeddings', z.get('arr_0')))
            if emb is not None and len(emb) > 0:
                embeddings.append(emb.astype(np.float32))
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
            continue
    
    return embeddings


def create_diluted_simulation(
    violence_sim_path: str,
    output_path: str,
    target_anomaly_pct: float = 5.0,
    total_duration_hours: float = 8.0,
    fps: float = 1.0,
    normal_emb_dir: str = None,
    seed: int = 42,
):
    """Interleave the source sim's violent runs with normal frames so they make up target_anomaly_pct."""
    np.random.seed(seed)
    
    print(f"Loading violence simulation from {violence_sim_path}...")
    violence_data = load_simulation(violence_sim_path)
    v_emb = violence_data['embeddings'].astype(np.float32)
    v_labels = violence_data['anticipation_labels']
    
    if v_labels.ndim > 1:
        v_labels = v_labels.max(axis=1)
    v_labels = (v_labels > 0).astype(np.int32)
    
    n_violent_frames = v_labels.sum()
    n_nonviolent_frames = len(v_labels) - n_violent_frames
    current_pct = 100.0 * n_violent_frames / len(v_labels)
    
    print(f"Original: {len(v_labels)} frames, {n_violent_frames} violent ({current_pct:.1f}%)")
    
    total_frames = int(total_duration_hours * 3600 * fps)
    target_violent_frames = int(total_frames * target_anomaly_pct / 100.0)
    
    if target_violent_frames > n_violent_frames:
        print(f"Warning: Not enough violent frames. Using all {n_violent_frames}")
        target_violent_frames = n_violent_frames
    
    target_normal_frames = total_frames - target_violent_frames
    
    print(f"Target: {total_frames} frames total, {target_violent_frames} violent ({target_anomaly_pct:.1f}%)")
    print(f"Need {target_normal_frames} normal frames")
    
    violent_indices = np.where(v_labels == 1)[0]
    nonviolent_indices = np.where(v_labels == 0)[0]
    
    if target_violent_frames < n_violent_frames:
        selected_violent = sample_contiguous_segments(violent_indices, target_violent_frames)
    else:
        selected_violent = violent_indices
    
    violent_emb = v_emb[selected_violent]
    violent_labels = np.ones(len(selected_violent), dtype=np.int32)
    
    print(f"Selected {len(violent_emb)} violent frames")
    
    normal_embeddings = []
    
    if len(nonviolent_indices) > 0:
        normal_embeddings.append(v_emb[nonviolent_indices])
        print(f"Added {len(nonviolent_indices)} non-violent frames from source")
    
    if normal_emb_dir and Path(normal_emb_dir).exists():
        external_emb = load_normal_embeddings(normal_emb_dir)
        if external_emb:
            for emb in external_emb:
                normal_embeddings.append(emb)
            total_external = sum(len(e) for e in external_emb)
            print(f"Added {total_external} frames from {len(external_emb)} external files")
    
    all_normal = np.vstack(normal_embeddings) if normal_embeddings else v_emb[nonviolent_indices]
    
    if len(all_normal) >= target_normal_frames:
        indices = np.random.choice(len(all_normal), target_normal_frames, replace=False)
        normal_emb = all_normal[indices]
    else:
        repeats = target_normal_frames // len(all_normal) + 1
        normal_emb = np.tile(all_normal, (repeats, 1))[:target_normal_frames]
        # Add small noise to avoid exact duplicates
        noise = np.random.randn(*normal_emb.shape) * 0.01
        normal_emb = (normal_emb + noise).astype(np.float32)
    
    normal_labels = np.zeros(len(normal_emb), dtype=np.int32)
    print(f"Using {len(normal_emb)} normal frames")
    
    n_violence_events = len(find_contiguous_segments(violent_indices))
    event_spacing = len(normal_emb) // (n_violence_events + 1)
    
    print(f"Interleaving {n_violence_events} violence events with ~{event_spacing} frame spacing")
    
    final_emb = []
    final_labels = []
    
    violent_segments = split_into_events(violent_emb, violent_labels, violent_indices)
    
    normal_chunk_size = len(normal_emb) // (len(violent_segments) + 1)
    normal_chunks = [normal_emb[i:i+normal_chunk_size] for i in range(0, len(normal_emb), normal_chunk_size)]
    
    for i, (v_seg, v_lab) in enumerate(violent_segments):
        if i < len(normal_chunks):
            final_emb.append(normal_chunks[i])
            final_labels.append(np.zeros(len(normal_chunks[i]), dtype=np.int32))
        final_emb.append(v_seg)
        final_labels.append(v_lab)
    
    if len(normal_chunks) > len(violent_segments):
        for chunk in normal_chunks[len(violent_segments):]:
            final_emb.append(chunk)
            final_labels.append(np.zeros(len(chunk), dtype=np.int32))
    
    final_emb = np.vstack(final_emb).astype(np.float32)
    final_labels = np.concatenate(final_labels).astype(np.int32)
    
    final_violent = final_labels.sum()
    final_pct = 100.0 * final_violent / len(final_labels)
    duration_hrs = len(final_labels) / fps / 3600
    
    print(f"\nFinal simulation:")
    print(f"  Frames: {len(final_labels)}")
    print(f"  Duration: {duration_hrs:.2f} hours")
    print(f"  Violent frames: {final_violent} ({final_pct:.1f}%)")
    print(f"  Normal frames: {len(final_labels) - final_violent} ({100-final_pct:.1f}%)")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        embeddings=final_emb,
        anticipation_labels=final_labels,
        fps=np.array([fps]),
        metadata=json.dumps({
            'source': violence_sim_path,
            'target_anomaly_pct': target_anomaly_pct,
            'actual_anomaly_pct': final_pct,
            'duration_hours': duration_hrs,
        })
    )
    print(f"\nSaved to {output_path}")


def sample_contiguous_segments(indices: np.ndarray, target_count: int) -> np.ndarray:
    """Keep a proportional prefix of each contiguous run of indices, up to target_count."""
    segments = find_contiguous_segments(indices)
    
    total = sum(len(s) for s in segments)
    if total <= target_count:
        return indices
    
    sampled = []
    ratio = target_count / total
    
    for seg in segments:
        n_keep = max(1, int(len(seg) * ratio))
        sampled.extend(seg[:n_keep])
    
    return np.array(sampled[:target_count])


def find_contiguous_segments(indices: np.ndarray) -> List[np.ndarray]:
    if len(indices) == 0:
        return []
    
    segments = []
    current = [indices[0]]
    
    for i in range(1, len(indices)):
        if indices[i] == indices[i-1] + 1:
            current.append(indices[i])
        else:
            segments.append(np.array(current))
            current = [indices[i]]
    
    segments.append(np.array(current))
    return segments


def split_into_events(emb: np.ndarray, labels: np.ndarray, original_indices: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    segments = find_contiguous_segments(original_indices)
    
    events = []
    offset = 0
    
    for seg in segments:
        n = len(seg)
        events.append((emb[offset:offset+n], labels[offset:offset+n]))
        offset += n
    
    return events


def main():
    parser = argparse.ArgumentParser(description="Create diluted simulation for false alarm analysis")
    parser.add_argument('--violence_sim', type=str, required=True,
                       help='Path to existing violence simulation NPZ')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for diluted simulation')
    parser.add_argument('--target_anomaly_pct', type=float, default=5.0,
                       help='Target percentage of violent frames (default: 5.0)')
    parser.add_argument('--total_duration_hours', type=float, default=8.0,
                       help='Total duration in hours (default: 8.0)')
    parser.add_argument('--fps', type=float, default=1.0,
                       help='Frames per second (default: 1.0)')
    parser.add_argument('--normal_emb_dir', type=str, default=None,
                       help='Optional directory with additional normal embeddings')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    create_diluted_simulation(
        violence_sim_path=args.violence_sim,
        output_path=args.output,
        target_anomaly_pct=args.target_anomaly_pct,
        total_duration_hours=args.total_duration_hours,
        fps=args.fps,
        normal_emb_dir=args.normal_emb_dir,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()