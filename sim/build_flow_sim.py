#!/usr/bin/env python3
"""Build an optical-flow simulation (plus a VLM+flow fused copy) as long as an existing VLM simulation.

Usage:
    python sim/build_flow_sim.py \
        --vlm_sim simulation_data/xd_violence_8hr_sim_violence.npz \
        --flow_dir data/processed/optical_flow \
        --output_path simulation_data/xd_violence_8hr_flow_sim.npz
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def build_flow_simulation(vlm_sim_path: str, flow_dir: str, output_path: str):
    print(f"Loading VLM simulation from {vlm_sim_path}...")
    vlm_data = np.load(vlm_sim_path, allow_pickle=True)
    
    vlm_embeddings = vlm_data['embeddings']
    labels = vlm_data['anticipation_labels']
    fps = float(vlm_data.get('fps', 1.0))
    
    video_info = vlm_data.get('video_info', None)
    video_names = vlm_data.get('video_names', None)
    
    print(f"VLM simulation: {len(vlm_embeddings)} frames, {vlm_embeddings.shape[1]} dims")
    print(f"Labels shape: {labels.shape}")
    
    flow_dir = Path(flow_dir)
    
    if video_info is not None:
        print("Found video_info in simulation, attempting to match videos...")
        all_flow_features = []
        
        try:
            video_info = video_info.item() if hasattr(video_info, 'item') else video_info
        except:
            pass
        
    
    # Known broken: flow rows are not aligned with VLM rows (docs/codebase/audio_pose_flow_fusion.md 4.3).
    
    print(f"\nLooking for optical flow features in {flow_dir}...")
    flow_files = sorted(list(flow_dir.rglob("*.npz")))
    print(f"Found {len(flow_files)} optical flow files")
    
    if len(flow_files) == 0:
        print("ERROR: No optical flow files found!")
        return
    
    sample_data = np.load(flow_files[0])
    sample_features = sample_data.get('features', sample_data.get('embeddings', sample_data.get('arr_0')))
    flow_dim = sample_features.shape[1] if sample_features.ndim > 1 else len(sample_features)
    print(f"Flow feature dimension: {flow_dim}")
    
    all_flow = []
    
    for flow_path in tqdm(flow_files, desc="Loading flow features"):
        try:
            data = np.load(flow_path, allow_pickle=True)
            features = data.get('features', data.get('embeddings', data.get('arr_0')))
            if features is not None and len(features) > 0:
                all_flow.append(features)
        except Exception as e:
            print(f"Error loading {flow_path}: {e}")
    
    if not all_flow:
        print("ERROR: No valid flow features loaded!")
        return
    
    all_flow = np.concatenate(all_flow, axis=0)
    print(f"\nTotal flow features: {len(all_flow)} frames")
    print(f"VLM embeddings: {len(vlm_embeddings)} frames")
    
    if len(all_flow) != len(vlm_embeddings):
        print(f"\nWARNING: Frame count mismatch!")
        print(f"  Flow: {len(all_flow)}, VLM: {len(vlm_embeddings)}")
        
        if len(all_flow) > len(vlm_embeddings):
            print(f"  Truncating flow to match VLM length")
            all_flow = all_flow[:len(vlm_embeddings)]
        else:
            print(f"  Padding flow with zeros to match VLM length")
            padding = np.zeros((len(vlm_embeddings) - len(all_flow), flow_dim), dtype=np.float32)
            all_flow = np.concatenate([all_flow, padding], axis=0)
    
    print(f"\nSaving to {output_path}...")
    np.savez_compressed(
        output_path,
        embeddings=all_flow.astype(np.float32),
        anticipation_labels=labels,
        fps=fps,
        feature_dim=flow_dim,
    )
    
    print(f"Done! Flow simulation saved with {len(all_flow)} frames, {flow_dim} dimensions")
    
    fused_path = output_path.replace('.npz', '_fused.npz')
    print(f"\nCreating pre-fused simulation at {fused_path}...")
    
    flow_mean = all_flow.mean(axis=0, keepdims=True)
    flow_std = all_flow.std(axis=0, keepdims=True) + 1e-8
    all_flow_norm = (all_flow - flow_mean) / flow_std
    
    fused = np.concatenate([vlm_embeddings, all_flow_norm], axis=1)
    print(f"Fused dimension: {fused.shape[1]} (VLM: {vlm_embeddings.shape[1]} + Flow: {flow_dim})")
    
    np.savez_compressed(
        fused_path,
        embeddings=fused.astype(np.float32),
        anticipation_labels=labels,
        fps=fps,
        vlm_dim=vlm_embeddings.shape[1],
        flow_dim=flow_dim,
    )
    
    print(f"Done! Pre-fused simulation saved")


def main():
    parser = argparse.ArgumentParser(description="Build Optical Flow Simulation")
    parser.add_argument('--vlm_sim', type=str, required=True,
                        help='Path to existing VLM simulation NPZ')
    parser.add_argument('--flow_dir', type=str, required=True,
                        help='Directory containing optical flow features')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output path for flow simulation NPZ')
    
    args = parser.parse_args()
    
    build_flow_simulation(args.vlm_sim, args.flow_dir, args.output_path)


if __name__ == '__main__':
    main()