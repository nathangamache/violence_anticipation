#!/usr/bin/env python3
"""Build a long-form surveillance stream: violent test clips separated by long non-violent padding (~1% violence)."""

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def classify_video_source(video_id: str) -> str:
    if '_x264' in video_id:
        return 'ucf_crime'
    if video_id.startswith('v='):
        return 'xd_violence'
    if re.search(r'\d{4}__#', video_id) or '__#' in video_id:
        return 'xd_violence'
    return 'unknown'


def extract_video_name(item) -> str:
    if isinstance(item, dict):
        path = item.get('graph_path') or item.get('label_path', '')
        return Path(path).stem
    return str(item)


def find_embedding_file(video_name: str, embeddings_dir: Path) -> Optional[Path]:
    source = classify_video_source(video_name)
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
    for ext in ['.npy', '.npz']:
        path = labels_dir / f"{video_name}{ext}"
        if path.exists():
            return path
    source = classify_video_source(video_name)
    for subdir in [source, 'ucf_crime', 'xd_violence']:
        for ext in ['.npy', '.npz']:
            path = labels_dir / subdir / f"{video_name}{ext}"
            if path.exists():
                return path
    return None


def load_embedding(path: Path) -> Optional[np.ndarray]:
    try:
        if path.suffix == '.npz':
            data = np.load(path)
            for key in ['emb', 'embedding', 'embeddings', 'features', 'arr_0']:
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
        print(f"Error loading {path}: {e}")
    return None


def load_labels(path: Path) -> Optional[np.ndarray]:
    try:
        if path.suffix == '.npz':
            data = np.load(path)
            for key in ['labels', 'label', 'arr_0']:
                if key in data.keys():
                    return data[key]
            if data.keys():
                return data[list(data.keys())[0]]
        else:
            return np.load(path)
    except Exception as e:
        print(f"Error loading labels {path}: {e}")
    return None


@dataclass
class VideoSegment:
    video_id: str
    embeddings: np.ndarray  # [T, D]
    labels: np.ndarray      # [T]
    has_violence: bool
    start_time: float = 0.0
    

class LongFormSimulationBuilder:
    
    def __init__(
        self,
        embeddings_dir: Path,
        labels_dir: Path,
        fps: float = 1.0,
        horizon_sec: float = 30.0,
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.labels_dir = Path(labels_dir)
        self.fps = fps
        self.horizon_sec = horizon_sec
        
    def load_videos(
        self, 
        index_path: str, 
        source_filter: Optional[str] = None
    ) -> Tuple[List[VideoSegment], List[VideoSegment]]:
        """Return (violent_videos, nonviolent_videos) from the index, optionally filtered by source."""
        with open(index_path) as f:
            items = json.load(f)
        
        violent_videos = []
        nonviolent_videos = []
        
        for item in items:
            video_name = extract_video_name(item)
            
            if source_filter and classify_video_source(video_name) != source_filter:
                continue
            
            emb_path = find_embedding_file(video_name, self.embeddings_dir)
            label_path = find_label_file(video_name, self.labels_dir)
            
            if emb_path is None or label_path is None:
                continue
            
            embeddings = load_embedding(emb_path)
            labels = load_labels(label_path)
            
            if embeddings is None or labels is None:
                continue
            
            if labels.ndim > 1:
                labels = labels.max(axis=1)
            labels = (labels > 0).astype(np.float32)
            
            min_len = min(len(embeddings), len(labels))
            embeddings = embeddings[:min_len]
            labels = labels[:min_len]
            
            has_violence = labels.max() > 0
            
            segment = VideoSegment(
                video_id=video_name,
                embeddings=embeddings,
                labels=labels,
                has_violence=has_violence,
            )
            
            if has_violence:
                violent_videos.append(segment)
            else:
                nonviolent_videos.append(segment)
        
        return violent_videos, nonviolent_videos
    
    def build_simulation(
        self,
        violent_videos: List[VideoSegment],
        nonviolent_videos: List[VideoSegment],
        target_duration_hours: float = 8.0,
        violence_percentage: float = 1.0,
        random_seed: int = 42,
    ) -> Dict:
        """Alternate uniform(0.5, 1.5) x average-padding stretches of non-violent clips with one violent clip."""
        np.random.seed(random_seed)
        
        target_frames = int(target_duration_hours * 3600 * self.fps)
        target_violent_frames = int(target_frames * violence_percentage / 100)
        target_nonviolent_frames = target_frames - target_violent_frames
        
        print(f"\nBuilding {target_duration_hours}-hour simulation:")
        print(f"  Target frames: {target_frames:,}")
        print(f"  Target violent frames: {target_violent_frames:,} ({violence_percentage}%)")
        print(f"  Target non-violent frames: {target_nonviolent_frames:,}")
        
        violent_frames_available = sum(len(v.embeddings) for v in violent_videos)
        nonviolent_frames_available = sum(len(v.embeddings) for v in nonviolent_videos)
        
        print(f"\nAvailable content:")
        print(f"  Violent: {len(violent_videos)} videos, {violent_frames_available:,} frames")
        print(f"  Non-violent: {len(nonviolent_videos)} videos, {nonviolent_frames_available:,} frames")
        
        stream_embeddings = []
        stream_labels = []
        stream_metadata = []
        
        current_time = 0.0
        
        violent_queue = violent_videos.copy()
        np.random.shuffle(violent_queue)
        
        nonviolent_queue = nonviolent_videos.copy()
        np.random.shuffle(nonviolent_queue)
        
        if len(violent_queue) > 0:
            avg_padding_frames = target_nonviolent_frames // len(violent_queue)
        else:
            avg_padding_frames = target_nonviolent_frames
        
        print(f"\nAverage non-violent padding between events: {avg_padding_frames} frames ({avg_padding_frames/self.fps/60:.1f} min)")
        
        violent_idx = 0
        nonviolent_idx = 0
        
        while len(stream_embeddings) < target_frames and violent_idx < len(violent_queue):
            
            padding_frames = int(avg_padding_frames * np.random.uniform(0.5, 1.5))
            frames_added = 0
            
            while frames_added < padding_frames and nonviolent_idx < len(nonviolent_queue):
                video = nonviolent_queue[nonviolent_idx]
                
                for t in range(len(video.embeddings)):
                    stream_embeddings.append(video.embeddings[t])
                    stream_labels.append(0.0)
                    stream_metadata.append({
                        'video_id': video.video_id,
                        'frame_idx': t,
                        'time': current_time,
                        'is_violent_video': False,
                    })
                    current_time += 1.0 / self.fps
                    frames_added += 1
                
                nonviolent_idx += 1
                
                if nonviolent_idx >= len(nonviolent_queue):
                    nonviolent_idx = 0
                    np.random.shuffle(nonviolent_queue)
            
            if violent_idx < len(violent_queue):
                video = violent_queue[violent_idx]
                
                for t in range(len(video.embeddings)):
                    stream_embeddings.append(video.embeddings[t])
                    stream_labels.append(float(video.labels[t]))
                    stream_metadata.append({
                        'video_id': video.video_id,
                        'frame_idx': t,
                        'time': current_time,
                        'is_violent_video': True,
                    })
                    current_time += 1.0 / self.fps
                
                violent_idx += 1
        
        stream_embeddings = np.array(stream_embeddings)
        stream_labels = np.array(stream_labels)
        
        horizon_frames = int(self.horizon_sec * self.fps)
        anticipation_labels = np.zeros(len(stream_labels))
        
        for t in range(len(stream_labels)):
            future_end = min(t + horizon_frames, len(stream_labels))
            anticipation_labels[t] = stream_labels[t:future_end].max()
        
        actual_violent = (stream_labels > 0).sum()
        actual_anticipation = (anticipation_labels > 0).sum()
        
        print(f"\nSimulation built:")
        print(f"  Total frames: {len(stream_embeddings):,}")
        print(f"  Duration: {len(stream_embeddings) / self.fps / 3600:.2f} hours")
        print(f"  Violent frames: {actual_violent:,} ({100*actual_violent/len(stream_labels):.2f}%)")
        print(f"  Anticipation-positive frames: {actual_anticipation:,} ({100*actual_anticipation/len(anticipation_labels):.2f}%)")
        print(f"  Violent events: {violent_idx}")
        
        return {
            'embeddings': stream_embeddings,
            'frame_labels': stream_labels,
            'anticipation_labels': anticipation_labels,
            'metadata': stream_metadata,
            'fps': self.fps,
            'horizon_sec': self.horizon_sec,
            'duration_sec': len(stream_embeddings) / self.fps,
            'n_violent_events': violent_idx,
        }
    
    def save_simulation(self, simulation: Dict, output_path: str):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            output_path,
            embeddings=simulation['embeddings'],
            frame_labels=simulation['frame_labels'],
            anticipation_labels=simulation['anticipation_labels'],
            fps=simulation['fps'],
            horizon_sec=simulation['horizon_sec'],
        )
        
        meta_path = output_path.with_suffix('.meta.json')
        meta = {
            'fps': simulation['fps'],
            'horizon_sec': simulation['horizon_sec'],
            'duration_sec': simulation['duration_sec'],
            'n_frames': len(simulation['embeddings']),
            'n_violent_events': simulation['n_violent_events'],
            'n_violent_frames': int((simulation['frame_labels'] > 0).sum()),
            'n_anticipation_frames': int((simulation['anticipation_labels'] > 0).sum()),
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"\nSaved simulation to:")
        print(f"  {output_path}")
        print(f"  {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Build long-form video simulation")
    
    parser.add_argument('--embeddings_dir', type=str, default='data/processed/vlm_emb',
                        help='Directory containing embeddings')
    parser.add_argument('--labels_dir', type=str, default='data/processed/labels_multih',
                        help='Directory containing labels')
    parser.add_argument('--test_index', type=str, default='test_index.json',
                        help='Test index file')
    parser.add_argument('--source_filter', type=str, default='xd_violence',
                        help='Filter to specific source')
    parser.add_argument('--duration_hours', type=float, default=8.0,
                        help='Target simulation duration in hours')
    parser.add_argument('--violence_pct', type=float, default=1.0,
                        help='Target violence percentage (e.g., 1.0 for 1%%)')
    parser.add_argument('--horizon_sec', type=float, default=30.0,
                        help='Anticipation horizon in seconds')
    parser.add_argument('--fps', type=float, default=1.0,
                        help='Frames per second')
    parser.add_argument('--output', type=str, default='./simulation_data/long_form_sim.npz',
                        help='Output path for simulation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    print("="*70)
    print("LONG-FORM VIDEO SIMULATION BUILDER")
    print("="*70)
    
    builder = LongFormSimulationBuilder(
        embeddings_dir=Path(args.embeddings_dir),
        labels_dir=Path(args.labels_dir),
        fps=args.fps,
        horizon_sec=args.horizon_sec,
    )
    
    print(f"\nLoading videos from {args.test_index}...")
    violent_videos, nonviolent_videos = builder.load_videos(
        args.test_index,
        source_filter=args.source_filter,
    )
    
    print(f"Loaded {len(violent_videos)} violent, {len(nonviolent_videos)} non-violent videos")
    
    if len(violent_videos) == 0:
        print("ERROR: No violent videos found!")
        return
    
    simulation = builder.build_simulation(
        violent_videos=violent_videos,
        nonviolent_videos=nonviolent_videos,
        target_duration_hours=args.duration_hours,
        violence_percentage=args.violence_pct,
        random_seed=args.seed,
    )
    
    builder.save_simulation(simulation, args.output)
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == '__main__':
    main()