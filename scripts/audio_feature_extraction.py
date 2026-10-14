#!/usr/bin/env python3
"""
GPU-Accelerated Audio Feature Extraction for Violence Anticipation

Uses torchaudio with GPU for faster feature extraction:
- Mel spectrograms computed on GPU
- MFCCs computed on GPU
- Batch processing for efficiency

Features extracted (per frame, ~92 dimensions):
- MFCCs (mean + std): 40
- Mel spectrogram stats: 4
- RMS energy: 2
- Zero-crossing rate: 2
- Spectral features: 8
- Onset strength: 3
- Chroma: 12
- Delta MFCCs: 20

Usage:
    # Single process
    python scripts/audio_feature_extraction_gpu.py \
        --videos_dir data/raw_videos/xd_violence/videos \
        --graphs_dir data/processed/graphs \
        --output_dir data/processed/audio_features \
        --source_filter xd_violence \
        --device cuda:0

    # Multi-GPU (8 processes, 4 per GPU)
    # GPU 0: ranks 0,1,2,3
    # GPU 1: ranks 4,5,6,7
    python scripts/audio_feature_extraction_gpu.py ... --device cuda:0 --rank 0 --world_size 8
    python scripts/audio_feature_extraction_gpu.py ... --device cuda:0 --rank 1 --world_size 8
    ...
"""

import argparse
import json
import os
import subprocess
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')

import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F


class GPUAudioFeatureExtractor:
    """
    GPU-accelerated audio feature extraction using torchaudio.
    """
    
    def __init__(
        self,
        device: str = 'cuda:0',
        target_sr: int = 16000,
        n_mfcc: int = 20,
        n_mels: int = 64,
        n_fft: int = 2048,
        hop_length: int = 512,
        output_fps: float = 1.0,
    ):
        self.device = torch.device(device)
        self.target_sr = target_sr
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.output_fps = output_fps
        
        # Initialize GPU transforms
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        ).to(self.device)
        
        self.mfcc_transform = T.MFCC(
            sample_rate=target_sr,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'n_mels': n_mels,
            },
        ).to(self.device)
        
        self.spectrogram = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0,
        ).to(self.device)
        
        # Feature names for documentation
        self.feature_names = self._build_feature_names()
    
    def _build_feature_names(self) -> List[str]:
        """Build list of feature names."""
        names = []
        
        # MFCCs
        for i in range(self.n_mfcc):
            names.append(f'mfcc_{i}_mean')
        for i in range(self.n_mfcc):
            names.append(f'mfcc_{i}_std')
        
        # Mel spectrogram stats
        names.extend(['mel_mean', 'mel_std', 'mel_max', 'mel_min'])
        
        # Energy
        names.extend(['rms_mean', 'rms_max'])
        
        # Zero crossing
        names.extend(['zcr_mean', 'zcr_std'])
        
        # Spectral features
        names.extend(['spectral_centroid_mean', 'spectral_centroid_std'])
        names.extend(['spectral_bandwidth_mean', 'spectral_bandwidth_std'])
        names.extend(['spectral_rolloff_mean', 'spectral_rolloff_std'])
        names.extend(['spectral_flatness_mean', 'spectral_flatness_std'])
        
        # Onset
        names.extend(['onset_strength_mean', 'onset_strength_max', 'onset_count'])
        
        # Chroma
        for i in range(12):
            names.append(f'chroma_{i}_mean')
        
        # Delta MFCCs
        for i in range(self.n_mfcc):
            names.append(f'delta_mfcc_{i}_mean')
        
        return names
    
    def get_feature_dim(self) -> int:
        """Get total feature dimension."""
        return len(self.feature_names)
    
    def _compute_spectral_centroid(self, spec: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Compute spectral centroid on GPU."""
        # spec: [n_freq, time]
        # freqs: [n_freq]
        spec_sum = spec.sum(dim=0) + 1e-10
        centroid = (spec * freqs.unsqueeze(1)).sum(dim=0) / spec_sum
        return centroid
    
    def _compute_spectral_bandwidth(self, spec: torch.Tensor, freqs: torch.Tensor, centroid: torch.Tensor) -> torch.Tensor:
        """Compute spectral bandwidth on GPU."""
        spec_sum = spec.sum(dim=0) + 1e-10
        deviation = freqs.unsqueeze(1) - centroid.unsqueeze(0)
        bandwidth = torch.sqrt((spec * deviation ** 2).sum(dim=0) / spec_sum)
        return bandwidth
    
    def _compute_spectral_rolloff(self, spec: torch.Tensor, freqs: torch.Tensor, roll_percent: float = 0.85) -> torch.Tensor:
        """Compute spectral rolloff on GPU."""
        spec_sum = spec.sum(dim=0, keepdim=True)
        cumsum = torch.cumsum(spec, dim=0)
        threshold = roll_percent * spec_sum
        
        # Find first frequency bin where cumsum exceeds threshold
        mask = cumsum >= threshold
        # Get index of first True per column
        rolloff_idx = mask.float().argmax(dim=0)
        rolloff = freqs[rolloff_idx]
        return rolloff
    
    def _compute_spectral_flatness(self, spec: torch.Tensor) -> torch.Tensor:
        """Compute spectral flatness on GPU."""
        # Geometric mean / arithmetic mean
        log_spec = torch.log(spec + 1e-10)
        geo_mean = torch.exp(log_spec.mean(dim=0))
        arith_mean = spec.mean(dim=0) + 1e-10
        flatness = geo_mean / arith_mean
        return flatness
    
    def _compute_zcr(self, waveform: torch.Tensor, frame_length: int, hop_length: int) -> torch.Tensor:
        """Compute zero-crossing rate on GPU."""
        # Pad and frame the signal
        n_frames = 1 + (waveform.shape[-1] - frame_length) // hop_length
        
        zcr_values = []
        for i in range(n_frames):
            start = i * hop_length
            end = start + frame_length
            frame = waveform[start:end]
            
            # Count zero crossings
            signs = torch.sign(frame)
            signs[signs == 0] = 1
            crossings = (signs[1:] != signs[:-1]).float().sum()
            zcr = crossings / (2 * frame_length)
            zcr_values.append(zcr)
        
        return torch.stack(zcr_values)
    
    def _compute_onset_strength(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Compute onset strength envelope on GPU."""
        # Convert to dB
        mel_db = 10 * torch.log10(mel_spec + 1e-10)
        
        # Compute difference along time axis
        diff = torch.diff(mel_db, dim=1)
        
        # Half-wave rectify and sum across frequency
        onset = torch.clamp(diff, min=0).mean(dim=0)
        
        return onset
    
    def _compute_chroma(self, spec: torch.Tensor, sr: int, n_fft: int) -> torch.Tensor:
        """Compute chroma features on GPU."""
        # Frequency bins
        freqs = torch.fft.rfftfreq(n_fft, 1/sr).to(self.device)
        
        # Convert to pitch classes (12 bins)
        # f = 440 * 2^((p-69)/12) => p = 69 + 12*log2(f/440)
        with torch.no_grad():
            # Avoid log of zero
            freqs_safe = torch.clamp(freqs, min=1e-10)
            pitches = 69 + 12 * torch.log2(freqs_safe / 440.0)
            chroma_bins = (pitches % 12).long()
            chroma_bins = torch.clamp(chroma_bins, 0, 11)
        
        # Aggregate into chroma
        n_time = spec.shape[1]
        chroma = torch.zeros(12, n_time, device=self.device)
        
        for i in range(12):
            mask = (chroma_bins == i)
            if mask.any():
                chroma[i] = spec[mask].sum(dim=0)
        
        # Normalize
        chroma = chroma / (chroma.sum(dim=0, keepdim=True) + 1e-10)
        
        return chroma
    
    def extract_frame_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract features from a single audio segment on GPU.
        
        Args:
            waveform: [samples] tensor on GPU
            
        Returns:
            Feature vector [feature_dim]
        """
        if waveform.numel() == 0 or waveform.abs().max() < 1e-10:
            return torch.zeros(self.get_feature_dim(), device=self.device)
        
        features = []
        
        try:
            # Ensure 2D for transforms: [1, samples]
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            # MFCCs: [1, n_mfcc, time]
            mfccs = self.mfcc_transform(waveform).squeeze(0)  # [n_mfcc, time]
            features.extend(mfccs.mean(dim=1).tolist())  # Mean per coefficient
            features.extend(mfccs.std(dim=1).tolist())   # Std per coefficient
            
            # Mel spectrogram: [1, n_mels, time]
            mel_spec = self.mel_spectrogram(waveform).squeeze(0)  # [n_mels, time]
            mel_db = 10 * torch.log10(mel_spec + 1e-10)
            features.extend([mel_db.mean().item(), mel_db.std().item(), 
                           mel_db.max().item(), mel_db.min().item()])
            
            # RMS energy
            rms = torch.sqrt(torch.mean(waveform ** 2, dim=-1))
            features.extend([rms.mean().item(), rms.max().item()])
            
            # Zero crossing rate (simplified - compute on CPU for stability)
            wav_1d = waveform.squeeze()
            signs = torch.sign(wav_1d)
            signs[signs == 0] = 1
            zcr = (signs[1:] != signs[:-1]).float().mean()
            features.extend([zcr.item(), 0.0])  # std placeholder
            
            # Power spectrogram for spectral features
            spec = self.spectrogram(waveform).squeeze(0)  # [n_freq, time]
            
            # Frequency bins
            freqs = torch.linspace(0, self.target_sr / 2, spec.shape[0], device=self.device)
            
            # Spectral centroid
            centroid = self._compute_spectral_centroid(spec, freqs)
            features.extend([centroid.mean().item() / self.target_sr, 
                           centroid.std().item() / self.target_sr])
            
            # Spectral bandwidth
            bandwidth = self._compute_spectral_bandwidth(spec, freqs, centroid)
            features.extend([bandwidth.mean().item() / self.target_sr,
                           bandwidth.std().item() / self.target_sr])
            
            # Spectral rolloff
            rolloff = self._compute_spectral_rolloff(spec, freqs)
            features.extend([rolloff.mean().item() / self.target_sr,
                           rolloff.std().item() / self.target_sr])
            
            # Spectral flatness
            flatness = self._compute_spectral_flatness(spec)
            features.extend([flatness.mean().item(), flatness.std().item()])
            
            # Onset strength
            onset = self._compute_onset_strength(mel_spec)
            onset_count = (onset > onset.mean() + onset.std()).sum().item()
            features.extend([onset.mean().item(), onset.max().item(), onset_count])
            
            # Chroma
            chroma = self._compute_chroma(spec, self.target_sr, self.n_fft)
            features.extend(chroma.mean(dim=1).tolist())
            
            # Delta MFCCs (temporal derivative)
            if mfccs.shape[1] > 2:
                delta_mfccs = mfccs[:, 1:] - mfccs[:, :-1]
                features.extend(delta_mfccs.mean(dim=1).tolist())
            else:
                features.extend([0.0] * self.n_mfcc)
            
        except Exception as e:
            # Return zeros on error
            return torch.zeros(self.get_feature_dim(), device=self.device)
        
        # Convert to tensor and ensure correct length
        features_tensor = torch.tensor(features, device=self.device, dtype=torch.float32)
        
        target_dim = self.get_feature_dim()
        if len(features_tensor) < target_dim:
            features_tensor = torch.nn.functional.pad(
                features_tensor, (0, target_dim - len(features_tensor))
            )
        elif len(features_tensor) > target_dim:
            features_tensor = features_tensor[:target_dim]
        
        # Replace NaN/Inf
        features_tensor = torch.nan_to_num(features_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features_tensor
    
    def extract_video_features(
        self,
        video_path: Path,
        num_frames: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Extract audio features from a video file.
        
        Args:
            video_path: Path to video file
            num_frames: Expected number of output frames
            
        Returns:
            features: [T, feature_dim] numpy array
            metadata: Dict with extraction info
        """
        metadata = {
            'video_path': str(video_path),
            'target_sr': self.target_sr,
            'output_fps': self.output_fps,
        }
        
        try:
            # Load audio using torchaudio
            waveform, sr = torchaudio.load(str(video_path))
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if sr != self.target_sr:
                resampler = T.Resample(sr, self.target_sr)
                waveform = resampler(waveform)
            
            waveform = waveform.squeeze(0)  # [samples]
            
            if waveform.numel() == 0:
                metadata['error'] = 'empty_audio'
                if num_frames:
                    return np.zeros((num_frames, self.get_feature_dim()), dtype=np.float32), metadata
                return np.array([]), metadata
            
            # Move to GPU
            waveform = waveform.to(self.device)
            
            duration = waveform.shape[0] / self.target_sr
            metadata['duration'] = duration
            metadata['original_samples'] = waveform.shape[0]
            
            # Determine number of output frames
            if num_frames is not None:
                n_output_frames = num_frames
            else:
                n_output_frames = int(duration * self.output_fps)
            
            n_output_frames = max(1, n_output_frames)
            metadata['n_output_frames'] = n_output_frames
            
            # Samples per output frame
            samples_per_frame = int(self.target_sr / self.output_fps)
            
            # Extract features for each frame
            features = []
            
            for i in range(n_output_frames):
                start_sample = int(i * samples_per_frame)
                end_sample = int((i + 1) * samples_per_frame)
                
                if start_sample >= waveform.shape[0]:
                    frame_features = torch.zeros(self.get_feature_dim(), device=self.device)
                else:
                    end_sample = min(end_sample, waveform.shape[0])
                    segment = waveform[start_sample:end_sample]
                    
                    if segment.shape[0] < samples_per_frame // 4:
                        frame_features = torch.zeros(self.get_feature_dim(), device=self.device)
                    else:
                        frame_features = self.extract_frame_features(segment)
                
                features.append(frame_features)
            
            # Stack and convert to numpy
            features = torch.stack(features).cpu().numpy().astype(np.float32)
            metadata['success'] = True
            
            return features, metadata
            
        except Exception as e:
            metadata['error'] = str(e)
            if num_frames:
                return np.zeros((num_frames, self.get_feature_dim()), dtype=np.float32), metadata
            return np.array([]), metadata


def build_video_index(root: str) -> Dict[str, str]:
    """Map video stem -> absolute path."""
    out = {}
    rp = Path(root)
    if not rp.exists():
        return out
    
    for ext in ['.mp4', '.avi', '.mkv', '.mov', '.webm']:
        for p in rp.rglob(f'*{ext}'):
            stem = p.stem
            out[stem] = str(p)
            out[stem.lower()] = str(p)
            if stem.endswith("_x264"):
                base = stem[:-5]
                out[base] = str(p)
                out[base.lower()] = str(p)
    
    return out


def classify_video_source(video_id: str) -> str:
    """Classify video source from ID."""
    if '_x264' in video_id:
        return 'ucf_crime'
    if video_id.startswith('v='):
        return 'xd_violence'
    if '__#' in video_id:
        return 'xd_violence'
    return 'unknown'


def main():
    parser = argparse.ArgumentParser(description="GPU-accelerated audio feature extraction")
    parser.add_argument('--videos_dir', type=str, required=True)
    parser.add_argument('--graphs_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='data/processed/audio_features')
    parser.add_argument('--source_filter', type=str, default=None,
                        choices=['xd_violence', 'ucf_crime'])
    parser.add_argument('--target_sr', type=int, default=16000)
    parser.add_argument('--output_fps', type=float, default=1.0)
    parser.add_argument('--n_mfcc', type=int, default=20)
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build video index
    print(f"[Rank {args.rank}] Building video index...")
    video_index = build_video_index(args.videos_dir)
    print(f"[Rank {args.rank}] Found {len(video_index) // 2} videos")
    
    # Get video IDs from graphs
    graphs_dir = Path(args.graphs_dir)
    graph_files = sorted([f for f in os.listdir(graphs_dir) 
                         if f.endswith('.npy') or f.endswith('.pt')])
    vid_ids = [os.path.splitext(f)[0] for f in graph_files]
    
    # Filter by source
    if args.source_filter:
        vid_ids = [v for v in vid_ids if classify_video_source(v) == args.source_filter]
    
    # Shard across processes
    vid_ids = vid_ids[args.rank::args.world_size]
    print(f"[Rank {args.rank}] Processing {len(vid_ids)} videos on {args.device}")
    
    # Initialize extractor
    extractor = GPUAudioFeatureExtractor(
        device=args.device,
        target_sr=args.target_sr,
        n_mfcc=args.n_mfcc,
        output_fps=args.output_fps,
    )
    
    feature_dim = extractor.get_feature_dim()
    print(f"[Rank {args.rank}] Feature dimension: {feature_dim}")
    
    # Process videos
    processed = 0
    skipped = 0
    errors = 0
    no_video = 0
    
    for vid_id in tqdm(vid_ids, desc=f"[Rank {args.rank}] Extracting"):
        source = classify_video_source(vid_id)
        out_subdir = output_dir / source
        out_subdir.mkdir(exist_ok=True)
        
        out_path = out_subdir / f"{vid_id}.npz"
        
        if args.skip_existing and out_path.exists():
            skipped += 1
            continue
        
        # Find video
        video_path = video_index.get(vid_id) or video_index.get(vid_id.lower())
        if not video_path:
            no_video += 1
            continue
        
        # Get T from graph
        graph_path = graphs_dir / f"{vid_id}.npy"
        if not graph_path.exists():
            graph_path = graphs_dir / f"{vid_id}.pt"
        
        try:
            if graph_path.suffix == '.npy':
                seq = np.load(graph_path, allow_pickle=True)
                T = len(seq)
            else:
                seq = torch.load(graph_path, map_location='cpu')
                T = len(seq)
        except Exception as e:
            errors += 1
            continue
        
        # Extract features
        try:
            features, metadata = extractor.extract_video_features(
                Path(video_path),
                num_frames=T,
            )
            
            if features.size == 0:
                errors += 1
                continue
            
            # Save
            np.savez_compressed(
                out_path,
                features=features,
                feature_dim=feature_dim,
            )
            
            processed += 1
            
        except Exception as e:
            print(f"[Rank {args.rank}] Error processing {vid_id}: {e}")
            errors += 1
    
    print(f"\n[Rank {args.rank}] Done!")
    print(f"  Processed: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  No video: {no_video}")
    print(f"  Errors: {errors}")


if __name__ == '__main__':
    main()