#!/usr/bin/env python3
"""
Check audio availability in XD-Violence videos.

This script analyzes videos to determine:
1. How many videos have audio streams
2. Audio properties (sample rate, channels, duration)
3. Whether audio is silent or has actual content

Usage:
    python scripts/check_audio_availability.py \
        --videos_dir data/raw_videos/xd_violence/videos \
        --output audio_analysis_report.json
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
from tqdm import tqdm

# Try to import audio processing libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not installed. Run: pip install librosa")

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


def get_video_info_ffprobe(video_path: Path) -> Dict:
    """
    Use ffprobe to get detailed video/audio stream information.
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            return {'error': 'ffprobe failed', 'stderr': result.stderr}
        
        return json.loads(result.stdout)
    
    except subprocess.TimeoutExpired:
        return {'error': 'ffprobe timeout'}
    except json.JSONDecodeError:
        return {'error': 'ffprobe output parse error'}
    except FileNotFoundError:
        return {'error': 'ffprobe not found'}
    except Exception as e:
        return {'error': str(e)}


def analyze_audio_content(video_path: Path, duration_sec: float = 10.0) -> Dict:
    """
    Analyze actual audio content to check if it's silent or has signal.
    
    Samples the first `duration_sec` seconds of audio.
    """
    if not LIBROSA_AVAILABLE:
        return {'error': 'librosa not available'}
    
    try:
        # Load a sample of audio
        y, sr = librosa.load(str(video_path), sr=16000, mono=True, duration=duration_sec)
        
        if len(y) == 0:
            return {'has_content': False, 'reason': 'empty audio'}
        
        # Calculate audio statistics
        rms = np.sqrt(np.mean(y ** 2))
        peak = np.max(np.abs(y))
        
        # Check if audio is essentially silent
        # RMS below -60dB is typically considered silent
        rms_db = 20 * np.log10(rms + 1e-10)
        
        # Calculate zero crossing rate (speech/music have higher ZCR than silence)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Spectral centroid (silent audio has very low centroid)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        has_content = rms_db > -50 and peak > 0.01
        
        return {
            'has_content': has_content,
            'rms': float(rms),
            'rms_db': float(rms_db),
            'peak': float(peak),
            'zero_crossing_rate': float(zcr),
            'spectral_centroid': float(spectral_centroid),
            'sample_rate': sr,
            'samples_analyzed': len(y),
            'duration_analyzed': len(y) / sr,
        }
    
    except Exception as e:
        return {'error': str(e)}


def check_video_audio(video_path: Path, analyze_content: bool = True) -> Dict:
    """
    Check if a video has audio and analyze its properties.
    """
    result = {
        'path': str(video_path),
        'filename': video_path.name,
        'has_audio_stream': False,
        'audio_streams': [],
    }
    
    # Get info via ffprobe
    info = get_video_info_ffprobe(video_path)
    
    if 'error' in info:
        result['error'] = info['error']
        return result
    
    # Check for audio streams
    streams = info.get('streams', [])
    audio_streams = [s for s in streams if s.get('codec_type') == 'audio']
    
    result['has_audio_stream'] = len(audio_streams) > 0
    result['num_audio_streams'] = len(audio_streams)
    
    for i, stream in enumerate(audio_streams):
        stream_info = {
            'index': stream.get('index'),
            'codec': stream.get('codec_name'),
            'sample_rate': stream.get('sample_rate'),
            'channels': stream.get('channels'),
            'channel_layout': stream.get('channel_layout'),
            'duration': stream.get('duration'),
            'bit_rate': stream.get('bit_rate'),
        }
        result['audio_streams'].append(stream_info)
    
    # Get format info
    format_info = info.get('format', {})
    result['duration'] = format_info.get('duration')
    result['format'] = format_info.get('format_name')
    result['size_bytes'] = format_info.get('size')
    
    # Analyze actual audio content
    if analyze_content and result['has_audio_stream'] and LIBROSA_AVAILABLE:
        content_analysis = analyze_audio_content(video_path)
        result['content_analysis'] = content_analysis
        result['has_audio_content'] = content_analysis.get('has_content', False)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Check audio availability in videos")
    parser.add_argument('--videos_dir', type=str, required=True,
                        help='Directory containing videos')
    parser.add_argument('--output', type=str, default='audio_analysis_report.json',
                        help='Output JSON file for detailed report')
    parser.add_argument('--analyze_content', action='store_true', default=True,
                        help='Analyze actual audio content (not just stream presence)')
    parser.add_argument('--no_analyze_content', action='store_false', dest='analyze_content',
                        help='Skip audio content analysis (faster)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of videos to check')
    parser.add_argument('--extensions', type=str, nargs='+', 
                        default=['.mp4', '.avi', '.mkv', '.mov', '.webm'],
                        help='Video file extensions to check')
    
    args = parser.parse_args()
    
    videos_dir = Path(args.videos_dir)
    
    if not videos_dir.exists():
        print(f"Error: Directory not found: {videos_dir}")
        return
    
    # Find all video files
    video_files = []
    for ext in args.extensions:
        video_files.extend(videos_dir.rglob(f'*{ext}'))
        video_files.extend(videos_dir.rglob(f'*{ext.upper()}'))
    
    video_files = sorted(set(video_files))
    
    if args.limit:
        video_files = video_files[:args.limit]
    
    print(f"Found {len(video_files)} video files")
    print(f"Librosa available: {LIBROSA_AVAILABLE}")
    print(f"Analyzing content: {args.analyze_content and LIBROSA_AVAILABLE}")
    print()
    
    # Analyze each video
    results = []
    stats = defaultdict(int)
    
    for video_path in tqdm(video_files, desc="Checking videos"):
        result = check_video_audio(video_path, analyze_content=args.analyze_content)
        results.append(result)
        
        # Update stats
        if 'error' in result:
            stats['errors'] += 1
        elif result['has_audio_stream']:
            stats['has_audio_stream'] += 1
            if result.get('has_audio_content', False):
                stats['has_audio_content'] += 1
            elif 'content_analysis' in result:
                stats['silent_audio'] += 1
        else:
            stats['no_audio_stream'] += 1
    
    # Print summary
    print("\n" + "="*60)
    print("AUDIO AVAILABILITY SUMMARY")
    print("="*60)
    print(f"Total videos checked: {len(video_files)}")
    print(f"Videos with audio stream: {stats['has_audio_stream']} ({100*stats['has_audio_stream']/len(video_files):.1f}%)")
    print(f"Videos with audio content: {stats['has_audio_content']} ({100*stats['has_audio_content']/len(video_files):.1f}%)")
    print(f"Videos with silent audio: {stats['silent_audio']} ({100*stats['silent_audio']/len(video_files):.1f}%)")
    print(f"Videos without audio stream: {stats['no_audio_stream']} ({100*stats['no_audio_stream']/len(video_files):.1f}%)")
    print(f"Errors: {stats['errors']} ({100*stats['errors']/len(video_files):.1f}%)")
    
    # Audio codec breakdown
    codec_counts = defaultdict(int)
    sample_rates = defaultdict(int)
    
    for r in results:
        for stream in r.get('audio_streams', []):
            codec_counts[stream.get('codec', 'unknown')] += 1
            sr = stream.get('sample_rate', 'unknown')
            sample_rates[sr] += 1
    
    if codec_counts:
        print("\nAudio codecs:")
        for codec, count in sorted(codec_counts.items(), key=lambda x: -x[1]):
            print(f"  {codec}: {count}")
    
    if sample_rates:
        print("\nSample rates:")
        for sr, count in sorted(sample_rates.items(), key=lambda x: -x[1]):
            print(f"  {sr} Hz: {count}")
    
    # Content analysis summary
    if args.analyze_content and LIBROSA_AVAILABLE:
        rms_values = []
        for r in results:
            ca = r.get('content_analysis', {})
            if 'rms_db' in ca:
                rms_values.append(ca['rms_db'])
        
        if rms_values:
            print(f"\nAudio loudness (RMS dB):")
            print(f"  Min: {min(rms_values):.1f} dB")
            print(f"  Max: {max(rms_values):.1f} dB")
            print(f"  Mean: {np.mean(rms_values):.1f} dB")
            print(f"  Median: {np.median(rms_values):.1f} dB")
    
    # Save detailed report
    report = {
        'summary': {
            'total_videos': len(video_files),
            'has_audio_stream': stats['has_audio_stream'],
            'has_audio_content': stats['has_audio_content'],
            'silent_audio': stats['silent_audio'],
            'no_audio_stream': stats['no_audio_stream'],
            'errors': stats['errors'],
            'codec_counts': dict(codec_counts),
            'sample_rates': dict(sample_rates),
        },
        'videos': results,
    }
    
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: {output_path}")
    
    # List videos with audio content
    videos_with_audio = [r['filename'] for r in results if r.get('has_audio_content', False)]
    if videos_with_audio:
        print(f"\nVideos with audio content ({len(videos_with_audio)}):")
        for v in videos_with_audio[:20]:  # Show first 20
            print(f"  {v}")
        if len(videos_with_audio) > 20:
            print(f"  ... and {len(videos_with_audio) - 20} more")


if __name__ == '__main__':
    main()