#!/usr/bin/env python3
"""
Violence Anticipation System - YouTube Video Analyzer

This script analyzes YouTube videos for violence anticipation using:
1. Qwen2.5-VL for frame embeddings
2. Temporal LSTM model for violence prediction

Usage:
    # Analyze a YouTube video (downloads temporarily)
    python analyze_youtube.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
    
    # Analyze a local video file
    python analyze_youtube.py --video /path/to/video.mp4
    
    # Download video for later use
    python analyze_youtube.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --download-only --output video.mp4
    
    # Adjust threshold and other parameters
    python analyze_youtube.py --url "URL" --threshold 0.3 --fps 1.0

Requirements:
    pip install yt-dlp opencv-python torch transformers qwen-vl-utils pillow matplotlib
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple, List
import subprocess

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Optional imports with graceful fallback
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ==============================================================================
# Model Definition (must match training)
# ==============================================================================

class TemporalViolenceModel(nn.Module):
    """Temporal LSTM model matching train_temporal_model.py architecture."""
    
    def __init__(
        self,
        input_dim: int = 3584,
        hidden_dim: int = 256,
        lstm_hidden: int = 128,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.frame_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True,
        )
        
        lstm_output_dim = lstm_hidden * 2
        
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        
        x_flat = x.view(B * T, D)
        encoded = self.frame_encoder(x_flat)
        encoded = encoded.view(B, T, -1)
        
        lstm_out, _ = self.lstm(encoded)
        
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)
        
        logits = self.classifier(context).squeeze(-1)
        return logits


def load_model(model_path: str, device: str) -> Tuple[nn.Module, dict]:
    """Load model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    # Detect dimensions from state dict
    input_dim = state_dict['frame_encoder.0.weight'].shape[1]
    hidden_dim = state_dict['frame_encoder.0.weight'].shape[0]
    lstm_hidden = state_dict['lstm.weight_hh_l0'].shape[1]
    num_layers = sum(1 for k in state_dict.keys() if 'lstm.weight_ih_l' in k and 'reverse' not in k)
    
    model = TemporalViolenceModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        lstm_hidden=lstm_hidden,
        num_lstm_layers=num_layers,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    config = {
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'lstm_hidden': lstm_hidden,
        'num_layers': num_layers,
        'sequence_length': checkpoint.get('args', {}).get('sequence_length', 30),
    }
    
    return model, config


# ==============================================================================
# Video Processing
# ==============================================================================

def download_youtube_video(url: str, output_path: Optional[str] = None, max_duration: Optional[int] = None) -> str:
    """Download YouTube video using yt-dlp.
    
    Args:
        url: YouTube video URL
        output_path: Output file path (optional)
        max_duration: Maximum duration in seconds to download (for long videos/streams)
    """
    import yt_dlp
    
    if output_path is None:
        output_path = tempfile.mktemp(suffix='.mp4')
    
    # First, get video info without downloading
    print(f"Fetching video info from: {url}")
    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info.get('title', 'Unknown')
        duration = info.get('duration', 0)
        is_live = info.get('is_live', False)
        was_live = info.get('was_live', False)
        
        print(f"  Title: {title}")
        print(f"  Duration: {duration}s ({duration/60:.1f} min)" if duration else "  Duration: Unknown (possibly live)")
        if is_live:
            print("  Status: Currently LIVE")
        elif was_live:
            print("  Status: Was a livestream")
    
    # Configure download options
    ydl_opts = {
        'outtmpl': output_path,
        'quiet': False,
        'no_warnings': False,
        # Prefer direct MP4 downloads over HLS/DASH to avoid fragment issues
        'format': 'best[ext=mp4][height<=720]/bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[height<=720]/best',
        # Force IPv4 (sometimes helps with fragment issues)
        'source_address': '0.0.0.0',
        # Extractor args to avoid JS requirement
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
    }
    
    # For livestreams or very long videos, limit duration
    if max_duration and (is_live or was_live or (duration and duration > max_duration)):
        print(f"  Limiting download to first {max_duration} seconds...")
        ydl_opts['download_ranges'] = lambda info, ydl: [{'start_time': 0, 'end_time': max_duration}]
        # Use ffmpeg for partial downloads
        ydl_opts['external_downloader'] = 'ffmpeg'
        ydl_opts['external_downloader_args'] = {'ffmpeg': ['-t', str(max_duration)]}
    
    # If it's a past livestream with HLS issues, try alternative approach
    if was_live:
        print("  Note: This was a livestream. Using alternative download method...")
        ydl_opts['format'] = 'best[height<=720]'
        # Skip unavailable fragments instead of failing
        ydl_opts['fragment_retries'] = 3
        ydl_opts['skip_unavailable_fragments'] = True
    
    print(f"Downloading video...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    # yt-dlp might add extension
    if not os.path.exists(output_path):
        for ext in ['.mp4', '.webm', '.mkv', '.mp4.part']:
            test_path = output_path.rsplit('.', 1)[0] + ext
            if os.path.exists(test_path):
                output_path = test_path
                break
    
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Downloaded: {output_path} ({file_size:.1f} MB)")
    else:
        raise RuntimeError(f"Download failed - file not found at {output_path}")
    
    return output_path


def extract_frames(video_path: str, fps: float = 1.0) -> List[np.ndarray]:
    """Extract frames from video at specified FPS."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    
    print(f"Video: {video_fps:.1f} FPS, {total_frames} frames, {duration:.1f}s duration")
    print(f"Extracting at {fps} FPS...")
    
    frame_interval = int(video_fps / fps) if video_fps > 0 else 1
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        frame_idx += 1
    
    cap.release()
    print(f"Extracted {len(frames)} frames")
    
    return frames


# ==============================================================================
# Embedding Generation
# ==============================================================================

def load_vlm_model(device: str = "cuda", vlm_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
    """Load Qwen2.5-VL model for embedding generation.
    
    Tries to use the project's QwenTeacher first for compatibility.
    Falls back to loading the model directly if QwenTeacher is not available.
    
    Args:
        device: Device to use (e.g., "cuda:0", "cuda:1")
        vlm_model_id: Model ID to load
    """
    # Try to use the project's QwenTeacher for perfect compatibility
    try:
        import sys
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from vlm.qwen_teacher import QwenTeacher, QwenTeacherConfig
        
        print(f"Loading VLM via QwenTeacher: {vlm_model_id}")
        print(f"  Device: {device}")
        
        # Configure for embedding extraction
        cfg = QwenTeacherConfig(
            model_id=vlm_model_id,
            device=device,
            dtype="bfloat16",
            prompt_mode="violence_embedding",  # Use violence-focused prompts
            context_mode="none",  # No temporal context needed for single frames
            attn_implementation="sdpa",
        )
        
        teacher = QwenTeacher(cfg)
        print("  Model loaded successfully (using QwenTeacher)")
        
        # Return teacher as both model and processor (we'll handle it differently)
        return teacher, None
        
    except ImportError as e:
        print(f"Warning: Could not import QwenTeacher: {e}")
        print("Falling back to direct model loading...")
    except Exception as e:
        print(f"Warning: Error initializing QwenTeacher: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to direct model loading...")
    
    # Fallback: Load model directly (uses smaller 3B model)
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    
    print("Loading Qwen2.5-VL model directly...")
    fallback_model = "Qwen/Qwen2.5-VL-3B-Instruct"  # Use smaller model for fallback
    print(f"  Using fallback model: {fallback_model}")
    print(f"  WARNING: This may not match your training embeddings!")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        fallback_model,
        torch_dtype=torch.float16,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(fallback_model)
    
    print("  Model loaded successfully (direct loading)")
    return model, processor


def generate_embeddings(
    frames: List[np.ndarray],
    vlm_model,
    vlm_processor,
    device: str = "cuda",
    batch_size: int = 1,
    target_dim: int = 3584,
    use_project_extractor: bool = True,
) -> np.ndarray:
    """Generate embeddings for frames using Qwen2.5-VL.
    
    This function uses the project's QwenTeacher for embedding extraction
    to ensure compatibility with trained models.
    
    Args:
        frames: List of RGB frames as numpy arrays
        vlm_model: Loaded QwenTeacher or Qwen2.5-VL model
        vlm_processor: Model processor (None if using QwenTeacher)
        device: Device to run inference on
        batch_size: Not used currently (frame-by-frame processing)
        target_dim: Target embedding dimension (3584 for training compatibility)
        use_project_extractor: Try to use project's QwenTeacher class
    
    Returns:
        embeddings: [num_frames, target_dim] array of embeddings
    """
    embeddings = []
    
    print(f"Generating embeddings for {len(frames)} frames...")
    
    # Check if we're using QwenTeacher
    try:
        from vlm.qwen_teacher import QwenTeacher
        is_qwen_teacher = isinstance(vlm_model, QwenTeacher)
    except ImportError:
        is_qwen_teacher = False
    
    if is_qwen_teacher:
        print("  Using QwenTeacher.infer_embedding_only() for extraction")
        teacher = vlm_model
        
        for i, frame in enumerate(frames):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Processing frame {i + 1}/{len(frames)}")
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame)
            
            # Use the teacher's embedding extraction method
            # This uses violence_embedding prompt mode by default
            embedding = teacher.infer_embedding_only(pil_image)
            
            if embedding is None:
                print(f"  Warning: Failed to extract embedding for frame {i+1}, using zeros")
                embedding = np.zeros(target_dim, dtype=np.float32)
            
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        print(f"  Generated embeddings: shape {embeddings.shape}")
        
        # Check dimension
        if embeddings.shape[1] != target_dim:
            print(f"  Note: Embedding dimension {embeddings.shape[1]} differs from target {target_dim}")
        
        return embeddings
    
    # Fallback: generic embedding extraction (may not match training)
    from qwen_vl_utils import process_vision_info
    
    print(f"  Using generic embedding extraction")
    print(f"  Target dimension: {target_dim}")
    print(f"  WARNING: Embeddings may not match training distribution!")
    
    for i, frame in enumerate(frames):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing frame {i + 1}/{len(frames)}")
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame)
        
        # Prepare message format with violence-focused prompt
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": "Describe this surveillance camera image, focusing on any people present, their body language, interactions, and the overall scene."}
            ]
        }]
        
        # Process inputs
        text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = vlm_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        # Get embeddings from hidden states
        with torch.no_grad():
            outputs = vlm_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            last_hidden = hidden_states[-1]
            hidden_dim = last_hidden.shape[-1]
            
            # Combine representations to reach target_dim
            embedding_parts = []
            embedding_parts.append(last_hidden.mean(dim=1))
            embedding_parts.append(last_hidden.max(dim=1).values)
            
            current_dim = hidden_dim * 2
            layer_idx = -2
            while current_dim < target_dim and abs(layer_idx) <= len(hidden_states):
                layer_hidden = hidden_states[layer_idx]
                layer_mean = layer_hidden.mean(dim=1)
                embedding_parts.append(layer_mean)
                current_dim += hidden_dim
                layer_idx -= 1
            
            full_embedding = torch.cat(embedding_parts, dim=-1)
            
            if full_embedding.shape[-1] > target_dim:
                full_embedding = full_embedding[:, :target_dim]
            elif full_embedding.shape[-1] < target_dim:
                padding = torch.zeros(
                    full_embedding.shape[0], 
                    target_dim - full_embedding.shape[-1],
                    device=full_embedding.device,
                    dtype=full_embedding.dtype
                )
                full_embedding = torch.cat([full_embedding, padding], dim=-1)
            
            embedding = full_embedding.cpu().numpy().astype(np.float32)
            embeddings.append(embedding[0])
    
    embeddings = np.array(embeddings)
    print(f"  Generated embeddings: shape {embeddings.shape}")
    
    return embeddings


# ==============================================================================
# Inference
# ==============================================================================

def run_inference(
    model: nn.Module,
    embeddings: np.ndarray,
    sequence_length: int = 30,
    device: str = "cuda",
) -> np.ndarray:
    """Run inference and return scores for each frame."""
    model.eval()
    n_frames = len(embeddings)
    scores = np.zeros(n_frames, dtype=np.float32)
    
    # Pad beginning
    pad_size = sequence_length - 1
    padded = np.vstack([
        np.zeros((pad_size, embeddings.shape[1]), dtype=np.float32),
        embeddings
    ])
    
    print(f"Running inference on {n_frames} frames...")
    
    with torch.no_grad():
        for i in range(n_frames):
            seq = padded[i:i + sequence_length]
            batch = torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
            logit = model(batch)
            scores[i] = torch.sigmoid(logit).cpu().numpy()
    
    return scores


# ==============================================================================
# Visualization
# ==============================================================================

def plot_results(
    scores: np.ndarray,
    threshold: float,
    fps: float,
    output_path: str,
    title: str = "Violence Anticipation Analysis",
):
    """Plot violence scores over time."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plot")
        return
    
    times = np.arange(len(scores)) / fps
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot scores
    ax.plot(times, scores, 'b-', linewidth=1.5, label='Violence Score')
    ax.axhline(y=threshold, color='r', linestyle='--', linewidth=1, label=f'Threshold ({threshold})')
    
    # Highlight regions above threshold
    above_threshold = scores >= threshold
    ax.fill_between(times, 0, 1, where=above_threshold, alpha=0.3, color='red', label='Alert Region')
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Violence Probability', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, times[-1] if len(times) > 0 else 1)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Plot saved to: {output_path}")


def print_alerts(scores: np.ndarray, threshold: float, fps: float):
    """Print detected alert periods."""
    above_threshold = scores >= threshold
    
    # Find contiguous alert regions
    alerts = []
    in_alert = False
    alert_start = 0
    
    for i, is_alert in enumerate(above_threshold):
        if is_alert and not in_alert:
            in_alert = True
            alert_start = i
        elif not is_alert and in_alert:
            in_alert = False
            alerts.append((alert_start, i - 1))
    
    if in_alert:
        alerts.append((alert_start, len(scores) - 1))
    
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Total duration: {len(scores) / fps:.1f} seconds")
    print(f"Threshold: {threshold}")
    print(f"Frames above threshold: {above_threshold.sum()} / {len(scores)} ({100 * above_threshold.mean():.1f}%)")
    print(f"Max score: {scores.max():.3f}")
    print(f"Mean score: {scores.mean():.3f}")
    
    if alerts:
        print(f"\nDetected {len(alerts)} alert period(s):")
        for i, (start, end) in enumerate(alerts):
            start_time = start / fps
            end_time = end / fps
            duration = end_time - start_time
            max_score = scores[start:end+1].max()
            print(f"  {i+1}. {start_time:.1f}s - {end_time:.1f}s (duration: {duration:.1f}s, max: {max_score:.3f})")
    else:
        print("\nNo violence anticipated in this video.")
    
    print("=" * 60)


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze YouTube videos for violence anticipation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze a YouTube video
    python analyze_youtube.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
    
    # Analyze a local video file
    python analyze_youtube.py --video /path/to/video.mp4
    
    # Download video only (for later analysis)
    python analyze_youtube.py --url "URL" --download-only --output video.mp4
    
    # Custom threshold and model
    python analyze_youtube.py --url "URL" --threshold 0.3 --model path/to/model.pt
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--url', type=str, help='YouTube video URL')
    input_group.add_argument('--video', type=str, help='Path to local video file')
    
    # Model options
    parser.add_argument('--model', type=str, default='outputs/outputs_temporal/best_temporal_model.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Detection threshold (default: 0.3)')
    
    # Processing options
    parser.add_argument('--fps', type=float, default=1.0,
                        help='Frames per second for analysis (default: 1.0)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (default: cuda:0)')
    parser.add_argument('--max-duration', type=int, default=300,
                        help='Max video duration in seconds to download (default: 300 = 5 min). Use 0 for no limit.')
    
    # Output options
    parser.add_argument('--output', type=str, help='Output path for downloaded video')
    parser.add_argument('--plot', type=str, default='analysis_plot.png',
                        help='Output path for plot (default: analysis_plot.png)')
    parser.add_argument('--save-scores', type=str, help='Save scores to numpy file')
    parser.add_argument('--save-embeddings', type=str, help='Save embeddings to numpy file')
    
    # Mode options
    parser.add_argument('--download-only', action='store_true',
                        help='Only download video, do not analyze')
    parser.add_argument('--skip-cleanup', action='store_true',
                        help='Do not delete temporary video file')
    
    # Embedding options
    parser.add_argument('--embeddings', type=str, 
                        help='Load pre-computed embeddings instead of generating')
    parser.add_argument('--embedding-dim', type=int, default=3584,
                        help='Target embedding dimension (default: 3584 to match training)')
    parser.add_argument('--vlm-model', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct',
                        help='VLM model to use (default: Qwen/Qwen2.5-VL-7B-Instruct)')
    
    args = parser.parse_args()
    
    # Check device
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print(f"Warning: {args.device} not available, using CPU")
        args.device = 'cpu'
    
    # Get video path
    video_path = None
    temp_video = False
    
    if args.url:
        output_path = args.output if args.output else tempfile.mktemp(suffix='.mp4')
        max_dur = args.max_duration if args.max_duration > 0 else None
        video_path = download_youtube_video(args.url, output_path, max_duration=max_dur)
        temp_video = not args.output and not args.skip_cleanup
        
        if args.download_only:
            print(f"\nVideo downloaded to: {video_path}")
            return
    else:
        video_path = args.video
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            sys.exit(1)
    
    try:
        # Load embeddings or generate them
        if args.embeddings:
            print(f"Loading embeddings from: {args.embeddings}")
            embeddings = np.load(args.embeddings)
            print(f"  Loaded embeddings: shape {embeddings.shape}")
        else:
            # Extract frames
            frames = extract_frames(video_path, args.fps)
            
            # Load VLM and generate embeddings
            vlm_model, vlm_processor = load_vlm_model(args.device, args.vlm_model)
            embeddings = generate_embeddings(
                frames, vlm_model, vlm_processor, args.device,
                target_dim=args.embedding_dim
            )
            
            # Clean up VLM model to free memory
            del vlm_model, vlm_processor
            torch.cuda.empty_cache()
            
            # Save embeddings if requested
            if args.save_embeddings:
                np.save(args.save_embeddings, embeddings)
                print(f"Embeddings saved to: {args.save_embeddings}")
        
        # Load temporal model
        print(f"\nLoading model from: {args.model}")
        model, config = load_model(args.model, args.device)
        print(f"  Input dim: {config['input_dim']}, Sequence length: {config['sequence_length']}")
        
        # Run inference
        scores = run_inference(model, embeddings, config['sequence_length'], args.device)
        
        # Save scores if requested
        if args.save_scores:
            np.save(args.save_scores, scores)
            print(f"Scores saved to: {args.save_scores}")
        
        # Print results
        print_alerts(scores, args.threshold, args.fps)
        
        # Plot results
        if args.plot:
            plot_results(scores, args.threshold, args.fps, args.plot)
        
    finally:
        # Cleanup temporary video
        if temp_video and video_path and os.path.exists(video_path):
            os.remove(video_path)
            print(f"Cleaned up temporary video file")


if __name__ == '__main__':
    main()