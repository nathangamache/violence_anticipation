"""Shared media decoding and the per-video feature file contract (spec: docs/FEATURES.md, section 0.1)."""

import json
import os
import subprocess
from dataclasses import dataclass

import numpy as np

FEATURE_ROOT = "data/features"


def feature_path(extractor, video_id):
    return os.path.join(FEATURE_ROOT, extractor, video_id + ".npz")


def tick_times(duration_sec):
    return np.arange(1, int(np.floor(duration_sec)) + 1, dtype=np.float32)


def save_features(extractor, video_id, t_sec, feat, valid, meta):
    path = feature_path(extractor, video_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(
        tmp,
        t_sec=np.asarray(t_sec, dtype=np.float32),
        feat=np.asarray(feat, dtype=np.float16),
        valid=np.asarray(valid, dtype=bool),
        meta=json.dumps(meta),
    )
    os.replace(tmp, path)
    return path


def load_features(extractor, video_id):
    with np.load(feature_path(extractor, video_id), allow_pickle=False) as z:
        return z["t_sec"], z["feat"].astype(np.float32), z["valid"], json.loads(str(z["meta"]))


@dataclass
class VideoInfo:
    path: str
    fps: float
    num_frames: int
    duration_sec: float
    width: int
    height: int


class Video:
    """Random-access decoder that samples frames by timestamp (seconds), not by index."""

    def __init__(self, path, num_threads=4, width=-1, height=-1):
        try:
            import torch  # noqa: F401  # decord loaded before torch segfaults on a later CUDA init
        except ImportError:
            pass
        import decord

        self.vr = decord.VideoReader(path, num_threads=num_threads, width=width, height=height)
        fps = float(self.vr.get_avg_fps()) or 25.0
        n = len(self.vr)
        h, w = self.vr[0].shape[:2]
        self.info = VideoInfo(path, fps, n, n / fps, w, h)

    def frame_index(self, t_sec):
        """Index of the last frame at or before time ``t_sec`` (causal)."""
        idx = np.floor(np.asarray(t_sec, dtype=np.float64) * self.info.fps + 1e-6).astype(np.int64)
        return np.clip(idx, 0, self.info.num_frames - 1)

    def frames_at(self, t_sec):
        """Decode RGB uint8 frames [N, H, W, 3] at the given times."""
        return self.vr.get_batch(self.frame_index(t_sec).tolist()).asnumpy()

    def sample_times(self, fps, end_sec=None):
        end = self.info.duration_sec if end_sec is None else min(end_sec, self.info.duration_sec)
        return np.arange(0.0, end, 1.0 / fps)


def has_audio(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return bool(out.stdout.strip())


def load_audio(path, sr=16000):
    if not has_audio(path):
        return None
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-map", "0:a:0", "-ac", "1", "-ar", str(sr),
         "-f", "f32le", "-"],
        capture_output=True,
    )
    if out.returncode != 0 or not out.stdout:
        return None
    return np.frombuffer(out.stdout, dtype=np.float32).copy()


def is_digital_silence(wav, thresh=1e-4):
    """UCF-Crime tracks are mostly digital silence; treat those as missing audio."""
    return wav is None or wav.size == 0 or float(np.abs(wav).max()) < thresh


def check_causal(extract_fn, video_path, cut_sec, atol=1e-2):
    """Re-extract on a copy cut at ``cut_sec`` and compare earlier ticks; ``extract_fn(path) -> (t_sec, feat, valid)``."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        cut = os.path.join(d, "cut" + os.path.splitext(video_path)[1])
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", video_path, "-t", str(cut_sec), "-c", "copy", cut],
                       check=True)
        t_full, f_full, _ = extract_fn(video_path)
        t_cut, f_cut, _ = extract_fn(cut)
    k = int(np.sum(t_cut <= cut_sec - 1))  # last tick may straddle the cut point
    diff = np.abs(f_full[:k].astype(np.float32) - f_cut[:k].astype(np.float32)).max() if k else 0.0
    return {"ticks_compared": k, "max_abs_diff": float(diff), "causal": bool(diff <= atol)}
