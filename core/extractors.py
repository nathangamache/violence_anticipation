import os
import cv2
import numpy as np
import torch
import librosa
import subprocess

from typing import List, Dict, Any
from PIL import Image

from transformers import VideoMAEModel, VideoMAEImageProcessor
import clip
from ultralytics import YOLO

from core.utils import get_device

# ==============================================================
# DEVICE / GLOBAL CONFIG
# ==============================================================

_DEVICE = get_device()
torch.set_grad_enabled(False)

# How densely to process frames
FRAME_STRIDE = 4      # process 1 in every 4 frames for tracks + actions
SCENE_STRIDE = 16     # process 1 in every 16 frames for scene CLIP

# Audio availability (for live mode)
try:
    import sounddevice as sd
    _AUDIO_AVAILABLE = True
except Exception:
    sd = None
    _AUDIO_AVAILABLE = False


# ==============================================================
# MODEL LOADING (GLOBAL)
# ==============================================================

print("[Load] YOLO + VideoMAE + CLIP (no DeepSORT)")

# --- YOLO person detector ---
_yolo_model = YOLO("yolov8n.pt")

# --- VideoMAE action encoder ---
_VIDEOMAE_NAME = "MCG-NJU/videomae-base"
_video_mae = VideoMAEModel.from_pretrained(_VIDEOMAE_NAME).to(_DEVICE)
_video_mae.half()
_video_mae.eval()

_video_mae_extractor = VideoMAEImageProcessor.from_pretrained(_VIDEOMAE_NAME)
_VMAE_NUM_FRAMES = getattr(_video_mae.config, "num_frames", 16)

# --- CLIP scene encoder ---
_clip_model, _clip_preprocess = clip.load("ViT-B/16", device=_DEVICE)
_clip_model.eval()


# ==============================================================
# HELPER FUNCTIONS
# ==============================================================

def crop_bbox(frame: np.ndarray, bbox: Any) -> np.ndarray | None:
    """
    Crop a bounding box [x1, y1, x2, y2] from a BGR frame.
    Returns None if the crop is invalid/empty.
    """
    if bbox is None:
        return None
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def _video_mae_embed_from_pil_frames(frames: List[Image.Image]) -> np.ndarray:
    """
    Given a list of PIL frames (length >= 1), run VideoMAE and return
    a single 768-D embedding (mean pooled over tokens).
    """
    if len(frames) == 0:
        return np.zeros((768,), dtype=np.float32)

    # Ensure we have exactly _VMAE_NUM_FRAMES frames by repeating last frame
    if len(frames) < _VMAE_NUM_FRAMES:
        frames = frames + [frames[-1]] * (_VMAE_NUM_FRAMES - len(frames))
    else:
        frames = frames[:_VMAE_NUM_FRAMES]

    inputs = _video_mae_extractor(frames, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(_DEVICE).half()  # (1, 3, T, H, W)

    with torch.no_grad():
        outputs = _video_mae(pixel_values)

    # outputs.last_hidden_state: (1, num_tokens, hidden_size)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
    return emb.cpu().numpy().astype(np.float32)


# ==============================================================
# OFFLINE: TRACK EXTRACTION (YOLO ONLY)
# ==============================================================

def extract_tracks(video_path: str) -> List[List[Dict[str, Any]]]:
    """
    Run YOLO person detection on the video with FRAME_STRIDE.
    Returns:
        tracks_by_frame: list over processed frames.
        Each element is a list of:
            { "track_id": int, "bbox": [x1, y1, x2, y2] }

    NOTE:
    We do NOT maintain track IDs across frames here. Each frame's
    people are enumerated 0..N-1. This is enough for building per-frame
    graphs in the current pipeline.
    """
    cap = cv2.VideoCapture(video_path)
    tracks_by_frame: List[List[Dict[str, Any]]] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_STRIDE != 0:
            frame_idx += 1
            continue

        results = _yolo_model(frame)[0]
        persons = []

        for i, box in enumerate(results.boxes):
            if int(box.cls) == 0:  # person class
                bbox = box.xyxy[0].cpu().numpy().tolist()
                persons.append({"track_id": i, "bbox": bbox})

        tracks_by_frame.append(persons)
        frame_idx += 1

    cap.release()
    return tracks_by_frame


# ==============================================================
# OFFLINE: ACTION FEATURES (VideoMAE)
# ==============================================================

def extract_actions(video_path: str, tracks_by_frame: List[List[Dict[str, Any]]]) -> List[Dict[int, np.ndarray]]:
    """
    For each processed frame (matching extract_tracks output), encode
    each person's crop with VideoMAE.

    Args:
        video_path: path to .mp4 file.
        tracks_by_frame: list over processed frames, each a list of
                         {track_id, bbox}.

    Returns:
        actions_by_frame: list over frames, each a dict:
            { track_id -> 768-D np.ndarray }
    """
    cap = cv2.VideoCapture(video_path)
    actions_by_frame: List[Dict[int, np.ndarray]] = []

    for t, persons in enumerate(tracks_by_frame):
        # Map back to original frame index: 0, FRAME_STRIDE, 2*FRAME_STRIDE, ...
        raw_frame_idx = t * FRAME_STRIDE
        cap.set(cv2.CAP_PROP_POS_FRAMES, raw_frame_idx)
        ret, frame = cap.read()
        if not ret:
            actions_by_frame.append({})
            continue

        frame_actions: Dict[int, np.ndarray] = {}

        if not persons:
            actions_by_frame.append(frame_actions)
            continue

        for person in persons:
            bbox = person.get("bbox", None)
            if bbox is None:
                continue

            crop = crop_bbox(frame, bbox)
            if crop is None:
                continue

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # For now we don't build real temporal windows;
            # we feed replicated single-frame clips to VideoMAE.
            frames_list = [pil_img] * _VMAE_NUM_FRAMES
            feat = _video_mae_embed_from_pil_frames(frames_list)
            frame_actions[person["track_id"]] = feat

        actions_by_frame.append(frame_actions)

    cap.release()
    return actions_by_frame


# ==============================================================
# OFFLINE: SCENE FEATURES (CLIP)
# ==============================================================

def extract_scene(video_path: str) -> np.ndarray:
    """
    Extract a single global CLIP scene embedding for the whole video
    by sampling frames with SCENE_STRIDE and averaging the features.

    Returns:
        scene_vec: (512,) np.ndarray
    """
    cap = cv2.VideoCapture(video_path)
    feats: List[np.ndarray] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % SCENE_STRIDE != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        img = _clip_preprocess(pil_img).unsqueeze(0).to(_DEVICE)

        with torch.no_grad():
            emb = _clip_model.encode_image(img).cpu().numpy()[0]

        feats.append(emb)
        frame_idx += 1

    cap.release()

    if len(feats) == 0:
        return np.zeros((512,), dtype=np.float32)

    feats_arr = np.stack(feats, axis=0)
    return feats_arr.mean(axis=0).astype(np.float32)


# ==============================================================
# OFFLINE: AUDIO PLACEHOLDER
# ==============================================================

def extract_audio_placeholder(dim: int = 64) -> np.ndarray:
    """
    For now, we are not doing heavy audio processing offline.
    Return a fixed-size zero vector as a placeholder.
    """
    return np.zeros((dim,), dtype=np.float32)


# (Optional) Real audio extractor using ffmpeg + librosa, if needed later.
def extract_audio_from_file(video_path: str, sr: int = 16000) -> np.ndarray:
    """
    Extract audio from an MP4 using ffmpeg (no sounddevice required).
    Returns a mean mel-spectrogram vector.
    """
    tmp_wav = "/tmp/va_temp_audio.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-loglevel", "quiet",
        tmp_wav,
    ]
    subprocess.run(cmd, check=True)

    audio, _ = librosa.load(tmp_wav, sr=sr)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    return mel.mean(axis=1).astype(np.float32)


# ==============================================================
# REALTIME: LIVE VERSIONS FOR WEBCAM / RTSP
# ==============================================================

def live_track(frame: np.ndarray) -> List[Dict[str, Any]]:
    """
    Live person detection on a single frame.
    Returns a list of {track_id, bbox} for that frame.
    (track_id is per-frame only, not stable across time).
    """
    results = _yolo_model(frame)[0]
    persons = []

    for i, box in enumerate(results.boxes):
        if int(box.cls) == 0:
            bbox = box.xyxy[0].cpu().numpy().tolist()
            persons.append({"track_id": i, "bbox": bbox})

    return persons


def live_actions(frame: np.ndarray, tracks: List[Dict[str, Any]]) -> Dict[int, np.ndarray]:
    """
    Live action features for one frame.
    Uses replicated single-frame clips as input to VideoMAE.
    """
    frame_actions: Dict[int, np.ndarray] = {}

    if not tracks:
        return frame_actions

    for person in tracks:
        bbox = person.get("bbox", None)
        if bbox is None:
            continue

        crop = crop_bbox(frame, bbox)
        if crop is None:
            continue

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        frames_list = [pil_img] * _VMAE_NUM_FRAMES
        feat = _video_mae_embed_from_pil_frames(frames_list)
        frame_actions[person["track_id"]] = feat

    return frame_actions


def live_scene(frame: np.ndarray) -> np.ndarray:
    """
    Live CLIP scene embedding for a single frame.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    img = _clip_preprocess(pil_img).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        emb = _clip_model.encode_image(img).cpu().numpy()[0]

    return emb.astype(np.float32)


def live_audio(seconds: float = 0.3, sr: int = 16000, dim: int = 64) -> np.ndarray:
    """
    Live audio embedding.
    If no audio device / PortAudio is available, returns zeros.
    """
    if not _AUDIO_AVAILABLE:
        return np.zeros((dim,), dtype=np.float32)

    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1)
    sd.wait()
    mel = librosa.feature.melspectrogram(y=audio.flatten(), sr=sr)
    return mel.mean(axis=1).astype(np.float32)
