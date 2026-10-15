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

_DEVICE = get_device()
torch.set_grad_enabled(False)

FRAME_STRIDE = 4      # tracks and VideoMAE actions
SCENE_STRIDE = 16     # CLIP scene

try:
    import sounddevice as sd
    _AUDIO_AVAILABLE = True
except Exception:
    sd = None
    _AUDIO_AVAILABLE = False


print("[Load] YOLO + VideoMAE + CLIP (no DeepSORT)")

_yolo_model = YOLO("yolov8n.pt")

_VIDEOMAE_NAME = "MCG-NJU/videomae-base"
_video_mae = VideoMAEModel.from_pretrained(_VIDEOMAE_NAME).to(_DEVICE)
_video_mae.half()
_video_mae.eval()

_video_mae_extractor = VideoMAEImageProcessor.from_pretrained(_VIDEOMAE_NAME)
_VMAE_NUM_FRAMES = getattr(_video_mae.config, "num_frames", 16)

_clip_model, _clip_preprocess = clip.load("ViT-B/16", device=_DEVICE)
_clip_model.eval()


def crop_bbox(frame: np.ndarray, bbox: Any) -> np.ndarray | None:
    """Crop [x1, y1, x2, y2] from a frame, clipped to its bounds; None if empty."""
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
    """Mean-pooled 768-d VideoMAE embedding of a clip padded/truncated to _VMAE_NUM_FRAMES frames."""
    if len(frames) == 0:
        return np.zeros((768,), dtype=np.float32)

    if len(frames) < _VMAE_NUM_FRAMES:
        frames = frames + [frames[-1]] * (_VMAE_NUM_FRAMES - len(frames))
    else:
        frames = frames[:_VMAE_NUM_FRAMES]

    inputs = _video_mae_extractor(frames, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(_DEVICE).half()  # (1, T, 3, H, W)

    with torch.no_grad():
        outputs = _video_mae(pixel_values)

    # outputs.last_hidden_state: (1, num_tokens, hidden_size)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
    return emb.cpu().numpy().astype(np.float32)


def extract_tracks(video_path: str) -> List[List[Dict[str, Any]]]:
    """YOLO person boxes every FRAME_STRIDE frames; track_id is the per-frame detection index, not a real track."""
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


def extract_actions(video_path: str, tracks_by_frame: List[List[Dict[str, Any]]]) -> List[Dict[int, np.ndarray]]:
    """768-d VideoMAE embedding per person crop, for each frame returned by extract_tracks."""
    cap = cv2.VideoCapture(video_path)
    actions_by_frame: List[Dict[int, np.ndarray]] = []

    for t, persons in enumerate(tracks_by_frame):
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

            # A single crop replicated into a static clip: VideoMAE gets no temporal context.
            frames_list = [pil_img] * _VMAE_NUM_FRAMES
            feat = _video_mae_embed_from_pil_frames(frames_list)
            frame_actions[person["track_id"]] = feat

        actions_by_frame.append(frame_actions)

    cap.release()
    return actions_by_frame


def extract_scene(video_path: str) -> np.ndarray:
    """Mean 512-d CLIP embedding over every SCENE_STRIDE-th frame of the whole video."""
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


def extract_audio_placeholder(dim: int = 64) -> np.ndarray:
    return np.zeros((dim,), dtype=np.float32)


def extract_audio_from_file(video_path: str, sr: int = 16000) -> np.ndarray:
    """Time-mean mel spectrogram of the video's audio track, decoded with ffmpeg."""
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


def live_track(frame: np.ndarray) -> List[Dict[str, Any]]:
    """YOLO person boxes for one frame; track_id is not stable across frames."""
    results = _yolo_model(frame)[0]
    persons = []

    for i, box in enumerate(results.boxes):
        if int(box.cls) == 0:
            bbox = box.xyxy[0].cpu().numpy().tolist()
            persons.append({"track_id": i, "bbox": bbox})

    return persons


def live_actions(frame: np.ndarray, tracks: List[Dict[str, Any]]) -> Dict[int, np.ndarray]:
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
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    img = _clip_preprocess(pil_img).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        emb = _clip_model.encode_image(img).cpu().numpy()[0]

    return emb.astype(np.float32)


def live_audio(seconds: float = 0.3, sr: int = 16000, dim: int = 64) -> np.ndarray:
    """Mel-mean microphone embedding, or zeros when sounddevice/PortAudio is unavailable."""
    if not _AUDIO_AVAILABLE:
        return np.zeros((dim,), dtype=np.float32)

    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1)
    sd.wait()
    mel = librosa.feature.melspectrogram(y=audio.flatten(), sr=sr)
    return mel.mean(axis=1).astype(np.float32)
