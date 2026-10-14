#!/usr/bin/env python3
"""One 768-d VideoMAE-base embedding per Kinetics clip from uniformly sampled frames."""
import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from transformers import VideoMAEModel, VideoMAEImageProcessor


BASE = os.path.expanduser("~/projects/violence_anticipation/data")

KIN_META = f"{BASE}/metadata/kinetics.json"
KIN_RAW  = f"{BASE}/raw_videos/kinetics/videos"
OUT_DIR  = f"{BASE}/processed/kinetics_actions"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)


print("[Kinetics] Loading VideoMAE model (base) for feature extraction...")

VIDEOMAE_NAME = "MCG-NJU/videomae-base"
videomae = VideoMAEModel.from_pretrained(VIDEOMAE_NAME).to(DEVICE)
videomae.half()
videomae.eval()

feature_extractor = VideoMAEImageProcessor.from_pretrained(VIDEOMAE_NAME)
VMAE_NUM_FRAMES = getattr(videomae.config, "num_frames", 16)


def safe_mkdir(p: str):
    if not os.path.exists(p):
        os.makedirs(p)


def _embed_clip_from_frames(frames: list[Image.Image]) -> np.ndarray:
    if len(frames) == 0:
        return np.zeros((768,), dtype=np.float32)

    if len(frames) < VMAE_NUM_FRAMES:
        frames = frames + [frames[-1]] * (VMAE_NUM_FRAMES - len(frames))
    else:
        frames = frames[:VMAE_NUM_FRAMES]

    inputs = feature_extractor(frames, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE).half()

    with torch.no_grad():
        outputs = videomae(pixel_values)

    emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
    return emb.cpu().numpy().astype(np.float32)


def preprocess_kinetics_for_videomae():
    safe_mkdir(OUT_DIR)

    if not os.path.exists(KIN_META):
        raise FileNotFoundError(f"Kinetics metadata not found at: {KIN_META}")

    with open(KIN_META, "r") as f:
        meta = json.load(f)

    print(f"[Kinetics] Loaded metadata for {len(meta)} clips.")
    keys = list(meta.keys())

    for clip_id in tqdm(keys, desc="Kinetics VideoMAE features"):
        out_path = os.path.join(OUT_DIR, f"{clip_id}.npy")
        if os.path.exists(out_path):
            continue

        vid_path = os.path.join(KIN_RAW, f"{clip_id}.mp4")
        if not os.path.exists(vid_path):
            print(f"[Skip] Missing Kinetics video: {vid_path}")
            continue

        cap = cv2.VideoCapture(vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if total_frames <= 0:
            cap.release()
            print(f"[Skip] No frames in: {vid_path}")
            continue

        num_samples = min(VMAE_NUM_FRAMES, total_frames)
        indices = np.linspace(0, total_frames - 1, num_samples).astype(int)

        frames: list[Image.Image] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            frames.append(pil_img)

        cap.release()

        if len(frames) == 0:
            print(f"[Skip] Could not read frames from: {vid_path}")
            continue

        feat = _embed_clip_from_frames(frames)
        np.save(out_path, feat)

    print(f"[Kinetics] Done. Features saved into {OUT_DIR}")


if __name__ == "__main__":
    preprocess_kinetics_for_videomae()
