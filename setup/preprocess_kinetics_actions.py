#!/usr/bin/env python3
"""Mean-pool per-person VideoMAE action embeddings into one feature per Kinetics clip."""
import os
import json
import numpy as np
from tqdm import tqdm

from core.extractors import (
    extract_tracks,
    extract_actions,
)


BASE = os.path.expanduser("~/projects/violence_anticipation/data")
KIN_VID_DIR = f"{BASE}/raw_videos/kinetics/videos"
KIN_META = f"{BASE}/metadata/kinetics.json"

OUT_DIR = f"{BASE}/processed/kinetics_actions"
OUT_FEATURES = os.path.join(OUT_DIR, "features.npy")
OUT_LABELS = os.path.join(OUT_DIR, "labels.npy")
OUT_INFO = os.path.join(OUT_DIR, "info.json")


def safe_mkdir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def preprocess_kinetics_actions():
    if not os.path.exists(KIN_META):
        raise FileNotFoundError(f"Missing metadata file: {KIN_META}")

    with open(KIN_META, "r") as f:
        meta = json.load(f)

    print(f"[Kinetics] Found {len(meta)} clips in metadata.")

    safe_mkdir(OUT_DIR)

    all_features = []
    all_labels = []
    all_clip_ids = []

    label_to_idx = {}

    for clip_id, info in tqdm(meta.items(), desc="Kinetics clips"):
        vid_path = os.path.join(KIN_VID_DIR, f"{clip_id}.mp4")
        if not os.path.exists(vid_path):
            print(f"[Skip] Missing video: {vid_path}")
            continue

        label_str = info.get("label", None)
        if label_str is None:
            print(f"[Skip] No label for clip {clip_id}")
            continue

        if label_str not in label_to_idx:
            label_to_idx[label_str] = len(label_to_idx)
        label_idx = label_to_idx[label_str]

        try:
            tracks_by_frame = extract_tracks(vid_path)
        except Exception as e:
            print(f"[Error] extract_tracks failed for {clip_id}: {e}")
            continue

        try:
            actions_by_frame = extract_actions(vid_path, tracks_by_frame)
        except Exception as e:
            print(f"[Error] extract_actions failed for {clip_id}: {e}")
            continue

        embeddings = []
        for frame_actions in actions_by_frame:
            for _, emb in frame_actions.items():
                arr = np.asarray(emb, dtype=np.float32)
                if arr.ndim == 0:
                    continue
                embeddings.append(arr)

        if len(embeddings) == 0:
            print(f"[Warn] No embeddings for clip {clip_id}, skipping.")
            continue

        clip_feat = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)

        all_features.append(clip_feat)
        all_labels.append(label_idx)
        all_clip_ids.append(clip_id)

    if len(all_features) == 0:
        raise RuntimeError("[Kinetics] No valid clips processed; all_features is empty.")

    features = np.stack(all_features, axis=0)
    labels = np.array(all_labels, dtype=np.int64)

    np.save(OUT_FEATURES, features)
    np.save(OUT_LABELS, labels)

    info = {
        "clip_ids": all_clip_ids,
        "label_to_idx": label_to_idx,
        "num_clips": int(features.shape[0]),
        "feat_dim": int(features.shape[1]),
    }
    with open(OUT_INFO, "w") as f:
        json.dump(info, f, indent=4)

    print(f"[Kinetics] Saved features → {OUT_FEATURES}")
    print(f"[Kinetics] Saved labels   → {OUT_LABELS}")
    print(f"[Kinetics] Saved info     → {OUT_INFO}")
    print(f"[Kinetics] Classes: {len(label_to_idx)}")


if __name__ == "__main__":
    preprocess_kinetics_actions()
