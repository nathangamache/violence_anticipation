#!/usr/bin/env python3
import os
import json
import subprocess
import numpy as np
from tqdm import tqdm

from core.extractors import (
    extract_tracks,
    extract_actions,
    extract_scene,
    extract_audio_placeholder,
)
from core.graph import build_graph_sequence
from scripts.generate_labels import generate_risk_curve


# ---------------------------------------------------------------------
# PATHS / CONFIG
# ---------------------------------------------------------------------

BASE = os.path.expanduser("~/projects/violence_anticipation/data")
RAW_VID = f"{BASE}/raw_videos"
META_DIR = f"{BASE}/metadata"
OUT = f"{BASE}/processed"

# Output folders
DIR_TRACKS = f"{OUT}/tracks"
DIR_ACTIONS = f"{OUT}/actions"
DIR_SCENES = f"{OUT}/scenes"
DIR_AUDIO = f"{OUT}/audio"
DIR_GRAPHS = f"{OUT}/graphs"
DIR_LABELS = f"{OUT}/labels"

# Where we store windowed (trimmed) clips
WINDOW_ROOT = f"{BASE}/windowed"

# How much of each video to actually process
POS_PRE_SEC = 30.0    # seconds *before* event for positives
POS_POST_SEC = 10.0   # seconds *after* event for positives
NEG_WINDOW_SEC = 40.0  # window length for negatives (no event)


# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------

def safe_mkdir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def atomic_save_json(path: str, obj):
    safe_mkdir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def atomic_save_npy(path: str, arr: np.ndarray | object):
    """
    Atomic npy save that avoids the `.npy`-on-`.tmp` suffix bug.

    np.save("foo.npy.tmp", ...) will actually write "foo.npy.tmp.npy".
    To prevent that, we open the file handle ourselves.
    """
    safe_mkdir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def load_actions(path: str):
    """
    Robust loader for actions saved via np.save (list[dict] or dict-of-dicts).
    """
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.dtype == object:
        # Scalar object (0-d)
        if loaded.shape == ():
            return loaded.item()
        # 1-D object array → list of whatever (dicts)
        return list(loaded)
    return loaded


def compute_window_frames(meta: dict) -> tuple[int, int]:
    """
    Given metadata with total_frames, fps, label, event_frame,
    compute [start_frame, end_frame] for the window we actually
    want to process.
    """
    fps = meta.get("fps", 25.0) or 25.0
    total = int(meta.get("total_frames", 0) or 0)
    label = int(meta.get("label", 0))
    event_frame = meta.get("event_frame", None)

    if total <= 0:
        # No idea, just say "whole video"
        return 0, 0

    if label == 1 and event_frame is not None and event_frame >= 0:
        event_frame = int(event_frame)
        pre_frames = int(POS_PRE_SEC * fps)
        post_frames = int(POS_POST_SEC * fps)

        start = max(0, event_frame - pre_frames)
        end = min(total - 1, event_frame + post_frames)
    else:
        # Negative or no event → fixed window from the middle
        window_frames = int(NEG_WINDOW_SEC * fps)
        if window_frames >= total:
            start = 0
            end = total - 1
        else:
            start = (total - window_frames) // 2
            end = start + window_frames - 1

    return int(start), int(end)


def get_windowed_video_path(vid_id: str, meta: dict) -> tuple[str, dict]:
    """
    Decide which temporal window to process, and ensure we have a
    trimmed clip for that window.

    Returns:
        window_path: path to actual .mp4 that extractors should use
        local_meta:  metadata adjusted to the window
                     (updated total_frames, event_frame, start_frame, end_frame)
    """
    src = os.path.join(RAW_VID, meta["source"], "videos", f"{vid_id}.mp4")
    if not os.path.exists(src):
        raise FileNotFoundError(f"Missing source video: {src}")

    start_frame, end_frame = compute_window_frames(meta)
    fps = meta.get("fps", 25.0) or 25.0
    total = int(meta.get("total_frames", 0) or 0)

    # If the window basically covers the whole video, skip trimming
    if start_frame <= 0 and (end_frame >= total - 1):
        local_meta = meta.copy()
        local_meta["start_frame"] = 0
        local_meta["end_frame"] = total - 1
        # event_frame already in clip coordinates
        return src, local_meta

    # Otherwise, trim a new clip or reuse existing one
    start_sec = start_frame / fps
    duration_sec = (end_frame - start_frame + 1) / fps

    dst_dir = os.path.join(WINDOW_ROOT, meta["source"])
    safe_mkdir(dst_dir)
    dst = os.path.join(dst_dir, f"{vid_id}.mp4")

    if not os.path.exists(dst):
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_sec),
            "-i", src,
            "-t", str(duration_sec),
            "-c", "copy",
            "-loglevel", "error",
            dst,
        ]
        subprocess.run(cmd, check=True)

    # Adjust metadata to window-coordinates
    local_meta = meta.copy()
    local_meta["start_frame"] = int(start_frame)
    local_meta["end_frame"] = int(end_frame)
    local_meta["total_frames"] = int(end_frame - start_frame + 1)

    if local_meta.get("label", 0) == 1 and local_meta.get("event_frame", None) is not None:
        # Shift event_frame so it's relative to window start
        local_meta["event_frame"] = max(0, int(local_meta["event_frame"]) - start_frame)

    return dst, local_meta


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------

def preprocess_anticipation():
    # ----------------------------------------------------
    # Load metadata (XD-Violence + UCF-Crime ONLY)
    # ----------------------------------------------------
    all_meta: dict[str, dict] = {}
    meta_files = ["xd_violence.json", "ucf_crime.json"]

    for fname in meta_files:
        path = os.path.join(META_DIR, fname)
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            part = json.load(f)
        all_meta.update(part)

    print(f"[Meta] Loaded metadata for {len(all_meta)} videos.")

    # ----------------------------------------------------
    # Prepare output directories
    # ----------------------------------------------------
    for d in [DIR_TRACKS, DIR_ACTIONS, DIR_SCENES, DIR_AUDIO, DIR_GRAPHS, DIR_LABELS]:
        safe_mkdir(d)

    print("[Preprocess] Using core.extractors (YOLO + VideoMAE + CLIP)")
    vid_ids = list(all_meta.keys())

    # ----------------------------------------------------
    # Iterate over videos
    # ----------------------------------------------------
    for idx, vid_id in enumerate(tqdm(vid_ids, desc="Processing videos")):
        meta = all_meta[vid_id]
        src_path = os.path.join(RAW_VID, meta["source"], "videos", f"{vid_id}.mp4")
        if not os.path.exists(src_path):
            print(f"[Skip] Missing video: {src_path}")
            continue

        # Decide which temporal window to use and construct / reuse a trimmed clip
        try:
            window_path, local_meta = get_windowed_video_path(vid_id, meta)
        except FileNotFoundError as e:
            print(f"[Skip] {e}")
            continue

        # -------- Tracks --------
        out_tracks = os.path.join(DIR_TRACKS, f"{vid_id}.json")
        if not os.path.exists(out_tracks):
            tracks = extract_tracks(window_path)
            atomic_save_json(out_tracks, tracks)
        else:
            with open(out_tracks, "r") as f:
                tracks = json.load(f)

        # If no frames / no people at all, just generate zeros and move on
        if not tracks:
            scene = extract_scene(window_path)
            audio = extract_audio_placeholder()
            graph_seq = []
            labels = generate_risk_curve(local_meta, graph_seq)

            out_scene = os.path.join(DIR_SCENES, f"{vid_id}.npy")
            out_audio = os.path.join(DIR_AUDIO, f"{vid_id}.npy")
            out_graph = os.path.join(DIR_GRAPHS, f"{vid_id}.npy")
            out_label = os.path.join(DIR_LABELS, f"{vid_id}.npy")

            if not os.path.exists(out_scene):
                atomic_save_npy(out_scene, scene)
            if not os.path.exists(out_audio):
                atomic_save_npy(out_audio, audio)
            if not os.path.exists(out_graph):
                atomic_save_npy(out_graph, np.array([], dtype=object))
            if not os.path.exists(out_label):
                atomic_save_npy(out_label, labels)
            continue

        # -------- Actions --------
        out_actions = os.path.join(DIR_ACTIONS, f"{vid_id}.npy")
        if not os.path.exists(out_actions):
            actions = extract_actions(window_path, tracks)
            # actions is list[dict]; np.save handles it as object array
            atomic_save_npy(out_actions, np.array(actions, dtype=object))
        else:
            actions = load_actions(out_actions)

        # -------- Scene --------
        out_scene = os.path.join(DIR_SCENES, f"{vid_id}.npy")
        if not os.path.exists(out_scene):
            scene = extract_scene(window_path)
            atomic_save_npy(out_scene, scene)
        else:
            scene = np.load(out_scene, allow_pickle=True)

        # -------- Audio (placeholder for now) --------
        out_audio = os.path.join(DIR_AUDIO, f"{vid_id}.npy")
        if not os.path.exists(out_audio):
            audio = extract_audio_placeholder()
            atomic_save_npy(out_audio, audio)
        else:
            audio = np.load(out_audio, allow_pickle=True)

        # -------- Graph construction --------
        out_graph = os.path.join(DIR_GRAPHS, f"{vid_id}.npy")
        if not os.path.exists(out_graph):
            graph_seq = build_graph_sequence(tracks, actions, scene, audio)
            atomic_save_npy(out_graph, np.array(graph_seq, dtype=object))
        else:
            graph_seq = np.load(out_graph, allow_pickle=True)
            if isinstance(graph_seq, np.ndarray) and graph_seq.dtype == object:
                graph_seq = list(graph_seq)

        # -------- Generate labels (using window-local meta) --------
        out_label = os.path.join(DIR_LABELS, f"{vid_id}.npy")
        if not os.path.exists(out_label):
            labels = generate_risk_curve(local_meta, graph_seq)
            atomic_save_npy(out_label, labels)
        else:
            labels = np.load(out_label, allow_pickle=True)

    print("[Done] Anticipation preprocessing complete.")


if __name__ == "__main__":
    preprocess_anticipation()
