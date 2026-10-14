#!/usr/bin/env python3
"""
Build frame-level label arrays aligned to graph sequence length T.

Option A behavior:
- UCF-Crime: ONLY write labels for videos that have valid temporal annotations
  in Temporal_Anomaly_Annotation_for_Testing_Videos.txt.
- XD-Violence: ONLY write labels for videos that have valid temporal annotations
  in --xd_temporal_file.
- If no valid temporal segment exists -> skip (no label file produced).

Tasks:
- detection:
    y[t] = 1 if raw_frame(t) is inside any (start,end) temporal segment.

- anticipation (PRE-only by default):
    y[t] = 1 if raw_frame(t) is within horizon window BEFORE any segment start:
           raw_frame(t) in [start - horizon_frames, start)  (strict anticipation)

    If --include_during_event is set:
           raw_frame(t) in [start - horizon_frames, end]    (early detection variant)

    If --ignore_during_event is set:
           y[t] = ignore_value (default -1.0) for any t where raw_frame(t) is inside
           the event segment [start, end] (ignore takes precedence over labeling)

For other sources (kinetics), we use metadata event_frame if present; else fallback to clip label.
Note: Kinetics behavior is kept simple/legacy-style.
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2


# ----------------------------
# Metadata
# ----------------------------
def load_metadata(meta_dir: str) -> Dict[str, dict]:
    meta = {}
    for fname in ["xd_violence.json", "ucf_crime.json", "kinetics.json"]:
        p = os.path.join(meta_dir, fname)
        if os.path.exists(p):
            with open(p, "r") as f:
                meta.update(json.load(f))
    return meta


# ----------------------------
# Temporal annotation parsing
# ----------------------------
def _add_alias_keys(segs: Dict[str, list], vid: str, value):
    """Add helpful aliases: lower(), strip _x264, strip .mp4, etc."""
    if not vid:
        return
    segs[vid] = value
    segs[vid.lower()] = value

    if vid.endswith(".mp4"):
        base = vid[:-4]
        segs[base] = value
        segs[base.lower()] = value

    stem = os.path.splitext(os.path.basename(vid))[0]
    segs[stem] = value
    segs[stem.lower()] = value

    if stem.endswith("_x264"):
        base2 = stem[:-5]
        segs[base2] = value
        segs[base2.lower()] = value


def parse_ucf_temporal(path: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Returns: vid_id -> list of (start_frame, end_frame) segments (ints, in frames).
    Keeps only valid segments where start>=0, end>start.
    Adds aliases without _x264.
    """
    segs: Dict[str, List[Tuple[int, int]]] = {}

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if not parts:
                continue

            fname = os.path.basename(parts[0])
            vid = os.path.splitext(fname)[0]  # remove .mp4

            nums = [int(x) for x in re.findall(r"-?\d+", " ".join(parts[1:]))]
            if len(nums) < 2:
                continue

            pairs = list(zip(nums[0::2], nums[1::2]))
            valid = [(s, e) for (s, e) in pairs if s >= 0 and e >= 0 and e > s]
            if not valid:
                continue

            _add_alias_keys(segs, vid, valid)

    return segs


def parse_xd_temporal(path: str) -> Dict[str, List[Tuple[float, float]]]:
    """
    Returns: vid_id -> list of (start, end) segments as floats.
    Units may be frames/seconds/normalized fractions; we decide later at build time.

    Supported formats:
      1) JSON dict: vid -> [[s,e], ...] OR [{"start":s,"end":e}, ...]
      2) Text/CSV-ish:
         vid_id <anything> s1 e1 s2 e2 ...
         # comments allowed
    """
    segs: Dict[str, List[Tuple[float, float]]] = {}
    if not path:
        return segs
    if not os.path.exists(path):
        print(f"[XD] WARNING: xd_temporal_file not found: {path}. XD will be skipped under Option A.")
        return segs

    # JSON mode
    if path.lower().endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            for k, v in data.items():
                vid = str(k)
                out: List[Tuple[float, float]] = []

                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            try:
                                s = float(item[0]); e = float(item[1])
                                if e > s and s >= 0 and e >= 0:
                                    out.append((s, e))
                            except Exception:
                                pass
                        elif isinstance(item, dict):
                            if "start" in item and "end" in item:
                                try:
                                    s = float(item["start"]); e = float(item["end"])
                                    if e > s and s >= 0 and e >= 0:
                                        out.append((s, e))
                                except Exception:
                                    pass

                if out:
                    _add_alias_keys(segs, vid, out)

        return segs

    # Text / CSV-ish mode
    float_re = re.compile(r"-?\d+(?:\.\d+)?")

    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = re.split(r"[,\t ]+", line)
            if not parts:
                continue

            vid = parts[0].strip()
            if not vid:
                continue

            nums = [float(x) for x in float_re.findall(" ".join(parts[1:]))]
            if len(nums) < 2:
                continue

            pairs = list(zip(nums[0::2], nums[1::2]))
            valid = [(s, e) for (s, e) in pairs if s >= 0 and e >= 0 and e > s]
            if not valid:
                continue

            _add_alias_keys(segs, vid, valid)

    return segs


# ----------------------------
# Video utilities
# ----------------------------
def get_total_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return max(tf, 0)


def map_t_to_raw_frame(t: int, T: int, total_frames: int) -> int:
    # match your VLM sampler convention (center of bin)
    raw = int(round(((t + 0.5) / float(T)) * float(total_frames)))
    return max(0, min(total_frames - 1, raw))


def infer_T_from_graph(graph_path: str) -> Optional[int]:
    try:
        if graph_path.endswith(".npy"):
            seq = np.load(graph_path, allow_pickle=True)
            return int(len(seq))
        elif graph_path.endswith(".pt"):
            import torch
            seq = torch.load(graph_path)
            return int(len(seq))
    except Exception:
        return None
    return None


# ----------------------------
# Segment conversion + label building
# ----------------------------
def segments_to_frame_pairs(
    segments: List[Tuple[float, float]],
    *,
    total_frames: int,
    fps: float,
    units: str = "auto",
) -> List[Tuple[int, int]]:
    """
    Convert float segments into integer raw-frame segments.

    units:
      - frames: endpoints are raw frame indices
      - seconds: endpoints are seconds within the clip
      - auto:
          * if max_end <= 1.0 -> treat as normalized fraction of clip [0,1]
          * elif max_end * fps <= total_frames * 1.25 -> treat as seconds
          * else treat as frames
    """
    if total_frames <= 0:
        return []
    if fps <= 0:
        fps = 25.0
    if not segments:
        return []

    max_end = max(e for (_, e) in segments)

    mode = (units or "auto").lower()
    if mode not in ("auto", "frames", "seconds"):
        mode = "auto"

    if mode == "auto":
        if max_end <= 1.0:
            mode = "normalized"
        else:
            if (max_end * fps) <= (total_frames * 1.25):
                mode = "seconds"
            else:
                mode = "frames"

    out: List[Tuple[int, int]] = []
    for (s, e) in segments:
        if mode == "frames":
            sf = int(round(s))
            ef = int(round(e))
        elif mode == "seconds":
            sf = int(round(s * fps))
            ef = int(round(e * fps))
        elif mode == "normalized":
            sf = int(round(s * total_frames))
            ef = int(round(e * total_frames))
        else:
            sf = int(round(s))
            ef = int(round(e))

        sf = max(0, min(total_frames - 1, sf))
        ef = max(0, min(total_frames - 1, ef))
        if ef > sf:
            out.append((sf, ef))

    return out


def build_ignore_mask_from_frame_segments(
    T: int,
    total_frames: int,
    seg_frames: List[Tuple[int, int]],
) -> np.ndarray:
    """
    ignore_mask[t] = True if raw_frame(t) is inside ANY segment [s, e].
    """
    ignore = np.zeros((T,), dtype=np.bool_)
    if T <= 0 or total_frames <= 0 or not seg_frames:
        return ignore

    for t in range(T):
        rf = map_t_to_raw_frame(t, T, total_frames)
        for (s, e) in seg_frames:
            if s <= rf <= e:
                ignore[t] = True
                break
    return ignore


def build_labels_detection_from_frame_segments(
    T: int,
    total_frames: int,
    seg_frames: List[Tuple[int, int]],
) -> np.ndarray:
    y = np.zeros((T,), dtype=np.float32)
    if T <= 0 or total_frames <= 0 or not seg_frames:
        return y

    for t in range(T):
        rf = map_t_to_raw_frame(t, T, total_frames)
        for (s, e) in seg_frames:
            if s <= rf <= e:
                y[t] = 1.0
                break
    return y


def build_labels_anticipation_from_frame_segments(
    T: int,
    total_frames: int,
    seg_frames: List[Tuple[int, int]],
    *,
    horizon_sec: float,
    fps: float,
    include_during_event: bool = False,
) -> np.ndarray:
    """
    Anticipation labels based on segment START times.
    Strict PRE-only: positives only BEFORE start: [start - H, start)
    If include_during_event: [start - H, end]
    """
    y = np.zeros((T,), dtype=np.float32)
    if T <= 0 or total_frames <= 0 or not seg_frames:
        return y

    if fps <= 0:
        fps = 25.0
    h_frames = int(round(max(0.0, horizon_sec) * fps))
    if h_frames <= 0:
        return y  # no anticipation window

    for t in range(T):
        rf = map_t_to_raw_frame(t, T, total_frames)
        for (s, e) in seg_frames:
            w0 = max(0, s - h_frames)
            if include_during_event:
                if w0 <= rf <= e:
                    y[t] = 1.0
                    break
            else:
                if w0 <= rf < s:
                    y[t] = 1.0
                    break
    return y


def build_labels_from_event_frame(
    T: int,
    event_frame: int,
    total_frames: int,
    horizon_sec: float,
    fps: float,
) -> np.ndarray:
    """
    Legacy: label a window leading up to an event_frame.
    For detection-like use: set horizon_sec=0.
    """
    y = np.zeros((T,), dtype=np.float32)
    if T <= 0 or total_frames <= 0 or fps <= 0:
        return y

    h_frames = int(round(max(0.0, horizon_sec) * fps))
    start = max(0, int(event_frame) - h_frames)
    end = int(event_frame)

    for t in range(T):
        rf = map_t_to_raw_frame(t, T, total_frames)
        if start <= rf <= end:
            y[t] = 1.0
    return y


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--meta_dir", default="data/metadata")

    ap.add_argument("--kinetics_root", default="data/raw_videos/kinetics/videos")
    ap.add_argument("--ucf_root", default="data/raw_videos/ucf_crime")
    ap.add_argument("--xd_root", default="data/raw_videos/xd_violence/videos")

    ap.add_argument("--ucf_temporal_file", default="data/raw_videos/ucf_crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
    ap.add_argument("--xd_temporal_file", default="data/raw_videos/xd_violence/annotations.txt")

    ap.add_argument("--xd_units", type=str, default="auto")  # frames|seconds|auto

    ap.add_argument("--task", type=str, default="detection", choices=["detection", "anticipation"])
    ap.add_argument("--horizon_sec", type=float, default=0.0)

    ap.add_argument("--include_during_event", action="store_true")

    # NEW: ignore points during the event segment [start, end]
    ap.add_argument("--ignore_during_event", action="store_true")
    ap.add_argument("--ignore_value", type=float, default=-1.0)

    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    meta = load_metadata(args.meta_dir)
    if not meta:
        raise RuntimeError(f"No metadata loaded from {args.meta_dir}")

    ucf_segments = parse_ucf_temporal(args.ucf_temporal_file)
    print(f"[UCF] Loaded temporal annotations for {len(ucf_segments)} keys (includes aliases).")

    xd_segments = parse_xd_temporal(args.xd_temporal_file)
    print(f"[XD] Loaded temporal annotations for {len(xd_segments)} keys (includes aliases).")

    def find_video_path(vid_id: str, source: str) -> Optional[str]:
        candidates = [vid_id, vid_id.lower()]
        if vid_id.endswith("_x264"):
            candidates += [vid_id[:-5], vid_id[:-5].lower()]

        if source == "kinetics":
            root = Path(args.kinetics_root)
        elif source == "ucf_crime":
            root = Path(args.ucf_root)
        elif source == "xd_violence":
            root = Path(args.xd_root)
        else:
            return None

        for c in candidates:
            p = root / f"{c}.mp4"
            if p.exists():
                return str(p)

        for p in root.rglob("*.mp4"):
            stem = p.stem
            if stem == vid_id or stem.lower() == vid_id.lower():
                return str(p)
            if stem.endswith("_x264") and (stem[:-5] == vid_id or stem[:-5].lower() == vid_id.lower()):
                return str(p)

        return None

    total = 0
    written = 0
    skipped_no_meta = 0
    skipped_ucf_no_temporal = 0
    skipped_xd_no_temporal = 0
    skipped_video_missing = 0
    skipped_bad_T = 0
    ignored_points_total = 0
    points_total = 0

    for fn in os.listdir(args.graphs_dir):
        if not (fn.endswith(".npy") or fn.endswith(".pt")):
            continue
        total += 1
        vid_id = os.path.splitext(fn)[0]
        info = meta.get(vid_id, None)
        if info is None:
            skipped_no_meta += 1
            continue

        source = info.get("source", "unknown")
        graph_path = os.path.join(args.graphs_dir, fn)
        out_path = os.path.join(args.out_dir, f"{vid_id}.npy")

        if os.path.exists(out_path) and not args.overwrite:
            continue

        T = infer_T_from_graph(graph_path)
        if T is None or T <= 0:
            skipped_bad_T += 1
            continue

        # Option A filters
        if source == "ucf_crime":
            segs = ucf_segments.get(vid_id) or ucf_segments.get(vid_id.lower())
            if not segs:
                skipped_ucf_no_temporal += 1
                continue

        if source == "xd_violence":
            segs = xd_segments.get(vid_id) or xd_segments.get(vid_id.lower())
            if not segs:
                skipped_xd_no_temporal += 1
                continue

        vpath = find_video_path(vid_id, source)
        if not vpath or not os.path.exists(vpath):
            skipped_video_missing += 1
            continue

        total_frames = int(info.get("total_frames", 0)) if info else 0
        if total_frames <= 0:
            total_frames = get_total_frames(vpath)
        if total_frames <= 0:
            skipped_video_missing += 1
            continue

        fps = float(info.get("fps", 25.0)) if info else 25.0
        task = args.task

        seg_frames: List[Tuple[int, int]] = []

        if source == "ucf_crime":
            segs = ucf_segments.get(vid_id) or ucf_segments.get(vid_id.lower())
            seg_frames = [(int(s), int(e)) for (s, e) in segs]

            if task == "detection":
                y = build_labels_detection_from_frame_segments(T, total_frames, seg_frames)
            else:
                y = build_labels_anticipation_from_frame_segments(
                    T, total_frames, seg_frames,
                    horizon_sec=float(args.horizon_sec),
                    fps=fps,
                    include_during_event=bool(args.include_during_event),
                )

        elif source == "xd_violence":
            segs = xd_segments.get(vid_id) or xd_segments.get(vid_id.lower())
            seg_frames = segments_to_frame_pairs(
                segs,
                total_frames=total_frames,
                fps=fps,
                units=args.xd_units,
            )

            if task == "detection":
                y = build_labels_detection_from_frame_segments(T, total_frames, seg_frames)
            else:
                y = build_labels_anticipation_from_frame_segments(
                    T, total_frames, seg_frames,
                    horizon_sec=float(args.horizon_sec),
                    fps=fps,
                    include_during_event=bool(args.include_during_event),
                )

        else:
            # kinetics: legacy behavior
            label = int(info.get("label", 0))
            event_frame = info.get("event_frame", None)

            if event_frame is not None and int(event_frame) > 0:
                y = build_labels_from_event_frame(
                    T=T,
                    event_frame=int(event_frame),
                    total_frames=total_frames,
                    horizon_sec=float(args.horizon_sec),
                    fps=fps,
                )
            else:
                y = np.ones((T,), dtype=np.float32) if label == 1 else np.zeros((T,), dtype=np.float32)

        # NEW: optionally ignore points during the event segment (UCF/XD only)
        if args.ignore_during_event and seg_frames and task == "anticipation":
            ignore_mask = build_ignore_mask_from_frame_segments(T, total_frames, seg_frames)
            # ignore takes precedence over “include_during_event”
            y = y.astype(np.float32, copy=False)
            y[ignore_mask] = float(args.ignore_value)

            ignored_points_total += int(ignore_mask.sum())
            points_total += int(T)

        np.save(out_path, y.astype(np.float32))
        written += 1

    print("\n========== LABEL BUILD SUMMARY ==========")
    print(f"task                    : {args.task}")
    if args.task == "anticipation":
        print(f"horizon_sec             : {args.horizon_sec}")
        print(f"include_during_event    : {bool(args.include_during_event)}")
        print(f"ignore_during_event     : {bool(args.ignore_during_event)} (value={args.ignore_value})")
    print(f"graphs_seen             : {total}")
    print(f"labels_written          : {written}")
    print(f"skipped_no_meta         : {skipped_no_meta}")
    print(f"skipped_bad_T           : {skipped_bad_T}")
    print(f"skipped_video_missing   : {skipped_video_missing}")
    print(f"skipped_ucf_no_temporal : {skipped_ucf_no_temporal}  (Option A)")
    print(f"skipped_xd_no_temporal  : {skipped_xd_no_temporal}   (Option A)")
    if args.ignore_during_event and points_total > 0:
        print(f"ignored_points_total    : {ignored_points_total} / {points_total} ({ignored_points_total / points_total:.3f})")
    print("========================================\n")


if __name__ == "__main__":
    main()
