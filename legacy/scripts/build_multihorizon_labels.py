#!/usr/bin/env python3
"""Build multi-horizon anticipation labels: <out_dir>/<vid_id>.npy [T, K] in {0, 1}.

With --ignore_inside_segments, steps inside a violence segment are -1. Also writes <out_dir>/labels_info.json.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_metadata(meta_dir: str) -> Dict[str, dict]:
    meta = {}
    for fname in ["xd_violence.json", "ucf_crime.json", "kinetics.json"]:
        p = os.path.join(meta_dir, fname)
        if os.path.exists(p):
            with open(p, "r") as f:
                meta.update(json.load(f))
    return meta


def _alias_keys(vid_id: str) -> List[str]:
    keys = []
    if not vid_id:
        return keys
    keys.append(vid_id)
    keys.append(vid_id.lower())
    if vid_id.endswith("_x264"):
        keys.append(vid_id[:-5])
        keys.append(vid_id[:-5].lower())
    if vid_id.endswith(".mp4"):
        keys.append(vid_id[:-4])
        keys.append(vid_id[:-4].lower())
    return list(dict.fromkeys(keys))


def parse_ucf_temporal(path: str) -> Dict[str, List[Tuple[int, int]]]:
    segs: Dict[str, List[Tuple[int, int]]] = {}
    if not path or not os.path.exists(path):
        return segs

    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            fname = os.path.basename(parts[0])
            vid = os.path.splitext(fname)[0]
            nums = [int(x) for x in re.findall(r"-?\d+", " ".join(parts[1:]))]
            if len(nums) < 2:
                continue
            pairs = list(zip(nums[0::2], nums[1::2]))
            valid = [(s, e) for (s, e) in pairs if s >= 0 and e >= 0 and e > s]
            if not valid:
                continue
            for k in _alias_keys(vid):
                segs[k] = valid
    return segs


def parse_xd_temporal(path: str) -> Dict[str, List[Tuple[float, float]]]:
    segs: Dict[str, List[Tuple[float, float]]] = {}
    if not path or not os.path.exists(path):
        return segs

    if path.lower().endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k, v in data.items():
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
                        elif isinstance(item, dict) and "start" in item and "end" in item:
                            try:
                                s = float(item["start"]); e = float(item["end"])
                                if e > s and s >= 0 and e >= 0:
                                    out.append((s, e))
                            except Exception:
                                pass
                if out:
                    for kk in _alias_keys(str(k)):
                        segs[kk] = out
        return segs

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
            nums = [float(x) for x in float_re.findall(" ".join(parts[1:]))]
            if len(nums) < 2:
                continue
            pairs = list(zip(nums[0::2], nums[1::2]))
            valid = [(s, e) for (s, e) in pairs if s >= 0 and e >= 0 and e > s]
            if not valid:
                continue
            for k in _alias_keys(vid):
                segs[k] = valid
    return segs


def map_t_to_raw_frame(t: int, T: int, total_frames: int) -> int:
    raw = int(round(((t + 0.5) / float(T)) * float(total_frames)))
    return max(0, min(total_frames - 1, raw))


def ucf_segments_to_seconds(segs: List[Tuple[int, int]], fps: float) -> List[Tuple[float, float]]:
    if not segs:
        return []
    if fps <= 0:
        fps = 25.0
    out = []
    for (s, e) in segs:
        out.append((float(s) / fps, float(e) / fps))
    out.sort()
    return out


def xd_segments_to_seconds(
    segs: List[Tuple[float, float]],
    *,
    fps: float,
    total_frames: int,
    units: str = "auto",
) -> List[Tuple[float, float]]:
    """Convert XD segments to seconds; units="auto" infers normalized, seconds or frames."""
    if not segs:
        return []
    if fps <= 0:
        fps = 25.0

    units = (units or "auto").lower()
    max_end = max(e for (_, e) in segs)

    mode = units
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

    out = []
    for (s, e) in segs:
        if mode == "frames":
            out.append((float(s) / fps, float(e) / fps))
        elif mode == "seconds":
            out.append((float(s), float(e)))
        elif mode == "normalized":
            out.append(((float(s) * total_frames) / fps, (float(e) * total_frames) / fps))
        else:
            out.append((float(s) / fps, float(e) / fps))
    out.sort()
    return out


def get_T_from_graph(graphs_dir: Path, vid_id: str) -> Optional[int]:
    npy = graphs_dir / f"{vid_id}.npy"
    pt  = graphs_dir / f"{vid_id}.pt"
    gpath = npy if npy.exists() else pt
    if not gpath.exists():
        for k in _alias_keys(vid_id):
            npy = graphs_dir / f"{k}.npy"
            pt  = graphs_dir / f"{k}.pt"
            gpath = npy if npy.exists() else pt
            if gpath.exists():
                break
        if not gpath.exists():
            return None

    try:
        if str(gpath).endswith(".npy"):
            seq = np.load(gpath, allow_pickle=True)
            return int(len(seq))
        else:
            import torch
            seq = torch.load(gpath)
            return int(len(seq))
    except Exception:
        return None


def parse_horizons(s: str) -> List[float]:
    hs = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        hs.append(float(part))
    hs = sorted(list(dict.fromkeys(hs)))
    if not hs:
        raise ValueError("No horizons parsed. Use --horizons_sec like 2,5,10,20,30,60")
    return hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_dir", required=True)
    ap.add_argument("--meta_dir", default="data/metadata")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--ucf_temporal_file", default="data/raw_videos/ucf_crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
    ap.add_argument("--xd_temporal_file", default="data/raw_videos/xd_violence/annotations.txt")
    ap.add_argument("--xd_units", default="auto", help="auto|frames|seconds")

    ap.add_argument("--horizons_sec", default="2,5,10,20,30,60,120")
    ap.add_argument("--ignore_inside_segments", action="store_true", help="Set -1 when t lies inside a violence segment (recommended).")

    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world_size", type=int, default=1)

    args = ap.parse_args()

    graphs_dir = Path(args.graphs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(args.meta_dir)
    if not meta:
        raise SystemExit(f"No metadata loaded from {args.meta_dir}")

    ucf_map = parse_ucf_temporal(args.ucf_temporal_file)
    xd_map  = parse_xd_temporal(args.xd_temporal_file)
    horizons = parse_horizons(args.horizons_sec)
    K = len(horizons)

    graph_files = [p.name for p in graphs_dir.iterdir() if p.name.endswith(".npy") or p.name.endswith(".pt")]
    vid_ids_all = sorted([os.path.splitext(f)[0] for f in graph_files])
    vid_ids = vid_ids_all[args.rank::args.world_size]
    print(f"[Index] graphs={len(graph_files)} shard vids={len(vid_ids)} rank={args.rank}/{args.world_size}")

    with (out_dir / "labels_info.json").open("w") as f:
        json.dump(
            {
                "task": "anticipation_multi_horizon",
                "horizons_sec": horizons,
                "ignore_inside_segments": bool(args.ignore_inside_segments),
                "ucf_temporal_file": args.ucf_temporal_file,
                "xd_temporal_file": args.xd_temporal_file,
                "xd_units": args.xd_units,
            },
            f,
            indent=2,
        )

    n_written = 0
    n_skipped = 0

    for vid_id in vid_ids:
        info = meta.get(vid_id, None)
        if info is None:
            for k in _alias_keys(vid_id):
                if k in meta:
                    info = meta[k]
                    break
        if info is None:
            n_skipped += 1
            continue

        source = info.get("source", "unknown")
        fps = float(info.get("fps", 25.0) or 25.0)
        total_frames = int(info.get("total_frames", 0) or 0)
        if fps <= 0:
            fps = 25.0
        if total_frames <= 0:
            n_skipped += 1
            continue

        T = get_T_from_graph(graphs_dir, vid_id)
        if not T or T <= 0:
            n_skipped += 1
            continue

        segs_sec: List[Tuple[float, float]] = []
        if source == "ucf_crime":
            segs = None
            for k in _alias_keys(vid_id):
                if k in ucf_map:
                    segs = ucf_map[k]
                    break
            if segs:
                segs_sec = ucf_segments_to_seconds(segs, fps=fps)

        elif source == "xd_violence":
            segs = None
            for k in _alias_keys(vid_id):
                if k in xd_map:
                    segs = xd_map[k]
                    break
            if segs:
                segs_sec = xd_segments_to_seconds(segs, fps=fps, total_frames=total_frames, units=args.xd_units)

        starts = [s for (s, _e) in segs_sec]
        def inside_any(tsec: float) -> bool:
            for (s, e) in segs_sec:
                if s <= tsec <= e:
                    return True
            return False

        def next_start(tsec: float) -> Optional[float]:
            for s in starts:
                if s >= tsec:
                    return s
            return None

        y = np.zeros((T, K), dtype=np.float32)

        for t in range(T):
            raw = map_t_to_raw_frame(t, T, total_frames)
            tsec = float(raw) / float(fps)

            if args.ignore_inside_segments and segs_sec and inside_any(tsec):
                y[t, :] = -1.0
                continue

            s = next_start(tsec) if starts else None
            if s is None:
                y[t, :] = 0.0
                continue

            d = float(s - tsec)  # seconds until next segment start
            for k, h in enumerate(horizons):
                y[t, k] = 1.0 if d <= float(h) else 0.0

        np.save(out_dir / f"{vid_id}.npy", y)
        n_written += 1

    print(f"[Done] wrote={n_written} skipped={n_skipped} out_dir={out_dir}")


if __name__ == "__main__":
    main()
