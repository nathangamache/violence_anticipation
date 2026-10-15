#!/usr/bin/env python3
"""Binary "violence now or within --horizon_sec" labels from risk labels (threshold, then future-OR)."""
import argparse, os, json
import numpy as np
from pathlib import Path

def load_metadata(meta_dir: str):
    meta = {}
    for fname in ["xd_violence.json", "ucf_crime.json", "kinetics.json"]:
        p = os.path.join(meta_dir, fname)
        if os.path.exists(p):
            with open(p, "r") as f:
                meta.update(json.load(f))
    return meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", default="data/processed/labels")
    ap.add_argument("--meta_dir", default="data/metadata")
    ap.add_argument("--out_dir", default="data/processed/labels_future_h30")
    ap.add_argument("--horizon_sec", type=float, default=30.0)
    ap.add_argument("--gt_thresh", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    meta = load_metadata(args.meta_dir)

    label_files = sorted(Path(args.labels_dir).glob("*.npy"))
    if not label_files:
        raise RuntimeError(f"No .npy labels found in {args.labels_dir}")

    n_done = 0
    for lp in label_files:
        vid_id = lp.stem
        y = np.load(lp).astype(np.float32)
        T = len(y)
        if T == 0:
            continue

        info = meta.get(vid_id, {})
        fps = float(info.get("fps", 25.0))
        total_frames = int(info.get("total_frames", 0))

        if fps > 0 and total_frames > 0:
            clip_sec = total_frames / fps
            dt = clip_sec / T
        else:
            dt = 1.0

        H = int(np.ceil(args.horizon_sec / max(dt, 1e-6)))

        y_bin = (y >= args.gt_thresh).astype(np.int32)

        y_future = np.zeros_like(y_bin, dtype=np.int32)
        for t in range(T):
            y_future[t] = int(y_bin[t : min(T, t + H + 1)].max())

        outp = os.path.join(args.out_dir, f"{vid_id}.npy")
        np.save(outp, y_future.astype(np.float32))
        n_done += 1

    print(f"[OK] Wrote {n_done} anticipation label files -> {args.out_dir}")

if __name__ == "__main__":
    main()
