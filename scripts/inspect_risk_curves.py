#!/usr/bin/env python3
"""Print and plot ground-truth risk curves to check that labels ramp up before the event.

Usage:
  python -m scripts.inspect_risk_curves --data_root data --video_id RoadAccidents039_x264
  python -m scripts.inspect_risk_curves --data_root data --auto_samples
"""

import os
import json
import argparse

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


def load_metadata(meta_dir):
    meta = {}
    for fname in ["xd_violence.json", "ucf_crime.json", "kinetics.json"]:
        path = os.path.join(meta_dir, fname)
        if os.path.exists(path):
            with open(path, "r") as f:
                meta.update(json.load(f))
    return meta


def pick_sample_videos(meta, k_pos_early=2, k_pos_late=2, k_neg=2):
    pos_early = []
    pos_late = []
    negs = []

    for vid, info in meta.items():
        label = info.get("label", 0)
        total_frames = info.get("total_frames", None)
        event_frame = info.get("event_frame", None)
        if total_frames is None or event_frame is None:
            continue

        frac = event_frame / max(1, total_frames)

        if label == 1:
            if frac <= 0.33 and len(pos_early) < k_pos_early:
                pos_early.append(vid)
            elif frac >= 0.67 and len(pos_late) < k_pos_late:
                pos_late.append(vid)
        else:
            if len(negs) < k_neg:
                negs.append(vid)

        if len(pos_early) >= k_pos_early and len(pos_late) >= k_pos_late and len(negs) >= k_neg:
            break

    return pos_early, pos_late, negs


def plot_curve(vid_id, labels, info, out_dir):
    label = info.get("label", 0)
    total_frames = info.get("total_frames", len(labels))
    event_frame = info.get("event_frame", 0)
    fps = info.get("fps", 25.0)

    nonzero_idxs = np.where(labels > 0)[0]
    first_nonzero = int(nonzero_idxs[0]) if nonzero_idxs.size > 0 else None

    print("=" * 80)
    print(f"Video: {vid_id}")
    print(f"  Clip label           : {'VIOLENT' if label == 1 else 'NON-VIOLENT'}")
    print(f"  Total frames (meta)  : {total_frames}")
    print(f"  Label array length   : {len(labels)}")
    print(f"  Event frame (meta)   : {event_frame}")
    if first_nonzero is not None:
        print(f"  First nonzero risk   : frame {first_nonzero} (~{first_nonzero / fps:.2f}s)")
    else:
        print("  First nonzero risk   : NONE (all zeros)")

    print("  Sample risk values (every ~10% of video):")
    for p in np.linspace(0, 1, num=11):
        idx = int(p * (len(labels) - 1))
        print(f"    frame {idx:4d} (p={p:4.2f}): {labels[idx]:.3f}")

    if not HAS_PLT:
        print("  (matplotlib not installed; skipping plots)")
        return

    os.makedirs(out_dir, exist_ok=True)
    t = np.arange(len(labels)) / fps
    plt.figure(figsize=(10, 4))
    plt.plot(t, labels, label="GT risk")
    if event_frame is not None:
        plt.axvline(event_frame / fps, color="red", linestyle="--", label="event_frame")
    plt.xlabel("Time (s)")
    plt.ylabel("Risk")
    plt.title(f"GT Risk Curve: {vid_id}")
    plt.legend()
    out_path = os.path.join(out_dir, f"{vid_id}_risk_curve.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Plot saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data",
                        help="Root data dir containing metadata/ and processed/labels/")
    parser.add_argument("--video_id", type=str, default=None,
                        help="Specific video ID to inspect (without extension).")
    parser.add_argument("--auto_samples", action="store_true",
                        help="If set, auto-pick a few pos/neg examples to inspect.")
    parser.add_argument("--labels_dir", type=str, default=None,
                        help="Override labels dir if not in data/processed/labels")
    args = parser.parse_args()

    meta_dir = os.path.join(args.data_root, "metadata")
    labels_dir = args.labels_dir or os.path.join(args.data_root, "processed", "labels")

    meta = load_metadata(meta_dir)
    if not meta:
        raise RuntimeError(f"No metadata found under {meta_dir}")

    if args.video_id:
        vids = [args.video_id]
    elif args.auto_samples:
        pos_early, pos_late, negs = pick_sample_videos(meta)
        print("[Auto-sample] Early-violence positives:", pos_early)
        print("[Auto-sample] Late-violence positives :", pos_late)
        print("[Auto-sample] Non-violent negatives   :", negs)
        vids = pos_early + pos_late + negs
    else:
        raise RuntimeError("Provide --video_id or --auto_samples")

    for vid in vids:
        label_path = os.path.join(labels_dir, f"{vid}.npy")
        if not os.path.exists(label_path):
            print(f"[Warn] Missing labels for {vid}: {label_path}")
            continue

        labels = np.load(label_path, allow_pickle=True)
        labels = np.asarray(labels, dtype=np.float32).reshape(-1)

        info = meta.get(vid, {})
        plot_curve(vid, labels, info, out_dir=os.path.join(args.data_root, "plots"))


if __name__ == "__main__":
    main()
