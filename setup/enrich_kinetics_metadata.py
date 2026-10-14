#!/usr/bin/env python3
"""Add fps, total_frames, a synthetic midpoint event_frame and label=1 to metadata/kinetics.json in place."""
import os
import json
import cv2

BASE = os.path.expanduser("~/projects/violence_anticipation/data")
KIN_VID_DIR = f"{BASE}/raw_videos/kinetics/videos"
KIN_META = f"{BASE}/metadata/kinetics.json"


def main():
    if not os.path.exists(KIN_META):
        raise FileNotFoundError(f"Metadata file not found: {KIN_META}")

    with open(KIN_META, "r") as f:
        meta = json.load(f)

    updated = 0
    missing = 0
    bad = 0

    for clip_id, info in meta.items():
        vid_path = os.path.join(KIN_VID_DIR, f"{clip_id}.mp4")
        if not os.path.exists(vid_path):
            print(f"[Kinetics] Skip missing video: {vid_path}")
            missing += 1
            continue

        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"[Kinetics] Cannot open video: {vid_path}")
            bad += 1
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        if total_frames <= 0:
            print(f"[Kinetics] Zero/unknown frames for: {vid_path}")
            bad += 1
            continue

        info["fps"] = float(fps)
        info["total_frames"] = int(total_frames)

        event_frame = max(0, min(total_frames - 1, total_frames // 2))
        info["event_frame"] = int(event_frame)
        info["label"] = 1
        info["source"] = "kinetics"  # must match the raw_videos/<source> folder name

        updated += 1

    with open(KIN_META, "w") as f:
        json.dump(meta, f, indent=4)

    print(f"[Kinetics Enrich] Updated: {updated}  Missing: {missing}  Bad: {bad}")
    print(f"[Kinetics Enrich] Saved → {KIN_META}")


if __name__ == "__main__":
    main()
