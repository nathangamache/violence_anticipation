#!/usr/bin/env python3
"""Build metadata/xd_violence.json (fps, total_frames, label, first event frame) from videos + annotations.txt."""
import os, json, cv2, re

BASE = os.path.expanduser("~/projects/violence_anticipation/data")
RAW = f"{BASE}/raw_videos/xd_violence"
VID_DIR = f"{RAW}/videos"
ANN_FILE = f"{RAW}/annotations.txt"
OUT_JSON = f"{BASE}/metadata/xd_violence.json"


def safe_mkdir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def parse_timestamp(ts):
    """'00-02-39' (HH-MM-SS) -> seconds."""
    h, m, s = [int(x) for x in ts.split("-")]
    return h*3600 + m*60 + s

def parse_clip_times(filename):
    """(start_s, end_s) from a name like 'v=u5SF4SlqNDQ__#00-02-39_00-03-41_label_G-0-0.mp4', else None."""
    m = re.search(r"#(\d\d-\d\d-\d\d)_(\d\d-\d\d-\d\d)", filename)
    if not m:
        return None
    start = parse_timestamp(m.group(1))
    end = parse_timestamp(m.group(2))
    return start, end


def main():
    metadata = {}
    safe_mkdir(os.path.dirname(OUT_JSON))

    ann = {}

    with open(ANN_FILE, "r") as f:
        for line in f:
            parts = line.strip().split()
            vid_id = parts[0]
            frames = [int(x) for x in parts[1:]]
            ann[vid_id] = frames

    for fname in os.listdir(VID_DIR):
        if not fname.endswith(".mp4"):
            continue

        vid_id = fname.replace(".mp4", "")
        full_path = os.path.join(VID_DIR, fname)

        cap = cv2.VideoCapture(full_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        clip_times = parse_clip_times(fname)
        if clip_times:
            clip_start_sec, clip_end_sec = clip_times
            clip_start_frame = int(clip_start_sec * fps)
            clip_end_frame = int(clip_end_sec * fps)
        else:
            clip_start_frame = 0
            clip_end_frame = total

        if vid_id not in ann:
            metadata[vid_id] = {
                "event_frame": 0,
                "total_frames": total,
                "label": 0,
                "fps": fps,
                "source": "xd_violence"
            }
            continue

        # Known bug: annotation frames are already clip-relative (docs/codebase/data_and_labels.md section 11).
        abs_frames = ann[vid_id]
        rel_frames = [f - clip_start_frame for f in abs_frames]

        rel_frames = [f for f in rel_frames if 0 <= f < total]

        if len(rel_frames) == 0:
            event_frame = 0
            label = 0
        else:
            event_frame = min(rel_frames)
            label = 1

        metadata[vid_id] = {
            "event_frame": event_frame,
            "total_frames": total,
            "label": label,
            "fps": fps,
            "source": "xd_violence"
        }

    with open(OUT_JSON, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[XD] Done. Saved metadata → {OUT_JSON}")


if __name__ == "__main__":
    main()
