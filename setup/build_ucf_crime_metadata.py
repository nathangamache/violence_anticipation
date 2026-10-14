#!/usr/bin/env python3
import os, json, cv2

BASE = os.path.expanduser("~/projects/violence_anticipation/data")
RAW_DIR = f"{BASE}/raw_videos/ucf_crime"
VID_DIR = f"{RAW_DIR}/videos"   # your mp4s live here
ANN_FILE = f"{RAW_DIR}/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
OUT_JSON = f"{BASE}/metadata/ucf_crime.json"


def safe_mkdir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def parse_annotation_line(line):
    """
    Example:
    Abuse028_x264.mp4  Abuse  165 240  -1 -1
    Format:
    video  category  start1 end1  start2 end2
    """
    parts = [p for p in line.strip().split(" ") if p != ""]
    if len(parts) < 4:
        return None

    video = parts[0].replace(".mp4", "")
    # parts[1] = category (ignored)

    try:
        start1 = int(parts[2])
        end1 = int(parts[3])
    except:
        return None

    # Optional second interval
    if len(parts) >= 6:
        try:
            start2 = int(parts[4])
            end2 = int(parts[5])
        except:
            start2, end2 = -1, -1
    else:
        start2, end2 = -1, -1

    starts = []
    if start1 >= 0:
        starts.append(start1)
    if start2 >= 0:
        starts.append(start2)

    if not starts:
        return None

    # FIRST violent frame
    return video, min(starts)


def main():
    safe_mkdir(os.path.dirname(OUT_JSON))

    # ----------------------------------------
    # STEP 1 — parse annotation file
    # ----------------------------------------
    violent_events = {}

    with open(ANN_FILE, "r") as f:
        for line in f:
            parsed = parse_annotation_line(line)
            if parsed is None:
                continue
            video, first_event_frame = parsed
            violent_events[video] = first_event_frame

    # ----------------------------------------
    # STEP 2 — Process all videos in /videos/
    # ----------------------------------------
    metadata = {}
    for f in os.listdir(VID_DIR):
        if not f.endswith(".mp4"):
            continue

        vid_id = f.replace(".mp4", "")
        path = os.path.join(VID_DIR, f)

        # get video info
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        if vid_id in violent_events:
            event_frame = violent_events[vid_id]
            label = 1
        else:
            event_frame = 0
            label = 0

        # clip the event frame to prevent > total_frames
        if event_frame > total_frames:
            event_frame = max(0, total_frames - 1)

        metadata[vid_id] = {
            "event_frame": event_frame,
            "total_frames": total_frames,
            "label": label,
            "fps": fps,
            "source": "ucf_crime"
        }

    # save metadata
    with open(OUT_JSON, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[UCF] Done. {len(metadata)} items saved → {OUT_JSON}")


if __name__ == "__main__":
    main()
