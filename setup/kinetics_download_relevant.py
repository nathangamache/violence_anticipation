#!/usr/bin/env python3
import os, json, subprocess, pandas as pd
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

BASE = os.path.expanduser("~/projects/violence_anticipation/data")
KINETICS_DIR = f"{BASE}/raw_videos/kinetics"

USE_TRAIN = True
USE_VAL = True

CSV_FILES = []
if USE_TRAIN:
    CSV_FILES.append(f"{KINETICS_DIR}/train.csv")
if USE_VAL:
    CSV_FILES.append(f"{KINETICS_DIR}/val.csv")

OUT_DIR = f"{KINETICS_DIR}/videos"
META_OUT = f"{BASE}/metadata/kinetics.json"

# Violence-relevant classes
TARGET_CLASSES = {
    "punching person",
    "kicking",
    "hitting",
    "slapping",
    "wrestling",
    "fighting stance",
    "grabbing person",
    "strangling person",
    "pushing person",
    "shooting weapon",
    "brandishing weapon",
    "pointing weapon",
    "aiming a gun",
    "choking person",
    "beating person",
    "threatening with object",
    "throwing object at person",
    "shoving"
}

# ============================================================
# UTILS
# ============================================================

def safe_mkdir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def download_clip(youtube_id, start, end, out_path):
    duration = end - start
    if duration <= 0:
        duration = 3

    cmd = [
        "yt-dlp",
        f"https://www.youtube.com/watch?v={youtube_id}",
        "--quiet",
        "--cookies",
        os.path.expanduser("~/projects/violence_anticipation/youtube_cookies.txt"),
        "--no-warnings",
        "-f", "mp4",
        "--download-sections", f"*{start}-{end}",
        "-o", out_path
    ]

    try:
        subprocess.run(cmd, check=True)
        return os.path.exists(out_path)
    except Exception:
        return False

# ============================================================
# MAIN
# ============================================================

def main():
    safe_mkdir(KINETICS_DIR)
    safe_mkdir(OUT_DIR)

    # Final metadata output
    metadata = {}

    # Load existing metadata if present (resume)
    if os.path.exists(META_OUT):
        with open(META_OUT, "r") as f:
            metadata = json.load(f)

    for csv_path in CSV_FILES:
        print(f"[KINETICS] Loading: {csv_path}")
        df = pd.read_csv(csv_path)

        df.columns = [c.lower() for c in df.columns]

        # Validate CSV header
        for col in ["label", "youtube_id", "time_start", "time_end"]:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in {csv_path}")

        # Filter by violence-related classes
        df = df[df["label"].isin(TARGET_CLASSES)]
        print(f"[KINETICS] {len(df)} relevant clips found in this file.")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
            youtube_id = row["youtube_id"]
            start = float(row["time_start"])
            end = float(row["time_end"])
            label = row["label"]

            clip_id = f"{youtube_id}_{int(start)}_{int(end)}"
            out_file = f"{OUT_DIR}/{clip_id}.mp4"

            # Skip if exists in metadata OR file already present
            if clip_id in metadata and os.path.exists(out_file):
                continue

            ok = download_clip(youtube_id, start, end, out_file)
            if ok:
                metadata[clip_id] = {
                    "youtube_id": youtube_id,
                    "start_time": start,
                    "end_time": end,
                    "label": label,
                    "source": "kinetics700"
                }

    # Save final metadata
    with open(META_OUT, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[KINETICS] Download complete → {META_OUT}")


if __name__ == "__main__":
    main()
