#!/usr/bin/env python3
import os
import pandas as pd

# ============================================
# CONFIG
# ============================================

BASE = os.path.expanduser("~/projects/violence_anticipation/data")
KINETICS_DIR = f"{BASE}/raw_videos/kinetics"

# You can change this to val.csv if needed
ANNOT_CSV = f"{KINETICS_DIR}/train.csv"

OUT_TXT = f"{BASE}/metadata/kinetics_labels.txt"

# ============================================
# MAIN
# ============================================

def main():
    print(f"[KINETICS] Loading CSV: {ANNOT_CSV}")

    if not os.path.exists(ANNOT_CSV):
        raise FileNotFoundError(f"CSV not found: {ANNOT_CSV}")

    df = pd.read_csv(ANNOT_CSV)

    # normalize header names
    df.columns = [c.lower() for c in df.columns]

    if "label" not in df.columns:
        raise ValueError("CSV does not contain a 'label' column.")

    labels = sorted(df["label"].unique())

    print(f"[KINETICS] Found {len(labels)} unique labels.")

    # ensure metadata folder exists
    out_dir = os.path.dirname(OUT_TXT)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # write to text file, one label per line
    with open(OUT_TXT, "w") as f:
        for lbl in labels:
            f.write(f"{lbl}\n")

    print(f"[KINETICS] Saved label list → {OUT_TXT}")


if __name__ == "__main__":
    main()
