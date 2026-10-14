#!/usr/bin/env python3
"""Write all Kinetics class names from train.csv to metadata/kinetics_labels.txt."""
import os
import pandas as pd

BASE = os.path.expanduser("~/projects/violence_anticipation/data")
KINETICS_DIR = f"{BASE}/raw_videos/kinetics"

ANNOT_CSV = f"{KINETICS_DIR}/train.csv"

OUT_TXT = f"{BASE}/metadata/kinetics_labels.txt"


def main():
    print(f"[KINETICS] Loading CSV: {ANNOT_CSV}")

    if not os.path.exists(ANNOT_CSV):
        raise FileNotFoundError(f"CSV not found: {ANNOT_CSV}")

    df = pd.read_csv(ANNOT_CSV)

    df.columns = [c.lower() for c in df.columns]

    if "label" not in df.columns:
        raise ValueError("CSV does not contain a 'label' column.")

    labels = sorted(df["label"].unique())

    print(f"[KINETICS] Found {len(labels)} unique labels.")

    out_dir = os.path.dirname(OUT_TXT)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(OUT_TXT, "w") as f:
        for lbl in labels:
            f.write(f"{lbl}\n")

    print(f"[KINETICS] Saved label list → {OUT_TXT}")


if __name__ == "__main__":
    main()
