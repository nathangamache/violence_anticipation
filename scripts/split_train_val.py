#!/usr/bin/env python3
import json
import random
import os

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
IN_PATH = "train_index.json"      # existing full index
TRAIN_OUT = "train_index.json"    # will be overwritten with 80%
VAL_OUT = "val_index.json"        # new file with 20%

TRAIN_FRAC = 0.8
SEED = 42                         # for reproducibility


def main():
    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(f"Could not find {IN_PATH}")

    with open(IN_PATH, "r") as f:
        items = json.load(f)

    if not isinstance(items, list):
        raise ValueError(f"{IN_PATH} must contain a JSON list, got {type(items)}")

    n = len(items)
    if n == 0:
        raise ValueError("Input index is empty, nothing to split.")

    rng = random.Random(SEED)
    rng.shuffle(items)

    n_train = int(round(TRAIN_FRAC * n))
    train_items = items[:n_train]
    val_items = items[n_train:]

    with open(TRAIN_OUT, "w") as f:
        json.dump(train_items, f, indent=2)

    with open(VAL_OUT, "w") as f:
        json.dump(val_items, f, indent=2)

    print(f"[Split] Total: {n}")
    print(f"[Split] Train: {len(train_items)} → {TRAIN_OUT}")
    print(f"[Split] Val:   {len(val_items)} → {VAL_OUT}")


if __name__ == "__main__":
    main()
