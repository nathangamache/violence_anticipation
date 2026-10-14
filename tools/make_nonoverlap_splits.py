#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

# Try to extract a stable "video/clip id" from an index item.
# Priority:
#   1) explicit id keys if present
#   2) basename of any known path-like key
#   3) fallback: hashable normalized json string (last resort)
ID_KEYS = ("video_id", "vid_id", "id", "uid", "name")
PATH_KEYS = ("npz_path", "emb_path", "jsonl_path", "path", "graph_path", "feat_path", "label_path")

def _basename_no_ext(p: str) -> str:
    base = os.path.basename(p)
    return os.path.splitext(base)[0]

def infer_group_id(item: Dict[str, Any], strip_regex: str | None = None) -> str:
    # 1) direct id keys
    for k in ID_KEYS:
        if k in item and item[k]:
            return str(item[k])

    # 2) common path keys
    for k in PATH_KEYS:
        if k in item and item[k]:
            gid = _basename_no_ext(str(item[k]))
            if strip_regex:
                gid = re.sub(strip_regex, "", gid)
            return gid

    # 3) fallback (should almost never be needed)
    return json.dumps(item, sort_keys=True)

def load_items(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list, got {type(data)}")
    return data

def stable_item_key(item: Dict[str, Any]) -> str:
    # Dedup based on a stable signature.
    # Prefer a path key if present; else fallback to normalized json.
    for k in PATH_KEYS:
        if k in item and item[k]:
            return f"{k}:{item[k]}"
    for k in ID_KEYS:
        if k in item and item[k]:
            return f"{k}:{item[k]}"
    return json.dumps(item, sort_keys=True)

def split_groups(
    group_ids: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[str], List[str], List[str]]:
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "ratios must sum to 1.0"
    n = len(group_ids)
    n_train = int(round(n * train_ratio))
    n_val   = int(round(n * val_ratio))
    # remainder goes to test to avoid off-by-one loss
    n_test  = n - n_train - n_val

    train_g = group_ids[:n_train]
    val_g   = group_ids[n_train:n_train + n_val]
    test_g  = group_ids[n_train + n_val:]
    assert len(test_g) == n_test
    return train_g, val_g, test_g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Input index json files (lists).")
    ap.add_argument("--out_dir", default="splits_fixed", help="Output directory.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train", type=float, default=0.80)
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument(
        "--strip_regex",
        type=str,
        default=None,
        help="Optional regex to strip from basename-derived IDs (use only if you know you need it).",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    # 1) load + union
    all_items: List[Dict[str, Any]] = []
    for p in args.inputs:
        items = load_items(p)
        print(f"[Load] {p}: {len(items)} items")
        all_items.extend(items)

    # 2) dedupe items
    uniq = {}
    for it in all_items:
        uniq[stable_item_key(it)] = it
    all_items = list(uniq.values())
    print(f"[Union] unique items: {len(all_items)}")

    # 3) group by video/clip id
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for it in all_items:
        gid = infer_group_id(it, strip_regex=args.strip_regex)
        groups[gid].append(it)

    group_ids = list(groups.keys())
    random.shuffle(group_ids)
    print(f"[Group] unique group_ids (videos/clips): {len(group_ids)}")

    # 4) split groups (non-overlapping by construction)
    train_g, val_g, test_g = split_groups(group_ids, args.train, args.val, args.test)

    def materialize(gids: List[str]) -> List[Dict[str, Any]]:
        out = []
        for gid in gids:
            out.extend(groups[gid])
        return out

    train_items = materialize(train_g)
    val_items   = materialize(val_g)
    test_items  = materialize(test_g)

    print(f"[Split] train groups={len(train_g)} items={len(train_items)}")
    print(f"[Split] val   groups={len(val_g)} items={len(val_items)}")
    print(f"[Split] test  groups={len(test_g)} items={len(test_items)}")

    # 5) write outputs
    out_train = os.path.join(args.out_dir, "train_index.json")
    out_val   = os.path.join(args.out_dir, "val_index.json")
    out_test  = os.path.join(args.out_dir, "test_index.json")
    out_eval  = os.path.join(args.out_dir, "eval_index.json")  # alias

    for p, items in [(out_train, train_items), (out_val, val_items), (out_test, test_items), (out_eval, test_items)]:
        with open(p, "w") as f:
            json.dump(items, f, indent=2)
        print(f"[Write] {p}: {len(items)} items")

    # 6) sanity check overlaps at group-id level
    train_set = set(train_g)
    val_set   = set(val_g)
    test_set  = set(test_g)
    print("[Check] group overlaps:",
          "train∩val", len(train_set & val_set),
          "train∩test", len(train_set & test_set),
          "val∩test", len(val_set & test_set))

if __name__ == "__main__":
    main()
