#!/usr/bin/env python3
import json, os, sys
from collections import defaultdict

def vid_id(item):
    for k in ("video_id","vid_id","id"):
        if k in item: return str(item[k])
    for k in ("npz_path","emb_path","path","graph_path"):
        if k in item:
            base=os.path.splitext(os.path.basename(item[k]))[0]
            return base
    return str(item)

def load(p):
    with open(p,"r") as f:
        data=json.load(f)
    return set(vid_id(x) for x in data)

splits = {
    "train": load("train_index.json"),
    "val": load("val_index.json"),
    "eval": load("eval_index.json"),
}

pairs=[("train","val"),("train","eval"),("val","eval")]
bad=False
for a,b in pairs:
    inter = splits[a] & splits[b]
    if inter:
        bad=True
        print(f"[BAD] overlap {a}∩{b}: {len(inter)} vids. Example:", list(sorted(inter))[:10])
    else:
        print(f"[OK] overlap {a}∩{b}: 0")
sys.exit(1 if bad else 0)
