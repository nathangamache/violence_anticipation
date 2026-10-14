"""Build student-distillation NPZs: graph-frame features as inputs, VLM teacher outputs as targets."""
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple

import numpy as np


def load_index(index_json: str) -> List[Dict[str, Any]]:
    with open(index_json, "r") as f:
        return json.load(f)


def norm_tag(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9\s_-]+", "", s)
    s = re.sub(r"\s+", "_", s)
    s = s.strip("_")
    return s


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def load_graph_frame_feature(graph_path: str, t: int, scene_dim_default=512, audio_dim_default=64) -> np.ndarray:
    """Step-t feature: [num_people, node mean, node max, scene, audio]."""
    seq = np.load(graph_path, allow_pickle=True)
    frames = list(seq) if isinstance(seq, np.ndarray) else seq
    t = max(0, min(int(t), len(frames) - 1))
    fr = frames[t]

    nodes = fr.get("nodes", [])
    if isinstance(nodes, dict):
        nodes = list(nodes.values())
    nodes = np.asarray(nodes, dtype=np.float32) if len(nodes) else np.zeros((0, 1), dtype=np.float32)

    if nodes.ndim == 1:
        nodes = nodes.reshape(1, -1)
    node_dim = int(nodes.shape[1]) if nodes.size else 772

    if nodes.size == 0:
        num_people = 0.0
        node_mean = np.zeros((node_dim,), np.float32)
        node_max = np.zeros((node_dim,), np.float32)
    else:
        num_people = float(nodes.shape[0])
        node_mean = nodes.mean(axis=0).astype(np.float32)
        node_max = nodes.max(axis=0).astype(np.float32)

    scene = fr.get("scene", None)
    if scene is None:
        scene = np.zeros((scene_dim_default,), np.float32)
    else:
        scene = np.asarray(scene, dtype=np.float32).flatten()

    audio = fr.get("audio", None)
    if audio is None:
        audio = np.zeros((audio_dim_default,), np.float32)
    else:
        audio = np.asarray(audio, dtype=np.float32).flatten()

    feat = np.concatenate([
        np.asarray([num_people], np.float32),
        node_mean,
        node_max,
        scene,
        audio,
    ], axis=0)

    return feat


def build_vocab(records: List[Dict[str, Any]], key: str, min_count: int, max_size: int) -> Dict[str, int]:
    c = Counter()
    for r in records:
        vals = r.get(key, [])
        if not isinstance(vals, list):
            continue
        for v in vals:
            t = norm_tag(str(v))
            if t:
                c[t] += 1
    items = [t for t, n in c.items() if n >= min_count]
    items.sort(key=lambda x: c[x], reverse=True)
    items = items[:max_size]
    return {t: i for i, t in enumerate(items)}


def encode_multihot(vals: List[str], vocab: Dict[str, int]) -> np.ndarray:
    y = np.zeros((len(vocab),), np.float32)
    if not isinstance(vals, list):
        return y
    for v in vals:
        t = norm_tag(str(v))
        if t in vocab:
            y[vocab[t]] = 1.0
    return y


def violent_to_float(v: Any) -> float:
    if isinstance(v, dict):
        lab = str(v.get("label", "uncertain")).lower()
        score = float(v.get("score", 0.5))
    else:
        lab = str(v).lower()
        score = 0.5

    if "yes" in lab:
        return max(score, 0.5)
    if "no" in lab:
        return min(score, 0.5)
    return 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_index", type=str, required=True)
    ap.add_argument("--val_index", type=str, required=True)
    ap.add_argument("--graphs_dir", type=str, required=True)
    ap.add_argument("--vlm_dir", type=str, default="data/processed/vlm_labels")
    ap.add_argument("--out_dir", type=str, default="data/processed/student")
    ap.add_argument("--min_count", type=int, default=10)
    ap.add_argument("--max_actions", type=int, default=200)
    ap.add_argument("--max_weapons", type=int, default=50)
    ap.add_argument("--max_scenes", type=int, default=200)
    ap.add_argument("--max_people", type=int, default=10, help="people_count will be clipped into [0..max_people]")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_items = load_index(args.train_index)
    val_items = load_index(args.val_index)

    def collect_records(index_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        recs = []
        for it in index_items:
            vid_id = os.path.splitext(os.path.basename(it["graph_path"]))[0]
            found = None
            for src in ["kinetics", "ucf_crime", "xd_violence", "unknown"]:
                p = os.path.join(args.vlm_dir, src, f"{vid_id}.jsonl")
                if os.path.exists(p):
                    found = p
                    break
            if not found:
                continue
            recs.extend(read_jsonl(found))
        return recs

    train_recs = collect_records(train_items)
    val_recs = collect_records(val_items)

    all_recs = train_recs + val_recs
    if not all_recs:
        raise RuntimeError("No VLM records found. Did you run vlm_label_dataset.py?")

    actions_vocab = build_vocab(all_recs, "actions", args.min_count, args.max_actions)
    weapons_vocab = build_vocab(all_recs, "weapons", args.min_count, args.max_weapons)
    scenes_vocab = build_vocab(all_recs, "scene", args.min_count, args.max_scenes)

    vocabs_path = os.path.join(args.out_dir, "vocabs.json")
    with open(vocabs_path, "w") as f:
        json.dump(
            {"actions": actions_vocab, "weapons": weapons_vocab, "scene": scenes_vocab},
            f, indent=2
        )
    print(f"[Vocab] Saved {vocabs_path}")
    print(f"[Vocab] actions={len(actions_vocab)} weapons={len(weapons_vocab)} scenes={len(scenes_vocab)}")

    graph_map = {}
    for it in train_items + val_items:
        vid_id = os.path.splitext(os.path.basename(it["graph_path"]))[0]
        graph_map[vid_id] = it["graph_path"]

    def build_npz(split_name: str, recs: List[Dict[str, Any]]):
        X = []
        y_violent = []
        y_people = []
        y_actions = []
        y_weapons = []
        y_scenes = []
        meta = []

        for r in recs:
            vid_id = r.get("vid_id")
            if not vid_id or vid_id not in graph_map:
                continue

            graph_path = graph_map[vid_id]
            if not os.path.exists(graph_path):
                gp = os.path.join(args.graphs_dir, f"{vid_id}.npy")
                if os.path.exists(gp):
                    graph_path = gp
                else:
                    continue

            if r.get("error") == "parse_failed":
                continue

            t = int(r.get("t", 0))

            try:
                feat = load_graph_frame_feature(graph_path, t)
            except Exception:
                continue

            X.append(feat)
            y_violent.append(violent_to_float(r.get("violent", {"label": "uncertain", "score": 0.5})))

            pc = r.get("people_count", 0)
            try:
                pc = int(pc)
            except Exception:
                pc = 0
            pc = max(0, min(pc, args.max_people))
            y_people.append(pc)

            y_actions.append(encode_multihot(r.get("actions", []), actions_vocab))
            y_weapons.append(encode_multihot(r.get("weapons", []), weapons_vocab))
            y_scenes.append(encode_multihot(r.get("scene", []), scenes_vocab))

            meta.append({"vid_id": vid_id, "t": t})

        if not X:
            raise RuntimeError(f"No samples for split={split_name}")

        X = np.stack(X, axis=0).astype(np.float32)
        y_violent = np.asarray(y_violent, np.float32)
        y_people = np.asarray(y_people, np.int64)
        y_actions = np.stack(y_actions, axis=0).astype(np.float32)
        y_weapons = np.stack(y_weapons, axis=0).astype(np.float32)
        y_scenes = np.stack(y_scenes, axis=0).astype(np.float32)

        out_npz = os.path.join(args.out_dir, f"{split_name}_student.npz")
        np.savez_compressed(
            out_npz,
            X=X,
            y_violent=y_violent,
            y_people=y_people,
            y_actions=y_actions,
            y_weapons=y_weapons,
            y_scenes=y_scenes,
            meta=json.dumps(meta),
        )
        print(f"[NPZ] {split_name}: X={X.shape} saved {out_npz}")

    build_npz("train", train_recs)
    build_npz("val", val_recs)


if __name__ == "__main__":
    main()
