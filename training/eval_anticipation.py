#!/usr/bin/env python3
"""
Anticipation-focused evaluation for ViolenceTransformer.

Computes:
- Global frame-level MSE + ROC-AUC (using PROBABILITIES)
- Time-To-Detection (TTD) for violent clips (approximate, using metadata)
- False-alarm duration for non-violent clips
- Per-class frame-level ROC-AUC
- Score percentile diagnostics (p50/p75/p90/p95/p99) for negative vs positive frames

Key contract (NO MORE DOUBLE SIGMOID):
- Model outputs LOGITS (we explicitly call return_logits=True)
- Eval applies sigmoid ONCE to get probabilities in [0, 1]
- Thresholds are interpreted in PROBABILITY space
"""

import os
import re
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from training.dataset import GraphSequenceDataset
from core.model import ViolenceTransformer
from core.utils import get_device


# ---------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------
def load_metadata(meta_dir: str):
    meta = {}
    for fname in ["xd_violence.json", "ucf_crime.json", "kinetics.json"]:
        path = os.path.join(meta_dir, fname)
        if os.path.exists(path):
            with open(path, "r") as f:
                meta.update(json.load(f))
    if not meta:
        print(f"[Warn] No metadata files found under {meta_dir}")
    return meta


def infer_class_name(vid_id: str, meta_info: dict):
    source = meta_info.get("source") if meta_info else None

    if source == "ucf_crime":
        base = vid_id.split("_")[0]
        cls = re.sub(r"\d+", "", base)
        return cls or "ucf_crime"
    elif source in ("xd_violence", "kinetics"):
        return source
    else:
        return source or "unknown"


# ---------------------------------------------------------------------
# Dim inference (from dataset, NOT raw graphs)
# ---------------------------------------------------------------------
def infer_dims_from_dataset(dataset: GraphSequenceDataset):
    node_dim = None
    scene_dim = None
    audio_dim = None

    for i in range(len(dataset)):
        frames, _labels = dataset[i]
        for f in frames:
            nodes = f.get("nodes", None)
            scene = f.get("scene", None)
            audio = f.get("audio", None)

            if node_dim is None and isinstance(nodes, torch.Tensor) and nodes.ndim == 2 and nodes.shape[0] > 0:
                node_dim = nodes.shape[1]

            if scene_dim is None and isinstance(scene, torch.Tensor) and scene.ndim == 1 and scene.shape[0] > 0:
                scene_dim = scene.shape[0]

            if audio_dim is None and isinstance(audio, torch.Tensor) and audio.ndim == 1 and audio.shape[0] > 0:
                audio_dim = audio.shape[0]

        if node_dim is not None and scene_dim is not None and audio_dim is not None:
            break

    if node_dim is None or scene_dim is None or audio_dim is None:
        raise RuntimeError(
            f"Could not infer dims from dataset. node_dim={node_dim}, scene_dim={scene_dim}, audio_dim={audio_dim}"
        )

    print(f"[Dims] node_dim={node_dim}, scene_dim={scene_dim}, audio_dim={audio_dim}")
    return node_dim, scene_dim, audio_dim


# ---------------------------------------------------------------------
# Score percentile diagnostics
# ---------------------------------------------------------------------
def _safe_percentiles(x: np.ndarray, ps=(50, 75, 90, 95, 99)):
    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return {p: float(np.percentile(x, p)) for p in ps}


def _print_pct_block(title: str, scores: np.ndarray):
    pct = _safe_percentiles(scores)
    n = int(np.isfinite(scores).sum())
    if pct is None:
        print(f"[Diag] {title}: n={n} (no finite samples)")
        return

    print(f"[Diag] {title}: n={n}")
    print(
        "       "
        f"p50={pct[50]:.3f}  p75={pct[75]:.3f}  p90={pct[90]:.3f}  p95={pct[95]:.3f}  p99={pct[99]:.3f}"
    )


def print_frame_score_percentiles(scores: np.ndarray, targets: np.ndarray, bin_thresh: float):
    scores = np.asarray(scores, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)

    neg_scores = scores[targets < bin_thresh]
    pos_scores = scores[targets >= bin_thresh]

    print("\n========== SCORE PERCENTILE DIAGNOSTICS (FRAME-LEVEL) ==========")
    print(f"[Diag] GT bin_thresh = {bin_thresh}")
    _print_pct_block("NEGATIVE FRAMES (GT < thresh)", neg_scores)
    _print_pct_block("POSITIVE FRAMES (GT >= thresh)", pos_scores)
    print("================================================================\n")


# ---------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------
def run_anticipation_eval(
    ckpt_path: str,
    index_path: str,
    data_root: str = "data",
    thresholds=(0.5,),
    device_str: str | None = None,
    max_videos: int | None = None,
    bin_thresh: float = 0.5,
):
    with open(index_path, "r") as f:
        index_items = json.load(f)

    if max_videos is not None:
        index_items = index_items[:max_videos]

    print(f"[Index] Loaded {len(index_items)} items from {index_path}")

    dataset = GraphSequenceDataset(index_path)
    print(f"[Dataset] Size = {len(dataset)} sequences")

    meta_dir = os.path.join(data_root, "metadata")
    metadata = load_metadata(meta_dir)

    node_dim, scene_dim, audio_dim = infer_dims_from_dataset(dataset)

    device = get_device() if device_str is None else torch.device(device_str)
    print(f"[Device] {device}")

    model = ViolenceTransformer(
        node_dim=node_dim,
        scene_dim=scene_dim,
        audio_dim=audio_dim,
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ).to(device)

    print(f"[Model] Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    thresholds = [float(t) for t in thresholds]

    all_scores = []   # probs
    all_targets = []

    stats = {
        theta: {
            "ttd_secs": [],
            "pos_clips": 0,
            "pos_hits": 0,
            "false_dur_secs": [],
            "neg_clips": 0,
        }
        for theta in thresholds
    }

    per_class_scores = defaultdict(list)   # probs
    per_class_targets = defaultdict(list)  # 0/1

    printed_predmode = False

    num_videos_to_eval = min(len(dataset), len(index_items))
    for idx in tqdm(range(num_videos_to_eval), desc="Videos", dynamic_ncols=True):
        item = index_items[idx]
        graph_path = item["graph_path"]
        vid_id = os.path.splitext(os.path.basename(graph_path))[0]

        frames_cpu, labels_cpu = dataset[idx]
        if not frames_cpu:
            print(f"[Warn] Empty frames for {vid_id}, skipping.")
            continue

        frames_cuda = []
        for f in frames_cpu:
            frames_cuda.append(
                {
                    "nodes": f["nodes"].to(device),
                    "edges": f["edges"].to(device),
                    "scene": f["scene"].to(device),
                    "audio": f["audio"].to(device),
                }
            )

        labels = torch.as_tensor(labels_cpu, dtype=torch.float32, device=device)
        if labels.dim() > 1:
            labels = labels.view(-1)

        with torch.no_grad():
            logits = model(frames_cuda, return_logits=True).view(-1)   # [T]
            probs = torch.sigmoid(logits)                              # [T] in (0,1)

        T = min(probs.shape[0], labels.shape[0])
        if T == 0:
            continue

        logits_np = logits[:T].detach().cpu().numpy()
        probs_np = probs[:T].detach().cpu().numpy()
        labels_np = labels[:T].detach().cpu().numpy()

        if not printed_predmode:
            printed_predmode = True
            print("[PredMode] Using logits from model -> applying sigmoid ONCE for evaluation.")
            print(f"[PredMode] logit range: min={float(np.min(logits_np)):.4f} max={float(np.max(logits_np)):.4f}")
            print(f"[PredMode] prob  range: min={float(np.min(probs_np)):.4f} max={float(np.max(probs_np)):.4f}")

        all_scores.append(probs_np)
        all_targets.append(labels_np)

        bin_labels = (labels_np >= bin_thresh).astype(np.int32)

        info = metadata.get(vid_id, {})
        cls_name = infer_class_name(vid_id, info)
        per_class_scores[cls_name].append(probs_np)
        per_class_targets[cls_name].append(bin_labels)

        clip_label = int(info.get("label", 0)) if info else int(bin_labels.max() > 0)
        fps = float(info.get("fps", 25.0))
        total_frames = int(info.get("total_frames", T))
        event_frame = info.get("event_frame", None)

        if total_frames <= 0 or fps <= 0:
            dt_sec = 1.0 / 25.0
        else:
            total_time_sec = total_frames / fps
            dt_sec = total_time_sec / T

        if clip_label == 1 and event_frame is not None and event_frame > 0 and total_frames > 0:
            event_proc_float = (float(event_frame) / float(total_frames)) * T
            event_proc_idx = int(round(event_proc_float))
            event_proc_idx = max(0, min(T - 1, event_proc_idx))

            for theta in thresholds:
                s = stats[theta]
                s["pos_clips"] += 1

                pre_mask = np.arange(T) <= event_proc_idx
                probs_pre = probs_np[pre_mask]

                alarm_idxs = np.where(probs_pre >= theta)[0]
                if alarm_idxs.size == 0:
                    continue

                first_i = int(alarm_idxs[0])
                t_event = event_proc_idx * dt_sec
                t_alarm = first_i * dt_sec
                ttd = max(0.0, t_event - t_alarm)

                s["ttd_secs"].append(ttd)
                s["pos_hits"] += 1

        elif clip_label == 0:
            for theta in thresholds:
                s = stats[theta]
                s["neg_clips"] += 1
                alarm_mask = probs_np >= theta
                false_frames = int(alarm_mask.sum())
                false_sec = false_frames * dt_sec
                s["false_dur_secs"].append(false_sec)

    if not all_scores:
        raise RuntimeError("[Eval] No valid sequences produced any predictions.")

    scores = np.concatenate(all_scores, axis=0)   # probs in [0,1]
    targets = np.concatenate(all_targets, axis=0)

    mse = float(np.mean((scores - targets) ** 2))
    bin_targets_global = (targets >= bin_thresh).astype(int)

    if np.unique(bin_targets_global).size < 2:
        auc_global = float("nan")
    else:
        auc_global = float(roc_auc_score(bin_targets_global, scores))

    print("\n========== GLOBAL FRAME-LEVEL METRICS ==========")
    print(f"Num frames:           {scores.shape[0]}")
    print(f"Frame-level MSE:      {mse:.6f}")
    if np.isnan(auc_global):
        print("Frame-level ROC-AUC:  N/A (only one class present)")
    else:
        print(f"Frame-level ROC-AUC:  {auc_global:.4f}")
    print("================================================\n")

    print_frame_score_percentiles(scores=scores, targets=targets, bin_thresh=bin_thresh)

    print("========== ANTICIPATION METRICS BY THRESHOLD ==========")
    for theta in thresholds:
        s = stats[theta]
        pos_clips = s["pos_clips"]
        neg_clips = s["neg_clips"]

        avg_ttd = float(np.mean(s["ttd_secs"])) if s["ttd_secs"] else 0.0
        hit_rate = (s["pos_hits"] / pos_clips) if pos_clips > 0 else 0.0
        avg_false = float(np.mean(s["false_dur_secs"])) if s["false_dur_secs"] else 0.0

        print(f"\nThreshold θ = {theta}")
        print(f"  Positive clips        : {pos_clips}")
        print(f"  Early-detected clips  : {s['pos_hits']} ({hit_rate*100:.1f}%)")
        print(f"  Avg TTD (seconds)     : {avg_ttd:.3f}")
        print(f"  Negative clips        : {neg_clips}")
        print(f"  Avg false alarm time  : {avg_false:.3f} s per negative clip")
    print("=======================================================\n")

    print("========== PER-CLASS FRAME-LEVEL ROC-AUC ==========")
    for cls, cls_scores_list in per_class_scores.items():
        cls_scores = np.concatenate(cls_scores_list, axis=0)
        cls_targets = np.concatenate(per_class_targets[cls], axis=0)

        if np.unique(cls_targets).size < 2:
            print(f"  {cls:20s} : N/A (only one class present)")
            continue

        auc_cls = float(roc_auc_score(cls_targets, cls_scores))
        print(f"  {cls:20s} : AUC = {auc_cls:.4f}  (frames={cls_scores.shape[0]})")
    print("===================================================\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--index", type=str, default="val_index.json")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--thresholds", type=str, default="0.5")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--bin_thresh", type=float, default=0.5)
    parser.add_argument("--max_videos", type=int, default=None)
    args = parser.parse_args()

    thresholds = [float(s) for s in args.thresholds.split(",") if s.strip()]

    run_anticipation_eval(
        ckpt_path=args.ckpt,
        index_path=args.index,
        data_root=args.data_root,
        thresholds=thresholds,
        device_str=args.device,
        max_videos=args.max_videos,
        bin_thresh=args.bin_thresh,
    )


if __name__ == "__main__":
    main()
