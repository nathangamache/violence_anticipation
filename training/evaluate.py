#!/usr/bin/env python3
import os
import json
import argparse
import math

import numpy as np
import yaml
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from training.dataset import GraphSequenceDataset
from core.model import ViolenceTransformer
from core.utils import get_device


def build_index_file(graphs_dir: str, labels_dir: str, out_json: str) -> str:
    """Write an index JSON of {graph_path, label_path} for every graph that has a label."""
    items = []
    for fn in os.listdir(graphs_dir):
        if not (fn.endswith(".npy") or fn.endswith(".pt")):
            continue

        vid_id = os.path.splitext(fn)[0]
        graph_path = os.path.join(graphs_dir, fn)
        label_path = os.path.join(labels_dir, f"{vid_id}.npy")

        if not os.path.exists(label_path):
            print(f"[Index] Skip (missing label): {vid_id}")
            continue

        items.append({"graph_path": graph_path, "label_path": label_path})

    with open(out_json, "w") as f:
        json.dump(items, f, indent=2)

    print(f"[Index] Wrote {len(items)} items → {out_json}")
    return out_json


def infer_dims(dataset: GraphSequenceDataset):
    node_dim = None
    scene_dim = None
    audio_dim = None

    for i in range(len(dataset)):
        frames, _ = dataset[i]
        for f in frames:
            if node_dim is None and hasattr(f["nodes"], "ndim") and f["nodes"].ndim == 2 and f["nodes"].shape[0] > 0:
                node_dim = f["nodes"].shape[1]

            if scene_dim is None and hasattr(f["scene"], "shape") and f["scene"].ndim == 1 and f["scene"].shape[0] > 0:
                scene_dim = f["scene"].shape[0]

            if audio_dim is None and hasattr(f["audio"], "shape") and f["audio"].ndim == 1 and f["audio"].shape[0] > 0:
                audio_dim = f["audio"].shape[0]

        if node_dim is not None and scene_dim is not None and audio_dim is not None:
            break

    if node_dim is None or scene_dim is None or audio_dim is None:
        raise RuntimeError(
            f"Could not infer dims from dataset. "
            f"node_dim={node_dim}, scene_dim={scene_dim}, audio_dim={audio_dim}"
        )

    print(f"[Dataset] Inferred dims: node={node_dim}, scene={scene_dim}, audio={audio_dim}")
    return node_dim, scene_dim, audio_dim


def evaluate_model(
    model: ViolenceTransformer,
    dataset: GraphSequenceDataset,
    device: torch.device,
    bin_thresh: float = 0.5,
    desc: str = "Eval",
):
    """Pooled frame-level MSE and ROC-AUC (targets binarized at bin_thresh)."""
    model.eval()
    all_scores = []
    all_targets = []

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=desc, dynamic_ncols=True):
            frames_cpu, labels_cpu = dataset[idx]

            frames_cuda = []
            for f in frames_cpu:
                nodes = torch.as_tensor(f["nodes"], dtype=torch.float32, device=device)
                edges = torch.as_tensor(f["edges"], dtype=torch.float32, device=device)
                scene = torch.as_tensor(f["scene"], dtype=torch.float32, device=device)
                audio = torch.as_tensor(f["audio"], dtype=torch.float32, device=device)

                frames_cuda.append(
                    {
                        "nodes": nodes,
                        "edges": edges,
                        "scene": scene,
                        "audio": audio,
                    }
                )

            labels = torch.as_tensor(labels_cpu, dtype=torch.float32, device=device)
            if labels.dim() > 1:
                labels = labels.view(-1)

            preds = model(frames_cuda)
            if preds.dim() > 1:
                preds = preds.view(-1)

            T = min(preds.shape[0], labels.shape[0])
            if T == 0:
                continue

            preds = preds[:T].detach().cpu().numpy()
            labels_np = labels[:T].detach().cpu().numpy()

            all_scores.append(preds)
            all_targets.append(labels_np)

    if not all_scores:
        raise RuntimeError("[Eval] No valid sequences produced any predictions.")

    scores = np.concatenate(all_scores, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    mse = float(np.mean((scores - targets) ** 2))

    bin_targets = (targets >= bin_thresh).astype(int)

    if np.unique(bin_targets).size < 2:
        print("[Eval] Only one class present in binarized labels; ROC-AUC undefined.")
        auc = float("nan")
    else:
        auc = float(roc_auc_score(bin_targets, scores))

    return {
        "mse": mse,
        "auc": auc,
        "num_frames": int(scores.shape[0]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to model checkpoint (e.g. checkpoints/model_epoch9.pt)",
    )
    parser.add_argument(
        "--bin_thresh",
        type=float,
        default=0.5,
        help="Threshold for binarizing risk labels when computing ROC-AUC.",
    )
    args = parser.parse_args()

    with open("configs/paths.yaml", "r") as f:
        paths_cfg = yaml.safe_load(f)

    graphs_dir = paths_cfg["paths"]["graphs"]
    labels_dir = paths_cfg["paths"]["labels"]

    if os.path.exists("val_index.json"):
        index_json = "val_index.json"
        print("[Eval] Using existing validation index: val_index.json")
    else:
        print("[Eval] No val_index.json found; building eval_index.json from full dataset...")
        index_json = build_index_file(
            graphs_dir=graphs_dir,
            labels_dir=labels_dir,
            out_json="eval_index.json",
        )

    dataset = GraphSequenceDataset(index_json)
    print(f"[Eval] Dataset size = {len(dataset)} sequences")

    node_dim, scene_dim, audio_dim = infer_dims(dataset)

    device = get_device()

    model = ViolenceTransformer(
        node_dim=node_dim,
        scene_dim=scene_dim,
        audio_dim=audio_dim,
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ).to(device)

    print(f"[Eval] Loading checkpoint: {args.ckpt}")
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)

    metrics = evaluate_model(
        model,
        dataset,
        device=device,
        bin_thresh=args.bin_thresh,
        desc="Eval (CLI)",
    )

    print("\n========== EVAL METRICS ==========")
    print(f"Num frames:           {metrics['num_frames']}")
    print(f"Frame-level MSE:      {metrics['mse']:.6f}")
    if math.isnan(metrics["auc"]):
        print("Frame-level ROC-AUC:  N/A (only one class present)")
    else:
        print(f"Frame-level ROC-AUC:  {metrics['auc']:.4f}")
    print("==================================\n")


if __name__ == "__main__":
    main()
