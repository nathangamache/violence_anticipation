#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from training.dataset import GraphSequenceDataset
from core.model import ViolenceTransformer
from training.train import build_index_file, infer_dims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument("--vid", default=None, help="Specific vid_id to inspect (no extension)")
    args = parser.parse_args()

    BASE = "data"
    GRAPHS = f"{BASE}/processed/graphs"
    LABELS = f"{BASE}/processed/labels"

    # Build / reuse index
    index_path = "train_index.json"
    if not os.path.exists(index_path):
        index_path = build_index_file(GRAPHS, LABELS, "train_index.json")

    dataset = GraphSequenceDataset(index_path)
    node_dim, scene_dim, audio_dim = infer_dims(dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViolenceTransformer(
        node_dim=node_dim,
        scene_dim=scene_dim,
        audio_dim=audio_dim,
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Pick an item
    if args.vid is None:
        idx = 0
        frames, labels = dataset[idx]
        vid_id = os.path.splitext(os.path.basename(dataset.items[idx]["graph_path"]))[0]
    else:
        # search by vid_id in dataset
        vid_id = args.vid
        idx = None
        for i, it in enumerate(dataset.items):
            if os.path.splitext(os.path.basename(it["graph_path"]))[0] == vid_id:
                idx = i
                break
        if idx is None:
            raise RuntimeError(f"vid_id {vid_id} not found in index")
        frames, labels = dataset[idx]

    print(f"[Inspect] vid_id = {vid_id}")
    labels_np = labels.numpy()
    print("GT labels shape:", labels_np.shape)
    print("GT min/max/mean:", labels_np.min(), labels_np.max(), labels_np.mean())
    print("GT any > 0.5?", (labels_np > 0.5).any())

    # Forward pass
    frames_cuda = []
    for f in frames:
        frames_cuda.append(
            {
                "nodes": f["nodes"].to(device),
                "edges": f["edges"].to(device),
                "scene": f["scene"].to(device),
                "audio": f["audio"].to(device),
            }
        )

    with torch.no_grad():
        preds = model(frames_cuda).cpu().numpy()

    T = min(len(preds), len(labels_np))
    preds = preds[:T]
    labels_np = labels_np[:T]

    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(labels_np, label="GT risk", linewidth=2)
    plt.plot(preds, label="Pred risk", alpha=0.7)
    plt.title(f"Risk curve for {vid_id}")
    plt.xlabel("Frame index")
    plt.ylabel("Risk")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
