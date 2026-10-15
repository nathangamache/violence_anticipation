#!/usr/bin/env python3
import os
import json
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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

            if scene_dim is None and hasattr(f["scene"], "numel") and f["scene"].numel() > 0:
                scene_dim = f["scene"].numel()

            if audio_dim is None and hasattr(f["audio"], "numel") and f["audio"].numel() > 0:
                audio_dim = f["audio"].numel()

        if node_dim is not None and scene_dim is not None and audio_dim is not None:
            break

    if node_dim is None or scene_dim is None or audio_dim is None:
        raise RuntimeError(
            f"Could not infer dims from dataset. node_dim={node_dim}, scene_dim={scene_dim}, audio_dim={audio_dim}"
        )

    print(f"[Dims] node_dim={node_dim}, scene_dim={scene_dim}, audio_dim={audio_dim}")
    return node_dim, scene_dim, audio_dim


def train():
    with open("configs/training.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    with open("configs/paths.yaml", "r") as f:
        paths_cfg = yaml.safe_load(f)

    training_cfg = cfg["training"]

    graphs_dir = paths_cfg["paths"]["graphs"]
    labels_dir = paths_cfg["paths"]["labels"]

    anticipation_window_sec = float(training_cfg.get("anticipation_window_sec", 30.0))
    anticipation_weight_factor = float(training_cfg.get("anticipation_weight_factor", 1.0))

    spam_penalty_lambda = float(training_cfg.get("spam_penalty_lambda", 0.0))
    spam_penalty_margin = float(training_cfg.get("spam_penalty_margin", 0.25))
    low_target_eps = float(training_cfg.get("low_target_eps", 1e-3))

    print(f"[Train] Anticipation window (labels) = {anticipation_window_sec:.1f}s")
    print(f"[Train] Anticipation weight factor   = {anticipation_weight_factor:.2f}")
    print(f"[Train] Spam penalty lambda         = {spam_penalty_lambda:.4f}")
    print(f"[Train] Spam penalty margin         = {spam_penalty_margin:.3f}")
    print(f"[Train] Low-target eps              = {low_target_eps:.6f}")

    index_json = build_index_file(
        graphs_dir=graphs_dir,
        labels_dir=labels_dir,
        out_json="train_index.json",
    )

    dataset = GraphSequenceDataset(index_json)
    print(f"[Train] Dataset size = {len(dataset)} sequences")

    node_dim, scene_dim, audio_dim = infer_dims(dataset)

    batch_size = int(training_cfg.get("batch_size", 1))
    if batch_size != 1:
        print("[Warn] batch_size>1 not supported with variable-length sequences. Forcing batch_size=1.")
        batch_size = 1

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(training_cfg.get("num_workers", 0)),
        collate_fn=lambda batch: batch,
    )

    device = get_device()
    print(f"[Train] Device = {device}")

    model = ViolenceTransformer(
        node_dim=node_dim,
        scene_dim=scene_dim,
        audio_dim=audio_dim,
        hidden_dim=int(training_cfg.get("hidden_dim", 256)),
        num_heads=int(training_cfg.get("num_heads", 4)),
        num_layers=int(training_cfg.get("num_layers", 2)),
        dropout=float(training_cfg.get("dropout", 0.1)),
    ).to(device)

    lr = float(training_cfg.get("lr", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))
    num_epochs = int(training_cfg.get("epochs", 20))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_spam = 0.0
        num_batches = 0

        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
            total=len(dataloader),
            dynamic_ncols=True,
        )

        for batch in progress:
            frames_cpu, labels_cpu = batch[0]

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

            labels = labels_cpu.to(device)
            if labels.dim() > 1:
                labels = labels.view(-1)

            optimizer.zero_grad()

            logits = model(frames_cuda, return_logits=True)
            preds = torch.sigmoid(logits)
            if preds.dim() > 1:
                preds = preds.view(-1)

            T = min(preds.shape[0], labels.shape[0])
            if T == 0:
                continue

            preds = preds[:T]
            labels_aligned = labels[:T]

            weights = 1.0 + anticipation_weight_factor * labels_aligned
            mse_loss = ((preds - labels_aligned) ** 2 * weights).mean()

            # "Spam" = false alarms: penalize preds above the margin on frames with GT risk ~0.
            if spam_penalty_lambda > 0.0:
                low_mask = (labels_aligned <= low_target_eps)
                if low_mask.any():
                    over = torch.relu(preds[low_mask] - spam_penalty_margin)
                    spam_loss = (over ** 2).mean()
                else:
                    spam_loss = torch.zeros((), device=device)
            else:
                spam_loss = torch.zeros((), device=device)

            loss = mse_loss + spam_penalty_lambda * spam_loss

            loss.backward()
            optimizer.step()

            loss_val = float(loss.item())
            epoch_loss += loss_val
            epoch_mse += float(mse_loss.item())
            epoch_spam += float(spam_loss.item())
            num_batches += 1

            progress.set_postfix(
                {
                    "loss": f"{loss_val:.4f}",
                    "mse": f"{float(mse_loss.item()):.4f}",
                    "spam": f"{float(spam_loss.item()):.4f}",
                }
            )

        denom = max(1, num_batches)
        print(f"[Epoch {epoch+1}] Avg Train Loss = {epoch_loss/denom:.6f}")
        print(f"[Epoch {epoch+1}] Avg MSE Loss   = {epoch_mse/denom:.6f}")
        print(f"[Epoch {epoch+1}] Avg Spam Loss  = {epoch_spam/denom:.6f} (unweighted)")

        ckpt_path = os.path.join("checkpoints", f"model_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"[Epoch {epoch+1}] Saved checkpoint → {ckpt_path}")


if __name__ == "__main__":
    train()
