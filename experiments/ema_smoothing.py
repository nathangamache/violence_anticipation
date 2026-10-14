"""
ema_smoothing.py
================
EMA (Exponential Moving Average) smoothing at three levels:
  1. Post-processing: smooth inference scores before thresholding
  2. Embedding-level: smooth VLM embeddings across time before feeding the model
  3. Hidden-state-level: smooth LSTM hidden states during inference (requires model surgery)

Usage:
    python ema_smoothing.py \
        --embeddings data/processed/vlm_emb \
        --labels data/processed/labels_multih \
        --checkpoint outputs/outputs_temporal/best_temporal_model.pt \
        --alphas 0.1 0.3 0.5 0.7 0.9

Outputs a CSV comparing raw vs. EMA-smoothed AUC at each alpha for all three levels.
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# EMA utilities
# ──────────────────────────────────────────────

def ema_1d(values: np.ndarray, alpha: float) -> np.ndarray:
    """EMA over a 1-D array (e.g., scalar scores over time)."""
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def ema_2d(matrix: np.ndarray, alpha: float) -> np.ndarray:
    """EMA along axis=0 for a 2-D array (T, D) — e.g., embeddings or hidden states."""
    out = np.empty_like(matrix, dtype=np.float32)
    out[0] = matrix[0]
    for i in range(1, len(matrix)):
        out[i] = alpha * matrix[i] + (1 - alpha) * out[i - 1]
    return out


# ──────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────

def load_video_pairs(vlm_dir: Path, labels_dir: Path, max_videos: int = 200):
    """Load (embeddings, labels) pairs from XD-Violence processed files."""
    vlm_files = sorted(vlm_dir.rglob("*.npz"))[:max_videos]
    pairs = []
    for vf in vlm_files:
        vid = vf.stem
        label_path = labels_dir / f"{vid}.npy"
        if not label_path.exists():
            label_path = labels_dir / "xd_violence" / f"{vid}.npy"
        if not label_path.exists():
            continue
        try:
            emb = np.load(vf)["emb"].astype(np.float32)
            lbl = np.load(label_path)
            if lbl.ndim > 1:
                lbl = lbl.max(axis=1)
            n = min(len(emb), len(lbl))
            pairs.append((emb[:n], (lbl[:n] > 0).astype(int)))
        except Exception:
            continue
    return pairs


def flatten_pairs(pairs):
    X = np.vstack([p[0] for p in pairs])
    y = np.concatenate([p[1] for p in pairs])
    return X, y


# ──────────────────────────────────────────────
# Level 1 – Post-processing EMA on scores
# ──────────────────────────────────────────────

def eval_score_ema(pairs, clf, alphas):
    """
    Train a logistic regression on raw embeddings, generate per-frame scores,
    then evaluate AUC both raw and after EMA smoothing at each alpha.
    """
    X, y = flatten_pairs(pairs)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # Split by video index to avoid leakage
    n_val = max(1, len(pairs) // 5)
    train_pairs = pairs[:-n_val]
    val_pairs = pairs[-n_val:]

    X_train, y_train = flatten_pairs(train_pairs)
    X_train_s = scaler.transform(X_train)
    clf.fit(X_train_s, y_train)

    results = {}
    raw_aucs = []
    for alpha in alphas:
        smoothed_aucs = []
        for emb, lbl in val_pairs:
            scores_raw = clf.predict_proba(scaler.transform(emb))[:, 1]
            scores_ema = ema_1d(scores_raw, alpha)
            if len(np.unique(lbl)) > 1:
                smoothed_aucs.append(roc_auc_score(lbl, scores_ema))

        results[alpha] = np.mean(smoothed_aucs) if smoothed_aucs else float("nan")

    # Raw baseline
    raw_aucs = []
    for emb, lbl in val_pairs:
        scores_raw = clf.predict_proba(scaler.transform(emb))[:, 1]
        if len(np.unique(lbl)) > 1:
            raw_aucs.append(roc_auc_score(lbl, scores_raw))
    results["raw"] = np.mean(raw_aucs) if raw_aucs else float("nan")

    return results


# ──────────────────────────────────────────────
# Level 2 – Embedding-level EMA before classification
# ──────────────────────────────────────────────

def eval_embedding_ema(pairs, alphas):
    """
    Apply EMA along the time axis to smooth embeddings before training/inference.
    Uses logistic regression as the downstream classifier.
    """
    n_val = max(1, len(pairs) // 5)
    train_pairs = pairs[:-n_val]
    val_pairs = pairs[-n_val:]

    results = {}

    for alpha in alphas:
        # Smooth training embeddings
        smoothed_train = [(ema_2d(emb, alpha), lbl) for emb, lbl in train_pairs]
        smoothed_val = [(ema_2d(emb, alpha), lbl) for emb, lbl in val_pairs]

        X_train, y_train = flatten_pairs(smoothed_train)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")
        clf.fit(X_train_s, y_train)

        aucs = []
        for emb_s, lbl in smoothed_val:
            scores = clf.predict_proba(scaler.transform(emb_s))[:, 1]
            if len(np.unique(lbl)) > 1:
                aucs.append(roc_auc_score(lbl, scores))
        results[alpha] = np.mean(aucs) if aucs else float("nan")

    # Raw baseline
    X_train_raw, y_train_raw = flatten_pairs(train_pairs)
    scaler_raw = StandardScaler()
    X_train_raw_s = scaler_raw.fit_transform(X_train_raw)
    clf_raw = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")
    clf_raw.fit(X_train_raw_s, y_train_raw)
    aucs_raw = []
    for emb, lbl in val_pairs:
        scores = clf_raw.predict_proba(scaler_raw.transform(emb))[:, 1]
        if len(np.unique(lbl)) > 1:
            aucs_raw.append(roc_auc_score(lbl, scores))
    results["raw"] = np.mean(aucs_raw) if aucs_raw else float("nan")

    return results


# ──────────────────────────────────────────────
# Level 3 – LSTM hidden-state EMA (model surgery)
# ──────────────────────────────────────────────

class EMAHookLSTM(nn.Module):
    """
    Wraps a trained LSTM model and applies EMA smoothing to hidden states
    output at each timestep before passing to the classifier head.
    This is done by hooking into the LSTM forward pass.
    """

    def __init__(self, base_model: nn.Module, alpha: float = 0.5):
        super().__init__()
        self.base_model = base_model
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        Applies EMA across timesteps on LSTM output before classifier.
        """
        # We need access to the LSTM and classifier separately.
        # Try common attribute names used in this project.
        model = self.base_model

        # Try to find the LSTM and classifier submodules
        lstm_module = getattr(model, "lstm", None)
        classifier = getattr(model, "classifier", None)
        feat_extractor = getattr(model, "feature_extractor", None)

        if lstm_module is None or classifier is None:
            # Fall back: just run the model normally
            return model(x)

        B, T, D = x.shape
        if feat_extractor is not None:
            x_flat = x.reshape(B * T, D)
            feats = feat_extractor(x_flat).reshape(B, T, -1)
        else:
            feats = x

        lstm_out, _ = lstm_module(feats)  # (B, T, hidden)

        # Apply EMA across time dimension on hidden states
        alpha = self.alpha
        ema_out = torch.zeros_like(lstm_out)
        ema_out[:, 0, :] = lstm_out[:, 0, :]
        for t in range(1, T):
            ema_out[:, t, :] = alpha * lstm_out[:, t, :] + (1 - alpha) * ema_out[:, t - 1, :]

        # Use last timestep
        last = ema_out[:, -1, :]

        # Handle attention pooling if present
        attention = getattr(model, "attention", None)
        if attention is not None:
            try:
                attn_weights = torch.softmax(attention(ema_out), dim=1)
                last = (ema_out * attn_weights).sum(dim=1)
            except Exception:
                pass  # Fall through to using last timestep

        return classifier(last)


def eval_hidden_state_ema(checkpoint_path: Path, pairs, alphas, device="cpu"):
    """
    Load the checkpoint, wrap with EMAHookLSTM, run inference on val pairs.
    """
    if not checkpoint_path.exists():
        print(f"  [SKIP] Checkpoint not found: {checkpoint_path}")
        return {}

    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
        base_model = ckpt.get("model", ckpt)
        if isinstance(base_model, dict):
            print("  [SKIP] Checkpoint is a state_dict without model class — cannot wrap.")
            return {}
        base_model.eval().to(device)
    except Exception as e:
        print(f"  [SKIP] Could not load checkpoint: {e}")
        return {}

    n_val = max(1, len(pairs) // 5)
    val_pairs = pairs[-n_val:]
    seq_len = 30

    results = {}

    for alpha in [None] + list(alphas):  # None = raw
        aucs = []
        model = EMAHookLSTM(base_model, alpha=alpha if alpha is not None else 1.0)
        model.eval()

        with torch.no_grad():
            for emb, lbl in val_pairs:
                if len(emb) < seq_len:
                    continue
                scores = []
                for i in range(seq_len, len(emb)):
                    seq = torch.tensor(
                        emb[i - seq_len:i], dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    out = model(seq)
                    if out.shape[-1] == 1:
                        scores.append(torch.sigmoid(out).item())
                    else:
                        scores.append(torch.softmax(out, dim=-1)[0, 1].item())
                scores = np.array(scores)
                lbl_seq = lbl[seq_len:]
                if len(np.unique(lbl_seq)) > 1:
                    aucs.append(roc_auc_score(lbl_seq, scores))

        key = "raw" if alpha is None else alpha
        results[key] = np.mean(aucs) if aucs else float("nan")

    return results


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EMA smoothing experiments")
    parser.add_argument("--embeddings", type=Path, default=Path("data/processed/vlm_emb"))
    parser.add_argument("--labels", type=Path, default=Path("data/processed/labels_multih"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/outputs_temporal/best_temporal_model.pt"))
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--max_videos", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("results/ema_results.csv"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EMA SMOOTHING EXPERIMENT")
    print("=" * 60)
    print(f"Loading embeddings from: {args.embeddings}")
    pairs = load_video_pairs(args.embeddings, args.labels, args.max_videos)
    print(f"Loaded {len(pairs)} video pairs")

    clf_base = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")

    # ── Level 1: Score EMA ────────────────────
    print("\n[Level 1] Post-processing score EMA")
    score_results = eval_score_ema(pairs, clf_base, args.alphas)
    print(f"  Raw AUC:  {score_results['raw']:.4f}")
    for a in args.alphas:
        delta = score_results[a] - score_results["raw"]
        print(f"  alpha={a:.1f}: {score_results[a]:.4f}  (Δ {delta:+.4f})")

    # ── Level 2: Embedding EMA ────────────────
    print("\n[Level 2] Embedding-level EMA (applied before classifier)")
    emb_results = eval_embedding_ema(pairs, args.alphas)
    print(f"  Raw AUC:  {emb_results['raw']:.4f}")
    for a in args.alphas:
        delta = emb_results[a] - emb_results["raw"]
        print(f"  alpha={a:.1f}: {emb_results[a]:.4f}  (Δ {delta:+.4f})")

    # ── Level 3: Hidden-state EMA ─────────────
    print("\n[Level 3] LSTM hidden-state EMA")
    hs_results = eval_hidden_state_ema(args.checkpoint, pairs, args.alphas, device=args.device)
    if hs_results:
        print(f"  Raw AUC:  {hs_results.get('raw', float('nan')):.4f}")
        for a in args.alphas:
            if a in hs_results:
                delta = hs_results[a] - hs_results.get("raw", 0)
                print(f"  alpha={a:.1f}: {hs_results[a]:.4f}  (Δ {delta:+.4f})")
    else:
        print("  (skipped — checkpoint not available or incompatible)")

    # ── Save CSV ──────────────────────────────
    import csv
    rows = []
    for level, res in [("score_ema", score_results), ("embedding_ema", emb_results), ("hidden_state_ema", hs_results)]:
        for key, val in res.items():
            rows.append({"level": level, "alpha": key, "auc": f"{val:.4f}"})

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["level", "alpha", "auc"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()