"""
algorithm_variations.py
========================
Three alternative algorithms evaluated against the current LSTM baseline:

  1. Transformer encoder — replaces LSTM with multi-head self-attention
  2. Logistic Regression (primary, not just baseline) — with calibration,
     temporal context features, and per-video evaluation
  3. Sliding window aggregation — pools a window of frame-level LR scores
     using mean, max, and percentile strategies

Usage:
    python algorithm_variations.py \
        --embeddings data/processed/vlm_emb \
        --labels data/processed/labels_multih \
        --checkpoint outputs/outputs_temporal/best_temporal_model.pt

Outputs: results/algorithm_comparison.csv  +  printed table
"""

import argparse
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# Shared data helpers
# ──────────────────────────────────────────────

def load_video_pairs(vlm_dir: Path, labels_dir: Path, max_videos: int = 300):
    pairs = []
    for vf in sorted(vlm_dir.rglob("*.npz"))[:max_videos]:
        vid = vf.stem
        for candidate in [
            labels_dir / f"{vid}.npy",
            labels_dir / "xd_violence" / f"{vid}.npy",
        ]:
            if candidate.exists():
                try:
                    emb = np.load(vf)["emb"].astype(np.float32)
                    lbl = np.load(candidate)
                    if lbl.ndim > 1:
                        lbl = lbl.max(axis=1)
                    n = min(len(emb), len(lbl))
                    pairs.append((emb[:n], (lbl[:n] > 0).astype(int)))
                except Exception:
                    pass
                break
    return pairs


def train_val_split(pairs, val_frac=0.2):
    n_val = max(1, int(len(pairs) * val_frac))
    return pairs[:-n_val], pairs[-n_val:]


def flatten(pairs):
    return np.vstack([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])


# ──────────────────────────────────────────────
# 1. Transformer Encoder
# ──────────────────────────────────────────────

class TransformerViolenceModel(nn.Module):
    """
    Replaces the bidirectional LSTM with a Transformer encoder.
    Architecture:
      - Linear projection: input_dim → d_model
      - Positional encoding (learnable)
      - N × TransformerEncoderLayer (multi-head attention + FFN)
      - CLS token pooling → classifier head
    """

    def __init__(
        self,
        input_dim: int = 3584,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        seq_len: int = 30,
        num_classes: int = 1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Learnable positional encoding (seq_len + 1 for CLS token)
        self.pos_emb = nn.Embedding(seq_len + 1, d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        B, T, D = x.shape

        # Project input
        x = self.input_proj(x)  # (B, T, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, d_model)

        # Add positional encodings
        positions = torch.arange(T + 1, device=x.device).unsqueeze(0)
        x = x + self.pos_emb(positions)

        # Transformer
        out = self.transformer(x)  # (B, T+1, d_model)

        # CLS token output → classifier
        cls_out = out[:, 0, :]  # (B, d_model)
        return self.classifier(cls_out)  # (B, 1)


def train_transformer(train_pairs, val_pairs, seq_len=30, epochs=30, lr=1e-3, device="cpu"):
    """Train the TransformerViolenceModel with early stopping."""

    # Scaler fitted on train embeddings
    X_tr, _ = flatten(train_pairs)
    scaler = StandardScaler()
    scaler.fit(X_tr)

    def make_sequences(pairs):
        Xs, ys = [], []
        for emb, lbl in pairs:
            emb_s = scaler.transform(emb)
            for i in range(seq_len, len(emb_s)):
                Xs.append(emb_s[i - seq_len:i])
                ys.append(lbl[i])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    X_train_seq, y_train_seq = make_sequences(train_pairs)
    X_val_seq, y_val_seq = make_sequences(val_pairs)

    # Class weights
    pos = y_train_seq.sum()
    neg = len(y_train_seq) - pos
    pos_weight = torch.tensor([neg / (pos + 1e-6)], dtype=torch.float32, device=device)

    model = TransformerViolenceModel(seq_len=seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = 0.0
    best_state = None
    patience = 7
    no_improve = 0

    batch_size = 256
    n = len(X_train_seq)

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = torch.tensor(X_train_seq[idx], device=device)
            yb = torch.tensor(y_train_seq[idx], device=device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # Validation AUC
        model.eval()
        with torch.no_grad():
            scores = []
            for start in range(0, len(X_val_seq), 512):
                xb = torch.tensor(X_val_seq[start:start + 512], device=device)
                scores.append(torch.sigmoid(model(xb)).cpu().numpy().flatten())
            scores = np.concatenate(scores)

        if len(np.unique(y_val_seq)) > 1:
            val_auc = roc_auc_score(y_val_seq, scores)
        else:
            val_auc = 0.0

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:3d}  loss={epoch_loss/max(1,n//batch_size):.4f}  val_auc={val_auc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler, best_val_auc


def eval_transformer(model, scaler, val_pairs, seq_len=30, device="cpu"):
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for emb, lbl in val_pairs:
            emb_s = scaler.transform(emb)
            for i in range(seq_len, len(emb_s)):
                seq = torch.tensor(emb_s[i - seq_len:i], dtype=torch.float32, device=device).unsqueeze(0)
                score = torch.sigmoid(model(seq)).item()
                all_scores.append(score)
                all_labels.append(lbl[i])
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    auc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else float("nan")
    ap = average_precision_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else float("nan")
    return auc, ap


# ──────────────────────────────────────────────
# 2. Logistic Regression as Primary Model
# ──────────────────────────────────────────────

def build_temporal_features(emb: np.ndarray, window: int = 10) -> np.ndarray:
    """
    Augment raw VLM embeddings with temporal context features:
      - Frame embedding (3584-dim)
      - Mean of last `window` frames (3584-dim)
      - Std of last `window` frames (3584-dim)  [flattened to scalar per-dim]
      - Delta (current − previous frame)
    Total: 3584 * 4 = 14336 dims (large, so we also provide a PCA-reduced version)
    """
    T, D = emb.shape
    feats = []
    for i in range(T):
        start = max(0, i - window)
        window_emb = emb[start:i + 1]
        mean_w = window_emb.mean(axis=0)
        std_w = window_emb.std(axis=0)
        delta = emb[i] - emb[i - 1] if i > 0 else np.zeros(D, dtype=np.float32)
        feats.append(np.concatenate([emb[i], mean_w, std_w, delta]))
    return np.array(feats, dtype=np.float32)


def eval_logistic_regression(train_pairs, val_pairs, use_temporal_features=False, window=10):
    """
    Logistic regression, optionally with temporal context features.
    Includes probability calibration via CalibratedClassifierCV.
    """
    if use_temporal_features:
        print("    Building temporal features (may take a moment)...")
        train_feats = [(build_temporal_features(emb, window), lbl) for emb, lbl in train_pairs]
        val_feats = [(build_temporal_features(emb, window), lbl) for emb, lbl in val_pairs]
    else:
        train_feats = train_pairs
        val_feats = val_pairs

    X_train, y_train = flatten(train_feats)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    # Base LR with calibration
    base_clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", solver="saga")
    clf = CalibratedClassifierCV(base_clf, cv=3, method="isotonic")
    clf.fit(X_train_s, y_train)

    per_video_aucs, per_video_aps = [], []
    for emb, lbl in val_feats:
        if use_temporal_features:
            scores = clf.predict_proba(scaler.transform(emb))[:, 1]
        else:
            scores = clf.predict_proba(scaler.transform(emb))[:, 1]
        if len(np.unique(lbl)) > 1:
            per_video_aucs.append(roc_auc_score(lbl, scores))
            per_video_aps.append(average_precision_score(lbl, scores))

    mean_auc = np.mean(per_video_aucs) if per_video_aucs else float("nan")
    mean_ap = np.mean(per_video_aps) if per_video_aps else float("nan")
    return mean_auc, mean_ap, clf, scaler


# ──────────────────────────────────────────────
# 3. Sliding Window Aggregation
# ──────────────────────────────────────────────

def sliding_window_aggregate(scores: np.ndarray, window: int, strategy: str = "mean") -> np.ndarray:
    """
    Aggregate frame-level scores over a sliding window.
      mean      — simple average
      max       — maximum (sensitive, good for detecting any violence)
      p75       — 75th percentile (robust to outliers)
      ewm       — exponentially weighted mean (equivalent to EMA but framed as aggregation)
    """
    out = np.empty_like(scores)
    for i in range(len(scores)):
        start = max(0, i - window + 1)
        window_scores = scores[start:i + 1]
        if strategy == "mean":
            out[i] = window_scores.mean()
        elif strategy == "max":
            out[i] = window_scores.max()
        elif strategy == "p75":
            out[i] = np.percentile(window_scores, 75)
        elif strategy == "ewm":
            # EWM within window
            alpha = 2 / (len(window_scores) + 1)
            v = window_scores[0]
            for s in window_scores[1:]:
                v = alpha * s + (1 - alpha) * v
            out[i] = v
        else:
            out[i] = window_scores.mean()
    return out


def eval_sliding_window(train_pairs, val_pairs, windows=(5, 10, 20, 30), strategies=("mean", "max", "p75", "ewm")):
    """
    Train a base LR classifier, then evaluate AUC/AP at all window × strategy combos.
    """
    X_train, y_train = flatten(train_pairs)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")
    clf.fit(X_train_s, y_train)

    results = {}

    # Baseline: no aggregation
    aucs_raw = []
    for emb, lbl in val_pairs:
        scores = clf.predict_proba(scaler.transform(emb))[:, 1]
        if len(np.unique(lbl)) > 1:
            aucs_raw.append(roc_auc_score(lbl, scores))
    results[("raw", 1)] = (np.mean(aucs_raw), float("nan"))

    for window in windows:
        for strategy in strategies:
            aucs, aps = [], []
            for emb, lbl in val_pairs:
                scores_raw = clf.predict_proba(scaler.transform(emb))[:, 1]
                scores_agg = sliding_window_aggregate(scores_raw, window, strategy)
                if len(np.unique(lbl)) > 1:
                    aucs.append(roc_auc_score(lbl, scores_agg))
                    aps.append(average_precision_score(lbl, scores_agg))
            results[(strategy, window)] = (
                np.mean(aucs) if aucs else float("nan"),
                np.mean(aps) if aps else float("nan"),
            )

    return results


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, default=Path("data/processed/vlm_emb"))
    parser.add_argument("--labels", type=Path, default=Path("data/processed/labels_multih"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/outputs_temporal/best_temporal_model.pt"))
    parser.add_argument("--max_videos", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("results/algorithm_comparison.csv"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 60)
    print("ALGORITHM VARIATIONS EXPERIMENT")
    print("=" * 60)

    pairs = load_video_pairs(args.embeddings, args.labels, args.max_videos)
    print(f"Loaded {len(pairs)} video pairs")
    train_pairs, val_pairs = train_val_split(pairs)
    print(f"Train: {len(train_pairs)} videos | Val: {len(val_pairs)} videos")

    all_results = []

    # ── Baseline: plain LR ────────────────────
    print("\n[Baseline] Plain logistic regression")
    auc, ap, _, _ = eval_logistic_regression(train_pairs, val_pairs, use_temporal_features=False)
    print(f"  AUC: {auc:.4f}  AP: {ap:.4f}")
    all_results.append({"model": "LR (baseline)", "auc": auc, "ap": ap})

    # ── 1. Transformer ────────────────────────
    print("\n[1] Transformer encoder")
    model, scaler, best_val_auc = train_transformer(train_pairs, val_pairs, device=str(device))
    t_auc, t_ap = eval_transformer(model, scaler, val_pairs, device=str(device))
    print(f"  AUC: {t_auc:.4f}  AP: {t_ap:.4f}  (best val during train: {best_val_auc:.4f})")
    all_results.append({"model": "Transformer", "auc": t_auc, "ap": t_ap})

    # Save transformer checkpoint
    torch.save({"model": model, "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_},
               "results/transformer_model.pt")
    print("  Saved to results/transformer_model.pt")

    # ── 2. LR as primary (with temporal features) ──
    print("\n[2] Logistic regression with temporal context features (window=10)")
    auc_t, ap_t, _, _ = eval_logistic_regression(train_pairs, val_pairs, use_temporal_features=True, window=10)
    print(f"  AUC: {auc_t:.4f}  AP: {ap_t:.4f}")
    all_results.append({"model": "LR + temporal features", "auc": auc_t, "ap": ap_t})

    # ── 3. Sliding window aggregation ─────────
    print("\n[3] Sliding window aggregation")
    sw_results = eval_sliding_window(train_pairs, val_pairs)
    best_sw = max(sw_results.items(), key=lambda x: x[1][0] if not np.isnan(x[1][0]) else 0)
    print(f"  Baseline (no window):  AUC={sw_results[('raw',1)][0]:.4f}")
    for (strategy, window), (auc_sw, ap_sw) in sorted(sw_results.items(), key=lambda x: -x[1][0]):
        if strategy == "raw":
            continue
        print(f"  window={window:2d} strategy={strategy:5s}  AUC={auc_sw:.4f}  AP={ap_sw:.4f}")
        all_results.append({"model": f"SlidingWindow_{strategy}_w{window}", "auc": auc_sw, "ap": ap_sw})

    print(f"\n  Best config: window={best_sw[0][1]}, strategy={best_sw[0][0]}, AUC={best_sw[1][0]:.4f}")

    # ── Summary table ─────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<35} {'AUC':>8} {'AP':>8}")
    print("-" * 55)
    for r in all_results:
        print(f"  {r['model']:<33} {r['auc']:>8.4f} {r['ap']:>8.4f}")

    # ── Save CSV ──────────────────────────────
    import csv
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "auc", "ap"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()