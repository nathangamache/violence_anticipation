"""Compare simple baselines (random, majority, LR, SVM, LDA, sliding window) with the LSTM on VLM embeddings.

Usage:
    python experiments/baseline_comparisons.py --embeddings data/processed/vlm_emb --labels data/processed/labels_multih \
        --lstm_checkpoint outputs/outputs_temporal/best_temporal_model.pt --ema_alpha 0.5
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")


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


def ema_1d(values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def compute_metrics(val_pairs, score_fn, ema_alpha=None) -> Dict:
    """Frame AUC/AP and mean per-video AUC (EMA-smoothed if ema_alpha) plus unsmoothed FA per 10 min on all-negative videos."""
    all_scores, all_labels = [], []
    per_video_aucs = []

    for emb, lbl in val_pairs:
        scores = score_fn(emb)
        if ema_alpha is not None:
            scores = ema_1d(scores, ema_alpha)
        if len(np.unique(lbl)) > 1:
            per_video_aucs.append(roc_auc_score(lbl, scores))
        all_scores.extend(scores.tolist())
        all_labels.extend(lbl.tolist())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    frame_auc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else float("nan")
    frame_ap = average_precision_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else float("nan")
    per_vid_auc = float(np.mean(per_video_aucs)) if per_video_aucs else float("nan")

    normal_scores = [scores for scores, lbl in
                     [(score_fn(emb), lbl) for emb, lbl in val_pairs]
                     if lbl.max() == 0]
    if normal_scores:
        fa_rate = compute_false_alarm_rate(normal_scores, threshold=0.5)
    else:
        fa_rate = float("nan")

    return {
        "frame_auc": round(frame_auc, 4),
        "frame_ap": round(frame_ap, 4),
        "per_vid_auc": round(per_vid_auc, 4),
        "fa_per_10min": round(fa_rate, 2),
    }


def compute_false_alarm_rate(
    normal_video_scores: List[np.ndarray],
    threshold: float = 0.5,
    fps: float = 1.0,
) -> float:
    """False alarms per 10 min of normal footage; an alarm is a contiguous above-threshold run."""
    total_duration_minutes = sum(len(s) for s in normal_video_scores) / (fps * 60)
    total_alarms = 0

    for scores in normal_video_scores:
        in_alarm = False
        for s in scores:
            if s > threshold and not in_alarm:
                total_alarms += 1
                in_alarm = True
            elif s <= threshold:
                in_alarm = False

    if total_duration_minutes == 0:
        return float("nan")
    return total_alarms / total_duration_minutes * 10.0


def run_random_baseline(val_pairs) -> Dict:
    rng = np.random.RandomState(42)

    def score_fn(emb):
        return rng.uniform(0, 1, len(emb)).astype(np.float32)

    return compute_metrics(val_pairs, score_fn)


def run_majority_class(train_pairs, val_pairs) -> Dict:
    """Constant score equal to the training positive rate."""
    _, y_train = flatten(train_pairs)
    majority_score = float(y_train.mean())

    def score_fn(emb):
        return np.full(len(emb), majority_score, dtype=np.float32)

    return compute_metrics(val_pairs, score_fn)


def run_mean_pooling_lr(train_pairs, val_pairs) -> Dict:
    """Per-video mean embedding -> LR, giving one constant score per video."""
    X_train_vid = np.array([p[0].mean(axis=0) for p in train_pairs])
    y_train_vid = np.array([int(p[1].max()) for p in train_pairs])
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train_vid)
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(X_s, y_train_vid)

    def score_fn(emb):
        vid_score = clf.predict_proba(scaler.transform(emb.mean(axis=0, keepdims=True)))[0, 1]
        return np.full(len(emb), vid_score, dtype=np.float32)

    return compute_metrics(val_pairs, score_fn)


def run_lr_raw(train_pairs, val_pairs, ema_alpha=None) -> Dict:
    X_train, y_train = flatten(train_pairs)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")
    clf.fit(X_s, y_train)

    def score_fn(emb):
        return clf.predict_proba(scaler.transform(emb))[:, 1].astype(np.float32)

    return compute_metrics(val_pairs, score_fn, ema_alpha=ema_alpha)


def run_svm(train_pairs, val_pairs) -> Dict:
    """LinearSVC with calibration for probability outputs."""
    X_train, y_train = flatten(train_pairs)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train)
    base = LinearSVC(max_iter=2000, C=1.0, class_weight="balanced")
    clf = CalibratedClassifierCV(base, cv=3)
    clf.fit(X_s, y_train)

    def score_fn(emb):
        return clf.predict_proba(scaler.transform(emb))[:, 1].astype(np.float32)

    return compute_metrics(val_pairs, score_fn)


def run_lda_calibrated(train_pairs, val_pairs) -> Dict:
    """LDA projection -> sigmoid centred between the class means."""
    X_train, y_train = flatten(train_pairs)
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(X_train, y_train).flatten()
    v_mean = X_lda[y_train == 1].mean()
    nv_mean = X_lda[y_train == 0].mean()
    midpoint = (v_mean + nv_mean) / 2
    scale = 4.0 / (abs(v_mean - nv_mean) + 1e-8)

    def score_fn(emb):
        proj = lda.transform(emb).flatten()
        return (1.0 / (1.0 + np.exp(-scale * (proj - midpoint)))).astype(np.float32)

    return compute_metrics(val_pairs, score_fn)


def run_lr_temporal_features(train_pairs, val_pairs, window=10) -> Dict:
    """LR with temporal context features (mean, std, delta of window)."""
    def build_features(emb, w):
        T, D = emb.shape
        feats = []
        for i in range(T):
            start = max(0, i - w)
            win = emb[start:i + 1]
            delta = emb[i] - emb[i - 1] if i > 0 else np.zeros(D)
            feats.append(np.concatenate([emb[i], win.mean(0), win.std(0) + 1e-8, delta]))
        return np.array(feats, dtype=np.float32)

    print("    Building temporal features for LR+temporal...")
    train_feats = [(build_features(emb, window), lbl) for emb, lbl in train_pairs]
    val_feats = [(build_features(emb, window), lbl) for emb, lbl in val_pairs]

    X_train, y_train = flatten(train_feats)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", solver="saga")
    clf.fit(X_s, y_train)

    def score_fn(emb):
        feats = build_features(emb, window)
        return clf.predict_proba(scaler.transform(feats))[:, 1].astype(np.float32)

    return compute_metrics(val_feats, score_fn)


def run_sliding_window(train_pairs, val_pairs, window=15, strategy="mean") -> Dict:
    """LR scores aggregated with sliding window."""
    X_train, y_train = flatten(train_pairs)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")
    clf.fit(X_s, y_train)

    def sliding(scores, w, strat):
        out = np.empty_like(scores)
        for i in range(len(scores)):
            s = max(0, i - w + 1)
            win = scores[s:i + 1]
            if strat == "mean":
                out[i] = win.mean()
            elif strat == "max":
                out[i] = win.max()
            elif strat == "p75":
                out[i] = np.percentile(win, 75)
            else:
                out[i] = win.mean()
        return out

    def score_fn(emb):
        raw = clf.predict_proba(scaler.transform(emb))[:, 1].astype(np.float32)
        return sliding(raw, window, strategy)

    return compute_metrics(val_pairs, score_fn)


def run_lstm(checkpoint_path: Path, train_pairs, val_pairs, ema_alpha=None, device="cpu", seq_len=30) -> Dict:
    if not checkpoint_path.exists():
        print(f"    [SKIP] {checkpoint_path} not found")
        return {"frame_auc": float("nan"), "frame_ap": float("nan"), "per_vid_auc": float("nan"), "fa_per_10min": float("nan")}

    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model = ckpt.get("model", None)
        if model is None:
            print("    [SKIP] No model in checkpoint")
            return {"frame_auc": float("nan"), "frame_ap": float("nan"), "per_vid_auc": float("nan"), "fa_per_10min": float("nan")}
        model.eval().to(device)
    except Exception as e:
        print(f"    [SKIP] {e}")
        return {"frame_auc": float("nan"), "frame_ap": float("nan"), "per_vid_auc": float("nan"), "fa_per_10min": float("nan")}

    X_train = np.vstack([p[0] for p in train_pairs])
    scaler = StandardScaler()
    scaler.fit(X_train)

    def score_fn(emb):
        emb_s = scaler.transform(emb)
        scores = np.full(len(emb), 0.0, dtype=np.float32)
        with torch.no_grad():
            for i in range(seq_len, len(emb_s)):
                seq = torch.tensor(emb_s[i - seq_len:i], dtype=torch.float32, device=device).unsqueeze(0)
                out = model(seq)
                if out.shape[-1] == 1:
                    scores[i] = torch.sigmoid(out).item()
                else:
                    scores[i] = torch.softmax(out, dim=-1)[0, 1].item()
        return scores

    return compute_metrics(val_pairs, score_fn, ema_alpha=ema_alpha)


def print_results_table(results: List[Dict]):
    print("\n" + "=" * 75)
    print("COMPREHENSIVE BASELINE COMPARISON")
    print("=" * 75)
    print(f"{'Model':<35} {'Frame AUC':>10} {'Frame AP':>10} {'Vid AUC':>10} {'FA/10min':>10}")
    print("-" * 75)

    results_sorted = sorted(results, key=lambda r: r.get("frame_auc", 0) or 0, reverse=True)
    for r in results_sorted:
        def fmt(v):
            return f"{v:.4f}" if not (v != v) else "   N/A"  # nan check
        print(f"  {r['model']:<33} {fmt(r['frame_auc']):>10} {fmt(r['frame_ap']):>10} "
              f"{fmt(r['per_vid_auc']):>10} {fmt(r['fa_per_10min']):>10}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, default=Path("data/processed/vlm_emb"))
    parser.add_argument("--labels", type=Path, default=Path("data/processed/labels_multih"))
    parser.add_argument("--lstm_checkpoint", type=Path, default=Path("outputs/outputs_temporal/best_temporal_model.pt"))
    parser.add_argument("--ema_alpha", type=float, default=0.5)
    parser.add_argument("--max_videos", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("results/baseline_comparison.csv"))
    parser.add_argument("--skip_slow", action="store_true",
                        help="Skip slow models (LR+temporal, Transformer) for a quick run")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = args.device

    print("=" * 65)
    print("LOADING DATA")
    print("=" * 65)
    pairs = load_video_pairs(args.embeddings, args.labels, args.max_videos)
    print(f"Loaded {len(pairs)} videos")
    train_pairs, val_pairs = train_val_split(pairs)
    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    results = []

    print("\n[B1] Random baseline")
    r = run_random_baseline(val_pairs)
    r["model"] = "Random"
    results.append(r)
    print(f"  AUC={r['frame_auc']:.4f}")

    print("[B2] Majority class")
    r = run_majority_class(train_pairs, val_pairs)
    r["model"] = "Majority class"
    results.append(r)
    print(f"  AUC={r['frame_auc']:.4f}")

    print("[B3] Mean pooling + LR")
    r = run_mean_pooling_lr(train_pairs, val_pairs)
    r["model"] = "Mean pooling + LR"
    results.append(r)
    print(f"  AUC={r['frame_auc']:.4f}")

    print("[B4] LR on raw VLM embeddings")
    r = run_lr_raw(train_pairs, val_pairs)
    r["model"] = "LR (raw embeddings)"
    results.append(r)
    print(f"  AUC={r['frame_auc']:.4f}")

    print("[B5] SVM on VLM embeddings")
    r = run_svm(train_pairs, val_pairs)
    r["model"] = "SVM (raw embeddings)"
    results.append(r)
    print(f"  AUC={r['frame_auc']:.4f}")

    print(f"[C1] LSTM (checkpoint)")
    r = run_lstm(args.lstm_checkpoint, train_pairs, val_pairs, device=device)
    r["model"] = "LSTM"
    results.append(r)
    print(f"  AUC={r['frame_auc']:.4f}")

    print(f"[C2] LSTM + EMA (alpha={args.ema_alpha})")
    r = run_lstm(args.lstm_checkpoint, train_pairs, val_pairs, ema_alpha=args.ema_alpha, device=device)
    r["model"] = f"LSTM + EMA (a={args.ema_alpha})"
    results.append(r)
    print(f"  AUC={r['frame_auc']:.4f}")

    print(f"[C3] LR + EMA (alpha={args.ema_alpha})")
    r = run_lr_raw(train_pairs, val_pairs, ema_alpha=args.ema_alpha)
    r["model"] = f"LR + EMA (a={args.ema_alpha})"
    results.append(r)
    print(f"  AUC={r['frame_auc']:.4f}")

    if not args.skip_slow:
        print("[C4] LR + temporal features (window=10)")
        r = run_lr_temporal_features(train_pairs, val_pairs, window=10)
        r["model"] = "LR + temporal (w=10)"
        results.append(r)
        print(f"  AUC={r['frame_auc']:.4f}")

    print("[C5] Sliding window (w=15, mean)")
    r = run_sliding_window(train_pairs, val_pairs, window=15, strategy="mean")
    r["model"] = "Sliding window (w=15)"
    results.append(r)
    print(f"  AUC={r['frame_auc']:.4f}")

    print("[C5b] Sliding window (w=15, p75)")
    r = run_sliding_window(train_pairs, val_pairs, window=15, strategy="p75")
    r["model"] = "Sliding window (w=15, p75)"
    results.append(r)
    print(f"  AUC={r['frame_auc']:.4f}")

    print("[C6] LDA-calibrated LR")
    r = run_lda_calibrated(train_pairs, val_pairs)
    r["model"] = "LDA calibrated"
    results.append(r)
    print(f"  AUC={r['frame_auc']:.4f}")

    print_results_table(results)

    import csv
    fieldnames = ["model", "frame_auc", "frame_ap", "per_vid_auc", "fa_per_10min"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()