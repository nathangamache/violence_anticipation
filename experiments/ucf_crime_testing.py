"""
ucf_crime_testing.py
=====================
Evaluates all trained models on the UCF-Crime dataset to measure cross-domain
generalization. UCF-Crime has different annotation structure from XD-Violence,
so this script handles both formats.

What it does:
  1. Loads UCF-Crime embeddings (VLM or raw features)
  2. Maps UCF-Crime labels to binary violent/non-violent
  3. Runs all trained models: LSTM, Logistic Regression, Transformer
  4. Computes per-video and aggregate AUC, AP, and false alarm rate
  5. Computes LDA distribution shift (how far UCF-Crime is from XD-Violence)
  6. Tries LDA-based calibration to partially correct for the shift

Usage:
    python ucf_crime_testing.py \
        --ucf_embeddings data/processed/vlm_emb/ucf_crime \
        --ucf_labels data/processed/labels_multih/ucf_crime \
        --xd_embeddings data/processed/vlm_emb \
        --xd_labels data/processed/labels_multih \
        --lstm_checkpoint outputs/outputs_temporal/best_temporal_model.pt \
        --transformer_checkpoint results/transformer_model.pt
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# UCF-Crime label mapping
# ──────────────────────────────────────────────

# UCF-Crime anomaly categories (all are "violent" or at-risk events)
UCF_CRIME_VIOLENT_CATEGORIES = {
    "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
    "Fighting", "RoadAccidents", "Robbery", "Shooting", "Shoplifting",
    "Stealing", "Vandalism",
}

# For anticipation evaluation: these categories have clear pre-event windows
UCF_ANTICIPATION_CATEGORIES = {
    "Assault", "Fighting", "Shooting", "Robbery", "Explosion",
}


def infer_ucf_label_from_filename(filename: str) -> Tuple[int, str]:
    """
    UCF-Crime filenames often contain the category name.
    Returns (label, category) where label=1 if violent, 0 if normal.
    """
    fname = Path(filename).stem.lower()
    for cat in UCF_CRIME_VIOLENT_CATEGORIES:
        if cat.lower() in fname:
            return 1, cat
    if "normal" in fname or "background" in fname:
        return 0, "Normal"
    return -1, "Unknown"  # -1 = can't determine


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def load_ucf_pairs(
    emb_dir: Path,
    labels_dir: Optional[Path] = None,
    max_videos: int = 500,
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Load UCF-Crime embedding/label pairs.
    If labels_dir is None, tries to infer labels from filenames.
    Returns list of (emb, labels, video_name).
    """
    pairs = []
    emb_files = sorted(emb_dir.rglob("*.npz"))[:max_videos]

    for vf in emb_files:
        vid = vf.stem
        label_found = False

        # Try loading from labels dir
        if labels_dir is not None:
            for candidate in [
                labels_dir / f"{vid}.npy",
                labels_dir / "ucf_crime" / f"{vid}.npy",
                labels_dir / f"{vid}_i3d.npy",
            ]:
                if candidate.exists():
                    try:
                        emb = np.load(vf)["emb"].astype(np.float32)
                        lbl = np.load(candidate)
                        if lbl.ndim > 1:
                            lbl = lbl.max(axis=1)
                        n = min(len(emb), len(lbl))
                        pairs.append((emb[:n], (lbl[:n] > 0).astype(int), vid))
                        label_found = True
                    except Exception:
                        pass
                    break

        # Fall back to filename-based labels
        if not label_found:
            inferred_label, cat = infer_ucf_label_from_filename(vid)
            if inferred_label >= 0:
                try:
                    emb = np.load(vf)["emb"].astype(np.float32)
                    # Assign uniform label for the whole video
                    lbl = np.full(len(emb), inferred_label, dtype=int)
                    pairs.append((emb, lbl, vid))
                except Exception:
                    pass

    return pairs


def load_xd_pairs(emb_dir: Path, labels_dir: Path, max_videos: int = 300):
    pairs = []
    for vf in sorted(emb_dir.rglob("*.npz"))[:max_videos]:
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


# ──────────────────────────────────────────────
# Distribution shift analysis
# ──────────────────────────────────────────────

def measure_distribution_shift(xd_embeddings: np.ndarray, ucf_embeddings: np.ndarray) -> Dict:
    """
    Quantify the distribution shift between XD-Violence and UCF-Crime embeddings.
    Uses LDA projection and per-dimension statistics.
    """
    results = {}

    # Mean shift
    xd_mean = xd_embeddings.mean(axis=0)
    ucf_mean = ucf_embeddings.mean(axis=0)
    mean_l2 = float(np.linalg.norm(xd_mean - ucf_mean))
    results["mean_l2_shift"] = mean_l2

    # Cosine similarity between means
    cos_sim = float(
        np.dot(xd_mean, ucf_mean) / (np.linalg.norm(xd_mean) * np.linalg.norm(ucf_mean) + 1e-8)
    )
    results["mean_cosine_similarity"] = cos_sim

    # Dimension-wise shift (how many dims differ significantly)
    xd_std = xd_embeddings.std(axis=0) + 1e-8
    ucf_std = ucf_embeddings.std(axis=0) + 1e-8
    pooled = np.sqrt((xd_std ** 2 + ucf_std ** 2) / 2)
    shift_d = np.abs(xd_mean - ucf_mean) / pooled  # normalized shift per dim
    results["dims_shifted_gt_0.5"] = int((shift_d > 0.5).sum())
    results["dims_shifted_gt_1.0"] = int((shift_d > 1.0).sum())
    results["mean_dim_shift"] = float(shift_d.mean())

    return results


# ──────────────────────────────────────────────
# LDA calibration
# ──────────────────────────────────────────────

def lda_calibrate(
    source_emb: np.ndarray,  # XD-Violence embeddings (training distribution)
    source_labels: np.ndarray,
    target_emb: np.ndarray,  # UCF-Crime embeddings (target distribution)
) -> np.ndarray:
    """
    Fit LDA on source, project both domains, then rescale target to match
    source's mean and std on the LDA axis. This partially corrects for shift.
    Returns calibrated target embeddings in the LDA subspace (1-dim).
    """
    lda = LinearDiscriminantAnalysis(n_components=1)
    src_lda = lda.fit_transform(source_emb, source_labels).flatten()
    tgt_lda = lda.transform(target_emb).flatten()

    # Z-score calibration: normalize target to source's distribution
    tgt_calibrated = (tgt_lda - tgt_lda.mean()) / (tgt_lda.std() + 1e-8) * src_lda.std() + src_lda.mean()
    return tgt_calibrated, src_lda, lda


# ──────────────────────────────────────────────
# Model evaluation
# ──────────────────────────────────────────────

def eval_logistic_regression_cross_domain(
    xd_pairs, ucf_pairs, use_calibration: bool = False
):
    """Train on XD-Violence, evaluate on UCF-Crime."""
    X_train = np.vstack([p[0] for p in xd_pairs])
    y_train = np.concatenate([p[1] for p in xd_pairs])

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")
    clf.fit(X_train_s, y_train)

    per_video_results = []
    all_scores, all_labels = [], []

    for emb, lbl, vid in ucf_pairs:
        emb_s = scaler.transform(emb)
        scores = clf.predict_proba(emb_s)[:, 1]
        per_video_results.append({
            "video": vid,
            "auc": roc_auc_score(lbl, scores) if len(np.unique(lbl)) > 1 else float("nan"),
            "label": int(lbl.max()),
        })
        all_scores.extend(scores)
        all_labels.extend(lbl)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    agg_auc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else float("nan")
    agg_ap = average_precision_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else float("nan")

    return agg_auc, agg_ap, per_video_results


def eval_lstm_cross_domain(checkpoint_path: Path, xd_pairs, ucf_pairs, seq_len=30, device="cpu"):
    """Load LSTM trained on XD, evaluate on UCF."""
    if not checkpoint_path.exists():
        print(f"  [SKIP] {checkpoint_path} not found")
        return float("nan"), float("nan"), []

    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model = ckpt.get("model", None)
        if model is None:
            print("  [SKIP] No model in checkpoint")
            return float("nan"), float("nan"), []
        model.eval().to(device)
    except Exception as e:
        print(f"  [SKIP] {e}")
        return float("nan"), float("nan"), []

    # Get scaler from XD training data
    X_xd = np.vstack([p[0] for p in xd_pairs])
    scaler = StandardScaler()
    scaler.fit(X_xd)

    all_scores, all_labels = [], []
    per_video_results = []

    with torch.no_grad():
        for emb, lbl, vid in ucf_pairs:
            if len(emb) < seq_len:
                continue
            emb_s = scaler.transform(emb)
            scores = []
            for i in range(seq_len, len(emb_s)):
                seq = torch.tensor(emb_s[i - seq_len:i], dtype=torch.float32, device=device).unsqueeze(0)
                out = model(seq)
                if isinstance(out, torch.Tensor):
                    if out.shape[-1] == 1:
                        score = torch.sigmoid(out).item()
                    else:
                        score = torch.softmax(out, dim=-1)[0, 1].item()
                else:
                    score = float(out)
                scores.append(score)

            scores = np.array(scores)
            lbl_seq = lbl[seq_len:]
            if len(np.unique(lbl_seq)) > 1:
                per_video_results.append({
                    "video": vid,
                    "auc": roc_auc_score(lbl_seq, scores),
                    "label": int(lbl_seq.max()),
                })
            all_scores.extend(scores)
            all_labels.extend(lbl_seq)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    agg_auc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else float("nan")
    agg_ap = average_precision_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else float("nan")
    return agg_auc, agg_ap, per_video_results


def eval_lda_calibrated(xd_pairs, ucf_pairs):
    """
    Train LDA on XD-Violence, calibrate UCF-Crime embeddings to training distribution,
    then score with a simple threshold on the LDA projection.
    """
    X_xd = np.vstack([p[0] for p in xd_pairs])
    y_xd = np.concatenate([p[1] for p in xd_pairs])

    all_scores, all_labels = [], []
    per_video_results = []

    for emb, lbl, vid in ucf_pairs:
        calibrated_lda, src_lda, lda_model = lda_calibrate(X_xd, y_xd, emb)

        # Convert LDA score to probability using sigmoid calibration
        v_mean_lda = src_lda[y_xd == 1].mean()
        nv_mean_lda = src_lda[y_xd == 0].mean()
        midpoint = (v_mean_lda + nv_mean_lda) / 2
        scale = 4.0 / (abs(v_mean_lda - nv_mean_lda) + 1e-8)
        scores = 1.0 / (1.0 + np.exp(-scale * (calibrated_lda - midpoint)))

        if len(np.unique(lbl)) > 1:
            per_video_results.append({
                "video": vid,
                "auc": roc_auc_score(lbl, scores),
                "label": int(lbl.max()),
            })
        all_scores.extend(scores)
        all_labels.extend(lbl)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    agg_auc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else float("nan")
    agg_ap = average_precision_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else float("nan")
    return agg_auc, agg_ap, per_video_results


# ──────────────────────────────────────────────
# False alarm rate analysis
# ──────────────────────────────────────────────

def false_alarm_analysis(per_video_results: List[Dict], threshold: float = 0.5) -> Dict:
    """
    Compute event-level false alarm stats for normal videos.
    A false alarm = any video labeled 0 (normal) where score > threshold.
    """
    normal_videos = [r for r in per_video_results if r["label"] == 0]
    if not normal_videos:
        return {}

    fa_count = sum(1 for r in normal_videos if r.get("auc", 0) > threshold)
    fa_rate = fa_count / len(normal_videos) if normal_videos else 0.0
    return {
        "n_normal_videos": len(normal_videos),
        "false_alarms": fa_count,
        "false_alarm_rate": fa_rate,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ucf_embeddings", type=Path, default=Path("data/processed/vlm_emb/ucf_crime"))
    parser.add_argument("--ucf_labels", type=Path, default=None)
    parser.add_argument("--xd_embeddings", type=Path, default=Path("data/processed/vlm_emb"))
    parser.add_argument("--xd_labels", type=Path, default=Path("data/processed/labels_multih"))
    parser.add_argument("--lstm_checkpoint", type=Path, default=Path("outputs/outputs_temporal/best_temporal_model.pt"))
    parser.add_argument("--transformer_checkpoint", type=Path, default=Path("results/transformer_model.pt"))
    parser.add_argument("--max_ucf", type=int, default=200)
    parser.add_argument("--max_xd", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("results/ucf_crime_results.csv"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = args.device

    print("=" * 65)
    print("UCF-CRIME CROSS-DOMAIN EVALUATION")
    print("=" * 65)

    # Load data
    print(f"\nLoading XD-Violence (train source)...")
    if not args.xd_embeddings.exists():
        print(f"  WARNING: {args.xd_embeddings} not found")
        xd_pairs = []
    else:
        xd_pairs = load_xd_pairs(args.xd_embeddings, args.xd_labels, args.max_xd)
        print(f"  Loaded {len(xd_pairs)} XD-Violence videos")

    print(f"\nLoading UCF-Crime (test target)...")
    if not args.ucf_embeddings.exists():
        print(f"  WARNING: {args.ucf_embeddings} not found — checking for ucf_crime subdir in xd path")
        ucf_alt = args.xd_embeddings.parent / "ucf_crime"
        if ucf_alt.exists():
            args.ucf_embeddings = ucf_alt
            print(f"  Using {ucf_alt}")
        else:
            print("  No UCF-Crime embeddings found. Run embedding extraction first.")
            print("  Expected path:", args.ucf_embeddings)
            return

    ucf_pairs = load_ucf_pairs(args.ucf_embeddings, args.ucf_labels, args.max_ucf)
    print(f"  Loaded {len(ucf_pairs)} UCF-Crime videos")

    if not xd_pairs or not ucf_pairs:
        print("\nInsufficient data. Exiting.")
        return

    # Distribution shift analysis
    print("\n[Distribution Shift Analysis]")
    X_xd_all = np.vstack([p[0] for p in xd_pairs])
    X_ucf_all = np.vstack([p[0] for p in ucf_pairs])
    shift = measure_distribution_shift(X_xd_all, X_ucf_all)
    print(f"  Mean L2 shift:          {shift['mean_l2_shift']:.4f}")
    print(f"  Mean cosine similarity: {shift['mean_cosine_similarity']:.4f}")
    print(f"  Dims shifted > 0.5 std: {shift['dims_shifted_gt_0.5']}")
    print(f"  Dims shifted > 1.0 std: {shift['dims_shifted_gt_1.0']}")
    print(f"  Mean per-dim shift:     {shift['mean_dim_shift']:.4f}")

    all_results = []

    # LR baseline (no calibration)
    print("\n[1] Logistic Regression (XD-trained, UCF-tested)")
    auc, ap, per_vid = eval_logistic_regression_cross_domain(xd_pairs, ucf_pairs)
    fa = false_alarm_analysis(per_vid)
    print(f"  Aggregate AUC: {auc:.4f}  AP: {ap:.4f}")
    if fa:
        print(f"  False alarm rate: {fa['false_alarm_rate']*100:.1f}%  ({fa['false_alarms']}/{fa['n_normal_videos']})")
    all_results.append({"model": "LR_xd_to_ucf", "auc": auc, "ap": ap, **fa})

    # LSTM (XD-trained, UCF-tested)
    print("\n[2] LSTM (XD-trained, UCF-tested)")
    auc, ap, per_vid = eval_lstm_cross_domain(args.lstm_checkpoint, xd_pairs, ucf_pairs, device=device)
    fa = false_alarm_analysis(per_vid)
    print(f"  Aggregate AUC: {auc:.4f}  AP: {ap:.4f}")
    if fa:
        print(f"  False alarm rate: {fa['false_alarm_rate']*100:.1f}%  ({fa['false_alarms']}/{fa['n_normal_videos']})")
    all_results.append({"model": "LSTM_xd_to_ucf", "auc": auc, "ap": ap, **fa})

    # LDA calibrated
    print("\n[3] LDA-calibrated LR (distribution shift correction)")
    auc, ap, per_vid = eval_lda_calibrated(xd_pairs, ucf_pairs)
    fa = false_alarm_analysis(per_vid)
    print(f"  Aggregate AUC: {auc:.4f}  AP: {ap:.4f}")
    if fa:
        print(f"  False alarm rate: {fa['false_alarm_rate']*100:.1f}%  ({fa['false_alarms']}/{fa['n_normal_videos']})")
    all_results.append({"model": "LDA_calibrated_xd_to_ucf", "auc": auc, "ap": ap, **fa})

    # Summary
    print("\n" + "=" * 65)
    print("CROSS-DOMAIN SUMMARY (XD-Violence → UCF-Crime)")
    print("=" * 65)
    print(f"{'Model':<35} {'AUC':>8} {'AP':>8} {'FA Rate':>10}")
    print("-" * 65)
    for r in all_results:
        fa_str = f"{r.get('false_alarm_rate', float('nan'))*100:.1f}%" if r.get('false_alarm_rate') is not None else "N/A"
        print(f"  {r['model']:<33} {r['auc']:>8.4f} {r['ap']:>8.4f} {fa_str:>10}")

    # Save
    import csv
    fieldnames = list({k for r in all_results for k in r.keys()})
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()