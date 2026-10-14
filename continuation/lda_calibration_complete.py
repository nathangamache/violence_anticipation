"""Compare LDA calibration and CORAL strategies on the hand-labelled YouTube clip probe_clip_01.

Usage: python continuation/lda_calibration_complete.py [--yt_embeddings NPY] [--train_data NPZ] [--output TXT]
"""

import argparse
from pathlib import Path

import numpy as np
import warnings

warnings.filterwarnings("ignore")

try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("sklearn not available — some methods will be skipped")


# Hand labels for probe_clip_01 in seconds, end inclusive (period_means slices end-exclusive, dropping each last second).
YT_LABELS = {
    "safe":        (0, 6),
    "anticipation": (7, 36),
    "violence":    (37, 45),
    "post":        (46, None),
}

def get_yt_period(t):
    if t <= 6:
        return "safe"
    elif t <= 36:
        return "anticipation"
    elif t <= 45:
        return "violence"
    else:
        return "post"

def period_means(scores):
    out = {}
    for period, (start, end) in YT_LABELS.items():
        end_idx = end if end is not None else len(scores)
        chunk = scores[start:end_idx]
        out[period] = float(chunk.mean()) if len(chunk) > 0 else float("nan")
    return out

def pattern_correct(means):
    """Return (violence above safe and post, anticipation above safe)."""
    violence_highest = (
        means["violence"] > means["safe"] and
        means["violence"] > means["post"]
    )
    anticipation_rises = means["anticipation"] > means["safe"]
    return violence_highest, anticipation_rises


def method_raw_lda(train_emb, train_labels, yt_emb):
    """Uncalibrated LDA projection through a sigmoid (baseline)."""
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(train_emb, train_labels).flatten()
    yt_lda = lda.transform(yt_emb).flatten()

    v_mean = X_lda[train_labels == 1].mean()
    nv_mean = X_lda[train_labels == 0].mean()
    midpoint = (v_mean + nv_mean) / 2
    scale = 4.0 / (abs(v_mean - nv_mean) + 1e-8)
    scores = 1.0 / (1.0 + np.exp(-scale * (yt_lda - midpoint)))
    return scores, X_lda, yt_lda, lda


def method_zscore_calibration(train_emb, train_labels, yt_emb):
    """Shift and scale the YouTube LDA projection to the training mean/std, then sigmoid."""
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(train_emb, train_labels).flatten()
    yt_lda = lda.transform(yt_emb).flatten()

    yt_calibrated = (
        (yt_lda - yt_lda.mean()) / (yt_lda.std() + 1e-8)
        * X_lda.std() + X_lda.mean()
    )

    v_mean = X_lda[train_labels == 1].mean()
    nv_mean = X_lda[train_labels == 0].mean()
    midpoint = (v_mean + nv_mean) / 2
    scale = 4.0 / (abs(v_mean - nv_mean) + 1e-8)
    scores = 1.0 / (1.0 + np.exp(-scale * (yt_calibrated - midpoint)))
    return scores, yt_calibrated


def method_percentile_matching(train_emb, train_labels, yt_emb, n_bins=20):
    """Map YouTube LDA percentiles onto training percentiles (1-D histogram matching), then sigmoid."""
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(train_emb, train_labels).flatten()
    yt_lda = lda.transform(yt_emb).flatten()

    percentiles = np.linspace(0, 100, n_bins + 1)
    train_pctls = np.percentile(X_lda, percentiles)
    yt_pctls = np.percentile(yt_lda, percentiles)

    yt_calibrated = np.interp(yt_lda, yt_pctls, train_pctls)

    v_mean = X_lda[train_labels == 1].mean()
    nv_mean = X_lda[train_labels == 0].mean()
    midpoint = (v_mean + nv_mean) / 2
    scale = 4.0 / (abs(v_mean - nv_mean) + 1e-8)
    scores = 1.0 / (1.0 + np.exp(-scale * (yt_calibrated - midpoint)))
    return scores, yt_calibrated


def method_coral(train_emb, train_labels, yt_emb, n_dims=256):
    """CORAL: re-color YouTube embeddings with the training covariance in an n_dims PCA subspace, then LDA + sigmoid."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_dims, random_state=42)
    train_pca = pca.fit_transform(train_emb).astype(np.float64)
    yt_pca = pca.transform(yt_emb).astype(np.float64)

    C_src = np.cov(train_pca.T) + np.eye(n_dims) * 1e-4

    C_tgt = np.cov(yt_pca.T) + np.eye(n_dims) * 1e-4

    from numpy.linalg import eigh

    def matrix_sqrt(C):
        vals, vecs = eigh(C)
        vals = np.maximum(vals, 0)
        return vecs @ np.diag(np.sqrt(vals)) @ vecs.T

    def matrix_invsqrt(C):
        vals, vecs = eigh(C)
        vals = np.maximum(vals, 1e-10)
        return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

    C_tgt_invsqrt = matrix_invsqrt(C_tgt)
    C_src_sqrt = matrix_sqrt(C_src)
    A = C_tgt_invsqrt @ C_src_sqrt

    yt_aligned = (yt_pca @ A).astype(np.float32)
    train_pca_f = train_pca.astype(np.float32)

    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(train_pca_f, train_labels).flatten()
    yt_lda = lda.transform(yt_aligned).flatten()

    v_mean = X_lda[train_labels == 1].mean()
    nv_mean = X_lda[train_labels == 0].mean()
    midpoint = (v_mean + nv_mean) / 2
    scale = 4.0 / (abs(v_mean - nv_mean) + 1e-8)
    scores = 1.0 / (1.0 + np.exp(-scale * (yt_lda - midpoint)))
    return scores, yt_lda


def method_relative_within_video(train_emb, train_labels, yt_emb):
    """Standardize the YouTube LDA projection by the clip's own mean/std, sidestepping the absolute domain shift."""
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(train_emb, train_labels).flatten()
    yt_lda = lda.transform(yt_emb).flatten()

    yt_norm = (yt_lda - yt_lda.mean()) / (yt_lda.std() + 1e-8)

    scores = 1.0 / (1.0 + np.exp(-yt_norm))
    return scores, yt_norm


def print_score_timeline(scores, title, n_seconds=None):
    n = n_seconds or len(scores)
    print(f"\n  {title}")
    print(f"  {'Time':>4}  {'Score':>6}  {'Bar':<22}  Period")
    print("  " + "-" * 55)
    for t in range(min(n, len(scores))):
        period = get_yt_period(t)
        bar = "█" * int(scores[t] * 20)
        marker = f"[{period[:3].upper()}]"
        print(f"  {t:3d}s  {scores[t]:>6.3f}  {bar:<22}  {marker}")


def print_period_summary(means, title):
    print(f"\n  {title}")
    for period in ["safe", "anticipation", "violence", "post"]:
        v = means.get(period, float("nan"))
        bar = "█" * int(v * 20) if not np.isnan(v) else ""
        print(f"    {period:<14}: {v:.3f}  {bar}")

    v_high, ant_rise = pattern_correct(means)
    print(f"    Violence highest: {'OK' if v_high else 'FAIL'}")
    print(f"    Anticipation rises: {'OK' if ant_rise else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yt_embeddings", type=Path, default=Path("probe_clip_01_embeddings.npy"))
    parser.add_argument("--train_data", type=Path, default=Path("simulation_data/xd_violence_vlm_clean_t02.npz"))
    parser.add_argument("--output", type=Path, default=Path("results/lda_calibration_results.txt"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("LDA CALIBRATION — COMPLETE EXPERIMENT")
    print("=" * 65)

    if not args.yt_embeddings.exists():
        print(f"ERROR: YouTube embeddings not found at {args.yt_embeddings}")
        return
    if not args.train_data.exists():
        print(f"ERROR: Training data not found at {args.train_data}")
        return

    yt_emb = np.load(args.yt_embeddings).astype(np.float32)
    sim = np.load(args.train_data, allow_pickle=True)
    train_emb = sim["embeddings"].astype(np.float32)
    train_labels = (sim["frame_labels"] > 0).astype(int)

    print(f"\nYouTube embeddings: {yt_emb.shape}  ({len(yt_emb)} seconds)")
    print(f"Training embeddings: {train_emb.shape}")
    print(f"Training class balance: {train_labels.mean()*100:.1f}% violent")

    results_log = []

    print("\n" + "─" * 55)
    print("[M0] Raw LDA (no calibration) — baseline")
    scores, X_lda, yt_lda, lda = method_raw_lda(train_emb, train_labels, yt_emb)
    means = period_means(scores)
    print_score_timeline(scores, "Per-second scores", n_seconds=min(60, len(scores)))
    print_period_summary(means, "Period means")
    results_log.append(("Raw LDA", means))

    print("\n" + "─" * 55)
    print("[M1] Z-score calibration (shift YouTube → training distribution)")
    scores, yt_cal = method_zscore_calibration(train_emb, train_labels, yt_emb)
    means = period_means(scores)
    print_score_timeline(scores, "Per-second scores", n_seconds=min(60, len(scores)))
    print_period_summary(means, "Period means")
    results_log.append(("Z-score calibration", means))

    print("\n" + "─" * 55)
    print("[M2] Percentile matching (histogram equalization on LDA axis)")
    scores, yt_cal = method_percentile_matching(train_emb, train_labels, yt_emb)
    means = period_means(scores)
    print_score_timeline(scores, "Per-second scores", n_seconds=min(60, len(scores)))
    print_period_summary(means, "Period means")
    results_log.append(("Percentile matching", means))

    print("\n" + "─" * 55)
    print("[M3] CORAL domain adaptation (covariance alignment, top-256 PCA dims)")
    try:
        scores, yt_lda_coral = method_coral(train_emb, train_labels, yt_emb, n_dims=256)
        means = period_means(scores)
        print_score_timeline(scores, "Per-second scores", n_seconds=min(60, len(scores)))
        print_period_summary(means, "Period means")
        results_log.append(("CORAL", means))
    except Exception as e:
        print(f"  CORAL failed: {e}")

    print("\n" + "─" * 55)
    print("[M4] Relative scoring (within-video normalization)")
    scores, _ = method_relative_within_video(train_emb, train_labels, yt_emb)
    means = period_means(scores)
    print_score_timeline(scores, "Per-second scores", n_seconds=min(60, len(scores)))
    print_period_summary(means, "Period means")
    results_log.append(("Relative (within-video)", means))

    print("\n" + "=" * 65)
    print("SUMMARY: CALIBRATION COMPARISON")
    print("=" * 65)
    print(f"  {'Method':<30} {'Safe':>6} {'Anticip':>8} {'Violnc':>8} {'Post':>6}  {'V>S':>4} {'A>S':>4}")
    print("  " + "-" * 75)
    for name, means in results_log:
        v_high, ant_rise = pattern_correct(means)
        print(
            f"  {name:<30}"
            f"  {means.get('safe', float('nan')):>6.3f}"
            f"  {means.get('anticipation', float('nan')):>8.3f}"
            f"  {means.get('violence', float('nan')):>8.3f}"
            f"  {means.get('post', float('nan')):>6.3f}"
            f"  {'OK' if v_high else 'FAIL':>4}"
            f"  {'OK' if ant_rise else 'FAIL':>4}"
        )

    print()
    print("  Key: V>S = violence scores higher than safe")
    print("       A>S = anticipation scores higher than safe")
    print()

    def score_method(means):
        s = 0
        if means["violence"] > means["safe"]: s += 2
        if means["anticipation"] > means["safe"]: s += 1
        if means["violence"] > means["anticipation"]: s += 1
        if means["post"] < means["violence"]: s += 1
        return s

    ranked = sorted(results_log, key=lambda x: score_method(x[1]), reverse=True)
    best = ranked[0]
    print(f"  Best method: {best[0]}")

    with open(args.output, "w") as f:
        f.write("LDA Calibration Experiment Results\n")
        f.write("=" * 65 + "\n\n")
        for name, means in results_log:
            v_high, ant_rise = pattern_correct(means)
            f.write(f"{name}\n")
            for period, val in means.items():
                f.write(f"  {period}: {val:.4f}\n")
            f.write(f"  violence_highest: {v_high}\n")
            f.write(f"  anticipation_rises: {ant_rise}\n\n")

    print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()