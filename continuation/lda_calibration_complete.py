"""
lda_calibration_complete.py
============================
Completes the LDA calibration experiment that was left unfinished.

Last session confirmed:
  - Training LDA: violent mean=+2.85, non-violent mean=-3.34, separation=6.19
  - YouTube mean on LDA axis: -5.13 (far outside both training classes)
  - Relative ordering within YouTube was partially correct, but absolute shift
    made all frames score as non-violent

This script:
  1. Runs the original LDA calibration (z-score shift to training distribution)
  2. Tests three additional calibration strategies
  3. Runs a proper domain adaptation experiment (CORAL covariance alignment)
  4. Evaluates all strategies on the YouTube video with known ground truth

YouTube ground truth (probe_clip_01):
  0-6s   = safe/neutral
  7-36s  = anticipation (confrontation building)
  37-45s = violence
  46+s   = post-violence (people dispersing)

Usage:
    python lda_calibration_complete.py \
        --yt_embeddings probe_clip_01_embeddings.npy \
        --train_data simulation_data/xd_violence_vlm_clean_t02.npz
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


# ──────────────────────────────────────────────
# YouTube ground truth
# ──────────────────────────────────────────────

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
    """Return mean score for each of the 4 YouTube periods."""
    out = {}
    for period, (start, end) in YT_LABELS.items():
        end_idx = end if end is not None else len(scores)
        chunk = scores[start:end_idx]
        out[period] = float(chunk.mean()) if len(chunk) > 0 else float("nan")
    return out

def pattern_correct(means):
    """
    Ideal pattern: violence > anticipation > safe, post < violence.
    Returns (violence_highest, anticipation_rises).
    """
    violence_highest = (
        means["violence"] > means["safe"] and
        means["violence"] > means["post"]
    )
    anticipation_rises = means["anticipation"] > means["safe"]
    return violence_highest, anticipation_rises


# ──────────────────────────────────────────────
# Calibration methods
# ──────────────────────────────────────────────

def method_raw_lda(train_emb, train_labels, yt_emb):
    """LDA projection with no calibration — the baseline that failed."""
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
    """
    Z-score the YouTube LDA projection to match training distribution,
    then apply sigmoid. This is the approach proposed at end of last session.
    """
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(train_emb, train_labels).flatten()
    yt_lda = lda.transform(yt_emb).flatten()

    # Shift YouTube to have same mean and std as training LDA scores
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
    """
    Match YouTube LDA percentiles to training LDA percentiles
    (histogram equalization on the 1-D LDA projection).
    More robust than z-score when distributions are non-Gaussian.
    """
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(train_emb, train_labels).flatten()
    yt_lda = lda.transform(yt_emb).flatten()

    # Build mapping: YouTube percentile → training percentile
    percentiles = np.linspace(0, 100, n_bins + 1)
    train_pctls = np.percentile(X_lda, percentiles)
    yt_pctls = np.percentile(yt_lda, percentiles)

    # Map each YouTube LDA score to its training equivalent
    yt_calibrated = np.interp(yt_lda, yt_pctls, train_pctls)

    v_mean = X_lda[train_labels == 1].mean()
    nv_mean = X_lda[train_labels == 0].mean()
    midpoint = (v_mean + nv_mean) / 2
    scale = 4.0 / (abs(v_mean - nv_mean) + 1e-8)
    scores = 1.0 / (1.0 + np.exp(-scale * (yt_calibrated - midpoint)))
    return scores, yt_calibrated


def method_coral(train_emb, train_labels, yt_emb, n_dims=256):
    """
    CORAL (CORrelation ALignment) domain adaptation.
    Aligns the covariance structure of YouTube embeddings to training embeddings
    before applying LDA + LR.

    This goes further than LDA calibration — it operates in the full
    embedding space before any dimensionality reduction.

    Steps:
      1. Whiten YouTube embeddings (remove its covariance)
      2. Re-color with training covariance
      3. Apply LDA + sigmoid on the aligned embeddings

    Uses top n_dims PCA components for tractability (full 3584-dim covariance
    is expensive to invert).
    """
    from sklearn.decomposition import PCA

    # PCA for tractability
    pca = PCA(n_components=n_dims, random_state=42)
    train_pca = pca.fit_transform(train_emb).astype(np.float64)
    yt_pca = pca.transform(yt_emb).astype(np.float64)

    # Training covariance
    C_src = np.cov(train_pca.T) + np.eye(n_dims) * 1e-4  # regularize

    # YouTube covariance
    C_tgt = np.cov(yt_pca.T) + np.eye(n_dims) * 1e-4

    # CORAL transform: align YouTube to training
    # yt_aligned = yt_pca @ C_tgt^{-1/2} @ C_src^{1/2}
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

    # LDA + sigmoid on aligned features
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
    """
    Instead of absolute calibration, score each frame relative to the
    video's own distribution. This sidesteps cross-domain shift entirely
    by asking 'is this frame more violent-looking than typical frames
    in this same video?'

    Uses the LDA direction from training, but normalizes the YouTube
    scores by the YouTube video's own mean and std — centering around 0.5.
    """
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(train_emb, train_labels).flatten()
    yt_lda = lda.transform(yt_emb).flatten()

    # Within-video normalization
    yt_norm = (yt_lda - yt_lda.mean()) / (yt_lda.std() + 1e-8)

    # Map to [0, 1] via sigmoid around 0
    # Frames above the video's mean LDA score get probability > 0.5
    scores = 1.0 / (1.0 + np.exp(-yt_norm))
    return scores, yt_norm


# ──────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

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

    # Load data
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

    # ── Method 0: Raw LDA (baseline from last session) ──
    print("\n" + "─" * 55)
    print("[M0] Raw LDA (no calibration) — baseline")
    scores, X_lda, yt_lda, lda = method_raw_lda(train_emb, train_labels, yt_emb)
    means = period_means(scores)
    print_score_timeline(scores, "Per-second scores", n_seconds=min(60, len(scores)))
    print_period_summary(means, "Period means")
    results_log.append(("Raw LDA", means))

    # ── Method 1: Z-score calibration ───────────────────
    print("\n" + "─" * 55)
    print("[M1] Z-score calibration (shift YouTube → training distribution)")
    scores, yt_cal = method_zscore_calibration(train_emb, train_labels, yt_emb)
    means = period_means(scores)
    print_score_timeline(scores, "Per-second scores", n_seconds=min(60, len(scores)))
    print_period_summary(means, "Period means")
    results_log.append(("Z-score calibration", means))

    # ── Method 2: Percentile matching ───────────────────
    print("\n" + "─" * 55)
    print("[M2] Percentile matching (histogram equalization on LDA axis)")
    scores, yt_cal = method_percentile_matching(train_emb, train_labels, yt_emb)
    means = period_means(scores)
    print_score_timeline(scores, "Per-second scores", n_seconds=min(60, len(scores)))
    print_period_summary(means, "Period means")
    results_log.append(("Percentile matching", means))

    # ── Method 3: CORAL domain adaptation ───────────────
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

    # ── Method 4: Relative within-video ─────────────────
    print("\n" + "─" * 55)
    print("[M4] Relative scoring (within-video normalization)")
    scores, _ = method_relative_within_video(train_emb, train_labels, yt_emb)
    means = period_means(scores)
    print_score_timeline(scores, "Per-second scores", n_seconds=min(60, len(scores)))
    print_period_summary(means, "Period means")
    results_log.append(("Relative (within-video)", means))

    # ── Summary ──────────────────────────────────────────
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

    # Find best method
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

    # Save text log
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