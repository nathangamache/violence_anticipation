"""
domain_adaptation.py
=====================
Systematic domain adaptation experiments to address the cross-domain
generalization failure (XD-Violence → UCF-Crime / YouTube).

Last session found:
  - Cross-domain AUC near 0.5 (random chance)
  - YouTube embeddings mean LDA score = -5.13 (outside both training classes)
  - 2,141 / 3,584 dimensions have Cohen's d > 0.5 between domains

Three approaches here:
  1. TCA (Transfer Component Analysis) — finds a subspace where source
     and target distributions are aligned. Computationally light.

  2. CORAL (CORrelation ALignment) — aligns second-order statistics.
     Already implemented in lda_calibration_complete.py but here we
     do a proper cross-domain eval, not just YouTube.

  3. Batch normalization adaptation — normalizes target domain embeddings
     using running statistics estimated from the target domain itself
     (test-time adaptation, no labels needed in target domain).

  4. Subspace-only classifier — trains only on dimensions where both
     domains overlap (low shift dimensions), discarding domain-specific ones.

Usage:
    python domain_adaptation.py \
        --source_embeddings data/processed/vlm_emb \
        --source_labels data/processed/labels_multih \
        --target_embeddings data/processed/vlm_emb/ucf_crime \
        --target_labels data/processed/labels_multih/ucf_crime

Or for YouTube single-video test:
    python domain_adaptation.py \
        --mode youtube \
        --source_embeddings data/processed/vlm_emb \
        --source_labels data/processed/labels_multih \
        --yt_embeddings probe_clip_01_embeddings.npy
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def load_pairs(emb_dir: Path, labels_dir: Path, max_videos: int = 300):
    pairs = []
    for vf in sorted(emb_dir.rglob("*.npz"))[:max_videos]:
        vid = vf.stem
        for candidate in [
            labels_dir / f"{vid}.npy",
            labels_dir / "xd_violence" / f"{vid}.npy",
            labels_dir / "ucf_crime" / f"{vid}.npy",
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


def flatten(pairs):
    return np.vstack([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])


# ──────────────────────────────────────────────
# 1. Subspace-only classifier
#    (train only on low-shift dimensions)
# ──────────────────────────────────────────────

def find_low_shift_dims(
    src_emb: np.ndarray,
    tgt_emb: np.ndarray,
    max_shift: float = 0.3,
    top_k: Optional[int] = None,
) -> np.ndarray:
    """
    Find dimensions where the source and target distributions overlap well.
    Uses normalized mean shift (Cohen's d style) as the shift measure.
    Returns array of dimension indices.
    """
    src_mean = src_emb.mean(axis=0)
    tgt_mean = tgt_emb.mean(axis=0)
    src_std = src_emb.std(axis=0) + 1e-8
    tgt_std = tgt_emb.std(axis=0) + 1e-8
    pooled_std = np.sqrt((src_std ** 2 + tgt_std ** 2) / 2)
    shift = np.abs(src_mean - tgt_mean) / pooled_std

    if top_k is not None:
        # Select top_k least-shifted dimensions
        return np.argsort(shift)[:top_k]
    else:
        return np.where(shift <= max_shift)[0]


def eval_subspace_classifier(
    src_pairs, tgt_pairs, max_shift=0.3, top_k=None, name="Subspace"
) -> Dict:
    X_src, y_src = flatten(src_pairs)
    X_tgt, y_tgt = flatten(tgt_pairs)

    low_shift = find_low_shift_dims(X_src, X_tgt, max_shift=max_shift, top_k=top_k)
    n_dims = len(low_shift)

    if n_dims == 0:
        print(f"  No dimensions below shift threshold {max_shift}")
        return {"method": name, "auc": float("nan"), "ap": float("nan"), "n_dims": 0}

    X_src_sub = X_src[:, low_shift]
    X_tgt_sub = X_tgt[:, low_shift]

    scaler = StandardScaler()
    X_src_s = scaler.fit_transform(X_src_sub)
    X_tgt_s = scaler.transform(X_tgt_sub)

    clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")
    clf.fit(X_src_s, y_src)

    scores = clf.predict_proba(X_tgt_s)[:, 1]
    auc = roc_auc_score(y_tgt, scores) if len(np.unique(y_tgt)) > 1 else float("nan")
    ap = average_precision_score(y_tgt, scores) if len(np.unique(y_tgt)) > 1 else float("nan")

    return {"method": name, "auc": auc, "ap": ap, "n_dims": n_dims}


# ──────────────────────────────────────────────
# 2. CORAL domain adaptation
# ──────────────────────────────────────────────

def coral_transform(src_emb: np.ndarray, tgt_emb: np.ndarray, n_pca_dims: int = 256) -> np.ndarray:
    """
    Apply CORAL to align target domain to source.
    Returns target embeddings in the aligned PCA subspace.
    """
    pca = PCA(n_components=n_pca_dims, random_state=42)
    src_pca = pca.fit_transform(src_emb).astype(np.float64)
    tgt_pca = pca.transform(tgt_emb).astype(np.float64)

    C_src = np.cov(src_pca.T) + np.eye(n_pca_dims) * 1e-4
    C_tgt = np.cov(tgt_pca.T) + np.eye(n_pca_dims) * 1e-4

    from numpy.linalg import eigh

    def matrix_sqrt(C):
        vals, vecs = eigh(C)
        return vecs @ np.diag(np.sqrt(np.maximum(vals, 0))) @ vecs.T

    def matrix_invsqrt(C):
        vals, vecs = eigh(C)
        return vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-10))) @ vecs.T

    A = matrix_invsqrt(C_tgt) @ matrix_sqrt(C_src)
    tgt_aligned = (tgt_pca @ A).astype(np.float32)
    return src_pca.astype(np.float32), tgt_aligned


def eval_coral(src_pairs, tgt_pairs, n_pca_dims=256) -> Dict:
    X_src, y_src = flatten(src_pairs)
    X_tgt, y_tgt = flatten(tgt_pairs)

    try:
        src_pca, tgt_aligned = coral_transform(X_src, X_tgt, n_pca_dims)
    except Exception as e:
        print(f"  CORAL failed: {e}")
        return {"method": "CORAL", "auc": float("nan"), "ap": float("nan")}

    scaler = StandardScaler()
    X_src_s = scaler.fit_transform(src_pca)
    X_tgt_s = scaler.transform(tgt_aligned)

    clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")
    clf.fit(X_src_s, y_src)

    scores = clf.predict_proba(X_tgt_s)[:, 1]
    auc = roc_auc_score(y_tgt, scores) if len(np.unique(y_tgt)) > 1 else float("nan")
    ap = average_precision_score(y_tgt, scores) if len(np.unique(y_tgt)) > 1 else float("nan")

    return {"method": "CORAL", "auc": auc, "ap": ap, "n_pca_dims": n_pca_dims}


# ──────────────────────────────────────────────
# 3. TCA (Transfer Component Analysis)
# ──────────────────────────────────────────────

def tca_transform(
    src_emb: np.ndarray,
    tgt_emb: np.ndarray,
    n_components: int = 64,
    mu: float = 1.0,
    kernel: str = "rbf",
    sigma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplified TCA: find a subspace (via kernel PCA) where the MMD between
    source and target is minimized.

    This is an approximation of Pan et al.'s TCA using kernel alignment.
    Full TCA requires solving a generalized eigenvalue problem — this version
    uses PCA on the concatenated aligned features as a practical substitute
    that captures the main idea.

    For a full TCA implementation, see:
    Pan, S.J. et al. (2011). Domain Adaptation via Transfer Component Analysis.
    """
    n_src = len(src_emb)
    n_tgt = len(tgt_emb)
    n = n_src + n_tgt

    # Reduce dims first for tractability
    pre_pca = PCA(n_components=min(256, src_emb.shape[1] // 4), random_state=42)
    all_emb = np.vstack([src_emb, tgt_emb])
    all_reduced = pre_pca.fit_transform(all_emb).astype(np.float64)

    src_r = all_reduced[:n_src]
    tgt_r = all_reduced[n_src:]

    # MMD-based alignment: normalize each domain, then PCA on combined
    scaler_src = StandardScaler()
    scaler_tgt = StandardScaler()
    src_norm = scaler_src.fit_transform(src_r)
    tgt_norm = scaler_tgt.fit_transform(tgt_r)

    # Re-color target to match source statistics (simple mean/cov match)
    # This is essentially CORAL in the pre-PCA space, which TCA also achieves
    src_mean = src_norm.mean(axis=0)
    src_std = src_norm.std(axis=0) + 1e-8
    tgt_aligned = (tgt_norm - tgt_norm.mean(axis=0)) / (tgt_norm.std(axis=0) + 1e-8) * src_std + src_mean

    # PCA on aligned combined features
    combined = np.vstack([src_norm, tgt_aligned])
    pca_final = PCA(n_components=n_components, random_state=42)
    pca_final.fit(combined)

    src_tca = pca_final.transform(src_norm).astype(np.float32)
    tgt_tca = pca_final.transform(tgt_aligned).astype(np.float32)

    return src_tca, tgt_tca


def eval_tca(src_pairs, tgt_pairs, n_components=64) -> Dict:
    X_src, y_src = flatten(src_pairs)
    X_tgt, y_tgt = flatten(tgt_pairs)

    try:
        src_tca, tgt_tca = tca_transform(X_src, X_tgt, n_components=n_components)
    except Exception as e:
        print(f"  TCA failed: {e}")
        return {"method": "TCA", "auc": float("nan"), "ap": float("nan")}

    scaler = StandardScaler()
    X_src_s = scaler.fit_transform(src_tca)
    X_tgt_s = scaler.transform(tgt_tca)

    clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")
    clf.fit(X_src_s, y_src)

    scores = clf.predict_proba(X_tgt_s)[:, 1]
    auc = roc_auc_score(y_tgt, scores) if len(np.unique(y_tgt)) > 1 else float("nan")
    ap = average_precision_score(y_tgt, scores) if len(np.unique(y_tgt)) > 1 else float("nan")

    return {"method": f"TCA (k={n_components})", "auc": auc, "ap": ap}


# ──────────────────────────────────────────────
# 4. Test-time batch normalization adaptation
#    (no labels needed in target domain)
# ──────────────────────────────────────────────

def eval_tta_batchnorm(src_pairs, tgt_pairs) -> Dict:
    """
    Test-Time Adaptation via feature normalization.
    Normalizes target domain using its own mean/std per feature dimension,
    then maps to source distribution before applying source-trained classifier.

    This is the simplest form of TTA and requires zero target labels.
    """
    X_src, y_src = flatten(src_pairs)
    X_tgt, y_tgt = flatten(tgt_pairs)

    # Train on source
    scaler_src = StandardScaler()
    X_src_s = scaler_src.fit_transform(X_src)
    clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")
    clf.fit(X_src_s, y_src)

    # Adapt target: normalize per-dimension to source distribution
    # Step 1: normalize target to zero mean, unit std
    scaler_tgt = StandardScaler()
    X_tgt_norm = scaler_tgt.fit_transform(X_tgt)

    # Step 2: re-scale to source mean/std
    X_tgt_adapted = X_tgt_norm * scaler_src.scale_ + scaler_src.mean_

    scores = clf.predict_proba(X_tgt_adapted)[:, 1]
    auc = roc_auc_score(y_tgt, scores) if len(np.unique(y_tgt)) > 1 else float("nan")
    ap = average_precision_score(y_tgt, scores) if len(np.unique(y_tgt)) > 1 else float("nan")

    return {"method": "TTA BatchNorm", "auc": auc, "ap": ap}


# ──────────────────────────────────────────────
# Naive baseline (no adaptation)
# ──────────────────────────────────────────────

def eval_no_adaptation(src_pairs, tgt_pairs) -> Dict:
    X_src, y_src = flatten(src_pairs)
    X_tgt, y_tgt = flatten(tgt_pairs)
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")
    clf.fit(scaler.fit_transform(X_src), y_src)
    scores = clf.predict_proba(scaler.transform(X_tgt))[:, 1]
    auc = roc_auc_score(y_tgt, scores) if len(np.unique(y_tgt)) > 1 else float("nan")
    ap = average_precision_score(y_tgt, scores) if len(np.unique(y_tgt)) > 1 else float("nan")
    return {"method": "No adaptation", "auc": auc, "ap": ap}


# ──────────────────────────────────────────────
# YouTube single-video mode
# ──────────────────────────────────────────────

def eval_youtube_mode(src_pairs, yt_emb_path: Path):
    """
    Test all adaptation methods on the YouTube video with known timestamps.
    Reports per-period score means rather than AUC (since ground truth is
    soft / period-level, not per-frame).
    """
    if not yt_emb_path.exists():
        print(f"YouTube embeddings not found: {yt_emb_path}")
        return

    yt_emb = np.load(yt_emb_path).astype(np.float32)
    X_src, y_src = flatten(src_pairs)

    print(f"\nYouTube embeddings: {yt_emb.shape}")

    periods = {
        "safe":         (0, 7),
        "anticipation": (7, 37),
        "violence":     (37, 46),
        "post":         (46, len(yt_emb)),
    }

    def period_means(scores):
        return {k: float(scores[s:e].mean()) for k, (s, e) in periods.items() if e <= len(scores)}

    def pattern_score(means):
        s = 0
        if means.get("violence", 0) > means.get("safe", 1): s += 2
        if means.get("anticipation", 0) > means.get("safe", 1): s += 1
        if means.get("violence", 0) > means.get("anticipation", 0): s += 1
        if means.get("post", 1) < means.get("violence", 0): s += 1
        return s

    results = []

    methods = [
        ("No adaptation", lambda X_s, y_s, X_t: _lr_scores(X_s, y_s, X_t)),
        ("TTA BatchNorm",  lambda X_s, y_s, X_t: _tta_scores(X_s, y_s, X_t)),
        ("CORAL",          lambda X_s, y_s, X_t: _coral_scores(X_s, y_s, X_t)),
        ("Low-shift dims", lambda X_s, y_s, X_t: _subspace_scores(X_s, y_s, X_t)),
        ("Relative (within-video)", lambda X_s, y_s, X_t: _relative_scores(X_s, y_s, X_t)),
    ]

    print(f"\n{'Method':<28}  {'Safe':>6} {'Anticip':>8} {'Violnc':>8} {'Post':>6}  Score")
    print("-" * 70)

    for name, method in methods:
        try:
            scores = method(X_src, y_src, yt_emb)
            means = period_means(scores)
            ps = pattern_score(means)
            print(f"  {name:<26}  {means.get('safe', float('nan')):>6.3f} "
                  f"{means.get('anticipation', float('nan')):>8.3f} "
                  f"{means.get('violence', float('nan')):>8.3f} "
                  f"{means.get('post', float('nan')):>6.3f}  {ps}/5")
            results.append((name, means, ps))
        except Exception as e:
            print(f"  {name:<26}  ERROR: {e}")

    best = max(results, key=lambda r: r[2]) if results else None
    if best:
        print(f"\n  Best method: {best[0]} (score {best[2]}/5)")


def _lr_scores(X_src, y_src, X_tgt):
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")
    clf.fit(scaler.fit_transform(X_src), y_src)
    return clf.predict_proba(scaler.transform(X_tgt))[:, 1].astype(np.float32)


def _tta_scores(X_src, y_src, X_tgt):
    scaler_src = StandardScaler()
    clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")
    clf.fit(scaler_src.fit_transform(X_src), y_src)
    scaler_tgt = StandardScaler()
    X_tgt_norm = scaler_tgt.fit_transform(X_tgt)
    X_tgt_adapted = X_tgt_norm * scaler_src.scale_ + scaler_src.mean_
    return clf.predict_proba(X_tgt_adapted)[:, 1].astype(np.float32)


def _coral_scores(X_src, y_src, X_tgt):
    src_pca, tgt_aligned = coral_transform(X_src, X_tgt, n_pca_dims=min(256, X_src.shape[1] // 4))
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")
    clf.fit(scaler.fit_transform(src_pca), y_src)
    return clf.predict_proba(scaler.transform(tgt_aligned))[:, 1].astype(np.float32)


def _subspace_scores(X_src, y_src, X_tgt):
    low_dims = find_low_shift_dims(X_src, X_tgt, top_k=512)
    X_src_sub = X_src[:, low_dims]
    X_tgt_sub = X_tgt[:, low_dims]
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")
    clf.fit(scaler.fit_transform(X_src_sub), y_src)
    return clf.predict_proba(scaler.transform(X_tgt_sub))[:, 1].astype(np.float32)


def _relative_scores(X_src, y_src, X_tgt):
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X_src, y_src)
    yt_lda = lda.transform(X_tgt).flatten()
    yt_norm = (yt_lda - yt_lda.mean()) / (yt_lda.std() + 1e-8)
    return (1.0 / (1.0 + np.exp(-yt_norm))).astype(np.float32)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cross_domain", "youtube"], default="cross_domain")
    parser.add_argument("--source_embeddings", type=Path, default=Path("data/processed/vlm_emb"))
    parser.add_argument("--source_labels", type=Path, default=Path("data/processed/labels_multih"))
    parser.add_argument("--target_embeddings", type=Path, default=Path("data/processed/vlm_emb/ucf_crime"))
    parser.add_argument("--target_labels", type=Path, default=None)
    parser.add_argument("--yt_embeddings", type=Path, default=Path("probe_clip_01_embeddings.npy"))
    parser.add_argument("--max_source", type=int, default=300)
    parser.add_argument("--max_target", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("results/domain_adaptation.csv"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("DOMAIN ADAPTATION EXPERIMENT")
    print("=" * 65)

    print(f"\nLoading source domain (XD-Violence)...")
    src_pairs = load_pairs(args.source_embeddings, args.source_labels, args.max_source)
    print(f"  Loaded {len(src_pairs)} source videos")

    if args.mode == "youtube":
        eval_youtube_mode(src_pairs, args.yt_embeddings)
        return

    print(f"Loading target domain (UCF-Crime)...")
    if not args.target_embeddings.exists():
        print(f"  Target dir not found: {args.target_embeddings}")
        print("  Falling back to YouTube mode.")
        eval_youtube_mode(src_pairs, args.yt_embeddings)
        return

    tgt_labels = args.target_labels or args.source_labels
    tgt_pairs = load_pairs(args.target_embeddings, tgt_labels, args.max_target)
    print(f"  Loaded {len(tgt_pairs)} target videos")

    if not src_pairs or not tgt_pairs:
        print("Not enough data.")
        return

    # Quantify shift
    X_src, _ = flatten(src_pairs)
    X_tgt, _ = flatten(tgt_pairs)
    scaler = StandardScaler()
    scaler.fit(X_src)
    shift = np.abs(scaler.transform(X_tgt).mean(axis=0))
    print(f"\nDistribution shift summary:")
    print(f"  Mean per-dim shift:   {shift.mean():.4f}")
    print(f"  Dims shifted > 0.5σ:  {(shift > 0.5).sum()}")
    print(f"  Dims shifted > 1.0σ:  {(shift > 1.0).sum()}")

    results = []

    print("\n[1] No adaptation (baseline)")
    r = eval_no_adaptation(src_pairs, tgt_pairs)
    results.append(r)
    print(f"  AUC={r['auc']:.4f}  AP={r['ap']:.4f}")

    print("[2] TTA BatchNorm (no labels needed)")
    r = eval_tta_batchnorm(src_pairs, tgt_pairs)
    results.append(r)
    print(f"  AUC={r['auc']:.4f}  AP={r['ap']:.4f}")

    print("[3] CORAL (covariance alignment)")
    r = eval_coral(src_pairs, tgt_pairs)
    results.append(r)
    print(f"  AUC={r['auc']:.4f}  AP={r['ap']:.4f}")

    print("[4] TCA (transfer component analysis)")
    r = eval_tca(src_pairs, tgt_pairs)
    results.append(r)
    print(f"  AUC={r['auc']:.4f}  AP={r['ap']:.4f}")

    print("[5] Subspace classifier (low-shift dims, max_shift=0.3)")
    r = eval_subspace_classifier(src_pairs, tgt_pairs, max_shift=0.3)
    results.append(r)
    print(f"  AUC={r['auc']:.4f}  AP={r['ap']:.4f}  dims={r.get('n_dims', '?')}")

    print("[6] Subspace classifier (top-512 least-shifted dims)")
    r = eval_subspace_classifier(src_pairs, tgt_pairs, top_k=512, name="Subspace top-512")
    results.append(r)
    print(f"  AUC={r['auc']:.4f}  AP={r['ap']:.4f}  dims=512")

    # Summary
    print("\n" + "=" * 65)
    print("DOMAIN ADAPTATION SUMMARY")
    print("=" * 65)
    results_sorted = sorted(results, key=lambda r: r.get("auc", 0) or 0, reverse=True)
    print(f"  {'Method':<35} {'AUC':>8} {'AP':>8}")
    print("  " + "-" * 55)
    for r in results_sorted:
        auc = r.get("auc", float("nan"))
        ap = r.get("ap", float("nan"))
        print(f"  {r['method']:<35} {auc:>8.4f} {ap:>8.4f}")

    # Save
    import csv
    fieldnames = ["method", "auc", "ap"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results_sorted)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()