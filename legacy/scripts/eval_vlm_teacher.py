#!/usr/bin/env python3
"""Evaluate VLM teacher violence scores (JSONL) against GT labels (.npy).

Global metrics and plots use all usable (score, gt) pairs; per-video aggregates skip degenerate videos.
Outputs in --out_dir: summary.json, per_video.csv, f1_vs_threshold.png, pr_curve.png, roc_curve.png.
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    precision_recall_fscore_support,
    f1_score,
)


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def extract_score_and_source(rec: Dict[str, Any]) -> Tuple[Optional[float], str]:
    source = str(rec.get("source", "unknown"))
    v = rec.get("violent", None)

    if isinstance(v, dict):
        s = safe_float(v.get("score", None), default=None)
        if s is None:
            lab = str(v.get("label", "uncertain")).lower()
            if "yes" in lab:
                s = 1.0
            elif "no" in lab:
                s = 0.0
            else:
                s = 0.5
        return float(np.clip(s, 0.0, 1.0)), source

    s = safe_float(v, default=None)
    if s is not None:
        return float(np.clip(s, 0.0, 1.0)), source

    return None, source


def iter_jsonl_records(jsonl_path: str):
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def find_jsonl_files(root: str) -> List[str]:
    rootp = Path(root)
    return [str(p) for p in rootp.rglob("*.jsonl")]


def _auc_safe(y: np.ndarray, s: np.ndarray) -> Tuple[float, float]:
    """(roc_auc, pr_auc), or NaNs when y is empty or single-class."""
    if y.size == 0:
        return float("nan"), float("nan")
    if np.unique(y).size < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def compute_baselines(pos_rate: float) -> Dict[str, float]:
    """Precision/recall/F1 of always-positive and always-negative predictors."""
    ap_prec = float(pos_rate)
    ap_rec = 1.0 if pos_rate > 0 else 0.0
    ap_f1 = (2 * ap_prec * ap_rec / (ap_prec + ap_rec)) if (ap_prec + ap_rec) > 0 else 0.0

    an_prec = 0.0
    an_rec = 0.0
    an_f1 = 0.0

    return {
        "always_pos_precision": ap_prec,
        "always_pos_recall": ap_rec,
        "always_pos_f1": ap_f1,
        "always_neg_precision": an_prec,
        "always_neg_recall": an_rec,
        "always_neg_f1": an_f1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl_root", type=str, default="data/processed/vlm_labels")
    ap.add_argument("--labels_dir", type=str, default="data/processed/labels")
    ap.add_argument("--out_dir", type=str, default="results/vlm_eval")
    ap.add_argument("--gt_thresh", type=float, default=0.5, help="Binarize GT labels as (y >= gt_thresh)")
    ap.add_argument("--num_thresholds", type=int, default=101, help="Threshold sweep points in [0,1]")
    ap.add_argument("--skip_parse_failed", action="store_true", help="Skip records with error=parse_failed")
    ap.add_argument("--min_samples_per_video", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    jsonl_files = find_jsonl_files(args.jsonl_root)
    if not jsonl_files:
        raise RuntimeError(f"No .jsonl files found under {args.jsonl_root}")

    global_scores: List[float] = []
    global_gts: List[int] = []
    global_sources: List[str] = []
    global_vids: List[str] = []

    per_vid_scores: Dict[str, List[float]] = {}
    per_vid_gts: Dict[str, List[int]] = {}
    per_vid_source: Dict[str, str] = {}

    n_records = 0
    n_used = 0
    n_skip_no_label = 0
    n_skip_oob = 0
    n_skip_parse_failed = 0
    n_skip_no_score = 0

    for jp in sorted(jsonl_files):
        vid_id_from_file = Path(jp).stem

        label_path = os.path.join(args.labels_dir, f"{vid_id_from_file}.npy")
        label_arr = None
        if os.path.exists(label_path):
            label_arr = np.load(label_path).astype(np.float32)

        for rec in iter_jsonl_records(jp):
            n_records += 1

            if args.skip_parse_failed and rec.get("error") == "parse_failed":
                n_skip_parse_failed += 1
                continue

            vid_id = str(rec.get("vid_id", vid_id_from_file))
            t = rec.get("t", None)
            t = int(t) if t is not None else None
            if t is None or t < 0:
                continue

            score, source = extract_score_and_source(rec)
            if score is None:
                n_skip_no_score += 1
                continue

            lp2 = os.path.join(args.labels_dir, f"{vid_id}.npy")
            if label_arr is None or lp2 != label_path:
                if not os.path.exists(lp2):
                    n_skip_no_label += 1
                    continue
                label_path = lp2
                label_arr = np.load(label_path).astype(np.float32)

            if t >= len(label_arr):
                n_skip_oob += 1
                continue

            gt_val = float(label_arr[t])
            gt_bin = 1 if gt_val >= args.gt_thresh else 0

            global_scores.append(score)
            global_gts.append(gt_bin)
            global_sources.append(source)
            global_vids.append(vid_id)

            per_vid_scores.setdefault(vid_id, []).append(score)
            per_vid_gts.setdefault(vid_id, []).append(gt_bin)
            per_vid_source.setdefault(vid_id, source)

            n_used += 1

    if n_used == 0:
        raise RuntimeError("No usable (score, gt) pairs found. Check label names and JSONL structure.")

    scores_all = np.asarray(global_scores, dtype=np.float32)
    gts_all = np.asarray(global_gts, dtype=np.int32)

    pos_rate = float(np.mean(gts_all)) if gts_all.size else float("nan")
    baselines = compute_baselines(pos_rate)

    roc_auc_all, pr_auc_all = _auc_safe(gts_all, scores_all)

    thresholds = np.linspace(0.0, 1.0, args.num_thresholds)
    f1s, precs, recs = [], [], []
    for th in thresholds:
        pred = (scores_all >= th).astype(np.int32)
        p, r, f1, _ = precision_recall_fscore_support(gts_all, pred, average="binary", zero_division=0)
        f1s.append(float(f1))
        precs.append(float(p))
        recs.append(float(r))

    f1s = np.asarray(f1s, dtype=np.float32)
    best_idx = int(np.argmax(f1s))
    best_th = float(thresholds[best_idx])
    best_f1 = float(f1s[best_idx])
    best_p = float(precs[best_idx])
    best_r = float(recs[best_idx])

    plt.figure()
    plt.plot(thresholds, f1s)
    plt.axvline(best_th)
    plt.title(f"F1 vs Threshold (ALL pairs)  best F1={best_f1:.3f} @ th={best_th:.2f}")
    plt.xlabel("Prediction threshold")
    plt.ylabel("F1 score")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "f1_vs_threshold.png"), dpi=200)
    plt.close()

    if np.unique(gts_all).size == 2:
        precision, recall, _ = precision_recall_curve(gts_all, scores_all)
        fpr, tpr, _ = roc_curve(gts_all, scores_all)

        plt.figure()
        plt.plot(recall, precision)
        title = f"Precision-Recall (ALL pairs)  AP={pr_auc_all:.4f}" if np.isfinite(pr_auc_all) else "Precision-Recall (ALL pairs)"
        plt.title(title)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "pr_curve.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.plot(fpr, tpr)
        title = f"ROC (ALL pairs)  AUC={roc_auc_all:.4f}" if np.isfinite(roc_auc_all) else "ROC (ALL pairs)"
        plt.title(title)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "roc_curve.png"), dpi=200)
        plt.close()

    per_rows = []
    for vid_id, s_list in per_vid_scores.items():
        y_list = per_vid_gts.get(vid_id, [])
        s = np.asarray(s_list, dtype=np.float32)
        y = np.asarray(y_list, dtype=np.int32)
        n_samples = int(len(s))

        pred = (s >= best_th).astype(np.int32) if n_samples else np.asarray([], dtype=np.int32)
        f1 = float(f1_score(y, pred, zero_division=0)) if n_samples else float("nan")
        p, r, _, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0) if n_samples else (float("nan"), float("nan"), None, None)
        p = float(p) if np.isfinite(p) else float("nan")
        r = float(r) if np.isfinite(r) else float("nan")

        roc_v, pr_v = _auc_safe(y, s)

        invalid_reasons = []
        if n_samples < args.min_samples_per_video:
            invalid_reasons.append("too_few_samples")
        if not np.isfinite(roc_v):
            invalid_reasons.append("roc_nan_or_single_class")
        if not np.isfinite(pr_v):
            invalid_reasons.append("pr_nan_or_single_class")

        valid_for_video_aggs = 1 if len(invalid_reasons) == 0 else 0

        per_rows.append({
            "vid_id": vid_id,
            "source": per_vid_source.get(vid_id, "unknown"),
            "n_samples": n_samples,
            "best_threshold_global": best_th,
            "f1_at_best_th": f1,
            "precision_at_best_th": p,
            "recall_at_best_th": r,
            "roc_auc": roc_v,
            "pr_auc": pr_v,
            "mean_score": float(np.mean(s)) if n_samples else float("nan"),
            "pos_rate_gt": float(np.mean(y)) if n_samples else float("nan"),
            "valid_for_video_aggs": valid_for_video_aggs,
            "invalid_reasons": ",".join(invalid_reasons),
        })

    per_rows.sort(key=lambda r: (r["valid_for_video_aggs"], r["f1_at_best_th"] if np.isfinite(r["f1_at_best_th"]) else -1.0), reverse=True)

    per_csv = os.path.join(args.out_dir, "per_video.csv")
    fieldnames = list(per_rows[0].keys()) if per_rows else ["vid_id"]
    with open(per_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for row in per_rows:
            w.writerow(row)

    valid_rows = [r for r in per_rows if r["valid_for_video_aggs"] == 1]

    def _mean_or_nan(vals):
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    video_aggs = {
        "n_videos_total": int(len(per_rows)),
        "n_videos_valid_for_video_aggs": int(len(valid_rows)),
        "mean_f1_valid": _mean_or_nan([r["f1_at_best_th"] for r in valid_rows]),
        "mean_precision_valid": _mean_or_nan([r["precision_at_best_th"] for r in valid_rows]),
        "mean_recall_valid": _mean_or_nan([r["recall_at_best_th"] for r in valid_rows]),
        "mean_roc_auc_valid": _mean_or_nan([r["roc_auc"] for r in valid_rows]),
        "mean_pr_auc_valid": _mean_or_nan([r["pr_auc"] for r in valid_rows]),
    }

    per_source = {}
    for src in sorted(set(global_sources)):
        mask = np.asarray([s == src for s in global_sources], dtype=bool)
        ys = gts_all[mask]
        ss = scores_all[mask]
        pr = float(np.mean(ys)) if ys.size else float("nan")
        roc_s, pr_s = _auc_safe(ys, ss)
        per_source[src] = {
            "n_pairs": int(ys.size),
            "pos_rate": pr,
            "roc_auc": roc_s,
            "pr_auc": pr_s,
        }

    baseline_f1 = baselines["always_pos_f1"]
    baseline_gap = float(best_f1 - baseline_f1)
    looks_like_allpos = (best_th <= 1e-9) and (abs(baseline_gap) <= 1e-3)

    summary = {
        "jsonl_root": args.jsonl_root,
        "labels_dir": args.labels_dir,
        "out_dir": args.out_dir,
        "gt_thresh": args.gt_thresh,
        "num_thresholds": args.num_thresholds,
        "min_samples_per_video": args.min_samples_per_video,
        "counts": {
            "jsonl_files": len(jsonl_files),
            "records_seen": n_records,
            "pairs_used_total": n_used,
            "skip_no_label": n_skip_no_label,
            "skip_oob_t": n_skip_oob,
            "skip_parse_failed": n_skip_parse_failed,
            "skip_no_score": n_skip_no_score,
        },
        "global_metrics_all_pairs": {
            "roc_auc": roc_auc_all,
            "pr_auc": pr_auc_all,
            "best_threshold_by_f1": best_th,
            "best_f1": best_f1,
            "best_precision": best_p,
            "best_recall": best_r,
            "n_samples": int(scores_all.shape[0]),
            "pos_rate_gt": pos_rate,
        },
        "global_baselines": baselines,
        "best_minus_always_pos_f1": baseline_gap,
        "warning_best_is_all_positive_baseline": bool(looks_like_allpos),
        "per_source_all_pairs": per_source,
        "video_aggregates_valid_only": video_aggs,
        "notes": [
            "Global metrics/plots are computed on ALL usable (score,gt) pairs (no per-video filtering).",
            "Per-video CSV still includes ALL videos. Many videos may have NaN AUC if their GT is single-class.",
            "Always-positive baseline is included to detect when 'best F1' is just class-imbalance gaming.",
        ],
    }

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== VLM Teacher Eval (UPDATED) =====")
    print(f"[Data] jsonl_files={len(jsonl_files)} records_seen={n_records} used_pairs={n_used}")
    print(f"[Global ALL] pos_rate={pos_rate:.4f}")
    if np.isfinite(roc_auc_all):
        print(f"[Global ALL] ROC-AUC={roc_auc_all:.4f}  PR-AUC={pr_auc_all:.4f}")
    else:
        print("[Global ALL] ROC-AUC/PR-AUC = N/A (global pool missing both classes)")
    print(f"[Best F1]  F1={best_f1:.4f} @ th={best_th:.2f} (P={best_p:.4f}, R={best_r:.4f})")
    print(f"[Baseline always+ ] F1={baselines['always_pos_f1']:.4f} (P={baselines['always_pos_precision']:.4f}, R=1.0000)")
    if looks_like_allpos:
        print("[WARN] Best F1 matches ALWAYS-POSITIVE baseline (class imbalance / OR-window effect).")
    print(f"[CSV] {per_csv}")
    print(f"[Out] {args.out_dir}")
    print("=====================================\n")


if __name__ == "__main__":
    main()
