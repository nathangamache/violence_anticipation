#!/usr/bin/env python3
"""Train the multi-horizon student MLP on VLM embeddings + hashed JSONL fields, aligned to label timestep t.

Inputs: emb_root/<source>/<vid>.npz (t, emb), jsonl_root/<source>/<vid>.jsonl, labels_dir/<vid>.npy [T,K].
"""

import argparse
import csv
import json
import math
import os
import random
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


def roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int32)
    if len(y_true) == 0 or y_true.min() == y_true.max():
        return float("nan")

    order = np.argsort(y_score)
    y = y_true[order]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    ranks = np.arange(1, len(y) + 1)
    sum_ranks_pos = ranks[y == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg + 1e-12)
    return float(auc)


def average_precision_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int32)
    if len(y_true) == 0 or y_true.min() == y_true.max():
        return float("nan")

    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp[-1] + 1e-12)

    rec_prev = np.concatenate([[0.0], rec[:-1]])
    ap = np.sum(prec * (rec - rec_prev))
    return float(ap)


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, th: float) -> dict:
    y_true = y_true.astype(np.int32)
    y_pred = (y_score >= th).astype(np.int32)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)

    tpr = recall
    fpr = fp / (fp + tn + 1e-12)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-12)
    mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0
    youden = tpr - fpr

    return dict(
        threshold=float(th),
        f1=float(f1),
        precision=float(precision),
        recall=float(recall),
        tp=tp, fp=fp, fn=fn, tn=tn,
        tpr=float(tpr),
        fpr=float(fpr),
        mcc=float(mcc),
        youden=float(youden),
        n_samples=int(len(y_true)),
        pos_rate=float(y_true.mean() if len(y_true) else 0.0),
    )


def sweep_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    num_thresholds: int = 101,
    th_min: float = 0.0,
    th_max: float = 0.999,
) -> List[dict]:
    """Metrics at evenly spaced thresholds in [th_min, th_max]."""
    out: List[dict] = []

    lo = float(max(0.0, min(1.0, th_min)))
    hi = float(max(0.0, min(1.0, th_max)))

    # Never include 1.0: sigmoid probs almost never reach it, so it degenerates to "never alert".
    hi = min(hi, 1.0 - 1e-6)
    if hi <= lo:
        hi = min(1.0 - 1e-6, lo + 1e-3)

    for th in np.linspace(lo, hi, num_thresholds, dtype=np.float32):
        out.append(metrics_at_threshold(y_true, y_score, float(th)))

    return out

def select_threshold_from_sweep(sweep: List[dict], mode: str,
                                precision_floor: float = 0.0,
                                f_beta: float = 1.0) -> dict:
    """Best sweep row by f1/youden/mcc/fbeta, or max recall at precision >= floor; ties -> higher threshold."""
    mode = (mode or "f1").lower()

    if mode == "precision_floor":
        pf = float(max(0.0, min(1.0, precision_floor)))
        candidates = [m for m in sweep if m["precision"] >= pf]
        if candidates:
            best = max(candidates, key=lambda d: (d["recall"], d["threshold"]))
            return best
        return max(sweep, key=lambda d: (d["precision"], d["recall"], d["threshold"]))

    if mode == "fbeta":
        b = float(f_beta)
        b = max(0.05, min(b, 5.0))
        b2 = b * b
        best = None
        best_val = -1e9
        for m in sweep:
            p = float(m["precision"]); r = float(m["recall"])
            f = (1.0 + b2) * p * r / (b2 * p + r + 1e-12)
            if (best is None) or (f > best_val) or (abs(f - best_val) < 1e-12 and m["threshold"] > best["threshold"]):
                best = m
                best_val = f
        return best or dict(threshold=0.5, f1=0.0, precision=0.0, recall=0.0)

    key = "f1" if mode == "f1" else ("youden" if mode == "youden" else "mcc")

    best = None
    for m in sweep:
        if best is None or m[key] > best[key]:
            best = m

    if best is not None:
        best_val = best[key]
        tied = [m for m in sweep if abs(m[key] - best_val) < 1e-12]
        if tied:
            best = max(tied, key=lambda d: d["threshold"])

    return best or dict(threshold=0.5, f1=0.0, precision=0.0, recall=0.0)

def baselines(y_true: np.ndarray) -> dict:
    y_true = y_true.astype(np.int32)
    p = float(y_true.mean()) if len(y_true) else 0.0

    ap_prec = p
    ap_rec = 1.0
    ap_f1 = (2 * ap_prec * ap_rec) / (ap_prec + ap_rec + 1e-12)

    an_prec = 0.0
    an_rec = 0.0
    an_f1 = 0.0

    return dict(
        always_pos_precision=float(ap_prec),
        always_pos_recall=float(ap_rec),
        always_pos_f1=float(ap_f1),
        always_neg_precision=float(an_prec),
        always_neg_recall=float(an_rec),
        always_neg_f1=float(an_f1),
        pos_rate=float(p),
    )


def evaluate_probs(
    y_true01: np.ndarray,
    probs: np.ndarray,
    num_thresholds: int = 101,
    threshold_select: str = "f1",
    th_min: float = 0.0,
    th_max: float = 0.999,
    precision_floor: float = 0.0,
    f_beta: float = 1.0
):
    roc = roc_auc_score_np(y_true01, probs)
    ap = average_precision_np(y_true01, probs)

    sweep = sweep_thresholds(
        y_true01,
        probs,
        num_thresholds=num_thresholds,
        th_min=th_min,
        th_max=th_max,
    )
    best = select_threshold_from_sweep(sweep, threshold_select, precision_floor=precision_floor, f_beta=f_beta)

    b = baselines(y_true01)
    best_minus_ap = float(best["f1"] - b["always_pos_f1"])
    warn_baseline = (best_minus_ap < 0.01)

    summary = dict(
        roc_auc=float(roc),
        pr_auc=float(ap),
        best_threshold=float(best["threshold"]),
        best_f1=float(best["f1"]),
        best_precision=float(best["precision"]),
        best_recall=float(best["recall"]),
        n_samples=int(len(y_true01)),
        pos_rate=float(y_true01.mean() if len(y_true01) else 0.0),
        threshold_select=str(threshold_select),
    )

    return summary, b, best, sweep, warn_baseline

_token_re = re.compile(r"[a-z0-9]+")


def _hash_idx(token: str, dim: int) -> int:
    h = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(h, 16) % dim


def hashed_multi_hot(tokens: List[str], dim: int) -> np.ndarray:
    v = np.zeros((dim,), dtype=np.float32)
    for t in tokens:
        t = (t or "").strip().lower()
        if not t:
            continue
        j = _hash_idx(t, dim)
        v[j] += 1.0
    if v.sum() > 0:
        v /= (v.sum() + 1e-12)
    return v


def tokenize_notes(s: str) -> List[str]:
    s = (s or "").lower()
    return _token_re.findall(s)[:40]


def parse_jsonl_to_tmap(jsonl_path: Path) -> Dict[int, dict]:
    tmap: Dict[int, dict] = {}
    if not jsonl_path.exists():
        return tmap
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            t = r.get("t", None)
            if t is None:
                continue
            try:
                t = int(t)
            except Exception:
                continue
            tmap[t] = r
    return tmap


def get_vlm_score(rec: dict) -> float:
    v = rec.get("violent", None)
    if isinstance(v, dict):
        s = v.get("score", None)
        if s is None:
            lab = str(v.get("label", "uncertain")).lower()
            return 1.0 if lab == "yes" else (0.0 if lab == "no" else 0.5)
        try:
            return float(s)
        except Exception:
            return 0.5
    if isinstance(v, str):
        lab = v.lower().strip()
        return 1.0 if lab == "yes" else (0.0 if lab == "no" else 0.5)
    return 0.5


def as_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i).lower().strip() for i in x][:8]
    if isinstance(x, str):
        return [x.lower().strip()]
    return []


def iter_npz_files(emb_root: Path) -> List[Tuple[str, str, Path]]:
    out = []
    for p in emb_root.rglob("*.npz"):
        source = p.parent.name
        vid_id = p.stem
        out.append((vid_id, source, p))
    out.sort()
    return out


class FlatSamples(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y  # [N, K], -1 = ignore

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.X[i]).float(),
            torch.from_numpy(self.y[i]).float(),
        )

class StudentMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, drop: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):  # [B, D] -> [B, K]
        return self.net(x)

def make_splits(video_ids: List[str], seed: int, val_frac: float, test_frac: float):
    rng = random.Random(seed)
    vids = list(video_ids)
    rng.shuffle(vids)
    n = len(vids)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test = set(vids[:n_test])
    val = set(vids[n_test:n_test + n_val])
    train = set(vids[n_test + n_val:])
    return train, val, test

def split_group_id(vid_id: str, source: str, mode: str) -> str:
    """Split grouping key: 'movie' strips the '__#' segment suffix, 'source_*' prefixes the dataset source."""
    vid_id = str(vid_id)
    source = str(source)
    base = vid_id.split("__#", 1)[0] if "__#" in vid_id else vid_id

    mode = (mode or "vid").lower()
    if mode == "vid":
        return vid_id
    if mode == "movie":
        return base
    if mode == "source_vid":
        return f"{source}/{vid_id}"
    if mode == "source_movie":
        return f"{source}/{base}"
    return vid_id

def load_metadata(meta_dir: str) -> Dict[str, dict]:
    meta = {}
    for fname in ["xd_violence.json", "ucf_crime.json", "kinetics.json"]:
        p = os.path.join(meta_dir, fname)
        if os.path.exists(p):
            with open(p, "r") as f:
                meta.update(json.load(f))
    return meta


def _alias_keys(vid_id: str) -> List[str]:
    keys = []
    if not vid_id:
        return keys
    keys.append(vid_id)
    keys.append(vid_id.lower())
    if vid_id.endswith("_x264"):
        keys.append(vid_id[:-5])
        keys.append(vid_id[:-5].lower())
    if vid_id.endswith(".mp4"):
        keys.append(vid_id[:-4])
        keys.append(vid_id[:-4].lower())
    return list(dict.fromkeys(keys))


def parse_ucf_temporal(path: str) -> Dict[str, List[Tuple[int, int]]]:
    segs: Dict[str, List[Tuple[int, int]]] = {}
    if not path or not os.path.exists(path):
        return segs

    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            fname = os.path.basename(parts[0])
            vid = os.path.splitext(fname)[0]
            nums = [int(x) for x in re.findall(r"-?\d+", " ".join(parts[1:]))]
            if len(nums) < 2:
                continue
            pairs = list(zip(nums[0::2], nums[1::2]))
            valid = [(s, e) for (s, e) in pairs if s >= 0 and e >= 0 and e > s]
            if not valid:
                continue
            for k in _alias_keys(vid):
                segs[k] = valid
    return segs


def parse_xd_temporal(path: str) -> Dict[str, List[Tuple[float, float]]]:
    segs: Dict[str, List[Tuple[float, float]]] = {}
    if not path or not os.path.exists(path):
        return segs

    if path.lower().endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k, v in data.items():
                out: List[Tuple[float, float]] = []
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            try:
                                s = float(item[0]); e = float(item[1])
                                if e > s and s >= 0 and e >= 0:
                                    out.append((s, e))
                            except Exception:
                                pass
                        elif isinstance(item, dict) and "start" in item and "end" in item:
                            try:
                                s = float(item["start"]); e = float(item["end"])
                                if e > s and s >= 0 and e >= 0:
                                    out.append((s, e))
                            except Exception:
                                pass
                if out:
                    for kk in _alias_keys(str(k)):
                        segs[kk] = out
        return segs

    float_re = re.compile(r"-?\d+(?:\.\d+)?")
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[,\t ]+", line)
            if not parts:
                continue
            vid = parts[0].strip()
            nums = [float(x) for x in float_re.findall(" ".join(parts[1:]))]
            if len(nums) < 2:
                continue
            pairs = list(zip(nums[0::2], nums[1::2]))
            valid = [(s, e) for (s, e) in pairs if s >= 0 and e >= 0 and e > s]
            if not valid:
                continue
            for k in _alias_keys(vid):
                segs[k] = valid
    return segs


def map_t_to_raw_frame(t: int, T: int, total_frames: int) -> int:
    """Label timestep -> raw frame: identity when T == total_frames, else proportional."""
    if total_frames > 0 and T == total_frames:
        return max(0, min(total_frames - 1, int(t)))

    if total_frames <= 0:
        return max(0, int(t))

    raw = int(round(((t + 0.5) / float(max(1, T))) * float(total_frames)))
    return max(0, min(total_frames - 1, raw))


def xd_segments_to_start_seconds(
    segs: List[Tuple[float, float]],
    *,
    fps: float,
    total_frames: int,
    units: str = "auto",
) -> List[float]:
    """XD segment starts in seconds, using the label builder's units='auto' heuristic."""
    if not segs:
        return []
    if fps <= 0:
        fps = 25.0

    units = (units or "auto").lower()
    max_end = max(e for (_, e) in segs)

    mode = units
    if mode not in ("auto", "frames", "seconds"):
        mode = "auto"

    if mode == "auto":
        if max_end <= 1.0:
            mode = "normalized"
        else:
            if (max_end * fps) <= (total_frames * 1.25):
                mode = "seconds"
            else:
                mode = "frames"

    starts_sec = []
    for (s, _e) in segs:
        if mode == "frames":
            sf = float(s)
            starts_sec.append(sf / fps)
        elif mode == "seconds":
            starts_sec.append(float(s))
        elif mode == "normalized":
            starts_sec.append((float(s) * total_frames) / fps)
        else:
            starts_sec.append(float(s) / fps)
    return sorted(starts_sec)


def ucf_segments_to_start_seconds(segs: List[Tuple[int, int]], fps: float) -> List[float]:
    if not segs:
        return []
    if fps <= 0:
        fps = 25.0
    return sorted([float(s) / fps for (s, _e) in segs])

def pairs_to_alert_times(pairs, *, threshold, T, total_frames, fps, min_consecutive=2):
    # pairs: [(t, prob)] sorted by t
    alert_times = []
    run = 0
    first_t_in_run = None

    for (t, p) in pairs:
        if p >= threshold:
            if run == 0:
                first_t_in_run = int(t)
            run += 1
            if run == min_consecutive:
                rf = map_t_to_raw_frame(first_t_in_run, T, total_frames)
                alert_times.append(float(rf) / fps)
                run = 0
                first_t_in_run = None
        else:
            run = 0
            first_t_in_run = None

    return sorted(alert_times)

def compute_anticipation_video_stats(
    *,
    split_name: str,
    vids: List[str],
    probs_by_vid_t: Dict[str, List[Tuple[int, float]]],
    labels_dir: Path,
    meta: Dict[str, dict],
    ucf_map: Dict[str, List[Tuple[int, int]]],
    xd_map: Dict[str, List[Tuple[float, float]]],
    xd_units: str,
    threshold: float,
    horizon_sec: float,
    label_col: int,
    grace_sec: float = 0.0,
    min_consecutive: int = 1,
    consecutive_t_gap: int = 1,
) -> dict:
    """First-alert video stats vs. onset s: early_false, in [s - horizon, s + grace], or late_or_miss."""
    lead_times = []

    n_videos = len(vids)
    n_with_meta = 0
    n_with_segments = 0

    n_any_alert = 0
    n_any_alert_in_window = 0

    n_first_in_window = 0
    n_first_early_false = 0
    n_first_late_or_miss = 0
    
    n_neg_videos = 0
    n_neg_false_alarm = 0
    n_neg_true_silent = 0

    grace_sec = float(max(0.0, grace_sec))
    horizon_sec = float(max(0.0, horizon_sec))

    for vid in vids:
        info = meta.get(vid, None)
        if info is None:
            for k in _alias_keys(vid):
                if k in meta:
                    info = meta[k]
                    break
        if info is None:
            continue

        n_with_meta += 1
        source = info.get("source", "unknown")
        fps = float(info.get("fps", 25.0) or 25.0)
        total_frames = int(info.get("total_frames", 0) or 0)

        y_path = labels_dir / f"{vid}.npy"
        if not y_path.exists():
            found = None
            for k in _alias_keys(vid):
                p = labels_dir / f"{k}.npy"
                if p.exists():
                    found = p
                    break
            if found is None:
                continue
            y_path = found

        y_arr = np.load(y_path).astype(np.float32)
        if y_arr.ndim != 2:
            raise SystemExit(f"Expected multi-horizon labels [T,K]. Got shape {y_arr.shape} for {y_path}")
        T, K_file = y_arr.shape
        if T <= 0 or total_frames <= 0 or fps <= 0:
            continue
        if label_col < 0 or label_col >= K_file:
            raise SystemExit(f"label_col {label_col} out of range for y shape {y_arr.shape} in {y_path}")

        yk = y_arr[:, label_col]
        valid = (yk >= 0.0)
        pos_ts = np.where((yk > 0.5) & valid)[0]
        has_pos_window = (len(pos_ts) > 0)

        starts_sec: List[float] = []
        if source == "ucf_crime":
            segs = None
            for k in _alias_keys(vid):
                if k in ucf_map:
                    segs = ucf_map[k]
                    break
            if segs:
                starts_sec = ucf_segments_to_start_seconds(segs, fps=fps)

        elif source == "xd_violence":
            segs = None
            for k in _alias_keys(vid):
                if k in xd_map:
                    segs = xd_map[k]
                    break
            if segs:
                starts_sec = xd_segments_to_start_seconds(segs, fps=fps, total_frames=total_frames, units=xd_units)

        if not starts_sec:
            # No annotation: first positive label ~ start - horizon, so start ~= t_pos0 + horizon.
            if has_pos_window:
                t0 = int(pos_ts[0])
                t0_sec = float(map_t_to_raw_frame(t0, T, total_frames)) / fps
                start_est = t0_sec + float(horizon_sec)

                duration_sec = float(total_frames) / fps
                start_est = max(0.0, min(start_est, duration_sec))

                starts_sec = [start_est]
            else:
                starts_sec = []

        if not starts_sec:
            n_neg_videos += 1
            pass
        else:
            starts_sec = sorted(starts_sec)
            n_with_segments += 1

        pairs = probs_by_vid_t.get(vid, [])
        if not pairs:
            for k in _alias_keys(vid):
                if k in probs_by_vid_t:
                    pairs = probs_by_vid_t[k]
                    break

        alert_times = []
        pairs = sorted(pairs, key=lambda x: x[0])

        K_req = int(max(1, min_consecutive))
        gap = int(max(1, consecutive_t_gap))

        if K_req <= 1:
            for (t, p) in pairs:
                if p >= threshold:
                    rf = map_t_to_raw_frame(int(t), T, total_frames)
                    alert_times.append(float(rf) / fps)
        else:
            run = 0
            last_t = None
            triggered_this_run = False

            for (t, p) in pairs:
                t = int(t)
                if p >= threshold:
                    if last_t is None:
                        run = 1
                    else:
                        dt = t - last_t
                        run = (run + 1) if (dt <= gap) else 1

                    last_t = t

                    if (not triggered_this_run) and (run >= K_req):
                        # Fire at the Kth confirming point, as an online system would.
                        rf = map_t_to_raw_frame(t, T, total_frames)
                        alert_times.append(float(rf) / fps)
                        triggered_this_run = True
                else:
                    run = 0
                    last_t = None
                    triggered_this_run = False

        alert_times.sort()
        
        if not starts_sec:
            if len(alert_times) > 0:
                n_neg_false_alarm += 1
            else:
                n_neg_true_silent += 1
            continue
        
        if len(alert_times) > 0:
            n_any_alert += 1

        # Anchor to the earliest onset s with a <= s + grace, so alerts in (s, s + grace] still count for s.
        def anchor_start_for(a: float) -> Optional[float]:
            for s in starts_sec:
                if a <= s + grace_sec:
                    return s
            return None

        any_in_window = False
        for a in alert_times:
            s = anchor_start_for(a)
            if s is None:
                continue
            w0 = s - horizon_sec
            w1 = s + grace_sec
            if w0 <= a <= w1:
                any_in_window = True
                break
        if any_in_window:
            n_any_alert_in_window += 1

        if not alert_times:
            n_first_late_or_miss += 1
            continue

        a0 = alert_times[0]
        s0 = anchor_start_for(a0)
        if s0 is None:
            n_first_late_or_miss += 1
            continue

        w0 = s0 - horizon_sec
        w1 = s0 + grace_sec

        if a0 < w0:
            n_first_early_false += 1
        elif w0 <= a0 <= w1:
            n_first_in_window += 1
            lead = max(0.0, float(s0 - a0))
            lead_times.append(lead)
        else:
            n_first_late_or_miss += 1

    lead_times = np.asarray(lead_times, dtype=np.float32)

    def q(x):
        return float(np.quantile(lead_times, x)) if len(lead_times) else float("nan")

    denom = max(1, n_with_segments)
    first_in_window_rate = float(n_first_in_window / denom)
    early_false_rate = float(n_first_early_false / denom)
    neg_false_alarm_rate = float(n_neg_false_alarm / max(1, n_neg_videos))

    return dict(
        split=str(split_name),
        threshold=float(threshold),
        horizon_sec=float(horizon_sec),
        grace_sec=float(grace_sec),

        n_videos=int(n_videos),
        n_with_meta=int(n_with_meta),
        n_with_segments=int(n_with_segments),

        n_any_alert_anytime=int(n_any_alert),
        n_any_alert_in_window=int(n_any_alert_in_window),

        n_first_alert_in_window=int(n_first_in_window),
        n_first_alert_early_false=int(n_first_early_false),
        n_first_alert_late_or_miss=int(n_first_late_or_miss),
        
        n_neg_videos=int(n_neg_videos),
        n_neg_false_alarm=int(n_neg_false_alarm),
        n_neg_true_silent=int(n_neg_true_silent),
        neg_false_alarm_rate=neg_false_alarm_rate,

        any_alert_rate=float(n_any_alert / max(1, n_with_segments)),
        any_alert_in_window_rate=float(n_any_alert_in_window / max(1, n_with_segments)),
        first_alert_in_window_rate=first_in_window_rate,
        early_false_rate=early_false_rate,

        lead_time_mean_sec=float(np.mean(lead_times)) if len(lead_times) else float("nan"),
        lead_time_median_sec=float(np.median(lead_times)) if len(lead_times) else float("nan"),
        lead_time_p10_sec=q(0.10),
        lead_time_p90_sec=q(0.90),
    )
    
def set_seed(seed: int):
    import os, random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def parse_horizons(s: str) -> List[float]:
    hs = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        hs.append(float(part))
    hs = sorted(list(dict.fromkeys(hs)))
    if not hs:
        raise ValueError("No horizons parsed. Use --horizons_sec like 2,5,10,20,30")
    return hs

def infer_t_stride_frames(ts: np.ndarray, vids: np.ndarray) -> int:
    """Median per-video spacing of `t` (embeddings every N frames -> N); 1 if it cannot be inferred."""
    if ts is None or vids is None or len(ts) < 2:
        return 1

    by_vid: Dict[str, List[int]] = {}
    for t, v in zip(ts, vids):
        by_vid.setdefault(str(v), []).append(int(t))

    med_diffs = []
    for _v, arr in by_vid.items():
        arr = sorted(set(arr))
        if len(arr) < 2:
            continue
        d = np.diff(np.asarray(arr, dtype=np.int64))
        d = d[d > 0]
        if len(d) == 0:
            continue
        med_diffs.append(float(np.median(d)))

    if not med_diffs:
        return 1

    stride = int(round(float(np.median(med_diffs))))
    return max(1, stride)


def sanitize_for_json(x):
    """Recursively replace NaN/Inf with None so metrics.json is strict JSON."""
    if isinstance(x, dict):
        return {k: sanitize_for_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [sanitize_for_json(v) for v in x]
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(x, (np.floating,)):
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(x, (np.integer,)):
        return int(x)
    return x

def apply_ema_by_video(
    probs: np.ndarray,
    vids: np.ndarray,
    ts: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Causal per-video EMA ordered by t: s = alpha*s_prev + (1-alpha)*p, so higher alpha smooths more; 0 disables."""
    if probs is None:
        return probs
    a = float(alpha or 0.0)
    if a <= 0.0 or len(probs) <= 1:
        return probs
    a = max(0.0, min(a, 0.999999))

    out = probs.astype(np.float32, copy=True)

    by_vid = {}
    for i in range(len(out)):
        v = str(vids[i])
        by_vid.setdefault(v, []).append(i)

    for v, idxs in by_vid.items():
        idxs.sort(key=lambda i: int(ts[i]))

        s = float(out[idxs[0]])
        out[idxs[0]] = s
        for i in idxs[1:]:
            x = float(out[i])
            s = a * s + (1.0 - a) * x
            out[i] = s

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_root", default="data/processed/vlm_emb")
    ap.add_argument("--jsonl_root", default="data/processed/vlm_labels")
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--meta_dir", default="data/metadata")
    ap.add_argument("--ucf_temporal_file", default="data/raw_videos/ucf_crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
    ap.add_argument("--xd_temporal_file", default="data/raw_videos/xd_violence/annotations.txt")
    ap.add_argument("--xd_units", default="auto", help="auto|frames|seconds for XD temporal file")

    ap.add_argument("--task", default="detection", choices=["detection", "anticipation"])
    ap.add_argument("--horizon_sec", type=float, default=0.0)

    ap.add_argument("--hash_dim_actions", type=int, default=64)
    ap.add_argument("--hash_dim_weapons", type=int, default=32)
    ap.add_argument("--hash_dim_scene", type=int, default=64)
    ap.add_argument("--hash_dim_notes", type=int, default=0, help="0 disables notes hashing")

    ap.add_argument("--include_vlm_score", action="store_true")
    ap.add_argument("--include_people_count", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--device", default="cuda:0")
    
    ap.add_argument("--split_group", default="source_movie",
                choices=["vid", "movie", "source_vid", "source_movie"],
                help="How to group samples for train/val/test splitting. source_movie is safest.")

    ap.add_argument("--label_smooth", type=float, default=0.0,
                    help="Label smoothing for BCE targets (0 disables). Try 0.01–0.05.")
    ap.add_argument("--video_balance", action="store_true",
                    help="Use WeightedRandomSampler so each video contributes roughly equally during training.")

    ap.add_argument("--grad_clip", type=float, default=1.0, help="Clip grad norm (0 disables).")

    ap.add_argument("--pos_weight_power", type=float, default=0.5,
                    help="Use (n_neg/n_pos)^power for BCE pos_weight. 1.0=old behavior, 0.5=sqrt (recommended).")
    ap.add_argument("--pos_weight_cap", type=float, default=20.0,
                    help="Hard cap for pos_weight to stop rare horizons from dominating.")

    ap.add_argument("--loss_balance", default="inv_sqrt_pos_weight",
                    choices=["none", "inv_pos_weight", "inv_sqrt_pos_weight"],
                    help="Reweight per-horizon loss so rare horizons don't dominate.")

    ap.add_argument("--normalize_inputs", action="store_true",
                    help="Z-score normalize features using TRAIN mean/std (recommended).")

    ap.add_argument("--num_thresholds", type=int, default=101)
    ap.add_argument("--sweep_th_min", type=float, default=0.0, help="Min threshold used in sweeps/selection.")
    ap.add_argument("--sweep_th_max", type=float, default=0.999, help="Max threshold used in sweeps/selection. Keep < 1.0.")
    ap.add_argument("--grace_sec", type=float, default=1.0, help="Tiny grace after segment start for anticipation success window (seconds).")
    ap.add_argument("--early_false_penalty", type=float, default=1.0, help="Penalty weight for early false alarms when selecting threshold via video_utility.")
    ap.add_argument("--threshold_select", default="f1",
                choices=["f1", "youden", "mcc", "video_utility", "precision_floor", "fbeta"])
    ap.add_argument("--precision_floor", type=float, default=0.75,
                    help="Used when --threshold_select precision_floor. Pick threshold with max recall s.t. precision>=floor.")
    ap.add_argument("--f_beta", type=float, default=0.5,
                    help="Used when --threshold_select fbeta. beta<1 favors precision; beta>1 favors recall.")
    
    ap.add_argument("--late_miss_penalty", type=float, default=1.0,
                help="Penalty weight for late/miss when selecting threshold via video_utility.")

    ap.add_argument("--min_consecutive", type=int, default=1,
                    help="Require K consecutive sampled timesteps >= threshold to trigger an alert (1 disables).")

    ap.add_argument("--t_stride", type=int, default=0,
                    help="Frame stride between embedding timesteps t. 0=auto-infer from npz 't' values (recommended).")

    ap.add_argument("--consecutive_gap_steps", type=int, default=0,
                    help="Gap in *embedding-steps* allowed between consecutive points. If >0, overrides --consecutive_t_gap using t_stride. "
                        "Example: stride=5 frames and gap_steps=1 => consecutive_t_gap=5.")

    ap.add_argument("--consecutive_t_gap", type=int, default=0,
                    help="Max allowed gap in raw t units between points to count as consecutive. 0=auto (uses inferred t_stride). "
                        "If your t is raw frame index and embeddings are every 5 frames, auto will pick ~5.")

    ap.add_argument("--horizons_sec", default="2,5,10,20,30,60,120",
                    help="Comma-separated horizons used by multi-horizon labels.")
    
    ap.add_argument("--select_horizon_sec", type=float, default=30.0, help="Which horizon column to use for threshold selection + video_utility stats.")
    ap.add_argument("--false_alarm_penalty", type=float, default=2.0, help="Penalty for alerts on negative videos (no positive window in selected horizon labels).")

    ap.add_argument("--ema_alpha", type=float, default=0.0, help="EMA smoothing on per-video predicted probabilities before thresholding/video_utility. 0 disables. Try 0.7–0.9.")

    args = ap.parse_args()
    
    set_seed(args.seed)
    
    horizons = parse_horizons(args.horizons_sec)
    K = len(horizons)

    if args.select_horizon_sec not in horizons:
        raise SystemExit(f"--select_horizon_sec {args.select_horizon_sec} must be in --horizons_sec {horizons}")
    sel_k = horizons.index(args.select_horizon_sec)

    emb_root = Path(args.emb_root)
    jsonl_root = Path(args.jsonl_root)
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = iter_npz_files(emb_root)
    if not files:
        raise SystemExit(f"No npz files found under {emb_root}")

    meta = load_metadata(args.meta_dir)
    ucf_segments = parse_ucf_temporal(args.ucf_temporal_file)
    xd_segments = parse_xd_temporal(args.xd_temporal_file)

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    vid_ids: List[str] = []
    sources: List[str] = []
    t_list: List[int] = []

    missing_labels = 0
    missing_jsonl = 0
    used_videos = set()

    for vid_id, source, npz_path in files:
        y_path = labels_dir / f"{vid_id}.npy"
        if not y_path.exists():
            missing_labels += 1
            continue

        y_arr = np.load(y_path).astype(np.float32)
        if y_arr.ndim != 2:
            raise SystemExit(f"Expected multi-horizon labels [T,K]. Got {y_arr.shape} at {y_path}")
        T, K_file = y_arr.shape
        if K_file != K:
            raise SystemExit(f"Label K mismatch at {y_path}: file K={K_file}, args K={K}")
        if T <= 0:
            continue

        try:
            z = np.load(npz_path)
            t_arr = z["t"].astype(np.int32)
            emb = z["emb"]
        except Exception:
            continue

        jsonl_path = jsonl_root / source / f"{vid_id}.jsonl"
        tmap = parse_jsonl_to_tmap(jsonl_path)
        if not jsonl_path.exists():
            missing_jsonl += 1

        for i in range(len(t_arr)):
            t = int(t_arr[i])
            if t < 0 or t >= T:
                continue
            gt_vec = y_arr[t]  # [K] in {0, 1, -1}
            if np.all(gt_vec < 0):
                continue
            rec = tmap.get(t, {})

            feats = [np.asarray(emb[i], dtype=np.float32)]

            if args.include_vlm_score:
                feats.append(np.array([get_vlm_score(rec)], dtype=np.float32))

            if args.include_people_count:
                pc = rec.get("people_count", 0)
                try:
                    pc = float(pc)
                except Exception:
                    pc = 0.0
                pc = max(0.0, min(pc, 10.0)) / 10.0
                feats.append(np.array([pc], dtype=np.float32))

            actions = as_list(rec.get("actions"))
            weapons = as_list(rec.get("weapons"))
            scene = as_list(rec.get("scene"))

            feats.append(hashed_multi_hot(actions, args.hash_dim_actions))
            feats.append(hashed_multi_hot(weapons, args.hash_dim_weapons))
            feats.append(hashed_multi_hot(scene, args.hash_dim_scene))

            if args.hash_dim_notes and args.hash_dim_notes > 0:
                notes = rec.get("notes", "")
                feats.append(hashed_multi_hot(tokenize_notes(str(notes)), args.hash_dim_notes))

            x = np.concatenate(feats, axis=0)

            X_list.append(x)
            y_list.append(gt_vec.astype(np.float32))
            vid_ids.append(vid_id)
            sources.append(source)
            t_list.append(t)

        used_videos.add(vid_id)

    if not X_list:
        raise SystemExit("No samples built. Check paths + that labels exist.")

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0).astype(np.float32)  # [N, K]
    vid_ids_arr = np.asarray(vid_ids, dtype=object)
    sources_arr = np.asarray(sources, dtype=object)
    t_arr_all = np.asarray(t_list, dtype=np.int32)
    
    inferred_stride = infer_t_stride_frames(t_arr_all, vid_ids_arr)
    t_stride_used = int(args.t_stride) if int(args.t_stride) > 0 else int(inferred_stride)
    t_stride_used = max(1, t_stride_used)

    if int(args.consecutive_gap_steps) > 0:
        args.consecutive_t_gap = int(args.consecutive_gap_steps) * t_stride_used

    if int(args.consecutive_t_gap) <= 0:
        args.consecutive_t_gap = t_stride_used

    print(f"[Stride] inferred_t_stride={inferred_stride} | using_t_stride={t_stride_used} | consecutive_t_gap={args.consecutive_t_gap}")

    # Group-level split so clips of one movie never straddle splits (leakage).
    group_ids = [split_group_id(v, s, args.split_group) for v, s in zip(vid_ids, sources)]
    uniq_groups = sorted(set(group_ids))

    train_groups, val_groups, test_groups = make_splits(uniq_groups, args.seed, args.val_frac, args.test_frac)

    split = np.array(["train"] * len(vid_ids), dtype=object)
    for i, g in enumerate(group_ids):
        if g in val_groups:
            split[i] = "val"
        elif g in test_groups:
            split[i] = "test"

    train_vids = set([vid_ids[i] for i in range(len(vid_ids)) if split[i] == "train"])
    val_vids   = set([vid_ids[i] for i in range(len(vid_ids)) if split[i] == "val"])
    test_vids  = set([vid_ids[i] for i in range(len(vid_ids)) if split[i] == "test"])

    with (out_dir / "splits_manifest.json").open("w") as f:
        json.dump({
            "split_group_mode": args.split_group,
            "n_groups": {"train": len(train_groups), "val": len(val_groups), "test": len(test_groups)},
            "n_vids": {"train": len(train_vids), "val": len(val_vids), "test": len(test_vids)},
            "groups": {
                "train": sorted(list(train_groups))[:5000],
                "val": sorted(list(val_groups))[:5000],
                "test": sorted(list(test_groups))[:5000],
            },
        }, f, indent=2)

    tr_mask = (split == "train")
    va_mask = (split == "val")
    te_mask = (split == "test")

    X_tr, y_tr = X[tr_mask], y[tr_mask]
    X_va, y_va = X[va_mask], y[va_mask]
    X_te, y_te = X[te_mask], y[te_mask]
    
    if args.normalize_inputs:
        x_mu = X_tr.mean(axis=0, keepdims=True).astype(np.float32)
        x_sig = X_tr.std(axis=0, keepdims=True).astype(np.float32)
        x_sig[x_sig < 1e-6] = 1.0
        X_tr = ((X_tr - x_mu) / x_sig).astype(np.float32)
        X_va = ((X_va - x_mu) / x_sig).astype(np.float32)
        X_te = ((X_te - x_mu) / x_sig).astype(np.float32)

    pos_weight_vec = np.ones((K,), dtype=np.float32)

    for k in range(K):
        yk = y_tr[:, k]
        m = (yk >= 0.0)
        if m.sum() == 0:
            pos_weight_vec[k] = 1.0
            continue

        n_pos = float((yk[m] > 0.5).sum())
        n_neg = float((yk[m] <= 0.5).sum())

        if n_pos <= 0:
            w = 1.0
        else:
            w = (n_neg / (n_pos + 1e-12))
            w = float(w ** float(args.pos_weight_power))
            w = min(w, float(args.pos_weight_cap))

        pos_weight_vec[k] = float(max(1.0, w))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = StudentMLP(in_dim=X.shape[1], out_dim=K, hidden=args.hidden, drop=args.dropout).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3, min_lr=1e-6)

    pos_weight_t = torch.tensor(pos_weight_vec, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t, reduction="none")

    if args.loss_balance == "inv_pos_weight":
        h_w = 1.0 / (pos_weight_vec + 1e-12)
    elif args.loss_balance == "inv_sqrt_pos_weight":
        h_w = 1.0 / (np.sqrt(pos_weight_vec + 1e-12))
    else:
        h_w = np.ones_like(pos_weight_vec, dtype=np.float32)

    h_w = (h_w / (h_w.mean() + 1e-12)).astype(np.float32)
    horizon_w_t = torch.tensor(h_w, device=device)

    train_ds = FlatSamples(X_tr, y_tr)
    val_ds   = FlatSamples(X_va, y_va)
    test_ds  = FlatSamples(X_te, y_te)

    if args.video_balance:
        tr_vids = vid_ids_arr[tr_mask]
        c = Counter([str(v) for v in tr_vids])
        w = np.array([1.0 / max(1, c[str(v)]) for v in tr_vids], dtype=np.float32)
        sampler = WeightedRandomSampler(weights=torch.from_numpy(w), num_samples=len(w), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, sampler=sampler, num_workers=2, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    val_loader  = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    def run_loader(loader):
        model.eval()
        all_logits = []
        all_y = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                logits = model(xb).detach().cpu().numpy()  # [B, K]
                all_logits.append(logits)
                all_y.append(yb.cpu().numpy())
        y_out = np.concatenate(all_y)
        l_out = np.concatenate(all_logits)
        probs = 1.0 / (1.0 + np.exp(-l_out))       # [N, K]
        return y_out, l_out, probs

    history = []
    best_val_pr = -1.0
    best_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            mask = (yb >= 0.0).float()
            yb01 = torch.clamp(yb, 0.0, 1.0)    # -1 -> 0 only so BCE is defined; masked out below

            eps = float(args.label_smooth or 0.0)
            if eps > 0.0:
                eps = max(0.0, min(eps, 0.2))
                yb01 = yb01 * (1.0 - eps) + 0.5 * eps

            loss_mat = crit(logits, yb01)       # [B, K]
            loss_mat = loss_mat * horizon_w_t
            loss = (loss_mat * mask).sum() / (mask.sum() + 1e-12)
            loss.backward()

            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))

            opt.step()

            bs = xb.shape[0]
            total_loss += float(loss.item()) * bs
            n += bs

        tr_loss = total_loss / max(1, n)

        yv, lv, pv = run_loader(val_loader)

        yv_k = yv[:, sel_k]
        pv_k_raw = pv[:, sel_k]

        pv_k = apply_ema_by_video(
            pv_k_raw,
            vid_ids_arr[va_mask],
            t_arr_all[va_mask],
            args.ema_alpha,
        )

        keep = (yv_k >= 0.0)
        yv_true = (yv_k[keep] > 0.5).astype(np.int32)
        pv_use  = pv_k[keep]


        val_summary, val_baselines, val_best, _val_sweep, warn_baseline = evaluate_probs(
            yv_true,
            pv_use,
            num_thresholds=args.num_thresholds,
            threshold_select=args.threshold_select,
            th_min=args.sweep_th_min,
            th_max=args.sweep_th_max,
            precision_floor=args.precision_floor,
            f_beta=args.f_beta,
        )

        history.append({"epoch": epoch, "train_loss": tr_loss, **val_summary})

        val_pr = val_summary["pr_auc"]
        if not math.isnan(val_pr):
            sched.step(val_pr)

        if not math.isnan(val_pr) and val_pr > best_val_pr:
            best_val_pr = val_pr
            torch.save({"model": model.state_dict(), "in_dim": X.shape[1]}, best_path)

        print(
            f"[E{epoch:02d}] loss={tr_loss:.4f} "
            f"val_pr={val_summary['pr_auc']:.4f} val_roc={val_summary['roc_auc']:.4f} "
            f"val_best_f1={val_summary['best_f1']:.4f} (oracle_f1_th={val_summary['best_threshold']:.2f})"
        )

    if best_path.exists():
        try:
            ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])

    yv, lv, pv = run_loader(val_loader)
    yt, lt, pt = run_loader(test_loader)

    yv_k = yv[:, sel_k]
    pv_k_raw = pv[:, sel_k]
    va_keep = (yv_k >= 0.0)

    yt_k = yt[:, sel_k]
    pt_k_raw = pt[:, sel_k]
    te_keep = (yt_k >= 0.0)

    pv_k = apply_ema_by_video(
        pv_k_raw,
        vid_ids_arr[va_mask],
        t_arr_all[va_mask],
        args.ema_alpha,
    )
    pt_k = apply_ema_by_video(
        pt_k_raw,
        vid_ids_arr[te_mask],
        t_arr_all[te_mask],
        args.ema_alpha,
    )

    yv_true = (yv_k[va_keep] > 0.5).astype(np.int32)
    pv_use  = pv_k[va_keep]

    yt_true = (yt_k[te_keep] > 0.5).astype(np.int32)
    pt_use  = pt_k[te_keep]

    def probs_by_vid(mask: np.ndarray, probs_local: np.ndarray) -> Dict[str, List[Tuple[int, float]]]:
        idx = np.where(mask)[0]
        out: Dict[str, List[Tuple[int, float]]] = {}
        for local_i, global_i in enumerate(idx):
            vid = str(vid_ids_arr[global_i])
            t = int(t_arr_all[global_i])
            out.setdefault(vid, []).append((t, float(probs_local[local_i])))
        for k in out:
            out[k].sort(key=lambda x: x[0])
        return out

    pv_by_vid = probs_by_vid(va_mask, pv_k)
    pt_by_vid = probs_by_vid(te_mask, pt_k)

    val_oracle, val_base, val_best, val_sweep, warn_val_baseline = evaluate_probs(        
        yv_true,
        pv_use,
        num_thresholds=args.num_thresholds,
        threshold_select=args.threshold_select,
        th_min=args.sweep_th_min,
        th_max=args.sweep_th_max,
        precision_floor=args.precision_floor,
        f_beta=args.f_beta,
    )
    test_oracle, test_base, test_best, test_sweep, warn_test_baseline = evaluate_probs(
        yt_true,
        pt_use,
        num_thresholds=args.num_thresholds,
        threshold_select=args.threshold_select,
        th_min=args.sweep_th_min,
        th_max=args.sweep_th_max,
        precision_floor=args.precision_floor,
        f_beta=args.f_beta,
    )

    if args.threshold_select == "video_utility" and args.task == "anticipation" and args.select_horizon_sec > 0:
        best_val_threshold = 0.5
        best_score = -1e9
        best_val_video_stats = None
        fallback_info = None

        lo = float(max(0.0, min(1.0, args.sweep_th_min)))
        hi = float(max(0.0, min(1.0, args.sweep_th_max)))
        hi = min(hi, 1.0 - 1e-6)
        if hi <= lo:
            hi = min(1.0 - 1e-6, lo + 1e-3)
        th_grid = np.linspace(lo, hi, args.num_thresholds, dtype=np.float32)

        sweep_csv = out_dir / "video_utility_sweep.csv"
        with sweep_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "threshold",
                "utility",
                "first_in_window_rate",
                "early_false_rate",
                "late_or_miss_rate",
                "any_alert_rate",
                "any_alert_in_window_rate",
                "lead_time_median_sec",
                "lead_time_mean_sec",
                "n_with_segments",
                "n_first_in_window",
                "n_first_early_false",
                "n_first_late_or_miss",
                "neg_false_alarm_rate",
                "n_neg_videos",
                "n_neg_false_alarm",
            ])

            for th in th_grid:
                th = float(th)

                st = compute_anticipation_video_stats(
                    split_name="val",
                    vids=sorted(list(val_vids)),
                    probs_by_vid_t=pv_by_vid,
                    labels_dir=labels_dir,
                    meta=meta,
                    ucf_map=ucf_segments,
                    xd_map=xd_segments,
                    xd_units=args.xd_units,
                    threshold=th,
                    horizon_sec=float(args.select_horizon_sec),
                    label_col=int(sel_k),
                    grace_sec=float(args.grace_sec),
                    min_consecutive=int(args.min_consecutive),
                    consecutive_t_gap=int(args.consecutive_t_gap),
                )

                denom = max(1, int(st["n_with_segments"]))

                any_in = float(st.get("any_alert_in_window_rate", 0.0))
                early = float(st.get("early_false_rate", 0.0))
                neg_fa = float(st.get("neg_false_alarm_rate", 0.0))

                # miss = no in-window alert at all (not based on the first alert)
                late_or_miss_rate = 1.0 - any_in

                score = (
                    any_in
                    - float(args.early_false_penalty) * early
                    - float(args.late_miss_penalty) * late_or_miss_rate
                    - float(args.false_alarm_penalty) * neg_fa
                )

                w.writerow([
                    float(th),
                    float(score),
                    float(st["first_alert_in_window_rate"]),
                    float(st["early_false_rate"]),
                    float(late_or_miss_rate),
                    float(st["any_alert_rate"]),
                    float(st["any_alert_in_window_rate"]),
                    float(st["lead_time_median_sec"]),
                    float(st["lead_time_mean_sec"]),
                    int(st["n_with_segments"]),
                    int(st["n_first_alert_in_window"]),
                    int(st["n_first_alert_early_false"]),
                    int(st["n_first_alert_late_or_miss"]),
                    float(st.get("neg_false_alarm_rate", 0.0)),
                    int(st.get("n_neg_videos", 0)),
                    int(st.get("n_neg_false_alarm", 0)),
                ])

                # Ties: prefer thresholds that alert at least once, then the higher threshold.
                deg = (int(st.get("n_any_alert_anytime", 0)) == 0)
                best_deg = (
                    best_val_video_stats is not None
                    and int(best_val_video_stats.get("n_any_alert_anytime", 0)) == 0
                )

                if (
                    (score > best_score)
                    or (abs(score - best_score) < 1e-12 and best_val_video_stats is None)
                    or (abs(score - best_score) < 1e-12 and best_deg and not deg)
                    or (abs(score - best_score) < 1e-12 and (best_deg == deg) and (th > best_val_threshold))
                ):
                    best_score = float(score)
                    best_val_threshold = float(th)
                    best_val_video_stats = st

        # Utility can be maximized by never alerting; fall back to the VAL oracle F1 threshold.
        if best_val_video_stats is not None and int(best_val_video_stats.get("n_any_alert_anytime", 0)) == 0:
            fb_th = float(val_best["threshold"])
            fallback_info = {
                "reason": "degenerate_no_alerts_from_video_utility",
                "fallback_mode": "f1_oracle_val",
                "fallback_threshold": fb_th,
                "video_utility_threshold": float(best_val_threshold),
                "video_utility_utility": float(best_score),
            }
            best_val_threshold = fb_th

            best_val_video_stats = compute_anticipation_video_stats(
                split_name="val",
                vids=sorted(list(val_vids)),
                probs_by_vid_t=pv_by_vid,
                labels_dir=labels_dir,
                meta=meta,
                ucf_map=ucf_segments,
                xd_map=xd_segments,
                xd_units=args.xd_units,
                threshold=float(best_val_threshold),
                horizon_sec=float(args.select_horizon_sec),
                label_col=int(sel_k),
                grace_sec=float(args.grace_sec),
                min_consecutive=int(args.min_consecutive),
                consecutive_t_gap=int(args.consecutive_t_gap),
            )

        val_video_selection = dict(
            mode="video_utility",
            utility=float(best_score),
            early_false_penalty=float(args.early_false_penalty),
            late_miss_penalty=float(args.late_miss_penalty),
            grace_sec=float(args.grace_sec),
            min_consecutive=int(args.min_consecutive),
            consecutive_t_gap=int(args.consecutive_t_gap),
            val_stats_at_selected_threshold=best_val_video_stats,
            sweep_csv=str(sweep_csv),
            false_alarm_penalty=float(args.false_alarm_penalty),
        )
        if fallback_info is not None:
            val_video_selection["fallback"] = fallback_info

    else:
        val_video_selection = None
        val_sel_summary, _, val_sel_best, _sweep_sel, _warn = evaluate_probs(
            yv_true,
            pv_use,
            num_thresholds=args.num_thresholds,
            threshold_select=args.threshold_select,
            th_min=args.sweep_th_min,
            th_max=args.sweep_th_max,
            precision_floor=args.precision_floor,
            f_beta=args.f_beta,
        )
        best_val_threshold = float(val_sel_best["threshold"])

    val_at_valth = metrics_at_threshold(yv_true, pv_use, best_val_threshold)
    test_at_valth = metrics_at_threshold(yt_true, pt_use, best_val_threshold)
    val_at_valth["roc_auc"] = float(roc_auc_score_np(yv_true, pv_use))
    val_at_valth["pr_auc"]  = float(average_precision_np(yv_true, pv_use))
    test_at_valth["roc_auc"] = float(roc_auc_score_np(yt_true, pt_use))
    test_at_valth["pr_auc"]  = float(average_precision_np(yt_true, pt_use))

    per_sample_csv = out_dir / "per_sample.csv"
    with per_sample_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "vid_id", "source", "t", "gt", "pred_prob_raw", "pred_prob_ema"])

        val_idx = np.where(va_mask)[0]
        for local_i, global_i in enumerate(val_idx):
            gt = float(y[global_i, sel_k])
            prob_raw = float(pv_k_raw[local_i])
            prob_ema = float(pv_k[local_i])
            if gt < 0:
                continue
            w.writerow(["val", vid_ids_arr[global_i], sources_arr[global_i], int(t_arr_all[global_i]), gt, prob_raw, prob_ema])

        test_idx = np.where(te_mask)[0]
        for local_i, global_i in enumerate(test_idx):
            gt = float(y[global_i, sel_k])
            prob_raw = float(pt_k_raw[local_i])
            prob_ema = float(pt_k[local_i])
            if gt < 0:
                continue
            w.writerow(["test", vid_ids_arr[global_i], sources_arr[global_i], int(t_arr_all[global_i]), gt, prob_raw, prob_ema])

    def write_per_video_csv(path: Path, probs_map: Dict[str, List[Tuple[int, float]]]):
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["vid_id", "pred_max_prob", "pred_mean_prob", "n_points"])
            for vid, pairs in sorted(probs_map.items()):
                ps = [p for (_t, p) in pairs]
                w.writerow([vid, float(np.max(ps)), float(np.mean(ps)), int(len(ps))])

    write_per_video_csv(out_dir / "per_video_val.csv", pv_by_vid)
    write_per_video_csv(out_dir / "per_video_test.csv", pt_by_vid)

    anticipation_stats = None
    if args.task == "anticipation" and args.select_horizon_sec > 0:
        val_vid_list = sorted(list(val_vids))
        test_vid_list = sorted(list(test_vids))

        anticipation_stats = {
            "val": compute_anticipation_video_stats(
                split_name="val",
                vids=val_vid_list,
                probs_by_vid_t=pv_by_vid,
                labels_dir=labels_dir,
                meta=meta,
                ucf_map=ucf_segments,
                xd_map=xd_segments,
                xd_units=args.xd_units,
                threshold=best_val_threshold,
                horizon_sec=float(args.select_horizon_sec),
                label_col=int(sel_k),
                grace_sec=float(args.grace_sec),
                min_consecutive=int(args.min_consecutive),
                consecutive_t_gap=int(args.consecutive_t_gap),
            ),
            "test": compute_anticipation_video_stats(
                split_name="test",
                vids=test_vid_list,
                probs_by_vid_t=pt_by_vid,
                labels_dir=labels_dir,
                meta=meta,
                ucf_map=ucf_segments,
                xd_map=xd_segments,
                xd_units=args.xd_units,
                threshold=best_val_threshold,
                horizon_sec=float(args.select_horizon_sec),
                label_col=int(sel_k),
                grace_sec=float(args.grace_sec),
                min_consecutive=int(args.min_consecutive),
                consecutive_t_gap=int(args.consecutive_t_gap),
            )
        }

    metrics = {
        "task": str(args.task),
        "horizon_sec": float(args.horizon_sec),
        "horizons_sec": horizons,
        "select_horizon_sec": float(args.select_horizon_sec),
        "labels_dir": str(labels_dir),
        "emb_root": str(emb_root),
        "jsonl_root": str(jsonl_root),
        "counts": {
            "npz_files_total": int(len(files)),
            "videos_used": int(len(used_videos)),
            "samples_total": int(len(y)),
            "missing_labels_npz": int(missing_labels),
            "missing_jsonl": int(missing_jsonl),
        },
        "feature_dim": int(X.shape[1]),
        "pos_weight_used": [float(x) for x in pos_weight_vec.tolist()],
        "split_videos": {
            "train": int(len(train_vids)),
            "val": int(len(val_vids)),
            "test": int(len(test_vids)),
        },
        "threshold_select_mode": str(args.threshold_select),
        "best_val_threshold_selected": float(best_val_threshold),

        "val_metrics_oracle": val_oracle,
        "test_metrics_oracle": test_oracle,

        "val_baselines": val_base,
        "test_baselines": test_base,
        "warning_val_best_is_all_positive_baseline": bool(warn_val_baseline),
        "warning_test_best_is_all_positive_baseline": bool(warn_test_baseline),

        "val_metrics_at_val_threshold": val_at_valth,
        "test_metrics_at_val_threshold": test_at_valth,

        "anticipation_video_stats_at_val_threshold": anticipation_stats,
        
        "grace_sec": float(args.grace_sec),
        "early_false_penalty": float(args.early_false_penalty),
        "val_threshold_selection_video": val_video_selection,
        
        "false_alarm_penalty": float(args.false_alarm_penalty),
        
        "t_stride_inferred_frames": int(inferred_stride),
        "t_stride_used_frames": int(t_stride_used),
        "min_consecutive": int(args.min_consecutive),
        "consecutive_t_gap_used": int(args.consecutive_t_gap),
        "consecutive_gap_steps_arg": int(args.consecutive_gap_steps),

        "history": history,
    }

    with (out_dir / "metrics.json").open("w") as f:
        json.dump(sanitize_for_json(metrics), f, indent=2, allow_nan=False)

    import matplotlib.pyplot as plt

    def _save(figpath: Path):
        plt.tight_layout()
        plt.savefig(figpath, dpi=150)
        plt.close()

    def plot_threshold_sweep(sweep: List[dict], out_path: Path, title: str, chosen_th: float):
        th = np.array([d["threshold"] for d in sweep], dtype=np.float32)
        f1 = np.array([d["f1"] for d in sweep], dtype=np.float32)
        prec = np.array([d["precision"] for d in sweep], dtype=np.float32)
        rec = np.array([d["recall"] for d in sweep], dtype=np.float32)
        fpr = np.array([d["fpr"] for d in sweep], dtype=np.float32)

        plt.figure()
        plt.plot(th, f1, label="F1")
        plt.plot(th, prec, label="Precision")
        plt.plot(th, rec, label="Recall")
        plt.plot(th, fpr, label="FPR")
        plt.axvline(chosen_th, linestyle="--", label=f"chosen_th={chosen_th:.2f}")
        plt.xlabel("threshold")
        plt.ylabel("metric")
        plt.title(title)
        plt.legend()
        _save(out_path)

    def plot_prob_hist(y_true01: np.ndarray, probs: np.ndarray, out_path: Path, title: str):
        pos = probs[y_true01 == 1]
        neg = probs[y_true01 == 0]
        plt.figure()
        if len(neg) > 0:
            plt.hist(neg, bins=50, alpha=0.5, label=f"neg (n={len(neg)})", density=True)
        if len(pos) > 0:
            plt.hist(pos, bins=50, alpha=0.5, label=f"pos (n={len(pos)})", density=True)
        plt.xlabel("predicted probability")
        plt.ylabel("density")
        plt.title(title)
        plt.legend()
        _save(out_path)

    def plot_calibration(y_true01: np.ndarray, probs: np.ndarray, out_path: Path, title: str, n_bins: int = 20):
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(probs, bins) - 1
        xs = []
        ys = []
        ns = []
        for b in range(n_bins):
            m = (bin_ids == b)
            if not np.any(m):
                continue
            xs.append(float(np.mean(probs[m])))
            ys.append(float(np.mean(y_true01[m])))
            ns.append(int(np.sum(m)))

        plt.figure()
        plt.plot([0, 1], [0, 1], linestyle="--", label="ideal")
        if xs:
            plt.plot(xs, ys, marker="o", label="model")
        plt.xlabel("mean predicted prob (bin)")
        plt.ylabel("empirical positive rate (bin)")
        plt.title(title + (f" | bins={n_bins}" if n_bins else ""))
        plt.legend()
        _save(out_path)

    def plot_confusion(y_true01: np.ndarray, probs: np.ndarray, th: float, out_path: Path, title: str):
        m = metrics_at_threshold(y_true01, probs, th)
        tn, fp, fn, tp = m["tn"], m["fp"], m["fn"], m["tp"]
        cm = np.array([[tn, fp],
                       [fn, tp]], dtype=np.int32)

        plt.figure()
        plt.imshow(cm)
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        plt.title(f"{title} | th={th:.2f}")

        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(int(v)), ha="center", va="center")

        plt.xlabel("Prediction")
        plt.ylabel("Ground truth")
        _save(out_path)

    def plot_per_video_scatter(csv_path: Path, out_path: Path, title: str):
        if not csv_path.exists():
            return
        means = []
        maxs = []
        with csv_path.open("r") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    means.append(float(row["pred_mean_prob"]))
                    maxs.append(float(row["pred_max_prob"]))
                except Exception:
                    pass
        if not means:
            return
        means = np.asarray(means, dtype=np.float32)
        maxs = np.asarray(maxs, dtype=np.float32)

        plt.figure()
        plt.scatter(means, maxs, s=10)
        plt.xlabel("per-video mean prob")
        plt.ylabel("per-video max prob")
        plt.title(title)
        _save(out_path)

    def plot_video_utility_sweep(csv_path: Path, out_prefix: Path, chosen_th: float):
        if not csv_path.exists():
            return

        th = []
        util = []
        first_in = []
        early = []
        neg_fa = []
        late = []
        lead_med = []

        with csv_path.open("r") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    th.append(float(row["threshold"]))
                    util.append(float(row["utility"]))
                    first_in.append(float(row["first_in_window_rate"]))
                    early.append(float(row["early_false_rate"]))
                    late.append(float(row["late_or_miss_rate"]))
                    neg_fa.append(float(row["neg_false_alarm_rate"]))
                    lead_med.append(float(row["lead_time_median_sec"]))
                except Exception:
                    continue

        if not th:
            return

        th = np.asarray(th, dtype=np.float32)
        util = np.asarray(util, dtype=np.float32)
        first_in = np.asarray(first_in, dtype=np.float32)
        early = np.asarray(early, dtype=np.float32)
        late = np.asarray(late, dtype=np.float32)
        neg_fa = np.asarray(neg_fa, dtype=np.float32)
        lead_med = np.asarray(lead_med, dtype=np.float32)

        plt.figure()
        plt.plot(th, util, label="utility")
        plt.axvline(chosen_th, linestyle="--", label=f"chosen_th={chosen_th:.2f}")
        plt.xlabel("threshold")
        plt.ylabel("utility")
        plt.title("Video utility sweep (VAL)")
        plt.legend()
        _save(Path(str(out_prefix) + "_utility.png"))

        plt.figure()
        plt.plot(th, first_in, label="first_in_window_rate")
        plt.plot(th, early, label="early_false_rate")
        plt.plot(th, late, label="late_or_miss_rate")
        plt.plot(th, neg_fa, label="neg_false_alarm_rate")
        plt.axvline(chosen_th, linestyle="--", label=f"chosen_th={chosen_th:.2f}")
        plt.xlabel("threshold")
        plt.ylabel("rate")
        plt.title("Video rates vs threshold (VAL)")
        plt.legend()
        _save(Path(str(out_prefix) + "_rates.png"))

        plt.figure()
        plt.plot(th, lead_med, label="lead_time_median_sec")
        plt.axvline(chosen_th, linestyle="--", label=f"chosen_th={chosen_th:.2f}")
        plt.xlabel("threshold")
        plt.ylabel("seconds")
        plt.title("Median lead time vs threshold (VAL)")
        plt.legend()
        _save(Path(str(out_prefix) + "_leadtime.png"))

    epochs = [h["epoch"] for h in history]
    losses = [h["train_loss"] for h in history]
    pr = [h["pr_auc"] for h in history]
    roc = [h["roc_auc"] for h in history]
    f1 = [h["best_f1"] for h in history]

    plt.figure()
    plt.plot(epochs, losses, label="train_loss")
    plt.plot(epochs, pr, label="val_pr_auc")
    plt.plot(epochs, roc, label="val_roc_auc")
    plt.plot(epochs, f1, label="val_best_f1 (oracle)")
    plt.xlabel("epoch")
    plt.legend()
    plt.title("Training curve")
    _save(out_dir / "train_curve.png")

    order = np.argsort(-pt_use)
    y_sorted = yt_true[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp[-1] + 1e-12)

    plt.figure()
    plt.plot(rec, prec)
    op = metrics_at_threshold(yt_true, pt_use, best_val_threshold)
    plt.scatter([op["recall"]], [op["precision"]])
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title(f"PR curve (test) | op@val_th={best_val_threshold:.2f} ({args.threshold_select})")
    _save(out_dir / "pr_curve.png")

    order = np.argsort(-pt_use)
    y_sorted = yt_true[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    tpr = tp / (tp[-1] + 1e-12)
    fpr = fp / (fp[-1] + 1e-12)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve (test)")
    _save(out_dir / "roc_curve.png")

    plot_threshold_sweep(val_sweep, out_dir / "val_threshold_sweep.png", "Threshold sweep (VAL)", best_val_threshold)
    plot_threshold_sweep(test_sweep, out_dir / "test_threshold_sweep.png", "Threshold sweep (TEST)", best_val_threshold)

    plot_prob_hist(yv_true, pv_use, out_dir / "val_prob_hist.png", "Prob histogram (VAL) | pos vs neg")
    plot_prob_hist(yt_true, pt_use, out_dir / "test_prob_hist.png", "Prob histogram (TEST) | pos vs neg")

    plot_calibration(yv_true, pv_use, out_dir / "val_calibration.png", "Calibration (VAL)")
    plot_calibration(yt_true, pt_use, out_dir / "test_calibration.png", "Calibration (TEST)")

    plot_confusion(yv_true, pv_use, best_val_threshold, out_dir / "val_confusion.png", "Confusion matrix (VAL)")
    plot_confusion(yt_true, pt_use, best_val_threshold, out_dir / "test_confusion.png", "Confusion matrix (TEST)")

    plot_per_video_scatter(out_dir / "per_video_val.csv", out_dir / "per_video_val_scatter.png", "Per-video scatter (VAL) | mean vs max prob")
    plot_per_video_scatter(out_dir / "per_video_test.csv", out_dir / "per_video_test_scatter.png", "Per-video scatter (TEST) | mean vs max prob")

    plot_video_utility_sweep(out_dir / "video_utility_sweep.csv", out_dir / "video_utility_sweep", best_val_threshold)

    print("\n==== DONE ====")
    print(f"Task={args.task} horizon_sec={args.horizon_sec}")
    print("VAL oracle:", val_oracle)
    print("TEST oracle:", test_oracle)
    print("VAL baselines:", val_base)
    print("TEST baselines:", test_base)
    print(f"Chosen threshold (VAL, mode={args.threshold_select}): {best_val_threshold:.3f}")
    print("VAL @ chosen threshold:", val_at_valth)
    print("TEST @ chosen threshold:", test_at_valth)
    if anticipation_stats:
        print("Anticipation video stats:", anticipation_stats)
    print(f"Wrote: {out_dir}/metrics.json + curves + csvs")


if __name__ == "__main__":
    main()
