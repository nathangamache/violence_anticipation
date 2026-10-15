"""VA30 evaluator: every metric of docs/BASELINE.md section 6.4 from one per-tick score table.

Usage (from the repo root):

    .venv/bin/python -m anticipation.eval.va30 --scores runs/my_model/scores.parquet \
        --out_dir benchmark/results/my_model [--name my_model] [--folds benchmark/folds_v2.json] \
        [--n_boot 1000] [--k 2 --n 3] [--restrict_to_scored] [--onset_sources official,hivau,ubi] \
        [--threshold_set val|nontest] [--horizons benchmark/extract_manifest_v1.csv] [--meta run_config.json]

Writes ``<out_dir>/metrics.json`` (all numbers with 95 % cluster-bootstrap CIs), ``<out_dir>/report.md``
(short human summary), ``amoc.png`` and ``reliability.png``, and ``events_op2.csv`` (per eligible event
outcome at the 2 FA/h operating point).

Score table contract (parquet or csv), one row per scored tick:

    video_id  str    catalog id ``<source>/<stem>`` (benchmark/videos.csv)
    t_sec     float  tick time in seconds; ticks are 1..floor(duration) (``anticipation.features.io.tick_times``)
    score     float  higher = violence more likely to start soon. Must use media in [0, t_sec] only.
                     A probability of "onset within 30 s" in [0, 1] also enables the calibration metrics.
    fold      int    fold whose model produced the score. Each fold's model must score its TEST fold and its
                     VALIDATION slice (thresholds are chosen on validation). -1 = fold-agnostic score (untrained
                     baselines): used as the val and test score for every fold.
    seed      int    training seed (optional; default 0). Metrics are computed per seed, then mean/std.

Ticks missing from the table are an error unless ``--allow_missing`` (then missing videos are dropped and
reported, gaps are sample-and-hold filled, and ticks before the first score get the global minimum score).
Also an error: duplicate (seed, fold, video_id, t_sec) keys, NaN / non-integer fold or seed, NaN / inf scores or
times, and a (seed, video) with both fold -1 rows and fold-specific rows.

Per-video evaluation horizons (``--horizons``, a csv with video_id and duration_sec, e.g. the extraction manifest):
a CLEAN NEGATIVE listed with a shorter duration than the catalog is evaluated as a video of that length. Footage
after the horizon is unobserved: no ticks, no FA/h hours, no scores required (rows past it are ignored), and the
usual right-censoring applies at the new end. Shortening any other role is an error. Without the flag nothing changes.

Protocol summary (exact definitions in docs/VA30.md):
  * Universe: fold videos with role ``violent`` (trusted onsets) or ``clean_negative``.
  * Out-of-fold: each video is scored by the model of its own test fold; metrics pool the 5 test folds.
  * Threshold-free metrics on pooled test ticks; thresholded metrics use, per fold, the threshold chosen on that
    fold's validation slice so that validation FA/h on clean negatives is 1, 2 or 5; outcomes are pooled.
  * Headline frame metrics use pre-violence ticks only (before the first annotated segment of a clip); the
    spec-literal all-segment versions are reported as ``*_all_segments``.
  * Success@30 is reported with two chance references: the Poisson formula and a position-oracle null
    (``position_null``); plus Success@30 on "late" events whose acceptance window excludes the clip start.
  * Alarm episodes do not merge across an event's clear_start (end of the previous segment's aftermath).
  * CIs: cluster bootstrap over test videos (default 1000 resamples, percentile 95 %); thresholds held fixed.
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from anticipation.data.catalog import EVENTS_JSON, REPO_ROOT, VIDEOS_CSV, file_sha256, load_catalog, load_events
from anticipation.data.folds import FOLDS_DEFAULT, Folds
from anticipation.data.labels import AFTERMATH_SEC, HORIZONS, blocker_spans, event_table, tick_labels

TRUSTED_ONSETS = ("official", "hivau", "ubi")
HORIZON_TOL_SEC = 1e-3     # a horizon within this of the catalog duration is not a shortening
SOURCE_KINDS = ("xd_movie", "xd_youtube", "ucf", "ubi", "yt")
SUCCESS_LEADS = (30, 10, 0)
SUCCESS_WINDOW_SEC = 90.0  # [t0 - 120, t0 - 30]
LATE_WINDOW_SEC = 30.0     # "late" events: acceptance window starts >= 30 s into the clip (clip start excluded)


@dataclass
class AlarmPolicy:
    """Event-level alarm policy (VA30 6.4.3). Only ``theta`` is chosen on validation; the rest is fixed."""

    k: int = 2                    # k-of-n smoothing: alarm active at t if >= k of ticks t-n+1..t exceed theta
    n: int = 3
    merge_gap_sec: float = 10.0   # an alarm that resumes after < 10 s of silence continues the same episode
    refractory_sec: float = 30.0  # no new alarm within 30 s of the previous alarm start
    window_sec: float = 120.0     # an alarm starting in [t0 - 120, t0) is an anticipation of onset t0
    aftermath_sec: float = AFTERMATH_SEC


@dataclass
class EvalConfig:
    policy: AlarmPolicy = field(default_factory=AlarmPolicy)
    horizons: tuple = HORIZONS
    far_lo: float = 30.0
    far_hi: float = 60.0
    neg_window: float = 120.0
    fa_targets: tuple = (1.0, 2.0, 5.0)
    n_boot: int = 1000
    boot_seed: int = 0
    cluster: str = "video"        # or "group" (resample catalog groups, more conservative)
    onset_sources: tuple = TRUSTED_ONSETS
    restrict_to_scored: bool = False
    allow_missing: bool = False
    amoc_points: int = 60
    reliability_bins: int = 10
    max_threshold_candidates: int = 5000   # unused since the exact threshold search (kept for compatibility)
    breakdowns: bool = True
    position_null: bool = True             # position-oracle reference curve for Success@30 (see position_null)
    threshold_bootstrap: int = 0           # > 0: also threshold-aware CIs with this many replicates (slow-ish)
    threshold_set: str = "val"             # "val" (default) or "nontest" (fold-agnostic tables only, see docs)
    horizons_csv: str = None               # per-video evaluation horizons (csv: video_id, duration_sec), see docs
    phase0_metrics: bool = False
    person_counts_root: str = None


# ---------------------------------------------------------------------------------------------------------
# Benchmark (ground truth + folds)
# ---------------------------------------------------------------------------------------------------------

def read_video_horizons(path):
    """``({video_id: duration_sec}, {"file", "sha256"})`` from a csv with (at least) video_id and duration_sec."""
    df = pd.read_csv(path, dtype={"video_id": str})
    missing = {"video_id", "duration_sec"} - set(df.columns)
    if missing:
        raise ValueError(f"horizons file {path} lacks columns {sorted(missing)}")
    d = pd.to_numeric(df["duration_sec"], errors="coerce").values.astype(np.float64)
    if df["video_id"].isna().any() or not np.isfinite(d).all() or (d <= 0).any():
        raise ValueError(f"horizons file {path}: every row needs a video_id and a finite duration_sec > 0")
    dup = df["video_id"].duplicated(keep=False)
    if dup.any():
        raise ValueError(f"horizons file {path}: {int(dup.sum())} rows repeat a video_id, e.g. "
                         f"{list(df.loc[dup, 'video_id'].head(3))}")
    ap = os.path.abspath(path)
    return dict(zip(df["video_id"], d.tolist())), {
        "file": os.path.relpath(ap, REPO_ROOT) if ap.startswith(REPO_ROOT) else str(path),
        "sha256": file_sha256(path)}


class Benchmark:
    """Catalog rows, segments and folds. ``Benchmark.load()`` reads the committed benchmark files.

    For tests, construct directly: ``videos`` is a DataFrame indexed by video_id with columns duration_sec, role,
    onset_source, source_kind, interpersonal, has_audio, group; ``events`` maps video_id -> {"segments": [...],
    optional "other_segments": [...], "mask_spans": [...]}; ``folds`` is an ``anticipation.data.folds.Folds``.

    ``video_horizons`` (``{video_id: seconds}`` or a csv path, see ``read_video_horizons``) caps the evaluated
    length of clean negatives whose footage was cut; ``duration(vid)`` is the evaluated length of every video.
    """

    def __init__(self, videos, events, folds, provenance=None, video_horizons=None):
        self.videos = videos
        self.events = events
        self.folds = folds
        self.provenance = provenance or {}
        self.video_horizons = {}
        self._labels = {}
        self._events = {}
        if video_horizons is not None:
            self._apply_horizons(video_horizons)

    def _apply_horizons(self, horizons):
        source = {"file": None, "sha256": None}
        if isinstance(horizons, (str, os.PathLike)):
            horizons, source = read_video_horizons(horizons)
        if not horizons and source["file"] is None:
            return
        dur = self.videos["duration_sec"].astype(float).to_dict()
        role = self.videos["role"].to_dict()
        applied, bad, n_unknown, n_not_shorter = {}, [], 0, 0
        for vid, h in horizons.items():
            h = float(h)
            if not np.isfinite(h) or h <= 0:
                raise ValueError(f"horizon of {vid} must be a finite number of seconds > 0, got {h}")
            if vid not in dur:
                n_unknown += 1
            elif h >= dur[vid] - HORIZON_TOL_SEC:
                n_not_shorter += 1
            elif role[vid] != "clean_negative":
                bad.append(f"{vid} (role {role[vid]}, catalog {dur[vid]:.3f} s -> horizon {h:.3f} s)")
            else:
                applied[vid] = h
        if bad:
            raise ValueError(f"horizons would shorten {len(bad)} video(s) that are not clean negatives (only capped "
                             f"clean negatives may be shortened; violent and hold-out videos were not capped): "
                             f"{'; '.join(bad[:5])}")
        self.video_horizons = applied
        self.provenance = {**self.provenance, "video_horizons": {
            **source, "n_listed": len(horizons), "n_not_in_catalog": n_unknown,
            "n_not_shorter_than_catalog": n_not_shorter, "n_shortened": len(applied),
            "shortened_hours_catalog": round(sum(dur[v] for v in applied) / 3600.0, 3),
            "shortened_hours_used": round(sum(applied.values()) / 3600.0, 3)}}

    def with_horizons(self, horizons):
        """Copy of this benchmark that evaluates the listed clean negatives only up to their horizon."""
        if self.provenance.get("video_horizons"):
            raise ValueError("this benchmark already has evaluation horizons")
        return Benchmark(self.videos, self.events, self.folds, self.provenance, video_horizons=horizons)

    def duration(self, vid):
        """Evaluated length of a video in seconds: the catalog duration unless a horizon shortens it."""
        h = self.video_horizons.get(vid)
        return float(self.videos.loc[vid, "duration_sec"]) if h is None else h

    @classmethod
    def load(cls, folds_path=None, videos_csv=VIDEOS_CSV, events_json=EVENTS_JSON, horizons=None):
        folds_path = folds_path or FOLDS_DEFAULT
        folds = Folds.load(folds_path)
        prov = {"folds_path": os.path.relpath(folds_path, REPO_ROOT) if folds_path.startswith(REPO_ROOT) else folds_path,
                "folds_content_sha256": folds.sha256, "videos_csv_sha256": file_sha256(videos_csv),
                "events_json_sha256": file_sha256(events_json)}
        cat = folds.d.get("catalog", {})
        if cat.get("videos_csv_sha256") and cat["videos_csv_sha256"] != prov["videos_csv_sha256"]:
            prov["note"] = ("videos.csv changed since the fold file was written (expected when the catalog is "
                            "re-run with new sources; fold membership is taken from the fold file)")
        return cls(load_catalog(videos_csv), load_events(events_json), folds, prov,
                   video_horizons=horizons)

    def segments(self, vid):
        if self.videos.loc[vid, "role"] != "violent":
            return []
        return self.events.get(vid, {}).get("segments", [])

    def excluded_spans(self, vid):
        """``(other_segments, mask_spans, soft_mask_spans)`` of a video: excluded time (never positive, negative
        or false alarm)."""
        e = self.events.get(vid, {})
        return e.get("other_segments") or [], e.get("mask_spans") or [], e.get("soft_mask_spans") or []

    def labels(self, vid):
        if vid not in self._labels:
            others, masks, soft = self.excluded_spans(vid)
            self._labels[vid] = tick_labels(self.duration(vid), self.segments(vid),
                                            other_segments=others, mask_spans=masks, soft_mask_spans=soft)
        return self._labels[vid]

    def event_rows(self, vid):
        if vid not in self._events:
            lab = self.labels(vid)
            self._events[vid] = event_table(lab.segments, blockers=blocker_spans(lab.other_segments, lab.mask_spans))
        return self._events[vid]

    def eval_ids(self, onset_sources=TRUSTED_ONSETS):
        out = []
        for vid in self.folds.video_ids:
            if vid not in self.videos.index:
                continue
            r = self.videos.loc[vid]
            if self.duration(vid) < 1.0:
                continue
            if r["role"] == "clean_negative" or (r["role"] == "violent" and r["onset_source"] in onset_sources):
                out.append(vid)
        return out


# ---------------------------------------------------------------------------------------------------------
# Score table
# ---------------------------------------------------------------------------------------------------------

def load_scores(path):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df


class ScoreTable:
    """Index of a score DataFrame by (seed, fold, video_id).

    ``horizons`` (``{video_id: seconds}``, ``Benchmark.video_horizons``): rows after a video's horizon are dropped
    before any validation, so they cannot raise or influence the table (``n_trimmed`` counts them).
    """

    def __init__(self, df, horizons=None):
        missing = {"video_id", "t_sec", "score"} - set(df.columns)
        if missing:
            raise ValueError(f"score table lacks columns {sorted(missing)}")
        df = df.copy()
        self.n_trimmed = 0
        if horizons:
            cut = df["video_id"].map(horizons)
            beyond = (cut.notna() & (pd.to_numeric(df["t_sec"], errors="coerce") > cut + 1e-6)).values
            self.n_trimmed = int(beyond.sum())
            df = df[~beyond]
        if "fold" not in df:
            df["fold"] = -1
        if "seed" not in df:
            df["seed"] = 0
        for c in ("fold", "seed"):
            v = pd.to_numeric(df[c], errors="coerce")
            if v.isna().any():
                raise ValueError(f"score table: {int(v.isna().sum())} rows with a missing / non-numeric '{c}' "
                                 f"(use -1 for fold-agnostic scores explicitly)")
            if (v != np.round(v)).any():
                raise ValueError(f"score table: non-integer '{c}' values")
            df[c] = v.astype(int)
        df["t_sec"] = df["t_sec"].astype(np.float64)
        df["score"] = df["score"].astype(np.float64)
        if not np.isfinite(df["score"].values).all():
            raise ValueError("score table contains NaN or inf scores")
        if not np.isfinite(df["t_sec"].values).all():
            raise ValueError("score table contains NaN or inf t_sec")
        dup = df.duplicated(["seed", "fold", "video_id", "t_sec"], keep=False)
        if dup.any():
            ex = df.loc[dup, ["seed", "fold", "video_id", "t_sec"]].head(3).to_dict("records")
            raise ValueError(f"score table: {int(dup.sum())} rows share a (seed, fold, video_id, t_sec) key, "
                             f"e.g. {ex}; every tick must be scored exactly once per model")
        fr = df.groupby(["seed", "video_id"])["fold"].agg(["min", "max"])
        mixed = fr[(fr["min"] == -1) & (fr["max"] >= 0)]
        if len(mixed):
            raise ValueError(f"score table: {len(mixed)} (seed, video) pairs have both fold -1 (fold-agnostic) and "
                             f"fold-specific rows, e.g. {list(mixed.index[:3])}; use one kind per video")
        df = df.sort_values(["seed", "fold", "video_id", "t_sec"], kind="mergesort").reset_index(drop=True)
        self.t = df["t_sec"].values
        self.s = df["score"].values
        self.index = {k: (int(v[0]), int(v[-1]) + 1) for k, v in
                      df.groupby(["seed", "fold", "video_id"], sort=False).indices.items()}
        self.seeds = sorted(df["seed"].unique().tolist())
        self.videos = set(df["video_id"].unique())
        self.global_min = float(self.s.min()) if len(self.s) else 0.0
        self.n_rows = len(df)
        self.is_probability = bool(len(self.s) and self.s.min() >= 0.0 and self.s.max() <= 1.0)
        self.fold_agnostic = bool(len(df) and (df["fold"] == -1).all())

    def get(self, seed, fold, vid):
        k = self.index.get((seed, fold, vid)) or self.index.get((seed, -1, vid))
        if k is None:
            return None
        return self.t[k[0]:k[1]], self.s[k[0]:k[1]]

    def aligned(self, seed, fold, vid, ticks):
        """Scores on ``ticks`` (sample-and-hold) and the number of ticks not exactly present / before any score."""
        g = self.get(seed, fold, vid)
        if g is None:
            return None, len(ticks), len(ticks)
        tr, sr = g
        idx = np.searchsorted(tr, ticks + 1e-6, side="right") - 1
        out = np.where(idx >= 0, sr[np.clip(idx, 0, None)], np.nan)
        exact = (idx >= 0) & (np.abs(tr[np.clip(idx, 0, None)] - ticks) < 1e-3)
        return out, int((~exact).sum()), int((idx < 0).sum())


# ---------------------------------------------------------------------------------------------------------
# Tick set (concatenated per-video arrays)
# ---------------------------------------------------------------------------------------------------------

class TickSet:
    def __init__(self, bench, vids, scores, folds_of=None):
        self.bench = bench
        self.vids = list(vids)
        V = len(self.vids)
        labs = [bench.labels(v) for v in self.vids]
        lens = np.array([lb.T for lb in labs], dtype=np.int64)
        self.offsets = np.r_[0, np.cumsum(lens)]
        self.vidx = np.repeat(np.arange(V), lens)
        cat = lambda xs, dt=np.float64: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)  # noqa: E731
        self.score = cat([scores[v] for v in self.vids])
        self.t = cat([lb.t_sec for lb in labs])
        self.tte = cat([lb.tte for lb in labs])
        self.masked = cat([lb.masked for lb in labs], bool)
        self.in_violence = cat([lb.in_violence for lb in labs], bool)
        self.remaining = cat([lb.remaining for lb in labs])
        self.pre = cat([lb.pre_violence for lb in labs], bool)       # before the first annotated segment
        self.excluded = cat([lb.excluded for lb in labs], bool)
        rows = bench.videos.loc[self.vids] if V else bench.videos.iloc[:0]
        self.role_v = rows["role"].values
        self.clean_v = self.role_v == "clean_negative"
        self.violent_v = self.role_v == "violent"
        self.source_v = rows["source_kind"].values
        self.onset_v = rows["onset_source"].values if "onset_source" in rows else np.full(V, "none")
        self.interp_v = rows["interpersonal"].values.astype(bool)
        self.audio_v = rows["has_audio"].values.astype(bool)
        self.group_v = rows["group"].values if "group" in rows else np.array(self.vids)
        self.duration_v = np.array([bench.duration(v) for v in self.vids], dtype=np.float64)
        self.fold_v = np.array([folds_of.get(v, -1) for v in self.vids]) if folds_of else np.full(V, -1)
        self.clean = self.clean_v[self.vidx] if V else np.zeros(0, bool)
        self.V = V
        self.events = [bench.event_rows(v) if self.violent_v[i] else [] for i, v in enumerate(self.vids)]
        # Excluded spans (FA-free windows): other segments + aftermath, hard and soft mask spans.
        self.blockers = [blocker_spans(lb.other_segments, lb.mask_spans) + [list(x) for x in lb.soft_mask_spans]
                         for lb in labs]

    def video_mask(self, mask_v):
        return np.asarray(mask_v, bool)[self.vidx]


# ---------------------------------------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------------------------------------

def bootstrap_weights(ts, n_boot, seed=0, cluster="video"):
    """``W [n_boot, V]``: how often each video is drawn in each resample (cluster bootstrap)."""
    rng = np.random.default_rng(seed)
    if n_boot <= 0 or ts.V == 0:
        return np.zeros((0, ts.V))
    if cluster in ("group", "split_group"):
        labels = ts.group_v
        if cluster == "split_group" and "split_group" in ts.bench.videos:
            labels = ts.bench.videos.loc[ts.vids, "split_group"].astype(str).values
        groups, ginv = np.unique(labels, return_inverse=True)
        G = len(groups)
        Wg = rng.multinomial(G, np.full(G, 1.0 / G), size=n_boot).astype(np.float64)
        return Wg[:, ginv]
    return rng.multinomial(ts.V, np.full(ts.V, 1.0 / ts.V), size=n_boot).astype(np.float64)


def ci(values, point):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    out = {"value": None if point is None or not np.isfinite(point) else float(point), "ci": None}
    if len(v) >= 10:
        out["ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return out


N_THREADS = min(16, os.cpu_count() or 1)


def _parallel(fn, n, min_per_thread=64):
    """Run ``fn(indices)`` over ``range(n)`` split across threads (numpy's searchsorted releases the GIL;
    every index writes a distinct row or column, so there are no races)."""
    k = max(1, min(N_THREADS, n // min_per_thread))
    if k == 1:
        fn(range(n))
        return
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(k) as ex:
        list(ex.map(fn, [range(i, n, k) for i in range(k)]))


class PairAUC:
    """AUROC from per-video pair counts so cluster-bootstrap resamples are matrix products.

    ``C[i, j]`` = number of (positive tick of video i, negative tick of video j) pairs ranked correctly
    (+0.5 for ties). AUROC(w) = wp C wn / ((wp . npos) (wn . nneg)); with w = 1 it equals the Mann-Whitney AUROC.
    """

    def __init__(self, score, vidx, pos, neg, V, neg_w=None):
        ps, pv = score[pos], vidx[pos]
        ns, nv = score[neg], vidx[neg]
        nw = np.ones(len(ns)) if neg_w is None else np.asarray(neg_w, np.float64)[neg]
        self.pvid, pinv = np.unique(pv, return_inverse=True)
        self.nvid, ninv = np.unique(nv, return_inverse=True)
        self.npos = np.bincount(pinv, minlength=len(self.pvid)).astype(np.float64)
        cnt = np.bincount(ninv, minlength=len(self.nvid))
        self.nneg = np.bincount(ninv, weights=nw, minlength=len(self.nvid)).astype(np.float64)
        self.n_pos, self.n_neg = len(ps), int((nw > 0).sum())
        Vp, Vn = len(self.pvid), len(self.nvid)
        C = np.zeros((Vp, Vn))
        if Vp and Vn:
            if Vn * len(ps) <= Vp * len(ns):
                order = np.argsort(ninv, kind="stable")
                bounds = np.r_[0, np.cumsum(cnt)]
                ns_sorted, nw_sorted = ns[order], nw[order]

                def cols(js):
                    for j in js:
                        o = np.argsort(ns_sorted[bounds[j]:bounds[j + 1]], kind="stable")
                        s_u = ns_sorted[bounds[j]:bounds[j + 1]][o]
                        cw = np.r_[0.0, np.cumsum(nw_sorted[bounds[j]:bounds[j + 1]][o])]
                        lt = np.searchsorted(s_u, ps, side="left")
                        le = np.searchsorted(s_u, ps, side="right")
                        C[:, j] = np.bincount(pinv, weights=cw[lt] + 0.5 * (cw[le] - cw[lt]), minlength=Vp)

                _parallel(cols, Vn)
            else:
                order = np.argsort(pinv, kind="stable")
                bounds = np.r_[0, np.cumsum(self.npos.astype(np.int64))]
                ps_sorted = ps[order]

                def rows(iis):
                    for i in iis:
                        p_v = np.sort(ps_sorted[bounds[i]:bounds[i + 1]])
                        gt = len(p_v) - np.searchsorted(p_v, ns, side="right")
                        eq = np.searchsorted(p_v, ns, side="right") - np.searchsorted(p_v, ns, side="left")
                        C[i, :] = np.bincount(ninv, weights=(gt + 0.5 * eq) * nw, minlength=Vn)

                _parallel(rows, Vp)
        self.C = C

    def value(self, w=None):
        if self.n_pos == 0 or self.n_neg == 0:
            return np.nan
        wp = np.ones(len(self.pvid)) if w is None else w[self.pvid]
        wn = np.ones(len(self.nvid)) if w is None else w[self.nvid]
        den = (wp @ self.npos) * (wn @ self.nneg)
        return float(wp @ self.C @ wn / den) if den > 0 else np.nan

    def boot(self, W):
        if self.n_pos == 0 or self.n_neg == 0 or len(W) == 0:
            return np.full(len(W), np.nan)
        Wp, Wn = W[:, self.pvid], W[:, self.nvid]
        num = ((Wp @ self.C) * Wn).sum(1)
        den = (Wp @ self.npos) * (Wn @ self.nneg)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)

    def stat(self, W):
        out = ci(self.boot(W), self.value())
        out.update(n_pos=int(self.n_pos), n_neg=int(self.n_neg), n_pos_videos=int(len(self.pvid)),
                   n_neg_videos=int(len(self.nvid)))
        return out


def average_precision(score, y, vidx, W=None, chunk=50):
    """sklearn-equivalent AP (ties grouped); with ``W [B, V]`` also the per-resample values (video weights).

    Ticks are sorted by decreasing score and grouped into "blocks": block k ends at the k-th distinct score that
    has a positive tick and contains every tick scored above the previous such score, down to and including
    that score (ties are one threshold, as in sklearn). Ticks below the last positive score never change AP.
    Per-video block counts (sparse ``[K, V]`` for positives and negatives) turn every resample into two sparse
    products: tp = Pos @ w, fp = Neg @ w, AP = sum_k tp_k * ctp_k / (ctp_k + cfp_k) / ctp_K.
    """
    from scipy import sparse

    y = np.asarray(y, bool)
    if len(score) == 0 or not y.any():
        return np.nan, np.full(0 if W is None else len(W), np.nan)
    order = np.argsort(-score, kind="stable")
    s, yy, v = score[order], y[order], vidx[order]
    gid = np.cumsum(np.r_[True, s[1:] != s[:-1]]) - 1        # distinct-score group of every tick
    pos_groups = np.unique(gid[yy])                            # groups holding at least one positive
    blk = np.searchsorted(pos_groups, gid, side="left")        # first positive group at or below this tick
    keep = blk < len(pos_groups)
    blk, yy, v = blk[keep], yy[keep], v[keep]
    K, V = len(pos_groups), int(vidx.max()) + 1 if W is None else W.shape[1]
    Pm = sparse.csr_matrix((np.ones(int(yy.sum())), (blk[yy], v[yy])), shape=(K, V))
    Nm = sparse.csr_matrix((np.ones(int((~yy).sum())), (blk[~yy], v[~yy])), shape=(K, V))

    def ap_cols(Wt):  # Wt [V, b] -> AP per column
        tp = np.asarray(Pm @ Wt)
        fp = np.asarray(Nm @ Wt)
        ctp, cfp = np.cumsum(tp, 0), np.cumsum(fp, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            prec = np.where(ctp + cfp > 0, ctp / (ctp + cfp), 0.0)
            P = ctp[-1]
            return np.where(P > 0, (tp * prec).sum(0) / P, np.nan)

    point = float(ap_cols(np.ones((V, 1)))[0])
    boots = np.full(0, np.nan)
    if W is not None and len(W):
        Wt = np.ascontiguousarray(W.T)
        boots = np.concatenate([ap_cols(Wt[:, c:c + chunk]) for c in range(0, len(W), chunk)])
    return point, boots


def weighted_quantiles(x, w, qs):
    x, w = np.asarray(x, float), np.asarray(w, float)
    keep = w > 0
    x, w = x[keep], w[keep]
    if len(x) == 0:
        return [np.nan] * len(qs)
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = np.cumsum(w) - 0.5 * w
    cw /= w.sum()
    return [float(np.interp(q, cw, x)) for q in qs]


# ---------------------------------------------------------------------------------------------------------
# Alarms and event-level outcomes
# ---------------------------------------------------------------------------------------------------------

def alarm_starts(score, t, vidx, offsets, theta, policy, reset=None):
    """Alarm episode starts ``(ep_video, ep_time)`` and the per-tick ``active`` mask for threshold ``theta``.

    1. raw(t) = score >= theta; 2. active(t) = at least k of the last n ticks raw (causal, per video);
    3. runs of active ticks; a run starting after < merge_gap_sec seconds of silence continues the episode;
    4. an episode starting < refractory_sec after the previous counted alarm start is absorbed.
    ``reset`` (optional bool per tick) starts a new epoch at that tick: a run that starts in a new epoch is a new
    episode whatever the gap, and the refractory clock restarts. The evaluator resets at every event's
    ``clear_start`` (end of the previous segment's aftermath), so an alarm that detected event k cannot swallow
    a fresh anticipation alarm for event k + 1. A run that is active across the reset is NOT restarted by it.
    """
    N = len(score)
    if N == 0:
        return np.zeros(0, np.int64), np.zeros(0), np.zeros(0, bool)
    raw = (score >= theta).astype(np.int64)
    cs = np.r_[0, np.cumsum(raw)]
    i = np.arange(N)
    vstart = offsets[vidx]
    lo = np.maximum(i + 1 - policy.n, vstart)
    active = (cs[i + 1] - cs[lo]) >= policy.k
    prev_active = np.r_[False, active[:-1]]
    prev_active[vstart == i] = False
    run_start = np.flatnonzero(active & ~prev_active)
    if len(run_start) == 0:
        return np.zeros(0, np.int64), np.zeros(0), active
    nxt_active = np.r_[active[1:], False]
    vend = offsets[vidx + 1] - 1
    nxt_active[vend == i] = False
    run_end = np.flatnonzero(active & ~nxt_active)
    rv = vidx[run_start]
    rs, re = t[run_start], t[run_end]
    epoch = np.cumsum(reset)[run_start] if reset is not None else np.zeros(len(run_start), np.int64)
    gap = np.r_[np.inf, rs[1:] - re[:-1] - 1.0]
    new_video = np.r_[True, rv[1:] != rv[:-1]]
    new_epoch = np.r_[True, epoch[1:] != epoch[:-1]]
    new_ep = new_video | new_epoch | (gap >= policy.merge_gap_sec)
    ev, et, eep = rv[new_ep], rs[new_ep], epoch[new_ep]
    keep = np.ones(len(et), bool)
    last_v, last_e, last_t = -1, -1, -np.inf
    for j in range(len(et)):
        if ev[j] != last_v or eep[j] != last_e:
            last_v, last_e, last_t = ev[j], eep[j], et[j]
            continue
        if et[j] - last_t < policy.refractory_sec:
            keep[j] = False
        else:
            last_t = et[j]
    return ev[keep], et[keep], active


def _first_at_or_after(ep_v, ep_t, q_v, q_t):
    """Index of the first episode of video ``q_v`` starting at or after ``q_t`` (vectorized), or -1."""
    if len(ep_v) == 0 or len(q_v) == 0:
        return np.full(len(q_v), -1)
    key = ep_v.astype(np.float64) * 1e7 + ep_t
    k = np.searchsorted(key, q_v.astype(np.float64) * 1e7 + q_t - 1e-9, side="left")
    ok = k < len(key)
    k2 = np.where(ok, k, 0)
    ok &= ep_v[k2] == q_v
    return np.where(ok, k2, -1)


class EventEval:
    """Per-video event bookkeeping for one TickSet (eligible events, FA windows, hours)."""

    def __init__(self, ts, policy):
        self.ts, self.p = ts, policy
        ev = []
        for i, rows in enumerate(ts.events):
            for r in rows:
                lo = max(r["onset"] - policy.window_sec, r["clear_start"], 0.0)
                ev.append((i, r["onset"], r["end"], r["clear_start"], lo, r["eligible30"], r["eligible60"],
                           r["index"], r.get("max_lead", r["onset"] - max(policy.k, np.ceil(r["clear_start"])))))
        a = np.array(ev, dtype=np.float64).reshape(-1, 9)
        self.ev_v = a[:, 0].astype(np.int64)
        self.ev_on, self.ev_end, self.ev_clear, self.ev_lo = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
        self.ev_e30, self.ev_e60 = a[:, 5].astype(bool), a[:, 6].astype(bool)
        self.ev_index = a[:, 7].astype(np.int64)
        self.ev_first = self.ev_index == 0
        self.ev_late = self.ev_lo >= LATE_WINDOW_SEC
        # Best lead reachable under THIS policy's k (labels assume k = 2): the Success@L ceiling.
        self.ev_maxlead = self.ev_on - np.maximum(float(policy.k), np.ceil(self.ev_clear - 1e-9))
        # Success@30 acceptance window length (for the per-event random reference).
        self.ev_w30 = np.maximum(0.0, (self.ev_on - 30.0) - self.ev_lo)
        # Epoch resets at each event's clear_start (see alarm_starts).
        self.reset = np.zeros(len(ts.score), bool)
        for j in np.flatnonzero(self.ev_clear > 0):
            i = self.ev_v[j]
            k = int(np.ceil(self.ev_clear[j] - 1e-9)) - 1          # tick index of the first tick >= clear_start
            if 0 <= k < ts.offsets[i + 1] - ts.offsets[i]:
                self.reset[ts.offsets[i] + k] = True
        # Windows where an alarm start is not a false alarm, merged per video: [t0 - 120, e + aftermath] of every
        # violent segment, plus excluded spans (other-anomaly segments + aftermath, mask spans) of every video.
        self.win = []
        self.hours_out = np.zeros(ts.V)
        wv, ws, we = [], [], []
        for i in range(ts.V):
            dur = ts.duration_v[i]
            w = [[max(0.0, r["onset"] - policy.window_sec), min(dur, r["end"] + policy.aftermath_sec)]
                 for r in ts.events[i]]
            w += [[max(0.0, s), min(dur, e)] for s, e in ts.blockers[i] if min(dur, e) > max(0.0, s)]
            w.sort()
            merged = []
            for s, e in w:
                if merged and s <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            self.win.append(np.array(merged).reshape(-1, 2))
            self.hours_out[i] = max(0.0, dur - sum(e - s for s, e in merged)) / 3600.0
            for s, e in merged:
                wv.append(i), ws.append(s), we.append(e)
        self.win_v, self.win_s, self.win_e = np.array(wv, np.int64), np.array(ws), np.array(we)
        self.win_key = self.win_v.astype(np.float64) * 1e7 + self.win_s
        self.clean_hours = float(self.hours_out[ts.clean_v].sum())

    def _outside(self, ep_v, ep_t):
        """Bool per episode: the start lies outside every window of its video (i.e. a false alarm)."""
        if len(ep_v) == 0:
            return np.zeros(0, bool)
        if len(self.win_v) == 0:
            return np.ones(len(ep_v), bool)
        k = np.searchsorted(self.win_key, ep_v.astype(np.float64) * 1e7 + ep_t + 1e-9, side="right") - 1
        k2 = np.clip(k, 0, None)
        inside = (k >= 0) & (self.win_v[k2] == ep_v) & (ep_t <= self.win_e[k2] + 1e-9)
        return ~inside

    def _starts(self, theta_v):
        ts = self.ts
        return alarm_starts(ts.score, ts.t, ts.vidx, ts.offsets, theta_v[ts.vidx], self.p, reset=self.reset)

    def _leads(self, ep_v, ep_t):
        k = _first_at_or_after(ep_v, ep_t, self.ev_v, self.ev_lo)
        st = np.where(k >= 0, ep_t[np.clip(k, 0, None)] if len(ep_t) else 0.0, np.inf)
        return np.where(st < self.ev_on, self.ev_on - st, np.nan)

    def outcomes(self, theta_v):
        """Outcomes for per-video thresholds ``theta_v [V]`` (one per fold mapped to its videos)."""
        ts = self.ts
        V = ts.V
        ep_v, ep_t, active = self._starts(theta_v)
        fa = np.bincount(ep_v[self._outside(ep_v, ep_t)], minlength=V).astype(float) if V else np.zeros(0)
        lead = self._leads(ep_v, ep_t)
        # Detection: any active tick in [max(t0 - 120, clear_start), e] (tick index = offset + t - 1).
        cs = np.r_[0, np.cumsum(active)]
        T_v = np.diff(ts.offsets)
        a = np.clip(np.ceil(self.ev_lo - 1e-9).astype(np.int64), 1, None)
        b = np.minimum(np.floor(self.ev_end + 1e-9).astype(np.int64), T_v[self.ev_v])
        off = ts.offsets[self.ev_v]
        det = (b >= a) & ((cs[off + np.maximum(b, a - 1)] - cs[off + a - 1]) > 0)
        active_ticks = np.bincount(ts.vidx[active], minlength=V).astype(float) if V else np.zeros(0)
        return {"fa": fa, "lead": lead, "det": det, "n_alarms": np.bincount(ep_v, minlength=V).astype(float),
                "active_ticks": active_ticks}

    def fa_clean(self, theta_v):
        """FA/h on clean negatives only (fast path for threshold search)."""
        ep_v, ep_t, _ = self._starts(theta_v)
        sel = self.ts.clean_v[ep_v] if len(ep_v) else np.zeros(0, bool)
        n = int(self._outside(ep_v[sel], ep_t[sel]).sum()) if sel.any() else 0
        return n / self.clean_hours if self.clean_hours > 0 else np.nan

    def fa_counts_clean(self, theta_v):
        """Per-video false-alarm counts on clean negatives ``[V]`` (zero for other videos)."""
        ep_v, ep_t, _ = self._starts(theta_v)
        out = np.zeros(self.ts.V)
        sel = self.ts.clean_v[ep_v] if len(ep_v) else np.zeros(0, bool)
        if sel.any():
            fa = self._outside(ep_v[sel], ep_t[sel])
            np.add.at(out, ep_v[sel][fa], 1.0)
        return out

    def fast(self, theta_v):
        """``(FA/h clean, Success@30 over eligible30, Success@30 over late eligible30)`` without bootstrap."""
        ep_v, ep_t, _ = self._starts(theta_v)
        sel = self.ts.clean_v[ep_v] if len(ep_v) else np.zeros(0, bool)
        n = int(self._outside(ep_v[sel], ep_t[sel]).sum()) if sel.any() else 0
        lead = self._leads(ep_v, ep_t)
        hit = np.nan_to_num(lead, nan=-1.0) >= 30
        e30, late = self.ev_e30, self.ev_e30 & self.ev_late
        return (n / self.clean_hours if self.clean_hours > 0 else np.nan,
                float(hit[e30].mean()) if e30.any() else np.nan, float(hit[late].mean()) if late.any() else np.nan)

    def summarize(self, out, W=None, ev_mask=None, clean_mask=None, all_mask=None):
        """Scalars (with bootstrap CIs) from ``outcomes``; masks restrict events / videos for breakdowns."""
        ts = self.ts
        ev_mask = np.ones(len(self.ev_v), bool) if ev_mask is None else ev_mask
        clean_mask = ts.clean_v if clean_mask is None else (clean_mask & ts.clean_v)
        all_mask = (ts.clean_v | ts.violent_v) if all_mask is None else all_mask
        W = np.zeros((0, ts.V)) if W is None else W
        res = {}

        def rate(sel, val):
            pt = val[sel].mean() if sel.any() else np.nan
            if len(W) and sel.any():
                Wv = W[:, self.ev_v[sel]]
                with np.errstate(invalid="ignore", divide="ignore"):
                    b = (Wv * val[sel]).sum(1) / Wv.sum(1)
            else:
                b = []
            r = ci(b, pt)
            r["n"] = int(sel.sum())
            return r

        lead = out["lead"]
        hit = np.isfinite(lead)
        for name, emask in (("eligible30", self.ev_e30), ("eligible60", self.ev_e60),
                            ("eligible30_late", self.ev_e30 & self.ev_late)):
            sel = emask & ev_mask
            block = {}
            for L in SUCCESS_LEADS:
                val = (hit & (np.nan_to_num(lead, nan=-1.0) >= L)) if L > 0 else hit
                block[f"success{L}"] = rate(sel, val.astype(float))
            L_cap = 60 if name == "eligible60" else 30
            block[f"success{L_cap}_ceiling"] = float((self.ev_maxlead[sel] >= L_cap).mean()) if sel.any() else None
            if name == "eligible60":
                val = hit & (np.nan_to_num(lead, nan=-1.0) >= 60)
                block["success60"] = rate(sel, val.astype(float))
            if name == "eligible30_late":
                res[name] = block
                continue
            ls = lead[sel & hit]
            lv = self.ev_v[sel & hit]
            q = weighted_quantiles(ls, np.ones(len(ls)), (0.25, 0.5, 0.75)) if len(ls) else [np.nan] * 3
            lb = {"median": [], "p25": [], "p75": [], "mean": []}
            for b in range(len(W)):
                wb = W[b, lv]
                if wb.sum() > 0:
                    qb = weighted_quantiles(ls, wb, (0.25, 0.5, 0.75))
                    lb["p25"].append(qb[0]); lb["median"].append(qb[1]); lb["p75"].append(qb[2])
                    lb["mean"].append(float((ls * wb).sum() / wb.sum()))
            block["lead_sec"] = {
                "n_alarmed": int(len(ls)), "median": ci(lb["median"], q[1]), "p25": ci(lb["p25"], q[0]),
                "p75": ci(lb["p75"], q[2]), "mTTA": ci(lb["mean"], float(ls.mean()) if len(ls) else np.nan)}
            res[name] = block

        def fa_rate(vmask, hours_v):
            n = out["fa"]
            hrs = hours_v[vmask].sum()
            pt = n[vmask].sum() / hrs if hrs > 0 else np.nan
            b = []
            if len(W) and vmask.any():
                with np.errstate(invalid="ignore", divide="ignore"):
                    b = (W[:, vmask] @ n[vmask]) / (W[:, vmask] @ hours_v[vmask])
            r = ci(b, pt)
            r.update(n_false_alarms=int(n[vmask].sum()), hours=round(float(hrs), 3))
            return r, b

        res["fa_per_hour_clean"], fa_b = fa_rate(clean_mask, self.hours_out)
        res["fa_per_hour_all"], _ = fa_rate(all_mask, self.hours_out)
        # Long clean videos (UCF normals up to 9 h) dominate the clean hours of a fold; this variant drops
        # clean videos over 1 h so a few very long CCTV recordings cannot carry the FA/h alone.
        res["fa_per_hour_clean_videos_under_1h"], _ = fa_rate(clean_mask & (ts.duration_v <= 3600.0), self.hours_out)
        far = res["fa_per_hour_clean"]["value"]
        sel30 = self.ev_e30 & ev_mask
        if far is not None:
            pr_nom = 1.0 - np.exp(-far * SUCCESS_WINDOW_SEC / 3600.0)
            pr_ev = float(np.mean(1.0 - np.exp(-far * self.ev_w30[sel30] / 3600.0))) if sel30.any() else np.nan
            b_nom = 1.0 - np.exp(-np.asarray(fa_b) * SUCCESS_WINDOW_SEC / 3600.0) if len(fa_b) else []
            res["random_reference_success30"] = {
                "nominal_W90": ci(b_nom, pr_nom), "per_event_window": pr_ev,
                "formula": "1 - exp(-FAR * W), FAR = test FA/h on clean negatives. Assumes alarm times independent "
                           "of position in the clip; compare with position_null_success30 too"}
        # Time in alarm on clean negatives: exposes degenerate near-always-on operating points, which the
        # episode-based FA/h undercounts (a merged alarm counts once however long it lasts).
        ticks_v = np.diff(ts.offsets).astype(float)
        cm = clean_mask
        pt = out["active_ticks"][cm].sum() / ticks_v[cm].sum() if ticks_v[cm].sum() > 0 else np.nan
        bt = []
        if len(W) and cm.any():
            with np.errstate(invalid="ignore", divide="ignore"):
                bt = (W[:, cm] @ out["active_ticks"][cm]) / (W[:, cm] @ ticks_v[cm])
        res["alarm_time_fraction_clean"] = ci(bt, pt)
        res["detection"] = rate(ev_mask.copy(), out["det"].astype(float))
        res["detection"]["definition"] = "any active alarm tick in [max(t0 - 120, clear_start), e], all events"
        return res


def threshold_aware_bootstrap(val_ee, ee, W, target, n_boot, seed=0, max_grid=300, cluster="video"):
    """CIs for Success@30 and test FA/h that include threshold-selection noise.

    In every replicate the validation (threshold-set) videos of every fold are resampled with replacement and
    theta is re-chosen with the same top-down rule on a grid of at most ``max_grid`` candidate scores per fold
    (all unique validation scores down to the theta where FA/h reaches 4x the target, or quantiles of them), then
    the test outcomes at those thetas are weighted with the replicate's test-video weights ``W[b]``.
    """
    ts = ee.ts
    rng = np.random.default_rng(seed + 104729)
    B = min(n_boot, len(W)) if len(W) else n_boot
    nf = len(val_ee)
    theta_b = np.zeros((B, nf))
    for f in range(nf):
        vee = val_ee[f]
        vts = vee.ts
        lo, _ = choose_threshold(vee, 4.0 * target)
        u = np.unique(vts.score)
        u = u[u >= lo][::-1]
        if len(u) > max_grid:
            u = np.unique(np.quantile(u, np.linspace(0, 1, max_grid)))[::-1]
        cand = np.r_[np.inf, u]
        Nm = np.stack([vee.fa_counts_clean(np.full(vts.V, c)) for c in cand])           # [K, Vval]
        hrs = np.where(vts.clean_v, vee.hours_out, 0.0)
        if cluster == "split_group":
            Wv = bootstrap_weights(vts, B, seed + 104729 * (f + 1), cluster="split_group")
        else:
            Wv = rng.multinomial(vts.V, np.full(vts.V, 1.0 / vts.V), size=B).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            fa = (Wv @ Nm.T) / (Wv @ hrs)[:, None]                                       # [B, K]
        exceed = ~(fa <= target)
        first = np.where(exceed.any(1), exceed.argmax(1), len(cand))
        theta_b[:, f] = cand[np.maximum(first - 1, 0)]
    cache = {}
    s30, far = np.full(B, np.nan), np.full(B, np.nan)
    hrs_t = np.where(ts.clean_v, ee.hours_out, 0.0)
    e30 = ee.ev_e30
    for b in range(B):
        key = tuple(theta_b[b])
        if key not in cache:
            ep_v, ep_t, _ = ee._starts(np.asarray(key)[ts.fold_v])
            sel = ts.clean_v[ep_v] if len(ep_v) else np.zeros(0, bool)
            fa_v = np.zeros(ts.V)
            if sel.any():
                np.add.at(fa_v, ep_v[sel][ee._outside(ep_v[sel], ep_t[sel])], 1.0)
            hit = (np.nan_to_num(ee._leads(ep_v, ep_t), nan=-1.0) >= 30).astype(float)
            cache[key] = (fa_v, hit)
        fa_v, hit = cache[key]
        w = W[b] if len(W) else np.ones(ts.V)
        we = w[ee.ev_v[e30]]
        s30[b] = (we * hit[e30]).sum() / we.sum() if we.sum() > 0 else np.nan
        far[b] = (w @ fa_v) / (w @ hrs_t) if (w @ hrs_t) > 0 else np.nan
    q = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]  # noqa: E731
    return {"n_boot": int(B), "success30_ci": q(s30), "fa_per_hour_clean_ci": q(far),
            "distinct_threshold_sets": len(cache),
            "theta_per_fold_p5_p50_p95": {str(f): [float(x) for x in np.percentile(theta_b[:, f], [5, 50, 95])]
                                          for f in range(nf)},
            "note": ("validation split_groups resampled and theta re-chosen; test split_groups resampled jointly"
                     if cluster == "split_group" else
                     "validation videos resampled and theta re-chosen per replicate; test videos resampled jointly")}


def choose_threshold(ee, target, max_cand=None, coarse=None):
    """Operating threshold on a validation TickSet: lower theta from +inf over the unique validation scores until
    the clean-negative FA/h first exceeds ``target``; return the lowest threshold before that point.

    FA/h is NOT monotone in theta: once alarms become near-continuous, merging turns them into one long
    episode per video and FA/h falls again (an always-on alarm counts once per video). A bottom-up search would
    pick that degenerate regime, so the search runs top-down over the EXACT unique scores: a geometric scan
    (1, 2, 4, 8, ... unique values below +inf) finds the first candidate over the target, then bisection over
    the unique scores between it and the last feasible candidate finds the boundary to one unique score. This
    equals the brute-force top-down rule whenever FA/h is monotone inside that last geometric bracket (always
    below saturation; tested against brute force). ``max_cand`` / ``coarse`` are ignored (kept for
    compatibility). Returns ``(theta, validation FA/h)``.
    """
    ts = ee.ts
    cand = np.r_[np.unique(ts.score), np.inf]
    top = len(cand) - 1
    cache = {}
    ones = np.ones(ts.V)

    def fa(ix):
        if ix not in cache:
            cache[ix] = ee.fa_clean(ones * cand[ix])
        return cache[ix]

    ok, bad = top, None
    step = 1
    while True:
        ix = max(0, top - step)
        if fa(ix) <= target:
            ok = ix
            if ix == 0:
                break
            step *= 2
        else:
            bad = ix
            break
    if bad is not None:
        lo, hi = bad, ok
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if fa(mid) <= target:
                hi = mid
            else:
                lo = mid
        ok = hi
    return float(cand[ok]), float(fa(ok))


# ---------------------------------------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------------------------------------

def _collect(bench, table, seed, vids_by_fold, cfg, report):
    """``{vid: aligned scores}`` for {fold: [vids]}; fills or raises on missing ticks."""
    scores, dropped = {}, []
    held = lead_fill = 0
    for f, vids in vids_by_fold.items():
        for v in vids:
            ticks = bench.labels(v).t_sec
            s, n_not_exact, n_before = table.aligned(seed, f, v, ticks)
            if s is None:
                dropped.append(v)
                continue
            held += n_not_exact - n_before
            lead_fill += n_before
            if n_before:
                s = np.where(np.isnan(s), table.global_min, s)
            scores[v] = s
    report["missing_videos"] = report.get("missing_videos", 0) + len(dropped)
    report["held_ticks"] = report.get("held_ticks", 0) + int(held)
    report["filled_leading_ticks"] = report.get("filled_leading_ticks", 0) + int(lead_fill)
    if (dropped or held or lead_fill) and not cfg.allow_missing:
        raise ValueError(f"seed {seed}: {len(dropped)} videos without scores (e.g. {dropped[:3]}), {held} ticks "
                         f"not exactly present, {lead_fill} ticks before the first score. Score every tick of "
                         f"every video or pass --allow_missing / --restrict_to_scored.")
    if dropped:
        report.setdefault("missing_video_examples", []).extend(dropped[:5])
    return scores


# Log-spaced bins of seconds since video start (b_time_fit, the source-blind position control).
TIME_EDGES = (0, 10, 20, 30, 45, 60, 90, 120, 180, 240, 300, 450, 600, 900, 1200, 1800, 3600, np.inf)
# Finer bins for the time matching, the position-oracle null and b_time_source_fit: 5 s steps up to 2 min.
TIME_MATCH_EDGES = tuple(range(0, 125, 5)) + (150, 180, 240, 300, 450, 600, 900, 1200, 1800, 3600, np.inf)


def time_strata(t, source=None, edges=TIME_MATCH_EDGES):
    """Stratum id per tick: time-since-start bin, crossed with the source kind when ``source`` is given."""
    nb = len(edges) - 1
    b = np.clip(np.searchsorted(np.asarray(edges[1:-1]), t, side="right"), 0, nb - 1)
    if source is None:
        return b
    _, sid = np.unique(np.asarray(source), return_inverse=True)
    return sid * nb + b


def time_match_weights(t, pos, neg, edges=TIME_EDGES, source=None):
    """Per-tick weights for negatives so their distribution over (source kind x) seconds-since-start matches the
    positives'; also the positives that have comparable negatives.

    w(stratum) = P_pos(stratum) / P_neg(stratum); negatives in strata without positives get 0, and positives in
    strata without negatives are dropped (returned mask), so a score that is any function of (source, time bin)
    alone has AUROC exactly 0.5 (up to within-bin effects). Weights depend only on labels. With ``source=None``
    this is the v1 source-blind matching. Returns ``(neg_weights [N], pos_mask [N])``.
    """
    st = time_strata(t, source, edges)
    n = int(st.max()) + 1 if len(st) else 1
    hn = np.bincount(st[neg], minlength=n).astype(float)
    pos2 = pos & (hn[st] > 0)
    hp = np.bincount(st[pos2], minlength=n).astype(float)
    if hp.sum() == 0 or hn.sum() == 0:
        return np.zeros(len(t)), pos2
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(hn > 0, (hp / hp.sum()) / (hn / hn.sum()), 0.0)
    return np.where(neg, r[st], 0.0), pos2


def _src_tick(ts):
    return ts.source_v[ts.vidx] if ts.V else np.zeros(0, object)


def threshold_free(ts, W, cfg, ev_mask_fn=None):
    """Far-window, per-horizon, detection AUROC/AP for one TickSet.

    Headline frame metrics use only PRE-VIOLENCE ticks (before the first annotated segment of the clip) on the
    positive and the violent-negative side: later segments' lead-ins come after violence the model has already
    seen, so "remembers past violence" would separate them from clean negatives without any anticipation (a
    memory oracle gets 0.70 far AUROC on the all-segments version). ``*_all_segments`` keep the spec's literal
    definitions for reference.
    """
    M = {}
    ok = ~ts.masked
    src = _src_tick(ts)
    far_all = ok & (ts.tte > cfg.far_lo) & (ts.tte <= cfg.far_hi)
    far_pos = far_all & ts.pre
    neg_clean = ts.clean & ok & (ts.remaining >= cfg.far_hi)
    neg_clean_unc = ts.clean & ok
    neg_sec_all = ok & (ts.tte > cfg.neg_window) & (ts.remaining >= cfg.far_hi)
    neg_sec = neg_sec_all & ts.pre
    w_st, pos_st = time_match_weights(ts.t, far_pos, neg_clean, TIME_MATCH_EDGES, source=src)
    tm = PairAUC(ts.score, ts.vidx, pos_st, neg_clean, ts.V, neg_w=w_st).stat(W)
    tm["definition"] = ("clean negatives re-weighted to the positives' (source kind x seconds-since-start) "
                        "distribution (5 s bins up to 120 s); positives without comparable negatives dropped")
    w_t, pos_t = time_match_weights(ts.t, far_pos, neg_clean, TIME_EDGES)
    M["far_window_auroc"] = {
        "clean": PairAUC(ts.score, ts.vidx, far_pos, neg_clean, ts.V).stat(W),
        "secondary": PairAUC(ts.score, ts.vidx, far_pos, neg_sec, ts.V).stat(W),
        "clean_time_matched": tm,
        "clean_time_matched_source_blind": PairAUC(ts.score, ts.vidx, pos_t, neg_clean, ts.V, neg_w=w_t).stat(W),
        "clean_uncensored": PairAUC(ts.score, ts.vidx, far_pos, neg_clean_unc, ts.V).stat(W),
        "clean_all_segments": PairAUC(ts.score, ts.vidx, far_all, neg_clean, ts.V).stat(W),
        "secondary_all_segments": PairAUC(ts.score, ts.vidx, far_all, neg_sec_all, ts.V).stat(W),
    }
    M["far_window_auroc"]["definition"] = {
        "clean": "positives: pre-violence unmasked ticks with 30 < tte <= 60 (first segment only); negatives: "
                 "clean-negative ticks with >= 60 s of footage left",
        "secondary": "negatives: pre-violence unmasked ticks of any universe video with no onset within 120 s",
        "clean_uncensored": "as clean, without the 60 s right-censoring of clean negatives",
        "clean_all_segments": "spec literal: positives of every segment (also after earlier violence)"}
    hz = {}
    for H in cfg.horizons:
        pos_all = ok & (ts.tte <= H)
        neg_all = ok & (ts.tte > cfg.neg_window) & (ts.remaining >= H)
        pos, neg = pos_all & ts.pre, neg_all & ts.pre
        a = PairAUC(ts.score, ts.vidx, pos, neg, ts.V).stat(W)
        w_h, pos_h = time_match_weights(ts.t, pos, neg, TIME_MATCH_EDGES, source=src)
        a_tm = PairAUC(ts.score, ts.vidx, pos_h, neg, ts.V, neg_w=w_h).stat(W)
        sel = pos | neg
        ap, apb = average_precision(ts.score[sel], pos[sel], ts.vidx[sel], W)
        hz[str(H)] = {"auroc": a, "auroc_time_matched": a_tm, "ap": ci(apb, ap),
                      "base_rate": float(pos.sum() / max(1, sel.sum())),
                      "auroc_all_segments": PairAUC(ts.score, ts.vidx, pos_all, neg_all, ts.V).stat(W)}
    M["horizon"] = hz
    det_pos = ts.in_violence & ts.video_mask(ts.violent_v)
    M["detection_auroc"] = PairAUC(ts.score, ts.vidx, det_pos, ts.clean & ok, ts.V).stat(W)
    return M


def calibration(ts, W, cfg, is_prob):
    if not is_prob:
        return {"note": "scores are not in [0, 1]; calibration not computed"}
    sel = ~ts.masked & ((ts.tte <= 30) | (ts.remaining >= 30))
    p = ts.score[sel]
    y = (ts.tte[sel] <= 30).astype(float)
    v = ts.vidx[sel]
    se_v = np.bincount(v, weights=(p - y) ** 2, minlength=ts.V)
    n_v = np.bincount(v, minlength=ts.V).astype(float)
    brier = se_v.sum() / n_v.sum() if n_v.sum() else np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        bb = (W @ se_v) / (W @ n_v) if len(W) else []
    base = y.mean() if len(y) else np.nan
    edges = np.linspace(0, 1, cfg.reliability_bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, cfg.reliability_bins - 1)
    rel = []
    ece = 0.0
    for k in range(cfg.reliability_bins):
        m = b == k
        if m.any():
            rel.append({"bin": [float(edges[k]), float(edges[k + 1])], "n": int(m.sum()),
                        "mean_pred": float(p[m].mean()), "frac_pos": float(y[m].mean())})
            ece += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return {"target": "P(onset within 30 s) on unmasked, uncensored ticks", "brier": ci(bb, brier),
            "brier_climatology": float(base * (1 - base)) if np.isfinite(base) else None,
            "base_rate": float(base), "ece": float(ece), "reliability": rel}


def position_null(ee, max_thresholds=50):
    """Position-oracle reference curve for Success@30: what alarms placed by (source kind x time bin) alone reach.

    Position-only policies (e.g. "raise one alarm at the start of every movie clip"):
      * ``hit``    P(tick lies in the Success@30 window [lo, t0 - 30] of an eligible30 event | stratum)
      * ``tte60``  P(onset within 60 s | stratum)  (what b_time_fit / b_time_source_fit fit on training folds)
      * ``tte120`` P(onset within 120 s | stratum)
        each with strata = time bin x source kind or time bin alone, and 5 s bins (``TIME_MATCH_EDGES``) or the
        coarse ``TIME_EDGES``; rates are CROSS-FITTED on the evaluated ticks (a fold's ticks get the rate
        estimated on the other folds' videos, add-one-half smoothing), so small test sets do not inflate the null
        by overfitting (in-sample rates when the TickSet has no fold labels)
      * ``start``  one alarm on ticks 1-3 of every clip of a subset of source kinds (every subset)
      * ``after``  alarm from time T on (score = t, i.e. B-time), for all clips and for each source kind alone
    Each score family is swept over at most ``max_thresholds`` global thresholds with the evaluator's alarm
    policy and the best thresholds are picked on the evaluated set itself (an optimistic choice, so the null errs
    towards being strict). The null is the upper CONCAVE HULL of all those (FA/h, Success@30) points (mixtures of position
    policies, e.g. different ones per fold, reach any point between two of them); ``position_null_success30`` at
    an operating point = the hull at that point's test FA/h (``null_at``). The Poisson
    formula 1 - exp(-FAR * W) assumes alarm times independent of position, which VA30 violates: onsets cluster
    early in their clips, so a model must beat this null, not only the Poisson level.
    """
    ts = ee.ts
    if ts.V == 0 or not ee.ev_e30.any():
        return {}
    N = len(ts.t)
    hit = np.zeros(N, bool)
    for j in np.flatnonzero(ee.ev_e30):
        i = ee.ev_v[j]
        a = max(1, int(np.ceil(ee.ev_lo[j] - 1e-9)))
        b = int(np.floor(ee.ev_on[j] - 30.0 + 1e-9))
        if b >= a:
            hit[ts.offsets[i] + a - 1: ts.offsets[i] + b] = True
    src = _src_tick(ts)
    targets = {"hit": hit.astype(float), "tte60": (~ts.masked & (ts.tte <= 60.0)).astype(float),
               "tte120": (~ts.masked & (ts.tte <= 120.0)).astype(float)}
    fams = {}
    fold_t = ts.fold_v[ts.vidx]
    folds = np.unique(ts.fold_v)
    cross = len(folds) > 1 and (ts.fold_v >= 0).all()
    for ename, edges in (("5s", TIME_MATCH_EDGES), ("coarse", TIME_EDGES)):
        for sname, sv in (("source", src), ("blind", None)):
            st = time_strata(ts.t, sv, edges)
            n = int(st.max()) + 1
            den_all = np.bincount(st, minlength=n).astype(float)
            for tname, y in targets.items():
                num_all = np.bincount(st, weights=y, minlength=n)
                if not cross:
                    fams[f"{tname}|{sname}|{ename}"] = ((num_all + 0.5) / (den_all + 1.0))[st]
                    continue
                score = np.zeros(N)
                for f in folds:     # cross-fitted: the rate of a fold's ticks comes from the other folds only
                    m = fold_t == f
                    num = num_all - np.bincount(st[m], weights=y[m], minlength=n)
                    den = den_all - np.bincount(st[m], minlength=n)
                    score[m] = ((num + 0.5) / (den + 1.0))[st[m]]
                fams[f"{tname}|{sname}|{ename}"] = score
    saved = ts.score
    pts = []
    try:
        for name, score in fams.items():
            ths = np.unique(score[score > 0])[::-1]
            if len(ths) > max_thresholds:
                ths = np.unique(np.quantile(ths, np.linspace(0, 1, max_thresholds)))[::-1]
            ts.score = score
            for th in ths:
                fa, s30, s30_late = ee.fast(np.full(ts.V, th))
                pts.append({"family": name, "theta": float(th), "fa_per_hour_clean": float(fa), "success30": s30,
                            "success30_late": s30_late})
        kinds = sorted(set(ts.source_v))
        for name, sel in [("after", None)] + [(f"after:{k}", k) for k in kinds]:
            sc = ts.t.astype(float) if sel is None else np.where(src == sel, ts.t, 0.0)
            ths = np.unique(np.quantile(sc[sc > 0], np.linspace(0, 1, max_thresholds)))[::-1] if (sc > 0).any() else []
            ts.score = sc
            for th in ths:
                fa, s30, s30_late = ee.fast(np.full(ts.V, th))
                pts.append({"family": name, "theta": float(th), "fa_per_hour_clean": float(fa), "success30": s30,
                            "success30_late": s30_late})
        early = ts.t <= 3.0
        for mask in range(1, 2 ** len(kinds)):
            sel = [k for b, k in enumerate(kinds) if mask >> b & 1]
            ts.score = (early & np.isin(src, sel)).astype(float)
            fa, s30, s30_late = ee.fast(np.full(ts.V, 0.5))
            pts.append({"family": "start:" + "+".join(sel), "theta": 0.5, "fa_per_hour_clean": float(fa),
                        "success30": s30, "success30_late": s30_late})
    finally:
        ts.score = saved
    pts = [p for p in pts if np.isfinite(p["fa_per_hour_clean"])]
    fams_n = {}
    for p in pts:
        fams_n[p["family"].split(":")[0]] = fams_n.get(p["family"].split(":")[0], 0) + 1
    return {"success30": _concave_hull([(p["fa_per_hour_clean"], np.nan_to_num(p["success30"]), p["family"])
                                        for p in pts]),
            "success30_late": _concave_hull([(p["fa_per_hour_clean"], np.nan_to_num(p["success30_late"]), p["family"])
                                             for p in pts]),
            "n_points_per_family": fams_n,
            "definition": "upper concave hull of (test FA/h clean, Success@30) over position-only oracle policies "
                          "and their mixtures; null_at() interpolates linearly"}


def _concave_hull(points):
    """Upper concave hull through (0, 0) of ``[(x, y, label)]``: ``[[x, y, label], ...]`` sorted by x.

    Mixing two position-only policies (e.g. per fold, or on a random half of the videos) reaches every point on the
    segment between them, so the least favourable position null is the concave majorant, not the staircase.
    """
    pts = sorted({(round(float(x), 6), round(float(y), 6), lab) for x, y, lab in points if np.isfinite(x)},
                 key=lambda q: (q[0], -q[1]))
    hull = [(0.0, 0.0, "origin")]
    for x, y, lab in pts:
        if y <= hull[-1][1] + 1e-12:
            continue                      # never below an earlier (cheaper) point
        while len(hull) >= 2:
            (x1, y1, _), (x2, y2, _) = hull[-2], hull[-1]
            if (y2 - y1) * (x - x1) <= (y - y1) * (x2 - x1) + 1e-15:   # hull[-1] lies on or below the chord
                hull.pop()
            else:
                break
        hull.append((x, y, lab))
    return [[x, y, lab] for x, y, lab in hull]


def null_at(curve, fa, key="success30"):
    """Value of the position null (``position_null``) at FA/h ``fa``: linear interpolation on the concave hull,
    flat beyond its last vertex."""
    if fa is None or not curve or not curve.get(key):
        return None
    h = np.array([[x, y] for x, y, _ in curve[key]], dtype=float)
    return float(np.interp(fa, h[:, 0], h[:, 1]))


def evaluate_seed(bench, table, seed, eval_set, cfg, cov):
    folds = bench.folds
    nf = folds.n_folds
    test_by_fold = {f: [v for v in folds.test_ids(f) if v in eval_set] for f in range(nf)}
    if cfg.threshold_set == "nontest":
        if not table.fold_agnostic:
            raise ValueError("threshold_set='nontest' is only valid for fold-agnostic (fold -1) score tables: a "
                             "trained model must choose thresholds on its validation slice")
        val_by_fold = {f: [v for v in folds.video_ids if v in eval_set and folds.fold_of(v) != f]
                       for f in range(nf)}
    else:
        val_by_fold = {f: [v for v in folds.val_ids(f) if v in eval_set] for f in range(nf)}
    cov_test, cov_val = cov.setdefault("test", {}), cov.setdefault("val", {})
    test_scores = _collect(bench, table, seed, test_by_fold, cfg, cov_test)
    folds_of = {v: f for f, vs in test_by_fold.items() for v in vs}
    test_vids = [v for f in range(nf) for v in test_by_fold[f] if v in test_scores]
    ts = TickSet(bench, test_vids, test_scores, folds_of)
    W = bootstrap_weights(ts, cfg.n_boot, cfg.boot_seed + 7919 * seed, cfg.cluster)
    t0 = time.time()
    M = {"counts": {
        "test_videos": ts.V, "violent_videos": int(ts.violent_v.sum()), "clean_negative_videos": int(ts.clean_v.sum()),
        "clean_negative_hours": round(float(ts.duration_v[ts.clean_v].sum()) / 3600, 2),
        "test_hours": round(float(ts.duration_v.sum()) / 3600, 2), "ticks": int(len(ts.score)),
    }}
    if bench.provenance.get("video_horizons"):
        cat_s = bench.videos.loc[ts.vids, "duration_sec"].astype(np.float64).values if ts.V else np.zeros(0)
        M["counts"]["video_horizons"] = {
            "clean_negative_videos_shortened": int(sum(v in bench.video_horizons
                                                       for v in np.array(ts.vids)[ts.clean_v])),
            "clean_negative_hours_used": round(float(ts.duration_v[ts.clean_v].sum()) / 3600, 2),
            "clean_negative_hours_catalog": round(float(cat_s[ts.clean_v].sum()) / 3600, 2)}
    M.update(threshold_free(ts, W, cfg))
    M["calibration"] = calibration(ts, W, cfg, table.is_probability)
    M["per_fold_far_window_auroc_clean"] = {}
    pa = PairAUC(ts.score, ts.vidx, ~ts.masked & ts.pre & (ts.tte > cfg.far_lo) & (ts.tte <= cfg.far_hi),
                 ts.clean & ~ts.masked & (ts.remaining >= cfg.far_hi), ts.V)
    for f in range(nf):
        w = (ts.fold_v == f).astype(float)
        M["per_fold_far_window_auroc_clean"][str(f)] = pa.value(w)

    # Event-level: thresholds per fold chosen on the fold's validation slice.
    ee = EventEval(ts, cfg.policy)
    M["counts"].update(eligible30_events=int(ee.ev_e30.sum()), eligible60_events=int(ee.ev_e60.sum()),
                       all_events=int(len(ee.ev_v)), eligible30_first_segment=int((ee.ev_e30 & ee.ev_first).sum()),
                       eligible30_late_window=int((ee.ev_e30 & ee.ev_late).sum()),
                       clean_negative_hours_unmasked=round(ee.clean_hours, 2))
    val_ee = {}
    for f in range(nf):
        vs = _collect(bench, table, seed, {f: val_by_fold[f]}, cfg, cov_val)
        vts = TickSet(bench, list(vs), vs)
        if not vts.clean_v.any():
            raise ValueError(f"fold {f} validation slice has no clean negatives")
        val_ee[f] = EventEval(vts, cfg.policy)
    M["counts"]["threshold_set"] = cfg.threshold_set
    M["counts"]["val_videos_per_fold"] = {str(f): val_ee[f].ts.V for f in range(nf)}
    M["counts"]["val_clean_negative_hours_per_fold"] = {str(f): round(val_ee[f].clean_hours, 2) for f in range(nf)}
    null = position_null(ee) if cfg.position_null else {}
    ops = {}
    op_out = {}
    for target in cfg.fa_targets:
        theta_f, val_fa = {}, {}
        for f in range(nf):
            theta_f[f], val_fa[f] = choose_threshold(val_ee[f], target)
        theta_v = np.array([theta_f[f] for f in ts.fold_v], dtype=np.float64)
        out = ee.outcomes(theta_v)
        key = f"fa{int(target) if float(target).is_integer() else target}"
        op = ee.summarize(out, W)
        op["target_val_fa_per_hour"] = target
        op["theta_per_fold"] = {str(f): theta_f[f] for f in range(nf)}
        op["val_fa_per_hour_per_fold"] = {str(f): val_fa[f] for f in range(nf)}
        far = op["fa_per_hour_clean"]["value"]
        op["position_null_success30"] = null_at(null, far)
        op["position_null_success30_late"] = null_at(null, far, "success30_late")
        if far is not None:
            op["eligible30_late"]["random_reference_success30"] = 1.0 - np.exp(-far * SUCCESS_WINDOW_SEC / 3600.0)
        if cfg.threshold_bootstrap > 0:
            op["threshold_aware_ci"] = threshold_aware_bootstrap(val_ee, ee, W, target, cfg.threshold_bootstrap,
                                                                 cfg.boot_seed + 7919 * seed, cluster=cfg.cluster)
        ops[key] = op
        op_out[key] = (theta_v, out)
    M["operating_points"] = ops
    M["position_null_curve"] = null

    # AMOC-style curve: one global threshold swept over pooled test scores (descriptive, not tuned).
    qs = np.unique(np.quantile(ts.score, np.linspace(0, 1, cfg.amoc_points))) if len(ts.score) else []
    curve = []
    for th in list(qs) + [np.inf]:
        o = ee.outcomes(np.full(ts.V, th))
        s = ee.summarize(o, None)
        curve.append({"theta": float(th), "fa_per_hour_clean": s["fa_per_hour_clean"]["value"],
                      "success30": s["eligible30"]["success30"]["value"],
                      "success30_late": s["eligible30_late"]["success30"]["value"],
                      "success0": s["eligible30"]["success0"]["value"],
                      "detection": s["detection"]["value"]})
    M["amoc_curve"] = curve

    if cfg.breakdowns:
        M["breakdowns"] = breakdowns(ts, ee, W, cfg, op_out.get("fa2") or next(iter(op_out.values())))
    M["_seconds"] = round(time.time() - t0, 1)
    if cfg.phase0_metrics:
        from anticipation.eval.phase0 import compute_phase0
        M["phase0"] = compute_phase0(ts, ee, val_ee, cfg, seed, M)
    M["_event_table_op2"] = event_rows_table(ts, ee, *(op_out.get("fa2") or next(iter(op_out.values()))))
    return M


def breakdowns(ts, ee, W, cfg, op):
    """Per source kind, interpersonal vs other violent classes, audio present vs absent, onset source (official /
    hivau / ubi: official onsets are the independently validated ground truth, HIVAU train onsets are not), and
    event index (first segment of the clip vs later segments, event-level only)."""
    theta_v, out = op
    res = {}
    far_pos = ~ts.masked & ts.pre & (ts.tte > cfg.far_lo) & (ts.tte <= cfg.far_hi)
    groups = {}
    for sk in SOURCE_KINDS:
        m = ts.source_v == sk
        if m.any():
            groups[f"source:{sk}"] = (m, m)
    groups["class:interpersonal"] = (ts.violent_v & ts.interp_v, ts.clean_v)
    groups["class:other_violent"] = (ts.violent_v & ~ts.interp_v, ts.clean_v)
    groups["audio:present"] = (ts.audio_v, ts.audio_v)
    groups["audio:absent"] = (~ts.audio_v, ~ts.audio_v)
    for osrc in TRUSTED_ONSETS:
        m = ts.violent_v & (ts.onset_v == osrc)
        if m.any():
            groups[f"onset:{osrc}"] = (m, ts.clean_v)
    for name, (pos_v, neg_v) in groups.items():
        pm, nm = ts.video_mask(pos_v), ts.video_mask(neg_v)
        b = {"videos_pos_side": int((pos_v & ts.violent_v).sum()), "videos_neg_side": int((neg_v & ts.clean_v).sum())}
        b["far_window_auroc_clean"] = PairAUC(ts.score, ts.vidx, far_pos & pm,
                                              ts.clean & ~ts.masked & nm & (ts.remaining >= cfg.far_hi), ts.V).stat(W)
        for H in (30,):
            pos = ~ts.masked & ts.pre & (ts.tte <= H) & pm
            neg = (~ts.masked & ts.pre & (ts.tte > cfg.neg_window) & (ts.remaining >= H)
                   & (nm | (pm & ts.video_mask(ts.violent_v))))
            sel = pos | neg
            ap, apb = average_precision(ts.score[sel], pos[sel], ts.vidx[sel], W)
            b[f"auroc{H}"] = PairAUC(ts.score, ts.vidx, pos, neg, ts.V).stat(W)
            b[f"ap{H}"] = ci(apb, ap)
            b[f"base_rate{H}"] = float(pos.sum() / max(1, sel.sum()))
        ev_mask = pos_v[ee.ev_v] if len(ee.ev_v) else np.zeros(0, bool)
        s = ee.summarize(out, W, ev_mask=ev_mask, clean_mask=neg_v)
        b["op_fa2"] = {"success30": s["eligible30"]["success30"], "success10": s["eligible30"]["success10"],
                       "success0": s["eligible30"]["success0"], "fa_per_hour_clean": s["fa_per_hour_clean"],
                       "detection": s["detection"]}
        res[name] = b
    for name, emask in (("event:first_segment", ee.ev_first), ("event:later_segment", ~ee.ev_first)):
        s = ee.summarize(out, W, ev_mask=emask)
        res[name] = {"op_fa2": {"success30": s["eligible30"]["success30"], "success10": s["eligible30"]["success10"],
                                "success0": s["eligible30"]["success0"], "fa_per_hour_clean": s["fa_per_hour_clean"],
                                "detection": s["detection"]},
                     "note": "event-level only (frame metrics use pre-violence ticks, i.e. first segments)"}
    # Per-horizon AUROC / AP per source (pooled per-horizon numbers live in M["horizon"]).
    per_src = {}
    for sk in SOURCE_KINDS:
        m = ts.source_v == sk
        if not m.any():
            continue
        vm = ts.video_mask(m)
        hz = {}
        for H in cfg.horizons:
            pos = ~ts.masked & ts.pre & (ts.tte <= H) & vm
            neg = ~ts.masked & ts.pre & (ts.tte > cfg.neg_window) & (ts.remaining >= H) & vm
            sel = pos | neg
            ap, apb = average_precision(ts.score[sel], pos[sel], ts.vidx[sel], W)
            hz[str(H)] = {"auroc": PairAUC(ts.score, ts.vidx, pos, neg, ts.V).stat(W), "ap": ci(apb, ap),
                          "base_rate": float(pos.sum() / max(1, sel.sum()))}
        per_src[sk] = hz
    res["horizon_by_source"] = per_src
    return res


def event_rows_table(ts, ee, theta_v, out):
    rows = []
    for j in range(len(ee.ev_v)):
        i = ee.ev_v[j]
        rows.append({"video_id": ts.vids[i], "fold": int(ts.fold_v[i]), "event_index": int(ee.ev_index[j]),
                     "onset_sec": round(float(ee.ev_on[j]), 3), "end_sec": round(float(ee.ev_end[j]), 3),
                     "window_start_sec": round(float(ee.ev_lo[j]), 3), "eligible30": bool(ee.ev_e30[j]),
                     "eligible60": bool(ee.ev_e60[j]), "late_window": bool(ee.ev_late[j]),
                     "theta": float(theta_v[i]),
                     "lead_sec": None if not np.isfinite(out["lead"][j]) else float(out["lead"][j]),
                     "detected": bool(out["det"][j]), "source_kind": ts.source_v[i],
                     "onset_source": ts.onset_v[i], "interpersonal": bool(ts.interp_v[i])})
    return rows


HEADLINE = [
    ("far_auroc_clean", lambda m: m["far_window_auroc"]["clean"]),
    ("far_auroc_secondary", lambda m: m["far_window_auroc"]["secondary"]),
    ("far_auroc_clean_time_matched", lambda m: m["far_window_auroc"]["clean_time_matched"]),
    ("far_auroc_clean_all_segments", lambda m: m["far_window_auroc"]["clean_all_segments"]),
    ("auroc30", lambda m: m["horizon"]["30"]["auroc"]),
    ("auroc30_time_matched", lambda m: m["horizon"]["30"]["auroc_time_matched"]),
    ("ap30", lambda m: m["horizon"]["30"]["ap"]),
    ("success30_at_fa2", lambda m: m["operating_points"]["fa2"]["eligible30"]["success30"]),
    ("success10_at_fa2", lambda m: m["operating_points"]["fa2"]["eligible30"]["success10"]),
    ("success0_at_fa2", lambda m: m["operating_points"]["fa2"]["eligible30"]["success0"]),
    ("success30_late_at_fa2", lambda m: m["operating_points"]["fa2"]["eligible30_late"]["success30"]),
    ("test_fa_per_hour_clean_at_fa2", lambda m: m["operating_points"]["fa2"]["fa_per_hour_clean"]),
    ("random_success30_at_fa2", lambda m: m["operating_points"]["fa2"].get("random_reference_success30", {}).get("nominal_W90", {})),
    ("position_null_success30_at_fa2",
     lambda m: {"value": m["operating_points"]["fa2"].get("position_null_success30")}),
    ("detection_auroc", lambda m: m["detection_auroc"]),
]


def summarize_seeds(per_seed):
    out = {}
    for name, get in HEADLINE:
        vals = []
        for m in per_seed.values():
            try:
                v = get(m).get("value")
            except (KeyError, AttributeError):
                v = None
            vals.append(np.nan if v is None else v)
        a = np.array(vals, float)
        out[name] = {"mean": float(np.nanmean(a)) if np.isfinite(a).any() else None,
                     "std": float(np.nanstd(a)) if np.isfinite(a).sum() > 1 else 0.0,
                     "per_seed": [None if not np.isfinite(x) else float(x) for x in a]}
    return out


def git_commit():
    """HEAD commit read from .git without running git (None if unavailable)."""
    try:
        gd = os.path.join(REPO_ROOT, ".git")
        head = open(os.path.join(gd, "HEAD")).read().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(None, 1)[1]
        p = os.path.join(gd, ref)
        if os.path.exists(p):
            return open(p).read().strip()
        packed = os.path.join(gd, "packed-refs")
        if os.path.exists(packed):
            for line in open(packed):
                if line.strip().endswith(ref):
                    return line.split()[0]
    except OSError:
        pass
    return None


def evaluate(scores, bench, cfg=None, name=None, extra_meta=None):
    """Full VA30 metrics dict for a score DataFrame (or ScoreTable) against a Benchmark."""
    cfg = cfg or EvalConfig()
    if cfg.horizons_csv:
        bench = bench.with_horizons(cfg.horizons_csv)
    table = scores if isinstance(scores, ScoreTable) else ScoreTable(scores, bench.video_horizons)
    eval_ids = bench.eval_ids(cfg.onset_sources)
    n_all = len(eval_ids)
    if cfg.restrict_to_scored:
        eval_ids = [v for v in eval_ids if v in table.videos]
    eval_set = set(eval_ids)
    per_seed, cov = {}, {}
    for seed in table.seeds:
        per_seed[str(seed)] = evaluate_seed(bench, table, seed, eval_set, cfg, cov)
    first = next(iter(per_seed.values()))
    if cfg.threshold_set not in ("val", "nontest"):
        raise ValueError(f"unknown threshold_set {cfg.threshold_set}")
    res = {
        "name": name,
        "benchmark": {"name": "VA30", "fold_file": bench.folds.d.get("name"), **bench.provenance},
        "run": {"created": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": git_commit(),
                "command": " ".join(sys.argv), **(extra_meta or {})},
        "config": {**{k: v for k, v in asdict(cfg).items()
                      if k != "policy" and not (k == "horizons_csv" and v is None)
                      and not (k in ("phase0_metrics", "person_counts_root") and not v)},
                   "policy": asdict(cfg.policy),
                   "threshold_rule": "per fold: lower theta from +inf over the exact unique scores of the threshold "
                                     "set until its FA/h on clean negatives first exceeds the target; keep the "
                                     "lowest theta before that (geometric top-down scan + bisection)",
                   "threshold_set_rule": ("validation slice of each fold" if cfg.threshold_set == "val" else
                                          "all non-test videos of each fold (fold-agnostic scores only)"),
                   "ci_note": "95 % percentile cluster bootstrap over test videos (or groups). Thresholds are held "
                              "fixed across resamples and the threshold-selection videos are not resampled, so the "
                              "CIs of thresholded metrics omit threshold-selection noise (see "
                              "val_clean_negative_hours_per_fold: one alarm = 1 / hours FA/h)."},
        "coverage": {"eval_videos_in_universe": n_all, "eval_videos_used": len(eval_ids),
                     "restricted_to_scored": cfg.restrict_to_scored, "score_rows": table.n_rows,
                     "seeds": table.seeds, "scores_are_probabilities": table.is_probability, **cov,
                     **({"score_rows_beyond_horizon_ignored": table.n_trimmed}
                        if bench.provenance.get("video_horizons") else {})},
        "summary_over_seeds": summarize_seeds(per_seed),
        "seeds": per_seed,
        "headline_seed": str(table.seeds[0]),
    }
    res["counts"] = first["counts"]
    return res


# ---------------------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------------------

def _fmt(x, pct=False, nd=3):
    if x is None:
        return "n/a"
    v = x.get("value") if isinstance(x, dict) else x
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    f = (lambda z: f"{100 * z:.1f} %") if pct else (lambda z: f"{z:.{nd}f}")
    s = f(v)
    if isinstance(x, dict) and x.get("ci"):
        s += f" [{f(x['ci'][0])}, {f(x['ci'][1])}]"
    return s


def write_report(res, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    m = res["seeds"][res["headline_seed"]]
    c = res["counts"]
    L = [f"# VA30 report: {res.get('name') or 'unnamed'}", "",
         f"Fold file `{res['benchmark'].get('fold_file')}` (content sha256 `{res['benchmark'].get('folds_content_sha256', '')[:12]}`), "
         f"seeds {res['coverage']['seeds']}, headline seed {res['headline_seed']}. "
         f"Test universe: {c['test_videos']} videos ({c['violent_videos']} violent, {c['clean_negative_videos']} clean "
         f"negatives = {c['clean_negative_hours']} h), {c['eligible30_events']} eligible30 events "
         f"({c['eligible60_events']} eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.",
         ""]
    if res["coverage"].get("restricted_to_scored"):
        L += [f"**Restricted to scored videos:** {res['coverage']['eval_videos_used']} of "
              f"{res['coverage']['eval_videos_in_universe']} universe videos. Not directly comparable with "
              f"full-universe results.", ""]
    vh = res["benchmark"].get("video_horizons")
    if vh:
        ch = c["video_horizons"]
        L += [f"**Capped clean negatives:** per-video horizons from `{vh.get('file')}` (sha256 "
              f"`{(vh.get('sha256') or '')[:12]}`) shorten {vh['n_shortened']} catalog clean negatives "
              f"({ch['clean_negative_videos_shortened']} in the test universe); footage after each horizon is "
              f"unobserved and excluded from every metric. Test clean-negative hours used: "
              f"{ch['clean_negative_hours_used']} of {ch['clean_negative_hours_catalog']} in the catalog.", ""]
    op = m["operating_points"]
    L += ["## Headline", "", "| Metric | Value |", "|---|---|",
          f"| Far-window AUROC (30-60 s vs clean negatives) | {_fmt(m['far_window_auroc']['clean'])} |",
          f"| Far-window AUROC (secondary: vs no onset within 120 s) | {_fmt(m['far_window_auroc']['secondary'])} |",
          f"| Far-window AUROC (clean, source x time matched negatives) | {_fmt(m['far_window_auroc']['clean_time_matched'])} |",
          f"| Far-window AUROC (clean, uncensored negatives) | {_fmt(m['far_window_auroc']['clean_uncensored'])} |",
          f"| Far-window AUROC (spec literal, all segments incl. after earlier violence) | {_fmt(m['far_window_auroc']['clean_all_segments'])} |",
          f"| AUROC / AP, H = 30 s (base rate {m['horizon']['30']['base_rate']:.3f}) | {_fmt(m['horizon']['30']['auroc'])} / {_fmt(m['horizon']['30']['ap'])} |",
          f"| Detection AUROC (in-violence vs clean negatives) | {_fmt(m['detection_auroc'])} |"]
    cal = m["calibration"]
    if "brier" in cal:
        L += [f"| Brier P(onset within 30 s) (climatology {cal['brier_climatology']:.4f}), ECE | {_fmt(cal['brier'], nd=4)}, {cal['ece']:.4f} |"]
    L += ["", "## Event-level operating points (theta chosen per fold on validation FA/h)", "",
          "Position null = best Success@30 that position-only policies (source kind x time since clip start, and "
          "their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only "
          "the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.", "",
          "| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | "
          "Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, o in op.items():
        e = o["eligible30"]
        rr = o.get("random_reference_success30", {}).get("nominal_W90")
        el = o.get("eligible30_late", {})
        tci = o.get("threshold_aware_ci", {}).get("success30_ci")
        tci = f"[{100 * tci[0]:.1f}, {100 * tci[1]:.1f}] %" if tci else "n/a"
        L.append(f"| {o['target_val_fa_per_hour']} | {_fmt(o['fa_per_hour_clean'], nd=2)} | {_fmt(o['fa_per_hour_all'], nd=2)} | "
                 f"{_fmt(e['success30'], pct=True)} | {tci} | {_fmt(rr, pct=True)} | {_fmt(o.get('position_null_success30'), pct=True)} | "
                 f"{_fmt(el.get('success30'), pct=True)} ({el.get('success30', {}).get('n', 0)}) | "
                 f"{_fmt(e['success10'], pct=True)} | {_fmt(e['success0'], pct=True)} | "
                 f"{_fmt(e['lead_sec']['median'], nd=1)} | {_fmt(o['detection'], pct=True)} |")
    L += ["", "## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)", "",
          "| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |", "|---|---|---|---|---|---|"]
    for H, h in m["horizon"].items():
        L.append(f"| {H} | {_fmt(h['auroc'])} | {_fmt(h.get('auroc_time_matched'))} | {_fmt(h.get('auroc_all_segments'))} | "
                 f"{_fmt(h['ap'])} | {h['base_rate']:.4f} |")
    if "breakdowns" in m:
        L += ["", "## Breakdowns (Success at the 2 FA/h operating point)", "",
              "| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |", "|---|---|---|---|---|---|---|"]
        for name, b in m["breakdowns"].items():
            if name == "horizon_by_source":
                continue
            if "far_window_auroc_clean" not in b:
                b = {"far_window_auroc_clean": None, "auroc30": None, **b}
            L.append(f"| {name} | {_fmt(b['far_window_auroc_clean'])} | {_fmt(b['auroc30'])} | "
                     f"{_fmt(b['op_fa2']['success30'], pct=True)} | {_fmt(b['op_fa2']['success0'], pct=True)} | "
                     f"{_fmt(b['op_fa2']['fa_per_hour_clean'], nd=2)} | {b['op_fa2']['success30'].get('n', 0)} |")
    if len(res["seeds"]) > 1:
        L += ["", "## Mean and std over seeds", "", "| Metric | mean | std |", "|---|---|---|"]
        for k, v in res["summary_over_seeds"].items():
            L.append(f"| {k} | {_fmt(v['mean'])} | {v['std']:.3f} |")
    if "phase0" in m:
        ext = m["phase0"]
        L += ["", "## Opt-in Phase 0 metrics", "", "CIs resample split_group; legacy metrics above retain their original configuration.",
              "", "| Metric | Value |", "|---|---|",
              f"| Band source x time matched AUROC (30,120] | {_fmt(ext['band_auroc'])} |",
              f"| Long band matched AUROC (60,120] | {_fmt(ext['band_long_auroc'])} |",
              f"| Event window-max matched AUROC (M8) | {_fmt(ext['window_max_auroc'])} |",
              f"| First-event window-max AUROC | {_fmt(ext['window_max_first_auroc'])} |",
              f"| Partial AMOC 0.5..2 FA/h (normalized) | {_fmt(ext['partial_amoc_0p5_2'])} |"]
        for name, stat in ext["breakdowns"].items():
            L.append(f"| Band AUROC {name} | {_fmt(stat)} |")
        people = ext["person_count"]
        if "coverage" in people:
            L += [f"| Person-stratified band AUROC (M2) | {_fmt(people['stratified_band_auroc'])} |",
                  f"| Person-stratified band AUROC, >=2 people | {_fmt(people['at_least_two_band_auroc'])} |",
                  "", f"Pose coverage: {people['coverage']}"]
        L += ["", "| Clean time target | Test warning time | Sustained@60 | Position null@60 | Sustained@90 | Position null@90 | Sustained@120 | Position null@120 |",
              "|---|---|---|---|---|---|---|---|"]
        for op in ext["sustained_warning"]["operating_points"].values():
            vals = [f"{100*op['target_clean_time_fraction']:.0f}%", _fmt(op["alarm_time_fraction_clean"], pct=True)]
            for lead in (60,90,120):
                null = op["position_null"][str(lead)]
                vals += [_fmt(op[f"success_sus{lead}"], pct=True), "n/a" if null is None else f"{100*null:.1f}%"]
            L.append("| " + " | ".join(vals) + " |")
    L += ["", f"Generated {res['run']['created']} by `anticipation/eval/va30.py` (git {str(res['run'].get('git_commit'))[:10]}).", ""]
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(L))


def write_plots(res, out_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    m = res["seeds"][res["headline_seed"]]
    cur = [p for p in m["amoc_curve"] if p["fa_per_hour_clean"] is not None and p["success30"] is not None]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    if cur:
        x = np.array([p["fa_per_hour_clean"] for p in cur])
        y = np.array([p["success30"] for p in cur])
        o = np.argsort(x)
        ax.plot(np.maximum(x[o], 1e-2), y[o], marker=".", label=res.get("name") or "model")
        xr = np.logspace(-2, 3, 100)
        ax.plot(xr, 1 - np.exp(-xr * SUCCESS_WINDOW_SEC / 3600), "--", color="gray", label="random (Poisson)")
        nul = (m.get("position_null_curve") or {}).get("success30") or []
        if len(nul) > 1:
            xs = np.r_[[max(x, 1e-2) for x, _, _ in nul[1:]], 1e3]
            ys = np.r_[[y for _, y, _ in nul[1:]], nul[-1][1]]
            xg = np.logspace(-2, 3, 200)
            ax.plot(xg, np.interp(xg, np.r_[0.0, xs], np.r_[0.0, ys]), color="tab:red", lw=1,
                    label="position null (oracle hull)")
        for k, op in m["operating_points"].items():
            fx = op["fa_per_hour_clean"]["value"]
            sy = op["eligible30"]["success30"]["value"]
            if fx is not None and sy is not None:
                ax.plot(max(fx, 1e-2), sy, "o", color="black")
                ax.annotate(k, (max(fx, 1e-2), sy), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("False alarms per hour (clean negatives, test)")
    ax.set_ylabel("Success@30 (eligible30 events)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("AMOC-style curve (global threshold sweep on test)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "amoc.png"), dpi=120)
    plt.close(fig)
    rel = m["calibration"].get("reliability")
    if rel:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.plot([r["mean_pred"] for r in rel], [r["frac_pos"] for r in rel], "o-")
        ax.set_xlabel("mean predicted P(onset within 30 s)")
        ax.set_ylabel("observed frequency")
        ax.set_title("Reliability", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "reliability.png"), dpi=120)
        plt.close(fig)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def _clean_nans(o):
    if isinstance(o, dict):
        return {k: _clean_nans(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_nans(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None if np.isnan(o) else ("inf" if o > 0 else "-inf")
    return o


def save_results(res, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for seed, m in res["seeds"].items():
        rows = m.pop("_event_table_op2", None)
        if rows is not None and seed == res["headline_seed"]:
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, "events_op2.csv"), index=False)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(_clean_nans(res), f, indent=1, default=_json_default)
    write_report(res, out_dir)
    write_plots(res, out_dir)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True, help="parquet or csv score table")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--folds", default=None, help="fold file (default benchmark/folds_v2.json)")
    ap.add_argument("--videos", default=VIDEOS_CSV)
    ap.add_argument("--events", default=EVENTS_JSON)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--boot_seed", type=int, default=0)
    ap.add_argument("--cluster", choices=["video", "group", "split_group"], default="video")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--fa_targets", default="1,2,5")
    ap.add_argument("--onset_sources", default=",".join(TRUSTED_ONSETS))
    ap.add_argument("--restrict_to_scored", action="store_true",
                    help="evaluate only universe videos that appear in the score table (reported as such)")
    ap.add_argument("--allow_missing", action="store_true")
    ap.add_argument("--no_breakdowns", action="store_true")
    ap.add_argument("--no_position_null", action="store_true")
    ap.add_argument("--threshold_set", default="val", choices=["val", "nontest"],
                    help="nontest: choose thresholds on all non-test videos of each fold (fold-agnostic scores only)")
    ap.add_argument("--threshold_bootstrap", type=int, default=0,
                    help="replicates for threshold-aware CIs of Success@30 and FA/h (validation resampled, theta "
                         "re-chosen per replicate); 0 = off")
    ap.add_argument("--horizons", default=None,
                    help="csv with video_id,duration_sec (e.g. benchmark/extract_manifest_v1.csv): clean negatives "
                         "listed with a shorter duration than the catalog are evaluated only up to it")
    ap.add_argument("--phase0_metrics", action="store_true", help="add band, M2, M8 and sustained-warning metrics")
    ap.add_argument("--person_counts_root", default=None, help="pose_interact root for opt-in M2")
    ap.add_argument("--meta", default=None,
                    help="JSON file merged into metrics.json 'run' (feature set + extraction config, model config, "
                         "training command, seeds): the BASELINE 6.6 reporting template")
    args = ap.parse_args(argv)

    t0 = time.time()
    bench = Benchmark.load(args.folds, args.videos, args.events, horizons=args.horizons)
    cfg = EvalConfig(policy=AlarmPolicy(k=args.k, n=args.n), n_boot=args.n_boot, boot_seed=args.boot_seed,
                     cluster=args.cluster, fa_targets=tuple(float(x) for x in args.fa_targets.split(",")),
                     onset_sources=tuple(args.onset_sources.split(",")), restrict_to_scored=args.restrict_to_scored,
                     allow_missing=args.allow_missing or args.restrict_to_scored, breakdowns=not args.no_breakdowns,
                     position_null=not args.no_position_null, threshold_set=args.threshold_set,
                     threshold_bootstrap=args.threshold_bootstrap, phase0_metrics=args.phase0_metrics,
                     person_counts_root=args.person_counts_root)
    df = load_scores(args.scores)
    meta = {"scores_file": args.scores, "scores_sha256": file_sha256(args.scores)}
    if args.meta:
        with open(args.meta) as f:
            meta["model_meta"] = json.load(f)
    res = evaluate(df, bench, cfg, name=args.name or os.path.basename(os.path.normpath(args.out_dir)), extra_meta=meta)
    save_results(res, args.out_dir)
    s = res["summary_over_seeds"]
    print(f"[va30] {res['name']}: far AUROC clean {s['far_auroc_clean']['mean']}, Success@30@2FA/h "
          f"{s['success30_at_fa2']['mean']} (random {s['random_success30_at_fa2']['mean']}, position null "
          f"{s['position_null_success30_at_fa2']['mean']}), "
          f"wrote {args.out_dir} in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
