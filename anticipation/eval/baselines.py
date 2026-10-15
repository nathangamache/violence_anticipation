"""VA30 control baselines (BASELINE.md section 6.5) scored with the VA30 evaluator.

Usage (from the repo root):

    .venv/bin/python -m anticipation.eval.baselines      # prior, time, time_fit, time_source_fit, memory, teacher
    .venv/bin/python -m anticipation.eval.baselines --which teacher --teacher_root \
        data/processed/vlm_labels_old --name b_teacher_old
    .venv/bin/python -m anticipation.eval.baselines --n_boot 200         # quicker CIs
    .venv/bin/python -m anticipation.eval.baselines --which teacher --threshold_set nontest --name b_teacher_nontest
        # zero-shot teacher with thresholds chosen on all non-test videos of each fold (more FA/h evidence)

Baselines (b_time_fit and b_time_source_fit are fitted per fold on training videos; the others are fold-agnostic,
so the same score is used for every fold's val and test):

    b_prior    constant 0.5 on every tick. Sanity check: AUROC 0.5, never alarms at 1/2/5 FA/h.
    b_time     score = seconds since video start (t_sec), exactly as specified. Its AUROC is far BELOW 0.5 on
               VA30 because onsets are early in their clips while clean-negative CCTV videos are long, so read it
               as |AUROC - 0.5| or look at b_time_fit.
    b_time_fit position-only model trained per fold: on the fold's TRAINING videos, P(onset within 60 s | bin of
               seconds since start) with log-spaced bins (va30.TIME_EDGES) and add-one-half smoothing; scored on
               the fold's validation and test videos. This is the real size of the position shortcut and the
               level a model must clearly beat. Its source-blind time-matched far-window AUROC is 0.5 by construction.
    b_time_source_fit  the same per (source kind x 5 s time bin) (va30.TIME_MATCH_EDGES): position plus provenance,
               which any visual model can read off the frames (CCTV vs movie vs YouTube). The strongest label-free
               shortcut control; the (source x time) matched far-window AUROC neutralizes it (0.5 by construction).
    b_memory   detection-memory oracle: 1 once any annotated segment (violent or other) has started at or before
               t, else 0. Uses only the past, has zero information about future onsets, and is exactly what a
               perfect detector with max-pool memory outputs. Controls the "later segments follow earlier
               violence" shortcut: 0.5 on the pre-violence headline frame metrics, about 0.7 on the spec-literal
               all-segments far-window AUROC.
    b_teacher  zero-shot Qwen2.5-VL-7B ``violent.score`` from processed/vlm_labels*/<source>/<vid>.jsonl,
               sample-and-hold onto the 1 Hz ticks. A record's time is raw_frame_idx / fps (fps from
               data/metadata/<source>.json, falling back to the catalog fps); the tick t score is the latest record
               with time <= t (causal); ticks before the first record get 0.0 (the teacher's "not violent").
               String verdicts map yes -> 1, no -> 0, other -> 0.5. Duplicate (vid, t) records (from resumed
               runs) keep the first. The teacher covers only the 1721 videos of the original pipeline, so it is
               evaluated on that subset (``--restrict_to_scored``); b_time is re-scored on the same subset as
               ``b_time__teacher_subset`` for an apples-to-apples comparison.

Each baseline writes benchmark/results/<name>/{metrics.json, report.md, amoc.png, reliability.png,
events_op2.csv} and, when small, scores.parquet (the per-tick score table).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from anticipation.data.annotations import DATA_ROOT
from anticipation.data.catalog import BENCHMARK_DIR, EVENTS_JSON, VIDEOS_CSV
from anticipation.eval.va30 import TIME_EDGES, TIME_MATCH_EDGES, Benchmark, EvalConfig, evaluate, save_results

RESULTS_DIR = os.path.join(BENCHMARK_DIR, "results")
TEACHER_ROOT = os.path.join(DATA_ROOT, "processed", "vlm_labels")
META_DIR = os.path.join(DATA_ROOT, "metadata")
TEACHER_SOURCES = {"xd_violence": "xd", "ucf_crime": "ucf"}
MAX_SAVED_SCORES_MB = 5.0


def _ticks(bench, vid):
    return bench.labels(vid).t_sec


def scores_prior(bench, vids):
    return pd.concat([pd.DataFrame({"video_id": v, "t_sec": _ticks(bench, v), "score": 0.5}) for v in vids],
                     ignore_index=True)


def scores_time(bench, vids):
    return pd.concat([pd.DataFrame({"video_id": v, "t_sec": _ticks(bench, v), "score": _ticks(bench, v)})
                      for v in vids], ignore_index=True)


def _time_bin(t, edges=TIME_EDGES):
    return np.clip(np.searchsorted(np.asarray(edges[1:-1]), t, side="right"), 0, len(edges) - 2)


def scores_time_fit(bench, vids, horizon=60.0, edges=TIME_EDGES, by_source=False):
    """Per-fold position-only model (see module docstring): P(onset within ``horizon`` | time bin), or
    | (source kind, time bin) with ``by_source``, on the fold's training videos. Returns the table and the rates."""
    eval_set = set(vids)
    folds = bench.folds
    nb = len(edges) - 1
    parts, rates = [], {}

    def key(v):
        return str(bench.videos.loc[v, "source_kind"]) if by_source else "all"

    for f in range(folds.n_folds):
        pos, tot = {}, {}
        for v in folds.train_ids(f):
            if v not in eval_set:
                continue
            lab = bench.labels(v)
            valid = ~lab.masked & ((lab.tte <= horizon) | (lab.remaining >= horizon))
            b = _time_bin(lab.t_sec[valid], edges)
            k = key(v)
            tot[k] = tot.get(k, np.zeros(nb)) + np.bincount(b, minlength=nb)
            pos[k] = pos.get(k, np.zeros(nb)) + np.bincount(b, weights=(lab.tte[valid] <= horizon).astype(float),
                                                            minlength=nb)
        rate = {k: (pos[k] + 0.5) / (tot[k] + 1.0) for k in tot}
        prior = (sum(pos.values()) + 0.5) / (sum(tot.values()) + 1.0)
        rates[str(f)] = {k: [round(float(x), 6) for x in r] for k, r in rate.items()}
        for v in sorted(set(folds.test_ids(f)) | set(folds.val_ids(f))):
            if v not in eval_set:
                continue
            ticks = _ticks(bench, v)
            r = rate.get(key(v), prior)
            parts.append(pd.DataFrame({"video_id": v, "t_sec": ticks, "score": r[_time_bin(ticks, edges)], "fold": f}))
    return pd.concat(parts, ignore_index=True), rates


def scores_memory(bench, vids):
    """Detection-memory oracle: 1 once any annotated segment (violent or other-anomaly) has started, else 0."""
    parts = []
    for v in vids:
        lab = bench.labels(v)
        parts.append(pd.DataFrame({"video_id": v, "t_sec": lab.t_sec, "score": (~lab.pre_violence).astype(float)}))
    return pd.concat(parts, ignore_index=True)


def teacher_score(rec):
    """Numeric violent score of one teacher JSONL record, or None if the record carries no verdict."""
    if "error" in rec:
        return None
    v = rec.get("violent")
    if isinstance(v, dict):
        s = v.get("score")
        if isinstance(s, (int, float)) and not isinstance(s, bool):
            return float(s)
        v = v.get("label")
    if isinstance(v, str):
        v = v.strip().lower()
        return 1.0 if v == "yes" else (0.0 if v == "no" else 0.5)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return None


def read_teacher(path, fps):
    """``(times_sec, scores)`` sorted by time from one teacher JSONL."""
    seen, ts, ss = set(), [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = rec.get("t")
            if t in seen or rec.get("raw_frame_idx") is None:
                continue
            s = teacher_score(rec)
            if s is None:
                continue
            seen.add(t)
            ts.append(float(rec["raw_frame_idx"]) / fps)
            ss.append(s)
    o = np.argsort(ts, kind="stable")
    return np.asarray(ts)[o], np.asarray(ss)[o]


def sample_and_hold(times, values, ticks, fill=0.0):
    idx = np.searchsorted(times, ticks + 1e-9, side="right") - 1
    return np.where(idx >= 0, values[np.clip(idx, 0, None)], fill)


def scores_teacher(bench, vids, root=TEACHER_ROOT):
    """Teacher score table for the videos in ``vids`` that have a teacher JSONL; plus a coverage report."""
    meta = {}
    for src in TEACHER_SOURCES:
        p = os.path.join(META_DIR, f"{src}.json")
        if os.path.exists(p):
            with open(p) as f:
                meta[src] = json.load(f)
    parts, cov = [], {"videos_requested": len(vids), "videos_with_teacher": 0, "fps_mismatch": 0,
                      "records": 0, "ticks_before_first_record": 0}
    inv = {v: k for k, v in TEACHER_SOURCES.items()}
    for vid in vids:
        src, stem = vid.split("/", 1)
        tsrc = inv.get(src)
        if tsrc is None:
            continue
        path = os.path.join(root, tsrc, stem + ".jsonl")
        if not os.path.exists(path):
            continue
        cat_fps = float(bench.videos.loc[vid, "fps"]) if "fps" in bench.videos else 0.0
        fps = float(meta.get(tsrc, {}).get(stem, {}).get("fps") or cat_fps or 25.0)
        if cat_fps and abs(fps - cat_fps) > 0.01:
            cov["fps_mismatch"] += 1
        times, vals = read_teacher(path, fps)
        if len(times) == 0:
            continue
        ticks = _ticks(bench, vid)
        cov["videos_with_teacher"] += 1
        cov["records"] += int(len(times))
        cov["ticks_before_first_record"] += int((ticks < times[0] - 1e-9).sum())
        parts.append(pd.DataFrame({"video_id": vid, "t_sec": ticks, "score": sample_and_hold(times, vals, ticks)}))
    return pd.concat(parts, ignore_index=True), cov


def run_one(name, df, bench, cfg, out_root, extra=None, save_scores=True):
    t0 = time.time()
    if "fold" not in df:
        df = df.assign(fold=-1)
    df = df.assign(seed=0)
    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)
    meta = {"baseline": name, **(extra or {})}
    if save_scores:
        p = os.path.join(out, "scores.parquet")
        df.to_parquet(p, index=False, compression="zstd")
        if os.path.getsize(p) > MAX_SAVED_SCORES_MB * 1e6:
            os.remove(p)
            meta["scores_file"] = "not saved (too large); regenerate with anticipation.eval.baselines"
        else:
            meta["scores_file"] = os.path.relpath(p, os.path.dirname(BENCHMARK_DIR))
    res = evaluate(df, bench, cfg, name=name, extra_meta=meta)
    save_results(res, out)
    s = res["summary_over_seeds"]
    print(f"[baselines] {name}: far AUROC clean {s['far_auroc_clean']['mean']:.3f}, AUROC30 {s['auroc30']['mean']:.3f}, "
          f"Success@30@2FA/h {s['success30_at_fa2']['mean']}, ({time.time() - t0:.0f} s) -> {out}", flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", default="prior,time,time_fit,time_source_fit,memory,teacher")
    ap.add_argument("--folds", default=None, help="fold file (default benchmark/folds_v2.json)")
    ap.add_argument("--videos", default=VIDEOS_CSV)
    ap.add_argument("--events", default=EVENTS_JSON)
    ap.add_argument("--out_root", default=RESULTS_DIR)
    ap.add_argument("--teacher_root", default=TEACHER_ROOT)
    ap.add_argument("--name", default=None, help="override the result folder name (single baseline only)")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--no_save_scores", action="store_true")
    ap.add_argument("--no_subset_controls", action="store_true",
                    help="teacher only: skip re-scoring b_time / b_time_source_fit on the teacher-covered subset")
    ap.add_argument("--threshold_set", default="val", choices=["val", "nontest"],
                    help="threshold-selection videos for the fold-agnostic baselines (see va30 --threshold_set)")
    ap.add_argument("--threshold_bootstrap", type=int, default=200,
                    help="replicates for the threshold-aware CIs (va30 --threshold_bootstrap); 0 = off")
    args = ap.parse_args(argv)

    bench = Benchmark.load(args.folds, args.videos, args.events)
    vids = bench.eval_ids()
    which = [w.strip() for w in args.which.split(",") if w.strip()]
    tb = args.threshold_bootstrap
    cfg = EvalConfig(n_boot=args.n_boot, threshold_bootstrap=tb)
    cfg_agn = EvalConfig(n_boot=args.n_boot, threshold_set=args.threshold_set, threshold_bootstrap=tb)
    save = not args.no_save_scores
    for w in which:
        name = args.name or f"b_{w}"
        if w == "prior":
            run_one(name, scores_prior(bench, vids), bench, cfg_agn, args.out_root, save_scores=save)
        elif w == "time":
            run_one(name, scores_time(bench, vids), bench, cfg_agn, args.out_root, save_scores=save)
        elif w == "time_fit":
            df, rates = scores_time_fit(bench, vids)
            run_one(name, df, bench, cfg, args.out_root, save_scores=save,
                    extra={"time_edges_sec": [str(x) for x in TIME_EDGES], "fitted_rates_per_fold": rates,
                           "target": "P(onset within 60 s | time bin) on unmasked, uncensored training ticks"})
        elif w == "time_source_fit":
            df, rates = scores_time_fit(bench, vids, edges=TIME_MATCH_EDGES, by_source=True)
            run_one(name, df, bench, cfg, args.out_root, save_scores=save,
                    extra={"time_edges_sec": [str(x) for x in TIME_MATCH_EDGES], "fitted_rates_per_fold": rates,
                           "target": "P(onset within 60 s | source kind, time bin) on unmasked, uncensored "
                                     "training ticks"})
        elif w == "memory":
            run_one(name, scores_memory(bench, vids), bench, cfg_agn, args.out_root, save_scores=save,
                    extra={"definition": "1 once any annotated segment has started at or before t (ground-truth "
                                         "detection memory; no information about future onsets)"})
        elif w == "teacher":
            df, cov = scores_teacher(bench, vids, args.teacher_root)
            sub_cfg = EvalConfig(n_boot=args.n_boot, restrict_to_scored=True, allow_missing=True,
                                 threshold_set=args.threshold_set, threshold_bootstrap=tb)
            extra = {"teacher_root": args.teacher_root, "teacher_coverage": cov,
                     "fill_before_first_record": 0.0}
            run_one(name, df, bench, sub_cfg, args.out_root, extra=extra, save_scores=save)
            if args.no_subset_controls:
                continue
            covered = sorted(df.video_id.unique())
            tag = name.removeprefix("b_")
            run_one(f"b_time__{tag}_subset", scores_time(bench, covered), bench, sub_cfg,
                    args.out_root, extra={"subset": f"videos covered by {args.teacher_root}"}, save_scores=False)
            df_ts, _ = scores_time_fit(bench, vids, edges=TIME_MATCH_EDGES, by_source=True)
            df_ts = df_ts[df_ts.video_id.isin(set(covered))]
            sub_cfg_fit = EvalConfig(n_boot=args.n_boot, restrict_to_scored=True, allow_missing=True,
                                     threshold_bootstrap=tb)
            run_one(f"b_time_source_fit__{tag}_subset", df_ts, bench, sub_cfg_fit, args.out_root,
                    extra={"subset": f"videos covered by {args.teacher_root}",
                           "note": "fitted on the full-universe training videos of each fold, scored on the subset"},
                    save_scores=False)
        else:
            raise SystemExit(f"unknown baseline {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
