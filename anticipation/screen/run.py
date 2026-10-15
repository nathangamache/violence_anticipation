"""Run the pre-registered modality screen on cached feature streams (docs/SCREEN.md).

    nice -n 10 .venv/bin/python -m anticipation.screen.run --dry_run
    nice -n 10 .venv/bin/python -m anticipation.screen.run --phase A --out benchmark/results/cue_screen
    .venv/bin/python -m anticipation.screen.run --decide A --out benchmark/results/cue_screen
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from anticipation.data.catalog import REPO_ROOT, file_sha256
from anticipation.eval import va30
from anticipation.screen import decision as D
from anticipation.screen import features as F
from anticipation.screen import metrics as M
from anticipation.screen import prereg
from anticipation.screen import probes as P
from anticipation.screen import sets as S

KINDS = ("violent", "clean", "h1")
AUC_NULL_KEYS_ZERO = ("m4n", "gain", "tail")


def log(*a):
    print("[screen]", *a, flush=True)


def jsonable(x, key=""):
    if isinstance(x, M.Stat):
        null = 0.0 if any(k in key for k in AUC_NULL_KEYS_ZERO) else 0.5
        return x.to_dict(null)
    if isinstance(x, dict):
        return {str(k): jsonable(v, str(k)) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v, key) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(jsonable(data), f, indent=1, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def cohort(ss, runner):
    out = {}
    for f in ss.screen_folds:
        out[f] = {"train": [v for v in ss.train_ids(f) if runner.covered(v)],
                  "val": [v for v in ss.val_ids(f) if runner.covered(v)],
                  "test": [v for v in ss.test_ids(f, kinds=KINDS) if runner.covered(v)]}
    return out


def clean_val_vector(runner, scores, vids):
    xs = []
    for v in vids:
        vt = runner.vt(v)
        if vt.kind != "clean" or v not in scores:
            continue
        keep = ~vt.masked & (vt.remaining >= 60)
        xs.append(scores[v][:vt.n][keep])
    return np.concatenate(xs) if xs else np.zeros(0)


def p0_columns(pool, runner, block, scores_unused=None):
    """P0: per-column time-matched band AUROC with the pre-declared direction (no fitting)."""
    if runner.width is None or runner.width > F.WIDE_COLUMNS:
        return {"skipped": "wide block" if runner.width else "no data"}
    d = pool.pair_def(("band", False), pool.band_pos(30, 120), pool.clean_neg())
    pos2, neg, w = d
    m = pos2 | (neg & (w > 0))
    X = np.concatenate([runner.raw(v)[:runner.vt(v).n] for v in pool.vids])
    y = pos2[m]
    ww = np.where(pos2, 1.0, w)[m]
    low = set(block.lower_is_risk)
    rows = []
    for j, name in enumerate(runner.columns):
        sign = -1.0 if name.split(":", 1)[-1] in low else 1.0
        a = P.weighted_auc(sign * X[m, j], y, ww)
        rows.append({"column": name, "direction": int(sign), "auc": None if not np.isfinite(a) else round(a, 4)})
    rows.sort(key=lambda r: -(r["auc"] or 0))
    return {"columns": rows, "n_above_0.58": sum(1 for r in rows if (r["auc"] or 0) >= 0.58)}


def persons_for_pool(source, pool):
    out = []
    for v in pool.vids:
        try:
            t, x, valid, meta = source.load("pose_interact", v)
        except (FileNotFoundError, OSError):
            return None
        cols = list(meta.get("columns") or [])
        if "n_persons_last" not in cols:
            return None
        n = pool.offsets[pool.index[v] + 1] - pool.offsets[pool.index[v]]
        a = np.zeros(n)
        k = min(n, len(x))
        a[:k] = x[:k, cols.index("n_persons_last")]
        out.append(a)
    return np.concatenate(out)


def write_scores(out_dir, ss, runner, per_fold, models):
    rows = []
    for f, sc in per_fold.items():
        for split in ("test", "val"):
            vids = sorted(set().union(*[set(sc[m][split]) for m in models if m in sc]))
            for v in vids:
                n = runner.vt(v).n
                d = {"video_id": v, "t_sec": runner.vt(v).t[:n], "fold": f, "split": split, "seed": 0}
                for m in models:
                    if m in sc and v in sc[m][split]:
                        d[m] = sc[m][split][v][:n]
                rows.append(pd.DataFrame(d))
    df = pd.concat(rows, ignore_index=True)
    ss.assert_no_frozen(df)
    df.rename(columns={"p1": "score"}).to_parquet(os.path.join(out_dir, "scores.parquet"), index=False)


def run_m6(ss, runner, per_fold, out_dir, n_threshold_boot=200, name="va30"):
    rows = []
    for f, sc in per_fold.items():
        for split in ("test", "val"):
            for v, s in sc["p1"][split].items():
                if ss.kind(v) == "h1":
                    continue
                vt = runner.vt(v)
                rows.append(pd.DataFrame({"video_id": v, "t_sec": vt.t, "score": s[:vt.n], "fold": f, "seed": 0}))
    df = pd.concat(rows, ignore_index=True)
    ss.assert_no_frozen(df)
    path = os.path.join(out_dir, "va30_scores.parquet")
    df.to_parquet(path, index=False)
    va30.main(["--scores", path, "--out_dir", os.path.join(out_dir, name), "--restrict_to_scored",
               "--horizons", S.MANIFEST_V1, "--threshold_bootstrap", str(n_threshold_boot), "--n_boot", "1000"])
    return path


def run_block(ss, source, block, out_root, args, prereg_sha=None):
    t0 = time.time()
    runner = P.BlockRun(ss, source, block, seed=0)
    cohorts = cohort(ss, runner)
    out_dir = os.path.join(out_root, block.name)
    os.makedirs(out_dir, exist_ok=True)
    per_fold, probe_info, nuis_info = {}, {}, {}
    test_vts = []
    for f, c in cohorts.items():
        if not c["train"] or not c["val"] or not c["test"]:
            raise ValueError(f"{block.name} fold {f}: missing train/val/test coverage")
        log(f"{block.name} fold {f}: train {len(c['train'])} val {len(c['val'])} test {len(c['test'])}")
        n_scores, n_info, nuis = P.run_nuisance(ss, source, c["train"], c["val"], c["test"])
        nuis_info[f] = n_info
        scores, info = runner.run_fold(f, c["train"], c["val"], c["test"], nuis=nuis)
        if n_scores is not None:
            scores["n"] = n_scores
        per_fold[f] = scores
        probe_info[f] = info
        test_vts += [runner.vt(v) for v in c["test"]]
    pool = M.Pool(ss, test_vts)
    models = ["p1", "p1s", "p1b", "p1_near", "p1_later", "b_detect", "n_plus_p1", "n", "shuffled", "time_source"]
    pooled, val_by_model = {}, {}
    for m in models:
        d = {}
        for f in cohorts:
            d.update(per_fold[f].get(m, {}).get("test", {}))
        if set(pool.vids) <= set(d):
            pooled[m] = pool.assemble(d)
            val_by_model[m] = {f: clean_val_vector(runner, per_fold[f][m]["val"], cohorts[f]["val"])
                               for f in cohorts if m in per_fold[f]}
    W = pool.boot_weights(args.n_boot, seed=0)
    persons = persons_for_pool(source, pool) if args.m2 else None
    res = M.evaluate_pool(pool, pooled, val_by_model, W, persons=persons, primary_audio=block.audio)
    res["p0"] = p0_columns(pool, runner, block)
    n_pos_events = len(pool.events)
    res["coverage"] = {
        "events_in_pool": n_pos_events, "pool_videos": pool.V,
        "pool_clean_videos": int((pool.kind_v == 0).sum()), "pool_h1_videos": int((pool.kind_v == 2).sum()),
        "valid_fraction_mean": float(np.mean(list(runner._vf.values()))) if runner._vf else None,
        "scene_missing": {str(f): i.get("scene_missing") for f, i in nuis_info.items()}}
    res["probes"] = {str(f): i for f, i in probe_info.items()}
    res["nuisance"] = {str(f): i for f, i in nuis_info.items()}
    if block.audio:
        sb = res["breakdowns"]
        res["soundtrack_flag"] = D.soundtrack_flag(sb.get("source_xd_movie").value if sb.get("source_xd_movie") else None,
                                                   sb.get("s_audio").value if sb.get("s_audio") else None)
    res["config"] = {"block": block.name, "phase": block.phase, "streams": list(block.streams), "columns": runner.columns,
                     "wide": bool(runner.wide), "n_boot": args.n_boot, "prereg_sha256": prereg_sha,
                     "soft_masks": ss.soft_mask_info, "seconds": round(time.time() - t0, 1),
                     "diegetic_filter_applied": False if block.audio else None}
    write_json(os.path.join(out_dir, "metrics.json"), res)
    write_scores(out_dir, ss, runner, per_fold, models)
    if args.m6:
        run_m6(ss, runner, per_fold, out_dir)
    log(f"{block.name}: M1 {res['m1'].value} done in {time.time() - t0:.0f} s")
    return res


def decide(out_root, phase):
    blocks, exempt = {}, []
    for b in F.BLOCKS.values():
        p = os.path.join(out_root, b.name, "metrics.json")
        if b.phase == phase and os.path.isfile(p):
            m = json.load(open(p))
            m["tail"] = (m.get("events") or {}).get("tail")
            blocks[b.name] = m
            if b.g2_exempt:
                exempt.append(b.name)
    result = D.decide_phase(blocks, exempt)
    write_json(os.path.join(out_root, f"decisions_phase_{phase}.json"), result)
    lines = ["| block | M1 | G1 | G2 | G3 | class | tags |", "|---|---|---|---|---|---|---|"]
    for k, r in result.items():
        v = (blocks[k].get("m1") or {}).get("value")
        lines.append(f"| {k} | {'' if v is None else f'{v:.3f}'} | {r['G1']} | {r['G2']} | {r['G3']} | {r['class']} | "
                     f"{', '.join(r['tags'])} |")
    with open(os.path.join(out_root, f"results_phase_{phase}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return result


def dry_run(ss, source, blocks, out_root):
    """Tick counts of set S and feature coverage per block (no metric is computed)."""
    info = {"counts": ss.counts(), "soft_masks": ss.soft_mask_info, "blocks": {}}
    groups = {"positive": ss.video_lists()["positive_videos"], "clean": ss.video_lists()["clean_negative_videos"],
              "h1": ss.h1_ids(), "h1p": ss.h1p_ids()}
    train_pool = sorted({v for f in ss.screen_folds for v in ss.train_ids(f)})
    groups["train_or_val_all"] = train_pool
    for b in blocks:
        row = {}
        for g, vids in groups.items():
            need = have = vids_with = 0
            for v in vids:
                if b.sources and ss.source_kind(v) not in b.sources:
                    continue
                n = ss.ticks(v).n if (g != "train_or_val_all" or ss.kind(v) != "other") else 0
                if not n:
                    continue
                need += n
                if all(source.has(s, v) for s in b.streams):
                    vids_with += 1
                    k = n
                    try:
                        k = min(n, min(source.n_ticks(s, v) for s in b.streams))
                    except Exception:
                        k = 0
                    have += k
            nv = len([v for v in vids if not (b.sources and ss.source_kind(v) not in b.sources)])
            row[g] = {"videos": nv, "with_features": vids_with, "ticks": int(need), "ticks_covered": int(have),
                      "tick_coverage": round(have / need, 4) if need else None}
        ev = ss.events()
        ok = set(v for v in groups["positive"] if all(source.has(s, v) for s in b.streams))
        row["events_covered"] = int(ev["video_id"].isin(ok).sum())
        row["first_segment_events_covered"] = int((ev["video_id"].isin(ok) & ev["first"]).sum())
        info["blocks"][b.name] = row
        log(f"{b.name}: " + ", ".join(f"{g} {r['with_features']}/{r['videos']} ({r['tick_coverage']})"
                                      for g, r in row.items() if isinstance(r, dict)))
    scene = sum(1 for v in groups["positive"] + groups["clean"] + groups["h1"] if source.scene(v) is not None)
    info["scene0p_videos_with_embedding"] = scene
    info["scene0p_videos_needed"] = sum(len(groups[g]) for g in ("positive", "clean", "h1"))
    write_json(os.path.join(out_root, "dry_run_coverage.json"), info)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["A", "B"])
    ap.add_argument("--blocks", default="all")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "benchmark", "results", "cue_screen"))
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--decide", choices=["A", "B"])
    ap.add_argument("--write_prereg", action="store_true")
    ap.add_argument("--m6", action="store_true", help="also score the VA30 event metrics (slow)")
    ap.add_argument("--m2", action="store_true", help="person-count stratified AUROC (needs pose_interact)")
    ap.add_argument("--folds", default=None)
    ap.add_argument("--feature_root", default=None)
    ap.add_argument("--prereg", default=prereg.PREREG_PATH)
    ap.add_argument("--allow_unregistered", action="store_true")
    args = ap.parse_args(argv)
    va30.N_THREADS = 4
    if args.decide:
        decide(args.out, args.decide)
        return 0
    ss = S.load_screen_set(args.folds)
    source = F.FeatureSource(ss, root=args.feature_root)
    if args.write_prereg:
        sha = prereg.write(ss, args.prereg, git_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip())
        log("wrote", args.prereg, sha)
        return 0
    names = [b.name for b in F.BLOCKS.values() if (args.phase is None or b.phase == args.phase)] if args.blocks == "all" \
        else args.blocks.split(",")
    blocks = [F.BLOCKS[n] for n in names]
    if args.dry_run:
        with threadpool_limits(limits=4):
            dry_run(ss, source, blocks, args.out)
        return 0
    if not args.allow_unregistered:
        if not os.path.isfile(args.prereg):
            raise SystemExit(f"no pre-registration at {args.prereg}; write and commit it first")
        bad = prereg.verify(ss, args.prereg)
        if bad:
            raise SystemExit(f"set S differs from the pre-registration in: {bad}")
    sha = file_sha256(args.prereg) if os.path.isfile(args.prereg) else None
    with threadpool_limits(limits=4):
        for b in blocks:
            run_block(ss, source, b, args.out, args, sha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
