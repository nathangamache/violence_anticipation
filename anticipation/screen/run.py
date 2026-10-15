"""Run the frozen modality screen (benchmark/screens/screen_set_v1.json) on cached feature streams (docs/SCREEN.md).

    nice -n 10 .venv/bin/python -m anticipation.screen.run --dry_run
    nice -n 10 .venv/bin/python -m anticipation.screen.run --phase A --out benchmark/results/cue_screen
    .venv/bin/python -m anticipation.screen.run --decide A --out benchmark/results/cue_screen
    nice -n 10 .venv/bin/python -m anticipation.screen.run --phase A --sensitivity S1 --out benchmark/results/cue_screen
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse
import json
import re
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from anticipation.data.catalog import REPO_ROOT
from anticipation.data.folds import Folds, content_hash
from anticipation.eval import va30
from anticipation.screen import decision as D
from anticipation.screen import features as F
from anticipation.screen import metrics as M
from anticipation.screen import prereg
from anticipation.screen import probes as P
from anticipation.screen import sets as S

KINDS = ("violent", "clean", "h1")
AVAILABLE_STATUS = F.AVAILABLE
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
    rows = []
    for j, name in enumerate(runner.columns):
        sign = F.p0_sign(name, block)
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


def folds_without_confirmation(out_dir, confirmation=1):
    """Write a copy of folds_v2 with the confirmation fold removed and later folds renumbered; returns (path, map)."""
    src = Folds.load()
    fmap = {f: i for i, f in enumerate(f for f in range(src.n_folds) if f != confirmation)}
    assignment = {v: fmap[f] for v, f in src.assignment.items() if f in fmap}
    val = {str(fmap[f]): [v for v in vs if v in assignment] for f, vs in src.val.items() if f in fmap}
    d = {"name": "folds_v2_without_confirmation", "version": src.d["version"], "n_folds": len(fmap),
         "assignment": assignment, "val": val, "content_sha256": content_hash(assignment, val)}
    path = os.path.join(out_dir, "folds_without_confirmation.json")
    with open(path, "w") as fh:
        json.dump(d, fh)
    return path, fmap


def run_m6(ss, runner, per_fold, out_dir, n_threshold_boot=200, name="va30", n_boot=1000, fa_targets="1,2,5"):
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
    folds_path, fmap = folds_without_confirmation(out_dir)
    df["fold"] = df["fold"].map(fmap)
    path = os.path.join(out_dir, "va30_scores.parquet")
    df.to_parquet(path, index=False)
    va30.main(["--scores", path, "--out_dir", os.path.join(out_dir, name), "--folds", folds_path, "--restrict_to_scored",
               "--horizons", S.MANIFEST_V1, "--threshold_bootstrap", str(n_threshold_boot), "--n_boot", str(n_boot),
               "--fa_targets", fa_targets])
    return path


def window_loss(ss, pool):
    """Events whose 30-120 s window ticks are removed by the soft masks (sensitivity runs only)."""
    ev = pool.events
    rows = []
    for k in range(len(ev)):
        vt = pool.vts[pool.ev_vi[k]]
        ia, ib = int(pool.ev_ia[k]), min(int(pool.ev_ib[k]), vt.n - 1)
        if ib < ia:
            rows.append((0, 0, bool(ev["first"].iloc[k]), str(ev["source_kind"].iloc[k])))
            continue
        hard = ss.bench.labels(vt.vid).masked[:vt.n][ia:ib + 1]
        base = ~hard
        after = ~vt.masked[ia:ib + 1]
        rows.append((int(base.sum()), int(after.sum()), bool(ev["first"].iloc[k]), str(ev["source_kind"].iloc[k])))
    arr = np.array([(a, b) for a, b, _, _ in rows], np.int64).reshape(-1, 2)
    first = np.array([r[2] for r in rows], bool)
    had = arr[:, 0] > 0
    lost = had & (arr[:, 1] == 0)
    by_src = {}
    for src in sorted({r[3] for r in rows}):
        sel = np.array([r[3] == src for r in rows], bool)
        by_src[src] = {"events_with_window": int((had & sel).sum()), "events_lost": int((lost & sel).sum())}
    return {"variant": ss.sensitivity, "events": len(rows), "events_with_window_ticks": int(had.sum()),
            "events_losing_whole_window": int(lost.sum()),
            "events_partly_masked": int((had & ~lost & (arr[:, 1] < arr[:, 0])).sum()),
            "first_segment_events_with_window_ticks": int((had & first).sum()),
            "first_segment_events_losing_whole_window": int((lost & first).sum()),
            "window_ticks_before": int(arr[:, 0].sum()), "window_ticks_after": int(arr[:, 1].sum()),
            "by_source": by_src}


SENS_KEYS = ("m1", "m1_long", "m3p", "m4n")


def _val(m, key):
    d = (m or {}).get(key)
    return d.get("value") if isinstance(d, dict) else None


def _ci(m, key):
    d = (m or {}).get(key)
    return d.get("ci") if isinstance(d, dict) else None


def _family_for_decision(root, names, pr):
    blocks = {}
    for n in names:
        p = os.path.join(root, n, "metrics.json")
        if not os.path.isfile(p):
            return None
        with open(p) as fh:
            m = json.load(fh)
        if (m.get("config") or {}).get("prereg_sha256") != pr.sha256:
            return None
        m["tail"] = (m.get("events_first") or {}).get("tail")
        blocks[n] = m
    return blocks


def sensitivity_report(sens_root, primary_root, names, variant, pr):
    """Per block M1 (band), M1-long, M3p, M4n of a sensitivity run, the difference to the primary M1, lost events and
    the label-noise flag; written next to the sensitivity metrics, never read by ``decide``."""
    rows, sens_m, prim_m = {}, {}, {}
    for n in names:
        sp, pp = (os.path.join(r, n, "metrics.json") for r in (sens_root, primary_root))
        if not os.path.isfile(sp):
            continue
        with open(sp) as fh:
            sens_m[n] = json.load(fh)
        if os.path.isfile(pp):
            with open(pp) as fh:
                m = json.load(fh)
            if (m.get("config") or {}).get("prereg_sha256") == pr.sha256:
                prim_m[n] = m
    phases = {r["name"]: r["phase"] for r in F.resolve_blocks(pr)}
    verdict = {}
    for ph in sorted({phases.get(n) for n in sens_m} - {None}):
        fam = [n for n in sens_m if phases.get(n) == ph and n in prim_m]
        if not fam:
            continue
        exempt = [n for n in fam if n in F.BLOCKS and F.BLOCKS[n].g2_exempt]
        sb, pb = _family_for_decision(sens_root, fam, pr), _family_for_decision(primary_root, fam, pr)
        if not sb or not pb:
            continue
        ds, dp = D.decide_phase(sb, exempt), D.decide_phase(pb, exempt)
        for n in fam:
            verdict[n] = {"primary_G1": dp[n]["G1"], "sensitivity_G1": ds[n]["G1"], "family": fam,
                          "signal_explained_by_onset_label_noise": bool(
                              variant == "S1" and dp[n]["G1"] == "PASS" and ds[n]["G1"] == "FAIL")}
    for n, m in sens_m.items():
        pm = prim_m.get(n)
        row = {k: {"value": _val(m, k), "ci": _ci(m, k)} for k in SENS_KEYS}
        p1, s1 = _val(pm, "m1"), _val(m, "m1")
        row["primary_m1"] = {"value": p1, "ci": _ci(pm, "m1")}
        row["m1_minus_primary_m1"] = None if p1 is None or s1 is None else s1 - p1
        row["m1_difference_note"] = ("point difference; the primary bootstrap replicates are not stored"
                                     if pm else "no primary metrics.json from this pre-registration")
        row["events"] = m.get("sensitivity")
        row["g1"] = verdict.get(n)
        rows[n] = row
    out = {"variant": variant, "primary_class_unchanged": True, "used_for_decision": False,
           "prereg_sha256": pr.sha256, "blocks": rows}
    write_json(os.path.join(sens_root, "sensitivity_report.json"), out)
    fmt = lambda x: "" if x is None else f"{x:.3f}"  # noqa: E731
    lines = [f"Sensitivity {variant} (secondary, never used for a decision)", "",
             "| block | M1 | primary M1 | diff | M1-long | M3p | M4n | events with window | lose whole window | G1 primary | G1 sens | label noise |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in rows.items():
        e, g = r["events"] or {}, r["g1"] or {}
        lines.append(f"| {n} | {fmt(r['m1']['value'])} | {fmt(r['primary_m1']['value'])} | {fmt(r['m1_minus_primary_m1'])} | "
                     f"{fmt(r['m1_long']['value'])} | {fmt(r['m3p']['value'])} | {fmt(r['m4n']['value'])} | "
                     f"{e.get('events_with_window_ticks', '')} | {e.get('events_losing_whole_window', '')} | "
                     f"{g.get('primary_G1', '')} | {g.get('sensitivity_G1', '')} | "
                     f"{'signal explained by onset label noise' if g.get('signal_explained_by_onset_label_noise') else ''} |")
    with open(os.path.join(sens_root, "sensitivity_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return out


def run_block(ss, source, block, out_root, args, prereg_sha=None):
    t0 = time.time()
    runner = P.BlockRun(ss, source, block, seed=0, require_cut=getattr(args, "require_cut", True))
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
    sens = getattr(ss, "sensitivity", None)
    if sens:
        res["sensitivity"] = window_loss(ss, pool)
    res["config"] = {"block": block.name, "phase": block.phase, "streams": list(block.streams), "columns": runner.columns,
                     "wide": bool(runner.wide), "n_boot": args.n_boot, "prereg_sha256": prereg_sha,
                     "soft_masks": ss.soft_mask_info, "status": AVAILABLE_STATUS, "seconds": round(time.time() - t0, 1),
                     "diegetic_filter_applied": False if block.audio else None, "interpretations": INTERPRETATIONS,
                     "cut_stream_incomplete": list(runner.cut_incomplete)}
    if sens:
        res["config"]["sensitivity"] = sens
    write_json(os.path.join(out_dir, "metrics.json"), res)
    write_scores(out_dir, ss, runner, per_fold, models)
    if args.m6:
        run_m6(ss, runner, per_fold, out_dir, getattr(args, "threshold_boot", 200), n_boot=args.n_boot,
               fa_targets=getattr(args, "fa_targets", "1,2,5"))
    log(f"{block.name}: M1 {res['m1'].value} done in {time.time() - t0:.0f} s")
    return res


class FamilyError(RuntimeError):
    pass


def family_names(phase, pr=None):
    """Blocks that form the Holm family of a phase: the AVAILABLE ones (NOT_AVAILABLE and gated blocks cannot be
    tested and do not count toward the family size)."""
    if pr is not None:
        return [r["name"] for r in F.resolve_blocks(pr) if r["phase"] == phase and r["status"] == F.AVAILABLE]
    return [b.name for b in F.BLOCKS.values() if b.phase == phase and not b.audio]


def decide(out_root, phase, pr=None, partial=False, family=None):
    """Holm adjustment and classes for a phase. Refuses unless every block of the family has a metrics.json produced
    under the registered pre-registration; ``partial`` labels the output PARTIAL_FAMILY instead of refusing."""
    names = list(family) if family is not None else family_names(phase, pr)
    blocks, exempt, missing, stale = {}, [], [], []
    for n in names:
        p = os.path.join(out_root, n, "metrics.json")
        if not os.path.isfile(p):
            missing.append(n)
            continue
        with open(p) as fh:
            m = json.load(fh)
        if pr is not None and (m.get("config") or {}).get("prereg_sha256") != pr.sha256:
            stale.append(n)
            continue
        if (m.get("config") or {}).get("sensitivity"):
            stale.append(n)
            continue
        m["tail"] = (m.get("events_first") or {}).get("tail")
        m["tail_all_segments"] = (m.get("events") or {}).get("tail")
        blocks[n] = m
        if n in F.BLOCKS and F.BLOCKS[n].g2_exempt:
            exempt.append(n)
    if not names:
        raise FamilyError(f"phase {phase}: no AVAILABLE block to decide")
    if (missing or stale) and not partial:
        raise FamilyError(f"phase {phase}: metrics.json missing for {missing}, from another pre-registration for {stale}; "
                          f"run them or pass --partial_family")
    if not blocks:
        raise FamilyError(f"phase {phase}: no usable metrics.json")
    family_status = "COMPLETE" if not (missing or stale) else "PARTIAL_FAMILY"
    result = D.decide_phase(blocks, exempt)
    unavailable = [r for r in F.resolve_blocks(pr) if r["phase"] == phase and r["status"] != F.AVAILABLE] if pr else []
    write_json(os.path.join(out_root, f"decisions_phase_{phase}.json"),
               {"prereg_sha256": pr.sha256 if pr else None, "family_status": family_status,
                "holm_family_blocks": sorted(blocks), "holm_family_tests": 3 * len(blocks),
                "no_metrics_file": missing, "stale_prereg": stale, "not_run": unavailable, "decisions": result})
    lines = [f"Holm family: {family_status}, {len(blocks)} blocks x 3 G1 tests", "",
             "| block | M1 | G1 | G2 | G3 | class | tags | M4b | first-frame flag | soundtrack flag |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda x: "" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))  # noqa: E731
    for k, r in result.items():
        v = (blocks[k].get("m1") or {}).get("value")
        lines.append(f"| {k} | {fmt(v)} | {r['G1']} | {r['G2']} | {r['G3']} | {r['class']} | {', '.join(r['tags'])} | "
                     f"{fmt(r['m4b'])} | {fmt(r['first_frame_flag'])} | {fmt(r['soundtrack_flag'])} |")
    for r in unavailable:
        lines.append(f"| {r['name']} |  | {r['status']} |  |  | {r['status']} |  |  |  |  |")
    for k in missing + stale:
        lines.append(f"| {k} |  | NOT_RUN |  |  | NOT_RUN |  |  |  |  |")
    with open(os.path.join(out_root, f"results_phase_{phase}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return result


NOT_IMPLEMENTED = {
    "P2_gru": "one-layer GRU (64 units, 3 seeds, G1 PASS only): the event-window MIL loss is not specified enough to "
              "implement before any block passes G1",
    "P1_full_dimensional_L2_sensitivity": "full-dimensional L2 on the windowed wide blocks (1152 or 2048 columns x 9 "
                                          "summaries) is not implemented; only the PLS32 reduction runs",
    "forward_selection": "greedy fusion with paired M8 gain needs the fused-model trainer",
    "confirmation_rule": "the single fold-1 evaluation of the fused model; fold 1 is unreachable in this harness by design",
    "M6_sustained_leads_90_120": "the VA30 evaluator (not modifiable here) reports sustained success at 60 s only",
    "M6_clean_alarm_time_operating_points": "operating points at 1, 2 and 5 percent clean alarm time are not selectable "
                                            "in the VA30 evaluator, which has FA/h targets (1, 2, 5) only",
    "M7_same_film_matched_AUROC": "needs a film-level grouping of XD movie clips",
    "M7_news_and_compilation_exclusion": "needs the compilation/news annotation",
    "M7_visible_conflict_vs_sudden": "needs a manual annotation",
    "M8_partial_AMOC_area": "reported by the VA30 evaluator with --m6, not by the screen metrics",
    "control_aggression_level_no_context": "no score table of an aggression-level reference model is available",
    "context_cue_G2_exemption": "weapon, police, counter and arrivals cues exempt from G2 only as class-conditional "
                                "heads; no block sets Block.g2_exempt and the prompt columns are judged inside "
                                "their blocks; an exempt G2 can never give ANTICIPATION",
    "control_b_teacher": "no score table of the zero-shot teacher on the screening cohort is available",
}
INTERPRETATIONS = {
    "tail_threshold_source": "the frozen text says tail_train_quantile; the 99.5th percentile is taken on the fold "
                             "model's clean VALIDATION ticks (training scores are in-sample). Needs a dated hashed "
                             "amendment before any block outcome is read",
    "G1_multiplicity": "one Holm family over the M1, standardized-pAUC and first-segment tail-hit tests of every "
                       "AVAILABLE block of the phase; the sparse route needs its adjusted p < 0.05",
    "G2_undefined_ratio": "M3p upper < 0.55 is FAIL even when the ratio is undefined; the ratio clauses need a "
                          "defined ratio",
    "G1_tail_events": "first-segment events only; the all-segment tail is reported as tail_all_segments",
}
IMPLEMENTED_NOTES = {
    "M6_event_threshold_CI_200_resamples": "VA30 evaluator with --m6, --threshold_bootstrap from the frozen file",
    "M6_fa_targets_1_2_5": "VA30 evaluator with --m6, --fa_targets from the frozen file",
}


def check_frozen_constants(pr):
    """Mismatches between the code constants and the numbers in the frozen probes, nuisance, metrics and gates."""
    bad = list(D.check_constants(pr))
    p1 = pr.data["probes"]["P1"]
    want = [
        ("P1 C grid", tuple(P.C_GRID), tuple(float(c) for c in p1["C"])),
        ("nuisance C grid", tuple(P.C_GRID), tuple(float(c) for c in pr.data["nuisance"]["C"])),
        ("P1 windows", tuple(F.WINDOWS), tuple(int(w) for w in p1["windows_sec"])),
        ("P1s slopes", tuple(F.SLOPE_WINDOWS), tuple(int(w) for w in pr.data["probes"]["P1s"]["slopes_sec"])),
        ("P1b baseline", F.BASELINE_SEC, pr.data["probes"]["P1b"]["baseline_sec"]),
        ("wide threshold", F.WIDE_COLUMNS, p1["wide_block_threshold"]),
        ("PLS components", F.PLS_COMPONENTS, int(re.search(r"PLS(\d+)", p1["reduction"]).group(1))),
        ("P1 seeds", (0,), tuple(p1["seeds"])),
        ("P1 near target", (0.0, 10.0), tuple(float(x) for x in pr.data["probes"]["P1_near"]["target_tte_sec"])),
        ("scene PCA", P.SCENE_PCA, int(re.search(r"PCA(\d+)", pr.data["nuisance"]["scene_reduction"]).group(1))),
        ("bootstrap seed", 0, pr.data["metrics"]["bootstrap"]["seed"]),
    ]
    bad += [f"{n}: code {a} frozen {b}" for n, a, b in want if a != b]
    summ = [x.replace("fraction_above_clean_train_p95", "frac") for x in p1["summaries"]]
    if summ != ["mean", "max", "frac"]:
        bad.append(f"P1 summaries: {p1['summaries']}")
    return bad


def prepare(args):
    """Load and verify the frozen file, build the screening set from it, compare with the previous in-code rules."""
    pr = prereg.load(args.prereg)
    bad = check_frozen_constants(pr)
    if bad:
        raise SystemExit("code constants differ from the pre-registration:\n  " + "\n  ".join(bad))
    ss = S.load_screen_set(pr)
    fatal, notes = prereg.verify_inputs(pr, ss.bench)
    if fatal:
        raise SystemExit("inputs differ from the pre-registration:\n  " + "\n  ".join(fatal))
    ref = S.load_screen_set(None, soft_masks_path=None)
    agree = prereg.agreement(pr, ss, ref, ss.bench)
    for c in agree["disagreements"]:
        log("DISAGREEMENT frozen vs previous computation:", {k: v for k, v in c.items() if k not in ("examples",)})
    return pr, ss, {"input_notes": notes, "agreement": agree}


def run_report(pr, ss, extra, statuses):
    return {"prereg": {"path": os.path.relpath(pr.path, REPO_ROOT), "sha256": pr.sha256},
            "screen_folds": list(pr.screen_folds), "confirmation_fold": pr.confirmation_fold,
            "label_policy": pr.data["label_policy"], "soft_masks": ss.soft_mask_info, "audio_gate": pr.data["audio_gate"],
            "blocks": statuses, "not_implemented": NOT_IMPLEMENTED, "interpretations": INTERPRETATIONS, "implemented_by_delegation": IMPLEMENTED_NOTES,
            "controls": {c: ("NOT_AVAILABLE" if "aggression" in c or c == "b_teacher" else "IMPLEMENTED")
                         for c in pr.data["controls"]},
            "counts": ss.counts(), **extra}


def dry_run(ss, source, blocks, out_root):
    """Tick counts of the frozen cohorts and feature coverage per block (no metric is computed)."""
    info = {"counts": ss.counts(), "blocks": {}}
    groups = {"positive": ss.video_lists()["positive_videos"], "clean": ss.video_lists()["clean_negative_videos"],
              "h1": ss.h1_ids(), "h1p": ss.h1p_ids(), "clean_vlm": sorted(ss.prereg.ids("clean_vlm_videos"))}
    for b in blocks:
        row = {}
        for g, vids in groups.items():
            need = have = vids_with = 0
            for v in vids:
                if b.sources and ss.source_kind(v) not in b.sources:
                    continue
                n = ss.ticks(v).n
                if not n:
                    continue
                need += n
                if all(source.has(s, v) for s in b.streams):
                    vids_with += 1
                    try:
                        k = min(n, min(source.n_ticks(s, v) for s in b.streams))
                    except Exception:
                        k = 0
                    have += k
            nv = len([v for v in vids if not (b.sources and ss.source_kind(v) not in b.sources)])
            no_cut = sum(1 for v in vids if (not (b.sources and ss.source_kind(v) not in b.sources))
                         and ss.ticks(v).n and not source.has(F.CUT_STREAM, v))
            row[g] = {"cut_stream_missing": no_cut, "videos": nv, "with_features": vids_with, "ticks": int(need), "ticks_covered": int(have),
                      "tick_coverage": round(have / need, 4) if need else None}
        ev = ss.events()
        ok = set(v for v in groups["positive"] if all(source.has(s, v) for s in b.streams))
        row["events_covered"] = int(ev["video_id"].isin(ok).sum())
        row["first_segment_events_covered"] = int((ev["video_id"].isin(ok) & ev["first"]).sum())
        info["blocks"][b.name] = row
        log(f"{b.name}: " + ", ".join(f"{g} {r['with_features']}/{r['videos']} ({r['tick_coverage']})"
                                      for g, r in row.items() if isinstance(r, dict)))
    every = groups["positive"] + groups["clean"] + groups["h1"]
    info["scene0p_videos_with_embedding"] = sum(1 for v in every if source.scene(v) is not None)
    info["scene0p_videos_needed"] = len(every)
    write_json(os.path.join(out_root, "dry_run_coverage.json"), info)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["A", "B", "C"])
    ap.add_argument("--blocks", default="all")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "benchmark", "results", "cue_screen"))
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--decide", choices=["A", "B", "C"])
    ap.add_argument("--m6", action="store_true", help="also score the VA30 event metrics (slow)")
    ap.add_argument("--m2", action="store_true", help="person-count stratified AUROC (needs pose_interact)")
    ap.add_argument("--partial_family", action="store_true", help="with --decide: allow missing blocks (labelled)")
    ap.add_argument("--sensitivity", choices=sorted(S.SENSITIVITY_VARIANTS), default=None,
                    help="amendment 03: apply soft_masks_v1.json (S1 without uca_weapon_only, S2 with); writes to "
                         "<out>/sensitivity_<variant>/ and never feeds --decide")
    ap.add_argument("--feature_root", default=None)
    ap.add_argument("--prereg", default=prereg.PREREG_PATH)
    args = ap.parse_args(argv)
    va30.N_THREADS = 4
    pr, ss, extra = prepare(args)
    args.n_boot = int(pr.data["metrics"]["bootstrap"]["n"])
    args.threshold_boot = int(pr.data["metrics"]["bootstrap"]["threshold_bootstrap"])
    args.fa_targets = ",".join(str(x) for x in pr.data["metrics"]["M6_fa_targets"])
    if args.sensitivity and (args.decide or args.dry_run):
        raise SystemExit("--sensitivity cannot be combined with --decide or --dry_run")
    primary_out = args.out
    if args.sensitivity:
        try:
            masks, info = S.load_sensitivity_masks(args.sensitivity)
        except S.SoftMaskError as e:
            raise SystemExit(str(e))
        ss.apply_sensitivity(masks, info)
        args.out = os.path.join(primary_out, f"sensitivity_{args.sensitivity}")
        args.m6 = False
        log(f"sensitivity {args.sensitivity}: {info['entries']} mask entries on {info['videos']} videos -> {args.out}")
    if args.decide:
        decide(args.out, args.decide, pr, partial=args.partial_family)
        return 0
    statuses = F.resolve_blocks(pr)
    for r in statuses:
        if r["status"] != F.AVAILABLE:
            log(f"{r['status']}: {r['phase']} {r['name']} ({r['reason']})")
    for k, v in NOT_IMPLEMENTED.items():
        log(f"NOT_IMPLEMENTED: {k} ({v})")
    write_json(os.path.join(args.out, "run_report.json"), run_report(pr, ss, extra, statuses))
    by_name = {r["name"]: r for r in statuses}
    if args.blocks == "all":
        names = [r["name"] for r in statuses if (args.phase is None and r["phase"] == "A") or r["phase"] == args.phase]
    else:
        names = args.blocks.split(",")
        unknown = [n for n in names if n not in by_name]
        if unknown:
            raise SystemExit(f"blocks not in the pre-registration: {unknown}")
    refused = [n for n in names if by_name[n]["status"] != F.AVAILABLE]
    if args.dry_run:
        names = [n for n in names if n not in refused]
    elif refused and args.blocks != "all":
        raise SystemExit("blocks that cannot run: " + ", ".join(f"{n} {by_name[n]['status']}" for n in refused))
    names = [n for n in names if n not in refused]
    if not names:
        raise SystemExit("nothing to run: " + ", ".join(f"{n} {by_name[n]['status']}" for n in refused))
    blocks = [F.BLOCKS[n] for n in names]
    source = F.FeatureSource(ss, root=args.feature_root)
    if args.dry_run:
        with threadpool_limits(limits=4):
            dry_run(ss, source, blocks, args.out)
        return 0
    with threadpool_limits(limits=4):
        for b in blocks:
            run_block(ss, source, b, args.out, args, pr.sha256)
        if args.sensitivity:
            sensitivity_report(args.out, primary_out, names, args.sensitivity, pr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
