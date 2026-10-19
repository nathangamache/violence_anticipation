"""External test of the frozen two-head model (benchmark/external/external_test_plan_v1.md).

    nice -n 10 .venv/bin/python -m anticipation.screen.external --out benchmark/results/external_v1

Trains both frozen heads as confirm.run_head does, checks the scorer on the fold-1 test videos, writes the fold-1
thresholds, scores CCTV-Fights and the frozen MEVA list, and computes the plan metrics. Nothing is tuned on external data.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse
import hashlib
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from anticipation.data.catalog import REPO_ROOT
from anticipation.eval import va30
from anticipation.screen import confirm as C
from anticipation.screen import features as F
from anticipation.screen import fusion as FU
from anticipation.screen import metrics as M
from anticipation.screen import probes as P
from anticipation.screen import run as R

CONFIG = os.path.join(REPO_ROOT, "benchmark", "results", "fusion_v2", "confirmation_config.json")
CONFIG_SHA = "8e4dc5eb944257c891c30d22b9c1173750d009ad8e965be0480f01bd0007ef41"
COMMITTED = os.path.join(REPO_ROOT, "benchmark", "results", "confirmation", "confirmation.json")
CCTV_DIR = os.path.join(REPO_ROOT, "benchmark", "external", "cctv_fights")
MEVA_SELECTION = os.path.join(REPO_ROOT, "data", "annotation", "meva", "selection.json")
FA_TARGETS = (1, 2, 5)
HEAD_STREAMS = {"early": "siglip2_prompts", "imminent": "siglip2_prompts_v2"}
HEAD_LEAD = {"early": 30, "imminent": 10}
HEAD_POS = {"early": (30.0, 120.0), "imminent": (0.0, 10.0)}
N_BOOT = 1000
MIN_MEVA = 300


def log(*a):
    print("[external]", *a, flush=True)


class FrozenHead:
    """The fitted transform and probe of one head, able to score any video that has its feature files."""

    def __init__(self, head, names, runners, probe):
        self.head, self.names, self.probe = head, names, probe
        self.blocks = [runners[n].block for n in names]
        self.pls = [runners[n].pls for n in names]
        self.thr = [runners[n].thr["p1"] for n in names]
        self.wide = [runners[n].wide_columns for n in names]
        self.cut_incomplete = []

    def score(self, source, vid):
        parts = []
        for block, pls, thr, wide in zip(self.blocks, self.pls, self.thr, self.wide):
            x, _, _ = F.load_block(source, block, vid)
            if x.shape[1] > wide:
                x = x.astype(np.float16).astype(np.float32)
            xt = pls.transform(x).astype(np.float32) if pls is not None else x
            flags, ok = F.cut_flags(source, vid, len(xt))
            if not ok:
                self.cut_incomplete.append(vid)
            parts.append(F.window_features(xt, F.last_cut_index(flags), thr))
        T = min(len(p) for p in parts)
        return self.probe.score(np.concatenate([p[:T] for p in parts], axis=1))


def train_head(head, hs, spec, cset, source, ss):
    """Same training path as confirm.run_head up to the probe; returns the frozen head and the fold-1 context."""
    C.check_blocks(hs)
    target = "near" if head == "imminent" else "band"
    names = [b["name"] for b in hs["blocks"]]
    runners = {n: P.BlockRun(cset, source, F.BLOCKS[n], seed=hs["seed"], require_cut=C.REQUIRE_CUT) for n in names}
    cov = lambda v: all(r.covered(v) for r in runners.values())  # noqa: E731
    train = [v for v in FU.training_list(ss) if cov(v)]
    test = [v for v in cset.confirmation_test_ids() if cov(v)]
    if len(train) != spec["training"]["n_videos"]:
        raise C.ConfirmError(f"{head}: {len(train)} training videos covered, spec has {spec['training']['n_videos']}")
    for r in runners.values():
        r.set_fold(train)
    vt = {v: cset.ticks(v).cut(min(r.vt(v).n for r in runners.values())) for v in train + test}
    p1 = hs["p1"]
    rng = np.random.default_rng(hs["seed"] + C.FINAL_RNG_SALT)
    plan = P.Plan([vt[v] for v in train], target, rng, p1["max_train_pos"], p1["max_train_neg"], cset.source_kind)
    if plan.empty:
        raise C.ConfirmError(f"{head}: no {target} training rows")
    X, y, w, _ = C.p1_rows(runners, names, plan, vt, None, cset.source_kind)
    probe = FU.fit_fixed(np.concatenate([X[n] for n in names], axis=1), y, w, p1["C"])
    ref = {v: probe.score(np.concatenate([runners[n].feats(v, "p1")[:vt[v].n] for n in names], axis=1)) for v in test}
    log(f"{head}: trained on {len(train)} videos ({int(y.sum())} positive rows), fold-1 test {len(test)}")
    return FrozenHead(head, names, runners, probe), {"vt": vt, "test": test, "ref": ref, "n_train": len(train)}


def fold1_check(head, fh, ctx, spec, cset, source, committed):
    vt, test = ctx["vt"], ctx["test"]
    scores = {v: fh.score(source, v)[:vt[v].n] for v in test}
    diff = max(float(np.max(np.abs(scores[v] - ctx["ref"][v]))) for v in test)
    pool = M.Pool(cset.fold1, [vt[v] for v in test])
    W = pool.boot_weights(spec["bootstrap"]["n"], seed=spec["bootstrap"]["seed"])
    if head == "imminent":
        evp = FU.near_event_pool(pool)
        st = FU.pool_stats_near(pool, evp, W, pool.assemble(scores), True)
    else:
        st = FU.pool_stats(pool, W, pool.assemble(scores), True)
    got = float(st["m1"].value)
    want = float(committed["heads"][head]["model"]["m1"]["value"])
    return scores, {"m1_value": got, "m1_committed": want, "abs_diff": abs(got - want), "match_4_decimals": bool(round(got, 4) == round(want, 4)),
                    "max_abs_score_diff_vs_runner_path": diff, "videos": len(test), "ticks": int(sum(len(s) for s in scores.values()))}


def validation_machinery(cset, spec, oof_path, test_scores, vt):
    """Benchmark, score table and eval set of the fold-1 pseudo-fold, built as confirm.fold1_events builds them."""
    df, folds, n_cal = C.calibrated_frame(cset, oof_path, test_scores, vt)
    em = spec["event_metrics"]
    bench = va30.Benchmark(cset.bench.videos, cset.bench.events, folds, {"folds_path": "fold1_confirmation"},
                           video_horizons=os.path.join(REPO_ROOT, em["horizons"]))
    cfg = va30.EvalConfig(n_boot=0, fa_targets=tuple(float(x) for x in em["fa_targets"]), restrict_to_scored=True, allow_missing=True)
    table = va30.ScoreTable(df, bench.video_horizons)
    eval_set = {v for v in bench.eval_ids(cfg.onset_sources) if v in table.videos}
    return bench, table, eval_set, cfg, n_cal


def fold1_thresholds(cset, spec, oof_path, test_scores, vt):
    bench, table, eval_set, cfg, n_cal = validation_machinery(cset, spec, oof_path, test_scores, vt)
    folds = bench.folds
    val_by = {0: [v for v in folds.val_ids(0) if v in eval_set]}
    test_by = {0: [v for v in folds.test_ids(0) if v in eval_set]}
    vs = va30._collect(bench, table, 0, val_by, cfg, {})
    val_ee = va30.EventEval(va30.TickSet(bench, list(vs), vs), cfg.policy)
    ts_scores = va30._collect(bench, table, 0, test_by, cfg, {})
    ts = va30.TickSet(bench, list(ts_scores), ts_scores, {v: 0 for v in ts_scores})
    ee = va30.EventEval(ts, cfg.policy)
    out = {}
    for t in FA_TARGETS:
        theta, val_fa = va30.choose_threshold(val_ee, float(t))
        o = ee.outcomes(np.full(ts.V, theta))
        fa = float(o["fa"][ts.clean_v].sum() / ee.clean_hours)
        e30 = ee.ev_e30
        hit = lambda L: float((np.nan_to_num(o["lead"], nan=-1.0)[e30] >= L).mean())  # noqa: E731
        out[f"fa{t}"] = {"target_fa_per_hour": t, "theta": theta, "val_fa_per_hour": val_fa, "calibration_videos": n_cal,
                         "fold1_test_fa_per_hour_clean": fa, "fold1_success30": hit(30), "fold1_success10": hit(10)}
    return out


def committed_ops(committed, head):
    ev = committed["heads"][head]["events"]["operating_points"]
    return {k: {"fa": v["test_fa_per_hour_clean"], "s30": (v.get("success30") or v.get("success") or {}).get("value")} for k, v in ev.items()}


def meva_candidates(root, selection=None):
    """MEVA clips of the selection that exist on disk and have both prompt feature files."""
    sel = json.load(open(selection or MEVA_SELECTION))
    out = []
    for c in sel["clips"]:
        f = c.get("file")
        if not f or not os.path.isfile(f):
            continue
        stem = os.path.splitext(os.path.basename(f))[0]
        if all(os.path.isfile(os.path.join(root, s, "meva", stem + ".npz")) for s in HEAD_STREAMS.values()):
            out.append({"video_id": "meva/" + stem, "clip": c["clip"], "site": c["site"], "camera": c["camera"],
                        "date": c["date"], "indoor_outdoor": c.get("indoor_outdoor")})
    return sorted(out, key=lambda r: r["video_id"])


def extractors_running():
    r = subprocess.run(["pgrep", "-af", "extract"], capture_output=True, text=True)
    me = str(os.getpid())
    return [ln for ln in r.stdout.splitlines() if "meva" in ln and "--extractor siglip2" in ln and not ln.startswith(me + " ")]


def wait_for_meva(root, max_hours, interval, selection=None):
    """Poll until the candidate count is unchanged over two polls and no extractor is running (or the time is up)."""
    t0, prev, stable = time.time(), -1, 0
    while True:
        n = len(meva_candidates(root, selection))
        busy = extractors_running()
        log(f"MEVA clips with both features: {n}; extractors running: {len(busy)}")
        stable = stable + 1 if (n == prev and n > 0 and not busy) else 0
        prev = n
        if stable >= 2:
            return n, "stable"
        if time.time() - t0 > max_hours * 3600:
            return n, "timeout"
        time.sleep(interval)


def sha_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def align(s, T):
    """Scores on exactly T ticks: trimmed, or the last score held when the feature file is shorter."""
    if len(s) >= T:
        return s[:T], 0
    return np.concatenate([s, np.full(T - len(s), s[-1])]), T - len(s)


def build_external(cctv, meva_rows, meva_ticks):
    vids = cctv.copy()
    vids["domain"] = vids["domain"].astype(str)
    vids["split_group"] = vids["split_group"].astype(str)
    keep = ["duration_sec", "role", "onset_source", "source_kind", "interpersonal", "has_audio", "group", "split_group", "domain"]
    vids = vids.set_index("video_id")[keep]
    rows = []
    for r in meva_rows:
        rows.append({"video_id": r["video_id"], "duration_sec": float(meva_ticks[r["video_id"]]), "role": "clean_negative",
                     "onset_source": "none", "source_kind": "meva", "interpersonal": False, "has_audio": False,
                     "group": f"meva:{r['site']}.{r['camera']}", "split_group": r["video_id"], "domain": "meva"})
    meva = pd.DataFrame(rows).set_index("video_id")[keep]
    return pd.concat([vids, meva])


def boot_weights(ts, n_boot, seed, cctv_cluster="split_group", meva_cluster="video"):
    """W [B, V]: CCTV-Fights videos and MEVA clips resampled independently (video clusters)."""
    rng = np.random.default_rng(seed)
    W = np.zeros((n_boot, ts.V))
    for sel, how in ((ts.violent_v, cctv_cluster), (ts.clean_v, meva_cluster)):
        idx = np.flatnonzero(sel)
        if not len(idx):
            continue
        if how == "split_group":
            lab = ts.bench.videos.loc[np.array(ts.vids)[idx], "split_group"].astype(str).values
        elif how == "camera":
            lab = ts.bench.videos.loc[np.array(ts.vids)[idx], "group"].astype(str).values
        else:
            lab = np.array(ts.vids)[idx]
        _, inv = np.unique(lab, return_inverse=True)
        G = inv.max() + 1
        W[:, idx] = rng.multinomial(G, np.full(G, 1.0 / G), size=n_boot)[:, inv]
    return W


def ci95(b):
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if len(b) > 10 else None


def auc_boot(score, pos, neg, vidx, W):
    """Tick AUROC (ties count half) of pos vs neg ticks with video-cluster bootstrap weights W [B, V]."""
    m = pos | neg
    s, yp, yn, vi = score[m], pos[m], neg[m], vidx[m]
    if not yp.any() or not yn.any():
        return {"value": None, "ci": None, "n_pos": int(yp.sum()), "n_neg": int(yn.sum())}
    _, g = np.unique(s, return_inverse=True)
    G = int(g.max()) + 1

    def one(w):
        pw = np.bincount(g, weights=w * yp, minlength=G)
        nw = np.bincount(g, weights=w * yn, minlength=G)
        if pw.sum() <= 0 or nw.sum() <= 0:
            return np.nan
        return float((pw * (np.cumsum(nw) - nw + 0.5 * nw)).sum() / (pw.sum() * nw.sum()))

    return {"value": one(np.ones(len(s))), "ci": ci95([one(W[b][vi]) for b in range(len(W))]),
            "n_pos": int(yp.sum()), "n_neg": int(yn.sum()), "n_pos_videos": int(len(np.unique(vi[yp]))),
            "n_neg_videos": int(len(np.unique(vi[yn])))}


def head_metrics(head, ts, W, thetas):
    """All plan metrics of one head on a TickSet of CCTV-Fights (evaluation set) plus MEVA clips."""
    ee = va30.EventEval(ts, va30.AlarmPolicy())
    dom = ts.bench.videos.loc[np.array(ts.vids), "domain"].values
    cctv_v = ts.violent_v
    meva_v = ts.clean_v
    ev_dom = dom[ee.ev_v]
    lead_main = HEAD_LEAD[head]
    res = {"counts": {"cctv_videos": int(cctv_v.sum()), "meva_clips": int(meva_v.sum()), "meva_hours": float(ee.hours_out[meva_v].sum()),
                      "eligible30_events": int(ee.ev_e30.sum()), "eligible60_events": int(ee.ev_e60.sum()),
                      "eligible30_by_domain": {d: int((ee.ev_e30 & (ev_dom == d)).sum()) for d in ("cctv", "non_cctv")}},
           "operating_points": {}}
    lo, hi = HEAD_POS[head]
    pos = cctv_v[ts.vidx] & ts.pre & ~ts.masked & (ts.tte > lo) & (ts.tte <= hi)
    far = cctv_v[ts.vidx] & ~ts.masked & ((ts.tte > 120) | ~np.isfinite(ts.tte))
    meva_t = meva_v[ts.vidx]
    dom_t = dom[ts.vidx]
    auc = {}
    for name, dsel in (("all", None), ("cctv", "cctv"), ("non_cctv", "non_cctv")):
        pm = pos if dsel is None else pos & (dom_t == dsel)
        fm = far if dsel is None else far & (dom_t == dsel)
        auc[name] = {"vs_meva": auc_boot(ts.score, pm, meva_t, ts.vidx, W),
                     "vs_far_same_videos": auc_boot(ts.score, pm, fm, ts.vidx, W[:, :] * cctv_v[None, :])}
    res["tick_auroc"] = auc
    res["score_quantiles"] = {k: {str(q): float(np.quantile(ts.score[m], q)) for q in (0.5, 0.9, 0.99, 0.999)} for k, m in
                              (("meva", meva_t), ("cctv_far", far), ("cctv_pos", pos)) if m.any()}
    hours_m = ee.hours_out[meva_v]
    hours_c = ee.hours_out[cctv_v]
    for key, theta in thetas.items():
        out = ee.outcomes(np.full(ts.V, theta))
        fa_m = fa_c = out["fa"]
        Wm, Wc = W[:, meva_v], W[:, cctv_v]
        far_m = float(fa_m[meva_v].sum() / hours_m.sum())
        far_b = (Wm @ fa_m[meva_v]) / (Wm @ hours_m)
        alarm_clip = (fa_m[meva_v] > 0).astype(float)
        far_c = float(fa_c[cctv_v].sum() / hours_c.sum())
        far_cb = (Wc @ fa_c[cctv_v]) / (Wc @ hours_c)
        lead = out["lead"]

        def succ(L, mask):
            sel = mask & (ee.ev_e60 if L >= 60 else ee.ev_e30)
            hit = np.nan_to_num(lead, nan=-1.0) >= L
            if not sel.any():
                return {"value": None, "ci": None, "n": 0}
            Wv = W[:, ee.ev_v[sel]]
            with np.errstate(invalid="ignore", divide="ignore"):
                b = (Wv * hit[sel]).sum(1) / Wv.sum(1)
            return {"value": float(hit[sel].mean()), "ci": ci95(b), "n": int(sel.sum()),
                    "median_lead_sec_of_alarmed": float(np.median(lead[sel][np.isfinite(lead[sel])])) if np.isfinite(lead[sel]).any() else None}

        rand = lambda L, far: 1 - np.exp(-np.asarray(far) * (120.0 - L) / 3600.0)  # noqa: E731
        op = {"theta": theta, "meva_fa_per_hour": {"value": far_m, "ci": ci95(far_b), "alarms": int(fa_m[meva_v].sum())},
              "meva_share_clips_with_alarm": {"value": float(alarm_clip.mean()), "ci": ci95((Wm @ alarm_clip) / Wm.sum(1))},
              "success": {}, "random_reference_at_meva_far": {}, "descriptive_cctv_far_proxy": {
                  "fa_per_hour_outside_event_windows": far_c, "ci": ci95(far_cb),
                  "random_reference": {f"L{L}": float(rand(L, far_c)) for L in (30, 10)}}}
        for L in (60, 30, 10, 0):
            op["success"][f"L{L}"] = {n: succ(L, np.ones(len(ev_dom), bool) if n == "all" else ev_dom == n) for n in ("all", "cctv", "non_cctv")}
            op["random_reference_at_meva_far"][f"L{L}"] = {"value": float(rand(L, far_m)), "ci": ci95(rand(L, far_b))}
        grp = ts.bench.videos.loc[np.array(ts.vids)[meva_v], "group"].values
        op["meva_by_camera_group"] = {g: {"clips": int((grp == g).sum()), "hours": float(hours_m[grp == g].sum()),
                                         "alarms": int(fa_m[meva_v][grp == g].sum()),
                                         "fa_per_hour": float(fa_m[meva_v][grp == g].sum() / hours_m[grp == g].sum())}
                                     for g in sorted(set(grp))}
        res["operating_points"][key] = op
    L = lead_main
    op2 = res["operating_points"]["fa2"]
    s = op2["success"][f"L{L}"]["all"]
    r = op2["random_reference_at_meva_far"][f"L{L}"]["value"]
    a = auc["all"]["vs_meva"]
    c1 = s["ci"] is not None and s["ci"][0] > r
    c2 = a["ci"] is not None and a["ci"][0] > 0.5
    res["verdict"] = {"headline_measure": f"Success@{L} on CCTV-Fights eligible30 at the 2 FA/h threshold",
                      "success_point": s["value"], "success_ci": s["ci"], "random_reference_at_meva_far": r,
                      "success_ci_lower_gt_random": bool(c1), "auroc_vs_meva_point": a["value"], "auroc_vs_meva_ci": a["ci"],
                      "auroc_vs_meva_ci_lower_gt_0.5": bool(c2), "transfers": bool(c1 and c2)}
    return res


def f3(d):
    if not d or d.get("value") is None:
        return "n/a"
    s = f"{d['value']:.3f}"
    return s + (f" [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]" if d.get("ci") else "")


def write_report(path, out):
    L = ["# External test v1: frozen two-head model on CCTV-Fights and MEVA", "",
         f"Plan: benchmark/external/external_test_plan_v1.md. Config sha256 {CONFIG_SHA}.", ""]
    ck = out["check"]
    L += ["## Correctness check on fold-1 test videos", "", "| head | M1 here | M1 committed | abs diff | match 4 dp |", "|---|---|---|---|---|"]
    for h, c in ck.items():
        L.append(f"| {h} | {c['m1_value']:.6f} | {c['m1_committed']:.6f} | {c['abs_diff']:.2e} | {c['match_4_decimals']} |")
    L += ["", "## Thresholds (fold-1 calibration slice)", "", "| head | target FA/h | theta | fold-1 test FA/h (here) |", "|---|---|---|---|"]
    for h, d in out["thresholds"].items():
        for k, v in d.items():
            L.append(f"| {h} | {v['target_fa_per_hour']} | {v['theta']:.4f} | {v['fold1_test_fa_per_hour_clean']:.4f} |")
    L += ["", f"MEVA list: {out['meva']['n_clips']} clips, {out['meva']['hours']:.2f} h, sha256 {out['meva']['sha256']}.", ""]
    for h, m in out["metrics"].items():
        c = m["counts"]
        L += [f"## {h} head", "", f"CCTV-Fights evaluation set videos {c['cctv_videos']}, eligible30 events {c['eligible30_events']} "
              f"(cctv {c['eligible30_by_domain']['cctv']}, non_cctv {c['eligible30_by_domain']['non_cctv']}), eligible60 {c['eligible60_events']}; "
              f"MEVA clips {c['meva_clips']}, {c['meva_hours']:.2f} h.", ""]
        Lm = HEAD_LEAD[h]
        L += [f"| target FA/h | theta | MEVA FA/h | MEVA clips with alarm | Success@{Lm} (all) | random at MEVA FA/h | Success@{Lm} cctv | Success@{Lm} non_cctv | Success@0 (all) | median lead (alarmed) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for k, op in m["operating_points"].items():
            s = op["success"][f"L{Lm}"]
            L.append(f"| {k[2:]} | {op['theta']:.3f} | {f3(op['meva_fa_per_hour'])} | {f3(op['meva_share_clips_with_alarm'])} | {f3(s['all'])} | "
                     f"{f3(op['random_reference_at_meva_far'][f'L{Lm}'])} | {f3(s['cctv'])} | {f3(s['non_cctv'])} | "
                     f"{f3(op['success']['L0']['all'])} | {s['all'].get('median_lead_sec_of_alarmed')} |")
        L += ["", "Other lead thresholds (all domains):", "", "| target FA/h | Success@60 (e60) | Success@30 | Success@10 | Success@0 |", "|---|---|---|---|---|"]
        for k, op in m["operating_points"].items():
            L.append(f"| {k[2:]} | " + " | ".join(f3(op["success"][f"L{x}"]["all"]) for x in (60, 30, 10, 0)) + " |")
        L += ["", "Tick AUROC:", "", "| subset | vs MEVA ticks | vs far ticks of same videos | n pos / n neg MEVA |", "|---|---|---|---|"]
        for n, a in m["tick_auroc"].items():
            L.append(f"| {n} | {f3(a['vs_meva'])} | {f3(a['vs_far_same_videos'])} | {a['vs_meva']['n_pos']} / {a['vs_meva']['n_neg']} |")
        L += ["", "Descriptive (CCTV-Fights far ticks as false-alarm proxy):", "", "| target FA/h | proxy FA/h | random Success@30 | random Success@10 |", "|---|---|---|---|"]
        for k, op in m["operating_points"].items():
            d = op["descriptive_cctv_far_proxy"]
            L.append(f"| {k[2:]} | {d['fa_per_hour_outside_event_windows']:.2f} | {d['random_reference']['L30']:.3f} | {d['random_reference']['L10']:.3f} |")
        v = m["verdict"]
        L += ["", f"Verdict ({v['headline_measure']}): success {v['success_point']} CI {v['success_ci']} vs random {v['random_reference_at_meva_far']:.4f} "
              f"-> lower bound above random: {v['success_ci_lower_gt_random']}; AUROC vs MEVA {v['auroc_vs_meva_point']} CI {v['auroc_vs_meva_ci']} "
              f"-> lower bound above 0.5: {v['auroc_vs_meva_ci_lower_gt_0.5']}. **Transfers: {v['transfers']}**", ""]
    L += ["## Decisions not fixed by the plan", ""] + [f"- {x}" for x in out["decisions"]] + [""]
    open(path, "w").write("\n".join(L))


DECISIONS = [
    "Both heads use the fold-1 thresholds unchanged; external data never enters choose_threshold.",
    "Scoring runs on the full 1 Hz feature file of each video with no label cut; shot-cut flags come from the motion stream where it exists "
    "(CCTV-Fights) and are all-zero for MEVA (no motion stream; fixed cameras).",
    "MEVA clips are clean negatives with duration equal to the number of feature ticks; no horizon cap and no excluded spans.",
    "Bootstrap: 1000 resamples, CCTV-Fights videos (by split_group, one per video) and MEVA clips resampled independently; seed 0. "
    "A camera-cluster MEVA bootstrap is not used for the headline.",
    "Success uses the VA30 alarm policy through va30.EventEval with a fixed threshold per head and target; leads and hits are as in VA30 "
    "(first alarm episode starting in [t0-120, t0)); hit means lead >= L. Success@60 uses eligible60 events, other L use eligible30.",
    "Random reference for Success@L is 1 - exp(-FAR*(120-L)/3600) at the observed MEVA FA/h (W = 90 s for L=30, 110 s for L=10).",
    "Tick AUROC positives: first-segment pre-violence unmasked CCTV-Fights ticks with tte in (30,120] (early) or (0,10] (imminent). "
    "Negatives: all MEVA ticks, or unmasked CCTV-Fights ticks with tte > 120 or no onset ahead. Unweighted (no time matching); pooled over ticks.",
    "CCTV-Fights headline population is the evaluation set (label_reliable and not overlap_main, 985 videos); all 1000 are scored in scores.parquet.",
    "Success criterion applied to the headline measure of each head (Success@30 early, Success@10 imminent) at the 2 FA/h threshold; "
    "AUROC against MEVA uses the head's own positive band over the whole evaluation set.",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "benchmark", "results", "external_v1"))
    ap.add_argument("--feature_root", default=None)
    ap.add_argument("--meva_selection", default=None)
    ap.add_argument("--max_wait_hours", type=float, default=3.0)
    ap.add_argument("--poll_sec", type=float, default=300.0)
    args = ap.parse_args(argv)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    root = args.feature_root or F.io.FEATURE_ROOT

    raw = open(CONFIG, "rb").read()
    if C.sha_bytes(raw) != CONFIG_SHA:
        raise SystemExit("config sha256 mismatch")
    spec = C.read_config(CONFIG, CONFIG_SHA, require_committed=False)
    committed = json.load(open(COMMITTED))
    va30.N_THREADS = 4
    pr, ss, _ = R.prepare(argparse.Namespace(prereg=C.default_prereg()))
    if spec["prereg"]["sha256"] != pr.sha256:
        raise SystemExit("spec made under another pre-registration")
    if FU.training_list_sha(ss) != spec["training"]["videos_sha256"]:
        raise SystemExit("training membership differs from the frozen spec")
    cset = C.ConfirmSet(ss)
    source = F.FeatureSource(cset, root=args.feature_root)
    cfg_dir = os.path.dirname(CONFIG)

    heads, ctxs, checks, thresholds, f1_scores = {}, {}, {}, {}, {}
    with threadpool_limits(limits=4):
        for h, hs in spec["heads"].items():
            heads[h], ctxs[h] = train_head(h, hs, spec, cset, source, ss)
            f1_scores[h], checks[h] = fold1_check(h, heads[h], ctxs[h], spec, cset, source, committed)
            log(h, "check", checks[h])
            if not checks[h]["match_4_decimals"]:
                R.write_json(os.path.join(out_dir, "check.json"), checks)
                raise SystemExit(f"{h}: fold-1 M1 does not reproduce; stopping before any external score")
        R.write_json(os.path.join(out_dir, "check.json"), checks)
        for h, hs in spec["heads"].items():
            oof = os.path.join(cfg_dir, hs["calibration"]["scores_file"])
            if FU.file_sha(oof) != hs["calibration"]["scores_sha256"]:
                raise SystemExit(f"{h}: calibration scores differ from the frozen spec")
            thresholds[h] = fold1_thresholds(cset, spec, oof, f1_scores[h], ctxs[h]["vt"])
            want = committed_ops(committed, h)
            for k, v in thresholds[h].items():
                v["committed_fold1_test_fa_per_hour"] = want[k]["fa"]
                v["fa_matches_committed"] = bool(abs(v["fold1_test_fa_per_hour_clean"] - want[k]["fa"]) < 1e-6)
            log(h, "thresholds", {k: (round(v["theta"], 4), round(v["fold1_test_fa_per_hour_clean"], 4), v["fa_matches_committed"]) for k, v in thresholds[h].items()})
        R.write_json(os.path.join(out_dir, "thresholds.json"), {"config_sha256": CONFIG_SHA, "heads": thresholds})
        if not all(v["fa_matches_committed"] for d in thresholds.values() for v in d.values()):
            raise SystemExit("fold-1 FA/h does not reproduce the committed values; stopping before any external score")
        del ctxs, f1_scores

        cctv = pd.read_csv(os.path.join(CCTV_DIR, "videos.csv"))
        cctv_events = json.load(open(os.path.join(CCTV_DIR, "events.json")))["videos"]
        cctv_all = cctv["video_id"].tolist()
        scores = {h: {} for h in heads}
        for h, fh in heads.items():
            for v in cctv_all:
                scores[h][v] = fh.score(source, v)
            log(h, "scored", len(cctv_all), "CCTV-Fights videos")

        if len(meva_candidates(root, args.meva_selection)) < MIN_MEVA or extractors_running():
            n, why = wait_for_meva(root, args.max_wait_hours, args.poll_sec, args.meva_selection)
        else:
            why = "stable"
        meva_rows = meva_candidates(root, args.meva_selection)
        meva_ticks = {}
        for r in meva_rows:
            meva_ticks[r["video_id"]] = len(np.load(os.path.join(root, "siglip2_prompts", r["video_id"] + ".npz"), allow_pickle=False)["t_sec"])
        meva_list = {"n_clips": len(meva_rows), "hours": sum(meva_ticks.values()) / 3600.0, "wait_outcome": why, "clips": meva_rows,
                     "ticks": meva_ticks}
        mp = os.path.join(out_dir, "meva_list.json")
        R.write_json(mp, meva_list)
        open(mp + ".sha256", "w").write(f"{sha_file(mp)}  meva_list.json\n")
        meva_list["sha256"] = sha_file(mp)
        log(f"MEVA list frozen: {len(meva_rows)} clips ({why}) sha256 {meva_list['sha256']}")
        for h, fh in heads.items():
            for r in meva_rows:
                scores[h][r["video_id"]] = fh.score(source, r["video_id"])
            log(h, "scored MEVA clips")

    thr = {h: {k: v["theta"] for k, v in thresholds[h].items()} for h in heads}
    ev_set = cctv[cctv["label_reliable"] & ~cctv["overlap_main"]].copy()
    ev_ids = list(ev_set["video_id"])
    bvids = build_external(ev_set, meva_rows, meva_ticks)
    bench = va30.Benchmark(bvids, {v: cctv_events[v] for v in ev_ids}, None, {"external": "cctv_fights evaluation set + meva"})
    ids = list(bvids.index)
    rows, metrics = [], {}
    held = 0
    for h in heads:
        al = {}
        for v in ids:
            T = bench.labels(v).T
            al[v], k = align(scores[h][v], T)
            held += k
        ts = va30.TickSet(bench, ids, al)
        W = boot_weights(ts, N_BOOT, 0)
        metrics[h] = head_metrics(h, ts, W, thr[h])
        log(h, "verdict", metrics[h]["verdict"])
    for h in heads:
        for v, s in scores[h].items():
            rows.append(pd.DataFrame({"video_id": v, "t_sec": np.arange(1, len(s) + 1, dtype=np.float32), "head": h, "score": s}))
    pd.concat(rows, ignore_index=True).to_parquet(os.path.join(out_dir, "scores.parquet"), index=False)
    res = {"config_sha256": CONFIG_SHA, "check": checks, "thresholds": {h: thresholds[h] for h in heads},
           "meva": {k: meva_list[k] for k in ("n_clips", "hours", "sha256", "wait_outcome")},
           "alignment": {"ticks_held_to_label_length": held, "cut_stream_incomplete_videos": {h: len(set(heads[h].cut_incomplete)) for h in heads}},
           "metrics": metrics, "decisions": DECISIONS}
    R.write_json(os.path.join(out_dir, "metrics.json"), res)
    write_report(os.path.join(out_dir, "report.md"), res)
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
