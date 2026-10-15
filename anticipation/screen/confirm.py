"""One-shot fold-1 confirmation of the frozen fusion model (docs/FUSION.md section 5).

    nice -n 10 .venv/bin/python -m anticipation.screen.confirm --confirm_once --config_sha256 <sha256>

Refuses unless the committed confirmation_config.json has the given sha256, --confirm_once is passed and
benchmark/results/confirmation/ does not exist. Fold 1 is read nowhere else in the repository.
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
from anticipation.data.folds import Folds
from anticipation.eval import va30
from anticipation.screen import features as F
from anticipation.screen import fusion as FU
from anticipation.screen import metrics as M
from anticipation.screen import probes as P
from anticipation.screen import run as R
from anticipation.screen import sets as S

default_prereg = FU.default_prereg
CONFIG_DEFAULT = os.path.join(FU.DEFAULT_OUT, "confirmation_config.json")
OUT_DEFAULT = os.path.join(REPO_ROOT, "benchmark", "results", "confirmation")
FINAL_RNG_SALT = 9000
REQUIRE_CUT = True


class ConfirmError(RuntimeError):
    """The confirmation preconditions are not met; nothing was read."""


def log(*a):
    print("[confirm]", *a, flush=True)


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def committed_bytes(path):
    """The bytes of ``path`` at git HEAD, or None when it is untracked there."""
    rel = os.path.relpath(os.path.abspath(path), REPO_ROOT)
    r = subprocess.run(["git", "-C", REPO_ROOT, "show", f"HEAD:{rel}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def read_config(path, expected_sha, require_committed=True):
    """The spec, after checking its sha256 against ``expected_sha``, its sidecar and (by default) git HEAD."""
    if not expected_sha:
        raise ConfirmError("no expected sha256 given")
    if not os.path.isfile(path):
        raise ConfirmError(f"no config at {path}")
    raw = open(path, "rb").read()
    got = sha_bytes(raw)
    if got != expected_sha.strip().lower():
        raise ConfirmError(f"config sha256 {got} differs from the expected {expected_sha}")
    side = path + ".sha256"
    if not os.path.isfile(side) or open(side).read().split()[0] != got:
        raise ConfirmError(f"missing or mismatching sidecar {side}")
    if require_committed:
        head = committed_bytes(path)
        if head is None or sha_bytes(head) != got:
            raise ConfirmError("the config is not committed at git HEAD with this exact content")
    return json.loads(raw)


class ConfirmSet(S.ScreenSet):
    """A screening set that can also read the frozen fold; built only inside a gated confirmation run."""

    def __init__(self, base):
        self.__dict__.update(base.__dict__)
        h1 = base.h1.reset_index(drop=True) if len(base.h1) else None
        self.fold1 = S.ScreenSet(base.bench, h1, screen_folds=(base.frozen_fold,), frozen_fold=-1)

    def guard(self, vid):
        return None

    def guard_fold(self, f):
        return None

    def ticks(self, vid):
        if self.fold_of(vid) == self.frozen_fold:
            return self.fold1.ticks(vid)
        return super().ticks(vid)

    def confirmation_test_ids(self):
        return self.fold1.test_ids(self.frozen_fold, kinds=("violent", "clean"))


class ConfirmFolds(Folds):
    """One pseudo-fold: the fold-1 test videos, with the calibration slice as its validation slice."""

    def __init__(self, test_ids, val_ids):
        d = {"name": "fold1_confirmation", "n_folds": 1, "assignment": {v: 0 for v in test_ids},
             "val": {"0": list(val_ids)}, "content_sha256": "confirmation"}
        super().__init__(d)
        self._all = sorted(set(test_ids) | set(val_ids))

    @property
    def video_ids(self):
        return self._all


def check_blocks(hs):
    for b in hs["blocks"]:
        reg = F.BLOCKS.get(b["name"])
        if reg is None or list(reg.streams) != b["streams"] or list(reg.include) != b["include"] \
                or list(reg.exclude) != b["exclude"]:
            raise ConfirmError(f"block {b['name']} differs from the frozen spec")


def p1_rows(runners, names, plan, vt, nuis, src):
    X, y, w = {n: [] for n in names}, [], []
    nr = []
    for v, (ti, yy, ww) in plan.by_video.items():
        for n in names:
            X[n].append(runners[n].feats(v, "p1")[:vt[v].n][ti])
        if nuis is not None:
            nr.append(nuis.features(vt[v], src(v))[ti])
        y.append(yy)
        w.append(ww)
    return {n: np.concatenate(a) for n, a in X.items()}, np.concatenate(y), np.concatenate(w), (np.concatenate(nr) if nr else None)


def final_p2(cset, runners, names, vt, train, test, hs, target):
    p2 = hs["p2"]
    z, dims = FU.gru_inputs(runners, names, vt, train, train + test)
    rng = np.random.default_rng(hs["seed"] + 7000)
    plan = P.Plan([vt[v] for v in train], target, rng, p2["max_train_pos"], p2["max_train_neg"], cset.source_kind)
    rows = FU.gru_rows(plan, vt)
    sub = rng.choice(len(rows), min(2000, len(rows)), replace=False)
    rows_va = [rows[i] for i in sub]
    bags, _ = FU.make_bags(cset, train, vt, rng, window=(120, 30) if target == "band" else (10, 0))
    scores = {}
    for seed in p2["seeds"]:
        model, _ = FU.train_gru(z, dims, rows, rows_va, bags, seed, epochs=int(p2["fixed_epochs"]), fixed_epochs=True)
        for v in test:
            scores.setdefault(v, []).append(FU.score_video(model, z[v], model.context))
        log(f"P2 seed {seed} trained")
    return {v: np.mean(a, axis=0) for v, a in scores.items()}


def calibrated_frame(cset, oof_path, test_scores, vt):
    """Score table of the fold-1 test videos plus the out-of-fold scores of the fold-1 validation slice, one pseudo-fold."""
    bench0 = cset.bench
    val_ids = {v for v in bench0.folds.val_ids(cset.frozen_fold) if v in bench0.videos.index}
    oof = pd.read_parquet(oof_path)
    cal = oof[(oof["split"] == "test") & oof["video_id"].isin(val_ids)]
    rows = [pd.DataFrame({"video_id": v, "t_sec": vt[v].t[:len(s)], "score": s[:vt[v].n], "fold": 0, "seed": 0})
            for v, s in test_scores.items()]
    df = pd.concat(rows + [cal[["video_id", "t_sec", "score"]].assign(fold=0, seed=0)], ignore_index=True)
    folds = ConfirmFolds(sorted(test_scores), sorted(set(cal["video_id"])))
    return df, folds, len(set(cal["video_id"]))


def fold1_events(cset, spec, oof_path, test_scores, vt, out_dir, imminent):
    """Event metrics on fold 1 with thresholds from out-of-fold scores of the folds_v2 validation slice of fold 1."""
    df, folds, n_cal = calibrated_frame(cset, oof_path, test_scores, vt)
    os.makedirs(out_dir, exist_ok=True)
    em = spec["event_metrics"]
    bench = va30.Benchmark(cset.bench.videos, cset.bench.events, folds, {"folds_path": "fold1_confirmation"},
                           video_horizons=os.path.join(REPO_ROOT, em["horizons"]))
    if imminent:
        res = FU.lead_report(bench, df, 10.0, tuple(em["fa_targets"]), em["n_boot"], em["threshold_bootstrap"], em["cluster"])
        res["calibration_videos"] = n_cal
        return res
    cfg = va30.EvalConfig(n_boot=em["n_boot"], cluster=em["cluster"], fa_targets=tuple(float(x) for x in em["fa_targets"]),
                          restrict_to_scored=True, allow_missing=True, threshold_bootstrap=em["threshold_bootstrap"])
    res = va30.evaluate(df, bench, cfg, name="fold1_confirmation", extra_meta={"calibration_videos": n_cal})
    va30.save_results(res, os.path.join(out_dir, "va30"))
    return res


def verdict_early(hs, stats, gain):
    rule = hs["confirmation_rule"]
    lower = stats["m1"].lower
    ok1 = lower is not None and lower > rule["m1_band_ci_lower_gt"]
    ok2 = gain.value is not None and gain.value > rule["m8_gain_over_n_point_gt"]
    return {"m1_lower": lower, "m1_lower_gt_0.5": bool(ok1), "m8_gain_point": gain.value, "m8_gain_point_gt_0": bool(ok2),
            "m8_gain_ci": gain.ci, "confirmed": bool(ok1 and ok2), "rule": rule["text"]}


def verdict_imminent(hs, stats, events):
    rule = hs["confirmation_rule"]
    lower = stats["m1"].lower
    ok1 = lower is not None and lower > rule["m1_near_ci_lower_gt"]
    op = (events or {}).get("operating_points", {}).get("fa2", {})
    ci = (op.get("threshold_aware_ci") or {}).get("success_ci")
    pn = op.get("position_null")
    ok2 = bool(ci is not None and pn is not None and ci[0] > pn)
    return {"m1_near_lower": lower, "m1_near_lower_gt_0.6": bool(ok1), "success10_fa2": (op.get("success") or {}).get("value"),
            "success10_threshold_aware_ci": ci, "position_null_at_test_fa": pn, "success10_ci_lower_gt_position_null": ok2,
            "confirmed": bool(ok1 and ok2), "rule": rule["text"]}


def verdict(spec, stats, n_stats, gain):
    """Early-head rule applied to precomputed statistics (kept for the unit tests)."""
    return verdict_early({"confirmation_rule": spec["confirmation_rule"]}, stats, gain)


def run_head(head, hs, spec, cset, source, ss, out_dir, oof_path, require_cut, events):
    t0 = time.time()
    check_blocks(hs)
    imminent = head == "imminent"
    target = "near" if imminent else "band"
    names = [b["name"] for b in hs["blocks"]]
    runners = {n: P.BlockRun(cset, source, F.BLOCKS[n], seed=hs["seed"], require_cut=require_cut) for n in names}
    cov = lambda v: all(r.covered(v) for r in runners.values())  # noqa: E731
    train = [v for v in FU.training_list(ss) if cov(v)]
    all_test = cset.confirmation_test_ids()
    test = [v for v in all_test if cov(v)]
    if not test:
        raise ConfirmError(f"{head}: no fold-1 test video has all frozen feature streams")
    log(f"{head}: train {len(train)} videos, fold-1 test {len(test)} of {len(all_test)}")
    for r in runners.values():
        r.set_fold(train)
    vt = {v: cset.ticks(v).cut(min(r.vt(v).n for r in runners.values())) for v in train + test}
    p1 = hs["p1"]
    rng = np.random.default_rng(hs["seed"] + FINAL_RNG_SALT)
    plan = P.Plan([vt[v] for v in train], target, rng, p1["max_train_pos"], p1["max_train_neg"], cset.source_kind)
    if plan.empty:
        raise ConfirmError(f"{head}: no {target} training rows")
    nuis = P.Nuisance(source, train)
    X, y, w, Xn = p1_rows(runners, names, plan, vt, nuis, cset.source_kind)
    feats = {v: [runners[n].feats(v, "p1")[:vt[v].n] for n in names] for v in test}
    nfe = {v: nuis.features(vt[v], cset.source_kind(v)) for v in test}
    probe_n = FU.fit_fixed(Xn, y, w, hs["nuisance"]["C"])
    probe_np = FU.fit_fixed(np.concatenate([Xn] + [X[n] for n in names], axis=1), y, w, p1["C"])
    s_n = {v: probe_n.score(nfe[v]) for v in test}
    s_np = {v: probe_np.score(np.concatenate([nfe[v]] + feats[v], axis=1)) for v in test}
    if hs["probe_type"] == "p2":
        s_model = final_p2(cset, runners, names, vt, train, test, hs, target)
    else:
        probe = FU.fit_fixed(np.concatenate([X[n] for n in names], axis=1), y, w, p1["C"])
        s_model = {v: probe.score(np.concatenate(feats[v], axis=1)) for v in test}
    pool = M.Pool(cset.fold1, [vt[v] for v in test])
    W = pool.boot_weights(spec["bootstrap"]["n"], seed=spec["bootstrap"]["seed"])
    if imminent:
        evp = FU.near_event_pool(pool)
        st = lambda sc, extra=True: FU.pool_stats_near(pool, evp, W, pool.assemble(sc), extra)  # noqa: E731
    else:
        st = lambda sc, extra=True: FU.pool_stats(pool, W, pool.assemble(sc), extra)  # noqa: E731
    stats, n_stats, np_stats = st(s_model), st(s_n, False), st(s_np, False)
    gain = M.diff(stats["m8"], n_stats["m8"])
    res = {"head": head, "probe": hs["probe_type"], "blocks": names,
           "coverage": {"fold1_test_videos": len(all_test), "scored": len(test), "uncovered": sorted(set(all_test) - set(test)),
                        "train_videos": len(train), "events": int(len(pool.events))},
           "model": FU.stat_row(stats), "nuisance": FU.stat_row(n_stats), "n_plus_model_m1": FU.sd(np_stats["m1"]),
           "m4n": FU.sd(M.diff(np_stats["m1"], n_stats["m1"]), 0.0), "m8_gain_over_n": FU.sd(gain, 0.0)}
    ev = None
    if events:
        em = fold1_events(cset, spec, oof_path, s_model, vt, os.path.join(out_dir, head), imminent)
        if imminent:
            ev = R.jsonable(em)
        else:
            ev = {"operating_points": R.jsonable(FU.event_summary(em)), "per_source": R.jsonable(FU.per_source(em)),
                  "counts": R.jsonable(em["seeds"][em["headline_seed"]]["counts"])}
        res["events"] = ev
    res["verdict"] = verdict_imminent(hs, stats, ev) if imminent else verdict_early(hs, stats, gain)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def run_confirmation(spec, ss, out_dir, feature_root=None, require_cut=True, events=True, spec_sha=None, config_dir=None):
    """Train both frozen heads on the screening data, score fold 1 once and apply the frozen rules of amendment 05."""
    if FU.training_list_sha(ss) != spec["training"]["videos_sha256"]:
        raise ConfirmError("training membership differs from the frozen spec")
    config_dir = config_dir or os.path.dirname(os.path.abspath(CONFIG_DEFAULT))
    oofs = {}
    for h, hs in spec["heads"].items():
        if hs is None:
            continue
        path = os.path.join(config_dir, hs["calibration"]["scores_file"])
        if FU.file_sha(path) != hs["calibration"]["scores_sha256"]:
            raise ConfirmError(f"{h}: calibration scores differ from the frozen spec")
        oofs[h] = path
    cset = ConfirmSet(ss)
    source = F.FeatureSource(cset, root=feature_root)
    res = {"spec_sha256": spec_sha, "heads": {}}
    for h, hs in spec["heads"].items():
        if hs is None:
            res["heads"][h] = {"head": h, "verdict": {"confirmed": False, "rule": "no head was frozen"}}
            continue
        res["heads"][h] = run_head(h, hs, spec, cset, source, ss, out_dir, oofs[h], require_cut, events)
    res["confirmed"] = {h: r["verdict"]["confirmed"] for h, r in res["heads"].items()}
    R.write_json(os.path.join(out_dir, "confirmation.json"), res)
    write_report(res, out_dir)
    return res


def write_report(res, out_dir):
    f3 = lambda d: "n/a" if not d or d.get("value") is None else f"{d['value']:.3f}" + (f" [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]" if d.get("ci") and d["ci"][0] is not None else "")  # noqa: E731
    L = ["# Fold-1 confirmation (single evaluation, both heads)", ""]
    for h, r in res["heads"].items():
        v = r["verdict"]
        L += [f"## {h} head", ""]
        if "model" not in r:
            L += [v["rule"], f"**Confirmed: {v['confirmed']}**", ""]
            continue
        imminent = h == "imminent"
        L += [f"Model: {r['probe']} on {r['blocks']}; fold-1 test videos scored {r['coverage']['scored']} of "
              f"{r['coverage']['fold1_test_videos']}; events {r['coverage']['events']}", ""]
        c2 = "M3p-near" if imminent else "M1 30-60"
        k2 = "m3p" if imminent else "m1_30_60"
        L += [f"| model | {'M1-near' if imminent else 'M1 band'} | {c2} | {'M8-near' if imminent else 'M8'} |", "|---|---|---|---|",
              f"| frozen model | {f3(r['model']['m1'])} | {f3(r['model'].get(k2))} | {f3(r['model']['m8'])} |",
              f"| nuisance N | {f3(r['nuisance']['m1'])} | {f3(r['nuisance'].get(k2))} | {f3(r['nuisance']['m8'])} |", "",
              f"M8 gain over N: {f3(r['m8_gain_over_n'])}; M4n: {f3(r['m4n'])}", "", f"Rule: {v['rule']}"]
        for k, val in v.items():
            if k not in ("rule",):
                L.append(f"- {k}: {val}")
        L.append("")
        ev = r.get("events")
        if ev and imminent:
            L += ["| target FA/h | test FA/h | Success@10 | threshold-aware CI | position null |", "|---|---|---|---|---|"]
            for o in ev["operating_points"].values():
                ci = (o.get("threshold_aware_ci") or {}).get("success_ci")
                pn = o.get("position_null")
                L.append(f"| {o['target_fa_per_hour']:g} | {o['test_fa_per_hour_clean']:.2f} | {f3(o['success'])} | "
                         f"{'n/a' if not ci else f'[{ci[0]:.3f}, {ci[1]:.3f}]'} | {'n/a' if pn is None else f'{pn:.3f}'} |")
        elif ev:
            L += ["| target FA/h | test FA/h | Success@30 | position null | random |", "|---|---|---|---|---|"]
            for o in ev["operating_points"].values():
                pn = o.get("position_null_success30")
                pn_v = pn.get("value") if isinstance(pn, dict) else pn
                L.append(f"| {o['target_fa_per_hour']:g} | {o['test_fa_per_hour_clean']:.2f} | {f3(o['success30'])} | "
                         f"{'n/a' if pn_v is None else f'{pn_v:.3f}'} | {f3(o['random_success30'])} |")
        L.append("")
    L += ["Confirmed: " + ", ".join(f"{h} {c}" for h, c in res["confirmed"].items()), ""]
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirm_once", action="store_true", help="required: acknowledges that fold 1 is read exactly once")
    ap.add_argument("--config", default=CONFIG_DEFAULT)
    ap.add_argument("--config_sha256", required=True)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--prereg", default=default_prereg())
    ap.add_argument("--feature_root", default=None)
    args = ap.parse_args(argv)
    if not args.confirm_once:
        raise SystemExit("refusing to read fold 1 without --confirm_once")
    out = os.path.abspath(args.out)
    if os.path.exists(out):
        raise SystemExit(f"refusing: {out} exists; the confirmation is a one-shot evaluation")
    try:
        spec = read_config(args.config, args.config_sha256)
    except ConfirmError as e:
        raise SystemExit(f"refusing: {e}")
    va30.N_THREADS = 4
    pr, ss, _ = R.prepare(args)
    if spec.get("version") != 2:
        raise SystemExit("refusing: only the two-head (version 2) spec is supported")
    if spec["prereg"]["sha256"] != pr.sha256:
        raise SystemExit("refusing: the frozen spec was made under another pre-registration")
    os.makedirs(out)
    with open(os.path.join(out, "STARTED"), "w") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} config {args.config_sha256}\n")
    with threadpool_limits(limits=4):
        res = run_confirmation(spec, ss, out, args.feature_root, require_cut=REQUIRE_CUT, spec_sha=args.config_sha256,
                               config_dir=os.path.dirname(os.path.abspath(args.config)))
    log("confirmed:", res["confirmed"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
