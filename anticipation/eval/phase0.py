"""Opt-in Phase 0 diagnostics with frozen VA30 labels and split-group uncertainty."""

import json
from copy import copy
from pathlib import Path

import numpy as np

from anticipation.eval.va30 import (PairAUC, TIME_EDGES, TIME_MATCH_EDGES, TickSet, alarm_starts,
                                    ci, time_match_weights, _concave_hull, file_sha256)


def group_weights(ts, n_boot, seed):
    rows = ts.bench.videos.loc[ts.vids]
    key = "split_group" if "split_group" in rows else "group"
    groups, inverse = np.unique(rows[key].astype(str), return_inverse=True)
    if not len(groups) or n_boot <= 0:
        return np.zeros((0, ts.V))
    weights = np.random.default_rng(seed).multinomial(len(groups), np.full(len(groups), 1 / len(groups)), n_boot)
    return weights[:, inverse].astype(float)


def band_stat(ts, W, lo=30, hi=120, keep=None, strata=None):
    valid = ~ts.masked if keep is None else ~ts.masked & keep
    pos = valid & ts.pre & (ts.tte > lo) & (ts.tte <= hi)
    neg = valid & ts.clean & (ts.remaining >= 60)
    source = ts.source_v[ts.vidx] if strata is None else strata
    weights, matched = time_match_weights(ts.t, pos, neg, TIME_MATCH_EDGES, source=source)
    out = PairAUC(ts.score, ts.vidx, matched, neg, ts.V, neg_w=weights).stat(W)
    out.update(n_positive_before_matching=int(pos.sum()), n_positive_dropped=int((pos & ~matched).sum()),
               interval_sec=[lo, hi], matching="source kind x TIME_MATCH_EDGES; additional strata where stated")
    return out


def person_counts(ts, root):
    counts = np.full(len(ts.t), np.nan)
    files = 0
    manifest = []
    frozen = None
    root = Path(root)
    if root.is_file():
        frozen = json.loads(root.read_text())
        root = Path(frozen["root"])
    for i, vid in enumerate(ts.vids):
        if frozen is not None and vid not in frozen["files"]:
            continue
        path = root / (vid + ".npz")
        if not path.exists():
            continue
        digest = file_sha256(path)
        if frozen is not None and digest != frozen["files"][vid]:
            raise ValueError(f"Pose file changed after snapshot: {path}")
        with np.load(path) as z:
            meta = json.loads(str(z["meta"].item()))
            columns = meta.get("columns", [])
            if "n_persons_last" not in columns:
                continue
            times, feat, valid = z["t_sec"], z["feat"], np.asarray(z["valid"], bool)
            values = np.asarray(feat[:, columns.index("n_persons_last")], float)
        if times.ndim != 1 or valid.shape != times.shape or len(np.unique(times)) != len(times):
            raise ValueError(f"Malformed pose timestamp/valid mask: {path}")
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError(f"Unsorted/nonfinite pose timestamps: {path}")
        start, end = ts.offsets[i:i + 2]
        ticks = ts.t[start:end]
        k = np.searchsorted(times, ticks)
        good = k < len(times)
        clipped = np.clip(k, 0, max(0, len(times) - 1))
        if not len(times):
            continue
        good &= times[clipped] == ticks
        good &= valid[clipped] & np.isfinite(values[clipped]) & (values[clipped] >= 0)
        counts[start:end][good] = np.rint(values[clipped[good]])
        files += 1
        manifest.append(f"{vid} {digest}")
    import hashlib
    return counts, {"input_sha256": hashlib.sha256("\n".join(manifest).encode()).hexdigest(), "pose_files": files, "valid_ticks": int(np.isfinite(counts).sum()),
                    "missing_or_invalid_ticks": int((~np.isfinite(counts)).sum()),
                    "column": "n_persons_last", "alignment": "exact timestamp only; missing is not zero people"}


def window_max_auc(ee, W, first_only=False):
    ts = ee.ts
    correct, pairs = np.zeros((ts.V, ts.V)), np.zeros((ts.V, ts.V))
    matched, unmatched = 0, 0
    windows = []
    for j in np.flatnonzero(ee.ev_e30 & (ee.ev_first if first_only else True)):
        i = ee.ev_v[j]
        lo = max(1, int(np.ceil(ee.ev_lo[j] - 1e-9)))
        hi = int(np.floor(ee.ev_on[j] - 30 + 1e-9))
        a, b = ts.offsets[i:i + 2]
        keep = (ts.t[a:b] >= lo) & (ts.t[a:b] <= hi) & ~ts.masked[a:b]
        if not keep.any():
            unmatched += 1
            continue
        positive = float(ts.score[a:b][keep].max())
        negatives = []
        for k in np.flatnonzero(ts.clean_v & (ts.source_v == ts.source_v[i])):
            ca, cb = ts.offsets[k:k + 2]
            select = (ts.t[ca:cb] >= lo) & (ts.t[ca:cb] <= hi)
            if select.sum() != hi - lo + 1 or ts.masked[ca:cb][select].any():
                continue
            negatives.append((k, float(ts.score[ca:cb][select].max())))
        if not negatives:
            unmatched += 1
            continue
        for k, negative in negatives:
            correct[i, k] += (float(positive > negative) + .5 * float(positive == negative)) / len(negatives)
            pairs[i, k] += 1 / len(negatives)
        matched += 1
        windows.append({"video_id": ts.vids[i], "event_index": int(ee.ev_index[j]), "start": lo,
                        "end": hi, "max_score": positive, "matched_clean_windows": len(negatives)})
    numerator, denominator = correct.sum(), pairs.sum()
    boot_n = np.sum((W @ correct) * W, axis=1)
    boot_d = np.sum((W @ pairs) * W, axis=1)
    out = ci(np.divide(boot_n, boot_d, out=np.full(len(W), np.nan), where=boot_d > 0),
             numerator / denominator if denominator else None)
    out.update(events_matched=matched, events_unmatched=unmatched, windows=windows,
               definition="Per-event max in [max(onset-120,clear_start,1),onset-30], versus same-source clean windows at identical clip times and length. Equal original-event weight; matched conditional Mann-Whitney AUROC. Masked/incomplete clean windows excluded.")
    return out


def warning_episodes(ts, theta_v, policy, reset=None):
    _, _, active = alarm_starts(ts.score, ts.t, ts.vidx, ts.offsets, theta_v[ts.vidx], policy, reset)
    warning = active.copy()
    episodes = []
    for i in range(ts.V):
        a, b = ts.offsets[i:i + 2]
        ticks = np.flatnonzero(active[a:b]) + a
        if not len(ticks):
            continue
        boundaries = np.flatnonzero(np.diff(ticks) > 1) + 1
        runs = np.split(ticks, boundaries)
        start, end = int(runs[0][0]), int(runs[0][-1])
        for run in runs[1:]:
            lo, hi = int(run[0]), int(run[-1])
            blocked = reset is not None and reset[end + 1:lo + 1].any()
            if ts.t[lo] - ts.t[end] - 1 < policy.merge_gap_sec and not blocked:
                warning[end:lo + 1] = True
                end = hi
            else:
                episodes.append((i, ts.t[start], ts.t[end]))
                start, end = lo, hi
        episodes.append((i, ts.t[start], ts.t[end]))
    return np.asarray(episodes, float).reshape(-1, 3), warning


def sustained_outcomes(ee, theta_v):
    episodes, active = warning_episodes(ee.ts, theta_v, ee.p, ee.reset)
    hits = {}
    for lead in (60, 90, 120):
        eligible = ee.ev_maxlead >= lead
        hit = np.zeros(len(ee.ev_v), bool)
        start_time = np.full(len(ee.ev_v), np.nan)
        for j in np.flatnonzero(eligible):
            lower = max(ee.ev_clear[j], 2.0)
            valid = ((episodes[:, 0] == ee.ev_v[j]) & (episodes[:, 1] >= lower)
                     & (episodes[:, 1] <= ee.ev_on[j] - lead + 1e-9)
                     & (episodes[:, 2] >= np.floor(ee.ev_on[j] - 30 + 1e-9)))
            if valid.any():
                hit[j] = True
                start_time[j] = episodes[valid, 1].min()
        hits[lead] = (eligible, hit, start_time)
    valid_clean = ee.ts.clean & ~ee.ts.masked
    active_v = np.bincount(ee.ts.vidx[active & valid_clean], minlength=ee.ts.V).astype(float)
    total_v = np.bincount(ee.ts.vidx[valid_clean], minlength=ee.ts.V).astype(float)
    return hits, active_v, total_v


def fraction_stat(numerator, denominator, W):
    a, b = W @ numerator, W @ denominator
    return ci(np.divide(a, b, out=np.full(len(W), np.nan), where=b > 0),
              numerator.sum() / denominator.sum() if denominator.sum() else None)


def sustained_summary(ee, theta_v, W):
    hits, active_v, total_v = sustained_outcomes(ee, theta_v)
    out = {"alarm_time_fraction_clean": fraction_stat(active_v, total_v, W)}
    for lead, (eligible, hit, starts) in hits.items():
        counts = np.bincount(ee.ev_v[eligible], minlength=ee.ts.V).astype(float)
        positive = np.bincount(ee.ev_v[hit], minlength=ee.ts.V).astype(float)
        stat = fraction_stat(positive, counts, W)
        stat["n_events"] = int(eligible.sum())
        out[f"success_sus{lead}"] = stat
        timing = []
        for j in np.flatnonzero(hit):
            vid = ee.ts.vids[ee.ev_v[j]]
            conflict = ee.ts.bench.events.get(vid, {}).get("t_conflict")
            if isinstance(conflict, list):
                index = int(ee.ev_index[j])
                conflict = conflict[index] if index < len(conflict) else None
            elif ee.ev_index[j] != 0:
                conflict = None
            if isinstance(conflict, (int,float)) and np.isfinite(conflict):
                timing.append({"video_id":vid,"event_index":int(ee.ev_index[j]),
                               "alarm_start_minus_conflict_sec":float(starts[j]-conflict)})
            elif conflict == "before_clip_start":
                timing.append({"video_id":vid,"event_index":int(ee.ev_index[j]),
                               "alarm_start_minus_conflict_lower_bound_sec":float(starts[j])})
        out[f"conflict_timing{lead}"] = timing
    return out


def time_fraction_threshold(ee, target):
    unique = np.r_[np.inf, np.unique(ee.ts.score)[::-1]]
    low, high = 0, len(unique) - 1
    clean = ee.ts.clean & ~ee.ts.masked
    def cost(index):
        _, active = warning_episodes(ee.ts, np.full(ee.ts.V, unique[index]), ee.p, ee.reset)
        return float(active[clean].mean()) if clean.any() else np.nan
    if not np.isfinite(cost(0)):
        raise ValueError("Sustained threshold selection needs valid clean ticks")
    while low < high:
        middle = (low + high + 1) // 2
        if cost(middle) <= target + 1e-12:
            low = middle
        else:
            high = middle - 1
    return float(unique[low]), float(cost(low))


def sustained_position_null(ee, max_thresholds=16):
    from anticipation.eval.va30 import EventEval
    ts = ee.ts
    src = ts.source_v[ts.vidx]
    kinds = sorted(set(ts.source_v))
    candidates = []
    for mask in range(1, 2 ** len(kinds)):
        selected = [k for j, k in enumerate(kinds) if mask >> j & 1]
        candidates.append(("source_constant", np.isin(src, selected).astype(float), [0.5]))
    for source in [None] + kinds:
        allowed = np.ones(len(ts.t), bool) if source is None else src == source
        for end in (30, 60, 90, 120, 180, 300, 600):
            candidates.append(("source_prefix", (allowed & (ts.t <= end)).astype(float), [.5]))
            candidates.append(("source_suffix", (allowed & (ts.t >= end)).astype(float), [.5]))
    target = np.zeros(len(ts.t), bool)
    for j in np.flatnonzero(ee.ev_maxlead >= 60):
        target |= ((ts.vidx == ee.ev_v[j]) & (ts.t >= max(ee.ev_clear[j], 2))
                   & (ts.t <= ee.ev_on[j] - 30))
    for edges in (TIME_EDGES, TIME_MATCH_EDGES):
        bins = np.searchsorted(edges[1:-1], ts.t, side="right")
        for by_source in (False, True):
            keys = np.array([f"{s}:{t}" for s, t in zip(src if by_source else np.full(len(src), "all"), bins)])
            _, inverse = np.unique(keys, return_inverse=True)
            rate = np.bincount(inverse, weights=target) / np.maximum(1, np.bincount(inverse))
            score = rate[inverse]
            thresholds = np.unique(np.quantile(score, np.linspace(0, 1, max_thresholds)))
            candidates.append(("fitted_source_time" if by_source else "fitted_time", score, thresholds))
    points = {lead: [] for lead in (60, 90, 120)}
    empty_W = np.zeros((0, ts.V))
    for family, score, thresholds in candidates:
        evaluator = copy(ee)
        evaluator.ts = copy(ts)
        evaluator.ts.score = score
        for threshold in thresholds:
            stats = sustained_summary(evaluator, np.full(ts.V, threshold), empty_W)
            fraction = stats["alarm_time_fraction_clean"]["value"]
            if fraction is None:
                continue
            for lead in points:
                success = stats[f"success_sus{lead}"]["value"]
                if success is not None:
                    points[lead].append((fraction, success, family))
    return {str(lead): _concave_hull(values) for lead, values in points.items()}


def partial_amoc(ee, W, thresholds):
    counts = np.bincount(ee.ev_v[ee.ev_e30], minlength=ee.ts.V).astype(float)
    hours = ee.hours_out * ee.ts.clean_v
    x, y = [], []
    for threshold in thresholds:
        outcome = ee.outcomes(np.full(ee.ts.V, threshold))
        hit = ee.ev_e30 & (np.nan_to_num(outcome["lead"], nan=-1) >= 30)
        positives = np.bincount(ee.ev_v[hit], minlength=ee.ts.V).astype(float)
        all_w = np.vstack([np.ones(ee.ts.V), W])
        denominators = all_w @ hours
        event_counts = all_w @ counts
        x.append(np.divide(all_w @ (outcome["fa"] * ee.ts.clean_v), denominators,
                           out=np.full(len(all_w), np.nan), where=denominators > 0))
        y.append(np.divide(all_w @ positives, event_counts, out=np.full(len(all_w), np.nan), where=event_counts > 0))
    areas = []
    for xx, yy in zip(np.array(x).T, np.array(y).T):
        if not np.isfinite(xx).any() or not np.isfinite(yy).any():
            areas.append(np.nan)
            continue
        hull = _concave_hull([(a, b, "threshold") for a, b in zip(xx, yy) if np.isfinite(a+b)])
        h = np.array([[a, b] for a, b, _ in hull])
        grid = np.unique(np.r_[.5, h[(h[:, 0] > .5) & (h[:, 0] < 2), 0], 2])
        values = np.interp(grid, h[:, 0], h[:, 1])
        areas.append(float(np.trapezoid(values, grid) / 1.5))
    result = ci(areas[1:], areas[0])
    result["definition"] = "Normalized area over FA/h 0.5..2 under the upper concave hull of the descriptive global-threshold AMOC; threshold grid held fixed in paired split-group resamples."
    return result


def sustained_bootstrap(ee, val_ee, cfg, seed, max_grid=300):
    n = cfg.threshold_bootstrap
    result = {target: {"fraction": [], **{lead: [] for lead in (60,90,120)}} for target in (.01,.02,.05)}
    W = group_weights(ee.ts, n, cfg.boot_seed + 7919*seed)
    choices = {}
    for fold, evaluator in val_ee.items():
        ts = evaluator.ts
        unique = np.unique(ts.score)[::-1]
        if len(unique) > max_grid:
            unique = unique[np.unique(np.linspace(0,len(unique)-1,max_grid).astype(int))]
        grid = np.r_[np.inf, unique]
        clean = ts.clean & ~ts.masked
        totals = np.bincount(ts.vidx[clean], minlength=ts.V)
        amounts = []
        for theta in grid:
            _, active = warning_episodes(ts,np.full(ts.V,theta),evaluator.p,evaluator.reset)
            amounts.append(np.bincount(ts.vidx[active & clean],minlength=ts.V))
        Wv = group_weights(ts,n,cfg.boot_seed+7919*seed+104729*(fold+1))
        denom = Wv@totals
        costs = np.divide(Wv@np.asarray(amounts).T,denom[:,None],
                          out=np.full((n,len(grid)),np.nan),where=denom[:,None]>0)
        choices[fold] = {target: np.where(denom>0,grid[np.maximum(0,(costs<=target+1e-12).sum(axis=1)-1)],np.nan)
                         for target in result}
    for target, values in result.items():
        for r in range(n):
            thresholds = np.array([choices[f][target][r] for f in ee.ts.fold_v])
            if np.isnan(thresholds).any():
                continue
            hits, active, total = sustained_outcomes(ee,thresholds)
            denominator = W[r]@total
            if denominator<=0:
                continue
            values["fraction"].append(float((W[r]@active)/denominator))
            for lead,(eligible,hit,_) in hits.items():
                counts=np.bincount(ee.ev_v[eligible],minlength=ee.ts.V)
                positive=np.bincount(ee.ev_v[hit],minlength=ee.ts.V)
                denominator=W[r]@counts
                if denominator>0:
                    values[lead].append(float((W[r]@positive)/denominator))
    return {target: {"n_boot":n,"max_candidate_thresholds":max_grid+1,
                    "note":"Validation and test split_groups resampled jointly; warning thresholds reselected on a capped candidate grid. Replicates without clean validation exposure are omitted.",
                    "alarm_time_fraction_clean_ci":ci(values["fraction"],None)["ci"],
                    **{f"success_sus{lead}_ci":ci(values[lead],None)["ci"] for lead in (60,90,120)}}
            for target,values in result.items()}


def compute_phase0(ts, ee, val_ee, cfg, seed, legacy):
    W = group_weights(ts, cfg.n_boot, cfg.boot_seed + 7919 * seed)
    long_v = np.array([any(e["index"] == 0 and e["max_lead"] >= 60 for e in rows) for rows in ts.events])
    out = {"ci_cluster": "split_group (fallback catalog group)",
           "band_auroc": band_stat(ts, W), "band_long_auroc": band_stat(ts, W, 60, 120, keep=ts.video_mask(long_v)),
           "band_near_auroc": band_stat(ts, W, 0, 10),
           "window_max_auroc": window_max_auc(ee, W),
           "window_max_first_auroc": window_max_auc(ee, W, True),
           "partial_amoc_0p5_2": partial_amoc(ee, W, [r["theta"] for r in legacy["amoc_curve"]])}
    out["breakdowns"] = {}
    for source, selected in (("real_world", ts.source_v != "xd_movie"), ("movie", ts.source_v == "xd_movie")):
        out["breakdowns"][source] = band_stat(ts, W, keep=ts.video_mask(selected))
    if cfg.person_counts_root:
        counts, coverage = person_counts(ts, cfg.person_counts_root)
        bins = np.select([counts == 0, counts == 1, counts == 2, (counts >= 3) & (counts <= 5), counts >= 6],
                         ["0", "1", "2", "3-5", "6+"], default="missing")
        strata = np.array([f"{s}:{b}" for s, b in zip(ts.source_v[ts.vidx], bins)])
        coverage["events_kept_per_source"] = {}
        eligible_first = {int(v) for v in ee.ev_v[ee.ev_e30 & ee.ev_first]}
        for source in sorted(set(ts.source_v)):
            keep = (~ts.masked & ts.pre & (ts.tte>30) & (ts.tte<=120)
                    & (counts>=2) & (ts.source_v[ts.vidx]==source))
            coverage["events_kept_per_source"][str(source)] = len(set(ts.vidx[keep]) & eligible_first)
        out["person_count"] = {"coverage": coverage,
             "stratified_band_auroc": band_stat(ts, W, keep=np.isfinite(counts), strata=strata),
             "at_least_two_band_auroc": band_stat(ts, W, keep=np.isfinite(counts) & (counts >= 2), strata=strata)}
    else:
        out["person_count"] = {"status": "not requested; set person_counts_root to pose_interact"}
    null = sustained_position_null(ee) if cfg.position_null else {}
    sustained = {"definition": "Merged 2-of-3 warning episodes; gaps shorter than merge_gap_sec count as active warning time. No count-based refractory suppression. Start>=max(clear_start,2), start<=onset-L, episode covers tick floor(onset-30).",
                 "position_null_curve": null, "operating_points": {}}
    threshold_ci = sustained_bootstrap(ee,val_ee,cfg,seed) if cfg.threshold_bootstrap>0 else {}
    for target in (.01, .02, .05):
        thresholds, validation = {}, {}
        for fold, evaluator in val_ee.items():
            thresholds[fold], validation[fold] = time_fraction_threshold(evaluator, target)
        theta = np.array([thresholds[f] for f in ts.fold_v])
        stats = sustained_summary(ee, theta, W)
        if target in threshold_ci:
            stats["threshold_aware_ci"] = threshold_ci[target]
        stats.update(target_clean_time_fraction=target, theta_per_fold=thresholds,
                     validation_clean_time_fraction=validation)
        fraction = stats["alarm_time_fraction_clean"]["value"]
        stats["position_null"] = {str(lead): float(np.interp(fraction, np.array(null[str(lead)])[:, 0],
                                                   np.array(null[str(lead)])[:, 1]))
                                  if fraction is not None and null.get(str(lead)) else None for lead in (60, 90, 120)}
        sustained["operating_points"][f"time{int(target*100)}pct"] = stats
    out["sustained_warning"] = sustained
    return out
