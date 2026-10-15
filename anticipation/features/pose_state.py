"""Identity-free interaction state per 1 Hz tick with causal long-window summaries (``pose_state``, ``pose_state_pairs``).

What
    CPU-only re-derivation from existing ``pose_raw`` files (no decoding). Spec and column meanings:
    docs/FEATURES.md, section "pose_state". Every column is named and defined in ``meta["column_docs"]``.
    ``pose_state``        147 columns. The first 88 are the VERSION 2 columns, unchanged and in the same order: 18
                          per-tick base features, 2 cut columns, long-window summaries over 30 / 60 / 120 / 300 s
                          (mean / max / fraction above ``ENG_THR``), 4 history fractions. Columns 88..146 are the
                          VERSION 4 density-controlled additions (7 per-tick + 13 per window), which use a
                          back-view-checked facing; the old columns keep their VERSION 2 facing.
    ``pose_state_pairs``  9 columns about the identity of the current most engaged pair (cumulative engaged
                          seconds, approach-retreat cycles, time since first engaged). Track ids only persist on
                          static cameras, so use it for UCF and UBI; it is computed for every video.

Causality
    Tick t uses sampled frames with time <= t and past ticks only; windows are cut at the clip start and at shot
    cuts (motion ``cut_1s``: a window never contains a tick at or before the latest cut tick), and means divide by
    the ticks actually in the window. ``--selftest`` re-runs on real pose_raw files truncated at T and compares.

Usage
    python -m anticipation.features.pose_state --all [--shard 0 --num_shards 4] [--overwrite]
    python -m anticipation.features.pose_state --video_ids ucf/Abuse006_x264
    python -m anticipation.features.pose_state --selftest [--n 4 --cut 60]
"""

import argparse
import glob
import json
import os
import time
import zlib

import numpy as np

from anticipation.features import io, pose

EXTRACTOR = "pose_state"
EXTRACTOR_PAIRS = "pose_state_pairs"
VERSION = 4
OUTPUT_VERSIONS = {EXTRACTOR: VERSION, EXTRACTOR_PAIRS: VERSION}
WINDOWS = (30, 60, 120, 300)
ENG_THR = 0.5
CLOSE_D = 1.5
PROX_FAR = 2.5
MUTUAL_YAW = 0.3
CONTACT_D = 0.25
BYST_R = 5.0
APPROACH_R = 8.0
APPROACH_V = 0.5
INTERPOSE_PERP = 0.4
RATE_SPAN = 5
HYST_CLOSE, HYST_FAR = 1.2, 2.0
PAIR_TRACK_D = 4.0
KPT_CONF = pose.DEFAULTS["kpt_conf"]
H_SMOOTH = pose.DEFAULTS["h_smooth_sec"]
PAIR_EPS = 0.05
MIN_DX = 0.3
MAX_H_RATIO = 1.5
MIN_PAIR_D = 0.3
FRAME_DT = 0.2
NOSE_CONF = 0.5
ISO_CAP = 5.0
DYAD_MAX_N = 4
EXP_N_CAP = 30
NULL_BOX = (16.0, 4.0)
NULL_P_FACE = 0.5
NULL_FACE_MAG = 0.9
NULL_H_JITTER = 0.15
NULL_SEED = 12345
NULL_TRIALS = 5000

BASE = [
    ("n_people", "mean number of tracked people per sampled frame in (t-1, t]"),
    ("pair_frac", "fraction of frames with at least 2 tracked people"),
    ("spread", "mean distance of people to their centroid in median body heights (0 with < 2 people)"),
    ("close_mutual", "1 if any pair is within 1.5 body heights and both heads face each other (yaw > 0.3), max over frames"),
    ("eng_max", "max over pairs and frames of engagement = orient * prox; orient = mean of the two heads' positive facing "
                "toward the other (unknown head counts 0), prox = clip((2.5 - d)/2, 0, 1), d in body heights"),
    ("contact", "1 if a wrist is within 0.25 body heights of the other person's torso, shoulder, elbow or hip (pairs < 2.5 bh)"),
    ("contact_mutual", "contact within a pair whose orient >= 0.25"),
    ("contact_dmin", "min wrist-to-other-person distance in body heights, capped at 3"),
    ("top_pair_dist", "distance of the most engaged pair in body heights (3 when no engaged pair)"),
    ("byst_n", "mean number of bystanders within 5 bh of the engaged pair midpoint"),
    ("byst_orient", "mean facing of those bystanders toward the pair midpoint (-1..1; unknown heads skipped; 0 when no bystander)"),
    ("ring_circ", "circularity of >= 3 bystanders around the pair midpoint: (1 - resultant length of angles) * "
                  "(1 - radius CV); 0 otherwise"),
    ("approach_n", "max number of people (within 8 bh of the pair) moving toward the pair midpoint faster than 0.5 bh/s"),
    ("approach_speed", "mean positive speed toward the pair midpoint over those people (bh/s)"),
    ("interpose", "max over third persons of (1 - perp/0.4) for a person of similar scale between the two engaged members "
                  "(projection 0.15..0.85 of the segment, perp distance in bh)"),
    ("pair_closing", "closing speed of the engaged pair over 0.4 s (bh/s, > 0 approaching)"),
    ("gather_rate", "(n_people_t - n_people_ref)/span in people/s; ref = tick t-5 clipped to the latest cut/clip start"),
    ("spread_rate", "same rate for spread (< 0 = people converging)"),
]
CUT_COLS = [
    ("shot_cut", "1 if motion detected a shot cut in [t-1, t] (0 when motion is missing)"),
    ("since_cut", "seconds since the latest cut tick or clip start, capped at 300"),
]
WIN_SPEC = [("eng_max", ("mean", "max", "frac")), ("close_mutual", ("mean",)), ("contact_mutual", ("mean", "max")),
            ("n_people", ("mean", "max")), ("byst_orient", ("mean", "max")), ("ring_circ", ("mean", "max")),
            ("approach_n", ("mean", "max")), ("interpose", ("mean", "max"))]
PAIR_COLS = [
    ("pair_dist", "distance (bh) of the current most engaged pair (3 when none)"),
    ("pair_engaged_now", "1 if that pair's engagement exceeds 0.5 at the last frame of the tick"),
    ("pair_cum_eng_sec", "cumulative seconds (frame gaps capped at 0.3 s) that pair has been engaged (> 0.5); resets at shot cuts"),
    ("pair_since_first_eng", "seconds since that pair first became engaged (0 if never); resets at shot cuts"),
    ("pair_age", "seconds since that pair was first seen within 4 bh; resets at shot cuts, broken by track id churn"),
    ("pair_cycles", "approach-retreat cycles of that pair: re-approaches (> 2.0 bh to < 1.2 bh) after a first close; near-constant 0 on real data"),
    ("pair_ever_engaged", "1 if that pair has been engaged before; resets at shot cuts"),
    ("max_cum_eng_sec", "largest cumulative engaged seconds over all tracked pairs; resets at shot cuts, inflated by id churn"),
    ("n_pairs_engaged", "distinct track-id pairs engaged since the latest cut; inflated by id churn"),
]


NEW_BASE = [
    ("eng_pair_frac", "pairs with soft engagement > 0.5 / all valid pairs (not gated by scale or duplicate rules); soft engagement = "
                      "max of the two back-view-checked positive facings (one-sided) * prox; 0 with no valid pair"),
    ("contact_pair_frac", "pairs in contact (wrist-to-body < 0.25 bh, pair < 2.5 bh, any facing) / all valid pairs"),
    ("people_engaged_frac", "people in at least one soft-engaged pair / n_people (0 with < 2 people)"),
    ("top_pair_isolation", f"distance from the most engaged pair's midpoint to the nearest third person of similar scale, in bh, "
                           f"capped at {ISO_CAP:g} (cap when the pair is alone; 0 when no pair is engaged)"),
    ("dyad_eng", f"checked-facing eng_max, only when 2 <= n_people <= {DYAD_MAX_N} in that frame, else 0"),
    ("dyad_scene", f"fraction of frames with 2 <= n_people <= {DYAD_MAX_N}"),
    ("eng_max_excess", "max(0, checked-facing eng_max - EXP_ENG_MAX[n_people]); the expectation is a fixed table from synthetic "
                       "random placement of n people (see meta null_model)"),
]
WIN_SPEC_NEW = [("eng_pair_frac", ("mean", "max")), ("contact_pair_frac", ("mean", "max")),
                ("people_engaged_frac", ("mean", "max")), ("top_pair_isolation", ("mean", "max")),
                ("dyad_eng", ("mean", "max")), ("dyad_scene", ("mean",)), ("eng_max_excess", ("mean", "max"))]


def window_columns(spec=None, hist=True):
    spec = WIN_SPEC if spec is None else spec
    cols = []
    for w in WINDOWS:
        for key, stats in spec:
            for s in stats:
                name = f"{key}_{s}_{w}s"
                doc = (f"fraction of ticks in the last {w} s with {key} > {ENG_THR}" if s == "frac"
                       else f"{s} of {key} over the last {w} s (cut at clip start and shot cuts; mean over ticks present)")
                cols.append((name, doc))
    if hist:
        for w in WINDOWS:
            cols.append((f"hist_{w}s", f"valid ticks (decoded frame in (t-1, t]) in the {w} s window / {w}; carries time-since-start/cut"))
    return cols


def old_columns():
    return BASE + CUT_COLS + window_columns()


def new_columns():
    return NEW_BASE + window_columns(WIN_SPEC_NEW, hist=False)


def columns():
    return old_columns() + new_columns()


def _pair_geometry(P, facing=None):
    """Pairwise normalized distance D, facing toward the other fi[i, j], orient and engagement matrices."""
    facing = P["facing"] if facing is None else facing
    c, h = P["center"], P["h"]
    n = len(c)
    D = np.linalg.norm(c[:, None] - c[None], axis=2) / (0.5 * (h[:, None] + h[None]))
    dxn = (c[None, :, 0] - c[:, None, 0]) / (0.5 * (h[:, None] + h[None]))
    sgn = np.where(np.abs(dxn) > MIN_DX, np.sign(dxn), 0.0)
    fi = facing[:, None] * sgn
    ratio = np.maximum(h[:, None], h[None]) / np.minimum(h[:, None], h[None])
    bad = (ratio > MAX_H_RATIO) | (D < MIN_PAIR_D)
    pos = np.where(np.isfinite(fi), np.clip(fi, 0.0, 1.0), 0.0)
    orient = 0.5 * (pos + pos.T)
    mutual = (np.nan_to_num(fi, nan=0.0) > MUTUAL_YAW) & (np.nan_to_num(fi.T, nan=0.0) > MUTUAL_YAW)
    prox = np.clip((PROX_FAR - D) / (PROX_FAR - 0.5), 0.0, 1.0)
    eng = np.where(bad, 0.0, orient * prox)
    mutual &= ~bad
    eye = np.eye(n, dtype=bool)
    eng[eye] = 0.0
    mutual[eye] = False
    return D, fi, orient, mutual, eng, bad


EXP_ENG_MAX = {
    2: 0.019, 3: 0.051, 4: 0.092, 5: 0.135, 6: 0.18, 7: 0.227, 8: 0.272, 9: 0.311, 10: 0.34, 11: 0.378,
    12: 0.399, 13: 0.423, 14: 0.442, 15: 0.465, 16: 0.485, 17: 0.5, 18: 0.517, 19: 0.534, 20: 0.544, 21: 0.559,
    22: 0.572, 23: 0.585, 24: 0.599, 25: 0.612, 26: 0.616, 27: 0.63, 28: 0.64, 29: 0.651, 30: 0.655,
}


def _null_expectation(n, trials=NULL_TRIALS, seed=NULL_SEED):
    """Mean max pair engagement of ``n`` randomly placed people with randomly oriented, partly unknown heads."""
    rng = np.random.RandomState(seed + n)
    tot = 0.0
    for _ in range(trials):
        c = rng.rand(n, 2) * np.array(NULL_BOX)
        h = 1.0 + rng.uniform(-NULL_H_JITTER, NULL_H_JITTER, n)
        f = np.where(rng.rand(n) < NULL_P_FACE, NULL_FACE_MAG * rng.choice([-1.0, 1.0], n), np.nan)
        tot += _pair_geometry(dict(center=c, h=h, facing=f))[4].max()
    return tot / trials


def face_ok_mask(conf):
    """Heads whose keypoints fit a front or profile view: confident nose above both ears, and an eye visible."""
    c = np.asarray(conf)
    eye = (c[:, pose.LEYE] >= KPT_CONF) | (c[:, pose.REYE] >= KPT_CONF)
    return (c[:, pose.NOSE] >= NOSE_CONF) & (c[:, pose.NOSE] > c[:, pose.LEAR]) & (c[:, pose.NOSE] > c[:, pose.REAR]) & eye


def exp_eng_max(n):
    return EXP_ENG_MAX.get(min(int(n), EXP_N_CAP), 0.0)


_TARGETS = [pose.LSH, pose.RSH, pose.LELB, pose.RELB, pose.LHIP, pose.RHIP]


def _contact_dist(P, i, j):
    """Min distance (in h_j) from a visible wrist of i to the torso center or visible shoulder/elbow/hip of j."""
    xy, vis, c, h = P["xy"], P["vis"], P["center"], P["h"]
    wr = [xy[i, w] for w in (pose.LWR, pose.RWR) if vis[i, w]]
    if not wr:
        return np.inf
    tg = [c[j]] + [xy[j, k] for k in _TARGETS if vis[j, k]]
    d = np.linalg.norm(np.asarray(wr)[:, None] - np.asarray(tg)[None], axis=2)
    return float(d.min() / h[j])


def _frame_record(P, ids, vel):
    """Base per-frame features from ``person_attributes`` output; also returns the top pair indices (or None)."""
    r = dict(n=len(ids), pair=0.0, spread=0.0, close_mutual=0.0, eng_max=0.0, contact=0.0, contact_mutual=0.0,
             contact_dmin=3.0, top_d=3.0, byst_n=0.0, byst_orient=np.nan, ring_circ=0.0, approach_n=0.0,
             approach_speed=np.nan, interpose=0.0, top=None, eng=None, D=None,
             eng_pair_frac=0.0, contact_pair_frac=0.0, people_engaged_frac=0.0, top_pair_isolation=0.0, dyad_eng=0.0,
             dyad_scene=0.0, eng_max_excess=0.0)
    n = len(ids)
    r["dyad_scene"] = float(2 <= n <= DYAD_MAX_N)
    if n == 0:
        return r
    c, h = P["center"], P["h"]
    if n < 2:
        return r
    r["pair"] = 1.0
    r["spread"] = float(np.linalg.norm(c - c.mean(0), axis=1).mean() / np.median(h))
    D, fi, orient, mutual, eng, bad = _pair_geometry(P)
    r["D"], r["eng"] = D, eng
    iu = np.triu_indices(n, 1)
    near = D[iu] < CLOSE_D
    r["close_mutual"] = float((near & mutual[iu]).any())
    em = eng[iu]
    a = int(np.argmax(em))
    r["eng_max"] = float(em[a])
    _, fi_c, _, _, eng_c, _ = _pair_geometry(P, P["facing_chk"])
    pos_c = np.where(np.isfinite(fi_c), np.clip(fi_c, 0.0, 1.0), 0.0)
    prox = np.clip((PROX_FAR - D) / (PROX_FAR - 0.5), 0.0, 1.0)
    soft = np.where(bad, 0.0, np.maximum(pos_c, pos_c.T) * prox)
    np.fill_diagonal(soft, 0.0)
    valid_pairs = ~bad[iu]
    n_valid = int(valid_pairs.sum())
    cd = np.inf
    cont = cont_mut = False
    n_cont = 0
    for x, (i, j) in enumerate(zip(*iu)):
        if D[i, j] >= PROX_FAR or bad[i, j]:
            continue
        d = min(_contact_dist(P, i, j), _contact_dist(P, j, i))
        cd = min(cd, d)
        if d < CONTACT_D:
            cont = True
            n_cont += 1
            cont_mut |= orient[i, j] >= 0.25
    r["contact"], r["contact_mutual"] = float(cont), float(cont_mut)
    r["contact_dmin"] = float(min(cd, 3.0))
    ec = eng_c[iu]
    eng_pairs = soft[iu] > ENG_THR
    mx = float(ec.max())
    if n_valid:
        r["eng_pair_frac"] = float(eng_pairs.sum()) / n_valid
        r["contact_pair_frac"] = n_cont / n_valid
    r["people_engaged_frac"] = float(np.unique(np.concatenate([iu[0][eng_pairs], iu[1][eng_pairs]])).size) / n
    r["eng_max_excess"] = max(0.0, mx - exp_eng_max(n))
    if n <= DYAD_MAX_N:
        r["dyad_eng"] = mx
    if mx >= PAIR_EPS:
        a2 = int(np.argmax(ec))
        p2, q2 = int(iu[0][a2]), int(iu[1][a2])
        hm2 = 0.5 * (h[p2] + h[q2])
        mid2 = 0.5 * (c[p2] + c[q2])
        iso = ISO_CAP
        for k in range(n):
            if k in (p2, q2) or max(h[k], hm2) / min(h[k], hm2) > MAX_H_RATIO:
                continue
            iso = min(iso, float(np.linalg.norm(mid2 - c[k])) / hm2)
        r["top_pair_isolation"] = iso
    if em[a] < PAIR_EPS:
        return r
    p, q = int(iu[0][a]), int(iu[1][a])
    r["top"] = (p, q)
    r["top_d"] = float(D[p, q])
    hm = 0.5 * (h[p] + h[q])
    mid = 0.5 * (c[p] + c[q])
    others = [k for k in range(n) if k not in (p, q)]
    ab = c[q] - c[p]
    L2 = float(ab @ ab)
    inter, byst, rad, ang, toward = 0.0, [], [], [], []
    for k in others:
        if max(h[k], hm) / min(h[k], hm) > MAX_H_RATIO:
            continue
        dv = mid - c[k]
        dist = float(np.linalg.norm(dv))
        rk = dist / hm
        if L2 > 1e-6:
            s = float((c[k] - c[p]) @ ab) / L2
            perp = abs(float(ab[0] * (c[k] - c[p])[1] - ab[1] * (c[k] - c[p])[0])) / np.sqrt(L2) / hm
            if 0.15 < s < 0.85:
                inter = max(inter, max(0.0, 1.0 - perp / INTERPOSE_PERP))
        if rk < BYST_R:
            byst.append(k)
            if rk > 0.3:
                rad.append(rk)
                ang.append(np.arctan2(-dv[1], -dv[0]))
        if rk < APPROACH_R and dist > 1e-6 and int(ids[k]) in vel:
            toward.append(float(vel[int(ids[k])] @ dv) / dist)
    r["interpose"] = inter
    r["byst_n"] = float(len(byst))
    if byst:
        f = np.array([P["facing"][k] * np.sign(mid[0] - c[k, 0]) for k in byst])
        if np.isfinite(f).any():
            r["byst_orient"] = float(np.nanmean(f))
    if len(rad) >= 3:
        R = float(np.hypot(np.mean(np.cos(ang)), np.mean(np.sin(ang))))
        cv = float(np.std(rad) / max(np.mean(rad), 1e-6))
        r["ring_circ"] = (1.0 - R) * max(0.0, 1.0 - cv)
    if toward:
        tw = np.asarray(toward)
        r["approach_n"] = float((tw > APPROACH_V).sum())
        r["approach_speed"] = float(np.clip(tw[tw > 0], 0, 3).mean()) if (tw > 0).any() else 0.0
    return r


def _prepare_frames(raw):
    """Per-frame person attributes (causal track heights) in time order."""
    vmeta = raw["meta"]["video"]
    ptr = raw["frame_ptr"]
    F = len(raw["frame_t"])
    ftime = raw["frame_idx"].astype(np.float64) / float(vmeta["fps"])
    track_h = pose.TrackHeight(H_SMOOTH)
    persons = []
    for k in range(F):
        sl = slice(ptr[k], ptr[k + 1])
        tid = raw["det_track_id"][sl]
        m = tid >= 0
        ids = tid[m]
        P = None
        if len(ids):
            box, kp = raw["det_box"][sl][m], raw["det_kpt"][sl][m]
            hf = pose.body_height(box, kp[..., :2].astype(np.float64), kp[..., 2] >= KPT_CONF)
            P = pose.person_attributes(box, kp, KPT_CONF, h=track_h(ftime[k], ids, hf))
            P["facing_chk"] = np.where(face_ok_mask(kp[..., 2]), P["facing"], np.nan)
        persons.append((ids, P, {int(x): i for i, x in enumerate(ids)}))
    return persons, ftime


def _frame_loop(raw, cut_ticks):
    """Per-frame records (and pair-state outputs) in time order; returns (records, pair_state_by_frame)."""
    persons, ftime = _prepare_frames(raw)
    frame_t = raw["frame_t"].astype(np.float64)
    ok = raw["frame_ok"]
    F = len(frame_t)
    gmc_ok = raw.get("frame_gmc_ok", np.ones(F, dtype=bool))
    cut_start = np.zeros(F, dtype=bool)
    if cut_ticks is not None:
        for i in np.nonzero(cut_ticks)[0]:
            ks = np.nonzero((frame_t > i + 1e-6) & (frame_t <= i + 1 + 1e-6))[0]
            if len(ks):
                cut_start[ks[0]] = True
    recs, pstates = [], []
    state, last_t = {}, None
    for k in range(F):
        if cut_start[k]:
            state = {}
            last_t = None
        ids, P, idx = persons[k]
        vel = {}
        if k >= 1 and ok[k] and ok[k - 1] and gmc_ok[k] and gmc_ok[k - 1] and not cut_start[k]:
            dt = float(ftime[k] - ftime[k - 1])
            ida, Pa, _ = persons[k - 1]
            if dt > 0 and Pa is not None and P is not None:
                _, _, vel = pose._step_stats(Pa, ida, P, ids, raw["frame_warp"][k].astype(np.float64), dt)
        rec = _frame_record(P, ids, vel) if ok[k] else None
        if rec is not None:
            rec["pair_closing"] = np.nan
            if rec["top"] is not None and k >= 2 and ok[k - 2] and not cut_start[k] and not cut_start[k - 1]:
                p, q = int(ids[rec["top"][0]]), int(ids[rec["top"][1]])
                m0 = persons[k - 2][2]
                gap = float(ftime[k] - ftime[k - 2])
                if p in m0 and q in m0 and gap >= pose.MIN_PAIR_GAP_SEC and gmc_ok[k] and gmc_ok[k - 1] \
                        and gmc_ok[k - 2]:
                    d0, d1 = pose._pair_dist_change(persons, k - 2, k, p, q)
                    rec["pair_closing"] = (d0 - d1) / gap
            dtf = FRAME_DT if last_t is None else min(max(frame_t[k] - last_t, 0.0), 1.5 * FRAME_DT)
            last_t = frame_t[k]
            _update_pairs(state, rec, ids, frame_t[k], dtf)
            pstates.append(_pair_snapshot(state, rec, ids, frame_t[k]))
        else:
            pstates.append(None)
        recs.append(rec)
    return recs, pstates


def _update_pairs(state, rec, ids, t, dt):
    if rec["D"] is None:
        return
    D, eng = rec["D"], rec["eng"]
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            key = (int(min(ids[i], ids[j])), int(max(ids[i], ids[j])))
            st = state.get(key)
            if st is None:
                if D[i, j] >= PAIR_TRACK_D:
                    continue
                st = state[key] = dict(first=t, cum=0.0, first_eng=None, mode=None, cycles=0, seen_close=False)
            d = D[i, j]
            if eng[i, j] > ENG_THR:
                st["cum"] += dt
                if st["first_eng"] is None:
                    st["first_eng"] = t
            if d < HYST_CLOSE:
                if st["mode"] == "far" and st["seen_close"]:
                    st["cycles"] += 1
                st["mode"], st["seen_close"] = "close", True
            elif d > HYST_FAR:
                st["mode"] = "far"


def _pair_snapshot(state, rec, ids, t):
    out = dict(pair_dist=3.0, engaged_now=0.0, cum=0.0, since_first=0.0, age=0.0, cycles=0.0, ever=0.0,
               max_cum=max([s["cum"] for s in state.values()], default=0.0),
               n_eng=float(sum(s["first_eng"] is not None for s in state.values())))
    if rec["top"] is None:
        return out
    i, j = rec["top"]
    st = state.get((int(min(ids[i], ids[j])), int(max(ids[i], ids[j]))))
    if st is None:
        return out
    out.update(pair_dist=rec["top_d"], engaged_now=float(rec["eng"][i, j] > ENG_THR), cum=st["cum"],
               since_first=0.0 if st["first_eng"] is None else float(t - st["first_eng"]), age=float(t - st["first"]),
               cycles=float(st["cycles"]), ever=float(st["first_eng"] is not None))
    return out


def _tick_base(recs, raw, ticks, n_ticks):
    frame_t = raw["frame_t"].astype(np.float64)
    k1 = np.searchsorted(frame_t, ticks + 1e-6, side="right")
    k0 = np.searchsorted(frame_t, ticks - 1.0 + 1e-6, side="right")
    names = [c for c, _ in BASE + NEW_BASE]
    col = {c: i for i, c in enumerate(names)}
    B = np.zeros((n_ticks, len(names)), dtype=np.float64)
    valid = np.zeros(n_ticks, dtype=bool)
    for i in range(n_ticks):
        R = [r for r in recs[k0[i]:k1[i]] if r is not None]
        if not R:
            continue
        valid[i] = True

        def get(key, fn, default=0.0):
            a = np.array([r[key] for r in R], dtype=np.float64)
            a = a[np.isfinite(a)]
            return fn(a) if len(a) else default

        B[i, col["n_people"]] = get("n", np.mean)
        B[i, col["pair_frac"]] = get("pair", np.mean)
        B[i, col["spread"]] = get("spread", np.mean)
        B[i, col["close_mutual"]] = get("close_mutual", np.max)
        B[i, col["eng_max"]] = get("eng_max", np.max)
        B[i, col["contact"]] = get("contact", np.max)
        B[i, col["contact_mutual"]] = get("contact_mutual", np.max)
        B[i, col["contact_dmin"]] = get("contact_dmin", np.min, 3.0)
        B[i, col["top_pair_dist"]] = get("top_d", np.mean, 3.0)
        B[i, col["byst_n"]] = get("byst_n", np.mean)
        B[i, col["byst_orient"]] = get("byst_orient", np.mean)
        B[i, col["ring_circ"]] = get("ring_circ", np.mean)
        B[i, col["approach_n"]] = get("approach_n", np.max)
        B[i, col["approach_speed"]] = get("approach_speed", np.mean)
        B[i, col["interpose"]] = get("interpose", np.max)
        B[i, col["pair_closing"]] = get("pair_closing", np.mean)
        B[i, col["eng_pair_frac"]] = get("eng_pair_frac", np.mean)
        B[i, col["contact_pair_frac"]] = get("contact_pair_frac", np.mean)
        B[i, col["people_engaged_frac"]] = get("people_engaged_frac", np.mean)
        B[i, col["top_pair_isolation"]] = get("top_pair_isolation", np.mean)
        B[i, col["dyad_eng"]] = get("dyad_eng", np.max)
        B[i, col["dyad_scene"]] = get("dyad_scene", np.mean)
        B[i, col["eng_max_excess"]] = get("eng_max_excess", np.max)
    return B, valid, col


def _rates(B, col, cut_idx):
    n = len(B)
    for i in range(n):
        ref = max(i - RATE_SPAN, cut_idx[i] + 1, 0)
        span = i - ref
        if span > 0:
            B[i, col["gather_rate"]] = (B[i, col["n_people"]] - B[ref, col["n_people"]]) / span
            B[i, col["spread_rate"]] = (B[i, col["spread"]] - B[ref, col["spread"]]) / span


def _windows(B, col, valid, cut_idx, spec=None):
    spec = WIN_SPEC if spec is None else spec
    n = len(B)
    out = []
    for w in WINDOWS:
        hist = np.zeros(n)
        block = np.zeros((n, sum(len(ss) for _, ss in spec)))
        for i in range(n):
            lo = max(i - w + 1, cut_idx[i] + 1, 0)
            idx = np.arange(lo, i + 1)
            idx = idx[valid[idx]]
            hist[i] = len(idx) / w
            if not len(idx):
                continue
            j = 0
            for key, ss in spec:
                x = B[idx, col[key]]
                for s in ss:
                    block[i, j] = x.mean() if s == "mean" else x.max() if s == "max" else (x > ENG_THR).mean()
                    j += 1
        out.append((block, hist))
    return out


def compute(raw, cut_ticks=None):
    """``pose_state`` and ``pose_state_pairs`` outputs from a ``pose_raw`` dict.

    ``cut_ticks``: bool [T] shot-cut flag per tick, or None when motion is missing (no resets).
    Returns ``{name: (t_sec, feat float32, valid, meta)}``.
    """
    vmeta = raw["meta"]["video"]
    ticks = io.tick_times(vmeta["duration_sec"])
    n = len(ticks)
    cuts = np.zeros(n, dtype=bool) if cut_ticks is None else np.asarray(cut_ticks[:n], dtype=bool)
    if len(cuts) < n:
        cuts = np.pad(cuts, (0, n - len(cuts)))
    cut_idx = np.full(n, -1, dtype=np.int64)
    last = -1
    for i in range(n):
        if cuts[i]:
            last = i
        cut_idx[i] = last
    recs, pst = _frame_loop(raw, cuts if cut_ticks is not None else None)
    B, valid, col = _tick_base(recs, raw, ticks, n)
    _rates(B, col, cut_idx)
    wins = _windows(B, col, valid, cut_idx)
    wins_new = _windows(B, col, valid, cut_idx, WIN_SPEC_NEW)
    n_old = len(BASE)
    since = np.minimum(np.arange(n) - cut_idx, 300).astype(np.float64)
    feat = np.concatenate([B[:, :n_old], cuts[:, None].astype(np.float64), since[:, None]]
                          + [w[0] for w in wins] + [w[1][:, None] for w in wins]
                          + [B[:, n_old:]] + [w[0] for w in wins_new], axis=1)
    cols = columns()
    assert feat.shape[1] == len(cols), (feat.shape, len(cols))
    base_meta = dict(
        version=VERSION, window_sec=float(max(WINDOWS)), video=vmeta, pose_raw_version=raw["meta"].get("version"),
        motion_cuts_used=cut_ticks is not None, n_cut_ticks=int(cuts.sum()),
        constants=dict(windows=list(WINDOWS), eng_thr=ENG_THR, close_d=CLOSE_D, prox_far=PROX_FAR,
                       mutual_yaw=MUTUAL_YAW, contact_d=CONTACT_D, byst_r=BYST_R, approach_r=APPROACH_R,
                       approach_v=APPROACH_V, interpose_perp=INTERPOSE_PERP, rate_span=RATE_SPAN,
                       hyst_close=HYST_CLOSE, hyst_far=HYST_FAR, pair_track_d=PAIR_TRACK_D, kpt_conf=KPT_CONF,
                       nose_conf=NOSE_CONF, iso_cap=ISO_CAP, dyad_max_n=DYAD_MAX_N),
        null_model=dict(box_bh=list(NULL_BOX), p_face_known=NULL_P_FACE, face_mag=NULL_FACE_MAG, h_jitter=NULL_H_JITTER,
                        seed=NULL_SEED, trials=NULL_TRIALS, n_cap=EXP_N_CAP,
                        exp_eng_max={str(k): v for k, v in sorted(EXP_ENG_MAX.items())}),
        units="distances in body heights (bh); invalid ticks have no decoded frame in (t-1, t]",
    )
    meta = dict(base_meta, extractor=EXTRACTOR, D=len(cols), columns=[c for c, _ in cols],
                column_docs=dict(cols), n_columns_v2=len(old_columns()),
                v2_columns_note="the first 88 columns are computed exactly as in VERSION 2 (unchecked head facing); "
                                "columns 88.. use facing checked by face_ok_mask (nose conf >= nose_conf and above both ears, "
                                "an eye visible), else unknown")
    out = {EXTRACTOR: (ticks, feat.astype(np.float32), valid, meta)}
    pf = np.zeros((n, len(PAIR_COLS)), dtype=np.float64)
    frame_t = raw["frame_t"].astype(np.float64)
    k1 = np.searchsorted(frame_t, ticks + 1e-6, side="right")
    k0 = np.searchsorted(frame_t, ticks - 1.0 + 1e-6, side="right")
    for i in range(n):
        S = [s for s in pst[k0[i]:k1[i]] if s is not None]
        if S:
            s = S[-1]
            pf[i] = [s["pair_dist"], s["engaged_now"], s["cum"], s["since_first"], s["age"], s["cycles"],
                     s["ever"], s["max_cum"], s["n_eng"]]
    pmeta = dict(base_meta, extractor=EXTRACTOR_PAIRS, D=len(PAIR_COLS), columns=[c for c, _ in PAIR_COLS],
                 column_docs=dict(PAIR_COLS),
                 note="track ids persist only on static cameras: intended for UCF and UBI; state resets at shot cuts")
    out[EXTRACTOR_PAIRS] = (ticks, pf.astype(np.float32), valid, pmeta)
    return out


def motion_cuts(video_id):
    """Per-tick shot-cut flags from the motion file, or None when it is missing or unreadable."""
    path = io.feature_path("motion", video_id)
    if not os.path.exists(path):
        return None
    try:
        t, f, _, meta = io.load_features("motion", video_id)
        return f[:, meta["columns"].index("cut_1s")] > 0.5
    except Exception:
        return None


def derive_and_save(video_path, video_id, device="cpu", outputs=(EXTRACTOR, EXTRACTOR_PAIRS), **cfg):
    """Registry saver: compute from pose_raw (+ motion cuts) and write both outputs."""
    raw = pose.load_raw(video_id)
    res = compute(raw, motion_cuts(video_id))
    paths = {name: io.save_features(name, video_id, *res[name]) for name in outputs}
    return dict(paths=paths, duration_sec=raw["meta"]["video"]["duration_sec"])


def derive_state(video_path, video_id, device="cpu", **cfg):
    return derive_and_save(video_path, video_id, device, outputs=(EXTRACTOR,))


def derive_pairs(video_path, video_id, device="cpu", **cfg):
    return derive_and_save(video_path, video_id, device, outputs=(EXTRACTOR_PAIRS,))


def truncate_raw(raw, cut_sec):
    """Copy of a pose_raw dict containing only frames with time <= ``cut_sec``."""
    F = int((raw["frame_t"].astype(np.float64) <= cut_sec + 1e-6).sum())
    n = int(raw["frame_ptr"][F])
    out = {}
    for k, v in raw.items():
        if k == "meta":
            continue
        if k == "frame_ptr":
            out[k] = v[:F + 1]
        elif k.startswith("det_"):
            out[k] = v[:n]
        else:
            out[k] = v[:F]
    out["meta"] = json.loads(json.dumps(raw["meta"]))
    out["meta"]["video"]["duration_sec"] = float(cut_sec)
    return out


def _is_current(video_id, outputs):
    mo = os.path.exists(io.feature_path("motion", video_id))
    for name in outputs:
        p = io.feature_path(name, video_id)
        if not os.path.exists(p):
            return False
        try:
            with np.load(p, allow_pickle=False) as z:
                m = json.loads(str(z["meta"]))
        except Exception:
            return False
        if m.get("version") != VERSION or (mo and not m.get("motion_cuts_used")):
            return False
    return True


def list_videos():
    root = os.path.join(io.FEATURE_ROOT, pose.EXTRACTOR_RAW)
    paths = sorted(glob.glob(os.path.join(root, "*", "*.npz")))
    return [os.path.relpath(p, root)[:-4] for p in paths if not p.endswith(".tmp.npz")]


def _synthetic_frames(rng, n, box, frames, face_ok=True, known=1.0):
    out = []
    for _ in range(frames):
        c = rng.rand(n, 2) * np.array(box)
        f = np.where(rng.rand(n) < known, 0.9 * rng.choice([-1.0, 1.0], n), np.nan)
        P = dict(center=c, h=np.ones(n), facing=f, facing_chk=f if face_ok else np.full(n, np.nan),
                 xy=np.zeros((n, 17, 2)), vis=np.zeros((n, 17), dtype=bool))
        out.append(_frame_record(P, np.arange(n), {}))
    return out


def _unit_checks_v3(P):
    res = {}
    rng = np.random.RandomState(7)
    crowd = _synthetic_frames(rng, 20, (16.0, 4.0), 300, known=NULL_P_FACE)
    m = lambda k: float(np.mean([r[k] for r in crowd]))
    print(f"crowd n=20: close_mutual={m('close_mutual'):.2f} eng_pair_frac={m('eng_pair_frac'):.3f} "
          f"people_engaged_frac={m('people_engaged_frac'):.2f} eng_max_excess={m('eng_max_excess'):.3f}")
    full = _synthetic_frames(rng, 20, (16.0, 4.0), 300)
    res["crowd_old_saturates"] = float(np.mean([r["close_mutual"] for r in full])) > 0.9
    res["crowd_new_low"] = m("eng_pair_frac") < 0.05 and m("people_engaged_frac") < 0.5
    res["crowd_excess_small"] = m("eng_max_excess") < 0.25
    dy = _frame_record(dict(center=np.array([[0.0, 0], [1.0, 0]]), h=np.ones(2), facing=np.array([0.9, -0.9]),
                            facing_chk=np.array([0.9, -0.9]), xy=np.zeros((2, 17, 2)), vis=np.zeros((2, 17), bool)),
                       np.arange(2), {})
    res["dyad_new_high"] = dy["eng_pair_frac"] == 1.0 and dy["people_engaged_frac"] == 1.0 and dy["dyad_eng"] > 0.5 \
        and dy["dyad_scene"] == 1.0 and dy["top_pair_isolation"] == ISO_CAP
    back = _frame_record(dict(center=np.array([[0.0, 0], [1.0, 0]]), h=np.ones(2), facing=np.array([0.9, -0.9]),
                              facing_chk=np.full(2, np.nan), xy=np.zeros((2, 17, 2)), vis=np.zeros((2, 17), bool)),
                         np.arange(2), {})
    res["back_view_not_mutual_new"] = back["close_mutual"] == 1.0 and back["eng_pair_frac"] == 0.0 \
        and back["dyad_eng"] == 0.0 and back["eng_max_excess"] == 0.0
    one = _frame_record(dict(center=np.array([[0.0, 0], [1.0, 0]]), h=np.ones(2), facing=np.array([0.9, np.nan]),
                             facing_chk=np.array([0.9, np.nan]), xy=np.zeros((2, 17, 2)), vis=np.zeros((2, 17), bool)),
                        np.arange(2), {})
    res["one_sided_facing_counts"] = one["eng_pair_frac"] == 1.0 and one["dyad_eng"] < 0.5 and one["close_mutual"] == 0.0
    n10 = 10
    xy = np.zeros((n10, 17, 2))
    vis = np.zeros((n10, 17), bool)
    xy[0, pose.LWR] = [1.0, 0.0]
    vis[0, pose.LWR] = True
    cen = np.array([[0.0, 0.0], [1.0, 0.0]] + [[10.0 + 5.0 * k, 0.0] for k in range(n10 - 2)])
    ct = _frame_record(dict(center=cen, h=np.ones(n10), facing=np.full(n10, np.nan), facing_chk=np.full(n10, np.nan),
                            xy=xy, vis=vis), np.arange(n10), {})
    res["contact_normalised_by_pairs"] = abs(ct["contact_pair_frac"] - 1.0 / 45) < 1e-9 and ct["contact"] == 1.0
    conf = np.full((4, 17), 0.1)
    conf[0, :5] = [0.9, 0.8, 0.8, 0.6, 0.6]
    conf[1, :5] = [0.4, 0.7, 0.7, 0.8, 0.8]
    conf[2, :5] = [0.9, 0.1, 0.1, 0.1, 0.1]
    conf[3, :5] = [0.8, 0.8, 0.1, 0.1, 0.6]
    res["face_ok_mask"] = face_ok_mask(conf).tolist() == [True, False, False, True]
    res["null_table_reproducible"] = abs(_null_expectation(10, 1500) - EXP_ENG_MAX[10]) < 0.03
    return res


def _unit_checks():
    """Synthetic checks: pair gating, dx margin, hist over valid ticks only, pair-time cap."""
    def P(centers, hs, facing):
        return dict(center=np.array(centers, float), h=np.array(hs, float), facing=np.array(facing, float))

    res = {}
    g = _pair_geometry(P([[0, 0], [0.001, 0]], [100, 100], [0.9, -0.9]))
    res["depth_stacked_no_facing"] = not g[3].any() and g[4].max() == 0.0
    g = _pair_geometry(P([[0, 0], [150, 0]], [100, 400], [-0.9, 0.9]))
    res["height_ratio_gate"] = bool(g[5][0, 1]) and g[4].max() == 0.0
    g = _pair_geometry(P([[0, 0], [5, 0]], [100, 100], [0.9, -0.9]))
    res["duplicate_gate"] = bool(g[5][0, 1]) and g[4].max() == 0.0
    g = _pair_geometry(P([[0, 0], [100, 0]], [100, 100], [0.9, -0.9]))
    res["true_facing_pair"] = bool(g[3][0, 1]) and g[4][0, 1] > 0.5
    col = {c: i for i, (c, _) in enumerate(BASE)}
    valid = np.array([True, False, False, True])
    (_, hist), = _windows(np.zeros((4, len(BASE))), col, valid, np.full(4, -1))[:1]
    res["hist_counts_valid_only"] = abs(hist[3] - 2 / WINDOWS[0]) < 1e-9 and abs(hist[2] - 1 / WINDOWS[0]) < 1e-9
    st = {}
    rec = dict(D=np.array([[0, 1.0], [1.0, 0]]), eng=np.array([[0, 0.9], [0.9, 0]]))
    ids = np.array([1, 2])
    _update_pairs(st, rec, ids, 1.0, min(5.0, 1.5 * FRAME_DT))
    res["pair_dt_cap"] = abs(st[(1, 2)]["cum"] - 1.5 * FRAME_DT) < 1e-9
    res.update(_unit_checks_v3(P))
    for k, v in res.items():
        print(f"unit {k:26s} {'OK' if v else 'FAIL'}")
    return all(res.values())


def selftest(n=4, cut=60, seed=0):
    """Truncate real pose_raw files at ``cut`` s and require identical rows for ticks <= cut - 1 (CPU)."""
    rng = np.random.RandomState(seed)
    vids = [v for v in list_videos() if v.split("/")[0] in ("ucf", "xd", "ubi")]
    rng.shuffle(vids)
    tested, ok_all = 0, _unit_checks()
    for v in vids:
        raw = pose.load_raw(v)
        if raw["meta"]["video"]["duration_sec"] < cut + 20 or raw["meta"]["n_dets"] < 300:
            continue
        cuts = motion_cuts(v)
        full = compute(raw, cuts)
        part = compute(truncate_raw(raw, cut), None if cuts is None else cuts[:int(cut)])
        for name in full:
            tf, ff, vf, _ = full[name]
            tp, fp, vp, _ = part[name]
            k = int(np.sum(tp <= cut - 1))
            d = float(np.nanmax(np.abs(ff[:k] - fp[:k])))
            dT = float(np.nanmax(np.abs(ff[:k + 1] - fp[:k + 1])))
            same = bool(np.array_equal(vf[:k], vp[:k]))
            ok = d <= 1e-5 and same
            ok_all &= ok
            print(f"{v:32s} {name:17s} ticks<= {cut - 1}: max_abs_diff={d:.2e} (incl. tick {cut}: {dT:.2e}) "
                  f"cuts={'yes' if cuts is not None else 'no'} n_cut={int(0 if cuts is None else cuts[:cut].sum())} "
                  f"nonzero={float((np.abs(ff[:k]) > 0).mean()):.2f} {'OK' if ok else 'FAIL'}")
        tested += 1
        if tested >= n:
            break
    print("selftest", "OK" if ok_all and tested else "FAIL", f"({tested} videos)")
    return ok_all and tested > 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--video_ids", nargs="*")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--cut", type=int, default=60)
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest(args.n, args.cut) else 1)
    vids = args.video_ids or (list_videos() if args.all else [])
    vids = [v for v in vids if zlib.crc32(v.encode()) % args.num_shards == args.shard]
    t0, done, skipped, failed = time.time(), 0, 0, 0
    for v in vids:
        if not args.overwrite and _is_current(v, (EXTRACTOR, EXTRACTOR_PAIRS)):
            skipped += 1
            continue
        try:
            derive_and_save(None, v)
            done += 1
        except Exception as e:
            failed += 1
            print("FAIL", v, repr(e), flush=True)
        if (done + failed) % 200 == 0 and done + failed:
            print(f"shard {args.shard}: done={done} skipped={skipped} failed={failed} {time.time() - t0:.0f}s", flush=True)
    print(f"shard {args.shard}: done={done} skipped={skipped} failed={failed} of {len(vids)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
