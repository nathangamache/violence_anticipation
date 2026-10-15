"""Screen metrics M1-M8 (MODALITY_PLAN 6.5) with cluster bootstrap over split_group."""

import numpy as np

from anticipation.eval.va30 import PairAUC, SOURCE_KINDS, TIME_MATCH_EDGES, time_match_weights
from anticipation.screen.features import range_max

SRC_CODE = {s: i for i, s in enumerate(SOURCE_KINDS)}
KIND_CODE = {"clean": 0, "violent": 1, "h1": 2}
BANDS = {"band": (30, 120), "band_30_60": (30, 60), "long": (60, 120), "near": (0, 10), "b0_10": (0, 10),
         "b10_30": (10, 30), "b30_60": (30, 60), "b60_120": (60, 120)}
CLASS_GROUPS = {"dyadic": ("fighting", "abuse", "assault"), "instrumental": ("robbery", "shooting", "arrest"),
                "riot": ("riot",)}
TAIL_PCT, TAIL_MIN_TICKS = 99.5, 3


class Stat:
    def __init__(self, value, boot=None, **extra):
        self.value = None if value is None or not np.isfinite(value) else float(value)
        self.boot = np.asarray(boot, float) if boot is not None else np.zeros(0)
        self.extra = extra

    def _finite(self):
        return self.boot[np.isfinite(self.boot)]

    @property
    def ci(self):
        b = self._finite()
        if len(b) < 10 or len(b) < 0.8 * len(self.boot):
            return None
        return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

    @property
    def lower(self):
        c = self.ci
        return None if c is None else c[0]

    @property
    def upper(self):
        c = self.ci
        return None if c is None else c[1]

    def p_le(self, null):
        """One-sided bootstrap p-value for H0: statistic <= null."""
        b = self._finite()
        return None if len(b) < 10 else float((1 + np.sum(b <= null)) / (len(b) + 1))

    def to_dict(self, null=None):
        d = {"value": self.value, "ci": self.ci, **self.extra}
        if null is not None:
            d["p_le_null"] = self.p_le(null)
        return d


def diff(a, b):
    n = min(len(a.boot), len(b.boot))
    v = None if a.value is None or b.value is None else a.value - b.value
    return Stat(v, a.boot[:n] - b.boot[:n])


class Pool:
    """Concatenated ticks of the scored test videos of all screening folds (violent, clean, H1)."""

    def __init__(self, ss, vts):
        self.ss, self.vts = ss, list(vts)
        self.vids = [v.vid for v in self.vts]
        self.index = {v: i for i, v in enumerate(self.vids)}
        self.V = len(self.vids)
        lens = np.array([v.n for v in self.vts], np.int64)
        self.offsets = np.r_[0, np.cumsum(lens)]
        self.vidx = np.repeat(np.arange(self.V), lens)
        cat = lambda a, dt=float: (np.concatenate(a).astype(dt) if len(a) else np.zeros(0, dt))  # noqa: E731
        self.t = cat([v.t for v in self.vts])
        self.tte = cat([v.tte for v in self.vts])
        self.pre = cat([v.pre for v in self.vts], bool)
        self.masked = cat([v.masked for v in self.vts], bool)
        self.remaining = cat([v.remaining for v in self.vts])
        self.kind_v = np.array([KIND_CODE[v.kind] for v in self.vts], np.int64)
        self.src_name_v = np.array([ss.source_kind(v) for v in self.vids])
        self.src_v = np.array([SRC_CODE.get(s, -1) for s in self.src_name_v], np.int64)
        groups = [ss.split_group.get(v, v) for v in self.vids]
        _, self.group_v = np.unique(np.array(groups, dtype=object).astype(str), return_inverse=True)
        self.fold_v = np.array([ss.fold_of(v) for v in self.vids], np.int64)
        self.kind = self.kind_v[self.vidx]
        self.src = self.src_v[self.vidx]
        h1p = set(ss.h1p_ids())
        self.h1p_v = np.array([v in h1p for v in self.vids], bool)
        self.h1_class_v = np.array([str(ss.h1.loc[v, "classes"]) if v in ss.h1.index else "" for v in self.vids])
        ev = ss.events()
        ev = ev[ev["video_id"].isin(self.index)].reset_index(drop=True) if len(ev) else ev
        self.events = ev
        self.ev_vi = np.array([self.index[v] for v in ev["video_id"]], np.int64) if len(ev) else np.zeros(0, np.int64)
        self.ev_ia = np.maximum(np.ceil(ev["win_lo"].values - 1e-9).astype(np.int64), 1) - 1 if len(ev) else np.zeros(0, np.int64)
        self.ev_ib = np.floor(ev["win_hi"].values + 1e-9).astype(np.int64) - 1 if len(ev) else np.zeros(0, np.int64)
        self.ev_src = self.src_v[self.ev_vi] if len(ev) else np.zeros(0, np.int64)
        self.clean_idx = np.flatnonzero(self.kind_v == KIND_CODE["clean"])
        ev_first = set(ev.loc[ev["first"], "video_id"]) if len(ev) else set()
        self.first_video_v = np.array([v in ev_first for v in self.vids], bool)
        self.audio_video_v = np.zeros(self.V, bool)
        if len(ev):
            self.audio_video_v[[self.index[v] for v in ev.loc[ev["s_audio"], "video_id"]]] = True
        self._defs = {}

    def sl(self, i):
        return slice(self.offsets[i], self.offsets[i + 1])

    def assemble(self, scores):
        out = np.empty(len(self.t))
        for i, v in enumerate(self.vids):
            x = scores[v]
            n = self.offsets[i + 1] - self.offsets[i]
            if len(x) < n:
                raise ValueError(f"{v}: {len(x)} scores for {n} ticks")
            out[self.sl(i)] = x[:n]
        return out

    def boot_weights(self, n_boot, seed=0):
        rng = np.random.default_rng(seed)
        G = int(self.group_v.max()) + 1 if self.V else 0
        if not n_boot or not G:
            return np.zeros((0, self.V))
        Wg = rng.multinomial(G, np.full(G, 1.0 / G), size=n_boot).astype(np.float64)
        return Wg[:, self.group_v]

    def clean_neg(self):
        return (self.kind == KIND_CODE["clean"]) & ~self.masked & (self.remaining >= 60)

    def band_pos(self, lo, hi, first=True, vsel=None):
        p = (self.kind == KIND_CODE["violent"]) & ~self.masked & (self.tte > lo) & (self.tte <= hi)
        p &= self.pre if first else ~self.pre
        if first:
            p &= self.first_video_v[self.vidx]
        if vsel is not None:
            p &= np.asarray(vsel, bool)[self.vidx]
        return p

    def pair_def(self, key, pos, neg, source=True, strata=None):
        """Label-only weights: ``(pos_mask, neg_mask, neg_weights)``; cached under ``key``."""
        if key not in self._defs:
            src = self.src
            if strata is not None:
                src = src * 1000 + strata
            w, pos2 = time_match_weights(self.t, pos, neg, TIME_MATCH_EDGES, source=src if source else None)
            self._defs[key] = (pos2, neg, w)
        return self._defs[key]

    def auc(self, score, d, W):
        pos2, neg, w = d
        pa = PairAUC(score, self.vidx, pos2, neg, self.V, neg_w=w)
        s = Stat(pa.value(), pa.boot(W), n_pos=int(pa.n_pos), n_neg=int(pa.n_neg), n_pos_videos=int(len(pa.pvid)))
        return s


def weighted_quantile(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][min(np.searchsorted(cw, q * cw[-1]), len(x) - 1)])


def sparse_pauc(pool, score, d, W, fpr_max=0.01, chunk=100):
    """McClish-standardized partial AUC at clean FPR <= fpr_max (time-matched negatives), with bootstrap."""
    pos2, neg, w = d
    if not pos2.any() or not (neg & (w > 0)).any():
        return Stat(None)
    negs = neg & (w > 0)
    q = weighted_quantile(score[negs], w[negs], 1 - min(0.05, 5 * fpr_max))
    top = np.flatnonzero((pos2 | negs) & (score >= q))
    top = top[np.argsort(-score[top], kind="stable")]
    vi, isneg = pool.vidx[top], negs[top]
    ispos = pos2[top]
    nw = np.where(isneg, w[top], 0.0)
    npos = np.bincount(pool.vidx[pos2], minlength=pool.V).astype(float)
    nneg = np.bincount(pool.vidx[negs], weights=w[negs], minlength=pool.V)
    amin, amax = fpr_max ** 2 / 2, fpr_max

    def run(Wm):
        out = np.empty(len(Wm))
        for a in range(0, len(Wm), chunk):
            wt = Wm[a:a + chunk][:, vi]
            dp = Wm[a:a + chunk] @ npos
            dn = Wm[a:a + chunk] @ nneg
            tpr = np.cumsum(wt * ispos, axis=1) / dp[:, None]
            nwt = wt * nw
            fpr = np.cumsum(nwt, axis=1) / dn[:, None]
            prev = fpr - nwt / dn[:, None]
            width = np.clip(np.minimum(fpr, fpr_max) - np.minimum(prev, fpr_max), 0, None)
            area = (tpr * width).sum(1)
            out[a:a + chunk] = 0.5 * (1 + (area - amin) / (amax - amin))
        return out

    return Stat(run(np.ones((1, pool.V)))[0], run(W) if len(W) else None)


def _clean_windows(pool, score, mode, thr_v=None):
    """[E, Vc] window max (or count above the video's threshold) over each event window in each clean video."""
    E, Vc = len(pool.events), len(pool.clean_idx)
    out = np.full((E, Vc), np.nan)
    for j, vi in enumerate(pool.clean_idx):
        x = score[pool.sl(vi)]
        m = pool.masked[pool.sl(vi)]
        sel = np.flatnonzero((pool.ev_src == pool.src_v[vi]) & (pool.ev_ib < len(x)) & (pool.ev_ib >= pool.ev_ia))
        if not len(sel):
            continue
        cm = np.r_[0, np.cumsum(m)]
        sel = sel[(cm[pool.ev_ib[sel] + 1] - cm[pool.ev_ia[sel]]) == 0]
        if mode == "max":
            out[sel, j] = range_max(x, pool.ev_ia[sel], pool.ev_ib[sel])
        else:
            cs = np.r_[0, np.cumsum(x > thr_v[vi])]
            out[sel, j] = cs[pool.ev_ib[sel] + 1] - cs[pool.ev_ia[sel]]
    return out


def _event_values(pool, score, mode, thr_v=None):
    xe = np.full(len(pool.events), np.nan)
    for e in range(len(pool.events)):
        sl = pool.sl(pool.ev_vi[e])
        x, m = score[sl], pool.masked[sl]
        a, b = pool.ev_ia[e], min(pool.ev_ib[e], len(x) - 1)
        if b < a:
            continue
        ok = ~m[a:b + 1]
        if not ok.any():
            continue
        xe[e] = x[a:b + 1][ok].max() if mode == "max" else float((x[a:b + 1][ok] > thr_v[pool.ev_vi[e]]).sum())
    return xe


def _event_stat(pool, xe, CM, W, mode, sel=None):
    """Event-level contrast against matched clean windows: AUROC (mode max) or hit-rate difference (count)."""
    valid = np.isfinite(CM) & np.isfinite(xe)[:, None]
    if mode == "max":
        A = np.where(valid, (xe[:, None] > CM) + 0.5 * (xe[:, None] == CM), 0.0)
        e_val = np.zeros(len(xe))
    else:
        A = np.where(valid, (CM >= TAIL_MIN_TICKS).astype(float), 0.0)
        e_val = (xe >= TAIL_MIN_TICKS).astype(float)
    Mv = valid.astype(float)
    inc = np.isfinite(xe) if sel is None else (np.isfinite(xe) & sel)

    def run(Wm):
        Wc = Wm[:, pool.clean_idx]
        den = Mv @ Wc.T
        with np.errstate(invalid="ignore", divide="ignore"):
            R = (A @ Wc.T) / den
        okm = np.isfinite(R) & inc[:, None]
        We = Wm[:, pool.ev_vi].T * okm
        R = np.where(okm, R, 0.0)
        denom = We.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            if mode == "max":
                return (We * R).sum(0) / denom
            h = (We * e_val[:, None]).sum(0) / denom
            c = (We * R).sum(0) / denom
            return h - c, h, c

    one = np.ones((1, pool.V))
    if mode == "max":
        return Stat(run(one)[0], run(W) if len(W) else None, n_events=int(inc.sum()))
    d, h, c = run(one)
    db = run(W)[0] if len(W) else None
    return Stat(d[0], db, hit_rate=float(h[0]), clean_rate=float(c[0]), n_events=int(inc.sum()))


def tail_thresholds(pool, clean_val_scores, pct=TAIL_PCT):
    """Per-video threshold: the pct-th percentile of the video's fold model on clean validation ticks
    (``clean_val_scores``: {fold: 1-D array})."""
    thr = {f: (float(np.percentile(x, pct)) if len(x) else np.inf) for f, x in clean_val_scores.items()}
    return np.array([thr.get(int(f), np.inf) for f in pool.fold_v])


def event_metrics(pool, score, W, thr_v=None, first_only=False):
    E = pool.events
    if not len(E):
        return {}
    sel = E["first"].values if first_only else None
    xe = _event_values(pool, score, "max")
    out = {"m8": _event_stat(pool, xe, _clean_windows(pool, score, "max"), W, "max", sel)}
    if thr_v is not None:
        xc = _event_values(pool, score, "count", thr_v)
        out["tail"] = _event_stat(pool, xc, _clean_windows(pool, score, "count", thr_v), W, "count", sel)
    return out


def ramp_stat(pool, score):
    """Paired within-video mean(tte 0-30) minus mean(tte 60-120) on S_ramp videos; share rising and Wilcoxon p."""
    from scipy.stats import wilcoxon

    ev = pool.events
    vids = sorted(set(pool.ev_vi[ev["s_ramp"].values])) if len(ev) else []
    diffs = []
    for i in vids:
        sl = pool.sl(i)
        ok = pool.pre[sl] & ~pool.masked[sl]
        tte, x = pool.tte[sl], score[sl]
        a = x[ok & (tte > 0) & (tte <= 30)]
        b = x[ok & (tte > 60) & (tte <= 120)]
        if len(a) and len(b):
            diffs.append(a.mean() - b.mean())
    d = np.array(diffs)
    if len(d) < 6 or np.allclose(d, 0):
        return {"n_videos": len(d), "share_rising": None, "wilcoxon_p": None}
    return {"n_videos": int(len(d)), "share_rising": float((d > 0).mean()), "mean_diff": float(d.mean()),
            "wilcoxon_p": float(wilcoxon(d).pvalue)}


def ratio_stat(num, den, floor=0.01):
    """(num - 0.5) / (den - 0.5) with bootstrap; undefined where the denominator is below ``floor``."""
    def f(a, b):
        a, b = np.asarray(a, float) - 0.5, np.asarray(b, float) - 0.5
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(b > floor, a / b, np.nan)
    n = min(len(num.boot), len(den.boot))
    v = f(num.value, den.value) if num.value is not None and den.value is not None else None
    return Stat(None if v is None else float(v), f(num.boot[:n], den.boot[:n]))


def holm(pvals):
    """Holm-adjusted p-values for ``{key: p}`` (None stays None)."""
    items = sorted(((p, k) for k, p in pvals.items() if p is not None))
    m = len(items)
    adj, run = {k: None for k in pvals}, 0.0
    for r, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - r) * p))
        adj[k] = run
    return adj


def evaluate_pool(pool, scores, val_scores, W, persons=None, primary_audio=False):
    """All screen statistics for one block. ``scores``: model -> pooled score array; ``val_scores``: model ->
    {fold: {video: scores}} (clean validation ticks set the sparse-cue threshold)."""
    P = pool
    neg = P.clean_neg()
    vsel = P.audio_video_v if primary_audio else None
    out = {}
    d_band = P.pair_def(("band", primary_audio), P.band_pos(30, 120, vsel=vsel), neg)
    d_6 = P.pair_def(("band_30_60", primary_audio), P.band_pos(30, 60, vsel=vsel), neg)
    s = scores["p1"]
    out["m1"] = P.auc(s, d_band, W)
    out["m1_30_60"] = P.auc(s, d_6, W)
    for m in ("p1s", "p1b", "n", "time_source", "shuffled", "b_detect", "n_plus_p1"):
        if m in scores:
            out[f"m1_{m}"] = P.auc(scores[m], d_band, W)
    if "p1_later" in scores:
        d = P.pair_def("later", P.band_pos(30, 120, first=False), neg)
        out["m1_later"] = P.auc(scores["p1_later"], d, W)
    if "p1_near" in scores:
        d = P.pair_def("near", P.band_pos(0, 10), neg)
        out["m1_near"] = P.auc(scores["p1_near"], d, W)
    ev = P.events
    long_v = np.zeros(P.V, bool)
    ramp_v = np.zeros(P.V, bool)
    if len(ev):
        long_v[P.ev_vi[ev["s_long60"].values]] = True
        ramp_v[P.ev_vi[ev["s_ramp"].values]] = True
    d_long = P.pair_def("long", P.band_pos(60, 120, vsel=long_v), neg)
    out["m1_long"] = P.auc(s, d_long, W)
    for b in ("b0_10", "b10_30", "b30_60", "b60_120"):
        lo, hi = BANDS[b]
        out[f"m5_{b}"] = P.auc(s, P.pair_def(b, P.band_pos(lo, hi), neg), W)
    out["m5_ramp"] = ramp_stat(P, s)
    first_tick = np.zeros(len(s))
    for i in range(P.V):
        sl = P.sl(i)
        if sl.stop > sl.start:
            first_tick[sl] = s[min(sl.start + 1, sl.stop - 1)]
    d_ramp = P.pair_def("ramp", P.band_pos(30, 120, vsel=ramp_v), neg)
    out["m4b_first_frame"] = P.auc(first_tick, d_ramp, W)
    out["m1_sparse_pauc"] = sparse_pauc(P, s, d_band, W)
    thr_v = tail_thresholds(P, val_scores["p1"]) if "p1" in val_scores else None
    out["events"] = event_metrics(P, s, W, thr_v)
    out["events_first"] = event_metrics(P, s, W, thr_v, first_only=True)
    if "n" in scores:
        en = event_metrics(P, scores["n"], W)
        out["events_n"] = en
        if "m8" in out["events"] and "m8" in en:
            out["m8_gain_over_n"] = diff(out["events"]["m8"], en["m8"])
        out["m4n"] = diff(out["m1_n_plus_p1"], out["m1_n"]) if "m1_n_plus_p1" in out else None
    out.update(specificity(P, s, d_band, W))
    out["breakdowns"] = breakdowns(P, s, neg, W, primary_audio)
    if persons is not None:
        out["m2"] = person_stratified(P, s, persons, neg, W)
    return out


def specificity(P, s, d_band, W):
    out = {}
    h1 = (P.kind == KIND_CODE["h1"]) & (P.tte > 30) & (P.tte <= 120)
    if not h1.any():
        return out
    ucf = P.src == SRC_CODE["ucf"]
    pos_u = P.band_pos(30, 120) & ucf
    d_u = P.pair_def("m1_ucf", pos_u, P.clean_neg() & ucf, source=False)
    out["m1_ucf"] = P.auc(s, d_u, W)
    h1p = h1 & P.h1p_v[P.vidx]
    if h1p.any():
        d = P.pair_def("m3p", pos_u, h1p, source=False)
        out["m3p"] = P.auc(s, d, W)
        out["m3p_ratio"] = ratio_stat(out["m3p"], out["m1_ucf"])
    d = P.pair_def("m3_all_h1", P.band_pos(30, 120), h1)
    out["m3_all_h1"] = P.auc(s, d, W)
    d = P.pair_def("h1_vs_clean", h1, P.clean_neg())
    out["h1_vs_clean"] = P.auc(s, d, W)
    per = {}
    for c in sorted({c for v in np.unique(P.h1_class_v[P.kind_v == KIND_CODE["h1"]]) for c in v.split("|") if c}):
        sel = h1 & np.array([c in x.split("|") for x in P.h1_class_v])[P.vidx]
        if sel.any():
            per[c] = P.auc(s, P.pair_def(("m3c", c), P.band_pos(30, 120), sel), W)
    out["m3_per_class"] = per
    return out


def breakdowns(P, s, neg, W, primary_audio=False):
    ev = P.events
    out = {}
    if not len(ev):
        return out

    def vsel_from(col):
        v = np.zeros(P.V, bool)
        v[P.ev_vi[ev[col].values]] = True
        return v

    subsets = {"s_real": vsel_from("s_real"), "s_offubi": vsel_from("s_offubi"), "s_audio": vsel_from("s_audio")}
    for name, sk in (("xd_movie", "xd_movie"), ("xd_youtube", "xd_youtube"), ("ucf", "ucf"), ("ubi", "ubi")):
        subsets[f"source_{name}"] = P.src_v == SRC_CODE[sk]
    classes = {}
    for vid, cls in zip(P.vids, P.ss.bench.videos.loc[P.vids, "classes"] if len(P.vids) else []):
        classes[vid] = str(cls).split("|")
    for g, members in CLASS_GROUPS.items():
        subsets[f"class_{g}"] = np.array([any(c in members for c in classes.get(v, [])) for v in P.vids])
    for name, v in subsets.items():
        pos = P.band_pos(30, 120, vsel=v & P.first_video_v)
        if pos.any():
            out[name] = P.auc(s, P.pair_def(("bd", name), pos, neg), W)
    return out


def person_stratified(P, s, persons, neg, W):
    """M2: matching also on a person-count stratum (0 / 1 / 2 / 3-5 / 6+), and the n_persons >= 2 restriction."""
    strata = np.digitize(persons, [1, 2, 3, 6])
    pos = P.band_pos(30, 120)
    out = {"stratified": P.auc(s, P.pair_def("m2", pos, neg, strata=strata), W)}
    two = persons >= 2
    out["n_persons_ge2"] = P.auc(s, P.pair_def("m2_ge2", pos & two, neg & two), W)
    return out
