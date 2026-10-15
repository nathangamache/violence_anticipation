"""Probe training for the screen: targets, time-matched training plans, L2 logistic regression with C chosen on the
fold's validation slice, the nuisance model N, the shuffled-video and position controls, and the per-block runner."""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from anticipation.eval.va30 import SOURCE_KINDS, TIME_MATCH_EDGES, time_match_weights, time_strata
from anticipation.screen import features as F

C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
DEFAULT_C = 0.1
MAX_TRAIN_POS = 25000
MAX_TRAIN_NEG = 50000
MAX_VAL_NEG = 60000
MAX_PLS_ROWS = 40000
MAX_THR_ROWS = 300000
CLIP_Z = 10.0
SCENE_PCA = 32
SCENE_T_SEC = 2.0
SRC_CODE = {s: i for i, s in enumerate(SOURCE_KINDS)}
TARGETS = ("band", "band_b", "near", "later", "detect")

MODELS = {
    "p1": ("p1", "band"), "p1_near": ("p1", "near"), "p1_later": ("p1", "later"), "b_detect": ("p1", "detect"),
    "p1s": ("p1s", "band"), "p1b": ("p1b", "band_b"), "n_plus_p1": ("n_plus_p1", "band"),
}


def target_masks(vt, target):
    ok = ~vt.masked
    clean = vt.kind == "clean"
    viol = vt.kind == "violent"
    neg = np.full(vt.n, clean) & ok & (vt.remaining >= 60)
    if target in ("band", "band_b"):
        pos = viol & vt.pre & ok & (vt.tte > 30) & (vt.tte <= 120)
        if target == "band_b":
            late = vt.t > F.BASELINE_SEC
            pos, neg = pos & late, neg & late
    elif target == "near":
        pos = viol & vt.pre & ok & (vt.tte > 0) & (vt.tte <= 10)
    elif target == "later":
        pos = viol & ~vt.pre & ok & (vt.tte > 30) & (vt.tte <= 120)
    elif target == "detect":
        pos = viol & vt.in_violence
        neg = np.full(vt.n, clean) & ok
    else:
        raise ValueError(target)
    return pos & np.full(vt.n, viol), neg


def weighted_auc(score, y, w):
    """Weighted AUROC of ``score`` for boolean ``y`` (ties count half); NaN when a class is empty."""
    y = np.asarray(y, bool)
    w = np.asarray(w, np.float64)
    if not y.any() or y.all() or w[~y].sum() <= 0:
        return np.nan
    _, g = np.unique(score, return_inverse=True)
    pw = np.bincount(g, weights=w * y)
    nw = np.bincount(g, weights=w * ~y)
    below = np.cumsum(nw) - nw
    return float((pw * (below + 0.5 * nw)).sum() / (pw.sum() * nw.sum()))


class Plan:
    """Training or validation rows of one target: (video index, tick index, label, weight), time matched."""

    def __init__(self, vts, target, rng, max_pos, max_neg, source_of):
        vi, ti, tt, yy, ss = [], [], [], [], []
        for i, vt in enumerate(vts):
            p, n = target_masks(vt, target)
            sel = np.flatnonzero(p | n)
            vi.append(np.full(len(sel), i))
            ti.append(sel)
            tt.append(vt.t[sel])
            yy.append(p[sel])
            ss.append(np.full(len(sel), SRC_CODE.get(source_of(vt.vid), -1)))
        self.empty = True
        self.by_video = {}
        if not vi or not sum(map(len, ti)):
            return
        vi, ti, tt, yy, ss = (np.concatenate(a) for a in (vi, ti, tt, yy, ss))
        w, pos_ok = time_match_weights(tt, yy, ~yy, TIME_MATCH_EDGES, source=ss)
        pos_idx = np.flatnonzero(pos_ok)
        neg_idx = np.flatnonzero((~yy) & (w > 0))
        if not len(pos_idx) or not len(neg_idx):
            return
        if max_pos and len(pos_idx) > max_pos:
            pos_idx = np.sort(rng.choice(pos_idx, max_pos, replace=False))
        if max_neg and len(neg_idx) > max_neg:
            neg_idx = np.sort(rng.choice(neg_idx, max_neg, replace=False))
        wn = w[neg_idx] * (len(pos_idx) / w[neg_idx].sum())
        idx = np.concatenate([pos_idx, neg_idx])
        weight = np.concatenate([np.ones(len(pos_idx)), wn])
        weight *= len(weight) / weight.sum()
        order = np.argsort(idx, kind="stable")
        idx, weight = idx[order], weight[order]
        self.n_pos, self.n_neg = len(pos_idx), len(neg_idx)
        v_sel, t_sel, y_sel = vi[idx], ti[idx], yy[idx]
        for i in np.unique(v_sel):
            m = v_sel == i
            self.by_video[vts[i].vid] = (t_sel[m], y_sel[m], weight[m])
        self.empty = False


class Probe:
    def __init__(self, mean, scale, coef, intercept, C, val_auc, converged):
        self.mean, self.scale, self.coef, self.intercept = mean, scale, coef, intercept
        self.C, self.val_auc, self.converged = C, val_auc, converged

    def score(self, X):
        z = np.clip((np.asarray(X, np.float32) - self.mean) / self.scale, -CLIP_Z, CLIP_Z)
        return (z @ self.coef + self.intercept).astype(np.float64)


def fit_probe(X, y, w, val=None, cs=C_GRID):
    """L2 logistic regression path; C is the value with the best validation callback score (ties: smaller C)."""
    mean = X.mean(0)
    scale = np.maximum(X.std(0), 1e-6)
    Z = np.clip((X - mean) / scale, -CLIP_Z, CLIP_Z).astype(np.float64)
    best, results = None, []
    clf = LogisticRegression(solver="lbfgs", max_iter=300, tol=1e-4, warm_start=True)
    for C in sorted(cs):
        clf.set_params(C=C)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit(Z, y.astype(int), sample_weight=w)
        probe = Probe(mean.astype(np.float32), scale.astype(np.float32), clf.coef_[0].copy(), float(clf.intercept_[0]),
                      C, np.nan, not any(issubclass(c.category, ConvergenceWarning) for c in caught))
        auc = val(probe) if val is not None else np.nan
        probe.val_auc = auc
        results.append({"C": C, "val_auc": None if not np.isfinite(auc) else float(auc), "converged": probe.converged})
        if np.isfinite(auc) and (best is None or auc > best.val_auc + 1e-12):
            best = probe
    if best is None:
        clf.set_params(C=DEFAULT_C)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit(Z, y.astype(int), sample_weight=w)
        best = Probe(mean.astype(np.float32), scale.astype(np.float32), clf.coef_[0].copy(), float(clf.intercept_[0]),
                     DEFAULT_C, np.nan, True)
        best.fallback = True
    best.path = results
    return best


class Nuisance:
    """Source-kind x time-bin indicators plus the static scene embedding (PCA fitted on training videos)."""

    def __init__(self, source, train_vids, k=SCENE_PCA):
        self.source = source
        embs = []
        for v in train_vids:
            e = source.scene(v)
            if e is not None:
                embs.append(e)
        self.n_scene = len(embs)
        self.dim = len(embs[0]) if embs else 0
        self.k = min(k, max(self.n_scene - 1, 0), self.dim)
        if self.k > 0:
            E = np.stack(embs).astype(np.float64)
            self.mean = E.mean(0)
            _, _, vt = np.linalg.svd(E - self.mean, full_matrices=False)
            self.comp = vt[:self.k].T.astype(np.float32)
        self.missing = []
        self._cache = {}

    def scene_vec(self, vid):
        if vid not in self._cache:
            e = self.source.scene(vid) if self.k > 0 else None
            if e is None:
                self.missing.append(vid)
                self._cache[vid] = (np.zeros(self.k, np.float32), 1.0)
            else:
                self._cache[vid] = (((e - self.mean) @ self.comp).astype(np.float32), 0.0)
        return self._cache[vid]

    def features(self, vt, src):
        T = vt.n
        nb = len(TIME_MATCH_EDGES) - 1
        bins = np.clip(np.searchsorted(np.asarray(TIME_MATCH_EDGES[1:-1]), vt.t, side="right"), 0, nb - 1)
        hot = np.zeros((T, len(SOURCE_KINDS) * nb), np.float32)
        hot[np.arange(T), SRC_CODE.get(src, 0) * nb + bins] = 1
        z, miss = self.scene_vec(vt.vid)
        z = np.tile(z, (T, 1))
        z[vt.t < SCENE_T_SEC] = 0
        return np.concatenate([hot, z, np.full((T, 1), miss, np.float32)], axis=1)


def time_source_rates(vts, source_of):
    """Position-only control: P(band positive | source kind, time-match bin) on training ticks, add-half smoothed."""
    pos, tot = {}, {}
    nb = len(TIME_MATCH_EDGES) - 1
    for vt in vts:
        p, n = target_masks(vt, "band")
        m = p | n
        if not m.any():
            continue
        src = source_of(vt.vid)
        b = time_strata(vt.t[m])
        tot[src] = tot.get(src, np.zeros(nb)) + np.bincount(b, minlength=nb)
        pos[src] = pos.get(src, np.zeros(nb)) + np.bincount(b, weights=p[m].astype(float), minlength=nb)
    prior = (sum(pos.values()) + 0.5) / (sum(tot.values()) + 1.0) if tot else 0.5
    return {s: (pos[s] + 0.5) / (tot[s] + 1.0) for s in tot}, prior


def time_source_score(rates, prior, vt, src):
    r = rates.get(src)
    b = time_strata(vt.t)
    return r[b] if r is not None else np.full(vt.n, float(np.mean(prior)))


class CutStreamError(RuntimeError):
    pass


class BlockRun:
    """Runs every probe and control of one block on the screening folds."""

    def __init__(self, ss, source, block, seed=0, max_pos=MAX_TRAIN_POS, max_neg=MAX_TRAIN_NEG,
                 wide_columns=F.WIDE_COLUMNS, n_components=F.PLS_COMPONENTS, cs=C_GRID, arms=("p1", "p1s", "p1b"),
                 require_cut=True):
        self.ss, self.source, self.block = ss, source, block
        self.seed, self.max_pos, self.max_neg = seed, max_pos, max_neg
        self.wide_columns, self.n_components, self.cs, self.arms = wide_columns, n_components, cs, arms
        self._raw, self._cut, self._vt, self._vf, self.columns = {}, {}, {}, {}, None
        self.require_cut, self.cut_incomplete = require_cut, []
        self.width = None

    def src(self, vid):
        return self.ss.source_kind(vid)

    def covered(self, vid):
        if self.block.sources and self.src(vid) not in self.block.sources:
            return False
        if self.block.audio and vid in self.ss.bench.videos.index and not self.ss.bench.videos.loc[vid, "has_audio"]:
            return False
        return all(self.source.has(s, vid) for s in self.block.streams)

    def raw(self, vid):
        if vid not in self._raw:
            n = self.ss.ticks(vid).n
            x, vf, names = F.load_block(self.source, self.block, vid, n)
            self.columns = names
            self.width = x.shape[1]
            self._raw[vid] = x.astype(np.float16) if x.shape[1] > self.wide_columns else x
            self._vf[vid] = vf
            self._vt[vid] = self.ss.ticks(vid).cut(len(x))
        x = self._raw[vid]
        return x.astype(np.float32) if x.dtype != np.float32 else x

    def vt(self, vid):
        self.raw(vid)
        return self._vt[vid]

    def cut_last(self, vid):
        if vid not in self._cut:
            flags, ok = F.cut_flags(self.source, vid, self.vt(vid).n)
            if not ok:
                if self.require_cut:
                    raise CutStreamError(f"{vid}: motion cut stream missing or shorter than the scored ticks")
                self.cut_incomplete.append(vid)
            self._cut[vid] = (F.last_cut_index(flags), ok)
        return self._cut[vid][0]

    def release(self, vid):
        self._raw.pop(vid, None)

    def set_fold(self, train_vids):
        """Fit the PLS reduction (wide blocks) and the clean-training 95th percentiles for this fold."""
        self.train_vids = train_vids
        self.pls = None
        self.wide = None
        for v in train_vids:
            self.raw(v)
            self.wide = self.width > self.wide_columns
            break
        rng = np.random.default_rng(self.seed)
        if self.wide:
            plan = Plan([self.vt(v) for v in train_vids], "band", rng, MAX_PLS_ROWS // 2, MAX_PLS_ROWS // 2, self.src)
            if plan.empty:
                raise ValueError("no band rows to fit the PLS reduction")
            X, y, w = [], [], []
            for v, (ti, yy, ww) in plan.by_video.items():
                X.append(self.raw(v)[ti])
                y.append(yy)
                w.append(ww)
            self.pls = F.Pls.fit(np.concatenate(X), np.concatenate(y).astype(float), np.concatenate(w),
                                 self.n_components)
        rows, rows_b = [], []
        for v in train_vids:
            vt = self.vt(v)
            if vt.kind != "clean":
                continue
            x = self.xform(v)
            keep = ~vt.masked
            rows.append(x[keep])
            rows_b.append(F.early_baseline_subtract(x)[keep & (vt.t > F.BASELINE_SEC)])
        if not rows:
            raise ValueError("no clean training ticks")
        step = max(1, int(np.ceil(sum(map(len, rows)) / MAX_THR_ROWS)))
        self.thr = {"p1": np.percentile(np.concatenate(rows)[::step], 95, axis=0).astype(np.float32),
                    "p1b": (np.percentile(np.concatenate(rows_b)[::step], 95, axis=0) if rows_b and sum(map(len, rows_b))
                    else np.zeros(rows[0].shape[1])).astype(np.float32)}
        self.thr["p1s"] = self.thr["p1"]

    def xform(self, vid):
        x = self.raw(vid)
        return self.pls.transform(x).astype(np.float32) if self.pls is not None else x

    def feats(self, vid, arm):
        x = self.xform(vid)
        cut = self.cut_last(vid)
        if arm == "p1b":
            x = F.early_baseline_subtract(x)
            cut = np.maximum(cut, F.BASELINE_SEC - 1)
        return F.window_features(x, cut, self.thr[arm],
                                 slopes=F.SLOPE_WINDOWS if arm == "p1s" else ())

    def multi(self, vid, nuis=None):
        out = {a: self.feats(vid, a) for a in self.arms}
        if nuis is not None and "p1" in out:
            vt = self.vt(vid)
            out["n_plus_p1"] = np.concatenate([nuis.features(vt, self.src(vid)), out["p1"]], axis=1)
        return out

    def run_fold(self, fold, train, val, test, nuis=None, log=None):
        rng = np.random.default_rng(self.seed + 1000 * fold)
        self.set_fold(train)
        vts_tr = [self.vt(v) for v in train]
        vts_va = [self.vt(v) for v in val]
        plans_tr = {t: Plan(vts_tr, t, rng, self.max_pos, self.max_neg, self.src) for t in TARGETS}
        plans_va = {t: Plan(vts_va, t, rng, None, MAX_VAL_NEG, self.src) for t in TARGETS}
        models = {m: spec for m, spec in MODELS.items() if spec[0] in (self.arms + (("n_plus_p1",) if nuis else ()))}
        data = {m: ([], [], []) for m in models}
        vdata = {m: ([], [], []) for m in models}
        for vids, plans, store in ((train, plans_tr, data), (val, plans_va, vdata)):
            for v in vids:
                need = [m for m, (_, t) in models.items() if v in plans[t].by_video]
                if not need:
                    continue
                Fm = self.multi(v, nuis)
                for m in need:
                    key, t = models[m]
                    ti, yy, ww = plans[t].by_video[v]
                    store[m][0].append(Fm[key][ti])
                    store[m][1].append(yy)
                    store[m][2].append(ww)
        probes, info = {}, {}
        for m in models:
            if not data[m][0]:
                info[m] = {"status": "no_training_rows"}
                continue
            X, y, w = (np.concatenate(a) for a in data[m])
            if vdata[m][0]:
                Xv, yv, wv = (np.concatenate(a) for a in vdata[m])
                val_fn = lambda p, Xv=Xv, yv=yv, wv=wv: weighted_auc(p.score(Xv), yv, wv)  # noqa: E731
            else:
                val_fn = None
            probes[m] = fit_probe(X, y, w, val_fn, self.cs)
            info[m] = {"C": probes[m].C, "val_auc": None if not np.isfinite(probes[m].val_auc) else float(probes[m].val_auc),
                       "converged": probes[m].converged, "path": probes[m].path, "n_train": int(len(y)),
                       "n_train_pos": int(y.sum()), "n_val": int(sum(map(len, vdata[m][1]))),
                       "val_fallback": bool(getattr(probes[m], "fallback", False))}
        del data, vdata
        rates, prior = time_source_rates(vts_tr, self.src)
        scores = {m: {"test": {}, "val": {}} for m in list(probes) + ["shuffled", "time_source"]}
        clean_test = {}
        for v in test:
            if self.vt(v).kind == "clean":
                clean_test.setdefault(self.src(v), []).append(v)
        for split, vids in (("test", test), ("val", val)):
            for v in vids:
                vt = self.vt(v)
                Fm = self.multi(v, nuis)
                for m, p in probes.items():
                    scores[m][split][v] = p.score(Fm[models[m][0]])
                scores["time_source"][split][v] = time_source_score(rates, prior, vt, self.src(v))
                if split == "test" and "p1" in probes:
                    donors = [u for u in clean_test.get(self.src(v), []) if u != v] or [u for us in clean_test.values() for u in us if u != v]
                    if donors:
                        u = donors[(sum(map(ord, v)) + self.seed) % len(donors)]
                        d = self.feats(u, "p1")
                        idx = np.minimum(np.arange(vt.n), len(d) - 1)
                        scores["shuffled"]["test"][v] = probes["p1"].score(d[idx])
        return scores, info


def run_nuisance(ss, source, train, val, test, seed=0, cs=C_GRID, max_pos=MAX_TRAIN_POS, max_neg=MAX_TRAIN_NEG):
    """Model N for one fold: source x time-bin indicators plus the PCA-reduced scene embedding."""
    rng = np.random.default_rng(seed)
    src = ss.source_kind
    nuis = Nuisance(source, train)
    vt = ss.ticks
    plan_tr = Plan([vt(v) for v in train], "band", rng, max_pos, max_neg, src)
    plan_va = Plan([vt(v) for v in val], "band", rng, None, MAX_VAL_NEG, src)
    gather = lambda plan: [np.concatenate(a) for a in zip(*[  # noqa: E731
        (nuis.features(vt(v), src(v))[ti], yy, ww) for v, (ti, yy, ww) in plan.by_video.items()])]
    if plan_tr.empty:
        return None, {"status": "no_training_rows"}, nuis
    X, y, w = gather(plan_tr)
    val_fn = None
    if not plan_va.empty:
        Xv, yv, wv = gather(plan_va)
        val_fn = lambda p: weighted_auc(p.score(Xv), yv, wv)  # noqa: E731
    probe = fit_probe(X, y, w, val_fn, cs)
    scores = {"test": {v: probe.score(nuis.features(vt(v), src(v))) for v in test},
              "val": {v: probe.score(nuis.features(vt(v), src(v))) for v in val}}
    info = {"C": probe.C, "val_auc": None if not np.isfinite(probe.val_auc) else float(probe.val_auc),
            "scene_videos_fit": nuis.n_scene, "scene_missing": sorted(set(nuis.missing))}
    return scores, info, nuis
