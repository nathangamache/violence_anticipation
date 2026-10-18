"""Pre-registered forward selection of screen blocks, the P1/P2 comparison and the frozen final model (docs/FUSION.md).

    nice -n 10 .venv/bin/python -m anticipation.screen.fusion --out benchmark/results/fusion
    nice -n 10 .venv/bin/python -m anticipation.screen.fusion --out benchmark/results/fusion --from_p1
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse
import copy
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from torch.utils.data import Dataset

from anticipation.data.catalog import REPO_ROOT
from anticipation.eval import va30
from anticipation.models.objectives import TRAIN_HAZARD_EDGES
from anticipation.models.temporal import CausalTemporalModel, fit_temporal
from anticipation.screen import features as F
from anticipation.screen import metrics as M
from anticipation.screen import probes as P
from anticipation.screen import run as R
from anticipation.screen import sets as S

DEFAULT_OUT = os.path.join(REPO_ROOT, "benchmark", "results", "fusion")
DECISIONS_PATH = os.path.join(REPO_ROOT, "benchmark", "results", "cue_screen", "decisions_phase_A.json")
SELECTABLE = ("ANTICIPATION", "PROVISIONAL")
P2_MAX_POS, P2_MAX_NEG, P2_MAX_VAL_NEG = 10000, 10000, 6000
P2_BAG_TICKS = 16
P2_HISTORY_DROPOUT = 0.25
P2_MODALITY_DROPOUT = 0.15
P2_DROPOUT = 0.1
P2_BATCH = 32
P2_BAG_BATCH = 2
SCORE_BATCH = 256
FINAL_RULE = ("P2 replaces P1 only if its seed-ensemble M8 gain over P1 has a paired-bootstrap CI lower bound > 0; "
              "otherwise P1 is kept")


def default_prereg():
    """screen_set_v2.json when it exists, else the v1 registration."""
    v2 = os.path.join(os.path.dirname(R.prereg.PREREG_PATH), "screen_set_v2.json")
    return v2 if os.path.isfile(v2) else R.prereg.PREREG_PATH


def log(*a):
    print("[fusion]", *a, flush=True)


def sd(stat, null=None):
    return None if stat is None else stat.to_dict(null)


def snap_median_c(cs):
    """The grid value nearest (in log10) to the median of per-fold C values."""
    m = float(np.median(np.log10(cs)))
    return float(min(P.C_GRID, key=lambda c: abs(np.log10(c) - m)))


def load_candidates(path, prereg_sha):
    """ANTICIPATION and PROVISIONAL block names from a complete, current decisions file."""
    with open(path) as f:
        d = json.load(f)
    if d.get("prereg_sha256") != prereg_sha:
        raise ValueError(f"{path} was produced under another pre-registration")
    if d.get("family_status") != "COMPLETE":
        raise ValueError(f"{path}: family status {d.get('family_status')}")
    by = {c: sorted(n for n, r in d["decisions"].items() if r["class"] == c) for c in SELECTABLE}
    unknown = [n for v in by.values() for n in v if n not in F.BLOCKS]
    if unknown:
        raise ValueError(f"no block registered for {unknown}")
    return by, d["decisions"]


def cat(parts):
    return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)


class FoldCache:
    """Cached P1 design rows and full-length window features of one fold, per block."""


class FusionData:
    """P1 design matrices of several blocks on one common cohort, built once per fold."""

    def __init__(self, ss, source, names, seed=0, require_cut=True, max_pos=P.MAX_TRAIN_POS, max_neg=P.MAX_TRAIN_NEG,
                 folds=None, include_h1=True, targets=("band",)):
        self.ss, self.source, self.names, self.seed = ss, source, list(names), seed
        self.targets = tuple(targets)
        self.max_pos, self.max_neg = max_pos, max_neg
        self.folds = tuple(ss.screen_folds if folds is None else folds)
        self.kinds = ("violent", "clean", "h1") if include_h1 else ("violent", "clean")
        self.runners = {n: P.BlockRun(ss, source, F.BLOCKS[n], seed=seed, require_cut=require_cut) for n in self.names}
        self.fc = {}

    def covered(self, v):
        return all(r.covered(v) for r in self.runners.values())

    def common_ticks(self, v):
        n = min(r.vt(v).n for r in self.runners.values())
        return self.ss.ticks(v).cut(n)

    def cohort(self, f):
        return {"train": [v for v in self.ss.train_ids(f) if self.covered(v)],
                "val": [v for v in self.ss.val_ids(f) if self.covered(v)],
                "test": [v for v in self.ss.test_ids(f, kinds=self.kinds) if self.covered(v)]}

    def plan_rows(self, plan, vt, nuis):
        rows = {n: [] for n in self.names}
        nrows, ys, ws, order = [], [], [], []
        for v, (ti, yy, ww) in plan.by_video.items():
            for n, r in self.runners.items():
                rows[n].append(r.feats(v, "p1")[:vt[v].n][ti])
            if nuis is not None:
                nrows.append(nuis.features(vt[v], self.ss.source_kind(v))[ti])
            ys.append(yy)
            ws.append(ww)
            order.append(v)
        out = {"X": {n: np.concatenate(a) for n, a in rows.items()}, "y": np.concatenate(ys), "w": np.concatenate(ws),
               "videos": order}
        out["nuis"] = np.concatenate(nrows) if nuis is not None else None
        return out

    def build_fold(self, f):
        ss = self.ss
        co = self.cohort(f)
        if not co["train"] or not co["val"] or not co["test"]:
            raise ValueError(f"fold {f}: missing train/val/test coverage")
        log(f"fold {f}: train {len(co['train'])} val {len(co['val'])} test {len(co['test'])}")
        n_scores, n_info, nuis = P.run_nuisance(ss, self.source, co["train"], co["val"], co["test"])
        for r in self.runners.values():
            r.set_fold(co["train"])
        c = FoldCache()
        c.fold, c.cohort, c.nuis_scores, c.nuis_info, c.nuis = f, co, n_scores, n_info, nuis
        c.vt = {v: self.common_ticks(v) for v in co["train"] + co["val"] + co["test"]}
        rng = np.random.default_rng(self.seed + 1000 * f)
        vts_tr = [c.vt[v] for v in co["train"]]
        vts_va = [c.vt[v] for v in co["val"]]
        plans_tr = {t: P.Plan(vts_tr, t, rng, self.max_pos, self.max_neg, ss.source_kind) for t in P.TARGETS}
        plans_va = {t: P.Plan(vts_va, t, rng, None, P.MAX_VAL_NEG, ss.source_kind) for t in P.TARGETS}
        c.plans_tr, c.plans_va, c.rows = plans_tr, plans_va, {}
        for t in self.targets:
            if plans_tr[t].empty:
                raise ValueError(f"fold {f}: no {t} training rows")
            c.rows[t] = {"tr": self.plan_rows(plans_tr[t], c.vt, nuis),
                         "va": self.plan_rows(plans_va[t], c.vt, nuis) if not plans_va[t].empty else None}
        c.full, c.nuis_full = {n: {} for n in self.names}, {}
        for v in co["val"] + co["test"]:
            for n, r in self.runners.items():
                c.full[n][v] = r.feats(v, "p1")[:c.vt[v].n]
            c.nuis_full[v] = nuis.features(c.vt[v], ss.source_kind(v))
        self.fc[f] = c
        return c

    def build(self):
        for f in self.folds:
            self.build_fold(f)

    def fit_set(self, f, names, with_n=False, target="band"):
        """P1 on the concatenation of ``names`` (optionally after the nuisance columns) for one fold; ``names`` may be
        empty with ``with_n`` (the nuisance model alone)."""
        c = self.fc[f]
        tr, va = c.rows[target]["tr"], c.rows[target]["va"]
        parts = lambda d: ([d["nuis"]] if with_n else []) + [d["X"][n] for n in names]  # noqa: E731
        X = cat(parts(tr))
        val_fn = None
        if va is not None:
            Xv = cat(parts(va))
            val_fn = lambda p: P.weighted_auc(p.score(Xv), va["y"], va["w"])  # noqa: E731
        probe = P.fit_probe(X, tr["y"], tr["w"], val_fn, P.C_GRID)
        scores = {"test": {}, "val": {}}
        for split in ("test", "val"):
            for v in c.cohort[split]:
                fe = ([c.nuis_full[v]] if with_n else []) + [c.full[n][v] for n in names]
                scores[split][v] = probe.score(cat(fe))
        info = {"C": probe.C, "val_auc": None if not np.isfinite(probe.val_auc) else float(probe.val_auc),
                "converged": probe.converged, "path": probe.path, "n_train": int(len(tr["y"])),
                "n_train_pos": int(tr["y"].sum()), "n_columns": int(X.shape[1])}
        return scores, info


def pool_stats(pool, W, score, with_first=True):
    """M1 (band, and 30-60 s) and the window-max event AUROC M8 of one pooled score vector."""
    neg = pool.clean_neg()
    out = {"m1": pool.auc(score, pool.pair_def(("band", False), pool.band_pos(30, 120), neg), W),
           "m1_30_60": pool.auc(score, pool.pair_def(("band_30_60", False), pool.band_pos(30, 60), neg), W)}
    if len(pool.events):
        xe, cm = M._event_values(pool, score, "max"), M._clean_windows(pool, score, "max")
        out["m8"] = M._event_stat(pool, xe, cm, W, "max")
        if with_first:
            out["m8_first"] = M._event_stat(pool, xe, cm, W, "max", pool.events["first"].values)
    return out


def fit_fixed(X, y, w, C):
    """L2 logistic probe at a given C (no validation slice)."""
    return P.fit_probe(X, y, w, lambda p: 1.0, (C,))


def near_event_pool(pool):
    """A shallow copy of ``pool`` whose event windows are [max(t0 - 10, clear_start), t0) for M8-near."""
    p = copy.copy(pool)
    ev = pool.events.copy()
    if len(ev):
        ev["win_lo"] = np.maximum(ev["onset"].values - 10.0, ev["clear_start"].values)
        ev["win_hi"] = ev["onset"].values
        p.ev_ia = np.maximum(np.ceil(ev["win_lo"].values - 1e-9).astype(np.int64), 1) - 1
        p.ev_ib = np.ceil(ev["win_hi"].values - 1e-9).astype(np.int64) - 2
    p.events = ev
    p._defs = {}
    return p


def pool_stats_near(pool, ev_pool, W, score, extra=True):
    """Imminent-head statistics: M1-near, M3p-near (UCF vs H1p ticks within 10 s), M8-near and, with ``extra``, the
    official/UBI-onset breakdowns."""
    neg = pool.clean_neg()
    pos = pool.band_pos(0, 10)
    out = {"m1": pool.auc(score, pool.pair_def("near_head", pos, neg), W)}
    h1n = (pool.kind == M.KIND_CODE["h1"]) & (pool.tte > 0) & (pool.tte <= 10) & pool.h1p_v[pool.vidx]
    if h1n.any():
        out["m3p"] = pool.auc(score, pool.pair_def("m3p_near", pos & (pool.src == M.SRC_CODE["ucf"]), h1n, source=False), W)
    if len(ev_pool.events):
        xe, cm = M._event_values(ev_pool, score, "max"), M._clean_windows(ev_pool, score, "max")
        out["m8"] = M._event_stat(ev_pool, xe, cm, W, "max")
        if extra:
            off = ev_pool.events["onset_source"].isin(["official", "ubi"]).values
            out["m8_offubi"] = M._event_stat(ev_pool, xe, cm, W, "max", off & ev_pool.events["first"].values)
            v = np.zeros(pool.V, bool)
            v[ev_pool.ev_vi[off & ev_pool.events["first"].values]] = True
            if (pos & v[pool.vidx]).any():
                out["m1_offubi"] = pool.auc(score, pool.pair_def("near_off", pos & v[pool.vidx], neg), W)
    return out


class Evaluator:
    """Pooled screening-fold statistics (M1, M8, M4n) of score sets with one shared bootstrap.

    ``head`` is ``early`` (band target 30-120 s) or ``imminent`` (target 0-10 s, M1-near/M3p-near/M8-near)."""

    def __init__(self, data, n_boot=1000, head="early"):
        self.data, self.head = data, head
        self.target = "band" if head == "early" else "near"
        vts = [data.fc[f].vt[v] for f in data.folds for v in data.fc[f].cohort["test"]]
        self.pool = M.Pool(data.ss, vts)
        self.ev_pool = near_event_pool(self.pool) if head == "imminent" else self.pool
        self.W = self.pool.boot_weights(n_boot, seed=0)
        self.cache = {}
        self.n_stats = None

    def pooled(self, per_fold):
        d = {}
        for f in self.data.folds:
            d.update(per_fold[f]["test"])
        return self.pool.assemble(d)

    def stats(self, score, with_first=True):
        if self.head == "imminent":
            return pool_stats_near(self.pool, self.ev_pool, self.W, score, with_first)
        return pool_stats(self.pool, self.W, score, with_first)

    def nuisance(self):
        if self.n_stats is None:
            if self.head == "imminent":
                per = {f: self.data.fit_set(f, [], with_n=True, target="near")[0] for f in self.data.folds}
            else:
                per = {f: {"test": self.data.fc[f].nuis_scores["test"]} for f in self.data.folds}
            self.n_stats = self.stats(self.pooled(per))
        return self.n_stats

    def detect(self, names):
        """B-detect: the violence-now probe on the same blocks, applied to pre-onset ticks (M1-near)."""
        per = {f: self.data.fit_set(f, names, target="detect")[0] for f in self.data.folds}
        return self.stats(self.pooled(per), with_first=False)["m1"]

    def evaluate(self, names):
        key = tuple(names)
        if key not in self.cache:
            t0 = time.time()
            per_fold, infos = {}, {}
            for f in self.data.folds:
                per_fold[f], infos[f] = self.data.fit_set(f, names, target=self.target)
            self.cache[key] = {"scores": per_fold, "info": infos, "stats": self.stats(self.pooled(per_fold)),
                               "seconds": round(time.time() - t0, 1)}
            log(f"{'+'.join(names)}: M1 {self.cache[key]['stats']['m1'].value:.4f} "
                f"M8 {self.cache[key]['stats']['m8'].value:.4f} ({time.time() - t0:.0f} s)")
        return self.cache[key]

    def vs_nuisance(self, names, res=None):
        """M4n (gain in M1 of N + set over N) and the M8 gain of the set over N."""
        res = res or self.evaluate(names)
        per_fold = {}
        for f in self.data.folds:
            per_fold[f], _ = self.data.fit_set(f, names, with_n=True, target=self.target)
        s_np = self.stats(self.pooled(per_fold), with_first=False)
        n = self.nuisance()
        return {"m4n": M.diff(s_np["m1"], n["m1"]), "m1_n_plus_set": s_np["m1"],
                "m8_gain_over_n": M.diff(res["stats"]["m8"], n["m8"]), "m1_n": n["m1"], "m8_n": n["m8"]}


def stat_row(s):
    return {k: sd(v) for k, v in s.items()}


def mean_path_c(infos):
    """The C with the best mean validation score over folds (ties: smaller C)."""
    tot = {}
    for f, i in infos.items():
        for p in i["path"]:
            if p["val_auc"] is not None:
                tot.setdefault(p["C"], []).append(p["val_auc"])
    best = max(sorted(tot), key=lambda c: (np.mean(tot[c]), -c))
    return float(best), {str(c): float(np.mean(v)) for c, v in sorted(tot.items())}


def forward_select(ev, anticipation, provisional):
    """Greedy P1 forward selection with the paired M8 gain rule (docs/FUSION.md section 2)."""
    steps = []
    start_stats = {}
    for n in anticipation:
        r = ev.evaluate([n])
        start_stats[n] = r
    start = max(anticipation, key=lambda n: (start_stats[n]["stats"]["m8"].value, n))
    steps.append({"step": 0, "action": "start", "rule": "best ANTICIPATION block by M8 point estimate (the frozen file "
                  "does not define 'best')", "chosen": start,
                  "candidates": {n: {**stat_row(start_stats[n]["stats"]), "C": {str(f): i["C"] for f, i in start_stats[n]["info"].items()}}
                                 for n in anticipation}})
    selected = [start]
    remaining = sorted((set(anticipation) | set(provisional)) - {start})
    k = 0
    while remaining:
        k += 1
        cur = ev.evaluate(selected)
        cands = {}
        for n in remaining:
            r = ev.evaluate(selected + [n])
            g = M.diff(r["stats"]["m8"], cur["stats"]["m8"])
            cands[n] = {"gain": g, "res": r}
        best = max(remaining, key=lambda n: (cands[n]["gain"].value if cands[n]["gain"].value is not None else -9, n))
        g = cands[best]["gain"]
        ok = g.lower is not None and g.lower > 0
        steps.append({"step": k, "action": "add" if ok else "stop", "chosen": best if ok else None,
                      "best_candidate": best, "selected_before": list(selected),
                      "candidates": {n: {"m8_gain": sd(c["gain"], 0.0), **stat_row(c["res"]["stats"]),
                                         "C": {str(f): i["C"] for f, i in c["res"]["info"].items()}} for n, c in cands.items()}})
        if not ok:
            break
        selected.append(best)
        remaining.remove(best)
    return selected, steps


class GruRows(Dataset):
    """Band windows (video, tick, label, weight) with the item layout of ``models.temporal.WindowDataset``."""

    def __init__(self, store, rows, dims, ctx, history_dropout=0.0):
        self.store, self.rows, self.dims, self.ctx, self.hd = store, rows, dims, ctx, history_dropout
        self.edges = np.cumsum([0] + list(dims))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        v, tick, y, w = self.rows[i]
        length = int(np.random.randint(1, min(self.ctx, tick + 1) + 1)) if self.hd and np.random.random() < self.hd else None
        x, hist = window(self.store[v], tick, self.ctx, length)
        return {"streams": [x[:, a:b] for a, b in zip(self.edges[:-1], self.edges[1:])],
                "present": np.ones((self.ctx, len(self.dims)), np.float32), "history": hist,
                "position": np.float32(0), "hazard": np.full(len(TRAIN_HAZARD_EDGES) - 1, -1, np.float32), "now": np.float32(0),
                "now_valid": np.float32(0), "weight": np.float32(1), "band": np.float32(y), "band_weight": np.float32(w)}


class GruBags(Dataset):
    """Event bags of the multiple-instance loss: positive window ticks and matched clean ticks, one bag each."""

    def __init__(self, store, bags, dims, ctx):
        self.rows = GruRows(store, [], dims, ctx)
        self.store, self.bags, self.dims, self.ctx = store, bags, dims, ctx

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, i):
        v, ticks, target = self.bags[i]
        mask = np.arange(P2_BAG_TICKS) < len(ticks)
        padded = np.pad(ticks, (0, P2_BAG_TICKS - len(ticks)), mode="edge")
        items = []
        for t in padded:
            x, hist = window(self.store[v], int(t), self.ctx, None)
            items.append((x, hist))
        edges = self.rows.edges
        return {"streams": [np.stack([x[:, a:b] for x, _ in items]) for a, b in zip(edges[:-1], edges[1:])],
                "present": np.ones((P2_BAG_TICKS, self.ctx, len(self.dims)), np.float32),
                "history": np.stack([h for _, h in items]), "position": np.zeros(P2_BAG_TICKS, np.float32),
                "bag_mask": mask, "target": np.float32(target)}


def window(z, tick, ctx, length=None):
    """The ``ctx`` x D causal window ending at ``tick`` (front zero-padded) and its history mask."""
    length = ctx if length is None else min(ctx, length)
    lo = max(0, tick - length + 1)
    n = tick - lo + 1
    x = np.zeros((ctx, z.shape[1]), np.float32)
    x[ctx - n:] = z[lo:tick + 1]
    hist = np.zeros(ctx, bool)
    hist[ctx - n:] = True
    return x, hist


def make_bags(ss, vids, vt, rng, max_ticks=None, window=(120, 30)):
    """Per training event, a bag of up to ``max_ticks`` ticks in [max(t0-120, clear_start), t0-30] and a bag of the same
    ticks in a matched clean video of the same source kind."""
    max_ticks = max_ticks or P2_BAG_TICKS
    clean = {}
    for v in vids:
        t = vt[v]
        if t.kind == "clean":
            clean.setdefault(ss.source_kind(v), []).append((v, ~t.masked & (t.remaining >= 60)))
    bags, missing = [], 0
    for v in vids:
        t = vt[v]
        if t.kind != "violent":
            continue
        for e in ss.bench.event_rows(v):
            if not e["eligible30"]:
                continue
            lo, hi = max(e["onset"] - window[0], e["clear_start"]), e["onset"] - window[1]
            inside = (t.t <= hi) if window[1] else (t.t < hi)
            ticks = np.flatnonzero((t.t >= lo) & inside & ~t.masked)
            if not len(ticks):
                continue
            ticks = np.sort(rng.choice(ticks, min(len(ticks), max_ticks), replace=False))
            match = [u for u, good in clean.get(ss.source_kind(v), []) if ticks[-1] < len(good) and good[ticks].all()]
            if not match:
                missing += 1
                continue
            u = match[int(rng.integers(len(match)))]
            bags += [(v, ticks, 1.0), (u, ticks.copy(), 0.0)]
    return bags, {"event_bags": len(bags) // 2, "events_without_matched_clean_window": missing}


def gru_inputs(runners, names, vt, train, vids, clip=P.CLIP_Z):
    """Per-tick columns of ``names`` standardized on the ``train`` videos, for every video of ``vids``, and the widths."""
    raw, dims = {}, None
    for v in vids:
        parts = [runners[n].xform(v)[:vt[v].n] for n in names]
        dims = [p.shape[1] for p in parts]
        raw[v] = np.concatenate(parts, axis=1)
    s1 = np.zeros(sum(dims))
    s2 = np.zeros(sum(dims))
    n = 0
    for v in train:
        x = raw[v].astype(np.float64)
        s1 += x.sum(0)
        s2 += (x * x).sum(0)
        n += len(x)
    mean = s1 / max(n, 1)
    scale = np.sqrt(np.maximum(s2 / max(n, 1) - mean * mean, 1e-8))
    z = {v: np.clip((x - mean) / scale, -clip, clip).astype(np.float32) for v, x in raw.items()}
    return z, dims


def score_video(model, z, ctx, batch=SCORE_BATCH):
    """Band logits of every tick of one video from its causal windows."""
    model.eval()
    n, D = z.shape
    out = np.zeros(n, np.float32)
    zp = np.concatenate([np.zeros((ctx - 1, D), np.float32), z])
    wins = np.lib.stride_tricks.sliding_window_view(zp, ctx, axis=0)
    edges = np.cumsum([0] + list(model.dims))
    with torch.no_grad():
        for a in range(0, n, batch):
            idx = np.arange(a, min(a + batch, n))
            x = torch.from_numpy(np.ascontiguousarray(wins[idx].transpose(0, 2, 1)))
            xs = [x[:, :, p:q] for p, q in zip(edges[:-1], edges[1:])]
            present = torch.ones(len(idx), ctx, len(model.dims))
            full = idx[0] >= ctx - 1
            hist = None
            if not full:
                lens = np.minimum(idx + 1, ctx)
                hist = torch.from_numpy(np.arange(ctx)[None, :] >= (ctx - lens)[:, None])
            _, _, band = model(xs, present, torch.zeros(len(idx)), hist, True)
            out[idx] = band.numpy()
    return out


def score_rows(model, store, rows, ctx, batch=SCORE_BATCH):
    """Band logits for (video, tick, ...) rows grouped by video."""
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r[0], []).append(i)
    out = np.zeros(len(rows), np.float32)
    model.eval()
    edges = np.cumsum([0] + list(model.dims))
    with torch.no_grad():
        for v, ix in by.items():
            ticks = [rows[i][1] for i in ix]
            for a in range(0, len(ix), batch):
                ws = [window(store[v], t, ctx) for t in ticks[a:a + batch]]
                x = torch.from_numpy(np.stack([w[0] for w in ws]))
                hist = torch.from_numpy(np.stack([w[1] for w in ws]))
                xs = [x[:, :, p:q] for p, q in zip(edges[:-1], edges[1:])]
                _, _, band = model(xs, torch.ones(len(ws), ctx, len(model.dims)), torch.zeros(len(ws)), hist, True)
                out[ix[a:a + batch]] = band.numpy()
    return out


def gru_rows(plan, vt):
    rows = []
    for v, (ti, yy, ww) in plan.by_video.items():
        rows += [(v, int(t), float(y), float(w)) for t, y, w in zip(ti, yy, ww)]
    return rows


def train_gru(z, dims, rows_tr, rows_va, bags, seed, ctx=180, hidden=64, epochs=20, patience=4, lr=1e-3,
              fixed_epochs=False, max_minutes=None):
    """One causal GRU (band BCE plus the event-window loss, coefficient 1 each); returns ``(model, history)``.

    ``fixed_epochs`` trains exactly ``epochs`` epochs and keeps the last (no validation slice exists)."""
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CausalTemporalModel(dims, "gru", hidden=hidden, layers=1, context=ctx, modality_dropout=P2_MODALITY_DROPOUT,
                                dropout=P2_DROPOUT, score_kind="band")
    model.dims = list(dims)
    train = GruRows(z, rows_tr, dims, ctx, P2_HISTORY_DROPOUT)
    val = GruRows(z, rows_va, dims, ctx)
    bag_ds = GruBags(z, bags, dims, ctx) if bags else None
    counter = {"n": 0}
    deadline = None if not max_minutes else time.monotonic() + 60 * max_minutes

    def check_time():
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("GRU wall budget reached")

    def selection(m):
        if fixed_epochs:
            counter["n"] += 1
            return float(counter["n"])
        s = score_rows(m, z, rows_va, ctx)
        y = np.array([r[2] > 0 for r in rows_va])
        w = np.array([r[3] for r in rows_va])
        a = P.weighted_auc(s, y, w)
        return a if np.isfinite(a) else 0.5

    hist = fit_temporal(model, train, val, "cpu", epochs=epochs, patience=(epochs + 1 if fixed_epochs else patience),
                        batch_size=P2_BATCH, lr=lr, auxiliary_weight=0.0, seed=seed, check_time=check_time,
                        selection_score=selection, band_weight=1.0, event_weight=1.0, event_dataset=bag_ds,
                        event_batch_size=P2_BAG_BATCH)
    return model, hist


def p2_fold(data, f, names, seeds, epochs, max_minutes=None, target="band"):
    """P2 for one fold: ``{seed: {split: {video: logits}}}`` plus per-seed training information."""
    c, ss = data.fc[f], data.ss
    z, dims = gru_inputs(data.runners, names, c.vt, c.cohort["train"], c.cohort["train"] + c.cohort["val"] + c.cohort["test"])
    rng = np.random.default_rng(data.seed + 5000 + f)
    vts_tr = [c.vt[v] for v in c.cohort["train"]]
    vts_va = [c.vt[v] for v in c.cohort["val"]]
    plan_tr = P.Plan(vts_tr, target, rng, P2_MAX_POS, P2_MAX_NEG, ss.source_kind)
    plan_va = P.Plan(vts_va, target, rng, None, P2_MAX_VAL_NEG, ss.source_kind)
    rows_tr, rows_va = gru_rows(plan_tr, c.vt), gru_rows(plan_va, c.vt)
    bags, bag_info = make_bags(ss, c.cohort["train"], c.vt, rng, window=(120, 30) if target == "band" else (10, 0))
    out, info = {}, {}
    for seed in seeds:
        t0 = time.time()
        model, hist = train_gru(z, dims, rows_tr, rows_va, bags, seed, epochs=epochs, max_minutes=max_minutes)
        sc = {"test": {}, "val": {}}
        for split in ("test", "val"):
            for v in c.cohort[split]:
                sc[split][v] = score_video(model, z[v], model.context)
        out[seed] = sc
        info[seed] = {"best_epoch": int(model.best_epoch), "history": hist, "train_rows": len(rows_tr),
                      "val_rows": len(rows_va), "seconds": round(time.time() - t0, 1), **bag_info}
        log(f"P2 fold {f} seed {seed}: best epoch {model.best_epoch} of {len(hist)}, {time.time() - t0:.0f} s")
    return out, info


def run_p2(data, ev, names, seeds, epochs, max_minutes=None):
    target = ev.target
    per_seed = {s: {} for s in seeds}
    infos = {}
    for f in data.folds:
        out, info = p2_fold(data, f, names, seeds, epochs, max_minutes, target)
        for s in seeds:
            per_seed[s][f] = out[s]
        infos[f] = info
    seed_stats = {s: ev.stats(ev.pooled(per_seed[s])) for s in seeds}
    ens = {f: {sp: {v: np.mean([per_seed[s][f][sp][v] for s in seeds], axis=0) for v in per_seed[seeds[0]][f][sp]}
               for sp in ("test", "val")} for f in data.folds}
    return {"scores": ens, "stats": ev.stats(ev.pooled(ens)), "seed_stats": seed_stats, "info": infos}


def write_scores(ss, data, per_fold, path):
    rows = []
    for f in data.folds:
        for split in ("test", "val"):
            for v, s in per_fold[f][split].items():
                n = data.fc[f].vt[v].n
                rows.append(pd.DataFrame({"video_id": v, "t_sec": data.fc[f].vt[v].t[:n], "score": s[:n], "fold": f,
                                          "split": split, "seed": 0}))
    df = pd.concat(rows, ignore_index=True)
    ss.assert_no_frozen(df)
    df.to_parquet(path, index=False)
    return df


def label_paths(ss):
    """The registration's catalog paths (v3 labels under screen_set_v2); empty means the VA30 defaults."""
    prov = ss.prereg.data["provenance"] if getattr(ss, "prereg", None) is not None else {}
    return {k: os.path.join(REPO_ROOT, prov[k]) for k in ("videos_csv", "events_json") if prov.get(k)}


def run_va30(ss, df, out_dir, name, fa_targets="1,2,5", threshold_boot=200, n_boot=1000, no_position_null=False):
    """VA30 event metrics on the screening folds (fold 1 removed from the fold file, ids renumbered)."""
    os.makedirs(out_dir, exist_ok=True)
    folds_path, fmap = R.folds_without_confirmation(out_dir)
    d = df[df["video_id"].map(ss.kind) != "h1"].copy()
    d["fold"] = d["fold"].map(fmap)
    path = os.path.join(out_dir, "va30_scores.parquet")
    d[["video_id", "t_sec", "score", "fold", "seed"]].to_parquet(path, index=False)
    labels = label_paths(ss)
    args = [a for k, flag in (("videos_csv", "--videos"), ("events_json", "--events")) if k in labels for a in (flag, labels[k])]
    args += ["--scores", path, "--out_dir", os.path.join(out_dir, name), "--folds", folds_path, "--restrict_to_scored",
            "--horizons", S.MANIFEST_V1, "--threshold_bootstrap", str(threshold_boot), "--n_boot", str(n_boot),
            "--fa_targets", fa_targets, "--cluster", "split_group"]
    if no_position_null:
        args.append("--no_position_null")
    va30.main(args)
    with open(os.path.join(out_dir, name, "metrics.json")) as fh:
        return json.load(fh)


class _NullShim:
    """Presents an EventEval to ``va30.position_null`` with Success@``lead`` in place of Success@30."""

    def __init__(self, ee, lead):
        self.ee, self.lead = ee, lead
        self.ts, self.ev_e30, self.ev_v, self.ev_lo = ee.ts, ee.ev_e30, ee.ev_v, ee.ev_lo
        self.ev_on = ee.ev_on + (30.0 - lead)

    def fast(self, theta_v):
        ee = self.ee
        ep_v, ep_t, _ = ee._starts(theta_v)
        sel = ee.ts.clean_v[ep_v] if len(ep_v) else np.zeros(0, bool)
        n = int(ee._outside(ep_v[sel], ep_t[sel]).sum()) if sel.any() else 0
        hit = np.nan_to_num(ee._leads(ep_v, ep_t), nan=-1.0) >= self.lead
        e, late = ee.ev_e30, ee.ev_e30 & ee.ev_late
        return (n / ee.clean_hours if ee.clean_hours > 0 else np.nan, float(hit[e].mean()) if e.any() else np.nan,
                float(hit[late].mean()) if late.any() else np.nan)


def threshold_aware_lead(val_ee, ee, W, target, n_boot, lead, seed=0, cluster="split_group", max_grid=300):
    """``va30.threshold_aware_bootstrap`` with Success@``lead`` (the original is fixed at 30 s)."""
    ts = ee.ts
    B = min(n_boot, len(W)) if len(W) else n_boot
    nf = len(val_ee)
    theta_b = np.zeros((B, nf))
    for f in range(nf):
        vee = val_ee[f]
        vts = vee.ts
        lo, _ = va30.choose_threshold(vee, 4.0 * target)
        u = np.unique(vts.score)
        u = u[u >= lo][::-1]
        if len(u) > max_grid:
            u = np.unique(np.quantile(u, np.linspace(0, 1, max_grid)))[::-1]
        cand = np.r_[np.inf, u]
        Nm = np.stack([vee.fa_counts_clean(np.full(vts.V, c)) for c in cand])
        hrs = np.where(vts.clean_v, vee.hours_out, 0.0)
        Wv = va30.bootstrap_weights(vts, B, seed + 104729 * (f + 1), cluster=cluster)
        with np.errstate(invalid="ignore", divide="ignore"):
            fa = (Wv @ Nm.T) / (Wv @ hrs)[:, None]
        exceed = ~(fa <= target)
        first = np.where(exceed.any(1), exceed.argmax(1), len(cand))
        theta_b[:, f] = cand[np.maximum(first - 1, 0)]
    cache, hit_rate, far = {}, np.full(B, np.nan), np.full(B, np.nan)
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
            cache[key] = (fa_v, (np.nan_to_num(ee._leads(ep_v, ep_t), nan=-1.0) >= lead).astype(float))
        fa_v, hit = cache[key]
        w = W[b] if len(W) else np.ones(ts.V)
        we = w[ee.ev_v[e30]]
        hit_rate[b] = (we * hit[e30]).sum() / we.sum() if we.sum() > 0 else np.nan
        far[b] = (w @ fa_v) / (w @ hrs_t) if (w @ hrs_t) > 0 else np.nan
    q = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]  # noqa: E731
    return {"n_boot": int(B), "success_ci": q(hit_rate), "fa_per_hour_clean_ci": q(far)}


def lead_report(bench, df, lead=10.0, fa_targets=(1, 2, 5), n_boot=1000, thr_boot=200, cluster="split_group"):
    """Success@``lead`` at validation-chosen thresholds with threshold-aware CIs, the position null and the random
    reference, plus official/UBI-onset and per-source breakdowns (VA30 machinery, evaluation horizons already applied)."""
    cfg = va30.EvalConfig(n_boot=n_boot, cluster=cluster, restrict_to_scored=True, allow_missing=True)
    table = va30.ScoreTable(df, bench.video_horizons)
    eval_set = {v for v in bench.eval_ids(cfg.onset_sources) if v in table.videos}
    folds, nf = bench.folds, bench.folds.n_folds
    test_by = {f: [v for v in folds.test_ids(f) if v in eval_set] for f in range(nf)}
    val_by = {f: [v for v in folds.val_ids(f) if v in eval_set] for f in range(nf)}
    cov = {}
    test_scores = va30._collect(bench, table, 0, test_by, cfg, cov)
    folds_of = {v: f for f, vs in test_by.items() for v in vs}
    ts = va30.TickSet(bench, [v for f in range(nf) for v in test_by[f] if v in test_scores], test_scores, folds_of)
    W = va30.bootstrap_weights(ts, n_boot, cfg.boot_seed, cluster)
    ee = va30.EventEval(ts, cfg.policy)
    val_ee = {}
    for f in range(nf):
        vs = va30._collect(bench, table, 0, {f: val_by[f]}, cfg, {})
        val_ee[f] = va30.EventEval(va30.TickSet(bench, list(vs), vs), cfg.policy)
    null = va30.position_null(_NullShim(ee, lead))
    e30 = ee.ev_e30
    onset_src = bench.videos.loc[np.array(ts.vids), "onset_source"].values[ee.ev_v] if ts.V else np.array([])
    src_kind = bench.videos.loc[np.array(ts.vids), "source_kind"].values[ee.ev_v] if ts.V else np.array([])

    def rate(hit, sel):
        sel = sel & e30
        pt = float(hit[sel].mean()) if sel.any() else None
        if sel.any() and len(W):
            Wv = W[:, ee.ev_v[sel]]
            with np.errstate(invalid="ignore", divide="ignore"):
                b = (Wv * hit[sel]).sum(1) / Wv.sum(1)
            b = b[np.isfinite(b)]
            return {"value": pt, "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if len(b) > 10 else None,
                    "n": int(sel.sum())}
        return {"value": pt, "ci": None, "n": int(sel.sum())}

    ops = {}
    for t in fa_targets:
        theta_f = {f: va30.choose_threshold(val_ee[f], t)[0] for f in range(nf)}
        theta_v = np.array([theta_f[f] for f in ts.fold_v], dtype=np.float64)
        out = ee.outcomes(theta_v)
        hit = np.nan_to_num(out["lead"], nan=-1.0) >= lead
        far = float(out["fa"][ts.clean_v].sum() / ee.clean_hours) if ee.clean_hours > 0 else None
        op = {"target_fa_per_hour": t, "test_fa_per_hour_clean": far, "success": rate(hit, np.ones(len(hit), bool)),
              "position_null": va30.null_at(null, far), "random_reference": None if far is None else 1 - float(np.exp(-far * (120 - lead) / 3600)),
              "official_ubi": rate(hit, np.isin(onset_src, ["official", "ubi"])),
              "by_source": {str(k): rate(hit, src_kind == k) for k in sorted(set(src_kind))},
              "median_lead_sec": float(np.nanmedian(out["lead"][e30])) if np.isfinite(out["lead"][e30]).any() else None}
        if thr_boot > 0:
            op["threshold_aware_ci"] = threshold_aware_lead(val_ee, ee, W, t, thr_boot, lead, cfg.boot_seed, cluster)
        ops[f"fa{t}"] = op
    return {"lead_sec": lead, "operating_points": ops, "counts": {"events": int(e30.sum()), "test_videos": ts.V},
            "coverage": cov}


def screening_bench(out_dir, ss=None):
    """The VA30 benchmark of the screening folds (fold 1 removed, ids renumbered) and the fold map."""
    os.makedirs(out_dir, exist_ok=True)
    path, fmap = R.folds_without_confirmation(out_dir)
    return va30.Benchmark.load(path, horizons=S.MANIFEST_V1, **(label_paths(ss) if ss is not None else {})), fmap


def event_summary(m):
    """Success@30 per FA/h target against the position null and the random reference, plus per-source rows."""
    s = m["seeds"][m["headline_seed"]]
    rows = {}
    for key, op in s["operating_points"].items():
        rows[key] = {"target_fa_per_hour": op["target_val_fa_per_hour"],
                     "test_fa_per_hour_clean": op["fa_per_hour_clean"]["value"],
                     "success30": op["eligible30"]["success30"], "success10": op["eligible30"]["success10"],
                     "success30_late": op["eligible30_late"]["success30"],
                     "median_lead_sec": op["eligible30"]["lead_sec"]["median"],
                     "position_null_success30": op.get("position_null_success30"),
                     "position_null_success30_late": op.get("position_null_success30_late"),
                     "random_success30": op["random_reference_success30"]["nominal_W90"],
                     "threshold_aware_ci": op.get("threshold_aware_ci")}
    return rows


def per_source(m):
    s = m["seeds"][m["headline_seed"]]
    out = {}
    for k, b in (s.get("breakdowns") or {}).items():
        if not k.startswith("source:"):
            continue
        out[k] = {kk: b[kk] for kk in b if kk.startswith("op_fa")} | {"far_window_auroc_clean": b.get("far_window_auroc_clean")}
    return out


HEAD_TARGETS = {
    "early": "band: first-segment pre-violence ticks with time to onset in (30, 120] vs clean ticks with >= 60 s left",
    "imminent": "near: first-segment pre-violence ticks with time to onset in (0, 10] vs clean ticks with >= 60 s left"}
HEAD_RULES = {
    "early": {"m1_band_ci_lower_gt": 0.5, "m8_gain_over_n_point_gt": 0.0,
              "text": "fold 1 band M1 CI lower > 0.50 and positive M8 point gain over the nuisance model N"},
    "imminent": {"m1_near_ci_lower_gt": 0.6, "success10_fa2_threshold_aware_ci_lower_gt_position_null": True,
                 "text": "fold 1 M1-near CI lower > 0.60 and Success@10 at the 2 FA/h validation operating point has a "
                         "threshold-aware CI lower bound above the position null at the same test FA/h (amendment 05)"}}


def freeze_head(head, sel, seed, max_pos, max_neg, p2_epochs_fixed=None):
    """The frozen spec of one head (docs/FUSION.md section 5); nothing from fold 1."""
    names = sel["final"]["blocks"]
    return {"head": head, "target": HEAD_TARGETS[head],
            "blocks": [{"name": n, "streams": list(F.BLOCKS[n].streams), "include": list(F.BLOCKS[n].include),
                        "exclude": list(F.BLOCKS[n].exclude)} for n in names],
            "probe_type": sel["final"]["probe"], "seed": seed,
            "p1": {"C": sel["final"]["p1_C"], "C_rule": "best mean validation M1 over the four screening folds",
                   "C_per_fold": sel["final"]["p1_C_per_fold"], "windows_sec": list(F.WINDOWS),
                   "summaries": ["mean", "max", "fraction_above_clean_train_p95"], "slopes": False,
                   "wide_threshold": F.WIDE_COLUMNS, "pls_components": F.PLS_COMPONENTS, "max_train_pos": max_pos,
                   "max_train_neg": max_neg, "target": HEAD_TARGETS[head] + "; time_match_weights by source kind x "
                   "TIME_MATCH_EDGES"},
            "nuisance": {"C": sel["nuisance_C"], "C_rule": "grid value nearest the median per-fold choice",
                         "scene_pca": P.SCENE_PCA, "scene_t_sec": P.SCENE_T_SEC},
            "p2": ({"hidden": 64, "layers": 1, "context_sec": 180, "seeds": sel["p2"]["seeds"], "lr": 1e-3,
                    "weight_decay": 1e-4, "batch_size": P2_BATCH, "fixed_epochs": p2_epochs_fixed,
                    "epochs_rule": "median best epoch over folds and seeds, trained to that many epochs without a validation slice",
                    "max_train_pos": P2_MAX_POS, "max_train_neg": P2_MAX_NEG, "history_dropout": P2_HISTORY_DROPOUT,
                    "modality_dropout": P2_MODALITY_DROPOUT, "dropout": P2_DROPOUT, "bag_ticks": P2_BAG_TICKS,
                    "bag_batch_size": P2_BAG_BATCH, "loss": "target BCE + event-window MIL, coefficient 1 each",
                    "score": "mean over seeds of target logits"} if sel["final"]["probe"] == "p2" else None),
            "calibration": {"source": "out-of-fold scores of the screening-fold models of this spec on the folds_v2 "
                            "validation slice of fold 1 (a subset of the screening folds)",
                            "scores_file": None, "scores_sha256": None},
            "confirmation_rule": HEAD_RULES[head], "selection": {"selected_order": names, "selection_sha256": None}}


def flat_to_head(flat):
    """Head fragment from the single-head spec written by the first version of ``run_selection``."""
    keys = ("blocks", "probe_type", "seed", "p1", "nuisance", "p2", "calibration", "confirmation_rule", "selection")
    out = {"head": "early", "target": HEAD_TARGETS["early"], **{k: flat[k] for k in keys}}
    out["confirmation_rule"] = HEAD_RULES["early"]
    return out


def common_spec(ss, pr_info, heads, amendment_sha=None):
    return {"version": 2, "prereg": pr_info, "amendment_05_sha256": amendment_sha,
            "training": {"folds": list(ss.screen_folds), "rule": "union of the frozen train and val memberships of the "
                         "screening folds (kinds violent and clean)",
                         "videos_sha256": training_list_sha(ss), "n_videos": len(training_list(ss))},
            "event_metrics": {"threshold_rule": "VA30 per-fold threshold: lowest theta whose FA/h on clean negatives of the "
                              "calibration slice does not exceed the target", "fa_targets": [1, 2, 5],
                              "horizons": "benchmark/extract_manifest_v1.csv", "restrict_to_scored": True,
                              "n_boot": 1000, "threshold_bootstrap": 200, "cluster": "split_group", "alarm_policy": "2 of 3"},
            "bootstrap": {"n": 1000, "seed": 0, "cluster": "split_group"}, "heads": heads}


def training_list(ss):
    ids = set()
    for f in ss.screen_folds:
        ids |= set(ss.train_ids(f)) | set(ss.val_ids(f))
    return sorted(ids)


def training_list_sha(ss):
    return S.list_sha256(training_list(ss))


def canonical_bytes(spec):
    return (json.dumps(R.jsonable(spec), indent=1, sort_keys=True, allow_nan=False) + "\n").encode()


def write_frozen(spec, out_dir):
    raw = canonical_bytes(spec)
    path = os.path.join(out_dir, "confirmation_config.json")
    with open(path, "wb") as f:
        f.write(raw)
    sha = hashlib.sha256(raw).hexdigest()
    with open(os.path.join(out_dir, "confirmation_config.json.sha256"), "w") as f:
        f.write(f"{sha}  confirmation_config.json\n")
    return path, sha


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def table_md(steps):
    lines = ["| step | candidate | M8 | M8 gain vs current (95% CI) | M1 band | selected |", "|---|---|---|---|---|---|"]
    fmt = lambda d: "" if not d or d.get("value") is None else f"{d['value']:.3f}"  # noqa: E731
    for s in steps:
        for n, c in s["candidates"].items():
            g = c.get("m8_gain")
            gtxt = "" if g is None else f"{g['value']:+.3f} [{g['ci'][0]:+.3f}, {g['ci'][1]:+.3f}]" if g.get("ci") else f"{g['value']:+.3f}"
            mark = "yes" if s.get("chosen") == n else ""
            lines.append(f"| {s['step']} {s['action']} | {n} | {fmt(c['m8'])} | {gtxt} | {fmt(c['m1'])} | {mark} |")
    return lines


AMENDMENT_05 = os.path.join(REPO_ROOT, "benchmark", "screens", "amendment_05_imminent_head.md")
GATES = {"N1": "M1-near point >= 0.70 and CI lower > 0.60", "N2": "M3p-near CI lower > 0.50",
         "N3": "paired gain of [N + block] over N on M1-near, CI lower > 0"}


def gate_row(stats, vsn):
    m1, m3p, m4n = stats["m1"], stats.get("m3p"), vsn["m4n"]
    n1 = m1.value is not None and m1.value >= 0.70 and m1.lower is not None and m1.lower > 0.60
    n2 = m3p is not None and m3p.lower is not None and m3p.lower > 0.50
    n3 = m4n.lower is not None and m4n.lower > 0
    return {"N1": n1, "N2": n2, "N3": n3, "pass": bool(n1 and n2 and n3), "m1_near": sd(m1), "m3p_near": sd(m3p),
            "m4n_near": sd(m4n, 0.0), "m8_near": sd(stats.get("m8"))}


def run_gates(ss, source, names, out, n_boot=1000, require_cut=True, max_pos=P.MAX_TRAIN_POS, max_neg=P.MAX_TRAIN_NEG):
    """N1-N3 of amendment 05 for every block, each on its own cohort; resumes from ``out/gates.json``."""
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "gates.json")
    done = json.load(open(path)) if os.path.isfile(path) else {"gates": GATES, "blocks": {}}
    for n in names:
        if n in done["blocks"]:
            continue
        log(f"gates: {n}")
        data = FusionData(ss, source, [n], require_cut=require_cut, max_pos=max_pos, max_neg=max_neg, targets=("near",))
        data.build()
        ev = Evaluator(data, n_boot, "imminent")
        res = ev.evaluate([n])
        vsn = ev.vs_nuisance([n], res)
        row = gate_row(res["stats"], vsn)
        row["coverage"] = {"pool_videos": ev.pool.V, "events": int(len(ev.pool.events))}
        done["blocks"][n] = row
        R.write_json(path, done)
        del data, ev
        log(f"gates: {n}: N1 {row['N1']} N2 {row['N2']} N3 {row['N3']}")
    return done


def run_selection(ss, source, anticipation, provisional, out, pr_info, n_boot=1000, p2_seeds=(0, 1, 2), p2_epochs=20,
                  skip_p2=False, from_p1=False, require_cut=True, p2_max_minutes=None, events=True, max_pos=P.MAX_TRAIN_POS,
                  max_neg=P.MAX_TRAIN_NEG, head="early", root=None):
    """Forward selection, P1/P2 comparison and event metrics of one head; writes ``head_spec.json``.

    ``anticipation`` holds the start candidates (the ANTICIPATION blocks, or the blocks that passed N1-N3) and
    ``provisional`` the further candidates. Returns ``(result, head_spec)``."""
    os.makedirs(out, exist_ok=True)
    root = root or out
    names = sorted(set(anticipation) | set(provisional))
    imminent = head == "imminent"
    data = FusionData(ss, source, names, require_cut=require_cut, max_pos=max_pos, max_neg=max_neg,
                      targets=("near", "detect") if imminent else ("band",))
    data.build()
    ev = Evaluator(data, n_boot, head)
    sel_path = os.path.join(out, "p1_selection.json")
    if from_p1:
        prev = json.load(open(sel_path))
        selected, steps = prev["selected"], prev["steps"]
    else:
        selected, steps = forward_select(ev, anticipation, provisional)
        R.write_json(sel_path, {"selected": selected, "steps": steps, "prereg": pr_info})
    log(f"{head} P1 selected:", selected)
    p1 = ev.evaluate(selected)
    vsn = ev.vs_nuisance(selected, p1)
    c_p1, c_path = mean_path_c(p1["info"])
    nuis_cs = [data.fit_set(f, [], True, "near")[1]["C"] for f in data.folds] if imminent else \
        [data.fc[f].nuis_info["C"] for f in data.folds]
    c_n = snap_median_c(nuis_cs)
    metric = "M8-near" if imminent else "M8"
    result = {"head": head, "prereg": pr_info, "anticipation": anticipation, "provisional": provisional, "steps": steps,
              "selected": selected,
              "p1": {"stats": stat_row(p1["stats"]), "vs_nuisance": R.jsonable(vsn),
                     "C_per_fold": {str(f): i["C"] for f, i in p1["info"].items()}, "C_val_path": c_path},
              "nuisance": {"stats": stat_row(ev.nuisance()), "C": c_n},
              "coverage": {"events": int(len(ev.pool.events)), "pool_videos": ev.pool.V},
              "interpretation": {"start": f"paired {metric} point estimate among the start candidates",
                                 "metric": f"{metric} over all eligible30 events (first and later segments)",
                                 "accept": f"largest paired {metric} gain, accepted only if the 95% bootstrap CI lower bound > 0",
                                 "final_rule": FINAL_RULE.replace("M8", metric)}}
    if imminent:
        result["b_detect_m1_near"] = sd(ev.detect(selected))
    p2, use_p2 = None, False
    if not skip_p2:
        p2 = run_p2(data, ev, selected, list(p2_seeds), p2_epochs, p2_max_minutes)
        gain = M.diff(p2["stats"]["m8"], p1["stats"]["m8"])
        best_epochs = [i["best_epoch"] for fi in p2["info"].values() for i in fi.values()]
        result["p2"] = {"seeds": list(p2_seeds), "stats": stat_row(p2["stats"]),
                        "seed_stats": {str(s): stat_row(v) for s, v in p2["seed_stats"].items()},
                        "m8_gain_over_p1": sd(gain, 0.0), "m1_gain_over_p1": sd(M.diff(p2["stats"]["m1"], p1["stats"]["m1"]), 0.0),
                        "best_epochs": best_epochs, "median_best_epoch": float(np.median(best_epochs)),
                        "train": {str(f): {str(s): {k: v for k, v in i.items() if k != "history"} for s, i in fi.items()}
                                  for f, fi in p2["info"].items()}}
        use_p2 = gain.lower is not None and gain.lower > 0
    final_scores = p2["scores"] if use_p2 else p1["scores"]
    result["final"] = {"probe": "p2" if use_p2 else "p1", "blocks": selected, "rule": result["interpretation"]["final_rule"],
                       "p1_C": c_p1, "p1_C_per_fold": {str(f): i["C"] for f, i in p1["info"].items()},
                       "stats": stat_row(p2["stats"] if use_p2 else p1["stats"])}
    result["nuisance_C"] = c_n
    scores_path = os.path.join(out, "oof_scores.parquet")
    df = write_scores(ss, data, final_scores, scores_path)
    log(f"{head} final model:", result["final"]["probe"], selected)
    if events and imminent:
        bench, fmap = screening_bench(os.path.join(out, "events"), ss)
        d = df[df["video_id"].map(ss.kind) != "h1"].copy()
        d["fold"] = d["fold"].map(fmap)
        result["events"] = R.jsonable(lead_report(bench, d, 10.0, (1, 2, 5), n_boot, 200))
    elif events:
        m = run_va30(ss, df, os.path.join(out, "events"), "va30_fa125", n_boot=n_boot)
        result["events"] = {"operating_points": R.jsonable(event_summary(m)),
                            "counts": R.jsonable(m["seeds"][m["headline_seed"]]["counts"]), "per_source": {}}
        for t in ("1", "2", "5"):
            mm = m if t == "2" else run_va30(ss, df, os.path.join(out, "events"), f"va30_fa{t}", t, 0, n_boot)
            result["events"]["per_source"][t] = R.jsonable(per_source(mm))
    spec = freeze_head(head, result, data.seed, max_pos, max_neg, int(round(result["p2"]["median_best_epoch"])) if use_p2 else None)
    spec["calibration"]["scores_file"] = os.path.relpath(scores_path, root)
    spec["calibration"]["scores_sha256"] = file_sha(scores_path)
    R.write_json(os.path.join(out, "selection.json"), result)
    spec["selection"]["selection_sha256"] = file_sha(os.path.join(out, "selection.json"))
    R.write_json(os.path.join(out, "head_spec.json"), spec)
    write_report(result, out)
    return result, spec


def assemble_config(ss, pr_info, heads, out):
    """Write the two-head ``confirmation_config.json`` (and its sha256 sidecar) from head fragments (None = no head)."""
    amend = file_sha(AMENDMENT_05) if os.path.isfile(AMENDMENT_05) else None
    spec = common_spec(ss, pr_info, heads, amend)
    path, sha = write_frozen(spec, out)
    return spec, path, sha


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--head", choices=["early", "imminent", "assemble"], default="early")
    ap.add_argument("--decisions", default=DECISIONS_PATH)
    ap.add_argument("--prereg", default=default_prereg())
    ap.add_argument("--feature_root", default=None)
    ap.add_argument("--from_p1", action="store_true", help="reuse the head's p1_selection.json instead of re-running forward selection")
    ap.add_argument("--skip_p2", action="store_true")
    ap.add_argument("--p2_epochs", type=int, default=20)
    ap.add_argument("--p2_max_minutes", type=float, default=None)
    args = ap.parse_args(argv)
    va30.N_THREADS = 4
    pr, ss, extra = R.prepare(args)
    pr_info = {"path": os.path.relpath(pr.path, REPO_ROOT) if pr.path.startswith(REPO_ROOT) else pr.path, "sha256": pr.sha256}
    n_boot = int(pr.data["metrics"]["bootstrap"]["n"])
    seeds = pr.data["probes"]["P2"]["seeds"]
    source = F.FeatureSource(ss, root=args.feature_root)
    if args.head == "assemble":
        heads = {}
        early_dir, imm_dir = args.out, os.path.join(args.out, "imminent")
        spec_e = os.path.join(early_dir, "head_spec.json")
        if os.path.isfile(spec_e):
            heads["early"] = json.load(open(spec_e))
        else:
            flat = os.path.join(early_dir, "confirmation_config.json")
            old = json.load(open(flat))
            if old.get("version") == 1:
                os.replace(flat, os.path.join(early_dir, "confirmation_config_early_only_v1.json"))
                os.replace(flat + ".sha256", os.path.join(early_dir, "confirmation_config_early_only_v1.json.sha256"))
                heads["early"] = flat_to_head(old)
            else:
                heads["early"] = old["heads"]["early"]
        spec_i = os.path.join(imm_dir, "head_spec.json")
        heads["imminent"] = json.load(open(spec_i)) if os.path.isfile(spec_i) else None
        spec, path, sha = assemble_config(ss, pr_info, heads, early_dir)
        log("confirmation_config.json sha256", sha)
        return 0
    with threadpool_limits(limits=4):
        if args.head == "early":
            by_class, _ = load_candidates(args.decisions, pr.sha256)
            if not by_class["ANTICIPATION"]:
                raise SystemExit("no ANTICIPATION block: the forward selection has no start block (no 30-second claim)")
            log("ANTICIPATION", by_class["ANTICIPATION"], "PROVISIONAL", by_class["PROVISIONAL"])
            run_selection(ss, source, by_class["ANTICIPATION"], by_class["PROVISIONAL"], args.out, pr_info, n_boot=n_boot,
                          p2_seeds=seeds, p2_epochs=args.p2_epochs, skip_p2=args.skip_p2, from_p1=args.from_p1,
                          p2_max_minutes=args.p2_max_minutes)
        else:
            out = os.path.join(args.out, "imminent")
            blocks = [r["name"] for r in F.resolve_blocks(pr) if r["phase"] == "A" and r["status"] == F.AVAILABLE]
            gates = run_gates(ss, source, blocks, out, n_boot)
            passing = sorted(n for n, r in gates["blocks"].items() if r["pass"])
            log("imminent head passing blocks:", passing)
            if not passing:
                R.write_json(os.path.join(out, "selection.json"), {"head": "imminent", "selected": [], "prereg": pr_info,
                                                                   "note": "no block passed N1-N3; no imminent head"})
                return 0
            run_selection(ss, source, passing, [], out, pr_info, n_boot=n_boot, p2_seeds=seeds, p2_epochs=args.p2_epochs,
                          skip_p2=args.skip_p2, from_p1=args.from_p1, p2_max_minutes=args.p2_max_minutes, head="imminent",
                          root=args.out)
    return 0


def write_report(res, out_dir, sha=None):
    imminent = res.get("head") == "imminent"
    f3 = lambda d: "n/a" if not d or d.get("value") is None else f"{d['value']:.3f}" + (f" [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]" if d.get("ci") and d["ci"][0] is not None else "")  # noqa: E731
    m8name = "M8-near" if imminent else "M8"
    L = [f"# Fusion ({res.get('head', 'early')} head): forward selection and final model", "",
         f"Pre-registration: `{res['prereg']['sha256']}`", "",
         f"Start candidates {res['anticipation']}, further candidates {res['provisional']}", "", "## Forward selection (P1)", ""]
    L += table_md(res["steps"])
    cols = ("M1-near", "M3p-near", m8name, "M8-near official/UBI first") if imminent else ("M1 band", "M1 30-60", "M8 (all)", "M8 (first segment)")
    L += ["", f"Selected, in order: {res['selected']}", "", "## Selected set", "", "| model | " + " | ".join(cols) + " |", "|---|---|---|---|---|"]
    if imminent:
        row = lambda n, s: f"| {n} | {f3(s['m1'])} | {f3(s.get('m3p'))} | {f3(s['m8'])} | {f3(s.get('m8_offubi'))} |"  # noqa: E731
    else:
        row = lambda n, s: f"| {n} | {f3(s['m1'])} | {f3(s['m1_30_60'])} | {f3(s['m8'])} | {f3(s.get('m8_first'))} |"  # noqa: E731
    L.append(row("N (nuisance)", res["nuisance"]["stats"]))
    L.append(row("P1", res["p1"]["stats"]))
    if "p2" in res:
        L.append(row("P2 (3-seed ensemble)", res["p2"]["stats"]))
        for s, st in res["p2"]["seed_stats"].items():
            L.append(row(f"P2 seed {s}", st))
        L += ["", f"P2 minus P1: {m8name} {f3(res['p2']['m8_gain_over_p1'])}, M1 {f3(res['p2']['m1_gain_over_p1'])}"]
    v = res["p1"]["vs_nuisance"]
    L += ["", f"P1 vs N: M4n {f3(v['m4n'])}, {m8name} gain over N {f3(v['m8_gain_over_n'])}"]
    if imminent:
        L += [f"B-detect (violence-now probe, M1-near): {f3(res.get('b_detect_m1_near'))}"]
    L += ["", f"Final model: **{res['final']['probe']}** on {res['final']['blocks']}. Rule: {res['final']['rule']}.", ""]
    ev = res.get("events")
    if ev and imminent:
        L += ["## Success@10 on the screening folds (post-selection)", "",
              "| FA/h target | test FA/h | Success@10 | threshold-aware CI | position null | random | official/UBI onsets | median lead (s) |",
              "|---|---|---|---|---|---|---|---|"]
        for k, o in ev["operating_points"].items():
            ci = (o.get("threshold_aware_ci") or {}).get("success_ci")
            pn, rd = o.get("position_null"), o.get("random_reference")
            L.append(f"| {o['target_fa_per_hour']:g} | {o['test_fa_per_hour_clean']:.2f} | {f3(o['success'])} | "
                     f"{'n/a' if not ci else f'[{ci[0]:.3f}, {ci[1]:.3f}]'} | {'n/a' if pn is None else f'{pn:.3f}'} | "
                     f"{'n/a' if rd is None else f'{rd:.3f}'} | {f3(o['official_ubi'])} | "
                     f"{'n/a' if o['median_lead_sec'] is None else round(o['median_lead_sec'], 1)} |")
        L += ["", "Per-source Success@10:", "", "| source | 1 FA/h | 2 FA/h | 5 FA/h |", "|---|---|---|---|"]
        for s_ in sorted(ev["operating_points"]["fa2"]["by_source"]):
            L.append(f"| {s_} | " + " | ".join(f3(ev["operating_points"][f"fa{t}"]["by_source"].get(s_)) for t in (1, 2, 5)) + " |")
    elif ev:
        L += ["## VA30 event metrics on the screening folds (post-selection)", "",
              "| FA/h target | test FA/h | Success@30 | position null | random | Success@10 | median lead (s) |", "|---|---|---|---|---|---|---|"]
        for k, o in ev["operating_points"].items():
            pn = o.get("position_null_success30")
            pn_v = pn.get("value") if isinstance(pn, dict) else pn
            L.append(f"| {o['target_fa_per_hour']:g} | {o['test_fa_per_hour_clean']:.2f} | {f3(o['success30'])} | "
                     f"{'n/a' if pn_v is None else f'{pn_v:.3f}'} | {f3(o['random_success30'])} | {f3(o['success10'])} | {f3(o['median_lead_sec'])} |")
        L += ["", "Per-source Success@30 at each target:", "", "| source | 1 FA/h | 2 FA/h | 5 FA/h |", "|---|---|---|---|"]
        for s_ in sorted(ev["per_source"]["2"]):
            cells = []
            for t in ("1", "2", "5"):
                b = ev["per_source"][t].get(s_, {})
                op = next((b[k] for k in b if k.startswith("op_fa")), None)
                cells.append(f3(op["success30"]) if op else "n/a")
            L.append(f"| {s_} | " + " | ".join(cells) + " |")
    if sha:
        L += ["", f"confirmation_config.json sha256: `{sha}`", ""]
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
