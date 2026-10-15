"""Block registry, feature access with the fold guard, causal window summaries (P1 arms) and weighted PLS."""

import os
from dataclasses import dataclass, field

import numpy as np

from anticipation.features import io
from anticipation.screen.sets import REPO_ROOT

WINDOWS = (10, 60, 180)
SLOPE_WINDOWS = (15, 30, 60)
BASELINE_SEC = 20
WIDE_COLUMNS = 256
PLS_COMPONENTS = 32
SCENE_STREAM = "siglip2_scene0p"
CUT_STREAM, CUT_COLUMN = "motion", "cut_1s"
CAMERA_COLUMNS = ("cam_flow_mag", "cam_shake", "cam_flow_mag_5s", "hist_dist_max", "cut_1s", "cuts_5s", "time_since_cut")


@dataclass(frozen=True)
class Block:
    name: str
    phase: str
    streams: tuple
    include: tuple = ()
    exclude: tuple = ()
    lower_is_risk: tuple = ()
    sources: tuple = ()
    audio: bool = False
    g2_exempt: bool = False
    note: str = ""


PROMPT_LOW = ("calm_street", "talk_calm", "handshake", "hug", "queue_calm", "restaurant_calm", "store_empty",
              "parking_empty", "office_work")
PROMPT_V2_LOW = ("phase_calm", "phase_deescalation", "walk_away", "apologize", "calm_two")

BLOCKS = {b.name: b for b in [
    Block("pose_interact", "A", ("pose_interact",)),
    Block("pose_state", "A", ("pose_state",)),
    Block("pose_state_pairs", "A", ("pose_state", "pose_state_pairs"), sources=("ucf", "ubi"),
          note="pair-identity variant; UCF and UBI only"),
    Block("siglip2_prompts_v1", "A", ("siglip2_prompts",), lower_is_risk=PROMPT_LOW),
    Block("siglip2_prompts_v2", "A", ("siglip2_prompts_v2",), lower_is_risk=PROMPT_V2_LOW),
    Block("siglip2", "A", ("siglip2",)),
    Block("vjepa2_l", "A", ("vjepa2_l",)),
    Block("motion_nocam", "A", ("motion",), exclude=CAMERA_COLUMNS),
    Block("motion_cam", "A", ("motion",), include=CAMERA_COLUMNS, note="shortcut block"),
    Block("audio_dsp", "B", ("audio_dsp",), lower_is_risk=("quiet_frac",), audio=True),
    Block("ast_events", "B", ("ast_events",), lower_is_risk=("Silence",), audio=True),
    Block("clap_prompts", "B", ("clap_prompts",), audio=True),
    Block("speech_tox", "B", ("speech_tox",), lower_is_risk=("kw_deescalate",), audio=True),
    Block("voice_affect", "B", ("voice_affect",), lower_is_risk=("valence", "e2v_happy", "e2v_neutral"), audio=True),
    Block("e2v_emb", "B", ("e2v_emb",), audio=True),
]}
PHASE_C = ("vlm_arm_a", "vlm_arm_b", "vlm_arm_c", "vlm_forecast", "vlm_hidden_probe")
EXTERNAL_REFERENCES = ("aggression_level", "b_teacher")


class FeatureSource:
    """1 Hz streams via ``io.load_features``; refuses videos of the frozen fold."""

    def __init__(self, screen_set, root=None, loader=None):
        self.ss = screen_set
        self.root = root
        self.loader = loader
        self._scene = {}

    def _path(self, stream, vid):
        return os.path.join(self.root or io.FEATURE_ROOT, stream, vid + ".npz")

    def has(self, stream, vid):
        self.ss.guard(vid)
        if self.loader is not None:
            return self.loader(stream, vid, probe=True)
        return os.path.isfile(self._path(stream, vid))

    def n_ticks(self, stream, vid):
        """Number of ticks of a stream without decoding its features."""
        self.ss.guard(vid)
        if self.loader is not None:
            return len(self.loader(stream, vid)[0])
        with np.load(self._path(stream, vid), allow_pickle=False) as z:
            return int(z["t_sec"].shape[0])

    def load(self, stream, vid):
        self.ss.guard(vid)
        if self.loader is not None:
            return self.loader(stream, vid)
        old = io.FEATURE_ROOT
        io.FEATURE_ROOT = self.root or old
        try:
            return io.load_features(stream, vid)
        finally:
            io.FEATURE_ROOT = old

    def scene(self, vid):
        """Static scene embedding of a video or None."""
        self.ss.guard(vid)
        if vid not in self._scene:
            if self.loader is not None:
                e = self.loader(SCENE_STREAM, vid, scene=True)
            else:
                p = self._path(SCENE_STREAM, vid)
                e = None
                if os.path.isfile(p):
                    with np.load(p, allow_pickle=False) as z:
                        e = np.asarray(z["emb"], dtype=np.float32).reshape(-1)
            self._scene[vid] = e
        return self._scene[vid]


def block_columns(block, meta_columns, width):
    cols = list(meta_columns) if meta_columns else [str(i) for i in range(width)]
    keep = np.arange(width)
    if block.include:
        keep = np.array([i for i in keep if cols[i] in block.include])
    if block.exclude:
        keep = np.array([i for i in keep if cols[i] not in block.exclude])
    return keep, [cols[i] for i in keep]


def load_block(source, block, vid, n_ticks=None):
    """``(x [T, D] float32, valid_fraction)`` aligned to ticks 1..T on the joint timeline of the block's streams.

    T is cut at ``n_ticks`` and at the last tick every stream has; ``columns`` come from the first stream's meta.
    """
    parts, names, tmax, valid = [], [], None, []
    for s in block.streams:
        t, x, v, meta = source.load(s, vid)
        keep, cols = block_columns(block, meta.get("columns"), x.shape[1])
        if not len(keep):
            raise ValueError(f"{block.name}: no columns selected in {s}")
        ti = np.round(t).astype(np.int64)
        if len(ti) and (np.diff(ti) != 1).any() or (len(ti) and ti[0] != 1):
            raise ValueError(f"{vid}/{s}: ticks are not 1..T")
        parts.append(x[:, keep])
        names += [f"{s}:{c}" for c in cols]
        valid.append(v)
        tmax = len(t) if tmax is None else min(tmax, len(t))
    if n_ticks is not None:
        tmax = min(tmax, n_ticks)
    x = np.concatenate([p[:tmax] for p in parts], axis=1).astype(np.float32)
    vf = float(np.mean(np.all([v[:tmax] for v in valid], axis=0))) if tmax else 0.0
    return x, vf, names


def cut_flags(source, vid, T):
    """Shot-cut flags per tick from the motion stream (all False when the stream is missing)."""
    try:
        t, x, v, meta = source.load(CUT_STREAM, vid)
    except (FileNotFoundError, OSError, KeyError):
        return np.zeros(T, bool), False
    cols = list(meta.get("columns") or [])
    if CUT_COLUMN not in cols:
        return np.zeros(T, bool), False
    c = x[:, cols.index(CUT_COLUMN)] > 0.5
    out = np.zeros(T, bool)
    n = min(T, len(c))
    out[:n] = c[:n]
    return out, True


def last_cut_index(cut):
    idx = np.where(cut, np.arange(len(cut)), -1)
    return np.maximum.accumulate(idx) if len(idx) else idx


def _csum(a):
    a = np.asarray(a)
    z = np.zeros((1,) + a.shape[1:], np.float64)
    return np.concatenate([z, np.cumsum(a, axis=0, dtype=np.float64)])


def window_left(T, W, cut_last):
    return np.maximum(np.maximum(np.arange(T) - W + 1, cut_last + 1), 0)


def range_max(x, left, right):
    """Max of ``x[left:right+1]`` per query (0 for empty windows); x is [T] or [T, D]."""
    T = len(x)
    out = np.zeros((len(left),) + x.shape[1:], x.dtype)
    n = right - left + 1
    if T == 0 or len(left) == 0:
        return out
    levels = [x]
    k = 1
    while (1 << k) <= T:
        prev, h = levels[-1], 1 << (k - 1)
        levels.append(np.maximum(prev[:-h], prev[h:]))
        k += 1
    kk = np.zeros(len(left), np.int64)
    ok = n > 0
    kk[ok] = np.floor(np.log2(n[ok])).astype(np.int64)
    for k in np.unique(kk[ok]):
        sel = np.flatnonzero(ok & (kk == k))
        out[sel] = np.maximum(levels[k][left[sel]], levels[k][right[sel] - (1 << k) + 1])
    return out


def window_features(x, cut_last, thr, windows=WINDOWS, slopes=()):
    """P1 summaries of ``x [T, D]``: per window mean, max and fraction above ``thr`` (stat-major columns),
    then slopes. Windows are cut at the clip start and after the latest shot cut and divided by the ticks they
    contain; an empty window gives zeros."""
    T, D = x.shape
    right = np.arange(T)
    cs = _csum(x)
    cf = _csum(x > thr[None, :])
    stats = {"mean": [], "max": [], "frac": []}
    for W in windows:
        left = window_left(T, W, cut_last)
        n = np.maximum(right - left + 1, 0)
        den = np.maximum(n, 1)[:, None]
        stats["mean"].append(((cs[right + 1] - cs[left]) / den).astype(np.float32))
        stats["frac"].append(((cf[right + 1] - cf[left]) / den).astype(np.float32))
        stats["max"].append(range_max(x, left, right))
    parts = stats["mean"] + stats["max"] + stats["frac"]
    if slopes:
        idx = np.arange(T, dtype=np.float64)[:, None]
        cxi = _csum(idx * x.astype(np.float64))
        ci = _csum(idx)
        cii = _csum(idx * idx)
        for W in slopes:
            left = window_left(T, W, cut_last)
            n = (right - left + 1).astype(np.float64)[:, None]
            sx = cs[right + 1] - cs[left]
            sxi = cxi[right + 1] - cxi[left]
            si = ci[right + 1] - ci[left]
            sii = cii[right + 1] - cii[left]
            den = n * sii - si * si
            with np.errstate(invalid="ignore", divide="ignore"):
                s = np.where((n >= 3) & (den > 1e-9), (n * sxi - si * sx) / den, 0.0)
            parts.append(s.astype(np.float32))
    return np.concatenate(parts, axis=1) if parts else np.zeros((T, 0), np.float32)


def feature_names(cols, windows=WINDOWS, slopes=()):
    names = [f"{c}|{stat}{W}" for stat in ("mean", "max", "frac") for W in windows for c in cols]
    return names + [f"{c}|slope{W}" for W in slopes for c in cols]


def early_baseline_subtract(x, seconds=BASELINE_SEC):
    """x minus the causal per-video early baseline (running mean over the first ``seconds`` ticks)."""
    T = len(x)
    cs = _csum(x)
    k = np.minimum(np.arange(T) + 1, seconds)
    base = cs[k] / k[:, None]
    return (x - base).astype(np.float32)


class Pls:
    """Weighted PLS1 (NIPALS); ``transform`` projects raw columns onto the components."""

    def __init__(self, mean, scale, rotation):
        self.mean, self.scale, self.rotation = mean, scale, rotation

    @classmethod
    def fit(cls, X, y, w, n_components=PLS_COMPONENTS):
        w = np.asarray(w, np.float64)
        w = w / w.sum()
        X = np.asarray(X, np.float64)
        mean = w @ X
        scale = np.sqrt(np.maximum(w @ np.square(X - mean), 1e-12))
        sw = np.sqrt(w)[:, None]
        Z = (X - mean) / scale * sw
        r = (np.asarray(y, np.float64) - w @ np.asarray(y, np.float64)) * sw[:, 0]
        W, P = [], []
        for _ in range(min(n_components, X.shape[1])):
            wk = Z.T @ r
            nrm = np.linalg.norm(wk)
            if nrm < 1e-10:
                break
            wk /= nrm
            tk = Z @ wk
            tt = tk @ tk
            if tt < 1e-12:
                break
            pk = Z.T @ tk / tt
            Z -= np.outer(tk, pk)
            r = r - (tk @ r / tt) * tk
            W.append(wk)
            P.append(pk)
        W, P = np.array(W).T, np.array(P).T
        R = W @ np.linalg.pinv(P.T @ W)
        return cls(mean.astype(np.float32), scale.astype(np.float32), R.astype(np.float32))

    def transform(self, X):
        return ((np.asarray(X, np.float32) - self.mean) / self.scale) @ self.rotation
