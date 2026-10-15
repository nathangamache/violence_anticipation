"""Cheap causal motion features at 1 Hz (extractor ``motion``).

What
    Frames are sampled at ``sample_fps`` (default 5) on the video clock (``io.Video.sample_times``; each time maps
    to the last decoded frame at or before it) and decoded directly at a small size (``width`` = 160 px, aspect
    kept). For every consecutive pair of sampled frames we compute the statistics below. Rates divide by the
    real time between the two decoded frames, ``dt = (frame_idx[k] - frame_idx[k-1]) / fps`` (``frame_dt``): at
    24 fps the 0.2 s sampling grid lands on frame steps 4, 5, 5, 5, 5, so the nominal 0.2 s would be 17 % off on
    every fifth pair. Pairs that map to the same decoded frame (videos below 5 fps) are skipped.

    - absolute grey-level frame difference (mean, fraction of pixels changing by more than 8 %),
    - dense Farneback optical flow, expressed in frame widths per second so it is resolution independent:
      magnitude statistics, the median flow vector (global translation = camera pan proxy), the residual flow
      after subtracting that median (foreground motion), and the direction entropy of the residual flow,
    - the Bhattacharyya distance between HSV colour histograms (shot-cut detector: a pair is a cut when the
      distance exceeds ``cut_hist_thresh`` = 0.45 AND the mean grey difference exceeds ``cut_diff_thresh`` = 0.06;
      the second test rejects flashing police lights, which change the histogram but not the image structure;
      flow and difference statistics skip cut pairs),
    - global brightness and contrast of the current frame.

    Per 1 Hz tick t these are aggregated over the frames in ``[t - 1 s, t]`` (short window), ``[t - 5 s, t]``
    (long window), the time since the last detected cut (capped at 30 s), and causal least-squares slopes over
    5 / 15 / 30 s of past ticks. Every column is documented in ``COLUMN_DOCS`` and stored in ``meta["columns"]``
    and ``meta["column_docs"]``.

Causality
    Tick t only uses sampled frames with time <= t, so the lookback is at most 30 s (slopes, time since cut)
    plus the 1 s short window: ``meta["window_sec"] = 31``. ``io.check_causal`` passes exactly (max_abs_diff 0).
    Decode errors: frames are decoded in chunks, and a chunk that raises is re-decoded one frame at a time
    (``decode_frames``), so only the frames that really fail are missing. A bad frame at time t_bad therefore
    never changes ticks before t_bad (version 1 dropped the whole 40 s chunk, i.e. it changed past ticks).

Why
    Motion energy, camera shake and cuts are cheap context for the learned encoders and an explicit control for
    shortcuts (handheld phone footage shakes before a fight because the filmer expected one; movie clips cut).
    ``rolling_slope`` (causal OLS slope over past ticks) is shared with ``pose.py``.

Usage
    # batch (preferred), CPU only; run several shards in parallel
    python -m anticipation.features.run_extract --extractor motion --video_paths a.mp4 b.mp4 --device cpu
    # one video: print shape, timing and a few ticks (writes nothing unless --save <video_id>)
    python -m anticipation.features.motion --video a.mp4 [--print_ticks 40:50] [--check_causal 60]
    # self-test on synthetic videos (CPU, about 10 s): decode-error locality and real-dt flow at 24 fps
    python -m anticipation.features.motion --selftest

Versions
    1: first release. 2: per-frame decode fallback (a bad frame no longer drops its 40 s chunk), rates divide by
    the real time between decoded frames instead of the nominal 1 / sample_fps, duplicate frames are skipped.
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

from anticipation.features import io

EXTRACTOR = "motion"
VERSION = 2
OUTPUT_VERSIONS = {EXTRACTOR: VERSION}  # run_extract recomputes outputs whose meta["version"] differs

DEFAULTS = dict(
    sample_fps=5.0,          # frames per second sampled from the video clock
    width=160,               # decode width in pixels (height keeps the aspect ratio, rounded to even)
    window_sec=1.0,          # short aggregation window [t - window_sec, t]
    long_window_sec=5.0,     # long aggregation window
    slope_horizons=(5, 15, 30),
    cut_hist_thresh=0.45,    # a pair is a cut when the HSV-histogram Bhattacharyya distance exceeds this
    cut_diff_thresh=0.06,    # and the mean grey-level difference exceeds this (rejects flashing police lights)
    diff_pixel_thresh=0.08,  # grey-level change (0..1) counted as a moving pixel
    since_cut_cap_sec=30.0,
    chunk=200,               # frames decoded per batch
    num_threads=2,           # decord decoding threads
)

SHORT_COLS = [
    ("diff_mean", "mean abs(frame difference) (grey 0..1) over non-cut pairs in [t-1, t]"),
    ("diff_max", "max over pairs of the mean abs(frame difference)"),
    ("motion_area", "fraction of pixels whose grey level changes by > diff_pixel_thresh (pair mean)"),
    ("flow_mag_mean", "Farneback flow magnitude, frame widths per second (pair mean of the pixel mean)"),
    ("flow_mag_std", "pair mean of the pixel std of flow magnitude"),
    ("flow_mag_p90", "pair mean of the pixel 90th percentile of flow magnitude"),
    ("flow_mag_max", "max over pairs of the pixel 99th percentile of flow magnitude"),
    ("cam_flow_mag", "magnitude of the median flow vector (global translation, camera pan proxy), pair mean"),
    ("cam_shake", "std over pairs of the median flow vector (x and y combined); needs >= 2 pairs"),
    ("resid_flow_mean", "flow magnitude after subtracting the median flow vector (foreground motion), pair mean"),
    ("resid_flow_p90", "90th percentile of the residual flow magnitude, pair mean"),
    ("flow_dir_entropy", "magnitude-weighted 8-bin direction entropy of residual flow (0..1) over moving pixels"),
    ("hist_dist_max", "max HSV-histogram Bhattacharyya distance between consecutive sampled frames"),
    ("cut_1s", "1 if any pair in [t-1, t] is a shot cut"),
    ("brightness_mean", "mean grey level (0..1) of frames in [t-1, t]"),
    ("brightness_delta", "abs(brightness(last frame) - brightness(first frame)) in [t-1, t]"),
    ("contrast_mean", "mean grey-level std (0..1) of frames in [t-1, t]"),
]
LONG_COLS = [
    ("diff_mean_5s", "diff_mean over [t-5, t]"),
    ("flow_mag_mean_5s", "flow_mag_mean over [t-5, t]"),
    ("resid_flow_mean_5s", "resid_flow_mean over [t-5, t]"),
    ("cam_flow_mag_5s", "cam_flow_mag over [t-5, t]"),
    ("cuts_5s", "number of shot cuts in [t-5, t]"),
    ("flow_burst", "flow_mag_mean - flow_mag_mean_5s (sudden motion onset)"),
    ("time_since_cut", "seconds since the last detected cut, capped at since_cut_cap_sec (cap if none)"),
]
SLOPE_KEYS = ["flow_mag_mean", "resid_flow_mean"]


def _columns(slope_horizons):
    cols = list(SHORT_COLS) + list(LONG_COLS)
    for key in SLOPE_KEYS:
        for h in slope_horizons:
            cols.append((f"{key}_slope{h}s", f"causal OLS slope per second of {key} over ticks [t-{h}, t]"))
    return cols


COLUMNS = [c for c, _ in _columns(DEFAULTS["slope_horizons"])]
COLUMN_DOCS = dict(_columns(DEFAULTS["slope_horizons"]))


# ----------------------------------------------------------------------------------------------------------------
# Shared helpers (also used by pose.py)
# ----------------------------------------------------------------------------------------------------------------

def rolling_slope(x, defined, horizon, min_points=3):
    """Causal least-squares slope (units per second) of a 1 Hz tick series over ticks ``[t - horizon, t]``.

    ``x`` [T] float, ``defined`` [T] bool (ticks where ``x`` is meaningful). Only defined ticks in the window enter
    the fit; windows with fewer than ``min_points`` defined ticks get slope 0. Exact O(T * horizon) computation
    in window-relative time, so it stays well conditioned for arbitrarily long videos.
    """
    x = np.asarray(x, dtype=np.float64)
    m = np.asarray(defined, dtype=bool) & np.isfinite(x)
    T = len(x)
    if T == 0:
        return np.zeros(0, dtype=np.float32)
    L = int(horizon) + 1
    pad = np.zeros(L - 1)
    xw = np.lib.stride_tricks.sliding_window_view(np.concatenate([pad, np.where(m, x, 0.0)]), L)  # [T, L]
    ww = np.lib.stride_tricks.sliding_window_view(np.concatenate([pad, m.astype(np.float64)]), L)
    s = np.arange(L, dtype=np.float64) - (L - 1)  # seconds relative to the window end
    n = ww.sum(1)
    st, stt = ww @ s, ww @ (s * s)
    sx, stx = xw.sum(1), xw @ s
    nn = np.maximum(n, 1.0)
    cov = stx - st * sx / nn
    var = stt - st * st / nn
    ok = (n >= min_points) & (var > 1e-9)
    slope = np.zeros(T)
    slope[ok] = cov[ok] / var[ok]
    return slope.astype(np.float32)


def window_bounds(frame_t, ticks, window_sec, eps=1e-6):
    """Index ranges [k0, k1) of sampled frames with time in [t - window_sec, t] for every tick t."""
    k1 = np.searchsorted(frame_t, np.asarray(ticks) + eps, side="right")
    k0 = np.searchsorted(frame_t, np.asarray(ticks) - window_sec - eps, side="left")
    return k0, k1


def open_scaled(video_path, width=None, max_side=None, num_threads=2):
    """Open ``io.Video`` decoding at a reduced size (aspect kept, even dimensions).

    ``width``: target width. ``max_side``: only downscale when the long side exceeds it. Returns
    ``(video, native_wh)``.
    """
    import decord

    probe = decord.VideoReader(video_path, num_threads=1)
    h, w = probe[0].shape[:2]
    del probe
    scale = 1.0
    if width is not None:
        scale = width / float(w)
    elif max_side is not None and max(w, h) > max_side:
        scale = max_side / float(max(w, h))
    if abs(scale - 1.0) < 1e-6:
        return io.Video(video_path, num_threads=num_threads), (w, h)
    tw = max(2, int(round(w * scale / 2.0)) * 2)
    th = max(2, int(round(h * scale / 2.0)) * 2)
    return io.Video(video_path, num_threads=num_threads, width=tw, height=th), (w, h)


def decode_frames(video, times):
    """Decode frames at ``times`` with ``io.Video``: ``(frames [N, H, W, 3] uint8 or None, ok [N] bool)``.

    Tries one batched decode first. If it raises (corrupt stream, or a frame count that overstates the decodable
    frames), every time is decoded on its own, so one bad frame costs only that frame (its row is zero and
    ``ok`` is False). ``frames`` is None only when no frame decodes. Keeps decode errors local in time, which
    the causality rule needs: a failure at t_bad must not invalidate frames before t_bad. Shared with pose.py.
    """
    times = np.asarray(times, dtype=np.float64)
    ok = np.ones(len(times), dtype=bool)
    try:
        return video.frames_at(times), ok
    except Exception:  # noqa: BLE001  (decord raises a bare DECORDError on corrupt streams)
        pass
    frames = None
    for j in range(len(times)):
        try:
            fr = video.frames_at(times[j:j + 1])[0]
        except Exception:  # noqa: BLE001
            ok[j] = False
            continue
        if frames is None:
            frames = np.zeros((len(times),) + fr.shape, dtype=np.uint8)
        frames[j] = fr
    return frames, ok


def frame_dt(frame_idx, fps):
    """[F] seconds between decoded frame k-1 and frame k (``(idx[k] - idx[k-1]) / fps``; entry 0 is NaN).

    Use this, not the nominal sample spacing, for rates: sampling 5 fps from a 24 / 23.976 / 29.97 fps video
    maps to frame steps of 4 or 5 (6 or 5), i.e. 0.167 s or 0.208 s instead of 0.2 s. 0 means both samples map
    to the same decoded frame (videos below ``sample_fps``); callers skip such pairs.
    """
    idx = np.asarray(frame_idx, dtype=np.float64)
    dt = np.full(len(idx), np.nan)
    dt[1:] = np.diff(idx) / float(fps)
    return dt


# ----------------------------------------------------------------------------------------------------------------
# Per-frame and per-pair measurements
# ----------------------------------------------------------------------------------------------------------------

_PAIR_KEYS = ["diff_mean", "motion_area", "flow_mean", "flow_std", "flow_p90", "flow_p99", "cam_x", "cam_y",
              "resid_mean", "resid_p90", "dir_entropy", "hist_dist"]


def _frame_measure(rgb):
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return grey, hist, float(grey.mean()) / 255.0, float(grey.std()) / 255.0


def _pair_measure(g0, g1, h0, h1, dt, diff_pixel_thresh):
    """Motion statistics between two consecutive sampled grey frames ``dt`` seconds apart."""
    width = g0.shape[1]
    diff = np.abs(g1.astype(np.float32) - g0.astype(np.float32)) / 255.0
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow /= float(width) * dt  # frame widths per second
    fx, fy = flow[..., 0], flow[..., 1]
    mag = np.sqrt(fx * fx + fy * fy)
    cx, cy = float(np.median(fx)), float(np.median(fy))
    rx, ry = fx - cx, fy - cy
    rmag = np.sqrt(rx * rx + ry * ry)
    moving = rmag > 0.05  # 5 % of the frame width per second
    if moving.sum() >= 20:
        ang = np.arctan2(ry[moving], rx[moving])
        bins = ((ang + np.pi) / (2 * np.pi) * 8).astype(np.int64) % 8
        hw = np.bincount(bins, weights=rmag[moving], minlength=8)
        p = hw / max(hw.sum(), 1e-12)
        nz = p[p > 0]
        ent = float(-(nz * np.log(nz)).sum() / np.log(8))
    else:
        ent = 0.0
    p90, p99 = np.percentile(mag, [90, 99])
    return dict(
        diff_mean=float(diff.mean()), motion_area=float((diff > diff_pixel_thresh).mean()),
        flow_mean=float(mag.mean()), flow_std=float(mag.std()), flow_p90=float(p90), flow_p99=float(p99),
        cam_x=cx, cam_y=cy, resid_mean=float(rmag.mean()), resid_p90=float(np.percentile(rmag, 90)),
        dir_entropy=ent, hist_dist=float(cv2.compareHist(h0, h1, cv2.HISTCMP_BHATTACHARYYA)),
    )


def measure_video(video_path, cfg):
    """Decode sampled frames and compute per-frame and per-pair measurements.

    Returns a dict with ``frame_t`` [F], ``frame_ok`` [F], ``bright`` [F], ``contrast`` [F], ``pair_ok`` [F]
    (pair ending at frame k with frame k-1), and one [F] array per ``_PAIR_KEYS`` entry (NaN where no pair),
    plus ``info`` (decoded VideoInfo) and ``native_wh``.
    """
    video, native_wh = open_scaled(video_path, width=cfg["width"], num_threads=cfg["num_threads"])
    frame_t = video.sample_times(cfg["sample_fps"])
    frame_idx = video.frame_index(frame_t)
    dts = frame_dt(frame_idx, video.info.fps)
    F = len(frame_t)
    out = {k: np.full(F, np.nan, dtype=np.float64) for k in _PAIR_KEYS}
    frame_ok = np.zeros(F, dtype=bool)
    pair_ok = np.zeros(F, dtype=bool)
    bright = np.full(F, np.nan)
    contrast = np.full(F, np.nan)
    prev = None  # (k, grey, hist)
    for s in range(0, F, cfg["chunk"]):
        frames, ok = decode_frames(video, frame_t[s:s + cfg["chunk"]])
        for j in range(len(ok)):
            k = s + j
            if not ok[j]:  # decode error: this frame is missing, the pair chain restarts after it
                prev = None
                continue
            grey, hist, b, c = _frame_measure(frames[j])
            frame_ok[k], bright[k], contrast[k] = True, b, c
            if prev is not None and prev[0] == k - 1 and dts[k] > 0:  # dt 0: same decoded frame, no pair
                m = _pair_measure(prev[1], grey, prev[2], hist, float(dts[k]), cfg["diff_pixel_thresh"])
                for key in _PAIR_KEYS:
                    out[key][k] = m[key]
                pair_ok[k] = True
            prev = (k, grey, hist)
    out.update(frame_t=frame_t, frame_idx=frame_idx, frame_ok=frame_ok, pair_ok=pair_ok, bright=bright,
               contrast=contrast, info=video.info, native_wh=native_wh)
    return out


# ----------------------------------------------------------------------------------------------------------------
# Tick aggregation
# ----------------------------------------------------------------------------------------------------------------

def _nanmean(a):
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else np.nan


def _nanmax(a):
    a = a[np.isfinite(a)]
    return float(a.max()) if a.size else np.nan


def is_cut(m, cfg):
    """[F] bool: the pair ending at frame k is a shot cut (colour histogram jump AND large grey-level change)."""
    with np.errstate(invalid="ignore"):
        return m["pair_ok"] & (m["hist_dist"] > cfg["cut_hist_thresh"]) & (m["diff_mean"] > cfg["cut_diff_thresh"])


def aggregate(m, duration_sec, cfg):
    """Per-tick features from ``measure_video`` output. Returns ``(t_sec, feat, valid, columns)``."""
    ticks = io.tick_times(duration_sec)
    T = len(ticks)
    cols = [c for c, _ in _columns(cfg["slope_horizons"])]
    idx = {c: i for i, c in enumerate(cols)}
    feat = np.zeros((T, len(cols)), dtype=np.float64)
    valid = np.zeros(T, dtype=bool)
    if T == 0:
        return ticks, feat.astype(np.float32), valid, cols

    frame_t, frame_ok, pair_ok = m["frame_t"], m["frame_ok"], m["pair_ok"]
    cut = is_cut(m, cfg)
    use = pair_ok & ~cut  # pairs that enter motion statistics
    s0, s1 = window_bounds(frame_t, ticks, cfg["window_sec"])
    l0, l1 = window_bounds(frame_t, ticks, cfg["long_window_sec"])
    cut_times = frame_t[cut]
    for i in range(T):
        a, b = s0[i], s1[i]
        # a pair ending at frame k uses frame k-1, so only pairs with k-1 >= a stay inside the window
        pk = np.arange(a + 1, b)
        pk_use = pk[use[pk]] if len(pk) else pk
        fk = np.arange(a, b)
        fk = fk[frame_ok[fk]]
        valid[i] = len(fk) >= 2
        f = feat[i]
        if len(pk_use):
            f[idx["diff_mean"]] = _nanmean(m["diff_mean"][pk_use])
            f[idx["diff_max"]] = _nanmax(m["diff_mean"][pk_use])
            f[idx["motion_area"]] = _nanmean(m["motion_area"][pk_use])
            f[idx["flow_mag_mean"]] = _nanmean(m["flow_mean"][pk_use])
            f[idx["flow_mag_std"]] = _nanmean(m["flow_std"][pk_use])
            f[idx["flow_mag_p90"]] = _nanmean(m["flow_p90"][pk_use])
            f[idx["flow_mag_max"]] = _nanmax(m["flow_p99"][pk_use])
            cxy = np.stack([m["cam_x"][pk_use], m["cam_y"][pk_use]], 1)
            f[idx["cam_flow_mag"]] = float(np.linalg.norm(cxy, axis=1).mean())
            f[idx["cam_shake"]] = float(np.sqrt(cxy.var(0).sum())) if len(pk_use) >= 2 else 0.0
            f[idx["resid_flow_mean"]] = _nanmean(m["resid_mean"][pk_use])
            f[idx["resid_flow_p90"]] = _nanmean(m["resid_p90"][pk_use])
            f[idx["flow_dir_entropy"]] = _nanmean(m["dir_entropy"][pk_use])
        pk_any = pk[pair_ok[pk]] if len(pk) else pk
        if len(pk_any):
            f[idx["hist_dist_max"]] = _nanmax(m["hist_dist"][pk_any])
            f[idx["cut_1s"]] = float(cut[pk_any].any())
        if len(fk):
            f[idx["brightness_mean"]] = _nanmean(m["bright"][fk])
            f[idx["brightness_delta"]] = abs(m["bright"][fk[-1]] - m["bright"][fk[0]])
            f[idx["contrast_mean"]] = _nanmean(m["contrast"][fk])
        # long window
        lk = np.arange(l0[i] + 1, l1[i])
        lk_use = lk[use[lk]] if len(lk) else lk
        if len(lk_use):
            f[idx["diff_mean_5s"]] = _nanmean(m["diff_mean"][lk_use])
            f[idx["flow_mag_mean_5s"]] = _nanmean(m["flow_mean"][lk_use])
            f[idx["resid_flow_mean_5s"]] = _nanmean(m["resid_mean"][lk_use])
            cxy = np.stack([m["cam_x"][lk_use], m["cam_y"][lk_use]], 1)
            f[idx["cam_flow_mag_5s"]] = float(np.linalg.norm(cxy, axis=1).mean())
            f[idx["flow_burst"]] = f[idx["flow_mag_mean"]] - f[idx["flow_mag_mean_5s"]]
        f[idx["cuts_5s"]] = float(cut[lk].sum()) if len(lk) else 0.0
        past = cut_times[cut_times <= ticks[i] + 1e-6]
        f[idx["time_since_cut"]] = (min(ticks[i] - past[-1], cfg["since_cut_cap_sec"]) if len(past)
                                    else cfg["since_cut_cap_sec"])
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    for key in SLOPE_KEYS:
        for h in cfg["slope_horizons"]:
            feat[:, idx[f"{key}_slope{h}s"]] = rolling_slope(feat[:, idx[key]], valid, h)
    return ticks, feat.astype(np.float32), valid, cols


def extract(video_path, device="cpu", **cfg):
    """Registry entry point. ``device`` is ignored (CPU only). Returns ``(t_sec, feat, valid, meta)``."""
    cfg = {**DEFAULTS, **cfg}
    cfg["slope_horizons"] = tuple(int(h) for h in cfg["slope_horizons"])
    m = measure_video(video_path, cfg)
    info = m["info"]
    t_sec, feat, valid, cols = aggregate(m, info.duration_sec, cfg)
    docs = dict(_columns(cfg["slope_horizons"]))
    meta = dict(
        extractor=EXTRACTOR, version=VERSION, model_id="opencv-farneback", D=len(cols),
        window_sec=float(max(cfg["window_sec"], cfg["long_window_sec"],
                             max(cfg["slope_horizons"]) + cfg["window_sec"], cfg["since_cut_cap_sec"])),
        columns=cols, column_docs=[docs[c] for c in cols], config={k: cfg[k] for k in DEFAULTS},
        video=dict(fps=info.fps, num_frames=info.num_frames, duration_sec=info.duration_sec,
                   native_wh=list(m["native_wh"]), decoded_wh=[info.width, info.height]),
        n_frames_sampled=int(len(m["frame_t"])), n_frames_ok=int(m["frame_ok"].sum()),
        n_cuts=int(is_cut(m, cfg).sum()),
        units="flow in frame widths per second; brightness/contrast/diff in grey level 0..1",
        dt="rates use the real time between the two decoded frames, (frame_idx[k] - frame_idx[k-1]) / fps",
    )
    return t_sec, feat, valid, meta


# ----------------------------------------------------------------------------------------------------------------
# Self-test (synthetic videos, CPU)
# ----------------------------------------------------------------------------------------------------------------

def synthetic_video(path, fps=24.0, seconds=20.0, size=(320, 180), shift_px=2.0, seed=0):
    """Write a test mp4: a smooth random texture translating right by ``shift_px`` pixels per frame."""
    rng = np.random.default_rng(seed)
    w, h = size
    n = int(round(fps * seconds))
    tex = cv2.GaussianBlur((rng.random((h, w + int(shift_px * n) + 8)) * 255).astype(np.float32), (0, 0), 2.0)
    tex = cv2.normalize(tex, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        x0 = int(round(shift_px * (n - 1 - i)))  # content moves right as i grows
        g = tex[:, x0:x0 + w]
        wr.write(np.dstack([g, g, g]))
    wr.release()
    return path


def selftest():
    """Checks for the version-2 fixes; raises AssertionError on failure. Returns a dict of the numbers."""
    import tempfile

    res = {}
    with tempfile.TemporaryDirectory() as d:
        # 1) real dt: constant 2 px/frame pan at 24 fps -> sampled frame steps 4,5,5,5,5. The median-flow pan
        #    speed must be constant per pair (nominal dt = 0.2 s would make every 5th pair 17 % slower).
        path = synthetic_video(os.path.join(d, "pan24.mp4"), fps=24.0, seconds=12.0)
        m = measure_video(path, {**DEFAULTS})
        steps = np.diff(m["frame_idx"])
        cam = np.abs(m["cam_x"][m["pair_ok"]])
        expect = 2.0 * 24.0 / 320.0  # frame widths per second
        res["pan_steps"] = sorted(set(steps.tolist()))
        res["pan_speed_min_max_expected"] = (float(cam.min()), float(cam.max()), expect)
        assert set(steps.tolist()) == {4, 5}, steps
        assert cam.max() / cam.min() < 1.06, (cam.min(), cam.max())
        assert abs(np.median(cam) / expect - 1) < 0.05, (np.median(cam), expect)

        # 2) decode-error locality: one undecodable frame at t_bad must not change any tick before t_bad.
        t_bad = 9.4
        t0, f0, v0, _ = extract(path)
        orig = io.Video.frames_at

        def failing(self, t_sec):
            if (self.frame_index(t_sec) == int(np.floor(t_bad * self.info.fps + 1e-6))).any():
                raise RuntimeError("simulated decode error")
            return orig(self, t_sec)

        io.Video.frames_at = failing
        try:
            t1, f1, v1, _ = extract(path)
        finally:
            io.Video.frames_at = orig
        before = t1 < t_bad
        diff_before = float(np.abs(f0[before] - f1[before]).max())
        res["decode_error_diff_before_t_bad"] = diff_before
        res["decode_error_ticks_lost"] = int((v0 & ~v1).sum())
        assert diff_before == 0.0 and np.array_equal(v0[before], v1[before]), diff_before
        assert int((v0 & ~v1).sum()) == 0  # a single missing frame leaves >= 2 frames in every 1 s window
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video")
    ap.add_argument("--print_ticks", default="", help="range a:b of tick indices to print")
    ap.add_argument("--check_causal", type=float, default=0.0, help="cut point in seconds for io.check_causal")
    ap.add_argument("--save", default="", help="video_id to save under FEATURE_ROOT/motion/<video_id>.npz")
    ap.add_argument("--selftest", action="store_true", help="run the synthetic self-test and exit")
    args = ap.parse_args()
    if args.selftest:
        print("motion selftest ok", json.dumps(selftest()))
        return
    if not args.video:
        ap.error("--video is required (or --selftest)")
    t0 = time.time()
    t_sec, feat, valid, meta = extract(args.video)
    wall = time.time() - t0
    dur = meta["video"]["duration_sec"]
    print(f"T={len(t_sec)} D={feat.shape[1]} valid={valid.mean():.3f} cuts={meta['n_cuts']} "
          f"wall={wall:.1f}s speed={dur / max(wall, 1e-9):.1f}x realtime")
    if args.print_ticks:
        a, b = (int(x) for x in args.print_ticks.split(":"))
        cols = meta["columns"]
        for i in range(a, min(b, len(t_sec))):
            print(int(t_sec[i]), json.dumps({c: round(float(v), 4) for c, v in zip(cols, feat[i])}))
    if args.check_causal:
        res = io.check_causal(lambda p: extract(p)[:3], args.video, args.check_causal)
        print("check_causal", res)
    if args.save:
        print(io.save_features(EXTRACTOR, args.save, t_sec, feat, valid, meta))


if __name__ == "__main__":
    main()
