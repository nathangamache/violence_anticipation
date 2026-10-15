"""Causal hand-crafted audio (DSP) features at 1 Hz ticks, plus the audio helpers shared by the audio extractors.

Extractor name: ``audio_dsp``. Output follows ``anticipation/features/io.py`` (one npz per video, ticks
1..floor(duration), ``feat`` float16 ``[T, D]``, ``valid`` bool ``[T]``, JSON ``meta`` with column names).

What it computes (all from 16 kHz mono audio, frames of 32 ms with a 10 ms hop, ``center=False``):

* window statistics over ``[t - window_sec, t]`` (default 3 s): RMS loudness in dBFS (mean / max / std /
  10th percentile), loudness of the window relative to the running median of all past 1-s blocks of the
  same video (causal "is it louder than usual here"), spectral centroid (kHz, mean / std), spectral flux
  (mean / max, half-wave rectified log-mel difference in dB), zero-crossing rate, fraction of near-silent
  frames, F0 from ``librosa.pyin`` on voiced frames (mean Hz, std and 5-95 percentile range in semitones),
  voiced fraction and onset rate (flux peaks per second);
* slopes (least squares, units per second) of six 1-s block series over the last 5, 15 and 30 s:
  loudness, centroid, flux, F0 (semitones), voiced fraction, onset rate.

Causality: every per-frame quantity uses only samples inside its own frame; pYIN (whose Viterbi pass is
non-causal over its input) and onset peak picking run independently on each 1-s block ``[k - 1, k]``; window
statistics only aggregate frames or blocks that lie fully inside ``[t - window_sec, t]``; slopes only use
blocks ``<= t``. The relative-loudness column uses the median over blocks ``[0, t]`` (the whole past, by
design). So the maximum look-back is 30 s except for that column; nothing ever looks past ``t``.

Validity: ``valid`` is False for every tick when the file has no audio stream or the whole track is digital
silence (UCF-Crime, see ``io.is_digital_silence``), and for single ticks whose window lies past the end of
the decoded audio, is digital silence (peak < 1e-4) or is near-silent (window RMS < ``NEAR_SILENCE_DB`` =
-65 dBFS: encoder noise or dither, inaudible). A track whose loudest 1-s block is below that level gets
``audio_status = "near_silence"`` (every whole-second window is then invalid). Invalid ticks have all-zero
features. Validity is decided per window only, so it is causal too.

Audio clock: ``io.load_audio`` returns samples from the audio stream's own start. ``load_track`` re-aligns
them to the video clock (``io.Video`` time 0 = first video frame) using the ffprobe ``start_time`` of both
streams. When audio starts after the video (by more than 1 ms), leading zeros are inserted, otherwise tick t
would see audio from after t. When it starts more than ``AV_EARLY_TOL_SEC`` (50 ms) earlier, the leading
audio is dropped. Shifts are whole 10 ms steps rounded towards "audio lags", so the aligned audio never leads
the video. Small early offsets (AAC encoder delay, about -21 ms on XD) are left alone.

Downmix: ``io.load_audio`` uses ffmpeg's default ``-ac 1`` matrix (stereo 0.707 (L + R); 5.1 FC + 0.707
(FL + FR) + 0.5 (BL + BR), LFE dropped). This is unnormalised (peaks can exceed 1.0; harmless in float) but
keeps centre-panned dialogue at the same level for stereo and 5.1 mixes, which a normalised matrix would not.
The channel count is stored in ``meta["audio_channels"]`` for layout and shortcut audits.

Shared helpers used by ``audio_events.py`` and ``voice.py``: ``load_track``, ``window_bounds``,
``tick_valid``, ``base_meta``, ``empty_result``, ``probe_streams``.

Usage (smoke test on one file; the shared runner ``anticipation/features/run_extract.py`` does the batch
work and writes ``FEATURE_ROOT/audio_dsp/<source>/<stem>.npz``):

    .venv/bin/python -m anticipation.features.audio_dsp <video.mp4> [--window 3] [--causal-cut 60]
    .venv/bin/python -m anticipation.features.audio_dsp --selftest     # synthetic checks of the shared helpers

From Python:

    from anticipation.features import audio_dsp
    t_sec, feat, valid, meta = audio_dsp.extract("/path/video.mp4", device="cpu")

CPU only (``device`` is accepted and ignored). Cost is dominated by pYIN: about 0.04 to 0.06 s of one CPU core per
second of audio, so run several worker processes in parallel for bulk extraction.
"""

import argparse
import json
import math
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from anticipation.features import io

EXTRACTOR = "audio_dsp"
VERSION = "1.1"              # 1.1: A/V start alignment, near-silence validity, ffprobe duration fallback
SR = 16000
SILENCE_THRESH = 1e-4        # same threshold as io.is_digital_silence
NEAR_SILENCE_DB = -65.0      # window RMS (dBFS) below this is inaudible (encoder noise, dither): invalid
END_TOLERANCE_SEC = 0.25     # a tick whose window ends at most this far past the decoded audio stays valid
AV_LATE_TOL_SEC = 0.001      # audio starting later than the video by more than this is always corrected
AV_EARLY_TOL_SEC = 0.05      # audio starting earlier by up to this is left alone (AAC encoder delay, muxer rounding)
AV_OFFSET_STEP_SEC = 0.01    # corrections are whole 10 ms steps, rounded so audio never leads the video

# Frame grid shared by every frame-level DSP quantity.
FRAME_LEN = 512              # 32 ms at 16 kHz
HOP = 160                    # 10 ms
N_MELS = 40
QUIET_DB = -60.0             # frames below this dBFS are "near silent" (centroid ignored, silence fraction)
DB_FLOOR = -100.0
ONSET_DELTA_DB = 2.0         # flux peak must exceed its trailing 100 ms mean by this many dB
ONSET_MIN_DB = -50.0         # and the frame must be at least this loud

# pYIN settings (per independent 1-s block).
F0_MIN, F0_MAX = 75.0, 800.0      # 75 Hz floor avoids locking onto 50/60 Hz hum and bass
F0_FRAME = 1024              # 64 ms
F0_HOP = 320                 # 20 ms: 2.7x faster than 10 ms; per-block F0 mean r=0.91, voiced frac r=0.93 vs 10 ms
F0_RESOLUTION = 0.1          # semitone fraction (librosa default; coarser settings lose most voiced frames)

SLOPE_WINDOWS = (5, 15, 30)
SLOPE_SERIES = ("loud_db", "centroid_khz", "flux_db", "f0_st", "voiced_frac", "onset_rate")

WINDOW_COLUMNS = [
    "rms_db_mean", "rms_db_max", "rms_db_std", "rms_db_p10", "rel_loud_db",
    "centroid_khz_mean", "centroid_khz_std", "flux_db_mean", "flux_db_max", "zcr_mean", "quiet_frac",
    "f0_mean_hz", "f0_std_st", "f0_range_st", "voiced_frac", "onset_rate",
]
COLUMNS = WINDOW_COLUMNS + [f"slope{k}_{s}" for k in SLOPE_WINDOWS for s in SLOPE_SERIES]


# ----------------------------------------------------------------------------------------------------------
# Shared helpers (also used by audio_events.py and voice.py)
# ----------------------------------------------------------------------------------------------------------

@dataclass
class AudioTrack:
    """Decoded mono audio of one video on the shared 1 Hz tick grid (aligned to the video clock).

    ``wav`` is zero-padded so every window slice is in range; ``n_samples`` is the real decoded length (after
    the A/V alignment shift). ``status`` is "ok", "near_silence" (audio present but no 1-s block reaches
    ``NEAR_SILENCE_DB``), "no_audio" (no stream or decode failure), "digital_silence" or "no_ticks". ``wav`` is
    None for the last three, so callers test ``track.wav is None`` to skip all-invalid videos.

    The remaining fields describe the stream for ``meta`` (``av_offset_sec`` = audio start minus video start
    from ffprobe; ``av_shift_sec`` = the correction applied; ``rms_db`` and ``loudest_block_db`` summarise the
    whole track and must never be used as features).
    """

    path: str
    sr: int
    t_sec: np.ndarray
    duration_sec: float
    wav: Optional[np.ndarray]
    n_samples: int
    status: str
    channels: int = 0
    layout: str = "none"
    av_offset_sec: Optional[float] = None
    av_shift_sec: float = 0.0
    rms_db: Optional[float] = None
    loudest_block_db: Optional[float] = None


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def probe_streams(video_path):
    """ffprobe summary of the first (non cover-art) video stream, the first audio stream and the container.

    Returns ``{"video": {"start", "duration"} | None, "audio": {"start", "duration", "channels", "layout"} |
    None, "format_duration": float | None}`` (times in seconds, None when ffprobe has no value). The audio
    stream is the one ``io.load_audio`` decodes (``-map 0:a:0``).
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=index,codec_type,start_time,duration,channels,channel_layout:stream_disposition=attached_pic"
         ":format=duration", "-of", "json", video_path],
        capture_output=True, text=True,
    )
    try:
        info = json.loads(out.stdout or "{}")
    except json.JSONDecodeError:
        info = {}
    video = audio = None
    for s in info.get("streams", []):
        kind = s.get("codec_type")
        if kind == "video" and video is None and not s.get("disposition", {}).get("attached_pic"):
            video = {"start": _num(s.get("start_time")), "duration": _num(s.get("duration"))}
        elif kind == "audio" and audio is None:
            audio = {"start": _num(s.get("start_time")), "duration": _num(s.get("duration")),
                     "channels": int(s.get("channels") or 0), "layout": s.get("channel_layout") or "unknown"}
    return {"video": video, "audio": audio, "format_duration": _num(info.get("format", {}).get("duration"))}


def video_duration(video_path, probe=None):
    """Duration with the same definition as ``io.Video`` (frames / fps) so audio ticks match video ticks.

    If decord cannot open the file, falls back to the ffprobe duration of the video stream, then of the audio
    stream, and only then of the container. The container duration is not used first because XD-Violence
    movie clips carry a chapter/data stream as long as the whole film (e.g. 1340 s for a 67 s clip).
    """
    try:
        return io.Video(video_path, num_threads=1).info.duration_sec
    except Exception:
        probe = probe or probe_streams(video_path)
        for d in ((probe["video"] or {}).get("duration"), (probe["audio"] or {}).get("duration"),
                  probe["format_duration"]):
            if d and d > 0:
                return float(d)
        return 0.0


def av_offset(probe):
    """``(measured, applied)`` audio-minus-video start offset in seconds (measured is None if unknown).

    Late audio (offset > ``AV_LATE_TOL_SEC``) is always corrected, because it would put future audio into a
    tick's window. Early audio is corrected only beyond ``AV_EARLY_TOL_SEC``; a stream-copy cut changes the
    typical XD offset (-21 ms) by about 1 ms, so a tolerance keeps those videos untouched and exact under
    ``io.check_causal``. The applied shift is ``ceil(offset / 10 ms) * 10 ms``: at least the offset when
    padding, at most its size when trimming, so the aligned audio lags the video by 0 to 10 ms, never leads.
    """
    v, a = probe["video"], probe["audio"]
    if not v or not a or v["start"] is None or a["start"] is None:
        return None, 0.0
    off = a["start"] - v["start"]
    if -AV_EARLY_TOL_SEC <= off <= AV_LATE_TOL_SEC:
        return off, 0.0
    return off, math.ceil(round(off / AV_OFFSET_STEP_SEC, 6)) * AV_OFFSET_STEP_SEC


def align_to_video(wav, shift_sec, sr):
    """Put sample 0 at video time 0: prepend zeros when audio starts late (shift > 0), drop audio before it."""
    k = int(round(shift_sec * sr))
    if k > 0:
        return np.concatenate([np.zeros(k, dtype=np.float32), wav])
    return wav[-k:] if k < 0 else wav


def _block_ms(wav, sr, chunk_blocks=256):
    """Mean square (float64) of consecutive 1-s blocks of ``wav``; a trailing partial block counts as one."""
    n_full = wav.size // sr
    ms = np.empty(n_full + (1 if wav.size % sr else 0), dtype=np.float64)
    for a in range(0, n_full, chunk_blocks):
        b = min(n_full, a + chunk_blocks)
        ms[a:b] = np.mean(np.square(wav[a * sr:b * sr].reshape(b - a, sr), dtype=np.float64), axis=1)
    if wav.size % sr:
        ms[-1] = np.mean(np.square(wav[n_full * sr:], dtype=np.float64))
    return ms


def _db(ms):
    return float(10.0 * np.log10(max(float(ms), 1e-20)))


def load_track(video_path, sr=SR, duration_sec=None):
    """Decode the audio of ``video_path`` at ``sr``, align it to the video clock and attach the tick grid.

    The waveform is zero-padded to ``floor(duration) * sr`` samples (plus nothing else), so ``wav[s:e]`` is
    always defined for any window ending at a tick. Use ``tick_valid`` to know which windows are real audio.
    """
    probe = probe_streams(video_path)
    if duration_sec is None:
        duration_sec = video_duration(video_path, probe)
    t_sec = io.tick_times(duration_sec)
    a = probe["audio"] or {}
    offset, shift = av_offset(probe)
    info = dict(channels=a.get("channels", 0), layout=a.get("layout", "none"),
                av_offset_sec=None if offset is None else round(offset, 6), av_shift_sec=round(shift, 3))
    if len(t_sec) == 0:
        return AudioTrack(video_path, sr, t_sec, duration_sec, None, 0, "no_ticks", **info)
    wav = io.load_audio(video_path, sr=sr)
    if wav is not None:
        wav = align_to_video(wav, shift, sr)
    if wav is None or wav.size == 0:
        return AudioTrack(video_path, sr, t_sec, duration_sec, None, 0, "no_audio", **info)
    if io.is_digital_silence(wav, SILENCE_THRESH):
        return AudioTrack(video_path, sr, t_sec, duration_sec, None, int(wav.size), "digital_silence", **info)
    wav = wav.astype(np.float32, copy=False)
    n = int(wav.size)
    blk = _block_ms(wav, sr)
    energy = blk[: n // sr].sum() * sr + (blk[-1] * (n % sr) if n % sr else 0.0)
    info.update(rms_db=round(_db(energy / n), 2), loudest_block_db=round(_db(blk.max()), 2))
    status = "near_silence" if info["loudest_block_db"] < NEAR_SILENCE_DB else "ok"
    need = int(t_sec[-1]) * sr
    if n < need:
        wav = np.concatenate([wav, np.zeros(need - n, dtype=np.float32)])
    return AudioTrack(video_path, sr, t_sec, duration_sec, wav, n, status, **info)


def window_bounds(t_sec, window_sec, sr):
    """Sample bounds ``[start, end)`` of the causal window ``[max(0, t - W), t]`` for each tick."""
    t = np.asarray(t_sec, dtype=np.float64)
    end = np.round(t * sr).astype(np.int64)
    start = np.round(np.maximum(0.0, t - window_sec) * sr).astype(np.int64)
    return start, end


def _hop_stats(wav, hop, chunk_hops=100000):
    """Per-hop peak |x| (float32) and sum of squares (float64) over consecutive ``hop``-sample hops."""
    H = wav.size // hop
    peak = np.empty(H, dtype=np.float32)
    ss = np.empty(H, dtype=np.float64)
    for a in range(0, H, chunk_hops):
        b = min(H, a + chunk_hops)
        x = wav[a * hop:b * hop].reshape(b - a, hop)
        peak[a:b] = np.abs(x).max(axis=1)
        ss[a:b] = np.sum(np.square(x, dtype=np.float64), axis=1)
    return peak, ss


def tick_valid(track, window_sec, tol_sec=END_TOLERANCE_SEC, min_rms_db=NEAR_SILENCE_DB):
    """Per-tick validity of the window ``[max(0, t - W), t]``.

    Valid means: the track has audio, the window ends at most ``tol_sec`` past the decoded audio, the window is
    not digital silence (peak >= 1e-4) and its RMS is at least ``min_rms_db`` dBFS (None disables the RMS test).
    Each decision uses only the window's own samples, so validity is causal.
    """
    T = len(track.t_sec)
    if track.wav is None:
        return np.zeros(T, dtype=bool)
    sr = track.sr
    start, end = window_bounds(track.t_sec, window_sec, sr)
    valid = (end <= track.n_samples + int(tol_sec * sr)) & (end > start)
    min_ms = 10.0 ** (min_rms_db / 10.0) if min_rms_db is not None else 0.0
    hop = sr // 100
    if hop > 0 and not (start % hop).any() and not (end % hop).any():
        peak, ss = _hop_stats(track.wav, hop)      # fast path: window bounds on the 10 ms grid (the usual case)
        for i in np.flatnonzero(valid):
            a, b = start[i] // hop, end[i] // hop
            valid[i] = peak[a:b].max() >= SILENCE_THRESH and ss[a:b].sum() / (end[i] - start[i]) >= min_ms
        return valid
    for i in np.flatnonzero(valid):
        seg = track.wav[start[i]:end[i]]
        valid[i] = np.abs(seg).max() >= SILENCE_THRESH and np.mean(np.square(seg, dtype=np.float64)) >= min_ms
    return valid


def base_meta(extractor, version, track, window_sec, columns, **extra):
    """Meta dict common to every audio extractor (``io.save_features`` serialises it to JSON).

    ``audio_rms_dbfs`` and ``audio_loudest_block_dbfs`` summarise the whole track (provenance and audits
    only; they are not causal features).
    """
    meta = {
        "extractor": extractor,
        "version": version,
        "window_sec": float(window_sec),
        "D": len(columns),
        "columns": list(columns),
        "sr": track.sr,
        "audio_status": track.status,
        "duration_sec": float(track.duration_sec),
        "audio_sec": float(track.n_samples / track.sr) if track.n_samples else 0.0,
        "n_ticks": int(len(track.t_sec)),
        "tick_rule": "tick t uses audio in [max(0, t - window_sec), t] only",
        "valid_rule": "window inside the decoded audio, peak >= %g, RMS >= %g dBFS" % (SILENCE_THRESH,
                                                                                      NEAR_SILENCE_DB),
        "audio_channels": int(track.channels),
        "audio_layout": track.layout,
        "downmix": "ffmpeg default -ac 1 matrix (io.load_audio; unnormalised, LFE dropped)",
        "av_offset_sec": track.av_offset_sec,
        "av_shift_sec": track.av_shift_sec,
        "audio_rms_dbfs": track.rms_db,
        "audio_loudest_block_dbfs": track.loudest_block_db,
    }
    meta.update(extra)
    return meta


def empty_result(track, n_cols, meta):
    """All-invalid output for a video without usable audio."""
    T = len(track.t_sec)
    meta = dict(meta, valid_frac=0.0)
    return track.t_sec, np.zeros((T, n_cols), dtype=np.float16), np.zeros(T, dtype=bool), meta


def pad_indices(sel, size):
    """Pad a batch of tick indices to ``size`` by repeating the last one (GPU extractors drop the extra rows).

    GPU kernels (cuBLAS, cuDNN, SDPA) can pick a different algorithm for a different batch size, so a tick
    computed in a short last batch (truncated video) differs from the same tick in a full batch at the 1e-2
    level in fp16. Fixed batch shapes make every tick independent of how long the video is.
    """
    sel = np.asarray(sel)
    if len(sel) >= size:
        return sel
    return np.concatenate([sel, np.full(size - len(sel), sel[-1], dtype=sel.dtype)])


# ----------------------------------------------------------------------------------------------------------
# Frame-level features
# ----------------------------------------------------------------------------------------------------------

def _frames(x, frame_len, hop):
    """Strided view [n, frame_len] of non-centred frames (frame j covers samples [j*hop, j*hop + frame_len))."""
    n = 1 + (len(x) - frame_len) // hop if len(x) >= frame_len else 0
    return np.lib.stride_tricks.as_strided(x, shape=(n, frame_len), strides=(x.strides[0] * hop, x.strides[0]))


def frame_features(wav, sr=SR, chunk_frames=6000):
    """Per-frame rms_db, centroid_hz, zcr, onset/flux envelope (dB) on the 10 ms grid, computed in chunks.

    flux[j] compares frame j with frame j - 1 (flux[0] = 0); callers must not use flux at the first frame of
    a window, because its predecessor lies 10 ms before the window.
    """
    import librosa

    n_frames = 1 + (len(wav) - FRAME_LEN) // HOP if len(wav) >= FRAME_LEN else 0
    rms_db = np.full(n_frames, DB_FLOOR, dtype=np.float32)
    centroid = np.zeros(n_frames, dtype=np.float32)
    zcr = np.zeros(n_frames, dtype=np.float32)
    logmel = np.zeros((n_frames, N_MELS), dtype=np.float32)
    window = np.hanning(FRAME_LEN + 1)[:-1].astype(np.float32)  # periodic Hann
    freqs = np.fft.rfftfreq(FRAME_LEN, 1.0 / sr).astype(np.float32)
    mel_fb = librosa.filters.mel(sr=sr, n_fft=FRAME_LEN, n_mels=N_MELS, fmin=50.0, fmax=sr / 2).astype(np.float32)
    for a in range(0, n_frames, chunk_frames):
        b = min(n_frames, a + chunk_frames)
        seg = wav[a * HOP: (b - 1) * HOP + FRAME_LEN]
        fr = _frames(seg, FRAME_LEN, HOP)
        ms = np.mean(fr.astype(np.float32) ** 2, axis=1)
        rms_db[a:b] = np.maximum(DB_FLOOR, 10.0 * np.log10(ms + 1e-12))
        sgn = np.signbit(fr)
        zcr[a:b] = np.mean(sgn[:, 1:] != sgn[:, :-1], axis=1)
        power = np.abs(np.fft.rfft(fr * window, axis=1)) ** 2
        tot = power.sum(axis=1)
        centroid[a:b] = np.where(tot > 0, (power @ freqs) / np.maximum(tot, 1e-20), 0.0)
        logmel[a:b] = 10.0 * np.log10(power @ mel_fb.T + 1e-10)
    flux = np.zeros(n_frames, dtype=np.float32)
    if n_frames > 1:
        flux[1:] = np.maximum(0.0, logmel[1:] - logmel[:-1]).mean(axis=1)
    return {"rms_db": rms_db, "centroid_hz": centroid, "zcr": zcr, "flux_db": flux}


def frame_range(start, end, frame_len=FRAME_LEN, hop=HOP):
    """Indices [a, b) of frames lying fully inside samples [start, end)."""
    a = -(-start // hop)
    b = (end - frame_len) // hop + 1
    return int(a), int(max(a, b))


def block_pitch(wav, n_blocks, sr=SR, valid_blocks=None):
    """pYIN on each 1-s block independently. Returns f0 [n_blocks, F] (NaN = unvoiced) and voiced flags."""
    import librosa

    n_f = 1 + (sr - F0_FRAME) // F0_HOP
    f0 = np.full((n_blocks, n_f), np.nan, dtype=np.float32)
    vf = np.zeros((n_blocks, n_f), dtype=bool)
    idx = np.arange(n_blocks) if valid_blocks is None else np.flatnonzero(valid_blocks)
    for a in range(0, len(idx), 256):
        sel = idx[a:a + 256]
        blocks = np.stack([wav[k * sr:(k + 1) * sr] for k in sel])
        f, v, _ = librosa.pyin(blocks, sr=sr, fmin=F0_MIN, fmax=F0_MAX, frame_length=F0_FRAME, hop_length=F0_HOP,
                               center=False, resolution=F0_RESOLUTION)
        f0[sel] = np.where(v, f, np.nan)
        vf[sel] = v
    return f0, vf


def block_onsets(flux, rms_db, n_blocks, sr=SR):
    """Onset count per 1-s block from peaks of the flux envelope, using only frames inside the block."""
    import librosa

    counts = np.zeros(n_blocks, dtype=np.float32)
    for k in range(n_blocks):
        a, b = frame_range(k * sr, (k + 1) * sr)
        a += 1  # flux at the first frame of the block looks at the previous block
        if b - a < 8:
            continue
        env = flux[a:b].astype(np.float32)
        peaks = librosa.util.peak_pick(env, pre_max=3, post_max=4, pre_avg=10, post_avg=1, delta=ONSET_DELTA_DB,
                                       wait=5)
        peaks = [p for p in peaks if rms_db[a + p] >= ONSET_MIN_DB]
        counts[k] = len(peaks)
    return counts


def _semitones(hz):
    return 12.0 * np.log2(np.asarray(hz, dtype=np.float64) / 55.0)


def _slope(y):
    """Least-squares slope of y over unit-spaced x, ignoring NaN; 0 when fewer than 3 finite points."""
    x = np.arange(len(y), dtype=np.float64)
    m = np.isfinite(y)
    if m.sum() < 3:
        return 0.0
    x, y = x[m], y[m]
    xc = x - x.mean()
    den = (xc ** 2).sum()
    return float((xc * (y - y.mean())).sum() / den) if den > 0 else 0.0


# ----------------------------------------------------------------------------------------------------------
# Extractor
# ----------------------------------------------------------------------------------------------------------

def extract(video_path, device="cpu", window_sec=3.0, track=None, **unused):
    """Return ``(t_sec [T], feat [T, D] float16, valid [T] bool, meta dict)`` for one video.

    ``window_sec`` must be a whole number of seconds (block-aligned pitch and onset statistics).
    ``track`` lets a caller reuse an already decoded 16 kHz ``AudioTrack``.
    """
    W = float(window_sec)
    if W < 1 or W != int(W):
        raise ValueError("audio_dsp window_sec must be a whole number of seconds >= 1")
    t_start = time.time()
    if track is None:
        track = load_track(video_path, sr=SR)
    meta = base_meta(EXTRACTOR, VERSION, track, W, COLUMNS, slope_windows_sec=list(SLOPE_WINDOWS),
                     max_lookback_sec=max(SLOPE_WINDOWS), full_past_columns=["rel_loud_db"],
                     frame_len=FRAME_LEN, hop=HOP, f0_range_hz=[F0_MIN, F0_MAX], pyin_frame=F0_FRAME, pyin_hop=F0_HOP,
                     onset_delta_db=ONSET_DELTA_DB, model="none (librosa DSP)")
    if unused:
        meta["ignored_cfg"] = sorted(unused)
    valid = tick_valid(track, W)
    if track.wav is None or not valid.any():     # no audio, or nothing audible: all rows invalid (and zero)
        meta["elapsed_sec"] = round(time.time() - t_start, 3)
        return empty_result(track, len(COLUMNS), meta)

    T = len(track.t_sec)
    sr = track.sr
    # 1-s blocks [k, k+1) (k = 0..T-1) correspond to ticks k+1; block k is usable if its tick window of 1 s is
    # (same rule as the windows: inside the audio, not digital silence, not near-silent).
    block_valid = tick_valid(track, 1.0)

    ff = frame_features(track.wav[: T * sr], sr)
    rms_db, cen, zcr, flux = ff["rms_db"], ff["centroid_hz"], ff["zcr"], ff["flux_db"]
    f0, vflag = block_pitch(track.wav, T, sr, valid_blocks=block_valid)
    onsets = block_onsets(flux, rms_db, T, sr)

    # 1-s block series (index k = tick k+1); NaN for invalid blocks.
    blk = {s: np.full(T, np.nan) for s in SLOPE_SERIES}
    for k in np.flatnonzero(block_valid):
        a, b = frame_range(k * sr, (k + 1) * sr)
        db = rms_db[a:b]
        loud = cen[a:b][db > QUIET_DB]
        blk["loud_db"][k] = db.mean()
        blk["centroid_khz"][k] = loud.mean() / 1000.0 if loud.size else 0.0
        blk["flux_db"][k] = flux[a + 1:b].mean()
        voiced = f0[k][np.isfinite(f0[k])]
        blk["f0_st"][k] = _semitones(voiced).mean() if voiced.size else np.nan
        blk["voiced_frac"][k] = vflag[k].mean()
        blk["onset_rate"][k] = onsets[k]

    feat = np.zeros((T, len(COLUMNS)), dtype=np.float32)
    starts, ends = window_bounds(track.t_sec, W, sr)
    past_loud = []
    for i in range(T):
        if np.isfinite(blk["loud_db"][i]):
            past_loud.append(blk["loud_db"][i])
        if not valid[i]:
            continue
        a, b = frame_range(starts[i], ends[i])
        db = rms_db[a:b]
        quiet = db <= QUIET_DB
        c = cen[a:b][~quiet] / 1000.0
        fl = flux[a + 1:b]
        k0 = int(starts[i] // sr)            # blocks k0..i lie fully inside the window
        f_win = f0[k0:i + 1].ravel()
        f_win = f_win[np.isfinite(f_win)]
        st = _semitones(f_win) if f_win.size else None
        row = [
            db.mean(), db.max(), db.std(), np.percentile(db, 10),
            db.mean() - (np.median(past_loud) if past_loud else db.mean()),
            c.mean() if c.size else 0.0, c.std() if c.size else 0.0,
            fl.mean() if fl.size else 0.0, fl.max() if fl.size else 0.0,
            zcr[a:b].mean(), quiet.mean(),
            f_win.mean() if f_win.size else 0.0,
            st.std() if st is not None else 0.0,
            (np.percentile(st, 95) - np.percentile(st, 5)) if st is not None else 0.0,
            vflag[k0:i + 1].mean(),
            onsets[k0:i + 1].sum() / float(i + 1 - k0),
        ]
        for k in SLOPE_WINDOWS:
            lo = max(0, i + 1 - k)
            row += [_slope(blk[s][lo:i + 1]) for s in SLOPE_SERIES]
        feat[i] = row
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    meta["valid_frac"] = float(valid.mean()) if T else 0.0
    meta["elapsed_sec"] = round(time.time() - t_start, 3)
    return track.t_sec, feat.astype(np.float16), valid, meta


# ----------------------------------------------------------------------------------------------------------
# Self-test (synthetic media, CPU only, about 20 s)
# ----------------------------------------------------------------------------------------------------------

def _synth_video(path, audio, sr, video_sec, audio_offset=0.0):
    """Black 25 fps H.264 video of ``video_sec`` s muxed with mono float ``audio`` as AAC.

    ``audio_offset`` delays the audio stream start (ffmpeg ``-itsoffset``). The container lasts as long as
    the longer stream.
    """
    raw = path + ".f32"
    np.asarray(audio, dtype=np.float32).tofile(raw)
    cmd = ["ffmpeg", "-y", "-v", "error", "-nostdin", "-f", "lavfi", "-i",
           f"color=c=black:s=64x64:r=25:d={video_sec}"]
    if audio_offset:
        cmd += ["-itsoffset", str(audio_offset)]
    cmd += ["-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", raw, "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", path]
    subprocess.run(cmd, check=True)
    os.remove(raw)
    return path


def selftest(verbose=True):
    """Check the shared audio helpers on synthetic videos; raises AssertionError on failure.

    1. A/V alignment: the audio stream starts 0.5 s after the video and a burst sits at container time 5.3 s.
       ``load_track`` must place it at 5.3 s on the video clock, so tick 5 cannot see it (``io.load_audio``
       alone puts it at about 4.8 s, inside tick 5's window: a future leak).
    2. Near-silence: a dither-only track (-80 dBFS RMS, peak above the digital-silence threshold) must get
       status "near_silence" and no valid tick; ``extract`` must return all-invalid zero rows.
    3. Window validity: dither / loud noise / dither (10 s each) must be valid exactly where the 3-s window
       overlaps the loud part (edges of +-1 tick excluded for AAC smearing).
    4. Duration fallback: with decord unavailable, a 6 s video with 10 s of audio must report about 6 s, not
       the 10 s container duration.
    """
    sr = 48000
    rng = np.random.default_rng(0)
    results = {}
    with tempfile.TemporaryDirectory() as d:
        # 1. A/V alignment
        n = 12 * sr
        x = 0.001 * rng.standard_normal(n)
        b0 = int(4.8 * sr)
        x[b0:b0 + int(0.05 * sr)] += 0.5 * np.sin(2 * np.pi * 1000 * np.arange(int(0.05 * sr)) / sr)
        p = _synth_video(os.path.join(d, "offset.mp4"), x, sr, 12, audio_offset=0.5)
        raw = io.load_audio(p, sr=SR)
        tr = load_track(p, sr=SR)
        t_raw = np.flatnonzero(np.abs(raw) > 0.1)[0] / SR
        t_new = np.flatnonzero(np.abs(tr.wav) > 0.1)[0] / SR
        start, end = window_bounds(tr.t_sec, 3.0, SR)
        leak_new = np.abs(tr.wav[start[4]:end[4]]).max() > 0.1          # tick 5 = index 4
        leak_raw = np.abs(raw[start[4]:end[4]]).max() > 0.1
        results["av_alignment"] = dict(av_offset_sec=tr.av_offset_sec, av_shift_sec=tr.av_shift_sec,
                                       burst_raw_sec=round(t_raw, 4), burst_aligned_sec=round(t_new, 4),
                                       tick5_sees_burst_raw=bool(leak_raw), tick5_sees_burst=bool(leak_new))
        assert abs(tr.av_shift_sec - 0.48) < 0.011, results["av_alignment"]
        assert abs(t_new - 5.3) < 0.015 and t_raw < 5.0, results["av_alignment"]
        assert leak_raw and not leak_new, results["av_alignment"]

        # 2. near-silent (dither-only) track
        p = _synth_video(os.path.join(d, "dither.mp4"), 1e-4 * rng.standard_normal(n), sr, 12)
        tr = load_track(p, sr=SR)
        v = tick_valid(tr, 3.0)
        t, f, vv, m = extract(p, track=tr)
        results["near_silence"] = dict(status=tr.status, rms_db=tr.rms_db, loudest_block_db=tr.loudest_block_db,
                                       peak=float(np.abs(tr.wav).max()), n_valid=int(v.sum()),
                                       extract_valid=int(vv.sum()), extract_status=m["audio_status"])
        assert tr.status == "near_silence" and not v.any() and not vv.any() and not f.any(), results["near_silence"]
        assert float(np.abs(tr.wav).max()) >= SILENCE_THRESH, "dither must pass the old digital-silence test"

        # 3. dither / loud / dither
        y = 1e-4 * rng.standard_normal(30 * sr)
        y[10 * sr:20 * sr] = 0.05 * rng.standard_normal(10 * sr)
        p = _synth_video(os.path.join(d, "sil_loud_sil.mp4"), y, sr, 30)
        tr = load_track(p, sr=SR)
        v = tick_valid(tr, 3.0)
        on = [int(t) for t in tr.t_sec[v]]
        results["window_validity"] = dict(status=tr.status, valid_ticks=on)
        expect_on = set(range(12, 22))                        # windows clearly overlapping [10, 20]
        expect_off = set(range(1, 10)) | set(range(25, 31))  # windows clearly inside the dither parts
        assert expect_on <= set(on) and not (expect_off & set(on)), results["window_validity"]

        # 4. ffprobe duration fallback (decord made to fail)
        p = _synth_video(os.path.join(d, "long_audio.mp4"), 0.05 * rng.standard_normal(10 * sr), sr, 6)
        probe = probe_streams(p)
        orig = io.Video

        class _Broken:
            def __init__(self, *a, **k):
                raise RuntimeError("decord disabled for the self-test")

        io.Video = _Broken
        try:
            d_fb = video_duration(p)
        finally:
            io.Video = orig
        results["duration_fallback"] = dict(fallback_sec=d_fb, decord_sec=video_duration(p),
                                            container_sec=probe["format_duration"])
        assert abs(d_fb - 6.0) < 0.1 and probe["format_duration"] > 9.5, results["duration_fallback"]
    if verbose:
        for k, r in results.items():
            print("PASS", k, json.dumps(r))
    return results


def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video", nargs="?")
    ap.add_argument("--window", type=float, default=3.0)
    ap.add_argument("--causal-cut", type=float, default=0.0, help="also run io.check_causal at this cut (s)")
    ap.add_argument("--selftest", action="store_true", help="run the synthetic checks of the shared helpers")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        if not args.video:
            return
    if not args.video:
        ap.error("video is required unless --selftest is given")
    t, f, v, m = extract(args.video, window_sec=args.window)
    print(json.dumps({k: m[k] for k in ("audio_status", "n_ticks", "valid_frac", "elapsed_sec")}))
    print("feat", f.shape, f.dtype, "valid", int(v.sum()), "/", len(v))
    if args.causal_cut > 0:
        print(io.check_causal(lambda p: extract(p, window_sec=args.window)[:3], args.video, args.causal_cut))


if __name__ == "__main__":
    _main()
