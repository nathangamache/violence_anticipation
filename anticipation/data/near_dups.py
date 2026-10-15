"""Near-duplicate detection for the VA30 catalog: re-encoded copies and time-shifted overlapping clips.

The catalog's byte-level dedupe (``catalog.dedupe``) only catches byte-identical files. UCF-Crime and XD-Violence
also contain re-encoded copies under different names (``ucf/Shooting015_x264`` = ``ucf/Arrest009_x264``, several
UCF normal pairs, XD YouTube re-uploads). Left alone they land in different folds (train/test leakage) and, for the
violent pair, carry conflicting labels. This module finds them from tiny grayscale thumbnails.

Usage (from the repo root; the first fingerprint pass took 20 min for 6720 videos with 12 workers, then it is
cached per file; matching takes about 2 min):

    .venv/bin/python -m anticipation.data.near_dups fingerprint [--workers 12]   # cached + resumable
    .venv/bin/python -m anticipation.data.near_dups match                        # writes benchmark/near_duplicates.json
    .venv/bin/python -m anticipation.data.near_dups all                          # both

    # detached (the rule for jobs over 10 minutes):
    setsid nohup .venv/bin/python -m anticipation.data.near_dups all \
        > data/logs/va30_near_dups.log 2>&1 &

Then re-run ``python -m anticipation.data.catalog``: it reads ``benchmark/near_duplicates.json``, marks the
non-canonical copy of every *identical* pair as role ``duplicate`` and merges every near-duplicate pair into one
``split_group`` (the fold grouping key of folds_v2).

Fingerprint of one video (``FP_DIR/<sha1(path)>.npz``, keyed by path, size and mtime):
    head  [<=60, 12, 16] uint8  frames at t = 0.5, 1.5, ..., 59.5 s (decoded at 32x24, 2x2 mean-pooled)
    frac  [9, 12, 16]    uint8  frames at 5 %, 15 %, ..., 95 % of the clip
Matching (``match``):
    * identical  durations within max(1 s, 1 %), and the 9 fraction-aligned frames agree: at least 5 informative
                 frame pairs (pixel std >= 4 in both), median Pearson correlation >= 0.97 and median mean absolute
                 difference <= 3 grey levels. Typical re-encodes: corr > 0.99, MAD < 1. The catalog keeps one
                 canonical copy and marks the others role ``duplicate``.
    * similar    the same with MAD > 3 (a brightness / colour shift, or one static camera filmed twice): grouped
                 into one split group, both copies kept.
    * overlap    the head sequences agree at one integer-second lag: candidates from locality-sensitive hashing of
                 the head frames (48-bit sign hash, 3 bands of 16 bits, buckets with > 40 videos ignored), then at
                 least max(6, 60 % of the overlapping informative frames) frames with corr >= 0.95 and MAD <= 6 at
                 the best lag. Catches trimmed / time-shifted copies whose overlap starts in the first minute, and
                 the same static CCTV scene filmed twice (grouping those is intended: same scene, same fold).
Limits: overlaps that begin after the first minute of both clips, and copies with different crops or letterboxing,
are not detected. The result is a lower bound on duplication.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

from anticipation.data.annotations import CACHE_DIR

FP_DIR = os.path.join(CACHE_DIR, "fingerprints")
HEAD_SEC = 60
N_FRAC = 9
THUMB_HW = (12, 16)   # rows, cols after 2x2 pooling of a 24x32 decode
MIN_STD = 4.0         # a frame is informative if its pixel std is at least this (0-255 grey levels)


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


NEAR_DUPS_JSON = os.path.join(_repo_root(), "benchmark", "near_duplicates.json")


def _log(*a):
    print(*a, flush=True)


def fp_path(path):
    return os.path.join(FP_DIR, hashlib.sha1(path.encode()).hexdigest() + ".npz")


def _pool2(frames):
    """[N, 24, 32, 3] uint8 -> [N, 12, 16] uint8 grey (2x2 mean)."""
    g = frames.astype(np.float32).mean(-1)
    n, h, w = g.shape
    g = g[:, : h // 2 * 2, : w // 2 * 2].reshape(n, h // 2, 2, w // 2, 2).mean((2, 4))
    return np.clip(np.round(g), 0, 255).astype(np.uint8)


def fingerprint_file(path):
    """``{"head", "frac", "fps", "n", "error"}`` for one video (never raises)."""
    st = os.stat(path)
    out = {"path": path, "size": st.st_size, "mtime_ns": st.st_mtime_ns, "error": ""}
    try:
        import decord

        vr = decord.VideoReader(path, num_threads=1, width=32, height=24)
        n = len(vr)
        fps = float(vr.get_avg_fps()) or 25.0
        dur = n / fps
        head_t = np.arange(0.5, min(HEAD_SEC, dur), 1.0)
        head_idx = [min(n - 1, int(round(s * fps))) for s in head_t]
        frac_idx = [min(n - 1, int(n * q)) for q in np.linspace(0.05, 0.95, N_FRAC)]
        head = _pool2(vr.get_batch(head_idx).asnumpy()) if head_idx else np.zeros((0,) + THUMB_HW, np.uint8)
        frac = _pool2(vr.get_batch(frac_idx).asnumpy()) if n else np.zeros((0,) + THUMB_HW, np.uint8)
        out.update(head=head, frac=frac, fps=fps, n=n)
        del vr
    except Exception as e:  # unreadable file: no fingerprint
        out.update(head=np.zeros((0,) + THUMB_HW, np.uint8), frac=np.zeros((0,) + THUMB_HW, np.uint8),
                   fps=0.0, n=0, error=f"{type(e).__name__}: {e}")
    return out


def _cached(path):
    p = fp_path(path)
    if not os.path.exists(p):
        return None
    try:
        z = np.load(p, allow_pickle=False)
        st = os.stat(path)
        if int(z["size"]) != st.st_size or int(z["mtime_ns"]) != st.st_mtime_ns:
            return None
        return z
    except Exception:
        return None


def _work(path):
    rec = fingerprint_file(path)
    tmp = fp_path(path) + ".tmp.npz"
    np.savez(tmp, head=rec["head"], frac=rec["frac"], fps=rec["fps"], n=rec["n"], size=rec["size"],
             mtime_ns=rec["mtime_ns"], error=np.array(rec["error"]), path=np.array(path))
    os.replace(tmp, fp_path(path))
    return path, rec["error"]


def video_table(videos_csv=None):
    """``[(video_id, path, source_kind, group, duration_sec, role)]`` from the catalog."""
    import pandas as pd

    from anticipation.data.catalog import VIDEOS_CSV

    df = pd.read_csv(videos_csv or VIDEOS_CSV)
    df = df[df.duration_sec > 0]
    return list(df[["video_id", "path", "source_kind", "group", "duration_sec", "role"]].itertuples(index=False,
                                                                                                    name=None))


def fingerprint_all(rows, workers=12, log=_log):
    os.makedirs(FP_DIR, exist_ok=True)
    todo = [r[1] for r in rows if _cached(r[1]) is None]
    log(f"[near_dups] fingerprints: {len(rows)} videos, {len(rows) - len(todo)} cached, {len(todo)} to compute")
    if not todo:
        return
    t0 = time.time()
    errors = 0
    # Longest files last so progress is steady; chunksize 1 keeps workers busy.
    with Pool(max(1, workers)) as pool:
        for i, (_, err) in enumerate(pool.imap_unordered(_work, todo, chunksize=1)):
            errors += bool(err)
            if (i + 1) % 250 == 0 or i + 1 == len(todo):
                log(f"[near_dups]   {i + 1}/{len(todo)} ({time.time() - t0:.0f} s, {errors} unreadable)")


# ---------------------------------------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------------------------------------

def _norm(frames):
    """uint8 [N, h, w] -> (unit-norm centred vectors [N, D], informative mask [N], raw float [N, D])."""
    x = frames.reshape(len(frames), -1).astype(np.float32)
    mu = x.mean(1, keepdims=True)
    sd = x.std(1, keepdims=True)
    z = (x - mu) / np.maximum(sd, 1e-6) / np.sqrt(x.shape[1])
    return z, sd[:, 0] >= MIN_STD, x


def compare_aligned(fa, fb):
    """Fraction-aligned comparison: ``(n_informative, median corr, median MAD)``."""
    za, ia, xa = _norm(fa)
    zb, ib, xb = _norm(fb)
    n = min(len(za), len(zb))
    if n == 0:
        return 0, 0.0, 255.0
    m = ia[:n] & ib[:n]
    if not m.any():
        return 0, 0.0, 255.0
    corr = (za[:n] * zb[:n]).sum(1)[m]
    mad = np.abs(xa[:n] - xb[:n]).mean(1)[m]
    return int(m.sum()), float(np.median(corr)), float(np.median(mad))


def best_lag(ha, hb, max_lag=None):
    """Best integer-second lag between two head sequences.

    Returns ``(lag, n_match, n_overlap, median_corr)`` where frame i of ``a`` aligns with frame i + lag of ``b``;
    ``n_match`` counts informative overlapping frames with corr >= 0.95 and MAD <= 6.
    """
    za, ia, xa = _norm(ha)
    zb, ib, xb = _norm(hb)
    if len(za) == 0 or len(zb) == 0:
        return 0, 0, 0, 0.0
    C = za @ zb.T                                   # [na, nb] correlations
    na, nb = C.shape
    best = (0, 0, 0, 0.0)
    for lag in range(-(na - 1), nb):
        i0, i1 = max(0, -lag), min(na, nb - lag)
        if i1 - i0 <= 0:
            continue
        ii = np.arange(i0, i1)
        jj = ii + lag
        m = ia[ii] & ib[jj]
        n_ov = int(m.sum())
        if n_ov < 6:
            continue
        c = C[ii, jj][m]
        mad = np.abs(xa[ii][m] - xb[jj][m]).mean(1)
        n_match = int(((c >= 0.95) & (mad <= 6.0)).sum())
        if n_match > best[1]:
            best = (int(lag), n_match, n_ov, float(np.median(c)))
    return best


def _sign_hash(frames):
    """48-bit sign hash per frame: 2x2-pooled (6x8) centred thumbnail > 0."""
    x = frames.astype(np.float32)
    n, h, w = x.shape
    p = x.reshape(n, h // 2, 2, w // 2, 2).mean((2, 4)).reshape(n, -1)
    bits = (p > p.mean(1, keepdims=True)).astype(np.uint64)
    weights = (np.uint64(1) << np.arange(bits.shape[1], dtype=np.uint64))
    return (bits * weights).sum(1).astype(np.uint64)


def find_pairs(rows, log=_log, max_bucket=40):
    """All near-duplicate pairs among catalog ``rows`` with cached fingerprints."""
    fps = {}
    for vid, path, kind, group, dur, role in rows:
        z = _cached(path)
        if z is None or int(z["n"]) == 0:
            continue
        fps[vid] = {"head": z["head"], "frac": z["frac"], "dur": float(dur), "group": group, "kind": kind,
                    "role": role}
    vids = sorted(fps)
    log(f"[near_dups] matching {len(vids)} fingerprinted videos")
    pairs = {}

    # (a) identical: duration-matched, fraction-aligned.
    order = sorted(vids, key=lambda v: fps[v]["dur"])
    durs = np.array([fps[v]["dur"] for v in order])
    n_cand_a = 0
    for i, a in enumerate(order):
        tol = max(1.0, 0.01 * durs[i])
        j = i + 1
        while j < len(order) and durs[j] - durs[i] <= tol:
            b = order[j]
            j += 1
            n_cand_a += 1
            n_inf, corr, mad = compare_aligned(fps[a]["frac"], fps[b]["frac"])
            if n_inf >= 5 and corr >= 0.97:
                # identical content (re-encode) vs same scene with a brightness / colour shift or a static camera
                # filmed twice: both are grouped, only identical copies are dropped from evaluation.
                pairs[tuple(sorted((a, b)))] = {"kind": "identical" if mad <= 3.0 else "similar", "lag_sec": 0,
                                                "corr": round(corr, 4), "mad": round(mad, 3), "n_frames": n_inf}
    log(f"[near_dups] aligned: {n_cand_a} duration-matched candidate pairs -> "
        f"{sum(p['kind'] == 'identical' for p in pairs.values())} identical, "
        f"{sum(p['kind'] == 'similar' for p in pairs.values())} similar")

    # (b) overlap: LSH on head frames, verified by the best lag.
    buckets = defaultdict(set)
    for v in vids:
        h = fps[v]["head"]
        if len(h) == 0:
            continue
        _, inf, _ = _norm(h)
        hs = _sign_hash(h)[inf]
        for x in np.unique(hs):
            x = int(x)
            for band in range(3):
                buckets[(band, (x >> (16 * band)) & 0xFFFF)].add(v)
    hits = defaultdict(int)
    skipped = 0
    for members in buckets.values():
        if len(members) < 2:
            continue
        if len(members) > max_bucket:
            skipped += 1
            continue
        ms = sorted(members)
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                hits[(ms[i], ms[j])] += 1
    cands = [p for p, c in hits.items() if c >= 3 and p not in pairs]
    log(f"[near_dups] overlap: {len(buckets)} LSH buckets ({skipped} oversized skipped), {len(cands)} candidates")
    n_ov = 0
    for a, b in cands:
        lag, n_match, n_overlap, corr = best_lag(fps[a]["head"], fps[b]["head"])
        if n_match >= max(6, int(np.ceil(0.6 * n_overlap))):
            pairs[(a, b)] = {"kind": "overlap", "lag_sec": lag, "corr": round(corr, 4), "mad": None,
                             "n_frames": n_match, "n_overlap": n_overlap}
            n_ov += 1
    log(f"[near_dups] overlap: {n_ov} verified pairs")
    out = []
    for (a, b), p in sorted(pairs.items()):
        out.append({"a": a, "b": b, **p, "same_group": fps[a]["group"] == fps[b]["group"],
                    "roles": [fps[a]["role"], fps[b]["role"]], "kinds": [fps[a]["kind"], fps[b]["kind"]]})
    return out, len(vids)


def clusters(pairs):
    """Connected components (union-find) of the pair list: ``[[vid, ...], ...]`` sorted."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in pairs:
        ra, rb = find(p["a"]), find(p["b"])
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    comp = defaultdict(list)
    for x in list(parent):
        comp[find(x)].append(x)
    return sorted(sorted(c) for c in comp.values())


def load_near_dups(path=NEAR_DUPS_JSON):
    """The committed near-duplicate file, or None."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["fingerprint", "match", "all"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--videos", default=None, help="catalog videos.csv (default benchmark/videos.csv)")
    ap.add_argument("--out", default=NEAR_DUPS_JSON)
    args = ap.parse_args(argv)
    rows = video_table(args.videos)
    if args.step in ("fingerprint", "all"):
        fingerprint_all(rows, workers=args.workers)
    if args.step in ("match", "all"):
        t0 = time.time()
        pairs, n = find_pairs(rows)
        cl = clusters(pairs)
        d = {"meta": {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "videos_fingerprinted": n,
                      "rules": {"identical": "durations within max(1 s, 1 %); >= 5 informative fraction-aligned "
                                             "frames, median corr >= 0.97, median MAD <= 3",
                                "overlap": "LSH head-frame candidates; >= max(6, 60 % of overlap) frames with "
                                           "corr >= 0.95 and MAD <= 6 at the best integer-second lag"},
                      "n_pairs": len(pairs), "n_identical": sum(p["kind"] == "identical" for p in pairs),
                      "n_overlap": sum(p["kind"] == "overlap" for p in pairs), "n_clusters": len(cl)},
             "pairs": pairs, "clusters": cl}
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out + ".tmp", "w") as f:
            json.dump(d, f, indent=1)
        os.replace(args.out + ".tmp", args.out)
        _log(f"[near_dups] wrote {args.out}: {d['meta']['n_pairs']} pairs ({d['meta']['n_identical']} identical, "
             f"{d['meta']['n_overlap']} overlap), {len(cl)} clusters ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
