"""Build benchmark/extract_manifest_v1.csv (bulk feature extraction order) and the 300 s clean-negative cuts.

Row order: violent eligible30, an early clean-negative sample, other violent, holdout, remaining clean negatives.
The sample holds at least --early_negative_frac of the clean-negative hours of every (fold, validation slice or
test, source) cell of --folds; 0 gives the original order.

Usage (from the repo root; see docs/FEATURES.md "Bulk extraction run v1"):
    nice -n 10 .venv/bin/python tools/build_extract_manifest.py [--workers 3] [--dry_run]
    .venv/bin/python tools/build_extract_manifest.py --reorder_only [--dry_run]   # re-order --out in place, no cuts
    nice -n 10 .venv/bin/python tools/build_extract_manifest.py --v2 [--workers 3] [--dry_run]
        # H1 hard negatives: other_anomaly videos with >= 30 s before first_anomaly_sec, cut at
        # min(first_anomaly_sec + 10, 300) s into cache/trunc_h1; writes benchmark/extract_manifest_v2.csv
"""

import argparse
import csv
import hashlib
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from anticipation.data.folds import FOLDS_DEFAULT, Folds  # noqa: E402
from anticipation.features import io  # noqa: E402

DATA_ROOT = os.environ.get("VA_DATA_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
TRUNC_ROOT = os.path.join(DATA_ROOT, "cache", "trunc300")
ROLES = ("violent", "holdout", "clean_negative")
CAP_SEC = 300
TRUNC_H1_ROOT = os.path.join(DATA_ROOT, "cache", "trunc_h1")
H1_MIN_LEAD_SEC = 30.0
H1_AFTER_SEC = 10.0
SALT = "extract_v1"
SALT_V2 = "extract_v2"
EARLY_FRAC = 0.25
PRIORITY_NAMES = ["violent_eligible30", "violent", "holdout", "clean_negative"]
COLUMNS = ["video_id", "source", "source_kind", "role", "priority", "eligible30", "has_audio", "path", "duration_sec",
           "fps", "num_frames", "orig_path", "orig_duration_sec", "cap_sec"]


def priority(r):
    if r["role"] == "violent":
        return 0 if r["eligible30"] == "True" else 1
    return 2 if r["role"] == "holdout" else 3


def hash_key(r):
    return hashlib.sha256(f"{SALT}:{r['video_id']}".encode()).hexdigest()


def order_key(r):
    return priority(r), hash_key(r)


def negative_cells(r, folds):
    """Cells of one clean negative: its test fold, and every fold whose validation slice holds it."""
    cells = [(folds.fold_of(r["video_id"]), "test", r["source"])]
    cells += [(f, "val", r["source"]) for f in range(folds.n_folds) if folds.role_in_fold(r["video_id"], f) == "val"]
    return cells


def early_negative_ids(rows, folds, frac):
    """Clean negatives, taken in hash order while they help a cell that holds less than ``frac`` of its hours."""
    neg = sorted((r for r in rows if r["role"] == "clean_negative"), key=hash_key)
    cells = {r["video_id"]: negative_cells(r, folds) for r in neg}
    total, got, keep = defaultdict(float), defaultdict(float), set()
    for r in neg:
        for c in cells[r["video_id"]]:
            total[c] += float(r["duration_sec"])
    for r in neg:
        cs = cells[r["video_id"]]
        if any(got[c] < frac * total[c] for c in cs):
            keep.add(r["video_id"])
            for c in cs:
                got[c] += float(r["duration_sec"])
    return keep


def arrange(rows, frac, folds_path):
    """Rows in run order: eligible30, early negative sample, other violent, holdout, other clean negatives."""
    early = early_negative_ids(rows, Folds.load(folds_path), frac) if frac > 0 else set()
    group = {0: 0, 1: 2, 2: 3}

    def key(r):
        p = priority(r)
        return (1 if r["video_id"] in early else 4) if p == 3 else group[p], hash_key(r)

    return sorted(rows, key=key), early


def truncate(src, dst, cap=CAP_SEC):
    if os.path.exists(dst):
        return "exists"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst[:-4] + ".tmp.mp4"
    subprocess.run(["nice", "-n", "10", "ffmpeg", "-v", "error", "-nostdin", "-y", "-i", src, "-map", "0", "-map", "-0:d",
                    "-c", "copy", "-t", str(cap), tmp], check=True)
    os.replace(tmp, dst)
    return "made"


def verify(r, dst, cap=CAP_SEC):
    """Truncated file: same fps, ~cap s, same frames at 0 / cap/2 / cap-1 s, audio kept when the source had a stream."""
    a, b = io.Video(r["path"], num_threads=1), io.Video(dst, num_threads=1)
    t = [0.0, cap / 2, cap - 1.0]
    same = bool((a.frames_at(t) == b.frames_at(t)).all())
    out = dict(dur=b.info.duration_sec, frames=b.info.num_frames, fps=b.info.fps, same_frames=same,
               fps_ok=abs(a.info.fps - b.info.fps) < 1e-6)
    out["audio_stream"] = io.has_audio(dst)
    out["audio_ok"] = out["audio_stream"] == io.has_audio(r["path"])
    if r["has_audio"] == "True":
        wav = io.load_audio(dst)
        out["audio_sec"] = 0.0 if wav is None else wav.size / 16000
        out["audio_ok"] &= wav is not None and abs(out["audio_sec"] - cap) < 1.0
    out["ok"] = same and out["fps_ok"] and out["audio_ok"] and cap - 1 <= out["dur"] <= cap + 1
    return out


def h1_cut_sec(r):
    return min(float(r["first_anomaly_sec"]) + H1_AFTER_SEC, float(CAP_SEC))


def h1_rows(catalog):
    with open(catalog, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["role"] == "other_anomaly" and r["first_anomaly_sec"]
                and float(r["first_anomaly_sec"]) >= H1_MIN_LEAD_SEC]
    rows.sort(key=lambda r: hashlib.sha256(f"{SALT_V2}:{r['video_id']}".encode()).hexdigest())
    return rows


def main_v2(args):
    rows = h1_rows(args.catalog)
    todo = [r for r in rows if h1_cut_sec(r) < float(r["duration_sec"])]
    print(f"{len(rows)} H1 videos, {len(todo)} to cut, {len(rows) - len(todo)} shorter than their cut point "
          "(used uncut)", flush=True)
    if args.dry_run:
        return 0

    def work(r):
        cap = h1_cut_sec(r)
        dst = os.path.join(TRUNC_H1_ROOT, r["video_id"] + ".mp4")
        try:
            return r["video_id"], verify_h1(r, dst, cap)
        except Exception as e:  # noqa: BLE001
            return r["video_id"], dict(ok=False, error=f"{type(e).__name__}: {e}")

    def verify_h1(r, dst, cap):
        truncate(r["path"], dst, cap)
        return verify(r, dst, cap)

    with ThreadPoolExecutor(args.workers) as ex:
        res = {}
        for k, (vid, v) in enumerate(ex.map(work, todo)):
            res[vid] = v
            if not v["ok"]:
                print(f"FAIL {vid} {v}", flush=True)
            if (k + 1) % 50 == 0:
                print(f"{k + 1}/{len(todo)} cut and verified", flush=True)
    bad = [vid for vid, v in res.items() if not v["ok"]]
    if bad:
        print(f"{len(bad)} truncations failed verification; manifest not written", flush=True)
        return 1

    out_rows = []
    for r in rows:
        o = {c: r.get(c, "") for c in COLUMNS}
        o.update(priority="h1_hard_negative", eligible30="False", orig_path=r["path"],
                 orig_duration_sec=r["duration_sec"], cap_sec="", cut_reason="none")
        if r["video_id"] in res:
            v = res[r["video_id"]]
            cap = h1_cut_sec(r)
            o.update(path=os.path.join(TRUNC_H1_ROOT, r["video_id"] + ".mp4"), duration_sec=f"{v['dur']:.3f}",
                     num_frames=v["frames"], cap_sec=f"{cap:.3f}",
                     cut_reason="cap300" if cap >= CAP_SEC else "first_anomaly+10")
        out_rows.append(o)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS + ["cut_reason"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {args.out}: {len(out_rows)} videos, "
          f"{sum(float(o['duration_sec']) for o in out_rows) / 3600:.2f} h, {sum(o['has_audio'] == 'True' for o in out_rows)} "
          "with real audio", flush=True)
    for key in ("source_kind", "cut_reason"):
        cnt = defaultdict(lambda: [0, 0.0])
        for o in out_rows:
            cnt[o[key]][0] += 1
            cnt[o[key]][1] += float(o["duration_sec"]) / 3600
        print("  " + "; ".join(f"{k}: {n} videos, {h:.2f} h" for k, (n, h) in sorted(cnt.items())), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=os.path.join(REPO, "benchmark", "videos.csv"))
    ap.add_argument("--out", default=os.path.join(REPO, "benchmark", "extract_manifest_v1.csv"))
    ap.add_argument("--folds", default=FOLDS_DEFAULT)
    ap.add_argument("--early_negative_frac", type=float, default=EARLY_FRAC)
    ap.add_argument("--v2", action="store_true", help="build the H1 hard-negative manifest v2 (default --out changes)")
    ap.add_argument("--reorder_only", action="store_true", help="re-order the rows of --out; no cuts, rows unchanged")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if args.v2:
        if args.out == ap.get_default("out"):
            args.out = os.path.join(REPO, "benchmark", "extract_manifest_v2.csv")
        return main_v2(args)

    if args.reorder_only:
        with open(args.out, newline="") as f:
            out_rows = list(csv.DictReader(f))
        return write_manifest(out_rows, args, {})

    with open(args.catalog, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["role"] in ROLES]
    rows.sort(key=order_key)
    long_neg = [r for r in rows if r["role"] == "clean_negative" and float(r["duration_sec"]) > CAP_SEC]
    print(f"{len(rows)} videos, {len(long_neg)} clean negatives > {CAP_SEC} s to cut", flush=True)
    if args.dry_run:
        return 0

    def work(r):
        dst = os.path.join(TRUNC_ROOT, r["video_id"] + ".mp4")
        try:
            return r["video_id"], truncate(r["path"], dst), verify(r, dst)
        except Exception as e:  # noqa: BLE001
            return r["video_id"], "error", dict(ok=False, error=f"{type(e).__name__}: {e}")

    with ThreadPoolExecutor(args.workers) as ex:
        res = {}
        for k, (vid, how, v) in enumerate(ex.map(work, long_neg)):
            res[vid] = v
            if not v["ok"]:
                print(f"FAIL {vid} {how} {v}", flush=True)
            if (k + 1) % 50 == 0:
                print(f"{k + 1}/{len(long_neg)} cut and verified", flush=True)
    bad = [vid for vid, v in res.items() if not v["ok"]]
    if bad:
        print(f"{len(bad)} truncations failed verification; manifest not written", flush=True)
        return 1

    out_rows = []
    for r in rows:
        o = {c: r.get(c, "") for c in COLUMNS}
        o.update(priority=PRIORITY_NAMES[priority(r)], orig_path=r["path"], orig_duration_sec=r["duration_sec"],
                 cap_sec="")
        if r["video_id"] in res:
            v = res[r["video_id"]]
            o.update(path=os.path.join(TRUNC_ROOT, r["video_id"] + ".mp4"), duration_sec=f"{v['dur']:.3f}",
                     num_frames=v["frames"], cap_sec=CAP_SEC)
        out_rows.append(o)
    return write_manifest(out_rows, args, res)


def write_manifest(out_rows, args, res):
    out_rows, early = arrange(out_rows, args.early_negative_frac, args.folds)
    if not args.dry_run:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(out_rows)

    hours = sum(float(o["duration_sec"]) for o in out_rows) / 3600
    audio_h = sum(float(o["duration_sec"]) for o in out_rows if o["has_audio"] == "True") / 3600
    cut = f"; cut durations {min(v['dur'] for v in res.values()):.2f} to {max(v['dur'] for v in res.values()):.2f} s, " \
          "all frames and audio checks ok" if res else ""
    print(f"{'would write' if args.dry_run else 'wrote'} {args.out}: {len(out_rows)} videos, {hours:.1f} h, "
          f"{audio_h:.1f} h with real audio{cut}", flush=True)
    for p in PRIORITY_NAMES:
        sel = [o for o in out_rows if o["priority"] == p]
        print(f"  {p}: {len(sel)} videos, {sum(float(o['duration_sec']) for o in sel) / 3600:.1f} h", flush=True)
    sel = [o for o in out_rows if o["video_id"] in early]
    if sel:
        print(f"  early clean-negative sample: {len(sel)} videos, "
              f"{sum(float(o['duration_sec']) for o in sel) / 3600:.1f} h", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
