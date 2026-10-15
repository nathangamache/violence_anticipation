"""Build benchmark/extract_manifest_v1.csv (bulk feature extraction order) and the 300 s clean-negative cuts.

Usage (from the repo root; see docs/FEATURES.md "Bulk extraction run v1"):
    nice -n 10 .venv/bin/python tools/build_extract_manifest.py [--workers 3] [--dry_run]
"""

import argparse
import csv
import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from anticipation.features import io  # noqa: E402

DATA_ROOT = "data"
TRUNC_ROOT = os.path.join(DATA_ROOT, "cache", "trunc300")
ROLES = ("violent", "holdout", "clean_negative")
CAP_SEC = 300
SALT = "extract_v1"
COLUMNS = ["video_id", "source", "source_kind", "role", "priority", "eligible30", "has_audio", "path", "duration_sec",
           "fps", "num_frames", "orig_path", "orig_duration_sec", "cap_sec"]


def priority(r):
    if r["role"] == "violent":
        return 0 if r["eligible30"] == "True" else 1
    return 2 if r["role"] == "holdout" else 3


def order_key(r):
    return priority(r), hashlib.sha256(f"{SALT}:{r['video_id']}".encode()).hexdigest()


def truncate(src, dst):
    if os.path.exists(dst):
        return "exists"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst[:-4] + ".tmp.mp4"
    subprocess.run(["nice", "-n", "10", "ffmpeg", "-v", "error", "-nostdin", "-y", "-i", src, "-map", "0", "-map", "-0:d",
                    "-c", "copy", "-t", str(CAP_SEC), tmp], check=True)
    os.replace(tmp, dst)
    return "made"


def verify(r, dst):
    """Truncated file: same fps, ~300 s, same frames at 0 / 150 / 299 s, audio kept when the source had a stream."""
    a, b = io.Video(r["path"], num_threads=1), io.Video(dst, num_threads=1)
    t = [0.0, 150.0, 299.0]
    same = bool((a.frames_at(t) == b.frames_at(t)).all())
    out = dict(dur=b.info.duration_sec, frames=b.info.num_frames, fps=b.info.fps, same_frames=same,
               fps_ok=abs(a.info.fps - b.info.fps) < 1e-6)
    out["audio_stream"] = io.has_audio(dst)
    out["audio_ok"] = out["audio_stream"] == io.has_audio(r["path"])
    if r["has_audio"] == "True":
        wav = io.load_audio(dst)
        out["audio_sec"] = 0.0 if wav is None else wav.size / 16000
        out["audio_ok"] &= wav is not None and abs(out["audio_sec"] - CAP_SEC) < 1.0
    out["ok"] = same and out["fps_ok"] and out["audio_ok"] and CAP_SEC - 1 <= out["dur"] <= CAP_SEC + 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=os.path.join(REPO, "benchmark", "videos.csv"))
    ap.add_argument("--out", default=os.path.join(REPO, "benchmark", "extract_manifest_v1.csv"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

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
        o.update(priority=["violent_eligible30", "violent", "holdout", "clean_negative"][priority(r)],
                 orig_path=r["path"], orig_duration_sec=r["duration_sec"], cap_sec="")
        if r["video_id"] in res:
            v = res[r["video_id"]]
            o.update(path=os.path.join(TRUNC_ROOT, r["video_id"] + ".mp4"), duration_sec=f"{v['dur']:.3f}",
                     num_frames=v["frames"], cap_sec=CAP_SEC)
        out_rows.append(o)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out_rows)

    durs = [v["dur"] for v in res.values()]
    hours = sum(float(o["duration_sec"]) for o in out_rows) / 3600
    audio_h = sum(float(o["duration_sec"]) for o in out_rows if o["has_audio"] == "True") / 3600
    print(f"wrote {args.out}: {len(out_rows)} videos, {hours:.1f} h, {audio_h:.1f} h with real audio; cut durations "
          f"{min(durs):.2f} to {max(durs):.2f} s, all frames and audio checks ok", flush=True)
    for p in ("violent_eligible30", "violent", "holdout", "clean_negative"):
        sel = [o for o in out_rows if o["priority"] == p]
        print(f"  {p}: {len(sel)} videos, {sum(float(o['duration_sec']) for o in sel) / 3600:.1f} h", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
