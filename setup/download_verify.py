#!/usr/bin/env python3
"""ffprobe a sample (or all) of the videos in a folder and summarize streams, durations and audio presence.

Used to sanity-check freshly downloaded datasets (XD-Violence train, UBI-Fights). It reads the media only
through ``ffprobe`` (no Python decoding of untrusted files) and never modifies anything.

For each probed file it reports: video codec, WxH, fps, duration (s) of the VIDEO stream, whether an audio stream
exists, and the audio codec / sample rate / channels. The container duration is reported separately because
XD-Violence movie clips carry a text/chapter data stream spanning the whole source movie, which inflates
``format.duration`` (e.g. 4,646 s for a 48 s clip). With ``--volume`` it also runs ``ffmpeg -af volumedetect`` on the first
``--volume-sec`` seconds to catch digital-silence tracks (UCF-Crime style, mean volume about -91 dB).

Usage (from the repo root)
    .venv/bin/python -I setup/download_verify.py data/raw_videos/xd_violence_train/videos --n 10
    .venv/bin/python -I setup/download_verify.py data/raw_videos/ubi_fights/videos --n 10 --recursive
    .venv/bin/python -I setup/download_verify.py <dir> --all --json /tmp/probe.json   # full audit
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys

VIDEO_EXT = (".mp4", ".avi", ".mkv", ".mov", ".webm", ".mpg", ".mpeg", ".m4v")


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
        capture_output=True, text=True, timeout=120,
    )
    if out.returncode != 0:
        return {"path": path, "error": out.stderr.strip()[:300]}
    d = json.loads(out.stdout)
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), None)
    a = next((s for s in d.get("streams", []) if s.get("codec_type") == "audio"), None)
    fps = None
    if v and v.get("avg_frame_rate", "0/0") not in ("0/0", ""):
        num, den = v["avg_frame_rate"].split("/")
        fps = float(num) / float(den) if float(den) else None
    fmt_dur = float(d["format"].get("duration", "nan"))
    vid_dur = float(v["duration"]) if v and v.get("duration") not in (None, "N/A") else float("nan")
    return {
        "path": path,
        "size_mb": round(int(d["format"].get("size", 0)) / 1e6, 2),
        # Video-stream duration. XD-Violence movie clips carry a chapter/text data stream that spans the whole
        # source movie, so the container (format) duration can be 100x too long; never trust it alone.
        "duration_s": round(vid_dur if vid_dur == vid_dur else fmt_dur, 2),
        "format_duration_s": round(fmt_dur, 2),
        "n_data_streams": sum(s.get("codec_type") == "data" for s in d.get("streams", [])),
        "video": None if v is None else f"{v.get('codec_name')} {v.get('width')}x{v.get('height')}",
        "fps": None if fps is None else round(fps, 3),
        "has_audio": a is not None,
        "audio": None if a is None else f"{a.get('codec_name')} {a.get('sample_rate')}Hz {a.get('channels')}ch",
    }


def mean_volume(path, seconds):
    out = subprocess.run(
        ["ffmpeg", "-v", "info", "-nostdin", "-t", str(seconds), "-i", path, "-map", "0:a:0",
         "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True, timeout=300,
    )
    m = re.search(r"mean_volume:\s*(-?[\d.]+|-inf) dB", out.stderr)
    return None if m is None else (float("-inf") if m.group(1) == "-inf" else float(m.group(1)))


def list_videos(root, recursive):
    if recursive:
        files = [os.path.join(dp, f) for dp, _, fs in os.walk(root) for f in fs]
    else:
        files = [os.path.join(root, f) for f in os.listdir(root)]
    return sorted(f for f in files if f.lower().endswith(VIDEO_EXT) and os.path.isfile(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--n", type=int, default=10, help="random sample size (ignored with --all)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--volume", action="store_true", help="also measure mean audio volume")
    ap.add_argument("--volume-sec", type=float, default=60.0)
    ap.add_argument("--json", default=None, help="write per-file results to this JSON path")
    args = ap.parse_args()

    files = list_videos(args.root, args.recursive)
    if not files:
        print(f"no video files under {args.root}")
        return 1
    sample = files if args.all else random.Random(args.seed).sample(files, min(args.n, len(files)))
    rows = []
    for f in sample:
        r = probe(f)
        if args.volume and r.get("has_audio"):
            r["mean_volume_db"] = mean_volume(f, args.volume_sec)
        rows.append(r)
        if "error" in r:
            print(f"ERROR {os.path.basename(f)}: {r['error']}")
        else:
            vol = f" vol {r['mean_volume_db']} dB" if "mean_volume_db" in r else ""
            print(f"{os.path.basename(f)[:70]:70s} {r['duration_s']:9.2f}s {r['video']} {r['fps']}fps "
                  f"audio={r['audio']}{vol}")

    ok = [r for r in rows if "error" not in r]
    durs = sorted(r["duration_s"] for r in ok)
    print(f"\nfiles in folder: {len(files)}; probed: {len(rows)}; errors: {len(rows) - len(ok)}")
    if ok:
        print(f"with audio stream: {sum(r['has_audio'] for r in ok)}/{len(ok)}")
        bad_fmt = [r for r in ok if r["format_duration_s"] > r["duration_s"] + 2]
        print(f"container duration longer than video stream by > 2 s: {len(bad_fmt)}/{len(ok)} "
              f"(data streams present in {sum(r['n_data_streams'] > 0 for r in ok)})")
        print(f"duration s: min {durs[0]:.1f}, median {durs[len(durs) // 2]:.1f}, max {durs[-1]:.1f}, "
              f"total {sum(durs) / 3600:.2f} h")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)
    return 0 if len(ok) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
