"""Ingest NTU CCTV-Fights (external test set): verify videos, build catalog/events, overlap check, manifest.

Usage (repo root; the catalog files are written to benchmark/external/cctv_fights/):
    python setup/ingest_cctv_fights.py verify [--workers 3]    # decode + audio probe -> cache/cctv_fights/probe.json
    python setup/ingest_cctv_fights.py transcode               # re-encode files that failed to open (crf 16 mp4)
    python setup/ingest_cctv_fights.py build                   # videos.csv, events.json, summary.json
    python setup/ingest_cctv_fights.py overlap [--workers 3]   # overlap_with_main.json + flag in videos.csv
    python setup/ingest_cctv_fights.py manifest                # extract_manifest.csv (+ audio-only manifest)
Design and caveats: docs/EXTERNAL_TEST.md.
"""

import argparse
import json
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anticipation.data import external_cctv as ext  # noqa: E402


def _probe(path):
    from anticipation.features import io

    rec = {"path": path, "error": ""}
    try:
        v = io.Video(path, num_threads=1)
        i = v.info
        rec.update(fps=i.fps, num_frames=i.num_frames, duration_sec=i.duration_sec, width=i.width, height=i.height)
        n = i.num_frames
        v.vr.get_batch(sorted({0, n // 2, n - 1})).asnumpy()
        del v
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return rec
    try:
        stream = io.has_audio(path)
        peak = 0.0
        silent = True
        if stream:
            wav = io.load_audio(path)
            silent = io.is_digital_silence(wav)
            peak = 0.0 if wav is None or wav.size == 0 else float(np.abs(wav).max())
        rec.update(audio_stream=bool(stream), audio_peak=peak, has_audio=bool(stream and not silent))
    except Exception as e:
        rec.update(audio_stream=False, audio_peak=0.0, has_audio=False, error=f"audio: {e}")
    return rec


def cmd_verify(a):
    gt = ext.load_groundtruth()
    paths = [ext.video_path(s) for s in sorted(gt)]
    with Pool(a.workers) as pool:
        recs = pool.map(_probe, paths, chunksize=4)
    out = {os.path.splitext(os.path.basename(r["path"]))[0]: r for r in recs}
    os.makedirs(os.path.dirname(ext.PROBE_JSON), exist_ok=True)
    with open(ext.PROBE_JSON, "w") as f:
        json.dump(out, f, indent=0)
    bad = [k for k, r in out.items() if r["error"] and "fps" not in r]
    print(f"probed {len(out)}, decode failures {len(bad)}: {bad}")
    dd = [(k, r["duration_sec"], gt[k]["duration"]) for k, r in out.items() if "fps" in r
          and abs(r["duration_sec"] - gt[k]["duration"]) > max(1.0, 0.02 * gt[k]["duration"])]
    print(f"duration mismatches (> max(1 s, 2 %)): {len(dd)}")
    for k, d, g in dd[:30]:
        print(f"  {k} decord {d:.1f} gt {g:.1f}")
    print(f"real audio: {sum(bool(r.get('has_audio')) for r in out.values())}")


def _nominal_fps(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
    num, _, den = out.partition("/")
    try:
        fps = float(num) / float(den or 1)
    except ValueError:
        fps = 0.0
    return fps if 5 <= fps <= 60 else 30.0


def cmd_transcode(a):
    """All .webm go to constant-frame-rate mp4: their frame rate / count metadata is wrong for decord."""
    with open(ext.PROBE_JSON) as f:
        probes = json.load(f)
    log_path = os.path.join(ext.OUT_DIR, "transcoded.json")
    done = json.load(open(log_path)) if os.path.exists(log_path) else {}
    orig_dir = os.path.join(ext.RAW_DIR, "videos_original")
    os.makedirs(orig_dir, exist_ok=True)
    for stem, r in sorted(probes.items()):
        src = r["path"]
        if not src.endswith(".webm") or (stem in done and done[stem]["ok"]):
            continue
        fps = _nominal_fps(src)
        dst = os.path.join(ext.RAW_DIR, "videos", stem + ".mp4")
        tmp = dst + ".tmp.mp4"
        cmd = ["nice", "-n", "10", "ffmpeg", "-v", "error", "-nostdin", "-y", "-i", src, "-map", "0:v:0", "-map", "0:a:0?",
               "-r", f"{fps:.5f}", "-vsync", "cfr", "-c:v", "libx264", "-crf", "16", "-preset", "medium",
               "-threads", "3", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", tmp]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0 or not os.path.exists(tmp):
            done[stem] = {"src": src, "ok": False, "stderr": p.stderr[-300:]}
        else:
            os.replace(src, os.path.join(orig_dir, os.path.basename(src)))
            os.replace(tmp, dst)
            done[stem] = {"src": os.path.join(orig_dir, os.path.basename(src)), "dst": dst, "ok": True, "fps": fps,
                          "decord_fps_before": r.get("fps"), "decord_frames_before": r.get("num_frames")}
        print(stem, done[stem]["ok"], flush=True)
    with open(log_path, "w") as f:
        json.dump(done, f, indent=1)
    print("re-run verify to refresh the probe of transcoded files")


def cmd_build(a):
    ext.build_catalog()


def cmd_overlap(a):
    ext.run_overlap(workers=a.workers)


def cmd_manifest(a):
    ext.write_manifests()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("verify", cmd_verify), ("transcode", cmd_transcode), ("build", cmd_build),
                     ("overlap", cmd_overlap), ("manifest", cmd_manifest)):
        p = sub.add_parser(name)
        p.add_argument("--workers", type=int, default=3)
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
