"""Static scene prior ``siglip2_scene0p``: SigLIP 2 embedding of the frame at t = 2 s with every person greyed out.

Output format and box source: docs/FEATURES.md, "scene0p".

Usage (spec: docs/FEATURES.md, "Bulk extraction run v2"):
    python -m anticipation.features.scene0p --selftest [--device cuda:1 --parity xd/<stem>]
    python -m anticipation.features.scene0p --manifest benchmark/extract_manifest_v1.csv benchmark/extract_manifest_v2.csv \
        --device cuda:1 --workers 3
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from anticipation.features import io, pose, siglip

EXTRACTOR = "siglip2_scene0p"
VERSION = 1
T_SEC = 2.0
GROW = 1.10
GREY = 128
BATCH = 32
OUT_ROOT = None


def out_path(video_id):
    return os.path.join(OUT_ROOT or io.FEATURE_ROOT, EXTRACTOR, video_id + ".npz")


def pose_frame(raw, t_sec=T_SEC):
    """Index of the latest sampled pose frame at or before ``t_sec`` (decoded ones preferred), or -1."""
    fps = float(raw["meta"]["video"]["fps"])
    real_t = raw["frame_idx"].astype(np.float64) / fps
    ok = np.flatnonzero((real_t <= t_sec + 1e-6) & raw["frame_ok"])
    return int(ok[-1]) if len(ok) else -1


def person_boxes(raw, k):
    """Boxes [n, 4] (x1, y1, x2, y2, native pixels) of pose sample ``k``."""
    a, b = int(raw["frame_ptr"][k]), int(raw["frame_ptr"][k + 1])
    return raw["det_box"][a:b].astype(np.float64)


def grey_boxes(frame, boxes, grow=GROW):
    """Copy of uint8 ``frame`` [H, W, 3] with each box (grown by ``grow`` about its centre, clipped) set to grey."""
    out = frame.copy()
    h, w = out.shape[:2]
    for x1, y1, x2, y2 in boxes:
        cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * grow, (y2 - y1) * grow
        a, b = int(np.floor(max(cx - bw / 2, 0))), int(np.floor(max(cy - bh / 2, 0)))
        c, d = int(np.ceil(min(cx + bw / 2, w))), int(np.ceil(min(cy + bh / 2, h)))
        if c > a and d > b:
            out[b:d, a:c] = GREY
    return out


def prepare(video_id, path):
    """Decode and mask one video: ``(masked frame, record)`` or ``(None, reason)`` when pose_raw is unusable."""
    if not os.path.exists(io.feature_path(pose.EXTRACTOR_RAW, video_id)):
        return None, "no_pose_raw"
    raw = pose.load_raw(video_id)
    k = pose_frame(raw)
    video = io.Video(path, num_threads=1)
    idx = int(video.frame_index(T_SEC))
    frame = video.vr[idx].asnumpy()
    if k < 0:
        boxes, pose_t = np.zeros((0, 4)), None
    else:
        boxes, pose_t = person_boxes(raw, k), float(raw["frame_idx"][k]) / video.info.fps
    return grey_boxes(frame, boxes), dict(video_id=video_id, t_frame=idx / video.info.fps, n_boxes=len(boxes),
                                          pose_frame_t=pose_t, raw_version=raw["meta"].get("version"),
                                          fps=video.info.fps, duration_sec=video.info.duration_sec)


def save(rec, emb):
    path = out_path(rec["video_id"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = dict(extractor=EXTRACTOR, version=VERSION, model_id=siglip.MODEL_ID, t_sec=T_SEC, grow=GROW, grey=GREY,
                box_source="pose_raw latest sample <= t_sec, tracked and untracked detections", D=int(emb.shape[0]),
                l2_normalized=True, **{k: v for k, v in rec.items() if k not in ("video_id", "t_frame", "n_boxes")})
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, emb=emb.astype(np.float16), t_frame=np.float64(rec["t_frame"]),
                        n_boxes=np.int64(rec["n_boxes"]), meta=json.dumps(meta))
    os.replace(tmp, path)


def embed(frames, model, device, dtype=torch.bfloat16):
    """uint8 frames [N, H, W, 3] -> L2-normalised float32 embeddings [N, 1152] (numpy), via the siglip2 pipeline."""
    px = torch.cat([siglip.normalize(siglip.resize_frames(f[None]), device, dtype) for f in frames])
    return siglip.embed_images(model, px, BATCH).cpu().numpy()


def read_manifests(paths):
    seen, rows = set(), []
    for p in paths:
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                if r["video_id"] not in seen:
                    seen.add(r["video_id"])
                    rows.append((r["video_id"], r["path"]))
    return rows


def run(manifests, device, workers, limit, overwrite):
    rows = read_manifests(manifests)
    if limit:
        rows = rows[:limit]
    todo = [r for r in rows if overwrite or not os.path.exists(out_path(r[0]))]
    print(f"[scene0p] {len(rows)} videos, {len(rows) - len(todo)} done, {len(todo)} to do", flush=True)
    log_dir = os.path.join(OUT_ROOT or io.FEATURE_ROOT, "_logs", "scene0p")
    os.makedirs(log_dir, exist_ok=True)
    model, _ = siglip.load_model(device)
    missing, failed, n_ok, t0 = [], [], 0, time.time()

    def work(r):
        try:
            return r[0], *prepare(*r)
        except Exception as e:  # noqa: BLE001
            return r[0], None, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(workers) as ex:
        for b0 in range(0, len(todo), BATCH):
            ok = []
            for vid, frame, rec in ex.map(work, todo[b0:b0 + BATCH]):
                if frame is not None:
                    ok.append((frame, rec))
                elif rec == "no_pose_raw":
                    missing.append(vid)
                else:
                    failed.append(f"{vid}\t{rec}")
            if ok:
                embs = embed([f for f, _ in ok], model, device)
                for (_, rec), e in zip(ok, embs):
                    save(rec, e)
                n_ok += len(ok)
            print(f"[scene0p] {min(b0 + BATCH, len(todo))}/{len(todo)} ok={n_ok} no_pose_raw={len(missing)} "
                  f"fail={len(failed)} {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(log_dir, "missing.txt"), "w") as f:
        f.write("".join(v + "\n" for v in missing))
    with open(os.path.join(log_dir, "failures.txt"), "w") as f:
        f.write("".join(v + "\n" for v in failed))
    return 0 if not failed else 1


def selftest(device, parity):
    fr = np.full((100, 200, 3), 50, dtype=np.uint8)
    out = grey_boxes(fr, np.array([[50.0, 20.0, 70.0, 60.0]]))
    assert out[40, 60, 0] == GREY and out[40, 47, 0] == 50 and out[40, 73, 0] == 50  # x: 60 +/- 11 -> [49, 71)
    assert out[40, 49, 0] == GREY and out[40, 70, 0] == GREY and out[40, 71, 0] == 50
    assert out[17, 60, 0] == 50 and out[18, 60, 0] == GREY and out[61, 60, 0] == GREY and out[62, 60, 0] == 50
    assert (fr == 50).all() and (grey_boxes(fr, np.zeros((0, 4))) == fr).all()
    edge = grey_boxes(fr, np.array([[-30.0, -30.0, 10.0, 10.0], [190.0, 90.0, 400.0, 300.0]]))
    assert edge[0, 0, 0] == GREY and edge[99, 199, 0] == GREY and edge[50, 100, 0] == 50
    raw = dict(meta=dict(video=dict(fps=24.0)), frame_idx=np.array([0, 5, 10, 48, 49, 60]),
               frame_ok=np.array([True, True, True, True, False, True]), frame_ptr=np.array([0, 1, 3, 3, 4, 4, 6]),
               det_box=np.arange(24, dtype=np.float32).reshape(6, 4))
    assert pose_frame(raw) == 3  # frame 48 = 2.0 s; frame 49 is later and undecoded; 60 is after 2 s
    assert pose_frame(dict(raw, frame_idx=np.array([60, 61, 62, 63, 64, 65]))) == -1
    assert (person_boxes(raw, 1) == raw["det_box"][1:3]).all() and len(person_boxes(raw, 2)) == 0
    print("scene0p selftest: geometry ok", flush=True)
    if device and parity:
        model, _ = siglip.load_model(device)
        path = [p for v, p in read_manifests([os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "benchmark", "extract_manifest_v1.csv")]) if v == parity][0]
        video = io.Video(path)
        e = embed(video.frames_at([2.0]), model, device)[0]
        t, feat, valid, _ = io.load_features("siglip2", parity)
        ref = feat[int(np.flatnonzero(t == 2.0)[0])]
        cos = float(e @ ref / (np.linalg.norm(e) * np.linalg.norm(ref)))
        assert cos > 0.999, cos
        print(f"scene0p selftest: unmasked tick-2 embedding matches siglip2 stream, cosine {cos:.6f}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", nargs="*", default=[])
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out_root", default="", help="write here instead of FEATURE_ROOT (tests); pose_raw is still read from FEATURE_ROOT")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--parity", default="", help="video_id for the --selftest stream comparison (needs --device)")
    args = ap.parse_args()
    global OUT_ROOT
    OUT_ROOT = args.out_root or None
    torch.set_num_threads(4)
    if args.selftest:
        return selftest(args.device, args.parity)
    if not args.manifest or not args.device:
        ap.error("--manifest and --device are required")
    return run(args.manifest, args.device, args.workers, args.limit, args.overwrite)


if __name__ == "__main__":
    sys.exit(main())
