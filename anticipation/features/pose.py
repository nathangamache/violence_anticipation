"""Person detection, multi-object tracking, 2D keypoints and interaction features (``pose_raw``, ``pose_interact``).

What
    One decode + inference pass writes two feature files per video:

    ``pose_raw`` (FEATURE_ROOT/pose_raw/<source>/<stem>.npz)
        Every sampled frame (default 5 fps on the video clock, ``io.Video.sample_times``; each time maps to the
        last decoded frame at or before it) goes through Ultralytics YOLO11x-pose (person boxes + 17 COCO
        keypoints) and a BoT-SORT tracker (Kalman filter + sparse-optical-flow camera-motion compensation, no
        ReID). Ragged per-detection arrays in CSR layout are stored next to the contract keys; see ``save_raw``.
        Coordinates are pixels of the ORIGINAL video resolution.
    ``pose_interact`` (FEATURE_ROOT/pose_interact/<source>/<stem>.npz)
        Per 1 Hz tick t, 44 engineered person / dyad / crowd features from the sampled frames in
        ``[t - window_sec, t]`` (default 3 s = 16 frames) plus 21 causal slopes over 5 / 15 / 30 s of past ticks:
        D = 65. Every column name and its definition is in ``meta["columns"]`` / ``meta["column_docs"]`` and in
        ``WINDOW_COLS`` below. Distances and speeds are normalized by body height h, so they are resolution and
        zoom invariant. ``interaction_features`` is pure numpy on a ``pose_raw`` dict, so features can be
        re-derived on CPU without re-running the detector (registry name ``pose_interact``).

Body height, anchors, head yaw (version 2; see ``body_height``, ``TrackHeight``, ``_anchor_pair``, ``head_yaw``)
    h is a per-track causal median (last ``h_smooth_sec`` = 1 s) of a per-frame estimate that takes the max of the
    box height and a robust keypoint estimate (second-largest of several segment cues: torso, shoulder width,
    upper arms, thighs, shins, ear span), so it does not collapse to the upper-body box when the hips are hidden.
    Velocities use, per step and per person, the SAME keypoints in both frames (torso keypoints visible in both,
    else head keypoints, else the box center), so keypoint flicker never turns into speed. Head yaw is
    sin(yaw) from the nose offset relative to the ear (else eye) span, a continuous value; a lone ear / eye
    gives a fixed profile value of +/-0.9.

Causality
    Frames are processed strictly in time order; the tracker state (track ids, Kalman filter, camera-motion
    compensation) at frame k depends on frames 0..k only. Detection runs in fixed-size batches whose last batch
    is padded by repeating its final frame, so results never depend on how long the video is. Tick t uses frames
    with time <= t only. The tracker carries identity from [0, t] (allowed by VA30, "media from [0, t]");
    numeric features look back at most ``window_sec + max(slope_horizons)`` = 33 s (``meta["window_sec"]``).
    A batch that fails to decode is re-decoded frame by frame (``motion.decode_frames``), so a bad frame only
    removes itself and never changes earlier frames (version 1 dropped its whole 6.4 s batch).

Why
    Research brief (docs/research/README.md): the cues predictable 30 s ahead are escalation cues: approach and
    shrinking distance, face-offs, gesturing, crowd gathering. These are the most interpretable leading
    signals and the first ablation stream. The old 540-d pose vectors (removed legacy script, see tag pre-cleanup) were
    on the wrong clock, had no tracking and no velocities; this replaces them.

Usage
    # batch (preferred): writes pose_raw + pose_interact, skips existing outputs
    python -m anticipation.features.run_extract --extractor pose --video_paths a.mp4 --device cuda:0
    # recompute pose_interact from saved pose_raw after editing the feature code (CPU, no detector)
    python -m anticipation.features.run_extract --extractor pose_interact --video_paths a.mp4 --overwrite
    # one video: timing, a few ticks, optional causality check (writes nothing unless --save <video_id>)
    python -m anticipation.features.pose --video a.mp4 --device cuda:0 --print_ticks 40:50 --check_causal 60
    # self-test of the version-2 fixes (CPU, no detector, a few seconds)
    python -m anticipation.features.pose --selftest

Versions (``RAW_VERSION`` for pose_raw, ``VERSION`` for pose_interact; run_extract recomputes stale outputs)
    pose_raw 1 -> 2: the recorded GMC warp is identity when GMC raised (it kept the previous frame's warp) and
        ``frame_gmc_ok`` is stored; per-frame decode fallback.
    pose_interact 1 -> 2: consistent per-step anchors, multi-cue per-track body height, continuous head yaw,
        camera translation of the frame center (not the image origin), real time between decoded frames for
        every rate, steps with an unknown camera warp skipped, closing speeds from consistent anchors.

Weights
    ``weights`` is a file name or path. Names are looked up in models/ultralytics and
    downloaded from the HF mirror ``Ultralytics/YOLO11`` if missing; on failure the repo's ``yolov8m-pose.pt``
    is used (the model actually used is recorded in meta). Ultralytics is AGPL-3.0.
"""

import argparse
import hashlib
import json
import logging
import os
import time

import numpy as np
import torch  # noqa: F401  (must be imported before decord is first used, or CUDA init later segfaults)

from anticipation.features import io
from anticipation.features.motion import decode_frames, open_scaled, rolling_slope, window_bounds

EXTRACTOR_RAW = "pose_raw"
EXTRACTOR = "pose_interact"
OUTPUTS = (EXTRACTOR_RAW, EXTRACTOR)
RAW_VERSION = 2  # pose_raw (detection + tracking pass)
VERSION = 2      # pose_interact (feature code)
OUTPUT_VERSIONS = {EXTRACTOR_RAW: RAW_VERSION, EXTRACTOR: VERSION}  # run_extract recomputes mismatching outputs

WEIGHTS_DIR = "models/ultralytics"
HF_REPO = "Ultralytics/YOLO11"
FALLBACK_WEIGHTS = "yolov8m-pose.pt"

DEFAULTS = dict(
    # detection / tracking (pose_raw)
    weights="yolo11x-pose.pt",
    sample_fps=5.0,
    imgsz=640,
    batch=32,               # frames per detector forward pass (the last batch is padded to this size)
    half=True,
    max_side=960,           # decode with the long side capped at this many pixels (YOLO runs at imgsz anyway)
    det_conf=0.1,           # detector threshold; low-score boxes feed BYTE's second association stage
    save_conf=0.25,         # untracked detections below this are not saved
    nms_iou=0.7,
    tracker="botsort",      # botsort (with camera-motion compensation) or bytetrack
    track_buffer_sec=5.0,   # keep lost tracks alive this long
    gmc_downscale=3,        # BoT-SORT camera-motion compensation: image downscale factor (ultralytics default 2)
    gmc_max_corners=300,    # and number of tracked corners (default 1000; 300 is 3x faster, similar warps)
    num_threads=4,          # decord threads
    # interaction features (pose_interact)
    window_sec=3.0,
    slope_horizons=(5, 15, 30),
    kpt_conf=0.3,           # keypoint visibility threshold
    h_smooth_sec=1.0,       # body height = causal per-track median of the per-frame estimates over this span
)
RAW_KEYS = ("weights", "sample_fps", "imgsz", "batch", "half", "max_side", "det_conf", "save_conf", "nms_iou",
            "tracker", "track_buffer_sec", "gmc_downscale", "gmc_max_corners")
INTERACT_KEYS = ("window_sec", "slope_horizons", "kpt_conf", "h_smooth_sec")

KPT_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
             "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
             "right_knee", "left_ankle", "right_ankle"]
NOSE, LEYE, REYE, LEAR, REAR, LSH, RSH, LWR, RWR, LHIP, RHIP = 0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12
LELB, RELB, LKNE, RKNE, LANK, RANK = 7, 8, 13, 14, 15, 16
HEAD = [0, 1, 2, 3, 4]
TORSO_KPTS = [LSH, RSH, LHIP, RHIP]

FAR = 10.0  # fill value (body heights) for distance features when fewer than 2 persons are tracked

# Body-height cues: h ~ ratio x segment length. Ratios come from box height / segment over 2,816 full-body
# upright people in the five smoke videos: about the 25th percentile per segment (torso: p25 3.33, median 3.46;
# shoulder width between p10 5.27 and p25 6.04), because rotation and foreshortening only shorten a segment.
# The keypoint estimate is the largest cue, or the second-largest when >= 3 cues exist (robust to one bad
# keypoint); h_frame = max(box height, keypoint estimate). On full-body people h_frame / box height has median
# 1.02 (p90 1.16 after smoothing).
H_TORSO = 3.4  # shoulder midpoint to hip midpoint
H_CUES = [((LSH, RSH), 5.5), ((LSH, LELB), 6.3), ((RSH, RELB), 6.3), ((LHIP, LKNE), 4.5), ((RHIP, RKNE), 4.5),
          ((LKNE, LANK), 4.7), ((RKNE, RANK), 4.7), ((LEAR, REAR), 10.0)]
CENTER_OFF_SH = 0.14     # torso center below the shoulder midpoint (= above the hip midpoint), in h (median)
CENTER_OFF_HEAD = 0.245  # torso center below the mean of the visible head keypoints, in h (median)
# Head yaw: tan(yaw) = k x (nose_x - mid_x) / half_span. Ears: k = head half-width / nose depth (about 7 / 11 cm).
# Eyes: fitted to the ear estimate on people with both ears and both eyes visible (k_eye = 1.25 k_ear).
YAW_EAR = 0.65
YAW_EYE = 0.8
YAW_PROFILE = 0.9    # |sin(yaw)| when only one ear / eye is visible next to the nose (3/4 to profile view)
MIN_PAIR_GAP_SEC = 0.35  # closing speeds compare frames at least this far apart (two sampled steps)

# (name, fill value when undefined, definition). "frame mean" = mean over sampled frames in [t - W, t];
# "step" = consecutive sampled frames (nominally 0.2 s; rates divide by the real time between the two decoded
# frames). Units: body heights (h) and seconds.
WINDOW_COLS = [
    # presence and detection quality
    ("n_persons_mean", 0.0, "tracked persons per frame, frame mean"),
    ("n_persons_max", 0.0, "max tracked persons in a frame"),
    ("n_persons_last", 0.0, "tracked persons in the last frame <= t"),
    ("n_det_hiconf_mean", 0.0, "person detections with conf >= 0.5 per frame (tracked or not), frame mean"),
    ("det_conf_mean", 0.0, "mean detection confidence of tracked persons"),
    ("det_conf_max", 0.0, "max detection confidence of tracked persons"),
    ("kpt_conf_mean", 0.0, "mean keypoint confidence of tracked persons (all 17 keypoints)"),
    ("kpt_visible_frac", 0.0, "fraction of keypoints with conf >= kpt_conf"),
    ("new_tracks", 0.0, "track ids first seen inside the window (churn, occlusion, cuts)"),
    ("track_age_mean", 0.0, "mean age in s of tracks in the last frame (since first seen, capped at 30)"),
    ("person_height_median", 0.0, "median body height / frame height, frame mean (camera distance)"),
    ("person_height_max", 0.0, "max body height / frame height"),
    # distances (>= 2 persons)
    ("pair_present_frac", 0.0, "fraction of frames with >= 2 tracked persons"),
    ("min_pair_dist_mean", FAR, "per-frame min pairwise torso distance / mean body height of the pair, frame mean"),
    ("min_pair_dist_min", FAR, "min over the window of the per-frame min pairwise distance"),
    ("mean_pair_dist", FAR, "mean pairwise normalized distance, frame mean"),
    ("n_close_pairs", 0.0, "pairs closer than 1 h, frame mean"),
    ("n_contact_pairs", 0.0, "pairs closer than 0.5 h, frame mean"),
    ("max_box_iou", 0.0, "max pairwise box IoU, frame mean (grappling / occlusion)"),
    ("crowd_spread", 0.0, "mean distance to the group centroid / median h, frame mean"),
    # dynamics (camera-motion compensated with the tracker's affine warp)
    ("closest_pair_closing_vel", 0.0, "closing speed (h/s, > 0 = approaching) of the closest pair in the last "
                                      "frame of the window with >= 2 tracked persons, from the pair's earliest "
                                      "co-occurrence in the window (>= 0.35 s apart), same keypoints at both ends"),
    ("max_pair_closing_vel", 0.0, "max closing speed over all pairs in that frame (last frame with >= 2 persons)"),
    ("mean_pair_closing_vel", 0.0, "mean closing speed over all pairs in that frame (last frame with >= 2 "
                                   "persons)"),
    ("crowd_convergence", 0.0, "decrease per s of crowd_spread over persons present in the first and the last "
                               "frame with >= 2 persons (>= 0.35 s apart)"),
    ("n_converging_people", 0.0, "persons in the last frame moving (> 0.2 h/s) toward the centroid of the "
                                 "others (cos > 0.5), window-mean velocity"),
    ("frac_converging_people", 0.0, "n_converging_people / persons in the last frame"),
    ("n_moving_people", 0.0, "persons in the last frame with window-mean speed > 0.2 h/s"),
    ("body_speed_mean", 0.0, "body speed in h/s over all person steps (camera compensated; per step the same "
                             "keypoints in both frames: torso, else head, else box center)"),
    ("body_speed_max", 0.0, "max body speed in h/s"),
    # pose cues
    ("mutual_facing_closest", 0.0, "mutual facing of the closest pair in [-1, 1] (1 = heads turned toward each "
                                   "other; head yaw = sin(yaw) from nose vs ear / eye span), frame mean"),
    ("mutual_facing_mean", 0.0, "mutual facing averaged over pairs closer than 2 h, frame mean"),
    ("faceoff_frac", 0.0, "fraction of frames whose closest pair is < 1.5 h apart with mutual facing > 0.3"),
    ("profile_frac", 0.0, "fraction of persons seen in profile (shoulder width < 0.1 h)"),
    ("arm_raise_rate", 0.0, "fraction of person-frames with a wrist above its shoulder"),
    ("wrist_above_head_rate", 0.0, "fraction of person-frames with a wrist above the head"),
    ("arm_toward_other_mean", 0.0, "reach toward the nearest person: (dist(shoulder, other) - dist(wrist, other)) / h, "
                                   "max over arms, person-frame mean"),
    ("arm_toward_other_max", 0.0, "max of the reach above"),
    ("wrist_to_other_min", FAR, "min distance from any wrist to another person's torso center / their h"),
    ("wrist_speed_mean", 0.0, "wrist speed relative to the own body anchor (h/s, max over arms), person-step "
                              "mean"),
    ("wrist_speed_max", 0.0, "max of the wrist speed above"),
    ("torso_lean_mean", 0.0, "torso angle from vertical / 90 deg in [0, 2] (0 upright, 1 horizontal, > 1 "
                             "shoulders below hips), person-frame mean"),
    # camera (from the tracker's sparse-optical-flow affine warp between sampled frames)
    ("cam_translation", 0.0, "speed of the frame center under the camera warp (A c + t - c), frame widths "
                             "per s, step mean"),
    ("cam_zoom", 0.0, "abs(log scale change) per s, step mean"),
    ("cam_shake", 0.0, "std over steps of the frame-center velocity (frame widths per s)"),
]
SLOPE_KEYS = ["n_persons_mean", "min_pair_dist_mean", "mean_pair_dist", "mutual_facing_closest",
              "arm_raise_rate", "wrist_speed_mean", "body_speed_mean"]


def columns(slope_horizons=DEFAULTS["slope_horizons"]):
    """[(name, fill, doc)] for every pose_interact column, in file order."""
    cols = list(WINDOW_COLS)
    for key in SLOPE_KEYS:
        for h in slope_horizons:
            cols.append((f"{key}_slope{h}s", 0.0, f"causal OLS slope per second of {key} over ticks [t-{h}, t] "
                                                  f"(ticks where it is defined; 0 if < 3)"))
    return cols


COLUMNS = [c for c, _, _ in columns()]

_MODELS = {}
_LOG = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------------------------
# Model, tracker and the detection + tracking pass
# ----------------------------------------------------------------------------------------------------------------

def resolve_weights(weights):
    """Local path for ``weights`` (path, file in WEIGHTS_DIR, HF mirror download, or the repo fallback)."""
    if os.path.isfile(weights):
        return weights
    local = os.path.join(WEIGHTS_DIR, os.path.basename(weights))
    if os.path.isfile(local):
        return local
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(HF_REPO, os.path.basename(weights), local_dir=WEIGHTS_DIR)
    except Exception as e:  # offline or name not on the mirror
        _LOG.warning("could not fetch %s (%s); falling back to %s", weights, e, FALLBACK_WEIGHTS)
        return FALLBACK_WEIGHTS


def _sha256(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()[:n]


class Detector:
    """YOLO pose model on one device with GPU letterboxing and CUDA NMS (no per-image CPU preprocessing).

    The letterbox reproduces ``ultralytics.data.augment.LetterBox`` (minimum rectangle, stride padding, value 114)
    so ``ops.scale_boxes`` / ``ops.scale_coords`` map results back exactly as ``YOLO.predict`` would.
    """

    def __init__(self, weights, device, half=True):
        import torch
        import torchvision  # noqa: F401  (ultralytics uses torchvision's CUDA NMS once it is imported)
        from ultralytics import YOLO
        from ultralytics.nn.autobackend import AutoBackend
        from ultralytics.utils import LOGGER

        LOGGER.setLevel(logging.ERROR)  # silence per-frame GMC "not enough matching points" warnings
        self.path = resolve_weights(weights)
        self.sha = _sha256(self.path)
        yolo = YOLO(self.path)
        self.kpt_shape = tuple(yolo.model.kpt_shape)
        self.nc = len(yolo.names)
        self.device = torch.device(device)
        self.half = bool(half)
        self.net = AutoBackend(yolo.model, device=self.device, fp16=self.half, fuse=True, verbose=False).eval()
        self.stride = int(max(self.net.stride)) if hasattr(self.net.stride, "__len__") else int(self.net.stride)

    def __call__(self, frames, imgsz=640, conf=0.1, iou=0.7, max_det=100):
        """``frames`` uint8 RGB [B, H, W, 3]. Returns per frame a float32 array [n, 6 + 51]
        (x1, y1, x2, y2, conf, cls, 17 x (x, y, kconf)) in ``frames`` pixel coordinates."""
        import torch
        import torch.nn.functional as Fnn
        from ultralytics.utils import ops
        from ultralytics.utils.nms import non_max_suppression

        B, H, W = frames.shape[:3]
        r = min(imgsz / H, imgsz / W)
        nw, nh = round(W * r), round(H * r)
        dw, dh = ((imgsz - nw) % self.stride) / 2, ((imgsz - nh) % self.stride) / 2
        top, bottom, left, right = round(dh - 0.1), round(dh + 0.1), round(dw - 0.1), round(dw + 0.1)
        with torch.inference_mode():
            x = torch.from_numpy(frames).to(self.device, non_blocking=True).permute(0, 3, 1, 2)
            x = x.half() if self.half else x.float()
            if (nh, nw) != (H, W):
                x = Fnn.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
            x = Fnn.pad(x, (left, right, top, bottom), value=114.0) / 255.0
            preds = self.net(x)
            dets = non_max_suppression(preds, conf, iou, classes=None, agnostic=False, max_det=max_det, nc=self.nc)
            out = []
            nk = self.kpt_shape[0] * self.kpt_shape[1]
            for d in dets:
                d = d.float()
                if not len(d):
                    out.append(np.zeros((0, 6 + nk), dtype=np.float32))
                    continue
                d[:, :4] = ops.scale_boxes(x.shape[2:], d[:, :4], (H, W))
                k = d[:, 6:].reshape(len(d), *self.kpt_shape).clone()
                d[:, 6:] = ops.scale_coords(x.shape[2:], k, (H, W)).reshape(len(d), nk)
                out.append(d.cpu().numpy())
        return out


def load_model(weights, device, half=True):
    """Cached ``Detector`` per (weights, device, half)."""
    key = (resolve_weights(weights), str(device), bool(half))
    if key not in _MODELS:
        _MODELS[key] = Detector(weights, device, half)
    return _MODELS[key]


def make_tracker(cfg):
    """Fresh BoT-SORT / ByteTrack instance with track ids starting at 1. Records each frame's GMC warp.

    After every ``tracker.update``, ``tracker.last_warp`` is the warp the tracker actually applied and
    ``tracker.last_gmc_ok`` is False when GMC raised. Ultralytics catches GMC exceptions inside ``update`` and
    uses the identity, so the warp is reset to identity BEFORE calling GMC (version 1 kept the previous frame's
    warp, a phantom camera motion). GMC also keeps its previous-frame state when it raises, so the warp of the
    following frame spans two steps; ``interaction_features`` skips both steps.
    """
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    tcfg = IterableSimpleNamespace(**YAML.load(check_yaml(f"{cfg['tracker']}.yaml")))
    tcfg.track_buffer = max(1, int(round(cfg["track_buffer_sec"] * cfg["sample_fps"])))
    tracker = (BOTSORT if cfg["tracker"] == "botsort" else BYTETracker)(args=tcfg, frame_rate=30)
    tracker.reset()  # also resets the global track-id counter
    tracker.last_warp = np.eye(2, 3)
    tracker.last_gmc_ok = True
    if hasattr(tracker, "gmc"):
        tracker.gmc.downscale = max(1, int(cfg["gmc_downscale"]))
        if hasattr(tracker.gmc, "feature_params"):
            tracker.gmc.feature_params["maxCorners"] = int(cfg["gmc_max_corners"])
        apply = tracker.gmc.apply

        def recording_apply(img, dets=None):
            tracker.last_warp = np.eye(2, 3)  # what the tracker uses if apply raises (it catches the exception)
            tracker.last_gmc_ok = False
            warp = apply(img, dets)
            tracker.last_warp = np.asarray(warp, dtype=np.float64) if warp is not None else np.eye(2, 3)
            tracker.last_gmc_ok = warp is not None
            return warp

        tracker.gmc.apply = recording_apply
    return tracker, vars(tcfg)


def _prefetch_detections(video, frame_t, detector, cfg, depth=2):
    """Background thread: decode a chunk and run the detector on it while the caller tracks the previous chunk.

    Yields ``(start, frames or None, dets or None, ok [n] bool)``. A chunk that raises is re-decoded frame by
    frame (``motion.decode_frames``); ``ok`` is False for frames that still fail (their detector slot gets a
    copy of a decoded frame and the result is never used), and ``frames`` is None when no frame decodes.
    Per-sample detector outputs do not depend on the other frames of a fixed-size batch, so a failure at t_bad
    leaves every earlier frame bit-identical.
    """
    import queue
    import threading

    B = int(cfg["batch"])
    q = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def work():
        try:
            for s in range(0, len(frame_t), B):
                if stop.is_set():
                    return
                frames, ok = decode_frames(video, frame_t[s:s + B])
                if frames is None:  # nothing in this chunk decoded: all its frames stay missing
                    q.put((s, None, None, ok))
                    continue
                n = len(frames)
                frames_in = frames
                if not ok.all():  # fill failed slots with the nearest earlier decoded frame (else the first one)
                    good = np.flatnonzero(ok)
                    src = good[np.clip(np.searchsorted(good, np.arange(n), side="right") - 1, 0, None)]
                    frames_in = frames[src]
                if n < B:  # pad to a fixed batch shape -> identical numerics for any video length
                    frames_in = np.concatenate([frames_in, np.repeat(frames_in[-1:], B - n, axis=0)])
                dets = detector(frames_in, imgsz=cfg["imgsz"], conf=cfg["det_conf"], iou=cfg["nms_iou"])[:n]
                q.put((s, frames, dets, ok))
        except BaseException as e:  # surface model errors in the caller
            q.put(e)
        finally:
            q.put(None)

    th = threading.Thread(target=work, daemon=True)
    th.start()
    try:
        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        while th.is_alive():  # drain so the worker can exit
            try:
                q.get(timeout=0.1)
            except queue.Empty:
                pass


def _map_tracks_to_dets(tracks, conf, cfg_t):
    """Map tracker outputs to detection rows.

    Ultralytics' ``idx`` column indexes the high-score subset (conf >= track_high_thresh) or the low-score
    subset (track_low_thresh < conf < track_high_thresh) the track was matched in, not the full detection list.
    """
    out = {}
    if len(tracks) == 0:
        return out
    hi = np.flatnonzero(conf >= cfg_t["track_high_thresh"])
    lo = np.flatnonzero((conf > cfg_t["track_low_thresh"]) & (conf < cfg_t["track_high_thresh"]))
    for row in tracks:
        tid, score, idx = int(row[4]), float(row[5]), int(row[7])
        pool = hi if score >= cfg_t["track_high_thresh"] else lo
        if 0 <= idx < len(pool) and abs(conf[pool[idx]] - score) < 1e-4:
            out[int(pool[idx])] = tid
        else:  # should not happen; fall back to the detection with the closest score
            j = int(np.argmin(np.abs(conf - score)))
            out.setdefault(j, tid)
    return out


def track_video(video_path, device="cuda:0", **cfg):
    """Detection + tracking pass. Returns the ``pose_raw`` dict (arrays + ``meta``)."""
    from ultralytics.engine.results import Boxes

    cfg = {**DEFAULTS, **cfg}
    detector = load_model(cfg["weights"], device, cfg["half"])
    video, native_wh = open_scaled(video_path, max_side=cfg["max_side"], num_threads=cfg["num_threads"])
    info = video.info
    sx, sy = native_wh[0] / info.width, native_wh[1] / info.height  # decoded -> native pixels
    frame_t = video.sample_times(cfg["sample_fps"])
    F = len(frame_t)
    tracker, tcfg = make_tracker(cfg)
    scale4 = np.array([sx, sy, sx, sy], dtype=np.float32)

    frame_ok = np.zeros(F, dtype=bool)
    gmc_ok = np.ones(F, dtype=bool)
    warps = np.tile(np.eye(2, 3, dtype=np.float32), (F, 1, 1))
    n_hi = np.zeros(F, dtype=np.int16)
    counts = np.zeros(F, dtype=np.int64)
    ids, boxes, confs, kpts = [], [], [], []
    for s, frames, dets, fok in _prefetch_detections(video, frame_t, detector, cfg):
        if frames is None:  # decode error: these frames stay missing and the tracker does not see them
            continue
        for j, d in enumerate(dets):
            if not fok[j]:  # this frame failed to decode: missing, not shown to the tracker
                continue
            k = s + j
            det = Boxes(d[:, :6], (info.height, info.width))
            tracker.last_warp, tracker.last_gmc_ok = np.eye(2, 3), True  # identity unless GMC runs (ByteTrack)
            tracks = tracker.update(det, frames[j])  # GMC only needs a grey image; RGB order is irrelevant
            frame_ok[k] = True
            gmc_ok[k] = tracker.last_gmc_ok
            A = tracker.last_warp
            warps[k] = [[A[0, 0], A[0, 1], A[0, 2] * sx], [A[1, 0], A[1, 1], A[1, 2] * sy]]
            conf = d[:, 4].astype(np.float32)
            n_hi[k] = int((conf >= 0.5).sum())
            tmap = _map_tracks_to_dets(tracks, conf, tcfg)
            keep = np.array([i for i in range(len(conf)) if i in tmap or conf[i] >= cfg["save_conf"]], dtype=int)
            if not len(keep):
                continue
            kp = d[keep, 6:].reshape(len(keep), 17, 3).astype(np.float32)  # decoded pixels + keypoint conf
            kp[..., 0] *= sx
            kp[..., 1] *= sy
            ids.append(np.array([tmap.get(int(i), -1) for i in keep], dtype=np.int32))
            boxes.append(d[keep, :4].astype(np.float32) * scale4)
            confs.append(conf[keep])
            kpts.append(kp)
            counts[k] = len(keep)
    cat = (lambda xs, shape, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(shape, dtype=dt))
    raw = dict(
        frame_t=frame_t.astype(np.float32),
        frame_idx=video.frame_index(frame_t).astype(np.int32),
        frame_ok=frame_ok,
        frame_ptr=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        frame_warp=warps,
        frame_gmc_ok=gmc_ok,
        frame_n_hiconf=n_hi,
        det_track_id=cat(ids, (0,), np.int32),
        det_box=cat(boxes, (0, 4), np.float32),
        det_conf=cat(confs, (0,), np.float32),
        det_kpt=cat(kpts, (0, 17, 3), np.float32),
    )
    raw["meta"] = dict(
        extractor=EXTRACTOR_RAW, version=RAW_VERSION, model_id=os.path.basename(detector.path),
        weights_path=detector.path, weights_sha256_16=detector.sha, config={k: cfg[k] for k in RAW_KEYS},
        tracker_config=tcfg,
        video=dict(fps=info.fps, num_frames=info.num_frames, duration_sec=info.duration_sec,
                   native_wh=list(native_wh), decoded_wh=[info.width, info.height]),
        keypoint_names=KPT_NAMES, coords="pixels of the native video resolution",
    )
    return raw


# ----------------------------------------------------------------------------------------------------------------
# pose_raw file format
# ----------------------------------------------------------------------------------------------------------------

RAW_FEAT_COLS = ["n_tracked_last", "n_tracked_mean_1s", "det_conf_max_1s"]


def raw_tick_summary(raw):
    """Contract keys for pose_raw: per tick [n tracked in last frame, mean n tracked in [t-1, t], max conf]."""
    dur = raw["meta"]["video"]["duration_sec"]
    ticks = io.tick_times(dur)
    ptr, tid, conf = raw["frame_ptr"], raw["det_track_id"], raw["det_conf"]
    F = len(raw["frame_t"])
    n_trk = np.array([(tid[ptr[k]:ptr[k + 1]] >= 0).sum() for k in range(F)], dtype=np.float32)
    cmax = np.array([conf[ptr[k]:ptr[k + 1]].max() if ptr[k + 1] > ptr[k] else 0.0 for k in range(F)])
    k0, k1 = window_bounds(raw["frame_t"].astype(np.float64), ticks, 1.0)
    feat = np.zeros((len(ticks), 3), dtype=np.float32)
    valid = np.zeros(len(ticks), dtype=bool)
    for i in range(len(ticks)):
        ks = np.arange(k0[i], k1[i])
        ks = ks[raw["frame_ok"][ks]]
        if len(ks):
            valid[i] = True
            feat[i] = [n_trk[ks[-1]], n_trk[ks].mean(), cmax[ks].max()]
    return ticks, feat, valid


def save_raw(video_id, raw):
    """Write pose_raw with the contract keys plus the ragged arrays (atomic, like ``io.save_features``).

    Keys: t_sec [T], feat [T, 3] f16 (``RAW_FEAT_COLS``), valid [T], meta (JSON), and
      frame_t [F] f32 sample times, frame_idx [F] i32 decoded frame index (use it for real time:
      frame_idx / meta["video"]["fps"]; recomputing it from the float32 frame_t can be off by one), frame_ok [F] bool,
      frame_ptr [F+1] i64 (detections of frame k are rows frame_ptr[k]:frame_ptr[k+1]),
      frame_warp [F, 2, 3] f32 affine camera warp the tracker applied from the previous frame it saw to frame k
      (native pixels; identity for ByteTrack and when GMC failed),
      frame_gmc_ok [F] bool False when GMC raised on frame k (version 2; then the warp of frame k is identity and
      that of the next frame spans two steps),
      frame_n_hiconf [F] i16 detections with conf >= 0.5,
      det_track_id [N] i32 (-1 = not tracked), det_box [N, 4] f32 xyxy, det_conf [N] f32,
      det_kpt [N, 17, 3] f32 (x, y native pixels, keypoint confidence).
    """
    t_sec, feat, valid = raw_tick_summary(raw)
    meta = dict(raw["meta"], D=len(RAW_FEAT_COLS), columns=RAW_FEAT_COLS, window_sec=1.0,
                n_frames=int(len(raw["frame_t"])), n_dets=int(len(raw["det_conf"])))
    path = io.feature_path(EXTRACTOR_RAW, video_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    arrays = {k: v for k, v in raw.items() if k != "meta"}
    np.savez_compressed(tmp, t_sec=t_sec.astype(np.float32), feat=feat.astype(np.float16), valid=valid,
                        meta=json.dumps(meta), **arrays)
    os.replace(tmp, path)
    return path


def load_raw(video_id=None, path=None):
    """Load a pose_raw file into the dict layout returned by ``track_video``."""
    path = path or io.feature_path(EXTRACTOR_RAW, video_id)
    with np.load(path, allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files if k not in ("t_sec", "feat", "valid", "meta")}
        raw["meta"] = json.loads(str(z["meta"]))
    return raw


# ----------------------------------------------------------------------------------------------------------------
# Interaction features (pure numpy)
# ----------------------------------------------------------------------------------------------------------------

def _mid(xy, vis, a, b):
    va, vb = vis[:, a], vis[:, b]
    cnt = va.astype(np.float32) + vb
    m = (xy[:, a] * va[:, None] + xy[:, b] * vb[:, None]) / np.maximum(cnt, 1)[:, None]
    return m, cnt > 0


def body_height(box, xy, vis):
    """Per-frame body height estimate [n] in pixels: ``max(box height, keypoint estimate)``.

    The keypoint estimate uses the segment cues ``H_TORSO`` / ``H_CUES`` (ratio x length, for segments whose two
    keypoints are visible): the largest cue, or the second-largest when >= 3 cues exist. Unlike the box height
    alone, it keeps the scale when the hips or legs are hidden or cut off by the frame edge (version 1 then fell
    back to the upper-body box, about 3x too small, and jumped when torso visibility flickered).
    """
    n = len(box)
    box_h = np.maximum(box[:, 3] - box[:, 1], 1.0).astype(np.float64)
    sh, sh_ok = _mid(xy, vis, LSH, RSH)
    hip, hip_ok = _mid(xy, vis, LHIP, RHIP)
    cues = np.full((n, 1 + len(H_CUES)), -np.inf)
    t = sh_ok & hip_ok
    cues[t, 0] = H_TORSO * np.linalg.norm(sh - hip, axis=1)[t]
    for c, ((a, b), ratio) in enumerate(H_CUES, 1):
        ok = vis[:, a] & vis[:, b]
        cues[ok, c] = ratio * np.linalg.norm(xy[ok, a] - xy[ok, b], axis=1)
    n_cue = np.isfinite(cues).sum(1)
    desc = -np.sort(-cues, axis=1)  # descending; -inf (missing cues) last
    kp_h = np.where(n_cue >= 3, desc[:, 1], desc[:, 0])
    return np.where(n_cue > 0, np.maximum(box_h, kp_h), box_h)


class TrackHeight:
    """Causal per-track body height: median of the track's per-frame ``body_height`` over the last ``sec`` s.

    Call once per frame, in time order, with the tracked ids. Uses only frames up to the current one, so it is
    causal. A brief occlusion of the hips or a raised arm (which stretches the box) does not move h.
    """

    def __init__(self, sec):
        from collections import deque

        self.sec = float(sec)
        self.hist = {}
        self._deque = deque

    def __call__(self, t, ids, h_frame):
        out = np.empty(len(ids), dtype=np.float64)
        for j, tid in enumerate(ids):
            q = self.hist.setdefault(int(tid), self._deque())
            q.append((float(t), float(h_frame[j])))
            while t - q[0][0] > self.sec + 1e-6:
                q.popleft()
            out[j] = np.median([x for _, x in q])
        return out


def head_yaw(xy, vis, h):
    """Head yaw proxy f = sin(yaw) in [-1, 1], > 0 = facing image right, NaN when unknown (no nose).

    - nose + both ears: tan(yaw) = YAW_EAR x (nose_x - ear_mid_x) / half ear span (near-frontal heads);
    - else nose + both eyes: the same with the eyes and YAW_EYE (3/4 views, one ear hidden);
    - else nose + one ear (else one eye): profile, f = +/-YAW_PROFILE with the sign of nose_x - that keypoint
      (unknown when they are within 0.01 h).
    Version 1 divided the nose-to-ear offset by 0.06 h and clipped; with one ear hidden it saturated at +/-1
    (84 % of such heads), so the value was nearly binary.
    """
    f = np.full(len(xy), np.nan)
    nose, nx = vis[:, NOSE], xy[:, NOSE, 0].astype(np.float64)
    profile = nose.copy()
    for a, b, k in ((LEYE, REYE, YAW_EYE), (LEAR, REAR, YAW_EAR)):  # ears last: they win when both are present
        ok = nose & vis[:, a] & vis[:, b]
        off = nx - 0.5 * (xy[:, a, 0] + xy[:, b, 0])
        half = np.maximum(0.5 * np.abs(xy[:, a, 0] - xy[:, b, 0]), 1e-3)
        tan = k * off / half
        f = np.where(ok, tan / np.sqrt(1.0 + tan * tan), f)
        profile &= ~ok
    one_ear = vis[:, LEAR] | vis[:, REAR]
    one_eye = vis[:, LEYE] | vis[:, REYE]
    ref = np.where(one_ear, np.where(vis[:, LEAR], xy[:, LEAR, 0], xy[:, REAR, 0]),
                   np.where(vis[:, LEYE], xy[:, LEYE, 0], xy[:, REYE, 0]))
    off = nx - ref
    profile &= (one_ear | one_eye) & (np.abs(off) > 0.01 * h)
    return np.where(profile, YAW_PROFILE * np.sign(off), f)


def person_attributes(box, kpt, kpt_conf, h=None):
    """Per-person geometry for one frame. ``box`` [n, 4] xyxy, ``kpt`` [n, 17, 3]. Returns a dict of arrays.

    ``h``: body heights [n] (``TrackHeight`` output); default: the per-frame ``body_height``.
    ``center`` (used for distances) is the torso center, else estimated from the shoulders / hips / head with
    fixed offsets in h (``CENTER_OFF_*``), else the box center.
    """
    xy, vis = kpt[..., :2].astype(np.float64), kpt[..., 2] >= kpt_conf
    if h is None:
        h = body_height(box, xy, vis)
    h = np.asarray(h, dtype=np.float64)
    sh, sh_ok = _mid(xy, vis, LSH, RSH)
    hip, hip_ok = _mid(xy, vis, LHIP, RHIP)
    torso_ok = sh_ok & hip_ok
    down = np.stack([np.zeros_like(h), h], 1)  # +y is down in the image
    head_vis = vis[:, HEAD]
    head = (xy[:, HEAD] * head_vis[..., None]).sum(1) / np.maximum(head_vis.sum(1), 1)[:, None]
    box_c = np.stack([(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2], 1).astype(np.float64)
    center = np.where(head_vis.any(1)[:, None], head + CENTER_OFF_HEAD * down, box_c)
    center = np.where(hip_ok[:, None], hip - CENTER_OFF_SH * down, center)
    center = np.where(sh_ok[:, None], sh + CENTER_OFF_SH * down, center)
    center = np.where(torso_ok[:, None], 0.5 * (sh + hip), center)
    facing = head_yaw(xy, vis, h)
    both_sh = vis[:, LSH] & vis[:, RSH]
    profile = both_sh & (np.abs(xy[:, LSH, 0] - xy[:, RSH, 0]) < 0.1 * h)
    # arms: per side (left, right) wrist above shoulder / above head
    head_vis = vis[:, HEAD]
    head_y = np.where(head_vis.any(1), np.where(head_vis, xy[:, HEAD, 1], np.inf).min(1), np.nan)
    raise_any = np.zeros(len(box), dtype=bool)
    above_any = np.zeros(len(box), dtype=bool)
    arm_seen = np.zeros(len(box), dtype=bool)
    for w, s in ((LWR, LSH), (RWR, RSH)):
        ok = vis[:, w] & vis[:, s]
        arm_seen |= ok
        raise_any |= ok & (xy[:, s, 1] - xy[:, w, 1] > 0.05 * h)
        above_any |= ok & np.isfinite(head_y) & (xy[:, w, 1] < head_y)
    v = sh - hip
    lean = np.where(torso_ok, np.arctan2(np.abs(v[:, 0]), -v[:, 1]) / (np.pi / 2), np.nan)  # [0, 2]
    return dict(center=center, box_c=box_c, h=h, facing=facing, profile=profile, both_sh=both_sh,
                raise_any=raise_any, above_any=above_any, arm_seen=arm_seen, lean=lean, xy=xy, vis=vis,
                kconf=kpt[..., 2].mean(1), kvis=vis.mean(1))


def _box_iou(b):
    x1 = np.maximum(b[:, None, 0], b[None, :, 0])
    y1 = np.maximum(b[:, None, 1], b[None, :, 1])
    x2 = np.minimum(b[:, None, 2], b[None, :, 2])
    y2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area[:, None] + area[None, :] - inter, 1e-6)


def _frame_stats(P, ids, box, conf, frame_h):
    """Scalar statistics of one frame from ``person_attributes`` output (tracked persons only)."""
    n = len(ids)
    st = dict(n=n, conf_mean=np.nan, conf_max=np.nan, kconf=np.nan, kvis=np.nan, h_med=np.nan, h_max=np.nan,
              min_d=np.nan, mean_d=np.nan, n_close=np.nan, n_contact=np.nan, max_iou=np.nan, spread=np.nan,
              mf_closest=np.nan, mf_mean=np.nan, faceoff=np.nan, wrist_other=np.nan,
              n_raise=0, n_above=0, n_arm=0, n_profile=0, n_sh=0, reach=[], lean=[], pair_d={}, closest=None)
    if n == 0:
        return st
    c, h = P["center"], P["h"]
    st.update(conf_mean=float(conf.mean()), conf_max=float(conf.max()), kconf=float(P["kconf"].mean()),
              kvis=float(P["kvis"].mean()), h_med=float(np.median(h)) / frame_h, h_max=float(h.max()) / frame_h,
              n_raise=int(P["raise_any"].sum()), n_above=int(P["above_any"].sum()), n_arm=int(P["arm_seen"].sum()),
              n_profile=int(P["profile"].sum()), n_sh=int(P["both_sh"].sum()),
              lean=[x for x in P["lean"] if np.isfinite(x)])
    if n < 2:
        return st
    dist = np.linalg.norm(c[:, None] - c[None], axis=2)
    D = dist / (0.5 * (h[:, None] + h[None]))
    iu = np.triu_indices(n, 1)
    d = D[iu]
    # mutual facing: each person's head yaw projected on the direction to the other
    sgn = np.sign(c[None, :, 0] - c[:, None, 0])  # [i, j] = +1 if j is right of i
    fi = P["facing"][:, None] * sgn               # i toward j
    pair_terms = np.stack([fi[iu], fi.T[iu]], 1)  # (i toward j, j toward i); NaN where a head yaw is unknown
    known = np.isfinite(pair_terms)
    n_known = known.sum(1)
    mf = np.where(known, pair_terms, 0.0).sum(1) / np.maximum(n_known, 1)
    mf[n_known == 0] = np.nan  # mean of the known terms
    a = int(np.argmin(d))
    ci, cj = iu[0][a], iu[1][a]
    near = d < 2.0
    st.update(min_d=float(d.min()), mean_d=float(d.mean()), n_close=float((d < 1.0).sum()),
              n_contact=float((d < 0.5).sum()), max_iou=float(_box_iou(box)[iu].max()),
              spread=float(np.linalg.norm(c - c.mean(0), axis=1).mean() / np.median(h)),
              mf_closest=float(mf[a]) if np.isfinite(mf[a]) else np.nan,
              mf_mean=float(np.nanmean(mf[near])) if np.isfinite(mf[near]).any() else np.nan,
              closest=(int(min(ids[ci], ids[cj])), int(max(ids[ci], ids[cj]))))
    st["faceoff"] = float(d[a] < 1.5 and np.isfinite(mf[a]) and mf[a] > 0.3)
    st["pair_d"] = {(int(min(ids[i], ids[j])), int(max(ids[i], ids[j]))): float(D[i, j]) for i, j in zip(*iu)}
    # arms relative to the nearest other person
    Dn = D + np.diag(np.full(n, np.inf))
    nn = Dn.argmin(1)
    xy, vis = P["xy"], P["vis"]
    reach, wmin = [], np.inf
    for i in range(n):
        best = -np.inf
        for w, s in ((LWR, LSH), (RWR, RSH)):
            if not vis[i, w]:
                continue
            others = np.arange(n) != i
            wd = np.linalg.norm(c[others] - xy[i, w], axis=1) / h[others]
            wmin = min(wmin, float(wd.min()))
            if vis[i, s]:
                cj_ = c[nn[i]]
                best = max(best, (np.linalg.norm(xy[i, s] - cj_) - np.linalg.norm(xy[i, w] - cj_)) / h[i])
        if np.isfinite(best):
            reach.append(float(best))
    st["reach"] = reach
    st["wrist_other"] = wmin if np.isfinite(wmin) else np.nan
    return st


def _anchor_pair(Pa, i, Pb, j):
    """Anchor of one person in frames a and b built from the SAME keypoints in both frames.

    Mean of the torso keypoints visible in both frames, else of the head keypoints visible in both, else the
    two box centers. Motion of the anchor is real motion; switching between anchor types from one frame to the
    next (version 1: torso center vs box center) turned keypoint flicker into speeds of about 1 h/s.
    """
    for grp in (TORSO_KPTS, HEAD):
        common = Pa["vis"][i, grp] & Pb["vis"][j, grp]
        if common.any():
            g = np.asarray(grp)[common]
            return Pa["xy"][i, g].mean(0), Pb["xy"][j, g].mean(0)
    return Pa["box_c"][i], Pb["box_c"][j]


def _step_stats(Pa, ida, Pb, idb, warp, dt):
    """Per-person velocities between consecutive sampled frames a -> b (camera compensated with ``warp``).

    ``dt`` is the real time between the two decoded frames. Body velocity = displacement of the person's
    ``_anchor_pair`` anchor; wrist speed = wrist motion relative to that anchor (gesturing, not walking).
    """
    A, t = warp[:, :2].astype(np.float64), warp[:, 2].astype(np.float64)
    ia = {int(x): i for i, x in enumerate(ida)}
    body, wrist, vel = [], [], {}
    for j, tid in enumerate(idb):
        i = ia.get(int(tid))
        if i is None:
            continue
        hm = 0.5 * (Pa["h"][i] + Pb["h"][j])
        anc_a, anc_b = _anchor_pair(Pa, i, Pb, j)
        v = (anc_b - (A @ anc_a + t)) / (hm * dt)
        body.append(float(np.linalg.norm(v)))
        vel[int(tid)] = v
        best = np.nan
        for w in (LWR, RWR):
            if Pa["vis"][i, w] and Pb["vis"][j, w]:
                rel_a = Pa["xy"][i, w] - anc_a
                rel_b = Pb["xy"][j, w] - anc_b
                sp = float(np.linalg.norm(rel_b - A @ rel_a) / (hm * dt))
                best = sp if not np.isfinite(best) else max(best, sp)
        if np.isfinite(best):
            wrist.append(best)
    return body, wrist, vel


def _pair_dist_change(persons, k0, k1, p, q):
    """Normalized distance of tracks p and q at frames k0 and k1, each person anchored on the same keypoints in
    both frames (``_anchor_pair``). Returns ``(d0, d1)`` in body heights."""
    (_, P0, m0), (_, P1, m1) = persons[k0], persons[k1]
    i0, j0, i1, j1 = m0[p], m0[q], m1[p], m1[q]
    ap0, ap1 = _anchor_pair(P0, i0, P1, i1)
    aq0, aq1 = _anchor_pair(P0, j0, P1, j1)
    d0 = np.linalg.norm(ap0 - aq0) / (0.5 * (P0["h"][i0] + P0["h"][j0]))
    d1 = np.linalg.norm(ap1 - aq1) / (0.5 * (P1["h"][i1] + P1["h"][j1]))
    return float(d0), float(d1)


def interaction_features(raw, **cfg):
    """Per-tick ``pose_interact`` features from a ``pose_raw`` dict. Returns ``(t_sec, feat, valid, meta)``."""
    cfg = {**DEFAULTS, **cfg}
    horizons = tuple(int(h) for h in cfg["slope_horizons"])
    W = float(cfg["window_sec"])
    vmeta = raw["meta"]["video"]
    frame_w, frame_h = float(vmeta["native_wh"][0]), float(vmeta["native_wh"][1])
    ticks = io.tick_times(vmeta["duration_sec"])
    cols = columns(horizons)
    names = [c for c, _, _ in cols]
    col = {c: i for i, c in enumerate(names)}
    T = len(ticks)
    feat = np.full((T, len(cols)), np.nan, dtype=np.float64)
    valid = np.zeros(T, dtype=bool)

    frame_t = raw["frame_t"].astype(np.float64)
    ptr, F = raw["frame_ptr"], len(raw["frame_t"])
    ok = raw["frame_ok"]
    # real time of each decoded frame: rates divide by real gaps (24 fps -> frame steps 4,5,5,5,5, not 0.2 s)
    ftime = raw["frame_idx"].astype(np.float64) / float(vmeta["fps"])
    gmc_ok = raw.get("frame_gmc_ok", np.ones(F, dtype=bool))  # pose_raw version 1 has no flag
    center_px = np.array([frame_w / 2.0, frame_h / 2.0])
    track_h = TrackHeight(cfg["h_smooth_sec"])
    # ---- per frame
    persons, fstats, first_seen = [], [], {}
    for k in range(F):
        sl = slice(ptr[k], ptr[k + 1])
        tid = raw["det_track_id"][sl]
        m = tid >= 0
        ids = tid[m]
        box, conf, kp = raw["det_box"][sl][m], raw["det_conf"][sl][m], raw["det_kpt"][sl][m]
        P = None
        if len(ids):
            h_frame = body_height(box, kp[..., :2].astype(np.float64), kp[..., 2] >= cfg["kpt_conf"])
            P = person_attributes(box, kp, cfg["kpt_conf"], h=track_h(ftime[k], ids, h_frame))
        persons.append((ids, P, {int(x): i for i, x in enumerate(ids)}))
        fstats.append(_frame_stats(P, ids, box, conf, frame_h) if P is not None else _frame_stats(None, [], None,
                                                                                                  None, frame_h))
        for x in ids:
            first_seen.setdefault(int(x), frame_t[k])
    # ---- per step (frame k-1 -> k)
    steps = [None] * F
    for k in range(1, F):
        if not (ok[k] and ok[k - 1]):
            continue
        dt = float(ftime[k] - ftime[k - 1])
        if dt <= 0 or not (gmc_ok[k] and gmc_ok[k - 1]):  # same decoded frame twice / camera warp unknown
            continue
        (ida, Pa, _), (idb, Pb, _) = persons[k - 1], persons[k]
        warp = raw["frame_warp"][k].astype(np.float64)
        cam_v = (warp[:, :2] @ center_px + warp[:, 2] - center_px) / frame_w / dt  # frame-center displacement
        sc = float(np.sqrt(abs(np.linalg.det(warp[:, :2]))))
        zoom = abs(np.log(sc)) / dt if sc > 0 else 0.0
        if Pa is not None and Pb is not None:
            body, wrist, vel = _step_stats(Pa, ida, Pb, idb, warp, dt)
        else:
            body, wrist, vel = [], [], {}
        steps[k] = dict(cam_v=cam_v, zoom=zoom, body=body, wrist=wrist, vel=vel)

    nanmean = (lambda a: float(np.mean(a)) if len(a) else np.nan)
    first_t = np.sort(np.array(list(first_seen.values()), dtype=np.float64))
    k0s, k1s = window_bounds(frame_t, ticks, W)
    for i, t in enumerate(ticks):
        ks = np.arange(k0s[i], k1s[i])
        ks = ks[ok[ks]]
        if not len(ks):
            continue
        S = [fstats[k] for k in ks]
        nper = np.array([s["n"] for s in S], dtype=np.float64)
        valid[i] = nper.max() > 0
        f = feat[i]
        f[col["n_persons_mean"]] = nper.mean()
        f[col["n_persons_max"]] = nper.max()
        f[col["n_persons_last"]] = nper[-1]
        f[col["n_det_hiconf_mean"]] = raw["frame_n_hiconf"][ks].mean()
        f[col["pair_present_frac"]] = float((nper >= 2).mean())
        cam = [steps[k] for k in ks[1:] if steps[k] is not None and k - 1 >= ks[0]]
        if cam:
            cv = np.stack([s["cam_v"] for s in cam])
            f[col["cam_translation"]] = float(np.linalg.norm(cv, axis=1).mean())
            f[col["cam_zoom"]] = float(np.mean([s["zoom"] for s in cam]))
            f[col["cam_shake"]] = float(np.sqrt(cv.var(0).sum())) if len(cam) >= 2 else 0.0
        if not valid[i]:
            continue

        def agg(key, fn=np.nanmean):
            a = np.array([s[key] for s in S], dtype=np.float64)
            return float(fn(a)) if np.isfinite(a).any() else np.nan

        f[col["det_conf_mean"]] = agg("conf_mean")
        f[col["det_conf_max"]] = agg("conf_max", np.nanmax)
        f[col["kpt_conf_mean"]] = agg("kconf")
        f[col["kpt_visible_frac"]] = agg("kvis")
        f[col["person_height_median"]] = agg("h_med")
        f[col["person_height_max"]] = agg("h_max", np.nanmax)
        f[col["min_pair_dist_mean"]] = agg("min_d")
        f[col["min_pair_dist_min"]] = agg("min_d", np.nanmin)
        f[col["mean_pair_dist"]] = agg("mean_d")
        f[col["n_close_pairs"]] = agg("n_close")
        f[col["n_contact_pairs"]] = agg("n_contact")
        f[col["max_box_iou"]] = agg("max_iou")
        f[col["crowd_spread"]] = agg("spread")
        f[col["mutual_facing_closest"]] = agg("mf_closest")
        f[col["mutual_facing_mean"]] = agg("mf_mean")
        f[col["faceoff_frac"]] = agg("faceoff")
        f[col["wrist_to_other_min"]] = agg("wrist_other", np.nanmin)
        n_arm = sum(s["n_arm"] for s in S)
        if n_arm:
            f[col["arm_raise_rate"]] = sum(s["n_raise"] for s in S) / n_arm
            f[col["wrist_above_head_rate"]] = sum(s["n_above"] for s in S) / n_arm
        n_sh = sum(s["n_sh"] for s in S)
        if n_sh:
            f[col["profile_frac"]] = sum(s["n_profile"] for s in S) / n_sh
        reach = [r for s in S for r in s["reach"]]
        if reach:
            f[col["arm_toward_other_mean"]] = float(np.mean(reach))
            f[col["arm_toward_other_max"]] = float(np.max(reach))
        lean = [x for s in S for x in s["lean"]]
        f[col["torso_lean_mean"]] = nanmean(lean)
        # tracks: churn and age
        kL = ks[-1]
        idsL, PL, _ = persons[kL]
        f[col["new_tracks"]] = float(np.searchsorted(first_t, frame_t[kL] + 1e-6, side="right")
                                     - np.searchsorted(first_t, frame_t[ks[0]] - 1e-6, side="left"))
        if len(idsL):
            f[col["track_age_mean"]] = float(np.mean([min(frame_t[kL] - first_seen[int(x)], 30.0) for x in idsL]))
        # steps inside the window: speeds, per-track mean velocity
        st = [steps[k] for k in ks[1:] if steps[k] is not None and k - 1 >= ks[0]]
        body = [b for s in st for b in s["body"]]
        wrist = [w for s in st for w in s["wrist"]]
        if body:
            f[col["body_speed_mean"]], f[col["body_speed_max"]] = float(np.mean(body)), float(np.max(body))
        if wrist:
            f[col["wrist_speed_mean"]], f[col["wrist_speed_max"]] = float(np.mean(wrist)), float(np.max(wrist))
        vsum = {}
        for s in st:
            for tid, v in s["vel"].items():
                acc = vsum.setdefault(tid, [np.zeros(2), 0])
                acc[0] += v
                acc[1] += 1
        if len(idsL):
            moving = conv = 0
            cL = PL["center"]
            for j, tid in enumerate(idsL):
                if int(tid) not in vsum:
                    continue
                v = vsum[int(tid)][0] / vsum[int(tid)][1]
                sp = float(np.linalg.norm(v))
                if sp <= 0.2:
                    continue
                moving += 1
                if len(idsL) >= 2:
                    d = np.delete(cL, j, axis=0).mean(0) - cL[j]
                    if np.dot(v, d) > 0.5 * sp * np.linalg.norm(d):
                        conv += 1
            f[col["n_moving_people"]] = moving
            if len(idsL) >= 2:
                f[col["n_converging_people"]] = conv
                f[col["frac_converging_people"]] = conv / len(idsL)
        # closing velocities: pairs in the last frame with >= 2 persons, versus their earliest co-occurrence at
        # least MIN_PAIR_GAP_SEC earlier; each person is anchored on the same keypoints at both ends
        kp_ = [k for k in ks if fstats[k]["n"] >= 2]
        if kp_:
            kL2 = kp_[-1]
            closing = {}
            for pair in fstats[kL2]["pair_d"]:
                for k in kp_:
                    gap = ftime[kL2] - ftime[k]
                    if gap < MIN_PAIR_GAP_SEC - 1e-6:
                        break
                    if pair in fstats[k]["pair_d"]:
                        d0, d1 = _pair_dist_change(persons, k, kL2, *pair)
                        closing[pair] = (d0 - d1) / gap
                        break
            if closing:
                vals = np.array(list(closing.values()))
                f[col["max_pair_closing_vel"]] = float(vals.max())
                f[col["mean_pair_closing_vel"]] = float(vals.mean())
                cp = fstats[kL2]["closest"]
                if cp in closing:
                    f[col["closest_pair_closing_vel"]] = float(closing[cp])
            # crowd convergence over persons present at both ends (same-keypoint anchors)
            kF2 = kp_[0]
            gap = ftime[kL2] - ftime[kF2]
            if gap >= MIN_PAIR_GAP_SEC - 1e-6:
                (_, PF, mF), (_, PL2, mL) = persons[kF2], persons[kL2]
                common = sorted(set(mF) & set(mL))
                if len(common) >= 2:
                    anc = [_anchor_pair(PF, mF[x], PL2, mL[x]) for x in common]
                    aF, aL = np.array([a for a, _ in anc]), np.array([b for _, b in anc])
                    hF = np.median([PF["h"][mF[x]] for x in common])
                    hL = np.median([PL2["h"][mL[x]] for x in common])
                    sF = np.linalg.norm(aF - aF.mean(0), axis=1).mean() / hF
                    sL = np.linalg.norm(aL - aL.mean(0), axis=1).mean() / hL
                    f[col["crowd_convergence"]] = float((sF - sL) / gap)

    defined = np.isfinite(feat)
    fills = np.array([fill for _, fill, _ in cols], dtype=np.float64)
    feat = np.where(defined, feat, fills[None, :])
    for key in SLOPE_KEYS:
        x, d = feat[:, col[key]], defined[:, col[key]] & valid
        for hz in horizons:
            feat[:, col[f"{key}_slope{hz}s"]] = rolling_slope(x, d, hz)
    feat = np.clip(feat, -1000.0, 1000.0).astype(np.float32)
    meta = dict(
        extractor=EXTRACTOR, version=VERSION, raw_version=raw["meta"].get("version"),
        model_id=raw["meta"]["model_id"], weights_sha256_16=raw["meta"].get("weights_sha256_16"), D=len(cols),
        window_sec=float(W + max(horizons)), frame_window_sec=W,
        history="tracker identity uses all frames in [0, t]; numeric features use [t - window_sec, t]",
        columns=names, column_docs=[doc for _, _, doc in cols], fill_values=[float(x) for x in fills],
        config={**raw["meta"]["config"], **{k: cfg[k] for k in INTERACT_KEYS}},
        video=vmeta,
        units="h = body height: per-track causal median over h_smooth_sec of max(box height, keypoint segment "
              "estimate); s = seconds (rates use the real time between decoded frames)",
        valid_rule="at least one tracked person in [t - frame_window_sec, t]",
    )
    meta["config"]["slope_horizons"] = list(horizons)
    return ticks, feat, valid, meta


# ----------------------------------------------------------------------------------------------------------------
# Registry entry points (see run_extract.py)
# ----------------------------------------------------------------------------------------------------------------

def _split_cfg(cfg):
    return {k: v for k, v in cfg.items() if k not in INTERACT_KEYS}, {k: v for k, v in cfg.items()
                                                                         if k in INTERACT_KEYS}


def extract(video_path, device="cuda:0", **cfg):
    """pose_interact for one video (runs detection + tracking). Returns ``(t_sec, feat, valid, meta)``."""
    raw_cfg, int_cfg = _split_cfg(cfg)
    return interaction_features(track_video(video_path, device, **raw_cfg), **int_cfg)


def extract_and_save(video_path, video_id, device="cuda:0", **cfg):
    """Registry ``pose``: one detector pass, writes pose_raw and pose_interact."""
    raw_cfg, int_cfg = _split_cfg(cfg)
    raw = track_video(video_path, device, **raw_cfg)
    p_raw = save_raw(video_id, raw)
    t, f, v, meta = interaction_features(raw, **int_cfg)
    p_int = io.save_features(EXTRACTOR, video_id, t, f, v, meta)
    return dict(paths={EXTRACTOR_RAW: p_raw, EXTRACTOR: p_int}, duration_sec=raw["meta"]["video"]["duration_sec"])


def derive_and_save(video_path, video_id, device="cuda:0", **cfg):
    """Registry ``pose_interact``: re-derive features from an existing pose_raw (CPU); track first if missing."""
    raw_cfg, int_cfg = _split_cfg(cfg)
    if os.path.exists(io.feature_path(EXTRACTOR_RAW, video_id)):  # any pose_raw version (meta["raw_version"])
        raw = load_raw(video_id)
    else:
        if not str(device).startswith("cuda"):
            raise FileNotFoundError(f"no pose_raw for {video_id}; run --extractor pose (or pass --device cuda:k)")
        raw = track_video(video_path, device, **raw_cfg)
        save_raw(video_id, raw)
    t, f, v, meta = interaction_features(raw, **int_cfg)
    path = io.save_features(EXTRACTOR, video_id, t, f, v, meta)
    return dict(paths={EXTRACTOR: path}, duration_sec=raw["meta"]["video"]["duration_sec"])


def check_raw_causal(video_path, cut_sec, device="cuda:0", **cfg):
    """Compare raw tracks of frames before ``cut_sec - 1`` between the full video and an ffmpeg-truncated copy."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        cut = os.path.join(d, "cut" + os.path.splitext(video_path)[1])
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", video_path, "-t", str(cut_sec), "-c", "copy",
                        cut], check=True)
        a = track_video(video_path, device, **cfg)
        b = track_video(cut, device, **cfg)
    k = int(np.sum(b["frame_t"] <= cut_sec - 1))
    na, nb = a["frame_ptr"][k], b["frame_ptr"][k]
    same_ids = na == nb and np.array_equal(a["det_track_id"][:na], b["det_track_id"][:nb])
    kd = float(np.abs(a["det_kpt"][:na] - b["det_kpt"][:nb]).max()) if same_ids and na else 0.0
    return dict(frames_compared=k, dets=int(na), same_detections_and_ids=bool(same_ids), max_kpt_diff=kd)


# ----------------------------------------------------------------------------------------------------------------
# Self-test (synthetic tracks, CPU only; no detector)
# ----------------------------------------------------------------------------------------------------------------

# A standing person facing the camera, keypoints in units of body height (x from the box center, y from the top)
_TEMPLATE = np.array([[0, .06], [.02, .045], [-.02, .045], [.045, .06], [-.045, .06], [.11, .19], [-.11, .19],
                      [.14, .34], [-.14, .34], [.15, .47], [-.15, .47], [.07, .49], [-.07, .49], [.07, .72],
                      [-.07, .72], [.07, .95], [-.07, .95]])


def _person(cx, top, h, hide=()):
    """(box [4], kpt [17, 3]) of the template person; keypoints in ``hide`` get confidence 0."""
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, 0], kp[:, 1], kp[:, 2] = cx + _TEMPLATE[:, 0] * h, top + _TEMPLATE[:, 1] * h, 0.9
    kp[list(hide), 2] = 0.0
    vis = kp[:, 2] > 0
    bottom = kp[vis, 1].max() + 0.04 * h  # the box ends below the lowest visible keypoint
    return np.array([cx - .17 * h, top, cx + .17 * h, bottom], dtype=np.float32), kp


def _synthetic_raw(frames, fps=24.0, sample_fps=5.0, wh=(640, 360), warps=None):
    """pose_raw dict from ``frames`` = [[(track_id, box, kpt), ...] per sampled frame]."""
    F = len(frames)
    frame_t = np.arange(F) / sample_fps
    counts = [len(fr) for fr in frames]
    rows = [r for fr in frames for r in fr]
    return dict(
        frame_t=frame_t.astype(np.float32), frame_idx=np.floor(frame_t * fps + 1e-6).astype(np.int32),
        frame_ok=np.ones(F, dtype=bool), frame_ptr=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        frame_warp=(np.tile(np.eye(2, 3), (F, 1, 1)) if warps is None else warps).astype(np.float32),
        frame_gmc_ok=np.ones(F, dtype=bool), frame_n_hiconf=np.array(counts, dtype=np.int16),
        det_track_id=np.array([r[0] for r in rows], dtype=np.int32),
        det_box=np.array([r[1] for r in rows], dtype=np.float32).reshape(-1, 4),
        det_conf=np.full(len(rows), 0.9, dtype=np.float32),
        det_kpt=np.array([r[2] for r in rows], dtype=np.float32).reshape(-1, 17, 3),
        meta=dict(model_id="synthetic", version=RAW_VERSION, config=dict(sample_fps=sample_fps),
                  video=dict(fps=fps, num_frames=int(F / sample_fps * fps), duration_sec=F / sample_fps,
                             native_wh=list(wh), decoded_wh=list(wh))),
    )


def selftest():
    """Checks for the version-2 fixes; raises AssertionError on failure. Returns a dict of the numbers."""
    import tempfile

    from anticipation.features import motion

    res = {}
    col = {c: i for i, c in enumerate(COLUMNS)}
    legs = (LHIP, RHIP, LKNE, RKNE, LANK, RANK)

    # 1) torso flicker: a person standing still whose hips/legs vanish on every other frame. Version 1 switched
    #    anchor (torso center <-> box center) and h (3 x torso <-> upper-body box): about 1 h/s of fake speed.
    frames = [[(1, *_person(320, 60, 200, hide=legs if k % 2 else ()))] for k in range(60)]
    _, f, v, _ = interaction_features(_synthetic_raw(frames))
    res["flicker_body_speed_max"] = float(f[v, col["body_speed_max"]].max())
    res["flicker_wrist_speed_max"] = float(f[v, col["wrist_speed_max"]].max())
    hm = f[v, col["person_height_median"]]
    res["flicker_height_max_over_min"] = float(hm.max() / hm.min())
    assert res["flicker_body_speed_max"] < 0.01 and res["flicker_wrist_speed_max"] < 0.01, res
    assert res["flicker_height_max_over_min"] < 1.1, res

    # 2) real dt at 24 fps: constant 100 px/s walk sampled on frame steps 4,5,5,5,5 -> constant per-step speed
    frames = [[(1, *_person(100 + 100.0 * np.floor(k / 5 * 24 + 1e-6) / 24, 60, 200))] for k in range(60)]
    _, f, v, _ = interaction_features(_synthetic_raw(frames))
    ratio = f[10:, col["body_speed_max"]] / f[10:, col["body_speed_mean"]]
    res["walk24_speed_max_over_mean"] = float(ratio.max())
    assert ratio.max() < 1.01, ratio  # nominal 0.2 s would give about 1.04 (0.833 v and 1.042 v alternate)

    # 3) head yaw: continuous, no saturation with one ear hidden. Head turned 30 deg to image right:
    #    ears at +/-7 cos(30) cm, nose at 11 sin(30) cm (h = 170 cm).
    box, kp = _person(320, 60, 170)
    yaw = np.deg2rad(30)
    kp[NOSE, 0] = 320 + 11 * np.sin(yaw)
    kp[LEAR, 0], kp[REAR, 0] = 320 + 7 * np.cos(yaw), 320 - 7 * np.cos(yaw)
    kp[LEYE, 0], kp[REYE, 0] = 320 + 8.5 * np.sin(yaw) + 3.2 * np.cos(yaw), 320 + 8.5 * np.sin(yaw) - 3.2 * np.cos(yaw)
    f_both = float(person_attributes(box[None], kp[None], 0.3)["facing"][0])
    kp1 = kp.copy()
    kp1[LEAR, 2] = 0.0  # one ear hidden: falls back to the eyes
    f_eyes = float(person_attributes(box[None], kp1[None], 0.3)["facing"][0])
    res["yaw30_ears_eyes_true"] = (f_both, f_eyes, float(np.sin(yaw)))
    assert abs(f_both - np.sin(yaw)) < 0.1 and abs(f_eyes - np.sin(yaw)) < 0.25, res["yaw30_ears_eyes_true"]

    # 4) camera: a pure 5 % zoom about the frame center is not translation (version 1: 0.14 frame widths / s)
    s = 1.05
    c = np.array([320.0, 180.0])
    zoom = np.array([[s, 0, c[0] - s * c[0]], [0, s, c[1] - s * c[1]]])
    frames = [[(1, *_person(320, 60, 200))] for _ in range(30)]
    _, f, _, _ = interaction_features(_synthetic_raw(frames, warps=np.tile(zoom, (30, 1, 1))))
    res["zoom_cam_translation"] = float(np.abs(f[3:, col["cam_translation"]]).max())
    res["zoom_cam_zoom"] = float(f[3:, col["cam_zoom"]].mean())
    assert res["zoom_cam_translation"] < 1e-3 and res["zoom_cam_zoom"] > 0.1, res

    # 5) GMC failure: the recorded warp must be the identity the tracker used, not the previous frame's warp
    import cv2
    from ultralytics.engine.results import Boxes

    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur((rng.random((360, 640)) * 255).astype(np.float32), (0, 0), 3)
    img = np.dstack([cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)] * 3)
    shifted = np.roll(img, 30, axis=1)
    tracker, _ = make_tracker(dict(DEFAULTS))
    empty = Boxes(np.zeros((0, 6), dtype=np.float32), (360, 640))
    tracker.update(empty, img)
    tracker.update(empty, shifted)
    res["gmc_shift_tx"] = float(tracker.last_warp[0, 2])
    est = cv2.estimateAffinePartial2D
    cv2.estimateAffinePartial2D = lambda *a, **k: (None, None)  # degenerate points: makes GMC raise
    try:
        tracker.update(empty, shifted)
    finally:
        cv2.estimateAffinePartial2D = est
    res["gmc_failed_tx_ok"] = (float(tracker.last_warp[0, 2]), bool(tracker.last_gmc_ok))
    assert res["gmc_shift_tx"] > 20 and np.allclose(tracker.last_warp, np.eye(2, 3)) and not tracker.last_gmc_ok

    # 6) decode-error locality in the detection prefetch (fake detector): a bad frame removes only itself
    def fake_detector(frames, **kw):
        return [np.full((1, 57), float(fr.mean()), dtype=np.float32) for fr in frames]

    with tempfile.TemporaryDirectory() as d:
        path = motion.synthetic_video(os.path.join(d, "v.mp4"), fps=24.0, seconds=12.0)
        cfg = dict(DEFAULTS)

        def run():
            video = io.Video(path, num_threads=1)
            ft = video.sample_times(cfg["sample_fps"])
            out = {}
            for s0, frames, dets, ok in _prefetch_detections(video, ft, fake_detector, cfg):
                for j in range(len(ok)):
                    out[s0 + j] = float(dets[j][0, 0]) if ok[j] else None
            return ft, out

        ft, good = run()
        t_bad = 7.0  # inside the second 32-frame batch (6.4 .. 12.8 s)
        orig = io.Video.frames_at

        def failing(self, t_sec):
            if (self.frame_index(t_sec) == int(np.floor(t_bad * self.info.fps + 1e-6))).any():
                raise RuntimeError("simulated decode error")
            return orig(self, t_sec)

        io.Video.frames_at = failing
        try:
            _, bad = run()
        finally:
            io.Video.frames_at = orig
        missing = [k for k, x in bad.items() if x is None]
        res["decode_error_missing_frames"] = [float(ft[k]) for k in missing]
        assert missing == [int(round(t_bad * cfg["sample_fps"]))], missing
        assert all(bad[k] == good[k] for k in good if k not in missing)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video")
    ap.add_argument("--selftest", action="store_true", help="run the synthetic CPU self-test and exit")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--weights", default=DEFAULTS["weights"])
    ap.add_argument("--print_ticks", default="", help="range a:b of tick indices to print")
    ap.add_argument("--print_cols", default="", help="comma-separated columns to print (default: all)")
    ap.add_argument("--check_causal", type=float, default=0.0, help="cut point in seconds for io.check_causal")
    ap.add_argument("--save", default="", help="video_id: also write pose_raw and pose_interact")
    args = ap.parse_args()
    if args.selftest:
        print("pose selftest ok", json.dumps(selftest()))
        return
    if not args.video:
        ap.error("--video is required (or --selftest)")
    t0 = time.time()
    raw = track_video(args.video, args.device, weights=args.weights)
    t1 = time.time()
    t_sec, feat, valid, meta = interaction_features(raw)
    t2 = time.time()
    dur = raw["meta"]["video"]["duration_sec"]
    print(f"frames={len(raw['frame_t'])} dets={len(raw['det_conf'])} tracks={len(np.unique(raw['det_track_id']))} "
          f"T={len(t_sec)} D={feat.shape[1]} valid={valid.mean():.3f} track={t1 - t0:.1f}s features={t2 - t1:.2f}s "
          f"speed={dur / (t2 - t0):.1f}x realtime")
    if args.print_ticks:
        a, b = (int(x) for x in args.print_ticks.split(":"))
        sel = args.print_cols.split(",") if args.print_cols else meta["columns"]
        ci = [meta["columns"].index(c) for c in sel]
        for i in range(a, min(b, len(t_sec))):
            print(int(t_sec[i]), int(valid[i]), json.dumps({meta["columns"][c]: round(float(feat[i, c]), 3)
                                                            for c in ci}))
    if args.check_causal:
        print("check_causal pose_interact", io.check_causal(
            lambda p: extract(p, args.device, weights=args.weights)[:3], args.video, args.check_causal))
        print("check_causal pose_raw", check_raw_causal(args.video, args.check_causal, args.device,
                                                       weights=args.weights))
    if args.save:
        print(save_raw(args.save, raw))
        print(io.save_features(EXTRACTOR, args.save, t_sec, feat, valid, meta))


if __name__ == "__main__":
    main()
