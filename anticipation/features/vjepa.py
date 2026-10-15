"""V-JEPA 2 ViT-L causal clip features at 1 Hz (extractor ``vjepa2_l``).

What
    For every integer tick t in 1..floor(duration) (see ``io.tick_times``) a clip of ``num_frames`` = 16 frames
    is sampled uniformly from the window ``[t - window_sec, t]`` with the last frame at t. With the default
    ``window_sec = 4`` the frame times are t - 3.75, t - 3.5, ..., t (4 fps). Each time maps to the last decoded
    frame at or before it (``io.Video.frame_index``), so the feature at tick t only uses media in [t - 4 s, t].
    Times before 0 map to frame 0 (the clip is padded by repeating the first frame), which is still causal.

    The clip goes through the frozen V-JEPA 2 ViT-L encoder (``facebook/vjepa2-vitl-fpc64-256``, 24 blocks,
    width 1024, tubelet 2x16x16, so 8 x 16 x 16 = 2048 patch tokens per 16-frame clip at 256 px). The
    predictor is never run. Two poolings are stored side by side in one file:

        feat[:, 0:1024]     "last_mean": mean over the 2048 tokens of the last layer (after the final LayerNorm)
        feat[:, 1024:2048]  "mid_mean":  mean over the 2048 tokens of the output of block ``mid_layer`` (12 of 24),
                                         each token LayerNorm-ed without affine parameters before pooling (the
                                         raw residual stream has outlier channels with |x| > 300, which would
                                         dominate the mean and exceed float16 precision)

    So D = 2048. ``split_columns(feat, meta)`` returns the two blocks. A model that wants only the last layer
    reads ``feat[:, :1024]``. The column layout is recorded in ``meta["column_groups"]``.

Preprocessing (bitwise identical to ``VJEPA2VideoProcessor`` for this checkpoint)
    1. resize so the short side is 292 px and the long side ``int(292 * long / short)``, with torchvision's uint8
       bilinear antialiased kernel on the CPU: the exact call the processor makes, so the result is uint8;
    2. center crop 256 x 256 at the processor's offsets ``int((size - 256) / 2)`` (truncation);
    3. on the GPU: float32 ``(x - 255 * mean) / (255 * std)`` with the ImageNet mean/std (the processor's fused
       rescale + normalize), then cast to the model dtype.
    ``check_processor`` (CLI ``--check-processor``) asserts that the float32 pixels equal the
    ``AutoVideoProcessor`` output exactly (max |diff| = 0). Steps 1 and 2 run in the background decode thread
    (``resize_frames``, about 0.7 ms per 360p frame, about 4 unique frames per tick).
    Version 1 resized in float32 on the GPU without rounding back to uint8 and used a round-half-even crop
    offset (1 px off on 640x360); on upsampled 320x240 UCF video that moved the features by about half the
    across-clip spread. Version 1 files are not comparable with version 2.

    ``crop="full"`` instead squashes the whole frame to 256 x 256 with the same uint8 kernel (keeps the frame
    edges of 16:9 video at the cost of aspect distortion); it is not the default because it is not what the
    checkpoint was evaluated with. Changing ``crop`` changes the features: use a different extractor name.

Decode errors
    The unique frames of a chunk of ticks are decoded with one ``get_batch`` call. If that call fails,
    ``decode_frames`` decodes each frame on its own, so a bad frame only invalidates the ticks whose clip uses
    it (at most about 4 ticks). ``meta["n_decode_errors"]`` counts invalid ticks, ``meta["n_bad_frames"]``
    undecodable frames. Invalid ticks are zero with ``valid = False``.

Why
    V-JEPA 2 is the best frozen motion / action-anticipation encoder available (docs/research/backbones.md
    section 3). It replaces the old VideoMAE-base clip features. The middle layer is cached because
    intermediate layers often probe better than the output layer for motion and interaction cues.

Speed
    Frames for a chunk of ticks are decoded once and shared by the overlapping clips. Clips run in fixed-size
    batches (the last batch is padded by repeating its final clip), so the feature of a clip never depends on
    how many ticks the video has; this makes ``io.check_causal`` exact and bitwise reproducible.

Usage
    # one video, print shapes and timing (does not write anything)
    .venv/bin/python -m anticipation.features.vjepa VIDEO.mp4 --device cuda:1

    # write FEATURE_ROOT/vjepa2_l/<video_id>.npz
    .venv/bin/python -m anticipation.features.vjepa VIDEO.mp4 --device cuda:1 --video-id xd/<stem>

    # causality check: re-extract on a copy truncated at 60 s and compare
    .venv/bin/python -m anticipation.features.vjepa VIDEO.mp4 --device cuda:1 --check-causal 60

    # preprocessing parity with the HF processor (no model needed; exits non-zero on a mismatch)
    .venv/bin/python -m anticipation.features.vjepa VIDEO.mp4 --check-processor

    # from Python (registry entry point used by run_extract.py)
    from anticipation.features import vjepa
    t_sec, feat, valid, meta = vjepa.extract(path, device="cuda:1")
"""

import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from anticipation.features import io

EXTRACTOR = "vjepa2_l"
VERSION = 2  # 2: processor-exact uint8 preprocessing, per-frame decode-error fallback (1: float resize, see above)
MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"

DEFAULTS = dict(
    window_sec=4.0,      # clip covers [t - window_sec, t]
    num_frames=16,       # frames per clip (the checkpoint uses tubelets of 2 frames)
    mid_layer=12,        # 1-based block index whose output is pooled into the "mid_mean" columns
    crop="center",       # "center" (processor default) or "full" (squash the whole frame)
    batch_size=16,       # clips per forward pass (fixed; last batch is padded)
    chunk_ticks=64,      # ticks whose frames are decoded together
    dtype="bf16",        # "bf16", "fp16" or "fp32" model weights and activations
    decode_threads=4,    # decord threads
)

_RESIZE_SHORT = 292      # int(256 * 256 / 224), from the checkpoint's video_preprocessor_config.json
_CROP = 256
_RESCALE = 0.00392156862745098  # processor rescale_factor (1 / 255)
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

_MODEL_CACHE = {}


def prefetch(iterable, depth=2):
    """Run ``iterable`` in a daemon thread, buffering up to ``depth`` items; re-raises producer errors.

    Used to overlap video decoding (decord releases the GIL) with GPU work. Shared with siglip.py.
    """
    import queue
    import threading

    q = queue.Queue(maxsize=depth)
    done = object()
    stop = threading.Event()  # set when the consumer stops early (e.g. a GPU error) so the worker exits

    def put(item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def worker():
        try:
            for item in iterable:
                if not put(item):
                    return
        except BaseException as e:  # noqa: BLE001 - forwarded to the consumer
            put(e)
            return
        put(done)

    threading.Thread(target=worker, daemon=True).start()
    try:
        while True:
            item = q.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()


def decode_frames(vr, indices):
    """Decode frames ``indices`` with a decord VideoReader: ``(frames [N, H, W, 3] uint8 or None, ok [N] bool)``.

    Tries one ``get_batch`` call first. If it raises (corrupt stream, or a frame count that overstates the
    decodable frames), each index is decoded on its own, so one bad frame costs only that frame: its row is zero
    and ``ok`` is False there. ``frames`` is None only if no frame decodes. Shared with siglip.py.
    """
    indices = [int(i) for i in indices]
    ok = np.ones(len(indices), dtype=bool)
    try:
        return vr.get_batch(indices).asnumpy(), ok
    except Exception:  # noqa: BLE001 - decord raises a bare DECORDError on corrupt streams
        pass
    frames = None
    for k, i in enumerate(indices):
        try:
            frame = vr.get_batch([i]).asnumpy()[0]
        except Exception:  # noqa: BLE001
            ok[k] = False
            continue
        if frames is None:
            frames = np.zeros((len(indices),) + frame.shape, dtype=np.uint8)
        frames[k] = frame
    return frames, ok


def load_model(device="cuda:1", dtype="bf16"):
    """Load (once per process) the V-JEPA 2 ViT-L model in eval mode on ``device``."""
    key = (device, dtype)
    if key not in _MODEL_CACHE:
        from transformers import VJEPA2Model

        model = VJEPA2Model.from_pretrained(MODEL_ID, dtype=_DTYPES[dtype], attn_implementation="sdpa")
        _MODEL_CACHE[key] = model.to(device).eval()
    return _MODEL_CACHE[key]


def clip_times(ticks, window_sec=4.0, num_frames=16):
    """Frame times [T, num_frames] for clips ending at each tick: t - window + (j + 1) * window / num_frames.

    The last column is exactly t (offset 0.0), so the last frame of each clip is the frame at the tick.
    """
    offsets = window_sec * (np.arange(1, num_frames + 1, dtype=np.float64) / num_frames - 1.0)
    return np.asarray(ticks, dtype=np.float64)[:, None] + offsets[None, :]


def resize_frames(frames, crop="center"):
    """uint8 frames [N, H, W, 3] (numpy) -> uint8 tensor [N, 3, 256, 256] on the CPU, as the processor does it.

    ``crop="center"``: torchvision ``resize`` (uint8, bilinear, antialias) of the short side to 292 and the long
    side to ``int(292 * long / short)``, then the processor's center crop offsets ``int((size - 256) / 2)``.
    ``crop="full"``: the same kernel squashes the whole frame to 256 x 256. Runs in the decode thread.
    """
    import torchvision.transforms.v2.functional as TF

    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)  # NCHW view of NHWC memory
    h, w = x.shape[-2:]
    bilinear = TF.InterpolationMode.BILINEAR
    if crop == "full":
        return TF.resize(x, [_CROP, _CROP], interpolation=bilinear, antialias=True).contiguous()
    if crop != "center":
        raise ValueError(f"unknown crop mode {crop!r}")
    if h <= w:
        nh, nw = _RESIZE_SHORT, int(_RESIZE_SHORT * w / h)
    else:
        nh, nw = int(_RESIZE_SHORT * h / w), _RESIZE_SHORT
    x = TF.resize(x, [nh, nw], interpolation=bilinear, antialias=True)
    top, left = int((nh - _CROP) / 2.0), int((nw - _CROP) / 2.0)  # HF center_crop truncates, it does not round
    return x[:, :, top:top + _CROP, left:left + _CROP].contiguous()


def normalize(x, device, dtype=torch.bfloat16):
    """uint8 [N, 3, S, S] tensor -> model input on ``device``: float32 ``(x - 255 mean) / (255 std)`` -> ``dtype``.

    The same float32 arithmetic as the processor's fused rescale + normalize, so the values are bitwise equal.
    """
    x = x.to(device, non_blocking=True).float()
    mean = (torch.tensor(_MEAN, device=device) * (1.0 / _RESCALE)).view(1, 3, 1, 1)
    std = (torch.tensor(_STD, device=device) * (1.0 / _RESCALE)).view(1, 3, 1, 1)
    return ((x - mean) / std).to(dtype)


def preprocess(frames, device, crop="center", dtype=torch.bfloat16):
    """uint8 frames [N, H, W, 3] (numpy) -> normalized tensor [N, 3, 256, 256] on ``device``."""
    return normalize(resize_frames(frames, crop), device, dtype)


def check_processor(video_path, device="cpu", n_clips=6):
    """Compare ``preprocess`` with ``AutoVideoProcessor`` (float32 pixels) on ``n_clips`` clips of a video.

    Returns ``{"clips", "max_abs_diff", "exact"}``; ``exact`` is True when every pixel value is bitwise equal.
    """
    from transformers import AutoVideoProcessor

    proc = AutoVideoProcessor.from_pretrained(MODEL_ID)
    video = io.Video(video_path)
    ticks = io.tick_times(video.info.duration_sec)
    sel = ticks[np.unique(np.linspace(0, len(ticks) - 1, min(n_clips, len(ticks))).astype(int))]
    diff = 0.0
    for t in sel:
        frames = video.frames_at(clip_times([t])[0])
        ref = proc([frames], return_tensors="pt")["pixel_values_videos"][0].float()
        mine = preprocess(frames, device, "center", torch.float32).cpu()
        if ref.shape != mine.shape:
            raise AssertionError(f"shape {tuple(mine.shape)} != processor {tuple(ref.shape)}")
        diff = max(diff, float((ref - mine).abs().max()))
    return {"clips": int(len(sel)), "resolution": f"{video.info.width}x{video.info.height}",
            "max_abs_diff": diff, "exact": diff == 0.0}


@torch.inference_mode()
def encode_clips(model, clips, mid_layer=12):
    """clips [B, F, 3, 256, 256] -> (last_mean [B, 1024], mid_mean [B, 1024]) as float32 tensors."""
    captured = {}

    def hook(_module, _inputs, outputs):
        captured["mid"] = outputs[0]

    handle = model.encoder.layer[mid_layer - 1].register_forward_hook(hook)
    try:
        last = model.encoder(pixel_values_videos=clips).last_hidden_state
    finally:
        handle.remove()
    mid = captured["mid"].float()
    mid = F.layer_norm(mid, mid.shape[-1:], eps=1e-6)
    return last.float().mean(dim=1), mid.mean(dim=1)


def extract(video_path, device="cuda:1", tick_frame_fn=None, tick_frame_prep=None, **cfg):
    """Extract ``vjepa2_l`` features for one video.

    Returns ``(t_sec [T] float32, feat [T, 2048] float16, valid [T] bool, meta dict)`` following io.py.
    Ticks whose clip contains an undecodable frame are zero with ``valid = False``.

    ``tick_frame_fn(c0, c1, x, ok)``, if given, is called once per decoded chunk with the frames at ticks
    c0..c1-1 (the last frame of each clip, identical to ``io.Video.frames_at(t_sec[c0:c1])``) and ``ok``
    [c1 - c0] bool, False where that frame failed to decode (its row is zero). ``x`` is the uint8 frames
    [c1 - c0, H, W, 3], or ``tick_frame_prep(frames)`` if ``tick_frame_prep`` is given; the prep function runs
    in the decode thread (``prompts.extract_visual`` passes ``siglip.resize_frames``). This lets another
    per-frame extractor reuse this decode. It is not called for chunks in which no frame decoded.
    """
    ignored = sorted(set(cfg) - set(DEFAULTS))  # tolerate generic runner kwargs; recorded in meta
    cfg = {**DEFAULTS, **{k: v for k, v in cfg.items() if k in DEFAULTS}}
    t_start = time.time()
    dtype = _DTYPES[cfg["dtype"]]
    model = load_model(device, cfg["dtype"])
    hidden = model.config.hidden_size

    video = io.Video(video_path, num_threads=cfg["decode_threads"])
    ticks = io.tick_times(video.info.duration_sec)
    T, B = len(ticks), cfg["batch_size"]
    feat = np.zeros((T, 2 * hidden), dtype=np.float16)
    valid = np.zeros(T, dtype=bool)
    bad_frames = set()

    all_times = clip_times(ticks, cfg["window_sec"], cfg["num_frames"])

    def decode_chunks():
        """Yield (c0, c1, pix uint8 [U, 3, 256, 256] or None, inv [C, F], ok [U], uniq [U], tick frames)."""
        for c0 in range(0, T, cfg["chunk_ticks"]):
            c1 = min(T, c0 + cfg["chunk_ticks"])
            idx = video.frame_index(all_times[c0:c1])                # [C, F] frame indices
            uniq, inv = np.unique(idx, return_inverse=True)
            inv = inv.reshape(idx.shape)
            frames, ok = decode_frames(video.vr, uniq)
            if frames is None:
                yield c0, c1, None, inv, ok, uniq, None
                continue
            tick = None
            if tick_frame_fn is not None:
                tick = frames[inv[:, -1]]                            # last clip frame = frame at the tick
                if tick_frame_prep is not None:
                    tick = tick_frame_prep(tick)
            yield c0, c1, resize_frames(frames, cfg["crop"]), inv, ok, uniq, tick

    # Decoding and resizing run in a background thread (decord and torch release the GIL) while the GPU encodes.
    for c0, c1, pix, inv, ok, uniq, tick in prefetch(decode_chunks()):
        bad_frames.update(uniq[~ok].tolist())
        if pix is None:
            continue
        if tick_frame_fn is not None:
            tick_frame_fn(c0, c1, tick, ok[inv[:, -1]])
        clip_ok = ok[inv].all(axis=1)                                # [C] every frame of the clip decoded
        pix = normalize(pix, device, dtype)                          # [U, 3, 256, 256]
        inv_t = torch.from_numpy(inv).to(device)
        for b0 in range(0, c1 - c0, B):
            sel = inv_t[b0:b0 + B]
            n = sel.shape[0]
            if n < B:  # pad to a fixed batch shape so results do not depend on the number of ticks
                sel = torch.cat([sel, sel[-1:].expand(B - n, -1)], dim=0)
            last, mid = encode_clips(model, pix[sel], cfg["mid_layer"])
            feat[c0 + b0:c0 + b0 + n] = torch.cat([last, mid], dim=1)[:n].cpu().numpy().astype(np.float16)
            valid[c0 + b0:c0 + b0 + n] = clip_ok[b0:b0 + n]
    feat[~valid] = 0

    meta = dict(
        extractor=EXTRACTOR,
        version=VERSION,
        model_id=MODEL_ID,
        model_revision=getattr(model.config, "_commit_hash", None),
        window_sec=float(cfg["window_sec"]),
        D=int(feat.shape[1]),
        column_groups={"last_mean": [0, hidden], "mid_mean": [hidden, 2 * hidden]},
        num_frames=int(cfg["num_frames"]),
        frame_step_sec=float(cfg["window_sec"]) / cfg["num_frames"],
        mid_layer=int(cfg["mid_layer"]),
        num_layers=int(model.config.num_hidden_layers),
        crop=cfg["crop"],
        input_size=_CROP,
        preprocessing="torchvision uint8 bilinear antialias resize (CPU) + HF center crop; == VJEPA2VideoProcessor",
        dtype=cfg["dtype"],
        cfg=cfg,
        pooling="last: mean of final-LayerNorm tokens; mid: mean of per-token LayerNorm(no affine) block output",
        video_fps=float(video.info.fps),
        duration_sec=float(video.info.duration_sec),
        n_decode_errors=int(T - valid.sum()),
        n_bad_frames=len(bad_frames),
        ignored_cfg=ignored,
        wall_sec=round(time.time() - t_start, 3),
    )
    return ticks, feat, valid, meta


def split_columns(feat, meta):
    """Return ``{"last_mean": feat[:, a:b], "mid_mean": feat[:, c:d]}`` using ``meta["column_groups"]``."""
    return {k: feat[:, a:b] for k, (a, b) in meta["column_groups"].items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--video-id", help="save to FEATURE_ROOT/vjepa2_l/<video-id>.npz (e.g. xd/<stem>)")
    ap.add_argument("--check-causal", type=float, metavar="CUT_SEC",
                    help="run io.check_causal with the video truncated at CUT_SEC")
    ap.add_argument("--check-processor", action="store_true",
                    help="only compare preprocessing with VJEPA2VideoProcessor (CPU) and exit; status 1 if not exact")
    ap.add_argument("--crop", default=DEFAULTS["crop"], choices=["center", "full"])
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--dtype", default=DEFAULTS["dtype"], choices=sorted(_DTYPES))
    args = ap.parse_args()
    if args.check_processor:
        res = check_processor(args.video)
        print("check_processor", json.dumps(res))
        sys.exit(0 if res["exact"] else 1)
    cfg = dict(crop=args.crop, batch_size=args.batch_size, dtype=args.dtype)

    load_model(args.device, args.dtype)  # exclude one-time model loading from the timing
    t_sec, feat, valid, meta = extract(args.video, args.device, **cfg)
    print(f"{EXTRACTOR}: T={len(t_sec)} D={feat.shape[1]} valid={int(valid.sum())}/{len(valid)} "
          f"wall={meta['wall_sec']:.1f}s speed={meta['duration_sec'] / meta['wall_sec']:.1f} video-s/s")
    if args.video_id:
        print("saved", io.save_features(EXTRACTOR, args.video_id, t_sec, feat, valid, meta))
    if args.check_causal:
        res = io.check_causal(lambda p: extract(p, args.device, **cfg)[:3], args.video, args.check_causal)
        print("check_causal", json.dumps(res))


if __name__ == "__main__":
    main()
