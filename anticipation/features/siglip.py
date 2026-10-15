"""SigLIP 2 per-frame semantic embeddings at 1 Hz (extractor ``siglip2``).

What
    For every integer tick t in 1..floor(duration) the frame at t (the last decoded frame at or before t,
    ``io.Video.frame_index``) is embedded with the frozen SigLIP 2 So400m/14 image tower
    (``google/siglip2-so400m-patch14-384``) and L2-normalized. window_sec = 0: the feature at tick t uses
    exactly one frame, at time t. Output D = 1152.

    The text tower of the same checkpoint scores text prompts against these embeddings; that is the separate
    extractor ``siglip2_prompts`` in ``prompts.py``. Prompt scores can be recomputed from saved ``siglip2``
    files without decoding any video (``python -m anticipation.features.prompts --rescore``).

Preprocessing (bitwise identical to ``SiglipImageProcessor`` on every local source resolution)
    squash the full frame to 384 x 384 with torchvision's uint8 bilinear antialiased kernel on the CPU (no crop, so
    the frame edges are kept; the result is uint8, as the processor's PIL resize) -> on the GPU, float32
    ``float32(x * (1 / 255))`` -> ``(x - 0.5) / 0.5`` (the processor's arithmetic) -> model dtype. The embedding is
    the MAP-head pooled output (``get_image_features``).
    torchvision's uint8 kernel is a port of Pillow's resampler: it equals ``SiglipImageProcessorFast`` exactly and
    the default (slow, PIL) ``SiglipImageProcessor`` exactly for 320x240, 640x360, 640x268, 640x272 and 360x360
    input. For some other sizes about 0.01 % (640x346) to 4 % (640x480, 854x480) of output pixels differ by one
    intensity level from PIL. ``check_processor`` (CLI ``--check-processor``) measures this on a video.
    The resize runs in the background decode thread (``resize_frames``, about 0.6 ms per frame).
    Version 1 resized in float32 on the GPU without rounding back to uint8. On 320x240 UCF video that alone
    lowered the cosine to the processor's embedding to a median of 0.965 (version 2: exact), so version 1 files
    are not comparable with version 2.

Why SigLIP 2 rather than PE-Core-L14-336
    docs/research/backbones.md ranks the two as near-equal for frozen semantics (PE-Core slightly ahead on
    video zero-shot) and says to pick one. SigLIP 2 was chosen because:
      1. it loads natively in the installed transformers 4.57 (``SiglipModel``), with no new dependency.
         PE-Core needs ``open_clip_torch`` (not installed) on top of timm, adding a second preprocessing and
         model-loading stack for one stream;
      2. it is trained with a sigmoid (per pair) loss, so each prompt score is an independent "is this
         concept present" score with a calibrated bias, instead of a softmax-relative CLIP score. The 40 to 60
         escalation prompts are multi-label concepts, so this matches the use;
      3. 384 px input with no center crop keeps small and peripheral people in 16:9 CCTV and phone video.
    If PE-Core is wanted later, add ``pe_core.py`` with the same interface and a new extractor name.

Gotchas
    * Text must be lowercased and padded to max_length = 64 (how SigLIP 2 was trained); ``embed_texts``
      does both.
    * Features are deterministic per frame: images are encoded in fixed-size batches (the last batch is
      padded), so ``io.check_causal`` is exact.
    * A frame that fails to decode only invalidates its own tick (``vjepa.decode_frames`` falls back to one
      frame at a time when a chunk's batch decode fails). ``meta["n_decode_errors"]`` counts invalid ticks.
    * ``logit_params`` reads ``logit_scale`` / ``logit_bias`` in float32 from the checkpoint file (109.894,
      -15.9326). The bf16 model holds rounded copies (108.5, -15.9375); version 1 prompt files stored those.

Usage
    .venv/bin/python -m anticipation.features.siglip VIDEO.mp4 --device cuda:1
    .venv/bin/python -m anticipation.features.siglip VIDEO.mp4 --device cuda:1 --video-id ucf/Fighting003_x264
    .venv/bin/python -m anticipation.features.siglip VIDEO.mp4 --device cuda:1 --check-causal 60
    .venv/bin/python -m anticipation.features.siglip VIDEO.mp4 --check-processor   # exits 1 unless exact

    from anticipation.features import siglip
    t_sec, feat, valid, meta = siglip.extract(path, device="cuda:1")
"""

import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from anticipation.features import io
from anticipation.features.vjepa import decode_frames, prefetch

EXTRACTOR = "siglip2"
VERSION = 2  # 2: processor-exact uint8 resize, per-frame decode-error fallback (1: float resize on the GPU)
MODEL_ID = "google/siglip2-so400m-patch14-384"

DEFAULTS = dict(
    batch_size=32,       # frames per forward pass (fixed; last batch is padded)
    chunk_ticks=128,     # ticks decoded together
    dtype="bf16",        # "bf16", "fp16" or "fp32"
    decode_threads=4,    # decord threads
)

_SIZE = 384
_RESCALE = 0.00392156862745098  # processor rescale_factor (1 / 255)
_MEAN = 0.5
_STD = 0.5
_TEXT_LEN = 64
_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

_CACHE = {}


def load_model(device="cuda:1", dtype="bf16"):
    """Load (once per process) the SigLIP 2 model and tokenizer; returns ``(model, tokenizer)``."""
    key = (device, dtype)
    if key not in _CACHE:
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained(MODEL_ID, dtype=_DTYPES[dtype], attn_implementation="sdpa")
        _CACHE[key] = (model.to(device).eval(), AutoTokenizer.from_pretrained(MODEL_ID))
    return _CACHE[key]


def resize_frames(frames):
    """uint8 frames [N, H, W, 3] (numpy) -> uint8 tensor [N, 3, 384, 384] on the CPU (squash, no crop).

    torchvision ``resize`` (uint8, bilinear, antialias), which matches the processor's PIL bilinear resize (see the
    module docstring). Runs in the decode thread.
    """
    import torchvision.transforms.v2.functional as TF

    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)  # NCHW view of NHWC memory
    return TF.resize(x, [_SIZE, _SIZE], interpolation=TF.InterpolationMode.BILINEAR, antialias=True).contiguous()


def normalize(x, device, dtype=torch.bfloat16):
    """uint8 [N, 3, 384, 384] tensor -> model input on ``device`` with the processor's arithmetic, cast to ``dtype``.

    ``SiglipImageProcessor`` computes ``float32(float64(x) * rescale_factor)`` and then ``(x - 0.5) / 0.5`` in
    float32; doing the same keeps the float32 values bitwise equal.
    """
    x = (x.to(device, non_blocking=True).double() * _RESCALE).float()
    return ((x - _MEAN) / _STD).to(dtype)


def preprocess(frames, device, dtype=torch.bfloat16):
    """uint8 frames [N, H, W, 3] (numpy) -> normalized tensor [N, 3, 384, 384] on ``device``."""
    return normalize(resize_frames(frames), device, dtype)


def check_processor(video_path, device="cpu", n_frames=24):
    """Compare ``preprocess`` with the default ``AutoImageProcessor`` (PIL) on ``n_frames`` tick frames of a video.

    Returns ``{"frames", "max_abs_diff", "frac_pixels_differ", "exact"}`` on float32 pixel values.
    """
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    video = io.Video(video_path)
    ticks = io.tick_times(video.info.duration_sec)
    sel = ticks[np.unique(np.linspace(0, len(ticks) - 1, min(n_frames, len(ticks))).astype(int))]
    frames = video.frames_at(sel)
    ref = proc(images=list(frames), return_tensors="pt")["pixel_values"].float()
    mine = preprocess(frames, device, torch.float32).cpu()
    if ref.shape != mine.shape:
        raise AssertionError(f"shape {tuple(mine.shape)} != processor {tuple(ref.shape)}")
    d = (ref - mine).abs()
    return {"frames": int(len(sel)), "resolution": f"{video.info.width}x{video.info.height}",
            "processor": type(proc).__name__, "max_abs_diff": float(d.max()),
            "frac_pixels_differ": float((d > 0).float().mean()), "exact": bool(d.max() == 0)}


@torch.inference_mode()
def embed_images(model, pixels, batch_size=32):
    """Normalized pixels [N, 3, 384, 384] -> L2-normalized float32 embeddings [N, 1152] (on the GPU)."""
    outs = []
    for b0 in range(0, pixels.shape[0], batch_size):
        x = pixels[b0:b0 + batch_size]
        n = x.shape[0]
        if n < batch_size:  # fixed batch shape -> results independent of how many frames the video has
            x = torch.cat([x, x[-1:].expand(batch_size - n, -1, -1, -1)], dim=0)
        emb = model.get_image_features(pixel_values=x)[:n].float()
        outs.append(F.normalize(emb, dim=-1))
    return torch.cat(outs, dim=0)


@torch.inference_mode()
def embed_texts(texts, device="cuda:1", dtype="bf16"):
    """Lowercase, pad to 64 tokens and encode; returns L2-normalized float32 embeddings [P, 1152] on CPU."""
    model, tok = load_model(device, dtype)
    enc = tok([t.lower() for t in texts], padding="max_length", max_length=_TEXT_LEN, truncation=True,
              return_tensors="pt").to(device)
    emb = model.get_text_features(input_ids=enc["input_ids"]).float()
    return F.normalize(emb, dim=-1).cpu()


_LOGIT_CACHE = {}


def logit_params(device=None, dtype=None):
    """SigLIP ``(logit_scale, logit_bias)`` in float32: P(text matches image) = sigmoid(scale * cos + bias).

    Read from the checkpoint file (``exp(logit_scale)`` = 109.894, ``logit_bias`` = -15.9326), not from the loaded
    model, whose bf16 copies are rounded to 108.5 and -15.9375. ``device`` and ``dtype`` are ignored (kept for
    backward compatibility).
    """
    if "v" not in _LOGIT_CACHE:
        from safetensors import safe_open
        from transformers.utils import cached_file

        with safe_open(cached_file(MODEL_ID, "model.safetensors"), "pt") as f:
            scale = f.get_tensor("logit_scale").float().exp()
            bias = f.get_tensor("logit_bias").float()
        _LOGIT_CACHE["v"] = (float(scale.item()), float(bias.item()))
    return _LOGIT_CACHE["v"]


def extract(video_path, device="cuda:1", **cfg):
    """Extract ``siglip2`` features for one video.

    Returns ``(t_sec [T] float32, feat [T, 1152] float16 (L2-normalized), valid [T] bool, meta dict)``.
    """
    ignored = sorted(set(cfg) - set(DEFAULTS))  # tolerate generic runner kwargs; recorded in meta
    cfg = {**DEFAULTS, **{k: v for k, v in cfg.items() if k in DEFAULTS}}
    t_start = time.time()
    dtype = _DTYPES[cfg["dtype"]]
    model, _ = load_model(device, cfg["dtype"])
    dim = model.config.vision_config.hidden_size

    video = io.Video(video_path, num_threads=cfg["decode_threads"])
    ticks = io.tick_times(video.info.duration_sec)
    T = len(ticks)
    feat = np.zeros((T, dim), dtype=np.float16)
    valid = np.zeros(T, dtype=bool)

    def decode_chunks():
        """Yield (c0, c1, uint8 [C, 3, 384, 384] or None, ok [C]); decoding and resizing run in this thread."""
        for c0 in range(0, T, cfg["chunk_ticks"]):
            c1 = min(T, c0 + cfg["chunk_ticks"])
            frames, ok = decode_frames(video.vr, video.frame_index(ticks[c0:c1]))
            yield c0, c1, None if frames is None else resize_frames(frames), ok

    for c0, c1, x, ok in prefetch(decode_chunks()):
        if x is None:
            continue
        emb = embed_images(model, normalize(x, device, dtype), cfg["batch_size"])
        feat[c0:c1] = emb.cpu().numpy().astype(np.float16)
        valid[c0:c1] = ok
    feat[~valid] = 0

    meta = make_meta(video.info.fps, video.info.duration_sec, dim, cfg, int(T - valid.sum()), ignored,
                     time.time() - t_start)
    return ticks, feat, valid, meta


def make_meta(fps, duration_sec, dim, cfg, n_decode_errors, ignored, wall_sec):
    """``siglip2`` meta dict (shared by ``extract`` and ``prompts.extract_visual``); ``cfg`` is the effective config."""
    model = next(iter(_CACHE.values()))[0] if _CACHE else None
    return dict(
        extractor=EXTRACTOR,
        version=VERSION,
        model_id=MODEL_ID,
        model_revision=getattr(getattr(model, "config", None), "_commit_hash", None),
        window_sec=0.0,
        D=int(dim),
        input_size=_SIZE,
        resize="squash full frame to 384x384, torchvision uint8 bilinear antialias (== SiglipImageProcessor PIL)",
        l2_normalized=True,
        dtype=cfg["dtype"],
        cfg=dict(cfg),
        video_fps=float(fps),
        duration_sec=float(duration_sec),
        n_decode_errors=int(n_decode_errors),
        ignored_cfg=list(ignored),
        wall_sec=round(wall_sec, 3),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--video-id", help="save to FEATURE_ROOT/siglip2/<video-id>.npz (e.g. ucf/<stem>)")
    ap.add_argument("--check-causal", type=float, metavar="CUT_SEC",
                    help="run io.check_causal with the video truncated at CUT_SEC")
    ap.add_argument("--check-processor", action="store_true",
                    help="only compare preprocessing with SiglipImageProcessor (CPU) and exit; status 1 if not exact")
    ap.add_argument("--dtype", default=DEFAULTS["dtype"], choices=sorted(_DTYPES))
    args = ap.parse_args()
    if args.check_processor:
        res = check_processor(args.video)
        print("check_processor", json.dumps(res))
        sys.exit(0 if res["exact"] else 1)
    cfg = dict(dtype=args.dtype)

    load_model(args.device, args.dtype)
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
