# Feature extractors (anticipation/features/)

Per-video, per-second, strictly causal features for the VA30 benchmark ([BASELINE.md](BASELINE.md) section 6).
Every stream (pose, motion, visual backbones, audio, speech) is written on the same 1 Hz tick grid, so a model
can stack them by row. This file has one overview section (contract, runner, conventions; owned by the pose
track) and one section per extractor family, each written by the track that built it.

Sections:
- [0. Overview: contract, runner, registry](#0-overview-contract-runner-registry)
- [Visual backbones (V-JEPA 2, SigLIP 2, prompt concept scores)](#visual-backbones-v-jepa-2-siglip-2-prompt-concept-scores)
- [Audio (sound events, CLAP prompts, voice affect, DSP)](#audio-sound-events-clap-prompts-voice-affect-dsp)
- [Speech content (Whisper transcripts, toxicity, verbal-escalation cues)](#speech-content-whisper-transcripts-toxicity-verbal-escalation-cues)
- [Pose and interaction (pose_raw, pose_interact)](#pose-and-interaction-pose_raw-pose_interact)
- [Motion (motion)](#motion-motion)
- [Bulk extraction run v1 (manifest, queue, lanes)](#bulk-extraction-run-v1)

New tracks: add a `## <Family> (<extractor names>)` section at the end, add it to this list, and register the
extractor in `run_extract.REGISTRY` (section 0.2).

---

## 0. Overview: contract, runner, registry

Written 2026-10-14 by the pose track. Code: `anticipation/features/io.py` (contract, do not change without
telling every track), `anticipation/features/run_extract.py` (shared batch runner).

### 0.1 The feature file contract (`io.py`)

One `.npz` per (extractor, video) at `FEATURE_ROOT/<extractor>/<video_id>.npz`, with
`FEATURE_ROOT = data/features` and `video_id = <source>/<stem>`.

| Key | Type | Meaning |
|---|---|---|
| `t_sec` | float32 [T] | tick times 1, 2, ..., floor(duration) in video seconds (`io.tick_times`); T = 0 for videos under 1 s |
| `feat` | float16 [T, D] | features of tick t |
| `valid` | bool [T] | False where the stream is absent (no audio, no person, decode error); invalid rows are filled, not NaN |
| `meta` | JSON str | at least `extractor`, `version`, `model_id`, `window_sec`, `D`; by convention also `columns` |

- **Causality:** the feature at tick t may only use media from `[t - window_sec, t]` (VA30 allows `[0, t]`;
  trackers that carry identity from the start say so in `meta["history"]`). Never pool over the whole video.
- **Clock:** duration = `num_frames / fps` from `io.Video` (decord). A frame time maps to the last frame at or
  before it (`io.Video.frame_index`), so nothing peeks into the future. Audio extractors use the same ticks.
- **Verification:** `io.check_causal(fn, path, cut_sec)` re-extracts on an ffmpeg stream-copy truncation and
  compares ticks `<= cut_sec - 1` (tolerance 1e-2). Each section below reports its own check results; most
  extractors match exactly (difference 0). The trick that makes GPU extractors exact: run fixed-size batches
  and pad the last batch by repeating its last item, so no feature depends on how long the video is.
- **Reading:** `t, feat, valid, meta = io.load_features("pose_interact", "xd/<stem>")` (feat comes back float32).
  Column names: `meta["columns"]`. Join streams on `t_sec` (all share the grid; T can differ by at most the
  last tick if a stream's decoder reports a slightly different duration, so join on `t_sec`, not row index).

Sources and ids follow `benchmark/videos.csv` (built by the benchmark track; 6,720 videos, 352.4 h in the
version of 2026-10-14 18:16: xd 4,749 / 215.9 h, ucf 1,879 / 126.6 h, yt 92 / 10.0 h; 4,751 videos are 24 fps,
1,894 are 30 fps, the rest 23.976 to 29.97 fps): `xd/<stem>`, `ucf/<stem>`, `yt/<youtube_id>` for the curated
YouTube set. Some smoke-test
files written by the visual and audio tracks use `curated/<id>` instead of `yt/<id>`; bulk runs from the
manifest use `yt/`.

### 0.2 The batch runner (`run_extract.py`)

```bash
# list registered extractors
.venv/bin/python -m anticipation.features.run_extract --list
# one shard of the manifest (resumable: existing outputs are skipped)
.venv/bin/python -m anticipation.features.run_extract --extractor pose \
    --manifest benchmark/videos.csv [--sources xd,ucf,yt] [--video_ids xd/<stem> ...] \
    --shard 0 --num_shards 6 --device cuda:0
# explicit files (source inferred from the path) or a list file; config overrides as key=value (JSON values)
.venv/bin/python -m anticipation.features.run_extract --extractor motion --video_paths a.mp4 b.mp4 --cpu_threads 1
.venv/bin/python -m anticipation.features.run_extract --extractor pose --video_list paths.txt --cfg weights=yolo11l-pose.pt
# smoke tests into a scratch root (logs go to <root>/_logs)
.venv/bin/python -m anticipation.features.run_extract --extractor pose --video_paths a.mp4 --feature_root /tmp/x
```

Behaviour:
- **Sharding:** a video belongs to shard `crc32(video_id) % num_shards`. This is stable when the manifest grows,
  so shards never overlap across restarts. It is unbalanced for tiny lists (8/7/3 for 18 videos), fine for
  thousands.
- **Skip / resume:** a video is skipped when all of the extractor's outputs exist. Outputs are written to
  `*.tmp.npz` and renamed, so a killed run never leaves a partial file. `--overwrite` recomputes everything
  (and recomputes it again after a restart, so it is not resumable).
- **Version check (opt-in per module):** a module that declares `OUTPUT_VERSIONS = {output: version}` (pose,
  motion) also has the `meta["version"]` of every existing output compared with it. A mismatching or
  unreadable file counts as missing and is recomputed (`stale=N` in the header line). So after a code change:
  bump `VERSION`, re-run the same command, and only stale files are redone; a crash in the middle resumes.
  `--no_version_check` restores existence-only skipping. Modules without `OUTPUT_VERSIONS` (currently the
  visual, audio and speech tracks) are checked for existence only; add the dict to opt in.
- **Errors exit with status 1 and a message:** a missing `--manifest` / `--video_list`, no input at all, an
  empty manifest, or a selection that is empty after `--sources` / `--video_ids` (for example the typo
  `--sources XD`; unknown sources and ids are also warned about). A detached bulk launch with a typo therefore
  fails visibly instead of finishing "successfully" with nothing done. An empty shard of a non-empty selection
  is fine (exit 0). `--selftest` checks these cases and the version logic.
- **Failures never stop the run.** Each failing video is logged with its traceback, and the loop continues.
  Logs live in `data/logs/extract/<extractor>/`:
  `shard<i>of<n>.jsonl` (one record per video: status, wall_sec, duration_sec, paths) and `failures.jsonl`
  (all shards, file-locked appends). `--skip_failed` skips ids already in `failures.jsonl`.
- `--limit N` (smoke tests), `--dry_run` (print the shard's work list), `--cpu_threads k` (caps OpenCV and
  torch threads; use 1 for many parallel CPU shards).
- **Order:** videos are processed sorted by `video_id` unless `--keep_order` is given; then each shard keeps
  the manifest's row order (the bulk run uses this to finish the priority videos first).
- **Native crashes:** before each video the runner writes its id to `shard<i>of<n>.current` in the log
  directory and removes it afterwards. If a run dies inside a video (segfault, SIGKILL, OOM killer, stall
  watchdog), the next run of the same shard finds the file, logs that video to `failures.jsonl` with
  `status: "crash"`, and skips it. Every later run also skips every video with a crash record, so a shard
  with several always-crashing videos still gets past all of them. `--retry_crashed` retries them (a video
  that crashes again is recorded again). SIGTERM is turned into a clean stop (the marker is removed, nothing
  is skipped).
- **Systemic failures:** after `--max_consecutive_fail` failures in a row (default 20, 0 = never) the runner
  exits with status 3 instead of failing every remaining video (a model that cannot load, GPU memory held by
  another job, a missing mount). Re-running retries those videos; they are `fail`, not `crash`, records.
- Manifest columns used: `video_id` (wins), `path`, `source`. Alternative names are also accepted (see the module
  docstring). Relative paths resolve against the data root, then the repo root.

Registry (`run_extract.REGISTRY`; `--extractor <name>`):

| Name | Module | Writes | Default device |
|---|---|---|---|
| `pose` | `pose.py` (`extract_and_save`) | `pose_raw`, `pose_interact` | cuda:0 |
| `pose_interact` | `pose.py` (`derive_and_save`) | `pose_interact` re-derived from existing `pose_raw` (CPU, about 200x realtime) | cpu |
| `motion` | `motion.py` (`extract`) | `motion` | cpu |
| `visual` | `prompts.py` (`extract_and_save`) | `vjepa2_l`, `siglip2`, `siglip2_prompts` (one decode) | cuda:0 |
| `vjepa2_l`, `siglip2`, `siglip2_prompts` | `vjepa.py`, `siglip.py`, `prompts.py` | one each | cuda:0 |
| `audio_events` | `audio_events.py` (`extract_and_save`) | `ast_events`, `ast_emb`, `clap_prompts` | cuda:0 |
| `voice` | `voice.py` (`extract_and_save`) | `voice_affect`, `e2v_emb` | cuda:0 |
| `audio_dsp` | `audio_dsp.py` (`extract`) | `audio_dsp` | cpu |
| `speech_tox` | `speech.py` (`extract`, with `transcript_path=speech.transcript_path(video_id)`) | `speech_tox` + transcripts JSON | cuda:0 |

Adding an extractor: write a module with `EXTRACTOR = "<name>"`, `VERSION`, `DEFAULTS` and
`extract(video_path, device="cuda:0", **cfg) -> (t_sec, feat, valid, meta)`, then add one line
`"<name>": ExtractorSpec("anticipation.features.<module>")`. A module that writes several outputs from one
pass instead exposes `extract_and_save(video_path, video_id, device, **cfg) -> {"paths": {...},
"duration_sec": s}` and is registered with `outputs=(...)` and `saver="extract_and_save"`. Per-video keyword
arguments derived from the id use `video_kwargs=(("kwarg", "module_function"),)`. An unregistered dotted
module path also works (`--extractor anticipation.features.foo`).

### 0.3 Bulk-run recipe (manual)

The bulk run of 2026-10-14 uses the queue in [Bulk extraction run v1](#bulk-extraction-run-v1) instead of the
commands below; they remain valid for one-off runs.

Long runs must be detached, logged and resumable; re-running the same command after a crash continues.
Expected wall time for the current manifest (352 h), measured on a shared machine: pose about 5.2 h (68x
aggregate with 6 processes on 2 GPUs), motion about 3.7 h (94x with 8 CPU processes). The runner has no
per-video timeout: a native crash or hang in decord / ffmpeg kills or stalls that shard without a
`failures.jsonl` entry, so check the shard logs and re-run the same command (it resumes; `--skip_failed`
skips videos that raised a Python error).

```bash
cd .
L=data/logs; mkdir -p $L
# pose: 3 processes per GPU (CPU-side tracking is the bottleneck, see the pose section)
for i in 0 1 2 3 4 5; do
  setsid nohup .venv/bin/python -m anticipation.features.run_extract --extractor pose \
    --manifest benchmark/videos.csv --shard $i --num_shards 6 --device cuda:$((i % 2)) \
    > $L/pose_shard$i.log 2>&1 &
done
# motion: CPU only, 8 single-threaded shards (better run after pose; both compete for CPU)
for i in 0 1 2 3 4 5 6 7; do
  setsid nohup .venv/bin/python -m anticipation.features.run_extract --extractor motion \
    --manifest benchmark/videos.csv --shard $i --num_shards 8 --cpu_threads 1 \
    > $L/motion_shard$i.log 2>&1 &
done
```

### 0.4 Global gotchas

- **Import torch before decord is first used.** If a process opens a `decord.VideoReader` before
  `import torch`, the later CUDA initialisation segfaults (reproduced; it is decord's bundled libraries).
  `pose.py`, `speech.py` and the runner (for `--device cuda:*`) import torch first. Do the same in any
  script that mixes `io.Video` with CUDA.
- `io.check_causal` truncates with `ffmpeg -c copy`. This keeps the decoded frames before the cut
  bit-identical (verified on XD, UCF, and YouTube mp4), so an exact extractor shows 0.0.
- `--feature_root` rebinds `io.FEATURE_ROOT` inside the runner process only. Use it for smoke tests so
  experiments never leave stale files in the real root (the runner would skip them later).
- Several tracks fill invalid rows with zeros or fixed fill values. Always pass `valid` to the model as a
  presence mask rather than relying on the fill value.
- **Rates need the real frame gap.** Sampling 5 fps from a 24 fps video lands on frame steps 4, 5, 5, 5, 5
  (0.167 s or 0.208 s, not 0.2 s); 23.976 and 29.97 fps behave the same way. Divide motions by
  `(frame_idx[k] - frame_idx[k-1]) / fps` (`motion.frame_dt`), never by `1 / sample_fps`.
- **Decode errors must stay local in time.** decord raises for a whole `get_batch` call when one frame in it
  is bad. Re-decode that chunk frame by frame (`motion.decode_frames`, `vjepa.decode_frames`); dropping the
  whole chunk invalidates ticks before the bad frame, which breaks causality. decord also cannot open some
  damaged files at all (for example a truncated faststart mp4: `av_read_frame failed` in `io.Video`); those
  are logged as failures.

---

## Visual backbones (V-JEPA 2, SigLIP 2, prompt concept scores)

Written 2026-10-14 by the visual-backbone track; revised the same day after review (all three extractors are
now version 2, see Gotchas). Code: `anticipation/features/vjepa.py`,
`anticipation/features/siglip.py`, `anticipation/features/prompts.py`. All three follow the `io.py` contract
(one npz per video at `FEATURE_ROOT/<extractor>/<source>/<stem>.npz`; `t_sec` = 1..floor(duration);
`feat` float16; `valid` bool; JSON `meta`).

### What is stored

| Extractor | Module | D | Window | Content |
|---|---|---|---|---|
| `vjepa2_l` | `vjepa.py` | 2048 | 4 s | V-JEPA 2 ViT-L (`facebook/vjepa2-vitl-fpc64-256`) on a 16-frame clip ending at t. Columns `0:1024` = mean of last-layer tokens (after the final LayerNorm); `1024:2048` = mean of block-12 (of 24) tokens, each token LayerNorm-ed without affine first. Layout in `meta["column_groups"]`; `vjepa.split_columns(feat, meta)` splits it. |
| `siglip2` | `siglip.py` | 1152 | 0 s | SigLIP 2 So400m/14 at 384 px (`google/siglip2-so400m-patch14-384`), image embedding of the frame at t, L2-normalized. |
| `siglip2_prompts` | `prompts.py` | 58 | 0 s | Cosine similarity between the `siglip2` embedding and each of 58 text prompts. Column names in `meta["columns"]`; the prompt list, version and hash are in `meta`. |

Sampling and causality:
- `vjepa2_l`: the 16 frame times for tick t are `t - 3.75, t - 3.5, ..., t` (4 fps over `[t - 4, t]`). Each time
  maps to the last frame at or before it (`io.Video.frame_index`). For t < 4 the times before 0 map to
  frame 0, so the first ticks repeat the first frame. That is still causal.
- `siglip2` / `siglip2_prompts`: the single frame at t (the same frame as the last frame of the V-JEPA clip).
- `io.check_causal` passes for all three on all four smoke videos (cuts at 51.7 to 60.5 s; version 2 re-run).
  The difference is exactly 0.0 for `vjepa2_l` and `siglip2`, and at most 1.5e-5 for the prompts (float32
  matmul order). Exactness comes from fixed-shape GPU batches: the last batch is padded by repeating its
  last item. So a clip's feature never depends on how many ticks the video has, and re-runs are bitwise
  reproducible.

Preprocessing reproduces each checkpoint's HF processor exactly (version 2, 2026-10-14 fix):
- The resize runs on the CPU in the background decode thread with torchvision's uint8 bilinear antialiased
  kernel, so the resized frames are uint8, as in the processors. Only the normalization runs on the GPU, with the
  processors' own float32 arithmetic.
- V-JEPA 2: short side to 292, long side `int(292 * long / short)`, center crop 256 at offsets
  `int((size - 256) / 2)` (HF truncates), then `(x - 255 mean) / (255 std)` with ImageNet mean/std. The float32
  pixels are bitwise identical to `VJEPA2VideoProcessor` (same torchvision call).
- SigLIP 2: squash the full frame to 384 x 384 (no crop), then `float32(x / 255)`, mean 0.5 / std 0.5. torchvision's
  uint8 kernel is a port of Pillow's resampler. The pixels are always identical to `SiglipImageProcessorFast`.
  They are bitwise identical to the default (PIL) `SiglipImageProcessor` on the checked real videos at
  320x240, 640x360, 360x640 and 640x346. Some sizes differ from PIL by one intensity level on a few pixels:
  1e-5 of pixels on a real 202x360 video, about 1e-4 on noise images at 640x346 and 640x266, and 1 to 4 % at
  640x480, 720x576 or 854x480. The local data is 640 px wide or less and mostly 640x360 or 320x240.
- Check on any video (CPU only, no model; exit status 1 unless exact):
  `python -m anticipation.features.vjepa VIDEO --check-processor` (and the same for `siglip`).
- Version 1 resized in float32 on the GPU without rounding to uint8, and its V-JEPA crop was 1 px off on
  640x360. On upsampled 320x240 UCF video this moved `vjepa2_l` by about half the across-clip spread and lowered
  the SigLIP cosine to the processor to a median of 0.965. Version 1 files are not comparable with version 2:
  check `meta["version"]`.
- Models run in bf16 with SDPA attention. Pooling and normalization run in float32.

### How to run

```bash
# one video, print shapes and speed (writes nothing without --video-id)
.venv/bin/python -m anticipation.features.vjepa   VIDEO.mp4 --device cuda:1 [--check-causal 60]
.venv/bin/python -m anticipation.features.siglip  VIDEO.mp4 --device cuda:1 [--check-causal 60]
.venv/bin/python -m anticipation.features.prompts VIDEO.mp4 --device cuda:1 [--check-causal 60]

# preprocessing parity with the HF processors (CPU only; exits 1 unless bitwise exact)
.venv/bin/python -m anticipation.features.vjepa  VIDEO.mp4 --check-processor
.venv/bin/python -m anticipation.features.siglip VIDEO.mp4 --check-processor

# bulk (registered in run_extract.py): all three from one decode
.venv/bin/python -m anticipation.features.run_extract --extractor visual --manifest benchmark/videos.csv \
    --shard 0 --num_shards 2 --device cuda:1

# all three visual extractors from ONE decode, saved under FEATURE_ROOT
.venv/bin/python -m anticipation.features.prompts VIDEO.mp4 --device cuda:1 --all-visual --video-id xd/<stem>

# prompt list; re-score every cached siglip2 file after editing PROMPTS (no video decoding, seconds)
.venv/bin/python -m anticipation.features.prompts --list
.venv/bin/python -m anticipation.features.prompts --rescore --device cuda:1 [--force]
```

Python and registry entry points for `run_extract.py`:

| Registry name | Call | Writes |
|---|---|---|
| `vjepa2_l` | `vjepa.extract(video_path, device, **cfg) -> (t_sec, feat, valid, meta)` | the caller saves |
| `siglip2` | `siglip.extract(video_path, device, **cfg)` | the caller saves |
| `siglip2_prompts` | `prompts.extract(video_path, device, **cfg)` | the caller saves |
| `visual` (recommended for bulk) | `prompts.extract_and_save(video_path, video_id, device, **cfg) -> {"paths": {...}, "duration_sec": s}` | `vjepa2_l`, `siglip2`, `siglip2_prompts` (same convention as `pose.extract_and_save`) |

`prompts.extract_visual(path, device)` returns the three outputs as a dict without saving. Its outputs are
bitwise identical to the single-extractor functions (verified on the four smoke videos). Each `extract`
ignores config keys it does not know and lists them in `meta["ignored_cfg"]`, so a runner can pass generic
kwargs. Models are cached per process: load once, then loop over videos. Config defaults are in each
module's `DEFAULTS`; the effective config is stored in `meta["cfg"]`.

Config routing in `extract_visual` / `extract_and_save` (registry `visual`; `prompts.split_visual_cfg`):
- V-JEPA-only keys (`window_sec`, `num_frames`, `mid_layer`, `crop`) go to V-JEPA.
- The decode keys `decode_threads` and `chunk_ticks` go to V-JEPA, which does the only decode (SigLIP embeds
  the tick frames V-JEPA already decoded).
- `dtype` and `batch_size` go to both models.
- `vjepa_cfg={...}` / `siglip_cfg={...}` set a key for one model only and win over flat keys. From the runner:
  `--cfg 'vjepa_cfg={"batch_size": 32}'` (values are JSON).
- Anything else, including SigLIP decode keys in `siglip_cfg`, is listed in `meta["ignored_cfg"]` of all
  three outputs.

Changing `window_sec`, `num_frames`, `mid_layer` or `crop` changes the features, so save the result under a new
extractor name. If the GPU loop raises mid-video, the background decode thread stops cleanly (no leaked
threads in a long-running runner).

### Prompt set (v1, 58 prompts, hash `5c8ea55e9fad`)

`prompts.PROMPTS` is a list of `(key, group, text)`:
- **escalation** (27): arguing face to face, pointing aggressively, shouting at someone, chest to chest,
  clenched fists, raised fist, walking aggressively toward someone, following someone, shoving, pushing
  away, grabbing clothes, being held back, someone separating two people, group standoff, crowd circle,
  surrounding a man, bystanders filming, road-rage driver argument, driver getting out, customer yelling at
  an employee, police confronting an agitated man, taking off a shirt to fight, knife, gun, bat.
- **violence_now** (8): punching, kicking someone on the ground, wrestling on the ground, brawl, being
  beaten, knocked down, police tackling, people running in panic.
- **calm** (12): calm street, calm talking, handshake, hug, queue, restaurant, empty store aisle, empty
  parking lot, traffic, office, sports, party.
- **context** (11): CCTV of a store, grainy surveillance footage, dashcam, bodycam, person filming on a
  phone, shaky phone video, hollywood movie scene, news broadcast, large text captions, outside a bar at
  night, nightclub. These are scene priors and **shortcut detectors**. If a model relies on them, suspect
  provenance learning (BASELINE.md 6.5).

To edit: change `PROMPTS`, bump `PROMPT_SET_VERSION`, and run `--rescore`. Never reuse a key for a new meaning,
because keys are column names. Text is lowercased and padded to 64 tokens, as SigLIP 2 was trained. A template
("a photo of {}.") was tried and changed the post-onset AUC by at most 0.03, so the raw sentences are used.
SigLIP is sigmoid-trained: `sigmoid(meta["logit_scale"] * cos + meta["logit_bias"])` (109.894, -15.9326, the
float32 checkpoint values; `siglip.logit_params` reads them from the safetensors file because the bf16 model
holds rounded copies, 108.5 / -15.9375, which version 1 files stored) is a per-prompt match probability.
Cosines are small (1st to 99th percentile -0.08 to 0.12 on the smoke videos), so feed them to models z-scored
per column.

### Why these models

- **V-JEPA 2 ViT-L**: the best frozen motion and action-anticipation encoder that runs in transformers 4.57
  (`VJEPA2Model`). It replaces VideoMAE-base. ViT-L is for ablations; `facebook/vjepa2-vitg-fpc64-256` would
  be the final model (about 3x the compute; see docs/research/backbones.md). The middle layer is cached
  because intermediate layers often probe better for motion. Its raw residual stream has outlier channels
  with |x| > 300, which is why each token is LayerNorm-ed before pooling.
- **SigLIP 2 instead of PE-Core-L14-336** (the research doc rates them near-equal and says to pick one):
  1. It loads natively in the installed transformers, with no new dependency. PE-Core needs
     `open_clip_torch`, which is not installed.
  2. Its sigmoid loss gives independent, calibrated per-prompt scores. That suits multi-label concept
     prompts better than softmax-relative CLIP scores.
  3. 384 px with no crop keeps peripheral and small people in 16:9 CCTV and phone video.
  A `pe_core.py` with the same interface could be added later under a new extractor name.

### Smoke test (2026-10-14, cuda:1; re-run for version 2 after the review fixes)

Videos, saved (version 2) under `FEATURE_ROOT/{vjepa2_l,siglip2,siglip2_prompts}/` with the canonical ids of
`benchmark/videos.csv`:

| video_id | Length | Resolution | Note |
|---|---|---|---|
| `xd/v=pL2HjXvWuPc__#1_label_B1-0-0` | 201.5 s | 640x360 | fight onset at 45.4 s (annotations.txt) |
| `ucf/Fighting003_x264` | 103.4 s | 320x240 | onset at 60.7 s |
| `ucf/Normal_Videos_641_x264` | 120.0 s | 320x240 | normal |
| `yt/holdout_004` | 253.9 s | 640x360 | convenience-store confrontation |

The first smoke run saved the curated video as `curated/holdout_004` (version 1). Those three files are stale and
not at a canonical id; nothing reads them, and they can be deleted.

Version 2 checks on all four videos. The scripts are not in the repo: `smoke_fix.py` and a proposed pytest file
`test_visual_features.py` (CPU only, 9 tests) are in the visual fixer's scratchpad
`<scratch>/fix_visual/`.
The in-repo check is `--check-processor`. Results:
- T = floor(duration), every tick valid, all three files `version` 2.
- `extract_visual` (one decode) is bitwise identical to `vjepa.extract`, `siglip.extract` and `prompts.extract`.
- Stored `siglip2` embeddings are bitwise identical to embeddings of the `SiglipImageProcessor` (PIL) pixels.
  The prompt-score correlation with that path is at least 0.99999 for every prompt (version 1 on UCF
  Fighting003: median 0.857, minimum 0.68).
- `vjepa2_l` `last_mean` vs the `VJEPA2VideoProcessor` pixels on 24 clips: mean L2 distance 0.011 (only bf16
  batch-mate noise; the pixels are identical). The across-clip spread is 8.4 to 15.4 (version 1: 4.06 on UCF
  and 1.84 on XD).
- `siglip.logit_params()` equals the float32 model's values (109.894, -15.9326).
- One simulated undecodable frame at 70 s (UCF Fighting003 and XD) invalidates only `vjepa2_l` ticks 70 to 73
  (the clips that contain it) and `siglip2` / prompts tick 70. All other ticks are bitwise unchanged
  (version 1: 39 and 103 of 103 ticks were lost).
- `extract_visual(..., decode_threads=8, foo=1)` reaches V-JEPA (`meta["cfg"]["decode_threads"] = 8`), and
  `foo` is listed in `ignored_cfg` of all three outputs.

Throughput (video-seconds per wall-second per GPU, decoding included, 679 s of video, version 1 code with the
machine idle):

| Path | Speed | 350 h of video on 1 GPU | on 2 GPUs |
|---|---|---|---|
| `vjepa2_l` alone | 53.6 | 6.5 h | 3.3 h |
| `siglip2` + prompts alone | 115.6 | 3.0 h | 1.5 h |
| both, separate decodes | 36.6 | 9.6 h | 4.8 h |
| both, `extract_visual` (one decode) | 39.6 | 8.8 h | 4.4 h |

- V-JEPA is GPU-bound. The encoder alone runs at about 68 clips/s (bf16, batch 16 to 32; one clip per
  video-second, about 1.6 TFLOP per clip).
- Decoding runs in a background thread (decord releases the GIL) and costs about 100 to 180 video-s/s at
  360p with 4 threads. All local sources are 640x360 or smaller.
- With the machine loaded (load average 30 on 16 cores, other tracks running), the same runs measured about
  half these speeds (`vjepa2_l` 26, `siglip2` 54, `extract_visual` 38 video-s/s). Budget for that.
- Version 2 moves the resize to the CPU decode thread. That costs about 0.7 ms per 360p frame and
  about 3.5 ms per video-second. An A/B run in one process at load average 39 measured `extract_visual`
  30.9 video-s/s for version 2 and 29.1 for the version 1 GPU resize, so there is no measurable cost. The version
  2 smoke run measured 21.5 (`extract_visual`), 23.0 (`vjepa2_l`) and 37.3 (`siglip2` + prompts) video-s/s at load
  average 40 to 100. Those numbers reflect the CPU contention, not the code.
- Storage: about 4 KB per video-second for `vjepa2_l` (about 5 GB for 350 h), 2 KB for `siglip2` (2.6 GB)
  and 0.1 KB for prompts.

Prompt scores around onsets (version 2; mean cosine over onset-45 to -35 s, onset-25 to 0 s, and onset to +30 s):
- XD fight (onset 45.4 s): `crowd_circle` -0.008, 0.027, 0.022 and `surround_person` -0.015, 0.016, 0.006
  (both rise in the 25 s before onset). `brawl` 0.031, 0.048, 0.080 and `punching` 0.008, 0.017, 0.044 rise
  after onset, and `kick_ground` goes from -0.058 just before onset to 0.009. The violence_now group mean
  separates post-onset ticks (first 30 s) from pre-onset ticks with AUC 0.96.
- UCF Fighting003 (320x240 grainy CCTV): only a weak post-onset rise (`kick_ground` 0.054, 0.058, 0.070;
  `brawl` 0.104, 0.106, 0.111; AUC 0.57). At this resolution SigLIP barely resolves the people. Expect pose and
  V-JEPA to carry UCF.
- V-JEPA drift (`last_mean` z-scored per dimension by the video's sd, RMS distance to the mean of the first
  20 s) jumps at the UCF onset: about 1.13 to 1.15 before it, 1.46 in the 14 s after it.
- On UCF Normal_Videos_641 the violence_now group mean is flat (0.057, 0.051, 0.056 over the three thirds),
  and its top prompt is `cctv_store`. The top prompt of `yt/holdout_004` is `filming_bystanders`.

### Gotchas

- `vjepa2_l` mean-pooled vectors share a large constant component. The `last_mean` norm is about 53 on every
  video (52.5 to 53.9), and the largest `last_mean` |value| is 22.5 to 26.6. Standardize per dimension using
  training-fold statistics before a head. Use the `last_mean` and `mid_mean` blocks separately or
  concatenated, but never pool them together.
- `mid_mean` dimension 407 (feat column 1431) is a massive-activation channel that survives the per-token
  LayerNorm. It is nearly constant: mean -29.4 to -29.7, per-video sd 0.019 on UCF / YouTube and 0.10 on XD. It
  holds the largest |value| in the file (29.6 to 29.7). Its float16 step is 0.0156, so storage noise is about
  24 % of its per-video sd on UCF and 4 % on XD. Every other column is below 1.1 %. Do not let a
  standardized column 1431 carry weight; dropping it is reasonable.
- Center crop: for 16:9 video V-JEPA sees only the central 49 % of the width (and 88 % of the height), because
  that is how the checkpoint was evaluated. `crop="full"` squashes the whole frame instead. It is not the default; if it is
  used, save it under a different extractor name (for example `vjepa2_l_full`). SigLIP always sees the
  full frame.
- `meta["wall_sec"]` includes model loading for the first video of a process.
- Early ticks (t < 4) of `vjepa2_l` repeat frame 0. Videos shorter than 1 s give T = 0 (empty arrays).
- `valid` is False only where frames failed to decode. For `vjepa2_l` that means any frame of the tick's 4 s
  clip; for `siglip2` / prompts, the frame at the tick. A failed chunk decode falls back to one frame at a
  time (`vjepa.decode_frames`), so one bad frame costs at most about 4 `vjepa2_l` ticks and 1 `siglip2` tick.
  `meta["n_decode_errors"]` counts invalid ticks and `meta["n_bad_frames"]` (`vjepa2_l`) counts undecodable
  frames. Invalid rows are zero. Visual features have no other absence mask.
- Versions: `vjepa2_l` 2, `siglip2` 2 and `siglip2_prompts` 2 (2026-10-14). Version 1 files used a float
  GPU resize (see Preprocessing) and bf16-rounded logit constants; they must not be mixed with version 2. The
  runner skips existing outputs, so pass `--overwrite` (or delete version-1 files) before a bulk run.
- The weights are in the HF cache (`~/.cache/huggingface/hub`): V-JEPA 2-L 1.3 GB, SigLIP 2 4.5 GB.
- `prompts.rescore_all` rewrites a prompts file when it is missing or stale: a different `prompt_set_hash`,
  prompts `VERSION`, or `backbone_version` than the `siglip2` file it is computed from. `--force` rewrites all
  of them. It warns about `siglip2` files older than `siglip.VERSION`, because those must be re-extracted from
  the video.

---

## Audio (sound events, CLAP prompts, voice affect, DSP)

Written 2026-10-14 by the audio track. Code: `anticipation/features/audio_events.py`,
`anticipation/features/voice.py`, `anticipation/features/audio_dsp.py`. All six outputs follow the `io.py`
contract (one npz per video at `FEATURE_ROOT/<extractor>/<source>/<stem>.npz`; `t_sec` = 1..floor(duration)
with duration = frames / fps from `io.Video`, so the audio ticks are the same ticks as the video extractors;
`feat` float16; `valid` bool; JSON `meta` with `columns`). This section covers only non-speech-content audio.
Transcripts and toxicity are in the speech track.

Current version: **1.1** for all six outputs (`meta["version"]`), from the review fix pass on 2026-10-14. It
adds A/V start alignment, near-silence validity, bit-exact truncation invariance on GPU and a safer duration
fallback. Nothing had been bulk-extracted at 1.0, so no 1.0 files should exist under the real
`FEATURE_ROOT`. If any do, recompute them with `--overwrite`.

### What is stored

| Extractor | Module | D | Window W | Content |
|---|---|---|---|---|
| `ast_events` | `audio_events.py` | 53 | 3 s | AST (`MIT/ast-finetuned-audioset-10-10-0.4593`, AudioSet mAP 0.459) **logits** of 53 curated AudioSet classes. Exact AudioSet names in `meta["columns"]` and `audio_events.CURATED_CLASSES`. `classes="all"` keeps all 527 (save under a new name). |
| `ast_emb` | `audio_events.py` | 768 | 3 s | AST pooled embedding (mean of CLS and distillation tokens, before the classifier LayerNorm), same forward pass as `ast_events`. |
| `clap_prompts` | `audio_events.py` | 532 | 5 s | LAION CLAP (`laion/larger_clap_general`). Columns 0..19 = cosine(audio, prompt) for the 20 prompts in `audio_events.CLAP_PROMPTS` (`meta["n_prompts"]` = 20). Columns 20..531 = the L2-normalised 512-d CLAP audio embedding. |
| `voice_affect` | `voice.py` | 28 | 4 s | Silero VAD (`vad_mean`, `vad_max`, `speech_frac`, `speech_gate`). audEERING `wav2vec2-large-robust-12-ft-emotion-msp-dim` arousal / dominance / valence (about 0..1), uniform-pooled and speech-weighted (`*_sp`). emotion2vec+ large 9-class posteriors (`e2v_<class>`, `e2v_sp_<class>`; classes angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown). |
| `e2v_emb` | `voice.py` | 1024 | 4 s | emotion2vec+ large utterance embedding (uniform mean of frame features) of the window. |
| `audio_dsp` | `audio_dsp.py` | 34 | 3 s (+ slopes up to 30 s) | Hand-crafted DSP, CPU only (column list below). |

Curated AST classes (53): Speech, Male speech, Female speech, Child speech, Conversation, Narration, Babbling,
Whispering; Shout, Bellow, Yell, Battle cry, Children shouting, Screaming, Crying/sobbing, Whimper, Wail/moan,
Groan, Grunt, Gasp, Pant, Laughter; Crowd, Hubbub, Chatter, Cheering, Applause, Chant, Children playing; Run,
Slap/smack, Whack/thwack, Thump/thud, Smash/crash, Breaking, Glass, Shatter, Bang; Gunshot/gunfire, Explosion,
Siren, Police car (siren), Emergency vehicle, Alarm; Vehicle, Car, Vehicle horn, Traffic noise, Music,
Television, Dog, Wind noise (microphone), Silence. Music, Television, Wind noise and Silence are there as
confounder and provenance indicators (movie soundtracks, overdubbed YouTube music). Do not drop them.

CLAP prompts (20): people arguing loudly; a heated argument between two men; a man shouting angrily; a woman
screaming; a crowd yelling; people yelling threats and insults; an angry confrontation on the street; people
cheering and shouting at a fight; punches and slaps hitting a person; people scuffling and grunting; glass
breaking; a gunshot; a police siren; a car horn honking; calm conversation; people talking quietly; people
laughing; music playing; traffic noise on a quiet street; silence. Raw CLAP cosines are small (about -0.1 to
0.3). Use them relative to each other or z-scored, not as probabilities.

`audio_dsp` columns (34). Window statistics over the frames lying fully inside `[t - 3, t]` (32 ms frames,
10 ms hop, `center=False`):
- `rms_db_mean/max/std/p10`: per-frame RMS in dBFS (floor -100).
- `rel_loud_db`: window mean dB minus the median of all past 1-s block means of the same video (blocks
  `<= t`). This is the only column that uses the whole past `[0, t]`.
- `centroid_khz_mean/std`: frames above -60 dBFS only.
- `flux_db_mean/max`: mean over 40 mel bands of the positive log-mel difference to the previous frame.
- `zcr_mean`, `quiet_frac` (fraction of frames at or below -60 dBFS).
- `f0_mean_hz`, `f0_std_st`, `f0_range_st` (5 to 95 percentile, semitones), `voiced_frac`: pYIN over 75 to
  800 Hz, frame 64 ms, hop 20 ms, run separately on each 1-s block.
- `onset_rate`: flux peaks per second (peak picking inside each 1-s block; delta 2 dB, frame at least -50 dBFS).

Then `slope{5,15,30}_{loud_db, centroid_khz, flux_db, f0_st, voiced_frac, onset_rate}`: least-squares slope
(units per second) of the 1-s block series over the last 5, 15 and 30 blocks (NaN blocks ignored; 0 with
fewer than 3 points).

### Validity (`valid`) and missing audio

- No audio stream, decode failure or a fully digital-silent track (`io.is_digital_silence`, max |x| < 1e-4;
  almost all UCF-Crime tracks) gives `meta["audio_status"]` = `no_audio` or `digital_silence`, `valid` all
  False and all-zero `feat`. UBI-Fights has no audio, so it also comes out all invalid.
- Otherwise a tick is invalid when its window ends more than 0.25 s past the decoded audio, when the window
  is digital silence (peak < 1e-4, muted stretches), or when it is **near-silent: window RMS below -65 dBFS**
  (`audio_dsp.NEAR_SILENCE_DB`). Invalid ticks are zero rows. The rule is in `audio_dsp.tick_valid` and uses
  only the window's own samples, so validity is causal.
- A track whose loudest 1-s block is below -65 dBFS gets `audio_status = "near_silence"`. With whole-second
  windows (all defaults) every tick of such a track is invalid, so for the VA30 breakdown it counts as
  audio-absent, like `digital_silence`. Treat `ok` as the only "audio present" status.
- Why -65 dBFS (added in v1.1 after review): encoder-noise tracks pass the peak test but are inaudible
  (UCF Robbery056 -88 dBFS, Explosion042 -75, Arrest033 -70 RMS). Before v1.1 they were valid on every tick,
  and `voice.py` normalises each window to unit variance, so the SER heads produced confident arousal values
  on silence (a dither file gave arousal 0.58 with `speech_frac` 0). In 130 random XD and curated videos,
  2.7 % of 3-s windows are below -80 dBFS (digital-silence stretches, already invalid before) and only
  0.03 % fall between -80 and -65 dBFS, so the threshold does not remove real quiet scenes.
- Early ticks (t < W) use the shorter window `[0, t]`. They are valid.
- `audio_dsp` 1-s block series (slopes, the `rel_loud_db` median, pitch) use the same rule per block, so
  near-silent blocks are missing (NaN, ignored) rather than measured.
- For the downstream head: use `valid` as the audio presence mask (VA30 breakdown "audio present versus
  absent"). In `voice_affect`, `speech_frac` and `speech_gate` say whether the SER columns describe actual
  speech. Without speech they describe music or noise.

### Audio clock, downmix and level (v1.1)

- **A/V start alignment.** `io.load_audio` returns samples from the audio stream's own start and ignores its
  `start_time`. If the audio stream starts later than the video, tick t would see audio from after t, and
  `io.check_causal` cannot notice because the cut file shifts the same way. `audio_dsp.load_track` therefore
  reads both streams' `start_time` with ffprobe (`probe_streams`) and re-aligns the samples to the video
  clock (`io.Video` time 0 = first video frame). Late audio (by more than 1 ms) is always padded with leading
  zeros. Early audio is trimmed only beyond 50 ms (`AV_EARLY_TOL_SEC`): XD's typical offset is -21 ms (AAC
  encoder delay, audio slightly early, which is the safe side), and a stream-copy cut changes it by about
  1 ms, so leaving it alone keeps those videos exact under `check_causal`. The applied shift is
  `ceil(offset / 10 ms) * 10 ms`, so the aligned audio lags the video by 0 to 10 ms and never leads.
  `meta["av_offset_sec"]` (measured) and `meta["av_shift_sec"]` (applied) record it. The `--selftest` of `audio_dsp` checks a 0.5 s late audio stream: a burst at 5.3 s lands at 4.82 s with
  plain `io.load_audio` (inside tick 5's window, a leak) and at 5.30 s after alignment.
- Offsets in the data (ffprobe over the whole manifest): all 92 yt videos are 0. XD spans -23 to +24 ms; only
  `xd/Mission.Impossible.II.2000__#00-45-41_00-48-20_label_A` is late (+24 ms, now shifted 30 ms). In UCF,
  36 videos are late by 23 to 69 ms (shifted; almost all of them are digital silence anyway). A
  stream-copy cut keeps these offsets exactly, and `check_causal` on the two late cases (XD +24 ms, UCF
  Abuse026 +69 ms) is 0.0.
- **Downmix (unchanged, by decision).** `io.load_audio` uses ffmpeg's default `-ac 1` matrix: stereo
  0.707 (L + R); 5.1 FC + 0.707 (FL + FR) + 0.5 (BL + BR), LFE dropped. It is unnormalised, so float samples
  can exceed 1.0 (harmless, nothing clips in float), and a phase-inverted stereo pair (L = -R) cancels to
  silence (theoretical; never seen in the data). The review proposed `aresample=rematrix_maxval=1`. We
  measured it and kept the default, because the default is the one that is level-consistent across layouts:
  the same centre-panned "dialogue" decodes to -28.1 dBFS from a stereo mix and -28.1 from a 5.1 mix with
  the default, but to -31.1 and -38.7 dBFS with `rematrix_maxval=1` (a 7.6 dB layout offset). On real data
  the layout and source offset exists either way (median 1-s block level, default / normalised: XD 5.1
  -29.0 / -39.7, XD stereo -17.7 / -20.7, curated stereo -21.9 / -24.9 dBFS); normalising widens it.
- **Consequences for models.** Absolute levels (AST/CLAP log-mel levels, `rms_db_*`) carry a source and
  layout signature (5.1 movies are about 11 dB quieter than stereo XD compilations). For VA30 prefer the
  within-video columns (`rel_loud_db`, the slopes), stratify audits by `meta["audio_channels"]`, and report
  the curated set separately. `meta` also stores `audio_layout`, `audio_rms_dbfs` and
  `audio_loudest_block_dbfs`; these summarise the whole track (provenance and audits only, never features).

### Causality

Every column at tick t is a pure function of the audio in `[max(0, t - W), t]`, except `audio_dsp` slopes
(look-back 30 s) and `rel_loud_db` (whole past). Nothing looks past t. Details:
- AST: the Kaldi fbank (25 ms frames, 10 ms hop, per-frame processing only) is computed once per video on the
  GPU. The frames lying fully inside the window are then sliced and zero-padded to 1024 frames, exactly as
  `ASTFeatureExtractor` pads short clips (verified: max |diff| 5e-5 versus the HF extractor).
- CLAP: the window is repeat-padded to 10 s, then a GPU log-mel equal to `ClapFeatureExtractor`
  (`repeatpad`, slaney mel) is computed. Embedding cosine versus the HF CPU path: 1.000000.
- Voice: Silero VAD is run per window with its recurrent state reset at the window start, so even the VAD
  sees only `[t - W, t]`. Every window is normalised on its own (zero mean, unit variance), as each model's
  extractor does per utterance.
- DSP: pYIN's Viterbi decoding and onset peak picking look ahead within their input, so they run
  independently on each 1-s block `[k - 1, k]`. Windows aggregate whole blocks.
- Validity (including near-silence) is decided per window; the audio is aligned to the video clock before
  any window is cut (see "Audio clock" above).
- **Bit-exact under truncation (v1.1).** A tick's value must not depend on how long the video is. Two GPU
  effects broke this in v1.0, and both are now fixed:
  - `ast_fbank` ran one GPU fbank per 300 s chunk, and the last chunk's length depended on the track length.
    cuFFT then changed frames by about 1e-6, which the fp16 AST pass amplified to 0.011, so `ast_emb` failed
    `io.check_causal` (tolerance 1e-2) on The.Hurt.Locker at cuts 61.5, 64.5, 66.5 and 70.2 s. Every chunk is
    now zero-padded to exactly 300 s + 240 samples, and the extra frames are cut off. Frames are independent,
    so values are unchanged.
  - The last GPU batch of a truncated video was smaller. In fp16, AST with batch 1 versus 64 differs by
    0.011, and the sliced CLAP similarity matmul differed by 1 ulp. Every batch of full-length windows
    (AST, CLAP, voice) is now padded to `batch_size` rows by repeating the last index
    (`audio_dsp.pad_indices`), and outputs are sliced only after the last matmul. Early ticks with shorter
    windows are batches of one in every run. Cost: about +18 % GPU work for AST and voice and +10 % for CLAP
    on the xd + yt manifest (mean 168 ticks per video).
  - `--selftest` in `audio_events` and `voice` checks this on synthetic audio with no video needed. The fbank
    of a prefix must equal the prefix of the fbank bit for bit, and AST/CLAP/voice on a 200 s track must
    equal its 66 or 71 s truncation exactly, fp16 included. Both self-tests fail with the v1.0 code: fbank
    prefix unequal at cuts 70.2 and 150 s; without padding, AST differs by 0.0078, CLAP by 1.3e-4 and
    e2v_emb by 2.6e-4.
- The resampler in ffmpeg (`io.load_audio`) has a lookahead of a few milliseconds. This is negligible, but
  strictly speaking the last few ms before t can depend on samples just after t.

### How to run

```bash
# one video, print shapes / top classes / time series (writes nothing); optional causality check
.venv/bin/python -m anticipation.features.audio_events VIDEO.mp4 --device cuda:0 [--which ast_events ast_emb clap_prompts] [--causal-cut 60]
.venv/bin/python -m anticipation.features.voice        VIDEO.mp4 --device cuda:0 [--window 4] [--causal-cut 60]
.venv/bin/python -m anticipation.features.audio_dsp    VIDEO.mp4 [--window 3] [--causal-cut 60]
# self-tests (run after any change to these modules; each raises on failure)
.venv/bin/python -m anticipation.features.audio_dsp    --selftest                  # CPU, synthetic videos: A/V alignment, near-silence, validity windows, duration fallback (~3 s)
.venv/bin/python -m anticipation.features.audio_events --selftest --device cuda:0  # fbank + AST + CLAP truncation invariance (~30 s)
.venv/bin/python -m anticipation.features.voice        --selftest --device cuda:0  # VAD + A/V/D + emotion2vec+ truncation invariance
```

Python entry points (models are loaded once per process and device, then cached):

| Call | Returns / writes |
|---|---|
| `audio_events.extract(path, device, which="ast_events", **cfg)` | `(t_sec, feat, valid, meta)` for one of `ast_events`, `ast_emb`, `clap_prompts` |
| `audio_events.extract_all(path, device, which=OUTPUTS, **cfg)` | dict name -> tuple; one decode and one AST pass for events + emb |
| `audio_events.extract_and_save(path, video_id, device, **cfg)` | writes all three; returns `{"paths", "duration_sec"}` (runner saver convention) |
| `voice.extract(path, device, which="voice_affect", **cfg)` / `voice.extract_all` / `voice.extract_and_save` | same pattern for `voice_affect`, `e2v_emb` |
| `audio_dsp.extract(path, device="cpu", window_sec=3, **cfg)` | `audio_dsp` tuple (`EXTRACTOR = "audio_dsp"`) |

Unknown cfg keys are ignored and listed in `meta["ignored_cfg"]`. Config keys that change features:
`ast_window_sec` (3), `clap_window_sec` (5), `classes` ("curated" or "all"), voice `window_sec` (4), dsp
`window_sec` (3, whole seconds). Save non-default runs under a new extractor name. Other keys: `batch_size`,
`fp16` (default True; fp16 vs fp32 AST logits differ by at most 0.015).

Registry lines (already present in `run_extract.REGISTRY`, which the pose track owns):

```python
"audio_events": ExtractorSpec("anticipation.features.audio_events",
                              outputs=("ast_events", "ast_emb", "clap_prompts"), saver="extract_and_save"),
"voice": ExtractorSpec("anticipation.features.voice", outputs=("voice_affect", "e2v_emb"), saver="extract_and_save"),
"audio_dsp": ExtractorSpec("anticipation.features.audio_dsp", default_device="cpu"),
```

Bulk plan: `audio_events` and `voice` on GPU, 1 to 2 processes per GPU, restricted to sources with audio
(`--sources xd,yt`; the curated set is `yt` in the manifest; running UCF/UBI only writes all-invalid files
quickly, which is also fine and keeps the masks explicit). `audio_dsp` on CPU with several shards (`--device cpu --num_shards 8`), since pYIN is
single-threaded.

### Smoke test and throughput (2026-10-14, cuda:0)

Four videos. Outputs went to a scratch `FEATURE_ROOT`, round-tripped through `io.save_features` and
`io.load_features`, and the tick grid was checked against `io.tick_times`.

| Video | T | `audio_status` | valid | `check_causal` (cut) |
|---|---|---|---|---|
| `xd/v=FW_qbiPH7UA__#1_label_B1-0-0` (fight, onset 68.9 s) | 136 | ok | 1.00 | pass, all 6 (60 s) |
| `xd/v=1202rYvw7go__#1_label_A` (normal) | 164 | ok | 1.00 | pass, all 6 (60 s) |
| `curated/holdout_044` (bus altercation, YouTube) | 308 | ok | 1.00 | pass, all 6 (90 s) |
| `ucf/Fighting003_x264` | 103 | digital_silence | 0.00 | pass, all 6 (40 s) |

The smoke run also passed through `run_extract.py` (dotted-module path for `audio_dsp`) and through both
`extract_and_save` savers.

**Re-check after the v1.1 fix pass** (`final_check.py` in the fixer's scratchpad; log
`data/logs/audio_fix_check.log`). Eight videos, all six outputs,
round trip through `io` into a scratch root, tick grid equal to `io.tick_times`, no NaN, invalid rows all zero.
`io.check_causal` (fp16 defaults) was run at 9 cuts: The.Hurt.Locker (5.1, offset -21 ms) at 61.5 and 66.5 s,
curated holdout_004 at 97.3 s, the four videos above, UCF Robbery056 (near-silent) at 40 s, and a synthetic
40 s video with audio 0.5 s late at 20.5 s. **All 54 checks give max |diff| 0.0 with identical `valid`
masks** (v1.0: `ast_emb` 0.0112 = fail on The.Hurt.Locker; `ast_events` 0.0078; voice 0.001). The Hurt
Locker is `ok` with 6 channels. holdout_004 is `digital_silence`. Robbery056 is `near_silence`, so all six
outputs are invalid (v1.0: valid on every tick). Other UCF: Arrest049 (-93 dBFS), Explosion042 (-75) and
Arrest033 (-70) are `near_silence`. Explosion043, RoadAccidents039 and Abuse015 (-64 to -54 dBFS RMS, with
audible blocks) stay `ok`. The reviewer's dither file is `near_silence` with 0 valid `voice_affect` ticks.
Duration fallback on the XD Desperado clip with decord disabled: 67.04 s (container 1340.3 s), 67 ticks.

Warm throughput re-measured after v1.1 on the same 20.5 min track (machine less loaded, load average 3):
AST 422x, CLAP 863x and voice 211x realtime, DSP 14.6x per core, so v1.1 is no slower on long tracks.
Batch padding costs about +18 % (AST, voice) and +10 % (CLAP) on the real length distribution (see
Causality). All GPU stages together still need about 2 to 3 GPU-hours for 200 h of audio.

Reference checks against the official preprocessing:
- AST input versus `ASTFeatureExtractor`: max |diff| 5e-5.
- CLAP GPU mel versus `ClapFeatureExtractor`: max 2e-3 dB, embedding cosine 1.000000.
- audEERING A/V/D versus the model-card path (`Wav2Vec2FeatureExtractor` + mean pool + head): max |diff| 1e-4.
- emotion2vec+ posteriors versus `funasr` `generate()`: 7e-5 (fp32) and 5e-4 (fp16).

Warm throughput: models loaded, a 20.5 min track, one GPU, machine shared with other jobs (load average
about 11 on 16 cores).

| Stage | x realtime | Cost for 200 h of audio |
|---|---|---|
| ffmpeg decode, 16 kHz / 48 kHz (CPU) | 565 / 513 | 0.35 / 0.39 CPU-h |
| `ast_events` + `ast_emb` (fp16, batch 64) | 204 | 1.0 GPU-h |
| `clap_prompts` (fp16, batch 32) | 525 | 0.4 GPU-h |
| `voice_affect` + `e2v_emb` (VAD + wav2vec2 + emotion2vec+, fp16) | 135 | 1.5 GPU-h |
| `audio_dsp` (one CPU core; pYIN is about 90 % of it) | 15 | 13 core-h (about 1.6 h wall on 8 cores) |

All GPU stages need about 3 GPU-hours for 200 h of audio (about 1.5 h on 2 GPUs). Peak GPU memory is
9 GB. DSP is the long pole and is CPU-bound: shard it across cores.

### Sanity: do the cues rise before the onset?

Script (scratch, not committed): all 21 XD-Violence test clips labelled B1 (fighting) or B5 (abuse) with a
first onset at least 30 s in (onsets from `annotations.txt`). The features were aligned at the onset t0 and
each video was demeaned by its own mean over `[t0 - 60, t0 - 30)`. Below: the mean change per bin, and the
fraction of videos whose `[t0 - 10, t0)` mean is above baseline.

| Feature | [-30,-10) | [-10,0) | [0,10) (violence) | up in [-10,0) |
|---|---|---|---|---|
| AST Shout (logit) | -0.12 | +0.20 | +0.50 | 0.62 |
| AST Yell | +0.01 | +0.16 | +0.50 | 0.62 |
| AST Screaming | -0.14 | -0.01 | +0.49 | 0.62 |
| AST Crowd | -0.06 | +0.49 | +0.99 | 0.62 |
| AST Music | +0.09 | +0.26 | +0.19 | 0.57 |
| CLAP "a man shouting angrily" (cosine) | +0.001 | +0.007 | +0.008 | 0.62 |
| voice arousal | -0.02 | +0.02 | +0.10 | 0.62 |
| voice e2v_sp_angry | -0.05 | -0.06 | -0.02 | 0.38 |
| DSP rms_db_mean (dB) | +2.3 | +2.7 | +7.5 | 0.81 |
| DSP rel_loud_db (dB) | +1.6 | +1.5 | +5.6 | 0.71 |
| DSP f0_mean_hz | -10 | +1.6 | +35 | 0.52 |

Reading:
- The features work as detectors. At onset, shout, yell, scream and crowd logits rise by about 0.5 to 1.0,
  loudness by 7.5 dB, F0 by 35 Hz and arousal by 0.1.
- Before onset there is a modest rise of loudness from about 30 s out (+2 to 3 dB, 81 % of videos). There is
  also a rise of shout, yell and crowd in the last 10 s.
- Vocal arousal barely moves before onset. emotion2vec anger does not rise at all (it is unreliable on
  movie and soundtrack audio).
- On the single required clip `v=FW_qbiPH7UA` (a music-dominated YouTube compilation), nothing rises before
  onset: Yell goes from -8.4 to -6.0 and loudness from -30 to -15 dBFS only after 70 s.
- n = 21 movie and compilation clips with overdubbed music, and no significance test. Treat this as
  "the signals are wired correctly and respond to violence", not as evidence of 30 s anticipation. The
  real-world curated set (once hand-annotated) is the right place to test the leading-cue hypothesis.

### Why these models

- **AST** rather than BEATs / EAT / PE-A-Frame. It is a native `transformers` class with safetensors and no
  `trust_remote_code`, it is BSD-3, and it is the reference AudioSet tagger. EAT-large (mAP 0.495,
  `worstchan/EAT-large_epoch20_finetune_AS2M`) is the drop-in upgrade but needs `trust_remote_code`. PE-AV /
  PE-A-Frame need a newer transformers than 4.57. Revisit if AST logits prove too noisy.
- **CLAP** gives open-vocabulary "argument / shouting / confrontation" scores that AudioSet has no class for.
  It is cheap (280x realtime).
- **audEERING A/V/D** is the standard dimensional SER model and fairly robust to noise. **emotion2vec+ large**
  is the strongest open categorical SER model (anger posterior). Both are on the backbones.md shortlist.
  Rising arousal and anger are true leading cues.
- **DSP** is cheap, interpretable and domain-robust. Loudness relative to the video's own past and the
  15 to 30 s slopes are the most anticipation-shaped features in this track.

### Gotchas

- Weights live in the HF cache (`~/.cache/huggingface/hub`): AST 0.35 GB, audEERING 0.66 GB, CLAP 0.78 GB,
  emotion2vec+ 1.9 GB.
- Duration fallback: if decord cannot open a file, `audio_dsp.video_duration` uses the ffprobe video-stream
  duration, then the audio-stream duration, and only then `format=duration`. XD movie clips carry a
  chapter/data stream as long as the whole film, so the container duration is wrong for them (Desperado clip:
  1340.3 s container versus 67.04 s video). Using it would create about 1270 spurious all-invalid ticks.
  `speech.video_duration` (speech track) still has the container-only fallback.
- `io.load_audio` itself is unchanged (it is shared). The A/V alignment lives in `audio_dsp.load_track`, so
  any other module that calls `io.load_audio` directly (the speech track does) does not get it. On the
  current data, such a module sees at most 24 ms of future audio on one XD clip and up to 69 ms on 36 UCF
  clips, which are mostly silent.
- Track-level `meta` fields (`audio_rms_dbfs`, `audio_loudest_block_dbfs`, `audio_status`, `valid_frac`)
  summarise the whole video. Never feed them to a model as features.
- `laion/larger_clap_general` has only a pickle `pytorch_model.bin` on `main`. transformers 4.57 refuses
  `torch.load` with torch 2.5 (CVE-2025-32434 guard, needs torch >= 2.6). The code pins `revision="refs/pr/2"`
  (HF SFconvertbot safetensors conversion, commit 16c8cc31). AST and audEERING ship safetensors on `main`.
- The audEERING model needs its custom regression head (`voice._w2v_classes`, copied from the model card).
  Its output order is (arousal, dominance, valence) per `config.id2label`. The code reorders by name.
- `import silero_vad` calls `torch.set_num_threads(1)` process-wide. `voice._vad` restores the previous value.
  Silero runs on the GPU, batched over windows (CPU batched is about 50x slower).
- `funasr` reconfigures root logging at INFO on import and logs every loaded tensor. `voice._e2v` silences
  this while loading and restores the root level. emotion2vec+'s `model.pt` is a pickle loaded by FunASR
  (official `emotion2vec/emotion2vec_plus_large` repo). FunASR calls `snapshot_download` on every load
  (cached; set `HF_HUB_OFFLINE=1` if the network is down).
- pYIN at resolution coarser than 0.1 semitone loses most voiced frames (voiced fraction 0.45 -> 0.13 at
  0.25). The 75 Hz floor stops F0 locking onto 50/60 Hz hum and bass. F0 on music-heavy XD clips is music
  pitch, not voice. Combine it with `voice_affect.speech_frac`.
- XD-Violence clips are movie scenes and YouTube compilations, often with an overdubbed soundtrack (AST
  `Music` high during fights). Music cues are a provenance shortcut risk. Check the `Music` / `Television`
  columns in shortcut audits and report the curated real-world set separately.
- SER models were trained on clean single-speaker speech. On street audio they are noisy. The speech-weighted
  `*_sp` columns use weights `p_vad + 0.01`, so they fall back smoothly to the uniform mean when nobody speaks.
- All features are float16 on disk. Logits and embeddings are well inside its range. CLAP cosines lose about
  1e-4 of precision.

Packages installed for this track (pip, serialized with the shared lock): `silero-vad 6.2.3`, `funasr 1.4.16`
plus its dependencies (modelscope 1.40.2, hydra-core 1.3.7, omegaconf 2.3.1, kaldiio 2.18.1, sentencepiece
0.2.2, umap-learn 0.5.12, pynndescent 0.6.0, cryptography 50.0.2, oss2, aliyun-python-sdk-core/kms,
jieba, jamo, jaconv, tiktoken, RapidFuzz, tensorboardX, torch-complex, crcmod, pycryptodome, jmespath,
antlr4-python3-runtime 4.9.3). torch, transformers and numpy were not changed.

---

## Speech content (Whisper transcripts, toxicity, verbal-escalation cues)

Written 2026-10-14 by the speech track; **version 2** (same day) after an independent review (changes in
"Review fixes (v2)" below). Code: `anticipation/features/speech.py` (extractor `speech_tox`, registered in
`run_extract.py`). Follows the `io.py` contract: one npz per video at
`FEATURE_ROOT/speech_tox/<source>/<stem>.npz` (`t_sec` = 1..floor(duration), duration = frames / fps from
`io.Video` so the ticks match every other stream; `feat` float16 `[T, 38]`; `valid` bool; JSON `meta` with
`columns`), plus a transcript sidecar `FEATURE_ROOT/speech_tox/<source>/<stem>.transcripts.json` for later
LLM use (escalation ratings over rolling transcripts).

Why this stream: verbal threats, insults, commands ("back up", "put your hands up") and challenges ("what are
you gonna do") are among the few cues that precede interpersonal violence by 5 to 120 s
(research/backbones.md section 5). Treat the features as **sparse and high-precision**: many clips have no
intelligible speech, a music overlay, or narration instead of the participants' voices.

### Pipeline (per video)

1. `io.load_audio` at 16 kHz mono. No audio stream or a fully digital-silent track (`io.is_digital_silence`)
   -> `meta["audio"]` = `no_audio` / `digital_silence`, `valid` all False, zero rows, no models loaded. An
   audio stream that ffmpeg cannot decode, a missing file or an unreadable duration **raise** (the runner logs
   the failure and retries it; nothing empty is saved).
2. Per-tick validity (causal): tick t is valid when t >= 2 s, t <= the decoded audio length, and the 12 s
   before it, (t-12, t], are audible: peak >= 1e-4 (not digital silence) and RMS >= -65 dBFS (not encoder
   noise), the same rule as the audio track. A track with no valid tick gets `audio = near_silence`.
3. Silero VAD v6 (ONNX, one CPU thread), **exact streaming** over the whole track: 32 ms frames.
4. ASR windows end every `stride_sec` = 2 s and cover the previous `window_sec` = 12 s (at most 30, enforced;
   shorter at the start). Gate per window:
   - `vad` gate: at least 0.5 s of frames with VAD prob > 0.5 -> Whisper, normal acceptance;
   - `weak` gate: otherwise, if the window's max VAD prob >= 0.15 -> Whisper, strict acceptance
     (no_speech_prob <= 0.4, avg_logprob >= -0.8, and at least 3 words, CJK characters counting 1/2 and
     repeats counted, unless the window's max VAD prob is >= 0.9);
   - else skipped (`asr_ran` = 0). Skipped windows are music-only, silence or noise, where Whisper hallucinates.
5. Whisper large-v3 (`openai/whisper-large-v3`, bf16). Encoder in fixed micro-batches of 16 windows. Decoder
   = a hand-written single-token step with a static KV cache replayed as a **CUDA graph** (same math as the
   HF decoder; logits match HF to 0.16 on a scale of 25 with identical argmax). Greedy, timestamp-mode
   decoding with OpenAI's timestamp rules, no conditioning on previous text.
   - Language id and `no_speech_prob` come from the logits at `<|startoftranscript|>` (as in OpenAI Whisper).
     The decoding language is the window's own argmax. A causal prior only breaks near-ties: if the argmax
     probability is below 0.6, the language that holds >= 50 % of the prior is used instead when it has
     >= 0.1 probability in this window. The prior averages the posteriors of earlier `vad`-gate windows with
     no_speech_prob < 0.5 that ended in the previous 30 s, skipping windows of earlier batches whose
     transcript was rejected (weight = max(VAD speech s, 0.5) x (1 - no_speech_prob)).
6. Hallucination filter (reason stored in the sidecar as `rejected`), in this order:
   - sound captions are stripped: `[...]`, `(...)`, free-standing `*...*` (censored words such as
     "f***ing" are kept), full-width brackets, sung lyrics between music
     symbols (U+266A music notes around a line), music symbols, and short caption sentences in many languages ("Музыка",
     "СПОКОЙНАЯ МУЗЫКА", "Música", "音乐", "applause"); nothing left -> `sound_caption` (or `empty`);
   - repeated n-grams (1 to 16 tokens, CJK characters are tokens) are **collapsed** to 2 copies (first and
     last copy kept); the number of removed copies is stored (`repeats`); only then compression ratio > 2.4
     rejects (`compression_ratio`);
   - OpenAI rule no_speech_prob > 0.6 and avg_logprob < -1.0 (`no_speech`); avg_logprob < -1.5
     (`low_logprob`); weak-gate thresholds (`weak_gate`);
   - sentences with subtitle credits / outros in many languages are removed ("Subtitles by the Amara.org
     community", "Thanks for watching", "Спасибо за просмотр", "字幕", "ご視聴", "Legendas pela ...");
     nothing left -> `credit_phrase`;
   - stock phrases: if every clause is a stock phrase ("Thank you.", "Merci.", "Obrigado.", "谢谢大家",
     "Bye-bye.", "¡Hola!", "you", "Oh") and the window has < 2 s of VAD speech, no_speech_prob > 0.2 or came
     through the weak gate -> `stock_phrase`;
   - weak-gate windows with fewer than 3 words and max VAD prob < 0.9 -> `weak_short`. In the smoke set,
     short real shouts ("Hold it.", "Come on.", "过来") had max VAD prob >= 0.94 and short music/noise
     hallucinations ("nine", "like", "ПЕСНЯ") at most 0.47.
7. Non-English transcripts are translated to English by a **text MT model**, Qwen2.5-7B-Instruct used as a
   plain few-shot completion (`Transcript: ...\nEnglish:`; greedy; fixed batch of 32 and fixed left-padded
   prompt length; stops at the first newline or at a per-row budget of min(128, 1.5 x native tokens + 16)
   tokens, so a row never depends on its batch; unique texts per video only). The translation is checked
   (`mt_rejected`): `mt_refusal` (commentary or refusal), `mt_untranslated` (> 20 % non-Latin letters),
   `mt_length` (more than 2.5 x the native words + 5), `mt_compression` (after repeat collapsing),
   `mt_trivial` (a single function word such as "The"), `mt_empty`, `mt_credit`. A rejected translation leaves
   `text_en = ""` and `en_text = 0`, and its raw output is kept as `raw_text_en`.
8. Text features per accepted window: detoxify `unbiased` (RoBERTa, Jigsaw unintended-bias) on the English
   text (native English or the MT output); detoxify `multilingual` (XLM-R) on the native text **only for its
   7 training languages** en, fr, es, it, pt, tr, ru (`mtox_lang_ok` = 1; zeros otherwise); regex keyword
   counts on the English text; pronoun and punctuation rates.
9. Tick t copies the features of the latest window ending at or before t (lag 0 to 1 s), adds VAD fractions
   and 30 s rolling maxima. Invalid ticks are zeroed.

### Columns (`meta["columns"]`, D = 38)

| Columns | Meaning |
|---|---|
| `vad_frac_1s`, `vad_frac_win`, `vad_frac_30s` | Fraction of 32 ms frames with VAD prob > 0.5 in (t-1, t], (t-12, t], (t-30, t] |
| `asr_ran` | 1 if the latest window passed a gate and Whisper ran |
| `has_text` | 1 if that window has an accepted transcript |
| `no_speech_prob` | Whisper P(<\|nospeech\|>) of the latest window (0 if not run) |
| `asr_conf` | exp(avg_logprob) of the accepted transcript (0 if none) |
| `lang_en_prob` | Whisper P(English) of the latest window (0 if not run) |
| `non_english` | 1 if the accepted transcript is non-English (English columns then come from the MT output) |
| `en_text` | 1 if English text exists for the window (native English, or an accepted translation). `tox_*`, `kw_*` and the rate columns are only meaningful when it is 1 |
| `words_per_sec` | words in the transcript / window length (CJK characters count 1/2; zh/ja/th/...: words of the translation) |
| `speech_rate` | words / VAD speech seconds (0 if < 0.5 s of VAD speech; capped at 8) |
| `repeats` | copies removed by repeat collapsing (shouted repetitions, decoder loops), capped at 10 |
| `tox_toxicity` .. `tox_sexual_explicit` (7) | detoxify `unbiased` probabilities: toxicity, severe_toxicity, obscene, identity_attack, insult, threat, sexual_explicit |
| `mtox_lang_ok` | 1 if the transcript language is one of detoxify-multilingual's 7 languages |
| `mtox_toxicity`, `mtox_insult`, `mtox_threat` | detoxify `multilingual` on the native text (0 when `mtox_lang_ok` = 0) |
| `kw_threat`, `kw_command`, `kw_challenge`, `kw_profanity`, `kw_deescalate` | regex match counts in the English text (lists in `speech.KEYWORDS`; overlapping patterns can count twice) |
| `you_rate` | second-person pronouns / words |
| `excl_rate`, `question_rate` | '!' and '?' per sentence |
| `roll_tox_max`, `roll_insult_max`, `roll_threat_max`, `roll_mtox_max` | max over windows ending in (t-30, t] |
| `roll_kw_hostile_max` | max over the same windows of kw_threat + kw_command + kw_challenge |
| `roll_words_per_sec` | mean words_per_sec over the same windows |

All zero when no window has text yet. A zero toxicity row means "nothing said", not "calm speech"; use
`has_text` / `en_text` as the speech-present masks (the `vad_frac_*` columns under-count shouting in noisy
audio, see Design decisions).

### Transcript sidecar (`<stem>.transcripts.json`)

`{extractor, version, model_id, mt_model_id, asr_window_sec, stride_sec, duration_sec, audio_sec, video_path,
audio, windows: [...]}`. One entry per window that ran Whisper, in time order, with: `start`, `end` (s),
`vad_speech_sec`, `vad_max_prob`, `gate` (`vad`/`weak`), `lang`, `lang_raw` (this window's argmax),
`lang_prior` (true if the prior changed it), `lang_prob`, `no_speech_prob`, `avg_logprob`,
`compression_ratio` (of the raw text), `text` (accepted, cleaned, repeats collapsed; "" if rejected),
`text_en`, `segments` (`[[t0, t1, text], ...]` raw Whisper segments in absolute seconds, not collapsed; t1 can
be null when the decode budget ran out), `rejected` (reason or null), `raw_text` (only when it differs from
`text`), `repeats`, `mt_rejected` and `raw_text_en` (translation problems), `tox`, `mtox` (7 detoxify scores
each, only when computed), `kw` (keyword counts).

**Causal use rule:** the sidecar holds the whole video. A prompt or feature for tick t may only use windows
with `end <= t`: `speech.windows_until(doc, t)` (or `speech.windows_until(meta["transcripts"], t)`). Windows
overlap 6x (12 s windows every 2 s); for a non-overlapping transcript take the windows with `end % 12 == 0`
among those, or the `segments` of the latest one. `speech.load_transcripts(video_id)` reads the file. Without
a `transcript_path` argument, `extract()` puts the same list into `meta["transcripts"]` instead.

### Validity, causality and meta

- `valid[t]` is decided from (t-12, t] only (step 2 of the pipeline). Version 1 decided digital silence on
  the whole track, so a track that was silent until after t was valid at t only because sound came later
  (reviewer's case; 7 UCF videos are silent for their first 30 s or more, e.g. Arrest039 for 69 s). Invalid
  rows are zero.
- At tick t every ASR, text, rolling and validity column uses audio in [t - 42 s, t] (`meta["window_sec"]` =
  42 = 12 s window + 30 s rolling/language horizon). The Silero VAD state runs from the start of the track
  (causal, unbounded look-back); `meta["history"]` says so, as section 0.1 asks.
- Whole-video aggregates (`n_windows`, `n_asr`, `n_text`, `n_translated`, `n_mt_rejected`, `languages`,
  `vad_speech_frac`, `n_valid`) live in `meta["summary_noncausal"]`: diagnostics only, never features.
  `duration_sec` and `audio_sec` are top-level metadata.
- Every GPU batch has a fixed shape (Whisper encoder 16, decoder 16, MT 32 x fixed prompt length, toxicity
  64 x 128 tokens, all padded), and windows are batched in time order, so a window's result does not depend
  on later windows: a truncated copy of the video reproduces the same transcripts, translations, features and
  `valid`.
- `io.check_causal` compares only `feat`; the speech smoke test also compares `valid` and every transcript
  field (results below).
- The ffmpeg resampler in `io.load_audio` has a few ms of lookahead (same caveat as the audio track).

### How to run

```bash
# shared runner (resumable). Run EVERY source: 490 UCF videos (37 h) have real audio, many with speech.
# Two processes per GPU overlap the CPU stages (audio decode, VAD, the Python decode loop) with GPU work;
# each process needs about 25 GB of GPU memory once a non-English video has loaded the MT model.
L=data/logs
for i in 0 1 2 3; do
  setsid nohup .venv/bin/python -m anticipation.features.run_extract --extractor speech_tox \
      --manifest benchmark/videos.csv --sources xd,yt,ucf --shard $i --num_shards 4 --device cuda:$((i % 2)) \
      > $L/speech_tox_$i.log 2>&1 &
done
# UBI-Fights has no audio: running it is cheap (all-invalid files, about 0.3 s each) and keeps the set complete.

# standalone, a few videos (writes npz + transcripts under FEATURE_ROOT/speech_tox/<source>/)
.venv/bin/python -m anticipation.features.speech --device cuda:1 --source xd --videos A.mp4 B.mp4

# model-free self-test of the filters, repeat collapsing, translation checks and keyword patterns (about 2 s)
.venv/bin/python -m anticipation.features.speech --selftest

# Python
from anticipation.features import speech
t_sec, feat, valid, meta = speech.extract(path, "cuda:1", transcript_path=speech.transcript_path("xd/<stem>"))
```

Config overrides (`--cfg key=value` in the runner, kwargs in `extract`): `window_sec` (<= 30), `stride_sec`
(3 or 4 is about 1.5x / 2x faster, features then lag up to 2 or 3 s), `batch_size`, `translate=false` (no MT:
non-English windows then have `en_text` = 0), `min_speech_sec`, `weak_gate_prob`, `weak_min_words`,
`max_repeats`, `lang_*`, `min_rms_db`, thresholds. Defaults in `speech.DEFAULT_CFG`; every run stores its full
cfg in `meta["cfg"]`. Version-1 files (D = 35) must not be mixed with version 2 (D = 38); check
`meta["version"]`.

### Smoke test and throughput (version 2, 2026-10-14, cuda:1)

Scratch script `smoke3.py` (not in the repo; log `logs/speech_smoke_v2.log`), final code, load average 5 to 11
(other agents' jobs). Wall time includes audio decode, VAD, ASR, MT and text models; model loading (15 to 30
s, once per process) is excluded. Format asserts on every video: dtypes, shape `[T, 38]`, ticks
1..floor(duration), finite values, zero invalid rows, JSON-serialisable meta.

| Video | Src | T | valid | windows / Whisper / text | Language | Wall (x realtime) |
|---|---|---|---|---|---|---|
| The.Bourne.Identity.2002 (B1, onset 80.5 s) | xd | 115 | 114 | 57 / 46 / 43 | en | 5.2 s (22.0x) |
| v=96N4XAJ7FKM (B1 hockey fight, onset 38.4 s) | xd | 202 | 201 | 101 / 101 / 99 | en | 9.8 s (20.6x) |
| Tropa.de.Elite.2.2010 (B2, onset 32.4 s) | xd | 60 | 59 | 30 / 30 / 30, all translated | pt | 8.5 s (7.1x) |
| A.Beautiful.Mind.2001 (normal) | xd | 240 | 239 | 120 / 95 / 88 | en | 8.5 s (28.1x) |
| holdout_003 (bodycam, neighbour fight) | yt | 575 | 574 | 287 / 279 / 277 | en | 27.3 s (21.1x) |
| holdout_072 (road-rage news report) | yt | 276 | 275 | 138 / 138 / 137 | en | 15.2 s (18.2x) |
| Assault001_x264 | ucf | 82 | 0 | digital silence, no models | - | 0.3 s |

- These 7 videos: **1471 s of audio in 75 s = 19.7x realtime** (version 1: 12.0x at load 26 to 43).
  All-Portuguese Tropa runs at 7x instead of 3.5x because text MT replaced Whisper's second (translate)
  decode. The 14 review clips below: 1904 s in 87 s = 21.8x (non-English clips 7 to 14x, music clips up to
  99x).
- Random sample as in the review (12 xd/yt videos with audio, `random.seed(7)`, current manifest, so not the
  same 12 videos): 1965 s of audio in 89 s = **22.1x** on one process at load 2 to 5; English videos 21 to 45x,
  non-English 10 to 14x (Election zh 9.9x, a Russian clip 14.3x). Two processes on the same GPU together:
  1965 s in 82 s = 23.8x, i.e. no gain on an idle machine (the GPU is the bottleneck). The review measured
  version 1 at 8.1x on its sample at load 25 to 100 (4.7x at load 102: VAD and audio decode slow down 3 to 4x),
  so expect **5 to 25x** depending on CPU contention; extra processes per GPU only help when the CPU is busy.
- **Estimate:** 262.6 h of footage with audio (benchmark/videos.csv: XD 215.8 h, curated 9.8 h, UCF 37.0 h;
  silent stretches are cheap). At 20x that is 13 GPU-hours (6.6 h on 2 GPUs); at the review's contended 8x,
  33 GPU-hours (16 h on 2 GPUs). Peak GPU memory per process: 26.2 GB allocated (27.2 GB reserved) once the
  MT model is loaded, about 10 GB before. Storage: about 0.7 KB of transcript JSON per audio second plus well
  under 100 MB of npz in total.
- Gates and filters over the 1289 Whisper windows of all 21 videos: 1148 `vad` gate (1135 accepted), 141
  `weak` gate (75 accepted). Rejections: weak_gate 17, sound_caption 15, empty 15, stock_phrase 14, weak_short 11,
  low_logprob 3, credit_phrase 3, no_speech 1, compression_ratio 0 (repeats are collapsed instead).
  243 accepted non-English windows went through MT; 1 translation was rejected (`mt_trivial`: "你" ->
  "You"). The language prior changed 26 windows.
- **Causality** (full vs ffmpeg stream-copy truncation; ticks <= cut - 1; `feat`, `valid` and every transcript
  field compared): max |diff| = **0.0**, `valid` equal and 0 transcript differences on all 7 checks: Tropa cut
  40 s (19 windows, all translated), City of God cut 30 s (pt, 13 translated), Yellow Sea cut 37 s (zh, 18
  translated), Bourne cut 70 s, UCF Arrest039 cut 100 s (silent until 69 s: 30 valid ticks in both), hockey
  with 0 to 40 s zeroed cut at 30 s (0 valid ticks in both; version 1 gave 29 valid ticks on the full copy),
  bodycam cut 200 s (91 windows).
- MT row invariance: the same texts in different batch compositions, orders and next to a long row gave
  identical translations.
- Runner integration (`run_extract --extractor speech_tox --feature_root <scratch>`): City of God and UCF
  Normal_Videos011 ok, a random-bytes `.mp4` failed with "cannot determine the duration" and was logged in
  `failures.jsonl` (nothing saved); the rerun skipped the 2 good files (version check via `OUTPUT_VERSIONS`)
  and retried the corrupt one. `io.load_features` and `speech.load_transcripts` / `windows_until` read the
  outputs.
- Model-free self-test (`--selftest`): 0 failures (28 hallucination strings rejected under both gates,
  shouted repetitions kept, the weak-gate short rule, censored words, 8 translation checks, 25 calm and 20
  hostile keyword sentences).

Transcripts around onsets (version 2, from the sidecar JSON):
- **Bourne Identity** (onset 80.5 s): [60, 72] "You! Red bag! The red bag! Stop right there! Put your hands
  up!" (`kw_command` = 2, toxicity 0.19), 8.5 to 20 s before onset.
- **Tropa de Elite 2** (Portuguese, onset 32.4 s): [0, 12] "Nada, porra. Tá dando cu agora? ..." -> MT
  "Nothing, fuck. Are you fucking now? ..." (toxicity 0.99); [24, 36] contains "Quem foi que disse que a
  comunidade é tua?" (a challenge, not caught by the English keyword list).
- **City of God** (pt): version 1 produced "I'll kill you! I'll kill you! I'll kill you!" (toxicity 0.99) and
  repeated "Give me the money!" with no source text. Version 2: [4, 16] "Caminhão aí! Sai, meu irmão! Desce!
  Não, cara, desce! Não, dinheiro não! Desce!" -> "Truck here! Get out, brother! Get down! No, man, get down!
  No, no money! Get down!" (`kw_command` = 4); [34, 46] "A polícia, mano! A polícia!" -> "The police, man! The
  police!".
- **Yellow Sea** (zh): version 1 translated "你还敢打我 来吧 来吧" as "I'll kill you! Wait! What?" (0.96); version 2
  gives "You dare hit me? Come on, come on." (toxicity 0.38, `kw_challenge` = 1).
- **Young and Dangerous** (zh): windows ending 64 to 68 were dropped in version 1 (compression ratio of a
  looping translation). Version 2 keeps "你去死吧 你大吉小好 你去死吧 你去死吧" (`repeats` = 2) -> "Go die, you big
  fool, go die, go die." (toxicity 0.999, threat 0.93).
- **The Intouchables** (label A): version 1 forced Russian on French windows and accepted "СПОКОЙНАЯ МУЗЫКА";
  version 2 decodes them as French ("Ça y est, les voilà. Bon, Philippe, on se réveille un peu ...").
  The first 10 s still give a Whisper hallucination "Сигарство" (VAD gate, logprob -0.8 to -1.0) -> "Cigarette.".
- **Music-only XD clips** (v=jltJrNKnrNQ, v=tT-SrQC6Ddw) and UCF Normal_Videos011: every "Музыка" / "Music" /
  "Música" / "¡Hola!" window is rejected (`sound_caption`, `stock_phrase`); 0 text windows on the music clip.
- **Bodycam holdout_003**: [156, 168] "You all get out! Get out of here! [speech removed]
  [speech removed] ..." (toxicity 0.99, `kw_command` = 2). **UCF Arrest039** (eligible, onset 240.5 s): silent until 69 s (invalid), then hostage
  negotiation from about 120 s: [speech removed] (toxicity 0.99).
- **Hockey v=96N4XAJ7FKM**: TV commentary; **holdout_072**: news-anchor narration throughout (see Gotchas).

### Review fixes (v2)

An independent review of version 1 reported ten issues; all were reproduced and fixed:

| Issue | Fix |
|---|---|
| Run plan skipped UCF, but 490 UCF videos (37 h) have real audio, many with speech | All sources in the run recipe; "UCF has no usable audio" removed |
| Whisper translate invented hostile content; mtox ~0 on unsupported languages | Text MT of the native transcript (Qwen2.5-7B-Instruct) with output checks; `mtox_lang_ok` mask, zeros outside the 7 languages |
| Bracketed / non-English captions ("(upbeat music)", "Музыка", "Merci.") accepted | Caption stripping, multilingual caption / credit / stock lists, weak-gate short-text rule, trivial-translation check |
| Language prior overrode clear current-window evidence | Prior only breaks near-ties; rejected windows excluded |
| Compression-ratio rule dropped shouted repetitions | Repeat collapsing first; `repeats` column; translations no longer dropped on loops |
| `valid` used the whole track (future leak) | Per-tick audibility of (t-12, t]; smoke compares `valid` on truncation |
| Keyword false positives | Patterns tightened; self-test sentences |
| Throughput estimate optimistic | Measured range 5 to 25x and the real hour count (262.6 h) |
| Missing / unreadable file saved as an empty success | Raises; the runner logs and retries |
| Meta and doc traps (window_sec, whole-video aggregates, sidecar rule, window_sec > 30, rejected translations) | `meta["history"]`, `summary_noncausal`, `windows_until`, `window_sec <= 30` check, `raw_text_en` |

### Design decisions and why

- **large-v3, not turbo**: turbo hallucinates more on noisy audio (research/backbones.md section 5).
- **Timestamp-mode decoding**: in `<|notimestamps|>` mode large-v3 often stops early and returns only a
  fragment of a full 12 s window ("the emotion of", "second time"). Measured on two XD clips: windows with
  fewer than 1 word per VAD-speech second dropped from 13 to 17 % to 0 to 1.5 %, at the same speed.
- **Weak gate instead of a pure VAD gate**: Silero under-detects shouting in noisy footage. In the bodycam
  video `holdout_003` it scored "[speech removed]" windows at 0.35 to 0.6 s of speech, and a
  shouting stretch at max prob 0.45. Whisper transcribes these correctly. The weak gate (max prob >= 0.15)
  recovers them, while music-only spans (max prob about 0.03) stay skipped. Because weak-gate windows are
  also where caption hallucinations ("Музыка", "Kampung Kampung") appear, v2 adds the 3-word minimum
  (waived when VAD was confident, max prob >= 0.9, so "Hold it." survives).
- **No words-per-VAD-second filter**: an early version rejected windows with > 8 words per VAD-speech second;
  it removed 24 windows of genuine shouting ("You f***ing animals!", "I hate you") in one video.
- **Collapse repeats instead of rejecting on compression ratio** (v2): the 2.4 rule rejected "Stop
  resisting!" x 6 (ratio 3.52), "Get on the ground!" x 5 and "Stop!" x 8, i.e. the escalation cue itself,
  and threw away the real prefix of decoder loops. Temperature fallback (OpenAI) was not used because it
  needs sampling, which breaks bit-exact causality checks.
- **Text MT instead of Whisper's translate task** (v2): Whisper translation is a second free-running decode
  of the audio and invented hostile content with no source: "O que é isso? A polícia! A polícia!" became
  "I'll kill you! I'll kill you! I'll kill you!" (toxicity 0.99, logprob -0.74, so no score gate catches it),
  "Give me the money!" appeared in 5 City of God windows without any source text, "你还敢打我 来吧 来吧" became
  "I'll kill you! Wait! What?". Translating the accepted native transcript keeps the English grounded in what
  was transcribed. Qwen2.5-7B-Instruct was chosen over 1.5B / 3B after a side-by-side on 26 review lines: 1.5B
  refused profanity and sanitised it ("Nada, porra" -> "No, bro"), 3B invented an insult for Korean noise
  ("Your gay, retard."); 7B as a few-shot completion was faithful on all lines (chat mode refused "我操 你去死"
  and sometimes echoed the source, hence completion mode). NLLB-200 / M2M100 / opus-mt ship only pickle
  `.bin` weights, which transformers 4.57 refuses to load with torch 2.5.
- **mtox masked outside its languages** (v2): detoxify-multilingual gave 0.0003 toxicity on "他媽的"; a 0 that
  means "unsupported" must not look like "calm".
- **Language prior only breaks near-ties** (v2): v1 summed up to 14 overlapping earlier windows against
  2 x the current one, so a few hallucinated windows forced the language: The Intouchables windows with raw
  `fr` were decoded as `ru` (p = 0.007) and produced "СПОКОЙНАЯ МУЗЫКА" instead of the French dialogue.
  The near-tie threshold is 0.6 (at 0.5, Yellow Sea's "你还敢打我 来吧 来吧" window, p(ko) between 0.5 and 0.6 and
  p(zh) 0.14, stayed Korean and came out as "아, 까맣다! 탐마! 비하시! 군!");
  in the smoke set the prior changed 26 of 1289 windows.
- **Streaming VAD, not segment-reset VAD**: a batched GPU variant that reset the state every 30 s (with 10 s
  warm-up) agreed with streaming on 98 % of frames of movie dialogue, but marked only 3 % of a bodycam shouting
  stretch as speech versus 45 % for streaming. Silero adapts to the background over tens of seconds.
- **Keyword patterns tightened** (v2): v1 fired on "We're going to hit the road", "You want some cake?",
  "Hit me up", "Stop by the store", "Drop the kids off", "Dick Cheney" and narration ("trying to stop her").
  Threat patterns now need a violent verb with a person object; utterance commands ("Stop!", "Turn around!",
  "Get back!") must stand alone between punctuation; bare "come on", "punk", "dick" were dropped. The
  self-test holds 25 calm and 20 hostile sentences (0 false positives, 0 misses). These remain heuristics with
  unknown precision on real transcripts; the planned LLM escalation rating over the sidecar should supersede
  them. Implicit threats ("Touch her again and you won't walk out of here") are covered only by a few
  patterns, and detoxify scores them low (0.04).
- **CUDA graph decoder**: on this shared machine (load average 20 to 28) the HF decoder spent about 60 ms of
  CPU launch overhead per token for 8 to 19 ms of GPU work. The graph makes the decoder GPU-bound (reading the
  cross-attention KV over all 1500 encoder positions dominates). Decoder batch 16 is as fast per window as 64
  and wastes less padding on short clips.
- **Detoxify via the pip package** (GitHub release checkpoints `toxic_debiased-c7548aa0.ckpt`,
  `multilingual_debiased-0b549669.ckpt`, cached in `~/.cache/torch/hub/checkpoints`): the HF copies
  (`unitary/unbiased-toxic-roberta`, `unitary/multilingual-toxic-xlm-roberta`) are pickle `.bin` only, which
  transformers 4.57 refuses to load with torch 2.5.

### Gotchas

- **Import torch before decord.** `import decord` followed by `import torch` and CUDA init segfaults
  (exit 139, no traceback). `speech.py` imports torch at module level and `run_extract.py` imports torch first
  for CUDA devices; any new script that calls `io.Video` before touching torch must do the same.
- `import silero_vad` calls `torch.set_num_threads(1)` process-wide; `speech.get_models` restores it.
- Write and read the transcript JSON with `encoding="utf-8"`: the default locale here is ASCII.
- **Translated text is lower precision than native English.** MT keeps profanity ("Caralho!" -> "Fuck!"),
  so `tox_*` on translated windows runs hotter than on comparable English, and Whisper's own errors on noisy
  non-English audio pass through ("Não para de fazer o sentimento bala" -> "the bullet feeling"). Use
  `non_english` as an interaction, and report non-English videos separately.
- Keyword lists are English-only; for non-English windows they run on the translation. Escalation phrased
  indirectly ("Who said the community is yours?") is not caught.
- Speech on curated YouTube news videos is often a **narrator** describing the incident ("police were not
  expecting this scene"), not the participants. That is a provenance shortcut risk (news narration precedes
  footage of violence). Report the curated set separately and audit narration-heavy videos.
- **Narration leaks outcomes.** News-style curated videos (e.g. `holdout_072`) and sports broadcasts (XD
  hockey clips) contain a narrator who announces the violence before it is shown. Text features can then
  "anticipate" through narration rather than the participants' escalation. Flag narrated videos (high
  `words_per_sec` with calm `tox_*`, anchor phrases) and report them separately; never tune on them.
- XD movie clips mix dialogue, score and dubbed languages. Sung lyrics marked with music symbols are dropped
  in v2, but Whisper does not always mark singing; check `tox_*` against the audio track's `Music` column in
  shortcut audits.
- `vad_frac_*` and `speech_rate` under-count shouting in noisy audio (Silero); prefer `has_text`, `asr_ran`.
- Gate and filter thresholds were tuned by eye on about 20 videos, not on labelled data.
- Weights: Whisper large-v3 3.1 GB (HF cache, fp16 safetensors), Qwen2.5-7B-Instruct 15 GB (HF cache,
  safetensors; loaded lazily on the first non-English window), detoxify 0.5 + 1.1 GB (torch hub cache),
  `roberta-base` / `xlm-roberta-base` tokenizers from HF.

Packages installed for this track (pip, serialized with the shared lock, `--no-deps`): `detoxify 0.5.2`,
`silero-vad 6.2.1`, `sentencepiece 0.2.1`. Note: this install ran 8 s after the audio track's funasr install
and replaced its `silero-vad 6.2.3` and `sentencepiece 0.2.2` with 6.2.1 / 0.2.1. The Silero model files are
byte-identical in 6.2.1 and 6.2.3 (same sha256 for `silero_vad.onnx` and `silero_vad.jit`; only optional
audio-I/O helpers changed), so VAD outputs are unaffected. torch, transformers and numpy were not changed.
Version 2 installed nothing (the Qwen2.5 weights come from the HF cache).

---

## Pose and interaction (pose_raw, pose_interact)

Written 2026-10-14 by the pose track; revised the same day after review (`pose_raw` version 2,
`pose_interact` version 2, see "Version history" below). Code: `anticipation/features/pose.py` (registry names
`pose` and `pose_interact`). Shares `rolling_slope`, `window_bounds`, `open_scaled`, `decode_frames` and
`frame_dt` with `motion.py`. Self-test (CPU, 2 s): `.venv/bin/python -m anticipation.features.pose --selftest`.

### What is stored

One detector + tracker pass writes two files per video.

| Extractor | D | Window | Content |
|---|---|---|---|
| `pose_raw` | 3 (+ ragged arrays) | 1 s | Every person detection on every sampled frame (5 fps): track id, box, confidence, 17 COCO keypoints. Contract keys hold a tiny per-tick summary (`n_tracked_last`, `n_tracked_mean_1s`, `det_conf_max_1s`). |
| `pose_interact` | 65 | 3 s of frames, slopes up to 30 s (`meta["window_sec"]` = 33) | 44 engineered person / dyad / crowd / camera features from the frames in `[t - 3, t]`, plus 21 causal slopes. |

`pose_raw` extra keys (CSR layout; load with `pose.load_raw(video_id)`, which returns a dict):

| Key | Shape, dtype | Meaning |
|---|---|---|
| `frame_t` | [F] f32 | sample times 0, 0.2, 0.4, ... (`io.Video.sample_times(5)`) |
| `frame_idx` | [F] i32 | decoded frame index used for each sample (last frame at or before the time). Real frame time = `frame_idx / meta["video"]["fps"]`. Use this array; recomputing it from the float32 `frame_t` is off by one for some samples (for example 1.8 s stored as 1.7999999) |
| `frame_ok` | [F] bool | False if the frame failed to decode (version 2: per frame; version 1: the whole 32-frame batch) |
| `frame_ptr` | [F+1] i64 | detections of frame k are rows `frame_ptr[k]:frame_ptr[k+1]` |
| `frame_warp` | [F, 2, 3] f32 | affine camera warp (BoT-SORT GMC) that the tracker applied from the previous frame it saw to frame k; translation in native pixels. Identity for ByteTrack and when GMC failed |
| `frame_gmc_ok` | [F] bool | version 2: False when GMC raised on frame k. Then frame k's warp is identity, and frame k+1's warp spans two steps, because GMC keeps its old reference frame. `pose_interact` skips both steps |
| `frame_n_hiconf` | [F] i16 | person detections with conf >= 0.5 (tracked or not) |
| `det_track_id` | [N] i32 | track id (from 1, reset per video); -1 = detection not confirmed by the tracker |
| `det_box` | [N, 4] f32 | x1, y1, x2, y2 in native-resolution pixels (raw detector box, not the Kalman box) |
| `det_conf` | [N] f32 | detection confidence |
| `det_kpt` | [N, 17, 3] f32 | x, y (native pixels), keypoint confidence; order in `meta["keypoint_names"]` |

Saved detections are all tracked ones plus untracked ones with conf >= 0.25. Size: 2.8 KB per video-second for
UCF CCTV and 5.8 KB for a 12-person XD street crowd, so about 3.5 to 7 GB for 350 h. `pose_interact` is about
0.13 KB per video-second.

### Pipeline (defaults in `pose.DEFAULTS`)

1. **Sampling:** 5 fps on the video clock. Frames are decoded with the long side capped at 960 px. All local
   sources are 640x360 or smaller, so they decode at native size.
2. **Detector:** Ultralytics **YOLO11x-pose**, fp16, `imgsz` 640, detection threshold 0.1 (low-score boxes
   feed BYTE's second association), NMS IoU 0.7, at most 100 people.
   - Weights: `models/ultralytics/yolo11x-pose.pt` (HF mirror `Ultralytics/YOLO11`).
     `yolo11l-pose.pt` is next to it. `resolve_weights` downloads missing names from the mirror and falls back
     to the repo's `yolov8m-pose.pt`. `meta["weights_sha256_16"]` records the file used.
   - The letterbox runs on the GPU and reproduces `ultralytics.data.augment.LetterBox` padding exactly. Boxes
     and keypoints match `YOLO.predict` to sub-pixel on 640x360. On 320x240 UCF the difference is 1 to 3 px,
     because torch bilinear upscaling is not cv2 bilinear.
   - torchvision is imported, so ultralytics uses CUDA NMS instead of its Python loop. That loop and the CPU
     preprocessing made the naive `model.predict` path 4x slower.
3. **Tracker:** BoT-SORT from ultralytics 8.3.228 with `botsort.yaml` thresholds (high 0.25, low 0.1, new
   track 0.25, match 0.8, fused score) and no ReID.
   - `track_buffer` = 5 s (25 sampled frames).
   - Camera-motion compensation (GMC sparse optical flow) runs at downscale 3 with 300 corners. The ultralytics
     defaults (downscale 2, 1000 corners) cost 15 ms per frame for similar warps.
   - The tracker sees frames strictly in time order. Its id counter is reset per video.
4. **Pipelining:** a background thread decodes and detects one 32-frame batch ahead of the tracker. The last
   batch is padded to 32 frames by repeating its final frame, so the numerics never depend on video length.
   A batch that fails to decode is re-decoded frame by frame (`motion.decode_frames`). Frames that still fail
   are marked `frame_ok = False` and are never shown to the tracker. Their detector slot gets a copy of a
   decoded frame and its result is discarded, so earlier frames stay bit-identical.
5. **Interaction features** (`interaction_features(raw)`, pure numpy, 100 to 280x realtime on one core):
   aggregated per tick over the sampled frames with time in `[t - 3, t]` (16 frames).

Definitions used by the columns (version 2):
- **Visibility:** keypoint conf >= 0.3.
- **Body height h** (`body_height`, `TrackHeight`):
  - Per frame: `max(box height, keypoint estimate)`. The keypoint estimate is ratio x segment length for every
    segment whose two keypoints are visible: torso (shoulder midpoint to hip midpoint) x 3.4, shoulder width
    x 5.5, upper arm x 6.3, thigh x 4.5, shin x 4.7, ear span x 10. It takes the largest cue, or the
    second-largest when there are 3 or more (so one bad keypoint cannot inflate it).
  - The ratios come from 2,816 full-body upright people in the smoke videos (box height / segment, about the
    25th percentile; segments only shrink under foreshortening).
  - h is then the **causal per-track median over the last 1 s** (`h_smooth_sec`), so occlusions and raised
    arms (which stretch the box) do not move it.
  - Why: version 1 fell back to the upper-body box when the hips were hidden. That was about 3x too small:
    median h / shoulder width was 6.9 with the torso visible and 1.9 without it. Now it is 7.1 vs 5.8
    (Fighting003), 6.7 vs 5.5 (XD) and 4.7 vs 5.5 (yt). On full-body people h / box height has median 1.02.
- **Position** (for distances): the torso center (mean of the shoulder and hip midpoints). Without hips:
  shoulder midpoint + 0.14 h down. Without shoulders: hip midpoint - 0.14 h. Without both: mean head
  keypoint + 0.245 h. Without any of these: the box center. The offsets are medians on full-body people.
- **Pair distance:** `dist(c_i, c_j) / mean(h_i, h_j)` in the image plane.
- **Head yaw** (`head_yaw`): `f = sin(yaw)` in [-1, 1], > 0 = facing image-right, unknown without the nose.
  - With both ears visible: `tan(yaw) = 0.65 (nose_x - ears_mid_x) / half ear span`. 0.65 is head half-width
    over nose depth.
  - Otherwise with both eyes: the same with the eyes and 0.8 (fitted to the ear estimate, corr 0.77).
  - Otherwise with one ear or one eye: profile, `f = +/-0.9` from the side of the nose that keypoint is on.
    That sign agrees with the eye estimate 95 % of the time.
  - Version 1 used `clip((nose_x - ears_mid_x) / 0.06 h)`. With one ear hidden this was +/-1 for 84 to 99 % of
    heads, a nearly binary "which ear is visible" signal. Now no head saturates, and the one-ear median |f|
    is 0.18 to 0.34.
  - **Mutual facing** of a pair = the mean of the known terms `f_i * sign(x_j - x_i)` and
    `f_j * sign(x_i - x_j)`: +1 = heads turned toward each other, -1 = turned away.
- **Velocities:** per track, between consecutive sampled frames, in body heights per second (h/s). They are
  divided by the **real time between the two decoded frames** (`frame_idx` difference / fps). At 24 fps
  the steps are 4 or 5 frames (0.167 or 0.208 s), not 0.2 s.
  - Each step uses one anchor per person, built from the **same keypoints in both frames** (`_anchor_pair`):
    the torso keypoints visible in both, else the head keypoints visible in both, else the two box centers.
    Version 1 switched between torso center and box center when keypoints flickered. That turned flicker
    into about 1 h/s of fake speed (median 0.83 to 1.09 h/s on flip steps against 0.02 to 0.17 on steady
    steps). Now flip steps match steady ones: 0.17 vs 0.12 (Fighting003), 0.13 vs 0.13 (XD), 0.011 vs 0.009 (yt).
  - Camera compensation: the previous anchor is mapped through the GMC warp first.
  - Wrist speed is wrist motion relative to the same anchor (gesturing rather than walking).
  - Steps are skipped when either frame is missing, when both samples hit the same decoded frame (video
    below 5 fps), or when GMC failed on either frame (`frame_gmc_ok`).
- **Closing speed of a pair:** `(d_first - d_last) / gap`. The end frame is the last frame in the window with
  >= 2 tracked persons. The start frame is the pair's earliest co-occurrence in the window at least 0.35 s
  (two sampled steps) earlier. Each person is anchored on the same keypoints at both ends, and the gap is in
  real time. Positive = approaching.
- **Camera** (`cam_*`): translation is the displacement of the frame center under the warp,
  `(A c + t - c) / frame width / dt`. Version 1 used the translation column `t`, i.e. the displacement of the
  image origin (top-left corner). Zoom and rotation then showed up as translation: about 2x too high
  (XD 0.223 vs 0.104 W/s), and a pure 5 % zoom read as 0.14 W/s. Zoom is `abs(log scale) / dt`.
- **Slopes:** `motion.rolling_slope`, the least-squares slope per second over ticks `[t - H, t]` where the
  feature is defined (valid tick and not a fill value). 0 when fewer than 3 such ticks.
- **Undefined values** (for example distances with fewer than 2 people) take the fill value in the table:
  10 body heights for distances, 0 otherwise. `meta["fill_values"]` lists them.
- **`valid` = at least one tracked person in `[t - 3, t]`.** Camera and person-count columns are still
  filled on invalid ticks.

Window columns (`pose.WINDOW_COLS`; also `meta["columns"]` and `meta["column_docs"]`):

| # | Column | Fill | Definition |
|---|---|---|---|
| 0 | `n_persons_mean` | 0 | tracked persons per frame, frame mean |
| 1 | `n_persons_max` | 0 | max tracked persons in a frame |
| 2 | `n_persons_last` | 0 | tracked persons in the last frame <= t |
| 3 | `n_det_hiconf_mean` | 0 | person detections with conf >= 0.5 per frame (tracked or not), frame mean |
| 4 | `det_conf_mean` | 0 | mean detection confidence of tracked persons |
| 5 | `det_conf_max` | 0 | max detection confidence of tracked persons |
| 6 | `kpt_conf_mean` | 0 | mean keypoint confidence of tracked persons (all 17 keypoints) |
| 7 | `kpt_visible_frac` | 0 | fraction of keypoints with conf >= kpt_conf |
| 8 | `new_tracks` | 0 | track ids first seen inside the window (churn, occlusion, cuts) |
| 9 | `track_age_mean` | 0 | mean age in s of tracks in the last frame (since first seen, capped at 30) |
| 10 | `person_height_median` | 0 | median body height / frame height, frame mean (camera distance) |
| 11 | `person_height_max` | 0 | max body height / frame height |
| 12 | `pair_present_frac` | 0 | fraction of frames with >= 2 tracked persons |
| 13 | `min_pair_dist_mean` | 10 | per-frame min pairwise torso distance / mean body height of the pair, frame mean |
| 14 | `min_pair_dist_min` | 10 | min over the window of the per-frame min pairwise distance |
| 15 | `mean_pair_dist` | 10 | mean pairwise normalized distance, frame mean |
| 16 | `n_close_pairs` | 0 | pairs closer than 1 h, frame mean |
| 17 | `n_contact_pairs` | 0 | pairs closer than 0.5 h, frame mean |
| 18 | `max_box_iou` | 0 | max pairwise box IoU, frame mean (grappling / occlusion) |
| 19 | `crowd_spread` | 0 | mean distance to the group centroid / median h, frame mean |
| 20 | `closest_pair_closing_vel` | 0 | closing speed (h/s, > 0 = approaching) of the closest pair in the last frame of the window with >= 2 tracked persons, from the pair's earliest co-occurrence in the window (>= 0.35 s apart), same keypoints at both ends |
| 21 | `max_pair_closing_vel` | 0 | max closing speed over all pairs in that frame (the last frame with >= 2 persons) |
| 22 | `mean_pair_closing_vel` | 0 | mean closing speed over all pairs in that frame |
| 23 | `crowd_convergence` | 0 | decrease per s of crowd_spread over persons present in both the first and the last frame with >= 2 persons (>= 0.35 s apart) |
| 24 | `n_converging_people` | 0 | persons in the last frame moving (> 0.2 h/s) toward the centroid of the others (cos > 0.5), window-mean velocity |
| 25 | `frac_converging_people` | 0 | n_converging_people / persons in the last frame |
| 26 | `n_moving_people` | 0 | persons in the last frame with window-mean speed > 0.2 h/s |
| 27 | `body_speed_mean` | 0 | body speed in h/s over all person steps (camera compensated; per step the same keypoints in both frames: torso, else head, else box center) |
| 28 | `body_speed_max` | 0 | max body speed in h/s |
| 29 | `mutual_facing_closest` | 0 | mutual facing of the closest pair in [-1, 1] (1 = heads turned toward each other; head yaw = sin(yaw) from nose vs ear / eye span), frame mean |
| 30 | `mutual_facing_mean` | 0 | mutual facing averaged over pairs closer than 2 h, frame mean |
| 31 | `faceoff_frac` | 0 | fraction of frames whose closest pair is < 1.5 h apart with mutual facing > 0.3 |
| 32 | `profile_frac` | 0 | fraction of persons seen in profile (shoulder width < 0.1 h) |
| 33 | `arm_raise_rate` | 0 | fraction of person-frames with a wrist above its shoulder |
| 34 | `wrist_above_head_rate` | 0 | fraction of person-frames with a wrist above the head |
| 35 | `arm_toward_other_mean` | 0 | reach toward the nearest person: (dist(shoulder, other) - dist(wrist, other)) / h, max over arms, person-frame mean |
| 36 | `arm_toward_other_max` | 0 | max of the reach above |
| 37 | `wrist_to_other_min` | 10 | min distance from any wrist to another person's torso center / their h |
| 38 | `wrist_speed_mean` | 0 | wrist speed relative to the own body anchor (h/s, max over arms), person-step mean |
| 39 | `wrist_speed_max` | 0 | max of the wrist speed above |
| 40 | `torso_lean_mean` | 0 | torso angle from vertical / 90 deg, range [0, 2] (0 upright, 1 horizontal, > 1 shoulders below hips: bent over or inverted; 0.1 % of XD persons), person-frame mean |
| 41 | `cam_translation` | 0 | speed of the frame center under the camera warp (A c + t - c), frame widths per s, step mean |
| 42 | `cam_zoom` | 0 | abs(log scale change) per s, step mean |
| 43 | `cam_shake` | 0 | std over steps of the frame-center velocity (frame widths per s) |

Slope columns (44 to 64): `<key>_slope{5,15,30}s` for `n_persons_mean`, `min_pair_dist_mean`, `mean_pair_dist`,
`mutual_facing_closest`, `arm_raise_rate`, `wrist_speed_mean`, `body_speed_mean`, in that order (key-major).

### Causality

- The tracker carries identity over `[0, t]`, which VA30 allows (`meta["history"]` says so). Numeric features
  look back at most 33 s.
- Version 2 code, four smoke videos, cut at half the duration. Every check gives **max_abs_diff = 0.0**:
  - `io.check_causal` (ffmpeg truncation; ticks `<= cut - 1`);
  - future perturbation: frames after the cut are replaced by noise while the length stays the same, and
    ticks `<= cut` are compared;
  - decode-error locality: one undecodable frame at 0.8 x the cut; ticks before it are compared and none
    loses validity;
  - `pose.check_raw_causal`: identical detections, track ids and keypoints for every frame before the cut.

| Video | Duration | Cut | `check_causal` | Perturbation | Decode error at | `pose_raw` |
|---|---|---|---|---|---|---|
| `xd/v=pL2HjXvWuPc__#1_label_B1-0-0` | 201.5 s | 100 s | 0.0 (99 ticks) | 0.0 (100 ticks) | 80.05 s: 0.0 (80 ticks) | identical, 496 frames, 4,775 detections |
| `ucf/Fighting003_x264` | 103.4 s | 51 s | 0.0 (50) | 0.0 (51) | 40.85 s: 0.0 (40) | identical, 251 frames, 1,014 detections |
| `ucf/Normal_Videos_050_x264` | 139.9 s | 69 s | 0.0 (68) | 0.0 (69) | 55.25 s: 0.0 (55) | identical (empty highway, no people) |
| `yt/holdout_072` | 277.0 s | 138 s | 0.0 (137) | 0.0 (138) | 110.45 s: 0.0 (110) | identical, 686 frames, 733 detections |

Version 1 would have failed the decode-error check: a failing frame dropped its whole 32-frame batch, i.e.
up to 6.4 s of earlier frames.

### How to run

```bash
# batch (writes pose_raw + pose_interact); see section 0.3 for the 6-process bulk recipe
.venv/bin/python -m anticipation.features.run_extract --extractor pose --manifest benchmark/videos.csv \
    --shard 0 --num_shards 6 --device cuda:0
# after changing the feature code: bump pose.VERSION, then re-derive pose_interact from pose_raw on CPU (no
# detector). Only files whose meta version differs are redone, so this resumes after a crash.
.venv/bin/python -m anticipation.features.run_extract --extractor pose_interact --manifest benchmark/videos.csv \
    --cpu_threads 1 --shard 0 --num_shards 8
# one video: timing, selected columns, causality checks (writes nothing unless --save <video_id>)
.venv/bin/python -m anticipation.features.pose --video VIDEO.mp4 --device cuda:0 --print_ticks 40:50 \
    --print_cols n_persons_mean,min_pair_dist_mean,closest_pair_closing_vel --check_causal 60
# synthetic self-test of the version-2 fixes (CPU, no detector, 2 s)
.venv/bin/python -m anticipation.features.pose --selftest
```

Versions: `pose.RAW_VERSION` for `pose_raw` (detection and tracking) and `pose.VERSION` for `pose_interact`
(feature code), both in `pose.OUTPUT_VERSIONS`. `pose_interact` meta records `raw_version`. Bump
`RAW_VERSION` only when the detection or tracking pass changes. That makes `--extractor pose` redo
everything (GPU). A feature-only change bumps `VERSION` and is redone on CPU with `--extractor pose_interact`.
Running `--extractor pose` after a feature-only bump also works, but reruns the detector.

Version history:
- `pose_raw` 2: the recorded GMC warp is the identity the tracker actually used when GMC raised. Version 1
  kept the previous frame's warp, a phantom camera motion (reproduced by forcing
  `cv2.estimateAffinePartial2D` to return None, which degenerate points do). `frame_gmc_ok` is new, and
  decoding falls back to one frame at a time. Detections on the smoke videos are unchanged from version 1.
- `pose_interact` 2:
  - same-keypoint step anchors and the per-track multi-cue body height;
  - continuous head yaw;
  - frame-center camera translation;
  - rates divided by the real gap between decoded frames;
  - steps with an unknown warp skipped;
  - closing speeds from same-keypoint anchors and a 0.35 s minimum gap (was 0.4 s nominal).
  `pose_interact` 2 also reads version-1 `pose_raw` files (no `frame_gmc_ok`: all steps are kept).

Python: `pose.extract(path, device)` returns the `pose_interact` tuple. `pose.track_video(path, device)`
returns the raw dict, `pose.interaction_features(raw)` turns it into the tuple, and `pose.load_raw(video_id)`
reads a saved `pose_raw`. Config overrides go through `--cfg` (for example `weights=yolo11l-pose.pt`,
`tracker=bytetrack`, `window_sec=5`). Changing a detection key needs `--extractor pose --overwrite`. Changing a
feature key (`window_sec`, `slope_horizons`, `kpt_conf`, `h_smooth_sec`) needs only `pose_interact
--overwrite`. Either way, save non-default variants under a new `--feature_root`. The version check compares
versions only, not configs, so a config variant written to the real root would be skipped as up to date.

### Smoke test and throughput (2026-10-14)

Saved under the real `FEATURE_ROOT` (`pose_raw`, `pose_interact`, `motion`; all version 2, regenerated
after the review fixes). The five videos:
- `xd/v=pL2HjXvWuPc__#1_label_B1-0-0`: street fight, onset 45.4 s. T = 201, `pose_interact` valid 0.97,
  9,476 saved detections.
- `ucf/Fighting003_x264`: onset 60.7 s. T = 103, valid 1.0.
- `ucf/Normal_Videos_050_x264`: empty highway. 0 people, `valid` all False, as expected.
- `ucf/Normal_Videos_150_x264`: normal footage with people, 2.9 persons per frame. T = 28, valid 1.0.
- `yt/holdout_072`: news report on a road-rage shooting. T = 276, valid 0.85.

Checks on these files:
- No GMC failures (`frame_gmc_ok` all True).
- Re-deriving `pose_interact` from the saved `pose_raw` on CPU reproduces the files bit for bit.
- A second runner call skips all five (`stale=0`).

The runner was also tested with a corrupt file, a missing file and a 0.6 s clip. The first two were logged
to `failures.jsonl` and the run continued. The short clip gave a valid file with T = 0. A 60 %-truncated
faststart mp4 fails at `io.Video` open (decord `av_read_frame failed`) and is logged as a failure. Such a
file is lost entirely, because decord cannot open it; salvaging the decodable prefix would need a change in
`io.Video`.

Throughput, in video-seconds per wall-second, including decoding and model loading. Other tracks were running,
with load average 15 to 24 on 16 cores.

| Configuration | Speed | Note |
|---|---|---|
| Detector alone, YOLO11x-pose, batch 32, 640x384 | 463 frames/s = 93x per GPU | YOLO11l-pose: 681 frames/s = 136x |
| 1 process, BoT-SORT | 18 to 25x | CPU-bound: GMC about 6 ms + tracker about 3 ms per sampled frame (crowded clip) |
| 1 process, `tracker=bytetrack` | 31.5x | no camera compensation (`cam_*` = 0, velocities not compensated) |
| 3 processes on one GPU | 41.7x aggregate | 13.9x each, 18 mixed xd/ucf/yt videos (61.8 min each) |
| **6 processes on 2 GPU** | **68x aggregate** | 3 per GPU |
| `pose_interact` re-derivation from `pose_raw` (CPU) | 100 to 280x per core (version 2; crowded clips are slowest) | 21 XD clips: 181x |

These numbers are for version 1. Version 2 changes nothing on the GPU path; the 5-video smoke rerun gave 19.6x
in one process including model load. Estimate: **the current 352 h manifest takes about 5.2 h on 2 GPUs with
6 processes.** With ByteTrack, or on an idle machine, expect roughly 1.5x faster. More than 3 processes per GPU
does not help while the CPU is shared.

### Sanity: do the cues move around real onsets?

All numbers below are version 2. Version 1 values are given in parentheses where the fixes changed them
materially.

Single clips (mean of the column per window, relative to onset o):

| Clip | Column | [o-40, o-20) | [o-20, o-5) | [o-5, o) | [o, o+10) |
|---|---|---|---|---|---|
| XD street fight | `n_persons_mean` | 6.9 | 11.7 | 11.7 | 9.2 |
| | `body_speed_mean` (h/s) | 0.31 | 0.15 | 0.22 | 0.22 (v1 0.27) |
| | `wrist_speed_mean` (h/s) | 0.41 | 0.27 | 0.31 | 0.31 |
| | `mutual_facing_closest` | 0.22 (v1 0.32) | 0.07 | -0.13 (v1 0.14) | -0.24 |
| | `faceoff_frac` | 0.15 (v1 0.30) | 0.23 | 0.16 | 0.03 |
| UCF Fighting003 | `n_persons_mean` | 4.4 | 4.1 | 5.9 | 5.4 |
| | `min_pair_dist_mean` (h) | 0.40 | 0.46 | 0.35 | 0.32 |
| | `max_pair_closing_vel` (h/s) | 0.21 (v1 0.28) | 0.13 | 0.18 | 0.45 (v1 0.65) |
| | `wrist_speed_mean` (h/s) | 0.26 | 0.28 | 0.20 | 0.36 (v1 0.40) |

All 21 XD test clips labelled B1 or B5 with first onset at least 30 s in (the same set as the audio
section). Version 2 features were re-derived from the version-1 `pose_raw` of these clips; the detections are
the same, and the clips have no flag for the GMC fix. Values are the mean change versus each video's own
`[t0 - 60, t0 - 30)` baseline (n = 19 videos with at least 3 baseline ticks). The last column is the
fraction of videos whose `[t0 - 10, t0)` mean is above baseline.

| Column | [-30, -10) | [-10, 0) | [0, 10) | up in [-10, 0) |
|---|---|---|---|---|
| `n_persons_mean` | +0.53 | +0.89 | +1.06 | 0.53 |
| `n_close_pairs` | +2.4 | +2.5 | +3.6 | 0.58 |
| `wrist_speed_mean` (h/s) | -0.02 | +0.08 (v1 +0.13) | +0.27 (v1 +0.44) | 0.53 (v1 0.68) |
| `body_speed_mean` (h/s) | -0.04 | +0.02 (v1 +0.05) | +0.10 (v1 +0.21) | 0.42 (v1 0.58) |
| `arm_raise_rate` | +0.035 | +0.062 | +0.083 | 0.68 |
| `arm_toward_other_max` (h) | +0.02 | +0.02 | +0.07 (v1 +0.14) | 0.47 |
| `new_tracks` | -0.5 | +0.6 | +2.1 | 0.58 |
| `max_pair_closing_vel` (h/s) | +0.003 | +0.003 | +0.015 (v1 +0.04) | 0.37 |
| `cam_translation` (W/s) | +0.009 | +0.030 (v1 -0.004) | +0.12 | 0.58 |
| `faceoff_frac` | -0.03 | -0.03 (v1 -0.09) | -0.01 | 0.37 |
| `mutual_facing_closest` | +0.01 | +0.02 (v1 -0.07) | -0.03 | 0.32 (v1 0.26) |
| On ticks with a pair in every frame (n = 10): `min_pair_dist_mean` (h) | -0.07 | -0.08 | -0.13 | 0.30 |
| same: `crowd_spread` | -0.15 | +0.00 | -0.04 | 0.50 |
| same: `mutual_facing_closest` | -0.02 | -0.09 (v1 -0.27) | -0.17 (v1 -0.26) | 0.20 (v1 0.10) |
| same: `faceoff_frac` | -0.03 | -0.09 (v1 -0.20) | -0.12 | 0.30 |

Reading:
- The features respond to violence: wrist and body speed, close pairs, arm raising, reach and track churn all
  rise at onset.
- Before onset, people gather (+0.5 to +0.9 persons) and come closer, arms are raised more, and gesturing
  speed rises a little in the last 10 s.
- **About half of the version-1 speed response was artefact.** Fights produce keypoint occlusion, and
  version 1 turned that flicker into speed. The onset response of `wrist_speed_mean` fell from +0.44 to
  +0.27 h/s and that of `body_speed_mean` from +0.21 to +0.10.
- **The "mutual facing falls before onset" finding of version 1 was mostly the saturated yaw proxy.**
  Over all ticks it is now flat (+0.02 in the last 10 s). On ticks with a pair in every frame it still
  falls, but by a third as much (-0.09 vs -0.27). It does not support the face-off hypothesis either way.
- n = 19 movie clips with no significance test. This shows the wiring is right; it is not evidence of 30 s
  anticipation. Test the leading-cue hypothesis on the curated real-world set once it is hand-annotated.

### Gotchas

- **All geometry is 2D.** People at different depths overlap in the image, so a street crowd reads as many
  "close pairs" (`min_pair_dist` about 0.15 to 0.2 h in the XD crowd clip). Normalizing by body height
  removes scale but not depth. Prefer within-video changes (slopes, demeaned values) over absolute levels.
- **Empty and low-resolution footage:** UCF Normal videos are often empty roads, which gives `valid` False
  everywhere. That is correct, so use `valid` as a mask. People in 320x240 CCTV are upscaled to 640 for the
  detector and still get low keypoint confidence. Check `kpt_conf_mean`.
- **Movie edits:** every cut ends tracks and starts new ones (`new_tracks` spikes). There is no explicit
  tracker reset at cuts. A spurious IoU match across a cut can create one large velocity step.
  `motion.cut_1s` flags those ticks.
- **GMC warps are noisy** in crowds, because RANSAC can lock onto the people. `cam_*` is a proxy; the `motion`
  extractor has a dense-flow camera estimate.
- **Body height is an estimate.** The cue ratios were calibrated on adults in the five smoke videos. Children
  and seated or crouching people make h smaller, so their normalized distances and speeds come out larger.
  A raised arm stretches the box, and h follows it only if the arm stays up for more than about half of the
  1 s median window.
- **Head yaw is a proxy.** It is continuous for near-frontal (both ears) and 3/4 (both eyes) views and a fixed
  +/-0.9 in profile. It is unknown without the nose (back views), so `mutual_facing_*` averages only the known
  terms.
- **Ultralytics tracker index bug, handled here.** The `idx` column of the tracker output indexes the high- or
  low-score subset, not the full detection list. `_map_tracks_to_dets` maps it back.
  `on_predict_postprocess_end` in ultralytics has this bug for second-stage matches.
- **Import torch before decord.** `pose.py` imports torch at module level for this reason (section 0.4).
- **License:** Ultralytics and the YOLO11 weights are AGPL-3.0. That is fine for research. A product would
  need RTMO via `rtmlib` (Apache-2.0) behind the same `pose_raw` format.
- `meta["video"]` records native and decoded sizes. Keypoints are always stored in native pixels.

---

## Motion (motion)

Written 2026-10-14 by the pose track; revised the same day after review (version 2). Code:
`anticipation/features/motion.py` (registry name `motion`, CPU only). It also hosts the shared helpers
`rolling_slope` (causal OLS slope over past ticks), `window_bounds`, `open_scaled` (decode `io.Video` at a
reduced size), `decode_frames` (batched decode with a per-frame fallback) and `frame_dt` (real time between
decoded frames). Self-test (CPU, 1 s): `.venv/bin/python -m anticipation.features.motion --selftest`.

Version history: 1 = first release. 2:
- A batch that fails to decode is retried frame by frame. Version 1 dropped the whole 200-frame (40 s) chunk,
  so one bad frame at 79.8 s invalidated ticks 41 to 80, i.e. it changed the past.
- Rates divide by the real time between the two decoded frames, not the nominal 0.2 s. A constant pan at
  24 fps measured 0.125 / 0.156 frame widths/s on alternating pairs in version 1, and a constant 0.150 now.
- Pairs that map to the same decoded frame are skipped.

### What is stored

`motion`, D = 30, `meta["window_sec"]` = 31 (short window 1 s, long window 5 s, slopes and time-since-cut up
to 30 s). `model_id` = `opencv-farneback`. `valid` = at least 2 decoded frames in `[t - 1, t]`.

Pipeline:
- Frames are sampled at 5 fps on the video clock and decoded directly at 160 px width (aspect kept) by decord.
  A frame that fails to decode is missing (`meta["n_frames_ok"]`); the pair chain restarts after it.
- For each consecutive pair (nominally 0.2 s apart; rates use the real gap `frame_dt`, 0.167 or 0.208 s at
  24 fps), on the grey image:
  - absolute frame difference;
  - Farneback dense flow (`pyr_scale` 0.5, levels 3, `winsize` 15, iterations 3, `poly_n` 5, sigma 1.2),
    expressed in frame widths per second so it is resolution-independent;
  - the median flow vector (camera pan proxy) and the residual flow after subtracting it (foreground motion);
  - an 8-bin magnitude-weighted direction entropy of the residual flow.
- **Shot cuts:** a pair is a cut when the HSV histogram Bhattacharyya distance is > 0.45 **and** the mean
  grey difference is > 0.06. The second test rejects flashing police lights, which pass the histogram test
  alone (seen on `yt/holdout_072`). Eight random flagged pairs from the rapidly edited Braveheart battle
  clip were all real cuts. **Flow and difference statistics skip cut pairs**; the cut columns report them.
- Columns:

| # | Column | Definition |
|---|---|---|
| 0 | `diff_mean` | mean abs(frame difference) (grey 0..1) over non-cut pairs in [t-1, t] |
| 1 | `diff_max` | max over pairs of the mean abs(frame difference) |
| 2 | `motion_area` | fraction of pixels whose grey level changes by > diff_pixel_thresh (pair mean) |
| 3 | `flow_mag_mean` | Farneback flow magnitude, frame widths per second (pair mean of the pixel mean) |
| 4 | `flow_mag_std` | pair mean of the pixel std of flow magnitude |
| 5 | `flow_mag_p90` | pair mean of the pixel 90th percentile of flow magnitude |
| 6 | `flow_mag_max` | max over pairs of the pixel 99th percentile of flow magnitude |
| 7 | `cam_flow_mag` | magnitude of the median flow vector (global translation, camera pan proxy), pair mean |
| 8 | `cam_shake` | std over pairs of the median flow vector (x and y combined); needs >= 2 pairs |
| 9 | `resid_flow_mean` | flow magnitude after subtracting the median flow vector (foreground motion), pair mean |
| 10 | `resid_flow_p90` | 90th percentile of the residual flow magnitude, pair mean |
| 11 | `flow_dir_entropy` | magnitude-weighted 8-bin direction entropy of residual flow (0..1) over moving pixels |
| 12 | `hist_dist_max` | max HSV-histogram Bhattacharyya distance between consecutive sampled frames |
| 13 | `cut_1s` | 1 if any pair in [t-1, t] is a shot cut |
| 14 | `brightness_mean` | mean grey level (0..1) of frames in [t-1, t] |
| 15 | `brightness_delta` | abs(brightness(last frame) - brightness(first frame)) in [t-1, t] |
| 16 | `contrast_mean` | mean grey-level std (0..1) of frames in [t-1, t] |
| 17 | `diff_mean_5s` | diff_mean over [t-5, t] |
| 18 | `flow_mag_mean_5s` | flow_mag_mean over [t-5, t] |
| 19 | `resid_flow_mean_5s` | resid_flow_mean over [t-5, t] |
| 20 | `cam_flow_mag_5s` | cam_flow_mag over [t-5, t] |
| 21 | `cuts_5s` | number of shot cuts in [t-5, t] |
| 22 | `flow_burst` | flow_mag_mean - flow_mag_mean_5s (sudden motion onset) |
| 23 | `time_since_cut` | seconds since the last detected cut, capped at since_cut_cap_sec (cap if none) |

Slope columns (24 to 29): `flow_mag_mean_slope{5,15,30}s`, `resid_flow_mean_slope{5,15,30}s` (key-major).
Units: flow in frame widths per second, grey levels 0 to 1.

### Causality, how to run, throughput

- Version 2 passes three checks with **max_abs_diff = 0.0** on all four smoke videos: XD fight, UCF
  Fighting003, UCF Normal_050 and `yt/holdout_072`.
  - `io.check_causal` with the cut at half the duration: 99, 50, 68 and 137 ticks compared.
  - Future perturbation: every frame after the cut is replaced by noise while the length stays the same.
    Ticks at or before the cut are compared: 100, 51, 69 and 138.
  - Decode-error locality: one undecodable frame at 0.8 x the cut (80.05, 40.85, 55.25 and 110.45 s). All
    ticks before it are identical and no tick loses validity.
  - The decode-error and real-dt fixes are also checked on synthetic videos by `motion.selftest()`.
- Run:

  ```bash
  .venv/bin/python -m anticipation.features.run_extract --extractor motion --manifest benchmark/videos.csv \
      --shard 0 --num_shards 8 --cpu_threads 1
  .venv/bin/python -m anticipation.features.motion --video VIDEO.mp4 --print_ticks 40:50 --check_causal 60
  ```

- Throughput (decode included): one process gives 22 to 32x realtime. Farneback is about 2.5 ms per pair
  and is about half the cost. More OpenCV threads do not speed it up, so parallelize with processes.
  8 single-threaded processes gave **94x aggregate** while other tracks were loading the machine (load
  average 42).
- Estimate: **the current 352 h manifest takes about 3.7 h on the CPU** at 94x. It competes with pose
  tracking for CPU, so run it after pose, or with fewer shards alongside it. Version 2 measured 30 to 34x per
  single-threaded process on the 21 XD sanity clips with the machine nearly idle.
- Size: about 0.07 KB per video-second.

### Sanity around onsets

21 XD B1/B5 clips, same protocol as the pose table: change versus the video's own `[t0 - 60, t0 - 30)`
baseline, n = 19. These are version 2 numbers; the real-dt fix moved the flow columns by at most 0.001.

| Column | [-30, -10) | [-10, 0) | [0, 10) | up in [-10, 0) |
|---|---|---|---|---|
| `flow_mag_mean` | +0.009 | +0.020 | +0.040 | 0.68 |
| `resid_flow_mean` | +0.007 | +0.019 | +0.038 | 0.68 |
| `cam_shake` | +0.008 | +0.009 | +0.017 | 0.79 |
| `diff_mean` | +0.012 | +0.021 | +0.051 | 0.79 |
| `cuts_5s` | -0.20 | -0.36 | +0.52 | 0.37 |

What the table shows:
- Motion roughly doubles after onset.
- Motion and camera shake rise modestly in the last 10 to 30 s.
- Cuts become rarer just before onset (the shot holds on the build-up) and more frequent after it.
- On the single XD street-fight clip, flow went 0.047 -> 0.085 -> 0.096 over `[o-20, o-5)`, `[o-5, o)` and
  `[o, o+10)`.
- On UCF Fighting003 (static CCTV), `cam_flow_mag` stays at 0 and flow only rises after onset (0.008 -> 0.021).

### Gotchas

- **Provenance shortcut risk.** Camera shake, cut rate and brightness tell phone video from CCTV from movies.
  Phone filmers start shaking because they expect a fight. Report CCTV and phone footage separately
  (research/README.md) and include these columns in shortcut audits.
- **Small decode size.** 160 px and 5 fps miss small, fast motions (fists) but capture global agitation.
  Fine-grained motion is the job of V-JEPA 2 and pose.
- `time_since_cut` saturates at 30 s and is 30 on footage without cuts (CCTV).
- Letterboxed movies: black bars have zero flow and lower `flow_mag_mean` uniformly; use per-video
  normalization.

---

## Bulk extraction run v1

Written 2026-10-14 by the extraction-ops track. Launched 2026-10-14 21:05 with `tools/extract_queue.sh start`;
stopped and resumed at 21:42 (no outputs lost) to deploy the review fixes: all crashed videos are skipped,
a shard stops after 20 failures in a row, and 27 cuts were re-made. Stopped and restarted a second time at 21:58
(2 s stop, no outputs lost) with the manifest rows re-ordered so that clean negatives exist early
([Run order and the early negative sample](#run-order-and-the-early-negative-sample)).
Code: `tools/build_extract_manifest.py` (manifest and 300 s cuts), `tools/extract_queue.sh` (queue),
`run_extract.py --keep_order` and the crash marker (section 0.2).

### Manifest (`benchmark/extract_manifest_v1.csv`)

Built from `benchmark/videos.csv` (sha256 `5a8f4ad2c38b...`, the catalog that `folds_v2.json` was built from;
`folds_v2.json` already contains all 973 UBI-Fights videos, so no new fold file was needed):

```bash
nice -n 10 .venv/bin/python tools/build_extract_manifest.py      # about 13 min, 3 parallel ffmpeg cuts
```

- **Included roles:** `violent`, `holdout` and `clean_negative` from every source (xd, ucf, ubi, yt).
  **Excluded:** `other_anomaly` (1388), `unlabelled` (48), `duplicate` (28), `contaminated` (1): 1465 videos,
  43.2 h.
- **Columns:** `video_id, source, source_kind, role, priority, eligible30, has_audio, path, duration_sec, fps,
  num_frames, orig_path, orig_duration_sec, cap_sec`. The runner uses `video_id`, `path` and `source`.
- **Order (run order, one row per video):** `violent_eligible30`, then an early clean-negative sample, then the
  other `violent`, then `holdout`, then the remaining `clean_negative`. Inside each group rows are sorted by
  `sha256("extract_v1:" + video_id)`, which is deterministic and mixes the sources and folds. With `--keep_order`
  every shard works through the manifest from the top and skips finished outputs. The `priority` column keeps its
  four values: the early sample is the `clean_negative` block right after `violent_eligible30` (rows 469 to 1555).

| Run order | `priority` | Videos | Hours |
|---|---|---|---|
| 1 | `violent_eligible30` | 468 | 33.1 |
| 2 | `clean_negative`, early sample | 1087 | 41.8 |
| 3 | `violent` (other violent) | 1661 | 39.3 |
| 4 | `holdout` (curated YouTube) | 92 | 10.0 |
| 5 | `clean_negative`, the rest | 2923 | 115.1 |
| | **Total** | **6231** | **239.3** (xd 155.6, ucf 42.3, ubi 31.5, yt 10.0) |

Clean negatives are 4010 videos and 156.9 h (303.5 h before the 300 s cap).

- **Real audio** (`has_audio`, a non-silent track): 4303 videos, **175.7 h** (xd 155.6, ucf 10.3, yt 9.8;
  UBI has none). The audio steps read a filtered copy, `logs/extract/queue/manifest_audio.csv`, that the
  queue writes at start. A video without an audio feature file therefore means "no real audio" (`has_audio`
  False). It does not mean "not extracted".
- **300 s cap on clean negatives.** 655 clean negatives are longer than 300 s (xd 383, ucf 138, ubi 134;
  201.2 h, up to 9 h each). They are cut to their first 300 s (54.6 h). Violent and holdout videos are never
  cut (the longest is 5721 s).
  - Command: `ffmpeg -i <orig> -map 0 -c copy -t 300` into
    `data/cache/trunc300/<video_id>.mp4` (22 GB). `-map 0` keeps all
    streams, including the UCF PNG cover stream and the XD data stream.
  - 27 XD movie cuts made that way gained a second data stream (`gpmd`, copied from the XD chapter track)
    and a container duration of 310 to 1229 s. They were re-cut on 2026-10-14 with
    `-map 0 -map -0:d` (the builder's current command) before any feature was extracted from them. The muxer
    still writes the chapter list as one text data stream, so all 655 cuts now have the same stream types as
    their originals and a container duration of at most 300.25 s. The extractors take the duration from
    decord in any case.
  - Stream copy is the same cut that `io.check_causal` uses. The decoded frames before the cut are
    bit-identical, so ticks 1 to 299 of a capped video equal those of the full video. Tick 300 is a boundary
    tick that can differ slightly: the copied audio ends at 299.95 s and the last frames differ at the cut.
    Checked with `audio_dsp` and `motion` on `xd/Hear.Me.2009__#01-09-30_01-16-59_label_A` (449 s): max
    difference 0.0 on ticks 1 to 299, small differences on tick 300 only, identical `valid`.
  - The manifest `path` points to the cut file. `cap_sec` = 300, and `orig_path` / `orig_duration_sec` keep
    the original. `duration_sec` and `num_frames` describe the cut file.
  - `video_id` is unchanged. The features of a capped negative are at the normal path and have T = 300.
  - Verified on all 655 cuts:
    - `io.Video` duration 300.00 to 300.08 s, with the same fps as the original;
    - decoded frames at 0, 150 and 299 s identical to the original;
    - an audio stream exactly when the original has one;
    - for the 407 cuts with real audio, the decoded audio is 300 s +/- 1 s.
- **Evaluation caveat (open).** `anticipation.eval.va30` takes each video's duration from `videos.csv` and
  requires a score for every tick. A model built on these features has no ticks after 300 s for the 655 capped
  negatives. Do not use `--allow_missing` for this case: it sample-and-hold fills the gap, which extends the
  last score over up to 9 h. The evaluator needs a way to score clean negatives on their first `cap_sec`
  seconds (clean hours then become 156.9 h instead of 303.5 h). The alternative is to extract the rest of those
  videos later; that is resumable with a manifest that points back to `orig_path`, after deleting the 300-tick
  files.

### Run order and the early negative sample

The trainer cannot set false-alarm thresholds until clean negatives exist in the validation slice and test fold
of every fold of `benchmark/folds_v2.json`. With the first order (violent, holdout, then negatives) they would
have appeared only after about 80 h of video. The early sample moves the part of them that covers every
fold early.

- **Rule** (`early_negative_ids` in `tools/build_extract_manifest.py`): a clean negative belongs to the cell
  (fold, `test`, source) of its test fold and to the cell (f, `val`, source) of every fold f whose validation
  slice holds it. Source is `xd`, `ucf` or `ubi`; that gives 5 folds x 2 x 3 = 30 cells. Clean negatives are taken
  in hash order, and one is kept while it helps a cell that holds less than 25 % of its hours (`--early_negative_frac
  0.25`). Every cell ends with at least 25 % of its hours (smallest 25.1 %; largest 43.7 %, `ubi` in the validation
  slice of fold 2), and the sample is 26.6 % of the clean-negative hours.
- **Size:** 1087 videos, 41.8 h: xd 635 (27.0 h), ucf 253 (8.2 h), ubi 199 (6.6 h). 697 of them (28.9 h) have real
  audio, which is all of xd. Videos are capped at 300 s, so a cell cannot overshoot by more than 0.083 h.
- **Order inside the sample** is the hash order, so a partly finished sample is spread over the cells.
- **Reproduce** (rows, paths, caps and columns are unchanged; only the row order differs; the cut files are not
  touched):

```bash
.venv/bin/python tools/build_extract_manifest.py --reorder_only            # re-order the existing manifest, seconds
nice -n 10 .venv/bin/python tools/build_extract_manifest.py                # full rebuild, default --early_negative_frac 0.25
```

  Both give the same order (`--early_negative_frac 0` gives the first order). Running `--reorder_only` on its
  own output changes nothing. The new manifest sha256 starts `f531a43df4c0`.
- **Applied** with the queue's own `stop`, a manifest swap and `start` (nothing in `extract_queue.sh` changed). The
  stop took 2 s: every shard got SIGTERM, abandoned its in-flight video through `KeyboardInterrupt` and removed its
  `.current` marker, so no video was recorded as crashed. Outputs are written atomically (`.tmp.npz`, then rename),
  so finished outputs were kept and skipped: `visual` skipped 423 videos (208 + 215), `pose` 608 (210 + 226 + 172),
  equal to the finished outputs (the 424th `visual` file is `curated/holdout_004`, which is not in the manifest).
  One `pose` video (`ucf/Robbery085_x264`) had `pose_raw` but not `pose_interact` when it was interrupted and is
  redone. The `audio` manifest the queue derives at start was rebuilt from the new order.
- **What is not re-ordered:** the steps inside a lane. `audio_events`, `voice` and `speech_tox` still start when
  `visual` has finished all 6231 videos, and `motion` and `audio_dsp` when `pose` has. Visual and pose features
  for the sample are therefore early; audio and motion features for it come after those two passes.

ETA, measured after the restart (aggregate of the per-process speeds in the shard logs):

@@ETA@@

### Lanes

All processes run under `nice -n 10`, pinned to cores 0-11 by `taskset`, so the 75 % cap does not depend on
`tools/cpu_governor.sh` running. Threads are capped with `OMP/MKL/OPENBLAS/NUMEXPR/NUMBA_NUM_THREADS` and
`--cpu_threads` (2 per GPU process, 1 per CPU process). The visual decoders keep their own 4 decord threads.

| Lane | Steps, in order | Processes | Videos | Speed (x realtime, aggregate) | ETA |
|---|---|---|---|---|---|
| `gpu0` (cuda:0, may run at 100 %) | `visual` (`vjepa2_l` + `siglip2` + `siglip2_prompts`, one decode) | 2 | all 6231 (239.3 h) | **34.6 measured** | 6.9 h |
| | `audio_events` (`ast_events` + `ast_emb` + `clap_prompts`, one pass) | 2 | real audio, 4303 (175.7 h) | about 120 planned | 1.5 h |
| | `voice` (`voice_affect` + `e2v_emb`) | 2 | real audio | about 90 planned | 2.0 h |
| | `speech_tox` | 2 | real audio, sources xd, yt, ucf | 12 planned (5 to 25 in the speech section) | 14.6 h (7 to 35) |
| `gpu1` (cuda:1, light) | `pose` (`pose_raw` + `pose_interact`) | 3 | all 6231 | **43.0 measured** | 5.6 h |
| `cpu` (starts when `gpu1` is done) | `motion` | 4 | all 6231 | about 110 (4 x 28 in the smoke test) | 2.2 h |
| | `audio_dsp` | 4 | real audio | about 52 (4 x 13 in the smoke test) | 3.4 h |

Planned rates come from the extractor sections and from the queue smoke test (one video per shard). `status`
replaces them with measured rates once a step is running.

Measured 7 minutes after launch, with the repo-cleanup jobs of another session also running:
- Speeds as in the table.
- The queue used 7.9 cores (instantaneous `top`); the machine was 58 to 66 % busy over all users.
- GPU memory: cuda:0 10.6 GB, cuda:1 10.4 GB.
- 0 failures.
- Each visual process runs at about 17x and each pose process at about 14x.

**ETA:**
- `gpu1`: done after about 5.6 h (2026-10-15 about 02:45).
- `cpu`: done after about 11 h (about 08:15).
- `gpu0`: done after about 25 h (2026-10-15 about 22:00), with a range of 17 to 45 h. Speech speed sets it,
  because speech slows down under CPU contention.
- Re-measured 8 minutes after the 21:42 resume (another session's VLM smoke test and pytest also running;
  machine 69 % busy, queue about 7.4 cores, cuda:0 10.6 GB, cuda:1 19.0 GB with that other job): visual 35.4x,
  pose 38.0x, 0 failures. New ETAs: `gpu1` about 03:20, `cpu` about 09:15, `gpu0` still about 22:00 on
  2026-10-15.
- Order of results: the eligible30 videos (33.1 h of video) have visual and pose features after about one
  hour. Every later step also processes them first, then the early negative sample, then the rest in manifest
  order.

Speech uses about 26 GB of GPU memory per process once the MT model is loaded, so about 52 GB on cuda:0 during
that step. While `visual` runs, cuda:1 carries only the pose detector (about 10 GB, low utilization).

### Commands

```bash
cd .
tools/extract_queue.sh start    # detached (setsid nohup); refuses to start twice
tools/extract_queue.sh status   # lane states, done / total videos and hours, failures, measured speed, ETA per step and lane
tools/extract_queue.sh stop     # SIGTERM to the process group (clean stop), SIGKILL after 60 s
.venv/bin/python tools/server_load.py --gpu   # machine load and GPU memory
# smoke test of the whole queue into a scratch root (1 video per shard, about 2 min)
EXTRACT_LIMIT=1 EXTRACT_FEATURE_ROOT=/tmp/x/f EXTRACT_LOG=/tmp/x/log tools/extract_queue.sh start
```

Other environment overrides: `EXTRACT_MANIFEST`, `EXTRACT_CORES` (default `0-11`), `EXTRACT_STALL_SEC` (default
5400), `EXTRACT_RETRY_SLEEP` (default 600), `EXTRACT_RETRY_CRASHED` (set to 1 to retry crashed videos).

Do not edit `tools/extract_queue.sh` in place while the queue runs: bash reads the script file lazily, so a
running queue can execute a mix of old and new text. Stop it, edit, and start it again (or replace the file
by rename, which keeps the running process on the old inode).

### Logs and status

Everything is under `data/logs/extract/`:
- `<lane>_<step>_shard<i>.log`: runner stdout and stderr, one line per video with the running speed. Each
  attempt starts with a `=== <time> attempt k` line and ends with `=== <time> exit <rc>`.
- `<extractor>/shard<i>of<n>.jsonl` and `<extractor>/failures.jsonl`: the runner's records (section 0.2).
- `queue/status.log`: every step start and finish. `queue/<lane>.status` holds the current state of each lane.
  `queue/queue.pgid` holds the process group. `queue/queue.log` holds the queue's own stderr: empty, except
  for bash's one-line notice when a shard dies from a signal (for example `Segmentation fault`).

### Failure handling and resume

- **Python errors:** a video that raises is logged to `failures.jsonl`, and the shard continues.
- **Native crashes:** a shard that exits with a signal or a status above 1 is restarted, up to 3 attempts. The
  runner's `.current` marker makes the restart skip the video it died on, and every run skips all videos
  with a crash record (section 0.2). They show up in the `fail` column of `status`. To retry them once the
  queue is done: `EXTRACT_RETRY_CRASHED=1 tools/extract_queue.sh start`.
- **Systemic failures:** a shard with 20 failures in a row exits with status 3 (section 0.2). The queue waits
  `EXTRACT_RETRY_SLEEP` seconds (default 600) and restarts it, up to 3 attempts; after that the shard gives
  up and the lane moves on. The likely case is `speech_tox` not fitting next to another user's job on cuda:0
  (it needs about 52 GB). Check `status` when speech starts, and re-run `start` when the GPU is free.
- **Stalls:** a watchdog kills a shard whose log has not changed for 90 minutes (`EXTRACT_STALL_SEC`), and
  the shard is restarted the same way. 90 minutes is far longer than any single video takes; the slowest case
  is a 1.6 h video in `speech_tox`.
- **Resume:** `tools/extract_queue.sh start` resumes after a stop, a crash or a reboot. Each lane restarts at
  its first step, and finished steps exit within seconds because every output is skipped. pose, motion and
  speech also version-check their outputs.
- **Retrying failures:** re-running retries failed videos. To skip them instead, run that step by hand with
  `--skip_failed`.
- **Changing process counts:** stop the queue, edit `STEPS` in the script, and start again. Outputs are per
  video, so a different shard count is safe once the old processes are gone.
