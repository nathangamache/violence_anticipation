# Realtime inference, YouTube analysis, and miscellaneous root files

> Paths in this file describe the tree before the 2026-10 cleanup. Surviving code is under `legacy/<same path>`;
> the honest runs are under `baselines/reference_runs/`; removed tracked files: `git show pre-cleanup:<path>`;
> ignored weights and arrays: `data/archive/repo_legacy/<same path>`
> until 2027-01-12. See [../REPO_CLEANUP_PLAN.md](../REPO_CLEANUP_PLAN.md).

Scope: `realtime/` (pipeline.py, rtsp_inference.py, inspect_checkpoint.py), `scripts/analyze_youtube.py`,
`mediamtx.yml`, `progress.sh`, `requirements.txt`, the root-level inference logs, the `probe_clip_01_*` artifacts,
`analysis_probe_clip_01.png`, `test_video.mp4` (deleted in the 2026-10 cleanup as a byte-identical duplicate of
`data/anticipation_curation/videos/holdout_025.mp4`), and the installed `.venv` packages. All paths are relative to the repo root
`.` unless absolute.

Everything below was checked against the code and files, or measured with cheap CPU-only scripts (no GPU, no training).
Claims marked **(inferred)** are reasoned but not directly verified.

---

## 0. TL;DR for the next agent

- **Only one deployment path works end to end: `realtime/rtsp_inference.py`.** It runs RTSP capture (1 fps), then Qwen2.5-VL-7B
  frame embeddings (3584-d) on `cuda:0`, then a BiLSTM+attention classifier over the last 32 embeddings on `cuda:1`, then a threshold alert.
  It ran live twice (Feb 26 and Mar 2, 2026) against a stream named `macbook`. It keeps up in real time: about 0.35 s per VLM embedding and about 2 ms per LSTM call.
- **`scripts/analyze_youtube.py`** is the offline equivalent for a YouTube URL or local file. It produced `probe_clip_01_embeddings.npy`,
  `probe_clip_01_scores.npy` and `analysis_probe_clip_01.png`. **Verified:** the scores were produced by
  `outputs/outputs_temporal/best_temporal_model.pt`. Rerunning it on CPU from the saved embeddings matches to a max abs diff of 2.5e-5.
- **`realtime/pipeline.py` is dead and broken.** It is the legacy graph/ViolenceTransformer + LLM-fusion pipeline. It uses a randomly
  initialised model, has shape and type mismatches, and has no `__main__`. Do not use it. Its test fails as well.
- **There is serious train/serve skew in every deployment path** (Section 6). The embedding recipe (prompt, context frames, pooling),
  the activation function (ReLU vs GELU), the time base (training embeddings are about 4 to 6.6 s apart, deployment samples at 1 s)
  and the sequence length (32 vs 30) all differ from training. Any accuracy number from `training/` does not carry over to these
  deployment scripts.
- **The real-world evidence is thin.** The RTSP logs have no ground truth. One YouTube video has hand-written ground truth (in a docstring),
  and the model separates it only weakly: frame AUC 0.72, and the highest score falls *after* the fight. Details are in Section 7.
- `requirements.txt` only covers `rtsp_inference.py`, and its transformers lower bound is wrong. Section 9 has a corrected, import-derived list.

---

## 1. File inventory

| Path | Size / shape | What it is | Status |
|---|---|---|---|
| `realtime/rtsp_inference.py` | 635 lines | Threaded live pipeline: RTSP, then Qwen2.5-VL embeddings (GPU0), then BiLSTM (GPU1), then alerts | Works. Has the skew bugs in Section 6 |
| `realtime/inspect_checkpoint.py` | 82 lines | CLI: prints a checkpoint's weight shapes and infers LSTM dims | Works (CPU) |
| `realtime/pipeline.py` | 136 lines | Legacy per-frame YOLO/VideoMAE/CLIP graph to `ViolenceTransformer` + Qwen2.5-3B text LLM fusion | Broken / dead |
| `realtime/__init__.py` | empty | package marker | |
| `scripts/analyze_youtube.py` | 634 lines | Offline: yt-dlp download, then 1 fps frames, QwenTeacher embeddings, temporal LSTM scores, plot | Works. Has the skew bugs in Section 6 |
| `inference_20260226_225311.log` | 1000 lines | 14-min live RTSP session log (written by rtsp_inference.py) | Evidence |
| `inference_20260302_185137.log` | 117 lines | 1m51s live RTSP session log | Evidence |
| `logs/inference_20260226_151941.log` | 32 lines | First live attempt. Never connected (auth URL on port 554) | Evidence |
| `probe_clip_01_embeddings.npy` | float32 (197, 3584), rows L2-normalised (norm = 1.0) | QwenTeacher embeddings of YouTube video `probe_clip_01` at nominal 1 fps | Reused by `continuation/lda_calibration_complete.py`, `continuation/domain_adaptation.py` |
| `probe_clip_01_scores.npy` | float32 (197,), range 0.5148 to 0.7674, mean 0.661 | Per-second scores from `outputs/outputs_temporal/best_temporal_model.pt` | Evidence |
| `analysis_probe_clip_01.png` | 2100x900 PNG | Plot of the scores with threshold 0.65 (Section 7.3) | Evidence |
| `mediamtx.yml` | 8 lines | Config for the MediaMTX RTSP server: path `macbook`, port 554, placeholder creds | Superseded (Section 8) |
| `progress.sh` | 8 lines | Counts `vlm_emb_violence` vs `vlm_emb` npz files and `vlm_labels_violence` vs `vlm_labels` jsonl files | Utility |
| `requirements.txt` | 6 packages | Minimal deps for rtsp_inference only | Incomplete (Section 9) |
| `test_video.mp4` | 416.96 s, 640x360 H.264 29.97 fps + AAC 44.1 kHz stereo, 20.8 MB | Not referenced by any code. Brand `mp42`, creation_time 2023-06-10, so probably a YouTube download (inferred) | Orphan test asset. Deleted in the 2026-10 cleanup as a byte-identical duplicate of `data/anticipation_curation/videos/holdout_025.mp4` |
| `test_holdout_025.mp4.part` / `.ytdl` | 0 bytes / 50 bytes | Aborted yt-dlp download (fragment index 0) from Mar 22 2026 | Junk (gitignored) |
| `snapshots/` | empty | Default `--snapshot_dir` of rtsp_inference. Always empty because of a bug (Section 10, item 6) | |

---

## 2. `realtime/rtsp_inference.py`: the live pipeline

### 2.1 Architecture

```
RTSP stream (cv2.VideoCapture, CAP_FFMPEG, BUFFERSIZE=1)
   |  Thread "RTSPCapture": reads every native frame, forwards one per 1/fps seconds of WALL-CLOCK time
   v
frame_queue (Queue maxsize=60; capture also drops oldest when qsize >= 30)
   |  Thread "EmbeddingExtractor" on cuda:0: Qwen2_5_VLForConditionalGeneration fp16
   |    BGR->RGB, resize to 448x448 (LANCZOS), chat template [image, "Describe the scene."], add_generation_prompt=False
   |    forward(output_hidden_states=True); embedding = hidden_states[-1][0].mean(dim=0)  -> (3584,) float32 CPU
   v
embed_queue (Queue maxsize=120)
   |  Thread "LSTMInference" on cuda:1: deque(maxlen=sequence_length=32)
   |    once the deque is full, runs on EVERY new embedding: ViolenceLSTM(1,32,3584) -> sigmoid prob
   |    is_alert = score >= threshold (0.5); alert callback only if (now - last_alert) > cooldown (30 s)
   v
AlertHandler.on_alert(pred): logs a WARNING banner, appends to alert_log, (never) saves a snapshot
```

Main thread: `ViolenceAnticipationPipeline.run_forever()` starts the LSTM thread, then the embedder, sleeps 2 s, then starts capture.
It logs `[Stats]` every 30 s. SIGINT and SIGTERM call `stop()` and `sys.exit(0)`.

Data containers: `Frame(image BGR ndarray, timestamp, frame_idx)`, `EmbeddedFrame(embedding (3584,), timestamp, frame_idx)`,
`Prediction(score, is_alert, timestamp, frame_idx, lead_time_budget)`. `frame_idx` counts *native* stream frames, so it advances
about 30 per sample at 30 fps.

`ViolenceLSTM` (lines 63-127) has the same parameter names and shapes as `TemporalViolenceModel` in `training/train_temporal_model.py`:

- `frame_encoder`: Linear(3584 to 256), LN, act, Dropout, Linear(256 to 256), LN, act, Dropout
- `lstm`: BiLSTM with hidden 128, 2 layers
- `attention`: Linear(256 to 64), Tanh, Linear(64 to 1), softmax over time
- `classifier`: Linear(256 to 64), act, Dropout, Linear(64 to 1)

It differs from training in two ways. First, **act is ReLU here but GELU in training**. Second, it appends `nn.Sigmoid()` (training
outputs logits). See Section 6.

### 2.2 How to run

```bash
# 1) An RTSP server must be publishing the camera. Past runs used rtsp://127.0.0.1:8554/macbook (MediaMTX default port).
# 2) Run from the repo root (the log file is created in the CWD at import time):
.venv/bin/python realtime/rtsp_inference.py \
    --rtsp rtsp://127.0.0.1:8554/macbook \
    --vlm_checkpoint models/Qwen2.5-VL-7B-Instruct \
    --lstm_checkpoint outputs/outputs_temporal/best_temporal_model.pt \
    --fps 1.0 --sequence_length 30 --threshold 0.5 --alert_cooldown 30 --frame_size 448 --snapshot_dir snapshots
```

CLI flags (`parse_args`, line 592):

| Flag | Required | Default |
|---|---|---|
| `--rtsp` | yes | |
| `--vlm_checkpoint` | yes | HF dir or hub id |
| `--lstm_checkpoint` | yes | |
| `--fps` | no | 1.0 |
| `--sequence_length` | no | 32. Training used 30, so pass 30 |
| `--threshold` | no | 0.5 |
| `--alert_cooldown` | no | 30.0 |
| `--frame_size` | no | 448 |
| `--snapshot_dir` | no | `snapshots` |

Requirements:

- **Two CUDA devices.** `cuda:0` and `cuda:1` are hard-coded at lines 523 and 531. The LSTM is tiny and could share GPU0.
- **Qwen2.5-VL-7B-Instruct weights.** Local copy: `models/Qwen2.5-VL-7B-Instruct` (repo symlink `models/`).
  It is loaded with `dtype=torch.float16, device_map="cuda:0", trust_remote_code=True`. The `dtype=` kwarg needs a recent transformers
  (4.57.5 is installed); older versions only knew `torch_dtype` (inferred from the transformers changelog).
- **An LSTM checkpoint whose `frame_encoder.0.weight` is (256, 3584)**, hidden 128, 2 layers. The loader takes either a dict with
  `model_state_dict` (plus an optional `model_config`, which no training script writes) or a raw state dict.

Compatible checkpoints (verified by loading on CPU):

| Checkpoint | Trained on (args.embeddings_dir) | Stored test ROC / val ROC |
|---|---|---|
| `outputs/outputs_temporal/best_temporal_model.pt` | `data/processed/vlm_emb` | 0.991 / 0.577 |
| `outputs/outputs_temporal_old/best_temporal_model.pt` | `data/processed/vlm_emb` | 0.957 / 0.600 |
| `outputs/outputs_temporal_old_emb/best_temporal_model.pt` | `data/processed/vlm_emb_old` | 0.947 / 0.665 |
| `outputs/outputs_temporal_violence/best_temporal_model.pt` | `data/processed/vlm_emb_violence` | 0.912 / 0.533 |
| `outputs/outputs_temporal_vlm_emb/best_temporal_model.pt` | `data/processed/vlm_emb` | 0.893 / 0.575 |
| `experiments/output/exp3_cross_dataset/{xd_only,ucf_only,combined}/best_temporal_model.pt` | (Apr 2026, after the live runs) | n/a |

The gap between test ROC (about 0.9 to 0.99) and val ROC (about 0.53 to 0.67) is itself a red flag about the training and evaluation
split. See the training docs.

NOT compatible:

- Fused, audio and projected checkpoints (`frame_encoder.0.weight` input dims 4124, 3606, 3675, 91, 256) fail with a size-mismatch error.
- **The root-level MLPs (`anticipation_mlp.pt`, `ucf_crime_mlp.pt`, `best_synthetic_anticipation.pt`, ...) have `net.*` keys.**
  Because of `load_state_dict(strict=False)` (line 403) they load with *zero* matching keys and **silently run a randomly
  initialised LSTM**.

The live logs do not record which checkpoint was used. `outputs/outputs_temporal/best_temporal_model.pt` is the most likely one,
since it is the analyze_youtube default and was the newest 3584-d model at the time (inferred).

### 2.3 Outputs

- stdout plus `inference_<YYYYmmdd_HHMMSS>.log` in the **current working directory**. The FileHandler is created at module import
  (line 34), so merely importing the module creates a log file. That is why logs sit in the repo root.
- One `[LSTM] frame=NNNNNN | score=0.xxxx | alert=YES/no | inference=X.Xms` line per inference. (Older logs used emoji markers instead
  of YES/no. The code was de-emojified on 2026-10-14.)
- An alert banner, with `[Stats]` every 30 s and `[Capture]` every 60 samples.
- An in-memory `alert_log`, printed by `summary()` on shutdown. Nothing is persisted except the log. No scores file and no snapshots.

---

## 3. `scripts/analyze_youtube.py`: offline YouTube / file analysis

Pipeline (`main`, line 514):

1. `--url`: `download_youtube_video()` (yt-dlp Python API). It prefers MP4 at 720p or below with the android/web player clients. It
   limits the download to the first `--max-duration` seconds (default 300) for long or live videos, using ffmpeg as an external
   downloader. Temp files come from `tempfile.mktemp`.
   `--video`: a local file.
   `--download-only` stops after the download.
2. `extract_frames(video, fps)`: keeps every `int(video_fps / fps)`-th frame as RGB, **all held in RAM**.
3. `load_vlm_model()`: tries the project's `vlm.qwen_teacher.QwenTeacher` with `prompt_mode="violence_embedding"`,
   `context_mode="none"`, bf16 and sdpa attention. `generate_embeddings()` then calls `teacher.infer_embedding_only(PIL)` per frame.
   That returns an image-token mean-pooled, **L2-normalised** 3584-d vector built from a system prompt plus the violence JSON prompt.
   If QwenTeacher fails, it falls back to **Qwen2.5-VL-3B**: hidden 2048, with mean, max and earlier-layer means concatenated and
   truncated or zero-padded to 3584. That output is meaningless for the trained heads, and the script only prints a warning.
4. `load_model()`: reads `checkpoint['model_state_dict']`, infers dims, and builds the GELU `TemporalViolenceModel` with strict loading.
   `sequence_length` comes from `checkpoint['args']` (30).
5. `run_inference()`: **left-pads the embedding sequence with 29 zero vectors**, slides a 30-frame window by one, and applies sigmoid,
   giving one score per sampled frame.
6. `print_alerts()` prints contiguous above-threshold regions. `plot_results()` saves a PNG (`--plot`, default `analysis_plot.png`).
   `--save-scores` and `--save-embeddings` write `.npy` files. `--embeddings` reuses saved embeddings, but `--url` or `--video` is still
   mandatory.

Example that reproduces the saved artifacts from the stored embeddings (CPU-safe, no VLM needed):

```bash
.venv/bin/python scripts/analyze_youtube.py --video data/anticipation_curation/videos/holdout_025.mp4 \
    --embeddings probe_clip_01_embeddings.npy --model outputs/outputs_temporal/best_temporal_model.pt \
    --device cpu --threshold 0.65 --plot /tmp/x.png --save-scores /tmp/s.npy
```

Full run with the local VLM:

```bash
.venv/bin/python scripts/analyze_youtube.py --url "https://www.youtube.com/watch?v=probe_clip_01" \
    --vlm-model models/Qwen2.5-VL-7B-Instruct --device cuda:0 --threshold 0.65 \
    --save-embeddings probe_clip_01_embeddings.npy --save-scores probe_clip_01_scores.npy --plot analysis_probe_clip_01.png
```

The default `--vlm-model` is the hub id `Qwen/Qwen2.5-VL-7B-Instruct`, which needs a network connection or an HF cache. Prefer the local path.

---

## 4. `realtime/pipeline.py`: legacy, non-functional

It was meant to run, per frame: `core.extractors.live_track` (YOLOv8n, track ids are per-frame indices and not stable),
`live_actions` (VideoMAE-base on a single crop replicated 16x), `live_audio` (sounddevice mel mean, or zeros), and `live_scene`
(CLIP ViT-B/16). It would then build a graph dict, keep a buffer of up to 120 frames, and start scoring after 30. It would score with
`ViolenceTransformer` plus an optional `LLMEmbedder("Qwen/Qwen2.5-3B-Instruct").predict_risk(graph_to_text(...))`, fused by
`LLMFusionHead(mode="linear")`.

Why it cannot work:

- Everything heavy happens at import time: `ViolenceTransformer().cuda()`, Qwen2.5-3B download and load, plus YOLO, VideoMAE and CLIP via
  `core.extractors`. Importing it, as `scripts/benchmark.py` and `tests/test_realtime_graph.py` do, loads about 4 models.
- `ViolenceTransformer()` is never given a checkpoint, so its weights are random. `LLMFusionHead` is also untrained (weights [0.7, 0.3]).
- `build_live_graph` returns `nodes` as a *list of dicts* and numpy `scene`/`audio`. `ViolenceTransformer.forward` calls `f["nodes"].to(device)`,
  which raises AttributeError. The model returns `[T]` logits, so `.item()` would fail for T > 1 anyway.
- `graph_to_text(latest_graph)` passes a dict where `core/graph.py:161` expects a list (`graph_seq[-max_frames:]`), which raises TypeError.
- `live_audio` returns 128-d (librosa default n_mels) when a mic exists, but the model's `audio_dim` is 64.
- `LLMEmbedder._extract_number` regexes the *whole decoded text including the prompt*. The prompt ends in "(0.0-1.0):", so if the LLM
  emits no number the score becomes **1.0**.
- There is no `if __name__ == "__main__"` and no argparse, although README section 8 documents `python realtime/pipeline.py --source ...`.
  Run from the repo root, the import of `core` would also fail because of sys.path.
- `cv2.imshow` needs a display, and the server is headless.
- `tests/test_realtime_graph.py` asserts `graph["nodes"].shape == (2, 770)` and an `edges` key, neither of which exists, so the test fails.

Treat `realtime/pipeline.py`, `scripts/benchmark.py` and `tests/test_realtime_graph.py` as dead code from the Nov 2025 graph era.

---

## 5. `realtime/inspect_checkpoint.py`

`.venv/bin/python -B realtime/inspect_checkpoint.py <ckpt.pt>` runs on CPU. It prints every weight shape, then infers `input_dim`,
`encoder_dim`, `lstm_hidden`, `bidirectional`, `num_layers` and `attention`, then prints suggested `ViolenceLSTM(...)` args. It supports
`frame_encoder.0.weight` or `feature_extractor.net.0.weight` naming, and a wrapped `model_state_dict` or a raw state dict.

Output for `outputs/outputs_temporal/best_temporal_model.pt`: input 3584, encoder 256, hidden 128, bidirectional, 2 layers, attention.

Limitation: shapes cannot reveal activation functions. The former header comment in rtsp_inference.py ("Confirmed architecture from
inspect_checkpoint.py ... ReLU") is how the ReLU/GELU mismatch slipped in (inferred). The script also ignores `args`, `val_metrics`
and `test_metrics` in the checkpoint.

---

## 6. Train/serve skew: why deployment scores do not match training metrics

| Aspect | Training embeddings `data/processed/vlm_emb` (used by `outputs_temporal`) | `analyze_youtube.py` | `rtsp_inference.py` |
|---|---|---|---|
| Producer | `scripts/vlm_label_dataset.py` via `QwenTeacher.infer_one` | `QwenTeacher.infer_embedding_only` | own code |
| Prompt | `default` prompt (jsonl has no prompt_mode field; vlm_labels_violence is the violence_embedding set) | `violence_embedding` JSON prompt + system prompt | "Describe the scene.", no system prompt, add_generation_prompt=False |
| Context | `context_mode="both"` per `data/processed/vlm_labels/*.jsonl`: previous-frame images and text in the input (inferred to apply to embeddings too) | none | none |
| Pooling | image-token mean pool (current QwenTeacher code) | image-token mean pool | mean over **all** tokens (text included) |
| L2 norm | yes (verified: row norms = 1.0) | yes | **no**. Measured effect small: scaling inputs x50 changes the score by at most 0.006 because LayerNorm absorbs scale |
| dtype / resolution | bf16, processor default resolution | bf16, native resolution | fp16, **forced 448x448** |
| Time between embeddings | **about 6.6 s (XD-Violence, 159 frames at 24 fps), about 4.3 s (UCF, 128 frames at 30 fps), about 2 s (curated)** | about 1 s | about 1 s |
| Window | 30 steps (about 130-200 s of real time); label = violence inside the window OR in the next 30 *steps* | 30 steps of about 1 s with zero left-padding | 32 steps of about 1 s |
| Activation | GELU | GELU (correct) | **ReLU** (measured: +0.037 mean shift in probability, corr 0.9995, on probe_clip_01 windows) |

Consequences:

1. **Time-scale mismatch is the largest conceptual bug.** The model learned patterns over about 2-3 minute windows, sampled every
   4-7 s. At deployment it sees 30 s of 1 fps frames. The "30 s horizon" in training (`horizon_frames = int(30 * fps=1.0)` = 30 steps)
   is really about 2-3 minutes of real time. The training label also includes current violence (`label = max(current, future)`), so
   the model is a detection-plus-anticipation hybrid, not a pure 30 s-ahead predictor.
2. The embedding recipes differ, so live and YouTube embeddings fall outside the training distribution. The continuation scripts
   independently found this: YouTube embeddings sit at an LDA score of -5.13 against training class means of +2.85 and -3.34, and
   2141 of 3584 dims have Cohen's d > 0.5 (recorded in former docstrings of `continuation/lda_calibration_complete.py` and `continuation/domain_adaptation.py`; this doc is now the record).
3. **Any new model must ship its own feature extractor and use identical code at train and serve time**: the same function, prompt,
   pooling, frame rate and window length.

---

## 7. Latency and real-world behaviour from the logs

### 7.1 Latency and throughput (measured from the logs, GPU)

| Stage | Measurement |
|---|---|
| Qwen2.5-VL-7B load | 9.1 s (Feb 26), 10.0 s (Mar 2) |
| LSTM load | about 0.6 s |
| VLM embedding per frame (fp16, 448x448) | median 0.35 s, range 0.15-0.57 s (95 logged samples) |
| LSTM inference | 1.6-2.2 ms. The first call is about 100 ms (CUDA warm-up) |
| Queues | `frame_q=0`, `embed_q=0` throughout, so the pipeline keeps up at 1 fps. VLM headroom is about 2-3 fps (inferred from 0.35 s/frame) |
| Capture | Native 30 fps. About 29.4 frames dropped per sampled frame (e.g. "60 sampled, 1759 dropped") |
| Inference rate | 825 inferences in 841 s, about 0.98 Hz |
| Cold-start to first score | 32.3 s after the stream opened (both runs). The 32-frame buffer must fill first |
| Per-score latency | about 0.4 s (embed + LSTM) after a frame is sampled |
| Score dynamics | Very smooth. Median step 0.0015-0.0066, max step 0.019 per second. Reaching 0.5 from baseline took about 55-60 s in both runs, so the BiLSTM-over-32 s design adds tens of seconds of detection lag |

### 7.2 What each live session shows

**No ground truth exists for any RTSP session.** The repo has no annotations and no notes about what happened on camera. The stream path
`macbook` suggests a laptop webcam pushed to an RTSP server, likely a staged or benign desk scene (inferred). So false-alarm and miss
rates **cannot** be computed. What can be read:

- **`logs/inference_20260226_151941.log`**: URL `rtsp://username:password@127.0.0.1:554/macbook` (matches `mediamtx.yml`). "Failed to
  open stream" 6 times, Ctrl+C after 30 s, 0 inferences. The setup was then switched to port 8554 without auth.
- **`inference_20260226_225311.log`** (22:53:11-23:08:03, about 14 min, `rtsp://127.0.0.1:8554/macbook`):
  - 825 inferences. Scores ranged 0.116-0.504, mean 0.208, median 0.186, 95th percentile 0.393.
  - There were two bumps: 22:55:17-22:56:22, rising to 0.504, and 22:57:00-22:57:21, peaking at 0.33. Otherwise scores stayed at 0.12-0.25.
  - 4 inferences reached 0.5 or more (0.5044, 0.5012, 0.5011, 0.5008, at 22:55:38-22:55:51). Only **1 alert** fired (22:55:37); the 30 s cooldown suppressed the rest.
  - The stream was lost at 23:07:48 and the reconnect failed. On shutdown the SIGINT handler re-entered 7 times (repeated Ctrl+C), and
    the capture thread took about 4 s to exit from its reconnect sleep.
  - If the scene was benign, this is one marginal false alarm in 14 min (inferred). The peak barely grazed the threshold either way.
- **`inference_20260302_185137.log`** (18:51:37-18:53:28, about 1m51s):
  - 75 inferences. The score climbed steadily from 0.212 (18:52:13) to a peak of 0.540 (18:53:13), then decayed to 0.395 at stop.
  - 10 consecutive inferences reached 0.5 or more (18:53:09-18:53:18), producing **1 alert** at 18:53:08.
  - The shape looks like a deliberately staged event about 50-60 s into the session (inferred), but no annotation confirms it.

Takeaways:

- The pipeline is stable and real-time.
- The model's output range on webcam footage is compressed (0.12-0.54) and sits just under the 0.5 threshold, so alerts are threshold
  grazes and very sensitive to calibration. The ReLU bug alone shifts scores by about +0.04.
- `Prediction.lead_time_budget` is just the cooldown constant (30 s), not an estimated time-to-violence.
- `continuation/time_to_violence.py` contains a WATCH/ALERT/IMMINENT/ACTIVE tiering and ETA heuristic with an "integration patch" for
  this file. It was **never merged** into rtsp_inference.py.

### 7.3 YouTube video `probe_clip_01` (the only deployment-style run with ground truth)

Ground truth, from `YT_LABELS` in `continuation/lda_calibration_complete.py`: 0-6 s safe, 7-36 s anticipation, 37-45 s violence,
46 s onward post-violence (people dispersing).

`probe_clip_01_scores.npy` has 197 frames at nominal 1 fps from `outputs/outputs_temporal/best_temporal_model.pt` (provenance verified).
Measured on CPU from the saved arrays:

- Frame-level ROC-AUC is 0.718 for "anticipation or violence (7-45 s)" vs everything else. For anticipation (7-36 s) vs safe plus post
  (violence excluded), it is 0.650.
- Every one of the 197 scores is at least 0.515. At the script's default threshold of 0.3, or at 0.5, **100% of frames are flagged**.
- The PNG used threshold 0.65:
  - The first alert came at t = 16 s, which is **21 s before violence onset** (37 s).
  - The score peaked at 0.741 at t = 43 s, during the fight.
  - 55 of 151 post-violence seconds were also flagged (alert regions 16-71 s, 86-93 s, about 97 s, and 177-197 s).
  - The **global maximum (0.767) is at t = 196 s, well after the event**.

| Threshold | First alert (s) | Frames flagged | Post-violence frames flagged (of 151) |
|---|---|---|---|
| 0.30 / 0.50 | 0 | 100% | 151 |
| 0.60 | 10 | 94.9% | 151 |
| 0.65 | 16 | 43.1% | 55 |
| 0.70 | 24 | 24.4% | 26 |
| 0.72 | 28 | 18.3% | 18 |

Caveats:

- The t = 0 score (0.515) is an artifact of the 29 zero-padded frames, which are out of distribution.
- `int(29.97 / 1.0) = 29` gives an effective rate of 1.033 fps, so plot time drifts by about 3% (about 1.5 s at t = 45). This assumes
  the source was 29.97 fps (inferred).
- One video is not a benchmark. The threshold of 0.65 was clearly chosen post hoc.

---

## 8. Misc root files

- **`mediamtx.yml`**: `rtsp: yes`, `rtspAddress: :554`, path `macbook` with `publishUser/publishPass/readUser/readPass` set to
  `username`/`password`. Port 554 requires root, and the only run that used it failed. Later runs used `rtsp://127.0.0.1:8554/macbook`
  without auth (MediaMTX defaults). No `mediamtx` binary exists on this server (`which mediamtx` is empty, nothing within depth 4 of `/`).
  The server ran elsewhere or was removed, or port 8554 was SSH-forwarded from the MacBook (inferred). The publisher was presumably
  something like `ffmpeg -f avfoundation -i 0 -f rtsp rtsp://<host>:8554/macbook` on the Mac (inferred).
- **`progress.sh`**: run it from the repo root. It currently prints "Embeddings Progress: 1721 / 1813 (94.9%)" and "Labels Progress: 1721 / 1721 (100.0%)".
  The denominator `vlm_emb` now includes 92 `curated` files that `vlm_emb_violence` never got, so 100% is unreachable.
- **`requirements.txt`**: torch>=2.1.0, torchvision>=0.16.0, transformers>=4.37.0, accelerate>=0.26.0, opencv-python>=4.8.0, Pillow>=10.0.0,
  and a commented-out `av`. This covers rtsp_inference.py only. `Qwen2_5_VLForConditionalGeneration` needs transformers 4.49 or later
  (inferred from the release history), and rtsp_inference's `dtype=` kwarg needs a recent version. numpy is not listed.
- **`.env`**: contains a single key, `YOUTUBE_API_KEY`, used by `setup/curate_youtube.py` via python-dotenv. **`youtube_cookies.txt`**
  holds YouTube cookies for `setup/kinetics_download_relevant.py`; the 2026-10 cleanup moved it out of the repo to
  `<cookies file outside the repo>` (directory mode 700, file mode 600). Both are secrets, both are gitignored, and their contents were not read.
- **`test_video.mp4`** (deleted in the 2026-10 cleanup as a duplicate; the same clip is `data/anticipation_curation/videos/holdout_025.mp4`):
  6:57 at 640x360, 29.97 fps, with stereo AAC audio. Usable as a quick smoke-test input for analyze_youtube
  (`--video data/anticipation_curation/videos/holdout_025.mp4`) and for audio pipelines. Its content and labels are unknown.

---

## 9. Dependencies: actual imports vs requirements.txt vs installed

The method was an AST scan of all 99 `.py` files outside `.venv`, `data`, `models` and LoRA dirs, then a comparison with `.venv/bin/pip list`
(Python 3.10.12, 135 packages).

| Import name | pip package (installed version) | Used by (count, examples) | In requirements.txt? |
|---|---|---|---|
| torch | torch 2.5.1+cu121 | 53 files | yes |
| numpy | numpy 2.2.6 | 71 files | **no** |
| tqdm | tqdm 4.67.1 | 35 | **no** |
| sklearn | scikit-learn 1.7.2 | 28 (sim/, experiments/, continuation/) | **no** |
| cv2 | opencv-python 4.12.0.88 | 21 | yes |
| matplotlib | matplotlib 3.10.7 | 16 | **no** |
| transformers | transformers 4.57.5 | 13 | yes (bound too low) |
| PIL | pillow 12.1.0 | 12 | yes |
| seaborn | seaborn 0.13.2 | 4 (training/train_temporal_model.py, sim/comprehensive_eval.py, experiments/) | **no** |
| yaml | PyYAML 6.0.3 | 3 (training/train.py, training/evaluate.py, scripts/build_llm_dataset.py) | **no** |
| ultralytics | ultralytics 8.3.228 | 3 (core/extractors.py, scripts/pose_feature_extraction.py, experiments/exp1_person_detection.py) | **no** |
| qwen_vl_utils | qwen-vl-utils 0.0.14 | 2 (analyze_youtube fallback, experiments/prompt_engineering_stats.py) | **no** |
| librosa | librosa 0.11.0 (+ soundfile 0.13.1, soxr) | 2 (core/extractors.py, scripts/check_audio_availability.py) | **no** |
| pandas | pandas 2.3.3 | 2 (setup/kinetics_*.py) | **no** |
| torchaudio | torchaudio 2.5.1+cu121 | scripts/audio_feature_extraction.py | **no** |
| torchvision | torchvision 0.20.1+cu121 | tuning/optical_flow_features.py | yes |
| scipy | scipy 1.15.3 | tuning/optical_flow_features.py | **no** |
| peft | peft 0.18.1 | tuning/lora_finetune_vlm.py | **no** |
| bitsandbytes | bitsandbytes 0.49.2 | tuning/lora_finetune_vlm.py | **no** |
| clip | OpenAI CLIP, "clip 1.0" (git install; README suggests `clip-anytorch`) | core/extractors.py | **no** |
| sounddevice | sounddevice 0.5.3 (needs PortAudio) | core/extractors.py (optional) | **no** |
| yt_dlp | yt-dlp 2026.3.17 (+ yt-dlp-ejs 0.8.0) | analyze_youtube. The CLI is also called via subprocess in setup/curate_youtube.py and setup/kinetics_download_relevant.py | **no** |
| dotenv | python-dotenv 1.2.1 | setup/curate_youtube.py (optional) | **no** |
| pydub | **NOT installed** | scripts/check_audio_availability.py (inside try/except, optional) | no |
| (implicit) | accelerate 1.12.0 | needed for `device_map=` in from_pretrained | yes |
| (implicit) | onnx 1.19.1 | `torch.onnx.export` in scripts/export_trace_graph.py | no |

Installed but never imported directly: deep-sort-realtime, decord, torch-geometric, onnxruntime, timm, einops, polars, av.
Some of these are transitive: lapx for ultralytics, av for torchvision/qwen-vl-utils video I/O (inferred). The venv also carries
**both cu11 and cu12 `nvidia-*` wheels**, leftovers from the README's cu118 install (inferred). `pytest` is **not installed**, although
README section 12 says `pytest -q`.

System binaries used via subprocess: `ffmpeg` (core/extractors.py, scripts/preprocess_anticipation.py, yt-dlp partial downloads),
`ffprobe` (scripts/check_audio_availability.py), and `yt-dlp` (needs `.venv/bin` on PATH). `ffmpeg` and `ffprobe` are present at `/usr/bin`.
Live mode also needs an RTSP server such as MediaMTX.

Suggested `requirements.txt` that reflects actual usage (pinned to the working venv):

```
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.5.1+cu121
torchvision==0.20.1+cu121
torchaudio==2.5.1+cu121
transformers==4.57.5        # >=4.49 for Qwen2.5-VL; rtsp_inference uses dtype= kwarg
accelerate==1.12.0
qwen-vl-utils==0.0.14
peft==0.18.1
bitsandbytes==0.49.2
numpy==2.2.6
scipy==1.15.3
scikit-learn==1.7.2
pandas==2.3.3
opencv-python==4.12.0.88
pillow==12.1.0
matplotlib==3.10.7
seaborn==0.13.2
tqdm==4.67.1
PyYAML==6.0.3
python-dotenv==1.2.1
librosa==0.11.0
soundfile==0.13.1
ultralytics==8.3.228
yt-dlp==2026.3.17
onnx==1.19.1
# optional / legacy graph pipeline:
# git+https://github.com/openai/CLIP.git   (provides `import clip`)
# sounddevice==0.5.3                       (needs system PortAudio)
# pydub                                    (optional in check_audio_availability.py; not installed)
# pytest                                   (tests/; not installed)
```

---

## 10. Bug and issue list (deployment area)

1. **rtsp_inference uses ReLU where training used GELU** (frame encoder and classifier). It loads silently because the parameter names
   match. Measured shift: about +0.037 in probability.
2. **rtsp_inference embedding recipe differs from training**: prompt, no system prompt, all-token pooling, no L2 normalisation,
   forced 448x448, fp16. See Section 6.
3. **The time base differs from training** for both deployment scripts: 1 s sampling vs 4-7 s spacing in the training embeddings.
4. `--sequence_length` defaults to 32, while the checkpoints were trained with 30.
5. `load_state_dict(strict=False)` means an incompatible checkpoint whose keys do not overlap (the root-level `*_mlp.pt` files) runs
   with **random weights and no error**.
6. **Alert snapshots are never saved.** `alert_callback(pred)` is called without the frame (line 459), and `EmbeddedFrame` does not carry
   the image. `snapshots/` is empty.
7. `lead_time_budget` is hard-wired to `alert_cooldown` and is not a prediction.
8. The log FileHandler is created at import time in the CWD, so importing the module (tests, notebooks) litters log files.
9. Repeated SIGINT re-enters `stop()` (7 times in the Feb 26 log). Shutdown blocks on thread joins of up to 5 + 10 + 5 s.
10. Sampling is based on wall clock (`time.time()`), so a file or replayed stream at faster than real time is sampled inconsistently.
11. `cuda:0` and `cuda:1` are hard-coded, so the script cannot run on 1 GPU without editing.
12. analyze_youtube zero-pads the first 29 windows, which inflates or deflates early scores (t = 0 is 0.515 on probe_clip_01).
13. analyze_youtube `frame_interval = int(video_fps / fps)` truncates (29.97 fps gives 1.033 fps, so the time axis drifts). It is 0 when
    `fps > video_fps`, which raises **ZeroDivisionError** on `frame_idx % 0`.
14. analyze_youtube extracts with prompt_mode `violence_embedding` and no context, but its default checkpoint `outputs_temporal` was trained on
    `vlm_emb` (default prompt, `context_mode=both`). The matching checkpoint for that recipe is `outputs/outputs_temporal_violence/`.
15. analyze_youtube's silent fallback to Qwen2.5-VL-3B (2048-d, padded to 3584) produces garbage features with only a printed warning.
16. analyze_youtube's default threshold of 0.3 flags 100% of frames on the one evaluated video. All frames are held in RAM, and it uses `tempfile.mktemp`.
17. `realtime/pipeline.py`, `scripts/benchmark.py` and `tests/test_realtime_graph.py` are broken legacy code (Section 4). README sections 8, 9 and 12 document them as working.
18. `LLMEmbedder._extract_number` parses the prompt text too, so it returns 1.0 when the LLM outputs no number.
19. `requirements.txt` is incomplete, and its transformers lower bound is too low (Section 9).

---

## 11. Guidance for building the next deployable model

- Reuse the **threading skeleton** of `rtsp_inference.py`: capture, then feature worker, then a light temporal head with drop-oldest
  queues. It is proven to run in real time at 1 fps on GPU with about 0.35 s per Qwen2.5-VL-7B frame.
- Do not copy model classes into the inference script. Import the exact training module class, or save a TorchScript or full-module
  checkpoint together with a `model_config` and a `feature_config` (prompt, pooling, fps, window length). Refuse to load with `strict=False`.
- Sample frames at the **same interval used to build the training features** and feed **the same window length**. Better still, make
  the training features 1 fps so that a "30 s horizon" really means 30 s.
- Log per-frame scores to a file (`.npy`/`.csv`) and save alert snapshots, so live sessions can be annotated and scored afterwards.
  Without ground truth, the existing logs cannot answer the false-alarm question.
- Evaluate on held-out *videos*, at the deployment sampling rate, with lead-time metrics (time of first alert before onset) and
  false alarms per hour on long negative footage. `probe_clip_01` plus `data/anticipation_curation/videos/holdout_025.mp4` (formerly `test_video.mp4`) can serve as smoke tests. They are not a benchmark.

## Notes moved from code comments

- `scripts/analyze_youtube.py` `generate_embeddings()`: the `batch_size` and `use_project_extractor` arguments are accepted but
  ignored. Frames are always embedded one at a time, and the QwenTeacher path is chosen by an `isinstance` check on the model.
