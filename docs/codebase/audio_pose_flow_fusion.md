# Audio, Pose, Optical Flow, and Fusion

> Paths in this file describe the tree before the 2026-10 cleanup. Surviving code is under `legacy/<same path>`;
> the honest runs are under `baselines/reference_runs/`; removed tracked files: `git show pre-cleanup:<path>`;
> ignored weights and arrays: `data/archive/repo_legacy/<same path>`
> until 2027-01-12. See [../REPO_CLEANUP_PLAN.md](../REPO_CLEANUP_PLAN.md).

Scope: every non-VLM modality in the repo (audio, YOLOv8 pose, optical flow), plus the scripts that fuse
them with the Qwen2.5-VL embeddings (early fusion by concatenation, and late fusion of per-modality model scores).

Everything below was checked against the code and the files on disk on 2026-10-14. Claims marked
**(inferred)** could not be verified directly. Paths are relative to the repo root
`.`, where `data/` is a symlink to
`data/`.

Related docs: `docs/codebase/temporal_and_student_models.md` covers `training/train_temporal_model.py`, which trained
every modality model listed here. `docs/codebase/realtime_and_misc.md` covers `core/` and `realtime/`.

---

## 0. Summary

1. **No multimodal result in this repo is trustworthy as it stands.** Audio and pose were extracted on the wrong
   clock: both assume one graph step equals one second. Fusion then pairs rows by index. So audio row *i* and pose
   row *i* do not describe the same moment as VLM row *i* or label row *i*. Section 1 explains the clocks.
2. **Leakage in every "matched/fused" simulation.** `simulation_data/xd_violence_{8hr_fused,vlm_audio,audio_only,vlm_matched}.npz`
   contain **all 772 XD-Violence videos: 626 train, 67 val and 79 test.** That means the `final/evaluate_*` numbers
   were measured mostly on training videos.
3. **The optical-flow evaluation is broken.** `sim/build_flow_sim.py` concatenated per-frame flow from the
   first 11 alphabetically-sorted videos and pasted it next to an unrelated 8-hour VLM simulation.
4. **The late-fusion "100 % detection, 0.08 FA/h" result is a metric artifact.** The audio model's score never
   drops below 0.252, so at threshold 0.1 the alert stays on permanently. The script counts false alarms as rising
   edges, so a permanently-on alert registers as a single false alarm.
5. **Only one fused dataset is correctly time-aligned:** `data/processed/fused_vlm_flow` (VLM + optical flow,
   subsampled with `np.linspace`). It also had the best validation ROC-AUC of any modality run (0.721).
6. **What to reuse:** the raw per-frame optical-flow files, which can be re-aggregated cheaply onto any clock;
   the 1-Hz audio files after dropping the zero rows; `yolov8m-pose.pt`; and the raw XD videos, which all have
   48 kHz AAC audio. Re-extract pose, on a real clock and with tracking.
7. **Quick check run while writing this doc** (correct time alignment, no model fitting). Each feature's ROC-AUC
   against "violence onset within 30 s":
   - Best single feature, pooled across videos: audio about 0.67, pose about 0.62, flow about 0.60.
   - Within a video (each feature demeaned per video): only about 0.54 to 0.58.

   So these hand-crafted features mostly tell videos apart rather than predicting onset. See section 7.4.

---

## 1. Timelines (read this first)

Every modality has its own row clock, and fusion pairs rows by index. These clocks were verified on disk.

| Stream | Where | Row *i* corresponds to | Notes |
|---|---|---|---|
| Graph / labels | `data/processed/graphs/<vid>.npy` (object array of dicts with `frame_idx, nodes, adj, scene, audio`), `data/processed/labels_multih/<vid>.npy` `[T,7]` | time `((i+0.5)/T)·duration` (`map_t_to_raw_frame` in `scripts/build_multihorizon_labels.py:123`) | T varies per video. Measured seconds per row: median 0.68, range 0.04 to 360 s. **Not 1 Hz.** |
| VLM embeddings | `data/processed/vlm_emb/xd_violence/<vid>.npz` (`emb [N,3584] f16`, `t`, `raw_frame_idx`) | graph step `t = 5·i`, frame `round(((t+0.5)/T)·total_frames)` (`scripts/vlm_label_dataset.py:552-583`) | N = `len(range(0,T,5))`. Measured seconds per row: median 3.17, range 0.21 to 756 s. |
| Audio (this area) | `data/processed/audio_features/xd_violence/<vid>.npz` (`features [T,91]`) | **seconds [i, i+1)** from t=0 | T rows are forced to equal the graph length. Rows past the end of the audio are all zero. |
| Pose (this area) | `data/processed/pose_features/xd_violence/<vid>.npz` (`features [N,540]`, `frame_indices`) | **second 5·i** (`frame_idx = int(5·i·fps)`, clamped to the last frame) | N = `len(range(0,T,5))`, the same count as VLM but on a different clock. |
| Optical flow (this area) | `data/processed/optical_flow/xd_violence/<vid>.npz` (`features [F-1,22]`, `frame_indices`) | video frame `i+1`: the flow between frames i and i+1 at native fps, usually 24 | 2,240,502 rows in total. This is the only stream that keeps exact frame indices. |

Example: `A.Beautiful.Mind.2001__#00-25-20_00-29-20_label_A` (240 s at 24 fps, T=181).
- VLM rows sit at frames 16, 175, 334 … 5745, covering the whole video.
- Pose rows sit at frames 0, 120, 240 … 4320, covering only the first 180 s.
- Audio has 181 rows covering seconds 0 to 181.
- The fused VLM+audio file keeps only audio rows 0 to 36, so the audio covers seconds 0 to 37 while the VLM
  rows span 0 to 240 s.

Consequences measured on disk:
- **71.6 %** of all audio rows are all-zero: 168,057 of 234,777. 490 files have trailing zeros. For those, the
  median non-zero prefix is only 38 % of the rows.
- **70 %** of pose rows repeat the clamped last frame: 33,006 of 47,312 rows, across 495 files. About half of
  those rows (16,799) are all-zero because no person was detected or the read failed.
- Real total duration of the 772 XD videos is **25.9 h** (computed from `data/metadata/xd_violence.json`).
  The simulations assume `fps=1.0` and report 13.14 h for the VLM-rate sims and 65.2 h for the audio sim.
  Any "FA/hour" or "lead time (s)" from those sims is therefore in rows, not seconds.

---

## 2. Files in this area

| File | Role | Ever run? |
|---|---|---|
| `scripts/audio_feature_extraction.py` | GPU (torchaudio) hand-crafted audio features, 91-d, 1 Hz | Yes. 667 of the 798 files came from this script (logs in `logs/audio_gpu_{0..7}.log`). |
| `scripts/check_audio_availability.py` | ffprobe + librosa scan for audio streams and silence | No report found on disk **(inferred: never run, or report deleted)** |
| `scripts/pose_feature_extraction.py` | YOLOv8m-pose → 540-d per-sample pose/interaction vector | Yes. 798 files. |
| `tuning/optical_flow_features.py` | Farneback (or RAFT) dense flow → 22 motion statistics per frame | Yes, extraction. 772 files. `--analyze` mode never run (no png on disk). |
| `scripts/build_fused_features.py` | VLM + pose early fusion (per-video z-score + concat) and simulation builders | Yes → `data/processed/fused_features`, `simulation_data/xd_violence_8hr_fused.npz` |
| `scripts/multimodal_fusion.py` | VLM / audio / pose early fusion (generalisation of the file above) | Yes → `data/processed/vlm_audio_features`, `simulation_data/xd_violence_vlm_audio.npz`, `simulation_data/xd_violence_audio_only.npz` |
| `tuning/fuse_features.py` | VLM + optical flow (+ optional VLM-QA) early fusion with `linspace` resampling | Yes → `data/processed/fused_vlm_flow`, `data/processed/fused_finetuned_flow` |
| `sim/build_flow_sim.py` | Builds the flow "simulation" that matches the 8-hr VLM sim. **Broken** (section 4.3). | Yes → `simulation_data/xd_violence_8hr_flow_sim{,_fused}.npz` |
| `scripts/late_fusion.py` | Trains per-modality models (shells out to the trainer), late-fuses scores, grid-searches weights | Yes → `final/late_fusion_matched`, `final/late_fusion_max`. `final/late_fusion_optimization` is empty. |
| `yolov8m-pose.pt` (53 MB) | Ultralytics pose weights, used only by `pose_feature_extraction.py` (default `--model yolov8m-pose.pt`, resolved relative to the working directory) | Yes |
| `yolov8n.pt` (6.5 MB) | Ultralytics person detector, used by `core/extractors.py:33` (graph tracks, every 4th frame, no tracking IDs) and `experiments/exp1_person_detection.py` | Yes |
| `core/extractors.py: extract_audio_placeholder` | Writes a **64-d zero vector** as the "audio" of every graph | `data/processed/audio/*.npy`: all 1,798 files are zeros (verified). `graph['audio']` is also zeros. |

Environment, as installed in `.venv`: ultralytics 8.3.228, torchaudio 2.5.1+cu121, opencv 4.12, librosa 0.11.
`/usr/bin/ffmpeg` and `/usr/bin/ffprobe` are present.

---

## 3. Audio

### 3.1 `scripts/audio_feature_extraction.py`
- **Class `GPUAudioFeatureExtractor`**:
  - Loads audio with `torchaudio.load(video)` (needs the ffmpeg backend), downmixes to mono, and resamples on the
    CPU to 16 kHz with a new `T.Resample` per video.
  - Cuts the audio into **non-overlapping 1-s windows** (`output_fps=1.0`). The window count is forced to
    `num_frames = len(graph)`.
  - Windows that start past the end of the audio, or are shorter than 0.25 s, become zero vectors. So do silent
    windows (max < 1e-10) and any window whose computation raises an exception.
- **Inside each window:** STFT with n_fft=2048, hop=512 (about 31 STFT frames), n_mels=64, n_mfcc=20.
- **Output:** `<output_dir>/<source>/<vid>.npz` holding `features [T,91] float32` and `feature_dim`.
  Source is inferred from the file name: `_x264` → ucf_crime; `v=` or `__#` → xd_violence.
- **Feature layout (91-d)**. The names are stored in the npz files that carry metadata.

| Cols | Content |
|---|---|
| 0–19 | MFCC 0–19, mean over the window |
| 20–39 | MFCC 0–19, std |
| 40–43 | log-mel dB mean, std, max, min |
| 44–45 | `rms_mean`, `rms_max`. **Identical in this script**, because RMS is taken over the whole window. |
| 46–47 | `zcr_mean`, `zcr_std`. **std is a hard-coded 0.0 placeholder.** |
| 48–55 | spectral centroid mean/std, bandwidth mean/std, rolloff mean/std (each divided by sr), then flatness mean/std |
| 56–58 | onset strength mean, max, and count of onset frames above mean+std |
| 59–70 | 12-bin chroma (crude FFT-bin → pitch-class mapping; the DC bin is included) |
| 71–90 | delta-MFCC means |

- **Run** (from the repo root, 8 ranks over 2 GPUs as was done; see the logs):
  ```bash
  .venv/bin/python scripts/audio_feature_extraction.py \
      --videos_dir data/raw_videos/xd_violence/videos --graphs_dir data/processed/graphs \
      --output_dir data/processed/audio_features --source_filter xd_violence \
      --device cuda:0 --rank 0 --world_size 8 --skip_existing
  ```
  The script fully loads every graph `.npy` (about 2 MB of pickled dicts each) just to read `len(graph)`.
- **Small problems:**
  - `_compute_zcr` is dead code, and the `json` and `subprocess` imports are unused.
  - Every feature calls `.item()`, which forces a GPU sync per feature (slow).

### 3.2 What is on disk (`data/processed/audio_features/xd_violence/`, 798 files, 24 MB)
- **Coverage:** all 798 XD videos in `data/raw_videos/xd_violence/videos`. 772 of them have VLM embeddings. The
  other 26 have an empty graph (T=0) and therefore a single zero row. There is no UCF-Crime or Kinetics audio.
- **Two different extractor versions are mixed:**
  - **131 files** store `features, feature_dim, feature_names, video_path, target_sr, output_fps, duration,
    original_samples, n_output_frames, success`. In these, `rms_mean ≠ rms_max` and `zcr_std ≠ 0`, so a
    different implementation produced them **(inferred: an older CPU/librosa version)**. Same 91 names.
  - **667 files** store only `features, feature_dim` and come from the current script (`rms_mean == rms_max`,
    `zcr_std == 0`).
  - The logs show 8 ranks that each skipped about 17 existing files and wrote about 83 new ones, with 0 errors.
- **71.6 % of rows are all-zero** (section 1). A zero row can mean "after the end of the audio",
  "silence" or "error"; the data does not distinguish them.
- Every XD video checked has an AAC 48 kHz stream (ffprobe of a sample). XD-Violence is an audio-visual
  dataset, so audio is genuinely available.

### 3.3 Other audio code
- `scripts/check_audio_availability.py`: ffprobe stream listing plus librosa analysis of the first 10 s.
  "Has content" means RMS > −50 dB and peak > 0.01. Writes `audio_analysis_report.json`.
- `core/extractors.py`:
  - `extract_audio_placeholder()` returns `zeros(64)`, so the graph pipeline never had real audio.
  - `extract_audio_from_file()` (mean mel over the whole file) is unused.
  - `live_audio()` returns a **128-d** mean mel (librosa's default n_mels), which does not match the 64-d
    placeholder used offline (realtime issue).

---

## 4. Pose, person detection and optical flow

### 4.1 `scripts/pose_feature_extraction.py` (YOLOv8m-pose)
- `PoseFeatureExtractor(model_name="yolov8m-pose.pt", conf_threshold=0.5, max_persons=10)` runs
  `YOLO(frame, conf=0.5)` on single frames read with `cv2.VideoCapture` seeks.
- **Sampling:** `t_steps = range(0, T_graph, every_t=5)` is treated as **seconds**, and
  `frame_idx = min(int(t·fps), total_frames-1)`. Short videos therefore repeat their last frame (section 1).
- **Output:** `<output_dir>/<source>/<vid>.npz` with `features [N,540] f32`, `frame_indices [N]`, `feature_dim=540`.
- **Feature layout (540-d, only 360 populated):**

| Cols | Content |
|---|---|
| 0–29 | Scene features: n/max_persons, n_persons, pairwise centre distance min/max/mean/std (divided by max(w,h)), fraction of close pairs (<0.15), fraction of "facing" pairs, fraction of overlapping boxes, position variance, box-height coefficient of variation, aspect-ratio std. **Only 12 are used; cols 12–29 are zero.** If fewer than 2 persons, cols 2–29 are zero. |
| 30–349 | 10 person slots × 32. Each slot holds 25 used values plus 7 zero-padding: arms raised L/R, arm extension L/R, stance ratio, crouch, lean, 6 joint angles/π (elbows, knees, shoulders), 4 arm-segment lengths divided by box height, facing ratio, vertical span, wrist positions relative to the box (4), mean and min keypoint confidence. Slot order is YOLO output order; there is **no tracking**. |
| 350–357 | Motion summary: mean, max and std of \|keypoint displacement\|/100 px versus the **previous sampled frame (5 "s" earlier)**, matched **by detection index**, then the top-5 per-coordinate mean motions |
| 358 | n_persons |
| 359 | detection flag |
| 360–539 | **Always zero.** `get_feature_dim()` = 10·(32+17)+30+20 = 540 over-counts the 360 values actually written. |

Raw keypoints (17×3 per person) are **not saved**. If no person is detected, the whole vector is zero.

- **Run:**
  ```bash
  .venv/bin/python scripts/pose_feature_extraction.py \
      --videos_dir data/raw_videos/xd_violence/videos --graphs_dir data/processed/graphs \
      --output_dir data/processed/pose_features --source_filter xd_violence \
      --every_t 5 --device cuda:0 [--rank R --world_size W]
  ```
- **On disk:** `data/processed/pose_features/xd_violence/`, 798 files, 5.6 MB, 47,312 rows.
  - 26 files are empty (T=0).
  - 58.5 % of rows have ≥ 1 person; mean 1.69 persons; maximum 10.
  - About 70 % of rows are clamped duplicates.
- **Bugs:**
  - Wrong clock (section 1).
  - Motion is computed against a frame 5 s earlier and matched by detection index, so it is not a velocity.
  - `prev_keypoints` is not reset when a frame has no detections.
  - Half of the "violence pose" descriptors default to neutral constants (0.5, π/2) when keypoint confidence is
    below 0.3.

### 4.2 Person detector `yolov8n.pt`
- `core/extractors.py:extract_tracks` detects persons on every 4th frame for the scene graphs. Track IDs are
  just the per-frame enumeration.
- `experiments/exp1_person_detection.py` gates temporal-model scores to 0 when no person is present. Result in
  `experiments/output/exp1_person_detection/exp1_results.json`: gating **hurt**. ROC 0.832 → 0.812, and false-alarm
  events per hour rose 8.2 → 10.3. Persons are present in about 84 % of frames (smoothed), and the presence rate
  is nearly the same in violent and non-violent videos (86 % vs 83 %).

### 4.3 Optical flow: `tuning/optical_flow_features.py` and `sim/build_flow_sim.py`
- **`OpticalFlowExtractor(method='farneback')`:**
  - Resizes frames to 256 px height and runs `cv2.calcOpticalFlowFarneback` (CPU) on **every consecutive frame
    pair** (`sample_rate=1`).
  - Produces 22 features per pair, in this order: `magnitude_mean, magnitude_std, magnitude_max, magnitude_p90,
    magnitude_p99, motion_intensity, flow_variance_x, flow_variance_y, flow_variance_total, directional_entropy,
    dominant_direction_strength, motion_concentration, motion_spread, center_motion_ratio, motion_acceleration,
    motion_consistency, flow_change_mean, flow_change_max, high_motion_area_ratio, num_high_motion_regions,
    chaos_score, impact_score`.
  - The history-based features (acceleration, consistency, flow_change) refer to adjacent frames, i.e. 1/24 s.
- **Output:** `<output_dir>/<rel path of embedding>/<vid>.npz` with `features [F-1,22] f32`, `frame_indices` (starting
  at 1), `feature_names`, and `video_info={total_frames,fps,width,height,sample_rate}` (pickled dict, so load with
  `allow_pickle=True`).
- **Run** (the video list comes from the embedding directory):
  ```bash
  .venv/bin/python tuning/optical_flow_features.py --videos_dir data/raw_videos/xd_violence/videos \
      --embeddings_dir data/processed/vlm_emb --output_dir data/processed/optical_flow \
      --source_filter xd_violence --use_pool --num_processes 16
  ```
  Alternatives: `--launch_parallel --num_workers N`, or `--video_path` for a single file.
- **On disk:** `data/processed/optical_flow/xd_violence/`, 772 files (the videos that have VLM embeddings),
  170 MB, 2,240,502 rows.
- **Problems:**
  - The RAFT path feeds [0,1] BGR tensors (torchvision RAFT expects [-1,1] RGB), and 256-px-high frames give
    widths that are not multiples of 8 **(inferred: RAFT would fail or be wrong)**.
  - `--method` is not forwarded to `--launch_parallel` workers or to the `--use_pool` path, which always uses
    Farneback.
  - `analyze_features()` pairs flow row *i* (a video frame) with label row *i* (a graph step) and uses a random
    per-row train/test split, which leaks across videos. It was never run.
- **`sim/build_flow_sim.py`, broken:**
  - It concatenates `features` from **all** flow files in sorted order, then truncates to the VLM sim length
    (32,092 rows). That length is reached after the first 11 files (A.Beautiful.Mind … Bad.Boys.II): about 22
    minutes of 24 fps video that has nothing to do with the 8-hr sim's content.
  - The `_fused` output concatenates **raw** VLM embeddings with z-scored flow. The model was trained on per-video
    z-scored VLM (`fuse_features.py`), so the inputs are also mis-scaled.
  - `sim/comprehensive_eval.py --optical_flow_sim` does the same naive concatenation at eval time.

---

## 5. Fusion scripts

### 5.1 Early fusion: shared behaviour
All three early-fusion scripts:
- **Align by truncation:** `feat[:min_len]`. `tuning/fuse_features.py` is the exception and resamples with
  `np.linspace` instead.
- **Normalize per video** (`concat_norm`): a z-score computed over the **whole video**. This is non-causal, since
  future frames shape the statistics. It also cannot be reproduced on a live stream, and it removes between-video
  offsets.
- **Store** `emb` as float16 under `<out>/<source>/<vid>.npz`, which `training/train_temporal_model.py` reads directly.

`build_simulation*` modes concatenate the videos into one timeline:
- `frame_labels = (labels_multih.max(axis=1) > 0)[:n]` and `anticipation_labels[t] = max(frame_labels[t:t+30])`,
  with `fps=1.0`, `video_boundaries` and `video_names`.
- The label source is **every** file in `labels_dir` that passes the source filter. **There is no train/val/test
  filter.**
- **Label semantics:** `labels_multih` has horizons [2,5,10,20,30,60,120] s, with −1 inside violence segments.
  `max(axis=1) > 0` therefore means "a violence onset within 120 s", and frames *inside* violence become 0
  (negative). The extra +30-row horizon is then added on top.
- **Truncation `[:n]` instead of `[::5]`:** VLM row k corresponds to label row 5k. Verified for
  `xd_violence_vlm_audio.npz`: `frame_labels == labels[:n]` for 772 of 772 videos, and 362 videos would differ if
  correctly subsampled.

### 5.2 `scripts/build_fused_features.py` (VLM + pose)
- **Modes:**
  - `build_embeddings`: per-video `concat_norm` of `[VLM 3584 | pose 540]` = 4124-d. Videos without pose are
    skipped.
  - `build_simulation`: from an existing sim that has boundaries and names.
  - `build_simulation_from_videos`: from directories.
  - `check_separability`: prints the cosine similarity of class-mean vectors; results are not saved.
- **Outputs:**
  - `data/processed/fused_features/xd_violence/` (772 files plus `fusion_metadata.json`; strategy concat_norm).
  - `simulation_data/xd_violence_8hr_fused.npz`: `[47310, 4124]`, 772 boundaries. **Misnamed:** it is not the
    8-hr sim, but all 772 videos concatenated.
- **Bugs in `build_fused_simulation`** (not used for any saved file **(inferred)**):
  - Hard-coded `pose_dim=350` zero padding against a real dimension of 540, so `vstack` fails if videos are mixed.
  - Padded videos use the un-normalized `concat` strategy.
  - Per-video truncation shrinks the timeline, but labels are only truncated at the end.

### 5.3 `scripts/multimodal_fusion.py` (VLM / audio / pose)
- `MultiModalFuser` with `--modalities` drawn from {vlm, audio, pose}.
- **Strategies:** `concat`, `concat_norm` and `weighted` (hard-coded weights vlm 1.0, audio 1.5, pose 1.0).
- **Modes:** `check_separability`, `build_embeddings` and `build_simulation`. A video is skipped unless every
  requested modality is present.
- **Outputs:**
  - `data/processed/vlm_audio_features/xd_violence/` (772 files, 3675-d).
  - `simulation_data/xd_violence_vlm_audio.npz` `[47310,3675]`.
  - `simulation_data/xd_violence_audio_only.npz` `[234751,91]`.

  The metadata stores `dims` as a pickled dict.
- Example:
  ```bash
  .venv/bin/python scripts/multimodal_fusion.py --mode build_simulation --vlm_dir data/processed/vlm_emb \
      --audio_dir data/processed/audio_features --labels_dir data/processed/labels_multih \
      --modalities vlm audio --source_filter xd_violence --output simulation_data/xd_violence_vlm_audio.npz
  ```

### 5.4 `tuning/fuse_features.py` (VLM + flow [+ QA])
- Loads the first matching key from `emb/embedding/embeddings/features/arr_0`.
- **Aligns by subsampling longer streams** with `np.linspace(0, len-1, min_len)`. For flow against VLM this lands
  within about 15 frames of the VLM `raw_frame_idx`, i.e. **correctly time-aligned** (verified on A.Beautiful.Mind).
  Each sampled value is still an instantaneous single-frame-pair statistic.
- Normalizes each stream with `standard` (per-video z-score), `minmax` or `l2`, then applies `concat` or `weighted`.
- **Outputs:**
  - `data/processed/fused_vlm_flow/` (flat, 772 files, `[N,3606]`, config `fusion_config.json`).
  - `data/processed/fused_finetuned_flow/`: its config points at `vlm_dir=data/processed/vlm_emb_finetuned`,
    which **does not exist**. The files therefore hold **flow only** (`[F-1, 22]` per-frame rows,
    `output_dim: 22`) despite the name.
- The VLM part of `fused_vlm_flow`, `fused_features` and `vlm_audio_features` matches the current
  `vlm_emb/xd_violence` after a per-video z-score (checked on 20 random videos).

### 5.5 `scripts/late_fusion.py`
- **`train_individual`:**
  - Shells out to `python training/train_temporal_model.py` with `--input_dim` hard-coded per modality
    {vlm 3584, audio 91, pose 540}, seq 30, 100 epochs, patience 15, lr 1e-4.
  - The command is a relative path run with the system `python`, so it must be launched from the repo root.
  - No outputs from this mode exist **(inferred)**.
- **`evaluate`:**
  - `load_model` re-implements the trainer's network but with **ReLU in place of GELU**. The `state_dict` still
    loads because the parameter indices match, so the outputs silently differ from the trained model.
  - `run_inference` builds an `[N,30,D]` array in RAM (about 20 GB for 47,310×30×3584 float32).
  - Modalities are "aligned" with `resample_to_timeline`, which interpolates by `fps`. Every sim declares
    `fps=1.0`, so in practice this just truncates the 234,751-row audio sim to the VLM sim length. The audio
    predictions therefore come from different videos and times than the VLM predictions.
  - **Events** come from `frame_labels` within boundaries. A detection window is
    `[event_start − 30, event_end)`, which includes the violence itself.
  - **False alarms** are counted as rising edges where `anticipation_labels == 0`.
  - The best threshold maximises `det − 0.1·FA/h` **on the evaluation set**.
  - `roc_auc_score` exceptions are swallowed and replaced with 0.5.
- **`optimize_weights`:** 2-modality grid search with weights 0 to 1 in steps of 0.1 and thresholds 0.3 to 0.7,
  also on the evaluation set. Its output folder, `final/late_fusion_optimization/`, is empty.
- **Strategies:** max, mean, weighted, product (geometric mean), noisy-OR.

---

## 6. Derived data and checkpoints (sizes and shapes)

| Path | Producer | Shape / count | Status |
|---|---|---|---|
| `data/processed/audio_features/xd_violence/*.npz` | audio_feature_extraction.py (+ older version) | 798 files, `[T,91]`, 234,777 rows, 24 MB | Real audio at 1 Hz from t=0. 71.6 % zero rows. Two extractor versions. |
| `data/processed/pose_features/xd_violence/*.npz` | pose_feature_extraction.py | 798 files, `[N,540]` (360 used), 47,312 rows, 5.6 MB | About 30 % of rows usable |
| `data/processed/optical_flow/xd_violence/*.npz` | optical_flow_features.py | 772 files, `[F-1,22]` per frame, 2.24 M rows, 170 MB | **Most reusable**: exact frame indices plus fps |
| `data/processed/fused_features/xd_violence/*.npz` | build_fused_features.py | 772, `[N,4124]` f16, 302 MB | VLM+pose, misaligned |
| `data/processed/vlm_audio_features/xd_violence/*.npz` | multimodal_fusion.py | 772, `[N,3675]` f16, 304 MB | VLM+audio, misaligned (audio = first N seconds only) |
| `data/processed/fused_vlm_flow/*.npz` | fuse_features.py | 772, `[N,3606]` f16, 301 MB | **Time-aligned** VLM+flow (per-video z-score) |
| `data/processed/fused_finetuned_flow/*.npz` | fuse_features.py | 772, `[F-1,22]` f16, 84 MB | Mislabeled: flow only |
| `data/processed/audio/*.npy` | core placeholder | 1,798 × `(64,)` | All zeros |
| `simulation_data/xd_violence_8hr_fused.npz` | build_fused_features (from_videos) | `[47310,4124]`, 772 videos, 632 MB | Leaky (all splits) and misaligned |
| `simulation_data/xd_violence_vlm_audio.npz` | multimodal_fusion | `[47310,3675]`, 635 MB | Leaky and misaligned |
| `simulation_data/xd_violence_audio_only.npz` | multimodal_fusion | `[234751,91]`, 23 MB | Leaky |
| `simulation_data/xd_violence_vlm_matched.npz` | VLM-only matched sim (different builder; video order differs) | `[47310,3584]`, 368 MB | Leaky. Labels are {0,1}, but they contained −1 when late fusion was run (see 7.3). |
| `simulation_data/xd_violence_8hr_flow_sim.npz` / `_fused.npz` | sim/build_flow_sim.py | `[32092,22]` / `[32092,3606]` | Garbage pairing (section 4.3) |
| `outputs/outputs_temporal_audio_only/best_temporal_model.pt` | train_temporal_model.py | 3.7 MB, input 91 | Trained on misaligned rows |
| `outputs/outputs_temporal_vlm_audio/best_temporal_model.pt` | " | 7.3 MB, input 3675 | " |
| `outputs/outputs_temporal_fused/best_temporal_model.pt` | " | 7.8 MB, input 4124 (VLM+pose) | " |
| `outputs_fused/best_temporal_model.pt` | " | 7.3 MB, input 3606 (VLM+flow) | Aligned inputs, but labels truncated `[:n]` |

---

## 7. Measured results

### 7.1 Training runs (`training/train_temporal_model.py`)
All runs used the same settings:
- Model: BiLSTM + attention, `seq_len=30` rows, stride 5, focal loss, `labels_dir=data/processed/labels_multih`,
  XD-Violence only.
- Splits: `train_index.json`, `val_index.json` and `test_index.json`.
- The "test" threshold, F1, precision and recall are **chosen on the test set**.

| Run dir | Input | Dim | Val ROC / PR | Test ROC / PR | Test F1 | Epochs (best) |
|---|---|---|---|---|---|---|
| `outputs/outputs_temporal_audio_only` | audio_features | 91 | 0.642 / 0.337 | 0.758 / 0.345 | 0.477 | 18 (3) |
| `outputs/outputs_temporal_vlm_audio` | vlm_audio_features | 3675 | 0.693 / 0.367 | 0.773 / 0.469 | 0.516 | 26 (11) |
| `outputs/outputs_temporal_fused` | fused_features (VLM+pose) | 4124 | 0.674 / 0.333 | 0.897 / 0.445 | 0.574 | 26 (11) |
| `outputs_fused` | fused_vlm_flow | 3606 | **0.721 / 0.426** | 0.920 / 0.698 | 0.615 | 28 (13) |
| *ref* `outputs/outputs_temporal_vlm_emb` | vlm_emb (raw) | 3584 | 0.575 / 0.356 | 0.893 / 0.720 | 0.759 | 20 (5) |
| *ref* `outputs/outputs_temporal` | vlm_emb (raw) | 3584 | 0.577 / 0.365 | 0.991 / 0.907 | 0.888 | 16 (1) |
| *ref* `outputs/outputs_temporal_old_emb` | vlm_emb_old | 3584 | 0.665 / 0.357 | 0.947 / 0.675 | 0.781 | 21 (6) |

How to read this table:
- **Overfitting.** Every fused model reaches a train ROC of 0.9999 to 1.0 within about 10 epochs. The large val/test gap follows
  from how small and skewed the splits are:
  - At VLM rate, only 348 / 37 / 45 train / val / test videos have ≥ 30 rows, which a 30-row window needs.
  - **One** non-violent video, `v=vHnAmx0UYbM__#1_label_A`, contributes 796 of the 1,020 test sequences.
  - In val, `v=pFamvR9CpYw__#00-06-36_00-15-20_label_B4-0-0` contributes 438 of the 608 sequences.
  - So test ROC mostly measures "this long video versus the positives from other videos".
- **Validation ROC is the less biased number.** On it, every fused variant (0.67 to 0.72) beats raw VLM
  (0.58 to 0.67).
- **The gain is probably not from the added modality (inferred).** The audio half of `vlm_audio_features` is
  badly misaligned, yet it still gains about +0.1. So much of the improvement plausibly comes from the per-video
  z-scoring of the VLM half, a non-causal normalisation, rather than from the extra modality.
- VLM+flow is the only correctly aligned fusion, and it is the best on validation.

### 7.2 Simulation evaluations (`sim/comprehensive_eval.py` → `final/<dir>/metrics.json`)
How `comprehensive_eval.py` scores runs:
- Scores are produced per row with a 30-row window. Only the first 29 rows of the whole sim get 0, so windows
  straddle video boundaries.
- **Events** are runs of `anticipation_labels == 1` over the concatenated timeline, **ignoring video boundaries**.
  This is why event counts differ (183, 269, 280) on the same 47,310 rows: the sims order the videos differently.
- An event counts as **detected** if there is any alert in `[start−30, end]`.
- **Lead time** is measured from the start of the anticipation run.
- "PR" in the table is `average_precision_score`.
- Which model and sim each folder used is **inferred** from the row counts and input dimensions.

| Folder | Sim (rows) | Model | Thr | ROC | PR | Det | FA/h* | Lead* (s) | Events |
|---|---|---|---|---|---|---|---|---|---|
| `final/evaluate_original` | xd_violence_8hr_sim (32,092) | VLM-only temporal | 0.388 | 0.834 | 0.131 | 84.6 % | 8.64 | 11.4 | 13 |
| `final/final_evaluation_optical_flow_fused` | 8hr_sim + 8hr_flow_sim (32,092) | `outputs_fused` (VLM+flow) | 0.2 | **0.557** | 0.024 | 46.2 % | 9.20 | 25.0 | 13 |
| `final/evaluate_vlm_matched` | xd_violence_vlm_matched (47,310) | VLM-only | 0.5 | 0.813 | 0.694 | 94.0 % | 16.6 | 18.6 | 183 |
| `final/evaluate_vlm_audio` | xd_violence_vlm_audio (47,310) | vlm_audio | 0.388 | 0.868 | 0.670 | 74.0 % | 22.8 | 14.8 | 269 |
| `final/evaluate_fused` | xd_violence_8hr_fused (47,310) | VLM+pose | 0.388 | 0.887 | 0.706 | 77.9 % | 20.8 | 14.8 | 280 |
| `final/evaluate_audio_only` | xd_violence_audio_only (234,751) | audio-only | 0.53 | **0.438** | 0.153 | 19.7 % | 3.33 | 6.2 | 476 |

\* Rows are assumed to be 1 s, which they are not (section 1).

What the table shows:
- The 47,310-row sims are 81 % training videos, so the VLM+pose and VLM+audio "gains" over VLM-only
  (0.887 and 0.868 vs 0.813) are confounded by memorisation: those fused models reached train AUC 1.0.
- Audio-only is below chance on its own sim.
- The flow-fused result is invalid (section 4.3).
- `evaluate_audio_only`'s sweep shows 100 % detection at thresholds 0.1 to 0.3 with only about 7 FA/h. Audio
  scores never fall below about 0.25, and false alarms are only counted on rising edges, so this is an artifact.

### 7.3 Late fusion (`scripts/late_fusion.py --mode evaluate --fusion_strategy max`)
Models and sims are not recorded in the JSON **(inferred: `outputs/outputs_temporal_*` VLM and audio checkpoints)**.

| Folder | Reported | Recomputed from `predictions.npz` (ROC, labels > 0) |
|---|---|---|
| `final/late_fusion_max` (VLM 8-hr sim, 32,092 rows) | VLM ROC 0.814; audio 0.540; max-fused 0.664; "best" fused = det 100 %, FA 0.11/h at thr 0.1 | same |
| `final/late_fusion_matched` (47,310 rows) | ROC **0.5 everywhere**, because labels had −1 values and `roc_auc_score` raised; "best" fused = det 100 %, FA 0.08/h at thr 0.1 | VLM **0.811**, audio **0.480**, max-fused **0.743** |

Notes:
- `audio_preds` min/max are 0.252/0.637, so at thr ≤ 0.25 the alert is permanently on and counts as a single false
  alarm. The "100 % detection" is meaningless.
- Late fusion with audio *lowered* ROC in both cases, as expected given the misaligned audio predictions.
- Late-fusion labels have −1 where the current `xd_violence_vlm_matched.npz` has 0. Otherwise they are identical,
  so the sim was probably re-saved after this run **(inferred)**.

### 7.4 Quick univariate check (run while writing this doc; no model fitting)
Method:
- 772 XD videos.
- Target: "violence onset within 30 s", taken from `labels_multih[:,4]`, with −1 rows dropped.
- Each label row t is mapped to `tsec = round(((t+0.5)/T)·total_frames)/fps`. That time indexes the audio row
  `floor(tsec)` (zero rows dropped), the nearest pose row within 2.5 s (clamped duplicates dropped), and the flow
  row at the nearest `frame_indices`.
- Score: per-feature ROC-AUC, pooled over all rows, and **within-video** (features demeaned per video; only
  videos containing both classes).

| Modality | Pooled best feature (AUC) | Within-video best (AUC) |
|---|---|---|
| Audio (91) | mfcc_4_mean 0.670, mfcc_8_mean 0.645 (n=180,551, 12.7 % positive) | mfcc_0_mean 0.549, onset_strength_mean 0.541 (285 videos) |
| Pose (272 non-constant cols) | motion summary cols 350–354 ≈ 0.62 (inverse), n_persons 0.61 (inverse) | person-slot 4/5 limb lengths and confidence ≈ 0.57 (more people present → onset) (286 videos) |
| Flow (22) | high_motion_area_ratio 0.600, directional_entropy 0.590 | high_motion_area_ratio 0.572 (inverse), magnitude_mean 0.542 (354 videos) |

The same audio data aligned the way the trainer aligned it (row index) gives a best feature of 0.62. Takeaway:
the hand-crafted features carry mostly video-level signal and very little within-video "pre-violence" signal.
Stronger pretrained audio and pose encoders, on a correct clock, are needed before fusion can help.

---

## 8. Issues, ranked

**Critical: these invalidate the reported multimodal numbers.**
1. **Wrong clock in audio and pose extraction.** Both use `range(T_graph)` as seconds. Labels and VLM use
   `((t+0.5)/T)·duration`. Result: 71.6 % zero audio rows and 70 % clamped pose rows.
2. **Fusion by truncation `[:min_len]`.** In `build_fused_features.py` and `multimodal_fusion.py` this pairs VLM
   row k (graph step 5k) with audio second k and pose second 5k.
3. **Label truncation `labels[:n]` instead of `labels[::5]`** in every simulation builder here (and in the
   trainer). 362 of 772 videos have misaligned labels in the sims.
4. **Train/val/test leakage** in the 47,310-row and 234,751-row sims (626 of 772 videos are train). `final/evaluate_*` is optimistic.
5. **`sim/build_flow_sim.py` pairs unrelated flow with the 8-hr VLM sim**, and feeds raw VLM to a model trained on
   z-scored VLM.
6. **Late fusion:**
   - Audio predictions are misaligned with VLM predictions.
   - ReLU replaces GELU in the re-implemented model.
   - The ROC exception is swallowed and replaced with 0.5.
   - Counting only rising edges rewards a permanently-on alarm.
   - Weights and thresholds are tuned on the evaluation set.
7. **`fused_finetuned_flow` contains no VLM features** (its source directory was missing).

**High**
8. **Non-causal per-video z-score normalisation.** It cannot be used in streaming deployment and probably
   explains most of the "fusion gain".
9. **Label semantics.** `max over horizons > 0` means "onset within 120 s"; frames inside violence count as
   negatives; and +30 rows is added on top.
10. **Test and val are dominated by one long video each**, and test metrics use thresholds chosen on test.
11. **Two audio extractor versions are mixed** in one directory.
12. **`fps=1.0` is assumed everywhere.** FA/h and lead times are in rows (real data is 25.9 h, not 13.1 h or 65.2 h).
13. **`comprehensive_eval` merges events across video boundaries**, so event counts depend on video order.

**Medium / low**
14. Pose:
    - No tracking; motion is computed 5 s apart and matched by detection index.
    - 180 always-zero columns; raw keypoints not saved.
    - The model path is relative to the working directory.
15. Audio:
    - `rms_max == rms_mean` and `zcr_std = 0`.
    - Silent, short, error and padding windows all produce the same zero vector.
    - Dead `_compute_zcr`.
16. Flow:
    - The RAFT path is mis-normalised.
    - `--method` is not propagated to workers.
    - `--analyze` aligns labels wrongly and leaks via a per-row split.
    - Features are instantaneous (single frame pair) rather than window-aggregated.
17. `build_fused_simulation`: hard-coded `pose_dim=350`, inconsistent strategy for padded videos.
18. `late_fusion.run_inference` uses roughly 20 GB of RAM; `train_individual` uses a relative path and the system
    python.
19. The graph `audio` channel and `data/processed/audio/*.npy` are all-zero placeholders. `live_audio` is 128-d
    against the 64-d placeholder.

---

## 9. Advice for the next model (what to reuse and what to redo)

1. **Pick one absolute clock** (e.g. 1 Hz or 2 Hz in video seconds) and build labels from the raw XD/UCF
   annotations on that clock. Do not reuse the `labels_multih` row grid unless you map it with
   `((t+0.5)/T)·duration`.
2. **Audio:** re-extract from `data/raw_videos/xd_violence/videos/*.mp4` (48 kHz AAC) with a pretrained encoder:
   - general sound events: PANNs CNN14, BEATs, AST or CLAP;
   - speech and shouting: wav2vec2, or the Whisper encoder (also gives transcripts and keyword cues).

   Use overlapping windows centred on clock ticks. If you keep the 91-d features, take
   `audio_features[...][:floor(duration)]` and treat row s as second s.
3. **Pose:** re-run `yolov8m-pose.pt` (or a newer pose model) with `model.track(...)` (ByteTrack) on the chosen
   clock. Store raw keypoints `[T, P, 17, 3]` plus track IDs, then compute interaction features (closing distance,
   approach velocity, raised arms toward another person) downstream. The existing pose files are about 30 %
   usable at best.
4. **Flow:** reuse `data/processed/optical_flow/xd_violence/*.npz` directly. They have exact `frame_indices` and
   `video_info['fps']`, so aggregate (mean/max/p90) over each window. No recompute is needed. RAFT, or features
   from a video backbone (VideoMAE or X3D), would be stronger.
5. **Normalize with train-split statistics only.** Build simulations per split. Cap or weight sequences per video
   so that no single video dominates val or test. Pick thresholds and fusion weights on val.
6. **For late fusion,** align the per-modality score streams on the shared clock *before* fusing. Count false
   alarms per alert episode with a cool-down, not as rising edges of a permanently-on signal.
