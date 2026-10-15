# Documentation index: start here

This folder is the map of the repository for future AI agents and developers. It was written on
2026-10-14 by auditing every script, result folder and data directory. Numbers are verified against the
files on disk unless they are marked **(inferred)**.

---

## 1. What the project is

Senior capstone project at LTU. The goal is to **anticipate interpersonal violence in video at least 30
seconds before it starts**, ideally from several modalities (video, audio, pose, speech). The target footage
is real-world: CCTV, bodycam, dashcam/road rage, and phone video of street confrontations.

**Honest status (read this before quoting any number in the repo):**

- After about five months of work (Nov 2025 to Apr 2026) and many approaches, **no model in the repo
  anticipates violence 30 s ahead.** The only leak-free result is a frame-level MLP on Qwen2.5-VL
  embeddings: test ROC-AUC **0.58 to 0.62** for "violence starts within 30 s" (positive rate 0.081),
  XD-Violence-only test AUC **0.50 to 0.56** (chance), median alarm lead **about 8 to 9 s**. See
  [BASELINE.md](BASELINE.md).
- The impressive numbers in the removed `final/`, `outputs/` and older `results/` folders (AUC 0.93 to 0.99,
  "0 false alarms per hour", "92 % detection") come from label misalignment, leaky splits, simulations
  built mostly from training videos, test-tuned thresholds and lenient event metrics. See
  [EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md) and [codebase/results_inventory.md](codebase/results_inventory.md).
- The main bottleneck is data. Only **112 of the 640 onset-annotated videos have 30 s or more of footage
  before violence starts** (49 of them interpersonal), and the test splits hold only 10 to 12 of them.

The root `README.md` was rewritten in the 2026-10 cleanup as a short pointer here. Its earlier version described a
graph-transformer system with DeepSORT, real audio, ONNX export and passing tests that was never actually built
(audit in [codebase/core_and_training.md](codebase/core_and_training.md) section 14).

---

## 2. Reading order for a new agent

1. This file (layout, data locations, pipelines).
2. [BASELINE.md](BASELINE.md): the honest number to beat, the evaluation flaws, and the protocol any new
   model must be scored with.
3. [EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md): what was tried, what it scored, and why it failed.
4. [codebase/data_and_labels.md](codebase/data_and_labels.md): where the onsets come from, the time-axis
   contract, label sets, splits and data bugs. **Required before touching any label or feature file.**
5. [MODALITY_PLAN.md](MODALITY_PLAN.md): the evidence-based decision on which cues and feature streams to
   invest in for 30 s+ anticipation, the pre-registered modality screen, and the work priority. **Read before
   adding any feature stream or model.**
6. [research/README.md](research/README.md): which public datasets and pretrained backbones to add.
7. The per-area codebase docs below, as needed.

---

## 3. Repository layout (one line per top-level entry)

Repo root: `.`. Layout after the 2026-10 cleanup (section 8).

| Entry | What it holds |
|---|---|
| `anticipation/` | New pipeline package: data catalog, onsets, labels, folds (`data/`), VA30 metrics and control baselines (`eval/`), 1 Hz feature extractors (`features/`). |
| `benchmark/` | VA30 artifacts: `videos.csv`, `events.json`, `folds_v1.json`/`folds_v2.json`, `near_duplicates.json`, `catalog_summary.json`, `results/<baseline>/`. |
| `baselines/` | Honest gen-2 reference: `train_student_original.py` (B0 trainer, the only correctly aligned legacy trainer) and `reference_runs/` (4 movie-grouped StudentMLP runs + 4 zero-shot Qwen teacher evals). |
| `docs/` | This documentation: README (index), BASELINE, VA30, FEATURES, DATA_ACQUISITION, EXPERIMENT_HISTORY, REPO_CLEANUP_PLAN, `codebase/` (audit of the pre-cleanup code), `research/`, `figures/`. |
| `legacy/` | Frozen code kept only as provenance for files in `data/` and for cited numbers; `legacy/README.md` explains how to run it with cwd=`legacy`. Includes the leaky gen-1 clip split `{train,val,test}_index.json`. |
| `setup/` | Data acquisition: XD train split, UBI-Fights, HIVAU onsets, self-test/verify, `curate_youtube.py` (curated YouTube hold-out). |
| `tests/` | VA30 unit tests (`python -m pytest tests -q`). |
| `tools/` | `server_load.py`, `cpu_governor.sh`, `contact_sheet.py`, `check_comment_only_diff.py`. |
| `data` -> `data/` | Symlink to videos, metadata, features and `archive/` (outside git). |
| `models` -> `models/` | Symlink to pretrained weights. |
| `.venv/` | Python 3.10 environment (ignored). |
| `CLAUDE.md`, `README.md`, `requirements.txt`, `.gitignore` | Agent instructions, short project pointer, dependency list of the kept code, ignore rules. |
| `.env` | `YOUTUBE_API_KEY` for `setup/curate_youtube.py` (ignored, mode 600). |
| `yolov8m-pose.pt` | Fallback pose weight hard-coded in `anticipation/features/pose.py` (ignored). |

---

## 4. Where the data and features live

Everything below is under `data/` = `data/`. Full catalog with
shapes in [codebase/data_and_labels.md](codebase/data_and_labels.md) sections 1 to 8.

| Path | Content | Notes |
|---|---|---|
| `raw_videos/xd_violence/videos/` + `annotations.txt` | 798 XD-Violence **test** clips (500 violent, 298 normal), 24 fps, real audio. Onsets: clip-relative frames at 24 fps (strip `.mp4` from 14 ids). | The 3,954-video XD train split is not downloaded. |
| `raw_videos/ucf_crime/videos/` + `Temporal_Anomaly_Annotation_for_Testing_Videos.txt` | 1000 UCF-Crime videos (950 anomaly + 50 normal); onsets only for 140 test anomalies (30 fps). | Audio is mostly digital silence. 950 extra normal videos in `Training-Normal-Videos/` and `Testing_Normal_Videos/` were never used. |
| `raw_videos/kinetics/videos/` | 700 ten-second Kinetics-700 clips (wrestling, slapping). | Useless for anticipation (no lead-in). |
| `anticipation_curation/` | 92 YouTube road-rage / bodycam / CCTV confrontation videos (10 h, all with audio) + `candidates.json`. | Best domain match. Onsets are LSTM pseudo-labels, **not ground truth**; need hand annotation. |
| `metadata/{ucf_crime,xd_violence,kinetics}.json` | `{vid: {event_frame, total_frames, label, fps, source}}`. | **Do not use XD `label`/`event_frame`** (274 of 500 violent clips wrongly labelled 0). fps/total_frames are fine. |
| `processed/labels_multih/` | `[T,7]` labels, horizons 2/5/10/20/30/60/120 s, `-1` inside violence. | Correct only when indexed by the embedding `t` key. T is an irregular graph-step axis, not seconds. |
| `processed/vlm_emb*/{xd_violence,ucf_crime}/` | Qwen2.5-VL-7B frame embeddings `[N,3584]` + `t` + `raw_frame_idx` (1721 videos; variants `_meanpool`, `_violence`, `_old`). `vlm_emb/curated/` has 92 YouTube videos at 0.5 fps. | One step is median about 3 s, range 0.2 s to 756 s. Use `labels[t]`, never `labels[:len(emb)]`. |
| `processed/vlm_labels*/` | Per-frame Qwen JSON verdicts (violent score, people, actions, weapons, scene). | Zero-shot violent score AUC about 0.67 overall, 0.78 on XD. |
| `processed/optical_flow/xd_violence/` | Farneback flow, `[F,22]` per raw frame with exact frame indices (772 videos). | The most reusable non-VLM feature. |
| `processed/audio_features/xd_violence/`, `processed/pose_features/xd_violence/` | 91-d hand-crafted audio, 540-d YOLOv8m-pose vectors. | Extracted on the wrong clock (71.6 % zero audio rows, 70 % clamped pose rows). Re-extract. |
| `processed/{graphs,tracks,actions,scenes,audio,labels}/` | Gen-1 graph pipeline outputs on 40 s windowed clips. | `audio/` is all zeros; `scenes/` leaks post-event frames. Low value. |
| `windowed/` | 40 s ffmpeg trims around the (buggy) metadata onset. | Inherits the XD metadata bug. |

Pretrained weights: `models/Qwen2.5-VL-7B-Instruct` (16 GB). `yolov8m-pose.pt` in the repo root (pose fallback).

---

## 5. The main pipelines (by generation)

Legacy entry points run from `legacy/` (cwd=`legacy`, see `legacy/README.md`). "Removed" scripts are recoverable with
`git show pre-cleanup:<path>`.

| Generation | Dates | Data flow | Entry points | Doc |
|---|---|---|---|---|
| 1. Graph transformer | Nov to Dec 2025 | windowed clip -> YOLOv8n boxes + VideoMAE crop embeddings + clip-mean CLIP + zero audio -> per-step graph -> `ViolenceTransformer` | `legacy/scripts/preprocess_anticipation.py`, `legacy/training/train.py` (dangerous), `legacy/training/evaluate.py` | [core_and_training.md](codebase/core_and_training.md) |
| 2. VLM teacher + embeddings | Jan to Feb 2026 | raw video -> every 5th graph step -> Qwen2.5-VL-7B JSON verdict + 3584-d embedding | `legacy/scripts/vlm_label_dataset.py`, `legacy/scripts/eval_vlm_teacher.py` | [vlm_and_llm.md](codebase/vlm_and_llm.md) |
| 3. Heads on frozen embeddings | Feb 2026 | `vlm_emb` (+ flow/audio/pose) + `labels_multih` -> MLP / BiLSTM / contrastive heads | `baselines/train_student_original.py` (honest), `legacy/training/train_temporal_model.py`; removed: `training/train_student.py`, `tuning/contrastive_*.py` | [temporal_and_student_models.md](codebase/temporal_and_student_models.md) |
| 4. Multimodal fusion | Feb 2026 | flow / audio / pose extraction -> early concat or late score fusion | removed (`tuning/optical_flow_features.py`, `scripts/{audio,pose}_feature_extraction.py`, `tuning/fuse_features.py`, `scripts/late_fusion.py`); superseded by `anticipation/features/` | [audio_pose_flow_fusion.md](codebase/audio_pose_flow_fusion.md) |
| 5. Simulation and evaluation | Feb to Apr 2026 | per-video embeddings stitched into "8-hour" streams -> event metrics, smoothing sweeps | removed (`sim/`, `scripts/evaluate_with_smoothing.py`, `experiments/run_experiments.py`); superseded by `anticipation/eval/` (VA30) | [simulation_and_evaluation.md](codebase/simulation_and_evaluation.md) |
| 6. Deployment and curation | Feb to Apr 2026 | RTSP or YouTube -> Qwen embedding at 1 fps -> BiLSTM -> alerts; YouTube search -> LSTM pseudo-labels | `legacy/realtime/rtsp_inference.py`, `legacy/scripts/analyze_youtube.py`, `setup/curate_youtube.py` (repo root; `analyze`/`export` need `PYTHONPATH=legacy`), `legacy/experiments/fast_analyze_curation.py` | [realtime_and_misc.md](codebase/realtime_and_misc.md) |
| 7. VA30 benchmark and 1 Hz features | Oct 2026 | raw annotations -> seconds-based onsets and labels, grouped folds -> causal 1 Hz features -> VA30 metrics | `anticipation/`, `benchmark/`, `setup/download_*`, `setup/build_hivau_onsets.py` | [VA30.md](VA30.md), [FEATURES.md](FEATURES.md), [DATA_ACQUISITION.md](DATA_ACQUISITION.md) |

---

## 6. Full document index

### Top level
- [BASELINE.md](BASELINE.md): honest baseline numbers, protocol, evaluation flaws, and the specification of
  the corrected benchmark (VA30) that every new model must be scored on.
- [EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md): every approach tried, in order, with its score, its
  result folder, and why it failed or plateaued.
- [VA30.md](VA30.md): the VA30 benchmark (catalog, events, folds, metrics, control baselines).
- [FEATURES.md](FEATURES.md): the causal 1 Hz feature extractors in `anticipation/features/`.
- [DATA_ACQUISITION.md](DATA_ACQUISITION.md): downloads of the XD train split, UBI-Fights and HIVAU onsets.
- [REPO_CLEANUP_PLAN.md](REPO_CLEANUP_PLAN.md): the 2026-10 cleanup plan and its execution record.

### Codebase (`docs/codebase/`)
- [data_and_labels.md](codebase/data_and_labels.md): datasets on disk, onset sources, pre-onset statistics,
  feature catalog, the time-axis contract, label sets, splits, data bugs.
- [core_and_training.md](codebase/core_and_training.md): gen-1 `ViolenceTransformer`, extractors, graph
  builder, training/eval scripts, README audit, CPU re-evaluation and leakage ablations.
- [vlm_and_llm.md](codebase/vlm_and_llm.md): Qwen2.5-VL teacher and embeddings, zero-shot quality, LoRA,
  VQA features, text-LLM head, circularity audit, inference cost.
- [temporal_and_student_models.md](codebase/temporal_and_student_models.md): student MLPs, BiLSTM,
  contrastive projectors, fusion builder, ensemble simulators; misalignment bug analysis.
- [audio_pose_flow_fusion.md](codebase/audio_pose_flow_fusion.md): audio, pose and flow extraction, the
  per-modality clocks, early and late fusion, univariate signal check.
- [simulation_and_evaluation.md](codebase/simulation_and_evaluation.md): every simulation builder and
  evaluator, metric definitions and their flaws, and a reference sketch for a corrected event metric.
- [results_inventory.md](codebase/results_inventory.md): every metric file in the repo, grouped by
  protocol, with trust levels and the folder that produced it.
- [realtime_and_misc.md](codebase/realtime_and_misc.md): RTSP live pipeline, YouTube analysis, latency,
  train/serve skew, root files, dependency list.

### Research (`docs/research/`)
- [research/README.md](research/README.md): short brief and prioritized shortlist.
- [research/datasets.md](research/datasets.md): public datasets with measured lead-in statistics.
- [research/backbones.md](research/backbones.md): frozen encoders for video, audio, speech, pose, VLM teacher.
- [research/literature.md](research/literature.md): anticipation literature, objectives, evaluation
  protocols, realistic accuracy ceilings.

---

## 7. Working conventions

- Python: `.venv/bin/python` (Python 3.10,
  torch 2.5.1+cu121, transformers 4.57.5, ultralytics 8.3.228). `requirements.txt` lists the imports of the
  kept code (`anticipation/`, `baselines/`, `setup/`, `tools/`, `tests/`) with the venv's installed versions;
  `legacy/` needs more (see [realtime_and_misc.md](codebase/realtime_and_misc.md) section 9).
- New code runs from the repo root; legacy code runs from `legacy/` (see `legacy/README.md`).
- GPUs: two data-centre GPUs. `nvidia-smi` is broken by a driver mismatch, but torch CUDA works. Many scripts
  default to `cuda:1`; pass `--device` explicitly.
- New evaluation code must follow [BASELINE.md](BASELINE.md) section 6. Do not reuse the archived
  `simulation_data/*.npz`, the removed `sim/comprehensive_eval.py` event metrics, or `legacy/*_index.json` as a benchmark.

---

## 8. Repository cleanup (2026-10)

Executed on 2026-10-14 following [REPO_CLEANUP_PLAN.md](REPO_CLEANUP_PLAN.md) (with the owner's amendments recorded
there). The working tree without `.venv/` and `.git/` went from about 6.0 GB to about 0.11 GB.

- **Moved to `legacy/`** (31 tracked files, `git mv`): `core/`, `vlm/`, `training/{train,dataset,evaluate,train_temporal_model}.py`,
  `configs/{paths,training}.yaml`, the producer scripts in `scripts/` (`preprocess_anticipation`, `generate_labels`,
  `vlm_label_dataset`, `build_multihorizon_labels`, `build_anticipation_labels`, `eval_vlm_teacher`, `analyze_youtube`),
  `realtime/{rtsp_inference,inspect_checkpoint}.py`, `experiments/fast_analyze_curation.py`,
  `setup/build_{ucf_crime,xdviolence}_metadata.py` and the root `{train,val,test}_index.json`.
- **Moved to `baselines/`**: `training/train_student_original.py` and the honest runs `runs/student_fix1`,
  `runs/student_fix2`, `outputs/student_mlp`, `results/anticipation_prboost_v3` plus `results/vlm_eval*`
  (now `baselines/reference_runs/<name>/`). `curate_anticipation_dataset.py` became `setup/curate_youtube.py`.
- **Removed tracked files** (`git rm`, 840 files: invalid result folders, superseded code, stale tests, logs):
  recover with `git checkout pre-cleanup -- <path>` (tag `pre-cleanup` = commit 9f14b92; oldest snapshot: commit 2164ca4).
- **Archived ignored artifacts** (6.1 GB, relative paths preserved, md5-verified): `data/archive/repo_legacy/`
  (`ARCHIVE_LIST.txt`, `MD5SUMS`, `README.md`). It holds the root MLPs, `yolov8n.pt`, the `probe_clip_01` arrays,
  `good_video_indices.npy`, all 100 gen-1 epoch checkpoints, every LoRA adapter (best, final and per-epoch),
  `simulation_data/`, the legacy BiLSTM/contrastive/student weights, projected embeddings, the unique score
  traces, and the two shell sessions that trained the synthetic MLPs. Grace period 90 days: **purge date
  2027-01-12**, after a review with the owner.
- **Backup of the trusted weights**: `data/archive/baseline_weights/` (the four `best.pt` of
  `baselines/reference_runs/`, with `MD5SUMS`). Never delete it.
- **Deleted outright** (only regenerable or duplicate data): `test_video.mp4` (byte-identical to
  `data/anticipation_curation/videos/holdout_025.mp4`), the 11 `final/*/scores.npy` byte-identical to the archived
  `final/smoothing_sweep/scores.npy`, the empty `.part`/`.ytdl` download stubs, empty folders (`snapshots/`, `.locks/`,
  `lora_finetuned/worker_*`, three empty `final/` runs) and `__pycache__`/`.pytest_cache`.
- **Secrets**: `.env` is mode 600; `youtube_cookies.txt` moved out of the repo to `<cookies file outside the repo>`
  (directory 700, file 600).
