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
- The impressive numbers in `final/`, `outputs/` and older `results/` folders (AUC 0.93 to 0.99,
  "0 false alarms per hour", "92 % detection") come from label misalignment, leaky splits, simulations
  built mostly from training videos, test-tuned thresholds and lenient event metrics. See
  [EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md) and [codebase/results_inventory.md](codebase/results_inventory.md).
- The main bottleneck is data. Only **112 of the 640 onset-annotated videos have 30 s or more of footage
  before violence starts** (49 of them interpersonal), and the test splits hold only 10 to 12 of them.

The root `README.md` is **stale**. It describes a graph-transformer system with DeepSORT, real audio,
ONNX export and passing tests that was never actually built (audit in
[codebase/core_and_training.md](codebase/core_and_training.md) section 14). Trust `docs/` instead.

---

## 2. Reading order for a new agent

1. This file (layout, data locations, pipelines).
2. [BASELINE.md](BASELINE.md): the honest number to beat, the evaluation flaws, and the protocol any new
   model must be scored with.
3. [EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md): what was tried, what it scored, and why it failed.
4. [codebase/data_and_labels.md](codebase/data_and_labels.md): where the onsets come from, the time-axis
   contract, label sets, splits and data bugs. **Required before touching any label or feature file.**
5. [research/README.md](research/README.md): which public datasets and pretrained backbones to add.
6. The per-area codebase docs below, as needed.

---

## 3. Repository layout (one line per top-level entry)

Repo root: `.`.

| Entry | What it holds |
|---|---|
| `core/` | Gen-1 graph pipeline: `model.py` (`ViolenceTransformer`, non-causal), `extractors.py` (YOLOv8n, VideoMAE, CLIP; loads models at import), `graph.py`, `utils.py`. |
| `training/` | `train.py` (gen-1; **overwrites `train_index.json`**), `evaluate.py`, `eval_anticipation.py`, `dataset.py`; gen-2 heads: `train_student_original.py` (**the only correctly aligned trainer**), `train_student.py`, `train_temporal_model.py` (BiLSTM, misaligned labels). |
| `vlm/` | `qwen_teacher.py`: Qwen2.5-VL-7B wrapper (JSON verdicts, 3584-d image-token embeddings). |
| `llm/` | Text-LLM decision heads on graph summaries (failed; leaky prompts). |
| `llm_data/` | `train.jsonl` / `val.jsonl` prompts and prediction files for the text-LLM head. |
| `scripts/` | Data and label builders (`build_multihorizon_labels.py`, `vlm_label_dataset.py`, `preprocess_anticipation.py`), feature extractors (audio, pose), fusion, `eval_vlm_teacher.py`, `evaluate_with_smoothing.py`, `late_fusion.py`, `analyze_youtube.py`, broken legacy tools (`benchmark.py`, `export_trace_graph.py`). |
| `setup/` | Metadata builders for UCF-Crime / XD-Violence (`build_xdviolence_metadata.py` has an onset bug), Kinetics download and finetune scripts. |
| `tuning/` | Contrastive projectors, optical-flow extraction, VLM+flow fusion, LoRA finetune of Qwen2.5-VL, VQA question features, ensemble and contrastive simulators. |
| `sim/` | "8-hour shift" stream builders and evaluators (`comprehensive_eval.py`, `live_simulator.py`, `temporal_simulator.py`). All have label and metric flaws. |
| `experiments/` | Mar to Apr 2026 experiment scripts (EMA, algorithm variations, UCF testing, person filtering, cross-dataset, curation pseudo-labelling) and `output/`. |
| `continuation/` | Time-to-violence ETA heuristic (unused), LDA calibration probe, domain adaptation (never run). |
| `realtime/` | `rtsp_inference.py` (the only working live path: Qwen embeddings + BiLSTM), `inspect_checkpoint.py`, broken `pipeline.py`. |
| `tools/` | Split builders and checkers (`make_nonoverlap_splits.py`, `check_split_overlap.py`), debug helpers. |
| `tests/` | Four stale test files. None pass; `pytest` is not installed. |
| `configs/` | YAML configs for gen-1. `model.yaml` and `features.yaml` are never read. |
| `curate_anticipation_dataset.py` | YouTube search/download pipeline that built `data/anticipation_curation/` (92 escalation videos). |
| `outputs/`, `outputs_fused/`, `runs/`, `results/`, `final/`, `simulation_results*/` | Result folders (metrics JSON/CSV, plots, checkpoints). Inventory and trust levels in [codebase/results_inventory.md](codebase/results_inventory.md). |
| `simulation_data/` | Stitched evaluation streams (`*.npz`, gitignored, 3.5 GB). **Do not use as a benchmark.** |
| `checkpoints/`, `best_checkpoints/` | Gen-1 `ViolenceTransformer` epoch checkpoints (Dec 2025, gitignored, no metrics). |
| `lora_finetuned/`, `lora_finetuned_v2/` | Collapsed LoRA adapters (gitignored, do not reuse). |
| `logs/`, root `inference_*.log` | VLM extraction, audio extraction, LoRA and RTSP demo logs. |
| `snapshots/` | Empty alert-snapshot folder of `rtsp_inference.py`. |
| Root `*.pt`, `*.npy` | Orphaned MLP state dicts (Mar 2), YOLO weights (`yolov8n.pt`, `yolov8m-pose.pt`), `probe_clip_01_*` YouTube probe arrays, `good_video_indices.npy` (unreferenced). |
| Root `*_index.json` | `train/val/test_index.json` (1438/180/180 clips, **movie-level leakage**); `eval_index.json` is identical to `test_index.json`. |
| `data` -> `data/` | Symlink to all videos, metadata and features (outside git). |
| `models` -> `models/` | Symlink to pretrained weights (only `Qwen2.5-VL-7B-Instruct`). |
| `docs/` | This documentation. |

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

Pretrained weights: `models/Qwen2.5-VL-7B-Instruct` (16 GB). YOLO weights in the repo root.

---

## 5. The main pipelines (by generation)

| Generation | Dates | Data flow | Entry points | Doc |
|---|---|---|---|---|
| 1. Graph transformer | Nov to Dec 2025 | windowed clip -> YOLOv8n boxes + VideoMAE crop embeddings + clip-mean CLIP + zero audio -> per-step graph -> `ViolenceTransformer` | `python -m scripts.preprocess_anticipation`, `python -m training.train` (dangerous), `python -m training.evaluate` | [core_and_training.md](codebase/core_and_training.md) |
| 2. VLM teacher + embeddings | Jan to Feb 2026 | raw video -> every 5th graph step -> Qwen2.5-VL-7B JSON verdict + 3584-d embedding | `python -m scripts.vlm_label_dataset`, `scripts/eval_vlm_teacher.py` | [vlm_and_llm.md](codebase/vlm_and_llm.md) |
| 3. Heads on frozen embeddings | Feb 2026 | `vlm_emb` (+ flow/audio/pose) + `labels_multih` -> MLP / BiLSTM / contrastive heads | `training/train_student_original.py` (honest), `training/train_temporal_model.py`, `training/train_student.py`, `tuning/contrastive_*.py` | [temporal_and_student_models.md](codebase/temporal_and_student_models.md) |
| 4. Multimodal fusion | Feb 2026 | flow / audio / pose extraction -> early concat or late score fusion | `tuning/optical_flow_features.py`, `scripts/audio_feature_extraction.py`, `scripts/pose_feature_extraction.py`, `tuning/fuse_features.py`, `scripts/late_fusion.py` | [audio_pose_flow_fusion.md](codebase/audio_pose_flow_fusion.md) |
| 5. Simulation and evaluation | Feb to Apr 2026 | per-video embeddings stitched into "8-hour" streams -> event metrics, smoothing sweeps | `sim/comprehensive_eval.py`, `scripts/evaluate_with_smoothing.py`, `experiments/run_experiments.py` | [simulation_and_evaluation.md](codebase/simulation_and_evaluation.md) |
| 6. Deployment and curation | Feb to Apr 2026 | RTSP or YouTube -> Qwen embedding at 1 fps -> BiLSTM -> alerts; YouTube search -> LSTM pseudo-labels | `realtime/rtsp_inference.py`, `scripts/analyze_youtube.py`, `curate_anticipation_dataset.py`, `experiments/fast_analyze_curation.py` | [realtime_and_misc.md](codebase/realtime_and_misc.md) |

---

## 6. Full document index

### Top level
- [BASELINE.md](BASELINE.md): honest baseline numbers, protocol, evaluation flaws, and the specification of
  the corrected benchmark (VA30) that every new model must be scored on.
- [EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md): every approach tried, in order, with its score, its
  result folder, and why it failed or plateaued.

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
  torch 2.5.1+cu121, transformers 4.57.5, ultralytics 8.3.228). `requirements.txt` is incomplete; see
  [realtime_and_misc.md](codebase/realtime_and_misc.md) section 9 for the real dependency list.
- Run from the repo root. Package-style scripts (`core`, `training`, `scripts`, `vlm` imports) need
  `python -m training.train` style invocation.
- GPUs: two data-centre GPUs. `nvidia-smi` is broken by a driver mismatch, but torch CUDA works. Many scripts
  default to `cuda:1`; pass `--device` explicitly.
- Never run `python -m training.train` unmodified: it overwrites `train_index.json` with every clip.
- New evaluation code must follow [BASELINE.md](BASELINE.md) section 6. Do not reuse
  `simulation_data/*.npz`, `sim/comprehensive_eval.py` event metrics, or the root index split as a benchmark.
