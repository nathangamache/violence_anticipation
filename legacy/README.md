# legacy/: frozen code from the Nov 2025 to Apr 2026 approaches

Kept only as provenance for files in `data/` and for numbers cited in `docs/`. Do not build new work on it.
What each approach scored and why it failed: `docs/EXPERIMENT_HISTORY.md` and `docs/codebase/*.md`.

| Path | Produced / role |
|---|---|
| `core/`, `training/{train,dataset,evaluate}.py`, `configs/` | gen-1 ViolenceTransformer (weights: `archive/repo_legacy/checkpoints/`) |
| `training/train_temporal_model.py` | BiLSTM behind `archive/repo_legacy/outputs/outputs_temporal/` and the curated pseudo-onsets |
| `scripts/preprocess_anticipation.py`, `scripts/generate_labels.py` | `data/windowed/`, `data/processed/{tracks,actions,scenes,audio,graphs,labels}/` |
| `scripts/vlm_label_dataset.py`, `vlm/qwen_teacher.py` | `data/processed/vlm_emb*/` and `vlm_labels*/` (inputs of B0 and B-teacher) |
| `scripts/build_multihorizon_labels.py` | `data/processed/labels_multih/` |
| `scripts/build_anticipation_labels.py`, `scripts/eval_vlm_teacher.py` | `data/processed/labels_future_h*/` and `baselines/reference_runs/vlm_eval*/` |
| `setup/build_{ucf_crime,xdviolence}_metadata.py` | `data/metadata/*.json` (XD `label`/`event_frame` are wrong for 274 clips) |
| `realtime/rtsp_inference.py`, `scripts/analyze_youtube.py` | live RTSP and offline YouTube demos |
| `experiments/fast_analyze_curation.py` | LSTM pseudo-onsets in `data/anticipation_curation/candidates.json` |
| `{train,val,test}_index.json` | leaky gen-1 clip split (1438/180/180); never a benchmark |

`archive/` above means `data/archive/` (outside the repo). The
`archive/repo_legacy/` part is kept until its purge date, 2027-01-12 (see its `README.md`).

## How to run

Imports use bare package names (`core`, `training`, `scripts`, `vlm`), and `configs/` plus `*_index.json` are opened
relative to the CWD, so run from this folder:

```bash
cd legacy && ln -sfn ../data data          # once; legacy/data is gitignored
../.venv/bin/python -m training.evaluate --ckpt data/archive/repo_legacy/checkpoints/model_epoch60.pt
../.venv/bin/python scripts/build_multihorizon_labels.py --help
```

`setup/curate_youtube.py analyze|export` (run from the repo root) needs `PYTHONPATH=legacy`.

## Warnings

- Never run `setup/build_{ucf_crime,xdviolence}_metadata.py`: they take no arguments (`--help` runs them), ignore the
  CWD and overwrite the live `data/metadata/{ucf_crime,xd_violence}.json`, which `baselines/` and
  `anticipation/eval/baselines.py` read.
- `scripts/preprocess_anticipation.py` also takes no arguments and writes into the live `data/processed/` and
  `data/windowed/` (it only fills missing files, but loads the GPU extractors).
- `scripts/build_anticipation_labels.py` (default `--out_dir data/processed/labels_future_h30`) and
  `experiments/fast_analyze_curation.py` (default `--output_dir data/anticipation_curation`) write into the live
  `data/` by default. Pass new output dirs.
- `training/train.py` rebuilds `train_index.json` from every clip. Never run it unmodified.
- `scripts/vlm_label_dataset.py` writes to `data/processed/vlm_labels` and `vlm_emb` by default; rerunning it
  overwrites the B0 and B-teacher inputs. Pass new output dirs.
- Default model paths such as `outputs/outputs_temporal/best_temporal_model.pt` now live under
  `data/archive/repo_legacy/`; pass them explicitly.
- `core/extractors.py` loads YOLOv8n, VideoMAE and CLIP onto the GPU at import (`yolov8n.pt` is re-downloaded if
  missing).

Removed legacy files: `git show pre-cleanup:<path>`.
