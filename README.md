# Violence anticipation

Senior capstone project (LTU): anticipate interpersonal violence in real-world video (CCTV, bodycam, road rage,
phone footage) **30 seconds or more before it starts**, ideally from video, audio, pose and speech.

**Honest status.** No model in this repo anticipates violence 30 s ahead yet. The only leak-free result is a
frame-level MLP on Qwen2.5-VL embeddings (`baselines/`): test ROC-AUC 0.58 to 0.62 for "violence starts within
30 s" (base rate 0.081), XD-Violence-only test AUC at chance (0.50 to 0.56), median alarm lead about 8 s. Older
headline numbers (AUC 0.93 to 0.99, "0 false alarms per hour") came from label misalignment and leakage. New models
are scored with the VA30 benchmark.

Start with [docs/README.md](docs/README.md), then [docs/BASELINE.md](docs/BASELINE.md) and
[docs/EXPERIMENT_HISTORY.md](docs/EXPERIMENT_HISTORY.md). Agent rules are in [CLAUDE.md](CLAUDE.md).

| Path | What it holds |
|---|---|
| `anticipation/` | New pipeline: data catalog, onsets, labels, folds; VA30 metrics; causal 1 Hz feature extractors |
| `benchmark/` | VA30 artifacts: video catalog, events, grouped folds, baseline results |
| `baselines/` | B0 trainer `train_student_original.py` and `reference_runs/` (the numbers to beat) |
| `setup/` | Dataset acquisition and the curated YouTube hold-out (`curate_youtube.py`) |
| `tests/`, `tools/` | VA30 unit tests; server-load, CPU-governor and doc-check helpers |
| `legacy/` | Frozen earlier code, kept only as provenance (see `legacy/README.md`) |
| `docs/` | All documentation |
| `data`, `models` | Symlinks to videos, features and pretrained weights outside the repo |

Python: `.venv/bin/python` (3.10). Tests: `.venv/bin/python -m pytest tests -q`.
