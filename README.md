# Violence anticipation in real-world video

Can a model warn that a fight is about to start, seconds before the first blow, from ordinary video (CCTV,
body-worn cameras, dashcams, phone footage)? This repository holds the research code, the VA30 benchmark
definition and the measured results for that question. It began as a solo senior capstone at Lawrence
Technological University (advisor Dr. Eric Martinson, fall 2025 to May 2026) and was rebuilt in October 2026.

> **Ethics and scope.** The model outputs an uncertain risk estimate meant for review by a person. It is not
> evidence of anyone's intent, it does not identify people, and no claim is made that it is safe or useful to
> deploy. All footage comes from published research datasets; no video, frame or face is stored in this
> repository (see [DATA.md](DATA.md)).

## Headline results (October 2026, labels v3)

Two frozen heads, both linear probes (P1) on SigLIP 2 prompt scores: an **early-warning head** (prompt set v1,
trained for onsets 30 to 120 s ahead) and an **imminent head** (prompt set v2, onsets within 10 s). They were
selected on development folds 0, 2, 3 and 4, then scored once on fold 1, which no step had touched before.

**Fold-1 confirmation (single run, 1,214 held-out videos, 75 eligible fights, 30.7 clean hours).** Both heads
passed their pre-registered rules.

| Head | AUROC (95% CI) | Threshold tuned for | False alarms per hour on fold 1 | Fights warned in time | Threshold-aware CI | Position null |
|---|---|---|---|---|---|---|
| Early, warned 30 s+ ahead | 0.905 [0.848, 0.948] (30-120 s window) | 2 / h | 0.72 | 22.7% | [13.7, 38.8]% | 4.3% |
| Early, warned 30 s+ ahead | | 5 / h | 2.64 | 36.0% | [23.8, 48.0]% (bootstrap) | 8.6% |
| Imminent, warned 10 s+ ahead | 0.893 [0.843, 0.938] (last 10 s) | 2 / h | 0.85 | 20.0% | [10.2, 39.2]% | 5.4% |
| Imminent, warned 10 s+ ahead | | 5 / h | 3.23 | 45.3% | [27.5, 58.6]% | 11.7% |

**Development folds (post-selection, 3,515 test videos, 344 eligible fights, 125.9 clean hours).** From
[`benchmark/results/fusion_v2/events/va30_fa125/report.md`](benchmark/results/fusion_v2/events/va30_fa125/report.md)
and [`benchmark/results/fusion_v2/imminent/report.md`](benchmark/results/fusion_v2/imminent/report.md):

| Head | Test FA/h (2 FA/h target) | Success | Threshold-aware CI | Position null | Random |
|---|---|---|---|---|---|
| Early (Success@30) | 2.12 | 31.4% | [23.9, 37.8]% | 9.5% | 5.2% |
| Imminent (Success@10) | 2.15 | 33.4% | [25.8, 41.6]% | 11.2% | 6.4% |

Full write-up: [docs/RESULTS_V3.md](docs/RESULTS_V3.md). Per-block screen:
[`benchmark/results/cue_screen_v2/results_phase_A.md`](benchmark/results/cue_screen_v2/results_phase_A.md).
Single-stream detail, including the crowd-matched check:
[`benchmark/results/single_block/siglip2_prompts_v1/`](benchmark/results/single_block/siglip2_prompts_v1/).

## How to read these numbers

- **Success@L** is the share of eligible fights (at least 30 s of calm footage before them) whose first alarm
  came at least L seconds before the onset. Alarms need 2 of 3 consecutive seconds above the threshold and count
  once per episode.
- **False alarms per hour** are counted on clean videos with no violence. The threshold is chosen on validation
  data for a target rate (1, 2 or 5 per hour). The test rate is reported as observed and can differ from the
  target; on fold 1 it came out lower.
- **Position null** is the best Success@L a policy can reach at the same false-alarm rate when it knows only the
  video source and how long the clip has been running. A model has to beat it, not only random alarms.
- **Threshold-aware CIs** also resample the validation data used to pick the threshold, so they are wider than
  plain bootstrap intervals. These are the intervals to quote.

## Caveats

- The development-fold numbers are post-selection: the same folds chose the streams. The fold-1 confirmation is
  the out-of-sample check, but it has only 75 events, so its intervals are wide.
- One seed and the linear probe (P1) only; the registered GRU alternative (P2) was not run.
- The score is roughly flat with time to onset: footage that later turns violent scores high from the start.
  Much of the early warning is recognition of the kind of clip and setting, not of a visible build-up. A score
  from the first frame alone already reaches AUROC 0.82 on the development folds.
- Movie and YouTube clips dominate both the hits and the false alarms. At the shared 2 FA/h threshold the
  development false-alarm rate is 4.34 per hour on XD-Violence movie clips and 0.09 per hour on UCF-Crime CCTV.
- The v3 onset labels were corrected by AI visual review (blind two-pass review with reconciliation by language
  models), not by human annotators. 162 of the 419 eligible events keep an onset the reviewers could not resolve.
- Audio was audited but never modelled.
- An external test on NTU CCTV-Fights plus never-violent MEVA footage is pre-registered
  ([plan](benchmark/external/external_test_plan_v1.md)) and not yet reported.

## Two eras

- **Capstone era, November 2025 to April 2026** (frozen in [`legacy/`](legacy/README.md)): a graph and temporal
  transformer, a Qwen2.5-VL zero-shot teacher with embedding students, LoRA finetuning, optical-flow, audio and
  pose fusion, long stitched simulations and an RTSP demo. Most headline numbers from that period (AUC 0.93 to
  0.99, "0 false alarms per hour") came from label misalignment and leakage. The only leak-free result was a
  frame-level MLP on Qwen embeddings with test AUROC 0.58 to 0.62 and a median warning lead of about 8 s
  ([docs/BASELINE.md](docs/BASELINE.md), [docs/EXPERIMENT_HISTORY.md](docs/EXPERIMENT_HISTORY.md)).
- **Rebuild, October 2026** (`anticipation/`, `benchmark/`): the VA30 benchmark (1 Hz ticks in seconds, grouped
  folds, clean negatives, position nulls, bootstrap CIs), causal 1 Hz feature extractors, blind visual correction
  of the onset labels (labels v3), a pre-registered modality screen with control baselines, two-head fusion and a
  single fold-1 confirmation. The rebuild was carried out with AI coding agents under the owner's direction; every
  decision, amendment and result is recorded under `benchmark/` and `docs/`.

## Repository map

| Path | What it holds |
|---|---|
| `anticipation/` | Pipeline: data catalog, onsets, labels and folds (`data/`), VA30 metrics (`eval/`), causal feature extractors (`features/`), the registered screen, fusion and confirmation (`screen/`) |
| `benchmark/` | VA30 definition (catalog, events, folds), labels v3, registrations and amendments (`screens/`), all result reports (`results/`) |
| `baselines/` | The leak-free capstone baseline and its reference runs |
| `setup/` | Dataset acquisition and rebuild scripts |
| `tests/`, `tools/` | Unit tests and helpers |
| `legacy/` | Frozen capstone-era code, kept as provenance |
| `docs/` | Documentation, starting at [docs/README.md](docs/README.md) |
| `report/` | Capstone report (draft) |

## Reproduce the headline table from a fresh clone

The committed out-of-fold scores of the early head are enough to regenerate its event-level report; no video or
feature file is needed.

```bash
git clone https://github.com/nathangamache/violence_anticipation && cd violence_anticipation
python3 -m venv .venv && .venv/bin/pip install -r requirements-eval.txt
PYTHONPATH=. .venv/bin/python -m anticipation.eval.va30 \
  --videos benchmark/v3/videos.csv --events benchmark/v3/events.json \
  --scores benchmark/results/fusion_v2/events/va30_scores.parquet \
  --folds benchmark/results/fusion_v2/events/folds_without_confirmation.json \
  --horizons benchmark/extract_manifest_v1.csv --restrict_to_scored \
  --threshold_bootstrap 200 --n_boot 1000 --fa_targets 1,2,5 --cluster split_group \
  --out_dir /tmp/va30_repro
diff <(grep -v Generated /tmp/va30_repro/report.md) <(grep -v Generated benchmark/results/fusion_v2/events/va30_fa125/report.md)
```

The diff should show only the provenance lines. Retraining the heads, the screen
(`python -m anticipation.screen.run`), fusion (`python -m anticipation.screen.fusion`) and the confirmation
(`python -m anticipation.screen.confirm`) need the datasets and extracted features: see [DATA.md](DATA.md),
[docs/FEATURES.md](docs/FEATURES.md) and [docs/FUSION.md](docs/FUSION.md). The full environment is in
`requirements.txt` (CUDA 12.1 builds of PyTorch). Local data lives under `./data` or the directory named by
`VA_DATA_ROOT`.

The published catalogs differ from the ones used in the experiments only in anonymised hold-out identifiers,
relative paths and redacted reviewer notes; [`benchmark/public_redaction.json`](benchmark/public_redaction.json)
lists every such file with both hashes, and the registration checks map one to the other.

## License and citation

Code: MIT ([LICENSE](LICENSE)). Data and derived labels: see [DATA.md](DATA.md). Author: Nathan Gamache, Lawrence
Technological University.
