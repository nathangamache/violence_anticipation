# Causal anticipation training

Status for the lead (2026-10-15):
- Observed-horizon evaluator and early clean negatives resolve the previous two blockers.
- Fold 1 is excluded from all new training, normalization, validation and evaluation.
- Section-8 band, extended-hazard, event-bag and causal-history changes are implemented.
- First-frame, source/time/scene nuisance and frozen-offset residual controls are available.
- Trainer + VA30 regression suite: 52 passed; expanded objective/control suite: 16 passed.
- One-epoch GRU diagnostic and corrected frozen historical-model replay completed.
- Registered Phase A now lives in `anticipation.screen.run`; this trainer does not replace its P1 grid.
- Person-masked scene controls support `siglip2_scene0p`; unmasked scene streams are diagnostic only.

This framework consumes cached 1 Hz features without loading media or backbones. Synthetic learning
and small real-data diagnostics establish integration, not real-world anticipation. Read
[MODALITY_PLAN.md](MODALITY_PLAN.md), [screen registration](../benchmark/screens/summary.md) and its
amendments before designing comparisons. The old 45-run trainer proposal is superseded; do not execute it.

## CLI

Use the isolated checkout and existing Python environment; do not install packages into it.

```bash
PY=.venv/bin/python
$PY tools/server_load.py --gpu
nice -n 10 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  $PY -m anticipation.train --streams vjepa2_l,siglip2_prompts,pose_interact \
  --model gru --context 180 --layers 1 --hidden 64 --device cpu \
  --horizons benchmark/extract_manifest_v1.csv --run_name gru_diagnostic_v1
```

Models: `stats_lgbm`, `stats_logreg`, `gru`, `transformer`, `time_only`, `first_frame`, `nuisance`,
`residual_logreg`. Stream names are arbitrary cached extractors, with independent validity masks.
Defaults: diagnostic phase, folds **0,2,3,4**, three seeds 0/1/2, band objective, 180 s context,
20 epochs, patience 4, CPU, at most four threads, 29-minute training/scoring budget. GPU use is explicit
and only `cuda:1` is accepted. Each run needs a fresh name; outputs are never overwritten. Individual
library fits, snapshots and evaluation can extend past a deadline check, so keep shared-server jobs small.

`--fold_ids 0,2,3,4` restricts models and videos; excluded-fold videos never enter fitting or validation.
`--phase synthetic` allows all five folds for custom synthetic fixtures only. Real fold 1 stays frozen.
`--phase screen` deliberately rejects execution and points to the registered `anticipation.screen.run`
harness, whose probe grid, cohorts and gates differ from this diagnostic trainer.

`--cohort ids.json` freezes coverage across comparisons. Training and validation sampling caps default
to 20,000 / 10,000 ticks; `0` uses all usable ticks. Every jointly observed validation/test tick is scored
regardless of these fitting caps. `--require_validation_metric` rejects missing matched validation
classes before fitting. Without it, an unavailable selection metric permits exactly one diagnostic epoch.

## Data and leakage boundaries

The trainer verifies `folds_v2.json` (fallback v1), never modifies folds, and uses only trusted violent
and clean-negative roles assigned to selected folds. Missing required files skip the entire video;
coverage records missing streams, invalid shapes, extractor metadata and file hashes. Snapshot membership
is fixed before training. Timestamps must be sorted unique integer seconds and streams join exactly.

Scores are emitted only at timestamps supplied by every required stream. An extractor's false validity
mask (for example missing audio) is retained as a separate presence indicator; its values become zero.
No interpolation or future filling is used. Feature means/scales and tabular scalers fit training data only;
there is no whole-video normalization. Per-stream temporal LayerNorm operates within each tick.

Labels reuse `anticipation.data.labels`, including exclusions, violence, aftermath and right censoring.
Clean-negative training duration is capped to its observed manifest horizon; violent durations are not
shortened. Snapshots are memory-mapped NumPy arrays with a bounded video cache. They may be deleted
after scoring; extraction files are never changed.

## Models and objectives

`stats_logreg` uses L2 logistic regression, C=0.1, Newton-CG. This solver avoids a reproducible native
SciPy LBFGS crash on float16-derived tables. `stats_lgbm` uses LightGBM if installed, otherwise sklearn
HistGradientBoosting, 15 leaves and 10 additional rounds/epoch at learning rate .05. Per-stream causal
mean/max/std/slope summaries use trailing 5/15/30/60 s plus presence fractions. These summaries are the
original baseline, **not** registered P1's 10/60/180 s summaries, PLS32 and C-grid.

Default `--objective band`: positives are unmasked pre-first-event ticks with onset **(30,120] s** ahead;
negatives are clean-negative ticks with at least 60 s remaining. Other ticks are excluded from tabular
band fitting. `--objective horizon30` retains the original y30 classifier, excluding ambiguous negative
ticks whose onset is 31–120 s away. Scores from either classifier need not be calibrated hazards.

Both temporal models apply per-stream LayerNorm, linear projection, GELU, presence masks and modality
dropout. Scalar streams project before LayerNorm to preserve variation. Fusion feeds a unidirectional
GRU or a Transformer with causal attention. Every score resets over a trailing finite 180–300 s context.
Incomplete histories are zero padded and explicitly masked: GRU packing removes padding; Transformer
keys mask it. `--history_dropout .25` randomly shortens training histories to reduce padding-length
shortcuts. Validation/inference use all available history. Absolute elapsed time enters only explicit
controls; context-relative Transformer encodings remain available.

There are ten censored hazard bins `(0,5],(5,10],(10,20],(20,30],(30,45],(45,60],(60,90],(90,120],
(120,180],(180,300]`, a violence-now head and a separate band head. Hazard BCE ignores censored and
post-event bins. Auxiliary now BCE excludes excluded spans. Far-positive distance down-weighting is
**off** by default; `--far_positive_downweight` restores positive-bin weight clip(30/tte,.25,1).
Temporal loss adds band BCE and event MIL (weights default 1 each) to hazard and auxiliary-now loss.

Event positive bags sample up to 16 ticks from `[max(onset-120,clear_start),onset-30]`, only observed
unmasked ticks. Each receives a clean bag from the same source at identical clock ticks, with sufficient
remaining footage. Unmatched events are counted and omitted from MIL. Normalized log-sum-exp pooling
allows a sparse positive warning within a bag. History never extends beyond its bag tick. All bags use
training videos only. `--event_weight 0` disables MIL; `--band_weight` controls tick-band loss.

Temporal `--score_head auto` chooses sigmoid(band) for the band objective and hazard30 otherwise.
`hazard30` computes 1-product(1-h) through 30 s; `hazard_band` computes survival(30)-survival(120).
The now head never receives anticipation credit. AdamW and gradient clipping train temporal models.

Checkpoint selection maximizes validation **source-and-time-matched** AUROC on (30,120] for band
or (30,60] for horizon30, using shared VA30 matching strata. Unmatched ticks have zero metric weight;
test outcomes and pooled AUROC never select a checkpoint. Earlier epochs win ties. Validation log
loss remains diagnostic. `--time_match_weights` defaults on for fitting; `--no-time_match_weights`
disables it. `--time_matched_training` also samples negatives to positive source/time-bin counts;
finite sample caps can perturb equality. Temporal hazard/now supervision retains its censoring semantics;
source/time matching weights apply to the tick band objective, and MIL pairs match exact clock time.

## Shortcut controls

`time_only` uses only elapsed seconds with the same tree backend and coverage pipeline.
`first_frame` broadcasts each required stream's tick-1 feature and validity mask; `--first_frame_tick`
changes the anchor, which contributes zeros until observed. No later feature value enters this control.
`--position_control` adds elapsed seconds to a primary model as an explicit diagnostic ablation.

`nuisance` fits source-kind × shared time-bin indicators plus training-only standardized scene PCA32
(or fewer components on tiny fixtures). The scene anchor is tick 2, zero before tick 2, using
`--scene_stream siglip2_scene0p` by default. This reads the extractor's single masked-frame `emb` format;
regular cached streams can be requested for an explicitly unmasked diagnostic. Scene hashes/missing
files and masking metadata are saved. PCA fits once per training-video set, never validation/test.

`residual_logreg` freezes the fitted nuisance logits as offsets, then learns a penalized logistic
correction from window features; predictions are sigmoid(nuisance logit + learned correction).
This is an offset residual baseline, not a substitute for the registered paired [N+block] minus N gate.
Use identical cohorts to compare controls, and report Success@30 with position null and random reference.

## Artifacts and observed-horizon evaluation

Each `benchmark/results/<run_name>/` stores exact config/git/package provenance, coverage, training
history, training-only normalizers, selected checkpoints and parquet columns
`video_id,t_sec,score,fold,seed`. Both validation and test scores retain the original model-fold ID.
Feature arrays, cache and weights remain ignored; commit reports/configs rather than model binaries.
Interrupted runs retain completed model scores, but automatic resume is not implemented.

The shared evaluator receives `Benchmark.load(..., horizons=manifest)` and `restrict_to_scored=True`.
A guard requires every video/model timeline to be a contiguous 1 Hz prefix reaching floor(observed
duration), allowing at most the last tick to be absent. Other internal gaps or truncation block evaluation.
For selected folds an in-memory copy remaps fold IDs for the evaluator, then restores original IDs in
reported fold maps; disk folds never change. Bootstrap clusters use `split_group` when provided.

Replay a selected-fold score file with:

```bash
nice -n 10 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  $PY -m anticipation.models.evaluation --scores benchmark/results/RUN/scores.parquet \
  --out_dir benchmark/results/RUN --folds benchmark/folds_v2.json \
  --horizons benchmark/extract_manifest_v1.csv --fold_ids 0,2,3,4 --threads 2
```

The exact command is saved in run_config. `--n_boot 1000` defaults to cluster intervals;
`--threshold_bootstrap 200` adds calibration uncertainty. Every validation slice requires observed
clean footage. Missing calibration classes preserve scores and return status `evaluation_unavailable`.
`--no_eval` explicitly saves scores alone. Partial cohorts are not the complete benchmark.

The earlier [exposure audit](../benchmark/results/trainer_evaluator_exposure_audit_20261014/audit.json)
demonstrated 12x exposure inflation before the horizon fix. It is historical evidence, not an active blocker.
`--horizons benchmark/extract_manifest_v1.csv` now removes unobserved capped-negative tails from
exposure and score holding. `anticipation.models.replay` can add newly extracted negatives to frozen
historical tabular checkpoints without refitting; those models predate the fold-1 freeze and are
explicitly historical diagnostics, even though their re-evaluation excludes fold 1.

## Extend and verify

Add implementations under `anticipation/models/`, register CLI routing, and reuse FeatureStore,
training-only Normalizer, shared labels and selected split lists. Return finite scores only at jointly
observed ticks; retain validation/test provenance. Select on matched validation AUROC and save config,
checkpoint and history. New models must pass precursor learning, post-onset-only credit and future
truncation checks. Temporal checkpoints record dimensions, hazard edges, score head and constructor
config; old six-bin states cannot be loaded into the extended ten-bin architecture directly.

```bash
nice -n 10 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  $PY -m pytest tests/test_train.py tests/test_train_plan.py -q
```

Tests cover all four primary models, a precursor starting 40 s before onset, post-onset-only signals,
future truncation/corruption, invalid stream masks, train-only normalization, censoring gradients,
source/time matching, five synthetic folds × three seeds, event bags, padded histories, train-only scene
PCA, fixed nuisance offsets and observed-horizon/frozen-fold guards.

## Results

Persisted synthetic runs below use the same 20-video fixture (10 violent, 10 clean), five
synthetic grouped folds and three seeds, with 100 bootstrap replicates. These are integration tests,
not numbers on the real VA30 v2 benchmark. Thresholds target **validation** 2 FA/h; the actual test rate
is shown separately. Values are means over seeds. The fixture plants the precursor at 40 seconds,
so only one third of the 30–60-second far window contains signal.

| Synthetic run | Far AUROC pooled / matched | Success@30 | Position null | Random reference | Test FA/h |
|---|---|---|---|---|---|
| [stats_lgbm (sklearn fallback)](../benchmark/results/trainer_synthetic_stats_20261014/report.md) | 0.666 / 0.664 | 100.0% | 30.0% | 11.5% | 4.91 |
| [GRU](../benchmark/results/trainer_synthetic_gru_20261014/report.md) | 0.622 / 0.624 | 100.0% | 66.7% | 23.8% | 10.91 |
| [Time-only control](../benchmark/results/trainer_synthetic_time_control_20261014/report.md) | 0.644 / 0.574 | 0.0% | 0.0% | 0.0% | 0.00 |

The short synthetic validation slice has only 220 seconds of clean footage per fold, so one alarm
already exceeds the target. Test FA/h can be much higher than the validation target. Zero seed
variance for the tree/control reflects this tiny fully sampled deterministic fixture. The time-only
model can still exceed 0.5 inside the evaluator's finite time bins; the event position null remains
necessary. Per-run reports contain bootstrap intervals and all operating points.

The [real SigLIP prompt diagnostic](../benchmark/results/trainer_siglip2_prompts_diagnostic_v2_20261014/run_config.json)
uses **folds_v2**, one fixed epoch per model, 5,000 training / 2,000 validation-loss tick caps,
and all five folds × three seeds. It writes **389,142 score rows across 345 videos** (344 violent,
one clean negative). No test metrics were selected or reported. Success@30, position null and random
reference are **unavailable** because validation folds 0–3 have no clean negatives. This run uses no
hyperparameter/checkpoint search and is not part of the modality screen. Checkpoint reload reproduces
the stored scores (see `checkpoint_reload.json` in the diagnostic and GRU run directories).

The [band GRU diagnostic](../benchmark/results/trainer_band_gru_diagnostic_20261015/report.md)
uses a frozen 50-video cohort (27 violent, 23 clean), original folds 0/2/3/4, seed 0, one epoch,
180 s context, 16 hidden units and four ticks per event bag. It is a bounded integration run,
with 256 training / 128 validation-selection ticks per model and 20 bootstrap replicates,
not a registered P2 comparison or a hyperparameter search. Training contained 10–15 matched
positive-event bags per fold. The 12,187 scores include validation slices and test folds.

| Real diagnostic | Far AUROC pooled / matched (30–60 s) | Success@30 | Position null | Random reference | Test FA/h |
|---|---|---|---|---|---|
| Band GRU, SigLIP prompts, 50 videos | .571 / .690 | 18.2% | 17.0% | 16.7% | 7.33 |

The operating point targets validation 2 FA/h; this tiny cohort has only 0.41 test clean hours and
wide intervals. Success@30's interval is 5.8–32.4%, so this does not establish gain over the position
null. The evaluator's existing Brier/reliability output assumes a 30 s probability and is **not a
calibration metric for the selected (30,120] band head**. The reported far AUROC is the evaluator's
30–60 s metric, distinct from the (30,120] checkpoint objective and registered M1.

The [historical SigLIP-prompt replay](../benchmark/results/trainer_horizon_replay_20261015/report.md)
adds available clean-negative scores using the old frozen tabular checkpoints, without refitting.
It covers 3,497 videos, 2,118,534 validation/test score rows and 125.86 observed clean-negative hours
(242.30 catalog hours); 527 evaluated negatives are shortened by the manifest. No fold-1 model or
video is evaluated, but the original models were trained before the prospective fold-1 freeze.
The results therefore remain historical diagnostics, not confirmation or registered screen outcomes.
All three original seeds are retained; the table shows seed means at validation-target 2 FA/h.

| Historical replay | Far AUROC pooled / matched | Success@30 | Position null | Random reference | Test FA/h |
|---|---|---|---|---|---|
| Frozen SigLIP prompts stats classifier, 3 seeds | .515 / .504 | 2.8% | 8.5% | 5.1% | 2.10 |

This replay shows no useful anticipation advantage. Twenty bootstrap replicates suffice only for
this diagnostic; use registered bootstrap settings for scientific comparisons. Both new runs include
`frozen_fold_check.json`, confirming zero scored videos assigned to real fold 1. The new GRU also
excludes fold 1 from preprocessing and fitting, unlike the explicitly historical replay checkpoints.
