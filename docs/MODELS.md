# Causal anticipation training

Status for the lead (2026-10-14):
- Framework and initial tests pushed in `a30af89`; 69 suite checks passed; added scalar-stream and cohort checks also pass.
- Models: window statistics + logistic/boosting, causal GRU/Transformer; time-only control added.
- Checkpoints now use validation source-and-time-matched far-window AUROC, never pooled AUROC.
- Missing matched validation classes permit one diagnostic epoch only; modality screening requires both.
- Score tables now contain only timestamps shared by all requested streams; modality valid masks remain separate.
- Blocker: VA30 fills missing timestamps and uses catalog durations for FA/h; capped negatives are unsafe to evaluate.
- Blocker: the current real-feature snapshot has 345 videos but only one clean negative; folds 0–3 lack validation negatives.
- Modality screen is pending clean-negative extraction, the evaluator fix and the incoming modality decision rule.

`anticipation.train` learns from the cached 1 Hz streams described in [FEATURES.md](FEATURES.md).
It uses the frozen VA30 folds, shared tick labels and shared evaluator. No raw media or backbone model
is loaded. This framework establishes a reproducible training path; synthetic learning checks are not
evidence of real-world anticipation.

## Run

Run from an isolated checkout with the `data` and `models` symlinks. Use the existing environment;
no additional packages are required. Check shared-server load before experiments.

```bash
PY=.venv/bin/python
$PY tools/server_load.py --gpu

nice -n 10 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  $PY -m anticipation.train \
  --streams vjepa2_l,siglip2_prompts,pose_interact \
  --model stats_logreg --run_name stats_multistream_v1

nice -n 10 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  $PY -m anticipation.train \
  --streams vjepa2_l,siglip2_prompts,pose_interact \
  --model gru --context 120 --device cuda:1 --run_name gru_multistream_v1
```

Models: `stats_lgbm`, `stats_logreg`, `gru`, `transformer`, `time_only`. Defaults are all five folds, seeds `0,1,2`,
20 epochs, patience 4, four CPU threads and CPU execution. GPU use must be explicit and only `cuda:1`
is accepted. Set `--threads 1` or `2` when extraction is busy. Every run needs a new `--run_name`;
existing output directories are never overwritten. `--max_minutes` defaults to 29 and accepts at most
30; training and scoring check this deadline between epochs, batches and videos. A single library fit,
feature snapshot or VA30 evaluation can run past a check, so keep experiments small on the shared server.

`--max_train_ticks 20000` and `--max_val_ticks 10000` bound the samples used for fitting/checkpoint
selection; `0` uses all usable ticks. Sampling is uniform without replacement, separately per seed,
after splitting and applying task masks. Every test and validation tick shared by all required stream timestamp grids is scored,
regardless of training/checkpoint sample limits. Reduce tick limits, epochs or hidden width for smoke tests. `--cohort ids.json` accepts a JSON array
of benchmark video IDs, hashes it into the run config and restricts the feature snapshot. Freeze one
cohort before each modality comparison and reuse it for feature models and the time-only control;
otherwise extraction progress can make nominally identical stream lists cover different videos.

Additional temporal settings: `--context 60..300`, `--hidden 64`, `--layers 2`, `--heads 4`,
`--batch_size 32`, `--lr 0.001`, `--dropout 0.1`, `--modality_dropout 0.15`,
`--auxiliary_weight 0.2`. `--position_control` explicitly adds elapsed video seconds (scaled by 300
for temporal heads); leave it off for the primary models. Context-relative positional encodings remain
available to the Transformer. Padding, missingness and changing visual content can still reveal clip
position indirectly; read source/time-matched AUROC and the position null in every VA30 report.

## Data and leakage boundaries

1. `Folds.load` verifies `benchmark/folds_v2.json` if present, otherwise `folds_v1.json`. The trainer
   never changes fold files. Only trusted-onset `violent` and `clean_negative` videos in the fold file
   enter training or evaluation. Other anomalies, unlabelled videos, duplicates and curated hold-out
   videos cannot become training negatives.
2. A video missing any required stream is skipped for the entire run. Unreadable files, inconsistent
   dimensions and videos with no common timestamps are also reported and skipped. Snapshot coverage is fixed
   before training so extraction progress cannot change membership between folds or seeds.
3. Streams are joined by exact `t_sec` onto `labels_for_video(...).t_sec`. Missing, invalid or nonfinite
   feature rows become zeros with a false per-stream presence mask. No interpolation or future filling
   occurs. Scores are emitted only at the intersection of required-stream timestamps, including rows
   where a modality's valid mask is false (for example absent audio). Snapshots retain the 1 Hz grid up
   to the last common feature timestamp, with gaps masked; labels retain the original catalog duration
   for right-censoring. Timestamps must be unique, sorted integer seconds. Per-video extraction metadata and source-file SHA256 values are saved in coverage.
4. Per-dimension means and population standard deviations use valid feature ticks of the current
   training folds only. Validation/test statistics never participate. Invalid rows remain zero after
   normalization. There is no whole-video normalization. A zero-variance dimension has minimum scale
   `1e-4`; a stream absent throughout training has zero means and the same scale floor.
5. Labels come directly from `anticipation.data.labels.labels_for_video`: excluded spans, violent
   segments, aftermath and right-censoring retain the benchmark's semantics. Only the shared evaluator
   assigns anticipation credit. Neither detection targets nor post-onset alarms count as anticipation.

Snapshots are `.npy` arrays under the run's `cache/`, opened with memory mapping and an eight-video
LRU. Feature extraction outputs are never modified. Snapshots avoid repeatedly decompressing `.npz`
files during window sampling, but require disk space proportional to the selected feature streams.

## Models and objectives

### Window statistics

For each stream and each trailing window of 5, 15, 30 and 60 ticks, compute per-dimension mean, maximum,
population standard deviation and OLS slope per second over valid rows only. The window includes the
current tick and excludes ticks at or before `t - window`. A window with no valid values yields zeros;
a slope with fewer than two observations is zero. Four presence fractions per stream describe missingness
without adding the video timestamp. Statistics use normalized features; a second `StandardScaler`
is fitted on the sampled training statistics only.

`stats_logreg` uses L2 logistic regression (`C=0.1`, Newton-CG, warm-start optimization).
`stats_lgbm` uses LightGBM when importable, otherwise sklearn `HistGradientBoostingClassifier`.
Both tree backends add 10 boosting rounds per epoch, with 15 leaves and learning rate 0.05.
Newton-CG avoids a reproducible native SciPy LBFGS crash on float16-cache-derived tables in the supplied environment.
Tabular models predict the uncensored, unmasked `y_30` target, using positives within 30 seconds
and negatives with no onset within 120 seconds, as in VA30's per-horizon metrics. Ticks with an onset
31–120 seconds away are excluded from fitting and checkpoint loss, so legitimate early precursor
warnings are not trained as false alarms. This output is a conditional classification probability;
it is not guaranteed to be calibrated to unconditional P(onset within 30 seconds). VA30 still reports
its Brier/reliability diagnostics on the full horizon population. Checkpoints minimize held-out
validation source-and-time-matched far-window AUROC, with no internal random validation split
and no test-set selection. Log loss is retained as a diagnostic only.

### Temporal models

Each sample contains a fixed trailing context ending at its scored tick. Contexts never cross videos;
short histories repeat the first available grid tick on the left. This avoids an explicit history-length
feature. A GRU resets its hidden state at every context, so its receptive field is bounded. The
Transformer uses a triangular causal attention mask and sinusoidal positions relative to the fixed
context; there are no absolute video timestamps unless the explicit control is enabled.

Each stream has LayerNorm, a learned linear projection and GELU. Scalar streams project before
LayerNorm so a one-dimensional input does not collapse to a constant; other streams normalize first. Its presence mask zeros the projection,
including its bias, and is concatenated into fusion. Modality dropout removes an entire stream across
a sample's context and changes the supplied mask consistently. The fused sequence enters a unidirectional
GRU or causal Transformer. The last token produces six discrete hazards and a separate violence-now logit.

Hazard bins are `(0,5], (5,10], (10,20], (20,30], (30,45], (45,60]` seconds. BCE is evaluated only where
`hazard_target >= 0`: bins after the event, right-censored bins and masked ticks have no hazard gradient.
Positive event-bin terms receive weight `clip(30 / time_to_onset, 0.25, 1)`; survival terms retain unit
weight, so distant positive events have less influence. Auxiliary violence-now BCE uses observed ticks
outside excluded spans, including actual violence and aftermath. Total loss is hazard BCE plus
`auxiliary_weight * now BCE`. AdamW and gradient clipping at 1 optimize the loss; validation
source-and-time-matched far-window AUROC selects the checkpoint. The hazard score is `1 - product(1 - h_j)` over the first four bins, computed in log space.
The detection head never substitutes for this score.

## Artifacts and evaluation

Each run writes `benchmark/results/<run_name>/`:

| Artifact | Contents |
|---|---|
| `run_config.json` | Exact parsed options, command, git commit/dirty state, seeds, benchmark hashes, split membership, run status and evaluation command |
| `coverage.json` | Accepted video metadata and feature hashes; missing/invalid videos and valid tick counts per stream |
| `training_history.json` | Backend, best epoch, sample counts, matched validation AUROC and diagnostic losses per fold/seed |
| `scores.parquet` | `video_id, t_sec, score, fold, seed`; every covered test and validation tick from its fold's selected model |
| `checkpoints/normalizer_fold*.npz` | Training-fold means, scales and valid tick counts |
| `checkpoints/fold*_seed*.pkl` or `.pt` | Selected tabular model or temporal state/config/dimensions |
| `metrics.json`, `report.md`, plots, event CSV | Shared VA30 output when threshold calibration is possible |
| `cache/` | Aligned feature snapshots, safe to remove after training/scoring |

Weights, raw feature arrays and cache files must not be committed. The checkpoints directory and NumPy
arrays are already ignored by the repository. Each finished fold/seed first writes a small parquet;
these are merged into `scores.parquet` after all models finish and then removed. Interrupted runs retain
completed model scores and history, but automatic resume is not implemented; use a new run name.

By default the trainer calls the existing VA30 CLI entrypoint in-process, with evaluator thread count
capped, `--restrict_to_scored`, the selected fold file, bootstrap settings and `--meta run_config.json`.
The equivalent module command is saved verbatim. To recompute a full-timeline report while also
capping the evaluator's internal Python thread pool:

```bash
nice -n 10 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  $PY -c 'from anticipation.eval import va30; va30.N_THREADS=4; va30.main()' \
  --scores benchmark/results/stats_multistream_v1/scores.parquet \
  --out_dir benchmark/results/stats_multistream_v1 \
  --folds benchmark/folds_v2.json --restrict_to_scored \
  --meta benchmark/results/stats_multistream_v1/run_config.json
```

`--n_boot 1000` is the reporting default; `--threshold_bootstrap 200` adds uncertainty from threshold
selection. Partial feature coverage is always reported and is not directly comparable to the full
benchmark. Every validation slice needs covered clean negatives for FA/h calibration. If these are
missing, the trainer preserves all completed scores, records `evaluation_unavailable` and returns 2;
it does not invent negatives, move validation videos or use test videos to choose a threshold.
`--no_eval` explicitly requests scoring alone and records `scored_without_evaluation`.


## Shortcut controls and checkpoint selection

`--model time_only --streams <the same streams>` runs the same folds, labels, sample selection,
validation metric, checkpointing and scoring pipeline, but its classifier sees only elapsed tick time.
It uses the same boosting backend as `stats_lgbm`; visual values, modality masks, IDs and source labels
are not classifier inputs. The stream list establishes identical feature coverage for the comparison.

`MatchedValidation` uses the shared evaluator's `time_match_weights` and `TIME_MATCH_EDGES`. Positives
are unmasked first-event ticks with `30 < time_to_onset <= 60`; negatives are observed clean-negative
ticks with at least 60 seconds remaining. Matching crosses source kind with time bins (5 seconds up to
120, the evaluator's wider bins afterward). Non-comparable positives and negatives without positive
mass have zero metric weight. If capped, validation metric sampling uses fixed seed 0, independent of
training seed, and never uses test data. `training_history.json` records candidate, sampled and matched
tick counts. Validation log loss cannot break AUROC ties: the earlier checkpoint wins. If a fold has no
matched validation positives/negatives, exactly one fixed diagnostic epoch is saved without tuning.
`--require_validation_metric` rejects such a run before fitting any fold; use it for the modality screen.

`--time_matched_training` pairs positive and negative training ticks within the same source/time bins,
without replacement, discarding unmatched ticks. Stats positives are `y30`; temporal positives have
an observed positive hazard bin (onset within 60 seconds). Each matched stratum contributes equal
positive and negative counts. Temporal violence/aftermath samples remain for the auxiliary head.
A final optional tick budget is a uniform subsample of this balanced pool, so finite sampling can
introduce small count differences. Validation and test populations are never balanced or resampled
for reported scores. This option changes class priors; read the calibration diagnostics accordingly.

## Partial-timeline evaluator blocker

Audited against evaluator commit `a30af89`: `TickSet` reads `duration_v` from the catalog at
`anticipation/eval/va30.py:280`; `EventEval` computes exposure as catalog duration minus excluded spans
at line 577. `_collect` at lines 854–877 sample-and-hold fills missing score ticks, including the suffix
after the last feature. `--restrict_to_scored` at lines 1376–1377 restricts video IDs only. Consequently,
300 scored seconds of a 3,600-second clean negative can contribute 1 hour rather than 0.0833 hours.
This changes both thresholds and test FA/h, and the fabricated held tail can affect alarms too.
[Reproduction](../benchmark/results/trainer_evaluator_exposure_audit_20261014/reproduce.py) and
[audit output](../benchmark/results/trainer_evaluator_exposure_audit_20261014/audit.json) demonstrate
exactly 3,300 held ticks and a 12x exposure inflation on a synthetic one-hour negative scored for 300 seconds.

The trainer does not modify the evaluator. It blocks automatic evaluation when any covered video has
fewer jointly scored ticks than catalog ticks, saves the exact partial score table and records
`evaluation_unavailable`. `coverage.json` records catalog/snapshot/scored tick counts; split coverage
records both actual scored clean hours and catalog hours. Once the lead fixes missing-tick exposure,
remove the conservative guard in `anticipation.train` and rerun the saved scores with the corrected
evaluator and `--restrict_to_scored`. No real partial-timeline FA/h or Success@30 numbers are valid yet.

## Modality screen

When negatives and the evaluator fix arrive, screen `vjepa2_l`, `siglip2`, `siglip2_prompts`,
`pose_interact`, `motion` singly, then pairs, with `stats_lgbm` and `gru`, all five v2 folds and seeds
`0,1,2`. Keep model/sampling settings identical across streams, require matched validation coverage,
and include the time-only control on each compared coverage cohort. Do not select on pooled AUROC or
test outcomes. Read the incoming `MODALITY_PLAN.md` decision rule before choosing modalities.
A [45-run plan](../benchmark/results/trainer_modality_screen_plan_20261014/screen_plan.json)
lists the 15 stream combinations times two feature models plus the time-only control. It is a
proposal pending the incoming decision rule; none of these real modality-screen jobs has launched.
Every comparison must show coverage, matched far-window AUROC, Success@30, the position null and the
random reference together. Different feature coverage is not an apples-to-apples modality comparison.

## Add a model

Implement the new model under `anticipation/models/`, add its name to the CLI choices and route it in
`train_run`. Reuse `FeatureStore`, `Normalizer`, `sample_ticks`, shared labels and the fold split lists.
Return one finite score per jointly observed tick. Fit only on training data, select on matched far-window
validation AUROC, preserve test/validation fold provenance, and save the selected checkpoint and history.
Add the model to the synthetic precursor, detection-only and truncation tests before running real data.

`Normalizer.load(path)` restores preprocessing. Tabular checkpoints contain a `StatsModel` with
`predict(video, normalizer)`. Temporal checkpoints contain dimensions, parsed config and state dict;
construct `CausalTemporalModel` with those settings, load its state, move it to the chosen device and call
`predict_temporal`. Load pickle/torch checkpoints only from trusted runs.

## Validation and results

```bash
nice -n 10 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  $PY -m pytest tests/test_train.py -q
```

The tests exercise all four models with a planted precursor starting 40 seconds before onset,
post-onset-only signals, future truncation and future-feature corruption. They check timestamp joins,
invalid-row normalization, manually computed window statistics, censored-bin gradients, missing-stream
coverage, and a complete five-fold/three-seed CLI run through VA30. Synthetic event thresholds are
selected on each fold's validation clean negatives using the shared alarm policy.

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

[Screen status table](../benchmark/results/trainer_modality_screen_plan_20261014/screen_results.csv)
lists every planned single-stream/pair/model/control combination, current file coverage and missing
validation negatives. Empty metric cells mean not run. Rerun coverage checks after extraction completes,
freeze each comparison's cohort, and follow the incoming modality decision rule before launching.

The synthetic feature arrays and checkpoints stay local and ignored. To regenerate the synthetic
inputs from their [recipe](../benchmark/results/trainer_synthetic_inputs_20261014/recipe.json):

```bash
nice -n 10 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY - <<'PYCODE'
import runpy
from pathlib import Path
from anticipation.features import io
root = Path("benchmark/results/trainer_synthetic_inputs_20261014")
_, _, features, _ = runpy.run_path("tests/test_train.py")["synthetic_benchmark"](root)
io.FEATURE_ROOT = str(root / "features")
for vid, (t, x, valid, meta) in features.items():
    io.save_features("synthetic", vid, t, x, valid, meta)
PYCODE
```

Then reuse a synthetic run's recorded command with a fresh `--run_name`. Never regenerate the
real benchmark fold files.
