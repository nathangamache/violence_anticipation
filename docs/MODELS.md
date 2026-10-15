# Causal anticipation training

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

Models: `stats_lgbm`, `stats_logreg`, `gru`, `transformer`. Defaults are all five folds, seeds `0,1,2`,
20 epochs, patience 4, four CPU threads and CPU execution. GPU use must be explicit and only `cuda:1`
is accepted. Set `--threads 1` or `2` when extraction is busy. Every run needs a new `--run_name`;
existing output directories are never overwritten. `--max_minutes` defaults to 29 and accepts at most
30; training and scoring check this deadline between epochs, batches and videos. A single library fit,
feature snapshot or VA30 evaluation can run past a check, so keep experiments small on the shared server.

`--max_train_ticks 20000` and `--max_val_ticks 10000` bound the samples used for fitting/checkpoint
selection; `0` uses all usable ticks. Sampling is uniform without replacement, separately per seed,
after splitting and applying task masks. Every test and validation tick is scored, regardless of the
training/checkpoint sample limits. Reduce tick limits, epochs or hidden width for smoke tests.

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
   dimensions and videos with no usable ticks are also reported and skipped. Snapshot coverage is fixed
   before training so extraction progress cannot change membership between folds or seeds.
3. Streams are joined by exact `t_sec` onto `labels_for_video(...).t_sec`. Missing, invalid or nonfinite
   feature rows become zeros with a false per-stream presence mask. No interpolation or future filling
   occurs. Missing final ticks do not shorten the catalog timeline. Timestamps must be unique, sorted
   integer seconds. Per-video extraction metadata and source-file SHA256 values are saved in coverage.
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
validation log loss, with no internal random validation split and no test-set selection.

### Temporal models

Each sample contains a fixed trailing context ending at its scored tick. Contexts never cross videos;
short histories repeat the first available grid tick on the left. This avoids an explicit history-length
feature. A GRU resets its hidden state at every context, so its receptive field is bounded. The
Transformer uses a triangular causal attention mask and sinusoidal positions relative to the fixed
context; there are no absolute video timestamps unless the explicit control is enabled.

Each stream has LayerNorm, a learned linear projection and GELU. Its presence mask zeros the projection,
including its bias, and is concatenated into fusion. Modality dropout removes an entire stream across
a sample's context and changes the supplied mask consistently. The fused sequence enters a unidirectional
GRU or causal Transformer. The last token produces six discrete hazards and a separate violence-now logit.

Hazard bins are `(0,5], (5,10], (10,20], (20,30], (30,45], (45,60]` seconds. BCE is evaluated only where
`hazard_target >= 0`: bins after the event, right-censored bins and masked ticks have no hazard gradient.
Positive event-bin terms receive weight `clip(30 / time_to_onset, 0.25, 1)`; survival terms retain unit
weight, so distant positive events have less influence. Auxiliary violence-now BCE uses observed ticks
outside excluded spans, including actual violence and aftermath. Total loss is hazard BCE plus
`auxiliary_weight * now BCE`. AdamW, gradient clipping at 1 and validation-loss early stopping select the
checkpoint. The hazard score is `1 - product(1 - h_j)` over the first four bins, computed in log space.
The detection head never substitutes for this score.

## Artifacts and evaluation

Each run writes `benchmark/results/<run_name>/`:

| Artifact | Contents |
|---|---|
| `run_config.json` | Exact parsed options, command, git commit/dirty state, seeds, benchmark hashes, split membership, run status and evaluation command |
| `coverage.json` | Accepted video metadata and feature hashes; missing/invalid videos and valid tick counts per stream |
| `training_history.json` | Backend, best epoch, sample counts and validation loss per fold/seed |
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
The equivalent command is saved verbatim. To recompute a report:

```bash
nice -n 10 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  $PY -m anticipation.eval.va30 \
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

## Add a model

Implement the new model under `anticipation/models/`, add its name to the CLI choices and route it in
`train_run`. Reuse `FeatureStore`, `Normalizer`, `sample_ticks`, shared labels and the fold split lists.
Return one finite probability per catalog tick. Fit and select only on the supplied training and
validation sets, preserve test/validation fold provenance, and save the selected checkpoint and history.
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

Results from the initial runs will be recorded below after validation.
