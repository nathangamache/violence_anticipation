# SCREEN: the pre-registered modality screen as implemented

Code: `anticipation/screen/`; tests: `tests/test_screen.py`; pre-registration: `benchmark/screens/screen_set_v1.json`
and `benchmark/screens/summary.md`. This file maps [MODALITY_PLAN.md](MODALITY_PLAN.md) section 6 to the code. Where
the plan left a choice open it is marked **Decision**. Evaluator machinery (`PairAUC`, `time_match_weights`,
`TIME_MATCH_EDGES`, horizons) comes from `anticipation/eval/va30.py` and is not modified.

## 1. Run

All commands run from the repo root with `.venv/bin/python`, CPU only (threads capped at 4; start under `nice -n 10`).

```bash
# 0. tests (about 30 s)
.venv/bin/python -m pytest tests/test_screen.py -q

# 1. set S counts and feature coverage per block, no metric computed
nice -n 10 .venv/bin/python -m anticipation.screen.run --dry_run --out benchmark/results/cue_screen

# 2. pre-registration (written once, committed before any result)
.venv/bin/python -m anticipation.screen.run --write_prereg

# 3. a phase (refuses to start unless set S still matches the pre-registered list hashes)
nice -n 10 .venv/bin/python -m anticipation.screen.run --phase A --out benchmark/results/cue_screen
nice -n 10 .venv/bin/python -m anticipation.screen.run --blocks pose_interact,motion_nocam   # subset
#   optional: --m6 (VA30 event metrics, slow), --m2 (person-count strata; needs pose_interact), --n_boot 1000

# 4. Holm adjustment across the phase and the decision classes
.venv/bin/python -m anticipation.screen.run --decide A --out benchmark/results/cue_screen
```

Outputs under `benchmark/results/cue_screen/` (plan 6.9):

| File | Content |
|---|---|
| `<block>/metrics.json` | every statistic (value, 95 % CI, one-sided bootstrap p), per-fold probe info (chosen C, validation AUROC), P0 table, coverage, config incl. pre-registration sha256 |
| `<block>/scores.parquet` | out-of-fold scores of the primary probe: `video_id, t_sec, score, fold, split, seed` plus one column per control model (`split` is `test` or `val`; the model of `fold` produced the row) |
| `<block>/va30_scores.parquet`, `<block>/va30/` | with `--m6`: the VA30 table (violent and clean videos only) and the evaluator output |
| `decisions_phase_<A|B>.json`, `results_phase_<A|B>.md` | gates, class and tags per block after the Holm adjustment |
| `dry_run_coverage.json` | tick counts and feature coverage per block on set S |

## 2. Set S (`sets.py`, plan 6.3)

* **Fold freeze.** `ScreenSet` refuses fold 1 everywhere: `test_ids(1)`, `ticks(v)`, `FeatureSource.load/has/scene` and
  `assert_no_frozen(score_table)` raise `FrozenFoldError`. **Decision:** fold-1 videos are also removed from every
  training and validation set of the screening folds (the plan only says no result may read fold 1; this is the
  stricter reading, so the final fused model can be trained on folds 0, 2, 3, 4 and tested once on fold 1).
* **Positives.** Eligible30 events of the trusted-onset videos in the test sets of folds 0, 2, 3, 4. Ticks run from 1 to
  `floor(last eligible onset + 10)` (masked ticks follow `Benchmark.labels`; soft masks add to them).
* **Clean negatives.** Every clean negative of the screening folds, ticks as capped by `extract_manifest_v1.csv`
  (`Benchmark.load(horizons=...)`).
* **H1.** `extract_manifest_v2.csv` videos (role `other_anomaly`) in the screening folds. Ticks 1 to
  `min(first anomaly, 300)`; `tte = first anomaly - t` with `first_anomaly_sec` from `videos.csv`. H1p = the UCF videos of
  the classes stealing, shoplifting, burglary, vandalism, arson. H1 videos are never training data and never enter
  the VA30 table.
* **Subsets.** S_first, S_real (not `xd_movie`), S_long60 (first segment, eligible60), S_ramp (first segment,
  `max_lead >= 90`), S_offubi (official or UBI onsets), S_audio (first segment, ucf or xd_youtube, real audio; the
  diegetic filter of plan 6.1 step 4 applies once `benchmark/audits/` has it; until then
  `diegetic_filter_applied: false` is written to `metrics.json`).
* **Soft masks.** If `benchmark/audits/soft_masks_v1.json` (`{video_id: [[start, end, reason], ...]}`) exists, those
  ticks are masked (never positive or negative) and `soft_mask_info` (applied flag, path, sha256) is recorded in the
  pre-registration and in every `metrics.json`. Absent file: not applied, recorded as such.

## 3. Features and probes (`features.py`, `probes.py`, plan 6.4)

Inputs are cached 1 Hz streams (`io.load_features`) joined on `t_sec`; a block's ticks stop at the shortest stream.
Block registry: `features.BLOCKS` (Phase A and B), `PHASE_C` (VLM arms; not run here), `EXTERNAL_REFERENCES`
(`aggression_level`, `b_teacher`: scored through the same metrics from a score table, not trained).

| Probe | Implementation |
|---|---|
| P0 | per column time-matched band AUROC on the pooled screening ticks; direction pre-declared (default higher = riskier, `Block.lower_is_risk` lists the exceptions); wide blocks skipped |
| P1 | per column causal mean, max and fraction above the clean-training 95th percentile over 10, 60, 180 s; windows cut at the clip start and after the latest shot cut (`motion:cut_1s`), divided by the ticks they contain (empty window = 0) |
| P1s | P1 plus least-squares slopes over 15, 30, 60 s |
| P1b | P1 on the column minus a causal early baseline (running mean over the first 20 s) |
| PLS | blocks wider than 256 columns: weighted PLS1 (NIPALS) to 32 components on the band target with time-match weights, fitted on the training folds. **Decision:** applied per tick before the windows (so the 95th percentiles and windows are on 32 columns, not 1152 or 2048). Full-dimension L2 (the plan's sensitivity check) is not implemented |
| Model | L2 logistic regression, C in {1e-3, 1e-2, 1e-1, 1, 10} (warm-started path), chosen by the time-matched AUROC on the fold's validation slice (ties: smaller C; no usable validation rows: C = 0.1 and `val_fallback`). Features are standardised on the training rows and clipped to +-10 |
| Target | band: first-segment pre-violence ticks with tte in (30, 120] vs clean ticks with >= 60 s left. Weights are `time_match_weights` (source kind x `TIME_MATCH_EDGES`), normalised so the negative mass equals the positive mass. **Decision:** rows are capped at 25,000 positives and 50,000 negatives per fold (uniform subsample, weights kept), 60,000 validation negatives |
| P1-near, P1-later | same machinery, targets tte in (0, 10] and later-segment lead-ins (tte in (30, 120], after earlier violence) |
| P2 | stub: the GRU is not implemented (only for blocks that pass G1) |

Each fold's model scores its test videos and its validation slice (all ticks), seed 0 (the logistic regression is
deterministic; subsampling uses `seed + 1000 * fold`).

### Controls

| Control | Where | Expected |
|---|---|---|
| `time_source` | source kind x time-bin positive rate fitted on the training folds (`b_time_source_fit` style) | M1 = 0.50 by construction |
| N | logistic regression on source x time-bin indicators (`TIME_EDGES` bins) + `siglip2_scene0p` embedding reduced to 32 PCA dimensions fitted on the training videos; a missing embedding gives zeros and a missing flag (recorded under `nuisance.<fold>.scene_missing`) | the scene and collection prior |
| N + block | N features concatenated with the P1 features, same target and grid | gain over N is M4n |
| `shuffled` | the P1 model applied to the features of a random clean test video of the same source kind (same ticks; short donors repeat their last tick) | 0.50 |
| `b_detect` | P1 features, target in-violence vs clean ticks (violence ticks exist only up to the last onset + 10 s), applied to pre-onset ticks | diagnostic |
| first-frame broadcast | the P1 score of tick 2 broadcast over the whole video, on S_ramp | diagnostic, never gates |

## 4. Metrics (`metrics.py`, plan 6.5)

Bootstrap: 1,000 resamples (default) of `split_group` clusters (`Pool.boot_weights`); every difference (M4n, M8 gain)
reuses the same resample matrix, so it is paired. Reported CI = 2.5th and 97.5th percentiles; `p_le_null` is the
one-sided bootstrap p-value `(1 + #{resample <= null}) / (B + 1)`. AUROCs are `PairAUC` with time-match weights; the
weights depend on labels only and are held fixed across resamples.

| ID | Key in `metrics.json` | Notes |
|---|---|---|
| M1 | `m1`, `m1_30_60` | band (30, 120] and (30, 60]; also for `p1s`, `p1b` and each control (`m1_<model>`) |
| M1-long | `m1_long` | tte (60, 120] on S_long60 videos |
| M1-sparse | `m1_sparse_pauc`, `events.tail` | McClish standardised pAUC at clean FPR <= 1 % (time-matched negatives); tail-hit difference: share of events with >= 3 ticks in [t0-120 (or clear start), t0-30] above the 99.5th percentile of the fold model on clean validation ticks, minus the same share over clean windows at the same absolute times in clean videos of the same source kind |
| M1-near | `m1_near` | P1-near scores, tte (0, 10] |
| M1-later | `m1_later` | P1-later scores, later-segment ticks |
| M2 | `m2` | with `--m2`: person-count stratum (0 / 1 / 2 / 3-5 / 6+ from `pose_interact:n_persons_last`) added to the matching; `n_persons_ge2` restriction |
| M3p | `m3p`, `m3p_ratio`, `m1_ucf`, `m3_per_class`, `m3_all_h1`, `h1_vs_clean` | UCF band ticks vs H1p ticks with time to anomaly in (30, 120], matched on time since start; ratio = (M3p - 0.5) / (M1_UCF - 0.5), undefined where the denominator is below 0.01 |
| M4n | `m4n` | `m1_n_plus_p1 - m1_n`, paired |
| M4b | `m4b_first_frame` | S_ramp |
| M5 | `m5_b0_10 ... m5_b60_120`, `m5_ramp` | bucket AUROCs; paired within-video mean(tte 0-30) minus mean(tte 60-120) on S_ramp videos, share rising, Wilcoxon p |
| M6 | `<block>/va30/metrics.json` | `--m6`: score table with `--restrict_to_scored --horizons benchmark/extract_manifest_v1.csv`, threshold-aware CI N = 200; H1 videos are not in the table; violent videos are scored only up to the last onset + 10 s, so their later hours are not scored |
| M7 | `breakdowns` | S_real, S_offubi, S_audio, per source kind, class groups (dyadic = fighting/abuse/assault, instrumental = robbery/shooting/arrest, riot). Not implemented: AUROC matched within the same film, news and compilation exclusion, visible-conflict vs sudden (needs annotation) |
| M8 | `events.m8`, `events_first.m8`, `m8_gain_over_n` | max score in the acceptance window `[max(t0-120, clear_start), t0-30]` against the max in same-time windows of clean videos of the same source kind (clean windows containing masked ticks are skipped). The partial AMOC area is not implemented (it is in the VA30 report with `--m6`) |

## 5. Decision rule (`decision.py`, plan 6.6)

Gates return PASS, FAIL or INCONCLUSIVE (plus NA / EXEMPT). Thresholds are constants at the top of `decision.py` and are
copied into the pre-registration.

* **G1** main route: PASS when the M1 point is >= 0.58 and the Holm-adjusted one-sided p < 0.05 across the blocks of the
  phase (`decide_phase`); FAIL when the CI upper bound is < 0.58. Sparse route: PASS when the standardised pAUC is >= 0.55
  with CI lower > 0.50 or the tail-hit difference has CI lower > 0; FAIL when pAUC CI upper < 0.55 and tail-hit
  difference CI upper < 0.05 (**Decision:** 0.05 is the "useful effect" of the sparse route). G1 passes if either route
  passes, fails if the main route fails and the sparse route fails or is unavailable.
* **G2** (M3p): PASS CI lower > 0.50 and ratio >= 0.5; FAIL CI upper < 0.55 or ratio < 0.25 (point estimates).
* **G3** (M4n): PASS gain >= +0.02 with CI lower > 0; FAIL CI upper < +0.02.
* **Classes:** ANTICIPATION, PROVISIONAL, CONTEXT/SHORTCUT, SHORT-HORIZON, INCONCLUSIVE, DROP exactly as in the plan;
  `short_horizon` is also reported as a flag next to the class. **Tags:** LONG-LEAD, RAMP.
* **Audio blocks:** the primary M1 uses the S_audio positives; `soundtrack_flag` is true when `xd_movie` M1 exceeds the
  S_audio M1 by more than 0.08. `Block.g2_exempt` marks context-cue blocks exempt from G2 (none in Phase A or B).
* **Not implemented:** forward selection (needs the fusion trainer), the confirmation run on fold 1 (by design
  unreachable here), P2, Phase C.

## 6. Tests (`tests/test_screen.py`)

Synthetic benchmark: 5 folds, 8 violent, 20 clean and 6 H1p-like videos per fold, a 4-column stream. Checks: a planted
band signal gives G1 PASS and G3 PASS with controls at 0.50; a scene-prior signal (video-level, present in the static
scene embedding) passes G1 but fails G3 (CONTEXT/SHORTCUT); a pure time-since-start signal is not a signal; the same
signal planted before the H1p anomalies fails G2; a cue in the last 10 s only is SHORT-HORIZON; every access path to
fold 1 raises; window features match a brute-force loop with shot cuts; slopes match `polyfit`; the early baseline
is causal; PLS recovers a planted direction; Holm and the class table; the sparse pAUC.
