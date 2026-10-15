# SCREEN: the frozen modality screen as implemented

## Status (2026-10-15)

No metric has been computed on a real feature block; the screen has not been run. Everything below was checked on
synthetic data and on real-data dry runs (tick counts, coverage, set membership) only.

* **Implemented:** the frozen cohorts and folds (fold 1 unreachable), probes P0, P1, P1s, P1b, P1-near, P1-later with
  the PLS32 reduction for wide blocks, controls (time/source fit, nuisance N with scene0p, shuffled clean video,
  B_detect, first-frame broadcast), metrics M1 to M5, M7 (partial), M8, the sparse route (standardised pAUC and tail
  hits), the Holm-adjusted gates G1 to G3, classes and tags, `run_report.json`.
* **AVAILABLE blocks (Phase A, 9):** `pose_interact`, `pose_state_identity_free`, `pose_state_pair_identity_ucf_ubi`,
  `siglip2_prompts_v1`, `siglip2_prompts_v2`, `siglip2`, `vjepa2_l`, `motion_no_camera`, `motion_camera_shortcut`.
  Real-data dry run: every Phase A block has 100 % feature and tick coverage on the positive, clean, H1, H1p and
  clean-VLM cohorts, and no video lacks the motion stream used for shot-cut resets.
* **NOT_AVAILABLE (no extractor):** `camera_wearer_diagnostic`, `overlap_and_duty_cycles`, `dialogue_acts`, the five
  Phase C VLM arms; controls `aggression_level_no_context`, `b_teacher`.
* **BLOCKED_AUDIO_GATE:** the six Phase B audio blocks until a diegetic audit is frozen.
* **NOT_IMPLEMENTED:** P2 GRU, full-dimensional L2 sensitivity, forward selection, the confirmation rule, M6 sustained
  leads 90/120 s and clean-alarm-time operating points, M7 same-film/news/visible-conflict parts, M8 partial AMOC
  area, the context-cue G2 exemption (full list with reasons in `run_report.json`).
* **Open before the first block is read:** the tail-hit threshold is taken on clean validation scores while the frozen
  JSON says `tail_train_quantile` (section 4). The frozen file cannot be edited from here, so a dated hashed amendment
  that fixes the meaning (or a change to out-of-fold training scores) is still owed; it is recorded under
  `interpretations` in `run_report.json` and every `metrics.json`.
* **Command to run Phase A** (after the dry run and the tests; CPU only, 4 threads, check `tools/server_load.py` first):

```bash
.venv/bin/python -m pytest tests/test_screen.py -q
nice -n 10 .venv/bin/python -m anticipation.screen.run --dry_run --phase A --out benchmark/results/cue_screen
nice -n 10 .venv/bin/python -m anticipation.screen.run --phase A --out benchmark/results/cue_screen
.venv/bin/python -m anticipation.screen.run --decide A --out benchmark/results/cue_screen
```

Code: `anticipation/screen/`; tests: `tests/test_screen.py`; pre-registration: `benchmark/screens/screen_set_v1.json`
(+ `screen_set_v1.sha256`, `summary.md`; sha256 `8831e16e...`, committed in 73c2d3f). This file maps the frozen JSON
(and [MODALITY_PLAN.md](MODALITY_PLAN.md) section 6) to the code. **The frozen file wins over anything in the code or
in the plan**; the harness reads cohorts, scored intervals, fold memberships, block names, probe grids, controls,
metric definitions, thresholds and gates from it and never writes it. Evaluator machinery (`PairAUC`,
`time_match_weights`, `TIME_MATCH_EDGES`, horizons) comes from `anticipation/eval/va30.py` and is not modified.

## 1. Run

All commands run from the repo root with `.venv/bin/python`, CPU only (threads capped at 4; start under `nice -n 10`).
Every command first calls `prereg.load`, which refuses to continue (`PreregError`) unless the sha256 of the JSON equals
the one in the sidecar `.sha256` file and the value pinned in `prereg.PINNED_SHA256`, then checks that the numeric
constants of the code (gate thresholds, C grid, windows, PLS width, tail quantile, bootstrap seed and so on) equal the
numbers in the JSON, that `folds_v2.json`, `videos.csv` and `events.json` hash to the registered values, and that no
fold-1 video is in any membership list. It then builds the screening set from the frozen lists, recomputes the sets
with the previous in-code rules and writes every difference to `run_report.json` (section 2).

```bash
# 0. tests (about 30 s)
.venv/bin/python -m pytest tests/test_screen.py -q

# 1. coverage per block on the frozen cohorts, no metric computed (also writes run_report.json)
nice -n 10 .venv/bin/python -m anticipation.screen.run --dry_run --phase A --out benchmark/results/cue_screen

# 2. a phase; blocks that cannot run are listed as NOT_AVAILABLE or BLOCKED_AUDIO_GATE and skipped
nice -n 10 .venv/bin/python -m anticipation.screen.run --phase A --out benchmark/results/cue_screen
nice -n 10 .venv/bin/python -m anticipation.screen.run --blocks pose_interact,motion_no_camera   # subset (refuses unavailable ones)
#   optional: --m6 (VA30 event metrics, slow), --m2 (person-count strata; needs pose_interact)

# 3. Holm adjustment across the phase and the decision classes (refuses unless every AVAILABLE block of the phase has a
#    metrics.json from this pre-registration; --partial_family writes a PARTIAL_FAMILY-labelled result instead)
.venv/bin/python -m anticipation.screen.run --decide A --out benchmark/results/cue_screen
```

There is no `--write_prereg`, `--n_boot`, `--folds` or `--allow_unregistered`: the bootstrap size (1000), seed (0),
threshold bootstrap (200) and FA/h targets (1, 2, 5) are the frozen values. A change needs a dated hashed amendment.

Outputs under `benchmark/results/cue_screen/` (plan 6.9):

| File | Content |
|---|---|
| `<block>/metrics.json` | every statistic (value, 95 % CI, one-sided bootstrap p), per-fold probe info (chosen C, validation AUROC), P0 table, coverage, config incl. pre-registration sha256 |
| `<block>/scores.parquet` | out-of-fold scores of the primary probe: `video_id, t_sec, score, fold, split, seed` plus one column per control model (`split` is `test` or `val`; the model of `fold` produced the row) |
| `<block>/va30_scores.parquet`, `<block>/va30/` | with `--m6`: the VA30 table (violent and clean videos only) and the evaluator output |
| `decisions_phase_<A|B>.json`, `results_phase_<A|B>.md` | gates, class and tags per block after the Holm adjustment |
| `dry_run_coverage.json` | tick counts and feature coverage per block on the frozen cohorts (positive, clean, h1, h1p, clean_vlm) |
| `run_report.json` | prereg path and sha256, label policy, status of all 23 frozen blocks, NOT_IMPLEMENTED list, control status, set counts, agreement checks against the previous in-code rules |

## 2. Cohorts, ticks and folds (`prereg.py`, `sets.py`)

* **Frozen lists.** `positive_videos` (379), `eligible30_events` (446), `first_events` (253), `real_first_events` (165),
  `long60_first_events` (100), `ramp90_first_events` (61), `official_ubi_first_events` (74),
  `audio_first_event_candidates` (86, not approved audio cases), `clean_cpu_videos` (3,220), `clean_vlm_videos` (600,
  150 per source, a subset of the CPU clean videos), `h1_videos` (328), `h1p_videos` (102), `teacher_videos` (419).
  `ScreenSet(prereg=...)` takes video kinds (violent, clean, h1) from these lists, events from `eligible30_events` and
  the S_* flags from the subset lists.
* **Scored ticks.** `scored_intervals_sec[video] = [1, end]`; the ticks are 1..floor(end). Labels (tte, masks) still come
  from `Benchmark.labels` (frozen v2 labels).
* **Folds.** `fold_membership[f].{train,val,test}` for f in 0, 2, 3, 4 are used as given (they hold only the frozen
  videos; H1 videos are filtered out of train and val). Fold 1 has no membership entry; `ScreenSet` refuses it
  (`FrozenFoldError`) in `test_ids`, `train_ids`, `val_ids`, `ticks`, `FeatureSource.load/has/scene` and
  `assert_no_frozen`, and `prereg.load` rejects a file that lists it.
* **Label policy.** No proposed masks in v2. A `soft_masks_v1.json` file is ignored (`soft_masks.applied` false with the
  reason is recorded in `run_report.json` and every `metrics.json`).
* **Audio gate.** Phase B is blocked until an independent diegetic-audio audit is frozen. The six audio blocks have
  extractors but resolve to `BLOCKED_AUDIO_GATE`; running them is refused. `s_audio` flags the 86 candidates only.

### Disagreements between the previous in-code rules and the frozen file (real data, 2026-10-15)

`prereg.agreement` recomputes every list with the old rules and reports the differences (`run_report.json`,
`agreement`). Result:

| Item | Frozen | Previous rules | Difference |
|---|---|---|---|
| positive videos, events, first events, S_real, S_long60, S_ramp, S_offubi, S_audio candidates | 379, 446, 253, 165, 100, 61, 74, 86 | same | none. The 373 positive videos that older documents quote came from the plan text, not from the code, which always gave 379 |
| clean CPU videos, H1, H1p | 3,220, 328, 102 | same | none (the old pre-registration text said 3,200 and 320 as plan values) |
| clean VLM videos | 600 | no rule | subset of the CPU clean list: yes |
| scored ticks per video | frozen interval | old rule | 31 H1 videos have one more frozen tick: when the first anomaly falls on an integer second the frozen interval includes the tick at tte 0 (old rule stopped one tick earlier). No violent or clean video differs |
| fold test / train / val membership | frozen lists only | all violent and clean videos of the folds | the old rule also counted the violent videos without an eligible30 event (about 320 per fold in test, about 860 in train, about 130 in val) and clean videos outside the frozen list. The frozen membership is smaller; frozen is used |

## 3. Blocks, features and probes (`features.py`, `probes.py`)

Inputs are cached 1 Hz streams (`io.load_features`) joined on `t_sec`; a block's ticks stop at the shortest stream.
`features.resolve_blocks(prereg)` returns the status of all 23 frozen block names; a name without an extractor is
reported, never dropped, and a registered block that the JSON does not name is an error.

| Frozen block | Streams and columns | Status |
|---|---|---|
| `pose_interact` | `pose_interact` (65) | AVAILABLE |
| `pose_state_identity_free` | `pose_state` (88) | AVAILABLE |
| `pose_state_pair_identity_ucf_ubi` | `pose_state` + `pose_state_pairs` (9); UCF and UBI videos only | AVAILABLE (assumption: the pair stream is added to the identity-free columns) |
| `siglip2_prompts_v1` | `siglip2_prompts` (58) | AVAILABLE |
| `siglip2_prompts_v2` | `siglip2_prompts_v2` (22) | AVAILABLE |
| `siglip2` | `siglip2` (1152, PLS32) | AVAILABLE |
| `vjepa2_l` | `vjepa2_l` (2048, PLS32) | AVAILABLE |
| `motion_no_camera` | `motion` without `cam_flow_mag, cam_shake, cam_flow_mag_5s, hist_dist_max, cut_1s, cuts_5s, time_since_cut` | AVAILABLE |
| `motion_camera_shortcut` | `motion`, only those 7 columns | AVAILABLE (the camera/cut column set is our choice; the JSON gives only the names) |
| `camera_wearer_diagnostic` | none | NOT_AVAILABLE |
| `audio_dsp`, `ast_events`, `clap_prompts`, `speech_tox`, `voice_affect`, `e2v_emb` | the same-named streams | BLOCKED_AUDIO_GATE |
| `overlap_and_duty_cycles`, `dialogue_acts` | none | NOT_AVAILABLE |
| `vlm_single_frame_A`, `vlm_short_clip_B`, `vlm_long_window_C`, `vlm_forecast`, `vlm_hidden_state_probe` | none | NOT_AVAILABLE |

| Probe | Implementation |
|---|---|
| P0 | per column time-matched band AUROC on the pooled screening ticks, no fitting; prompt blocks only: higher = risk except concepts containing calm, intervention, separat(e/ion), walk_away (the "retreat" cue), apolog(y/ize), deescalat(e/ion), which are lower = risk (the JSON lists calm, intervention, separation, retreat, apology, deescalation; mapping retreat to `walk_away` is ours). Generic embedding blocks have no P0 arm |
| P1 | per column causal mean, max and fraction above the clean-training 95th percentile over 10, 60, 180 s; windows cut at the clip start and after the latest shot cut (`motion:cut_1s`), divided by the ticks they contain; the cut tick itself keeps its own value (a window is never empty, so no fabricated zero). A missing or short motion stream fails the block (`CutStreamError`; `cut_stream_incomplete` is recorded in the config) |
| P1s | P1 plus least-squares slopes over 15, 30, 60 s |
| P1b | P1 on the column minus the mean of ticks 1..10, applied only after tick 10; training rows, scores and the M1 pairs of ticks <= 10 are masked (target `band_b`); no future fill. Its windows start at tick 11 (cut index 9), so they are divided by post-baseline history only. `m1_p1b` lives on the t > 10 set; `m1_p1_late` is P1 on the same set and `p1b_gain_over_p1` the paired difference |
| PLS | blocks wider than 256 columns: train-only weighted PLS1 to 32 components on the band target with time-match weights, applied per tick before the windows. The full-dimensional L2 sensitivity is NOT_IMPLEMENTED |
| Model | L2 logistic regression, C in {1e-3, 1e-2, 1e-1, 1, 10}, chosen by the time-matched AUROC on the fold's validation slice (ties: smaller C). Features standardised on the training rows and clipped to +-10 |
| Target | band: first-segment pre-violence ticks with tte in (30, 120] vs clean ticks with >= 60 s left, `time_match_weights` over source kind x `TIME_MATCH_EDGES`. Rows are capped at 25,000 positives and 50,000 negatives per fold (uniform subsample), 60,000 validation negatives (our decision; not in the JSON) |
| P1-near, P1-later | tte in (0, 10] and later-segment lead-ins |
| P2 | NOT_IMPLEMENTED |

Seed 0 only (the logistic regression is deterministic; subsampling uses `seed + 1000 * fold`).

### Controls

| Frozen control | Status | Where |
|---|---|---|
| `b_time_source_fit` | implemented | source kind x `TIME_MATCH_EDGES` bin positive rate from training folds; M1 = 0.50 by construction |
| `nuisance_N_source_time_Bscene0p` | implemented | logistic regression on source x `TIME_MATCH_EDGES` indicators + `siglip2_scene0p` (tick 2, people greyed out) reduced to training-only PCA32; N + block gives M4n. The scene frame is at 2 s, so the scene components of tick 1 are set to 0 (no information from the future) |
| `shuffled_same_source_clean_video` | implemented | the P1 model applied to the features of a random clean test video of the same source kind |
| `B_detect` | implemented | in-violence vs clean ticks, applied to pre-onset ticks |
| `first_frame_broadcast_on_ramp90` | implemented | tick-2 score broadcast over the video on S_ramp; diagnostic |
| `aggression_level_no_context`, `b_teacher` | NOT_AVAILABLE | no score tables for the screening cohort |

## 4. Metrics (`metrics.py`)

Bootstrap: 1,000 resamples (frozen) of `split_group` clusters (`Pool.boot_weights`); every difference (M4n, M8 gain)
reuses the same resample matrix, so it is paired. Reported CI = 2.5th and 97.5th percentiles; `p_le_null` is the
one-sided bootstrap p-value `(1 + #{resample <= null}) / (B + 1)`. AUROCs are `PairAUC` with time-match weights; the
weights depend on labels only and are held fixed across resamples.

| ID | Key in `metrics.json` | Notes |
|---|---|---|
| M1 | `m1`, `m1_30_60`, `m1_ramp` | band (30, 120] and (30, 60]; also for `p1s`, `p1b` (ticks > 10 only) and each control (`m1_<model>`) |
| M1-long | `m1_long` | tte (60, 120] on S_long60 videos |
| M1-sparse | `m1_sparse_pauc`, `events.tail` | McClish standardised pAUC at clean FPR <= 1 % (time-matched negatives; tied scores form one diagonal ROC step, so a constant or mostly-zero cue gives 0.5; matches sklearn `max_fpr` on continuous scores); tail-hit difference: share of events with >= 3 ticks in [t0-120 (or clear start), t0-30] above the 99.5th percentile of the fold model on clean validation ticks (the JSON says "tail_train_quantile"; the fold's clean validation slice is our reading, since training scores are overfit; a validation threshold uses no test data, so this is a protocol deviation, not a leak; an amendment is owed), minus the same share over clean windows at the same absolute times in clean videos of the same source kind |
| M1-near | `m1_near` | P1-near scores, tte (0, 10] |
| M1-later | `m1_later` | P1-later scores, later-segment ticks |
| M2 | `m2` | with `--m2`: person-count stratum (frozen bins 0 / 1 / 2 / 3-5 / 6+ from `pose_interact:n_persons_last`) added to the matching; `n_persons_ge2` restriction |
| M3p | `m3p`, `m3p_ratio`, `m1_ucf`, `m3_per_class`, `m3_all_h1`, `h1_vs_clean` | UCF band ticks vs H1p ticks with time to anomaly in (30, 120], matched on time since start; ratio = (M3p - 0.5) / (M1_UCF - 0.5), undefined where the denominator is not above 0, as frozen (G2 then rests on the M3p upper bound alone) |
| M4n | `m4n` | `m1_n_plus_p1 - m1_n`, paired |
| M4b | `m4b_first_frame`, `m1_ramp_gain_over_m4b` | S_ramp; M1 minus the broadcast on the same ticks (paired). Not gating; `decide` copies both into the decision row and sets `first_frame_flag` when the difference CI includes 0 or the broadcast CI lower exceeds 0.55 |
| M5 | `m5_b0_10 ... m5_b60_120`, `m5_ramp` | bucket AUROCs; paired within-video mean(tte 0-30) minus mean(tte 60-120) on S_ramp videos, share rising, Wilcoxon p |
| M6 | `<block>/va30/metrics.json` | `--m6`: score table with `--restrict_to_scored --horizons benchmark/extract_manifest_v1.csv`, threshold CI with 200 resamples, FA/h targets 1, 2, 5 (both read from the JSON); H1 videos are not in the table; violent videos are scored only up to the last onset + 10 s, so their later hours are not scored |
| M7 | `breakdowns` | S_real, S_offubi, S_audio, per source kind, class groups (dyadic = fighting/abuse/assault, instrumental = robbery/shooting/arrest, riot). NOT_IMPLEMENTED parts: AUROC matched within the same film, news and compilation exclusion, visible-conflict vs sudden |
| M8 | `events.m8`, `events_first.m8`, `m8_gain_over_n` | max score in the acceptance window `[max(t0-120, clear_start), t0-30]` against the max in same-time windows of clean videos of the same source kind (clean windows containing masked ticks are skipped). Windows are tick-exact to the M1 band: the last tick has tte strictly > 30 (an integer onset no longer adds the tte = 30 tick) and tte = 120 is included. An event window drops its masked ticks while a clean window containing any masked tick is dropped whole and ignores `remaining >= 60`, so "same observed length" is approximate; this asymmetry is accepted. The partial AMOC area is not implemented (it is in the VA30 report with `--m6`) |

## 5. Decision rule (`decision.py`)

Gates return PASS, FAIL or INCONCLUSIVE (plus NA / EXEMPT). The threshold constants at the top of `decision.py` are
compared with the numbers in the JSON at every start (`decision.check_constants`, `run.check_frozen_constants`); a
mismatch aborts the run.

* **G1** main route: PASS when the M1 point is >= 0.58, the CI exists and the Holm-adjusted one-sided p < 0.05.
  Sparse route: PASS when the standardised pAUC is >= 0.55 (one-sided p against 0.5) or the first-segment tail-hit
  difference is positive (one-sided p against 0), each with its own adjusted p < 0.05. **Multiplicity (our choice,
  stricter than the frozen "Holm across blocks"):** one Holm family over the three G1 tests (M1, pAUC, first-segment
  tail hits) of every AVAILABLE block of the phase, so adding the sparse route cannot inflate the family-wise false-pass
  rate. NOT_AVAILABLE and gated blocks cannot be tested and do not count toward the family size. G1 is PASS if either
  route passes, FAIL if the main-route CI upper is < 0.58 and the sparse route did not pass, else INCONCLUSIVE. The
  tail-hit route uses `events_first` only; the all-segment tail is stored as `tail_all_segments` and does not gate.
  `decide` refuses an incomplete family or a `metrics.json` whose `prereg_sha256` is not the registered one, and writes
  the family (`holm_family_blocks`, `family_status`) into `decisions_phase_<X>.json`.
* **G2** (M3p): PASS CI lower > 0.50 and a defined ratio >= 0.5; FAIL CI upper < 0.55 (also with an undefined ratio,
  our reading of the frozen sentence: only the ratio clauses are undecidable) or ratio < 0.25; otherwise INCONCLUSIVE.
* **G3** (M4n): PASS gain >= +0.02 with CI lower > 0; FAIL CI upper < +0.02.
* **Classes** (frozen spelling): ANTICIPATION, PROVISIONAL, CONTEXT_SHORTCUT, SHORT_HORIZON, DROP, INCONCLUSIVE.
  **Tags:** LONG_LEAD, RAMP. `short_horizon` is also reported as a flag.
* **Audio blocks:** blocked by the audio gate. When the audit exists, the primary M1 uses the S_audio positives and
  `soundtrack_flag` is true when `xd_movie` M1 exceeds the S_audio M1 by more than 0.08.
* **Context-cue exemption** from G2 (weapon, police, counter, arrivals heads only): NOT_IMPLEMENTED (listed in
  `run_report.json`). `Block.g2_exempt` is unused; if set it caps the class at PROVISIONAL.
* **Edge cases:** a PASS or FAIL needs a non-null CI; thresholds compare with a 1e-9 tolerance (`soundtrack_flag`
  needs a difference strictly above 0.08); NA gates follow the frozen class table (not FAIL) but are listed in
  `unevaluated_gates` and G3 NA can never give ANTICIPATION; `soundtrack_flag` is copied into the decision rows.

## 6. NOT_IMPLEMENTED and NOT_AVAILABLE (listed in every run, in `run_report.json`)

| Item | Reason |
|---|---|
| P2 GRU | the event-window MIL loss is underspecified; only blocks with G1 PASS qualify |
| P1 full-dimensional L2 sensitivity | not implemented; only PLS32 runs |
| forward selection, confirmation rule | need the fused-model trainer; fold 1 is unreachable here by design |
| M6 sustained leads 90 and 120 s, clean alarm-time operating points 1/2/5 % | the VA30 evaluator offers 60 s and FA/h targets only |
| M7 same-film AUROC, news/compilation exclusion, visible-conflict vs sudden | need annotations |
| M8 partial AMOC area | only in the VA30 report with `--m6` |
| controls `aggression_level_no_context`, `b_teacher` | no score tables for the cohort |
| blocks `camera_wearer_diagnostic`, `overlap_and_duty_cycles`, `dialogue_acts`, five VLM arms | NOT_AVAILABLE (no extractor) |
| Phase B audio blocks | BLOCKED_AUDIO_GATE until the diegetic audit is frozen |

Implemented by delegation to `va30.main` under `--m6`: the event-threshold CI with 200 resamples and FA/h targets 1, 2, 5.

## 7. Tests (`tests/test_screen.py`)

Synthetic benchmark: 5 folds, 8 violent, 20 clean and 6 H1p-like videos per fold, a 4-column stream. Checks: a planted
band signal gives G1 PASS and G3 PASS with controls at 0.50; a scene-prior signal passes G1 but fails G3
(CONTEXT_SHORTCUT); a pure time-since-start signal is not a signal; the same signal planted before the H1p anomalies
fails G2; a cue in the last 10 s only is SHORT_HORIZON; window features, slopes, the causal early baseline (masked
first ticks), PLS, Holm, the class table and the frozen G1/G2 variants; P0 directions and the P1b target.

Review fixes (each has a test): tied-score pAUC and the sklearn match, pooled Holm for the sparse route, G2 FAIL with an undefined ratio, the incomplete or stale Holm family, first-segment tail hits, M8 window edges, P1b pairing and windows, cut-stream failure, cut-tick windows, tick-1 scene masking, edge cases, M4b reporting.

Frozen-registration tests: a tampered JSON, tampered sidecar, missing sidecar, forged JSON plus sidecar (pinned hash)
and a tampered copy passed to `run.main` are all refused before any output is written; the committed file loads, its
numbers equal the code constants (a changed number is detected) and its 23 block names resolve to statuses; a synthetic
frozen file drives the cohorts (its lists replace the computed ones, a dropped clean or H1 video and a changed scored
interval are used and reported by `agreement`); a full frozen run reads no feature, label or score row of fold 1 and
only videos listed in the file; a fold-1 video in a membership list is fatal.
