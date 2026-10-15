# VA30 report: trainer_horizon_replay_20261015

Fold file `folds_v2` (content sha256 `8ebdf6ec2e54`), seeds [0, 1, 2], headline seed 0. Test universe: 3497 videos (277 violent, 3220 clean negatives = 125.86 h), 326 eligible30 events (131 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 3497 of 4916 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 653 catalog clean negatives (527 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 125.86 of 242.3 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.518 [0.476, 0.571] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.519 [0.479, 0.572] |
| Far-window AUROC (clean, source x time matched negatives) | 0.505 [0.470, 0.551] |
| Far-window AUROC (clean, uncensored negatives) | 0.523 [0.482, 0.574] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.477 [0.437, 0.536] |
| AUROC / AP, H = 30 s (base rate 0.017) | 0.515 [0.470, 0.552] / 0.017 [0.013, 0.022] |
| Detection AUROC (in-violence vs clean negatives) | 0.414 [0.358, 0.512] |
| Brier P(onset within 30 s) (climatology 0.0288), ECE | 0.5929 [0.5736, 0.6154], 0.6748 |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 0.45 [0.32, 0.51] | 0.44 [0.31, 0.49] | 0.3 % [0.0 %, 0.9 %] | n/a | 1.1 % [0.8 %, 1.3 %] | 4.7 % | 0.0 % [0.0 %, 0.0 %] (145) | 0.3 % [0.0 %, 0.9 %] | 0.3 % [0.0 %, 0.9 %] | 36.5 [36.5, 36.5] | 0.5 % [0.2 %, 1.1 %] |
| 2.0 | 2.01 [1.73, 2.23] | 1.96 [1.68, 2.19] | 2.1 % [1.2 %, 3.8 %] | n/a | 4.9 % [4.2 %, 5.4 %] | 8.3 % | 0.0 % [0.0 %, 0.0 %] (145) | 2.5 % [1.2 %, 4.1 %] | 2.5 % [1.2 %, 4.1 %] | 41.4 [36.5, 66.8] | 2.0 % [1.4 %, 2.9 %] |
| 5.0 | 4.09 [3.63, 4.42] | 3.99 [3.50, 4.35] | 6.1 % [5.0 %, 9.6 %] | n/a | 9.7 % [8.7 %, 10.5 %] | 13.1 % | 0.0 % [0.0 %, 0.0 %] (145) | 7.7 % [5.9 %, 10.9 %] | 7.7 % [5.9 %, 10.9 %] | 43.3 [37.9, 52.2] | 6.7 % [5.2 %, 8.8 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.508 [0.474, 0.542] | 0.508 [0.483, 0.543] | 0.500 [0.459, 0.549] | 0.003 [0.002, 0.004] | 0.0027 |
| 10 | 0.507 [0.466, 0.543] | 0.508 [0.476, 0.544] | 0.498 [0.454, 0.547] | 0.005 [0.004, 0.007] | 0.0053 |
| 20 | 0.509 [0.465, 0.545] | 0.508 [0.473, 0.544] | 0.495 [0.452, 0.546] | 0.011 [0.008, 0.014] | 0.0108 |
| 30 | 0.515 [0.470, 0.552] | 0.509 [0.475, 0.547] | 0.495 [0.450, 0.545] | 0.017 [0.013, 0.022] | 0.0168 |
| 60 | 0.516 [0.479, 0.557] | 0.511 [0.483, 0.551] | 0.492 [0.450, 0.540] | 0.035 [0.026, 0.045] | 0.0330 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.600 [0.543, 0.703] | 0.556 [0.513, 0.630] | 4.7 % [1.2 %, 8.1 %] | 5.8 % [1.6 %, 9.2 %] | 2.73 [1.98, 3.24] | 86 |
| source:xd_youtube | 0.424 [0.387, 0.502] | 0.448 [0.361, 0.494] | 2.0 % [0.4 %, 5.9 %] | 2.0 % [0.4 %, 5.9 %] | 2.64 [2.07, 2.97] | 100 |
| source:ucf | 0.529 [0.449, 0.598] | 0.476 [0.427, 0.538] | 1.2 % [0.0 %, 3.8 %] | 1.2 % [0.0 %, 3.8 %] | 1.20 [0.88, 1.59] | 80 |
| source:ubi | 0.660 [0.543, 0.726] | 0.738 [0.674, 0.808] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.16 [0.07, 0.27] | 60 |
| class:interpersonal | 0.503 [0.460, 0.556] | 0.506 [0.466, 0.540] | 1.4 % [0.1 %, 2.7 %] | 1.8 % [0.1 %, 3.1 %] | 2.01 [1.73, 2.23] | 279 |
| class:other_violent | 0.602 [0.540, 0.709] | 0.561 [0.494, 0.661] | 6.4 % [1.4 %, 13.4 %] | 6.4 % [1.4 %, 13.4 %] | 2.01 [1.73, 2.23] | 47 |
| audio:present | 0.497 [0.472, 0.571] | 0.490 [0.438, 0.537] | 3.3 % [1.8 %, 5.8 %] | 3.8 % [1.8 %, 6.3 %] | 2.75 [2.31, 3.08] | 212 |
| audio:absent | 0.593 [0.518, 0.650] | 0.596 [0.541, 0.663] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.24 [0.17, 0.41] | 114 |
| onset:official | 0.524 [0.431, 0.593] | 0.497 [0.399, 0.604] | 2.2 % [0.0 %, 6.8 %] | 2.2 % [0.0 %, 6.8 %] | 2.01 [1.73, 2.23] | 45 |
| onset:hivau | 0.514 [0.473, 0.581] | 0.502 [0.454, 0.536] | 2.7 % [1.3 %, 4.6 %] | 3.2 % [1.4 %, 5.1 %] | 2.01 [1.73, 2.23] | 221 |
| onset:ubi | 0.529 [0.410, 0.597] | 0.595 [0.534, 0.683] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 2.01 [1.73, 2.23] | 60 |
| event:first_segment | n/a | n/a | 3.7 % [2.3 %, 6.2 %] | 4.2 % [2.3 %, 6.7 %] | 2.01 [1.73, 2.23] | 189 |
| event:later_segment | n/a | n/a | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 2.01 [1.73, 2.23] | 137 |

## Mean and std over seeds

| Metric | mean | std |
|---|---|---|
| far_auroc_clean | 0.515 | 0.007 |
| far_auroc_secondary | 0.517 | 0.007 |
| far_auroc_clean_time_matched | 0.504 | 0.003 |
| far_auroc_clean_all_segments | 0.476 | 0.006 |
| auroc30 | 0.510 | 0.004 |
| auroc30_time_matched | 0.506 | 0.002 |
| ap30 | 0.017 | 0.000 |
| success30_at_fa2 | 0.028 | 0.005 |
| success10_at_fa2 | 0.034 | 0.008 |
| success0_at_fa2 | 0.034 | 0.008 |
| success30_late_at_fa2 | 0.000 | 0.000 |
| test_fa_per_hour_clean_at_fa2 | 2.103 | 0.082 |
| random_success30_at_fa2 | 0.051 | 0.002 |
| position_null_success30_at_fa2 | 0.085 | 0.002 |
| detection_auroc | 0.406 | 0.006 |

Generated 2026-10-15T14:42:51 by `anticipation/eval/va30.py` (git 5575d031cb).
