# VA30 report: phase_c_aggression_imminent

Fold file `phase_c_pilot_without_original_fold1` (content sha256 `8ebdf6ec2e54`), seeds [0], headline seed 0. Test universe: 112 videos (44 violent, 68 clean negatives = 2.79 h), 44 eligible30 events (44 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 112 of 112 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 0 catalog clean negatives (0 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 2.79 of 2.79 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.726 [0.634, 0.817] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.691 [0.607, 0.774] |
| Far-window AUROC (clean, source x time matched negatives) | 0.834 [0.622, 0.927] |
| Far-window AUROC (clean, uncensored negatives) | 0.715 [0.623, 0.805] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.726 [0.634, 0.817] |
| AUROC / AP, H = 30 s (base rate 0.120) | 0.749 [0.668, 0.832] / 0.346 [0.217, 0.533] |
| Detection AUROC (in-violence vs clean negatives) | 0.802 [0.701, 0.888] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 1.09 [0.00, 3.09] | 1.55 [0.00, 3.90] | 6.8 % [0.0 %, 15.0 %] | [0.0, 26.1] % | 2.7 % [0.0 %, 7.4 %] | 30.0 % | 0.0 % [0.0 %, 0.0 %] (14) | 11.4 % [2.4 %, 21.7 %] | 13.6 % [4.3 %, 24.5 %] | 26.1 [12.1, 70.4] | 12.8 % [4.2 %, 22.7 %] |
| 2.0 | 1.09 [0.00, 3.09] | 1.55 [0.00, 3.90] | 6.8 % [0.0 %, 15.0 %] | [0.0, 38.7] % | 2.7 % [0.0 %, 7.4 %] | 30.0 % | 0.0 % [0.0 %, 0.0 %] (14) | 11.4 % [2.4 %, 21.7 %] | 13.6 % [4.3 %, 24.5 %] | 26.1 [12.1, 70.4] | 12.8 % [4.2 %, 22.7 %] |
| 5.0 | 2.18 [0.39, 4.27] | 3.72 [1.48, 6.48] | 18.2 % [7.5 %, 30.0 %] | [4.4, 50.1] % | 5.3 % [1.0 %, 10.1 %] | 50.0 % | 14.3 % [0.0 %, 35.3 %] (14) | 25.0 % [12.8 %, 38.6 %] | 25.0 % [12.8 %, 38.6 %] | 70.3 [26.5, 96.2] | 23.4 % [11.9 %, 35.9 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.757 [0.671, 0.841] | 0.675 [0.533, 0.801] | 0.757 [0.671, 0.841] | 0.121 [0.041, 0.284] | 0.0192 |
| 10 | 0.762 [0.677, 0.844] | 0.675 [0.534, 0.802] | 0.762 [0.677, 0.844] | 0.175 [0.079, 0.363] | 0.0387 |
| 20 | 0.759 [0.679, 0.839] | 0.666 [0.530, 0.796] | 0.759 [0.679, 0.839] | 0.271 [0.153, 0.472] | 0.0786 |
| 30 | 0.749 [0.668, 0.832] | 0.656 [0.524, 0.783] | 0.749 [0.668, 0.832] | 0.346 [0.217, 0.533] | 0.1197 |
| 60 | 0.719 [0.633, 0.800] | 0.636 [0.520, 0.755] | 0.719 [0.633, 0.800] | 0.490 [0.362, 0.637] | 0.2466 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.575 [0.439, 0.753] | 0.810 [0.625, 0.932] | 0.0 % [0.0 %, 0.0 %] | 33.3 % [0.0 %, 100.0 %] | 3.35 [0.00, 8.83] | 6 |
| source:xd_youtube | 0.828 [0.689, 0.993] | 0.857 [0.708, 0.994] | 25.0 % [0.0 %, 100.0 %] | 25.0 % [0.0 %, 100.0 %] | 0.00 [0.00, 0.00] | 4 |
| source:ucf | 0.878 [0.591, 1.000] | 0.663 [0.527, 0.798] | 5.9 % [0.0 %, 15.2 %] | 8.8 % [0.0 %, 19.4 %] | 0.00 [0.00, 0.00] | 34 |
| class:interpersonal | 0.741 [0.649, 0.830] | 0.767 [0.686, 0.850] | 7.5 % [0.0 %, 16.7 %] | 15.0 % [4.7 %, 27.3 %] | 1.09 [0.00, 3.09] | 40 |
| class:other_violent | 0.574 [0.144, 0.889] | 0.569 [0.153, 0.876] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 1.09 [0.00, 3.09] | 4 |
| audio:present | 0.709 [0.586, 0.816] | 0.777 [0.660, 0.882] | 8.7 % [0.0 %, 21.7 %] | 17.4 % [3.4 %, 35.3 %] | 1.16 [0.00, 3.31] | 23 |
| audio:absent | 0.875 [0.528, 1.000] | 0.633 [0.462, 0.822] | 4.8 % [0.0 %, 15.4 %] | 9.5 % [0.0 %, 23.8 %] | 0.00 [0.00, 0.00] | 21 |
| onset:official | 0.737 [0.593, 0.850] | 0.794 [0.651, 0.906] | 6.7 % [0.0 %, 21.4 %] | 20.0 % [0.0 %, 42.9 %] | 1.09 [0.00, 3.09] | 15 |
| onset:hivau | 0.720 [0.601, 0.829] | 0.742 [0.643, 0.844] | 6.9 % [0.0 %, 17.7 %] | 10.3 % [0.0 %, 23.1 %] | 1.09 [0.00, 3.09] | 29 |
| event:first_segment | n/a | n/a | 6.8 % [0.0 %, 15.0 %] | 13.6 % [4.3 %, 24.5 %] | 1.09 [0.00, 3.09] | 44 |
| event:later_segment | n/a | n/a | n/a | n/a | 1.09 [0.00, 3.09] | 0 |

Generated 2026-10-16T15:43:55 by `anticipation/eval/va30.py` (git d25ab37745).
