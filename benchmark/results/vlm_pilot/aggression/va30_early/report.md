# VA30 report: phase_c_aggression_early

Fold file `phase_c_pilot_without_original_fold1` (content sha256 `8ebdf6ec2e54`), seeds [0], headline seed 0. Test universe: 112 videos (44 violent, 68 clean negatives = 2.79 h), 44 eligible30 events (44 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 112 of 112 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 0 catalog clean negatives (0 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 2.79 of 2.79 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.702 [0.609, 0.792] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.668 [0.579, 0.760] |
| Far-window AUROC (clean, source x time matched negatives) | 0.808 [0.600, 0.908] |
| Far-window AUROC (clean, uncensored negatives) | 0.689 [0.596, 0.776] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.702 [0.609, 0.792] |
| AUROC / AP, H = 30 s (base rate 0.120) | 0.715 [0.635, 0.793] / 0.304 [0.198, 0.477] |
| Detection AUROC (in-violence vs clean negatives) | 0.770 [0.677, 0.855] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 0.73 [0.00, 1.97] | 0.62 [0.00, 1.64] | 9.1 % [2.1 %, 18.6 %] | [2.3, 20.0] % | 1.8 % [0.0 %, 4.8 %] | 22.3 % | 7.1 % [0.0 %, 23.1 %] (14) | 11.4 % [2.5 %, 21.7 %] | 15.9 % [6.2 %, 27.9 %] | 35.5 [5.3, 80.8] | 14.9 % [5.9 %, 26.1 %] |
| 2.0 | 0.73 [0.00, 1.97] | 0.62 [0.00, 1.64] | 9.1 % [2.1 %, 18.6 %] | [2.6, 32.0] % | 1.8 % [0.0 %, 4.8 %] | 22.3 % | 7.1 % [0.0 %, 23.1 %] (14) | 11.4 % [2.5 %, 21.7 %] | 15.9 % [6.2 %, 27.9 %] | 35.5 [5.3, 80.8] | 14.9 % [5.9 %, 26.1 %] |
| 5.0 | 1.45 [0.33, 2.98] | 2.17 [0.69, 4.14] | 13.6 % [4.0 %, 24.5 %] | [6.2, 45.7] % | 3.6 % [0.8 %, 7.2 %] | 37.7 % | 7.1 % [0.0 %, 23.1 %] (14) | 15.9 % [4.8 %, 27.3 %] | 20.5 % [8.7 %, 33.3 %] | 68.2 [16.8, 91.8] | 19.1 % [8.5 %, 30.8 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.736 [0.654, 0.814] | 0.643 [0.502, 0.783] | 0.736 [0.654, 0.814] | 0.105 [0.044, 0.266] | 0.0192 |
| 10 | 0.740 [0.657, 0.815] | 0.641 [0.501, 0.779] | 0.740 [0.657, 0.815] | 0.160 [0.083, 0.330] | 0.0387 |
| 20 | 0.722 [0.640, 0.798] | 0.616 [0.487, 0.748] | 0.722 [0.640, 0.798] | 0.233 [0.136, 0.414] | 0.0786 |
| 30 | 0.715 [0.635, 0.793] | 0.609 [0.483, 0.735] | 0.715 [0.635, 0.793] | 0.304 [0.198, 0.477] | 0.1197 |
| 60 | 0.691 [0.613, 0.766] | 0.600 [0.485, 0.722] | 0.691 [0.613, 0.766] | 0.441 [0.322, 0.596] | 0.2466 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.554 [0.423, 0.762] | 0.726 [0.554, 0.887] | 0.0 % [0.0 %, 0.0 %] | 16.7 % [0.0 %, 66.7 %] | 2.23 [0.00, 6.05] | 6 |
| source:xd_youtube | 0.775 [0.551, 0.998] | 0.848 [0.683, 1.000] | 50.0 % [0.0 %, 100.0 %] | 50.0 % [0.0 %, 100.0 %] | 0.00 [0.00, 0.00] | 4 |
| source:ucf | 0.813 [0.453, 0.973] | 0.626 [0.504, 0.765] | 5.9 % [0.0 %, 15.2 %] | 11.8 % [2.9 %, 25.0 %] | 0.00 [0.00, 0.00] | 34 |
| class:interpersonal | 0.705 [0.611, 0.796] | 0.724 [0.642, 0.807] | 10.0 % [2.3 %, 20.5 %] | 17.5 % [6.8 %, 30.6 %] | 0.73 [0.00, 1.97] | 40 |
| class:other_violent | 0.674 [0.401, 0.938] | 0.643 [0.348, 0.916] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.73 [0.00, 1.97] | 4 |
| audio:present | 0.671 [0.552, 0.782] | 0.730 [0.616, 0.833] | 13.0 % [0.0 %, 29.2 %] | 21.7 % [5.6 %, 40.0 %] | 0.78 [0.00, 2.13] | 23 |
| audio:absent | 0.820 [0.378, 1.000] | 0.600 [0.444, 0.784] | 4.8 % [0.0 %, 15.4 %] | 9.5 % [0.0 %, 23.5 %] | 0.00 [0.00, 0.00] | 21 |
| onset:official | 0.741 [0.624, 0.848] | 0.761 [0.638, 0.869] | 13.3 % [0.0 %, 33.3 %] | 20.0 % [0.0 %, 42.9 %] | 0.73 [0.00, 1.97] | 15 |
| onset:hivau | 0.681 [0.568, 0.791] | 0.710 [0.613, 0.806] | 6.9 % [0.0 %, 17.9 %] | 13.8 % [3.3 %, 28.1 %] | 0.73 [0.00, 1.97] | 29 |
| event:first_segment | n/a | n/a | 9.1 % [2.1 %, 18.6 %] | 15.9 % [6.2 %, 27.9 %] | 0.73 [0.00, 1.97] | 44 |
| event:later_segment | n/a | n/a | n/a | n/a | 0.73 [0.00, 1.97] | 0 |

Generated 2026-10-16T15:43:52 by `anticipation/eval/va30.py` (git d25ab37745).
