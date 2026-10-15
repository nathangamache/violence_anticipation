# VA30 report: va30

Fold file `folds_v2_without_confirmation` (content sha256 `5c5149555ca8`), seeds [0], headline seed 0. Test universe: 3599 videos (379 violent, 3220 clean negatives = 125.86 h), 446 eligible30 events (181 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 3599 of 4916 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 653 catalog clean negatives (527 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 125.86 of 242.3 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.878 [0.854, 0.900] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.873 [0.850, 0.895] |
| Far-window AUROC (clean, source x time matched negatives) | 0.863 [0.838, 0.888] |
| Far-window AUROC (clean, uncensored negatives) | 0.870 [0.847, 0.893] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.893 [0.875, 0.911] |
| AUROC / AP, H = 30 s (base rate 0.022) | 0.885 [0.869, 0.902] / 0.222 [0.185, 0.267] |
| Detection AUROC (in-violence vs clean negatives) | 0.928 [0.897, 0.953] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 1.20 [0.98, 1.43] | 1.50 [1.17, 1.90] | 16.8 % [13.4 %, 20.4 %] | [10.7, 21.9] % | 2.9 % [2.4 %, 3.5 %] | 6.5 % | 15.0 % [10.1 %, 20.9 %] (206) | 24.4 % [20.4 %, 29.0 %] | 26.2 % [22.0 %, 30.9 %] | 34.2 [31.2, 38.7] | 32.3 % [27.6 %, 37.2 %] |
| 2.0 | 1.97 [1.66, 2.31] | 2.39 [1.96, 2.88] | 24.7 % [20.7 %, 29.0 %] | [19.5, 30.2] % | 4.8 % [4.1 %, 5.6 %] | 8.4 % | 22.3 % [16.9 %, 28.8 %] (206) | 33.0 % [28.4 %, 37.7 %] | 35.9 % [31.2 %, 40.6 %] | 36.8 [34.0, 39.4] | 41.1 % [36.0 %, 46.3 %] |
| 5.0 | 4.63 [4.16, 5.12] | 5.08 [4.50, 5.71] | 35.4 % [30.8 %, 40.0 %] | [29.6, 40.8] % | 10.9 % [9.9 %, 12.0 %] | 14.8 % | 27.7 % [21.5 %, 34.0 %] (206) | 45.7 % [41.0 %, 50.4 %] | 48.2 % [43.4 %, 53.1 %] | 40.0 [36.6, 44.0] | 57.7 % [52.5 %, 62.6 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.889 [0.872, 0.906] | 0.877 [0.859, 0.894] | 0.895 [0.877, 0.909] | 0.051 [0.040, 0.067] | 0.0035 |
| 10 | 0.887 [0.869, 0.903] | 0.874 [0.855, 0.891] | 0.892 [0.873, 0.906] | 0.092 [0.072, 0.120] | 0.0071 |
| 20 | 0.885 [0.868, 0.902] | 0.871 [0.853, 0.890] | 0.891 [0.874, 0.906] | 0.160 [0.129, 0.198] | 0.0144 |
| 30 | 0.885 [0.869, 0.902] | 0.870 [0.852, 0.889] | 0.891 [0.873, 0.905] | 0.222 [0.185, 0.267] | 0.0222 |
| 60 | 0.882 [0.864, 0.900] | 0.866 [0.846, 0.885] | 0.887 [0.867, 0.904] | 0.342 [0.295, 0.394] | 0.0437 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.823 [0.772, 0.871] | 0.832 [0.798, 0.865] | 34.7 % [26.8 %, 44.0 %] | 54.2 % [45.4 %, 63.9 %] | 3.61 [2.86, 4.46] | 118 |
| source:xd_youtube | 0.848 [0.799, 0.894] | 0.865 [0.824, 0.900] | 26.0 % [18.8 %, 34.2 %] | 38.6 % [31.2 %, 46.9 %] | 2.33 [1.76, 2.90] | 127 |
| source:ucf | 0.907 [0.865, 0.945] | 0.901 [0.868, 0.930] | 19.1 % [12.6 %, 26.3 %] | 24.3 % [17.2 %, 31.8 %] | 0.65 [0.30, 1.07] | 115 |
| source:ubi | 0.967 [0.939, 0.988] | 0.968 [0.950, 0.983] | 16.3 % [8.2 %, 25.7 %] | 22.1 % [12.6 %, 33.3 %] | 0.05 [0.00, 0.19] | 86 |
| class:interpersonal | 0.879 [0.853, 0.905] | 0.885 [0.867, 0.904] | 22.6 % [18.4 %, 27.2 %] | 32.6 % [27.8 %, 37.5 %] | 1.97 [1.66, 2.31] | 380 |
| class:other_violent | 0.872 [0.817, 0.920] | 0.888 [0.849, 0.922] | 36.4 % [25.7 %, 47.9 %] | 54.5 % [41.7 %, 67.1 %] | 1.97 [1.66, 2.31] | 66 |
| audio:present | 0.853 [0.822, 0.882] | 0.862 [0.841, 0.884] | 28.3 % [23.7 %, 33.8 %] | 42.7 % [37.1 %, 49.0 %] | 2.67 [2.23, 3.13] | 286 |
| audio:absent | 0.930 [0.892, 0.962] | 0.932 [0.905, 0.955] | 18.1 % [12.2 %, 24.7 %] | 23.8 % [17.1 %, 30.9 %] | 0.32 [0.13, 0.60] | 160 |
| onset:official | 0.840 [0.761, 0.913] | 0.889 [0.843, 0.929] | 25.0 % [14.5 %, 36.5 %] | 42.9 % [30.4 %, 56.3 %] | 1.97 [1.66, 2.31] | 56 |
| onset:hivau | 0.874 [0.846, 0.901] | 0.875 [0.853, 0.897] | 27.0 % [22.3 %, 32.5 %] | 38.5 % [33.4 %, 44.3 %] | 1.97 [1.66, 2.31] | 304 |
| onset:ubi | 0.947 [0.917, 0.968] | 0.946 [0.927, 0.964] | 16.3 % [8.2 %, 25.7 %] | 22.1 % [12.6 %, 33.3 %] | 1.97 [1.66, 2.31] | 86 |
| event:first_segment | n/a | n/a | 27.3 % [22.2 %, 32.9 %] | 38.3 % [32.6 %, 44.7 %] | 1.97 [1.66, 2.31] | 253 |
| event:later_segment | n/a | n/a | 21.2 % [15.6 %, 27.8 %] | 32.6 % [25.9 %, 40.3 %] | 1.97 [1.66, 2.31] | 193 |

Generated 2026-10-15T13:19:27 by `anticipation/eval/va30.py` (git 00cd04bb59).
