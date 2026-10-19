# VA30 report: va30_fa5

Fold file `folds_v2_without_confirmation` (content sha256 `5c5149555ca8`), seeds [0], headline seed 0. Test universe: 3515 videos (295 violent, 3220 clean negatives = 125.86 h), 344 eligible30 events (158 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 3515 of 4894 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 653 catalog clean negatives (527 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 125.86 of 242.3 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.852 [0.823, 0.881] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.849 [0.820, 0.878] |
| Far-window AUROC (clean, source x time matched negatives) | 0.838 [0.807, 0.868] |
| Far-window AUROC (clean, uncensored negatives) | 0.849 [0.821, 0.878] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.872 [0.850, 0.894] |
| AUROC / AP, H = 30 s (base rate 0.017) | 0.866 [0.842, 0.890] / 0.148 [0.111, 0.190] |
| Detection AUROC (in-violence vs clean negatives) | 0.933 [0.905, 0.953] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5.0 | 4.99 [4.02, 5.89] | 5.49 [4.49, 6.51] | 39.0 % [33.0 %, 44.3 %] | n/a | 11.7 % [9.6 %, 13.7 %] | 16.1 % | 36.9 % [28.1 %, 45.3 %] (157) | 51.5 % [45.6 %, 57.1 %] | 55.8 % [49.8 %, 61.5 %] | 42.6 [36.5, 47.7] | 63.6 % [58.0 %, 69.5 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.871 [0.848, 0.893] | 0.866 [0.843, 0.888] | 0.884 [0.862, 0.904] | 0.031 [0.022, 0.044] | 0.0028 |
| 10 | 0.871 [0.846, 0.894] | 0.865 [0.841, 0.889] | 0.881 [0.858, 0.901] | 0.058 [0.042, 0.080] | 0.0055 |
| 20 | 0.870 [0.846, 0.892] | 0.863 [0.839, 0.887] | 0.878 [0.855, 0.899] | 0.107 [0.079, 0.144] | 0.0113 |
| 30 | 0.866 [0.842, 0.890] | 0.858 [0.832, 0.883] | 0.873 [0.849, 0.895] | 0.148 [0.111, 0.190] | 0.0174 |
| 60 | 0.860 [0.836, 0.885] | 0.848 [0.823, 0.874] | 0.864 [0.838, 0.888] | 0.242 [0.190, 0.299] | 0.0355 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.788 [0.722, 0.855] | 0.813 [0.748, 0.877] | 52.7 % [41.2 %, 65.3 %] | 80.2 % [71.6 %, 89.4 %] | 10.58 [7.58, 14.23] | 91 |
| source:xd_youtube | 0.849 [0.792, 0.896] | 0.876 [0.839, 0.908] | 45.3 % [34.3 %, 55.8 %] | 67.0 % [56.7 %, 75.5 %] | 4.91 [3.82, 6.17] | 106 |
| source:ucf | 0.913 [0.875, 0.946] | 0.912 [0.877, 0.943] | 20.9 % [13.0 %, 29.9 %] | 25.3 % [16.0 %, 35.4 %] | 0.52 [0.24, 0.81] | 91 |
| source:ubi | 0.888 [0.823, 0.939] | 0.877 [0.803, 0.936] | 33.9 % [23.6 %, 45.0 %] | 44.6 % [31.1 %, 57.9 %] | 1.76 [1.09, 2.47] | 56 |
| class:interpersonal | 0.852 [0.818, 0.881] | 0.865 [0.839, 0.890] | 37.3 % [30.9 %, 43.0 %] | 52.6 % [46.2 %, 58.8 %] | 4.99 [4.02, 5.89] | 287 |
| class:other_violent | 0.856 [0.785, 0.908] | 0.876 [0.827, 0.911] | 47.4 % [31.9 %, 59.7 %] | 71.9 % [55.2 %, 84.4 %] | 4.99 [4.02, 5.89] | 57 |
| audio:present | 0.836 [0.800, 0.871] | 0.856 [0.822, 0.886] | 45.6 % [38.2 %, 52.5 %] | 67.1 % [60.1 %, 73.3 %] | 6.58 [5.28, 7.98] | 228 |
| audio:absent | 0.889 [0.844, 0.926] | 0.889 [0.852, 0.923] | 25.9 % [18.4 %, 33.1 %] | 33.6 % [25.5 %, 42.7 %] | 1.22 [0.84, 1.64] | 116 |
| onset:official | 0.839 [0.779, 0.895] | 0.861 [0.823, 0.896] | 42.3 % [28.6 %, 55.0 %] | 53.8 % [39.0 %, 68.6 %] | 4.99 [4.02, 5.89] | 52 |
| onset:hivau | 0.846 [0.813, 0.876] | 0.864 [0.834, 0.892] | 39.4 % [32.5 %, 46.1 %] | 58.9 % [51.4 %, 65.2 %] | 4.99 [4.02, 5.89] | 236 |
| onset:ubi | 0.907 [0.860, 0.947] | 0.893 [0.833, 0.941] | 33.9 % [23.6 %, 45.0 %] | 44.6 % [31.1 %, 57.9 %] | 4.99 [4.02, 5.89] | 56 |
| event:first_segment | n/a | n/a | 43.1 % [35.6 %, 50.6 %] | 59.9 % [52.4 %, 67.0 %] | 4.99 [4.02, 5.89] | 197 |
| event:later_segment | n/a | n/a | 33.3 % [25.4 %, 41.0 %] | 50.3 % [41.3 %, 58.9 %] | 4.99 [4.02, 5.89] | 147 |

Generated 2026-10-18T19:54:37 by `anticipation/eval/va30.py` (git a68aa73c61).
