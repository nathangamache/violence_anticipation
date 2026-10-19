# VA30 report: va30_fa125

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
| 1.0 | 1.16 [0.87, 1.46] | 1.53 [1.12, 2.00] | 20.6 % [15.5 %, 26.0 %] | [15.0, 30.6] % | 2.9 % [2.1 %, 3.6 %] | 7.3 % | 22.9 % [15.4 %, 30.9 %] (157) | 30.8 % [24.9 %, 36.8 %] | 34.3 % [28.3 %, 40.6 %] | 34.8 [30.4, 43.5] | 42.0 % [35.7 %, 48.0 %] |
| 2.0 | 2.12 [1.65, 2.59] | 2.55 [2.01, 3.19] | 31.4 % [26.2 %, 36.5 %] | [23.9, 37.8] % | 5.2 % [4.0 %, 6.3 %] | 9.5 % | 38.2 % [29.4 %, 46.6 %] (157) | 42.4 % [36.3 %, 48.5 %] | 45.3 % [39.1 %, 52.2 %] | 42.6 [36.9, 51.3] | 51.6 % [45.0 %, 57.8 %] |
| 5.0 | 4.99 [4.02, 5.89] | 5.49 [4.49, 6.51] | 39.0 % [33.0 %, 44.3 %] | [33.8, 45.9] % | 11.7 % [9.6 %, 13.7 %] | 16.1 % | 36.9 % [28.1 %, 45.3 %] (157) | 51.5 % [45.6 %, 57.1 %] | 55.8 % [49.8 %, 61.5 %] | 42.6 [36.5, 47.7] | 63.6 % [58.0 %, 69.5 %] |

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
| source:xd_movie | 0.788 [0.722, 0.855] | 0.813 [0.748, 0.877] | 40.7 % [29.4 %, 52.6 %] | 64.8 % [53.1 %, 77.2 %] | 4.34 [2.91, 6.10] | 91 |
| source:xd_youtube | 0.849 [0.792, 0.896] | 0.876 [0.839, 0.908] | 34.9 % [25.0 %, 45.2 %] | 50.9 % [39.8 %, 60.6 %] | 2.27 [1.68, 2.93] | 106 |
| source:ucf | 0.913 [0.875, 0.946] | 0.912 [0.877, 0.943] | 16.5 % [9.0 %, 24.5 %] | 22.0 % [13.4 %, 30.3 %] | 0.09 [0.00, 0.21] | 91 |
| source:ubi | 0.888 [0.823, 0.939] | 0.877 [0.803, 0.936] | 33.9 % [21.7 %, 47.1 %] | 41.1 % [27.8 %, 56.3 %] | 0.67 [0.32, 1.11] | 56 |
| class:interpersonal | 0.852 [0.818, 0.881] | 0.865 [0.839, 0.890] | 30.0 % [24.9 %, 35.6 %] | 41.8 % [35.9 %, 47.8 %] | 2.12 [1.65, 2.59] | 287 |
| class:other_violent | 0.856 [0.785, 0.908] | 0.876 [0.827, 0.911] | 38.6 % [20.9 %, 54.2 %] | 63.2 % [42.3 %, 78.3 %] | 2.12 [1.65, 2.59] | 57 |
| audio:present | 0.836 [0.800, 0.871] | 0.856 [0.822, 0.886] | 34.2 % [27.6 %, 41.1 %] | 52.6 % [44.7 %, 59.7 %] | 2.84 [2.20, 3.54] | 228 |
| audio:absent | 0.889 [0.844, 0.926] | 0.889 [0.852, 0.923] | 25.9 % [17.6 %, 33.7 %] | 31.0 % [22.1 %, 40.4 %] | 0.41 [0.21, 0.64] | 116 |
| onset:official | 0.839 [0.779, 0.895] | 0.861 [0.823, 0.896] | 30.8 % [18.4 %, 43.3 %] | 40.4 % [26.7 %, 54.2 %] | 2.12 [1.65, 2.59] | 52 |
| onset:hivau | 0.846 [0.813, 0.876] | 0.864 [0.834, 0.892] | 30.9 % [24.2 %, 37.9 %] | 47.5 % [39.7 %, 55.0 %] | 2.12 [1.65, 2.59] | 236 |
| onset:ubi | 0.907 [0.860, 0.947] | 0.893 [0.833, 0.941] | 33.9 % [21.7 %, 47.1 %] | 41.1 % [27.8 %, 56.3 %] | 2.12 [1.65, 2.59] | 56 |
| event:first_segment | n/a | n/a | 29.9 % [23.6 %, 36.6 %] | 44.7 % [37.3 %, 52.4 %] | 2.12 [1.65, 2.59] | 197 |
| event:later_segment | n/a | n/a | 33.3 % [25.6 %, 41.1 %] | 46.3 % [37.0 %, 55.3 %] | 2.12 [1.65, 2.59] | 147 |

Generated 2026-10-18T19:52:31 by `anticipation/eval/va30.py` (git a68aa73c61).
