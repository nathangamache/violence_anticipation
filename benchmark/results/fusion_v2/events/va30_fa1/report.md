# VA30 report: va30_fa1

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
| 1.0 | 1.16 [0.87, 1.46] | 1.53 [1.12, 2.00] | 20.6 % [15.5 %, 26.0 %] | n/a | 2.9 % [2.1 %, 3.6 %] | 7.3 % | 22.9 % [15.4 %, 30.9 %] (157) | 30.8 % [24.9 %, 36.8 %] | 34.3 % [28.3 %, 40.6 %] | 34.8 [30.4, 43.5] | 42.0 % [35.7 %, 48.0 %] |

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
| source:xd_movie | 0.788 [0.722, 0.855] | 0.813 [0.748, 0.877] | 28.6 % [16.8 %, 40.3 %] | 49.5 % [35.8 %, 64.6 %] | 2.50 [1.63, 3.61] | 91 |
| source:xd_youtube | 0.849 [0.792, 0.896] | 0.876 [0.839, 0.908] | 28.3 % [18.9 %, 37.9 %] | 44.3 % [33.3 %, 55.3 %] | 1.22 [0.83, 1.73] | 106 |
| source:ucf | 0.913 [0.875, 0.946] | 0.912 [0.877, 0.943] | 4.4 % [0.9 %, 8.9 %] | 11.0 % [4.8 %, 18.2 %] | 0.00 [0.00, 0.00] | 91 |
| source:ubi | 0.888 [0.823, 0.939] | 0.877 [0.803, 0.936] | 19.6 % [10.4 %, 31.4 %] | 28.6 % [16.9 %, 43.1 %] | 0.31 [0.06, 0.58] | 56 |
| class:interpersonal | 0.852 [0.818, 0.881] | 0.865 [0.839, 0.890] | 18.8 % [14.0 %, 24.2 %] | 30.7 % [24.8 %, 37.1 %] | 1.16 [0.87, 1.46] | 287 |
| class:other_violent | 0.856 [0.785, 0.908] | 0.876 [0.827, 0.911] | 29.8 % [13.7 %, 43.4 %] | 52.6 % [34.4 %, 66.2 %] | 1.16 [0.87, 1.46] | 57 |
| audio:present | 0.836 [0.800, 0.871] | 0.856 [0.822, 0.886] | 25.0 % [18.4 %, 32.0 %] | 41.2 % [32.9 %, 48.9 %] | 1.59 [1.17, 2.02] | 228 |
| audio:absent | 0.889 [0.844, 0.926] | 0.889 [0.852, 0.923] | 12.1 % [6.6 %, 18.6 %] | 20.7 % [13.3 %, 29.0 %] | 0.16 [0.03, 0.30] | 116 |
| onset:official | 0.839 [0.779, 0.895] | 0.861 [0.823, 0.896] | 17.3 % [5.9 %, 29.6 %] | 30.8 % [17.4 %, 44.8 %] | 1.16 [0.87, 1.46] | 52 |
| onset:hivau | 0.846 [0.813, 0.876] | 0.864 [0.834, 0.892] | 21.6 % [15.5 %, 28.2 %] | 36.4 % [29.0 %, 44.1 %] | 1.16 [0.87, 1.46] | 236 |
| onset:ubi | 0.907 [0.860, 0.947] | 0.893 [0.833, 0.941] | 19.6 % [10.4 %, 31.4 %] | 28.6 % [16.9 %, 43.1 %] | 1.16 [0.87, 1.46] | 56 |
| event:first_segment | n/a | n/a | 19.8 % [13.9 %, 26.3 %] | 31.5 % [24.1 %, 38.9 %] | 1.16 [0.87, 1.46] | 197 |
| event:later_segment | n/a | n/a | 21.8 % [15.0 %, 29.2 %] | 38.1 % [28.9 %, 47.2 %] | 1.16 [0.87, 1.46] | 147 |

Generated 2026-10-18T19:53:31 by `anticipation/eval/va30.py` (git a68aa73c61).
