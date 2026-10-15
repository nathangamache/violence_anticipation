# VA30 report: va30_fa125

Fold file `folds_v2_without_confirmation` (content sha256 `5c5149555ca8`), seeds [0], headline seed 0. Test universe: 3599 videos (379 violent, 3220 clean negatives = 125.86 h), 446 eligible30 events (181 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 3599 of 4916 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 653 catalog clean negatives (527 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 125.86 of 242.3 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.842 [0.810, 0.872] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.838 [0.806, 0.867] |
| Far-window AUROC (clean, source x time matched negatives) | 0.828 [0.793, 0.859] |
| Far-window AUROC (clean, uncensored negatives) | 0.839 [0.807, 0.869] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.851 [0.823, 0.877] |
| AUROC / AP, H = 30 s (base rate 0.022) | 0.862 [0.835, 0.885] / 0.198 [0.153, 0.246] |
| Detection AUROC (in-violence vs clean negatives) | 0.910 [0.880, 0.934] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 1.16 [0.86, 1.51] | 1.59 [1.14, 2.15] | 23.5 % [19.1 %, 28.3 %] | [15.2, 30.5] % | 2.9 % [2.1 %, 3.7 %] | 6.4 % | 24.8 % [19.1 %, 30.7 %] (206) | 32.3 % [26.9 %, 37.9 %] | 35.7 % [30.4 %, 41.3 %] | 35.2 [32.4, 41.4] | 44.8 % [39.6 %, 50.3 %] |
| 2.0 | 2.26 [1.78, 2.81] | 2.69 [2.13, 3.41] | 31.2 % [25.8 %, 36.6 %] | [23.6, 36.1] % | 5.5 % [4.3 %, 6.8 %] | 9.0 % | 31.1 % [24.0 %, 38.5 %] (206) | 43.5 % [37.9 %, 49.4 %] | 46.2 % [40.6 %, 52.2 %] | 35.8 [33.2, 41.1] | 53.7 % [48.4 %, 59.0 %] |
| 5.0 | 5.11 [4.23, 6.20] | 5.51 [4.62, 6.61] | 38.1 % [33.3 %, 42.9 %] | [32.4, 44.5] % | 12.0 % [10.0 %, 14.4 %] | 15.9 % | 32.5 % [26.4 %, 39.7 %] (206) | 48.4 % [43.4 %, 53.7 %] | 52.2 % [46.8 %, 57.6 %] | 39.2 [34.8, 43.0] | 62.5 % [57.6 %, 67.1 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.873 [0.848, 0.896] | 0.867 [0.841, 0.890] | 0.879 [0.859, 0.900] | 0.048 [0.036, 0.068] | 0.0035 |
| 10 | 0.870 [0.845, 0.892] | 0.864 [0.838, 0.886] | 0.876 [0.854, 0.896] | 0.084 [0.063, 0.114] | 0.0071 |
| 20 | 0.864 [0.838, 0.887] | 0.857 [0.831, 0.880] | 0.871 [0.849, 0.893] | 0.147 [0.111, 0.191] | 0.0144 |
| 30 | 0.862 [0.835, 0.885] | 0.852 [0.825, 0.876] | 0.867 [0.843, 0.889] | 0.198 [0.153, 0.246] | 0.0222 |
| 60 | 0.853 [0.826, 0.878] | 0.840 [0.811, 0.865] | 0.856 [0.831, 0.879] | 0.291 [0.234, 0.347] | 0.0437 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.813 [0.754, 0.869] | 0.829 [0.773, 0.882] | 46.6 % [33.0 %, 58.2 %] | 72.0 % [60.4 %, 83.2 %] | 4.80 [3.15, 7.08] | 118 |
| source:xd_youtube | 0.844 [0.785, 0.894] | 0.859 [0.816, 0.894] | 31.5 % [23.2 %, 41.2 %] | 44.1 % [35.0 %, 52.8 %] | 2.25 [1.64, 2.91] | 127 |
| source:ucf | 0.880 [0.823, 0.931] | 0.879 [0.830, 0.922] | 14.8 % [8.2 %, 21.9 %] | 22.6 % [15.2 %, 30.4 %] | 0.13 [0.00, 0.28] | 115 |
| source:ubi | 0.837 [0.726, 0.923] | 0.903 [0.851, 0.945] | 31.4 % [21.1 %, 42.7 %] | 45.3 % [34.8 %, 58.2 %] | 0.83 [0.46, 1.22] | 86 |
| class:interpersonal | 0.833 [0.796, 0.868] | 0.859 [0.829, 0.885] | 28.4 % [23.6 %, 33.9 %] | 42.4 % [37.2 %, 48.2 %] | 2.26 [1.78, 2.81] | 380 |
| class:other_violent | 0.886 [0.841, 0.925] | 0.880 [0.841, 0.910] | 47.0 % [29.0 %, 61.3 %] | 68.2 % [52.7 %, 79.5 %] | 2.26 [1.78, 2.81] | 66 |
| audio:present | 0.835 [0.793, 0.869] | 0.847 [0.813, 0.877] | 35.3 % [28.4 %, 42.3 %] | 52.4 % [45.3 %, 59.2 %] | 2.99 [2.31, 3.81] | 286 |
| audio:absent | 0.859 [0.796, 0.909] | 0.893 [0.856, 0.924] | 23.8 % [17.0 %, 30.9 %] | 35.0 % [27.6 %, 43.0 %] | 0.51 [0.30, 0.74] | 160 |
| onset:official | 0.830 [0.761, 0.886] | 0.841 [0.789, 0.883] | 26.8 % [14.3 %, 40.9 %] | 46.4 % [31.5 %, 60.3 %] | 2.26 [1.78, 2.81] | 56 |
| onset:hivau | 0.841 [0.801, 0.876] | 0.857 [0.824, 0.887] | 31.9 % [25.4 %, 38.8 %] | 46.4 % [39.4 %, 53.0 %] | 2.26 [1.78, 2.81] | 304 |
| onset:ubi | 0.862 [0.765, 0.934] | 0.916 [0.873, 0.950] | 31.4 % [21.1 %, 42.7 %] | 45.3 % [34.8 %, 58.2 %] | 2.26 [1.78, 2.81] | 86 |
| event:first_segment | n/a | n/a | 34.0 % [26.4 %, 40.7 %] | 47.8 % [40.7 %, 54.6 %] | 2.26 [1.78, 2.81] | 253 |
| event:later_segment | n/a | n/a | 27.5 % [21.0 %, 34.8 %] | 44.0 % [36.7 %, 52.5 %] | 2.26 [1.78, 2.81] | 193 |

Generated 2026-10-15T18:07:58 by `anticipation/eval/va30.py` (git 56b27f65df).
