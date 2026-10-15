# VA30 report: va30_fa5

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
| 5.0 | 5.11 [4.23, 6.20] | 5.51 [4.62, 6.61] | 38.1 % [33.3 %, 42.9 %] | n/a | 12.0 % [10.0 %, 14.4 %] | 15.9 % | 32.5 % [26.4 %, 39.7 %] (206) | 48.4 % [43.4 %, 53.7 %] | 52.2 % [46.8 %, 57.6 %] | 39.2 [34.8, 43.0] | 62.5 % [57.6 %, 67.1 %] |

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
| source:xd_movie | 0.813 [0.754, 0.869] | 0.829 [0.773, 0.882] | 56.8 % [46.3 %, 67.3 %] | 78.8 % [69.0 %, 89.5 %] | 10.44 [7.45, 14.84] | 118 |
| source:xd_youtube | 0.844 [0.785, 0.894] | 0.859 [0.816, 0.894] | 40.2 % [31.3 %, 49.5 %] | 54.3 % [45.5 %, 62.9 %] | 5.00 [3.93, 6.27] | 127 |
| source:ucf | 0.880 [0.823, 0.931] | 0.879 [0.830, 0.922] | 18.3 % [11.8 %, 25.4 %] | 26.1 % [18.9 %, 34.5 %] | 0.73 [0.38, 1.19] | 115 |
| source:ubi | 0.837 [0.726, 0.923] | 0.903 [0.851, 0.945] | 36.0 % [26.0 %, 46.9 %] | 47.7 % [36.7 %, 59.5 %] | 2.28 [1.51, 3.17] | 86 |
| class:interpersonal | 0.833 [0.796, 0.868] | 0.859 [0.829, 0.885] | 35.5 % [30.4 %, 40.6 %] | 48.7 % [43.4 %, 54.3 %] | 5.11 [4.23, 6.20] | 380 |
| class:other_violent | 0.886 [0.841, 0.925] | 0.880 [0.841, 0.910] | 53.0 % [39.7 %, 63.6 %] | 72.7 % [58.6 %, 83.5 %] | 5.11 [4.23, 6.20] | 66 |
| audio:present | 0.835 [0.793, 0.869] | 0.847 [0.813, 0.877] | 43.7 % [37.4 %, 50.0 %] | 59.8 % [53.2 %, 66.1 %] | 6.60 [5.39, 8.14] | 286 |
| audio:absent | 0.859 [0.796, 0.909] | 0.893 [0.856, 0.924] | 28.1 % [21.0 %, 35.5 %] | 38.8 % [31.0 %, 46.6 %] | 1.60 [1.12, 2.13] | 160 |
| onset:official | 0.830 [0.761, 0.886] | 0.841 [0.789, 0.883] | 44.6 % [31.0 %, 58.3 %] | 55.4 % [40.8 %, 68.9 %] | 5.11 [4.23, 6.20] | 56 |
| onset:hivau | 0.841 [0.801, 0.876] | 0.857 [0.824, 0.887] | 37.5 % [31.5 %, 43.6 %] | 53.0 % [46.3 %, 59.3 %] | 5.11 [4.23, 6.20] | 304 |
| onset:ubi | 0.862 [0.765, 0.934] | 0.916 [0.873, 0.950] | 36.0 % [26.0 %, 46.9 %] | 47.7 % [36.7 %, 59.5 %] | 5.11 [4.23, 6.20] | 86 |
| event:first_segment | n/a | n/a | 45.1 % [38.2 %, 51.7 %] | 59.3 % [52.2 %, 65.7 %] | 5.11 [4.23, 6.20] | 253 |
| event:later_segment | n/a | n/a | 29.0 % [22.9 %, 35.8 %] | 43.0 % [35.9 %, 50.5 %] | 5.11 [4.23, 6.20] | 193 |

Generated 2026-10-15T18:09:15 by `anticipation/eval/va30.py` (git 56b27f65df).
