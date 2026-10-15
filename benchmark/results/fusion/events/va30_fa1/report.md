# VA30 report: va30_fa1

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
| 1.0 | 1.16 [0.86, 1.51] | 1.59 [1.14, 2.15] | 23.5 % [19.1 %, 28.3 %] | n/a | 2.9 % [2.1 %, 3.7 %] | 6.4 % | 24.8 % [19.1 %, 30.7 %] (206) | 32.3 % [26.9 %, 37.9 %] | 35.7 % [30.4 %, 41.3 %] | 35.2 [32.4, 41.4] | 44.8 % [39.6 %, 50.3 %] |

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
| source:xd_movie | 0.813 [0.754, 0.869] | 0.829 [0.773, 0.882] | 34.7 % [22.2 %, 46.1 %] | 58.5 % [46.3 %, 71.1 %] | 2.66 [1.67, 4.03] | 118 |
| source:xd_youtube | 0.844 [0.785, 0.894] | 0.859 [0.816, 0.894] | 23.6 % [16.5 %, 31.9 %] | 33.1 % [24.4 %, 42.2 %] | 1.07 [0.70, 1.47] | 127 |
| source:ucf | 0.880 [0.823, 0.931] | 0.879 [0.830, 0.922] | 11.3 % [5.6 %, 17.4 %] | 16.5 % [9.6 %, 24.1 %] | 0.13 [0.00, 0.28] | 115 |
| source:ubi | 0.837 [0.726, 0.923] | 0.903 [0.851, 0.945] | 24.4 % [16.0 %, 32.5 %] | 33.7 % [24.4 %, 43.2 %] | 0.31 [0.10, 0.57] | 86 |
| class:interpersonal | 0.833 [0.796, 0.868] | 0.859 [0.829, 0.885] | 21.1 % [17.1 %, 25.5 %] | 32.6 % [28.0 %, 38.2 %] | 1.16 [0.86, 1.51] | 380 |
| class:other_violent | 0.886 [0.841, 0.925] | 0.880 [0.841, 0.910] | 37.9 % [21.2 %, 50.6 %] | 53.0 % [36.4 %, 66.1 %] | 1.16 [0.86, 1.51] | 66 |
| audio:present | 0.835 [0.793, 0.869] | 0.847 [0.813, 0.877] | 26.2 % [19.9 %, 32.6 %] | 40.6 % [33.4 %, 47.9 %] | 1.55 [1.14, 2.05] | 286 |
| audio:absent | 0.859 [0.796, 0.909] | 0.893 [0.856, 0.924] | 18.8 % [13.3 %, 24.5 %] | 26.9 % [20.1 %, 33.7 %] | 0.24 [0.10, 0.41] | 160 |
| onset:official | 0.830 [0.761, 0.886] | 0.841 [0.789, 0.883] | 19.6 % [9.5 %, 31.8 %] | 32.1 % [19.0 %, 45.8 %] | 1.16 [0.86, 1.51] | 56 |
| onset:hivau | 0.841 [0.801, 0.876] | 0.857 [0.824, 0.887] | 24.0 % [18.3 %, 30.4 %] | 36.8 % [30.5 %, 43.5 %] | 1.16 [0.86, 1.51] | 304 |
| onset:ubi | 0.862 [0.765, 0.934] | 0.916 [0.873, 0.950] | 24.4 % [16.0 %, 32.5 %] | 33.7 % [24.4 %, 43.2 %] | 1.16 [0.86, 1.51] | 86 |
| event:first_segment | n/a | n/a | 23.3 % [16.9 %, 29.6 %] | 35.2 % [28.6 %, 42.2 %] | 1.16 [0.86, 1.51] | 253 |
| event:later_segment | n/a | n/a | 23.8 % [18.4 %, 29.8 %] | 36.3 % [30.0 %, 43.3 %] | 1.16 [0.86, 1.51] | 193 |

Generated 2026-10-15T18:08:37 by `anticipation/eval/va30.py` (git 56b27f65df).
