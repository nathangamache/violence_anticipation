# VA30 report: va30

Fold file `folds_v2_without_confirmation` (content sha256 `5c5149555ca8`), seeds [0], headline seed 0. Test universe: 3599 videos (379 violent, 3220 clean negatives = 125.86 h), 446 eligible30 events (181 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 3599 of 4916 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 653 catalog clean negatives (527 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 125.86 of 242.3 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.835 [0.808, 0.860] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.831 [0.804, 0.856] |
| Far-window AUROC (clean, source x time matched negatives) | 0.819 [0.790, 0.846] |
| Far-window AUROC (clean, uncensored negatives) | 0.832 [0.805, 0.856] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.851 [0.830, 0.870] |
| AUROC / AP, H = 30 s (base rate 0.022) | 0.845 [0.824, 0.864] / 0.189 [0.154, 0.233] |
| Detection AUROC (in-violence vs clean negatives) | 0.897 [0.864, 0.923] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 1.28 [1.05, 1.51] | 1.53 [1.21, 1.88] | 19.3 % [15.6 %, 23.2 %] | [15.0, 25.1] % | 3.1 % [2.6 %, 3.7 %] | 6.7 % | 18.9 % [13.2 %, 25.2 %] (206) | 27.8 % [23.3 %, 32.4 %] | 31.6 % [27.2 %, 36.7 %] | 33.7 [31.1, 39.7] | 36.6 % [32.1 %, 41.2 %] |
| 2.0 | 2.34 [2.02, 2.66] | 2.70 [2.29, 3.15] | 27.1 % [23.1 %, 31.6 %] | [21.3, 32.4] % | 5.7 % [4.9 %, 6.4 %] | 9.2 % | 24.8 % [19.1 %, 31.0 %] (206) | 37.7 % [33.3 %, 42.4 %] | 41.0 % [36.3 %, 45.7 %] | 37.5 [33.0, 41.8] | 46.7 % [42.3 %, 51.5 %] |
| 5.0 | 5.04 [4.58, 5.54] | 5.43 [4.87, 6.01] | 37.0 % [32.7 %, 41.5 %] | [32.7, 41.7] % | 11.8 % [10.8 %, 12.9 %] | 15.8 % | 33.5 % [27.4 %, 40.2 %] (206) | 48.7 % [44.3 %, 53.3 %] | 53.4 % [48.8 %, 58.4 %] | 41.1 [34.6, 43.9] | 61.0 % [56.3 %, 65.5 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.854 [0.833, 0.874] | 0.849 [0.827, 0.869] | 0.870 [0.853, 0.885] | 0.045 [0.033, 0.063] | 0.0035 |
| 10 | 0.852 [0.831, 0.871] | 0.845 [0.824, 0.865] | 0.865 [0.848, 0.882] | 0.083 [0.063, 0.111] | 0.0071 |
| 20 | 0.847 [0.827, 0.867] | 0.838 [0.817, 0.859] | 0.860 [0.841, 0.877] | 0.142 [0.112, 0.179] | 0.0144 |
| 30 | 0.845 [0.824, 0.864] | 0.834 [0.812, 0.854] | 0.856 [0.837, 0.873] | 0.189 [0.154, 0.233] | 0.0222 |
| 60 | 0.841 [0.819, 0.861] | 0.826 [0.803, 0.846] | 0.849 [0.828, 0.867] | 0.277 [0.233, 0.325] | 0.0437 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.816 [0.767, 0.860] | 0.824 [0.791, 0.856] | 41.5 % [32.6 %, 50.4 %] | 61.9 % [52.3 %, 70.6 %] | 4.93 [4.08, 5.77] | 118 |
| source:xd_youtube | 0.828 [0.767, 0.878] | 0.856 [0.820, 0.891] | 31.5 % [23.7 %, 39.7 %] | 46.5 % [37.6 %, 55.4 %] | 2.11 [1.59, 2.63] | 127 |
| source:ucf | 0.892 [0.854, 0.927] | 0.869 [0.828, 0.907] | 9.6 % [4.2 %, 15.5 %] | 15.7 % [8.7 %, 22.5 %] | 0.34 [0.08, 0.70] | 115 |
| source:ubi | 0.811 [0.708, 0.896] | 0.842 [0.769, 0.912] | 24.4 % [15.7 %, 35.3 %] | 38.4 % [28.9 %, 48.3 %] | 1.24 [0.59, 2.02] | 86 |
| class:interpersonal | 0.830 [0.799, 0.858] | 0.842 [0.818, 0.865] | 25.3 % [21.2 %, 29.8 %] | 38.9 % [33.9 %, 44.1 %] | 2.34 [2.02, 2.66] | 380 |
| class:other_violent | 0.861 [0.819, 0.898] | 0.863 [0.832, 0.890] | 37.9 % [26.2 %, 48.3 %] | 53.0 % [40.6 %, 64.6 %] | 2.34 [2.02, 2.66] | 66 |
| audio:present | 0.836 [0.803, 0.865] | 0.847 [0.821, 0.870] | 32.5 % [27.2 %, 37.7 %] | 48.3 % [42.3 %, 53.9 %] | 2.96 [2.53, 3.37] | 286 |
| audio:absent | 0.827 [0.772, 0.874] | 0.839 [0.799, 0.881] | 17.5 % [11.8 %, 24.0 %] | 28.1 % [21.1 %, 35.0 %] | 0.87 [0.48, 1.34] | 160 |
| onset:official | 0.832 [0.775, 0.881] | 0.844 [0.796, 0.884] | 25.0 % [14.1 %, 36.1 %] | 35.7 % [23.7 %, 47.5 %] | 2.34 [2.02, 2.66] | 56 |
| onset:hivau | 0.828 [0.793, 0.858] | 0.837 [0.811, 0.862] | 28.3 % [23.4 %, 33.4 %] | 42.8 % [36.8 %, 48.9 %] | 2.34 [2.02, 2.66] | 304 |
| onset:ubi | 0.881 [0.810, 0.939] | 0.897 [0.848, 0.942] | 24.4 % [15.7 %, 35.3 %] | 38.4 % [28.9 %, 48.3 %] | 2.34 [2.02, 2.66] | 86 |
| event:first_segment | n/a | n/a | 28.5 % [23.1 %, 34.1 %] | 42.7 % [36.8 %, 48.7 %] | 2.34 [2.02, 2.66] | 253 |
| event:later_segment | n/a | n/a | 25.4 % [19.7 %, 31.5 %] | 38.9 % [32.2 %, 45.8 %] | 2.34 [2.02, 2.66] | 193 |

Generated 2026-10-15T13:13:38 by `anticipation/eval/va30.py` (git 00cd04bb59).
