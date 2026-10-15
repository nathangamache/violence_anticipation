# VA30 report: va30

Fold file `folds_v2_without_confirmation` (content sha256 `5c5149555ca8`), seeds [0], headline seed 0. Test universe: 3599 videos (379 violent, 3220 clean negatives = 125.86 h), 446 eligible30 events (181 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 3599 of 4916 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 653 catalog clean negatives (527 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 125.86 of 242.3 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.842 [0.813, 0.868] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.838 [0.809, 0.864] |
| Far-window AUROC (clean, source x time matched negatives) | 0.828 [0.797, 0.856] |
| Far-window AUROC (clean, uncensored negatives) | 0.839 [0.811, 0.865] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.851 [0.827, 0.873] |
| AUROC / AP, H = 30 s (base rate 0.022) | 0.862 [0.842, 0.882] / 0.198 [0.158, 0.246] |
| Detection AUROC (in-violence vs clean negatives) | 0.910 [0.882, 0.932] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 1.16 [0.94, 1.38] | 1.59 [1.19, 2.08] | 23.5 % [19.5 %, 27.8 %] | [15.9, 29.8] % | 2.9 % [2.3 %, 3.4 %] | 6.4 % | 24.8 % [19.1 %, 31.4 %] (206) | 32.3 % [27.8 %, 37.2 %] | 35.7 % [31.0 %, 40.7 %] | 35.2 [32.6, 41.2] | 44.8 % [40.2 %, 50.2 %] |
| 2.0 | 2.26 [1.94, 2.59] | 2.69 [2.25, 3.20] | 31.2 % [26.7 %, 36.1 %] | [25.2, 35.5] % | 5.5 % [4.7 %, 6.3 %] | 9.0 % | 31.1 % [24.6 %, 38.4 %] (206) | 43.5 % [38.8 %, 48.4 %] | 46.2 % [41.3 %, 51.2 %] | 35.8 [33.1, 41.2] | 53.7 % [49.1 %, 58.7 %] |
| 5.0 | 5.11 [4.59, 5.64] | 5.51 [4.91, 6.20] | 38.1 % [33.5 %, 42.7 %] | [33.5, 42.5] % | 12.0 % [10.8 %, 13.2 %] | 15.9 % | 32.5 % [26.4 %, 38.7 %] (206) | 48.4 % [43.8 %, 53.3 %] | 52.2 % [47.4 %, 57.0 %] | 39.2 [34.8, 42.9] | 62.5 % [58.1 %, 66.9 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.873 [0.852, 0.893] | 0.867 [0.847, 0.888] | 0.879 [0.861, 0.896] | 0.048 [0.036, 0.068] | 0.0035 |
| 10 | 0.870 [0.849, 0.890] | 0.864 [0.843, 0.883] | 0.876 [0.858, 0.893] | 0.084 [0.064, 0.115] | 0.0071 |
| 20 | 0.864 [0.844, 0.885] | 0.857 [0.835, 0.877] | 0.871 [0.853, 0.888] | 0.147 [0.115, 0.188] | 0.0144 |
| 30 | 0.862 [0.842, 0.882] | 0.852 [0.830, 0.873] | 0.867 [0.848, 0.884] | 0.198 [0.158, 0.246] | 0.0222 |
| 60 | 0.853 [0.831, 0.874] | 0.840 [0.816, 0.863] | 0.856 [0.836, 0.875] | 0.291 [0.242, 0.346] | 0.0437 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.813 [0.760, 0.858] | 0.829 [0.796, 0.859] | 46.6 % [38.2 %, 56.1 %] | 72.0 % [63.6 %, 80.2 %] | 4.80 [3.93, 5.70] | 118 |
| source:xd_youtube | 0.844 [0.795, 0.889] | 0.859 [0.823, 0.893] | 31.5 % [22.9 %, 39.7 %] | 44.1 % [35.1 %, 52.8 %] | 2.25 [1.72, 2.82] | 127 |
| source:ucf | 0.880 [0.820, 0.931] | 0.879 [0.829, 0.922] | 14.8 % [8.7 %, 22.1 %] | 22.6 % [15.7 %, 30.8 %] | 0.13 [0.00, 0.28] | 115 |
| source:ubi | 0.837 [0.739, 0.917] | 0.903 [0.851, 0.947] | 31.4 % [21.4 %, 42.5 %] | 45.3 % [34.4 %, 58.0 %] | 0.83 [0.44, 1.30] | 86 |
| class:interpersonal | 0.833 [0.800, 0.863] | 0.859 [0.835, 0.882] | 28.4 % [23.6 %, 33.3 %] | 42.4 % [37.3 %, 47.6 %] | 2.26 [1.94, 2.59] | 380 |
| class:other_violent | 0.886 [0.845, 0.919] | 0.880 [0.851, 0.906] | 47.0 % [35.9 %, 58.0 %] | 68.2 % [57.1 %, 79.4 %] | 2.26 [1.94, 2.59] | 66 |
| audio:present | 0.835 [0.799, 0.866] | 0.847 [0.822, 0.872] | 35.3 % [29.5 %, 41.5 %] | 52.4 % [46.6 %, 58.7 %] | 2.99 [2.54, 3.43] | 286 |
| audio:absent | 0.859 [0.801, 0.909] | 0.893 [0.856, 0.925] | 23.8 % [17.4 %, 30.7 %] | 35.0 % [28.0 %, 42.7 %] | 0.51 [0.29, 0.76] | 160 |
| onset:official | 0.830 [0.759, 0.889] | 0.841 [0.791, 0.883] | 26.8 % [15.7 %, 38.5 %] | 46.4 % [33.3 %, 59.2 %] | 2.26 [1.94, 2.59] | 56 |
| onset:hivau | 0.841 [0.805, 0.872] | 0.857 [0.832, 0.882] | 31.9 % [26.3 %, 37.7 %] | 46.4 % [40.5 %, 52.3 %] | 2.26 [1.94, 2.59] | 304 |
| onset:ubi | 0.862 [0.774, 0.933] | 0.916 [0.876, 0.953] | 31.4 % [21.4 %, 42.5 %] | 45.3 % [34.4 %, 58.0 %] | 2.26 [1.94, 2.59] | 86 |
| event:first_segment | n/a | n/a | 34.0 % [28.7 %, 40.4 %] | 47.8 % [41.7 %, 54.2 %] | 2.26 [1.94, 2.59] | 253 |
| event:later_segment | n/a | n/a | 27.5 % [20.9 %, 34.1 %] | 44.0 % [36.7 %, 52.7 %] | 2.26 [1.94, 2.59] | 193 |

Generated 2026-10-15T13:07:24 by `anticipation/eval/va30.py` (git 00cd04bb59).
