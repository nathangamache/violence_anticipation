# VA30 report: va30

Fold file `folds_v2_without_confirmation` (content sha256 `5c5149555ca8`), seeds [0], headline seed 0. Test universe: 3599 videos (379 violent, 3220 clean negatives = 125.86 h), 446 eligible30 events (181 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 3599 of 4916 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 653 catalog clean negatives (527 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 125.86 of 242.3 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.611 [0.581, 0.639] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.607 [0.576, 0.634] |
| Far-window AUROC (clean, source x time matched negatives) | 0.584 [0.549, 0.615] |
| Far-window AUROC (clean, uncensored negatives) | 0.612 [0.582, 0.639] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.600 [0.573, 0.628] |
| AUROC / AP, H = 30 s (base rate 0.022) | 0.592 [0.567, 0.617] / 0.030 [0.026, 0.035] |
| Detection AUROC (in-violence vs clean negatives) | 0.638 [0.597, 0.677] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 1.02 [0.83, 1.25] | 1.01 [0.81, 1.21] | 3.1 % [1.6 %, 5.0 %] | [1.8, 5.5] % | 2.5 % [2.0 %, 3.1 %] | 6.1 % | 3.4 % [1.0 %, 6.5 %] (206) | 3.8 % [2.1 %, 5.9 %] | 4.5 % [2.6 %, 6.6 %] | 40.4 [22.4, 50.6] | 3.6 % [2.3 %, 5.1 %] |
| 2.0 | 1.94 [1.66, 2.23] | 1.96 [1.68, 2.26] | 5.8 % [3.5 %, 8.4 %] | [3.9, 8.5] % | 4.7 % [4.1 %, 5.4 %] | 8.3 % | 6.3 % [3.2 %, 10.2 %] (206) | 7.4 % [4.9 %, 10.0 %] | 8.5 % [5.7 %, 11.4 %] | 36.6 [31.6, 42.7] | 7.6 % [5.5 %, 9.8 %] |
| 5.0 | 5.48 [4.99, 6.00] | 5.50 [5.01, 6.02] | 10.1 % [7.0 %, 12.9 %] | [6.2, 13.2] % | 12.8 % [11.7 %, 13.9 %] | 16.8 % | 9.7 % [5.9 %, 13.6 %] (206) | 14.3 % [10.8 %, 17.8 %] | 16.4 % [12.7 %, 20.0 %] | 37.5 [30.0, 43.3] | 15.0 % [12.0 %, 18.3 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.605 [0.575, 0.634] | 0.584 [0.552, 0.615] | 0.588 [0.561, 0.618] | 0.005 [0.004, 0.006] | 0.0035 |
| 10 | 0.605 [0.577, 0.633] | 0.585 [0.555, 0.615] | 0.594 [0.568, 0.622] | 0.010 [0.009, 0.012] | 0.0071 |
| 20 | 0.597 [0.570, 0.623] | 0.576 [0.547, 0.604] | 0.593 [0.567, 0.619] | 0.020 [0.017, 0.024] | 0.0144 |
| 30 | 0.592 [0.567, 0.617] | 0.568 [0.540, 0.596] | 0.588 [0.562, 0.612] | 0.030 [0.026, 0.035] | 0.0222 |
| 60 | 0.597 [0.571, 0.621] | 0.568 [0.539, 0.595] | 0.588 [0.563, 0.612] | 0.059 [0.050, 0.069] | 0.0437 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.640 [0.590, 0.686] | 0.609 [0.565, 0.653] | 16.1 % [8.9 %, 23.4 %] | 22.9 % [14.2 %, 30.9 %] | 2.20 [1.69, 2.76] | 118 |
| source:xd_youtube | 0.583 [0.522, 0.639] | 0.577 [0.531, 0.617] | 4.7 % [1.6 %, 8.5 %] | 7.9 % [3.8 %, 13.1 %] | 3.02 [2.45, 3.59] | 127 |
| source:ucf | 0.538 [0.467, 0.609] | 0.517 [0.455, 0.577] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.73 [0.40, 1.13] | 115 |
| source:ubi | 0.672 [0.564, 0.763] | 0.637 [0.547, 0.717] | 1.2 % [0.0 %, 4.1 %] | 1.2 % [0.0 %, 4.1 %] | 0.10 [0.00, 0.26] | 86 |
| class:interpersonal | 0.594 [0.559, 0.625] | 0.579 [0.550, 0.606] | 3.7 % [1.8 %, 5.6 %] | 5.8 % [3.4 %, 8.3 %] | 1.94 [1.66, 2.23] | 380 |
| class:other_violent | 0.696 [0.632, 0.755] | 0.660 [0.606, 0.709] | 18.2 % [8.7 %, 29.2 %] | 24.2 % [13.6 %, 35.4 %] | 1.94 [1.66, 2.23] | 66 |
| audio:present | 0.612 [0.576, 0.643] | 0.597 [0.567, 0.625] | 8.7 % [5.4 %, 12.2 %] | 12.9 % [8.8 %, 17.2 %] | 2.72 [2.32, 3.12] | 286 |
| audio:absent | 0.613 [0.544, 0.674] | 0.583 [0.530, 0.635] | 0.6 % [0.0 %, 2.2 %] | 0.6 % [0.0 %, 2.2 %] | 0.11 [0.03, 0.23] | 160 |
| onset:official | 0.614 [0.549, 0.676] | 0.587 [0.525, 0.644] | 7.1 % [1.7 %, 14.1 %] | 10.7 % [2.8 %, 20.8 %] | 1.94 [1.66, 2.23] | 56 |
| onset:hivau | 0.615 [0.579, 0.650] | 0.602 [0.572, 0.630] | 6.9 % [4.1 %, 10.2 %] | 10.2 % [6.8 %, 13.8 %] | 1.94 [1.66, 2.23] | 304 |
| onset:ubi | 0.583 [0.485, 0.672] | 0.558 [0.480, 0.632] | 1.2 % [0.0 %, 4.1 %] | 1.2 % [0.0 %, 4.1 %] | 1.94 [1.66, 2.23] | 86 |
| event:first_segment | n/a | n/a | 5.1 % [2.7 %, 8.0 %] | 7.5 % [4.5 %, 11.0 %] | 1.94 [1.66, 2.23] | 253 |
| event:later_segment | n/a | n/a | 6.7 % [3.0 %, 10.4 %] | 9.8 % [5.6 %, 14.5 %] | 1.94 [1.66, 2.23] | 193 |

Generated 2026-10-15T13:35:10 by `anticipation/eval/va30.py` (git 00cd04bb59).
