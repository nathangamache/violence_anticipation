# VA30 report: va30

Fold file `folds_v2_without_confirmation` (content sha256 `5c5149555ca8`), seeds [0], headline seed 0. Test universe: 3599 videos (379 violent, 3220 clean negatives = 125.86 h), 446 eligible30 events (181 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 3599 of 4916 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 653 catalog clean negatives (527 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 125.86 of 242.3 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.625 [0.586, 0.661] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.621 [0.583, 0.656] |
| Far-window AUROC (clean, source x time matched negatives) | 0.574 [0.533, 0.611] |
| Far-window AUROC (clean, uncensored negatives) | 0.600 [0.565, 0.634] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.610 [0.580, 0.640] |
| AUROC / AP, H = 30 s (base rate 0.022) | 0.592 [0.564, 0.620] / 0.029 [0.025, 0.035] |
| Detection AUROC (in-violence vs clean negatives) | 0.608 [0.569, 0.647] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 0.78 [0.62, 0.95] | 0.80 [0.64, 0.96] | 1.8 % [0.7 %, 3.1 %] | [0.5, 3.2] % | 1.9 % [1.5 %, 2.3 %] | 5.5 % | 2.4 % [0.5 %, 4.4 %] (206) | 3.1 % [1.6 %, 4.8 %] | 3.1 % [1.6 %, 4.8 %] | 36.5 [24.5, 74.5] | 3.1 % [1.7 %, 4.6 %] |
| 2.0 | 1.93 [1.69, 2.18] | 1.89 [1.66, 2.13] | 3.1 % [1.6 %, 4.7 %] | [1.1, 4.8] % | 4.7 % [4.1 %, 5.3 %] | 8.2 % | 4.9 % [2.1 %, 7.9 %] (206) | 4.5 % [2.6 %, 6.4 %] | 4.9 % [2.9 %, 7.0 %] | 36.8 [27.8, 51.1] | 4.8 % [3.2 %, 6.6 %] |
| 5.0 | 5.17 [4.75, 5.58] | 5.07 [4.67, 5.46] | 7.6 % [5.1 %, 10.1 %] | [5.2, 10.3] % | 12.1 % [11.2 %, 13.0 %] | 16.1 % | 11.2 % [7.0 %, 15.3 %] (206) | 10.8 % [8.0 %, 13.7 %] | 11.7 % [8.6 %, 14.6 %] | 39.0 [30.3, 47.8] | 12.0 % [9.6 %, 14.5 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.578 [0.546, 0.607] | 0.556 [0.521, 0.589] | 0.550 [0.521, 0.579] | 0.004 [0.004, 0.005] | 0.0035 |
| 10 | 0.577 [0.546, 0.607] | 0.553 [0.520, 0.586] | 0.552 [0.524, 0.579] | 0.009 [0.008, 0.010] | 0.0071 |
| 20 | 0.583 [0.554, 0.614] | 0.554 [0.523, 0.586] | 0.555 [0.528, 0.581] | 0.018 [0.016, 0.022] | 0.0144 |
| 30 | 0.592 [0.564, 0.620] | 0.554 [0.523, 0.585] | 0.561 [0.535, 0.587] | 0.029 [0.025, 0.035] | 0.0222 |
| 60 | 0.611 [0.581, 0.641] | 0.560 [0.528, 0.593] | 0.582 [0.557, 0.607] | 0.066 [0.055, 0.090] | 0.0437 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.587 [0.555, 0.620] | 0.565 [0.531, 0.599] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 1.22 [0.82, 1.64] | 118 |
| source:xd_youtube | 0.623 [0.545, 0.693] | 0.601 [0.547, 0.659] | 3.1 % [0.7 %, 6.6 %] | 6.3 % [2.3 %, 10.7 %] | 1.51 [1.16, 1.88] | 127 |
| source:ucf | 0.669 [0.591, 0.740] | 0.599 [0.543, 0.655] | 3.5 % [0.8 %, 7.0 %] | 4.3 % [0.9 %, 8.2 %] | 2.75 [2.17, 3.40] | 115 |
| source:ubi | 0.618 [0.478, 0.748] | 0.595 [0.498, 0.681] | 7.0 % [2.2 %, 12.6 %] | 10.5 % [4.6 %, 16.5 %] | 3.16 [2.46, 3.88] | 86 |
| class:interpersonal | 0.610 [0.568, 0.653] | 0.576 [0.546, 0.608] | 3.7 % [1.9 %, 5.5 %] | 5.8 % [3.4 %, 8.1 %] | 1.93 [1.69, 2.18] | 380 |
| class:other_violent | 0.696 [0.624, 0.758] | 0.672 [0.618, 0.719] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 1.93 [1.69, 2.18] | 66 |
| audio:present | 0.623 [0.586, 0.661] | 0.591 [0.561, 0.623] | 2.1 % [0.7 %, 3.9 %] | 3.5 % [1.5 %, 5.7 %] | 1.46 [1.21, 1.73] | 286 |
| audio:absent | 0.643 [0.563, 0.723] | 0.616 [0.560, 0.669] | 5.0 % [1.9 %, 8.5 %] | 7.5 % [3.7 %, 11.3 %] | 3.03 [2.52, 3.50] | 160 |
| onset:official | 0.722 [0.644, 0.784] | 0.653 [0.594, 0.702] | 1.8 % [0.0 %, 6.0 %] | 1.8 % [0.0 %, 6.0 %] | 1.93 [1.69, 2.18] | 56 |
| onset:hivau | 0.617 [0.575, 0.660] | 0.593 [0.560, 0.624] | 2.3 % [0.7 %, 3.9 %] | 3.9 % [1.9 %, 6.2 %] | 1.93 [1.69, 2.18] | 304 |
| onset:ubi | 0.544 [0.412, 0.674] | 0.526 [0.435, 0.608] | 7.0 % [2.2 %, 12.6 %] | 10.5 % [4.6 %, 16.5 %] | 1.93 [1.69, 2.18] | 86 |
| event:first_segment | n/a | n/a | 3.2 % [1.1 %, 5.5 %] | 4.3 % [2.0 %, 7.0 %] | 1.93 [1.69, 2.18] | 253 |
| event:later_segment | n/a | n/a | 3.1 % [1.0 %, 5.7 %] | 5.7 % [2.7 %, 9.0 %] | 1.93 [1.69, 2.18] | 193 |

Generated 2026-10-15T12:42:09 by `anticipation/eval/va30.py` (git 00cd04bb59).
