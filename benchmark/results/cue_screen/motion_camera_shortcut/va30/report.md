# VA30 report: va30

Fold file `folds_v2_without_confirmation` (content sha256 `5c5149555ca8`), seeds [0], headline seed 0. Test universe: 3599 videos (379 violent, 3220 clean negatives = 125.86 h), 446 eligible30 events (181 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 3599 of 4916 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 653 catalog clean negatives (527 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 125.86 of 242.3 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.624 [0.592, 0.654] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.624 [0.592, 0.655] |
| Far-window AUROC (clean, source x time matched negatives) | 0.600 [0.564, 0.634] |
| Far-window AUROC (clean, uncensored negatives) | 0.626 [0.594, 0.656] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.592 [0.563, 0.623] |
| AUROC / AP, H = 30 s (base rate 0.022) | 0.613 [0.587, 0.639] / 0.037 [0.030, 0.045] |
| Detection AUROC (in-violence vs clean negatives) | 0.665 [0.621, 0.714] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 1.43 [1.18, 1.70] | 1.47 [1.22, 1.74] | 4.9 % [2.9 %, 7.3 %] | [3.0, 7.7] % | 3.5 % [2.9 %, 4.2 %] | 7.0 % | 7.3 % [3.8 %, 11.4 %] (206) | 7.8 % [5.3 %, 10.6 %] | 9.0 % [6.3 %, 11.9 %] | 36.0 [27.2, 46.8] | 10.3 % [7.6 %, 13.1 %] |
| 2.0 | 2.40 [2.08, 2.77] | 2.43 [2.11, 2.78] | 7.8 % [5.3 %, 10.6 %] | [4.7, 10.3] % | 5.8 % [5.1 %, 6.7 %] | 9.4 % | 9.7 % [5.7 %, 14.5 %] (206) | 11.2 % [8.2 %, 14.5 %] | 13.5 % [10.3 %, 17.1 %] | 36.3 [29.3, 44.9] | 14.6 % [11.5 %, 17.8 %] |
| 5.0 | 5.68 [5.18, 6.28] | 5.65 [5.14, 6.23] | 12.3 % [9.0 %, 15.5 %] | [9.1, 15.4] % | 13.2 % [12.2 %, 14.5 %] | 17.3 % | 14.1 % [9.2 %, 19.2 %] (206) | 17.7 % [13.8 %, 21.7 %] | 20.6 % [16.6 %, 24.6 %] | 34.5 [30.3, 39.0] | 22.3 % [18.8 %, 26.1 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.624 [0.592, 0.655] | 0.611 [0.575, 0.643] | 0.602 [0.572, 0.633] | 0.007 [0.006, 0.010] | 0.0035 |
| 10 | 0.624 [0.594, 0.654] | 0.613 [0.580, 0.644] | 0.605 [0.577, 0.635] | 0.013 [0.011, 0.017] | 0.0071 |
| 20 | 0.620 [0.594, 0.647] | 0.610 [0.582, 0.639] | 0.603 [0.577, 0.631] | 0.025 [0.021, 0.031] | 0.0144 |
| 30 | 0.613 [0.587, 0.639] | 0.599 [0.571, 0.628] | 0.596 [0.570, 0.623] | 0.037 [0.030, 0.045] | 0.0222 |
| 60 | 0.617 [0.591, 0.642] | 0.597 [0.568, 0.625] | 0.593 [0.568, 0.619] | 0.069 [0.057, 0.083] | 0.0437 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.730 [0.684, 0.774] | 0.671 [0.620, 0.717] | 22.9 % [15.2 %, 30.5 %] | 40.7 % [31.6 %, 50.0 %] | 3.58 [2.93, 4.34] | 118 |
| source:xd_youtube | 0.557 [0.489, 0.619] | 0.589 [0.539, 0.635] | 6.3 % [2.3 %, 10.8 %] | 9.4 % [4.5 %, 15.3 %] | 3.55 [2.89, 4.28] | 127 |
| source:ucf | 0.641 [0.572, 0.717] | 0.624 [0.560, 0.688] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.30 [0.09, 0.55] | 115 |
| source:ubi | 0.544 [0.439, 0.657] | 0.541 [0.438, 0.635] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 86 |
| class:interpersonal | 0.597 [0.562, 0.629] | 0.593 [0.566, 0.621] | 3.4 % [1.6 %, 5.2 %] | 7.4 % [4.7 %, 10.2 %] | 2.40 [2.08, 2.77] | 380 |
| class:other_violent | 0.755 [0.692, 0.814] | 0.710 [0.643, 0.773] | 33.3 % [22.0 %, 45.6 %] | 48.5 % [36.9 %, 60.3 %] | 2.40 [2.08, 2.77] | 66 |
| audio:present | 0.636 [0.596, 0.672] | 0.629 [0.597, 0.660] | 12.2 % [8.4 %, 16.3 %] | 21.0 % [16.2 %, 26.5 %] | 3.36 [2.90, 3.86] | 286 |
| audio:absent | 0.611 [0.536, 0.681] | 0.594 [0.532, 0.653] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.14 [0.03, 0.27] | 160 |
| onset:official | 0.651 [0.596, 0.710] | 0.582 [0.514, 0.652] | 8.9 % [2.2 %, 16.1 %] | 10.7 % [3.7 %, 19.0 %] | 2.40 [2.08, 2.77] | 56 |
| onset:hivau | 0.636 [0.597, 0.672] | 0.639 [0.607, 0.670] | 9.9 % [6.5 %, 13.4 %] | 17.8 % [13.5 %, 22.7 %] | 2.40 [2.08, 2.77] | 304 |
| onset:ubi | 0.520 [0.459, 0.590] | 0.509 [0.445, 0.572] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 2.40 [2.08, 2.77] | 86 |
| event:first_segment | n/a | n/a | 7.1 % [4.0 %, 10.4 %] | 14.2 % [10.1 %, 18.9 %] | 2.40 [2.08, 2.77] | 253 |
| event:later_segment | n/a | n/a | 8.8 % [4.8 %, 13.4 %] | 12.4 % [7.5 %, 17.7 %] | 2.40 [2.08, 2.77] | 193 |

Generated 2026-10-15T13:38:30 by `anticipation/eval/va30.py` (git 00cd04bb59).
