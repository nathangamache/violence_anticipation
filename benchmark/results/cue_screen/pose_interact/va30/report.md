# VA30 report: va30

Fold file `folds_v2_without_confirmation` (content sha256 `5c5149555ca8`), seeds [0], headline seed 0. Test universe: 3599 videos (379 violent, 3220 clean negatives = 125.86 h), 446 eligible30 events (181 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 3599 of 4916 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 653 catalog clean negatives (527 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 125.86 of 242.3 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.645 [0.605, 0.680] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.644 [0.605, 0.678] |
| Far-window AUROC (clean, source x time matched negatives) | 0.633 [0.591, 0.668] |
| Far-window AUROC (clean, uncensored negatives) | 0.639 [0.599, 0.675] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.660 [0.631, 0.689] |
| AUROC / AP, H = 30 s (base rate 0.022) | 0.646 [0.618, 0.673] / 0.039 [0.032, 0.046] |
| Detection AUROC (in-violence vs clean negatives) | 0.681 [0.621, 0.739] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 1.37 [1.16, 1.60] | 1.36 [1.16, 1.57] | 4.5 % [2.4 %, 6.5 %] | [1.9, 6.3] % | 3.4 % [2.9 %, 3.9 %] | 6.9 % | 5.3 % [2.5 %, 8.5 %] (206) | 7.2 % [4.9 %, 9.6 %] | 7.8 % [5.6 %, 10.5 %] | 32.6 [23.3, 42.0] | 9.6 % [7.5 %, 12.1 %] |
| 2.0 | 2.10 [1.82, 2.38] | 2.12 [1.85, 2.40] | 6.1 % [3.9 %, 8.3 %] | [3.8, 9.0] % | 5.1 % [4.4 %, 5.8 %] | 8.6 % | 6.8 % [3.6 %, 10.2 %] (206) | 10.3 % [7.6 %, 13.4 %] | 11.4 % [8.6 %, 14.6 %] | 31.7 [23.5, 40.8] | 13.0 % [10.4 %, 15.8 %] |
| 5.0 | 4.82 [4.34, 5.30] | 4.81 [4.36, 5.28] | 13.0 % [9.9 %, 16.2 %] | [9.1, 16.2] % | 11.3 % [10.3 %, 12.4 %] | 15.2 % | 10.7 % [6.8 %, 14.7 %] (206) | 19.7 % [16.1 %, 23.7 %] | 21.3 % [17.5 %, 25.3 %] | 34.5 [30.0, 39.1] | 23.1 % [19.6 %, 26.7 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.648 [0.617, 0.679] | 0.640 [0.606, 0.670] | 0.650 [0.623, 0.676] | 0.007 [0.005, 0.008] | 0.0035 |
| 10 | 0.649 [0.620, 0.678] | 0.640 [0.607, 0.668] | 0.655 [0.629, 0.680] | 0.012 [0.010, 0.015] | 0.0071 |
| 20 | 0.648 [0.619, 0.676] | 0.637 [0.607, 0.666] | 0.657 [0.632, 0.681] | 0.025 [0.021, 0.029] | 0.0144 |
| 30 | 0.646 [0.618, 0.673] | 0.632 [0.601, 0.662] | 0.655 [0.630, 0.679] | 0.039 [0.032, 0.046] | 0.0222 |
| 60 | 0.647 [0.617, 0.675] | 0.633 [0.599, 0.661] | 0.657 [0.633, 0.681] | 0.077 [0.064, 0.090] | 0.0437 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.629 [0.575, 0.674] | 0.644 [0.600, 0.688] | 11.0 % [5.3 %, 16.8 %] | 27.1 % [19.5 %, 35.4 %] | 3.42 [2.81, 4.12] | 118 |
| source:xd_youtube | 0.669 [0.596, 0.734] | 0.626 [0.569, 0.678] | 8.7 % [3.9 %, 14.6 %] | 11.0 % [5.7 %, 17.2 %] | 2.91 [2.37, 3.50] | 127 |
| source:ucf | 0.653 [0.576, 0.722] | 0.658 [0.601, 0.710] | 0.9 % [0.0 %, 2.9 %] | 1.7 % [0.0 %, 4.5 %] | 0.13 [0.00, 0.30] | 115 |
| source:ubi | 0.657 [0.530, 0.754] | 0.708 [0.631, 0.774] | 2.3 % [0.0 %, 5.0 %] | 3.5 % [0.0 %, 7.7 %] | 0.21 [0.00, 0.50] | 86 |
| class:interpersonal | 0.630 [0.587, 0.669] | 0.637 [0.605, 0.668] | 4.7 % [2.5 %, 6.9 %] | 8.7 % [6.1 %, 11.5 %] | 2.10 [1.82, 2.38] | 380 |
| class:other_violent | 0.718 [0.635, 0.787] | 0.692 [0.639, 0.742] | 13.6 % [5.5 %, 21.9 %] | 27.3 % [16.7 %, 37.7 %] | 2.10 [1.82, 2.38] | 66 |
| audio:present | 0.657 [0.615, 0.695] | 0.640 [0.605, 0.672] | 8.4 % [5.1 %, 11.8 %] | 16.1 % [11.8 %, 20.6 %] | 2.91 [2.53, 3.32] | 286 |
| audio:absent | 0.639 [0.567, 0.705] | 0.682 [0.634, 0.726] | 1.9 % [0.0 %, 4.0 %] | 3.1 % [0.6 %, 6.1 %] | 0.16 [0.03, 0.33] | 160 |
| onset:official | 0.660 [0.578, 0.727] | 0.630 [0.570, 0.687] | 10.7 % [3.6 %, 19.2 %] | 19.6 % [10.2 %, 30.4 %] | 2.10 [1.82, 2.38] | 56 |
| onset:hivau | 0.651 [0.605, 0.694] | 0.652 [0.617, 0.686] | 6.2 % [3.5 %, 9.2 %] | 12.2 % [8.5 %, 16.4 %] | 2.10 [1.82, 2.38] | 304 |
| onset:ubi | 0.586 [0.467, 0.685] | 0.634 [0.564, 0.697] | 2.3 % [0.0 %, 5.0 %] | 3.5 % [0.0 %, 7.7 %] | 2.10 [1.82, 2.38] | 86 |
| event:first_segment | n/a | n/a | 5.5 % [2.9 %, 8.4 %] | 11.5 % [7.9 %, 15.6 %] | 2.10 [1.82, 2.38] | 253 |
| event:later_segment | n/a | n/a | 6.7 % [3.6 %, 10.5 %] | 11.4 % [7.1 %, 16.1 %] | 2.10 [1.82, 2.38] | 193 |

Generated 2026-10-15T12:17:48 by `anticipation/eval/va30.py` (git 00cd04bb59).
