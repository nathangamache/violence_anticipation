# VA30 report: phase_c_B_early

Fold file `phase_c_pilot_without_original_fold1` (content sha256 `8ebdf6ec2e54`), seeds [0], headline seed 0. Test universe: 249 videos (100 violent, 149 clean negatives = 5.37 h), 100 eligible30 events (100 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 249 of 249 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 0 catalog clean negatives (0 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 5.37 of 5.37 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.722 [0.660, 0.785] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.698 [0.636, 0.760] |
| Far-window AUROC (clean, source x time matched negatives) | 0.722 [0.660, 0.786] |
| Far-window AUROC (clean, uncensored negatives) | 0.725 [0.664, 0.781] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.726 [0.665, 0.787] |
| AUROC / AP, H = 30 s (base rate 0.139) | 0.686 [0.623, 0.746] / 0.376 [0.283, 0.485] |
| Detection AUROC (in-violence vs clean negatives) | 0.744 [0.674, 0.806] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 1.13 [0.37, 2.07] | 2.08 [0.86, 3.79] | 28.0 % [18.3 %, 38.5 %] | [20.2, 39.6] % | 2.8 % [0.9 %, 5.0 %] | 6.9 % | 25.8 % [10.7 %, 44.0 %] (31) | 30.0 % [20.4 %, 40.2 %] | 32.0 % [21.8 %, 42.3 %] | 70.9 [50.2, 78.0] | 30.2 % [20.6 %, 39.6 %] |
| 2.0 | 1.13 [0.37, 2.07] | 2.08 [0.86, 3.79] | 29.0 % [19.4 %, 38.8 %] | [20.2, 45.8] % | 2.8 % [0.9 %, 5.0 %] | 6.9 % | 25.8 % [10.7 %, 44.0 %] (31) | 31.0 % [21.4 %, 41.2 %] | 33.0 % [22.7 %, 43.4 %] | 68.6 [51.1, 77.3] | 31.1 % [21.3 %, 40.6 %] |
| 5.0 | 4.14 [1.44, 7.46] | 5.61 [2.94, 8.70] | 34.0 % [24.7 %, 43.9 %] | [25.1, 57.5] % | 9.8 % [3.5 %, 17.0 %] | 17.4 % | 38.7 % [21.2 %, 57.1 %] (31) | 38.0 % [28.0 %, 47.5 %] | 44.0 % [34.7 %, 54.7 %] | 68.5 [47.8, 76.8] | 41.5 % [32.6 %, 51.4 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.691 [0.619, 0.752] | 0.674 [0.600, 0.742] | 0.695 [0.623, 0.755] | 0.132 [0.076, 0.229] | 0.0221 |
| 10 | 0.692 [0.621, 0.754] | 0.675 [0.601, 0.744] | 0.697 [0.627, 0.757] | 0.201 [0.130, 0.306] | 0.0447 |
| 20 | 0.676 [0.609, 0.744] | 0.656 [0.581, 0.727] | 0.681 [0.615, 0.747] | 0.290 [0.205, 0.404] | 0.0910 |
| 30 | 0.686 [0.623, 0.746] | 0.668 [0.597, 0.736] | 0.691 [0.629, 0.752] | 0.376 [0.283, 0.485] | 0.1386 |
| 60 | 0.689 [0.630, 0.747] | 0.668 [0.605, 0.732] | 0.694 [0.638, 0.750] | 0.541 [0.440, 0.638] | 0.2826 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.768 [0.676, 0.888] | 0.760 [0.657, 0.890] | 52.2 % [26.7 %, 76.9 %] | 52.2 % [26.7 %, 76.9 %] | 4.51 [1.41, 8.27] | 23 |
| source:xd_youtube | 0.714 [0.604, 0.811] | 0.716 [0.606, 0.830] | 29.0 % [14.3 %, 46.7 %] | 41.9 % [25.0 %, 60.0 %] | 0.50 [0.00, 1.72] | 31 |
| source:ucf | 0.647 [0.505, 0.798] | 0.634 [0.509, 0.754] | 14.7 % [3.7 %, 27.6 %] | 14.7 % [3.7 %, 27.6 %] | 0.00 [0.00, 0.00] | 34 |
| source:ubi | 0.784 [0.575, 0.975] | 0.586 [0.406, 0.768] | 25.0 % [0.0 %, 53.3 %] | 25.0 % [0.0 %, 53.3 %] | 0.00 [0.00, 0.00] | 12 |
| class:interpersonal | 0.718 [0.650, 0.786] | 0.684 [0.615, 0.753] | 25.6 % [15.8 %, 35.7 %] | 30.5 % [20.2 %, 41.3 %] | 1.13 [0.37, 2.07] | 82 |
| class:other_violent | 0.742 [0.601, 0.860] | 0.719 [0.563, 0.857] | 44.4 % [17.6 %, 72.2 %] | 44.4 % [17.6 %, 72.2 %] | 1.13 [0.37, 2.07] | 18 |
| audio:present | 0.741 [0.673, 0.809] | 0.723 [0.650, 0.799] | 34.3 % [21.9 %, 46.8 %] | 40.3 % [27.1 %, 52.9 %] | 1.73 [0.58, 3.21] | 67 |
| audio:absent | 0.677 [0.541, 0.813] | 0.611 [0.499, 0.729] | 18.2 % [7.4 %, 34.2 %] | 18.2 % [7.4 %, 34.2 %] | 0.00 [0.00, 0.00] | 33 |
| onset:official | 0.747 [0.600, 0.875] | 0.820 [0.696, 0.926] | 46.7 % [20.0 %, 75.0 %] | 46.7 % [20.0 %, 75.0 %] | 1.13 [0.37, 2.07] | 15 |
| onset:hivau | 0.700 [0.630, 0.770] | 0.681 [0.606, 0.756] | 26.0 % [15.7 %, 36.7 %] | 31.5 % [20.5 %, 42.3 %] | 1.13 [0.37, 2.07] | 73 |
| onset:ubi | 0.821 [0.628, 0.964] | 0.603 [0.421, 0.782] | 25.0 % [0.0 %, 53.3 %] | 25.0 % [0.0 %, 53.3 %] | 1.13 [0.37, 2.07] | 12 |
| event:first_segment | n/a | n/a | 29.0 % [19.4 %, 38.8 %] | 33.0 % [22.7 %, 43.4 %] | 1.13 [0.37, 2.07] | 100 |
| event:later_segment | n/a | n/a | n/a | n/a | 1.13 [0.37, 2.07] | 0 |

Generated 2026-10-16T15:43:17 by `anticipation/eval/va30.py` (git d25ab37745).
