# VA30 report: phase_c_hidden_A_early

Fold file `phase_c_pilot_without_original_fold1` (content sha256 `8ebdf6ec2e54`), seeds [0], headline seed 0. Test universe: 249 videos (100 violent, 149 clean negatives = 5.36 h), 100 eligible30 events (100 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 249 of 249 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 0 catalog clean negatives (0 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 5.36 of 5.36 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.794 [0.723, 0.853] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.724 [0.663, 0.782] |
| Far-window AUROC (clean, source x time matched negatives) | 0.775 [0.695, 0.846] |
| Far-window AUROC (clean, uncensored negatives) | 0.792 [0.731, 0.846] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.794 [0.723, 0.852] |
| AUROC / AP, H = 30 s (base rate 0.139) | 0.743 [0.684, 0.798] / 0.270 [0.211, 0.360] |
| Detection AUROC (in-violence vs clean negatives) | 0.792 [0.725, 0.852] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 7.53 [4.31, 10.84] | 8.50 [5.41, 11.77] | 28.0 % [20.2 %, 36.9 %] | [21.6, 44.2] % | 17.2 % [10.2 %, 23.7 %] | 27.9 % | 16.1 % [3.8 %, 31.0 %] (31) | 32.0 % [23.5 %, 41.8 %] | 34.0 % [25.6 %, 43.6 %] | 63.5 [58.3, 72.1] | 38.7 % [29.9 %, 48.0 %] |
| 2.0 | 7.72 [4.50, 11.17] | 8.98 [6.02, 12.40] | 32.0 % [24.8 %, 41.1 %] | [23.0, 50.9] % | 17.6 % [10.6 %, 24.4 %] | 28.4 % | 25.8 % [12.5 %, 41.9 %] (31) | 36.0 % [28.1 %, 45.6 %] | 39.0 % [30.6 %, 48.3 %] | 66.9 [57.6, 74.5] | 43.4 % [34.8 %, 52.5 %] |
| 5.0 | 10.36 [6.45, 14.46] | 12.35 [8.52, 16.62] | 39.0 % [29.9 %, 48.5 %] | [27.2, 58.5] % | 22.8 % [14.9 %, 30.3 %] | 35.8 % | 32.3 % [17.2 %, 50.0 %] (31) | 43.0 % [33.3 %, 52.4 %] | 47.0 % [37.6 %, 56.2 %] | 68.3 [59.2, 74.3] | 51.9 % [42.9 %, 60.6 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.752 [0.692, 0.808] | 0.725 [0.652, 0.792] | 0.749 [0.689, 0.804] | 0.060 [0.039, 0.098] | 0.0221 |
| 10 | 0.750 [0.689, 0.804] | 0.722 [0.649, 0.788] | 0.748 [0.687, 0.802] | 0.107 [0.077, 0.157] | 0.0447 |
| 20 | 0.748 [0.689, 0.801] | 0.720 [0.650, 0.784] | 0.747 [0.688, 0.800] | 0.195 [0.148, 0.274] | 0.0910 |
| 30 | 0.743 [0.684, 0.798] | 0.714 [0.644, 0.779] | 0.741 [0.683, 0.797] | 0.270 [0.211, 0.360] | 0.1386 |
| 60 | 0.728 [0.668, 0.785] | 0.683 [0.605, 0.759] | 0.727 [0.667, 0.784] | 0.446 [0.373, 0.547] | 0.2826 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.657 [0.561, 0.775] | 0.648 [0.559, 0.752] | 47.8 % [33.3 %, 70.6 %] | 52.2 % [37.0 %, 75.0 %] | 23.47 [12.99, 32.62] | 23 |
| source:xd_youtube | 0.837 [0.714, 0.933] | 0.790 [0.689, 0.875] | 45.2 % [28.6 %, 62.1 %] | 54.8 % [37.9 %, 71.9 %] | 4.98 [1.19, 10.13] | 31 |
| source:ucf | 0.760 [0.593, 0.899] | 0.691 [0.570, 0.805] | 17.6 % [6.1 %, 32.4 %] | 23.5 % [10.3 %, 38.7 %] | 1.67 [0.00, 4.18] | 34 |
| source:ubi | 0.839 [0.630, 0.992] | 0.822 [0.657, 0.955] | 8.3 % [0.0 %, 28.6 %] | 16.7 % [0.0 %, 41.7 %] | 3.01 [0.00, 7.48] | 12 |
| class:interpersonal | 0.800 [0.728, 0.860] | 0.751 [0.691, 0.807] | 31.7 % [23.0 %, 41.7 %] | 40.2 % [30.9 %, 50.7 %] | 7.72 [4.50, 11.17] | 82 |
| class:other_violent | 0.765 [0.668, 0.854] | 0.757 [0.676, 0.848] | 33.3 % [13.6 %, 60.0 %] | 33.3 % [13.6 %, 60.0 %] | 7.72 [4.50, 11.17] | 18 |
| audio:present | 0.786 [0.701, 0.857] | 0.735 [0.664, 0.801] | 40.3 % [30.4 %, 50.9 %] | 46.3 % [35.8 %, 58.9 %] | 10.40 [5.65, 15.76] | 67 |
| audio:absent | 0.791 [0.637, 0.919] | 0.751 [0.643, 0.847] | 15.2 % [3.2 %, 28.1 %] | 24.2 % [9.7 %, 40.7 %] | 2.70 [0.48, 5.57] | 33 |
| onset:official | 0.739 [0.621, 0.853] | 0.777 [0.692, 0.862] | 33.3 % [10.0 %, 61.5 %] | 53.3 % [26.7 %, 83.3 %] | 7.72 [4.50, 11.17] | 15 |
| onset:hivau | 0.800 [0.729, 0.861] | 0.754 [0.691, 0.811] | 35.6 % [26.0 %, 45.9 %] | 39.7 % [29.8 %, 49.4 %] | 7.72 [4.50, 11.17] | 73 |
| onset:ubi | 0.825 [0.737, 0.905] | 0.818 [0.729, 0.899] | 8.3 % [0.0 %, 28.6 %] | 16.7 % [0.0 %, 41.7 %] | 7.72 [4.50, 11.17] | 12 |
| event:first_segment | n/a | n/a | 32.0 % [24.8 %, 41.1 %] | 39.0 % [30.6 %, 48.3 %] | 7.72 [4.50, 11.17] | 100 |
| event:later_segment | n/a | n/a | n/a | n/a | 7.72 [4.50, 11.17] | 0 |

Generated 2026-10-16T15:58:04 by `anticipation/eval/va30.py` (git d25ab37745).
