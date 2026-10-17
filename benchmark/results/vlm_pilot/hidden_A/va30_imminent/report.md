# VA30 report: phase_c_hidden_A_imminent

Fold file `phase_c_pilot_without_original_fold1` (content sha256 `8ebdf6ec2e54`), seeds [0], headline seed 0. Test universe: 249 videos (100 violent, 149 clean negatives = 5.36 h), 100 eligible30 events (100 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 249 of 249 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 0 catalog clean negatives (0 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 5.36 of 5.36 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.771 [0.690, 0.842] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.711 [0.645, 0.776] |
| Far-window AUROC (clean, source x time matched negatives) | 0.749 [0.671, 0.825] |
| Far-window AUROC (clean, uncensored negatives) | 0.779 [0.713, 0.842] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.772 [0.691, 0.843] |
| AUROC / AP, H = 30 s (base rate 0.139) | 0.735 [0.674, 0.794] / 0.233 [0.182, 0.324] |
| Detection AUROC (in-violence vs clean negatives) | 0.790 [0.721, 0.850] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 8.48 [4.76, 12.25] | 10.27 [6.67, 14.10] | 31.0 % [22.3 %, 40.4 %] | [24.3, 46.4] % | 19.1 % [11.2 %, 26.4 %] | 30.5 % | 19.4 % [6.2 %, 35.5 %] (31) | 36.0 % [26.9 %, 45.0 %] | 37.0 % [27.9 %, 46.2 %] | 63.9 [56.0, 74.8] | 39.6 % [30.5 %, 48.7 %] |
| 2.0 | 9.04 [5.25, 12.91] | 10.91 [7.33, 14.72] | 32.0 % [23.4 %, 41.1 %] | [24.3, 49.1] % | 20.2 % [12.3 %, 27.6 %] | 32.1 % | 19.4 % [6.2 %, 35.5 %] (31) | 37.0 % [28.0 %, 46.2 %] | 38.0 % [28.9 %, 47.3 %] | 65.3 [56.0, 75.5] | 40.6 % [31.6 %, 49.6 %] |
| 5.0 | 11.49 [7.27, 15.67] | 15.56 [11.37, 20.25] | 44.0 % [33.3 %, 54.6 %] | [31.7, 62.3] % | 25.0 % [16.6 %, 32.4 %] | 39.0 % | 29.0 % [13.0 %, 45.5 %] (31) | 48.0 % [37.5 %, 58.7 %] | 50.0 % [40.2 %, 60.2 %] | 73.2 [66.6, 77.3] | 56.6 % [47.1 %, 66.1 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.748 [0.687, 0.805] | 0.715 [0.642, 0.788] | 0.746 [0.685, 0.804] | 0.044 [0.032, 0.070] | 0.0221 |
| 10 | 0.744 [0.681, 0.802] | 0.712 [0.639, 0.783] | 0.744 [0.681, 0.802] | 0.086 [0.064, 0.135] | 0.0447 |
| 20 | 0.740 [0.680, 0.798] | 0.709 [0.636, 0.781] | 0.740 [0.680, 0.798] | 0.163 [0.125, 0.240] | 0.0910 |
| 30 | 0.735 [0.674, 0.794] | 0.703 [0.631, 0.775] | 0.735 [0.675, 0.795] | 0.233 [0.182, 0.324] | 0.1386 |
| 60 | 0.716 [0.652, 0.780] | 0.669 [0.591, 0.752] | 0.716 [0.652, 0.780] | 0.406 [0.334, 0.517] | 0.2826 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.649 [0.517, 0.787] | 0.641 [0.533, 0.760] | 56.5 % [35.7 %, 80.0 %] | 60.9 % [41.2 %, 81.2 %] | 25.28 [16.13, 33.86] | 23 |
| source:xd_youtube | 0.806 [0.663, 0.911] | 0.768 [0.646, 0.864] | 38.7 % [21.9 %, 57.7 %] | 48.4 % [30.0 %, 66.7 %] | 8.47 [2.47, 16.74] | 31 |
| source:ucf | 0.754 [0.594, 0.902] | 0.700 [0.593, 0.808] | 14.7 % [3.2 %, 27.6 %] | 20.6 % [8.1 %, 36.0 %] | 0.00 [0.00, 0.00] | 34 |
| source:ubi | 0.811 [0.558, 1.000] | 0.810 [0.621, 0.964] | 16.7 % [0.0 %, 42.9 %] | 16.7 % [0.0 %, 42.9 %] | 3.01 [0.00, 7.48] | 12 |
| class:interpersonal | 0.782 [0.705, 0.854] | 0.745 [0.684, 0.803] | 31.7 % [21.3 %, 42.1 %] | 37.8 % [27.6 %, 48.3 %] | 9.04 [5.25, 12.91] | 82 |
| class:other_violent | 0.714 [0.610, 0.819] | 0.727 [0.632, 0.827] | 33.3 % [12.5 %, 57.1 %] | 38.9 % [16.7 %, 61.5 %] | 9.04 [5.25, 12.91] | 18 |
| audio:present | 0.776 [0.681, 0.861] | 0.738 [0.660, 0.808] | 38.8 % [27.4 %, 50.8 %] | 46.3 % [34.7 %, 57.1 %] | 13.00 [7.52, 18.83] | 67 |
| audio:absent | 0.752 [0.591, 0.905] | 0.732 [0.630, 0.835] | 18.2 % [5.9 %, 33.3 %] | 21.2 % [8.0 %, 37.5 %] | 1.62 [0.00, 3.99] | 33 |
| onset:official | 0.747 [0.604, 0.865] | 0.787 [0.698, 0.873] | 40.0 % [16.7 %, 64.3 %] | 60.0 % [37.5 %, 83.3 %] | 9.04 [5.25, 12.91] | 15 |
| onset:hivau | 0.773 [0.691, 0.844] | 0.739 [0.672, 0.801] | 32.9 % [23.2 %, 43.1 %] | 37.0 % [26.5 %, 47.2 %] | 9.04 [5.25, 12.91] | 73 |
| onset:ubi | 0.793 [0.693, 0.884] | 0.797 [0.704, 0.880] | 16.7 % [0.0 %, 42.9 %] | 16.7 % [0.0 %, 42.9 %] | 9.04 [5.25, 12.91] | 12 |
| event:first_segment | n/a | n/a | 32.0 % [23.4 %, 41.1 %] | 38.0 % [28.9 %, 47.3 %] | 9.04 [5.25, 12.91] | 100 |
| event:later_segment | n/a | n/a | n/a | n/a | 9.04 [5.25, 12.91] | 0 |

Generated 2026-10-16T15:58:10 by `anticipation/eval/va30.py` (git d25ab37745).
