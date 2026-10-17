# VA30 report: phase_c_A_early

Fold file `phase_c_pilot_without_original_fold1` (content sha256 `8ebdf6ec2e54`), seeds [0], headline seed 0. Test universe: 249 videos (100 violent, 149 clean negatives = 5.37 h), 100 eligible30 events (100 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 249 of 249 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 0 catalog clean negatives (0 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 5.37 of 5.37 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.749 [0.690, 0.805] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.700 [0.642, 0.755] |
| Far-window AUROC (clean, source x time matched negatives) | 0.740 [0.677, 0.804] |
| Far-window AUROC (clean, uncensored negatives) | 0.756 [0.699, 0.809] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.751 [0.692, 0.806] |
| AUROC / AP, H = 30 s (base rate 0.139) | 0.729 [0.676, 0.785] / 0.315 [0.244, 0.421] |
| Detection AUROC (in-violence vs clean negatives) | 0.779 [0.717, 0.841] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 6.39 [3.82, 9.33] | 8.33 [5.54, 11.49] | 46.0 % [36.7 %, 55.8 %] | [37.9, 59.4] % | 14.8 % [9.1 %, 20.8 %] | 24.7 % | 51.6 % [34.6 %, 70.0 %] (31) | 49.0 % [39.2 %, 58.8 %] | 51.0 % [41.6 %, 60.6 %] | 74.2 [64.0, 83.3] | 51.9 % [42.6 %, 61.1 %] |
| 2.0 | 6.96 [4.35, 10.06] | 9.29 [6.34, 12.68] | 47.0 % [37.3 %, 57.0 %] | [38.5, 59.8] % | 16.0 % [10.3 %, 22.2 %] | 26.3 % | 51.6 % [34.6 %, 70.0 %] (31) | 51.0 % [41.6 %, 60.4 %] | 53.0 % [43.7 %, 62.5 %] | 73.2 [64.0, 82.4] | 53.8 % [44.7 %, 62.8 %] |
| 5.0 | 10.34 [7.08, 14.23] | 13.46 [10.05, 17.49] | 53.0 % [42.1 %, 62.9 %] | [41.5, 64.0] % | 22.8 % [16.2 %, 29.9 %] | 35.8 % | 38.7 % [22.6 %, 55.3 %] (31) | 57.0 % [46.5 %, 66.4 %] | 59.0 % [48.9 %, 68.5 %] | 74.5 [68.5, 78.0] | 66.0 % [57.1 %, 74.5 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.741 [0.683, 0.796] | 0.723 [0.651, 0.788] | 0.737 [0.679, 0.791] | 0.073 [0.049, 0.127] | 0.0221 |
| 10 | 0.729 [0.671, 0.787] | 0.711 [0.640, 0.780] | 0.728 [0.671, 0.785] | 0.131 [0.091, 0.208] | 0.0447 |
| 20 | 0.731 [0.675, 0.790] | 0.715 [0.648, 0.783] | 0.731 [0.676, 0.789] | 0.232 [0.173, 0.332] | 0.0910 |
| 30 | 0.729 [0.676, 0.785] | 0.713 [0.647, 0.779] | 0.730 [0.677, 0.785] | 0.315 [0.244, 0.421] | 0.1386 |
| 60 | 0.708 [0.653, 0.763] | 0.684 [0.619, 0.744] | 0.709 [0.654, 0.764] | 0.484 [0.405, 0.584] | 0.2826 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.728 [0.643, 0.818] | 0.703 [0.612, 0.810] | 65.2 % [44.4 %, 90.9 %] | 73.9 % [57.1 %, 92.3 %] | 11.73 [6.29, 19.59] | 23 |
| source:xd_youtube | 0.723 [0.615, 0.818] | 0.723 [0.627, 0.806] | 41.9 % [25.7 %, 59.5 %] | 54.8 % [38.2 %, 70.4 %] | 8.46 [3.05, 15.97] | 31 |
| source:ucf | 0.708 [0.562, 0.842] | 0.657 [0.540, 0.780] | 26.5 % [12.1 %, 42.3 %] | 26.5 % [12.1 %, 42.3 %] | 3.33 [0.00, 7.34] | 34 |
| source:ubi | 0.897 [0.782, 0.995] | 0.917 [0.831, 0.990] | 83.3 % [57.1 %, 100.0 %] | 83.3 % [57.1 %, 100.0 %] | 3.01 [0.00, 7.11] | 12 |
| class:interpersonal | 0.760 [0.699, 0.818] | 0.751 [0.694, 0.809] | 43.9 % [32.9 %, 54.8 %] | 50.0 % [39.2 %, 61.1 %] | 6.96 [4.35, 10.06] | 82 |
| class:other_violent | 0.691 [0.527, 0.819] | 0.661 [0.494, 0.794] | 61.1 % [36.4 %, 86.7 %] | 66.7 % [44.4 %, 88.2 %] | 6.96 [4.35, 10.06] | 18 |
| audio:present | 0.729 [0.660, 0.799] | 0.718 [0.651, 0.784] | 44.8 % [33.8 %, 57.1 %] | 53.7 % [41.7 %, 64.7 %] | 9.24 [5.37, 14.14] | 67 |
| audio:absent | 0.791 [0.672, 0.902] | 0.751 [0.650, 0.852] | 51.5 % [34.2 %, 69.2 %] | 51.5 % [34.2 %, 69.2 %] | 2.70 [0.55, 5.29] | 33 |
| onset:official | 0.676 [0.501, 0.825] | 0.740 [0.579, 0.877] | 66.7 % [40.0 %, 88.9 %] | 73.3 % [50.0 %, 93.8 %] | 6.96 [4.35, 10.06] | 15 |
| onset:hivau | 0.734 [0.669, 0.795] | 0.708 [0.647, 0.769] | 37.0 % [26.8 %, 48.6 %] | 43.8 % [33.3 %, 54.5 %] | 6.96 [4.35, 10.06] | 73 |
| onset:ubi | 0.926 [0.874, 0.968] | 0.942 [0.905, 0.973] | 83.3 % [57.1 %, 100.0 %] | 83.3 % [57.1 %, 100.0 %] | 6.96 [4.35, 10.06] | 12 |
| event:first_segment | n/a | n/a | 47.0 % [37.3 %, 57.0 %] | 53.0 % [43.7 %, 62.5 %] | 6.96 [4.35, 10.06] | 100 |
| event:later_segment | n/a | n/a | n/a | n/a | 6.96 [4.35, 10.06] | 0 |

Generated 2026-10-16T15:43:04 by `anticipation/eval/va30.py` (git d25ab37745).
