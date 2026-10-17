# VA30 report: phase_c_hidden_C_early

Fold file `phase_c_pilot_without_original_fold1` (content sha256 `8ebdf6ec2e54`), seeds [0], headline seed 0. Test universe: 249 videos (100 violent, 149 clean negatives = 5.36 h), 100 eligible30 events (100 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 249 of 249 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 0 catalog clean negatives (0 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 5.36 of 5.36 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.840 [0.775, 0.896] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.765 [0.706, 0.822] |
| Far-window AUROC (clean, source x time matched negatives) | 0.808 [0.735, 0.876] |
| Far-window AUROC (clean, uncensored negatives) | 0.844 [0.788, 0.894] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.838 [0.771, 0.896] |
| AUROC / AP, H = 30 s (base rate 0.139) | 0.780 [0.722, 0.837] / 0.299 [0.231, 0.429] |
| Detection AUROC (in-violence vs clean negatives) | 0.838 [0.777, 0.893] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 2.26 [0.93, 3.79] | 3.21 [1.82, 4.78] | 24.0 % [15.8 %, 33.3 %] | [16.0, 44.2] % | 5.5 % [2.3 %, 9.0 %] | 10.9 % | 25.8 % [11.1 %, 42.9 %] (31) | 24.0 % [15.8 %, 33.3 %] | 24.0 % [15.8 %, 33.3 %] | 68.1 [50.9, 82.4] | 26.4 % [18.2 %, 35.1 %] |
| 2.0 | 2.26 [0.93, 3.79] | 3.53 [2.02, 5.30] | 24.0 % [15.8 %, 33.3 %] | [16.3, 46.8] % | 5.5 % [2.3 %, 9.0 %] | 10.9 % | 25.8 % [11.1 %, 42.9 %] (31) | 24.0 % [15.8 %, 33.3 %] | 24.0 % [15.8 %, 33.3 %] | 72.7 [55.3, 80.3] | 27.4 % [19.3 %, 35.8 %] |
| 5.0 | 4.33 [2.42, 6.57] | 7.38 [5.08, 9.97] | 40.0 % [30.6 %, 49.5 %] | [22.8, 53.6] % | 10.3 % [5.9 %, 15.1 %] | 18.1 % | 22.6 % [9.1 %, 38.7 %] (31) | 43.0 % [34.0 %, 53.1 %] | 45.0 % [35.7 %, 54.9 %] | 73.8 [64.2, 81.0] | 57.5 % [48.5 %, 67.4 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.789 [0.731, 0.843] | 0.753 [0.680, 0.822] | 0.787 [0.728, 0.843] | 0.063 [0.045, 0.119] | 0.0221 |
| 10 | 0.789 [0.732, 0.843] | 0.752 [0.681, 0.821] | 0.786 [0.729, 0.843] | 0.120 [0.087, 0.204] | 0.0447 |
| 20 | 0.785 [0.728, 0.841] | 0.747 [0.675, 0.816] | 0.782 [0.723, 0.839] | 0.216 [0.162, 0.330] | 0.0910 |
| 30 | 0.780 [0.722, 0.837] | 0.743 [0.671, 0.813] | 0.777 [0.718, 0.837] | 0.299 [0.231, 0.429] | 0.1386 |
| 60 | 0.765 [0.706, 0.824] | 0.719 [0.644, 0.796] | 0.763 [0.700, 0.824] | 0.488 [0.403, 0.616] | 0.2826 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.762 [0.595, 0.910] | 0.739 [0.593, 0.880] | 17.4 % [0.0 %, 38.9 %] | 17.4 % [0.0 %, 38.9 %] | 4.51 [0.00, 9.69] | 23 |
| source:xd_youtube | 0.905 [0.830, 0.963] | 0.853 [0.779, 0.922] | 25.8 % [11.4 %, 43.6 %] | 25.8 % [11.4 %, 43.6 %] | 1.00 [0.00, 2.89] | 31 |
| source:ucf | 0.779 [0.642, 0.896] | 0.683 [0.577, 0.789] | 26.5 % [11.5 %, 41.2 %] | 26.5 % [11.5 %, 41.2 %] | 3.34 [0.81, 6.46] | 34 |
| source:ubi | 0.826 [0.586, 0.997] | 0.838 [0.657, 0.973] | 25.0 % [0.0 %, 50.0 %] | 25.0 % [0.0 %, 50.0 %] | 1.00 [0.00, 2.96] | 12 |
| class:interpersonal | 0.840 [0.774, 0.899] | 0.782 [0.722, 0.837] | 24.4 % [15.4 %, 34.0 %] | 24.4 % [15.4 %, 34.0 %] | 2.26 [0.93, 3.79] | 82 |
| class:other_violent | 0.840 [0.741, 0.925] | 0.838 [0.736, 0.921] | 22.2 % [5.0 %, 46.2 %] | 22.2 % [5.0 %, 46.2 %] | 2.26 [0.93, 3.79] | 18 |
| audio:present | 0.855 [0.782, 0.914] | 0.797 [0.722, 0.861] | 23.9 % [14.1 %, 35.0 %] | 23.9 % [14.1 %, 35.0 %] | 2.02 [0.37, 4.07] | 67 |
| audio:absent | 0.801 [0.659, 0.922] | 0.755 [0.653, 0.850] | 24.2 % [10.7 %, 40.0 %] | 24.2 % [10.7 %, 40.0 %] | 2.70 [0.73, 4.86] | 33 |
| onset:official | 0.813 [0.677, 0.920] | 0.813 [0.694, 0.914] | 40.0 % [15.4 %, 64.7 %] | 40.0 % [15.4 %, 64.7 %] | 2.26 [0.93, 3.79] | 15 |
| onset:hivau | 0.840 [0.773, 0.902] | 0.790 [0.730, 0.847] | 20.5 % [11.9 %, 30.9 %] | 20.5 % [11.9 %, 30.9 %] | 2.26 [0.93, 3.79] | 73 |
| onset:ubi | 0.873 [0.794, 0.942] | 0.872 [0.798, 0.939] | 25.0 % [0.0 %, 50.0 %] | 25.0 % [0.0 %, 50.0 %] | 2.26 [0.93, 3.79] | 12 |
| event:first_segment | n/a | n/a | 24.0 % [15.8 %, 33.3 %] | 24.0 % [15.8 %, 33.3 %] | 2.26 [0.93, 3.79] | 100 |
| event:later_segment | n/a | n/a | n/a | n/a | 2.26 [0.93, 3.79] | 0 |

Generated 2026-10-16T16:10:38 by `anticipation/eval/va30.py` (git d25ab37745).
