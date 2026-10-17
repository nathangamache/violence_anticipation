# VA30 report: phase_c_B_imminent

Fold file `phase_c_pilot_without_original_fold1` (content sha256 `8ebdf6ec2e54`), seeds [0], headline seed 0. Test universe: 249 videos (100 violent, 149 clean negatives = 5.37 h), 100 eligible30 events (100 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 249 of 249 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 0 catalog clean negatives (0 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 5.37 of 5.37 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.775 [0.711, 0.832] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.729 [0.668, 0.784] |
| Far-window AUROC (clean, source x time matched negatives) | 0.756 [0.690, 0.813] |
| Far-window AUROC (clean, uncensored negatives) | 0.767 [0.706, 0.818] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.778 [0.714, 0.834] |
| AUROC / AP, H = 30 s (base rate 0.139) | 0.796 [0.742, 0.844] / 0.424 [0.333, 0.540] |
| Detection AUROC (in-violence vs clean negatives) | 0.826 [0.763, 0.877] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 3.76 [1.57, 6.51] | 5.13 [2.72, 8.03] | 28.0 % [18.9 %, 37.3 %] | [21.2, 41.6] % | 9.0 % [3.8 %, 15.0 %] | 16.1 % | 29.0 % [13.3 %, 46.5 %] (31) | 34.0 % [23.8 %, 43.4 %] | 37.0 % [26.7 %, 47.0 %] | 63.1 [37.5, 74.8] | 34.9 % [25.0 %, 44.5 %] |
| 2.0 | 3.95 [1.74, 6.67] | 5.45 [3.01, 8.30] | 34.0 % [24.8 %, 43.8 %] | [21.3, 46.5] % | 9.4 % [4.3 %, 15.4 %] | 16.8 % | 35.5 % [18.5 %, 53.6 %] (31) | 42.0 % [31.8 %, 52.0 %] | 46.0 % [36.1 %, 56.1 %] | 58.0 [38.7, 72.1] | 43.4 % [34.2 %, 52.9 %] |
| 5.0 | 5.45 [2.56, 8.86] | 7.69 [4.57, 11.49] | 39.0 % [29.9 %, 48.2 %] | [29.9, 53.4] % | 12.7 % [6.2 %, 19.9 %] | 22.0 % | 41.9 % [24.2 %, 59.3 %] (31) | 45.0 % [35.1 %, 54.5 %] | 50.0 % [40.2 %, 59.8 %] | 67.1 [51.0, 76.3] | 50.0 % [40.8 %, 59.4 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.798 [0.743, 0.849] | 0.781 [0.720, 0.837] | 0.799 [0.744, 0.849] | 0.146 [0.096, 0.238] | 0.0221 |
| 10 | 0.798 [0.745, 0.846] | 0.780 [0.720, 0.837] | 0.800 [0.746, 0.847] | 0.233 [0.161, 0.341] | 0.0447 |
| 20 | 0.798 [0.745, 0.847] | 0.777 [0.717, 0.833] | 0.799 [0.746, 0.847] | 0.341 [0.256, 0.457] | 0.0910 |
| 30 | 0.796 [0.742, 0.844] | 0.773 [0.714, 0.826] | 0.798 [0.746, 0.845] | 0.424 [0.333, 0.540] | 0.1386 |
| 60 | 0.760 [0.707, 0.811] | 0.725 [0.664, 0.783] | 0.762 [0.710, 0.814] | 0.578 [0.488, 0.679] | 0.2826 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.782 [0.687, 0.884] | 0.816 [0.743, 0.903] | 56.5 % [37.5 %, 76.2 %] | 69.6 % [47.4 %, 91.9 %] | 10.83 [4.58, 19.45] | 23 |
| source:xd_youtube | 0.721 [0.599, 0.834] | 0.777 [0.670, 0.867] | 29.0 % [14.7 %, 46.9 %] | 54.8 % [37.8 %, 73.3 %] | 3.48 [0.00, 9.00] | 31 |
| source:ucf | 0.781 [0.639, 0.899] | 0.752 [0.656, 0.839] | 17.6 % [6.1 %, 30.8 %] | 20.6 % [7.7 %, 34.5 %] | 0.00 [0.00, 0.00] | 34 |
| source:ubi | 0.827 [0.654, 0.986] | 0.898 [0.777, 0.986] | 50.0 % [22.2 %, 81.8 %] | 50.0 % [22.2 %, 81.8 %] | 2.01 [0.00, 6.19] | 12 |
| class:interpersonal | 0.778 [0.708, 0.840] | 0.811 [0.755, 0.863] | 34.1 % [24.1 %, 44.4 %] | 46.3 % [35.4 %, 57.1 %] | 3.95 [1.74, 6.67] | 82 |
| class:other_violent | 0.760 [0.641, 0.862] | 0.759 [0.614, 0.876] | 33.3 % [12.5 %, 55.6 %] | 44.4 % [18.7 %, 69.6 %] | 3.95 [1.74, 6.67] | 18 |
| audio:present | 0.769 [0.698, 0.834] | 0.797 [0.732, 0.852] | 35.8 % [24.5 %, 48.2 %] | 53.7 % [40.9 %, 66.7 %] | 5.49 [2.41, 9.84] | 67 |
| audio:absent | 0.771 [0.648, 0.887] | 0.799 [0.711, 0.884] | 30.3 % [15.4 %, 48.2 %] | 30.3 % [15.4 %, 48.2 %] | 1.08 [0.00, 3.40] | 33 |
| onset:official | 0.804 [0.697, 0.899] | 0.889 [0.825, 0.942] | 46.7 % [21.7 %, 71.4 %] | 60.0 % [28.6 %, 84.6 %] | 3.95 [1.74, 6.67] | 15 |
| onset:hivau | 0.748 [0.673, 0.814] | 0.768 [0.706, 0.824] | 28.8 % [18.7 %, 39.1 %] | 42.5 % [31.8 %, 53.8 %] | 3.95 [1.74, 6.67] | 73 |
| onset:ubi | 0.902 [0.791, 0.974] | 0.947 [0.904, 0.979] | 50.0 % [22.2 %, 81.8 %] | 50.0 % [22.2 %, 81.8 %] | 3.95 [1.74, 6.67] | 12 |
| event:first_segment | n/a | n/a | 34.0 % [24.8 %, 43.8 %] | 46.0 % [36.1 %, 56.1 %] | 3.95 [1.74, 6.67] | 100 |
| event:later_segment | n/a | n/a | n/a | n/a | 3.95 [1.74, 6.67] | 0 |

Generated 2026-10-16T15:43:23 by `anticipation/eval/va30.py` (git d25ab37745).
