# VA30 report: phase_c_C_imminent

Fold file `phase_c_pilot_without_original_fold1` (content sha256 `8ebdf6ec2e54`), seeds [0], headline seed 0. Test universe: 249 videos (100 violent, 149 clean negatives = 5.37 h), 100 eligible30 events (100 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 249 of 249 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 0 catalog clean negatives (0 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 5.37 of 5.37 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.788 [0.725, 0.845] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.746 [0.689, 0.798] |
| Far-window AUROC (clean, source x time matched negatives) | 0.762 [0.690, 0.826] |
| Far-window AUROC (clean, uncensored negatives) | 0.787 [0.727, 0.838] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.792 [0.727, 0.848] |
| AUROC / AP, H = 30 s (base rate 0.139) | 0.796 [0.744, 0.842] / 0.405 [0.305, 0.533] |
| Detection AUROC (in-violence vs clean negatives) | 0.838 [0.782, 0.896] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 1.88 [0.57, 3.68] | 3.04 [1.61, 4.97] | 23.0 % [15.1 %, 30.6 %] | [17.6, 38.5] % | 4.6 % [1.4 %, 8.8 %] | 9.6 % | 19.4 % [6.2 %, 34.6 %] (31) | 29.0 % [19.6 %, 39.0 %] | 30.0 % [20.6 %, 40.2 %] | 59.1 [35.7, 70.3] | 34.0 % [24.8 %, 43.5 %] |
| 2.0 | 1.88 [0.57, 3.68] | 3.04 [1.61, 4.97] | 25.0 % [17.0 %, 33.3 %] | [18.1, 43.0] % | 4.6 % [1.4 %, 8.8 %] | 9.6 % | 22.6 % [8.3 %, 39.3 %] (31) | 30.0 % [20.6 %, 40.0 %] | 32.0 % [22.8 %, 42.4 %] | 59.2 [37.0, 70.2] | 35.8 % [27.2 %, 45.8 %] |
| 5.0 | 4.89 [2.84, 7.32] | 6.73 [4.50, 9.63] | 38.0 % [28.3 %, 47.9 %] | [24.7, 58.1] % | 11.5 % [6.8 %, 16.7 %] | 20.0 % | 25.8 % [11.1 %, 44.0 %] (31) | 46.0 % [36.0 %, 55.7 %] | 49.0 % [39.0 %, 59.4 %] | 61.0 [44.2, 70.0] | 54.7 % [45.3 %, 64.0 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.802 [0.748, 0.856] | 0.773 [0.708, 0.837] | 0.805 [0.752, 0.858] | 0.124 [0.077, 0.222] | 0.0221 |
| 10 | 0.799 [0.745, 0.850] | 0.769 [0.707, 0.835] | 0.802 [0.748, 0.853] | 0.199 [0.133, 0.317] | 0.0447 |
| 20 | 0.796 [0.743, 0.846] | 0.767 [0.705, 0.830] | 0.799 [0.745, 0.849] | 0.315 [0.222, 0.448] | 0.0910 |
| 30 | 0.796 [0.744, 0.842] | 0.767 [0.702, 0.827] | 0.798 [0.746, 0.845] | 0.405 [0.305, 0.533] | 0.1386 |
| 60 | 0.768 [0.712, 0.816] | 0.729 [0.662, 0.791] | 0.771 [0.714, 0.818] | 0.561 [0.457, 0.671] | 0.2826 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.715 [0.571, 0.855] | 0.771 [0.648, 0.892] | 39.1 % [20.0 %, 63.0 %] | 47.8 % [25.0 %, 75.0 %] | 5.41 [0.00, 13.39] | 23 |
| source:xd_youtube | 0.802 [0.687, 0.895] | 0.819 [0.709, 0.906] | 22.6 % [8.1 %, 38.5 %] | 29.0 % [13.3 %, 46.9 %] | 0.00 [0.00, 0.00] | 31 |
| source:ucf | 0.739 [0.596, 0.869] | 0.747 [0.646, 0.846] | 11.8 % [2.8 %, 23.3 %] | 17.6 % [5.7 %, 32.4 %] | 1.66 [0.00, 4.00] | 34 |
| source:ubi | 0.885 [0.746, 0.987] | 0.870 [0.687, 0.986] | 41.7 % [14.3 %, 71.4 %] | 50.0 % [22.2 %, 81.8 %] | 2.01 [0.00, 5.02] | 12 |
| class:interpersonal | 0.783 [0.713, 0.846] | 0.797 [0.742, 0.849] | 22.0 % [13.3 %, 30.7 %] | 30.5 % [19.8 %, 42.1 %] | 1.88 [0.57, 3.68] | 82 |
| class:other_violent | 0.814 [0.666, 0.920] | 0.825 [0.678, 0.928] | 38.9 % [13.6 %, 64.3 %] | 38.9 % [13.6 %, 64.3 %] | 1.88 [0.57, 3.68] | 18 |
| audio:present | 0.783 [0.706, 0.853] | 0.810 [0.743, 0.870] | 25.4 % [15.0 %, 36.1 %] | 32.8 % [21.1 %, 45.8 %] | 1.73 [0.00, 4.20] | 67 |
| audio:absent | 0.791 [0.679, 0.892] | 0.767 [0.674, 0.860] | 24.2 % [10.3 %, 40.0 %] | 30.3 % [15.0 %, 46.9 %] | 2.16 [0.53, 4.24] | 33 |
| onset:official | 0.854 [0.759, 0.925] | 0.875 [0.769, 0.945] | 13.3 % [0.0 %, 30.8 %] | 26.7 % [0.0 %, 53.9 %] | 1.88 [0.57, 3.68] | 15 |
| onset:hivau | 0.757 [0.681, 0.827] | 0.779 [0.718, 0.834] | 24.7 % [15.1 %, 34.9 %] | 30.1 % [19.8 %, 41.0 %] | 1.88 [0.57, 3.68] | 73 |
| onset:ubi | 0.890 [0.796, 0.965] | 0.876 [0.697, 0.979] | 41.7 % [14.3 %, 71.4 %] | 50.0 % [22.2 %, 81.8 %] | 1.88 [0.57, 3.68] | 12 |
| event:first_segment | n/a | n/a | 25.0 % [17.0 %, 33.3 %] | 32.0 % [22.8 %, 42.4 %] | 1.88 [0.57, 3.68] | 100 |
| event:later_segment | n/a | n/a | n/a | n/a | 1.88 [0.57, 3.68] | 0 |

Generated 2026-10-16T15:43:35 by `anticipation/eval/va30.py` (git d25ab37745).
