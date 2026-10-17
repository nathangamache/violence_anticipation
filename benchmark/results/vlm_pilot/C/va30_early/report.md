# VA30 report: phase_c_C_early

Fold file `phase_c_pilot_without_original_fold1` (content sha256 `8ebdf6ec2e54`), seeds [0], headline seed 0. Test universe: 249 videos (100 violent, 149 clean negatives = 5.37 h), 100 eligible30 events (100 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 249 of 249 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 0 catalog clean negatives (0 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 5.37 of 5.37 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.804 [0.741, 0.862] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.766 [0.708, 0.824] |
| Far-window AUROC (clean, source x time matched negatives) | 0.780 [0.711, 0.848] |
| Far-window AUROC (clean, uncensored negatives) | 0.798 [0.738, 0.853] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.806 [0.744, 0.864] |
| AUROC / AP, H = 30 s (base rate 0.139) | 0.799 [0.749, 0.849] / 0.433 [0.327, 0.562] |
| Detection AUROC (in-violence vs clean negatives) | 0.832 [0.772, 0.892] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 3.01 [1.28, 5.35] | 4.17 [2.36, 6.40] | 18.0 % [10.8 %, 25.2 %] | [12.4, 36.2] % | 7.2 % [3.2 %, 12.5 %] | 13.5 % | 16.1 % [3.8 %, 29.6 %] (31) | 22.0 % [13.3 %, 30.4 %] | 24.0 % [16.0 %, 32.8 %] | 59.6 [36.3, 70.0] | 27.4 % [18.9 %, 36.4 %] |
| 2.0 | 3.01 [1.28, 5.35] | 4.17 [2.36, 6.40] | 18.0 % [10.8 %, 25.2 %] | [12.9, 38.3] % | 7.2 % [3.2 %, 12.5 %] | 13.5 % | 16.1 % [3.8 %, 29.6 %] (31) | 22.0 % [13.3 %, 30.4 %] | 24.0 % [16.0 %, 32.8 %] | 59.6 [36.3, 70.0] | 27.4 % [18.9 %, 36.4 %] |
| 5.0 | 5.45 [3.33, 8.07] | 7.85 [5.43, 10.77] | 34.0 % [23.8 %, 44.2 %] | [18.5, 50.5] % | 12.7 % [8.0 %, 18.3 %] | 22.0 % | 16.1 % [3.8 %, 31.0 %] (31) | 38.0 % [27.7 %, 48.2 %] | 40.0 % [29.5 %, 50.0 %] | 61.6 [47.6, 71.4] | 50.9 % [41.0 %, 60.2 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.806 [0.748, 0.864] | 0.781 [0.715, 0.846] | 0.809 [0.751, 0.865] | 0.139 [0.081, 0.266] | 0.0221 |
| 10 | 0.796 [0.742, 0.852] | 0.770 [0.705, 0.833] | 0.799 [0.744, 0.854] | 0.225 [0.143, 0.360] | 0.0447 |
| 20 | 0.796 [0.745, 0.847] | 0.768 [0.707, 0.828] | 0.799 [0.746, 0.849] | 0.345 [0.248, 0.478] | 0.0910 |
| 30 | 0.799 [0.749, 0.849] | 0.769 [0.709, 0.829] | 0.801 [0.751, 0.852] | 0.433 [0.327, 0.562] | 0.1386 |
| 60 | 0.780 [0.728, 0.835] | 0.743 [0.681, 0.808] | 0.783 [0.730, 0.837] | 0.595 [0.490, 0.708] | 0.2826 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.767 [0.650, 0.907] | 0.755 [0.635, 0.888] | 21.7 % [5.0 %, 41.2 %] | 26.1 % [5.0 %, 53.0 %] | 11.73 [4.22, 22.41] | 23 |
| source:xd_youtube | 0.839 [0.733, 0.928] | 0.839 [0.746, 0.921] | 22.6 % [7.7 %, 37.9 %] | 32.3 % [15.6 %, 50.0 %] | 1.00 [0.00, 2.79] | 31 |
| source:ucf | 0.733 [0.584, 0.869] | 0.747 [0.644, 0.842] | 8.8 % [0.0 %, 18.8 %] | 11.8 % [2.8 %, 22.9 %] | 0.00 [0.00, 0.00] | 34 |
| source:ubi | 0.937 [0.847, 0.993] | 0.932 [0.852, 0.990] | 25.0 % [0.0 %, 50.0 %] | 33.3 % [8.3 %, 60.0 %] | 1.00 [0.00, 3.66] | 12 |
| class:interpersonal | 0.805 [0.741, 0.871] | 0.811 [0.756, 0.866] | 17.1 % [8.9 %, 25.6 %] | 24.4 % [15.0 %, 34.2 %] | 3.01 [1.28, 5.35] | 82 |
| class:other_violent | 0.795 [0.653, 0.914] | 0.764 [0.601, 0.904] | 22.2 % [5.3 %, 47.1 %] | 22.2 % [5.3 %, 47.1 %] | 3.01 [1.28, 5.35] | 18 |
| audio:present | 0.822 [0.756, 0.889] | 0.812 [0.754, 0.871] | 20.9 % [11.5 %, 30.7 %] | 26.9 % [16.2 %, 39.7 %] | 4.33 [1.78, 8.20] | 67 |
| audio:absent | 0.765 [0.633, 0.875] | 0.769 [0.671, 0.863] | 12.1 % [2.8 %, 24.3 %] | 18.2 % [6.7 %, 31.2 %] | 0.54 [0.00, 1.90] | 33 |
| onset:official | 0.836 [0.737, 0.912] | 0.853 [0.737, 0.939] | 6.7 % [0.0 %, 20.0 %] | 20.0 % [0.0 %, 47.1 %] | 3.01 [1.28, 5.35] | 15 |
| onset:hivau | 0.774 [0.703, 0.844] | 0.778 [0.723, 0.837] | 19.2 % [10.9 %, 27.6 %] | 23.3 % [14.1 %, 32.4 %] | 3.01 [1.28, 5.35] | 73 |
| onset:ubi | 0.940 [0.882, 0.981] | 0.935 [0.882, 0.978] | 25.0 % [0.0 %, 50.0 %] | 33.3 % [8.3 %, 60.0 %] | 3.01 [1.28, 5.35] | 12 |
| event:first_segment | n/a | n/a | 18.0 % [10.8 %, 25.2 %] | 24.0 % [16.0 %, 32.8 %] | 3.01 [1.28, 5.35] | 100 |
| event:later_segment | n/a | n/a | n/a | n/a | 3.01 [1.28, 5.35] | 0 |

Generated 2026-10-16T15:43:29 by `anticipation/eval/va30.py` (git d25ab37745).
