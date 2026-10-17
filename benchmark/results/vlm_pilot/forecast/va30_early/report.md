# VA30 report: phase_c_forecast_early

Fold file `phase_c_pilot_without_original_fold1` (content sha256 `8ebdf6ec2e54`), seeds [0], headline seed 0. Test universe: 249 videos (100 violent, 149 clean negatives = 5.37 h), 100 eligible30 events (100 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 249 of 249 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 0 catalog clean negatives (0 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 5.37 of 5.37 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.766 [0.701, 0.829] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.726 [0.662, 0.787] |
| Far-window AUROC (clean, source x time matched negatives) | 0.738 [0.669, 0.805] |
| Far-window AUROC (clean, uncensored negatives) | 0.753 [0.691, 0.812] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.765 [0.698, 0.828] |
| AUROC / AP, H = 30 s (base rate 0.139) | 0.777 [0.719, 0.833] / 0.347 [0.261, 0.473] |
| Detection AUROC (in-violence vs clean negatives) | 0.825 [0.764, 0.876] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 2.63 [0.81, 5.10] | 3.20 [1.31, 5.43] | 20.0 % [12.0 %, 28.3 %] | [13.0, 31.3] % | 6.4 % [2.0 %, 12.0 %] | 12.2 % | 19.4 % [6.2 %, 35.3 %] (31) | 23.0 % [14.6 %, 31.6 %] | 26.0 % [16.7 %, 35.4 %] | 66.3 [36.3, 77.8] | 24.5 % [15.5 %, 33.6 %] |
| 2.0 | 2.63 [0.81, 5.10] | 3.20 [1.31, 5.43] | 20.0 % [12.0 %, 28.3 %] | [13.0, 32.1] % | 6.4 % [2.0 %, 12.0 %] | 12.2 % | 19.4 % [6.2 %, 35.3 %] (31) | 23.0 % [14.6 %, 31.6 %] | 26.0 % [16.7 %, 35.4 %] | 66.3 [36.3, 77.8] | 24.5 % [15.5 %, 33.6 %] |
| 5.0 | 4.89 [2.51, 7.51] | 5.93 [3.48, 8.77] | 26.0 % [15.9 %, 35.7 %] | [15.7, 43.4] % | 11.5 % [6.1 %, 17.1 %] | 20.0 % | 22.6 % [10.0 %, 39.1 %] (31) | 32.0 % [22.2 %, 42.7 %] | 37.0 % [26.3 %, 49.0 %] | 52.5 [33.1, 73.7] | 36.8 % [26.5 %, 48.0 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.794 [0.737, 0.844] | 0.749 [0.682, 0.808] | 0.783 [0.725, 0.834] | 0.084 [0.056, 0.156] | 0.0221 |
| 10 | 0.791 [0.735, 0.843] | 0.746 [0.678, 0.807] | 0.784 [0.728, 0.835] | 0.151 [0.104, 0.246] | 0.0447 |
| 20 | 0.785 [0.730, 0.839] | 0.738 [0.670, 0.802] | 0.781 [0.725, 0.836] | 0.258 [0.188, 0.377] | 0.0910 |
| 30 | 0.777 [0.719, 0.833] | 0.729 [0.660, 0.794] | 0.775 [0.717, 0.830] | 0.347 [0.261, 0.473] | 0.1386 |
| 60 | 0.750 [0.691, 0.809] | 0.706 [0.637, 0.775] | 0.749 [0.689, 0.807] | 0.517 [0.416, 0.634] | 0.2826 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.598 [0.476, 0.729] | 0.644 [0.541, 0.756] | 30.4 % [9.1 %, 50.0 %] | 39.1 % [12.5 %, 62.5 %] | 12.63 [4.35, 22.19] | 23 |
| source:xd_youtube | 0.810 [0.691, 0.902] | 0.803 [0.707, 0.885] | 12.9 % [3.1 %, 26.3 %] | 19.4 % [6.5 %, 34.6 %] | 0.00 [0.00, 0.00] | 31 |
| source:ucf | 0.741 [0.614, 0.863] | 0.739 [0.631, 0.842] | 11.8 % [2.7 %, 23.3 %] | 14.7 % [3.4 %, 27.3 %] | 0.00 [0.00, 0.00] | 34 |
| source:ubi | 0.940 [0.836, 0.999] | 0.974 [0.928, 0.999] | 41.7 % [14.3 %, 72.7 %] | 50.0 % [20.0 %, 80.0 %] | 0.00 [0.00, 0.00] | 12 |
| class:interpersonal | 0.779 [0.710, 0.843] | 0.798 [0.739, 0.853] | 20.7 % [12.7 %, 29.5 %] | 28.0 % [18.4 %, 38.0 %] | 2.63 [0.81, 5.10] | 82 |
| class:other_violent | 0.700 [0.557, 0.825] | 0.699 [0.557, 0.826] | 16.7 % [0.0 %, 40.0 %] | 16.7 % [0.0 %, 40.0 %] | 2.63 [0.81, 5.10] | 18 |
| audio:present | 0.761 [0.679, 0.839] | 0.768 [0.696, 0.836] | 19.4 % [9.7 %, 29.2 %] | 25.4 % [13.2 %, 37.9 %] | 4.04 [1.17, 7.73] | 67 |
| audio:absent | 0.782 [0.657, 0.890] | 0.806 [0.712, 0.898] | 21.2 % [7.9 %, 35.3 %] | 27.3 % [12.5 %, 42.9 %] | 0.00 [0.00, 0.00] | 33 |
| onset:official | 0.729 [0.590, 0.842] | 0.773 [0.663, 0.864] | 13.3 % [0.0 %, 33.3 %] | 20.0 % [0.0 %, 38.9 %] | 2.63 [0.81, 5.10] | 15 |
| onset:hivau | 0.746 [0.668, 0.816] | 0.760 [0.693, 0.822] | 17.8 % [9.3 %, 27.0 %] | 23.3 % [13.7 %, 34.1 %] | 2.63 [0.81, 5.10] | 73 |
| onset:ubi | 0.929 [0.874, 0.974] | 0.949 [0.916, 0.978] | 41.7 % [14.3 %, 72.7 %] | 50.0 % [20.0 %, 80.0 %] | 2.63 [0.81, 5.10] | 12 |
| event:first_segment | n/a | n/a | 20.0 % [12.0 %, 28.3 %] | 26.0 % [16.7 %, 35.4 %] | 2.63 [0.81, 5.10] | 100 |
| event:later_segment | n/a | n/a | n/a | n/a | 2.63 [0.81, 5.10] | 0 |

Generated 2026-10-16T15:43:42 by `anticipation/eval/va30.py` (git d25ab37745).
