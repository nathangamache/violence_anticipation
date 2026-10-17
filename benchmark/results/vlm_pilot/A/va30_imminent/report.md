# VA30 report: phase_c_A_imminent

Fold file `phase_c_pilot_without_original_fold1` (content sha256 `8ebdf6ec2e54`), seeds [0], headline seed 0. Test universe: 249 videos (100 violent, 149 clean negatives = 5.37 h), 100 eligible30 events (100 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 249 of 249 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 0 catalog clean negatives (0 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 5.37 of 5.37 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.695 [0.636, 0.759] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.650 [0.592, 0.707] |
| Far-window AUROC (clean, source x time matched negatives) | 0.691 [0.630, 0.759] |
| Far-window AUROC (clean, uncensored negatives) | 0.698 [0.641, 0.760] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.699 [0.639, 0.762] |
| AUROC / AP, H = 30 s (base rate 0.139) | 0.671 [0.613, 0.728] / 0.320 [0.249, 0.422] |
| Detection AUROC (in-violence vs clean negatives) | 0.737 [0.663, 0.801] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 7.52 [4.44, 11.18] | 10.90 [7.74, 14.89] | 50.0 % [40.2 %, 59.6 %] | [40.3, 61.3] % | 17.1 % [10.5 %, 24.4 %] | 27.9 % | 45.2 % [26.9 %, 62.5 %] (31) | 56.0 % [46.2 %, 65.6 %] | 57.0 % [47.2 %, 66.3 %] | 65.5 [58.5, 80.6] | 60.4 % [51.5 %, 69.5 %] |
| 2.0 | 8.09 [4.91, 11.91] | 11.70 [8.38, 15.65] | 52.0 % [42.2 %, 61.6 %] | [40.3, 61.7] % | 18.3 % [11.6 %, 25.8 %] | 29.4 % | 45.2 % [26.9 %, 62.5 %] (31) | 57.0 % [47.3 %, 67.0 %] | 58.0 % [48.6 %, 67.6 %] | 67.3 [61.0, 83.3] | 61.3 % [52.9 %, 70.2 %] |
| 5.0 | 8.28 [4.98, 12.17] | 12.18 [8.77, 16.14] | 52.0 % [41.2 %, 62.5 %] | [42.4, 64.5] % | 18.7 % [11.7 %, 26.2 %] | 30.0 % | 41.9 % [22.2 %, 59.5 %] (31) | 56.0 % [46.1 %, 66.0 %] | 57.0 % [47.3 %, 66.7 %] | 70.0 [61.8, 83.0] | 62.3 % [53.8 %, 71.4 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.696 [0.637, 0.753] | 0.685 [0.617, 0.746] | 0.697 [0.638, 0.753] | 0.076 [0.054, 0.126] | 0.0221 |
| 10 | 0.685 [0.627, 0.741] | 0.675 [0.608, 0.738] | 0.687 [0.628, 0.742] | 0.140 [0.101, 0.215] | 0.0447 |
| 20 | 0.677 [0.619, 0.734] | 0.665 [0.602, 0.727] | 0.679 [0.623, 0.735] | 0.242 [0.183, 0.338] | 0.0910 |
| 30 | 0.671 [0.613, 0.728] | 0.659 [0.597, 0.719] | 0.674 [0.616, 0.729] | 0.320 [0.249, 0.422] | 0.1386 |
| 60 | 0.656 [0.601, 0.708] | 0.640 [0.578, 0.698] | 0.658 [0.604, 0.711] | 0.483 [0.406, 0.572] | 0.2826 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.701 [0.610, 0.801] | 0.671 [0.570, 0.776] | 73.9 % [57.1 %, 88.2 %] | 82.6 % [66.7 %, 96.3 %] | 18.95 [10.94, 29.43] | 23 |
| source:xd_youtube | 0.642 [0.523, 0.751] | 0.644 [0.537, 0.746] | 58.1 % [40.6 %, 75.0 %] | 64.5 % [47.2 %, 80.5 %] | 10.45 [4.20, 18.71] | 31 |
| source:ucf | 0.679 [0.544, 0.804] | 0.592 [0.479, 0.709] | 26.5 % [12.1 %, 41.2 %] | 29.4 % [15.4 %, 45.2 %] | 0.83 [0.00, 2.79] | 34 |
| source:ubi | 0.862 [0.730, 0.975] | 0.918 [0.839, 0.984] | 66.7 % [37.5 %, 93.3 %] | 75.0 % [46.1 %, 100.0 %] | 0.00 [0.00, 0.00] | 12 |
| class:interpersonal | 0.708 [0.642, 0.771] | 0.680 [0.618, 0.741] | 51.2 % [40.5 %, 61.9 %] | 57.3 % [46.7 %, 68.2 %] | 8.09 [4.91, 11.91] | 82 |
| class:other_violent | 0.629 [0.480, 0.765] | 0.666 [0.523, 0.795] | 55.6 % [29.4 %, 76.5 %] | 61.1 % [38.1 %, 82.4 %] | 8.09 [4.91, 11.91] | 18 |
| audio:present | 0.657 [0.581, 0.734] | 0.619 [0.549, 0.691] | 58.2 % [46.3 %, 69.1 %] | 64.2 % [51.6 %, 75.3 %] | 12.13 [7.41, 17.90] | 67 |
| audio:absent | 0.772 [0.659, 0.879] | 0.769 [0.676, 0.860] | 39.4 % [23.1 %, 56.4 %] | 45.5 % [29.0 %, 61.8 %] | 0.54 [0.00, 1.79] | 33 |
| onset:official | 0.657 [0.497, 0.803] | 0.718 [0.572, 0.852] | 53.3 % [27.8 %, 79.0 %] | 66.7 % [38.5 %, 88.9 %] | 8.09 [4.91, 11.91] | 15 |
| onset:hivau | 0.676 [0.604, 0.753] | 0.635 [0.571, 0.701] | 49.3 % [37.8 %, 60.4 %] | 53.4 % [42.9 %, 64.3 %] | 8.09 [4.91, 11.91] | 73 |
| onset:ubi | 0.858 [0.748, 0.964] | 0.916 [0.834, 0.980] | 66.7 % [37.5 %, 93.3 %] | 75.0 % [46.1 %, 100.0 %] | 8.09 [4.91, 11.91] | 12 |
| event:first_segment | n/a | n/a | 52.0 % [42.2 %, 61.6 %] | 58.0 % [48.6 %, 67.6 %] | 8.09 [4.91, 11.91] | 100 |
| event:later_segment | n/a | n/a | n/a | n/a | 8.09 [4.91, 11.91] | 0 |

Generated 2026-10-16T15:43:10 by `anticipation/eval/va30.py` (git d25ab37745).
