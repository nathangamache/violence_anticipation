# VA30 report: phase_c_hidden_B_early

Fold file `phase_c_pilot_without_original_fold1` (content sha256 `8ebdf6ec2e54`), seeds [0], headline seed 0. Test universe: 249 videos (100 violent, 149 clean negatives = 5.36 h), 100 eligible30 events (100 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 249 of 249 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 0 catalog clean negatives (0 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 5.36 of 5.36 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.811 [0.741, 0.870] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.729 [0.664, 0.787] |
| Far-window AUROC (clean, source x time matched negatives) | 0.776 [0.693, 0.851] |
| Far-window AUROC (clean, uncensored negatives) | 0.808 [0.748, 0.864] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.814 [0.747, 0.872] |
| AUROC / AP, H = 30 s (base rate 0.139) | 0.768 [0.711, 0.822] / 0.301 [0.235, 0.397] |
| Detection AUROC (in-violence vs clean negatives) | 0.833 [0.773, 0.885] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 4.71 [2.23, 7.42] | 6.58 [4.05, 9.45] | 26.0 % [17.9 %, 35.3 %] | [18.3, 40.0] % | 11.1 % [5.4 %, 16.9 %] | 19.4 % | 16.1 % [3.7 %, 30.6 %] (31) | 28.0 % [19.4 %, 37.6 %] | 28.0 % [19.4 %, 37.6 %] | 66.9 [55.0, 77.9] | 33.0 % [25.0 %, 42.2 %] |
| 2.0 | 4.71 [2.23, 7.42] | 6.74 [4.12, 9.60] | 26.0 % [17.9 %, 35.3 %] | [18.3, 40.7] % | 11.1 % [5.4 %, 16.9 %] | 19.4 % | 16.1 % [3.7 %, 30.6 %] (31) | 28.0 % [19.4 %, 37.6 %] | 28.0 % [19.4 %, 37.6 %] | 66.9 [55.0, 77.9] | 33.0 % [25.0 %, 42.2 %] |
| 5.0 | 6.97 [4.02, 10.11] | 9.30 [6.38, 12.42] | 34.0 % [25.2 %, 44.2 %] | [23.8, 52.9] % | 16.0 % [9.6 %, 22.3 %] | 26.3 % | 16.1 % [4.3 %, 31.3 %] (31) | 38.0 % [28.7 %, 48.9 %] | 41.0 % [31.4 %, 51.9 %] | 66.4 [56.9, 73.3] | 51.9 % [42.6 %, 61.9 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.788 [0.735, 0.839] | 0.757 [0.688, 0.821] | 0.787 [0.734, 0.838] | 0.069 [0.049, 0.120] | 0.0221 |
| 10 | 0.788 [0.734, 0.838] | 0.757 [0.688, 0.824] | 0.788 [0.735, 0.839] | 0.129 [0.093, 0.201] | 0.0447 |
| 20 | 0.776 [0.719, 0.829] | 0.747 [0.675, 0.814] | 0.776 [0.719, 0.828] | 0.223 [0.169, 0.310] | 0.0910 |
| 30 | 0.768 [0.711, 0.822] | 0.740 [0.666, 0.807] | 0.768 [0.711, 0.822] | 0.301 [0.235, 0.397] | 0.1386 |
| 60 | 0.742 [0.680, 0.799] | 0.701 [0.624, 0.775] | 0.743 [0.682, 0.801] | 0.470 [0.394, 0.566] | 0.2826 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.696 [0.583, 0.820] | 0.720 [0.623, 0.841] | 43.5 % [22.7 %, 70.0 %] | 43.5 % [22.7 %, 70.0 %] | 14.44 [5.77, 23.35] | 23 |
| source:xd_youtube | 0.874 [0.773, 0.946] | 0.817 [0.729, 0.891] | 16.1 % [4.7 %, 31.0 %] | 19.4 % [6.5 %, 34.6 %] | 0.00 [0.00, 0.00] | 31 |
| source:ucf | 0.743 [0.566, 0.906] | 0.682 [0.557, 0.802] | 26.5 % [12.9 %, 42.3 %] | 29.4 % [15.4 %, 45.5 %] | 3.34 [0.00, 8.79] | 34 |
| source:ubi | 0.828 [0.627, 0.982] | 0.839 [0.694, 0.964] | 16.7 % [0.0 %, 40.0 %] | 16.7 % [0.0 %, 40.0 %] | 5.02 [0.00, 11.06] | 12 |
| class:interpersonal | 0.820 [0.750, 0.882] | 0.774 [0.714, 0.830] | 23.2 % [14.5 %, 33.3 %] | 25.6 % [16.7 %, 36.1 %] | 4.71 [2.23, 7.42] | 82 |
| class:other_violent | 0.767 [0.644, 0.882] | 0.797 [0.678, 0.902] | 38.9 % [15.4 %, 66.7 %] | 38.9 % [15.4 %, 66.7 %] | 4.71 [2.23, 7.42] | 18 |
| audio:present | 0.815 [0.734, 0.880] | 0.772 [0.704, 0.836] | 28.4 % [18.1 %, 40.0 %] | 29.9 % [19.3 %, 41.7 %] | 4.62 [1.56, 8.15] | 67 |
| audio:absent | 0.791 [0.645, 0.920] | 0.765 [0.663, 0.861] | 21.2 % [8.1 %, 36.1 %] | 24.2 % [10.7 %, 40.0 %] | 4.87 [1.50, 9.05] | 33 |
| onset:official | 0.764 [0.636, 0.871] | 0.814 [0.726, 0.886] | 33.3 % [9.1 %, 57.1 %] | 33.3 % [9.1 %, 57.1 %] | 4.71 [2.23, 7.42] | 15 |
| onset:hivau | 0.815 [0.741, 0.877] | 0.774 [0.713, 0.833] | 26.0 % [17.1 %, 36.7 %] | 28.8 % [19.1 %, 40.0 %] | 4.71 [2.23, 7.42] | 73 |
| onset:ubi | 0.848 [0.755, 0.926] | 0.875 [0.793, 0.943] | 16.7 % [0.0 %, 40.0 %] | 16.7 % [0.0 %, 40.0 %] | 4.71 [2.23, 7.42] | 12 |
| event:first_segment | n/a | n/a | 26.0 % [17.9 %, 35.3 %] | 28.0 % [19.4 %, 37.6 %] | 4.71 [2.23, 7.42] | 100 |
| event:later_segment | n/a | n/a | n/a | n/a | 4.71 [2.23, 7.42] | 0 |

Generated 2026-10-16T16:01:51 by `anticipation/eval/va30.py` (git d25ab37745).
