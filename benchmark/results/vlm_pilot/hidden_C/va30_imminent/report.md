# VA30 report: phase_c_hidden_C_imminent

Fold file `phase_c_pilot_without_original_fold1` (content sha256 `8ebdf6ec2e54`), seeds [0], headline seed 0. Test universe: 249 videos (100 violent, 149 clean negatives = 5.36 h), 100 eligible30 events (100 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 249 of 249 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 0 catalog clean negatives (0 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 5.36 of 5.36 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.860 [0.805, 0.908] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.796 [0.741, 0.847] |
| Far-window AUROC (clean, source x time matched negatives) | 0.852 [0.795, 0.900] |
| Far-window AUROC (clean, uncensored negatives) | 0.862 [0.812, 0.907] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.861 [0.807, 0.908] |
| AUROC / AP, H = 30 s (base rate 0.139) | 0.823 [0.774, 0.869] / 0.404 [0.322, 0.536] |
| Detection AUROC (in-violence vs clean negatives) | 0.889 [0.840, 0.927] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 4.52 [2.48, 6.89] | 7.54 [5.17, 10.18] | 49.0 % [38.9 %, 59.0 %] | [40.6, 59.6] % | 10.7 % [6.0 %, 15.8 %] | 18.7 % | 32.3 % [16.0 %, 50.0 %] (31) | 50.0 % [40.2 %, 59.8 %] | 53.0 % [42.7 %, 63.7 %] | 71.3 [64.2, 76.8] | 65.1 % [56.2 %, 73.7 %] |
| 2.0 | 4.71 [2.65, 7.11] | 7.86 [5.65, 10.52] | 48.0 % [38.0 %, 58.3 %] | [40.4, 60.5] % | 11.1 % [6.4 %, 16.3 %] | 19.4 % | 29.0 % [13.9 %, 46.9 %] (31) | 49.0 % [39.3 %, 59.2 %] | 52.0 % [41.6 %, 62.8 %] | 70.8 [62.8, 76.6] | 65.1 % [56.2 %, 73.7 %] |
| 5.0 | 5.65 [3.45, 8.19] | 9.63 [7.17, 12.52] | 53.0 % [42.4 %, 63.9 %] | [38.6, 61.9] % | 13.2 % [8.3 %, 18.5 %] | 22.5 % | 32.3 % [16.2 %, 50.0 %] (31) | 54.0 % [43.3 %, 64.8 %] | 55.0 % [44.4 %, 65.5 %] | 72.5 [70.0, 78.5] | 71.7 % [63.2 %, 79.8 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.842 [0.794, 0.884] | 0.834 [0.772, 0.884] | 0.842 [0.794, 0.884] | 0.110 [0.085, 0.205] | 0.0221 |
| 10 | 0.834 [0.784, 0.878] | 0.825 [0.762, 0.876] | 0.834 [0.785, 0.878] | 0.188 [0.144, 0.305] | 0.0447 |
| 20 | 0.828 [0.779, 0.874] | 0.819 [0.757, 0.871] | 0.828 [0.779, 0.873] | 0.309 [0.239, 0.439] | 0.0910 |
| 30 | 0.823 [0.774, 0.869] | 0.815 [0.753, 0.867] | 0.823 [0.774, 0.869] | 0.404 [0.322, 0.536] | 0.1386 |
| 60 | 0.804 [0.750, 0.853] | 0.788 [0.725, 0.844] | 0.804 [0.751, 0.853] | 0.583 [0.494, 0.693] | 0.2826 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.855 [0.752, 0.933] | 0.863 [0.759, 0.941] | 65.2 % [41.7 %, 85.0 %] | 73.9 % [50.0 %, 92.6 %] | 5.42 [0.91, 11.44] | 23 |
| source:xd_youtube | 0.859 [0.754, 0.938] | 0.816 [0.716, 0.894] | 45.2 % [28.6 %, 63.0 %] | 48.4 % [32.1 %, 65.7 %] | 3.99 [1.47, 7.45] | 31 |
| source:ucf | 0.854 [0.739, 0.942] | 0.768 [0.663, 0.863] | 41.2 % [25.0 %, 57.7 %] | 44.1 % [27.8 %, 60.0 %] | 5.01 [0.90, 10.58] | 34 |
| source:ubi | 0.864 [0.688, 0.996] | 0.897 [0.790, 0.980] | 41.7 % [14.3 %, 72.8 %] | 41.7 % [14.3 %, 72.8 %] | 5.02 [0.00, 11.66] | 12 |
| class:interpersonal | 0.858 [0.801, 0.908] | 0.825 [0.771, 0.872] | 47.6 % [36.8 %, 58.3 %] | 51.2 % [39.7 %, 62.7 %] | 4.71 [2.65, 7.11] | 82 |
| class:other_violent | 0.868 [0.787, 0.938] | 0.878 [0.794, 0.947] | 50.0 % [26.9 %, 78.6 %] | 55.6 % [33.3 %, 80.0 %] | 4.71 [2.65, 7.11] | 18 |
| audio:present | 0.859 [0.796, 0.913] | 0.819 [0.753, 0.876] | 52.2 % [40.3 %, 64.9 %] | 56.7 % [43.8 %, 69.1 %] | 4.05 [1.85, 6.89] | 67 |
| audio:absent | 0.864 [0.757, 0.947] | 0.838 [0.752, 0.909] | 39.4 % [22.2 %, 57.1 %] | 42.4 % [25.8 %, 60.6 %] | 5.95 [1.77, 10.14] | 33 |
| onset:official | 0.840 [0.745, 0.917] | 0.854 [0.770, 0.930] | 46.7 % [23.1 %, 72.7 %] | 46.7 % [23.1 %, 72.7 %] | 4.71 [2.65, 7.11] | 15 |
| onset:hivau | 0.868 [0.809, 0.916] | 0.834 [0.776, 0.882] | 49.3 % [38.2 %, 60.6 %] | 54.8 % [43.3 %, 66.2 %] | 4.71 [2.65, 7.11] | 73 |
| onset:ubi | 0.836 [0.683, 0.955] | 0.864 [0.760, 0.957] | 41.7 % [14.3 %, 72.8 %] | 41.7 % [14.3 %, 72.8 %] | 4.71 [2.65, 7.11] | 12 |
| event:first_segment | n/a | n/a | 48.0 % [38.0 %, 58.3 %] | 52.0 % [41.6 %, 62.8 %] | 4.71 [2.65, 7.11] | 100 |
| event:later_segment | n/a | n/a | n/a | n/a | 4.71 [2.65, 7.11] | 0 |

Generated 2026-10-16T16:10:44 by `anticipation/eval/va30.py` (git d25ab37745).
