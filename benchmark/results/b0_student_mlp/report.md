# VA30 report: b0_student_mlp

Fold file `folds_v2` (content sha256 `8ebdf6ec2e54`), seeds [0, 1, 2], headline seed 0. Test universe: 1031 videos (693 violent, 338 clean negatives = 15.17 h), 198 eligible30 events (73 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 1031 of 6139 universe videos. Not directly comparable with full-universe results.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.894 [0.858, 0.923] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.862 [0.822, 0.896] |
| Far-window AUROC (clean, source x time matched negatives) | 0.761 [0.660, 0.847] |
| Far-window AUROC (clean, uncensored negatives) | 0.882 [0.846, 0.910] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.900 [0.870, 0.927] |
| AUROC / AP, H = 30 s (base rate 0.146) | 0.865 [0.837, 0.889] / 0.486 [0.411, 0.570] |
| Detection AUROC (in-violence vs clean negatives) | 0.894 [0.870, 0.914] |
| Brier P(onset within 30 s) (climatology 0.1436), ECE | 0.1592 [0.1380, 0.1829], 0.1390 |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 1.48 [0.89, 2.18] | 4.00 [3.13, 5.05] | 16.7 % [11.0 %, 22.0 %] | [10.1, 26.4] % | 3.6 % [2.2 %, 5.3 %] | 15.9 % | 8.9 % [3.2 %, 15.7 %] (90) | 22.7 % [16.6 %, 28.6 %] | 27.3 % [20.6 %, 33.3 %] | 37.2 [29.1, 41.3] | 29.1 % [26.0 %, 32.1 %] |
| 2.0 | 2.63 [1.80, 3.60] | 5.54 [4.43, 6.82] | 22.2 % [16.0 %, 28.1 %] | [14.4, 32.8] % | 6.4 % [4.4 %, 8.6 %] | 27.9 % | 17.8 % [9.9 %, 26.8 %] (90) | 30.8 % [23.7 %, 37.2 %] | 34.3 % [27.2 %, 40.9 %] | 40.2 [30.6, 48.1] | 35.6 % [32.4 %, 38.8 %] |
| 5.0 | 6.61 [5.04, 8.56] | 9.90 [8.13, 11.84] | 35.9 % [29.3 %, 42.5 %] | [25.6, 46.3] % | 15.2 % [11.8 %, 19.3 %] | 50.2 % | 30.0 % [20.2 %, 40.2 %] (90) | 48.5 % [40.9 %, 55.6 %] | 53.0 % [45.5 %, 60.2 %] | 36.8 [33.0, 41.7] | 53.1 % [49.5 %, 56.4 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.854 [0.828, 0.877] | 0.724 [0.679, 0.772] | 0.810 [0.782, 0.834] | 0.207 [0.163, 0.276] | 0.0421 |
| 10 | 0.859 [0.833, 0.881] | 0.721 [0.673, 0.771] | 0.819 [0.790, 0.844] | 0.311 [0.253, 0.393] | 0.0721 |
| 20 | 0.865 [0.838, 0.887] | 0.720 [0.669, 0.773] | 0.832 [0.803, 0.857] | 0.426 [0.355, 0.512] | 0.1148 |
| 30 | 0.865 [0.837, 0.889] | 0.713 [0.660, 0.768] | 0.837 [0.807, 0.864] | 0.486 [0.411, 0.570] | 0.1457 |
| 60 | 0.865 [0.833, 0.891] | 0.697 [0.635, 0.758] | 0.846 [0.812, 0.874] | 0.586 [0.508, 0.670] | 0.2110 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.718 [0.563, 0.848] | 0.751 [0.699, 0.801] | 33.3 % [16.0 %, 51.9 %] | 58.3 % [38.1 %, 78.3 %] | 4.13 [2.57, 6.14] | 24 |
| source:xd_youtube | 0.790 [0.653, 0.886] | 0.867 [0.813, 0.906] | 16.7 % [4.0 %, 31.0 %] | 36.7 % [19.2 %, 55.2 %] | 1.96 [0.97, 3.26] | 30 |
| source:ucf | 0.784 [0.620, 0.899] | 0.718 [0.570, 0.829] | 21.5 % [14.5 %, 29.2 %] | 29.9 % [22.4 %, 37.7 %] | 1.50 [0.00, 5.09] | 144 |
| class:interpersonal | 0.892 [0.855, 0.925] | 0.866 [0.837, 0.890] | 23.1 % [16.5 %, 29.8 %] | 34.1 % [26.1 %, 41.4 %] | 2.63 [1.80, 3.60] | 173 |
| class:other_violent | 0.904 [0.827, 0.958] | 0.890 [0.859, 0.915] | 16.0 % [3.3 %, 30.0 %] | 36.0 % [17.4 %, 54.6 %] | 2.63 [1.80, 3.60] | 25 |
| audio:present | 0.872 [0.815, 0.919] | 0.858 [0.831, 0.884] | 26.7 % [17.8 %, 34.9 %] | 40.6 % [30.8 %, 49.5 %] | 2.64 [1.79, 3.68] | 101 |
| audio:absent | 0.684 [0.475, 0.876] | 0.700 [0.499, 0.841] | 17.5 % [10.1 %, 25.9 %] | 27.8 % [18.8 %, 37.8 %] | 2.40 [0.00, 8.04] | 97 |
| onset:official | 0.826 [0.759, 0.881] | 0.832 [0.797, 0.863] | 24.0 % [14.1 %, 32.9 %] | 45.3 % [33.3 %, 56.4 %] | 2.63 [1.80, 3.60] | 75 |
| onset:hivau | 0.939 [0.914, 0.960] | 0.916 [0.892, 0.937] | 21.1 % [14.1 %, 29.2 %] | 27.6 % [19.2 %, 36.5 %] | 2.63 [1.80, 3.60] | 123 |
| event:first_segment | n/a | n/a | 25.0 % [17.4 %, 33.0 %] | 34.2 % [25.7 %, 42.6 %] | 2.63 [1.80, 3.60] | 120 |
| event:later_segment | n/a | n/a | 17.9 % [9.2 %, 28.1 %] | 34.6 % [24.0 %, 45.6 %] | 2.63 [1.80, 3.60] | 78 |

## Mean and std over seeds

| Metric | mean | std |
|---|---|---|
| far_auroc_clean | 0.891 | 0.006 |
| far_auroc_secondary | 0.862 | 0.004 |
| far_auroc_clean_time_matched | 0.758 | 0.015 |
| far_auroc_clean_all_segments | 0.897 | 0.005 |
| auroc30 | 0.869 | 0.004 |
| auroc30_time_matched | 0.718 | 0.006 |
| ap30 | 0.491 | 0.008 |
| success30_at_fa2 | 0.205 | 0.024 |
| success10_at_fa2 | 0.281 | 0.025 |
| success0_at_fa2 | 0.316 | 0.028 |
| success30_late_at_fa2 | 0.156 | 0.018 |
| test_fa_per_hour_clean_at_fa2 | 2.360 | 0.291 |
| random_success30_at_fa2 | 0.057 | 0.007 |
| position_null_success30_at_fa2 | 0.251 | 0.031 |
| detection_auroc | 0.893 | 0.002 |

Generated 2026-10-14T21:36:35 by `anticipation/eval/va30.py` (git eb9e51707b).
