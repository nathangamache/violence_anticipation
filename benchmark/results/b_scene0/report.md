# VA30 report: b_scene0

Fold file `folds_v2` (content sha256 `8ebdf6ec2e54`), seeds [0, 1, 2], headline seed 0. Test universe: 1031 videos (693 violent, 338 clean negatives = 15.17 h), 198 eligible30 events (73 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 1031 of 6139 universe videos. Not directly comparable with full-universe results.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.792 [0.737, 0.844] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.769 [0.716, 0.819] |
| Far-window AUROC (clean, source x time matched negatives) | 0.711 [0.623, 0.793] |
| Far-window AUROC (clean, uncensored negatives) | 0.785 [0.733, 0.834] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.801 [0.748, 0.848] |
| AUROC / AP, H = 30 s (base rate 0.146) | 0.780 [0.746, 0.813] / 0.466 [0.397, 0.550] |
| Detection AUROC (in-violence vs clean negatives) | 0.826 [0.786, 0.861] |
| Brier P(onset within 30 s) (climatology 0.1436), ECE | 0.2233 [0.1872, 0.2638], 0.2079 |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 1.21 [0.71, 1.78] | 1.28 [0.79, 1.79] | 12.6 % [8.2 %, 17.4 %] | [8.1, 22.7] % | 3.0 % [1.8 %, 4.4 %] | 13.0 % | 0.0 % [0.0 %, 0.0 %] (90) | 13.6 % [8.9 %, 18.6 %] | 13.6 % [8.9 %, 18.6 %] | 42.2 [35.8, 52.9] | 25.8 % [22.5 %, 29.3 %] |
| 2.0 | 1.89 [1.21, 2.63] | 1.95 [1.34, 2.62] | 19.2 % [14.0 %, 24.9 %] | [11.3, 27.0] % | 4.6 % [3.0 %, 6.4 %] | 20.1 % | 0.0 % [0.0 %, 0.0 %] (90) | 20.7 % [15.2 %, 26.7 %] | 20.7 % [15.2 %, 26.7 %] | 37.4 [35.0, 45.8] | 36.1 % [32.6 %, 40.0 %] |
| 5.0 | 4.32 [3.30, 5.49] | 3.95 [3.13, 4.93] | 27.3 % [21.4 %, 33.3 %] | [19.7, 35.0] % | 10.2 % [7.9 %, 12.8 %] | 41.4 % | 0.0 % [0.0 %, 0.0 %] (90) | 28.8 % [22.5 %, 35.2 %] | 28.8 % [22.5 %, 35.2 %] | 42.2 [37.2, 48.9] | 55.7 % [52.0 %, 59.8 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.773 [0.742, 0.802] | 0.690 [0.649, 0.733] | 0.740 [0.710, 0.768] | 0.224 [0.178, 0.304] | 0.0421 |
| 10 | 0.778 [0.746, 0.807] | 0.690 [0.647, 0.734] | 0.748 [0.717, 0.777] | 0.313 [0.257, 0.397] | 0.0721 |
| 20 | 0.779 [0.747, 0.810] | 0.687 [0.641, 0.733] | 0.756 [0.723, 0.787] | 0.407 [0.340, 0.494] | 0.1148 |
| 30 | 0.780 [0.746, 0.813] | 0.686 [0.636, 0.733] | 0.762 [0.728, 0.795] | 0.466 [0.397, 0.550] | 0.1457 |
| 60 | 0.777 [0.738, 0.814] | 0.676 [0.621, 0.729] | 0.766 [0.728, 0.801] | 0.552 [0.479, 0.639] | 0.2110 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.784 [0.624, 0.909] | 0.704 [0.635, 0.767] | 16.7 % [3.8 %, 35.0 %] | 20.8 % [5.3 %, 38.9 %] | 3.10 [1.66, 4.72] | 24 |
| source:xd_youtube | 0.604 [0.421, 0.786] | 0.748 [0.641, 0.842] | 20.0 % [5.7 %, 36.7 %] | 20.0 % [5.7 %, 36.7 %] | 0.58 [0.12, 1.15] | 30 |
| source:ucf | 0.702 [0.578, 0.813] | 0.679 [0.564, 0.757] | 19.4 % [13.1 %, 26.3 %] | 20.8 % [14.5 %, 27.8 %] | 6.01 [2.01, 11.73] | 144 |
| class:interpersonal | 0.789 [0.730, 0.847] | 0.786 [0.751, 0.821] | 19.1 % [13.6 %, 25.3 %] | 19.7 % [13.9 %, 25.7 %] | 1.89 [1.21, 2.63] | 173 |
| class:other_violent | 0.812 [0.703, 0.890] | 0.771 [0.712, 0.823] | 20.0 % [5.9 %, 38.1 %] | 28.0 % [12.8 %, 47.6 %] | 1.89 [1.21, 2.63] | 25 |
| audio:present | 0.744 [0.663, 0.825] | 0.755 [0.710, 0.797] | 15.8 % [9.5 %, 23.4 %] | 16.8 % [10.0 %, 24.2 %] | 1.57 [0.97, 2.33] | 101 |
| audio:absent | 0.762 [0.616, 0.886] | 0.693 [0.539, 0.795] | 22.7 % [14.9 %, 31.6 %] | 24.7 % [16.2 %, 33.8 %] | 7.21 [1.93, 15.68] | 97 |
| onset:official | 0.734 [0.638, 0.816] | 0.748 [0.700, 0.793] | 22.7 % [12.8 %, 32.9 %] | 25.3 % [15.3 %, 35.8 %] | 1.89 [1.21, 2.63] | 75 |
| onset:hivau | 0.830 [0.772, 0.884] | 0.821 [0.787, 0.852] | 17.1 % [10.2 %, 23.9 %] | 17.9 % [10.9 %, 24.8 %] | 1.89 [1.21, 2.63] | 123 |
| event:first_segment | n/a | n/a | 31.7 % [23.6 %, 40.3 %] | 34.2 % [25.8 %, 42.9 %] | 1.89 [1.21, 2.63] | 120 |
| event:later_segment | n/a | n/a | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 1.89 [1.21, 2.63] | 78 |

## Mean and std over seeds

| Metric | mean | std |
|---|---|---|
| far_auroc_clean | 0.792 | 0.000 |
| far_auroc_secondary | 0.769 | 0.000 |
| far_auroc_clean_time_matched | 0.711 | 0.000 |
| far_auroc_clean_all_segments | 0.801 | 0.000 |
| auroc30 | 0.780 | 0.000 |
| auroc30_time_matched | 0.686 | 0.000 |
| ap30 | 0.466 | 0.000 |
| success30_at_fa2 | 0.192 | 0.000 |
| success10_at_fa2 | 0.207 | 0.000 |
| success0_at_fa2 | 0.207 | 0.000 |
| success30_late_at_fa2 | 0.000 | 0.000 |
| test_fa_per_hour_clean_at_fa2 | 1.888 | 0.000 |
| random_success30_at_fa2 | 0.046 | 0.000 |
| position_null_success30_at_fa2 | 0.201 | 0.000 |
| detection_auroc | 0.826 | 0.000 |

Generated 2026-10-14T21:37:09 by `anticipation/eval/va30.py` (git eb9e51707b).
