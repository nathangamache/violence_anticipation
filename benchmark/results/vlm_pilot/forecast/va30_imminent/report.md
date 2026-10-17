# VA30 report: phase_c_forecast_imminent

Fold file `phase_c_pilot_without_original_fold1` (content sha256 `8ebdf6ec2e54`), seeds [0], headline seed 0. Test universe: 249 videos (100 violent, 149 clean negatives = 5.37 h), 100 eligible30 events (100 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 249 of 249 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 0 catalog clean negatives (0 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 5.37 of 5.37 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.747 [0.670, 0.816] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.711 [0.645, 0.776] |
| Far-window AUROC (clean, source x time matched negatives) | 0.708 [0.627, 0.783] |
| Far-window AUROC (clean, uncensored negatives) | 0.734 [0.664, 0.800] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.747 [0.673, 0.816] |
| AUROC / AP, H = 30 s (base rate 0.139) | 0.759 [0.700, 0.815] / 0.330 [0.253, 0.447] |
| Detection AUROC (in-violence vs clean negatives) | 0.799 [0.736, 0.855] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 3.95 [1.70, 6.71] | 4.97 [2.62, 7.46] | 21.0 % [12.6 %, 29.7 %] | [13.9, 33.0] % | 9.4 % [4.2 %, 15.4 %] | 16.8 % | 19.4 % [6.9 %, 34.8 %] (31) | 24.0 % [15.7 %, 32.6 %] | 28.0 % [17.5 %, 39.2 %] | 66.3 [33.6, 89.0] | 28.3 % [18.7 %, 39.2 %] |
| 2.0 | 4.14 [1.83, 6.90] | 5.13 [2.80, 7.61] | 21.0 % [12.6 %, 29.7 %] | [13.9, 34.6] % | 9.8 % [4.5 %, 15.9 %] | 17.4 % | 19.4 % [6.9 %, 34.8 %] (31) | 24.0 % [15.7 %, 32.6 %] | 28.0 % [17.5 %, 39.2 %] | 66.3 [33.6, 89.0] | 28.3 % [18.7 %, 39.2 %] |
| 5.0 | 5.08 [2.41, 8.02] | 6.09 [3.42, 8.75] | 26.0 % [17.3 %, 35.6 %] | [17.1, 44.0] % | 11.9 % [5.8 %, 18.2 %] | 20.7 % | 22.6 % [9.1 %, 39.1 %] (31) | 28.0 % [19.2 %, 37.4 %] | 32.0 % [22.1 %, 44.0 %] | 62.5 [40.4, 77.1] | 33.0 % [23.3 %, 44.1 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.775 [0.717, 0.829] | 0.731 [0.666, 0.794] | 0.764 [0.705, 0.818] | 0.076 [0.053, 0.128] | 0.0221 |
| 10 | 0.771 [0.713, 0.827] | 0.727 [0.659, 0.793] | 0.763 [0.703, 0.822] | 0.138 [0.098, 0.219] | 0.0447 |
| 20 | 0.763 [0.704, 0.819] | 0.715 [0.645, 0.782] | 0.758 [0.697, 0.816] | 0.240 [0.176, 0.348] | 0.0910 |
| 30 | 0.759 [0.700, 0.815] | 0.708 [0.640, 0.776] | 0.755 [0.696, 0.813] | 0.330 [0.253, 0.447] | 0.1386 |
| 60 | 0.734 [0.672, 0.791] | 0.682 [0.611, 0.751] | 0.732 [0.671, 0.789] | 0.502 [0.413, 0.615] | 0.2826 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.544 [0.414, 0.686] | 0.608 [0.493, 0.726] | 26.1 % [4.5 %, 48.0 %] | 43.5 % [14.3 %, 70.8 %] | 19.85 [11.09, 29.54] | 23 |
| source:xd_youtube | 0.804 [0.677, 0.900] | 0.802 [0.695, 0.884] | 19.4 % [6.5 %, 35.5 %] | 22.6 % [8.8 %, 38.2 %] | 0.00 [0.00, 0.00] | 31 |
| source:ucf | 0.732 [0.602, 0.865] | 0.726 [0.623, 0.831] | 11.8 % [2.7 %, 22.6 %] | 17.6 % [5.9 %, 30.0 %] | 0.00 [0.00, 0.00] | 34 |
| source:ubi | 0.949 [0.864, 0.997] | 0.973 [0.929, 0.997] | 41.7 % [14.3 %, 71.4 %] | 41.7 % [14.3 %, 71.4 %] | 0.00 [0.00, 0.00] | 12 |
| class:interpersonal | 0.757 [0.677, 0.830] | 0.778 [0.717, 0.835] | 22.0 % [13.5 %, 31.0 %] | 30.5 % [20.0 %, 42.3 %] | 4.14 [1.83, 6.90] | 82 |
| class:other_violent | 0.697 [0.547, 0.814] | 0.691 [0.545, 0.818] | 16.7 % [0.0 %, 40.0 %] | 16.7 % [0.0 %, 40.0 %] | 4.14 [1.83, 6.90] | 18 |
| audio:present | 0.739 [0.646, 0.823] | 0.746 [0.672, 0.814] | 20.9 % [10.7 %, 32.3 %] | 29.9 % [16.4 %, 44.4 %] | 6.35 [2.99, 10.71] | 67 |
| audio:absent | 0.777 [0.656, 0.885] | 0.809 [0.714, 0.896] | 21.2 % [8.1 %, 35.5 %] | 24.2 % [10.0 %, 38.7 %] | 0.00 [0.00, 0.00] | 33 |
| onset:official | 0.794 [0.670, 0.895] | 0.796 [0.692, 0.886] | 6.7 % [0.0 %, 22.2 %] | 20.0 % [0.0 %, 38.9 %] | 4.14 [1.83, 6.90] | 15 |
| onset:hivau | 0.705 [0.620, 0.785] | 0.731 [0.663, 0.796] | 20.5 % [11.4 %, 30.6 %] | 27.4 % [15.9 %, 40.7 %] | 4.14 [1.83, 6.90] | 73 |
| onset:ubi | 0.938 [0.894, 0.976] | 0.952 [0.917, 0.980] | 41.7 % [14.3 %, 71.4 %] | 41.7 % [14.3 %, 71.4 %] | 4.14 [1.83, 6.90] | 12 |
| event:first_segment | n/a | n/a | 21.0 % [12.6 %, 29.7 %] | 28.0 % [17.5 %, 39.2 %] | 4.14 [1.83, 6.90] | 100 |
| event:later_segment | n/a | n/a | n/a | n/a | 4.14 [1.83, 6.90] | 0 |

Generated 2026-10-16T15:43:47 by `anticipation/eval/va30.py` (git d25ab37745).
