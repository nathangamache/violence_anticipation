# VA30 report: va30

Fold file `folds_v2_without_confirmation` (content sha256 `5c5149555ca8`), seeds [0], headline seed 0. Test universe: 3599 videos (379 violent, 3220 clean negatives = 125.86 h), 446 eligible30 events (181 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 3599 of 4916 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 653 catalog clean negatives (527 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 125.86 of 242.3 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.794 [0.763, 0.824] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.790 [0.760, 0.820] |
| Far-window AUROC (clean, source x time matched negatives) | 0.785 [0.753, 0.814] |
| Far-window AUROC (clean, uncensored negatives) | 0.789 [0.759, 0.819] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.794 [0.768, 0.820] |
| AUROC / AP, H = 30 s (base rate 0.022) | 0.781 [0.756, 0.807] / 0.105 [0.084, 0.139] |
| Detection AUROC (in-violence vs clean negatives) | 0.786 [0.734, 0.834] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 1.25 [1.02, 1.48] | 1.38 [1.12, 1.65] | 11.2 % [8.6 %, 14.3 %] | [7.7, 15.2] % | 3.1 % [2.5 %, 3.6 %] | 6.6 % | 9.7 % [6.1 %, 13.8 %] (206) | 15.2 % [11.8 %, 19.0 %] | 16.6 % [13.1 %, 20.3 %] | 36.9 [30.9, 42.7] | 15.6 % [12.7 %, 18.6 %] |
| 2.0 | 2.01 [1.72, 2.30] | 2.13 [1.82, 2.43] | 16.4 % [13.0 %, 20.0 %] | [12.2, 21.0] % | 4.9 % [4.2 %, 5.6 %] | 8.4 % | 15.0 % [10.7 %, 20.2 %] (206) | 21.3 % [17.7 %, 25.4 %] | 23.8 % [19.9 %, 27.8 %] | 42.5 [33.2, 47.7] | 22.9 % [19.5 %, 26.6 %] |
| 5.0 | 4.68 [4.21, 5.12] | 4.88 [4.38, 5.40] | 25.3 % [21.3 %, 29.6 %] | [21.4, 29.1] % | 11.0 % [10.0 %, 12.0 %] | 14.9 % | 20.9 % [15.8 %, 26.9 %] (206) | 30.3 % [25.9 %, 34.9 %] | 33.6 % [28.9 %, 38.7 %] | 44.2 [36.9, 47.3] | 35.3 % [31.1 %, 39.7 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.788 [0.761, 0.812] | 0.781 [0.753, 0.805] | 0.787 [0.764, 0.808] | 0.020 [0.015, 0.032] | 0.0035 |
| 10 | 0.788 [0.762, 0.813] | 0.782 [0.755, 0.806] | 0.788 [0.765, 0.809] | 0.038 [0.029, 0.057] | 0.0071 |
| 20 | 0.783 [0.757, 0.808] | 0.777 [0.752, 0.802] | 0.787 [0.764, 0.808] | 0.071 [0.057, 0.100] | 0.0144 |
| 30 | 0.781 [0.756, 0.807] | 0.774 [0.748, 0.799] | 0.786 [0.764, 0.806] | 0.105 [0.084, 0.139] | 0.0222 |
| 60 | 0.786 [0.761, 0.811] | 0.774 [0.749, 0.799] | 0.786 [0.765, 0.808] | 0.185 [0.152, 0.231] | 0.0437 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.750 [0.696, 0.797] | 0.741 [0.698, 0.784] | 22.0 % [15.3 %, 29.4 %] | 36.4 % [28.1 %, 45.6 %] | 3.25 [2.67, 3.99] | 118 |
| source:xd_youtube | 0.707 [0.639, 0.767] | 0.727 [0.669, 0.778] | 13.4 % [7.8 %, 20.3 %] | 19.7 % [12.1 %, 27.7 %] | 2.50 [1.97, 3.07] | 127 |
| source:ucf | 0.866 [0.814, 0.916] | 0.837 [0.790, 0.880] | 16.5 % [10.1 %, 23.2 %] | 19.1 % [12.4 %, 26.1 %] | 0.82 [0.46, 1.18] | 115 |
| source:ubi | 0.890 [0.823, 0.942] | 0.844 [0.774, 0.904] | 12.8 % [6.0 %, 20.8 %] | 18.6 % [10.7 %, 27.6 %] | 0.16 [0.00, 0.35] | 86 |
| class:interpersonal | 0.786 [0.749, 0.820] | 0.780 [0.751, 0.807] | 12.9 % [9.6 %, 16.5 %] | 20.3 % [16.3 %, 24.5 %] | 2.01 [1.72, 2.30] | 380 |
| class:other_violent | 0.834 [0.771, 0.888] | 0.790 [0.723, 0.846] | 36.4 % [25.8 %, 47.8 %] | 43.9 % [31.8 %, 56.7 %] | 2.01 [1.72, 2.30] | 66 |
| audio:present | 0.753 [0.715, 0.789] | 0.752 [0.722, 0.783] | 17.8 % [13.6 %, 22.5 %] | 26.6 % [21.5 %, 32.0 %] | 2.61 [2.23, 3.03] | 286 |
| audio:absent | 0.878 [0.830, 0.919] | 0.842 [0.796, 0.884] | 13.8 % [8.9 %, 19.8 %] | 18.8 % [13.1 %, 24.9 %] | 0.57 [0.34, 0.80] | 160 |
| onset:official | 0.786 [0.712, 0.848] | 0.777 [0.710, 0.838] | 12.5 % [4.2 %, 22.6 %] | 19.6 % [10.5 %, 30.6 %] | 2.01 [1.72, 2.30] | 56 |
| onset:hivau | 0.784 [0.745, 0.821] | 0.778 [0.748, 0.808] | 18.1 % [14.2 %, 22.6 %] | 26.0 % [21.0 %, 31.0 %] | 2.01 [1.72, 2.30] | 304 |
| onset:ubi | 0.859 [0.790, 0.916] | 0.814 [0.744, 0.876] | 12.8 % [6.0 %, 20.8 %] | 18.6 % [10.7 %, 27.6 %] | 2.01 [1.72, 2.30] | 86 |
| event:first_segment | n/a | n/a | 18.6 % [14.1 %, 24.0 %] | 26.1 % [20.9 %, 31.7 %] | 2.01 [1.72, 2.30] | 253 |
| event:later_segment | n/a | n/a | 13.5 % [9.2 %, 18.5 %] | 20.7 % [15.1 %, 26.9 %] | 2.01 [1.72, 2.30] | 193 |

Generated 2026-10-15T13:27:53 by `anticipation/eval/va30.py` (git 00cd04bb59).
