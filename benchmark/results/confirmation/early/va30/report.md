# VA30 report: fold1_confirmation

Fold file `fold1_confirmation` (content sha256 ``), seeds [0], headline seed 0. Test universe: 1214 videos (424 violent, 790 clean negatives = 31.03 h), 75 eligible30 events (29 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 1655 of 1655 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 653 catalog clean negatives (126 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 31.03 of 61.22 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.895 [0.843, 0.936] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.892 [0.840, 0.932] |
| Far-window AUROC (clean, source x time matched negatives) | 0.891 [0.839, 0.932] |
| Far-window AUROC (clean, uncensored negatives) | 0.896 [0.847, 0.936] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.901 [0.861, 0.933] |
| AUROC / AP, H = 30 s (base rate 0.041) | 0.888 [0.862, 0.917] / 0.386 [0.314, 0.459] |
| Detection AUROC (in-violence vs clean negatives) | 0.948 [0.921, 0.975] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 0.33 [0.10, 0.68] | 0.52 [0.15, 1.02] | 16.0 % [7.9 %, 25.0 %] | [1.9, 30.6] % | 0.8 % [0.2 %, 1.7 %] | 3.4 % | 14.7 % [3.1 %, 26.3 %] (34) | 20.0 % [10.7 %, 29.2 %] | 24.0 % [14.1 %, 32.6 %] | 36.3 [23.5, 51.6] | 29.6 % [25.7 %, 34.0 %] |
| 2.0 | 0.72 [0.34, 1.15] | 0.97 [0.44, 1.66] | 22.7 % [13.0 %, 31.8 %] | [13.7, 38.8] % | 1.8 % [0.8 %, 2.8 %] | 4.3 % | 26.5 % [12.0 %, 40.0 %] (34) | 30.7 % [18.9 %, 41.1 %] | 34.7 % [21.1 %, 46.7 %] | 32.3 [26.1, 51.8] | 41.2 % [36.2 %, 46.8 %] |
| 5.0 | 2.64 [1.75, 3.68] | 2.85 [1.89, 3.93] | 36.0 % [23.8 %, 48.0 %] | [25.0, 51.8] % | 6.4 % [4.3 %, 8.8 %] | 8.6 % | 47.1 % [25.9 %, 65.9 %] (34) | 45.3 % [30.6 %, 56.8 %] | 48.0 % [33.3 %, 60.0 %] | 35.4 [31.3, 48.9] | 59.5 % [53.6 %, 65.2 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.886 [0.859, 0.916] | 0.861 [0.832, 0.898] | 0.883 [0.854, 0.914] | 0.201 [0.147, 0.266] | 0.0113 |
| 10 | 0.886 [0.860, 0.916] | 0.864 [0.837, 0.900] | 0.883 [0.854, 0.913] | 0.280 [0.220, 0.350] | 0.0197 |
| 20 | 0.890 [0.865, 0.918] | 0.870 [0.843, 0.904] | 0.889 [0.862, 0.917] | 0.353 [0.285, 0.424] | 0.0318 |
| 30 | 0.888 [0.862, 0.917] | 0.868 [0.841, 0.902] | 0.888 [0.861, 0.916] | 0.386 [0.314, 0.459] | 0.0405 |
| 60 | 0.887 [0.860, 0.915] | 0.872 [0.844, 0.904] | 0.888 [0.860, 0.917] | 0.456 [0.373, 0.533] | 0.0607 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.766 [0.592, 0.964] | 0.847 [0.799, 0.906] | 25.0 % [0.0 %, 55.6 %] | 68.8 % [38.9 %, 100.0 %] | 0.52 [0.00, 1.58] | 16 |
| source:xd_youtube | 0.940 [0.886, 0.973] | 0.895 [0.841, 0.937] | 30.0 % [13.8 %, 50.0 %] | 33.3 % [17.2 %, 54.5 %] | 1.30 [0.48, 2.41] | 30 |
| source:ucf | 0.891 [0.809, 0.961] | 0.903 [0.852, 0.948] | 10.0 % [0.0 %, 25.0 %] | 10.0 % [0.0 %, 25.0 %] | 0.17 [0.00, 0.54] | 20 |
| source:ubi | 0.940 [0.802, 1.000] | 0.941 [0.892, 0.978] | 22.2 % [0.0 %, 50.0 %] | 33.3 % [0.0 %, 70.0 %] | 0.21 [0.00, 0.70] | 9 |
| class:interpersonal | 0.894 [0.844, 0.937] | 0.891 [0.864, 0.918] | 20.0 % [10.9 %, 29.5 %] | 32.3 % [18.9 %, 44.1 %] | 0.72 [0.34, 1.15] | 65 |
| class:other_violent | 0.900 [0.715, 0.997] | 0.882 [0.844, 0.944] | 40.0 % [0.0 %, 71.4 %] | 50.0 % [0.0 %, 88.3 %] | 0.72 [0.34, 1.15] | 10 |
| audio:present | 0.887 [0.823, 0.942] | 0.876 [0.844, 0.911] | 28.0 % [15.7 %, 42.6 %] | 44.0 % [26.8 %, 59.3 %] | 0.95 [0.44, 1.63] | 50 |
| audio:absent | 0.906 [0.837, 0.961] | 0.916 [0.873, 0.953] | 12.0 % [0.0 %, 26.7 %] | 16.0 % [3.3 %, 32.0 %] | 0.21 [0.00, 0.54] | 25 |
| onset:official | 0.916 [0.840, 0.976] | 0.887 [0.827, 0.938] | 21.1 % [0.0 %, 39.2 %] | 42.1 % [15.4 %, 66.7 %] | 0.72 [0.34, 1.15] | 19 |
| onset:hivau | 0.880 [0.811, 0.933] | 0.884 [0.858, 0.915] | 23.4 % [12.1 %, 35.6 %] | 31.9 % [16.2 %, 48.6 %] | 0.72 [0.34, 1.15] | 47 |
| onset:ubi | 0.927 [0.804, 0.998] | 0.933 [0.888, 0.968] | 22.2 % [0.0 %, 50.0 %] | 33.3 % [0.0 %, 70.0 %] | 0.72 [0.34, 1.15] | 9 |
| event:first_segment | n/a | n/a | 18.6 % [7.4 %, 31.0 %] | 34.9 % [19.6 %, 50.0 %] | 0.72 [0.34, 1.15] | 43 |
| event:later_segment | n/a | n/a | 28.1 % [13.6 %, 41.9 %] | 34.4 % [18.2 %, 50.0 %] | 0.72 [0.34, 1.15] | 32 |

Generated 2026-10-18T20:37:22 by `anticipation/eval/va30.py` (git 39ae77b6b3).
