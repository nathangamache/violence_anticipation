# VA30 report: va30

Fold file `folds_v2_without_confirmation` (content sha256 `5c5149555ca8`), seeds [0], headline seed 0. Test universe: 1491 videos (158 violent, 1333 clean negatives = 42.54 h), 201 eligible30 events (85 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 1491 of 4916 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 653 catalog clean negatives (218 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 42.54 of 131.74 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.597 [0.531, 0.657] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.590 [0.527, 0.648] |
| Far-window AUROC (clean, source x time matched negatives) | 0.526 [0.460, 0.587] |
| Far-window AUROC (clean, uncensored negatives) | 0.566 [0.502, 0.625] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.630 [0.572, 0.687] |
| AUROC / AP, H = 30 s (base rate 0.028) | 0.618 [0.568, 0.670] / 0.053 [0.039, 0.079] |
| Detection AUROC (in-violence vs clean negatives) | 0.565 [0.496, 0.639] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 1.03 [0.71, 1.39] | 1.02 [0.71, 1.37] | 6.0 % [3.1 %, 9.4 %] | [3.3, 10.8] % | 2.6 % [1.7 %, 3.4 %] | 5.4 % | 9.7 % [4.2 %, 15.8 %] (103) | 10.4 % [5.5 %, 16.2 %] | 11.9 % [6.6 %, 17.8 %] | 29.5 [19.0, 47.8] | 11.2 % [6.4 %, 16.2 %] |
| 2.0 | 2.19 [1.73, 2.69] | 2.19 [1.74, 2.68] | 8.0 % [4.2 %, 11.9 %] | [4.6, 11.8] % | 5.3 % [4.2 %, 6.5 %] | 7.0 % | 11.7 % [5.2 %, 17.9 %] (103) | 11.4 % [6.9 %, 16.1 %] | 12.9 % [7.9 %, 18.0 %] | 31.4 [27.5, 52.9] | 15.0 % [9.4 %, 21.0 %] |
| 5.0 | 5.31 [4.62, 6.03] | 5.22 [4.57, 5.90] | 9.0 % [5.5 %, 13.1 %] | [5.1, 14.1] % | 12.4 % [10.9 %, 14.0 %] | 11.3 % | 9.7 % [4.1 %, 16.1 %] (103) | 14.9 % [10.6 %, 19.7 %] | 17.9 % [13.0 %, 22.8 %] | 29.5 [16.0, 43.2] | 20.4 % [14.7 %, 26.4 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.584 [0.530, 0.637] | 0.549 [0.494, 0.607] | 0.575 [0.522, 0.634] | 0.007 [0.005, 0.017] | 0.0043 |
| 10 | 0.589 [0.537, 0.642] | 0.551 [0.497, 0.609] | 0.578 [0.525, 0.635] | 0.015 [0.011, 0.026] | 0.0087 |
| 20 | 0.608 [0.557, 0.661] | 0.558 [0.505, 0.618] | 0.597 [0.546, 0.651] | 0.032 [0.024, 0.050] | 0.0180 |
| 30 | 0.618 [0.568, 0.670] | 0.554 [0.501, 0.612] | 0.608 [0.558, 0.661] | 0.053 [0.039, 0.079] | 0.0280 |
| 60 | 0.610 [0.558, 0.659] | 0.541 [0.489, 0.596] | 0.612 [0.563, 0.665] | 0.095 [0.070, 0.137] | 0.0557 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:ucf | 0.612 [0.533, 0.683] | 0.579 [0.516, 0.641] | 5.2 % [1.7 %, 9.5 %] | 7.0 % [2.5 %, 12.3 %] | 1.98 [1.34, 2.57] | 115 |
| source:ubi | 0.549 [0.430, 0.675] | 0.681 [0.592, 0.762] | 11.6 % [5.2 %, 18.6 %] | 20.9 % [12.7 %, 29.6 %] | 2.44 [1.80, 3.20] | 86 |
| class:interpersonal | 0.601 [0.533, 0.665] | 0.628 [0.576, 0.680] | 8.4 % [4.5 %, 12.7 %] | 13.7 % [8.5 %, 19.0 %] | 2.19 [1.73, 2.69] | 190 |
| class:other_violent | 0.559 [0.299, 0.750] | 0.505 [0.262, 0.731] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 2.19 [1.73, 2.69] | 11 |
| audio:present | 0.608 [0.482, 0.729] | 0.524 [0.424, 0.640] | 9.8 % [2.4 %, 18.8 %] | 12.2 % [2.5 %, 23.7 %] | 0.89 [0.19, 1.70] | 41 |
| audio:absent | 0.587 [0.509, 0.661] | 0.637 [0.580, 0.691] | 7.5 % [3.5 %, 12.1 %] | 13.1 % [7.5 %, 18.8 %] | 2.38 [1.86, 2.93] | 160 |
| onset:official | 0.621 [0.441, 0.770] | 0.594 [0.431, 0.735] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 2.19 [1.73, 2.69] | 15 |
| onset:hivau | 0.619 [0.538, 0.693] | 0.592 [0.522, 0.661] | 6.0 % [1.9 %, 10.7 %] | 8.0 % [2.8 %, 13.8 %] | 2.19 [1.73, 2.69] | 100 |
| onset:ubi | 0.543 [0.423, 0.671] | 0.676 [0.584, 0.757] | 11.6 % [5.2 %, 18.6 %] | 20.9 % [12.7 %, 29.6 %] | 2.19 [1.73, 2.69] | 86 |
| event:first_segment | n/a | n/a | 4.9 % [1.0 %, 9.7 %] | 9.7 % [4.4 %, 15.6 %] | 2.19 [1.73, 2.69] | 103 |
| event:later_segment | n/a | n/a | 11.2 % [4.9 %, 17.5 %] | 16.3 % [8.3 %, 23.3 %] | 2.19 [1.73, 2.69] | 98 |

Generated 2026-10-15T12:56:06 by `anticipation/eval/va30.py` (git 00cd04bb59).
