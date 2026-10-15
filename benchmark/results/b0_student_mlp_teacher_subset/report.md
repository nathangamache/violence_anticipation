# VA30 report: b0_student_mlp_teacher_subset

Fold file `folds_v2` (content sha256 `8ebdf6ec2e54`), seeds [0], headline seed 0. Test universe: 1031 videos (693 violent, 338 clean negatives = 15.17 h), 198 eligible30 events (73 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 1031 of 6139 universe videos. Not directly comparable with full-universe results.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.681 [0.625, 0.735] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.672 [0.620, 0.722] |
| Far-window AUROC (clean, source x time matched negatives) | 0.712 [0.657, 0.760] |
| Far-window AUROC (clean, uncensored negatives) | 0.679 [0.624, 0.732] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.697 [0.649, 0.748] |
| AUROC / AP, H = 30 s (base rate 0.146) | 0.671 [0.641, 0.702] / 0.294 [0.256, 0.349] |
| Detection AUROC (in-violence vs clean negatives) | 0.810 [0.781, 0.838] |
| Brier P(onset within 30 s) (climatology 0.1436), ECE | 0.1598 [0.1413, 0.1816], 0.1534 |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 0.54 [0.14, 1.06] | 1.44 [0.91, 2.04] | 11.6 % [7.6 %, 16.1 %] | [5.8, 14.8] % | 1.3 % [0.4 %, 2.6 %] | 6.0 % | 11.1 % [5.4 %, 17.3 %] (90) | 21.7 % [15.8 %, 27.6 %] | 23.7 % [17.5 %, 30.1 %] | 29.8 [23.4, 38.3] | 47.4 % [43.9 %, 50.7 %] |
| 2.0 | 0.54 [0.14, 1.06] | 1.44 [0.91, 2.04] | 11.6 % [7.6 %, 16.1 %] | [6.7, 16.9] % | 1.3 % [0.4 %, 2.6 %] | 6.0 % | 11.1 % [5.4 %, 17.3 %] (90) | 21.7 % [15.8 %, 27.6 %] | 23.7 % [17.5 %, 30.1 %] | 29.8 [23.4, 38.3] | 47.4 % [43.9 %, 50.7 %] |
| 5.0 | 1.82 [1.10, 2.60] | 2.56 [1.86, 3.45] | 13.6 % [8.9 %, 18.7 %] | [10.4, 24.4] % | 4.4 % [2.7 %, 6.3 %] | 19.4 % | 8.9 % [3.6 %, 14.7 %] (90) | 22.7 % [16.7 %, 28.8 %] | 24.7 % [18.3 %, 30.9 %] | 31.8 [26.0, 40.9] | 50.4 % [46.9 %, 53.7 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.677 [0.650, 0.704] | 0.658 [0.629, 0.688] | 0.674 [0.647, 0.701] | 0.141 [0.114, 0.177] | 0.0421 |
| 10 | 0.674 [0.644, 0.701] | 0.655 [0.625, 0.686] | 0.676 [0.647, 0.702] | 0.194 [0.164, 0.238] | 0.0721 |
| 20 | 0.674 [0.646, 0.703] | 0.656 [0.623, 0.687] | 0.679 [0.651, 0.706] | 0.258 [0.223, 0.307] | 0.1148 |
| 30 | 0.671 [0.641, 0.702] | 0.651 [0.616, 0.684] | 0.680 [0.651, 0.707] | 0.294 [0.256, 0.349] | 0.1457 |
| 60 | 0.671 [0.639, 0.704] | 0.646 [0.607, 0.683] | 0.683 [0.650, 0.713] | 0.372 [0.323, 0.433] | 0.2110 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.791 [0.666, 0.918] | 0.730 [0.669, 0.788] | 37.5 % [17.1 %, 59.1 %] | 66.7 % [44.8 %, 85.0 %] | 1.24 [0.37, 2.34] | 24 |
| source:xd_youtube | 0.652 [0.513, 0.780] | 0.629 [0.559, 0.705] | 13.3 % [2.9 %, 25.0 %] | 26.7 % [9.7 %, 42.3 %] | 0.23 [0.00, 0.82] | 30 |
| source:ucf | 0.714 [0.650, 0.771] | 0.645 [0.571, 0.699] | 6.9 % [3.1 %, 10.9 %] | 16.0 % [10.4 %, 22.1 %] | 0.00 [0.00, 0.00] | 144 |
| class:interpersonal | 0.679 [0.620, 0.737] | 0.670 [0.638, 0.700] | 11.0 % [6.5 %, 15.8 %] | 23.7 % [17.2 %, 29.7 %] | 0.54 [0.14, 1.06] | 173 |
| class:other_violent | 0.692 [0.553, 0.831] | 0.686 [0.627, 0.746] | 16.0 % [3.7 %, 30.8 %] | 24.0 % [5.6 %, 43.5 %] | 0.54 [0.14, 1.06] | 25 |
| audio:present | 0.689 [0.615, 0.760] | 0.676 [0.639, 0.711] | 16.8 % [10.2 %, 23.9 %] | 30.7 % [21.7 %, 39.3 %] | 0.57 [0.15, 1.14] | 101 |
| audio:absent | 0.721 [0.647, 0.790] | 0.653 [0.543, 0.723] | 6.2 % [2.1 %, 11.4 %] | 16.5 % [8.6 %, 24.5 %] | 0.00 [0.00, 0.00] | 97 |
| onset:official | 0.698 [0.624, 0.780] | 0.697 [0.658, 0.736] | 20.0 % [11.2 %, 29.8 %] | 37.3 % [26.2 %, 48.1 %] | 0.54 [0.14, 1.06] | 75 |
| onset:hivau | 0.669 [0.599, 0.739] | 0.656 [0.618, 0.694] | 6.5 % [2.5 %, 10.5 %] | 15.4 % [9.2 %, 22.1 %] | 0.54 [0.14, 1.06] | 123 |
| event:first_segment | n/a | n/a | 12.5 % [7.0 %, 18.5 %] | 23.3 % [15.8 %, 30.8 %] | 0.54 [0.14, 1.06] | 120 |
| event:later_segment | n/a | n/a | 10.3 % [4.3 %, 17.1 %] | 24.4 % [15.2 %, 34.4 %] | 0.54 [0.14, 1.06] | 78 |

Generated 2026-10-14T21:37:20 by `anticipation/eval/va30.py` (git eb9e51707b).
