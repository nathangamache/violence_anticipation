# VA30 report: b_teacher_old

Fold file `folds_v2` (content sha256 `8ebdf6ec2e54`), seeds [0], headline seed 0. Test universe: 1031 videos (693 violent, 338 clean negatives = 15.17 h), 198 eligible30 events (73 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 1031 of 6139 universe videos. Not directly comparable with full-universe results.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.669 [0.627, 0.706] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.658 [0.617, 0.695] |
| Far-window AUROC (clean, source x time matched negatives) | 0.669 [0.625, 0.709] |
| Far-window AUROC (clean, uncensored negatives) | 0.662 [0.620, 0.700] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.687 [0.651, 0.720] |
| AUROC / AP, H = 30 s (base rate 0.146) | 0.673 [0.652, 0.694] / 0.314 [0.272, 0.365] |
| Detection AUROC (in-violence vs clean negatives) | 0.825 [0.805, 0.843] |
| Brier P(onset within 30 s) (climatology 0.1436), ECE | 0.1678 [0.1496, 0.1885], 0.1622 |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.0 % [0.0 %, 0.0 %] | [0.0, 5.0] % | 0.0 % [0.0 %, 0.0 %] | 0.0 % | 0.0 % [0.0 %, 0.0 %] (90) | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | n/a | 0.0 % [0.0 %, 0.0 %] |
| 2.0 | 0.47 [0.14, 0.84] | 0.62 [0.27, 0.97] | 3.0 % [0.9 %, 5.6 %] | [0.0, 7.8] % | 1.2 % [0.3 %, 2.1 %] | 5.3 % | 2.2 % [0.0 %, 5.7 %] (90) | 4.5 % [2.0 %, 7.5 %] | 4.5 % [2.0 %, 7.5 %] | 43.0 [26.9, 56.5] | 3.1 % [2.1 %, 4.2 %] |
| 5.0 | 3.24 [2.24, 4.39] | 4.21 [3.22, 5.38] | 10.6 % [6.4 %, 15.1 %] | [3.9, 17.8] % | 7.8 % [5.5 %, 10.4 %] | 34.2 % | 11.1 % [5.0 %, 17.9 %] (90) | 16.2 % [10.6 %, 22.1 %] | 17.2 % [11.8 %, 23.0 %] | 36.7 [28.8, 47.3] | 24.1 % [21.5 %, 26.9 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.685 [0.664, 0.706] | 0.673 [0.648, 0.697] | 0.671 [0.652, 0.690] | 0.140 [0.113, 0.176] | 0.0421 |
| 10 | 0.678 [0.657, 0.699] | 0.664 [0.640, 0.690] | 0.671 [0.650, 0.689] | 0.201 [0.168, 0.245] | 0.0721 |
| 20 | 0.675 [0.656, 0.696] | 0.660 [0.636, 0.686] | 0.673 [0.652, 0.692] | 0.271 [0.231, 0.318] | 0.1148 |
| 30 | 0.673 [0.652, 0.694] | 0.655 [0.629, 0.682] | 0.674 [0.651, 0.694] | 0.314 [0.272, 0.365] | 0.1457 |
| 60 | 0.671 [0.649, 0.695] | 0.650 [0.619, 0.680] | 0.678 [0.655, 0.702] | 0.399 [0.352, 0.457] | 0.2110 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.757 [0.654, 0.855] | 0.744 [0.712, 0.777] | 12.5 % [0.0 %, 28.6 %] | 16.7 % [3.7 %, 36.9 %] | 1.24 [0.40, 2.33] | 24 |
| source:xd_youtube | 0.603 [0.530, 0.656] | 0.645 [0.595, 0.692] | 0.0 % [0.0 %, 0.0 %] | 6.7 % [0.0 %, 17.2 %] | 0.12 [0.00, 0.39] | 30 |
| source:ucf | 0.671 [0.614, 0.723] | 0.615 [0.574, 0.663] | 2.1 % [0.0 %, 4.8 %] | 2.1 % [0.0 %, 4.8 %] | 0.00 [0.00, 0.00] | 144 |
| class:interpersonal | 0.668 [0.622, 0.711] | 0.664 [0.642, 0.686] | 2.3 % [0.5 %, 4.9 %] | 3.5 % [1.1 %, 6.4 %] | 0.47 [0.14, 0.84] | 173 |
| class:other_violent | 0.674 [0.571, 0.785] | 0.721 [0.681, 0.763] | 8.0 % [0.0 %, 20.0 %] | 12.0 % [0.0 %, 26.1 %] | 0.47 [0.14, 0.84] | 25 |
| audio:present | 0.687 [0.632, 0.744] | 0.700 [0.673, 0.726] | 5.0 % [1.0 %, 9.5 %] | 7.9 % [3.0 %, 13.4 %] | 0.50 [0.15, 0.89] | 101 |
| audio:absent | 0.648 [0.578, 0.716] | 0.584 [0.540, 0.642] | 1.0 % [0.0 %, 3.4 %] | 1.0 % [0.0 %, 3.4 %] | 0.00 [0.00, 0.00] | 97 |
| onset:official | 0.672 [0.614, 0.723] | 0.704 [0.678, 0.731] | 5.3 % [1.1 %, 10.9 %] | 9.3 % [3.7 %, 16.7 %] | 0.47 [0.14, 0.84] | 75 |
| onset:hivau | 0.666 [0.610, 0.724] | 0.654 [0.627, 0.681] | 1.6 % [0.0 %, 4.2 %] | 1.6 % [0.0 %, 4.2 %] | 0.47 [0.14, 0.84] | 123 |
| event:first_segment | n/a | n/a | 4.2 % [0.8 %, 8.3 %] | 5.0 % [1.5 %, 9.2 %] | 0.47 [0.14, 0.84] | 120 |
| event:later_segment | n/a | n/a | 1.3 % [0.0 %, 4.1 %] | 3.8 % [0.0 %, 8.8 %] | 0.47 [0.14, 0.84] | 78 |

Generated 2026-10-14T20:37:55 by `anticipation/eval/va30.py` (git 9326588dc9).
