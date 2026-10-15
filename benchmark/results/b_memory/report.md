# VA30 report: b_memory

Fold file `folds_v2` (content sha256 `8ebdf6ec2e54`), seeds [0], headline seed 0. Test universe: 6139 videos (2129 violent, 4010 clean negatives = 303.53 h), 551 eligible30 events (218 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.500 [0.500, 0.500] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.500 [0.500, 0.500] |
| Far-window AUROC (clean, source x time matched negatives) | 0.500 [0.500, 0.500] |
| Far-window AUROC (clean, uncensored negatives) | 0.500 [0.500, 0.500] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.722 [0.698, 0.748] |
| AUROC / AP, H = 30 s (base rate 0.022) | 0.500 [0.500, 0.500] / 0.022 [0.020, 0.025] |
| Detection AUROC (in-violence vs clean negatives) | 1.000 [1.000, 1.000] |
| Brier P(onset within 30 s) (climatology 0.0348), ECE | 0.0480 [0.0404, 0.0568], 0.0480 |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.0 % [0.0 %, 0.0 %] | [0.0, 0.0] % | 0.0 % [0.0 %, 0.0 %] | 0.0 % | 0.0 % [0.0 %, 0.0 %] (254) | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | n/a | 86.8 % [85.4 %, 88.2 %] |
| 2.0 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.0 % [0.0 %, 0.0 %] | [0.0, 0.0] % | 0.0 % [0.0 %, 0.0 %] | 0.0 % | 0.0 % [0.0 %, 0.0 %] (254) | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | n/a | 86.8 % [85.4 %, 88.2 %] |
| 5.0 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.0 % [0.0 %, 0.0 %] | [0.0, 0.0] % | 0.0 % [0.0 %, 0.0 %] | 0.0 % | 0.0 % [0.0 %, 0.0 %] (254) | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | n/a | 86.8 % [85.4 %, 88.2 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.685 [0.675, 0.696] | 0.006 [0.006, 0.007] | 0.0064 |
| 10 | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.688 [0.677, 0.699] | 0.011 [0.010, 0.012] | 0.0111 |
| 20 | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.693 [0.682, 0.704] | 0.018 [0.016, 0.020] | 0.0178 |
| 30 | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.695 [0.683, 0.708] | 0.022 [0.020, 0.025] | 0.0224 |
| 60 | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.701 [0.687, 0.715] | 0.031 [0.027, 0.035] | 0.0312 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 143 |
| source:xd_youtube | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 162 |
| source:ucf | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 144 |
| source:ubi | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 102 |
| class:interpersonal | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 472 |
| class:other_violent | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 79 |
| audio:present | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 352 |
| audio:absent | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 199 |
| onset:official | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 75 |
| onset:hivau | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 374 |
| onset:ubi | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 102 |
| event:first_segment | n/a | n/a | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 312 |
| event:later_segment | n/a | n/a | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 239 |

Generated 2026-10-14T20:37:19 by `anticipation/eval/va30.py` (git 9326588dc9).
