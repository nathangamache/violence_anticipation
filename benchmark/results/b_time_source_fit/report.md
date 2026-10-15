# VA30 report: b_time_source_fit

Fold file `folds_v2` (content sha256 `8ebdf6ec2e54`), seeds [0], headline seed 0. Test universe: 6139 videos (2129 violent, 4010 clean negatives = 303.53 h), 551 eligible30 events (218 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.858 [0.829, 0.883] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.857 [0.828, 0.882] |
| Far-window AUROC (clean, source x time matched negatives) | 0.496 [0.462, 0.528] |
| Far-window AUROC (clean, uncensored negatives) | 0.831 [0.802, 0.858] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.786 [0.751, 0.819] |
| AUROC / AP, H = 30 s (base rate 0.022) | 0.903 [0.886, 0.918] / 0.149 [0.115, 0.196] |
| Detection AUROC (in-violence vs clean negatives) | 0.685 [0.640, 0.726] |
| Brier P(onset within 30 s) (climatology 0.0348), ECE | 0.0347 [0.0305, 0.0395], 0.0224 |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.0 % [0.0 %, 0.0 %] | [0.0, 0.0] % | 0.0 % [0.0 %, 0.0 %] | 0.0 % | 0.0 % [0.0 %, 0.0 %] (254) | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | n/a | 0.0 % [0.0 %, 0.0 %] |
| 2.0 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.0 % [0.0 %, 0.0 %] | [0.0, 0.0] % | 0.0 % [0.0 %, 0.0 %] | 0.0 % | 0.0 % [0.0 %, 0.0 %] (254) | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | n/a | 0.0 % [0.0 %, 0.0 %] |
| 5.0 | 3.58 [3.19, 3.98] | 3.47 [3.09, 3.85] | 17.1 % [13.9 %, 20.4 %] | [11.2, 20.3] % | 8.6 % [7.7 %, 9.5 %] | 17.1 % | 0.0 % [0.0 %, 0.0 %] (254) | 17.1 % [13.9 %, 20.4 %] | 17.1 % [13.9 %, 20.4 %] | 41.7 [39.1, 49.0] | 18.8 % [17.4 %, 20.1 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.912 [0.898, 0.925] | 0.499 [0.483, 0.515] | 0.839 [0.820, 0.857] | 0.054 [0.040, 0.075] | 0.0064 |
| 10 | 0.910 [0.895, 0.922] | 0.499 [0.483, 0.514] | 0.836 [0.817, 0.856] | 0.088 [0.066, 0.121] | 0.0111 |
| 20 | 0.906 [0.890, 0.919] | 0.498 [0.483, 0.515] | 0.833 [0.812, 0.854] | 0.127 [0.097, 0.170] | 0.0178 |
| 30 | 0.903 [0.886, 0.918] | 0.498 [0.482, 0.515] | 0.832 [0.810, 0.853] | 0.149 [0.115, 0.196] | 0.0224 |
| 60 | 0.901 [0.882, 0.917] | 0.498 [0.479, 0.516] | 0.834 [0.809, 0.857] | 0.192 [0.149, 0.250] | 0.0312 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.699 [0.612, 0.772] | 0.795 [0.730, 0.841] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 143 |
| source:xd_youtube | 0.779 [0.719, 0.837] | 0.869 [0.841, 0.894] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 162 |
| source:ucf | 0.901 [0.854, 0.934] | 0.930 [0.904, 0.950] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 144 |
| source:ubi | 0.923 [0.891, 0.949] | 0.936 [0.919, 0.950] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 102 |
| class:interpersonal | 0.847 [0.815, 0.875] | 0.893 [0.874, 0.909] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 472 |
| class:other_violent | 0.909 [0.875, 0.938] | 0.942 [0.926, 0.956] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 79 |
| audio:present | 0.821 [0.774, 0.864] | 0.883 [0.854, 0.907] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 352 |
| audio:absent | 0.896 [0.860, 0.924] | 0.929 [0.912, 0.943] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 199 |
| onset:official | 0.861 [0.810, 0.904] | 0.900 [0.876, 0.922] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 75 |
| onset:hivau | 0.851 [0.817, 0.881] | 0.904 [0.886, 0.919] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 374 |
| onset:ubi | 0.894 [0.858, 0.923] | 0.906 [0.886, 0.923] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 102 |
| event:first_segment | n/a | n/a | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 312 |
| event:later_segment | n/a | n/a | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 239 |

Generated 2026-10-14T20:35:55 by `anticipation/eval/va30.py` (git 9326588dc9).
