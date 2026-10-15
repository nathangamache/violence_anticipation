# VA30 report: b_time_fit

Fold file `folds_v2` (content sha256 `8ebdf6ec2e54`), seeds [0], headline seed 0. Test universe: 6139 videos (2129 violent, 4010 clean negatives = 303.53 h), 551 eligible30 events (218 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.863 [0.837, 0.887] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.862 [0.836, 0.886] |
| Far-window AUROC (clean, source x time matched negatives) | 0.498 [0.466, 0.532] |
| Far-window AUROC (clean, uncensored negatives) | 0.840 [0.812, 0.866] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.779 [0.745, 0.812] |
| AUROC / AP, H = 30 s (base rate 0.022) | 0.912 [0.900, 0.923] / 0.167 [0.157, 0.178] |
| Detection AUROC (in-violence vs clean negatives) | 0.705 [0.645, 0.754] |
| Brier P(onset within 30 s) (climatology 0.0348), ECE | 0.0332 [0.0297, 0.0369], 0.0182 |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.0 % [0.0 %, 0.0 %] | [0.0, 0.0] % | 0.0 % [0.0 %, 0.0 %] | 0.0 % | 0.0 % [0.0 %, 0.0 %] (254) | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | n/a | 0.0 % [0.0 %, 0.0 %] |
| 2.0 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.0 % [0.0 %, 0.0 %] | [0.0, 0.0] % | 0.0 % [0.0 %, 0.0 %] | 0.0 % | 0.0 % [0.0 %, 0.0 %] (254) | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | n/a | 0.0 % [0.0 %, 0.0 %] |
| 5.0 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.0 % [0.0 %, 0.0 %] | [0.0, 0.0] % | 0.0 % [0.0 %, 0.0 %] | 0.0 % | 0.0 % [0.0 %, 0.0 %] (254) | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | n/a | 0.0 % [0.0 %, 0.0 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.923 [0.914, 0.932] | 0.500 [0.486, 0.515] | 0.843 [0.827, 0.860] | 0.060 [0.055, 0.066] | 0.0064 |
| 10 | 0.920 [0.911, 0.929] | 0.500 [0.485, 0.515] | 0.840 [0.823, 0.858] | 0.098 [0.091, 0.106] | 0.0111 |
| 20 | 0.915 [0.905, 0.926] | 0.498 [0.483, 0.513] | 0.835 [0.816, 0.855] | 0.141 [0.132, 0.151] | 0.0178 |
| 30 | 0.912 [0.900, 0.923] | 0.498 [0.482, 0.513] | 0.833 [0.813, 0.853] | 0.167 [0.157, 0.178] | 0.0224 |
| 60 | 0.907 [0.893, 0.921] | 0.497 [0.481, 0.515] | 0.831 [0.808, 0.854] | 0.213 [0.201, 0.227] | 0.0312 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.779 [0.723, 0.827] | 0.851 [0.829, 0.870] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 143 |
| source:xd_youtube | 0.785 [0.727, 0.838] | 0.876 [0.849, 0.900] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 162 |
| source:ucf | 0.907 [0.865, 0.939] | 0.932 [0.906, 0.951] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 144 |
| source:ubi | 0.929 [0.898, 0.954] | 0.941 [0.923, 0.955] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 102 |
| class:interpersonal | 0.856 [0.827, 0.881] | 0.907 [0.894, 0.920] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 472 |
| class:other_violent | 0.894 [0.860, 0.923] | 0.930 [0.916, 0.942] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 79 |
| audio:present | 0.827 [0.784, 0.866] | 0.894 [0.875, 0.914] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 352 |
| audio:absent | 0.903 [0.869, 0.930] | 0.932 [0.917, 0.945] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 199 |
| onset:official | 0.878 [0.831, 0.919] | 0.916 [0.898, 0.933] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 75 |
| onset:hivau | 0.857 [0.827, 0.884] | 0.913 [0.899, 0.925] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 374 |
| onset:ubi | 0.875 [0.831, 0.913] | 0.904 [0.882, 0.924] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 102 |
| event:first_segment | n/a | n/a | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 312 |
| event:later_segment | n/a | n/a | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 239 |

Generated 2026-10-14T20:34:18 by `anticipation/eval/va30.py` (git 9326588dc9).
