# VA30 report: b_time_source_fit__teacher_subset

Fold file `folds_v2` (content sha256 `8ebdf6ec2e54`), seeds [0], headline seed 0. Test universe: 1031 videos (693 violent, 338 clean negatives = 15.17 h), 198 eligible30 events (73 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 1031 of 6139 universe videos. Not directly comparable with full-universe results.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.755 [0.691, 0.810] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.751 [0.686, 0.804] |
| Far-window AUROC (clean, source x time matched negatives) | 0.490 [0.411, 0.563] |
| Far-window AUROC (clean, uncensored negatives) | 0.752 [0.691, 0.804] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.705 [0.641, 0.760] |
| AUROC / AP, H = 30 s (base rate 0.146) | 0.847 [0.816, 0.873] / 0.445 [0.403, 0.491] |
| Detection AUROC (in-violence vs clean negatives) | 0.593 [0.545, 0.639] |
| Brier P(onset within 30 s) (climatology 0.1436), ECE | 0.1328 [0.1180, 0.1523], 0.0706 |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.0 % [0.0 %, 0.0 %] | [0.0, 0.0] % | 0.0 % [0.0 %, 0.0 %] | 0.0 % | 0.0 % [0.0 %, 0.0 %] (90) | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | n/a | 0.0 % [0.0 %, 0.0 %] |
| 2.0 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.0 % [0.0 %, 0.0 %] | [0.0, 0.0] % | 0.0 % [0.0 %, 0.0 %] | 0.0 % | 0.0 % [0.0 %, 0.0 %] (90) | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | n/a | 0.0 % [0.0 %, 0.0 %] |
| 5.0 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.0 % [0.0 %, 0.0 %] | [0.0, 0.0] % | 0.0 % [0.0 %, 0.0 %] | 0.0 % | 0.0 % [0.0 %, 0.0 %] (90) | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | n/a | 0.0 % [0.0 %, 0.0 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.878 [0.856, 0.897] | 0.499 [0.467, 0.530] | 0.799 [0.770, 0.823] | 0.219 [0.191, 0.255] | 0.0421 |
| 10 | 0.870 [0.846, 0.891] | 0.499 [0.467, 0.531] | 0.789 [0.759, 0.816] | 0.311 [0.277, 0.354] | 0.0721 |
| 20 | 0.857 [0.830, 0.881] | 0.498 [0.462, 0.534] | 0.781 [0.749, 0.810] | 0.400 [0.360, 0.447] | 0.1148 |
| 30 | 0.847 [0.816, 0.873] | 0.498 [0.462, 0.536] | 0.775 [0.741, 0.805] | 0.445 [0.403, 0.491] | 0.1457 |
| 60 | 0.828 [0.789, 0.863] | 0.497 [0.455, 0.541] | 0.766 [0.725, 0.802] | 0.527 [0.484, 0.573] | 0.2110 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.620 [0.466, 0.771] | 0.831 [0.779, 0.877] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 24 |
| source:xd_youtube | 0.889 [0.807, 0.952] | 0.881 [0.835, 0.918] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 30 |
| source:ucf | 0.586 [0.467, 0.686] | 0.796 [0.709, 0.856] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 144 |
| class:interpersonal | 0.744 [0.672, 0.803] | 0.833 [0.798, 0.863] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 173 |
| class:other_violent | 0.826 [0.705, 0.931] | 0.910 [0.865, 0.943] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 25 |
| audio:present | 0.777 [0.703, 0.838] | 0.849 [0.812, 0.880] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 101 |
| audio:absent | 0.559 [0.405, 0.682] | 0.817 [0.709, 0.875] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 97 |
| onset:official | 0.753 [0.668, 0.828] | 0.833 [0.791, 0.870] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 75 |
| onset:hivau | 0.756 [0.672, 0.830] | 0.859 [0.826, 0.890] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 123 |
| event:first_segment | n/a | n/a | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 120 |
| event:later_segment | n/a | n/a | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 78 |

Generated 2026-10-14T20:37:47 by `anticipation/eval/va30.py` (git 9326588dc9).
