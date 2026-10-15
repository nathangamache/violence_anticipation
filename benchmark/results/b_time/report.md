# VA30 report: b_time

Fold file `folds_v2` (content sha256 `8ebdf6ec2e54`), seeds [0], headline seed 0. Test universe: 6139 videos (2129 violent, 4010 clean negatives = 303.53 h), 551 eligible30 events (218 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.135 [0.112, 0.161] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.136 [0.112, 0.161] |
| Far-window AUROC (clean, source x time matched negatives) | 0.499 [0.465, 0.533] |
| Far-window AUROC (clean, uncensored negatives) | 0.157 [0.132, 0.185] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.216 [0.185, 0.250] |
| AUROC / AP, H = 30 s (base rate 0.022) | 0.085 [0.074, 0.097] / 0.012 [0.011, 0.014] |
| Detection AUROC (in-violence vs clean negatives) | 0.294 [0.245, 0.354] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 0.95 [0.85, 1.04] | 0.94 [0.84, 1.03] | 0.2 % [0.0 %, 0.6 %] | [0.0, 1.5] % | 2.3 % [2.1 %, 2.6 %] | 4.5 % | 0.4 % [0.0 %, 1.2 %] (254) | 0.2 % [0.0 %, 0.6 %] | 0.4 % [0.0 %, 0.9 %] | 28.3 [9.6, 47.0] | 0.6 % [0.3 %, 1.0 %] |
| 2.0 | 1.84 [1.66, 2.01] | 1.86 [1.68, 2.02] | 2.2 % [1.0 %, 3.5 %] | [0.7, 3.1] % | 4.5 % [4.1 %, 4.9 %] | 8.8 % | 4.7 % [2.3 %, 7.6 %] (254) | 2.7 % [1.5 %, 4.1 %] | 2.9 % [1.6 %, 4.3 %] | 51.2 [29.5, 88.6] | 2.7 % [2.1 %, 3.5 %] |
| 5.0 | 4.18 [3.77, 4.57] | 4.32 [3.91, 4.71] | 4.5 % [3.0 %, 6.3 %] | [2.9, 6.8] % | 9.9 % [9.0 %, 10.8 %] | 19.8 % | 9.8 % [6.6 %, 13.6 %] (254) | 6.4 % [4.6 %, 8.6 %] | 8.2 % [6.2 %, 10.4 %] | 32.7 [24.8, 53.8] | 8.8 % [7.7 %, 10.0 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.074 [0.066, 0.083] | 0.497 [0.482, 0.511] | 0.155 [0.138, 0.170] | 0.003 [0.003, 0.004] | 0.0064 |
| 10 | 0.077 [0.068, 0.087] | 0.497 [0.483, 0.511] | 0.157 [0.140, 0.174] | 0.006 [0.005, 0.007] | 0.0111 |
| 20 | 0.082 [0.072, 0.092] | 0.497 [0.482, 0.512] | 0.162 [0.143, 0.180] | 0.010 [0.008, 0.011] | 0.0178 |
| 30 | 0.085 [0.074, 0.097] | 0.498 [0.482, 0.513] | 0.164 [0.144, 0.184] | 0.012 [0.011, 0.014] | 0.0224 |
| 60 | 0.090 [0.077, 0.104] | 0.498 [0.480, 0.515] | 0.165 [0.143, 0.188] | 0.017 [0.015, 0.019] | 0.0312 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.214 [0.168, 0.270] | 0.143 [0.124, 0.165] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 1.32 [1.01, 1.61] | 143 |
| source:xd_youtube | 0.214 [0.161, 0.272] | 0.121 [0.098, 0.148] | 3.1 % [0.7 %, 5.8 %] | 4.9 % [1.9 %, 8.3 %] | 2.37 [2.11, 2.61] | 162 |
| source:ucf | 0.091 [0.060, 0.133] | 0.066 [0.047, 0.092] | 2.8 % [0.6 %, 5.5 %] | 2.8 % [0.6 %, 5.5 %] | 1.48 [1.15, 1.89] | 144 |
| source:ubi | 0.068 [0.044, 0.097] | 0.057 [0.044, 0.074] | 2.9 % [0.0 %, 6.7 %] | 3.9 % [0.9 %, 7.6 %] | 1.90 [1.68, 2.15] | 102 |
| class:interpersonal | 0.142 [0.117, 0.169] | 0.090 [0.078, 0.103] | 2.5 % [1.2 %, 4.1 %] | 3.4 % [1.9 %, 5.1 %] | 1.84 [1.66, 2.01] | 472 |
| class:other_violent | 0.103 [0.075, 0.135] | 0.067 [0.056, 0.081] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 1.84 [1.66, 2.01] | 79 |
| audio:present | 0.170 [0.132, 0.213] | 0.102 [0.083, 0.122] | 2.0 % [0.8 %, 3.5 %] | 2.8 % [1.4 %, 4.6 %] | 1.85 [1.58, 2.11] | 352 |
| audio:absent | 0.094 [0.069, 0.127] | 0.065 [0.053, 0.080] | 2.5 % [0.5 %, 4.9 %] | 3.0 % [1.0 %, 5.5 %] | 1.83 [1.64, 2.04] | 199 |
| onset:official | 0.120 [0.080, 0.166] | 0.081 [0.065, 0.100] | 1.3 % [0.0 %, 4.3 %] | 2.7 % [0.0 %, 6.8 %] | 1.84 [1.66, 2.01] | 75 |
| onset:hivau | 0.141 [0.114, 0.170] | 0.085 [0.073, 0.097] | 2.1 % [0.8 %, 3.7 %] | 2.7 % [1.2 %, 4.5 %] | 1.84 [1.66, 2.01] | 374 |
| onset:ubi | 0.121 [0.083, 0.162] | 0.093 [0.073, 0.114] | 2.9 % [0.0 %, 6.7 %] | 3.9 % [0.9 %, 7.6 %] | 1.84 [1.66, 2.01] | 102 |
| event:first_segment | n/a | n/a | 1.3 % [0.3 %, 2.6 %] | 1.3 % [0.3 %, 2.6 %] | 1.84 [1.66, 2.01] | 312 |
| event:later_segment | n/a | n/a | 3.3 % [1.3 %, 5.9 %] | 5.0 % [2.4 %, 7.9 %] | 1.84 [1.66, 2.01] | 239 |

Generated 2026-10-14T20:32:46 by `anticipation/eval/va30.py` (git 9326588dc9).
