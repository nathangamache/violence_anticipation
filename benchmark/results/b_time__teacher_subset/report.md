# VA30 report: b_time__teacher_subset

Fold file `folds_v2` (content sha256 `8ebdf6ec2e54`), seeds [0], headline seed 0. Test universe: 1031 videos (693 violent, 338 clean negatives = 15.17 h), 198 eligible30 events (73 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 1031 of 6139 universe videos. Not directly comparable with full-universe results.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.318 [0.257, 0.387] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.310 [0.253, 0.372] |
| Far-window AUROC (clean, source x time matched negatives) | 0.499 [0.420, 0.587] |
| Far-window AUROC (clean, uncensored negatives) | 0.303 [0.245, 0.367] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.399 [0.337, 0.467] |
| AUROC / AP, H = 30 s (base rate 0.146) | 0.179 [0.151, 0.211] / 0.090 [0.077, 0.107] |
| Detection AUROC (in-violence vs clean negatives) | 0.422 [0.377, 0.468] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 1.21 [0.71, 1.67] | 1.49 [1.01, 1.95] | 0.5 % [0.0 %, 1.6 %] | [0.0, 2.6] % | 3.0 % [1.8 %, 4.1 %] | 13.0 % | 1.1 % [0.0 %, 3.6 %] (90) | 1.0 % [0.0 %, 2.5 %] | 1.5 % [0.0 %, 3.4 %] | 23.7 [1.8, 68.7] | 1.8 % [1.0 %, 3.0 %] |
| 2.0 | 2.36 [1.73, 2.98] | 2.67 [2.08, 3.27] | 1.5 % [0.0 %, 3.4 %] | [0.0, 3.5] % | 5.7 % [4.2 %, 7.2 %] | 25.1 % | 3.3 % [0.0 %, 7.3 %] (90) | 3.0 % [0.9 %, 5.6 %] | 3.5 % [1.1 %, 6.4 %] | 19.8 [9.0, 83.7] | 2.9 % [1.8 %, 4.2 %] |
| 5.0 | 5.06 [4.31, 5.82] | 5.28 [4.53, 6.07] | 4.0 % [1.5 %, 7.1 %] | [1.5, 7.0] % | 11.9 % [10.2 %, 13.6 %] | 45.6 % | 8.9 % [3.5 %, 14.8 %] (90) | 7.1 % [3.8 %, 10.7 %] | 8.1 % [4.6 %, 11.7 %] | 29.4 [22.6, 45.4] | 5.1 % [3.8 %, 6.8 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.138 [0.119, 0.160] | 0.497 [0.467, 0.529] | 0.228 [0.204, 0.255] | 0.025 [0.021, 0.030] | 0.0421 |
| 10 | 0.149 [0.127, 0.172] | 0.497 [0.465, 0.529] | 0.240 [0.215, 0.269] | 0.043 [0.037, 0.051] | 0.0721 |
| 20 | 0.166 [0.141, 0.194] | 0.498 [0.462, 0.534] | 0.255 [0.227, 0.288] | 0.069 [0.060, 0.082] | 0.1148 |
| 30 | 0.179 [0.151, 0.211] | 0.498 [0.461, 0.536] | 0.267 [0.237, 0.303] | 0.090 [0.077, 0.107] | 0.1457 |
| 60 | 0.211 [0.175, 0.251] | 0.498 [0.456, 0.545] | 0.293 [0.256, 0.335] | 0.136 [0.115, 0.165] | 0.2110 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.350 [0.218, 0.497] | 0.164 [0.121, 0.215] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 1.65 [0.71, 2.70] | 24 |
| source:xd_youtube | 0.111 [0.048, 0.200] | 0.111 [0.079, 0.152] | 0.0 % [0.0 %, 0.0 %] | 3.3 % [0.0 %, 11.1 %] | 2.66 [1.81, 3.38] | 30 |
| source:ucf | 0.398 [0.303, 0.520] | 0.199 [0.141, 0.285] | 2.1 % [0.0 %, 4.6 %] | 4.2 % [1.3 %, 7.6 %] | 3.01 [0.79, 5.04] | 144 |
| class:interpersonal | 0.325 [0.258, 0.401] | 0.189 [0.159, 0.223] | 1.7 % [0.0 %, 3.8 %] | 3.5 % [1.1 %, 6.5 %] | 2.36 [1.73, 2.98] | 173 |
| class:other_violent | 0.266 [0.110, 0.400] | 0.137 [0.094, 0.185] | 0.0 % [0.0 %, 0.0 %] | 4.0 % [0.0 %, 13.6 %] | 2.36 [1.73, 2.98] | 25 |
| audio:present | 0.269 [0.197, 0.351] | 0.162 [0.129, 0.199] | 2.0 % [0.0 %, 5.2 %] | 4.0 % [0.9 %, 8.2 %] | 2.36 [1.73, 2.97] | 101 |
| audio:absent | 0.433 [0.308, 0.589] | 0.178 [0.122, 0.286] | 1.0 % [0.0 %, 3.4 %] | 3.1 % [0.0 %, 6.7 %] | 2.40 [0.00, 4.94] | 97 |
| onset:official | 0.257 [0.178, 0.351] | 0.158 [0.126, 0.196] | 0.0 % [0.0 %, 0.0 %] | 4.0 % [0.0 %, 9.4 %] | 2.36 [1.73, 2.98] | 75 |
| onset:hivau | 0.358 [0.283, 0.441] | 0.198 [0.162, 0.237] | 2.4 % [0.0 %, 5.4 %] | 3.3 % [0.7 %, 6.6 %] | 2.36 [1.73, 2.98] | 123 |
| event:first_segment | n/a | n/a | 0.8 % [0.0 %, 2.7 %] | 2.5 % [0.0 %, 5.5 %] | 2.36 [1.73, 2.98] | 120 |
| event:later_segment | n/a | n/a | 2.6 % [0.0 %, 6.5 %] | 5.1 % [1.2 %, 10.4 %] | 2.36 [1.73, 2.98] | 78 |

Generated 2026-10-14T20:37:37 by `anticipation/eval/va30.py` (git 9326588dc9).
