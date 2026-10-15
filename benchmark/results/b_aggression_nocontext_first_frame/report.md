# VA30 report: b_aggression_nocontext_first_frame

Fold file `folds_v2` (content sha256 `8ebdf6ec2e54`), seeds [0], headline seed 0. Test universe: 1031 videos (693 violent, 338 clean negatives = 15.17 h), 198 eligible30 events (73 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 1031 of 6139 universe videos. Not directly comparable with full-universe results.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.549 [0.473, 0.619] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.567 [0.493, 0.633] |
| Far-window AUROC (clean, source x time matched negatives) | 0.616 [0.516, 0.698] |
| Far-window AUROC (clean, uncensored negatives) | 0.541 [0.470, 0.609] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.560 [0.492, 0.629] |
| AUROC / AP, H = 30 s (base rate 0.146) | 0.593 [0.544, 0.641] / 0.232 [0.182, 0.307] |
| Detection AUROC (in-violence vs clean negatives) | 0.664 [0.615, 0.708] |
| Brier P(onset within 30 s) (climatology 0.1436), ECE | 0.1542 [0.1307, 0.1803], 0.1110 |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 0.34 [0.07, 0.68] | 0.41 [0.15, 0.72] | 6.1 % [2.9 %, 9.7 %] | [2.4, 9.5] % | 0.8 % [0.2 %, 1.7 %] | 3.8 % | 0.0 % [0.0 %, 0.0 %] (90) | 6.1 % [2.9 %, 9.7 %] | 6.1 % [2.9 %, 9.7 %] | 46.8 [36.2, 73.3] | 18.5 % [15.4 %, 21.4 %] |
| 2.0 | 0.34 [0.07, 0.68] | 0.41 [0.15, 0.72] | 6.1 % [2.9 %, 9.7 %] | [3.2, 11.0] % | 0.8 % [0.2 %, 1.7 %] | 3.8 % | 0.0 % [0.0 %, 0.0 %] (90) | 6.1 % [2.9 %, 9.7 %] | 6.1 % [2.9 %, 9.7 %] | 46.8 [36.2, 73.3] | 18.5 % [15.4 %, 21.4 %] |
| 5.0 | 2.76 [1.91, 3.71] | 2.26 [1.56, 3.03] | 10.1 % [6.0 %, 14.6 %] | [4.9, 14.1] % | 6.7 % [4.7 %, 8.9 %] | 29.3 % | 0.0 % [0.0 %, 0.0 %] (90) | 10.1 % [6.0 %, 14.6 %] | 10.1 % [6.0 %, 14.6 %] | 53.5 [39.6, 70.0] | 26.9 % [22.9 %, 30.2 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.607 [0.559, 0.654] | 0.642 [0.599, 0.678] | 0.605 [0.561, 0.647] | 0.090 [0.066, 0.154] | 0.0421 |
| 10 | 0.607 [0.558, 0.655] | 0.643 [0.599, 0.681] | 0.606 [0.559, 0.649] | 0.139 [0.105, 0.208] | 0.0721 |
| 20 | 0.600 [0.551, 0.648] | 0.637 [0.588, 0.676] | 0.600 [0.553, 0.646] | 0.195 [0.151, 0.267] | 0.1148 |
| 30 | 0.593 [0.544, 0.641] | 0.631 [0.579, 0.673] | 0.595 [0.548, 0.643] | 0.232 [0.182, 0.307] | 0.1457 |
| 60 | 0.593 [0.538, 0.645] | 0.628 [0.574, 0.671] | 0.594 [0.542, 0.646] | 0.312 [0.248, 0.395] | 0.2110 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.639 [0.462, 0.796] | 0.673 [0.582, 0.751] | 16.7 % [3.7 %, 33.4 %] | 16.7 % [3.7 %, 33.4 %] | 0.83 [0.18, 1.78] | 24 |
| source:xd_youtube | 0.614 [0.435, 0.783] | 0.625 [0.533, 0.723] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.12 [0.00, 0.38] | 30 |
| source:ucf | 0.627 [0.494, 0.723] | 0.687 [0.608, 0.737] | 5.6 % [2.2 %, 9.9 %] | 5.6 % [2.2 %, 9.9 %] | 0.00 [0.00, 0.00] | 144 |
| class:interpersonal | 0.534 [0.457, 0.608] | 0.571 [0.519, 0.622] | 4.0 % [1.3 %, 7.3 %] | 4.0 % [1.3 %, 7.3 %] | 0.34 [0.07, 0.68] | 173 |
| class:other_violent | 0.647 [0.475, 0.805] | 0.671 [0.594, 0.753] | 20.0 % [5.0 %, 35.7 %] | 20.0 % [5.0 %, 35.7 %] | 0.34 [0.07, 0.68] | 25 |
| audio:present | 0.560 [0.463, 0.654] | 0.610 [0.546, 0.669] | 5.0 % [1.0 %, 9.6 %] | 5.0 % [1.0 %, 9.6 %] | 0.36 [0.08, 0.72] | 101 |
| audio:absent | 0.677 [0.536, 0.774] | 0.748 [0.668, 0.794] | 7.2 % [2.4 %, 13.0 %] | 7.2 % [2.4 %, 13.0 %] | 0.00 [0.00, 0.00] | 97 |
| onset:official | 0.608 [0.502, 0.709] | 0.649 [0.581, 0.709] | 8.0 % [2.5 %, 14.1 %] | 8.0 % [2.5 %, 14.1 %] | 0.34 [0.07, 0.68] | 75 |
| onset:hivau | 0.509 [0.419, 0.593] | 0.536 [0.479, 0.592] | 4.9 % [1.6 %, 9.3 %] | 4.9 % [1.6 %, 9.3 %] | 0.34 [0.07, 0.68] | 123 |
| event:first_segment | n/a | n/a | 10.0 % [4.8 %, 15.7 %] | 10.0 % [4.8 %, 15.7 %] | 0.34 [0.07, 0.68] | 120 |
| event:later_segment | n/a | n/a | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.34 [0.07, 0.68] | 78 |

## Opt-in Phase 0 metrics

CIs resample split_group; legacy metrics above retain their original configuration.

| Metric | Value |
|---|---|
| Band source x time matched AUROC (30,120] | 0.620 [0.509, 0.711] |
| Long band matched AUROC (60,120] | 0.621 [0.492, 0.730] |
| Event window-max matched AUROC (M8) | 0.624 [0.553, 0.691] |
| First-event window-max AUROC | 0.612 [0.541, 0.677] |
| Partial AMOC 0.5..2 FA/h (normalized) | 0.085 [0.051, 0.124] |
| Band AUROC real_world | 0.617 [0.498, 0.713] |
| Band AUROC movie | 0.677 [0.496, 0.838] |
| Person-stratified band AUROC (M2) | 0.589 [0.473, 0.692] |
| Person-stratified band AUROC, >=2 people | 0.589 [0.464, 0.705] |

Pose coverage: {'input_sha256': '2998d79a600a0f22cba689b90d0ad6162eb49304a0b04b0e3a8da194a3b9ddbb', 'pose_files': 1031, 'valid_ticks': 106618, 'missing_or_invalid_ticks': 26206, 'column': 'n_persons_last', 'alignment': 'exact timestamp only; missing is not zero people', 'events_kept_per_source': {'ucf': 51, 'xd_movie': 14, 'xd_youtube': 16}}

| Clean time target | Test warning time | Sustained@60 | Position null@60 | Sustained@90 | Position null@90 | Sustained@120 | Position null@120 |
|---|---|---|---|---|---|---|---|
| 1% | 1.7 % [0.2 %, 4.0 %] | 9.6 % [3.6 %, 17.5 %] | 15.6% | 11.6 % [2.9 %, 22.5 %] | 19.3% | 9.4 % [0.0 %, 20.8 %] | 22.6% |
| 2% | 1.9 % [0.3 %, 4.2 %] | 9.6 % [3.6 %, 17.5 %] | 16.8% | 11.6 % [2.9 %, 22.5 %] | 20.8% | 9.4 % [0.0 %, 20.8 %] | 24.0% |
| 5% | 1.9 % [0.3 %, 4.2 %] | 9.6 % [3.6 %, 17.5 %] | 16.8% | 11.6 % [2.9 %, 22.5 %] | 20.8% | 9.4 % [0.0 %, 20.8 %] | 24.0% |

Generated 2026-10-15T15:02:22 by `anticipation/eval/va30.py` (git b26a629c56).
