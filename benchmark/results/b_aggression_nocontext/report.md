# VA30 report: b_aggression_nocontext

Fold file `folds_v2` (content sha256 `8ebdf6ec2e54`), seeds [0], headline seed 0. Test universe: 1031 videos (693 violent, 338 clean negatives = 15.17 h), 198 eligible30 events (73 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 1031 of 6139 universe videos. Not directly comparable with full-universe results.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.637 [0.583, 0.689] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.635 [0.581, 0.685] |
| Far-window AUROC (clean, source x time matched negatives) | 0.718 [0.647, 0.776] |
| Far-window AUROC (clean, uncensored negatives) | 0.625 [0.572, 0.677] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.646 [0.594, 0.695] |
| AUROC / AP, H = 30 s (base rate 0.146) | 0.623 [0.588, 0.657] / 0.280 [0.236, 0.329] |
| Detection AUROC (in-violence vs clean negatives) | 0.809 [0.783, 0.832] |
| Brier P(onset within 30 s) (climatology 0.1436), ECE | 0.1474 [0.1280, 0.1705], 0.0908 |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 0.88 [0.36, 1.54] | 2.26 [1.56, 3.06] | 9.1 % [5.6 %, 13.2 %] | [3.2, 13.9] % | 2.2 % [0.9 %, 3.8 %] | 9.5 % | 10.0 % [4.4 %, 16.7 %] (90) | 14.6 % [10.1 %, 20.0 %] | 17.2 % [12.0 %, 22.7 %] | 31.6 [18.9, 57.0] | 25.2 % [21.6 %, 28.5 %] |
| 2.0 | 1.01 [0.48, 1.67] | 2.82 [2.01, 3.74] | 12.1 % [8.0 %, 16.9 %] | [5.6, 16.4] % | 2.5 % [1.2 %, 4.1 %] | 10.9 % | 14.4 % [7.8 %, 21.4 %] (90) | 20.7 % [15.4 %, 26.7 %] | 24.2 % [18.1 %, 30.6 %] | 29.9 [20.9, 40.9] | 35.7 % [32.0 %, 39.0 %] |
| 5.0 | 1.01 [0.48, 1.67] | 2.82 [2.01, 3.74] | 12.1 % [8.0 %, 16.9 %] | [8.0, 23.5] % | 2.5 % [1.2 %, 4.1 %] | 10.9 % | 14.4 % [7.8 %, 21.4 %] (90) | 20.7 % [15.4 %, 26.7 %] | 24.2 % [18.1 %, 30.6 %] | 29.9 [20.9, 40.9] | 35.7 % [32.0 %, 39.0 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.643 [0.608, 0.674] | 0.670 [0.635, 0.701] | 0.646 [0.616, 0.676] | 0.133 [0.102, 0.171] | 0.0421 |
| 10 | 0.634 [0.600, 0.665] | 0.663 [0.629, 0.694] | 0.641 [0.611, 0.670] | 0.186 [0.148, 0.228] | 0.0721 |
| 20 | 0.628 [0.594, 0.660] | 0.658 [0.623, 0.691] | 0.637 [0.605, 0.667] | 0.245 [0.203, 0.291] | 0.1148 |
| 30 | 0.623 [0.588, 0.657] | 0.654 [0.618, 0.689] | 0.636 [0.603, 0.666] | 0.280 [0.236, 0.329] | 0.1457 |
| 60 | 0.631 [0.593, 0.667] | 0.656 [0.617, 0.694] | 0.642 [0.606, 0.676] | 0.359 [0.307, 0.416] | 0.2110 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.737 [0.651, 0.845] | 0.708 [0.658, 0.751] | 29.2 % [12.5 %, 47.4 %] | 66.7 % [45.5 %, 85.7 %] | 1.24 [0.23, 2.61] | 24 |
| source:xd_youtube | 0.669 [0.557, 0.757] | 0.627 [0.543, 0.705] | 13.3 % [3.0 %, 26.7 %] | 20.0 % [6.4 %, 36.0 %] | 1.04 [0.34, 2.07] | 30 |
| source:ucf | 0.755 [0.659, 0.822] | 0.650 [0.594, 0.704] | 9.0 % [4.8 %, 13.8 %] | 18.1 % [12.2 %, 24.0 %] | 0.00 [0.00, 0.00] | 144 |
| class:interpersonal | 0.638 [0.581, 0.691] | 0.616 [0.578, 0.653] | 11.6 % [7.0 %, 16.7 %] | 23.1 % [16.9 %, 29.7 %] | 1.01 [0.48, 1.67] | 173 |
| class:other_violent | 0.634 [0.503, 0.754] | 0.656 [0.589, 0.717] | 16.0 % [4.0 %, 28.6 %] | 32.0 % [10.5 %, 52.0 %] | 1.01 [0.48, 1.67] | 25 |
| audio:present | 0.649 [0.571, 0.711] | 0.649 [0.599, 0.691] | 14.9 % [8.2 %, 22.1 %] | 28.7 % [19.8 %, 38.1 %] | 1.07 [0.51, 1.77] | 101 |
| audio:absent | 0.793 [0.681, 0.866] | 0.664 [0.600, 0.733] | 9.3 % [3.9 %, 15.6 %] | 19.6 % [12.4 %, 27.7 %] | 0.00 [0.00, 0.00] | 97 |
| onset:official | 0.673 [0.602, 0.735] | 0.668 [0.620, 0.713] | 17.3 % [9.1 %, 26.3 %] | 33.3 % [21.9 %, 45.1 %] | 1.01 [0.48, 1.67] | 75 |
| onset:hivau | 0.613 [0.541, 0.681] | 0.589 [0.546, 0.628] | 8.9 % [4.1 %, 14.2 %] | 18.7 % [12.7 %, 25.4 %] | 1.01 [0.48, 1.67] | 123 |
| event:first_segment | n/a | n/a | 10.8 % [5.9 %, 17.0 %] | 22.5 % [15.1 %, 29.8 %] | 1.01 [0.48, 1.67] | 120 |
| event:later_segment | n/a | n/a | 14.1 % [7.0 %, 21.9 %] | 26.9 % [17.7 %, 37.2 %] | 1.01 [0.48, 1.67] | 78 |

## Opt-in Phase 0 metrics

CIs resample split_group; legacy metrics above retain their original configuration.

| Metric | Value |
|---|---|
| Band source x time matched AUROC (30,120] | 0.749 [0.675, 0.805] |
| Long band matched AUROC (60,120] | 0.781 [0.693, 0.849] |
| Event window-max matched AUROC (M8) | 0.744 [0.683, 0.796] |
| First-event window-max AUROC | 0.711 [0.643, 0.773] |
| Partial AMOC 0.5..2 FA/h (normalized) | 0.120 [0.073, 0.172] |
| Band AUROC real_world | 0.757 [0.677, 0.819] |
| Band AUROC movie | 0.760 [0.664, 0.866] |
| Person-stratified band AUROC (M2) | 0.737 [0.671, 0.795] |
| Person-stratified band AUROC, >=2 people | 0.735 [0.675, 0.787] |

Pose coverage: {'input_sha256': '2998d79a600a0f22cba689b90d0ad6162eb49304a0b04b0e3a8da194a3b9ddbb', 'pose_files': 1031, 'valid_ticks': 106618, 'missing_or_invalid_ticks': 26206, 'column': 'n_persons_last', 'alignment': 'exact timestamp only; missing is not zero people', 'events_kept_per_source': {'ucf': 51, 'xd_movie': 14, 'xd_youtube': 16}}

| Clean time target | Test warning time | Sustained@60 | Position null@60 | Sustained@90 | Position null@90 | Sustained@120 | Position null@120 |
|---|---|---|---|---|---|---|---|
| 1% | 0.2 % [0.1 %, 0.3 %] | 0.0 % [0.0 %, 0.0 %] | 4.1% | 0.0 % [0.0 %, 0.0 %] | 6.2% | 0.0 % [0.0 %, 0.0 %] | 6.3% |
| 2% | 0.9 % [0.6 %, 1.3 %] | 8.2 % [2.8 %, 15.6 %] | 9.9% | 7.0 % [0.0 %, 16.0 %] | 12.7% | 3.1 % [0.0 %, 10.0 %] | 16.0% |
| 5% | 2.0 % [1.4 %, 2.6 %] | 8.2 % [2.8 %, 15.6 %] | 17.3% | 7.0 % [0.0 %, 16.0 %] | 21.5% | 3.1 % [0.0 %, 10.0 %] | 24.7% |

Generated 2026-10-15T15:02:01 by `anticipation/eval/va30.py` (git b26a629c56).
