# VA30 report: trainer_band_gru_diagnostic_20261015

Fold file `folds_v2` (content sha256 `8ebdf6ec2e54`), seeds [0], headline seed 0. Test universe: 50 videos (27 violent, 23 clean negatives = 0.41 h), 33 eligible30 events (13 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 50 of 4916 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 653 catalog clean negatives (1 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 0.41 of 0.42 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.571 [0.375, 0.762] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.561 [0.393, 0.708] |
| Far-window AUROC (clean, source x time matched negatives) | 0.690 [0.476, 0.849] |
| Far-window AUROC (clean, uncensored negatives) | 0.622 [0.446, 0.752] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.632 [0.495, 0.819] |
| AUROC / AP, H = 30 s (base rate 0.350) | 0.648 [0.458, 0.789] / 0.568 [0.356, 0.814] |
| Detection AUROC (in-violence vs clean negatives) | 0.697 [0.569, 0.843] |
| Brier P(onset within 30 s) (climatology 0.2120), ECE | 0.2351 [0.2282, 0.2403], 0.1570 |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 7.33 [1.35, 17.53] | 18.37 [11.65, 27.51] | 18.2 % [5.8 %, 32.4 %] | n/a | 16.7 % [3.3 %, 35.4 %] | 17.0 % | 13.3 % [0.0 %, 25.2 %] (15) | 33.3 % [19.3 %, 51.3 %] | 33.3 % [19.3 %, 51.3 %] | 47.0 [28.8, 57.3] | 41.2 % [31.8 %, 52.0 %] |
| 2.0 | 7.33 [1.35, 17.53] | 18.37 [11.65, 27.51] | 18.2 % [5.8 %, 32.4 %] | n/a | 16.7 % [3.3 %, 35.4 %] | 17.0 % | 13.3 % [0.0 %, 25.2 %] (15) | 33.3 % [19.3 %, 51.3 %] | 33.3 % [19.3 %, 51.3 %] | 47.0 [28.8, 57.3] | 41.2 % [31.8 %, 52.0 %] |
| 5.0 | 7.33 [1.35, 17.53] | 18.37 [11.65, 27.51] | 18.2 % [5.8 %, 32.4 %] | n/a | 16.7 % [3.3 %, 35.4 %] | 17.0 % | 13.3 % [0.0 %, 25.2 %] (15) | 33.3 % [19.3 %, 51.3 %] | 33.3 % [19.3 %, 51.3 %] | 47.0 [28.8, 57.3] | 41.2 % [31.8 %, 52.0 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.675 [0.471, 0.828] | 0.719 [0.540, 0.868] | 0.597 [0.502, 0.726] | 0.252 [0.094, 0.540] | 0.0625 |
| 10 | 0.683 [0.482, 0.834] | 0.724 [0.542, 0.868] | 0.602 [0.504, 0.732] | 0.365 [0.172, 0.676] | 0.1256 |
| 20 | 0.665 [0.471, 0.802] | 0.728 [0.553, 0.852] | 0.589 [0.482, 0.718] | 0.485 [0.277, 0.754] | 0.2430 |
| 30 | 0.648 [0.458, 0.789] | 0.714 [0.542, 0.836] | 0.572 [0.462, 0.705] | 0.568 [0.356, 0.814] | 0.3503 |
| 60 | 0.605 [0.462, 0.747] | 0.677 [0.542, 0.804] | 0.535 [0.438, 0.651] | 0.688 [0.527, 0.878] | 0.5517 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:ucf | 0.759 [0.487, 0.999] | 0.664 [0.548, 0.846] | 6.2 % [0.0 %, 11.8 %] | 18.8 % [6.0 %, 37.4 %] | 5.87 [0.00, 18.51] | 16 |
| source:ubi | 0.694 [0.570, 0.895] | 0.791 [0.669, 0.958] | 29.4 % [13.2 %, 53.3 %] | 47.1 % [33.4 %, 63.5 %] | 8.37 [0.00, 20.27] | 17 |
| class:interpersonal | 0.571 [0.375, 0.762] | 0.648 [0.458, 0.789] | 18.2 % [5.8 %, 32.4 %] | 33.3 % [19.3 %, 51.3 %] | 7.33 [1.35, 17.53] | 33 |
| class:other_violent | n/a | n/a | n/a | n/a | 7.33 [1.35, 17.53] | 0 |
| audio:present | 0.843 [0.773, 0.930] | 0.766 [0.615, 0.950] | 11.1 % [0.0 %, 20.0 %] | 33.3 % [14.9 %, 61.9 %] | 0.00 [0.00, 0.00] | 9 |
| audio:absent | 0.527 [0.227, 0.871] | 0.618 [0.415, 0.791] | 20.8 % [8.4 %, 38.8 %] | 33.3 % [19.8 %, 50.3 %] | 9.73 [1.35, 19.76] | 24 |
| onset:official | 0.554 [0.361, 0.906] | 0.604 [0.214, 0.994] | 0.0 % [0.0 %, 0.0 %] | 50.0 % [0.0 %, 100.0 %] | 7.33 [1.35, 17.53] | 2 |
| onset:hivau | 0.460 [0.228, 0.739] | 0.514 [0.330, 0.678] | 7.1 % [0.0 %, 12.5 %] | 14.3 % [0.0 %, 26.9 %] | 7.33 [1.35, 17.53] | 14 |
| onset:ubi | 0.767 [0.641, 0.867] | 0.840 [0.732, 0.947] | 29.4 % [13.2 %, 53.3 %] | 47.1 % [33.4 %, 63.5 %] | 7.33 [1.35, 17.53] | 17 |
| event:first_segment | n/a | n/a | 16.7 % [4.6 %, 32.2 %] | 44.4 % [20.8 %, 67.0 %] | 7.33 [1.35, 17.53] | 18 |
| event:later_segment | n/a | n/a | 20.0 % [4.0 %, 31.9 %] | 20.0 % [4.0 %, 31.9 %] | 7.33 [1.35, 17.53] | 15 |

Generated 2026-10-15T14:41:02 by `anticipation/eval/va30.py` (git e857da56bc).
