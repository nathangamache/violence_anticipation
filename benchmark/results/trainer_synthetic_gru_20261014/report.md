# VA30 report: trainer_synthetic_gru_20261014

Fold file `synthetic` (content sha256 `91e72a2d7736`), seeds [0, 1, 2], headline seed 0. Test universe: 20 videos (10 violent, 10 clean negatives = 0.61 h), 10 eligible30 events (10 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 20 of 20 universe videos. Not directly comparable with full-universe results.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.640 [0.414, 0.828] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.648 [0.439, 0.816] |
| Far-window AUROC (clean, source x time matched negatives) | 0.643 [0.417, 0.829] |
| Far-window AUROC (clean, uncensored negatives) | 0.641 [0.414, 0.832] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.640 [0.414, 0.828] |
| AUROC / AP, H = 30 s (base rate 0.128) | 1.000 [1.000, 1.000] / 1.000 [1.000, 1.000] |
| Detection AUROC (in-violence vs clean negatives) | 1.000 [1.000, 1.000] |
| Brier P(onset within 30 s) (climatology 0.0776), ECE | 0.0496 [0.0294, 0.0758], 0.1537 |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 13.09 [4.09, 24.55] | 10.94 [4.26, 19.95] | 100.0 % [100.0 %, 100.0 %] | n/a | 27.9 % [9.7 %, 45.9 %] | 80.0 % | 100.0 % [100.0 %, 100.0 %] (1) | 100.0 % [100.0 %, 100.0 %] | 100.0 % [100.0 %, 100.0 %] | 39.0 [39.0, 39.0] | 100.0 % [100.0 %, 100.0 %] |
| 2.0 | 13.09 [4.09, 24.55] | 10.94 [4.26, 19.95] | 100.0 % [100.0 %, 100.0 %] | n/a | 27.9 % [9.7 %, 45.9 %] | 80.0 % | 100.0 % [100.0 %, 100.0 %] (1) | 100.0 % [100.0 %, 100.0 %] | 100.0 % [100.0 %, 100.0 %] | 39.0 [39.0, 39.0] | 100.0 % [100.0 %, 100.0 %] |
| 5.0 | 13.09 [4.09, 24.55] | 10.94 [4.26, 19.95] | 100.0 % [100.0 %, 100.0 %] | n/a | 27.9 % [9.7 %, 45.9 %] | 80.0 % | 100.0 % [100.0 %, 100.0 %] (1) | 100.0 % [100.0 %, 100.0 %] | 100.0 % [100.0 %, 100.0 %] | 39.0 [39.0, 39.0] | 100.0 % [100.0 %, 100.0 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.0214 |
| 10 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.0428 |
| 20 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.0855 |
| 30 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.1283 |
| 60 | 0.824 [0.720, 0.908] | 0.822 [0.708, 0.916] | 0.820 [0.719, 0.902] | 0.783 [0.667, 0.899] | 0.2566 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:ucf | 0.640 [0.414, 0.828] | 1.000 [1.000, 1.000] | 100.0 % [100.0 %, 100.0 %] | 100.0 % [100.0 %, 100.0 %] | 13.09 [4.09, 24.55] | 10 |
| class:interpersonal | 0.640 [0.414, 0.828] | 1.000 [1.000, 1.000] | 100.0 % [100.0 %, 100.0 %] | 100.0 % [100.0 %, 100.0 %] | 13.09 [4.09, 24.55] | 10 |
| class:other_violent | n/a | n/a | n/a | n/a | 13.09 [4.09, 24.55] | 0 |
| audio:present | n/a | n/a | n/a | n/a | n/a | 0 |
| audio:absent | 0.640 [0.414, 0.828] | 1.000 [1.000, 1.000] | 100.0 % [100.0 %, 100.0 %] | 100.0 % [100.0 %, 100.0 %] | 13.09 [4.09, 24.55] | 10 |
| onset:official | 0.640 [0.414, 0.828] | 1.000 [1.000, 1.000] | 100.0 % [100.0 %, 100.0 %] | 100.0 % [100.0 %, 100.0 %] | 13.09 [4.09, 24.55] | 10 |
| event:first_segment | n/a | n/a | 100.0 % [100.0 %, 100.0 %] | 100.0 % [100.0 %, 100.0 %] | 13.09 [4.09, 24.55] | 10 |
| event:later_segment | n/a | n/a | n/a | n/a | 13.09 [4.09, 24.55] | 0 |

## Mean and std over seeds

| Metric | mean | std |
|---|---|---|
| far_auroc_clean | 0.622 | 0.027 |
| far_auroc_secondary | 0.626 | 0.027 |
| far_auroc_clean_time_matched | 0.624 | 0.027 |
| far_auroc_clean_all_segments | 0.622 | 0.027 |
| auroc30 | 1.000 | 0.001 |
| auroc30_time_matched | 1.000 | 0.001 |
| ap30 | 0.999 | 0.001 |
| success30_at_fa2 | 1.000 | 0.000 |
| success10_at_fa2 | 1.000 | 0.000 |
| success0_at_fa2 | 1.000 | 0.000 |
| success30_late_at_fa2 | 1.000 | 0.000 |
| test_fa_per_hour_clean_at_fa2 | 10.909 | 2.041 |
| random_success30_at_fa2 | 0.238 | 0.039 |
| position_null_success30_at_fa2 | 0.667 | 0.125 |
| detection_auroc | 1.000 | 0.000 |

Generated 2026-10-14T21:55:05 by `anticipation/eval/va30.py` (git e8e6d12c4b).
