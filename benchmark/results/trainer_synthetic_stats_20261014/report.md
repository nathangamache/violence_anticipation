# VA30 report: trainer_synthetic_stats_20261014

Fold file `synthetic` (content sha256 `91e72a2d7736`), seeds [0, 1, 2], headline seed 0. Test universe: 20 videos (10 violent, 10 clean negatives = 0.61 h), 10 eligible30 events (10 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 20 of 20 universe videos. Not directly comparable with full-universe results.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.666 [0.489, 0.850] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.670 [0.519, 0.837] |
| Far-window AUROC (clean, source x time matched negatives) | 0.664 [0.488, 0.844] |
| Far-window AUROC (clean, uncensored negatives) | 0.665 [0.489, 0.849] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.666 [0.489, 0.850] |
| AUROC / AP, H = 30 s (base rate 0.128) | 1.000 [0.999, 1.000] / 0.994 [0.972, 1.000] |
| Detection AUROC (in-violence vs clean negatives) | 1.000 [0.999, 1.000] |
| Brier P(onset within 30 s) (climatology 0.0776), ECE | 0.0321 [0.0211, 0.0462], 0.0763 |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 4.91 [0.00, 9.55] | 4.86 [1.22, 8.48] | 100.0 % [100.0 %, 100.0 %] | n/a | 11.5 % [0.0 %, 21.2 %] | 30.0 % | 100.0 % [100.0 %, 100.0 %] (1) | 100.0 % [100.0 %, 100.0 %] | 100.0 % [100.0 %, 100.0 %] | 39.0 [39.0, 49.4] | 100.0 % [100.0 %, 100.0 %] |
| 2.0 | 4.91 [0.00, 9.55] | 4.86 [1.22, 8.48] | 100.0 % [100.0 %, 100.0 %] | n/a | 11.5 % [0.0 %, 21.2 %] | 30.0 % | 100.0 % [100.0 %, 100.0 %] (1) | 100.0 % [100.0 %, 100.0 %] | 100.0 % [100.0 %, 100.0 %] | 39.0 [39.0, 49.4] | 100.0 % [100.0 %, 100.0 %] |
| 5.0 | 4.91 [0.00, 9.55] | 4.86 [1.22, 8.48] | 100.0 % [100.0 %, 100.0 %] | n/a | 11.5 % [0.0 %, 21.2 %] | 30.0 % | 100.0 % [100.0 %, 100.0 %] (1) | 100.0 % [100.0 %, 100.0 %] | 100.0 % [100.0 %, 100.0 %] | 39.0 [39.0, 49.4] | 100.0 % [100.0 %, 100.0 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 1.000 [0.999, 1.000] | 1.000 [1.000, 1.000] | 0.999 [0.998, 1.000] | 0.967 [0.876, 1.000] | 0.0214 |
| 10 | 1.000 [0.999, 1.000] | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | 0.982 [0.927, 1.000] | 0.0428 |
| 20 | 1.000 [0.999, 1.000] | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | 0.990 [0.960, 1.000] | 0.0855 |
| 30 | 1.000 [0.999, 1.000] | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | 0.994 [0.972, 1.000] | 0.1283 |
| 60 | 0.835 [0.759, 0.919] | 0.833 [0.744, 0.925] | 0.832 [0.759, 0.912] | 0.789 [0.721, 0.876] | 0.2566 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:ucf | 0.666 [0.489, 0.850] | 1.000 [0.999, 1.000] | 100.0 % [100.0 %, 100.0 %] | 100.0 % [100.0 %, 100.0 %] | 4.91 [0.00, 9.55] | 10 |
| class:interpersonal | 0.666 [0.489, 0.850] | 1.000 [0.999, 1.000] | 100.0 % [100.0 %, 100.0 %] | 100.0 % [100.0 %, 100.0 %] | 4.91 [0.00, 9.55] | 10 |
| class:other_violent | n/a | n/a | n/a | n/a | 4.91 [0.00, 9.55] | 0 |
| audio:present | n/a | n/a | n/a | n/a | n/a | 0 |
| audio:absent | 0.666 [0.489, 0.850] | 1.000 [0.999, 1.000] | 100.0 % [100.0 %, 100.0 %] | 100.0 % [100.0 %, 100.0 %] | 4.91 [0.00, 9.55] | 10 |
| onset:official | 0.666 [0.489, 0.850] | 1.000 [0.999, 1.000] | 100.0 % [100.0 %, 100.0 %] | 100.0 % [100.0 %, 100.0 %] | 4.91 [0.00, 9.55] | 10 |
| event:first_segment | n/a | n/a | 100.0 % [100.0 %, 100.0 %] | 100.0 % [100.0 %, 100.0 %] | 4.91 [0.00, 9.55] | 10 |
| event:later_segment | n/a | n/a | n/a | n/a | 4.91 [0.00, 9.55] | 0 |

## Mean and std over seeds

| Metric | mean | std |
|---|---|---|
| far_auroc_clean | 0.666 | 0.000 |
| far_auroc_secondary | 0.670 | 0.000 |
| far_auroc_clean_time_matched | 0.664 | 0.000 |
| far_auroc_clean_all_segments | 0.666 | 0.000 |
| auroc30 | 1.000 | 0.000 |
| auroc30_time_matched | 1.000 | 0.000 |
| ap30 | 0.994 | 0.000 |
| success30_at_fa2 | 1.000 | 0.000 |
| success10_at_fa2 | 1.000 | 0.000 |
| success0_at_fa2 | 1.000 | 0.000 |
| success30_late_at_fa2 | 1.000 | 0.000 |
| test_fa_per_hour_clean_at_fa2 | 4.909 | 0.000 |
| random_success30_at_fa2 | 0.115 | 0.000 |
| position_null_success30_at_fa2 | 0.300 | 0.000 |
| detection_auroc | 1.000 | 0.000 |

Generated 2026-10-14T21:49:04 by `anticipation/eval/va30.py` (git e8e6d12c4b).
