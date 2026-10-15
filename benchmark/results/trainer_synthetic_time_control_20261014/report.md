# VA30 report: trainer_synthetic_time_control_20261014

Fold file `synthetic` (content sha256 `91e72a2d7736`), seeds [0, 1, 2], headline seed 0. Test universe: 20 videos (10 violent, 10 clean negatives = 0.61 h), 10 eligible30 events (10 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 20 of 20 universe videos. Not directly comparable with full-universe results.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.644 [0.521, 0.748] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.665 [0.546, 0.771] |
| Far-window AUROC (clean, source x time matched negatives) | 0.574 [0.444, 0.697] |
| Far-window AUROC (clean, uncensored negatives) | 0.715 [0.598, 0.809] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.644 [0.521, 0.748] |
| AUROC / AP, H = 30 s (base rate 0.128) | 0.808 [0.767, 0.852] / 0.271 [0.149, 0.515] |
| Detection AUROC (in-violence vs clean negatives) | 0.741 [0.629, 0.863] |
| Brier P(onset within 30 s) (climatology 0.0776), ECE | 0.0704 [0.0472, 0.0995], 0.0521 |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.0 % [0.0 %, 0.0 %] | n/a | 0.0 % [0.0 %, 0.0 %] | 0.0 % | 0.0 % [0.0 %, 0.0 %] (1) | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | n/a | 0.0 % [0.0 %, 0.0 %] |
| 2.0 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.0 % [0.0 %, 0.0 %] | n/a | 0.0 % [0.0 %, 0.0 %] | 0.0 % | 0.0 % [0.0 %, 0.0 %] (1) | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | n/a | 0.0 % [0.0 %, 0.0 %] |
| 5.0 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.0 % [0.0 %, 0.0 %] | n/a | 0.0 % [0.0 %, 0.0 %] | 0.0 % | 0.0 % [0.0 %, 0.0 %] (1) | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | n/a | 0.0 % [0.0 %, 0.0 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.803 [0.705, 0.886] | 0.410 [0.264, 0.620] | 0.831 [0.728, 0.912] | 0.060 [0.022, 0.182] | 0.0214 |
| 10 | 0.809 [0.735, 0.880] | 0.398 [0.272, 0.575] | 0.836 [0.755, 0.906] | 0.108 [0.045, 0.263] | 0.0428 |
| 20 | 0.810 [0.772, 0.857] | 0.377 [0.289, 0.477] | 0.834 [0.791, 0.886] | 0.190 [0.097, 0.394] | 0.0855 |
| 30 | 0.808 [0.767, 0.852] | 0.412 [0.336, 0.494] | 0.827 [0.780, 0.871] | 0.271 [0.149, 0.515] | 0.1283 |
| 60 | 0.721 [0.646, 0.779] | 0.502 [0.420, 0.575] | 0.720 [0.637, 0.780] | 0.395 [0.218, 0.637] | 0.2566 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:ucf | 0.644 [0.521, 0.748] | 0.808 [0.767, 0.852] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 10 |
| class:interpersonal | 0.644 [0.521, 0.748] | 0.808 [0.767, 0.852] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 10 |
| class:other_violent | n/a | n/a | n/a | n/a | 0.00 [0.00, 0.00] | 0 |
| audio:present | n/a | n/a | n/a | n/a | n/a | 0 |
| audio:absent | 0.644 [0.521, 0.748] | 0.808 [0.767, 0.852] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 10 |
| onset:official | 0.644 [0.521, 0.748] | 0.808 [0.767, 0.852] | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 10 |
| event:first_segment | n/a | n/a | 0.0 % [0.0 %, 0.0 %] | 0.0 % [0.0 %, 0.0 %] | 0.00 [0.00, 0.00] | 10 |
| event:later_segment | n/a | n/a | n/a | n/a | 0.00 [0.00, 0.00] | 0 |

## Mean and std over seeds

| Metric | mean | std |
|---|---|---|
| far_auroc_clean | 0.644 | 0.000 |
| far_auroc_secondary | 0.665 | 0.000 |
| far_auroc_clean_time_matched | 0.574 | 0.000 |
| far_auroc_clean_all_segments | 0.644 | 0.000 |
| auroc30 | 0.808 | 0.000 |
| auroc30_time_matched | 0.412 | 0.000 |
| ap30 | 0.271 | 0.000 |
| success30_at_fa2 | 0.000 | 0.000 |
| success10_at_fa2 | 0.000 | 0.000 |
| success0_at_fa2 | 0.000 | 0.000 |
| success30_late_at_fa2 | 0.000 | 0.000 |
| test_fa_per_hour_clean_at_fa2 | 0.000 | 0.000 |
| random_success30_at_fa2 | 0.000 | 0.000 |
| position_null_success30_at_fa2 | 0.000 | 0.000 |
| detection_auroc | 0.741 | 0.000 |

Generated 2026-10-14T21:49:37 by `anticipation/eval/va30.py` (git e8e6d12c4b).
