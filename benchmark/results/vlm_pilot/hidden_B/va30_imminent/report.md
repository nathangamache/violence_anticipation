# VA30 report: phase_c_hidden_B_imminent

Fold file `phase_c_pilot_without_original_fold1` (content sha256 `8ebdf6ec2e54`), seeds [0], headline seed 0. Test universe: 249 videos (100 violent, 149 clean negatives = 5.36 h), 100 eligible30 events (100 eligible60). Values are point estimates with 95 % cluster-bootstrap CIs over videos.

**Restricted to scored videos:** 249 of 249 universe videos. Not directly comparable with full-universe results.

**Capped clean negatives:** per-video horizons from `benchmark/extract_manifest_v1.csv` (sha256 `f531a43df4c0`) shorten 0 catalog clean negatives (0 in the test universe); footage after each horizon is unobserved and excluded from every metric. Test clean-negative hours used: 5.36 of 5.36 in the catalog.

## Headline

| Metric | Value |
|---|---|
| Far-window AUROC (30-60 s vs clean negatives) | 0.834 [0.769, 0.887] |
| Far-window AUROC (secondary: vs no onset within 120 s) | 0.756 [0.695, 0.810] |
| Far-window AUROC (clean, source x time matched negatives) | 0.810 [0.743, 0.869] |
| Far-window AUROC (clean, uncensored negatives) | 0.842 [0.788, 0.889] |
| Far-window AUROC (spec literal, all segments incl. after earlier violence) | 0.836 [0.772, 0.888] |
| AUROC / AP, H = 30 s (base rate 0.139) | 0.796 [0.743, 0.843] / 0.358 [0.278, 0.493] |
| Detection AUROC (in-violence vs clean negatives) | 0.880 [0.830, 0.920] |

## Event-level operating points (theta chosen per fold on validation FA/h)

Position null = best Success@30 that position-only policies (source kind x time since clip start, and their mixtures) reach at the same test FA/h; Success@30 must beat it (CI lower bound above it), not only the Poisson random reference. Late = events whose acceptance window starts >= 30 s into the clip.

| Val FA/h target | Test FA/h clean | Test FA/h all | Success@30 | Success@30 threshold-aware CI | Random (Poisson) | Position null | Success@30 late (n) | Success@10 | Success@0 | Median lead (s) | Detection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 3.96 [2.01, 6.28] | 5.61 [3.54, 8.05] | 25.0 % [16.5 %, 34.0 %] | [18.9, 50.0] % | 9.4 % [4.9 %, 14.5 %] | 16.8 % | 9.7 % [0.0 %, 22.2 %] (31) | 29.0 % [20.4 %, 38.5 %] | 31.0 % [21.3 %, 41.1 %] | 65.0 [56.7, 72.3] | 36.8 % [27.5 %, 46.2 %] |
| 2.0 | 4.52 [2.36, 6.86] | 7.06 [4.76, 9.66] | 34.0 % [24.8 %, 43.8 %] | [21.4, 54.8] % | 10.7 % [5.7 %, 15.8 %] | 18.7 % | 16.1 % [3.8 %, 31.4 %] (31) | 39.0 % [29.8 %, 49.4 %] | 41.0 % [31.2 %, 51.1 %] | 68.7 [60.7, 74.7] | 47.2 % [37.4 %, 56.2 %] |
| 5.0 | 8.29 [5.34, 11.54] | 13.80 [10.31, 18.01] | 52.0 % [43.2 %, 62.0 %] | [32.7, 62.4] % | 18.7 % [12.5 %, 25.1 %] | 30.0 % | 41.9 % [25.0 %, 60.0 %] (31) | 58.0 % [48.6 %, 68.0 %] | 60.0 % [51.1 %, 69.3 %] | 71.1 [61.1, 78.8] | 67.9 % [59.8 %, 76.1 %] |

## Per-horizon (pooled pre-violence ticks; negatives = no onset within 120 s)

| H (s) | AUROC | AUROC time-matched | AUROC all segments | AP | base rate |
|---|---|---|---|---|---|
| 5 | 0.829 [0.781, 0.872] | 0.804 [0.744, 0.862] | 0.829 [0.781, 0.873] | 0.086 [0.060, 0.178] | 0.0221 |
| 10 | 0.819 [0.770, 0.864] | 0.794 [0.733, 0.855] | 0.819 [0.771, 0.864] | 0.156 [0.111, 0.276] | 0.0447 |
| 20 | 0.807 [0.754, 0.852] | 0.783 [0.723, 0.843] | 0.807 [0.756, 0.853] | 0.270 [0.202, 0.399] | 0.0910 |
| 30 | 0.796 [0.743, 0.843] | 0.771 [0.708, 0.832] | 0.796 [0.744, 0.843] | 0.358 [0.278, 0.493] | 0.1386 |
| 60 | 0.768 [0.708, 0.819] | 0.730 [0.654, 0.798] | 0.769 [0.710, 0.821] | 0.522 [0.434, 0.641] | 0.2826 |

## Breakdowns (Success at the 2 FA/h operating point)

| Subset | far AUROC (clean) | AUROC@30 | Success@30 | Success@0 | FA/h clean | n eligible30 |
|---|---|---|---|---|---|---|
| source:xd_movie | 0.766 [0.671, 0.866] | 0.784 [0.700, 0.879] | 43.5 % [22.6 %, 72.0 %] | 60.9 % [38.9 %, 87.5 %] | 11.74 [5.17, 20.36] | 23 |
| source:xd_youtube | 0.842 [0.728, 0.932] | 0.824 [0.736, 0.905] | 29.0 % [13.5 %, 46.9 %] | 35.5 % [19.2 %, 53.8 %] | 2.99 [0.35, 7.06] | 31 |
| source:ucf | 0.825 [0.694, 0.936] | 0.733 [0.625, 0.838] | 32.4 % [17.9 %, 48.3 %] | 35.3 % [20.0 %, 51.4 %] | 3.34 [0.00, 7.31] | 34 |
| source:ubi | 0.823 [0.589, 0.997] | 0.848 [0.685, 0.980] | 33.3 % [8.3 %, 63.7 %] | 33.3 % [8.3 %, 63.7 %] | 1.00 [0.00, 2.96] | 12 |
| class:interpersonal | 0.835 [0.770, 0.889] | 0.797 [0.740, 0.847] | 31.7 % [21.5 %, 42.0 %] | 39.0 % [27.8 %, 50.0 %] | 4.52 [2.36, 6.86] | 82 |
| class:other_violent | 0.830 [0.709, 0.929] | 0.860 [0.766, 0.936] | 44.4 % [20.0 %, 75.0 %] | 50.0 % [25.0 %, 80.0 %] | 4.52 [2.36, 6.86] | 18 |
| audio:present | 0.839 [0.769, 0.899] | 0.805 [0.742, 0.860] | 34.3 % [23.4 %, 47.4 %] | 43.3 % [31.5 %, 56.7 %] | 5.49 [2.61, 9.08] | 67 |
| audio:absent | 0.806 [0.672, 0.933] | 0.779 [0.683, 0.872] | 33.3 % [16.7 %, 50.0 %] | 36.4 % [20.0 %, 53.3 %] | 2.70 [0.52, 5.33] | 33 |
| onset:official | 0.815 [0.701, 0.910] | 0.852 [0.753, 0.932] | 40.0 % [16.6 %, 64.8 %] | 60.0 % [31.2 %, 84.6 %] | 4.52 [2.36, 6.86] | 15 |
| onset:hivau | 0.833 [0.766, 0.888] | 0.799 [0.741, 0.850] | 32.9 % [23.0 %, 44.4 %] | 38.4 % [28.1 %, 49.4 %] | 4.52 [2.36, 6.86] | 73 |
| onset:ubi | 0.865 [0.774, 0.930] | 0.893 [0.814, 0.950] | 33.3 % [8.3 %, 63.7 %] | 33.3 % [8.3 %, 63.7 %] | 4.52 [2.36, 6.86] | 12 |
| event:first_segment | n/a | n/a | 34.0 % [24.8 %, 43.8 %] | 41.0 % [31.2 %, 51.1 %] | 4.52 [2.36, 6.86] | 100 |
| event:later_segment | n/a | n/a | n/a | n/a | 4.52 [2.36, 6.86] | 0 |

Generated 2026-10-16T16:01:57 by `anticipation/eval/va30.py` (git d25ab37745).
