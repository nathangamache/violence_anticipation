# External test v1: frozen two-head model on CCTV-Fights and MEVA

Plan: benchmark/external/external_test_plan_v1.md. Config sha256 8e4dc5eb944257c891c30d22b9c1173750d009ad8e965be0480f01bd0007ef41.

## Correctness check on fold-1 test videos

| head | M1 here | M1 committed | abs diff | match 4 dp |
|---|---|---|---|---|
| early | 0.904818 | 0.904818 | 0.00e+00 | True |
| imminent | 0.892968 | 0.892968 | 0.00e+00 | True |

## Thresholds (fold-1 calibration slice)

| head | target FA/h | theta | fold-1 test FA/h (here) |
|---|---|---|---|
| early | 1 | 4.5682 | 0.3262 |
| early | 2 | 3.9627 | 0.7176 |
| early | 5 | 2.8661 | 2.6422 |
| imminent | 1 | 4.3134 | 0.4241 |
| imminent | 2 | 3.6132 | 0.8481 |
| imminent | 5 | 2.4902 | 3.2293 |

MEVA list: 355 clips, 29.58 h, sha256 8fce9ffcfe03531a9a1cad936129b41471dc2ee14dda02413a635e013874cd0b.

## early head

CCTV-Fights evaluation set videos 985, eligible30 events 145 (cctv 90, non_cctv 55), eligible60 41; MEVA clips 355, 29.58 h.

| target FA/h | theta | MEVA FA/h | MEVA clips with alarm | Success@30 (all) | random at MEVA FA/h | Success@30 cctv | Success@30 non_cctv | Success@0 (all) | median lead (alarmed) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4.568 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.166 [0.101, 0.240] | 0.000 [0.000, 0.000] | 0.156 [0.075, 0.255] | 0.182 [0.085, 0.298] | 0.283 [0.199, 0.364] | 31.72 |
| 2 | 3.963 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.241 [0.173, 0.315] | 0.000 [0.000, 0.000] | 0.167 [0.091, 0.253] | 0.364 [0.234, 0.500] | 0.331 [0.250, 0.414] | 36.602999999999994 |
| 5 | 2.866 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.359 [0.275, 0.441] | 0.000 [0.000, 0.000] | 0.256 [0.173, 0.344] | 0.527 [0.379, 0.667] | 0.524 [0.441, 0.605] | 37.8 |

Other lead thresholds (all domains):

| target FA/h | Success@60 (e60) | Success@30 | Success@10 | Success@0 |
|---|---|---|---|---|
| 1 | 0.220 [0.085, 0.368] | 0.166 [0.101, 0.240] | 0.255 [0.182, 0.333] | 0.283 [0.199, 0.364] |
| 2 | 0.220 [0.086, 0.366] | 0.241 [0.173, 0.315] | 0.324 [0.247, 0.408] | 0.331 [0.250, 0.414] |
| 5 | 0.390 [0.244, 0.541] | 0.359 [0.275, 0.441] | 0.483 [0.396, 0.570] | 0.524 [0.441, 0.605] |

Tick AUROC:

| subset | vs MEVA ticks | vs far ticks of same videos | n pos / n neg MEVA |
|---|---|---|---|
| all | 0.963 [0.942, 0.978] | 0.398 [0.337, 0.459] | 2214 / 106498 |
| cctv | 0.962 [0.933, 0.981] | 0.418 [0.333, 0.500] | 1415 / 106498 |
| non_cctv | 0.965 [0.940, 0.985] | 0.358 [0.283, 0.443] | 799 / 106498 |

Descriptive (CCTV-Fights far ticks as false-alarm proxy):

| target FA/h | proxy FA/h | random Success@30 | random Success@10 |
|---|---|---|---|
| 1 | 16.88 | 0.344 | 0.403 |
| 2 | 19.75 | 0.390 | 0.453 |
| 5 | 18.79 | 0.375 | 0.437 |

Verdict (Success@30 on CCTV-Fights eligible30 at the 2 FA/h threshold): success 0.2413793103448276 CI [0.17331034482758623, 0.3146921975662133] vs random 0.0000 -> lower bound above random: True; AUROC vs MEVA 0.962893892023673 CI [0.9420577203034198, 0.9777392202599815] -> lower bound above 0.5: True. **Transfers: True**

## imminent head

CCTV-Fights evaluation set videos 985, eligible30 events 145 (cctv 90, non_cctv 55), eligible60 41; MEVA clips 355, 29.58 h.

| target FA/h | theta | MEVA FA/h | MEVA clips with alarm | Success@10 (all) | random at MEVA FA/h | Success@10 cctv | Success@10 non_cctv | Success@0 (all) | median lead (alarmed) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4.313 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.138 [0.085, 0.197] | 0.000 [0.000, 0.000] | 0.111 [0.053, 0.178] | 0.182 [0.078, 0.300] | 0.152 [0.096, 0.217] | 39.3215 |
| 2 | 3.613 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.283 [0.205, 0.366] | 0.000 [0.000, 0.000] | 0.267 [0.175, 0.365] | 0.309 [0.167, 0.442] | 0.317 [0.237, 0.401] | 36.39975 |
| 5 | 2.490 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.503 [0.420, 0.585] | 0.000 [0.000, 0.000] | 0.444 [0.345, 0.552] | 0.600 [0.469, 0.731] | 0.545 [0.456, 0.626] | 36.6333 |

Other lead thresholds (all domains):

| target FA/h | Success@60 (e60) | Success@30 | Success@10 | Success@0 |
|---|---|---|---|---|
| 1 | 0.146 [0.050, 0.273] | 0.097 [0.054, 0.145] | 0.138 [0.085, 0.197] | 0.152 [0.096, 0.217] |
| 2 | 0.220 [0.115, 0.362] | 0.207 [0.139, 0.273] | 0.283 [0.205, 0.366] | 0.317 [0.237, 0.401] |
| 5 | 0.390 [0.233, 0.558] | 0.379 [0.301, 0.463] | 0.503 [0.420, 0.585] | 0.545 [0.456, 0.626] |

Tick AUROC:

| subset | vs MEVA ticks | vs far ticks of same videos | n pos / n neg MEVA |
|---|---|---|---|
| all | 0.984 [0.978, 0.988] | 0.481 [0.440, 0.524] | 4737 / 106498 |
| cctv | 0.986 [0.980, 0.991] | 0.477 [0.420, 0.532] | 1901 / 106498 |
| non_cctv | 0.982 [0.977, 0.988] | 0.453 [0.399, 0.509] | 2836 / 106498 |

Descriptive (CCTV-Fights far ticks as false-alarm proxy):

| target FA/h | proxy FA/h | random Success@30 | random Success@10 |
|---|---|---|---|
| 1 | 11.47 | 0.249 | 0.296 |
| 2 | 20.06 | 0.394 | 0.458 |
| 5 | 26.12 | 0.479 | 0.550 |

Verdict (Success@10 on CCTV-Fights eligible30 at the 2 FA/h threshold): success 0.2827586206896552 CI [0.2052937680421124, 0.3658622466506355] vs random 0.0000 -> lower bound above random: True; AUROC vs MEVA 0.9837258894252249 CI [0.9783940093605151, 0.9884321073483059] -> lower bound above 0.5: True. **Transfers: True**

## Decisions not fixed by the plan

- Both heads use the fold-1 thresholds unchanged; external data never enters choose_threshold.
- Scoring runs on the full 1 Hz feature file of each video with no label cut; shot-cut flags come from the motion stream where it exists (CCTV-Fights) and are all-zero for MEVA (no motion stream; fixed cameras).
- MEVA clips are clean negatives with duration equal to the number of feature ticks; no horizon cap and no excluded spans.
- Bootstrap: 1000 resamples, CCTV-Fights videos (by split_group, one per video) and MEVA clips resampled independently; seed 0. A camera-cluster MEVA bootstrap is not used for the headline.
- Success uses the VA30 alarm policy through va30.EventEval with a fixed threshold per head and target; leads and hits are as in VA30 (first alarm episode starting in [t0-120, t0)); hit means lead >= L. Success@60 uses eligible60 events, other L use eligible30.
- Random reference for Success@L is 1 - exp(-FAR*(120-L)/3600) at the observed MEVA FA/h (W = 90 s for L=30, 110 s for L=10).
- Tick AUROC positives: first-segment pre-violence unmasked CCTV-Fights ticks with tte in (30,120] (early) or (0,10] (imminent). Negatives: all MEVA ticks, or unmasked CCTV-Fights ticks with tte > 120 or no onset ahead. Unweighted (no time matching); pooled over ticks.
- CCTV-Fights headline population is the evaluation set (label_reliable and not overlap_main, 985 videos); all 1000 are scored in scores.parquet.
- Success criterion applied to the headline measure of each head (Success@30 early, Success@10 imminent) at the 2 FA/h threshold; AUROC against MEVA uses the head's own positive band over the whole evaluation set.
