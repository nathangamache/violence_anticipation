# External test plan v1: NTU CCTV-Fights + MEVA, frozen two-head model

Written 2026-10-18, after the fold-1 confirmation (config sha256
`8e4dc5eb944257c891c30d22b9c1173750d009ad8e965be0480f01bd0007ef41`, both heads confirmed) and BEFORE any model
score exists for any CCTV-Fights or MEVA video. Extends `docs/EXTERNAL_TEST.md` section 6, which was written on
2026-10-15 before the screen; where the two differ, this file wins.

## Models

Both frozen heads of the confirmation config, trained exactly as in `anticipation.screen.confirm.run_head` (same
3,515 screening training videos, same blocks, C values, seeds and row caps): early head = P1 on
`siglip2_prompts_v1` (band target), imminent head = P1 on `siglip2_prompts_v2` (near target). No refitting, no
recalibration, no smoothing change. Correctness check before scoring external videos: the scorer, applied to the
fold-1 test videos, must reproduce the committed fold-1 M1 (early 0.9048, imminent 0.8930) to 4 decimals.

## Thresholds

The fold-1 confirmation thresholds: for each head and each target (1, 2, 5 FA/h), `va30.choose_threshold` on the
calibration slice (out-of-fold scores of the folds_v2 validation slice of fold 1) exactly as in
`confirm.fold1_events`. They are written to the output before any external score is computed. Default 2-of-3
alarm rule and one alarm per episode, as in VA30.

## Populations

- **Fights:** CCTV-Fights `evaluation_set` (985 videos, label reliable, no overlap with the main catalog);
  eligible30 events (145) for Success@30 and Success@10, eligible60 (41) for Success@60 when reported. Labels and
  eligibility from `benchmark/external/cctv_fights/events.json` unchanged.
- **Clean negatives:** the MEVA clips listed in `data/annotation/meva/selection.json` that were downloaded and have
  both prompt features; the list is frozen with its sha256 in the output before scoring. MEVA is scripted
  activity by actors at a training facility, fixed EO cameras, no audio; the acquisition excluded any clip whose
  annotations contain an aggression-like activity type. No MEVA clip is ever used for training or selection.

## Metrics (both heads; 95% bootstrap CIs over videos)

1. Success@30 (early head) and Success@10 (imminent head) on CCTV-Fights eligible30 events at each threshold; also
   Success@10 for the early head and Success@0 for both; median lead; per domain (CCTV vs non-CCTV).
2. False alarms per hour on MEVA (all clips; also per site/camera group) at each threshold, and the share of
   MEVA clips with at least one false alarm.
3. Random-alarm reference at the MEVA false-alarm rate, `1 - exp(-FAR * W)` with W = 90 s (Success@30) or 110 s
   (Success@10). No position null (different domains).
4. Tick-level AUROC: CCTV-Fights pre-onset ticks in (30, 120] s (early) and (0, 10] s (imminent) against (a) MEVA
   ticks and (b) far-from-onset ticks (tte > 120 s or no onset ahead) of the same CCTV-Fights videos; CCTV vs
   non-CCTV separately. (b) controls for domain; (a) is the deployment-like comparison.
5. Descriptive only: the same metrics with CCTV-Fights far-from-onset ticks as a false-alarm proxy (EXTERNAL_TEST.md
   section 6.5).

## What counts as success (stated before scoring)

A head transfers if, at its 2 FA/h threshold, its success rate's lower 95% bound on CCTV-Fights exceeds the random
reference computed at the observed MEVA false-alarm rate, and its tick AUROC against MEVA has a lower bound above
0.5. Anything else is reported as a failure to transfer, with the numbers.
