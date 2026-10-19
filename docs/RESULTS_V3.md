# Results on labels v3 (October 2026)

This is the current result set. It supersedes the interim v1-label numbers in [FUSION.md](FUSION.md) and
[SCREEN_RESULTS.md](SCREEN_RESULTS.md). Every number below is copied from a committed report; the source file is
named with each table.

## 1. Pipeline

1. **Labels v3** ([ONSET_CORRECTION.md](ONSET_CORRECTION.md)). A purpose-built Qwen2.5-VL scan flagged suspicious
   onsets; then every one of the 615 eligible or newly eligible events was reviewed on contact sheets, with blind
   two-pass review for any label change and reconciliation of disagreements. The reviewers were language models
   (Claude Sonnet and GPT-based agents), not people. Result: 419 events with at least 30 s of clean lead-in
   (11.4 h of lead-in), 162 of them with an onset the reviewers could not resolve.
2. **Registration v2** (`benchmark/screens/screen_set_v2.json`, sha256 `25bf720b`), frozen before any v3 score.
3. **Phase A screen** on folds 0, 2, 3, 4 with linear probes and a scene-only nuisance model.
4. **Fusion**, forward selection per head, P1 probes only (the GRU alternative was not run).
5. **Fold-1 confirmation**, run once with the frozen config (sha256 `8e4dc5eb`).

## 2. Phase A screen

Source: [`benchmark/results/cue_screen_v2/results_phase_A.md`](../benchmark/results/cue_screen_v2/results_phase_A.md)
and the per-block `metrics.json` files. M1 is the time-matched AUROC for ticks 30 to 120 s before onset.

| Block | M1 | Class |
|---|---|---|
| siglip2_prompts_v1 | 0.844 | ANTICIPATION |
| siglip2_prompts_v2 | 0.813 | ANTICIPATION |
| siglip2 | 0.847 | CONTEXT_SHORTCUT |
| vjepa2_l | 0.780 | CONTEXT_SHORTCUT |
| pose_interact | 0.614 | CONTEXT_SHORTCUT |
| pose_state_identity_free | 0.559 | INCONCLUSIVE |
| motion_camera_shortcut | 0.557 | INCONCLUSIVE |
| motion_no_camera | 0.548 | INCONCLUSIVE |
| pose_state_pair_identity_ucf_ubi | 0.475 | DROP |

The scene-only nuisance model reaches M1 0.771. A score taken from the first frame of each clip reaches 0.819 for
prompts v1: most of the signal is present from the start of the clip.

## 3. Fusion (development folds, post-selection)

Sources: [`benchmark/results/fusion_v2/report.md`](../benchmark/results/fusion_v2/report.md),
[`benchmark/results/fusion_v2/imminent/report.md`](../benchmark/results/fusion_v2/imminent/report.md) and
[`benchmark/results/fusion_v2/events/va30_fa125/report.md`](../benchmark/results/fusion_v2/events/va30_fa125/report.md).

- **Early head:** forward selection kept `siglip2_prompts_v1` alone (adding v2 changed event-level M8 by -0.001).
- **Imminent head:** of the blocks screened for the 10 s gates, only `siglip2_prompts_v2` passed all three.
  Prompts v1 passed the signal and scene-gain gates but failed specificity against non-violent incidents with
  people (M3p-near 0.598, lower bound 0.496).

| Head | Target | Test FA/h | Success | Threshold-aware CI | Position null |
|---|---|---|---|---|---|
| Early (Success@30) | 1 / h | 1.16 | 20.6% | [15.0, 30.6]% | 7.3% |
| Early (Success@30) | 2 / h | 2.12 | 31.4% | [23.9, 37.8]% | 9.5% |
| Early (Success@30) | 5 / h | 4.99 | 39.0% | [33.8, 45.9]% | 16.1% |
| Imminent (Success@10) | 1 / h | 1.20 | 25.9% | [17.3, 34.2]% | 8.9% |
| Imminent (Success@10) | 2 / h | 2.15 | 33.4% | [25.8, 41.6]% | 11.2% |
| Imminent (Success@10) | 5 / h | 4.71 | 48.3% | [38.9, 55.9]% | 17.7% |

By source at 2 FA/h (early head): false alarms per hour 4.34 on XD movie clips, 2.27 on XD YouTube clips, 0.67 on
UBI-Fights CCTV and 0.09 on UCF-Crime CCTV; Success@30 40.7%, 34.9%, 33.9% and 16.5%.

A crowd-matched check
([`single_block/siglip2_prompts_v1/crowd_match.json`](../benchmark/results/single_block/siglip2_prompts_v1/crowd_match.json))
compares pre-onset seconds only with clean seconds that have the same number of tracked people: AUROC 0.864 against
0.869 unmatched, and 0.890 when both sides have 4 or more people. Crowd size is not what the stream relies on.

## 4. Fold-1 confirmation (single run)

Source: [`benchmark/results/confirmation/report.md`](../benchmark/results/confirmation/report.md). 1,214 test
videos, 75 eligible fights, 790 clean clips (30.7 h). Thresholds come from the validation slice and were fixed
before fold 1 was read.

| Head | AUROC | Event-level AUROC (M8) | Gain over scene-only (M8) | Rule | Confirmed |
|---|---|---|---|---|---|
| Early | 0.905 [0.848, 0.948] | 0.906 | +0.155 [0.081, 0.226] | M1 lower bound > 0.50 and positive M8 gain | yes |
| Imminent | 0.893 [0.843, 0.938] | 0.919 | +0.136 [0.057, 0.206] | M1-near lower bound > 0.60 and Success@10 above the position null at 2 FA/h | yes |

| Head | Target | Fold-1 FA/h | Success | Threshold-aware CI | Position null |
|---|---|---|---|---|---|
| Early (Success@30) | 1 / h | 0.33 | 16.0% | [1.9, 30.6]% | 3.4% |
| Early (Success@30) | 2 / h | 0.72 | 22.7% | [13.7, 38.8]% | 4.3% |
| Early (Success@30) | 5 / h | 2.64 | 36.0% | [25.0, 51.8]% | 8.6% |
| Imminent (Success@10) | 1 / h | 0.42 | 10.7% | [1.6, 22.4]% | 4.0% |
| Imminent (Success@10) | 2 / h | 0.85 | 20.0% | [10.2, 39.2]% | 5.4% |
| Imminent (Success@10) | 5 / h | 3.23 | 45.3% | [27.5, 58.6]% | 11.7% |

The chart data behind the fold-1 ROC curves and confusion matrices comes from a deterministic replay of the
confirmation that reproduced every committed metric (`benchmark/results/confirmation_replay/`).

## 6. External test (CCTV-Fights + MEVA, pre-registered)

Plan: [`benchmark/external/external_test_plan_v1.md`](../benchmark/external/external_test_plan_v1.md), frozen before
any external score. Results: [`benchmark/results/external_v1/report.md`](../benchmark/results/external_v1/report.md).
Both frozen heads were applied unchanged with the fold-1 thresholds. The scorer first reproduced the committed
fold-1 AUROCs exactly (0.9048 early, 0.8930 imminent). Fights: NTU CCTV-Fights evaluation set (985 videos, 145
eligible events). Clean footage: 355 MEVA clips (29.6 h of scripted multi-person activity, fixed cameras).

| Head | Threshold | MEVA FA/h | Success on CCTV-Fights | AUROC vs MEVA seconds | AUROC vs calm seconds of the same videos |
|---|---|---|---|---|---|
| Early (Success@30) | 2 / h | 0.00 | 24.1% [17.3, 31.5]% | 0.963 [0.942, 0.978] | 0.398 [0.337, 0.459] |
| Imminent (Success@10) | 2 / h | 0.00 | 28.3% [20.5, 36.6]% | 0.984 [0.978, 0.988] | 0.481 [0.440, 0.524] |

Both heads pass the pre-registered criterion, but only trivially. MEVA raised no alarm at all, so the random
reference is zero. MEVA is also easy to separate from fight footage. The informative control is the last column:
within the same CCTV-Fights videos, the seconds before a fight do not score higher than calm seconds far from any
fight, and for the early head they score lower. Counted on those calm seconds, the alarm rate is about 20 per hour,
and random alarms at that rate would reach Success@30 0.39 and Success@10 0.46, above the observed values. The
external test therefore confirms that the model separates fight videos from calm footage, and gives no evidence
that it detects the build-up to a fight inside a video. This matches the flat score-versus-time curve on VA30.

## 7. What the results do and do not show

- Both heads beat source-and-time controls by a wide margin out of sample. This is a real, if modest, warning
  signal at low false-alarm rates.
- The warning comes mostly from the type of footage and setting: the score is roughly flat over the two minutes
  before onset, and the first frame of a clip is almost as informative as the seconds just before the fight.
- Movie and YouTube clips carry most hits and most false alarms; real CCTV is quiet but gets fewer early warnings.
- On an external fight dataset, the score does not separate pre-fight seconds from calm seconds of the same videos.
- Results use one seed, linear probes only, AI-reviewed onsets and no audio model.
