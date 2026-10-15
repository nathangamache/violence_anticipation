# VA30 onset correction (labels v3)

Status (2026-10-15, America/Detroit): **IN PROGRESS; confirmation remains frozen.**

| Item | Value |
|---|---|
| Original eligible30 events | 551 (official 75, HIVAU 374, UBI 102) |
| Scan | Full scan running; calibration complete; cuda:0 only, bf16, batch 16 |
| Visual review / corrected labels | Not started / not released |
| Corrections by verdict and onset source | Pending visual review |
| Median shift / scan miss rate | Pending visual review |
| New eligible30 counts per fold | Pending corrected labels |
| Registration v2 / Phase A v3 | Not created / not run |
| Fold-1 model evaluation | Not run; reserved for the lead |

## Method fixed before curation

Only raw videos, the purpose-specific Qwen scan and visual inspection may inform corrections. No files under
`benchmark/results/`, existing feature arrays, prompt scores or fusion outputs are read during curation. The
requested audit documents describe known failure modes; their model rankings are not used to select events.
Every original eligible30 event is scanned, including fold 1. Original benchmark and registration files remain
unchanged. Work happens in the separate `~/projects/va_onsets` checkout.

The onset definition is the first physical aggression against a person, including a shove, grab, punch, slap,
kick, object strike, discharge toward a person, object thrown at a person, vehicle used as a weapon or police
force. Arguing, pointing and squaring up alone do not qualify (`YOUTUBE_HOLDOUT.md`, section 2).

`anticipation.data.onset_scan` scores one raw frame per prompt with local Qwen2.5-VL-7B-Instruct. It uses
bf16 on cuda:0, approximately 448-square pixels with preserved aspect ratio, two CPU threads and one decoder
thread. Sampling is one second apart from `max(clear_start, onset - 120)` through `onset + 15`, truncated at the
video boundary. The score is `sigmoid(logp_Yes - logp_No)`, the probability conditional on the two answer tokens;
it is not the unnormalised vocabulary probability of Yes. Both token log-probabilities are retained.

Calibration samples 800 clean-labelled videos (one random frame each) and 400 annotated violent segments,
without consulting model scores, using seed 301015. The threshold is the higher-interpolated 98th percentile
of clean scores; the comparison is strict `>`. Report both the empirical clean false-positive rate and the
annotated-violence sensitivity. These calibration classes can themselves contain label noise. Threshold selection
uses the clean sample only. Three high frames in a five-frame window wholly before onset flag EARLY_VIOLENCE;
the candidate is the first high frame of the earliest qualifying window. No high frame during the observed
`[onset, onset + 15]` samples flags NO_VIOLENCE_AT_ONSET.

Scan manifests, calibration frames, per-event frame probabilities, flags and the deterministic review queue
are stored outside git in `data/annotation/onset_v3/`. The GPU run has a 10,500-second cumulative wall-time cap
and checkpoints after each event. An incomplete scan cannot be treated as complete.

Visual review covers every flag and a seed-301015 random sample of 40 non-flagged events. Render 1-second
contact sheets with context and 0.5-second sheets around candidate onsets using `tools/contact_sheet.py`.
Changed labels require two blind passes; reconcile disagreements on fine sheets. Keep uncertain cases as
`unclear` rather than inventing a correction. These are AI visual judgements, not human ground truth.

`anticipation.data.apply_corrections` validates source hashes, review completion, original event identity,
correction direction and the two-pass evidence before writing separate outputs under `benchmark/v3/`.
The correction table contains visually reviewed events only; unselected, non-flagged events retain their
original labels without a claim of visual verification. It removes no-violence segments, normalises remaining segments and recomputes `labels.event_table` with
existing other-anomaly and cross-clip blockers. A video without remaining violence becomes `unlabelled`,
never a clean negative. Existing source-film masks are retained; their regeneration is not yet implemented.

## Results

Pending the complete scan and visual review. No corrected labels or replacement registration have been released.

## Implementation checks

The screen tests passed (49 tests), as did the correction, routing, blind-review reconciliation and
registration-rebuild checks (14 tests). No benchmark model result file has been opened. The v2 registration
alias deliberately refuses to load until the completed registration has a pinned hash.

## Scan calibration checkpoint

The fixed threshold is **0.022977370768785477**. On the calibration sample, strict `P(Yes) > threshold`
selects 15 of 800 clean-labelled frames (**1.875%**) and 192 of 400 annotated violent frames (**48%**).
The clean sample fixes the threshold; the violent sample only measures sensitivity. The full event scan
contains **44,132 frames**. Calibration and the first two event checks took 550.6 seconds; the resumed scan
has a 9,900-second cumulative runtime cap, leaving room for initial validation within about three GPU-hours.

The first implementation checkpoint is commit `25e7062`. Label curation and registration completion remain
pending; this checkpoint does not authorise reading fold-1 model outcomes.
