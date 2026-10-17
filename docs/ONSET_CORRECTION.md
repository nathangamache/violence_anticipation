# VA30 onset correction (labels v3)

Status (updated 2026-10-17, America/Detroit): **IN PROGRESS; confirmation remains frozen.**

| Item | Value |
|---|---|
| Original eligible30 events | 551 (official 75, HIVAU 374, UBI 102) |
| Scan | Complete: 551 events, 44,132 frames, 1.86 tracked GPU-hours |
| Initial review queue | 419 flagged + 40 random non-flagged = 459 events |
| Newly eligible follow-ups | Direct blind review as discovered; currently 1 queued |
| Visual review / corrected labels | Pass A is reviewing all 459 cases; Pass B is reviewing the owner-approved subset / not released |
| Corrections by verdict and onset source | Pending visual review |
| Median shift / scan miss rate | Pending visual review |
| New eligible30 counts per fold | Pending corrected labels |
| Keep audit | Seeded 10% sample; disagreement rate pending |
| Registration v2 / Phase A v3 | Not created / not run |
| Fold-1 model evaluation | Not run; reserved for the lead |

Review protocol update (2026-10-17): the owner approved a time-budget change. Pass A reviews all 459 original
visual-review cases and every newly eligible follow-up. Pass B reviews each A verdict that would change a label,
a seeded random 10% sample of A's keep and unclear verdicts, and every follow-up. The fixed audit seed is 301018;
the selected sample is 10% rounded up. Pass B remains blind to A's judgments. Its 33 original cases completed
before this change stay in the record. Reconcile every A/B disagreement on 0.5-second sheets. A label change
requires agreement or a documented reconciliation. A single-pass keep remains a keep. If disagreement among the
sampled A keeps exceeds 15%, stop label release and report the rate. Final one-pass/two-pass counts and the keep
audit rate will be recorded here and per event in `onset_corrections.json` before labels are released.

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

Calibration samples 800 clean-labelled videos (one random frame each), excludes frames inside existing
catalog hard/soft/other-anomaly masks (14 exclusions; 786 usable clean frames), and samples 400 annotated violent segments,
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
Three CPU workers prepare cached contact sheets, with one decoder thread each. Passes A and B use
independent shuffled identifiers and separate model contexts, blind to each other. Review work order may
prioritize cases with another completed pass solely to make reconciliation available sooner; no verdict or
case-specific observation is shared. Under the owner-approved 2026-10-17 protocol, A covers all 459 cases;
B covers A's label-changing verdicts, the seeded 10% keep/unclear audit, and every newly eligible follow-up.
Previously completed B reviews remain recorded even when they fall outside that selected subset. Every label
change needs agreement from both passes or a documented reconciliation. Reconcile disagreements on 0.5-second
sheets. Single-pass keeps remain keeps. Stop before release if the sampled keep disagreement rate is above 15%.
Keep uncertain cases as `unclear` rather than inventing a correction. These are AI visual judgements, not human
ground truth.

Each review targets one existing episode. Prior/following annotated segments, the current clear-start and
segment end are advisory context, not hard boundaries. Earlier aggression continuous with the target episode
can move its onset across an old boundary. A completed earlier episode or separate replay must not replace
the target onset. Reviewers record a separate, previously unannotated episode in the lead-in as an additional
span for reconciliation instead of absorbing it into the target. Completed reviews are independently audited
under this episode-scope clarification and marked `scope_protocol: episode_v2`. Autonomous animal attacks
are marked `unclear` because the benchmark targets interpersonal violence; human-directed animal force,
including a deployed police dog, counts as physical aggression. Incidental sports/dance contact and
swings into open space do not establish aggression against a person. Clear directed aggressive contact counts
without requiring proof of hostile intent; unresolved contact is `unclear`.

Separate missed episodes or omitted continuations beyond an old endpoint require explicit visual
reconciliation on fine sheets before being added to the segment list. Overlapping additions merge with the
existing episode rather than inventing another onset. Their ten-second aftermath ends the target's clear history when eligibility is recomputed.
The non-flagged sample counts an event requiring such an addition as a scan miss even if its target onset
is kept. Additional spans are retained alongside the target verdict in the correction artifacts.
`no_violence` requires the entire original annotated segment to lack physical aggression. If its first
physical aggression occurs later within that segment, use `later` even across a montage cut; use `unclear`
when the first force cannot be resolved. A separately annotated following event is not substituted for it.

`anticipation.data.apply_corrections` validates source hashes, review completion, original event identity,
correction direction and the two-pass evidence before writing separate outputs under `benchmark/v3/`.
The correction table covers all 551 scanned events plus direct visual follow-ups. Unselected, non-flagged events retain their original
labels with `visually_reviewed: false`, empty passes and an explicit scan-only evidence basis. Counts distinguish
these retained labels from visually verified keeps. It removes no-violence segments, normalises remaining segments and recomputes `labels.event_table` with
existing other-anomaly and cross-clip blockers. A video without remaining violence becomes `unlabelled`,
never a clean negative. XD cross-clip hard/soft masks are regenerated with the catalog's existing `source_masks`
rules, followed by eligibility recomputation for every affected clip. Existing contaminated clips are not
promoted to clean negatives. Before corrections, this reconstruction reproduces the masks of all 3,124 XD
source-span records exactly. H1/H1p screening cohorts and intervals remain frozen.

Corrections can make a previously ineligible original event eligible, for example after removing a false
preceding segment. These events receive two blind visual follow-up passes with the same onset definition,
1-second lead-in sheets and 0.5-second onset sheets, without Qwen rescoring. Separate follow-up queues and
records retain `scanned: false`; they do not alter the original 551-event scan or its random control sample.
Stable hashed blind identifiers allow the follow-up queue to grow. Finalization refuses a catalog with a
newly eligible original event missing both follow-up reviews. Added spans already have two passes and
explicit reconciliation. Eligibility is checked again after follow-up corrections before registration.

The v2 registration builder rebuilds positive events, subsets and intervals from the v3 catalog while
preserving the v1 fold/validation memberships and exact clean/H1/H1p cohorts and intervals. Preserving H1
intervals avoids reintroducing the documented one-tick difference from the older computed-interval rule.

## Results

The scan is complete. Visual review is in progress under the revised protocol. No corrected labels or replacement
registration have been released.

## Implementation checks

The screen tests passed (49 tests), as did the correction, routing, blind-review reconciliation and
registration-rebuild, calibration-mask and separate-episode, cross-clip-mask and newly eligible follow-up checks (19 tests). No benchmark model result file has been opened. The v2 registration
alias deliberately refuses to load until the completed registration has a pinned hash.

## Scan calibration checkpoint

The final threshold is **0.01798621006309986**. On the usable calibration sample, strict `P(Yes) > threshold`
selects 15 of 786 unmasked clean-labelled frames (**1.908%**) and 214 of 400 annotated violent frames (**53.5%**).
The clean sample fixes the threshold; the violent sample only measures sensitivity. The full event scan
contains **44,132 frames**. Calibration and the first two event checks took 550.6 seconds; the resumed scan
has a 9,900-second cumulative runtime cap, leaving room for initial validation within about three GPU-hours.

An initial implementation omitted the catalog exclusion masks from clean calibration. A metadata-only check
found 14 affected frames among the sampled 800. These were excluded deterministically before visual review,
and all event flags were recomputed from cached raw-frame probabilities. The superseded threshold was
0.022977370768785477 (15/800 clean frames, 192/400 annotated violent frames). Both the initial calibration
and the excluded frame list are retained outside git. This fixes sample eligibility; no screen model score
was consulted, and formal blinded visual review had not begun when the threshold was corrected.

The first implementation checkpoint is commit `25e7062`. Label curation and registration completion remain
pending; this checkpoint does not authorise reading fold-1 model outcomes.

## Completed scan (2026-10-16)

All 551 original eligible30 events have scores for all 44,132 planned frames. Original source hashes match
the scan manifest. Tracked calibration and scan runtime is 6,689.8 seconds (1.86 hours), plus initial
validation/loading overhead; GPU 0 was the only inference device.

| Onset source | Events | Early flag | No-violence-at-onset flag | Either flag | Random non-flagged review |
|---|---:|---:|---:|---:|---:|
| official | 75 | 46 | 3 | 48 | 7 |
| hivau | 374 | 239 | 60 | 280 | 31 |
| ubi | 102 | 85 | 13 | 91 | 2 |

There are 370 EARLY_VIOLENCE flags and 76 NO_VIOLENCE_AT_ONSET flags, with 27 events carrying both,
for 419 unique flagged events. Adding the 40 random non-flagged controls gives 459 visual-review cases.
Flags are screening candidates, not verdicts. The scan does not itself establish label accuracy.
