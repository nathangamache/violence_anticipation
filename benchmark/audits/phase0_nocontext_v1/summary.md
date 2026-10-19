# Phase 0 no-context label audit proposals

Status: **PROPOSAL_NOT_APPLIED**. No `events.json`, `videos.csv`, onsets, masks or clear-history boundaries were changed. The assistant reviewed all 54 sampled contact sheets; these are proposals for lead adjudication, not independent human ground truth. Sparse stills cannot establish intention or rule out violence between sampled frames.

The selection uses the no-context record's **`violent.score >= 0.8`**, not its `aggression_level`, at `raw_frame_idx / catalog fps`, on unmasked pre-violence records with `30 < time_to_onset <= 120`. This finds 54 first-event cases. The expected 53 is reproduced by excluding one supplemental UCF case: its onset is 31.333 s, leaving only 29.333 s maximum alarm lead on the 1 Hz, 2-of-3 clock. Two further mixed-anomaly cases (`V036`, `V046`) fail clear-history eligibility. All 54 are retained and reviewed; 51 are VA30 eligible30. This keeps record-time selection separate from event-metric eligibility.

| Cohort | (a) Non-violent contact/posturing | (b) Possible violence before annotated onset | (c) False positive/unclear |
|---|---:|---:|---:|
| All 54 record-time candidates | 18 | 17 | 19 |
| 53 with onset at least 32 s | 18 | 17 | 18 |
| 51 VA30 eligible30 cases | 17 | 17 | 17 |
| UCF, all selected cases | 9 | 12 | 14 |
| XD movie, all selected cases | 8 | 2 | 3 |
| XD YouTube source, all selected cases | 1 | 3 | 2 |

Category (b) has 10 high-, 6 medium- and 1 low-confidence calls. Several use caption corroboration, explicitly identified in each row; `V042` (a movie clip) is particularly uncertain. Category (a) includes weapon threats, coercion and ordinary holding, so it does not establish a genuine escalation cue. Three purported YouTube-source cases (`V050`, `V052`, `V054`) appear cinematic; source-kind is only a proxy for real-world footage. The model-selected audit is not representative of all lead-ins.

Earlier violence is substantial but **does not dominate** this 53-case cohort. The contact-state hypothesis remains unverified, and these results cannot justify treating every high contact score as anticipation. Label disagreement also depends on the violence definition: for example, `V022` is categorized as armed coercion without a demonstrated strike here, whereas the separate aggression audit proposes earlier violence. The lead should adjudicate such disagreements and uncertain calls before any catalog amendment.

The existing `../label_audit_v1.csv` uses a different selection: `aggression_level >= 0.8` across all eligible events. The reproduction script recovers its 51 candidates: 27 with a high frame in the clear lead-in and 24 in earlier annotated/excluded spans. It is preserved intact and must not be conflated with this first-event `violent.score` audit.

## UCA soft-mask proposal

`uca_soft_masks_proposal.json` checks the **first violent segment per UCF video**, requiring caption end at least 10 s before onset and caption start at or after clear_start. The proposed mask runs from the earliest vetted strike-caption start to the annotated onset; it does not replace the onset or silently redefine clear_start. Across the full available lead-in, 23 strike-word candidates become **19 physical-strike/firearm-discharge proposals** (15 eligible30), 1 restraint-only adjudication case, and 3 rejected matches. The rejected matches describe kicking a bag, a car hitting a pole, and being “kicked out” of a shop. One restraint/dragging case is kept separate. There are **17 weapon-only sensitivity flags**; weapon display alone is retained in the primary protocol. This differs from the other audit's all-event unit of analysis.

`visual_b_masks_proposal.json` separately records the 17 visual/caption-supported category (b) proposals with confidence. They can overlap the UCA proposals and must not be summed as distinct events. No proposal is applied automatically, especially low-confidence cases. Every artifact records catalog and input checksums; adjacent `.sha256` files cover the JSON outputs.

## Reproduction and review

From the repo root, using the project Python and the shared-server thread limits:

```
$PY benchmark/audits/phase0_nocontext_v1/reproduce.py --help
$PY benchmark/audits/phase0_nocontext_v1/reproduce.py --skip_render
$PY benchmark/audits/phase0_nocontext_v1/review_proposals.py
```

`reproduce.py` rebuilds the selection and caption candidates; omit `--skip_render` to regenerate contact sheets using `tools/contact_sheet.py`, or use `--render_only` to render the existing selection. `visual_reviews.json` is the recorded visual judgment, not a newly inferred label. `review_proposals.py` reapplies those judgments and the explicit caption adjudications. Contact sheets sample the high-scoring frame and annotated onset neighborhoods; each JPEG is smaller than 300 KB. `early_frames_proposal.csv` is the compact review index; the JSON retains selection provenance and clock eligibility.
