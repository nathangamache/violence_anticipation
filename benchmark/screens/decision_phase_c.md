# Phase C decision: NO-GO for VLM extraction beyond the pilot

Date: 2026-10-17. Decided by the lead under `docs/MODALITY_PLAN.md` section 6.7 and the frozen registration
`benchmark/screens/screen_set_v1.json` (+ amendments 01-05), from the pilot report
`benchmark/results/vlm_pilot/pilot_report.md` (commit `d485c48`; interim v1 labels; folds 0, 2, 3, 4; fold 1 untouched).

## Criteria (best arm must meet all)

| Criterion | Outcome |
|---|---|
| (i) paired band M1 gain over the no-context `aggression_level` on the 112 shared teacher videos (CI lower > 0), or G3 PASS where `aggression_level` has G3 FAIL | Met only by `hidden_A` (+0.097 [0.038, 0.190]); rating arms -0.016 to -0.108, all CIs include 0; `aggression_level` G3 is INCONCLUSIVE, not FAIL |
| (ii) G2 not FAIL | Rating arms PASS; `hidden_A` FAIL (so no arm meets (i) and (ii) together) |
| (iii) S_real band M1 >= 0.60, CI lower > 0.50 | All arms PASS (0.688-0.817) |
| (iv) no contamination flag | UCF flagged in violent and H1 strata, so UBI and the curated hold-out decide: UBI is strong (C band 0.920, near 0.931) but on the hold-out arm B reaches Success@10 0.114 [0.000, 0.271] vs a position null of 0.477 (Success@30 0.023 vs 0.118), despite near-window M1 0.877 |

## Decision

**NO-GO.** No arm meets (i)-(iii) jointly, and the decisive real-footage check fails at the event level. For the
imminent head (amendment 05), the rating arms pass N1-N3 on the pilot set, but their event results (C: Success@10
0.300 at 1.88 FA/h) do not exceed the already selected SigLIP 2 stream on the screening folds (0.294 at 1.48 FA/h,
0.408 at 2.44 FA/h) at roughly 10x the compute per video-second.

Consequences, as the plan specifies: SigLIP 2 (embedding and prompt set v2) remains the semantic stream; the VLM is
used only offline as a labeller (it already drives the onset scan for labels v3). No extraction of the rest of set S;
the remaining Phase C GPU budget is released. The pilot features and evaluator stay in the repository for reference.

Lesson recorded for the report: high near-onset classification (frame AUROC) does not imply a usable >= 10 s alarm
lead; event-level metrics with position nulls are necessary.
