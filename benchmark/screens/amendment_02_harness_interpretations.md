# Amendment 02 to screen_set_v1: harness interpretations

Date: 2026-10-15. Registration amended: `benchmark/screens/screen_set_v1.json`
(sha256 `8831e16e664c895271b9cd77e0bab68733ca85a47655708998ab51d5e625797b`).

**No block outcome had been read when this amendment was written.** Only coverage dry runs exist
(`benchmark/results/cue_screen/dry_run_coverage.json`, `run_report.json`); no metric has been computed on a real
feature block. Harness code at commit `fd08c26` (`anticipation/screen/`).

The frozen file wins wherever it is explicit. These are the places where it is ambiguous or where the harness
deliberately departs from it, fixed now so they cannot be chosen after seeing results.

## Departures

1. **Tail-hit threshold source (M1-sparse).** The frozen field is `tail_train_quantile`. The harness takes the
   99.5th percentile of the fold model's scores on the clean **validation** slice instead of on clean training
   ticks, because training scores are in-sample and their upper tail is biased by the fit. Folds, cohorts and the
   quantile are unchanged.
2. **G1 multiplicity (stricter).** One Holm family covers the M1 test, the standardized-pAUC test and the
   first-segment tail-hit test of every AVAILABLE block in a phase. The sparse route passes only with its own
   Holm-adjusted p < 0.05 in addition to the frozen point thresholds.

## Interpretations of ambiguous text

3. **G2 with an undefined specificity ratio.** `M3p upper < 0.55` is FAIL even when the ratio is undefined; the
   ratio clauses (PASS with ratio >= 0.5, FAIL with ratio < 0.25) need a defined ratio.
4. **Sparse tail events.** G1's tail-hit route uses first-segment events only, like the main route; the
   all-segment tail is reported as `tail_all_segments` and does not gate.
5. **P0 directions.** Only the frozen low-risk concepts are lower = risk (calm, intervention, separation, retreat,
   apology, de-escalation); "retreat" is mapped to the prompt concept `walk_away`. All other columns are
   higher = risk.
6. **Block definitions.** `pose_state_pair_identity_ucf_ubi` = `pose_state` + `pose_state_pairs` columns, UCF and
   UBI videos only. `motion_camera_shortcut` = motion columns `cam_flow_mag`, `cam_shake`, `cam_flow_mag_5s`,
   `hist_dist_max`, `cut_1s`, `cuts_5s`, `time_since_cut`; `motion_no_camera` = all other motion columns.
7. **P1b.** The early baseline is ticks 1-10; P1b windows start at tick 11 and its M1 is compared with P1 on the
   same t > 10 ticks.
8. **Nuisance model N.** The person-masked scene embedding (frame at 2 s) contributes zeros at tick 1, where it is
   not yet observed.
9. **Exemptions and missing gates.** The context-cue G2 exemption is not implemented; an exempt G2 could at most
   give PROVISIONAL. A gate that cannot be evaluated keeps the frozen class table and is listed in
   `unevaluated_gates`.
10. **Training caps.** At most 25,000 positive and 50,000 negative training ticks per fold (seeded); PLS is applied
    per tick before the window summaries.

Not implemented (reported in every run): P2 GRU, full-dimensional L2 sensitivity, forward selection and the
fold-1 confirmation (a separate fused-model step), M6 sustained leads at 90/120 s and clean-alarm-time operating
points, M7 film-matched / news-compilation / visible-conflict splits, and the aggression-level and teacher
reference controls on the screening cohort.
