# Evaluator handoff for the trainer (2026-10-15)

- Preregistration is frozen in `screen_set_v1.json`; SHA256 `8831e16e664c895271b9cd77e0bab68733ca85a47655708998ab51d5e625797b`.
- Screen folds 0/2/3/4 only. Exclude confirmation fold 1 from training and validation as well as test.
- Opt-in in-process API: `EvalConfig(phase0_metrics=True, cluster="split_group", person_counts_root="data/features/pose_interact", horizons_csv="benchmark/extract_manifest_v1.csv")`.
- `phase0_metrics` adds `seeds[seed].phase0`: band, person-stratified band, M8, partial AMOC, sustained warnings at 1/2/5% warning-time budgets with their own position null.
- Existing configs and metric values remain unchanged when the option is absent. `--cluster split_group` is explicit and also applies to validation threshold bootstrap.
- A JSON `person_counts_root` with `root` and a `files` SHA256 mapping freezes pose coverage across arms. Directory roots remain supported.
- Label audit masks will be proposals only, never silently applied to v2. Audio and H1 extraction prerequisites remain separate.
- No-context reference and proposal artifacts are in progress. Do not read new screen block outcomes before Phase 0 prerequisites and the H1/diegetic audits are satisfied.
- A rebase conflicted with the trainer's `docs/MODELS.md` rewrite; it was aborted, and that file was left intact. Integration updates live here to avoid simultaneous status edits.

## Completed reference and proposal handoff

- Full no-context aggression and first-frame references are in `benchmark/results/b_aggression_nocontext*/`; covered core subset: 1,031 videos / 198 eligible30 events. These are five-fold reference readouts, not candidate screens.
- Band AUROC aggression / first frame: 0.749 [0.675, 0.805] / 0.620 [0.509, 0.711]. Paired gain: 0.129 [0.056, 0.210]; S_ramp90 gain: 0.172 [0.072, 0.292].
- M3p aggression: 0.723 [0.664, 0.781], specificity ratio 0.849. Sustained warning points all remain below their own position nulls. Full CIs, M2/M8, source-proxy and operating-point tables are in `docs/VA30.md` section 11.
- The optional long-band positive cohort filter preserves all clean controls; paired first-frame comparisons use identical source/time weights and split-group draws. Real-world/movie breakdowns now include M8 and all event operating points.
- `benchmark/audits/phase0_nocontext_v1/summary.md` records 54 reviewed cases (53 onset>=32 s; 51 VA30 eligible), a/b/c core53 counts 18/17/18, 19 UCA strike/discharge proposals and 17 weapon-only flags. They are PROPOSALS ONLY and overlap the visual mask proposals; v2 catalog is unchanged.
- The separate existing `label_audit_v1.csv` uses aggression_level across all eligible events, whereas this audit uses first-event violent.score. Its 51/27/24 counts are reproduced and it is left intact. Lead adjudication is required for disagreements and uncertain calls.
- The preregistration SHA256 remains 8831e16e664c895271b9cd77e0bab68733ca85a47655708998ab51d5e625797b. No candidate screen metrics were read by this evaluator/baseline agent to select the registration or these comparisons.
- Audio/diegetic and broader human spot-check prerequisites remain with their respective owners; these artifacts do not declare those prerequisites satisfied. Existing screen amendments remain intact.
