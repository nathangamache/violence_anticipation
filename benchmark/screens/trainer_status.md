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
