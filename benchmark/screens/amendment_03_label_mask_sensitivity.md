# Amendment 03 to screen_set_v1: label-mask sensitivity analysis

Date: 2026-10-15. Registration amended: `benchmark/screens/screen_set_v1.json`
(sha256 `8831e16e664c895271b9cd77e0bab68733ca85a47655708998ab51d5e625797b`).

**No block outcome had been read when this amendment was written.**

## Why

The label audit (`docs/LABEL_AUDIT.md`, `benchmark/audits/label_audit_v1.csv`) found that early high aggression
scores in the 30-120 s lead window are mostly label noise: of 27 audited events, 13 showed violence already under way
before the annotated onset (10 of them HIVAU-onset UCF videos), 9 were teacher errors or unclear, and only 5 were
genuine non-violent escalation (all movie clips). UCA dense captions add 34 events whose violent captions end at
least 10 s before the onset.

## What changes

Nothing in the primary analysis. The frozen label policy (no proposed masks in v2) stays primary and decides every
gate and class.

Added: a **secondary sensitivity analysis**, reported next to the primary one and never used for a decision:
- **S1:** apply `benchmark/audits/soft_masks_v1.json` (sha256 in the file's sidecar) without the `uca_weapon_only`
  entries: masked ticks are removed from positives and negatives exactly like `masked` ticks in
  `anticipation/data/labels.py`.
- **S2:** the same including the `uca_weapon_only` entries.
- For each Phase A block, report M1 (band), M1-long, M3p and M4n under S1 and S2 with the same probes,
  folds and bootstrap as the primary run, plus the difference to the primary M1. Events whose whole 30-120 s window
  is masked drop out; report how many.
- Interpretation rule, fixed now: a block whose primary G1 PASS turns into a G1 FAIL under S1 is reported as
  "signal explained by onset label noise". Its primary class is unchanged.
