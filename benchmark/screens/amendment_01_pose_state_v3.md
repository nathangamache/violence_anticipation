# Amendment 01: pose_state density-controlled columns (extractor VERSION 4)

Date: 2026-10-15 (revised the same day, before any screen outcome)

## Status of the screen

No screen outcome had been read when this amendment was written or revised. `screen_set_v1.json` (hash in `screen_set_v1.sha256`) has not been run: no block score, AUROC, gate result or
per-cue number from the pre-registered screen has been looked at.

## What changed

Block `pose_state_identity_free` (extractor `pose_state`) goes from 88 to 147 columns (extractor VERSION 2 to 4; a VERSION 3 with the same 147 column names was produced and reviewed earlier the same day and is superseded, none of its files remain).

- The 88 VERSION 2 columns are unchanged in name, order and computation (they keep the unchecked head facing).
- 59 columns are appended: 7 per-tick (`eng_pair_frac`, `contact_pair_frac`, `people_engaged_frac`,
  `top_pair_isolation`, `dyad_eng`, `dyad_scene`, `eng_max_excess`) and 13 window summaries for each of the 30 / 60 /
  120 / 300 s windows (52).
- The appended columns use a back-view-checked head facing (nose confidence >= 0.5 and above both ears, an eye visible;
  otherwise unknown, never mutual) and are normalised by the number of valid pairs / people, or corrected by a fixed
  synthetic null table (`EXP_ENG_MAX`, random placement of n people in a 16 x 4 bh box; no dataset or label involved).
- VERSION 4 versus VERSION 3: `eng_pair_frac` and `people_engaged_frac` use a one-sided soft engagement (the larger of the
  two checked facings, times proximity) instead of strict mutual engagement, and `contact_pair_frac` no longer requires a
  facing. Reason: the strict fractions were nonzero in about 2% of ticks at n >= 5 and so were decorrelated from crowd
  size only by being nearly always 0. No other column definition changed.
- `pose_state_pairs` content is unchanged; its files were regenerated and carry version 4.

Why: reviewer evidence that the interaction columns (`close_mutual`, `eng_max`, `contact_mutual`, `interpose` and
their windows) are dominated by crowd density (pair counts grow as n^2; in UCF frames with 5 or more people normal
scenes score like fights), that people seen from behind get a hallucinated nose and a false mutual facing, and that
contact saturates in dense scenes. The additions let the block be read without the density shortcut. Definitions:
`docs/FEATURES.md`, section "VERSION 4 additions".

## Design-time use of labels (disclosure)

The VERSION 4 fraction definitions were chosen by looking at within-stratum clean-vs-violent AUC and the correlation
with n_people on a design sample of 300 random clean negatives and 300 interpersonal violent videos (seed 1), and checked
on a second sample (seed 2, 91 videos shared). This is reviewer-style diagnostics on the extractor, not the pre-registered
screen; no screen block, gate or fold result was computed. Result: the new fractions are alive (24 to 33% nonzero at
n >= 5), rho with n_people 0.29 to 0.36 versus 0.55 to 0.57 for old `eng_max`, and within-stratum AUC 0.50 to 0.58, similar to the old columns (0.51 to 0.59). They are density-reduced, not density-free, and not strongly discriminative
within a stratum. `eng_max_excess_max_*` windows remain partly density coupled (rho 0.52 for the 30 s window over all
ticks) and `top_pair_isolation` has no stable sign across strata (AUC 0.50 / 0.45 / 0.40 at n = 2 / 3-4 / >= 5).

## What is not verified

Equivalence of the 88 old columns to the VERSION 2 output is not verified. The VERSION 2 files were overwritten and the
extractor file was never committed, so no VERSION 2 code or data exists; only a 2-video comparison made before the
overwrite supports it. Available evidence: on 20 random videos the first 88 columns of a VERSION 4 recomputation equal
the stored VERSION 3 files exactly and do not change when the checked facing is forced to unknown, so they do not read it.

## What did not change

Block names, the probes, the gates, the folds, cohort and set S in `screen_set_v1.json` are unchanged. The block still
means "all columns of the `pose_state` extractor", so it now includes the 59 added columns. The block remains
identity-free; `pose_state_pair_identity_ucf_ubi` is unchanged apart from inheriting the same extractor columns.

## Artifact hash

`anticipation/features/pose_state.py` sha256:
`b528a9d47a7f9a7fca79b411e6d4a7d1f7a2fcb209cec08c306aaeba6aac10b7`
