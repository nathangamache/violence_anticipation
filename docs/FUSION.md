# Fusion: two-head forward selection and one-shot fold-1 confirmation

Status 2026-10-15: **interim results on v1 labels** (screen_set_v1, folds_v2). Onset labels are being corrected (v3) and a
new registration will follow, so every number below is to be re-run (section 7). P2 (GRU) was **not** run on real data
in this interim pass; the P1 results stand as the selected models by default of the rule. Fold 1 was never read.

Code: `anticipation/screen/fusion.py` (selection, gates, spec), `anticipation/screen/confirm.py` (one-shot fold 1),
tests `tests/test_fusion.py` (synthetic only). Results: `benchmark/results/fusion/` (early head at the root, imminent
head in `imminent/`). Rules: `benchmark/screens/screen_set_v1.json`, `summary.md`, amendments 01 to 03 and
`amendment_05_imminent_head.md`.

## 1. Design

Two heads on the same machinery as the screen (same `ScreenSet`, P1 probe, window summaries, PLS32 for wide blocks,
validation-slice C choice, time-match weights, 1000-resample paired cluster bootstrap over `split_group`).

| | early-warning head | imminent head (primary, amendment 05) |
|---|---|---|
| target | first-segment ticks with time to onset in (30, 120] vs clean (>= 60 s left) | same with time to onset in (0, 10] |
| candidate blocks | ANTICIPATION and PROVISIONAL classes of Phase A (read from `decisions_phase_A.json`) | blocks passing gates N1, N2, N3 (all nine Phase A blocks tested; CONTEXT_SHORTCUT blocks are eligible) |
| selection metric | paired M8 (window max over [max(t0-120, clear_start), t0-30] vs time-matched clean windows) | paired M8-near (window [max(t0-10, clear_start), t0)) |
| nuisance control N | source x time-bin + person-masked scene, trained on the band target | same, trained on the near target |
| event metric | VA30 Success@30 at 1/2/5 FA/h (position null, random) | Success@10 at 1/2/5 FA/h with threshold-aware CI, own position null at lead 10 (VA30 only provides the Success@30 null) |

Interpretations fixed in code before results: the start block is the one with the largest paired M8 point estimate
(the frozen text does not define "best"); M8 uses all eligible30 events (first-segment M8 is reported); a candidate is
accepted only if the largest paired gain has a 95% CI lower bound > 0 (no multiplicity adjustment across candidates,
none is registered); P2 replaces P1 only if the 3-seed ensemble's paired M8 gain over P1 has CI lower > 0. P2 is a
one-layer causal GRU, 64 units, 180 s, band/near BCE plus event-window MIL loss (coefficient 1), lr 1e-3, weight decay
1e-4, batch 32, up to 20 epochs, patience 4, validation M1 selection, history dropout 0.25, modality dropout 0.15, built
from `anticipation.models.temporal` components on the screen's cohorts and plans. The screen's `p1` reproduces exactly
(siglip2_prompts_v1 alone: M1 0.8350, M8 0.8805, identical to `cue_screen`).

Imminent gates (per block, on its own cohort): N1 M1-near point >= 0.70 and CI lower > 0.60; N2 M3p-near (UCF
first-segment ticks tte in (0,10] vs H1p ticks time-to-anomaly in (0,10], matched on time since start) CI lower > 0.50;
N3 paired M1-near gain of [N + block] over N, CI lower > 0. Reported only: B-detect, official/UBI-onset breakdown.
Sensitivity S1/S2 and Phase C blocks were not run for the imminent head.

## 2. Early-warning head: forward selection (v1 labels, interim)

| step | candidate | M8 | paired M8 gain vs current (95% CI) | M1 band | result |
|---|---|---|---|---|---|
| 0 start | siglip2_prompts_v1 | 0.881 | | 0.835 | start |
| 0 start | siglip2_prompts_v2 | 0.872 | | 0.829 | |
| 1 | pose_interact | 0.886 | +0.005 [-0.004, +0.015] | 0.839 | best gain, CI includes 0: stop |
| 1 | motion_no_camera | 0.886 | +0.005 [-0.003, +0.013] | 0.864 | |
| 1 | siglip2_prompts_v2 | 0.884 | +0.004 [-0.005, +0.011] | 0.841 | |
| 1 | motion_camera_shortcut | 0.880 | -0.001 [-0.006, +0.003] | 0.837 | |

Selected: `siglip2_prompts_v1` alone, P1, C chosen by mean validation M1 over folds. M1 band 0.835 [0.799, 0.866],
M8 0.881; nuisance N: M1 0.784, M8 0.771; M4n +0.066 [0.025, 0.105]; M8 gain over N +0.109 [0.075, 0.143].
Post-selection VA30 events on the screening folds (not a confirmation):

| FA/h target | test FA/h | Success@30 | position null | random | Success@10 | median lead (s) |
|---|---|---|---|---|---|---|
| 1 | 1.16 | 0.235 [0.191, 0.283] | 0.064 | 0.029 | 0.323 | 35.2 |
| 2 | 2.26 | 0.312 [0.258, 0.366] | 0.090 | 0.055 | 0.435 | 35.8 |
| 5 | 5.11 | 0.381 [0.333, 0.429] | 0.159 | 0.120 | 0.484 | 39.2 |

Per source Success@30 at 2 FA/h: xd_movie 0.466, xd_youtube 0.315, ubi 0.314, ucf 0.148.

## 3. Imminent head: gates and forward selection (v1 labels, interim)

| block | M1-near [95% CI] | M3p-near CI | M4n-near [CI] | M8-near | N1 | N2 | N3 |
|---|---|---|---|---|---|---|---|
| siglip2 | 0.912 [0.889, 0.934] | [0.536, 0.722] | +0.114 [0.083, 0.144] | 0.928 | pass | pass | pass |
| siglip2_prompts_v2 | 0.860 [0.831, 0.888] | [0.556, 0.732] | +0.093 [0.065, 0.122] | 0.904 | pass | pass | pass |
| siglip2_prompts_v1 | 0.886 [0.860, 0.910] | [0.480, 0.661] | +0.106 [0.077, 0.135] | 0.924 | pass | FAIL | pass |
| vjepa2_l | 0.827 [0.797, 0.861] | [0.376, 0.556] | +0.051 [0.017, 0.086] | 0.848 | pass | FAIL | pass |
| motion_camera_shortcut | 0.635 [0.592, 0.676] | [0.487, 0.654] | +0.020 [0.003, 0.038] | 0.674 | FAIL | FAIL | pass |
| pose_interact | 0.653 [0.620, 0.687] | [0.436, 0.607] | +0.005 [-0.015, 0.026] | 0.693 | FAIL | FAIL | FAIL |
| pose_state_pair (UCF, UBI) | 0.616 [0.551, 0.678] | [0.386, 0.582] | -0.044 | 0.657 | FAIL | FAIL | FAIL |
| motion_no_camera | 0.600 [0.563, 0.633] | [0.442, 0.612] | +0.015 [-0.004, 0.034] | 0.631 | FAIL | FAIL | FAIL |
| pose_state_identity_free | 0.576 [0.535, 0.617] | [0.409, 0.598] | -0.022 | 0.601 | FAIL | FAIL | FAIL |

Only the two SigLIP 2 blocks pass. Forward selection: start siglip2 (M8-near 0.928) vs siglip2_prompts_v2 (0.904); adding
v2 gains +0.003 [-0.001, +0.006], not accepted. Selected: `siglip2` alone, P1. M1-near 0.912 [0.889, 0.934], M3p-near
0.626 [0.536, 0.722], M8-near 0.928 [0.907, 0.945] (official/UBI first-segment 0.930); nuisance N: M1-near 0.794,
M8-near 0.805; M4n +0.114, M8-near gain over N +0.123 [0.096, 0.152]. B-detect (violence-now probe on pre-onset
ticks, M1-near) 0.903: the near signal is largely the "violence-like scene" signal, not a separate precursor.

Success@10 on the screening folds (post-selection, not a confirmation):

| FA/h target | test FA/h | Success@10 | threshold-aware CI | position null | random | official/UBI onsets |
|---|---|---|---|---|---|---|
| 1 | 1.48 | 0.294 | [0.230, 0.392] | 0.079 | 0.044 | 0.239 |
| 2 | 2.44 | 0.408 | [0.346, 0.493] | 0.103 | 0.072 | 0.387 |
| 5 | 5.19 | 0.518 | [0.458, 0.591] | 0.170 | 0.147 | 0.500 |

Per source at 2 FA/h: xd_movie 0.559, xd_youtube 0.480, ubi 0.302, ucf 0.252.

## 4. P1 versus P2

Not run on real data in this interim pass (decision of the lead: labels are being corrected). A first pass of the early
head was interrupted after about 10 of 12 fold-seed trainings (about 10 to 20 minutes each on 4 CPU threads); validation
M1 peaked after 1 to 11 epochs, so the GRU overfits quickly. Nothing from that partial run is used. The rule and code
are in place and tested on synthetic data; with P1 selected the config freezes `probe_type: p1`.

## 5. Frozen spec and confirmation

`benchmark/results/fusion/confirmation_config.json` (version 2; sidecar `.sha256`) freezes both heads: blocks in
order, probe type, C (mean validation M1 over folds), the nuisance C (median per-fold choice), P2 hyperparameters (only
if P2 is selected), seeds, caps, the training set (union of the frozen train and val memberships of folds 0, 2, 3, 4,
kinds violent and clean, with its list hash), the calibration scores of each head (`oof_scores.parquet` with sha256),
the event-metric settings and each head's confirmation rule. Interim sha256 (v1 labels):
`f304036b19afdf159362a6344ddf7f071e0eb50f27b0b8539f7660d819c93727`. It is interim; do not commit or use it for the
confirmation, regenerate on the final labels (section 7).

Threshold rule: the VA30 per-fold rule (lowest threshold whose FA/h on clean negatives of the calibration slice stays at
or below the target; 2-of-3 alarm policy). The final model is trained on all screening videos, so there is no held-out
validation slice for it. Calibration therefore uses **out-of-fold scores of the screening-fold models** (same spec) on
the `folds_v2` validation slice of fold 1, which lies inside the screening folds. The final model's scale can differ
slightly from the fold models'; this is a caveat of the one-shot design.

Confirmation rules (`confirm.py`, both applied in one run):
- early head: fold-1 band M1 CI lower > 0.50 and positive point M8 gain over N (CI reported);
- imminent head: fold-1 M1-near CI lower > 0.60 and Success@10 at the 2 FA/h operating point has a threshold-aware CI
  lower bound above the position null at the same test FA/h.

Safety: refuses without `--confirm_once`; refuses unless `--config_sha256` equals the config's hash, the sidecar and
the committed content at git HEAD; refuses if `benchmark/results/confirmation/` exists (the directory is created
before any fold-1 read, so a crash also blocks a rerun); fold-1 access exists only in `ConfirmSet`, built inside the
gated run. Tests use synthetic data only.

## 6. Caveats

- First-frame and scene state: the top blocks are SigLIP 2 scene-state signals (Phase A first-frame flag True for all
  of them). The near-head B-detect of 0.90 and the strong nuisance baselines (N M1-near 0.79) show much of the score is
  "this looks like a violent-type scene", which N only partly absorbs. M4n and M8 gain over N are the honest numbers.
- Label noise: early high scores in the 30 to 120 s window are partly violence already under way before the annotated
  onset (amendment 03, `docs/LABEL_AUDIT.md`); v3 labels will change every number here. HIVAU onsets are noisier than
  official ones: the official/UBI breakdown is co-primary.
- Movies: xd_movie clips score highest (Success@10 0.56 at 2 FA/h) but share films with clean clips and have a
  soundtrack/editing prior; real-world sources (ucf 0.25, ubi 0.30) are the credible figures.
- Selection is post-hoc for the imminent head (amendment 05 was written after Phase A outcomes); only fold 1 is
  untouched. Gate multiplicity is not adjusted for the imminent head (none registered).
- The synthetic tests do not validate real-feature behaviour; the real pipeline reproduces the screen's M1/M8 exactly.

## 7. Re-run on corrected labels (v3 / screen_set_v2)

`--prereg` defaults to `benchmark/screens/screen_set_v2.json` when it exists, else v1. `--decisions` must point to the
new Phase A decisions file. Commands (CPU, 4 threads; the first two can run concurrently):

```bash
PY=.venv/bin/python
nice -n 10 $PY -m anticipation.screen.fusion --head early --out benchmark/results/fusion_v2 --decisions <decisions_phase_A.json> --skip_p2
nice -n 10 $PY -m anticipation.screen.fusion --head imminent --out benchmark/results/fusion_v2 --skip_p2
$PY -m anticipation.screen.fusion --head assemble --out benchmark/results/fusion_v2
```

Drop `--skip_p2` for the P1 versus P2 comparison (about 4 hours per head). `--from_p1` reuses a head's saved
`p1_selection.json`. `--head assemble` writes the two-head `confirmation_config.json` and its sha256; commit it, then
(lead only, once):

```bash
nice -n 10 $PY -m anticipation.screen.confirm --confirm_once --config benchmark/results/fusion_v2/confirmation_config.json \
  --config_sha256 <sha256> --out benchmark/results/confirmation
```

Unit tests: `.venv/bin/python -m pytest tests/test_fusion.py -q` (synthetic world with one useful, one redundant and one
noise block; fold 1 untouchable; both-head confirmation).
