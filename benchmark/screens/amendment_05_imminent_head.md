# Amendment 05 to screen_set_v1: add a short-horizon "imminent" head (onset within 10 s)

Date: 2026-10-15. Registration amended: `benchmark/screens/screen_set_v1.json`
(sha256 `8831e16e664c895271b9cd77e0bab68733ca85a47655708998ab51d5e625797b`).

## Why

The project owner re-centred the goal: in real footage, visible warning signs usually precede violence by about
10 s (our hold-out annotation: median 4.6 s from first visible escalation to onset, 75th percentile 17.1 s), and
30 s or more is a stretch goal. The system therefore gets two heads: an **imminent head** (onset within 10 s,
primary) and the registered **early-warning head** (onset 30-120 s ahead, the band target of section 6).

## Disclosure

Unlike amendments 01-04, this one is written after Phase A outcomes on the screening folds 0, 2, 3, 4 were read,
including the short-horizon numbers (M1-near, Success@10, B-detect) in `benchmark/results/cue_screen/`. Block
selection for the imminent head on the screening folds is therefore not blind. **Fold 1 has not been read and
remains the single confirmation.** Every decision below is fixed before fold 1 is touched.

## Imminent head: gates on the screening folds (per Phase A block, and per Phase C block when available)

Target and probe: P1 exactly as registered, with target y = 1 for first-segment pre-violence ticks with time to
onset in (0, 10], y = 0 for clean ticks with at least 60 s left, VA30 time-match weights; H1 never in training.

- **N1 signal:** M1-near (time-matched AUROC, tte in (0, 10]) point >= 0.70 and CI lower > 0.60.
- **N2 specificity:** M3p-near = UCF first-segment ticks with tte in (0, 10] vs H1p ticks with time to anomaly in
  (0, 10], matched on time since start; CI lower > 0.50.
- **N3 beyond scene prior:** paired gain of [N + block] over N on M1-near, CI lower > 0.
- A block enters the imminent head only if N1, N2 and N3 all pass. Phase A classes (CONTEXT_SHORTCUT etc.) apply to
  the early-warning head only; they do not bar a block from the imminent head.
- Reported, never gating: B-detect (violence-now probe applied to pre-onset ticks), official-onset breakdown,
  amendment-03 sensitivity S1/S2.

## Imminent head: forward selection and final model

- Start from the passing block with the largest paired M8-near (event-level window-max over
  [max(t0 - 10, clear_start), t0) vs time-matched 10 s clean windows); add passing blocks greedily while the paired
  M8-near gain has CI lower > 0. Then P1 vs P2 (causal GRU) by paired M8-near.
- The early-warning head keeps the registered forward selection (M8 over the 30-120 s band).
- Alarm policy per head: thresholds chosen on the screening folds' validation slices at 1, 2 and 5 FA/h, default
  2-of-3 smoothing, as in VA30.

## Confirmation on fold 1 (single run, both heads)

- **Imminent head confirmed** if fold-1 M1-near CI lower > 0.60 and Success@10 at the 2 FA/h validation operating
  point has a threshold-aware CI lower bound above the position null at the same test FA/h.
- **Early-warning head confirmed** by the registered rule (fold-1 band M1 CI lower > 0.50 and positive M8 gain
  over N).
- Headline metric of the project: Success@10 at 1 and 2 FA/h, with Success@30 reported as the stretch result;
  official-onset breakdowns co-primary.
