# SCREEN_RESULTS: Phase A modality screen, results and post-hoc audit

> Superseded: these are interim results on v1 labels. Current results (labels v3, screen_set_v2, fusion_v2, fold-1 confirmation): see docs/RESULTS_V3.md.

Run date 2026-10-15. Pre-registration `benchmark/screens/screen_set_v1.json` (sha256 `8831e16e...`) plus amendments 01 to 03.
Harness and protocol: [SCREEN.md](SCREEN.md); plan: [MODALITY_PLAN.md](MODALITY_PLAN.md) section 6. Raw outputs:
`benchmark/results/cue_screen/` (`results_phase_A.md`, `decisions_phase_A.json`, one folder per block). Everything under
"Audit" and "Sensitivity" is **post-hoc and exploratory**; it never changes a pre-registered class. Fold 1 (confirmation) was
not read by anything described here.

## 1. Summary in plain language

* **What was run.** Nine Phase A feature blocks (pose, motion, V-JEPA 2, SigLIP 2 embedding, two SigLIP 2 text-prompt
  readouts) were scored with the same probe on folds 0, 2, 3, 4 (379 violent videos, 446 eligible events, 3,220 clean
  videos). The main number, M1, is a time-matched AUROC: first-segment pre-violence ticks 30 to 120 s before the onset
  against clean-video ticks, matched on source kind and time since clip start.
* **Pre-registered outcome.** Only the two SigLIP 2 prompt blocks are ANTICIPATION (M1 0.835 and 0.829). SigLIP 2 itself
  (0.862) and V-JEPA 2 (0.801) pass G1 but are CONTEXT_SHORTCUT (they also fire on non-violent anomalies, G2 FAIL). Pose
  interaction and the two motion blocks are PROVISIONAL with weak M1 (0.59 to 0.64). The two pose_state blocks are
  INCONCLUSIVE. A nuisance model that knows only source kind, time since start and the first scene already reaches
  0.784, so the prompts add +0.07 to +0.08 on top of it.
* **What the numbers do not show.** (a) The prompt score is flat in time to onset: bucket AUROCs 0.85 (0-10 s), 0.83
  (10-30 s), 0.82 (30-60 s), 0.84 (60-120 s). Nothing rises toward the onset (ramp test p = 0.15). (b) Broadcasting the
  score of the first frame over the whole clip gives 0.821 against M1 0.829 on the same clips (difference CI -0.03 to
  +0.07). (c) A probe trained to detect violence that is already happening, applied to the pre-onset band, scores 0.840,
  as high as the anticipation probe. The signal behaves like a clip-level "this looks like fight content" detector.
* **Audit verdict (post-hoc, one reviewer, not blinded to scores).** For the prompts the signal is **mixed**: of the 30
  highest-scoring band ticks in positives (one per video), 11 already show violence or its immediate aftermath (annotation
  noise or earlier unlabelled violence), 7 show visible pre-violence conflict (tension, restraint, standoff, crowd
  confrontation), 8 are provenance (title cards, news chyrons, watermarks, phone or CCTV overlays) and 4 are only movie
  look. The highest-scoring clean ticks are film scenes (15 of 30), sports and news broadcasts (7), and clips with fight-like
  content labelled clean (7). For SigLIP 2 it is mostly **provenance**: 15 of 30 positive ticks and 22 of 30 clean ticks are
  overlays, graphics or title cards. Only 3 of the 25 positives with the highest first-frame score show a conflict at the
  clip start.
* **Label-noise sensitivity (amendment 03).** No block is flagged. Masking the audited UCF lead-ins (S1) lowers the
  prompts' M1 by 0.01 to 0.02 and leaves siglip2 and V-JEPA 2 unchanged. The masks do not cover XD movie and YouTube clips,
  where most of the audited "violence already under way" cases sit.
* **Honest reading.** There is a real, significant, source-matched signal that survives the pre-registered controls, but it
  is not evidence that conflict state 30 s ahead has been detected. Most of it is content and provenance of the clip, plus
  onset-label noise. Success@30 of 19% at 1.3 FA/h (prompts v2) beats the position-only null (6.7%), and the same
  threshold fires 4.9 FA/h on movie clips against 0.3 on UCF.
* **Next.** Forward selection (pre-registered rule) is open for the two prompt blocks and the three PROVISIONAL blocks, but
  by the point estimates only the prompt blocks are candidates to carry anything (M8 gain over N +0.10 versus negative
  for pose and motion). See section 9.

## 2. Per-block results

Gates: G1 on M1, G2 on M3p (UCF band ticks vs people-focused H1p), G3 on M4n (gain over nuisance N). CIs are 95%
cluster bootstrap (1,000 resamples, `split_group`). All Holm-adjusted G1 p-values for passing blocks are 0.027.

| Block | Class (tags) | G1 | G2 | G3 | M1 band | M1-long (60-120 s) | M1-near (0-10 s) | M3p | M4n | M4b | M1 minus M4b (ramp clips) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| siglip2_prompts_v1 | ANTICIPATION (LONG_LEAD) | PASS | INCONCLUSIVE | PASS | 0.835 [0.80, 0.87] | 0.844 [0.80, 0.88] | 0.886 [0.86, 0.91] | 0.642 [0.53, 0.74] | 0.066 [0.02, 0.11] | 0.817 [0.76, 0.87] | 0.023 [-0.04, 0.08] |
| siglip2_prompts_v2 | ANTICIPATION (LONG_LEAD) | PASS | PASS | PASS | 0.829 [0.80, 0.86] | 0.842 [0.81, 0.88] | 0.860 [0.83, 0.89] | 0.764 [0.68, 0.84] | 0.081 [0.05, 0.12] | 0.821 [0.77, 0.87] | 0.023 [-0.03, 0.07] |
| siglip2 | CONTEXT_SHORTCUT (LONG_LEAD, RAMP) | PASS | FAIL | PASS | 0.862 [0.83, 0.89] | 0.862 [0.82, 0.90] | 0.912 [0.89, 0.93] | 0.594 [0.48, 0.71] | 0.075 [0.03, 0.12] | 0.843 [0.79, 0.89] | 0.043 [0.00, 0.09] |
| vjepa2_l | CONTEXT_SHORTCUT (LONG_LEAD) | PASS | FAIL | INCONCLUSIVE | 0.801 [0.77, 0.84] | 0.821 [0.78, 0.86] | 0.827 [0.80, 0.86] | 0.551 [0.43, 0.67] | 0.031 [-0.01, 0.08] | 0.764 [0.69, 0.83] | 0.064 [0.01, 0.12] |
| pose_interact | PROVISIONAL (LONG_LEAD) | PASS | PASS | INCONCLUSIVE | 0.639 [0.60, 0.68] | 0.644 [0.59, 0.69] | 0.653 [0.62, 0.69] | 0.631 [0.53, 0.73] | -0.002 [-0.03, 0.02] | 0.570 [0.49, 0.65] | 0.078 [-0.00, 0.16] |
| motion_no_camera | PROVISIONAL (LONG_LEAD) | PASS | PASS | INCONCLUSIVE | 0.588 [0.55, 0.62] | 0.590 [0.54, 0.64] | 0.600 [0.56, 0.63] | 0.624 [0.53, 0.72] | 0.006 [-0.01, 0.02] | 0.507 [0.43, 0.58] | 0.093 [0.01, 0.18] |
| motion_camera_shortcut | PROVISIONAL (RAMP) | PASS | INCONCLUSIVE | INCONCLUSIVE | 0.587 [0.54, 0.63] | 0.570 [0.51, 0.63] | 0.635 [0.59, 0.68] | 0.592 [0.49, 0.69] | 0.013 [-0.00, 0.03] | 0.564 [0.48, 0.64] | 0.019 [-0.08, 0.11] |
| pose_state_identity_free | INCONCLUSIVE | INCONCLUSIVE | FAIL | FAIL | 0.560 [0.51, 0.61] | 0.541 [0.47, 0.61] | 0.576 [0.53, 0.62] | 0.505 [0.41, 0.61] | -0.063 [-0.11, -0.02] | 0.564 [0.49, 0.63] | -0.017 [-0.13, 0.09] |
| pose_state_pair_identity_ucf_ubi | INCONCLUSIVE | INCONCLUSIVE | FAIL | FAIL | 0.520 [0.45, 0.59] | 0.513 [0.43, 0.60] | 0.616 [0.55, 0.68] | 0.458 [0.36, 0.56] | -0.068 [-0.12, -0.02] | 0.537 [0.43, 0.65] | 0.007 [-0.13, 0.15] |

* M4b is the first-frame broadcast on the 58 to 61 ramp clips (onset at least 90 s in); the first-frame flag is true for
  every block except motion_no_camera. The prompts, SigLIP 2 and V-JEPA 2 are within 0.02 to 0.06 of their M1 on those
  clips, so most of their band signal is already present at the clip start.
* M3p for prompts v1 (0.642, CI 0.53 to 0.74) leaves G2 INCONCLUSIVE; v2 passes with ratio 0.66 (CI 0.46 to 0.82).
  Both still score non-violent H1 anomalies and clean clips highly (H1 vs clean 0.73 and 0.75; all-H1 M3 0.68 and 0.66).
* pose_state_pair_identity_ucf_ubi is evaluated on UCF and UBI only, so its nuisance N (0.929) is not comparable to the other
  blocks' N (0.784).

### Event level (M6, VA30 evaluator; thresholds from validation, `--m6`)

Success@30 with 95% video-bootstrap CI at the test FA/h on clean videos (threshold chosen on validation for 1, 2 and
5 FA/h), the position-only null at the same FA/h and the Poisson random reference. All 446 eligible events.

| Block | about 1 FA/h | about 2 FA/h | about 5 FA/h |
|---|---|---|---|
| siglip2_prompts_v1 | 23.5% [19, 28] at 1.2 FA/h; null 6.4%, random 2.9% | 31.2% [27, 36] at 2.3; null 9.0%, random 5.5% | 38.1% [33, 43] at 5.1; null 15.9%, random 12.0% |
| siglip2_prompts_v2 | 19.3% [16, 23] at 1.3 FA/h; null 6.7%, random 3.1% | 27.1% [23, 32] at 2.3; null 9.2%, random 5.7% | 37.0% [33, 41] at 5.0; null 15.8%, random 11.8% |
| siglip2 | 16.8% [13, 20] at 1.2 FA/h; null 6.5%, random 2.9% | 24.7% [21, 29] at 2.0; null 8.4%, random 4.8% | 35.4% [31, 40] at 4.6; null 14.8%, random 10.9% |
| vjepa2_l | 11.2% [9, 14] at 1.3 FA/h; null 6.6%, random 3.1% | 16.4% [13, 20] at 2.0; null 8.4%, random 4.9% | 25.3% [21, 30] at 4.7; null 14.9%, random 11.0% |
| pose_interact | 4.5% [2, 6] at 1.4 FA/h; null 6.9%, random 3.4% | 6.1% [4, 8] at 2.1; null 8.6%, random 5.1% | 13.0% [10, 16] at 4.8; null 15.2%, random 11.3% |
| motion_no_camera | 3.1% [2, 5] at 1.0 FA/h; null 6.1%, random 2.5% | 5.8% [4, 8] at 1.9; null 8.3%, random 4.7% | 10.1% [7, 13] at 5.5; null 16.8%, random 12.8% |
| motion_camera_shortcut | 4.9% [3, 7] at 1.4 FA/h; null 7.0%, random 3.5% | 7.8% [5, 11] at 2.4; null 9.4%, random 5.8% | 12.3% [9, 16] at 5.7; null 17.3%, random 13.2% |
| pose_state_identity_free | 1.8% [1, 3] at 0.8 FA/h; null 5.5%, random 1.9% | 3.1% [2, 5] at 1.9; null 8.2%, random 4.7% | 7.6% [5, 10] at 5.2; null 16.1%, random 12.1% |
| pose_state_pair_identity_ucf_ubi | 6.0% [3, 9] at 1.0 FA/h; null 5.4%, random 2.6% | 8.0% [4, 12] at 2.2; null 7.0%, random 5.3% | 9.0% [5, 13] at 5.3; null 11.3%, random 12.4% |

Only the four SigLIP 2 / V-JEPA 2 blocks clear the position null (CI lower bound above the null) at every operating point;
pose and motion are at or below it, even though they are PROVISIONAL on M1. Median alarm lead for prompts v2 at 1 FA/h is
33.7 s (CI 31 to 40); the 5.0 FA/h row is not a usable alarm rate. The position null is the VA30 construct (oracle
position-only policies); the Poisson reference is the cruder one. The CI above is a fixed-threshold CI; the
threshold-aware CI for prompts v2 at 1 FA/h is 15 to 25%.

## 3. Controls

All on the same ticks and weights as M1 (95% CI in brackets). N, time/source and shuffled are the pre-registered controls;
B-detect is a diagnostic (probe trained on in-violence versus clean, applied to the pre-onset band).

| Block | N (source x time + first scene) | time/source fit | shuffled clean video | N + block | B-detect | M8 (all events) / M8 gain over N | sparse pAUC | tail hit rate vs clean rate |
|---|---|---|---|---|---|---|---|---|
| siglip2_prompts_v1 | 0.784 [0.74, 0.82] | 0.472 [0.43, 0.52] | 0.500 [0.45, 0.55] | 0.850 [0.82, 0.88] | 0.871 [0.84, 0.90] | 0.881 / +0.109 [0.07, 0.14] | 0.561 | 0.33 vs 0.03 |
| siglip2_prompts_v2 | 0.784 | 0.472 | 0.510 [0.46, 0.56] | 0.865 [0.84, 0.89] | 0.840 [0.81, 0.87] | 0.872 / +0.101 [0.06, 0.13] | 0.545 | 0.33 vs 0.03 |
| siglip2 | 0.784 | 0.472 | 0.517 [0.46, 0.57] | 0.860 [0.83, 0.89] | 0.895 [0.87, 0.92] | 0.894 / +0.123 [0.09, 0.15] | 0.570 | 0.30 vs 0.02 |
| vjepa2_l | 0.784 | 0.472 | 0.493 [0.45, 0.54] | 0.815 [0.78, 0.85] | 0.819 [0.78, 0.86] | 0.805 / +0.034 [-0.00, 0.07] | 0.537 | 0.19 vs 0.03 |
| pose_interact | 0.784 | 0.472 | 0.503 [0.46, 0.55] | 0.783 [0.75, 0.82] | 0.670 [0.63, 0.70] | 0.683 / -0.088 [-0.14, -0.04] | 0.505 | 0.07 vs 0.02 |
| motion_no_camera | 0.784 | 0.472 | 0.517 [0.47, 0.56] | 0.790 [0.75, 0.83] | 0.590 [0.55, 0.62] | 0.613 / -0.159 [-0.22, -0.10] | 0.502 | 0.07 vs 0.02 |
| motion_camera_shortcut | 0.784 | 0.472 | 0.489 [0.45, 0.53] | 0.797 [0.76, 0.83] | 0.611 [0.58, 0.65] | 0.639 / -0.132 [-0.19, -0.08] | 0.506 | 0.08 vs 0.03 |
| pose_state_identity_free | 0.784 | 0.472 | 0.505 [0.46, 0.55] | 0.722 [0.68, 0.76] | 0.600 [0.56, 0.64] | 0.606 / -0.165 [-0.22, -0.12] | 0.506 | 0.02 vs 0.01 |
| pose_state_pair_identity_ucf_ubi | 0.929 [0.90, 0.95] | 0.461 [0.40, 0.53] | 0.483 [0.42, 0.55] | 0.861 [0.81, 0.91] | 0.635 [0.56, 0.71] | 0.588 / -0.316 [-0.37, -0.26] | 0.502 | 0.03 vs 0.02 |

* The time/source control (0.47) and the shuffled-video control (0.49 to 0.52) sit at chance: position and provenance by
  source kind alone do not explain M1, and the probe does not score a random clean video's features as a pre-onset signal.
* N is the strongest control: 0.784 from source kind, time since start and the first scene (people greyed out). It is
  the reason the first-frame reading matters; a clip's look at second 2 is worth 0.78 on its own.
* For pose and motion, N + block is not above N (0.78 to 0.80), and the event-level M8 is below N. They are PROVISIONAL by
  the frozen table (G3 INCONCLUSIVE, not FAIL) but carry no evidence of adding to N.
* B-detect applied to the pre-onset band gets 0.84 to 0.90 for the SigLIP 2 family, the same as or above the anticipation
  probe. For pose and motion B-detect is similar to or below M1 (they carry a weak signal of any kind).

## 4. Per-source and breakdown results for the ANTICIPATION blocks

Band AUROC (time-matched within each source kind). S_real excludes xd_movie. Event columns are VA30 at about 2 FA/h.

| Block | S_real | S_offubi | S_audio | xd_movie | xd_youtube | ucf | ubi | dyadic | instrumental | riot |
|---|---|---|---|---|---|---|---|---|---|---|
| siglip2_prompts_v1 | 0.848 | 0.815 | 0.848 | 0.811 | 0.845 | 0.874 | 0.785 | 0.819 | 0.849 | 0.826 |
| siglip2_prompts_v2 | 0.844 | 0.811 | 0.836 | 0.798 | 0.808 | 0.902 | 0.807 | 0.813 | 0.856 | 0.768 |
| siglip2 (context shortcut) | 0.885 | 0.885 | 0.854 | 0.801 | 0.838 | 0.899 | 0.966 | 0.869 | 0.840 | 0.857 |
| vjepa2_l (context shortcut) | 0.825 | 0.824 | 0.762 | 0.728 | 0.703 | 0.897 | 0.880 | 0.811 | 0.847 | 0.594 |

| Block (VA30, 2 FA/h target) | Success@30 xd_movie (FA/h) | xd_youtube | ucf | ubi |
|---|---|---|---|---|
| siglip2_prompts_v1 | 46.6% (4.8) | 31.5% (2.2) | 14.8% (0.13) | 31.4% (0.83) |
| siglip2_prompts_v2 | 41.5% (4.9) | 31.5% (2.1) | 9.6% (0.34) | 24.4% (1.24) |

* The signal is present in every source kind and in the real-world subset, so it is not only a movie effect; the AUROC is
  within-source because the matching is by source kind. The shared threshold, though, is very unequal: movies fire 4.9 false
  alarms per hour and UCF 0.3, so Success@30 is highest where alarms are cheapest.
* Events with HIVAU onsets (304 of 446) are unvalidated. Official and UBI onsets give the same picture for prompts v2
  (far-window AUROC 0.83 and 0.88).
* Not implemented: AUROC matched within the same film, news and compilation exclusion, visible-conflict versus sudden
  (needs annotation). Any film-versus-same-film shortcut therefore cannot be excluded, and 133 of 135 eligible movie videos
  share a film with clean clips.

## 5. Post-hoc audit (exploratory): what drives the scores

Triggered by the 0.85 rule for siglip2 (0.862) and by M4b for the prompts. Scores are the saved out-of-fold P1 scores
(`scores.parquet`, folds 0, 2, 3, 4 only). Contact sheets use `tile_w` 192 and frames at t-6, t-3, t (the tick and its
causal past). One reviewer (Claude) classified each tile; the classes are subjective and not blinded to the score. Files
in `benchmark/results/cue_screen/audit/`: the sheets (`*_1..3.jpg`), per-tick classes (`top_*.csv`, column
`reviewer_class`), `first_frame_top25_v2.csv`, `quant_results.json`, `recompute_check.json`, and `scripts/`.

Selection: the highest-scoring tick per video, top 30 videos, in clean negatives (ticks with at least 60 s left) and in
first-segment positives with time to onset in (30, 120]. One tick per video avoids 30 neighbouring ticks of one clip.

### 5.1 Contact sheet classification

| Set (30 ticks each) | Violence already under way | Visible pre-violence conflict | Provenance (title card, watermark, chyron, broadcast graphics, CCTV or phone overlay) | Movie look only | Fight-like content in a clean-labelled clip | Other |
|---|---|---|---|---|---|---|
| prompts v2, positives band | 11 | 7 | 8 | 4 | - | - |
| prompts v2, clean | - | - | 7 | 15 | 7 | 1 |
| siglip2, positives band | 1 | 3 | 15 | 11 | - | - |
| siglip2, clean | - | - | 22 | 8 | - | - |

Examples (prompts v2, positives band): a sports title card (t=1, 39 s before onset; provenance);
a movie swordfight, a movie shootout, a custodial-setting struggle (CCTV clip)
and a riot with smoke and shields, all with violence visible 32 to 72 s before the annotated onset
(under way: label noise or earlier unlabelled violence); news-graphics cases (broadcast headline graphics) count as provenance; a movie scene in which a man is restrained, a movie
armed standoff and a street group around parked cars (conflict). Clean top ticks: dramatic movie
scenes and a boxing bout labelled normal, a sports scuffle, a movie clip with a rifle aimed at a wounded person
(fight-like content in a "clean" clip), news broadcasts, a sports broadcast with scoreboard.
Examples (siglip2): clean top ticks are almost all title cards (show and episode title cards),
news and sports-interview graphics, sports-broadcast idents and a UCF "Normal" camera info screen; positive top ticks are
outlet watermarks, CCTV timestamp overlays, pillarboxed phone video and title
cards (a caption naming people and a venue).

Findings:

* **siglip2 is mostly provenance.** 37 of 60 audited ticks are overlays, graphics or title cards, and in clean videos 22 of
  30 are. Its class (CONTEXT_SHORTCUT) is consistent with this. Its 0.862 comes with no sign of future-frame leakage (the
  features are causal by extractor check, shuffled control 0.52, time/source 0.47), so the audit finding is a shortcut and
  not a leak.
* **The prompts are mixed.** The best single prompt columns in the P0 table are `grab` 0.73, `phase_violence` 0.73,
  `phase_contact` 0.73, `held_back` 0.72 and `fight_stance` 0.71; `phase_tension` is 0.60 and `phase_argument` 0.64. The
  score tracks physical-contact and violence content, not a calm-to-tension build-up. 11 of 30 top positive ticks already
  show violence, which is the label-noise failure documented in `docs/LABEL_AUDIT.md`.
* **False alarms in clean videos are content-driven.** Fight-like or action content in clean-labelled clips (7 of 30,
  mainly sports scuffles and film combat) and film look (15 of 30) dominate. The clean negatives are not all peaceful:
  some "negative" film clips contain combat or weapons.

### 5.2 Context concept columns (siglip2_prompts_v2)

Correlation of the prompts v2 score with each context column, over the 280,473 clean ticks used as negatives (Spearman
rho): ctx_cctv -0.12, ctx_movie +0.06, ctx_news +0.15, ctx_captions -0.19, ctx_phone -0.14. Within sources the range is
-0.35 to +0.24 (largest: xd_youtube with phone -0.35 and cctv -0.31). The context columns themselves are near chance as
band predictors: AUROC ctx_phone 0.54, ctx_news 0.54, ctx_movie 0.53, ctx_cctv 0.47, ctx_captions 0.39 (so captions are
higher in clean ticks). For comparison the strongest violence columns reach 0.73.

Band AUROC after keeping only ticks whose five context columns are all at or below their clean 75th percentile (both
positives and negatives; time-matching recomputed on the kept ticks):

| Subset | prompts v2 | siglip2 | positive ticks |
|---|---|---|---|
| all ticks (primary M1) | 0.829 | 0.862 | 8,499 |
| all five context columns low (<= clean p75) | 0.820 | 0.875 | 1,618 |
| ctx_movie low | 0.854 | 0.880 | 5,734 |
| ctx_news low | 0.820 | 0.872 | 5,880 |
| ctx_captions low | 0.827 | 0.858 | 7,109 |
| ctx_cctv low | 0.820 | 0.846 | 6,483 |
| ctx_phone low | 0.812 | 0.832 | 6,052 |
| phase_violence at or below clean p90 | 0.788 | 0.821 | 5,041 |
| phase_violence at or below clean p50 | 0.785 | 0.803 | 1,592 |
| xd_movie / xd_youtube / ucf / ubi | 0.798 / 0.808 / 0.902 / 0.807 | 0.801 / 0.838 / 0.899 / 0.966 | 2,265 / 2,357 / 2,815 / 1,062 |
| S_real (non-movie) | 0.844 | 0.885 | 6,234 |
| ucf and ubi only | 0.866 | 0.917 | 3,877 |

Reading: the prompts v2 signal does not depend on the five context concepts (it stays 0.81 to 0.85), and it falls only to
0.79 when ticks that look violent now are removed, so it is not only "violence is already visible". These context columns
are a coarse view of provenance (five text concepts); the contact sheets show watermarks and title cards the columns do not
name. The siglip2 embedding is higher in the CCTV subsets (UBI 0.966), where camera or site identity could separate
positive from normal videos; the audit cannot distinguish that from real content.

### 5.3 The first-frame question

For the 61 ramp clips (onset at least 90 s in), the tick-2 score of prompts v2 is above the clean 90th percentile at tick 2
(1.26) for 25 clips (43 above p75, 55 above p50). Classification of the first frame (t=2, plus t=1 and t=6) of those 25:

| First frame shows | Clips |
|---|---|
| a conflict or violence already visible (a person restrained by officers, a riot, an armed group) | 3 |
| a group of people in a street or lobby (possible confrontation, not clearly one) | 6 |
| provenance: title card or news graphic (3), outlet watermark (2) | 5 |
| a CCTV or camera look with nothing visible (empty room, a single person, a shop) | 7 |
| movie look without conflict | 4 |

Reading: a conflict-state interpretation of the first-frame signal is supported by at most 3 clear and 6 ambiguous clips
of 25; 16 are provenance, camera or film look. The first-frame result is mainly about what kind of clip this is, which
fits the N control (0.784) and the flat time profile.

### 5.4 Independent recomputation of M1

M1 (band, time-matched) recomputed from `scores.parquet` with own code (labels and tick masks from the frozen set, fold 1
excluded by the harness, weights from `anticipation.eval.va30.time_match_weights` with source kind and
`TIME_MATCH_EDGES`, a separate weighted rank AUROC):

| Quantity | Recomputed | metrics.json | n positive ticks |
|---|---|---|---|
| prompts v2 M1 | 0.829385 | 0.829385 | 8,499 |
| prompts v2 M1 (30-60 s) | 0.819114 | 0.819114 | 4,650 |
| prompts v2 N, time/source, shuffled, N + block | 0.7845, 0.4721, 0.5104, 0.8654 | same | |
| pose_interact M1 | 0.638820 | 0.638820 | 8,499 |
| pose_interact M1 (30-60 s) | 0.633028 | 0.633028 | 4,650 |
| pose_interact N + block | 0.7825 | same | |

All agree to better than 1e-12 (see `audit/recompute_check.json`). Bootstrap CIs were not recomputed.

## 6. Sensitivity to onset label noise (amendment 03; secondary, never used for a class)

Same probes, folds, seeds and bootstrap as the primary run, with the soft masks of `benchmark/audits/soft_masks_v1.json`
removing masked ticks from positives, negatives and training. S1 drops the weapon-only entries (47 entries, 42 videos),
S2 keeps all 66 (54 videos). S1 removes 1,298 of 15,588 band ticks (14,290 left); S2 removes 1,562 (14,026 left). Events that
lose their whole 30 to 120 s window: 8 in S1 (5 first-segment) and 11 in S2 (7 first-segment); partly masked: 18 and 24; all
the whole-window losses are UCF events. Amendment 03 rule: a primary G1 PASS turning into a G1 FAIL under S1 is "signal
explained by onset label noise".

**Result: no block is flagged** (`signal_explained_by_onset_label_noise` is false for all nine; the primary class is
unchanged in every case).

| Block | Primary M1 | S1 M1 [CI] | S1 diff | S2 M1 [CI] | S2 diff | S1 M4n | S2 M4n | S1 M3p | S2 M3p | G1 primary / S1 / S2 |
|---|---|---|---|---|---|---|---|---|---|---|
| siglip2_prompts_v1 | 0.835 | 0.811 [0.77, 0.85] | -0.024 | 0.814 [0.78, 0.85] | -0.021 | 0.064 [0.02, 0.11] | 0.071 [0.02, 0.12] | 0.584 [0.47, 0.69] | 0.595 [0.47, 0.71] | PASS / PASS / PASS |
| siglip2_prompts_v2 | 0.829 | 0.818 [0.78, 0.85] | -0.012 | 0.811 [0.78, 0.84] | -0.018 | 0.096 [0.06, 0.13] | 0.099 [0.06, 0.13] | 0.737 [0.65, 0.81] | 0.714 [0.62, 0.80] | PASS / PASS / PASS |
| siglip2 | 0.862 | 0.858 [0.82, 0.89] | -0.005 | 0.855 [0.82, 0.89] | -0.008 | 0.097 [0.05, 0.14] | 0.099 [0.05, 0.15] | 0.583 [0.46, 0.70] | 0.587 [0.46, 0.71] | PASS / PASS / PASS |
| vjepa2_l | 0.801 | 0.802 [0.77, 0.84] | +0.000 | 0.793 [0.76, 0.83] | -0.008 | 0.046 [-0.01, 0.10] | 0.037 [-0.02, 0.09] | 0.526 [0.41, 0.64] | 0.503 [0.39, 0.62] | PASS / PASS / PASS |
| pose_interact | 0.639 | 0.621 [0.57, 0.66] | -0.018 | 0.628 [0.58, 0.67] | -0.011 | -0.004 [-0.03, 0.02] | 0.003 [-0.03, 0.03] | 0.596 [0.48, 0.71] | 0.624 [0.51, 0.73] | PASS / PASS / PASS |
| motion_no_camera | 0.588 | 0.579 [0.54, 0.62] | -0.009 | 0.575 [0.53, 0.61] | -0.013 | 0.005 [-0.01, 0.02] | 0.005 [-0.01, 0.03] | 0.591 [0.49, 0.68] | 0.609 [0.51, 0.70] | PASS / PASS / INCONCLUSIVE |
| motion_camera_shortcut | 0.587 | 0.588 [0.54, 0.63] | +0.001 | 0.585 [0.54, 0.63] | -0.002 | 0.015 [-0.00, 0.03] | 0.012 [-0.01, 0.03] | 0.624 [0.52, 0.72] | 0.636 [0.53, 0.72] | PASS / PASS / PASS |
| pose_state_identity_free | 0.560 | 0.565 [0.51, 0.62] | +0.005 | 0.561 [0.50, 0.61] | +0.001 | -0.043 [-0.09, -0.00] | -0.061 [-0.12, -0.01] | 0.531 [0.42, 0.64] | 0.515 [0.39, 0.64] | INCONCLUSIVE x3 |
| pose_state_pair_identity_ucf_ubi | 0.520 | 0.539 [0.47, 0.61] | +0.019 | 0.508 [0.43, 0.58] | -0.012 | -0.031 [-0.07, -0.00] | -0.059 [-0.10, -0.02] | 0.515 [0.40, 0.63] | 0.458 [0.34, 0.58] | INCONCLUSIVE / INCONCLUSIVE / FAIL |

* The differences are point differences (the primary replicates are not stored) and small: at most -0.024. The prompts
  lose 0.012 to 0.024, which is within the CI half-width (0.03 to 0.04); siglip2 and V-JEPA 2 are unchanged. M1-long moves
  by under 0.02 for every block that passes G1.
* S2 changes two G1 labels, neither of them a PASS to FAIL: `motion_no_camera` PASS to INCONCLUSIVE (point estimate 0.575, now
  below the 0.58 threshold) and `pose_state_pair_identity_ucf_ubi`
  INCONCLUSIVE to FAIL. They show that the two weakest blocks are not robust to a few percent of masked ticks.
* G2 for prompts v1 stays INCONCLUSIVE (M3p 0.58 to 0.60, CI lower 0.47) and v2 holds (M3p 0.71 to 0.74, CI lower above 0.6).
* Limit of this check: the masks come from UCA captions and the label audit, so they only touch UCF lead-ins (all events
  that lose their whole window are UCF). Onset noise in the XD clips (the movie and YouTube cases where violence is visible
  before the annotated onset, which dominate the audit in 5.1) is not masked here. S1/S2 therefore say the signal is not an
  artefact of the UCF caption-visible noise; they do not say it is robust to onset noise elsewhere.

## 7. Caveats

* **HIVAU onsets.** 304 of the 446 events have HIVAU onsets, which were not human-validated. Violence already visible
  30 to 120 s before the annotated onset (11 of 30 top ticks in the audit; 13 of 27 in the earlier label audit, mostly
  HIVAU) inflates every semantic block. Official and UBI onsets give prompt AUROCs of 0.83 and 0.88, no lower.
* **Movie clips.** 87 of the positive videos are movie clips whose clean counterparts come from the same films. The
  same-film AUROC is not implemented. Prompts and siglip2 also fire strongly on film look in clean negatives.
* **Label noise in negatives.** Some clean-labelled clips contain fights, combat sports or weapons, which lowers AUROCs; it
  does not create the signal.
* **Pose density confound.** `pose_interact` best P0 columns are `wrist_speed_max` 0.63, `torso_lean_mean`, `body_speed_max`,
  `n_persons_max` 0.61: activity and crowd size. Amendment 01 documents that the interaction columns track crowd density. No
  person-count-stratified M2 was run, and pose has no evidence of adding to N (M4n -0.002, M8 gain -0.088).
* **First-frame signal.** Not gating (flag only), but it dominates: M4b is 0.82 for prompts and 0.84 for siglip2.
* **Flat time profile.** For the prompts the bucket AUROCs do not rise toward the onset; for siglip2 they fall slightly
  from 0.89 (0-10 s) to 0.86 (60-120 s). The ramp test is significant only for siglip2 (p = 0.02, 62% rising) and is not
  significant for the prompts (p = 0.15 and 0.42).
* **Screening only.** Four folds, seed 0, one probe family (PLS32 plus L2 logistic for wide blocks), post-selection for these
  folds. The audit used a single reviewer. P2 (GRU), the full-dimension L2 check, forward selection and the confirmation
  rule are not implemented. The tail-hit threshold deviation and the Holm family choice are documented in `SCREEN.md`.
* **Not a deployment result.** The thresholds fire 1.3 false alarms per hour on clean video at 19% Success@30; movies fire
  4.9 FA/h at the same threshold.

## 8. What the audit changes and does not change

No pre-registered class changes. For writing up: describe the prompts as "significantly above nuisance N and the position
null, but not distinguishable from a first-frame and violence-content detector; 11 of 30 top positive ticks show violence
already under way". Describe siglip2 as dominated by provenance. Do not claim 30 s anticipation of conflict from these
numbers.

## 9. Forward selection (pre-registered rule), what goes in next

Rule: start from the best ANTICIPATION block, add ANTICIPATION or PROVISIONAL blocks greedily while the paired M8 gain has
CI lower > 0, then evaluate the selected set once on fold 1. It is not implemented in the harness (it needs the fused
trainer).

* Candidates: `siglip2_prompts_v1`, `siglip2_prompts_v2` (ANTICIPATION); `pose_interact`, `motion_no_camera`,
  `motion_camera_shortcut` (PROVISIONAL). Excluded: `siglip2`, `vjepa2_l` (CONTEXT_SHORTCUT), the two `pose_state` blocks
  (INCONCLUSIVE; kept for the long-form data).
* Starting block: v1 has the higher M8 (0.881 vs 0.872) and Success@30 (23.5% vs 19.3% at 1 FA/h) but G2 INCONCLUSIVE;
  v2 has G2 PASS and a smaller specificity problem. The gap is far inside the CIs. The tie rule should be fixed before the
  fused run (it is not in the frozen file).
* Expected effect of PROVISIONAL blocks: their M4n, M8 gain over N and Success@30 relative to the position null give no sign
  of a positive contribution; they pass because the M1 gate is low (0.58). They should still be tested by the rule (a
  complementary signal may exist) but expectations should be low.
* Suggested additions before the fold-1 evaluation (not changing the rule): report M8 gain over a first-frame-only model
  next to the gain over N, and report results with the S1 label mask.

## 10. Files

| Path | Content |
|---|---|
| `benchmark/results/cue_screen/results_phase_A.md`, `decisions_phase_A.json` | pre-registered classes and gates |
| `benchmark/results/cue_screen/<block>/{metrics.json,scores.parquet,va30/}` | primary outputs |
| `benchmark/results/cue_screen/sensitivity_S1`, `sensitivity_S2` | amendment 03 runs |
| `benchmark/results/cue_screen/audit/` | contact sheets, per-tick classes, context correlations, recomputation check, scripts |
