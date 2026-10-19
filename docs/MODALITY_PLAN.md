# Modality plan: which cues and streams to invest in for 30 s+ violence anticipation

Audience: the project owner and future agents deciding which feature streams, labels, data and screens to
spend time and GPU on. Written 2026-10-14 by the final editor of the modality review, after one synthesis
draft and three independent critiques (source checking, feasibility on our data, missed opportunities).
This file is the decision document. Where it disagrees with the four research reports it is built on
([research/escalation_precursors.md](research/escalation_precursors.md),
[research/modality_evidence.md](research/modality_evidence.md),
[research/cue_extractors.md](research/cue_extractors.md),
[research/pre_onset_evidence_our_data.md](research/pre_onset_evidence_our_data.md)), this file wins. The
reasons are in section 11.

**Number conventions.** Benchmark numbers are VA30 version 2 ([VA30.md](VA30.md)):
- 551 eligible30 events in 468 videos, 312 of them the first violent segment of their clip.
- Zero-shot teacher Success@30 is 11.6 % at 0.54 false alarms per hour (FA/h). The position null at that rate
  is 6.0 %.
- The position-plus-source control (`b_time_source_fit`) reaches far-window AUROC 0.86.

The figures "489 events", "11.1 % at 0.66 FA/h vs 1.6 % random" and "position-only 0.76" are from VA30 v1 and
are superseded. Every count in section 2 was recomputed from `benchmark/events.json`, `benchmark/videos.csv` and
`benchmark/folds_v2.json` on 2026-10-14; section 12 has the code. Figures marked "(reviewer)" were measured by a
critique agent with scratch scripts and have not been re-run here.

---

## 0. Executive summary

1. **What can be seen 30 s or more before violence, when anything can, is a conflict state, not a ramp.** That
   means arguing, posturing, low-level contact, and context specific to the kind of violence. In our data it shows
   up in a minority of events.
   - **The one measured far-window signal that is specific to violence** is the no-context Qwen2.5-VL aggression
     rating. Its time-matched AUROC is 0.67 to 0.71 at 30 to 120 s before onset, and 0.52 before non-violent
     anomalies.
   - **That signal has three limits.**
     - **Partly a scene prior.** The first frame's rating alone (broadcast over the clip) reaches 0.56 to 0.63.
     - **Concentrated in a subset.** Most of it sits in the 41 % of events that have an early violent-looking
       frame.
     - **Not yet tested at the event level.**
   - **Nothing else is measured beyond 30 s on real violence footage.** The context-carrying teacher's
     Success@30 above the null comes mostly from movie clips: 8.0 % outside movies, against a pooled null of
     6.0 %.
2. **Ranking by expected contribution to 30 s anticipation on the target footage:**
   - **Measured, with caveats:**
     1. Vision conflict-state semantics from a video-LLM, read by a trained probe on hidden states and
        log-probabilities.
   - **Hypotheses (weak or indirect evidence):**
     2. Pose and track contact and engagement aggregates.
     3. SigLIP 2 contact and aggression prompts.
     4. Vocal prosody and conversational dynamics.
     5. Speech dialogue acts.
     6. Event history.
     7. Context cues specific to the kind of violence.
     8. Camera-wearer cues on bodycam, dashcam and phone footage.
   - **Short-horizon only:** V-JEPA 2, optical flow and sound events. They belong in the "violence now" and 10 s
     heads.

   The screen in section 6 decides which hypotheses survive.
3. **The largest potential lead on bodycam, phone and road-rage footage is in the verbal channel, but nobody has
   measured it on violence.**
   - **The published support** covers non-violent outcomes (arrests, online personal attacks), acted scenes, or
     the current conflict state.
   - **Our own loudness check** shows a rise only inside the last 30 s, on 21 movie clips.
   - **VA30 cannot settle it.** The screening folds hold 86 first-segment events with real-world audio, and many
     of those are narrated news edits.
   - **Three things can settle it:**
     - an audio-type audit;
     - a human forecasting study with sound off and then on (section 4.5);
     - long-form bodycam, dashcam and livestream footage with audio (section 4.3).
4. **The biggest levers are measurement, labels, data and the training objective, not another extractor.**
   - **Measurement.** VA30 cannot credit an alarm that starts more than 120 s before onset; add a sustained-warning
     metric.
   - **Labels.**
     - Some HIVAU onsets mark the walk-up, and some come after violence has already started.
     - The strongest early "contact" signal may be unlabelled violence; audit it.
     - Precursor timelines and hindsight state labels turn the conflict state into a supervised target.
   - **Data.** Only 54 first-segment events have 120 s or more of lead-in. Long-form, domain-matched footage is the
     only way to test minute-scale leads and the audio hypothesis.
   - **Objective.** Use an event-level band objective (onset 30 to 120 s ahead) and longer hazard bins, and stop
     down-weighting far-from-onset positives.
5. **Drop:**
   - face, facial expression and gaze;
   - NTU-trained skeleton action classifiers;
   - world-model surprise features and V-JEPA 2.1;
   - open-vocabulary weapon detectors on CCTV;
   - context-carrying VLM verdicts as features;
   - camera-operator behaviour in pooled models;
   - generic toxicity as the main speech target.

   Voice emotion is demoted to a low-prior candidate, not used as a negative control.
6. **The screen is pre-registered.**
   - **Folds.** It runs on VA30 folds 0, 2, 3 and 4. Fold 1 is frozen for one confirmation run.
   - **Three gates,** each with a PASS, INCONCLUSIVE or FAIL outcome:
     - time-matched AUROC for onset 30 to 120 s ahead;
     - specificity against people-focused non-violent anomalies;
     - gain over a nuisance model of source, time and scene.
   - **Sparse cues** have their own tail-based route through the first gate.
   - **No block is dropped for lack of power.** An underpowered result is called INCONCLUSIVE.
7. **Realistic outcome:**
   - **Success@30** of about 10 to 20 % at 1 FA/h on VA30. It must beat the position null and the late-event null
     at the same FA/h.
   - **Time-matched AUROC** of 0.65 to 0.75.
   - **Minute-scale leads** only on the minority of events with a visible build-up, and only measurable with the
     sustained-warning metric.
   - **Deployment.** False alarms on real confrontations (bars, police stops, traffic disputes) will be several
     times VA30's.
8. **First actions need no new GPU job** (section 6.1). The VLM pilot waits until they are done.
   - Score the existing no-context aggression rating through VA30.
   - Audit the 53 "early contact" events.
   - Mask the UCF lead-ins that already contain violence.
   - Extend the evaluator (`cap_sec`, the band AUROC, the sustained-warning metric).
   - Pre-register the screen.
   - Rescore SigLIP prompts.

---

## 1. The corrected evidence base

The earlier drafts overstated several sources. This table records what each source actually supports. Sources
were opened by the reviewers unless marked; extracted texts were in the session scratchpad
(`<tmp>/.../scratchpad/pdfs/`, temporary).

| Source as used before | What it actually shows | Consequence for this plan |
|---|---|---|
| Philpot 2020: public conflicts last a mean of 3.27 min (n = 219) | Recording runs from the moment the CCTV operator "noticed a potential for violence" until the incident dissipated or police separated the parties. Clips range "from the mildest animated disagreements to grave physical violence". SD 4.20 is larger than the mean | Conflicts are visible for minutes, but the duration includes the violence and its aftermath, and many conflicts never turned violent. **Not a lead time.** |
| Lindegaard 2022: first bystander intervention about 92 s in (n = 80) | The paper says "variation is large". Medium aggression (punches, kicks, shoves) came before the first intervention in 45 of 80 conflicts | Violence often comes **before** the 92 s mark. Removed from the lead-time evidence. |
| Rho 2023 (PNAS): the officer's first 45 words (about 27 s) predict escalation, 70.8 % balanced accuracy (577 stops) | Numbers correct. But the transcripts are professional (not ASR), only officer speech is used, the outcome is search, handcuffing or arrest, and force occurred in only 2 of 588 stops. "Outcomes minutes later" is not in the paper | Early speech predicts **non-violent** escalation outcomes. It is zero direct evidence about violence. |
| DeToxy: toxicity falls to 0.26 F1 "without trigger words" | Test-T is **non-toxic** utterances that each contain a trigger term. It measures false positives on benign profanity. The gold-to-ASR drop (0.80 to 0.67-0.68 F1) is correct. The "bodycam WER 32 %" comes from one 484-word sample | Toxicity scores will fire on ordinary street swearing. This is worse for us, and confirms that toxicity is the wrong main target. |
| Kim 2012: next 30 s conflict level 41.5 % vs 33.3 % chance | Political TV debates. Manual segmentation; 38.7 % with automatic diarization. The next-clip gain is mostly persistence of the current state | Not forecasting evidence. |
| Lefter 2022: F0 and intensity rise with escalation | Acted corpora; it classifies the current escalation level and forecasts nothing | Supports prosody as a **current-state** cue only. |
| Kooij 2016: audio caught 7 of 13 aggression events vs 4 of 13 for video motion | Professional actors on a train platform; the task is detection "shortly after they have started"; the target includes verbal aggression | Detection evidence, not lead evidence. |
| Repo loudness check: "the only cue already up 30 s out" | The +2.3 dB is the mean over [t0-30, t0-10) against a [t0-60, t0-30) baseline. The "81 % of clips up" refers to [t0-10, t0). F0 was flat (-10 Hz, then +1.6 Hz). The 21 clips are movie and compilation clips with music. FEATURES.md itself says this is not evidence of 30 s anticipation | Loudness rises **inside** the last 30 s: a lead of about 10 to 30 s at best. The F0 rationale is not supported. |
| Ganesh 2026: box geometry probe AUC 0.72, beats pose | Its "pre-onset separation" is a video-level anomalous-vs-normal AUC on truncated videos (a collection prior; the paper links it to title cards and watermarks). Full-frame context keeps more of the separation (up to 91 %) than box geometry (79 %). The G7 vs P11 difference has a CI that includes 0 | No evidence that geometry anticipates. "Prefer box-level over joint-angle features" stays only as a robustness choice. |
| ASK-HINT (74.5 to 89.8 AUC with structured questions), HiProbe-VAD (hidden-state probes), rank compression (probability weighting) | All detection. ASK-HINT is an arXiv preprint with no code and no confirmed venue | They support **how** to read a VLM (probes, log-probabilities, structured items), not that it anticipates. |
| (omitted before) RiskCueBench, arXiv 2601.03369v2 | Protest-risk F1: Qwen3-VL-8B-Instruct 0.48 vs random 0.44 (Thinking variant 0.57, humans 0.97). Shuffling or reversing frames changes F1 by only 1 to 3 points | Zero-shot **forecasting** with the chosen model is near chance, and a multi-frame input buys little zero-shot. The primary VLM readout must be a trained probe, and a single-frame arm is needed. |
| STCH (CVPR 2026): window AUC 0.69 at 60 s | Page returned HTTP 403; no code. Per [research/literature.md](research/literature.md), its average prediction horizon is about 11 s and its AUC is nearly flat from 2 s to 60 s (0.718 to 0.692). It was trained on UCF-Crime, where a first frame alone gives about 0.72, and reports no time-matched or hard-negative control | Not usable as evidence for pre-onset cues. Kept only as a published reference number. |
| Repo teacher Success@30 11.6 % "works on silent CCTV" | 9 of its 23 hits come from 24 `xd_movie` events. Outside movies: 14 of 174 (8.0 %), against a pooled null of 6.0 %. UCF: 6.9 % (144 events) | Event-level anticipation on real CCTV is **not shown**. Real-world footage (ucf, ubi, xd_youtube) becomes the primary readout, movies secondary. |
| No-context aggression "passes both controls" | Actual vs first-frame broadcast, time-matched: 60-120 s 0.714 vs 0.627 [0.52, 0.74]; 30-60 s 0.670 vs 0.556 [0.44, 0.68]; 10-30 s 0.676 vs 0.629. No paired CI. The broadcast accounts for about 33 to 73 % of the above-chance separation. Without the 41 % of events that have an early violent-looking frame, 30-60 s falls to 0.56 | It passes the non-violent-anomaly control (0.52) but **only partly** passes the scene-prior control. |
| Contact tags ("grabbing", "pushing") in 6 to 8 % of pre-onset frames vs 0.1 to 0.2 % of clean frames | UCA captions describe violence before about 19 HIVAU onsets. Only 10 of the 34 UCF events with an early violent-looking frame have a matching human caption of violence or a weapon | The strongest early "state" cue may be **unlabelled violence**. Audit it before building on it (section 6.1). |
| Blunsden 2009: pre-fight recall 0.20 on continuous video | A 2009 bag-of-cuboids method on acted data, with a non-causal plus or minus 50-frame window | Weak support for dropping skeleton classifiers; they are dropped for other reasons (section 5). |

Verified as stated:
- **Potegal 2025:** about half of 100 dyadic fights were brief unilateral assaults, a quarter of those sneak
  attacks. Caveat: filmers cut the build-up.
- **Crowner 2005:** antecedents before 60 % of assaults vs 10 % of control periods.
- **Axon body cameras:** the pre-event buffer has no audio by default; it is 30 s by default and configurable
  from 0 to 120 s.
- **PULS:** its advantage peaks at T-0.5 s.
- **Hawkes 2503.15821:** autistic psychiatric inpatients, and aggression includes self-injury.

**What remains.**
- **Measured.** A vision signal in our own data that reflects a current state. It is partly scene prior, present
  in a minority of events, and not yet scored at the event level.
- **Hypothesis.** Every other route to a 30 s+ lead.
- **The ranking that follows** is therefore a judgement about where the remaining signal most likely is, not a
  ranking of measured effects.

---

## 2. Data facts that limit every modality (verified 2026-10-14)

| Fact | Value | Why it matters |
|---|---|---|
| Eligible30 events | 551 in 468 videos; 312 first-segment, 239 later-segment | First segments carry the pre-violence headline; later segments are lulls after violence |
| First-segment onset time | median 51.4 s into the clip; 204 of 312 at or before 70 s | For most events, the 30-60 s window lies in the first 30 s of the clip, so first-frame controls are confounded (section 6.5) |
| First-segment lead (`max_lead`) | >= 60 s: 119; >= 90 s: 76; >= 120 s: 54; >= 180 s: 23; >= 300 s: 8 | 180 to 300 s memory can be tested on only a few dozen events |
| All-event lead | >= 120 s: 87; >= 180 s: 41; >= 300 s: 19 | Leads above 120 s are not credited by the current metric (section 4.1) |
| First segments by source | xd_movie 103, ucf 89, xd_youtube 77, ubi 43 | |
| First segments by onset source | hivau 217, official 52, ubi 43 (official + UBI = 95) | HIVAU onsets are unvalidated; official + UBI is co-primary |
| First segments by kind of violence | dyadic (fight, assault, abuse) 184; instrumental (robbery, arrest, shooting) 98; riot 30 | One in three has no dyadic build-up |
| Real-world audio (ucf + xd_youtube, `has_audio`) | 209 events, 105 first-segment (xd_youtube 77, ucf 28); with >= 60 s lead 52, >= 90 s 30; official onsets 21 | Audio blocks are power-limited, and an unknown share is narration or music |
| Pose coverage (reviewer, 367 `pose_raw` files) | Median share of 30-120 s frames with >= 2 tracked people: UCF 0.21, UBI 0.95, xd_movie 0.55, xd_youtube 0.61. Median share of person-time in tracks >= 30 s long: XD 0.0, UBI 0.17, UCF 0.61 | Pair-identity features fail on XD (cuts, camera motion), and two people are rarely detected on UCF |
| Non-violent anomaly videos with >= 30 s before the anomaly (H1) | 399 (xd_youtube 228, ucf 156, xd_movie 15). 271 are road or explosion scenes (car accident 165, explosion 90, road accident 13, both 3). 128 are people-focused, all UCF (stealing 42, shoplifting 25, burglary 24, arson 20, vandalism 17). 8.2 h before the anomaly at a 300 s cap (9.3 h if cut at the first anomaly + 10 s). None are in `extract_manifest_v1` | A people detector passes "violent vs all H1" trivially; the specificity gate uses the 128 people-focused videos |
| Known label noise | UCA describes a strike that ends >= 10 s before the onset in 18 of 89 UCF first-segment events (reviewer); about 19 genuine cases across all UCF HIVAU-onset videos. In XD, HIVAU onsets often include the build-up (the first strike mention comes a median 3.7 s after the onset, more than 10 s after in 30 %) | Some "lead-in" is violence, and some "violence" is lead-in |
| Short-lead events | 817 non-eligible events with 10 to 30 s of clear lead-in (4.0 h) | Usable to train state heads; never in the evaluation |
| Teacher-covered subset | 1,031 videos, 198 eligible events, 338 clean negatives (15.2 h) | The only subset with the Qwen2.5-VL reference scores |
| Curated YouTube hold-out | about 8 events with >= 30 s of pre-onset footage (13 to 15 projected at 92 videos); 28 of 55 annotated videos are news, bodycam breakdowns or compilations; 125 clips show escalation without violence | Cannot measure Success@30; a descriptive check and a read-only specificity readout |
| Screening folds (0, 2, 3, 4; fold 1 frozen) | 446 events, 253 first-segment; >= 60 s 100, >= 90 s 61, >= 120 s 45; official + UBI 74; UCF 69; real-world (non-movie) 165; S_audio 86; people-focused H1 102 | The sample sizes behind every confidence interval in section 6.8 |

---

## 3. Ranked modalities

Columns:
- **Evidence for a 30 s+ lead on violence:** M = measured in this repo; P = published on violence; N = published
  only on non-violent outcomes, acted data or the current state; none.
- **Expected contribution:** a projection, stated as the time-matched AUROC for onset 30 to 120 s ahead ("band
  AUROC", section 6.5) of the block alone where a number is meaningful.

| Rank | Modality | Evidence for 30 s+ lead | Expected contribution | Expected lead | Status | Cost |
|---|---|---|---|---|---|---|
| 1 | Conflict-state semantics from a video-LLM (stage and cue checklist from log-probabilities; trained probe on mid-layer hidden states) | M, partial: no-context aggression 0.67-0.71, partly scene prior; P: none verified; RiskCueBench says zero-shot forecasting is near chance | Highest. Band AUROC 0.65-0.75 for the probe; gain over the scene nuisance model +0.02-0.06 | State-level, 30-120 s, only when the conflict is in frame (a minority of events) | Needs building (`vlm_escalation`); the Qwen2.5-VL aggression rating on disk is the baseline | Pilot 10-15 GPU-h; screen set about 25-30 GPU-h total in HF (estimate; timing test first) |
| 2 | Interaction and contact state from pose and tracks (identity-free aggregates first) | N and M weak: behavioural studies code these cues but give no time to the blow; UCA behaviour cues significant only at 10-30 s; the 30-120 s contact tags may be label noise | Medium on UBI and phone footage, low on UCF (people rarely detected) and on XD (tracks break). Band AUROC 0.55-0.65 alone; value lies in complementarity and interpretability | Mostly 10-30 s; 30 s+ only for sustained engagement | Raw features built (`pose_raw`, `pose_interact`); window integrators need building (CPU) | CPU only |
| 3 | Frame concept scores, SigLIP 2 prompt set v2 (contact, aggression, conflict phases) | none direct (transferred from the Qwen aggression result) | Medium-low; a cheap 1 Hz proxy and the fallback semantic stream. Band AUROC 0.55-0.65 | State-level plateau | Built; v2 needs a rescore only | Seconds (rescore) |
| 4 | Vocal prosody and conversational dynamics (loudness relative to the video's own past, shouting duty cycle, overlapped speech, speaker count) | M weak: loudness rises inside the last 30 s (21 movie clips); N: current-state and acted studies | Small and power-limited on VA30; larger on bodycam, phone and dashcam with diegetic audio | 10-30 s measured; longer in verbal build-ups (hypothesis) | Mostly built (`audio_dsp`, `ast_events`); duty cycles and pyannote overlap need building | CPU, plus under 0.2 GPU-h per 100 h |
| 5 | Speech content scored as escalation dialogue acts (command without reason, challenge, insult at a person, threat, refusal, de-escalation) | N only (police stops with human transcripts; online text) | Near zero measurable on VA30; the highest upside on bodycam and phone footage; sparse; narration leakage risk | Hypothesis: tens of seconds to minutes | Transcripts queued (`speech_tox`); scorer needs building | 3-5 GPU-h for the 176 h audio scope |
| 6 | Event history (time since the last "violence now" burst, burst counts over 5 and 15 min, antagonists still in view) | P on non-video data (Hawkes model, AUROC 0.85 at 1 min to 0.78 at 25 min on inpatients) and VA30 structure | Medium for later-segment events, but only on continuous single-scene footage; nothing for the pre-violence headline | Minutes after a first burst | Needs building (model side) | None beyond training |
| 7 | Context cues for particular kinds of violence (weapon shown, police or security, counter and cash, mask, people arriving) | M from human captions: UCA at 30-60 s, weapon 19 vs 1 %, police 19 vs 0 %, arrivals 34 vs 10 % for robbery and shooting (n = 20-56); partly the crime itself starting early | Medium for robbery, arrest and shooting (about 30 % of first segments); low for fights | 30-120 s for robbery and arrest | Carried by ranks 1 and 3 (no new detector) | None extra |
| 8 | Camera-wearer cues on bodycam, dashcam and phone footage (looming nearest person, approach toward the camera, hands raised at the lens, wearer stepping back, vehicle stopping, near-field vs far-field voice) | none (domain logic: the wearer is one party of the conflict and is invisible to person-to-person geometry) | Unknown; possibly large on bodycam and dashcam; cannot be tested on VA30 (almost no such footage) | Seconds to tens of seconds (hypothesis) | Needs building (CPU, from `pose_raw` and `motion`) | CPU |
| 9 | Generic motion and video embeddings and sound events (V-JEPA 2-L, optical flow, AST, CLAP) | M against: flow +0.009 at 10-30 s; shout and crowd rise only in the last 10 s | Strong for detection and the last 10 s; feeds the "violence now" head and event history | 0-10 s | Built | Already paid |
| 10 | Crowd-level group features for riots (density trend over 60-300 s, flow coherence, opposing-flow index, two-group standoff) | none (HIVAU first-clip terms: police, crowd, gear, gather) | Low; riots are 30 of 312 first segments | Unknown | Needs building (CPU) | CPU |
| cond. | Audio-visual omni teacher (Qwen3-Omni-30B-A3B) and open-vocabulary audio events (PE-A-Frame) | none for lead | Only if rank 4 or 5 is not FAIL in the screen | Unknown | Needs a separate transformers 5.x venv | 15-25 GPU-h (Omni, audio subsets); about 0.5 GPU-h per 100 h (PE-A-Frame) |
| cond. | V-JEPA 2-g | none | Only if V-JEPA 2-L reaches SHORT-HORIZON or better | 0-10 s | Not built | 6-10 GPU-h per 100 h |

### 3.1 Rank 1: video-LLM conflict-state rating

- **Extractor:** new `anticipation/features/vlm_escalation.py` following the `io.py` contract. Model
  `Qwen/Qwen3-VL-8B-Instruct` in HF transformers 4.57.5 (existing venv), BF16. Store the prompt hash in `meta` and
  never mix models or prompts in one column.
- **Prompt:** the stage 0-5 plus 16-cue checklist in [research/cue_extractors.md](research/cue_extractors.md)
  section 5.1.4. Prefill `{"stage": ` and read the softmax over the digit tokens; cue probabilities come from the
  digit string.
- **Pilot arms, at matched cost where possible:**
  - **A, single frame:** the frame at t, queried every 2 s. Added because RiskCueBench shows that frame order
    barely matters zero-shot.
  - **B, short clip:** 8 frames at 1 fps over [t-7, t], queried every 2 s. This is the earlier design.
  - **C, long window:** 32 frames at 0.25 fps over [t-124, t], plus the last 4 s at 1 fps, queried every 8 s.
    About the same visual tokens per video-second as B.
    - Uses Qwen3-VL text timestamps.
    - Adds trajectory items: the same two people approach repeatedly; the conflict has lasted over a minute; a
      person held back is still agitated; someone left and returned; an intervention failed.
    - The window is truncated at the clip start and the available history is stored, so the column can be masked
      and is never a hidden clock.
  - **Forecast column:** P("a physical fight starts within 60 s" = yes), on arm B's visual prefix with its own
    prompt hash. Its marginal cost is a few percent. Treat it as a test, not a bet, given RiskCueBench.
  - **No text carried between queries in any arm.** The legacy failure came from carrying verdicts forward.
- **Readout:** mean-pooled visual-token states and the last prompt token's state at layers 18 and 27, LayerNorm'd,
  stored as float16.
  - **Primary output:** a probe trained on VA30 folds over the hidden states and the log-probability columns.
  - **Secondary:** the decoded stage.
  - The hidden-state probe is high-capacity and can learn which collection a video comes from, so it is screened
    as its own block, separate from the log-probability columns.
- **Gating:** skip queries with no tracked person over [t-7, t]. Pose misses small people on UCF, so ablate the
  gating on **all** UCF events in the pilot, not on a random 10 %.
- **Contamination probe** (about 1,000 queries): from calm first frames, ask "which UCF-Crime category is this
  video?". If accuracy is above chance on violent and H1 videos (permutation test), the model has seen the
  datasets: flag it, and weight UBI and the hold-out as the cleaner checks.
- **Baseline to beat:** the existing no-context `aggression_level` (Qwen2.5-VL, `processed/vlm_labels_violence`),
  not `b_teacher`, which is the collection prior.
- **Later (phase 2):**
  - a second image tile with the one or two most engaged dyads cropped (people are about 2x4 visual tokens on
    320x240 CCTV);
  - distilling hindsight state labels into the probe or a LoRA;
  - `Qwen/Qwen3.5-9B` with thinking off in a vLLM venv, once the NVML mismatch is fixed.

### 3.2 Rank 2: interaction and contact state from pose

- **New CPU module on `pose_raw`** (for example `pose_state`).
  - **Identity-free per-tick features first:**
    - any close pair (within 1.5 body heights) that is mutually oriented;
    - the maximum pair score;
    - near-contact (a wrist on the other person's torso or arm);
    - people count and gathering rate;
    - bystander orientation convergence and ring circularity around the most engaged pair;
    - people moving fast toward that pair;
    - third-party interposition.
  - **Then causal summaries over 30, 60, 120 and 300 s windows.** Each window is cut at the clip start and divided
    by the history actually available, min(W, t). Windows reset at shot cuts (use the cut detector in `motion`).
- **Pair-identity variants** (cumulative pair seconds, approach-retreat cycles, time since first engagement):
  score them only on UCF and UBI, where tracks persist. Appearance ReID and track stitching are phase 2.
- **Shortcut.** Cumulative 300 s features mostly measure time since the clip start unless they are normalised
  and evaluated time-matched. In deployment a fixed camera has unlimited history, so cap the windows at 300 s.

### 3.3 Rank 3: SigLIP 2 prompt set v2

Rewrite the prompts around two groups:
- **Conflict phases:** calm, tension, argument, face-off, contact, intervention, separation, violence,
  de-escalation.
- **Contact and aggression:** people pushing each other, a man grabbing another man, fighting stance, raised
  fists.

Keep the context prompts (CCTV, movie, news, captions) only as shortcut detectors. Bump `PROMPT_SET_VERSION` and
run `--rescore` on the cached embeddings; no video decoding is needed.

### 3.4 Ranks 4 and 5: the verbal channel

- **Only diegetic audio counts.** Every audio and speech block is judged only on footage whose audio is the scene
  itself (section 6.1, step 4).
  - UCF CCTV rarely records sound, so a UCF track may be narration or music.
  - XD YouTube includes news edits and compilations. A narrator saying "watch what happens next" leaks the outcome
    into speech features.
- **Prosody** (from existing streams):
  - loudness relative to the video's own past;
  - the shouting duty cycle (share of the last 60 and 180 s with AST Shout or Yell above a threshold);
  - slopes over 15, 30, 60 and 180 s;
  - plus the `pyannote/segmentation-3.0` overlapped-speech ratio and speaker count per second.
- **Dialogue acts:**
  - Model: a local `Qwen2.5-7B-Instruct` (or `Qwen3-8B` with thinking off).
  - Input: the rolling 60 to 120 s VAD-gated Whisper transcript.
  - Output: log-probabilities for command, reason given, challenge or dare, insult aimed at a person, threat,
    refusal, de-escalation, plus a stage 0-4.
  - Filter hallucinations (no-speech probability, repetition, text over non-speech stretches).
  - Keep detoxify as a secondary column only.
- **Expectation:** these blocks will mostly come out INCONCLUSIVE on VA30. The decisive tests are the human study
  (4.5) and long-form bodycam data (4.3).

### 3.5 Ranks 6 to 10

- **Event history.** Build from the "violence now" head (V-JEPA, flow, AST, VLM stage 5).
  - Score it only on continuous single-scene footage, using a shot detector to mark same-scene events.
  - 125 of the 239 later-segment events are XD, often a different scene in a compilation; there, history learns
    "this is a fight compilation". Report XD compilations separately.
  - Report event history only in the later-segment and deployment breakdowns, never in the pre-violence headline.
- **Context cues for particular kinds of violence.** Use them as auxiliary heads or class-conditional features.
  - Judge them per pooled group: instrumental (robbery, arrest, shooting) vs dyadic.
  - Exempt them from the specificity gate: robbery precursors (counter, entry, concealment) also come before
    shoplifting and burglary.
- **Camera-wearer cues.** Build them for models trained and evaluated on bodycam, dashcam or phone footage.
  - Keep them out of the pooled VA30 anticipation head, where camera behaviour is a provenance shortcut.
  - Inside their own domain they are legitimate precursors, because deployment is that same domain.
- **V-JEPA, flow and sound events.** Route them to the violence-now and 10 s heads. Keep the camera-motion columns
  in a separate shortcut block.
- **Riot crowd features.** Low priority; report them per class.

---

## 4. Levers larger than the choice of modality

### 4.1 Measurement: credit long leads

- **The current rule cannot credit long leads.** VA30 accepts an alarm only if it starts in
  [max(t0 - 120, clear_start), t0), and an alarm already running when that window opens earns nothing.
  - A conflict-state detector that fires when a 3-minute conflict begins is scored as a miss.
  - The same alarm counts in `fa_per_hour_all`.
- **Add a secondary sustained-warning metric.** For lead L in {60, 90, 120} on events with `max_lead >= L`:
  - **Success_sus@L** = an alarm episode (default 2-of-3 smoothing, 10 s merges) that starts in
    [max(clear_start, 2), t0 - L] and stays active through t0 - 30.
  - **Operating points:** set by `alarm_time_fraction_clean` (1, 2 and 5 % of clean time), not by FA/h. A sustained
    alarm costs time switched on, not a count of false alarms.
  - **Position null:** recompute it for this variant, since alarms at the clip start now earn credit.
  - **Reference to the true start of the conflict:** report alarm start relative to the annotated `t_conflict`
    where one exists (4.2).
- **Report two alarm tiers:**
  - **watch:** time fraction up to 5 %, with sustained Success@60 and @90;
  - **alarm:** 1 FA/h, with Success@30.

  This matches the evidence: the state gives the lead, the triggers give the precision.

### 4.2 Labels: what the onset is, and what the state is

- **Precursor timeline annotation** (the highest-value human task). Cover all 100 screen-fold events with at
  least 60 s of lead, plus a stratified 50 more, plus the hold-out. For each, record:
  - `t_conflict`: the first visible sign of the conflict, or "before clip start";
  - `t_contact`: the first non-strike contact (push, grab, chest to chest);
  - `t_strike`: the first punch, kick or weapon use;
  - `t_audible`: the first raised voice, insult or threat, with the sound on;
  - codes: sudden attack, audience, circle, filming, intervention, retreat, weapon visible.

  The distribution of `t_strike - t_conflict` is the empirical ceiling for Success@30. The sudden share is the
  floor of misses. Suggested location: `benchmark/precursors/` (CSV + README).
- **Onset definition.** The owner's target is physical violence. VA30's onsets mix the walk-up (some XD HIVAU
  onsets) with first blows and, in some UCF HIVAU onsets, come after earlier violence.
  - Keep VA30 v2 as the frozen primary benchmark.
  - Report a `t_strike` secondary onset only on human-verified events.
  - If it is adopted, version it as VA30 v3.
  - Do not re-anchor onsets with a model trained on the same noisy onsets; that is circular.
- **Label-noise masking** before the screen (section 6.1, step 3).
- **Hindsight state labels (phase 2).** An offline, non-causal VLM that sees the whole clip, including the violence,
  back-annotates the stage per tick plus `t_conflict` and `t_contact`.
  - Model: `Qwen3-VL-32B` in HF on one GPU.
  - Scope: violent videos, H1 lead-ins and a sample of negatives.
  - Cost: about 5-10 GPU-h (estimate).
  - Validate against the human timelines.
  - These labels are training targets only, never features. They replace the circular plan of training the state
    head on the same causal VLM's outputs.

### 4.3 Data: the only route to minute-scale leads and to the audio question

- **Long-form domain-matched set** (new; kept channel-disjoint from the 92-video hold-out).
  - Sources: unedited police-agency bodycam releases (queries for full, unedited footage), uncut dashcam, and
    livestream VODs.
  - Duration: 300 to 5,400 s.
  - Labelling: a VA30-trained violence-now head proposes onsets; humans verify only the onset and `t_conflict`.
  - Target: at least 200 events with 60 s or more of lead-in, all with diegetic audio.
  - Why bodycam: Willits and Makin report force at a mean of about 5 min into bodycam encounters, so leads of
    minutes exist there.
  - `setup/curate_youtube.py` defaults to `--max_duration 600` and short-form queries. Both select against long
    lead-ins and must change for this pool.
- **NTU CCTV-Fights:** requested on 2026-10-14, awaiting approval
  ([DATA_ACQUISITION.md](DATA_ACQUISITION.md) section 5). Its long untrimmed CCTV is what the long-window pose
  features need.
- **CamNuvem** (486 robbery CCTV videos with long approach phases). Check the licence and onset quality before use.
- **H2, confrontation negatives** (at least 100): arguments that end without contact, police stops without force,
  road disputes, bar quarrels broken up. Annotate them "no contact".
  - Until they exist, the hold-out's 125 escalation-without-violence clips serve as a read-only specificity
    readout. They are never used for decisions.
- **H0, scene-matched calm negatives** (phase 2): bar and club CCTV at night, nightlife streets, routine traffic
  stops on bodycam.
- **Full-length negatives for a subset.** The 300 s cap removes long sequences that self-supervised forecasting
  pretraining and long-window features need.

### 4.4 Objective

See section 8. In short:
- an event-level band objective;
- hazard bins extended to 300 s;
- no distance-based down-weighting;
- time-matched sample weights;
- training with random history truncation.

### 4.5 Human forecasting study (the cheapest direct test of audio's lead)

- **Clips:** 60 eligible events with diegetic audio (XD YouTube plus long-form clips once curated) and 60
  time-matched negatives drawn from clean videos and H2.
- **Cuts:** each clip is cut at t0 - 30 and at t0 - 60.
- **Raters and order:** two or three raters each give P(fight within 60 s), first with the sound off, then on.
  Counterbalance clip order across raters.
- **Outputs:**
  - human AUROC with sound vs without, which measures audio's added lead directly;
  - the overall human AUROC, which sets a realistic ceiling.
- **Cost:** about 4-6 h of owner time per rater.

---

## 5. What to drop or demote

| Item | Decision | Reason |
|---|---|---|
| Face detection, facial expression, gaze, mutual gaze | Drop | Faces are about 10-14 px on UCF and about 22 px on XD street footage. FER is about 63 % on AffectNet even at full resolution. There is no lead-time evidence, the same signs accompany fear, and close-ups make it a movie shortcut. Revisit only on phone or bodycam close-ups (faces >= 48 px) after human onsets exist |
| NTU-120 skeleton action classifiers (PoseC3D, CTR-GCN, InfoGCN) | Drop | Their aggressive classes (punch, kick, push) are the onset itself. They are trained on a lab domain, and pyskl has been unmaintained since 2023. Blunsden's 0.20 recall is only weak supporting evidence |
| World-model surprise features, V-JEPA 2.1 | Drop | Surprise peaks at T-0.5 s (PULS); V-JEPA 2.1 has no anticipation evidence |
| Open-vocabulary weapon, phone or police detectors on CCTV | Drop | Guns are about 16 px; cross-dataset AP50 is 4-12. The VLM checklist carries these cues |
| Context-carrying VLM verdicts (`vlm_labels`, `b_teacher` style) as features | Drop | A static per-video collection prior: 0.72 from the first frame alone, 0.71 before non-violent anomalies |
| Camera-operator behaviour (shake, zoom, start of filming) in pooled models | Drop from the pooled anticipation head; keep for audits and inside bodycam, dashcam and phone models | A phone-only provenance cue that does not transfer to CCTV. Within its own domain the wearer's motion is a legitimate cue |
| Generic toxicity (detoxify) as the main speech target | Demote to a secondary column | It fires on benign profanity (DeToxy Test-T) and loses about 0.12 F1 on ASR text |
| Voice emotion and arousal (emotion2vec anger, w2v2 arousal) | Demote: screen as a low-prior candidate on diegetic audio, not as a negative control | The only negative result is on 21 movie clips where SER is unreliable; that test is invalid, not negative. Extraction is already paid |
| Joint-angle pose descriptors in place of box-level and group geometry | Deprioritise | A robustness choice (keypoint flicker at 54-99 px), not evidence-based (section 1, Ganesh) |
| More generic backbones (ImageBind, InternVideo2, extra CLIP variants) | Do not add before the screen | No lead evidence; more collection-prior capacity |
| Tuning on XD movie build-ups and movie audio | Avoid; report `xd_movie` separately | Soundtrack swells and editing grammar act as precursors |
| "Nothing ramps, so drop slope features" | Rejected as a premise | It rests on 24-52 videos per bin. Slopes stay as a pre-registered probe arm (P1s) |

---

## 6. The empirical screening experiment

Goal: measure with identical machinery how much 30 to 120 s anticipation signal each candidate block carries
after the known shortcuts are removed:
- position and provenance;
- scene and collection prior;
- people present;
- unlabelled violence in the lead-in;
- detection memory.

Everything except the VLM pilot runs on CPU. Outputs go to `benchmark/results/cue_screen/`.

### 6.1 Phase 0: prerequisites (blocking; CPU or human; no new GPU job)

1. **Evaluator work** (`anticipation/eval/va30.py`, by the agent assigned to evaluation, with unit tests):
   - `cap_sec` support, so the 655 capped clean negatives are scored on their first 300 s;
   - the band metric (time-matched AUROC, pre-violence ticks with time to onset in (30, 120]);
   - the sustained-warning metric of 4.1 with its own position null;
   - the event-level window-max AUROC (M8);
   - a person-count-stratified time-matched AUROC (M2).
2. **Score the no-context `aggression_level`** (`processed/vlm_labels_violence`, CPU, `--restrict_to_scored`,
   teacher subset) through the full VA30 evaluator, with the event metrics. Also score its first-frame broadcast,
   its people-focused H1 control, and its real-world vs movie breakdown. This is the reference every VLM arm must
   beat.
3. **Label audit.**
   - **Contact sheets** (`tools/contact_sheet.py`): the 53 events with a no-context frame scored 0.8 or more in the
     30 to 120 s window. Classify each as:
     - (a) non-violent contact or posturing;
     - (b) violence before the annotated onset;
     - (c) teacher false positive or unclear.

     If (b) dominates, mask those lead-ins and downgrade the contact-state hypothesis (ranks 2 and 3) in the
     write-up.
   - **Soft-mask UCF lead-ins** from the start of a UCA strike caption that ends at least 10 s before the onset, up
     to the onset (about 18 to 22 events). Keep weapon-only captions, but flag them for a sensitivity run.
   - **Fill in the HIVAU spot-check CSV:** 50 HIVAU and 30 official sheets are ready.
4. **Audio-type audit** of the videos behind the 209 real-world audio events and the 92 hold-out videos. Classes:
   diegetic, narration, music, mixed. Use the AST Music and "Narration, monologue" scores plus about 3 h of
   listening. Also tag news, compilation and bodycam-breakdown videos. The audio subset for Phase B is the
   diegetic one.
5. **Pre-register.** Write `screen_set_v1.json` (with its sha256), plus `summary.md` holding:
   - the block list;
   - the probe grids;
   - the thresholds and decision rule of 6.6;
   - the confirmation fold (fold 1).

   Do this before any block result is read.
6. **Add the 399 H1 videos to a manifest v2.** Cut each with a stream copy at min(first anomaly + 10, 300) s, about
   9.3 h. Run pose, visual and motion on them, and audio where present (under 1 GPU-h).

### 6.2 Folds and confirmation

- **Screening folds:** VA30 `folds_v2` folds 0, 2, 3 and 4, giving 4-fold grouped cross-validation. Each fold's
  validation slice tunes probe hyperparameters and alarm thresholds.
- **Fold 1 is frozen.** No screen result may read it. It is used once, for the final fused model.
  - Why fold 1: it removes the fewest first-segment events (59) and only 19 of the 119 long-lead events, against
    31 for fold 4.
  - It was chosen on composition only, before any result.
- **The curated hold-out is never used for a decision.**

### 6.3 Screening set S

| Part | Content | Use |
|---|---|---|
| Positives | The 373 eligible30 videos in the screening folds (446 events, 253 first-segment). Ticks 1 to (last eligible onset + 10 s), about 16.7 h. Masked ticks follow `Benchmark.labels`, plus the label-noise soft masks of 6.1 | Headline frame metrics: first-segment pre-violence ticks. Event metrics: all 446 events, split first vs later segment |
| Clean negatives (CPU blocks) | Every clean negative in the screening folds (about 3,200 videos, about 125 h at the 300 s cap) | All CPU blocks |
| Clean negatives (VLM) | 600 videos, 150 per source kind, including every teacher-covered clean negative in the screening folds; first 300 s (about 25 h) | VLM blocks only; past about 500 negatives the AUROC CI barely narrows |
| H1 | Every H1 video in the screening folds (about 320); ticks 1 to the first anomaly, at most 300 s | Diagnostics, per class |
| H1p (people-focused H1) | The 102 stealing, shoplifting, burglary, vandalism and arson videos in the screening folds (all UCF) | Specificity gate G2 |
| S_first / S_real | 253 first-segment events / 165 of them not from movies | Primary / real-world primary readout |
| S_long60 / S_ramp | 100 first-segment events with >= 60 s lead / 61 with >= 90 s | LONG-LEAD tag; ramp and first-frame diagnostics |
| S_offubi | 74 first-segment events with official or UBI onsets | Co-primary (unvalidated HIVAU onsets) |
| S_audio | 86 first-segment real-world audio events (ucf + xd_youtube), restricted to diegetic audio after 6.1 | Phase B |
| S_teacher | Teacher-covered videos in the screening folds | Head-to-head with the Qwen2.5-VL references |
| VLM pilot set P | S_long60 events, ticks from max(1, t0 - 300) to t0 + 10 (about 4 h); 150 clean negatives (about 6 h); H1p (about 2.5 h) | Arm comparison and go/no-go (6.7) |

### 6.4 Candidate blocks, controls and probes

**Phase A** (CPU; starts when the visual and pose lanes finish and H1 is extracted):
- `pose_interact` (as is);
- `pose_state`, identity-free;
- `pose_state` pair-identity variant (UCF and UBI only);
- `siglip2_prompts` v1 and v2;
- `siglip2` embedding;
- `vjepa2_l`;
- `motion` without its camera columns;
- the camera columns as a separate shortcut block;
- camera-wearer cues (diagnostic only; almost no bodycam data);
- references on S_teacher: `aggression_level` and `b_teacher`.

**Phase B** (after the audio lanes; on diegetic S_audio):
- `audio_dsp`, `ast_events`, `clap_prompts`, `speech_tox`;
- voice affect and emotion2vec (as a regular candidate);
- the overlap ratio and duty cycles;
- the dialogue-act scorer.

**Phase C** (cuda:1 after pose; section 6.7): the VLM arms A, B and C, the forecast column, and the hidden-state
probe.

**Controls**, scored with the same machinery:
- `b_time_source_fit` (position and provenance).
- **Nuisance model N:** L2 logistic regression on source-kind x time-bin indicators plus B-scene0p. B-scene0p is the
  SigLIP 2 image embedding of the frame at tick 2 with every pose-detected person box greyed out, reduced to 32
  dimensions by PCA on the training folds. Masking people keeps a clip that opens mid-conflict from counting as
  "scene".
- **Shuffled-video control:** each video's features are replaced by a random clean video's from the same source kind
  at the same ticks (expect 0.50).
- **B-detect** (diagnostic): the block's probe trained on in-violence vs clean ticks, then applied to pre-onset
  ticks.
- First-frame broadcast on S_ramp (diagnostic).
- `io.check_causal`, already enforced in every extractor.

**Probes.** Identical for every block. Grids are fixed in the pre-registration.

- **P0 (no fitting):** each column with a pre-declared risk direction. It ranks columns and feeds the sparse-cue
  route.
- **P1 (primary):**
  - **Features per tick:**
    - causal summaries over the last 10, 60 and 180 s: mean, max, and the fraction of ticks above the column's 95th
      percentile on clean training ticks;
    - each window is cut at the clip start, divided by the history actually available, and reset at shot cuts.
  - **Arms:**
    - P1s adds 15, 30 and 60 s slopes;
    - P1b subtracts a per-video early baseline (an ablation, not the default, because it cancels the state in clips
      that start mid-conflict).
  - **Wide blocks** (more than 256 columns: SigLIP embedding, V-JEPA, VLM hidden states) are reduced with PLS to 32
    components. The PLS is fit on the training folds with the band target and time-match weights. Unsupervised PCA
    would keep the high-variance scene and collection directions. A full-dimension L2 run is a sensitivity check.
  - **Model:** L2 logistic regression; C in {1e-3, 1e-2, 1e-1, 1, 10}, chosen on the fold's validation slice.
  - **Target, the band:**
    - positives: first-segment pre-violence ticks with time to onset in (30, 120];
    - negatives: clean ticks with at least 60 s left;
    - sample weights: VA30 `time_match_weights` (source kind x time bin), so the probe cannot learn position;
    - H1 is not in training; it is the specificity test.
  - **Side probes:**
    - P1-near: time to onset in (0, 10] (short-horizon classification);
    - P1-later: later-segment lead-ins (event history), reported separately.
  - Each fold's model scores its test fold and its validation slice, as the VA30 threshold rule needs.
- **P2 (robustness; only for blocks that pass G1):**
  - one-layer causal GRU with 64 units over up to 180 s of the same inputs, masked by available history;
  - band target plus the event-level loss of section 8;
  - 3 seeds.

  It checks that temporal modelling does not change the ranking.

### 6.5 Metrics

All metrics use 95 % cluster-bootstrap CIs over `split_group`, with 1,000 resamples. Differences use paired
resamples.

| ID | Metric | Role |
|---|---|---|
| M1 | Band AUROC: time-matched (source kind x `TIME_MATCH_EDGES` bin) AUROC of first-segment pre-violence ticks with time to onset in (30, 120] vs clean ticks with >= 60 s left. Also (30, 60], VA30-compatible | Gate G1 |
| M1-long | Same for time to onset in (60, 120], on S_long60 | LONG-LEAD tag |
| M1-sparse | Time-matched standardized pAUC at clean FPR <= 1 % (McClish), and the per-event tail hit rate: share of events with >= 3 ticks in [t0-120, t0-30] above the clean-training 99.5th percentile, minus the same share for time-matched 90 s clean windows | Alternative route through G1 for sparse cues |
| M1-near | Same as M1 for time to onset in (0, 10] | SHORT-HORIZON |
| M2 | Time-matched AUROC with a person-count stratum (0 / 1 / 2 / 3-5 / 6+) added to the matching; also the n_persons >= 2 restriction, with the events kept per source | Diagnostic |
| M3p | Specificity: UCF first-segment band ticks vs H1p ticks with time to anomaly in (30, 120], matched on time since start. Plus the specificity ratio (M3p - 0.5) / (M1_UCF - 0.5). Also per H1 class, all H1, and H1 vs clean | Gate G2 |
| M4n | Paired gain in M1 of [N + block] over N | Gate G3 |
| M4b | First-frame broadcast M1, on S_ramp only | Diagnostic (on clips with an early onset the first 10 s are pre-onset signal, so it cannot gate) |
| M5 | AUROC per time-to-onset bucket (0-10, 10-30, 30-60, 60-120 s); on S_ramp, paired within-video mean(0-30) minus mean(60-120), with the share of videos rising and a Wilcoxon p-value | Diagnostic (RAMP tag) |
| M6 | VA30 event metrics (`--restrict_to_scored`, `cap_sec`, per-fold validation thresholds, threshold-aware CI with N = 200): Success@30 / @10 / @60 at 1 and 2 FA/h; late-event Success@30; position null recomputed on the scored universe; sustained Success@60 / @90 at 1-5 % alarm time; `alarm_time_fraction_clean`; median lead | Ranking and reporting only, never a gate |
| M7 | Breakdowns: S_real (primary) vs xd_movie (secondary; plus AUROC matched within the same film, since 133 of 135 eligible movie videos share a film with clean clips); S_offubi; dyadic vs instrumental vs riot; first vs later segment; diegetic audio; news and compilations excluded; visible-conflict vs sudden (once annotated) | Reporting |
| M8 | Event-level window-max AUROC: per eligible event, the maximum score in [max(t0-120, clear_start), t0-30], against the maximum in time-matched 90 s clean windows. Also the partial AMOC area between 0.5 and 2 FA/h | Forward selection (threshold-free) |

### 6.6 Decision rule (pre-registered)

Each gate has three outcomes. PASS needs the CI to exclude the null; FAIL needs the CI to exclude a useful
effect; anything else is INCONCLUSIVE.

| Gate | PASS | FAIL |
|---|---|---|
| G1 signal (M1) | point >= 0.58 and CI lower > 0.50 after Holm adjustment across the blocks of the same phase (one-sided bootstrap p-values) | CI upper < 0.58 |
| G1 sparse route (M1-sparse) | standardized pAUC point >= 0.55 with CI lower > 0.50, or tail-hit difference CI lower > 0 | both CIs exclude a useful effect |
| G2 specificity (M3p) | CI lower > 0.50 and specificity ratio >= 0.5 (point estimates) | CI upper < 0.55, or ratio < 0.25 |
| G3 beyond scene prior (M4n) | point >= +0.02 and CI lower > 0 | CI upper < +0.02 |

Classes:
- **ANTICIPATION** (enters the 30 s head): G1 PASS (either route), G3 PASS, and G2 not FAIL.
- **PROVISIONAL:** G1 PASS, neither G2 nor G3 FAIL, but not ANTICIPATION. It may enter forward selection and must
  hold on the confirmation fold.
- **CONTEXT / SHORTCUT:** G1 PASS but G2 FAIL or G3 FAIL. Excluded from the anticipation head; kept for audits.
  It may be re-tested after training with hard negatives.
- **SHORT-HORIZON:** G1 not PASS, but M1-near CI lower > 0.60. It feeds the violence-now and 10 s heads and event
  history.
- **INCONCLUSIVE:** G1 INCONCLUSIVE. Not dropped; carried to the long-form data and the human study. This is the
  expected result for most audio and speech blocks.
- **DROP:** G1 FAIL and not SHORT-HORIZON.

Tags:
- **LONG-LEAD:** M1-long point >= 0.58 with CI lower > 0.50.
- **RAMP:** M5 Wilcoxon p < 0.05 with a majority of videos rising.

Exemptions:
- Context-cue columns for particular kinds of violence (weapon, police, counter, arrivals) are exempt from G2. They
  are judged per class group and can enter only as auxiliary heads or class-conditional features.
- Audio and speech blocks are judged on diegetic S_audio. Flag a soundtrack shortcut if a block's `xd_movie` M1
  exceeds its S_audio M1 by more than 0.08.

Forward selection:
- Start from the best ANTICIPATION block.
- Add ANTICIPATION or PROVISIONAL blocks greedily while the paired M8 gain has CI lower > 0.
- Then evaluate the selected set once on fold 1.

### 6.7 VLM pilot and go/no-go

1. **Timing test:** 200 queries per arm. Projected costs above assume about 1 to 1.5 queries/s in HF for arm B.
   The legacy teacher measured 0.17 queries/s with about 300 output tokens; this design decodes about 20.
   - **Hard cap: 35 GPU-h for all of Phase C.**
   - If the projection exceeds the cap, drop arm A first, then cut the pilot negatives to 100.
2. **Contamination probe** (about 1,000 queries; section 3.1).
3. **Pilot:** arms A, B and C, the forecast column and hidden states, all on set P (about 12.5 h of video;
   estimated 10-15 GPU-h). Ablate pose gating on all UCF events in P.
4. **GO** to the rest of the VLM screening set (about 30 h more; estimated 10-16 GPU-h) only if the best arm meets
   all of these:
   - (i) a paired M1 gain over the no-context `aggression_level` on the shared S_teacher videos with CI lower > 0,
     **or** G3 PASS where `aggression_level` has G3 FAIL;
   - (ii) G2 not FAIL;
   - (iii) S_real M1 point >= 0.60 with CI lower > 0.50;
   - (iv) no contamination flag. If flagged, it may still go, but UBI and the hold-out become the deciding checks.

   Among passing arms, choose the one with the highest M8; on a tie, the cheapest.
5. **NO-GO:** SigLIP v2 stays the semantic stream, and the VLM is used only offline as the hindsight labeller of
   section 4.2.

### 6.8 Project-level rules and power

- **No surviving block.** If no block reaches ANTICIPATION or PROVISIONAL, these data do not support a 30 s claim.
  The headline becomes Success@10 and @20, sustained-warning results are reported descriptively, and the write-up
  says so.
- **Confirmation on fold 1.** The final fused model is evaluated once on fold 1. Two conditions must hold there:
  - M1 CI lower > 0.50;
  - a positive M8 point gain over N.

  Event-level significance claims use all five folds and are labelled "post-selection" for folds 0, 2, 3 and 4. A
  real confirmation needs new data (NTU CCTV-Fights, the long-form set).
- **Leakage audit:** any M1 above 0.85 triggers one before anything is reported.

Approximate 95 % CI half-widths on the screening folds (scaled from the reviewer's Hanley-McNeil estimates, which
reproduce the teacher's empirical plus or minus 0.052):

| Quantity | Events | Half-width |
|---|---|---|
| M1, first segments | 253 | +-0.04 |
| M1-long | 100 | +-0.06 |
| Ramp / M4b | 61 | +-0.08 |
| S_offubi | 74 | +-0.07 |
| M3p (UCF vs H1p) | 69 vs 102 | +-0.09 |
| S_audio (before the diegetic filter) | 86 | +-0.07 (about +-0.10 after it) |
| Confirmation fold 1 | 59 | +-0.08 |
| Success@30 near 15 %, fixed threshold | 446 | +-3.3 points (threshold-aware CIs are 1.5-2x wider) |

The earlier all-of-five rule had an estimated 5 to 10 % power for a realistically useful block. The
three-outcome gates above fix that by separating "no effect" from "too few events".

### 6.9 Outputs

All under `benchmark/results/cue_screen/`:
- `screen_set_v1.json` + sha256;
- `summary.md`, holding the pre-registration first and the results table after;
- one `<block>/metrics.json` per block;
- per-tick out-of-fold scores (`video_id, t_sec, score, fold, seed`) as parquet.

---

## 7. Work and extraction priority (ordered)

1. **Now, no GPU.**
   - Phase 0 steps 1 to 5: evaluator extensions, `aggression_level` through VA30, the label audit and masks, the
     audio-type audit started, pre-registration.
   - SigLIP prompt set v2 written and rescored as soon as the `siglip2` embeddings exist.
2. **Keep the running extraction queue unchanged.** Pose and visual lanes first, eligible30 videos first; then the
   audio and speech lanes.
3. **Data actions now.**
   - Follow up the NTU CCTV-Fights request.
   - Check CamNuvem's licence and onsets.
   - Start the long-form curation pool (new queries, longer `--max_duration`, channel-disjoint from the hold-out).
4. **Manifest v2 with the 399 H1 videos:** pose, visual, motion, and audio where present (under 1 GPU-h).
5. **After pose finishes (CPU):**
   - shot cuts from `motion`;
   - `pose_state` (identity-free first);
   - camera-wearer features;
   - crowd features for riots (low priority).
6. **Phase A screen (CPU).**
7. **Phase C VLM pilot on cuda:1:** timing test, then contamination probe, then arms on set P. Go/no-go, then the
   winning arm on the rest of S. Hard cap 35 GPU-h.
8. **Audio lanes finish.**
   - Duty cycles and the pyannote overlap ratio (CPU or under 0.2 GPU-h per 100 h).
   - The dialogue-act scorer with the narration and hallucination filters (3-5 GPU-h).
   - Phase B screen on diegetic S_audio.
9. **Human work in parallel:**
   - the precursor timeline annotation (4.2);
   - the human forecasting study with sound off and on (4.5);
   - H2 curation (at least 100);
   - the HIVAU spot-check;
   - continued hold-out annotation, adding `t_audible`.
10. **Phase 2, after the screen:**
    - hindsight state labels (Qwen3-VL-32B, about 5-10 GPU-h, validated on the human timelines);
    - appearance ReID and backward tracking from the onset on UCF and UBI (per-pair hazard with same-frame
      negatives, combined by multiple-instance learning);
    - the VLM dyad-crop tile;
    - probe or LoRA distillation of the hindsight labels;
    - self-supervised forecasting pretraining of the temporal head (needs full-length negatives for a subset);
    - UCA caption targets;
    - conditional streams: PE-A-Frame and Qwen3-Omni (if rank 4 or 5 is not FAIL), V-JEPA 2-g (if V-JEPA 2-L is
      SHORT-HORIZON or better);
    - the VLM on fold 1 and the full scope after a GO.
11. **Do not extract:** face, FER or gaze; NTU skeleton classifiers; surprise features; V-JEPA 2.1; open-vocabulary
    weapon detectors on CCTV.

---

## 8. Modeling implications

- **Objective.**
  - Keep the censored discrete-time hazard, but extend the bins beyond (45, 60] to (60, 90], (90, 120], (120, 180]
    and (180, 300]. The censoring logic in `labels.py` already handles it.
  - Add a band head P(onset in (30, 120]).
  - Add an event-level multiple-instance loss: a soft maximum (log-sum-exp or noisy-OR) of the band probability over
    the ticks in [max(t0-120, clear_start), t0-30] of each eligible event, against clean ticks.
    - This matches Success@30.
    - It lets the model fire where a state is visible instead of forcing every far tick positive.
  - **Do not** down-weight positives by distance from onset; that pushes learning toward the last 10 to 30 s. Once
    hindsight labels exist, treat far ticks with a visible state as full positives and far ticks before sudden
    attacks as unlabelled (positive-unlabelled learning).
  - Weight negatives with the VA30 time-match weights, or include source x time bin as a nuisance input dropped at
    scoring.
- **Two stages.**
  - **Stage A:** P(conflict active). Supervised by the human precursor timelines, the hindsight labels and UCA
    captions; never by the same causal VLM whose outputs are inputs.
  - **Stage B:** hazard given the state. Inputs: engagement duration, contact, challenges, audience, asymmetry,
    intervention and separation history.

  Lead comes from Stage A, specificity from Stage B. The 817 short-lead events (10 to 30 s, 4.0 h) can train
  Stage A; evaluation stays on eligible events.
- **Context length.**
  - Causal history up to 300 s, with every window feature normalised by available history and masked where
    incomplete.
  - Train with random history truncation, so outputs do not depend on the distance to the clip start; this also
    matches deployment, where history is unlimited.
  - 180 to 300 s memory can be evaluated on only 45 first-segment screen events with at least 120 s of lead. Treat it
    as exploratory.
- **Hard negatives.**
  - Add H1 to training after the screen, and evaluate specificity on held-out folds.
  - Add H2 when it exists, so the hazard head learns that most conflicts end peacefully.
  - Phase 2: same-frame negatives from backward tracking, which share camera, scene and time with the positives.
- **Collection prior vs real scene risk.** VA30's scene prior is a collection artefact and must not be learned from
  VA30. Real scene risk (nightlife, police stops) belongs in deployment-time calibration: per-scene FA/h budgets from
  the VLM's scene type, plus the per-camera baseline.
- **Fusion.**
  - Per-stream LayerNorm, presence masks and modality dropout.
  - The vision-only path must stand alone: audio and speech are absent on CCTV and on bodycam pre-event buffers.
  - Camera-wearer cues only in domain-specific models.
- **VLM outputs.**
  - Use probes and log-probabilities, never decoded text.
  - Freeze and hash prompts, tune only on validation slices, and never mix models or prompts in one column.
- **Two-sided state variables.** Treat intervention and retreat as state variables with learned effects, never as
  fixed negative evidence. Forceful intervention roughly halves continuation over the next few seconds, but turning
  away can trigger the attack.
- **Event history.** Make it first-class only on continuous single-scene footage, and report it on the later-segment
  and deployment breakdowns.
- **Auxiliary heads:**
  - violence now (fed by V-JEPA, flow and sound events);
  - contact happened;
  - phase or stage;
  - class-specific cues;
  - stage forecast at t + 30, + 60 and + 120 s;
  - optionally, UCA caption prediction 30 to 60 s ahead on UCF.
- **Reporting.**
  - Success@10, @20, @30 and @60 with lead distributions.
  - The sustained-warning tier (4.1).
  - The position null and late-event null at the same FA/h.
  - Breakdowns: real-world vs movie; official + UBI vs HIVAU; dyadic vs instrumental vs riot; first vs later
    segment; diegetic audio; visible-conflict vs sudden (diagnostic only).
  - FA/h on confrontation negatives as its own line.

---

## 9. Realistic expectations for 30 s anticipation

**Ceilings set by the data and the literature.**
- About half of one-on-one fights in internet videos are unilateral assaults with little visible escalation, and a
  quarter of those are sneak attacks. 40 % of video-recorded inpatient assaults had no antecedent in the prior 5 min.
- In VA30, 128 of 312 first segments are instrumental violence (robbery, arrest, shooting) or riots, with no dyadic
  build-up.
- A visible aggressive state 30 s or more ahead exists in about 13 to 41 % of events, depending on the detector.
  The 41 % comes with a 20 % false-positive window rate.
- In the real-world YouTube clips, visible escalation reaches 30 s in only 13 to 20 % (median 8.8 to 18.6 s), and
  that is judged from frames without sound.
- Only 54 first-segment events (45 in the screening folds) have 120 s or more of lead-in.

**Targets** (state them relative to the nulls at the same operating point):

| Quantity | Realistic | Reference |
|---|---|---|
| Time-matched band / far-window AUROC, fused | 0.65-0.75 | No-context aggression 0.67-0.71 (partly scene prior); STCH 0.62-0.69 at 60 s (unverified); above 0.85 means a leakage audit |
| Gain over the scene nuisance model N | +0.03 to +0.08 | If it is at or below +0.02, the "anticipation" is a scene prior |
| Success@30 at 1 FA/h, all eligible events | about 10-20 % | Position null about 5-7 % at that rate; teacher 11.6 % at 0.54 FA/h (null 6.0 %); outside movies 8.0 % |
| Success@30 at 1 FA/h, real-world (non-movie) events | about 8-15 % | Teacher: UCF 6.9 % |
| Late-event Success@30 | must exceed its own null (about 7 % at 2 FA/h) | The cleanest check that alarms are not clip-start alarms |
| Success@60 (eligible60) | at most about 10 % | Only sustained-conflict events |
| Sustained Success@60 / @90, watch tier | Unknown; the only metric that can show minute-scale lead | Dominated by the minority with a visible build-up |
| Success@10, Success@20 | Clearly higher; the honest secondary headline | |
| Events with a human-annotated visible conflict >= 30 s | Higher (perhaps 25-45 %) | Diagnostic only: the subset is conditioned on the cue existing |

**Audio.** Unknown. It might extend leads on bodycam, phone and road-rage footage. The human study (4.5) and the
long-form set (4.3) are the tests; VA30 cannot answer it.

**Deployment.**
- VA30 clean negatives are calm footage. In bars, police stops and traffic disputes, non-violent conflicts look like
  pre-violent ones until the last 10 to 30 s, so deployment FA/h will be several times VA30's.
- Bodycam pre-event buffers have no audio by default, so the verbal channel misses the first 30 s of a bodycam lead.

**If the screen fails.** If every vision block fails G2 or G3, the conclusion is that 30 s anticipation at low FA/h
is not supported by these datasets. The capstone then centres on Success@10 and @20, the measured ceilings, the
shortcut analysis, and the sustained-warning results.

---

## 10. Open risks

1. **The early "contact" state may be label noise** (unlabelled violence before HIVAU onsets). Section 6.1 step 3
   decides it; if so, ranks 2 and 3 lose their main supporting evidence.
2. **Scene or collection prior dressed as anticipation.** G3 and the people-masked nuisance model are the defence,
   but scene type within a source is only partly captured by one masked frame.
3. **HIVAU onset style.** 217 of 312 first segments rest on unvalidated HIVAU onsets that are sometimes early and
   sometimes late. S_offubi is co-primary until the spot-check is done.
4. **Power.**
   - Long-lead (100 events) and audio (86, fewer after the diegetic filter) analyses are small. Expect INCONCLUSIVE.
   - The confirmation fold (59 events) can confirm direction, not significance.
5. **Narration, music and burned-in text.** Speech features and the VLM can read the outcome from a narrator or burned-in
   headline text. Mitigations: audio-type filtering, results with news and compilation clips excluded, and an ablation with
   burned-in text masked.
6. **VLM pretraining contamination** with UCF-Crime, UCA, HIVAU or XD. The contamination probe tests it; UBI and new
   data are the cleaner checks.
7. **Compute.** HF throughput is an estimate. The vLLM path is blocked by the NVML driver mismatch. Phase C has a
   35 GPU-h cap.
8. **Domain gap.** There is almost no bodycam or dashcam training data. Camera-wearer and verbal-channel claims
   cannot be made until the long-form set exists.
9. **Multiple comparisons.** About 15 blocks, several subsets and forward selection. Mitigations: Holm on G1, the
   frozen fold 1, and labelling the other folds' numbers post-selection.
10. **Specificity in deployment** is unmeasured until H2 exists. The 125 hold-out escalation clips are a weak
    stand-in: edited, and some continue past a cut.
11. **Licensing.**
    - NTU CCTV-Fights forbids redistribution and derived datasets.
    - Curated YouTube footage is for research only.
    - Ultralytics (pose) is AGPL; WavLM MSP-dim is non-commercial.

---

## 11. Review log: criticisms of the draft and how they were handled

| Criticism (critique number) | Decision | Reason or change |
|---|---|---|
| Philpot 3.27 min and Lindegaard 92 s used as lead times (0) | Accepted | Section 1; neither is used as lead evidence |
| Rho 2023 overstated; "minutes later" not in the paper (0) | Accepted | Section 1; listed as non-violent-outcome evidence |
| DeToxy Test-T misquoted; bodycam WER from one sample (0) | Accepted | Section 1; toxicity demoted further |
| Kim, Lefter and Kooij are current-state, acted or detection studies (0) | Accepted | Section 1; prosody lead is listed as 10-30 s measured, longer as a hypothesis |
| Repo loudness result misquoted; F0 flat (0) | Accepted (checked against FEATURES.md) | Section 1; F0 rationale dropped |
| Ganesh read as geometry anticipation (0) | Accepted | Box-over-joint preference kept only as a robustness choice |
| RiskCueBench omitted (0) | Accepted | Trained probe as the primary readout; single-frame arm A added |
| STCH unverifiable and lacks controls (0) | Accepted | Not used as cue evidence |
| Teacher Success@30 is mostly movies (0, 1) | Accepted (arithmetic checked) | Real-world readout primary; same-film matched AUROC for movies |
| No-context aggression "passes both controls" overstated (0) | Accepted | Reworded to "partly passes the scene-prior control" |
| Contact tags may be label noise (0) | Accepted | Blocking audit in 6.1; rank 2 and 3 evidence marked weak |
| Narration and overlay leakage (0, 1) | Accepted | Audio-type audit, narration filter, exclusion breakdown, text-masking ablation |
| Long windows truncated by the clip start encode position (0, 1) | Accepted | Normalisation by available history, cut resets, history truncation in training, long memory exploratory |
| M4 first-frame broadcast confounded by early onsets (0, 1) | Accepted | M4b is a diagnostic on S_ramp; gate G3 uses the people-masked nuisance model |
| H1 mostly road scenes; hours overstated (0, 1) | Accepted (verified: 271 of 399 road or explosion; 8.2 / 9.3 h) | G2 uses the 128 people-focused videos (102 in the screening folds) |
| VLM go/no-go against the wrong baseline (0) | Accepted | Baseline is the no-context `aggression_level`, scored through VA30 in Phase 0 |
| Acceptance window cannot credit long leads (0, 2) | Accepted | Sustained-warning metric and two alarm tiers (4.1) |
| P1 target dominated by near-onset ticks (0, 1) | Accepted | Band target (30, 120] with time-match weights |
| S_audio underpowered; drop would be an artefact (0, 1) | Accepted | INCONCLUSIVE class; the human study and long-form data decide |
| Multiple comparisons; oracle thresholds (0, 1) | Accepted | Holm on G1, frozen confirmation fold, per-fold validation thresholds, M6 for ranking only |
| Voice emotion dropped on an invalid test (0) | Accepted | Demoted to a low-prior candidate |
| Stale v1 numbers (0) | Accepted | v2 used throughout; v1 figures named and superseded at the top |
| Targets lack a null; "30-50 % on visible-conflict events" circular (0) | Accepted | Targets stated against the nulls; the visible-conflict subset is diagnostic only |
| Blunsden is weak support for dropping skeleton models (0) | Partly accepted | The drop stands, for other reasons (classes are the onset itself, lab domain, cost) |
| Pose rarely sees two people on UCF; XD tracks fragment (1) | Accepted (reviewer measurement) | Identity-free features first; pair-identity on UCF and UBI only; shot-cut resets |
| Event history on XD compilations learns provenance (1) | Accepted | Only on continuous single-scene footage |
| Class-context cues penalised by G2 (1) | Accepted | Exempt from G2; judged per class group |
| Early-baseline normalisation cancels the state (1) | Accepted | Ablation arm P1b, not the default |
| Mask caption-flagged early violence; B-detect control (1) | Accepted | Phase 0 step 3; B-detect kept as a diagnostic, not a gate, because a state detector trained on violence can legitimately respond to precursors |
| Pooled out-of-fold thresholds; position null not recomputed on S (1) | Accepted | VA30 evaluator with per-fold validation thresholds and `--restrict_to_scored` (it recomputes the null) |
| Score CPU blocks on all clean negatives (1) | Accepted | Every clean negative in the screening folds; 600 for the VLM |
| VLM history truncated to [t0-180, t0+10] (1) | Accepted | Pilot from max(1, t0-300); full set from tick 1 |
| Gating ablation on 10 % holds about 9 UCF events (1) | Accepted | All UCF events in the pilot |
| VLM throughput unmeasured; contamination (1) | Accepted | Timing test, 35 GPU-h cap, contamination probe |
| All-of-five gate has 5-10 % power (1) | Accepted | Three-outcome gates; M2 and M4b diagnostics |
| Freeze fold 4 for confirmation (1) | Modified | Fold 1 frozen instead: fold 4 holds 31 of 119 long-lead events, fold 1 only 19 (chosen on composition, before any result) |
| "No cue ramps" used as a design premise (1) | Accepted | Slopes kept as arm P1s; RAMP tag diagnostic |
| Status message counted 489 events (1) | Accepted | 551 (VA30 v2) used throughout |
| Re-anchor onsets with a violence-now head and relabel the conflict phase (2) | Partly rejected | Machine re-anchoring with a head trained on the same noisy onsets is circular, and UCF HIVAU errors go the other way. Instead: a human-verified `t_strike` secondary onset (4.2); VA30 v2 stays the frozen primary benchmark |
| No plan for long-lead domain data (2) | Accepted | Section 4.3 |
| NTU CCTV-Fights "never submitted" (2) | Rejected as a fact | DATA_ACQUISITION.md section 5: submitted 2026-10-14, awaiting approval. |
| Use the hold-out's 125 non-violent escalation clips (2) | Accepted, read-only | A specificity readout, never a decision input |
| The VLM design drops what STCH credits; long-window query and forecast column (2) | Partly accepted | Arm C and the forecast column added at matched cost. STCH's ablation is unverifiable and its memory is a learned module, not a long prompt, so the long query is a test, not a premise |
| Dyad modelling: ReID, backward tracking, same-frame negatives, camera-wearer cues, crops (2) | Accepted, staged | Camera-wearer cues as rank 8 (domain models only); ReID, backward tracking and crops in phase 2 for UCF and UBI, given track fragmentation on XD |
| Circular pseudo-labels; use a hindsight teacher; 817 short-lead events unused (2) | Accepted (817 events, 4.0 h verified) | 4.2 and section 8 |
| Hazard bins stop at 60 s; down-weighting pushes away from long lead (2) | Accepted | Bins to 300 s, band head, multiple-instance loss, no distance down-weighting |
| PCA-64 keeps scene directions; sparse cues diluted (2) | Accepted | PLS-32 with a full-dimension sensitivity run; M1-sparse route |
| Self-supervised minute-scale forecasting pretraining (2) | Accepted, phase 2 | Needs full-length negatives for a subset |
| UCA and HIVAU captions underused (2) | Accepted, phase 2 | Auxiliary caption targets; gains expected mainly at 10-30 s |
| Separate collection prior from real scene risk; H0 negatives (2) | Accepted | Deployment calibration (section 8); H0 in phase 2 |
| Riots and shootings written off (2) | Accepted, low priority | Rank 10 crowd features; per-class reporting |
| Human audio on/off forecasting study (2) | Accepted | Section 4.5 |
| Two alarm tiers (2) | Accepted | Section 4.1 |
| "The draft's shortcut controls (M2-M4, H1) are sound and should stay as is" (2) | Rejected | Critiques 0 and 1 showed M4 is confounded and H1 is mostly road scenes; both were rewritten |
| The draft proposed saving the plan as `docs/research/modality_plan.md` | Superseded | The assigned output is `docs/MODALITY_PLAN.md`, linked from `docs/README.md` |

---

## 12. Reproducing the counts in section 2

CPU only, seconds. Run from the repo root with the venv Python (`nice -n 10 .venv/bin/python -I script.py`).

```python
import json, numpy as np, pandas as pd
R = "benchmark/"
ev = json.load(open(R + "events.json"))["videos"]
vc = pd.read_csv(R + "videos.csv").set_index("video_id")
asg = json.load(open(R + "folds_v2.json"))["assignment"]
rows = []
for vid, d in ev.items():
    if d.get("role") != "violent":
        continue
    for e in d.get("events") or []:
        if e.get("eligible30"):
            r = vc.loc[vid]
            rows.append(dict(vid=vid, idx=e["index"], onset=e["onset"], max_lead=e["max_lead"],
                             sk=r.source_kind, audio=bool(r.has_audio), os=d.get("onset_source"),
                             fold=asg.get(vid), cls=r.classes))
E = pd.DataFrame(rows); F = E[E.idx == 0]
print(len(E), len(F), F.onset.median(), [(L, int((F.max_lead >= L).sum())) for L in (60, 90, 120, 180, 300)])
print(F.sk.value_counts().to_dict(), F.os.value_counts().to_dict())
A = E[E.audio & E.sk.isin(["ucf", "xd_youtube"])]; print(len(A), int((A.idx == 0).sum()))
oa = vc[(vc.role == "other_anomaly") & (vc.first_anomaly_sec >= 30)]
print(len(oa), oa.classes.value_counts().to_dict(), np.minimum(oa.first_anomaly_sec, 300).sum() / 3600)
S = F[F.fold != 1]; print("screen folds:", len(S), int((S.max_lead >= 60).sum()), int((S.max_lead >= 90).sum()))
```

The pose coverage figures and the UCA early-strike count in section 2 come from reviewer scratch scripts
(`pose_cov.py`, `pose_cov2.py`, `early_strike.py` in the session scratchpad, temporary). Re-run them after the pose
lane finishes, because coverage was measured on the first 367 `pose_raw` files.
