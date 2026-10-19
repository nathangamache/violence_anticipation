# Cue extractors: the best 2024-2026 model for each violence precursor cue

Audience: the agent or developer who decides which feature extractors to run next for 30 s violence
anticipation on VA30 ([../VA30.md](../VA30.md)). Written 2026-10-14 (Investigation 3). Scope: frozen inference on
two data-centre GPUs, no fine-tuning of the extractor itself. It complements [backbones.md](backbones.md) (a backbone
survey) by asking a narrower question: **for each cue that could precede violence by 30 s or more, which
extractor measures it best on our footage, what will it cost, and is it worth it?**

How to read the evidence:
- **measured** = computed in this repo (file or script named).
- **paper** = number from the cited paper or model card. **secondary** = from a third-party summary.
- **est.** = back-of-envelope estimate (FLOPs, token counts, reference speeds). Re-measure before planning a
  long job around it.
- The web search budget ran out at the end of this investigation. A few numbers (marked "unverified") could
  not be re-checked against a primary source.

---

## 0. TL;DR

1. **The 30 s signal we can measure today is a persistent state, not a ramp.** The only control above the
   position null on VA30 is the zero-shot Qwen2.5-VL-7B `violent` score. Its AUROC against clean-negative ticks is
   0.67 to 0.70 in **every** time-to-onset bin from 120 s to 0 s, and in 35 videos with at least 90 s of
   lead-in its mean over the last 30 s exceeds its mean over 60 to 120 s before onset in only 17 % of videos
   (equal in 63 %) (**measured**, section 3.1). It recognises "this scene is a risky or confrontational one", not
   "violence is about to start". Extractors must be judged on two things: separating the pre-onset **state** from
   **scene-matched** negatives, and showing a **within-video ramp**. Section 3.3 gives a cheap screening protocol.
2. **Biggest gain per GPU-hour: replace the single-frame "violent?" question with a multi-frame structured
   escalation rating.** Ask for an escalation stage (0 calm to 5 physical) and a fixed-order checklist of about 16
   precursor cues. Score both from **token log-probabilities**, not sampled text, and pool mid-layer hidden
   states in the same pass. Structured, fine-grained questions are the largest lever in published VLM anomaly
   work: Qwen2.5-VL-7B on UCF-Crime goes from 74.5 to 89.8 AUC (ASK-HINT, **paper**), and InternVL2-8B from
   65.0 to 86.6 (VERA, **paper**).
3. **Model choice.** Use `Qwen/Qwen3-VL-8B-Instruct` now. It runs in the existing venv (transformers 4.57.5;
   the README requires >= 4.57.0, **measured** import OK). Upgrade to `Qwen/Qwen3.5-9B` (non-thinking mode)
   through vLLM once a separate venv works. On its card Qwen3.5-9B has VideoMME 78.4 (without subtitles) and
   MVBench 74.4, against 73.3 and 72.0 for Qwen3-VL-30B-A3B (**paper**). Use `Qwen/Qwen3.6-27B` or
   `Qwen/Qwen3-Omni-30B-A3B-Instruct` (audio + video) only as offline teachers on subsets. InternVL3.5-8B,
   VideoLLaMA3, LLaVA-Video and Gemma 3 are weaker on video or cost more tokens per frame.
4. **vLLM can be installed, but only in a separate venv, and the development host's NVML was broken.** `nvidia-smi` failed with
   "Driver/library version mismatch" (**measured**). vLLM's
   platform detection uses NVML and may refuse to start until the nvidia modules are reloaded or the host reboots.
   The HF transformers path works now. Estimated throughput for an 8-frame query: about 5 queries/s with vLLM
   against about 1 to 1.5 queries/s with HF batched generate (**est.**). The legacy teacher pipeline measured
   about 6 s per frame.
5. **Audio (only where audio exists: bodycam, phone, dashcam, XD).** Keep the DSP loudness and F0 slopes. They
   are the only audio cue that rises 30 s out: +2 to 3 dB relative loudness, up in 81 % of 21 XD fight clips
   (**measured**, FEATURES.md). Add `facebook/pe-a-frame-base`, an open-vocabulary detector at 25 Hz that takes
   prompts such as "people arguing loudly" (AudioSet-Strong AUROC 0.96, DESED 0.98, **paper**). It needs
   transformers 5.x, so it goes in the separate venv. Add a pyannote overlapped-speech ratio. Stop investing in
   speech emotion: anger posteriors did not rise before onset.
6. **Speech content.** Keep Whisper large-v3 + toxicity. Add an LLM escalation rating of the rolling 60 s
   transcript (local `Qwen2.5-7B-Instruct`): cheap (about 2 GPU-h per 100 h, **est.**), sparse, and high
   precision in bodycam and phone footage.
7. **Pose.** Keep YOLO11x-pose + BoT-SORT engineered features. They are cheap and domain-robust, and people
   gather and raise their arms before onset (**measured**, n = 19). Top-down ViTPose refinement is an optional
   fix for small people (UCF median person height 75 px). **Skip skeleton action recognisers**
   (PoseC3D, CTR-GCN or InfoGCN on NTU-120) for four reasons:
   - they are trained on a lab domain;
   - their aggressive classes (punch, kick, push) are the onset itself;
   - pyskl has been unmaintained since 2023;
   - a 2026 study found pose representations never beat coarse box geometry.
8. **Skip face analysis.** Faces are about 10 to 14 px in UCF CCTV and about 22 px in XD street clips
   (**measured**). Facial expression recognition (FER) is only about 63 % accurate on AffectNet even at full
   resolution (HSEmotion, **paper**). Gaze and mutual-gaze models need head crops of about 64 px or more.
9. **Open-vocabulary detection has low value for a 30 s lead.** UCF-Crime handguns average 16 px. Supervised
   CCTV detectors reach 43 to 57 AP50, and cross-dataset transfer to UCF reaches only 4 to 12 AP50 (CCTV-Gun,
   **paper**). Let the VLM checklist carry weapon presence. If a detector is wanted for raised phones or police,
   use YOLOE-26 (ultralytics 8.3.228, already installed; **measured** import OK).
10. **Temporal and world models.** Keep V-JEPA 2-L. Its published anticipation result (EK100) is a 1 s horizon,
    and the only V-JEPA-based anomaly-anticipation paper (PULS) has its advantage peak at T-0.5 s. Do not
    build "surprise" features. Use future-feature regression as an auxiliary training loss instead.

---

## 1. Scope and method

The question: for each precursor cue, which extractor is best? Each candidate was judged on:
- accuracy on low-resolution surveillance, bodycam and phone footage;
- speed and cost per 100 h of video on one GPU;
- license;
- ease of setup (HF transformers preferred; the repo venv is torch 2.5.1+cu121, transformers 4.57.5,
  ultralytics 8.3.228).

The cues come from [backbones.md](backbones.md) section 2, [literature.md](literature.md), the onset sanity
tables in [../FEATURES.md](../FEATURES.md), and CCTV and bodycam studies of conflicts. The behavioural literature
gives the **sequence** of escalation (verbal, posturing, pushing, hitting) but almost no **timing**:
- Levine et al. 2011 coded 42 CCTV incidents as escalating or conciliatory acts.
- Ejbye-Ernst et al. 2022 coded 46 Amsterdam CCTV conflicts. That footage had no sound, and 36 of the 165
  collected clips were dropped as too low-resolution to code.
- Philpot et al. 2020 found that bystander intervention is the norm in public conflicts.
- Willits and Makin 2018 coded time to force in bodycam footage.

No source found during this investigation reports a typical time from first verbal aggression to the first
blow. Lead times below therefore rest on our own measurements wherever possible.

Checked locally (CPU only, no GPU jobs):
- the resolution distribution of all 7,696 catalog videos;
- person and face sizes from the 5 pose smoke files;
- which model families transformers 4.57.5 imports;
- the NVML state;
- the lead-time profile of the zero-shot teacher on VA30 (script in Appendix A).

---

## 2. What our footage allows (measured)

| Source kind | Resolution (videos) | Real audio | Person box height (px, p10 / p50 / p90) | Face width (px, about 1.3 x ear-to-ear) |
|---|---|---|---|---|
| `ucf` (CCTV) | 320x240 (all 1,879) | 490 of 1,879 | UCF fighting clip: 54 / 75 / 99; UCF normal clip: 62 / 88 / 144 | about 14 and 11 |
| `ubi` (CCTV) | 640x360 (all 973) | none | not measured | not measured |
| `xd_movie` | 640x266 to 640x360 (letterboxed) | all | not measured | not measured |
| `xd_youtube` | 640x360 (2,163), 480x360, 320x240 | all but 4 | XD fight clip: 100 / 172 / 229 | about 22 |
| `yt` (curated hold-out) | 640x360 (80 of 92) | 90 of 92 | curated news-style clip: 98 / 259 / 288 | about 92 |

Sources: `benchmark/videos.csv` and `FEATURE_ROOT/pose_raw/*/*.npz` (high-confidence detections). The person and
face sizes come from only 4 videos with people, so treat them as indicative.

Consequences:
- **Faces are unusable on CCTV.** An 11 to 14 px face on UCF and about 22 px on XD street footage are far
  below the 48 to 112 px inputs that FER, head-pose and gaze models are trained on. Expression recognition at
  12 to 14 px is a research problem: a specialised low-resolution network reports 75.5 % on 7-class RAF-DB
  downsampled to 14x14, on curated frontal faces (GME-Net, **paper**).
- **Weapons and phones are tiny.** Handguns average 16 px in UCF frames (CCTV-Gun, **paper**).
- **Pose works but is noisy** at 54 to 99 px person height. Keypoint flicker was half the version-1 speed
  response (FEATURES.md, pose section).
- **Audio exists for 313 of 468 eligible30 videos (67 %), but 272 of those are XD.** Only 41 of the 196 CCTV
  eligible videos have audio (UCF 41 of 127, UBI 0 of 69) (**measured**, `benchmark/videos.csv`). Real CCTV
  usually has no audio (the Amsterdam CCTV study had none). So audio and speech cues can only help bodycam,
  phone and dashcam footage, plus XD, where movie soundtracks are a shortcut risk.
- **Everything is 640 px wide or less.** Qwen3-VL at native resolution therefore costs only 70 to 220
  visual tokens per pair of frames (section 5.1.5).

---

## 3. What the internal evidence says about lead time

### 3.1 The zero-shot VLM teacher is a state detector (measured)

Data: `benchmark/results/b_teacher/scores.parquet` (Qwen2.5-VL-7B `violent.score`, sample-and-hold to 1 Hz,
1,031 teacher-covered universe videos). Positives are unmasked pre-violence ticks of violent videos, by
time-to-onset bin. Negatives are unmasked ticks of clean negatives (53,316 ticks; mean score 0.052; score > 0 on
10.5 %). Script in Appendix A.

| Time to onset (s) | Ticks | Mean score | Fraction > 0 | AUROC vs clean | AUROC, UCF only |
|---|---|---|---|---|---|
| (0, 5] | 2,443 | 0.247 | 0.47 | 0.683 | 0.713 |
| (5, 10] | 1,752 | 0.238 | 0.45 | 0.675 | 0.708 |
| (10, 20] | 2,406 | 0.246 | 0.47 | 0.683 | 0.711 |
| (20, 30] | 1,558 | 0.225 | 0.44 | 0.667 | 0.713 |
| (30, 45] | 1,466 | 0.227 | 0.45 | 0.672 | 0.711 |
| (45, 60] | 946 | 0.252 | 0.48 | 0.690 | 0.717 |
| (60, 90] | 1,320 | 0.282 | 0.50 | 0.700 | 0.717 |
| (90, 120] | 933 | 0.277 | 0.46 | 0.685 | 0.717 |

**Within-video ramp.** Take the 35 violent videos with at least 30 pre-violence ticks in (60, 120] s and at
least 20 in (0, 30] s. The mean score is 0.269 far from onset and 0.252 near it. Near is higher in 17 % of
videos, equal in 63 % and lower in 20 %.

This agrees with the per-horizon table in `benchmark/results/b_teacher/report.md`, where AUROC is 0.671 to
0.677 for every H from 5 to 60 s.

Interpretation:
- The teacher earns its Success@30 (11.6 % at 0.54 FA/h, median lead 29.8 s) by recognising a risky scene early
  and alarming once. It cannot tell 90 s from 5 s before onset.
- Within UCF alone (one source, one resolution) the separation is still 0.71. So it is not only provenance.
  But it may partly be **scene type**: UCF anomaly videos show people in stores and streets, while many UCF
  normal videos show empty roads.
- A people-present control (section 3.3) is needed before any feature is credited with "escalation".

### 3.2 Hand-crafted streams around onsets (from FEATURES.md, measured)

21 XD-Violence B1/B5 test clips with a first onset at least 30 s in. Values are the change against each
video's own [t0 - 60, t0 - 30) baseline; n = 19 to 21; no significance tests.

| Stream | [-30, -10) s | [-10, 0) s | Fraction of videos up in [-10, 0) | Reading |
|---|---|---|---|---|
| DSP `rms_db_mean` | +2.3 dB | +2.7 dB | 0.81 | **The only signal already up 30 s out** |
| DSP `rel_loud_db` | +1.6 dB | +1.5 dB | 0.71 | same, relative to the video's own past |
| AST Shout / Yell / Crowd logits | -0.12 / +0.01 / -0.06 | +0.20 / +0.16 / +0.49 | 0.62 | last 10 s only |
| voice arousal / e2v anger | -0.02 / -0.05 | +0.02 / -0.06 | 0.62 / 0.38 | flat; SER is unreliable on movie audio |
| pose `n_persons_mean` | +0.53 | +0.89 | 0.53 | people gather |
| pose `arm_raise_rate` | +0.035 | +0.062 | 0.68 | gesturing builds |
| pose `n_close_pairs` | +2.4 | +2.5 | 0.58 | |
| pose `mutual_facing_closest`, `faceoff_frac` | about 0 | about 0 | 0.32 to 0.37 | no face-off signal from 2D pose |
| motion `flow_mag_mean` / `cam_shake` | +0.009 / +0.008 | +0.020 / +0.009 | 0.68 / 0.79 | modest; camera shake is a shortcut risk |

### 3.3 Consequences for choosing extractors, and a screening protocol

At 30 s or more ahead, the measurable thing is that **a confrontation or risk state is present**. A ramp shows
up only in the last 10 to 30 s (loudness, gathering, gesturing). The best extractors therefore:
1. recognise the escalation state **specifically**, telling an argument apart from a busy store;
2. keep working on silent, low-resolution CCTV;
3. add a ramp that tells the temporal head when the state is worsening.

Before training any model on a new stream, screen each candidate column or score. All steps are CPU only and
take minutes per stream:
1. `far_window_auroc.clean_time_matched` from the VA30 evaluator, with the stream as the score.
2. **People-present matched negatives (new).** Restrict clean-negative ticks to those with at least 2 tracked
   people in the last 3 s (`pose_interact` `n_persons_mean >= 2`). This separates "people are interacting" from
   "people are escalating".
3. **Within-video ramp:** for eligible events with at least 90 s of lead-in, compare the mean over (0, 30] with
   the mean over (60, 120] using a paired Wilcoxon test. Report the fraction of videos where it rises.
4. Breakdown per source kind and footage type (CCTV, movie, YouTube; the curated hold-out once annotated).
5. Single-stream Success@30 at 1 and 2 FA/h against the position null (VA30 section 5.6).

A stream that passes step 2 or step 3 is worth adding to the fused model. One that passes neither is a scene
prior and a shortcut risk.

---

## 4. Cue-by-cue: what to use

Lead-time column: **measured** where section 3 supports it, otherwise "hypothesised" (behavioural sequence,
no timing data). Priority is for 30 s anticipation on VA30 and the curated real-world hold-out.

| Cue | Lead (evidence) | Observable in | Covered now by | Best extractor to add | Value for 30 s | Priority |
|---|---|---|---|---|---|---|
| Confrontation or risk state ("people arguing face to face", standoff) | whole pre-onset window; flat AUROC 0.67-0.70 from 120 s to 0 s (**measured**, teacher) | all footage | Qwen2.5-VL single-frame score (old), SigLIP 2 prompts | **Qwen3-VL-8B multi-frame stage rating + checklist + hidden states** (5.1) | high | 1 |
| Verbal conflict: raised voices, loudness above the video's own past | 10-30 s (+2-3 dB, 81 % up; **measured**); F0 and intensity rise with escalation level (Lefter 2022, **paper**) | bodycam, phone, dashcam; rarely CCTV | `audio_dsp` slopes, AST, CLAP | keep DSP; add `facebook/pe-a-frame-base` prompts ("people arguing loudly", "a man yelling at someone") | medium-high where audio exists | 2 |
| Overlapping speech, turn-taking breakdown | hypothesised 10-60 s | bodycam, phone | none | `pyannote/segmentation-3.0` overlap ratio | medium (unvalidated on street audio) | 4 |
| Threats, insults, commands ("back up", "drop it", slurs) | hypothesised 5-120 s; sparse | bodycam, phone with intelligible speech | Whisper large-v3 + toxicity (`speech_tox`) | add LLM escalation rating of the rolling 60 s transcript (`Qwen2.5-7B-Instruct`, local) | medium (high precision, low recall) | 3 |
| Crowd gathering, encircling, people converging | 10-40 s (+0.5 to +0.9 persons; **measured**) | CCTV (wide view) best; phone partial | `pose_interact` counts, convergence, circle score | keep; VLM checklist item `crowd_gathering_or_circle` | medium | (have) |
| Approach, close proximity, squaring up | hypothesised 5-60 s; 2D facing proxies flat (**measured**) | needs people >= ~60 px | `pose_interact` dyadic features | VLM checklist (`face_to_face`, `squaring_up`); optional ViTPose top-down refinement | medium | 1 (via VLM) |
| Aggressive gesturing, pointing, arm raising | 0-30 s (arm raise +0.035 then +0.062; **measured**) | all, if people visible | `pose_interact` | keep; VLM checklist | medium | (have) |
| Third parties separating or holding back | hypothesised 0-60 s; intervention is common and often de-escalates (Levine 2011, Philpot 2020) | CCTV, phone | `pose_interact` "between" score | VLM checklist `someone_holding_back_or_separating` | medium (two-sided cue) | 1 (via VLM) |
| Pushing, shoving, grabbing clothing | 0-15 s; often the annotated onset itself | all | pose speed and contact | VLM stage 4 | low for 30 s (detection) | (VLM) |
| Weapon displayed | hypothesised 1-30 s | bodycam, phone close-ups; not CCTV (16 px guns) | SigLIP prompts (knife, gun, bat) | VLM checklist `weapon_visible`; YOLOE-26 only for verification | low on CCTV, medium on bodycam | 6 |
| Bystanders raising phones to film | hypothesised 5-60 s; no study found | CCTV, phone | SigLIP `filming_bystanders` | VLM checklist `bystanders_filming` | low-medium | (VLM) |
| Police or security confronting someone, non-compliance | hypothesised 5-120 s (bodycam: suspect resistance predicts time to force, Willits and Makin 2018) | bodycam, CCTV | SigLIP `police_confronting` | VLM checklist + transcript rating ("stop resisting", commands) | medium for bodycam | (VLM) |
| Road rage: vehicle stops, driver gets out, follows | hypothesised 5-60 s (practitioner guidance only) | dashcam, phone | SigLIP `driver_getting_out` | VLM checklist `driver_exits_vehicle_toward_other` | medium for dashcam | (VLM) |
| Facial anger, staring, mutual gaze | unknown; Penchala 2026 face AUC 95 at about 4 s on XD movie clips (shortcut risk) | phone or news close-ups only (about 90 px faces) | none | none recommended (5.2) | very low on CCTV | skip |
| Filmer behaviour (filming starts, shake, zoom) | varies; `cam_shake` up in 79 % (**measured**) | phone only | `motion` | none; keep separate for shortcut audits | shortcut risk | (have) |
| Generic motion and interaction embedding | mostly last 10 s; detection-like | all | `vjepa2_l` | keep; ViT-g only if probes show value | low-medium | 7 |

---

## 5. Family evaluations

### 5.1 Video-LLM / VLM escalation rating

#### 5.1.1 Why this family ranks first

- **Internal.** It is the only control above the position null on VA30 (section 3.1, VA30.md section 7.1), and
  it covers silent CCTV.
- **Published.** Every method with a usable 20 to 60 s horizon is VLM-based:
  - STCH (Wang et al., CVPR 2026: Qwen2-VL-7B + LoRA + entity kinematics + memory bank) reaches window AUC 0.69
    (UCF) and 0.62 (XD) at 60 s. Removing the memory bank drops UCF AUC-60s from 0.692 to 0.601 (**paper**, via
    literature.md).
  - AssistPDA reports a 29.2 s average advance time on UCF and XD, with no false-alarm rate (**paper**).
- **Prompt structure dominates zero-shot quality.**
  - ASK-HINT: plain abstract prompt vs 6 compressed action-centric questions on UCF-Crime: Qwen2.5-VL-7B 74.5 ->
    89.8 AUC, InternVL2.5-8B 76.6 -> 87.4, InternVideo2.5 77.0 -> 89.1. An uncurated pool of 42 questions gives
    67.2 (**paper**).
  - VERA: plain questions on InternVL2-8B give 53.1 and 65.0 AUC; learned guiding questions give 86.6 (**paper**).
- **Plain zero-shot MLLMs are conservative.** They lean towards "normal", with high precision and collapsed
  recall. On ShanghaiTech, class-specific instructions raise peak F1 from 0.09 to 0.64 (arXiv 2603.04727,
  **paper**). Continuous scores from log-probabilities avoid the hard yes/no threshold.
- **Hidden states beat outputs.** HiProbe-VAD (ACM MM 2025) reports that intermediate-layer MLLM states are
  more linearly separable for anomalies than the output layer (**paper**, abstract).

#### 5.1.2 Candidates

| Model (HF id) | Size | Video input | License | Runs in current venv | Video evidence | Verdict |
|---|---|---|---|---|---|---|
| `Qwen/Qwen2.5-VL-7B-Instruct` (local at `models/`) | 8.3B | native video, 28 px per token, 2-frame temporal patches | Apache-2.0 | yes | VideoMME about 65 (older report) | keep for continuity with B-teacher only |
| **`Qwen/Qwen3-VL-8B-Instruct`** | 8.8B (DeepStack, text timestamps such as `<3.0 seconds>`) | native video; patch 16, spatial merge 2, temporal patch 2 = **32 px per token per 2 frames** (config) | Apache-2.0 | **yes** (needs transformers >= 4.57.0; `qwen-vl-utils` 0.0.14 installed) | report: 8B "competitive with Qwen2.5-VL-72B" on video; tables are images | **use now** |
| `Qwen/Qwen3-VL-32B-Instruct` | 33B | same | Apache-2.0 | yes, but needs TP=2 or 4-bit | stronger; about 4x the cost | teacher on subsets |
| `Qwen/Qwen3-VL-30B-A3B-Instruct` | 30B MoE, 3B active | same | Apache-2.0 | slow in HF (MoE); vLLM recommended | VideoMME 73.3 (w/o sub), MVBench 72.0, MLVU 78.9, LVBench 59.2 (Qwen3.5 card) | only with vLLM |
| **`Qwen/Qwen3.5-9B`** | about 10B; hybrid Gated DeltaNet + gated attention; thinking mode by default | native video, early fusion | Apache-2.0 | **no** (transformers main/5.x, vLLM nightly at release) | **VideoMME 78.4 (w/o sub) / 84.5 (w/ sub), MVBench 74.4, MLVU 84.4, LVBench 70.0** (card) | **best 8-10B option; use through vLLM with thinking disabled** |
| `Qwen/Qwen3.5-4B` | 4B | same | Apache-2.0 | no | VideoMME 76.9 (w/o sub), MVBench 71.2 (card) | cheap alternative |
| `Qwen/Qwen3.6-27B` (Apr 2026), `Qwen/Qwen3.6-35B-A3B` | 27B dense / 35B MoE | native video | Apache-2.0 | no | VideoMME (w/ sub) 87.7 / 86.6 (vendor, **secondary**) | offline teacher and pseudo-labeller on subsets |
| `Qwen/Qwen3-Omni-30B-A3B-Instruct` | 30B MoE, 3B active; **audio + video** (`use_audio_in_video`) | native | Apache-2.0 | module present in 4.57.5; card asks for transformers main; 15 s video needs about 79 GB in BF16 HF (card) | WorldSense 54.0 (AV), VoiceBench 85.5 (card) | AV teacher on audio-bearing subsets (yt, XD YouTube, bodycam) |
| `OpenGVLab/InternVL3_5-8B-HF` | 8B (Qwen3 LLM) | frames as images; **256 tokens per 448 px tile per frame**, no temporal merge | Apache-2.0 | yes (>= 4.52.1) | VideoMME 66.0 / 68.6, MVBench 72.1, MLVU 70.2 (paper) | weaker video, about 2x the tokens of Qwen3-VL; skip |
| `DAMO-NLP-SG/VideoLLaMA3-7B` | 8B | video-specialised | Apache-2.0 | `trust_remote_code` | about 64-66 on VideoMME (paper / third party) | superseded by Qwen3-VL and Qwen3.5 |
| LLaVA-Video-7B-Qwen2 | 8B | needs the LLaVA-NeXT repo | Apache-2.0 | no | about 62-63 VideoMME (third party) | superseded |
| `google/gemma-3-12b-it` | 4/12/27B | images only (frames as separate images) | Gemma terms (custom) | yes | no native video | skip |
| `Qwen/Qwen3-VL-Embedding-2B` (Jan 2026) | 2B | video clip -> 2048-d text-aligned embedding (1 fps, up to 64 frames) | Apache-2.0 | yes (Qwen3-VL architecture; needs the repo's embedding wrapper) | MMEB-V2 video overall 61.9 (2B), 67.1 (8B); used as the semantic space in PULS | optional clip-level "prompt score" stream, cheap |
| API-only (Qwen3.5-Omni, Gemini, GPT) | - | - | proprietary | no | - | out of scope (frozen local inference) |

Notes:
- Qwen3.5-9B must be run with `enable_thinking=False`. Thinking mode adds hundreds of tokens per call and
  makes log-probability scoring of the first answer token impossible.
- Qwen3.5 uses linear-attention layers. In HF transformers without the flash-linear-attention kernels these
  fall back to slow torch code (**inference**, not measured). Use vLLM.
- Version note: the Qwen3.5 and Qwen3.6 cards were read on 2026-10-14. A secondary source claims a Qwen3.8-27B
  open release (Aug 2026), which was not verified. Check the Qwen HF organisation before pinning, and pick a
  checkpoint at least two weeks old.

#### 5.1.3 Multi-frame clip vs single frame

- Escalation is temporal: approaching, following, a gesture building, a crowd converging. Single frames cannot
  show direction of change.
- Fine-grained temporal reasoning is still the weakest skill of video LLMs (TemporalBench: GPT-4o about 38.5 %,
  about 30 points below humans; Vinoground: best model about 50 to 56 % against about 90 % for humans, **paper**).
  Many benchmarks reward single-frame cues; Vinoground reports that more sampled frames help up to a point.
- For detection, frame count matters little: ASK-HINT varies 8 to 256 frames and stays at 88.8 to 89.8 AUC.
- For anticipation, **long context matters more than frames per query**: STCH loses 9 AUC points at 60 s
  without its memory bank.

Recommendation:
- Each query gets **8 frames at 1 fps covering [t - 7, t]**, at native resolution (these videos are small), and
  runs every 2 s with sample-and-hold to 1 Hz.
- Long context (60 to 120 s) belongs to the trainable causal temporal head that reads the per-tick VLM
  outputs, not to the VLM prompt. Feeding the previous rating back into the prompt invites error feedback and
  leaks the alarm state.

#### 5.1.4 Structured prompt and scoring (proposed v1)

System: "You analyse surveillance, body-camera, dash-camera or phone footage. Judge only what is visible in
these frames; the last frame is the present. Do not predict the future."

User: the 8 frames, then the following instructions.

```
Rate the most severe interpersonal interaction visible NOW.
stage: 0 calm or no interaction of concern; 1 tension (people watching each other, one person agitated, a
crowd's attention focused); 2 verbal conflict (arguing, shouting, aggressive gestures at a distance);
3 threat display (face to face, chest out, finger in face, following or approaching aggressively, weapon
shown, someone being held back); 4 imminent (closing to arm's length, pushing, shoving, grabbing clothes,
swinging); 5 physical violence in progress (punching, kicking, wrestling, beating).
Then answer each item with 1 (visible) or 0 (not visible), in this exact order, as one 16-digit string:
1 arguing_face_to_face  2 pointing_or_aggressive_gesture  3 approaching_or_following_someone
4 invading_personal_space  5 squaring_up_posture  6 pushing_or_grabbing  7 someone_holding_back_or_separating
8 crowd_gathering_or_circle  9 bystanders_filming_with_phones  10 weapon_visible
11 police_or_security_confronting_person  12 person_refusing_or_resisting  13 driver_exits_vehicle_toward_other
14 person_backing_away_or_fleeing  15 person_on_ground  16 violence_now
Answer as JSON: {"stage": <digit>, "cues": "<16 digits>"}
```

Scoring:
- Prefill the assistant turn with `{"stage": ` so the first generated token is the stage digit.
  - Read the softmax over the digit tokens "0" to "5". Store `E[stage]` and `P(stage >= 2)`, `P(stage >= 3)`,
    `P(stage >= 4)`.
  - In HF use `generate(..., output_scores=True, return_dict_in_generate=True)`. In vLLM use
    `SamplingParams(logprobs=20)` with `continue_final_message=True`.
- Qwen tokenizers split numbers into single digits (verify once), so each checklist digit is one token. Its
  probability of "1" gives 16 continuous cue scores for about 20 decoded tokens.
- **Why this matters:** the old teacher's scores are coarse (0, 0.1, 0.5, 0.8, 0.9). Many ticks tie, and some
  FA/h targets are unreachable on VA30 (VA30.md section 9). Log-probability scores are continuous and calibratable.
- Store per tick: stage probabilities (6), cue probabilities (16), the raw JSON, and `valid`. Follow the io.py
  contract, e.g. extractor `vlm_escalation_q3vl8b`, with `window_sec` 8 and `meta` holding the prompt hash.

Hidden states (same HF prefill):
- Mean-pool the visual-token states at about 1/2 and 3/4 depth (Qwen3-VL-8B has 36 layers: use 18 and 27).
- Also keep the last prompt token's state at those layers. LayerNorm each vector before pooling: Qwen2.5-VL
  states have large-norm outlier channels (see the vlm_and_llm doc).
- About 4 x 4096 floats per tick, so store as float16 (about 32 KB per tick, about 6 GB per 100 h at 0.5 Hz).
- vLLM does not expose per-layer states. If the ratings run in vLLM, do the hidden-state pass as a separate HF
  prefill-only job.

Causality: frames only from [t - 7, t]. Every tick's query is independent, so the io.py truncation check
(`check_causal`) applies directly. Use fixed-size batches padded by repetition, as the other GPU extractors do.

Gating (saves cost on empty CCTV): skip the query when `pose_interact` shows fewer than 1 tracked person over
[t - 7, t]. Store stage 0 with a `gated` flag. This is causal, but it ties the VLM stream to pose. Ablate it.

#### 5.1.5 Throughput and cost (est., one GPU, per 100 h of video)

Token math for Qwen3-VL-8B (config: patch 16, spatial merge 2, temporal patch 2):

| Footage | Frame size used | Tokens per 2 frames | 8-frame query (visual + about 350 text + about 20 out) |
|---|---|---|---|
| 640x360 (UBI, XD YouTube, yt) | 640x352 | 20 x 11 = 220 | about 880 visual, about 1,250 total |
| 320x240 (UCF) | 320x224 | 10 x 7 = 70 | about 280 visual, about 650 total |

The 8.2B LLM prefill is about 2 x 8.2e9 x 1,250, or about 20 TFLOP, so about 0.15 s at 40 % of GPU BF16 peak.
The vision tower adds about 0.03 s. Decoding about 20 tokens costs about 0.02 s per query when vLLM batches 64
queries.

| Path | Queries/s (est.) | 100 h at one query per 2 s (180k queries) |
|---|---|---|
| vLLM, Qwen3-VL-8B or Qwen3.5-9B | about 5 | **about 10 GPU-h** |
| HF transformers 4.57.5, batched generate (batch 8, SDPA, no paged KV) | about 1 to 1.5 | about 35 to 50 GPU-h |
| HF prefill-only hidden-state pass | about 4 | about 12 GPU-h |
| vLLM, Qwen3-VL-32B or Qwen3.6-27B (TP=2 or 4-bit) | about 1.2 to 1.5 | about 35 to 45 GPU-h |
| vLLM, Qwen3-VL-30B-A3B or Qwen3.6-35B-A3B (3B active; 60-70 GB weights) | about 8 | about 6 GPU-h |
| vLLM, InternVL3.5-8B (256 tokens per frame) | about 2.5 | about 20 GPU-h |
| vLLM, Qwen3-Omni-30B-A3B thinker, 8 s of audio + 8 frames | about 2 to 4 | about 15 to 25 GPU-h (audio subsets only) |
| HF, Qwen3-VL-Embedding-2B clip embedding | about 30 | about 2 GPU-h |
| Legacy reference (**measured docstring**): Qwen2.5-VL-7B teacher at native resolution, about 300 output tokens, second forward pass | about 0.17 | about 300 GPU-h at the same rate |

The ratings matter most on the violent videos and on clean negatives with people. With pose gating, the
239 h extraction scope (violent + YouTube + 5-minute negatives) is about 15 to 25 GPU-h in vLLM, or about
3 to 5 days of one GPU in HF (**est.**).

#### 5.1.6 Can vLLM be installed here?

| Check | Finding |
|---|---|
| Driver | the 580 driver branch supports CUDA 13.0 user space. |
| vLLM wheels | vLLM 0.30.x / 0.31 (Sep to Oct 2026) ship CUDA 13.0 wheels on PyPI and CUDA 12.9 wheels as release assets (**secondary**). They pin their own torch (much newer than 2.5.1), so they **cannot share the repo venv**. |
| Disk | a vLLM venv is about 10-15 GB. |
| Python | 3.10 in the repo venv. Recent vLLM supports 3.10+, but confirm in the release notes of the chosen version. |
| **Blocker** | `nvidia-smi` failed: "Failed to initialize NVML: Driver/library version mismatch" (user library newer than the kernel module). torch CUDA works (README section 7), but vLLM's CUDA platform detection calls NVML and can fail with "Failed to infer device type" (vllm issue #13946). The standard fix is reloading the nvidia kernel modules or rebooting. That needs administrator rights and stops all GPU jobs. |

Plan:
1. Create `.venv-vllm` with `python -m venv`.
2. `pip install vllm==<a release at least 2 weeks old>`.
3. Run a 5-minute smoke test (`vllm.LLM("Qwen/Qwen3-VL-8B-Instruct")` on one clip).

If it fails at platform detection, request the module reload and keep the HF path meanwhile. SGLang and LMDeploy
would probably hit the same NVML problem (**inference**). The same venv can hold transformers 5.x for
PE-A-Frame (5.5) and Qwen3.5.

#### 5.1.7 Risks

- **Scene-type confound.** The teacher's separation may partly be "people present". Use the people-present
  screen (3.3 step 2).
- **Overlays and provenance.** Burned-in captions and news tickers sway VLMs. Text overlays are a documented
  hallucination source (arXiv 2604.17375, title only). Keep the SigLIP `large_text_captions` and `news_broadcast`
  prompts for shortcut audits.
- **Prompt overfitting.** Freeze the prompt (record its hash) before scoring the test folds. Prompt tuning on
  VA30 validation folds only.
- **Calibration drift between models.** Never mix rating files from different models or prompts in one column.
- **Thinking models.** Qwen3.5 and Qwen3.6 default to thinking. Always disable it for extraction.

### 5.2 Face: detection, expression, gaze

| Task | Best candidates | Accuracy (paper) | License | On our footage |
|---|---|---|---|---|
| Face detection | SCRFD-10G-KPS / 34G (insightface) | WIDER Hard 82.8 / 85.3 AP at VGA. Hard is dominated by small faces: 51.9 % of WIDER validation faces are under 16 px. | code MIT; **insightface model weights are for non-commercial research** | finds 10-14 px faces on UCF only sometimes, and the crops are useless downstream |
| | YOLO5Face / YOLOv8-face | YOLO5Face-x6 WIDER Hard 86.6 | GPL / AGPL | same limit |
| Expression | HSEmotion / EmotiEffLib `enet_b2_8` | AffectNet 8-class 63.0 %, 7-class 66.3 % at 224 px; users report about 61.6 % | Apache-2.0 code | 7-class FER at 12-14 px is about 50-75 % even for dedicated low-resolution networks on curated faces (GME-Net) |
| | POSTER++, DDAMFN, EmoNet | similar AffectNet range | various | same |
| Head pose | 6DRepNet | AFLW2000 MAE 3.97, BIWI 3.47 (on 640x480 sources) | MIT | no published low-resolution evaluation |
| Gaze target | Gaze-LLE (CVPR 2025; frozen DINOv2 + 2.9M decoder) | state of the art on GazeFollow; VideoAttentionTarget AUC 0.937 cross-dataset | code repository | needs head boxes; one person at a time |
| Mutual gaze | LAEO-Net++ (head tracks, 64x64 crops) | AP 86.7 UCO-LAEO, 68.7 AVA-LAEO (10-frame) | code repository | 64 px head crops vs our 10-22 px heads |
| Body orientation (proxy for facing) | MEBOW (COCO-MEBOW, about 130k labelled bodies) | designed for cases where pose is hard, e.g. low resolution | code repository | the most plausible facing cue for CCTV; still untested here |

**Verdict: skip faces for VA30.**
- The only paper that found faces strongest (Penchala 2026: face-only AUC 95.2) used XD movie and news clips
  with a horizon of about 4 s. That is the setting where provenance and actor close-ups are shortcuts.
- If facing or staring is wanted, the cheapest robust route is the VLM checklist (`face_to_face`,
  `squaring_up_posture`) plus the existing 2D facing proxies. Option: MEBOW-style body orientation on person
  crops for people at least 80 px tall.
- An optional curated-set experiment, only for phone or news footage with faces of 60 px or more:
  SCRFD-10G -> EmotiEffLib `enet_b0_8_va_mtl` (valence / arousal). Cost: about 1-2 GPU-h per 100 h at 2 fps
  (**est.**).

### 5.3 Open-vocabulary detection (weapons, raised phones, police, context)

| Model | Access | Accuracy (paper) | Speed | Notes |
|---|---|---|---|---|
| **YOLOE-26 L/X** (Ultralytics, 2026) | `ultralytics` 8.3.228 already installed (`YOLOE` imports) | LVIS minival 40.6 AP (X, text prompts); YOLOE-26L 36.8 (secondary) | about 6 ms on a T4 for L (secondary), so cheap on an GPU | AGPL-3.0; text, visual or prompt-free modes |
| YOLOE-11 / YOLOE-v8 (ICCV 2025) | same | YOLOE-v8-S +3.5 AP over YOLO-Worldv2-S on LVIS at 1.4x speed | real-time | superseded by YOLOE-26 |
| YOLO-World v2 | ultralytics | LVIS 35.4 AP (v1-L, 52 FPS V100) | real-time | "significantly inferior on small objects" in street-scene tests |
| Grounding DINO (T/B), MM-Grounding-DINO | HF transformers (`grounding_dino`, `mm_grounding_dino` present in 4.57.5) | best accuracy of the open models in street-scene tests | about 3 FPS in one comparison (secondary) | Apache-2.0 |
| LLMDet (CVPR 2025) | mmdet code | LVIS minival AP_r 37.3 (Swin-T) to 45.1 (Swin-L) | slow | heavy dependencies |
| OWLv2 (`google/owlv2-base-patch16-ensemble`) | HF transformers | strong long-tail | about 8 FPS (secondary) | Apache-2.0 |
| Florence-2 | HF (`florence2` present) | captioning plus grounding | generation-based, slow | MIT |
| Grounding DINO 1.5 Pro, DINO-X | **API only** | - | - | not runnable locally |

What the evidence says for weapons on CCTV (CCTV-Gun benchmark):
- Handguns average **16 px in UCF-Crime frames** (25 px in MGD, 47 px in 1080p USRT).
- Trained and tested on UCF: AP50 43.4 to 57.4.
- Trained on the other two datasets and tested on UCF: **AP50 3.7 to 11.7**.
- Zero-shot open-vocabulary detectors will not do better on 320x240 CCTV.

Verdict: **low value for 30 s on VA30.**
- Weapon display is usually seconds before the onset.
- Small objects defeat the detectors, and a false knife on CCTV is a false alarm generator.
- Let the VLM checklist report `weapon_visible` and `bystanders_filming_with_phones`.
- If a detector stream is wanted for bodycam or phone footage (raised phones, police uniforms, bats or bottles
  in hand), run YOLOE-26-L at 2 fps with text prompts. Keep only detections that persist across 3 or more
  frames of the same track, and store max confidence per tick. Cost: about 0.5 GPU-h per 100 h (**est.**).
- MM-Grounding-DINO-T at 2 fps would cost about 10-20 GPU-h per 100 h (**est.**) for little expected gain.

### 5.4 Skeleton interaction models and pose estimators

Skeleton action recognisers (PoseC3D, CTR-GCN, InfoGCN, ST-GCN++):

**Relevant NTU-120 mutual classes.** Of the 26 mutual classes, a handful look like precursors: point finger
(A54), walking towards (A59), follow (A116) and grab stuff (A109). The rest of the aggressive ones are the
violence itself: punch or slap (A50), kick (A51), push (A52), hit with object (A106), wield knife (A107).

**Accuracy.** PoseC3D on NTU-120 2D HRNet keypoints reaches 85.9 % (X-Sub, joint) and 86.7 % (two-stream)
(pyskl model zoo). Two-person interaction subsets reach about 91-97 % (**paper**). All of this is lab RGB-D
footage: frontal Kinect views, actors, clean skeletons.

**Domain gap evidence.**
- Ganesh 2026: pose representations never beat 7-d box geometry for pre-onset separability.
- A YOLO11-pose pushing detector falls from precision 0.98 (frontal, controlled) to 0.72 on real high-angle
  surveillance (arXiv 2605.02659).
- Pure-skeleton fight detectors top out around 90 % on RWF-2000, a clip-level detection task.

**Engineering.** pyskl is "not maintained" (last change 2023) and needs an old mmcv. 2D COCO-17 NTU
checkpoints exist only through pyskl. CTR-GCN and InfoGCN official checkpoints use 3D Kinect skeletons.

**Verdict: skip.** The precursor classes (pointing, approaching, following) are already approximated by the
engineered `pose_interact` features and by the VLM checklist. If a learned skeleton stream is wanted later,
train a small ST-GCN++ or CTR-GCN in plain PyTorch on our own tracked YOLO poses with VA30 labels (468 events
means overfitting risk).

Pose estimator upgrades (optional):
- **Top-down ViTPose** (HF transformers `vitpose`, Apache-2.0; e.g. `usyd-community/vitpose-plus-base`) on
  the existing BoT-SORT boxes, for people under about 120 px tall (UCF, UBI). Top-down crops are upsampled to
  256x192, which helps small people more than the one-stage model at 640 px input (**inference**; measure
  keypoint flicker before and after). Cost: about 1-2 GPU-h per 100 h at 5 fps (**est.**).
- RTMO-l (rtmlib, Apache-2.0) is the crowd-robust one-stage alternative ([backbones.md](backbones.md)). It only
  matters if Ultralytics' AGPL license becomes a problem.

### 5.5 Audio and speech

| Need | Current | Best candidate | Evidence | Cost per 100 h (one GPU) | Verdict |
|---|---|---|---|---|---|
| Sound events: shouting, yelling, crowd | AST 527-class logits (mAP 45.9) | EAT-large AS2M (mAP about 48.6-49.5); BEATs or OpenBEATs; Dasheng-1.2B (best X-ARES average) | +3 mAP over AST at best; shouting rises only in the last 10 s (3.2) | about 0.5-1 GPU-h | low gain; skip |
| **Open-vocabulary "argument" detection, 25 Hz** | CLAP prompts (5 s clips) | **`facebook/pe-a-frame-base`** (PE-A-Frame: AudioSet-Strong AUROC 0.96, DESED 0.98, UrbanSED 0.89; one embedding every 40 ms; free-text queries) | the only model that scores "people arguing", "a heated argument" or "a man yelling at someone" frame by frame | about 0.3-0.5 GPU-h (**est.**) | **add** (needs transformers 5.x: `PeAudioFrameLevelModel`, 48 kHz mono) |
| Overlapping speech, turn-taking, speaker count | Silero VAD only | `pyannote/segmentation-3.0` (VAD + overlap; pyannote.audio 3.3.x, MIT, gated) or `pyannote/speaker-diarization-community-1` (CC-BY-4.0; pyannote.audio 4.x; better speaker counting, same segmentation as 3.1) | escalation literature (overlap, interruptions); untested on street audio | segmentation: under 0.2 GPU-h or a few CPU-h; community-1: about 2-3 GPU-h (**est.**) | add segmentation-3.0 overlap ratio per second |
| Vocal arousal and anger | audEERING A/V/D + emotion2vec+ large | WavLM MSP-dim (non-commercial) | anger flat and arousal barely moves before onset (3.2) | - | **no further investment** |
| Loudness, F0, slopes | `audio_dsp` (34 columns) | - | the only audio cue up 30 s out (81 %) | CPU | keep; feed 15-30 s slopes to the head |
| ASR | Whisper large-v3 (`speech_tox`, measured 19.7x realtime, about 5-13 GPU-h per 100 h) | `nvidia/parakeet-tdt-0.6b-v3` (RTFx 3,333 vs 146 for Whisper large-v3; English average WER 6.3 vs 7.4; 25 European languages; claimed robust at -5 dB SNR, **secondary**) | speed only; Whisper covers more languages | negligible | keep Whisper; Parakeet only if real-time is needed |
| Toxicity from text | detoxify unbiased + XLM-R toxicity | - | - | - | keep |
| **Verbal escalation rating** | none | LLM over the rolling 60 s transcript: `Qwen2.5-7B-Instruct` (cached) or `Qwen/Qwen3-8B` (non-thinking) -> stage 0-4 + flags (insult, threat, command, de-escalation) as log-probabilities | DeepSafety (ASR text + vocal features) F1 0.85; Lefter 2022 3-level escalation UAR 63.8 % from speech | about 1.5-3 GPU-h (**est.**: about 15k queries at about 0.4 s) | **add** |
| Audio-only toxicity (no ASR) | none | MuTox (SONAR speech encoder + MLP; AUC about 0.80 supervised, 0.77 zero-shot, **secondary**) | multilingual; robust where ASR fails | under 1 GPU-h | low priority (fairseq2 dependency) |
| Audio LLM description | none | `Qwen/Qwen3-Omni-30B-A3B-Captioner` (Apache-2.0); Audio Flamingo 3 (MMAU about 72-73; research-only license) | - | high | teacher or label QA only |

Audio caveats:
- XD audio is often a movie soundtrack or overdubbed music. AST `Music` rises during fights, a provenance
  shortcut. Report audio results on the curated real-world set and on `xd_youtube` separately from `xd_movie`.
- CCTV (UCF, UBI, most real cameras) is silent. Audio features must carry a presence mask (already the io.py
  convention) and modality dropout in the head.

### 5.6 Temporal video models and world models

| Approach | Evidence on horizon | Cost per 100 h | Verdict |
|---|---|---|---|
| V-JEPA 2 ViT-L (`facebook/vjepa2-vitl-fpc64-256`, have) | best frozen action anticipation, but EK100 is a **1 s** horizon (R@5 32.7 L, 39.7 g384) | **measured** 1.9 GPU-h idle (53.6 video-s/s), 3.8 loaded | keep as the motion stream |
| V-JEPA 2 ViT-g (`facebook/vjepa2-vitg-fpc64-256`) | anticipation scales with size on EK100 (32.7 -> 38.0) | about 3x L: about 6-10 GPU-h (**est.**) | only if the L probe shows a far-window gain after the screen in 3.3 |
| V-JEPA 2.1 (dense features, 384 px) | no anticipation evidence; official weights via torch.hub; HF ports `apiantonio/vjepa2.1-vit-large-384` (`trust_remote_code`) | similar to L at 384 px (more tokens) | skip for now |
| Predictor "surprise" (latent prediction error) | PULS (V-JEPA 2 -> Qwen3-VL-Embedding space): advantage peaks at **T-0.5 s** (+8.9 pp); a JEPA driving-surprise study: AP 0.512 vs 0.489 for a random encoder | about 1-2 GPU-h extra | skip as an extractor |
| Future-feature regression (predict the frozen embedding at t + 5 / t + 10 s) | recommended auxiliary objective (literature.md 9.1); uses unlabelled footage | training-time only | **use in the head**, not as a feature |
| VideoMAE (v1 base, local) / VideoMAE V2 (CC-BY-NC) | frozen motion features weak (SSv2 frozen 56.1 vs 77.3 for V-JEPA 2-g384) | - | skip |
| InternVideo2-1B | strong balanced encoder (V-JEPA 2 Table 4 average 87.0) | about V-JEPA-g | second motion stream only if needed |
| Qwen3-VL-Embedding-2B clip embeddings | text-aligned video embeddings; the semantic space in PULS (chunk AUROC 0.899 UCF, 0.816 XD) | about 2 GPU-h (**est.**) | optional; a multi-frame alternative to SigLIP 2 prompt scores |

---

## 6. Cost summary: per 100 h of video on one GPU

| Stream | Model | Status | GPU-h per 100 h | Source |
|---|---|---|---|---|
| VLM stage + checklist ratings (0.5 Hz, 8 frames) | Qwen3-VL-8B (HF) / Qwen3.5-9B (vLLM) | **new, priority 1** | 35-50 (HF) / about 10 (vLLM) | est. |
| VLM hidden states (prefill only) | Qwen3-VL-8B (HF) | new, priority 1 | about 12 | est. |
| Transcript escalation rating | Qwen2.5-7B-Instruct (HF) | new, priority 3 | about 1.5-3 | est. |
| Open-vocabulary audio events | PE-A-Frame base | new, priority 2 | about 0.3-0.5 | est. |
| Overlap ratio | pyannote segmentation-3.0 | new, priority 4 | under 0.2 (or CPU) | est. |
| AV teacher (subsets with audio) | Qwen3-Omni-30B-A3B thinker (vLLM) | optional | about 15-25 | est. |
| Clip embeddings | Qwen3-VL-Embedding-2B | optional | about 2 | est. |
| Raised phones / police / weapons | YOLOE-26-L, 2 fps | optional, low | about 0.5 | est. |
| Pose refinement | ViTPose-plus-base top-down | optional | about 1-2 | est. |
| V-JEPA 2-g | `facebook/vjepa2-vitg-fpc64-256` | conditional | about 6-10 | est. |
| (have) V-JEPA 2-L | | built | 1.9-3.8 | measured |
| (have) SigLIP 2 + prompts | | built | about 0.9-1.9 | measured |
| (have) AST + CLAP + voice | | built | about 1.5 | measured |
| (have) Whisper + toxicity | | built | about 5-13 | measured |
| (have) Pose (YOLO11x-pose + BoT-SORT) | | built | about 1.5 h wall on 2 GPUs (CPU-bound) | measured |
| Face (SCRFD + FER), skeleton classifiers, MuTox, V-JEPA 2.1, predictor surprise | | **skip** | - | - |

---

## 7. Recommended plan (ordered)

1. **Screen what already exists (CPU, hours).**
   - Once the bulk features land, run the protocol in 3.3 on `siglip2_prompts`, `pose_interact`, `audio_dsp`,
     `ast_events`, `speech_tox` and the teacher.
   - Record per-column time-matched far AUROC, people-present AUROC and ramp fraction in a table under
     `benchmark/results/cue_screen/`.
   - This tells the temporal-model work which columns carry state and which carry a ramp.
2. **VLM escalation extractor v1 (HF, existing venv).** Implement `anticipation/features/vlm_escalation.py`
   following the io.py contract:
   - model `Qwen/Qwen3-VL-8B-Instruct`, BF16, SDPA;
   - 8 frames at 1 fps ending at t, native resolution, every 2 s with sample-and-hold;
   - prompt as in 5.1.4 with a stored hash; log-probability stage and cue scores;
   - mid-layer pooled hidden states in the same pass; fixed-size padded batches; a `check_causal` test.
3. **Pilot it on the teacher subset first** (1,031 videos, 37.0 h, about 13-19 GPU-h in HF, **est.**) and compare it with
   `b_teacher` on VA30. Go on to the full scope only if it beats `b_teacher` on time-matched far AUROC (CI) **and**
   shows a ramp or a people-present separation that the teacher lacks.
4. **Separate modern venv** (`.venv-vllm`: vLLM at a release at least 2 weeks old, transformers 5.x).
   - Smoke-test it first. If NVML blocks it, request the module reload.
   - Then run Qwen3.5-9B (thinking off) for the full scope (about 15-25 GPU-h with gating, **est.**),
     PE-A-Frame base, and optionally Qwen3-Omni on the audio-bearing subsets.
5. **Speech and audio additions:** the transcript escalation rating (local Qwen2.5-7B-Instruct) and the
   pyannote segmentation-3.0 overlap ratio. Install pyannote.audio 3.3.x in the repo venv only after checking
   that it does not upgrade torch (`pip install --dry-run`). Otherwise use the modern venv.
6. **Do not build:** face or FER, gaze, skeleton action classifiers, predictor-surprise features, or V-JEPA 2.1.
   Revisit faces only on the curated real-world set after human onsets exist.
7. **Train-time use of the new streams:**
   - per-stream LayerNorm, presence masks and modality dropout;
   - explicit 5 / 15 / 30 s slopes of the VLM stage probabilities, loudness and crowd counts;
   - a future-feature regression auxiliary loss;
   - always score against `b_time_source_fit`, the position null and `b_teacher` (VA30).

---

## 8. Risks and open questions

- **Is the 30 s signal escalation or scene type?** Section 3.1 cannot tell. The people-present screen and the
  curated real-world hold-out (once annotated) are the tests. If every stream is flat in time and separates only
  "people vs no people", 30 s anticipation at low FA/h is not achievable with these data, and the write-up
  should say so.
- **HIVAU onsets mark the walk-up in some videos** (VA30.md 2.5). That moves part of the escalation phase inside
  "violence", which shortens measurable lead. Treat the `onset:official` breakdown as co-primary.
- **VLM prompt leakage.** Any prompt tuning must use validation folds only. Store the prompt hash in `meta`.
- **Licensing.** Ultralytics (pose, YOLOE) is AGPL-3.0, insightface weights are non-commercial, and Audio
  Flamingo 3 and WavLM MSP-dim are research-only. All are fine for a capstone; none suits a commercial product.
- **Unverified items:** the Qwen3.8 release; Qwen3.6 VideoMME numbers (vendor, secondary); Parakeet noise
  robustness (secondary); exact YOLOE-26 checkpoint file names. Check before use.

---

## 9. Sources

VLMs and video-LLMs
- Qwen3-VL technical report, arXiv 2511.21631: https://arxiv.org/abs/2511.21631 ; model card https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct ; config https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/raw/main/config.json ; README (transformers >= 4.57.0, vLLM >= 0.11.0) https://github.com/QwenLM/Qwen3-VL ; vLLM recipe https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen3-VL.html
- Qwen3.5-9B card (video table, thinking default): https://huggingface.co/Qwen/Qwen3.5-9B ; Qwen3.5 blog https://qwen.ai/blog?hl=en-US&id=qwen3.5 ; Qwen3.6-27B blog https://qwen.ai/blog?id=qwen3.6-27b
- Qwen3-Omni-30B-A3B-Instruct card (memory table, AV benchmarks): https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct ; Captioner https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner ; vLLM supported models https://docs.vllm.ai/en/latest/models/supported_models/
- Qwen3-VL-Embedding: https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B , arXiv 2601.04720 https://arxiv.org/abs/2601.04720
- InternVL3.5: arXiv 2508.18265 https://arxiv.org/html/2508.18265v1 ; HF card https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF
- VideoLLaMA 3: arXiv 2501.13106 https://arxiv.org/abs/2501.13106 ; Video Panels third-party comparison https://arxiv.org/html/2509.23724v1
- ASK-HINT (fine-grained prompting for VLM VAD, WACV 2026): https://arxiv.org/abs/2510.02155
- VERA (CVPR 2025): https://arxiv.org/abs/2412.01095
- HiProbe-VAD (ACM MM 2025): https://arxiv.org/abs/2507.17394
- "Are Multimodal LLMs Ready for Surveillance?": https://arxiv.org/abs/2603.04727
- Cerberus (cascaded VLM VAD, 57.7 fps on L40S): https://arxiv.org/abs/2510.16290
- AssistPDA: https://arxiv.org/abs/2503.21904 ; STCH (Wang et al., CVPR 2026): https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Streaming_Video_Crime_Anticipation_with_Spatio-Temporal_Causal_Reasoning_CVPR_2026_paper.html
- TemporalBench: https://arxiv.org/abs/2410.10818 ; Vinoground: https://arxiv.org/abs/2410.02763
- VARS (danger rating 0-10): https://arxiv.org/abs/2410.19642 ; text-overlay hallucination: https://arxiv.org/abs/2604.17375
- vLLM releases https://github.com/vllm-project/vllm/releases ; GPU install https://docs.vllm.ai/en/stable/getting_started/installation/gpu/ ; device inference failure https://github.com/vllm-project/vllm/issues/13946 ; NVML mismatch fix https://forums.developer.nvidia.com/t/how-to-fix-nvml-driver-library-version-mismatch-without-rebooting/269998

Face and gaze
- SCRFD: https://arxiv.org/abs/2105.04714 ; insightface model zoo https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md ; YOLO5Face https://arxiv.org/abs/2105.12931
- HSEmotion: https://www.sciencedirect.com/science/article/pii/S2665963822001178 ; low-resolution FER (GME-Net) https://arxiv.org/abs/2511.05938
- Gaze-LLE: https://arxiv.org/abs/2412.09586 ; LAEO-Net++: https://arxiv.org/abs/2101.02136 ; 6DRepNet: https://arxiv.org/abs/2202.12555 ; MEBOW: https://arxiv.org/abs/2011.13688

Open-vocabulary detection and weapons
- YOLOE: https://arxiv.org/abs/2503.07465 ; YOLO26 / YOLOE-26: https://docs.ultralytics.com/models/yolo26 , https://arxiv.org/abs/2606.03748 ; YOLO-World: https://arxiv.org/abs/2401.17270
- Grounding DINO 1.5: https://arxiv.org/abs/2405.10300 ; LLMDet: https://github.com/iSEE-Laboratory/LLMDet ; open-vocabulary detectors in unusual street scenes: https://arxiv.org/abs/2408.11221
- CCTV-Gun benchmark: https://arxiv.org/abs/2303.10703 ; ACF armed CCTV dataset: https://pmc.ncbi.nlm.nih.gov/articles/PMC9572610/

Skeleton
- pyskl: https://github.com/kennymckormick/pyskl ; PoseC3D model zoo https://github.com/kennymckormick/pyskl/blob/main/configs/posec3d/README.md ; PoseC3D paper https://arxiv.org/abs/2104.13586
- Moderate violence (pushing) with YOLO11-pose: https://arxiv.org/abs/2605.02659
- Ganesh 2026, pre-event separability: https://arxiv.org/abs/2608.27879 (summarised in [literature.md](literature.md) 2.3)

Audio and speech
- PE-A-Frame: https://huggingface.co/facebook/pe-a-frame-large ; PE README tables https://github.com/facebookresearch/perception_models/blob/main/apps/pe/README.md ; transformers PE Audio docs https://huggingface.co/docs/transformers/model_doc/pe_audio ; PE-AV paper https://arxiv.org/abs/2512.19687
- EAT: https://arxiv.org/abs/2401.03497 ; OpenBEATs: https://arxiv.org/abs/2507.14129
- pyannote community-1: https://huggingface.co/pyannote/speaker-diarization-community-1 ; pyannote.audio https://pypi.org/project/pyannote.audio/
- emotion2vec+: https://huggingface.co/emotion2vec/emotion2vec_plus_large
- Canary-1B-v2 and Parakeet-TDT-0.6B-v3: https://arxiv.org/abs/2509.14128 ; Open ASR Leaderboard: https://arxiv.org/abs/2510.06961
- MuTox: https://arxiv.org/abs/2401.05060 ; Audio Flamingo 3: https://arxiv.org/abs/2507.08128
- Lefter et al. 2022, escalating negative interactions from speech: https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2022.749804/full

Temporal and world models
- V-JEPA 2: https://arxiv.org/abs/2506.09985 ; V-JEPA 2.1: https://arxiv.org/abs/2603.14482 ; HF port https://huggingface.co/apiantonio/vjepa2.1-vit-large-384
- PULS / Latent Clarity: https://arxiv.org/abs/2607.03558

Behavioural studies of conflicts on video
- Levine, Taylor and Best 2011: https://doi.org/10.1177/0956797611398495
- Philpot et al. 2020 (intervention is the norm): https://eprints.lancs.ac.uk/id/eprint/134891/
- Ejbye-Ernst, Lindegaard and Bernasco 2022 (46 Amsterdam CCTV conflicts, no sound): https://pmc.ncbi.nlm.nih.gov/articles/PMC9682270
- Willits and Makin 2018 (bodycam, time to force): https://journals.sagepub.com/doi/abs/10.1177/0022427817701257

Internal
- [../VA30.md](../VA30.md) section 7.1, `benchmark/results/b_teacher/report.md`, [../FEATURES.md](../FEATURES.md)
  (onset sanity tables, throughput), `benchmark/videos.csv`, `FEATURE_ROOT/pose_raw/`, docs/codebase/vlm_and_llm.md
  section 9 (legacy teacher cost).

---

## Appendix A: teacher lead-time profile script (CPU, about 1 min)

```python
# Lead-time profile of the zero-shot Qwen2.5-VL teacher (b_teacher) on VA30.
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from anticipation.eval.va30 import Benchmark
from sklearn.metrics import roc_auc_score
bench = Benchmark.load(); ids = set(bench.eval_ids())
s = pd.read_parquet("benchmark/results/b_teacher/scores.parquet")
bins = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 45), (45, 60), (60, 90), (90, 120)]
pos, neg = {}, []
for vid, g in s.groupby("video_id"):
    if vid not in ids: continue
    L = bench.labels(vid); t = np.asarray(L.t_sec)
    sc = pd.Series(g.score.values, index=g.t_sec.values).reindex(t).ffill().fillna(0).values
    m = ~np.asarray(L.masked); role = bench.videos.loc[vid, "role"]
    if role == "clean_negative": neg.append(sc[m])
    elif role == "violent":
        pv = np.asarray(L.pre_violence) & m; tte = np.asarray(L.tte)
        for lo, hi in bins:
            k = pv & (tte > lo) & (tte <= hi)
            if k.any(): pos.setdefault((lo, hi), []).append(sc[k])
neg = np.concatenate(neg)
for b in bins:
    p = np.concatenate(pos[b])
    print(b, len(p), round(p.mean(), 3),
          round(roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(neg))], np.r_[p, neg]), 3))
```

The within-video ramp: for violent videos with at least 30 pre-violence ticks in (60, 120] and at least 20 in
(0, 30], compare the per-video means. Use the same score on the people-present negatives once `pose_interact`
covers the clean negatives.
