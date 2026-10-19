# Literature Review: Violence / Anomaly Anticipation and Early Detection

Last updated: 2026-10-14. Scope: what the published literature says about predicting violence (or crime,
aggression, abnormal behavior) **before onset** in real-world video, which training objectives work for
long-horizon anticipation, how to evaluate honestly, and what accuracy is realistic for a 30 s+ horizon.

This document is meant as a foundation for future agents working on this repo. Numbers marked
"(paper)" were read from the paper text or its official tables; numbers marked "(secondary)" come from a
citing paper or a summary and should be re-checked before being quoted in the capstone write-up.

---

## 0. Key takeaways (read this first)

1. **Almost nobody reports reliable violence anticipation at 30 s+.** Most "anticipation" benchmarks use
   horizons of 0.5-8 s. The only peer-reviewed video work evaluating 20 s and 60 s horizons on
   UCF-Crime / XD-Violence (Wang et al., CVPR 2026) reaches **window AUC 0.69 at 60 s on UCF-Crime and
   0.62 at 60 s on XD-Violence (cross-domain)**. A strong weakly-supervised detector (UR-DMU) used as an
   anticipator scores about **0.54 / 0.51** at 60 s, i.e. near chance.
2. **Short-horizon numbers are also modest.** Semi-supervised anomaly anticipation on NWPU Campus: frame AUC
   about 0.69-0.71 at 0.5-3 s (humans: 0.90 at 3 s). Abnormal-behavior forecasting on AHB-F: Top-1 mAP about
   67 % (1-3 s) and 64 % (4-8 s).
3. **High reported pre-onset accuracy is suspicious.** A 2026 diagnostic study showed that "pre-onset"
   separability on UCF-Crime / XD-Violence is partly driven by **provenance artifacts** (title cards,
   platform watermarks, clip length) because violent videos come from YouTube/news while normal videos come
   from CCTV. This is directly relevant to this repo, which mines YouTube/phone footage.
4. **What is realistically predictable 30 s ahead is escalation, not the punch.** Literature on aggression
   in speech (raised F0 / intensity), interaction geometry (approach, encircling, crowding) and pre-crime
   behavior supports predicting an *elevated-risk state*. Treat the target as "violence onset within the next
   H seconds" with a calibrated risk, not as a hard binary.
5. **Use a censored, multi-horizon (discrete-time hazard / survival) objective**, with masked post-onset
   frames, plus an auxiliary "violence now" detection head, plus early-tolerant weighting borrowed from
   accident anticipation (exponential loss) and early action prediction (Aliakbarian et al.).
6. **Evaluate at the event level with lead time and false alarms per hour** (seizure-prediction and
   accident-anticipation style), with per-video (ideally per-source) splits, cluster-bootstrap CIs,
   length-matched negatives, and a random-predictor baseline. Window-level accuracy on balanced clips is
   not an honest headline metric.
7. **Fusion:** mid-level (token) fusion of a strong frozen visual encoder (whole-frame context beats person
   crops) + audio gives consistent but modest gains; pose kinematics help mainly as a supplementary channel
   (or as a training-only modality, as in pi-VAD).

---

## 1. Terminology

| Term | Meaning | Typical horizon in literature |
|---|---|---|
| Detection (VAD / violence detection) | Is violence happening in this frame/snippet? | 0 s |
| Online / streaming detection | Same, but causal (no future frames) | 0 s |
| Early detection / early recognition | Recognize an event from its first part (partially observed event) | event already started |
| Anticipation / forecasting / prediction | Predict that an event will start in (t, t+H] using only frames up to t, all of which are normal | 0.5-8 s typical; 20-60 s rare |
| Lead time / time-to-event (TTE) | t_onset - t_alarm | - |
| Horizon H | How far ahead the label looks | - |

The repo goal (30 s+ before onset) is anticipation at the far end of anything published.

---

## 2. Work directly on violence / crime / aggression anticipation

### 2.1 Streaming Video Crime Anticipation with Spatio-Temporal Causal Reasoning (Wang et al., CVPR 2026)
- Link: https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Streaming_Video_Crime_Anticipation_with_Spatio-Temporal_Causal_Reasoning_CVPR_2026_paper.html
- Idea: streaming VLM (Qwen2-VL-7B + LoRA r=64) with a Spatio-Temporal Causal Hypergraph (STCH) built over
  tracked entities (speed changes, distances such as gun-to-victim) and a memory bank for long context.
  New STCRC dataset: 73K QA samples over 5 "causal precursor" tasks (avg. 11 s prediction horizon for the
  temporal task) built from UCF-Crime; XD-Violence used only for cross-domain test.
- Inputs: 2 FPS, 8-frame streaming windows.
- Metrics: TimeDiff (MAE of predicted time-to-crime, s), weighted F1 of crime type, AUC at short (2 s),
  medium (20 s), long (60 s) windows.
- Numbers (paper, Table 3):

| Method | UCF TimeDiff (s) | UCF AUC-2s | UCF AUC-20s | UCF AUC-60s | XD TimeDiff | XD AUC-2s | XD AUC-20s | XD AUC-60s |
|---|---|---|---|---|---|---|---|---|
| UR-DMU (WSVAD detector) | 81.5 | 0.603 | 0.581 | 0.535 | 76.3 | 0.573 | 0.533 | 0.506 |
| GPT-4o (sliding window) | 86.0 | 0.581 | 0.543 | 0.537 | 55.3 | 0.618 | 0.583 | 0.565 |
| Holmes-VAU | 61.2 | 0.636 | 0.625 | 0.609 | 37.3 | 0.553 | 0.527 | 0.514 |
| Flash-VStream (streaming VLM) | 58.8 | 0.652 | 0.606 | 0.597 | 35.5 | 0.623 | 0.600 | 0.583 |
| **STCH (ours)** | **55.8** | **0.718** | **0.696** | **0.692** | **31.0** | **0.659** | **0.639** | **0.622** |

- Ablation: removing the memory bank drops UCF AUC-60s 0.692 -> 0.601 (long context matters).
- Code: not found. Compute: H200 GPUs.
- Relevance: **the single most relevant benchmark for "30 s+" claims.** It sets a realistic ceiling
  (about 0.6-0.7 window AUC at 20-60 s) and shows a pure detector is near chance far from onset.

### 2.2 Can We Anticipate Violence? Multimodal Learning from Pre-Incident Behavioral Cues (Penchala et al., arXiv 2609.40014, Sep 2026)
- Link: https://arxiv.org/abs/2609.40014
- Idea: binary Normal vs Risky classification of **pre-incident** windows (incident frames excluded) from
  443 manually labeled XD-Violence clips. T=32 frames from an observation window of at most about 4 s.
  Modalities: face-region appearance (Swin/ViT/DeiT-Tiny), audio (log-magnitude STFT -> 3-layer CNN),
  body motion (YOLO-pose COCO-17 keypoints -> 12 kinematic descriptors -> temporal transformer).
  Source-level split (clips from one source video stay in one partition).
- Numbers (paper, held-out test, 91 clips):

| Modalities | Acc | Bal. Acc | F1 | ROC-AUC |
|---|---|---|---|---|
| Audio only | 67.0 | 73.1 | 70.0 | 78.8 |
| Motion (pose) only | 82.4 | 76.1 | 87.9 | 86.2 |
| Face only (Swin-T) | 89.0 | 87.4 | 91.9 | 95.2 |
| Audio + Face (Swin-T) | 89.0 | 86.4 | 92.1 | 96.1 |
| Audio + Face + Motion (DeiT-T) | **91.2** | **89.0** | **93.7** | **96.4** |

- Caveats: tiny test set (91 clips), manual "risk" labels assigned from the same cues, short horizon
  (seconds), no false-alarm-per-hour or lead-time analysis, and XD-Violence sources (movies/news/YouTube)
  are exactly the setting where provenance shortcuts exist (see 2.3). No code.
- Relevance: confirms tri-modal (appearance + audio + pose) fusion beats any single modality for
  pre-incident cues; audio alone is weak but complementary.

### 2.3 What Do Interaction Representations Actually Measure? Pre-Event Separability in Weakly-Supervised Violence Detection (Ganesh, arXiv 2608.27879, Aug 2026)
- Link: https://arxiv.org/abs/2608.27879
- Idea: fixed pipeline (YOLOv8m + BoT-SORT + RTMPose-m; DINOv2 ViT-B/14 for appearance), compare five
  interaction representations (box geometry G7, pose descriptors, learned joint encoders, person-crop
  appearance APP, whole-frame context CTX) on UCF-Crime (15 anomalous videos) and XD-Violence (137).
  Then score anomalous videos **using only frames before the annotated onset**.
- Numbers (paper): full-video MIL AUC on XD: G7 0.829, APP 0.924, CTX 0.959. Retained separation
  ((AUC_pre - 0.5)/(AUC_full - 0.5)) on XD MIL: G7 39 %, APP 50 %, CTX 71 %. Pose representations never beat
  coarse box geometry. Whole-frame context >= person crops.
- Controls: truncating normal videos to the pre-onset length distribution reduced XD CTX retention from
  98 % to 70 % (clip length itself was a cue). Inspection found **title cards and platform watermarks** in
  pre-onset windows (e.g. a 39 s watermark-only pre-onset window in one UCF clip) that never appear in the
  CCTV normals.
- Recommendations from the paper: pre-onset-only scoring, length control, provenance audit,
  cluster-bootstrap CIs by video. No code.
- Relevance: **mandatory reading for this repo.** Any model trained on YouTube violence vs. other-source
  normals can learn the source, not the escalation.

### 2.4 AssistPDA: online video anomaly Prediction, Detection and Analysis (arXiv 2503.21904, 2025)
- Link: https://arxiv.org/abs/2503.21904
- Idea: Qwen2-VL-2B with a spatio-temporal relation distillation module, streaming at 2 FPS. Dataset
  VAPDA-127K from UCF-Crime + XD-Violence (2,415 videos). An LLM (Qwen2.5-72B) reads 1 FPS BLIP-2 captions up
  to the onset and labels "the earliest frame where the anomaly could have been predicted" (then human
  review).
- Numbers (paper): anomaly-prediction weighted F1 64.7 %, **average advance time (AAT) 29.2 s**; Holmes-VAD
  47.9 % / 15.7 s; Qwen2-VL 30.7 % / 11.6 s.
- Caveats: AAT formula is not given and **no false-alarm rate on normal videos is reported**, so the 29 s is
  not comparable to an event-level lead time at fixed FA/h. Code "to be released".
- Relevance: the LLM-based "earliest predictable frame" labeling is a usable recipe for creating precursor
  labels on our own data.

### 2.5 Latent Clarity / PULS (Samad, arXiv 2607.03558, Jul 2026)
- Link: https://arxiv.org/abs/2607.03558
- Idea: distill V-JEPA 2 features into a Qwen3-VL-Embedding semantic space and predict future latents
  (world-model style). Chunk-level AUROC 0.899 UCF-Crime, 0.816 XD-Violence (zero-shot transfer).
  Peak anticipatory advantage **+8.9 pp at T-0.5 s** only.
- Relevance: future-latent prediction in a frozen foundation-model space is cheap and works at short
  horizons; the gain disappears quickly as the horizon grows. No code.

### 2.6 Early violence recognition / pre-crime behavior (older and smaller)
- **Early Violence Recognition Using Knowledge Distillation** (Vaishy, Basak, Gautam; Springer):
  https://link.springer.com/chapter/10.1007/978-3-031-93697-5_5 . Teacher on full RWF-2000 clips distills into
  a student that sees only the first part of 5 s clips (3D ShuffleNetV2 + TCN). This is early *recognition*
  (violence already in progress), not anticipation. Teacher-student distillation from "full-view" to
  "partial-view" is a reusable trick.
- **Suspicious behavior / criminal intention at early stages of shoplifting** (Martinez-Mascorro et al.,
  arXiv 2005.02142; Computation 2021): https://arxiv.org/abs/2005.02142 . Segment UCF-Crime shoplifting
  videos into a Pre-Crime Behavior phase and classify it with a 3D CNN: about 75 % accuracy on detecting
  that a crime is about to happen; small dataset. Useful precedent for "pre-crime window" labeling.
- **Pre-fight detection** (Blunsden & Fisher, BEHAVE, hierarchical AdaBoost on cuboid features): historical
  precedent; performance dropped on continuous video because pre-fight frames are confused with normal
  activity or fighting (cited by 2609.40014).

### 2.7 Escalation and aggression in audio / speech
- **A Cross-Corpus Speech-Based Analysis of Escalating Negative Interactions** (Lefter, Baird, Stappen,
  Schuller; Frontiers in Computer Science 2022): https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2022.749804/full .
  3-level escalation (low/medium/high) from speech using openSMILE ComParE (6,373 features) or DeepSpectrum
  embeddings + linear SVM. Best cross-corpus UAR 63.8 % (chance 33.3 %); ComParE 2021 Escalation
  sub-challenge baseline 59.0 %, winner 63.2 %. **F0 (pitch) and intensity rise with escalation level**
  across corpora. Data are Dutch role-play (trains, service desk).
- **Automatic audio-visual fusion for aggression detection using meta-information** (Lefter, Burghouts,
  Rothkrantz, AVSS 2012): https://research.tudelft.nl/en/publications/automatic-audio-visual-fusion-for-aggression-detection-using-meta/ .
  Simple late fusion of unimodal detectors did not reliably predict multimodal aggression; an intermediate
  layer of meta-features helped. Supports learned mid-level fusion.
- **DeepSafety** (arXiv 2206.11822): https://arxiv.org/abs/2206.11822 . Fuses verbal (ASR text), vocal (MFCC)
  features to rate violence in conversations, F1 0.85. Supports adding a Whisper-transcript channel.
- Relevance: verbal escalation (shouting, insults, threats, crowd chanting) is plausibly the longest-lead
  cue available in phone footage; it is cheap to extract (audio tagger + ASR + text classifier).

### 2.8 Non-video analogs that do achieve 1 min+ lead time
- **Predicting aggression to others in youth with autism using a wearable biosensor** (Goodwin et al.,
  Autism Research 2019; replicated by Imbiriba et al., JAMA Netw Open 2023):
  https://onlinelibrary.wiley.com/doi/abs/10.1002/aur.2151 , https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10739066/ .
  3 min of physiology -> aggression **1 min ahead**, AUROC 0.71 (population model), 0.84 (person-specific).
- **Agitation prediction in dementia** (multimodal wristband + video, arXiv 2411.08882; JMIR Aging 2025):
  pre-agitation patterns at least 6 min before episodes, but from wristband physiology, tiny pilot (n=3).
- Relevance: even with direct physiological signals, 1-minute-ahead aggression prediction lands around
  AUROC 0.7-0.84. Video-only 30-60 s prediction should not be expected to beat that by a wide margin.

---

## 3. Video anomaly anticipation / forecasting benchmarks

### 3.1 NWPU Campus (Cao et al., CVPR 2023; TPAMI 2025)
- Links: https://arxiv.org/abs/2305.13611 , https://campusvad.github.io/ , TPAMI: https://dl.acm.org/doi/10.1109/TPAMI.2024.3461718
- 43 scenes, 28 anomaly classes, 16 h, semi-supervised (train on normal only). First dataset defined for
  video anomaly anticipation (VAA): predict whether an anomaly occurs in the next alpha frames.
- Numbers on VAA (frame AUC %, from Uni-DSM, arXiv 2610.11149, paper):

| Horizon | 0.5 s | 1 s | 1.5 s | 2 s | 2.5 s | 3 s |
|---|---|---|---|---|---|---|
| FBAE (Cao et al.) | 65.8 | 65.3 | 64.9 | 64.6 | 64.2 | 64.0 |
| SSAE (TPAMI) | 69.5 | 69.3 | 68.7 | 68.2 | 67.1 | 66.9 |
| Uni-DSM (2026) | 71.3 | 71.1 | 70.5 | 70.2 | 69.5 | 69.1 |
| Human volunteers | - | - | - | - | - | 90.4 |

- Relevance: monotone decay of AUC with horizon; even at 3 s the best model is about 20 AUC points below
  humans. Semi-supervised (normal-only) training is not enough for 30 s+.

### 3.2 AHB-F and SIaT (Majhi et al., WACV 2025) and D3P (Merilleau et al., WACV 2026)
- Links: https://openaccess.thecvf.com/content/WACV2025/papers/Majhi_Guess_Future_Anomalies_from_Normalcy_Forecasting_Abnormal_Behavior_in_Real-World_WACV_2025_paper.pdf ,
  data/code https://github.com/snehashismajhi/AHB-F ;
  D3P https://openaccess.thecvf.com/content/WACV2026/papers/Merilleau_Denoise_Divide_Distill_and_Predict_D3P_Towards_Forecasting_Long-horizon_Real-world_WACV_2026_paper.pdf
  (code announced at github.com/quentinmerilleau/D3P, not reachable on 2026-10-14).
- AHB-F: 1,470 untrimmed real-world CCTV videos (1,091 train / 379 test) merged from UCF-Crime, LAD-2000,
  UCF-CrimeV2, UBI-Fights, CCTV-Fights; 26 human-anomaly categories; avg video 102 s, avg anomaly 43 s.
  Protocol: deterministic reference points so that the observation contains only normal frames; Top-1
  frame mAP for future seconds 1-3 ("short") and 4-8 ("long").
- Method SIaT: scene + object (CLIP text-enriched) encoders, temporal and object interaction modules, and a
  NULL module for uncertainty of the normal past. Best with **200 observed frames**; longer observation
  degrades (noise from the distant past).
- Numbers (paper, Top-1 mAP %): action-anticipation baselines (OadTR, FUTR, LSTR, TeSTra) with
  scene+object features get about 62.7-63.3 short / 59.8-61.3 long. SIaT 64.8 / 61.7. D3P (differential
  attention past encoder + chunked autoregressive future decoder with distillation) **67.3 / 63.9**.
- Relevance: the only benchmark built for "observe normal, forecast abnormal" in real CCTV; horizons
  stop at 8 s. The SIaT observation-length ablation suggests that more context is not automatically better.

---

## 4. Weakly-supervised video anomaly detection (WSVAD) state of the art

These are *detectors*, but they are the natural teacher / backbone for an anticipation model, and their
frame scores can be used to pseudo-label onsets in training videos (only the test splits of UCF-Crime and
XD-Violence have frame-level labels).

Datasets: **UCF-Crime** (1,900 CCTV videos, 13 classes, 290 test videos with frame labels; metric frame
AUC). **XD-Violence** (4,754 videos, 217 h, 6 violence classes, sources: movies, sports, games, hand-held,
CCTV, car cams, news; 800 test videos with frame labels; metric frame AP; has audio). Note: XD test has
about 23 % positive frames, which inflates AP relative to other datasets (VERA authors).

| Method (venue) | Idea | Features | UCF AUC | XD AP | Code |
|---|---|---|---|---|---|
| MIL ranking, Sultani (CVPR'18) | Top-scoring snippet of anomalous bag > top of normal bag | C3D | 75.41 | 75.68 (reimpl.) | https://github.com/WaqasSultani/AnomalyDetectionCVPR2018 |
| HL-Net, Wu et al. (ECCV'20) | XD-Violence paper; holistic + localized graph branches, A+V | I3D + VGGish | 82.44 | 78.64 | https://github.com/Roc-Ng/XDVioDet |
| RTFM (ICCV'21) | Top-k feature-magnitude MIL | I3D | 84.30 | 77.81 | https://github.com/tianyu0207/RTFM |
| MSL (AAAI'22) | Multi-sequence learning | I3D / VideoSwin | 85.30 | 78.28 | - |
| S3R (ECCV'22) | Self-supervised sparse representation | I3D | 85.99 | 80.26 | - |
| MACIL-SD (ACM MM'22) | Modality-aware contrastive + self-distillation, A+V | I3D + VGGish | - | 83.40 | https://github.com/JustinYuu/MACIL_SD |
| MGFN (AAAI'23) | Magnitude-contrastive glance-focus | I3D | 86.98 | 79.19 | - |
| UR-DMU (AAAI'23) | Dual memory units (normal/abnormal) + uncertainty | I3D (+audio) | 86.97 | 81.66 (81.77 A+V) | https://github.com/henrryzh1/UR-DMU |
| CLIP-TSA (ICIP'23) | CLIP features + temporal self-attention | CLIP ViT | 87.58 | 82.19 | - |
| HyperVD (2023) | Hyperbolic graph A+V fusion | I3D + VGGish | - | 85.67 | https://github.com/xiaogangpeng/HyperVD |
| VadCLIP (AAAI'24) | Frozen CLIP, dual branch (binary + text-aligned classes), learnable prompts | CLIP ViT-B/16 | 88.02 | 84.51 | https://github.com/nwpu-zxr/VadCLIP |
| TPWNG, Yang et al. (CVPR'24) | Text-prompt pseudo labels | CLIP | 87.79 | 83.68 | - |
| PEL4VAD (TIP'24) | Temporal context aggregation + knowledge-based prompts | I3D | 86.76 | 85.59 | https://github.com/yujiangpu20/PEL4VAD |
| Cross-modal fusion + attention (arXiv 2412.20455) | A+V attention fusion | I3D + VGGish | - | 86.34 | - |
| Align-first-then-fuse (KBS 2025, arXiv 2501.07496) | Align modalities before fusion | RGB + audio + flow | - | 86.07 (85.15 RGB+A) | - |
| pi-VAD (CVPR'25, arXiv 2505.13123) | Pose, depth, panoptic masks, flow, text used **only at training** and induced into RGB | I3D | 90.33 | 85.37 | announced (repo not reachable) |
| LAVAD (CVPR'24) | Training-free: frame captions -> LLM scoring | BLIP-2 + LLM | 80.28 | 62.01 | https://github.com/lucazanella/lavad |
| VERA (CVPR'25) | Verbalized learning of guiding questions for a frozen VLM | VLM | 86.55 | 70.54 (secondary) | https://github.com/vera-framework/VERA |
| Holmes-VAD (2024) | Anomaly-focused sampler + instruction-tuned MLLM (VAD-Instruct50k) | MLLM | 89.51 | 90.67 (secondary) | https://github.com/pipixin321/HolmesVAD |
| Holmes-VAU (CVPR'25) | Long-video multi-granular understanding (HIVAU-70k) | InternVL2-2B | 88.96 | 87.68 | https://github.com/pipixin321/HolmesVAU |

Observations:
- Detection is near saturation (UCF AUC about 88-90, XD AP about 85-90) but **a good detector is a poor
  anticipator far from onset** (UR-DMU about 0.54 AUC at 60 s, see 2.1).
- Training-free LLM/VLM pipelines (LAVAD, VERA) lag trained detectors on XD AP by 15-25 points.
- The top-k MIL / magnitude objective is the standard way to learn from video-level labels; it can be
  reused for anticipation as "the max pre-onset window of a violent video should outrank the max window of
  a source-matched normal video".

---

## 5. Audio-visual fusion (XD-Violence and others)

- Audio adds a modest but consistent amount for detection on XD-Violence (e.g. UR-DMU 81.66 -> 81.77 AP;
  dedicated A+V methods MACIL-SD 83.40, HyperVD 85.67, cross-modal attention 86.34, align-then-fuse 86.07
  with flow).
- **AViS-Mamba** (arXiv 2604.03329): https://arxiv.org/abs/2604.03329 . The visual stream modulates an audio
  Mamba encoder through a learned routing gate; adaptive AV-InfoNCE. 88.6 % acc on NTU-CCTV, 75.7 % on DVD;
  robust when audio is degraded or missing.
- **Gated Fusion Networks for Multi-Modal Violence Detection** (AI 2025, doi 10.3390/ai6100259):
  per-modality gates over RGB, flow, audio.
- **AFFT** (WACV 2023, action anticipation): https://github.com/zeyun-zhong/AFFT . Mid-level transformer
  fusion beats score fusion; adding audio improves EK100 anticipation by 0.6 MT5R (16.4 -> 17.0, paper).
- In 2609.40014, audio alone is the weakest pre-incident cue (AUC 0.79) but improves appearance (0.952 ->
  0.961).
- Practical guidance: fuse at the token level with a learned gate, apply **modality dropout** in training
  (phone footage often has music overlays, muted audio or wind noise), and keep the audio pathway causal
  (only past audio, as AFFT does).

---

## 6. Pose and interaction representations

- **SPIL / LG-SPIL** (ECCV 2020; arXiv 2308.13866): skeletons as 3D point clouds with interaction learning;
  about 89-90 % acc on RWF-2000 (detection). Expensive 3D skeleton estimation.
- **DIFEM** (arXiv 2412.05386): hand-designed keypoint interaction features (joint velocities, inter-person
  distances), lightweight.
- **Ganesh 2026** (2.3): pose descriptors never beat coarse box geometry; whole-frame DINOv2 context beat
  person crops. In 2609.40014, pose-only AUC 0.86 vs face-appearance 0.95.
- **pi-VAD** (CVPR 2025): pose as a *training-only* auxiliary modality distilled into RGB features gives
  the best UCF AUC (90.33).
- Guidance: use pose/tracks for **interpretable escalation features** that plausibly lead violence by tens
  of seconds (number of people converging, minimum inter-person distance, approach velocity, encirclement /
  crowd ring formation, arm raising), not as the main backbone.

---

## 7. General action anticipation methods that transfer

| Method | Idea | Horizon | Dataset / numbers | Code |
|---|---|---|---|---|
| RULSTM (Furnari & Farinella, ICCV'19, TPAMI'20) | "Rolling" LSTM summarizes the past, "unrolling" LSTM rolls into the future; modality attention over RGB, flow, objects; sequence-completion pretraining | 0.25-2 s (1 s main) | EK100 val action MT5R 14.0, test 11.2 (from AFFT table) | https://github.com/fpv-iplab/rulstm |
| AVT (Girdhar & Grauman, ICCV'21) | Causal video transformer; losses = next-action CE + **future-feature regression** + current action CE | 1 s | EK100 val MT5R 14.9 (RGB), 15.9 (AVT+, RGB+obj); test 12.6 | https://github.com/facebookresearch/AVT |
| MeMViT (Wu et al., CVPR'22) | Cache keys/values of past clips as memory; 30x longer temporal support for 4.5 % more compute; causal for anticipation | 1 s (long context) | EK100 val MT5R 17.7 (RGB only) | https://github.com/facebookresearch/MeMViT |
| AFFT (Zhong et al., WACV'23) | Mid-level multimodal fusion transformer (RGB, flow, objects, audio), predicts future features autoregressively | 1 s | EK100 val 18.5, test 14.9 (Swin + all modalities) | https://github.com/zeyun-zhong/AFFT |
| LSTR (NeurIPS'21) | Long-term memory (512 s compressed) + short-term memory (8 s) encoder-decoder for online detection/anticipation | up to 2 s | THUMOS'14 OAD mAP 69.5 | https://github.com/amazon-research/long-short-term-transformer |
| TeSTra (ECCV'22) | LSTR with temporal-smoothing kernels, 142.8 FPS streaming | up to 2 s | THUMOS'14 71.2 | https://github.com/zhaoyue-zephyrus/TeSTra |
| MAT (ICCV'23) | Memory-and-anticipation transformer, jointly models past/present/future | up to 2 s | THUMOS'14 71.6 | https://github.com/Echo0125/Memory-and-Anticipation-Transformer |
| Long-term anticipation (Ego4D LTA, Breakfast, survey arXiv 2410.14045) | Predict sequences of future actions over minutes | 30 s - minutes | sequence edit-distance metrics | - |

Transferable ideas:
- **Future-feature regression** (AVT, AFFT, PULS): predicting the embedding at t+k is a free
  self-supervised signal on all (unlabeled) footage.
- **Long memory with compression** (LSTR, MeMViT, STCH memory bank): for 30-60 s horizons, the model must see
  at least 60-120 s of past; compress old context instead of extending the short window.
- **Multimodal mid-level fusion with modality attention** (RULSTM MATT, AFFT).

---

## 8. Objectives and evaluation borrowed from other early-warning domains

### 8.1 Early-event detection and early action prediction
- **Max-Margin Early Event Detectors (MMED)** (Hoai & De la Torre, CVPR 2012 / IJCV 2014):
  https://www3.cs.stonybrook.edu/~minhhoai/papers/MMED_CVPR12.pdf . Structured SVM trained on simulated
  frame-by-frame arrival so partial events are scored; introduced the **AMOC curve** (timeliness vs
  false-positive rate) for early detection.
- **Encouraging LSTMs to Anticipate Actions Very Early** (Aliakbarian et al., ICCV 2017):
  https://arxiv.org/abs/1703.07023 . Loss with a false-negative term at constant weight and a **false-positive
  term whose weight grows linearly over time**, so early ambiguity is tolerated while the model is still
  pushed to commit early.

### 8.2 Traffic accident anticipation (dashcam)
- **DSA-RNN** (Chan et al., ACCV 2016): https://yuxng.github.io/Papers/2016/chan_accv16.pdf . Introduced the
  **exponential anticipation loss**: per-frame BCE on positive videos weighted by
  exp(-max(0, (t_acc - t)/fps)); plain CE on negatives. Metrics: AP and **mean time-to-accident (mTTA)**
  averaged over thresholds, and TTA at 80 % recall.
- Numbers (from arXiv 2511.08640 Table 1; AP % / mTTA s):

| Method | DAD | CCD | A3D |
|---|---|---|---|
| DSA (ACCV'16) | 48.1 / 1.34 | 98.7 / 3.08 | 92.3 / 2.95 |
| AdaLEA (CVPR'18) | 52.3 / 3.44 | 99.2 / 3.45 | 92.9 / 3.16 |
| UString (TIV'22) | 53.7 / 3.53 | 99.5 / 3.74 | 93.2 / 3.24 |
| DSTA (ACM MM'20) | 52.9 / 3.21 | 99.1 / 3.54 | 93.5 / 2.87 |
| GSC (TIV'24) | 58.2 / 2.76 | 99.3 / 3.58 | 94.9 / 2.62 |
| Predict & Resist (2025, actor-critic alert timing) | 91.2 / 4.59 | - | - |

- UString code: https://github.com/Cogito2012/UString . "Predict and Resist" (arXiv 2511.08640) learns
  **when to raise the alert** with an actor-critic and time-weighted rewards that trade early alerts against
  false alarms.
- Lesson: there is an explicit **AP vs. lead-time trade-off** (stronger exponential decay -> earlier but less
  accurate). Report both.

### 8.3 Driver maneuver anticipation
- **Brain4Cars** (Jain et al., ICCV 2015 / arXiv 1601.00740): https://arxiv.org/abs/1601.00740 . Sensory-fusion
  LSTM over cabin video + road video + vehicle dynamics; exponential loss; anticipates maneuvers **3.5 s**
  ahead with 90.5 % precision / 87.4 % recall.

### 8.4 Seizure prediction (best-developed honest evaluation framework)
- Seizure prediction horizon (SPH) = minimum lead between alarm and the window in which the event is
  expected; seizure occurrence period (SOP) = window after SPH in which the event must occur for the alarm to
  count. Sensitivity is event-level; specificity is **false predictions per hour (FPR/h)** measured on
  interictal data only. Significance is tested against a **random (Poisson) predictor** with the same alarm
  rate: P_random = 1 - exp(-FPR * SOP). Typical CHB-MIT setting SPH 5 min, SOP 30 min; one RF study reports
  82 % sensitivity at 0.08 FP/h. Review: https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2023.1175305/full ;
  long-horizon caveat: https://www.sciencedirect.com/science/article/abs/pii/S0920121106003901 .

### 8.5 Deep survival analysis
- **DeepHit** (AAAI 2018): https://cdn.aaai.org/ojs/11842/11842-13-15370-1-2-20201228.pdf . Discrete-time
  distribution over event time with competing risks + ranking loss. Review: https://arxiv.org/abs/2305.14961 .
- Discrete-time hazard models turn "time to event" into a set of conditional binary problems and handle
  **right censoring** (normal footage that ends before anything happens) properly.

---

## 9. Recommendations for this project

### 9.1 Label formulation and training objective

Recommended primary target: **censored discrete-time hazard over multiple horizons**, evaluated every
1 s step t, using only inputs up to t.

- Let t0 be the onset of the next violent event in the video (first frame of physical violence). Bins
  (seconds until onset): b1 = (0,5], b2 = (5,10], b3 = (10,20], b4 = (20,30], b5 = (30,45], b6 = (45,60].
- The model outputs per-bin hazards h_j(t) = P(onset in b_j | no onset in earlier bins).
  P(onset within H) = 1 - prod_{j: b_j within H} (1 - h_j). Monotonic across horizons by construction, so the
  30 s and 60 s risks are always consistent.
- Loss (negative log-likelihood):
  - pre-onset step with t0 - t in b_k: -sum_{j<k} log(1 - h_j) - log h_k
  - step with t0 - t > 60 s, or any step in a normal video with at least 60 s of remaining footage:
    -sum_j log(1 - h_j)
  - step near the end of a normal video (fewer than 60 s remaining): **right-censored**, only bins fully
    observed contribute (do not invent "no violence" labels beyond the end of the clip).
  - steps during and shortly after an ongoing violent event: **masked** for the anticipation head (they go
    to the detection head instead). The "Can We Anticipate Violence" paper notes that standard detectors
    count early responses as false alarms; masking avoids teaching the model that.
- Auxiliary heads (shared encoder):
  1. "Violence now" detection head trained on in-event frames (standard WSVAD MIL / top-k if only video
     labels exist). Strong, easy supervision that stabilizes the encoder.
  2. Time-to-onset regression (log seconds, censored, e.g. Weibull/AFT likelihood) to report TimeDiff/MAE
     comparable to Wang et al. (CVPR 2026).
  3. Future-feature regression (AVT-style): predict the frozen encoder embedding at t+5 s / t+10 s; usable on
     unlabeled footage.
- Weighting: down-weight far-from-onset positive steps (exponential or linear ramp as in DSA / Aliakbarian)
  so the model is not forced to "find" violence 50 s early in videos where no cue exists; that pressure is
  exactly what makes models latch onto provenance shortcuts.
- Ranking regularizer (optional): within a violent video, risk at step t2 should be >= risk at step t1 for
  t1 < t2 < t0 (soft margin); across videos, top-k pre-onset risk of a violent video > top-k risk of a
  **source-matched** normal video (MIL ranking adapted to anticipation).
- Shortcut suppression: add a gradient-reversal source/domain classifier (CCTV vs phone vs broadcast,
  or per-channel) on the shared embedding, and mask frame borders / burned-in text where possible.

Simpler fall-back (if time is short): independent sigmoid heads for y_H(t) = 1[onset in (t, t+H]] for
H in {5, 10, 20, 30, 60}, with post-onset masking and censoring as above, plus a monotonicity penalty
(max(0, p_H - p_H') for H < H').

### 9.2 Model architecture suggested by the literature
- Frozen encoders, 1-2 FPS tokens:
  - Visual: whole-frame CLIP/SigLIP or DINOv2 (Ganesh: context >= crops), optionally VideoMAE v2 / InternVideo2 /
    V-JEPA 2 for motion. The repo already has CLIP and VideoMAE extractors.
  - Audio: AudioSet-pretrained tagger embeddings (PANNs / BEATs / CLAP / VGGish) plus explicit prosody
    (F0, RMS intensity, which rise with escalation) and AudioSet class scores (shout, yell, scream, crowd,
    slap).
  - Speech: Whisper ASR -> text toxicity / threat / argument classifier score per window (verbal escalation).
  - Pose/tracks: YOLO-pose or RTMPose + tracker -> interaction geometry (count, min distance, approach speed,
    convergence, encirclement).
- Fusion: token-level cross-modal transformer with learned modality gates and modality dropout (AFFT,
  AViS-Mamba style), not late score averaging.
- Temporal model: causal transformer or Mamba with a compressed long-term memory of 60-180 s plus a short
  window of about 8-16 s (LSTR / MeMViT / STCH memory bank). The SIaT ablation suggests tuning, not maxing,
  the observation length.
- Optional VLM branch: a small video-LLM (Qwen2-VL-2B class) fine-tuned with LoRA on precursor captions /
  QA (AssistPDA, STCRC recipe), used as a teacher or as an extra feature, not as the only model.

### 9.3 Labels and data
- Only the **test** splits of UCF-Crime (290 videos) and XD-Violence (800) have frame-level onsets. For
  training onsets use: UCA (UCF-Crime Annotation, 23,542 timestamped event sentences,
  https://xuange923.github.io/Surveillance-Video-Understanding), HIVAU-70k (Holmes-VAU), AHB-F (complete
  temporal labels for 1,470 CCTV videos), and pseudo-onsets from a strong detector (VadCLIP / Holmes-VAD frame
  scores) refined by hand on a subset.
- Filter XD-Violence to real-world sources (CCTV, hand-held, car cam) when the target is real footage; movie
  and game clips have edited pre-onset structure.
- Normal footage must come from the **same kinds of sources** as the violent footage (phone videos of
  crowds, arguments that did not become fights, street CCTV), otherwise the model learns provenance.
- Use an LLM/VLM to propose "earliest predictable moment" and precursor descriptions (AssistPDA / STCRC
  recipe), then human-verify a subset; these become auxiliary text targets and help interpretability.
- Note that phone footage has a selection effect: people start filming because they expect a fight. Camera
  behavior (zooming, approaching, crowd ring) is a real precursor in this distribution but will not transfer
  to fixed CCTV. Report CCTV and phone results separately.

### 9.4 Honest evaluation protocol
1. **Splits:** by source video at minimum; preferably by uploader/channel or compilation, so near-duplicate
   clips and shared watermarks never cross splits. K-fold with **cluster bootstrap CIs over videos**.
2. **Window-level metrics per horizon:** AUROC and AUPRC for "onset within H" at H = 5, 10, 20, 30, 60 s,
   computed only on causal pre-onset steps and normal steps (in-event steps excluded). Report a
   **far-window metric**: steps with 30 <= t0 - t <= 60 vs normal steps. That is the actual 30 s+ claim.
3. **Event-level metrics (primary):** define an alarm policy (threshold + k-of-n smoothing + refractory
   period). For each violent event: TP if the first alarm falls in [t0 - 60 s, t0 - 30 s] (strict) or
   [t0 - 60 s, t0) (lenient). Report sensitivity, **lead-time distribution** (median, % with >= 30 s), and
   **false alarms per hour** on normal footage and on pre-onset footage outside the window. Plot an
   AMOC-style curve (lead time / sensitivity vs FA per hour), plus mTTA and TTA at 80 % recall.
4. **Random-predictor baseline:** for alarm rate FAR (per hour) and an acceptance window W (hours),
   P_random = 1 - exp(-FAR * W). Example: FAR = 2/h, W = 30 s gives about 1.7 % chance sensitivity. A model is
   only interesting if it beats this by a wide margin at the same FAR.
5. **Shortcut audits (Ganesh 2026):** length-matched negatives; pre-onset-only scoring; "blind" baselines
   (first 2 s only, frame borders only, audio-only, video metadata only); manual inspection of the highest-
   scoring far-from-onset windows for title cards, watermarks, letterboxing, music overlays.
6. **Calibration:** Brier score and reliability diagrams; alarms depend on thresholds.
7. **Report detection and anticipation separately.** A detector that fires 0-2 s after onset is not an
   anticipator; count those as detections, not early warnings.
8. Do not use balanced-clip accuracy as the headline; report at realistic base rates (violence is rare).

### 9.5 Realistic accuracy expectations from the literature

| Setting | Horizon | Best reported | Source |
|---|---|---|---|
| WSVAD detection (UCF / XD) | 0 s | AUC 88-90 / AP 85-90 | pi-VAD, Holmes-VAD/VAU, VadCLIP |
| Campus anomaly anticipation (NWPU) | 0.5-3 s | frame AUC 0.71-0.69 (human 0.90 at 3 s) | Uni-DSM |
| Abnormal-behavior forecasting (AHB-F) | 1-3 s / 4-8 s | Top-1 mAP 67.3 / 63.9 | D3P |
| Crime anticipation, UCF-Crime | 2 / 20 / 60 s | AUC 0.718 / 0.696 / 0.692 | STCH (CVPR'26) |
| Crime anticipation, XD-Violence (cross-domain) | 2 / 20 / 60 s | AUC 0.659 / 0.639 / 0.622 | STCH (CVPR'26) |
| Detector used as anticipator (UR-DMU) | 60 s | AUC 0.535 (UCF), 0.506 (XD) | STCH table |
| Time-to-crime regression | - | MAE 55.8 s (UCF), 31.0 s (XD) | STCH (CVPR'26) |
| Pre-incident Normal vs Risky (curated 443 XD clips) | few s | AUC 0.964, acc 91.2 | Penchala 2026 |
| VLM anomaly prediction | AAT 29 s | weighted F1 64.7 (no FA/h) | AssistPDA |
| Traffic accident anticipation | mTTA 1.3-4.6 s | AP 48-91 (DAD) | DSA ... Predict & Resist |
| Driver maneuver | 3.5 s | P 90.5 / R 87.4 | Brain4Cars |
| Aggression from wearable physiology | 60 s | AUROC 0.71 (pop.), 0.84 (personal) | Goodwin 2019 |

Interpretation for this repo:
- On an honest per-video, source-controlled split, a **window AUROC of about 0.65-0.75 for "onset within
  30-60 s"** would match or beat the best published video results. A value above about 0.85 at 30 s+ should
  trigger a leakage / shortcut audit before it is believed.
- Event-level performance will be the more meaningful capstone number: e.g. "X % of fights get an alarm at
  least 30 s ahead at Y false alarms per hour of normal footage, versus Z % for a random predictor at the same
  alarm rate".

---

## 10. Reference list (links)

Violence / crime anticipation
- Wang et al., Streaming Video Crime Anticipation with Spatio-Temporal Causal Reasoning, CVPR 2026: https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Streaming_Video_Crime_Anticipation_with_Spatio-Temporal_Causal_Reasoning_CVPR_2026_paper.html
- Penchala et al., Can We Anticipate Violence?, arXiv 2609.40014: https://arxiv.org/abs/2609.40014
- Ganesh, Pre-Event Separability in Weakly-Supervised Violence Detection, arXiv 2608.27879: https://arxiv.org/abs/2608.27879
- AssistPDA, arXiv 2503.21904: https://arxiv.org/abs/2503.21904
- Latent Clarity / PULS, arXiv 2607.03558: https://arxiv.org/abs/2607.03558
- Early Violence Recognition Using Knowledge Distillation: https://link.springer.com/chapter/10.1007/978-3-031-93697-5_5
- Martinez-Mascorro et al., shoplifting pre-crime behavior: https://arxiv.org/abs/2005.02142 , https://www.mdpi.com/2079-3197/9/2/24
- Lefter et al., escalation in speech (Frontiers 2022): https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2022.749804/full
- Lefter, Burghouts, Rothkrantz, AV aggression fusion (AVSS 2012): https://research.tudelft.nl/en/publications/automatic-audio-visual-fusion-for-aggression-detection-using-meta/
- DeepSafety (audio-text violence in conversations): https://arxiv.org/abs/2206.11822
- Goodwin et al. 2019, aggression prediction from wearables: https://onlinelibrary.wiley.com/doi/abs/10.1002/aur.2151 ; Imbiriba et al. 2023: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10739066/
- Dementia agitation prediction (multimodal): https://arxiv.org/abs/2411.08882

Anomaly anticipation / forecasting
- Cao et al., NWPU Campus, CVPR 2023: https://arxiv.org/abs/2305.13611 ; TPAMI 2025: https://dl.acm.org/doi/10.1109/TPAMI.2024.3461718
- Uni-DSM, arXiv 2610.11149: https://arxiv.org/abs/2610.11149
- Majhi et al., AHB-F / SIaT, WACV 2025: https://openaccess.thecvf.com/content/WACV2025/papers/Majhi_Guess_Future_Anomalies_from_Normalcy_Forecasting_Abnormal_Behavior_in_Real-World_WACV_2025_paper.pdf ; https://github.com/snehashismajhi/AHB-F
- Merilleau et al., D3P, WACV 2026: https://openaccess.thecvf.com/content/WACV2026/papers/Merilleau_Denoise_Divide_Distill_and_Predict_D3P_Towards_Forecasting_Long-horizon_Real-world_WACV_2026_paper.pdf

WSVAD and audio-visual violence detection
- Sultani et al., MIL ranking, CVPR 2018: https://github.com/WaqasSultani/AnomalyDetectionCVPR2018
- Wu et al., XD-Violence / HL-Net, ECCV 2020: https://github.com/Roc-Ng/XDVioDet
- RTFM: https://github.com/tianyu0207/RTFM
- UR-DMU: https://github.com/henrryzh1/UR-DMU
- MACIL-SD: https://arxiv.org/abs/2207.05500 , https://github.com/JustinYuu/MACIL_SD
- HyperVD: https://arxiv.org/abs/2305.18797 , https://github.com/xiaogangpeng/HyperVD
- VadCLIP: https://github.com/nwpu-zxr/VadCLIP
- PEL4VAD: https://arxiv.org/abs/2306.14451 , https://github.com/yujiangpu20/PEL4VAD
- Cross-modal fusion and attention for WSVAD: https://arxiv.org/abs/2412.20455
- Aligning First, Then Fusing: https://arxiv.org/abs/2501.07496
- pi-VAD (Just Dance with pi!), CVPR 2025: https://arxiv.org/abs/2505.13123
- LAVAD, CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/papers/Zanella_Harnessing_Large_Language_Models_for_Training-free_Video_Anomaly_Detection_CVPR_2024_paper.pdf , https://github.com/lucazanella/lavad
- VERA, CVPR 2025: https://arxiv.org/abs/2412.01095
- Holmes-VAD: https://arxiv.org/abs/2406.12235 , https://holmesvad.github.io
- Holmes-VAU, CVPR 2025: https://arxiv.org/abs/2412.06171 , https://github.com/pipixin321/HolmesVAU
- AViS-Mamba: https://arxiv.org/abs/2604.03329
- UCA (UCF-Crime Annotation): https://arxiv.org/abs/2309.13925
- CUVA (causation of video anomalies), CVPR 2024: https://github.com/fesvhtr/CUVA
- Awesome-Video-Anomaly-Detection list: https://github.com/Junxi-Chen/Awesome-Video-Anomaly-Detection

Pose-based violence
- SPIL (ECCV 2020): https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490069.pdf ; LG-SPIL: https://arxiv.org/abs/2308.13866
- DIFEM: https://arxiv.org/abs/2412.05386

Action anticipation / online detection
- RULSTM: https://arxiv.org/abs/2005.02190 , https://github.com/fpv-iplab/rulstm
- AVT: https://arxiv.org/abs/2106.02036 , https://github.com/facebookresearch/AVT
- MeMViT: https://arxiv.org/abs/2201.08383 , https://github.com/facebookresearch/MeMViT
- AFFT: https://arxiv.org/abs/2210.12649 , https://github.com/zeyun-zhong/AFFT
- LSTR: https://arxiv.org/abs/2107.03377
- TeSTra: https://arxiv.org/abs/2209.09236
- MAT: https://arxiv.org/abs/2308.07893
- Human Action Anticipation survey: https://arxiv.org/abs/2410.14045

Early detection objectives and evaluation from other domains
- Hoai & De la Torre, MMED: https://www3.cs.stonybrook.edu/~minhhoai/papers/MMED_CVPR12.pdf
- Aliakbarian et al., ICCV 2017: https://arxiv.org/abs/1703.07023
- Chan et al., DSA-RNN accident anticipation: https://yuxng.github.io/Papers/2016/chan_accv16.pdf
- Accident anticipation survey: https://arxiv.org/abs/2308.15985
- UString: https://arxiv.org/abs/2008.00334 , https://github.com/Cogito2012/UString
- Predict and Resist (actor-critic alert timing): https://arxiv.org/abs/2511.08640
- MsFIN: https://arxiv.org/abs/2509.19227
- Brain4Cars: https://arxiv.org/abs/1601.00740
- Seizure prediction evaluation: https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2023.1175305/full
- DeepHit: https://cdn.aaai.org/ojs/11842/11842-13-15370-1-2-20201228.pdf ; deep survival review: https://arxiv.org/abs/2305.14961
