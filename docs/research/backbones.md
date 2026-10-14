# Pretrained backbones and feature extractors for violence anticipation (2023-2026)

Research date: 2026-10-14. Scope: frozen pretrained encoders plus a small trainable temporal head, trained on
hundreds to a few thousand real-world videos (YouTube, phone and CCTV street fights and assaults), on two data-centre GPUs.
The target is to predict physical violence **30+ seconds before onset** from video and audio.

How confident each claim is: numbers taken from model cards and papers carry a link. Throughput figures marked
"est." are back-of-envelope FLOP estimates, not measurements. Re-measure them on our hardware before relying on them.

---

## 0. TL;DR

1. **Video motion stream: V-JEPA 2** (`facebook/vjepa2-vitg-fpc64-256`, use ViT-L `facebook/vjepa2-vitl-fpc64-256` for fast ablations).
   Among frozen encoders it is the best at motion and interaction (SSv2, Diving-48) and the best published at
   *action anticipation* (EK100 frozen probe 39.7 R@5 versus 27.6 for the previous best). It replaces the repo's old `MCG-NJU/videomae-base`.
2. **Per-frame semantic stream: Perception Encoder `facebook/PE-Core-L14-336`** (Apache-2.0, 1024-d CLIP space).
   It also provides **text-prompt "concept scores"** ("two men arguing face to face", "a crowd forming a circle", ...).
   With a small dataset these interpretable features are the most robust. It replaces the old CLIP ViT-B/16.
3. **Audio and audio-visual stream: Meta PE-AV** (`facebook/pe-av-large`, Dec 2025, Apache-2.0). It is a joint
   audio/video/text embedding and currently state of the art. Pair it with **`facebook/pe-a-frame-base`**, a
   25 Hz open-vocabulary sound-event detector ("people shouting", "crowd yelling", "slap") that produces an
   interpretable audio escalation timeline. Add an AudioSet-tagger for 527 fixed class logits: EAT-large or BEATs.
4. **Paralinguistics: `emotion2vec/emotion2vec_plus_large`** (anger posterior and 50 Hz frame features) plus a
   WavLM arousal/dominance regressor, plus cheap DSP loudness and pitch features. Rising vocal arousal is a true *leading* cue.
5. **Speech content:** faster-whisper **large-v3** (not turbo, because turbo hallucinates more on noisy audio) with VAD
   and word timestamps, then **detoxify** (`unbiased-toxic-roberta`: toxicity, threat, insult, obscene), then a small LLM
   that rates escalation over a rolling 30-60 s transcript. pyannote `community-1` adds overlapped-speech ratio and speaker count.
6. **Pose:** **RTMO-l** (one-stage, crowd-robust, Apache-2.0 via `rtmlib`) or YOLO11/YOLO26-pose (AGPL), plus a
   tracker. Turn the output into **hand-engineered dyadic interaction features** (distance, approach velocity, mutual facing,
   arm raises, crowd-circle formation). These are the most interpretable leading indicators and the most robust to domain shift.
7. **Video-LLM as teacher, not as the online scorer:** Qwen3-VL-8B or Qwen3-Omni-30B-A3B (audio+video, Apache-2.0),
   run sparsely to give escalation ratings, pseudo-labels and hidden-state probes.
8. **Skip or deprioritise:** ImageBind (non-commercial, far behind PE-AV), LanguageBind, X-CLIP, UMT (all superseded),
   VideoMAE V2 (CC-BY-NC and much weaker frozen motion features than V-JEPA 2), and VideoPrism (JAX-only friction).
   InternVideo2-1B is a strong second motion stream if V-JEPA 2 plateaus.

Section 2 explains the most important modelling point. Most "violence" features fire only **once violence is happening**.
The 30-second-ahead signal lives in **confrontation and escalation cues**: approach and proximity, face-offs,
gesturing, crowd gathering, raised voices and verbal threats. The stack above is chosen to measure those cues explicitly.

---

## 1. Where the repo is today (for context)

The existing code (see `core/`, `training/`, `llm/`, `vlm/`) uses:

| Modality | Current extractor in repo | Recommended replacement |
|---|---|---|
| Video clip | `MCG-NJU/videomae-base` (2022, K400) | V-JEPA 2 ViT-g/L (or V-JEPA 2.1) |
| Frame semantics | CLIP ViT-B/16 | PE-Core-L14-336 (or SigLIP 2 so400m) + prompt concept scores |
| Audio | log-mel spectrogram + small CNN | PE-AV audio + PE-A-Frame SED + EAT/BEATs AudioSet logits |
| Voice affect | none | emotion2vec+ large, WavLM arousal/dominance, DSP loudness/pitch |
| Speech content | none | Whisper large-v3 + detoxify + LLM escalation rating |
| Pose | `yolov8m-pose.pt` (+ DeepSORT) | RTMO-l or YOLO11x/26-pose + BoT-SORT, dyadic features |
| VLM | Qwen2.5-VL-7B teacher / LoRA | Qwen3-VL-8B or Qwen3-Omni-30B-A3B as sparse teacher + hidden-state probe |

Data in `data/raw_videos/` currently spans `kinetics`, `ucf_crime` and `xd_violence`.

---

## 2. Leading cues versus concurrent cues

This section matters more than which backbone you pick.

A clip-level "is this violent" encoder learns punches, grappling, falls, screams and impact sounds. All of these
happen **at or after onset**, so they cannot carry a 30 s lead. For anticipation, the features must capture states
that typically *precede* contact by 10-120 s.

### 2.1 Leading (pre-violence / escalation) cues

| Cue | Typical lead | Best measured by |
|---|---|---|
| Two people closing distance, sustained close proximity (< ~1 body-height) | 5-60 s | Pose + tracking (dyadic distance, approach velocity) |
| Face-off / "squaring up": torsos mutually oriented, chest-to-chest | 5-60 s | Pose (shoulder/hip vectors), V-JEPA 2 motion tokens |
| Aggressive gesturing: pointing, arms above shoulders, chest puffing, hand in face | 5-60 s | Pose (wrist height/velocity), V-JEPA 2 |
| Agitation: pacing, high limb jerk, weight shifting | 10-120 s | Pose kinematics (speed/accel/jerk), optical-flow energy |
| Crowd convergence / circle forming around a dyad, phones raised to film | 10-120 s | Detector/pose counts + spatial layout; PE-Core prompt scores |
| Third parties stepping between or holding someone back | 5-60 s | Pose (person between dyad), VLM teacher |
| Raised vocal effort, shouting, rising arousal and anger | 10-120 s | emotion2vec+, WavLM A/D, PE-A-Frame "shouting", RMS/F0 |
| Turn-taking breakdown (overlapping speech, rapid exchanges) | 10-120 s | pyannote overlap ratio, VAD |
| Verbal threats, insults, challenges ("what you gonna do", "back up", slurs) | 5-120 s | Whisper + detoxify (`threat`, `insult`) + LLM rating |
| Weapon brandished (before use) | 1-30 s | PE-Core/SigLIP prompt scores, detector |
| Scene and setting priors (bar or club exit, street at night, school hallway) | weak prior | PE-Core/SigLIP scene embedding |

### 2.2 Concurrent or post-onset cues (detection only, no lead)

These include punches, kicks, grappling, people falling, impact sounds, screaming, glass breaking, gunshots,
crowd scattering and running away, heavy camera shake from the filmer running, and blood.

These remain useful in two ways. They are an **auxiliary training target**: multi-task "violence now" plus "violence
in 30 s" heads regularise the shared representation. They are also useful for **label QA**, to find the exact onset
time so pre-onset windows are not contaminated.

### 2.3 Implications

- Feed the temporal head **30-120 s of history** as per-second tokens, and give it **trend and derivative features**
  (slope of arousal, slope of dyadic distance). Anticipation is about *change*.
- All features must be computed **causally**: a feature at time t may use only frames and audio up to t.
  Clip encoders such as V-JEPA 2 must use clips that *end* at t.
- At a 30 s horizon, honest performance comes mostly from detecting that **a confrontation phase is already under way**.
  A calm scene that turns violent with no build-up is close to unpredictable. Consider adding a label for
  "confrontation/escalation present" in addition to time-to-onset.
- Beware **shortcuts**. In phone-filmed YouTube fights the camera operator often starts filming *because* a confrontation is
  starting. Movie clips (much of XD-Violence) have soundtracks, cuts and letterboxing. Domain-invariant cues such as pose geometry and
  vocal arousal generalise better than raw appearance embeddings. Split train and test **by source video**, and ideally by source domain.

---

## 3. Video backbones

### 3.1 Comparison table

| Model | HF id(s) | Params | Input | Feature dim | License | Notes for this task |
|---|---|---|---|---|---|---|
| **V-JEPA 2** (Meta, Jun 2025) | `facebook/vjepa2-vitl-fpc64-256`, `-vith-fpc64-256`, `-vitg-fpc64-256`, `-vitg-fpc64-384` | 0.3B / 0.6B / 1B | 16-64 frames, 256 or 384 px, tubelet 2x16x16 | 1024 (L), 1280 (H), 1408 (g); patch tokens | MIT per HF card (repo code MIT) | Best frozen motion features; SOTA frozen anticipation on EK100. HF `AutoModel` + `get_vision_features`. **Primary pick.** |
| **V-JEPA 2.1** (Meta, Mar 2026) | official via `torch.hub` `facebookresearch/vjepa2` (`vjepa2_1_vit_*_384`); community HF ports e.g. `apiantonio/vjepa2.1-vit-large-384` | 80M / 0.3B / 1B / 2B | 384 px | as above (+ G: 1664) | MIT code | Dense, temporally consistent features (dense predictive loss on all tokens). Reported EK100 action R@5 ~40.8 (secondary source). Try once the V-JEPA 2 pipeline works. |
| **InternVideo2** s2-1B (Shanghai AI Lab, 2024) | `OpenGVLab/InternVideo2-Stage2_1B-224p-f4` (gated); distilled S14/B14/L14 in `OpenGVLab/InternVideo2_distillation_models` | 1B (6B also) | 4-8 frames, 224 px | 1408 | Apache-2.0 metadata (6B: MIT); gated | Strongest balanced encoder (avg 87.0 in V-JEPA 2 Table 4). Custom code, awkward loading. Good **second motion stream**. |
| **InternVideo2.5** | `OpenGVLab/InternVideo2_5_Chat_8B` | 8B | long video | LLM | MIT-ish | A video **MLLM**, not a feature encoder. Use only as a VLM teacher, if at all. |
| **VideoMAE V2** | `OpenGVLab/VideoMAEv2-Large`, `-Huge`, `-giant` | 0.3B-1B | 16 frames, 224 | 1024-1408 | **CC-BY-NC-4.0** | Frozen features weak on motion (SSv2 frozen 56.1 versus 77.3 for V-JEPA 2-g384). Needs fine-tuning to shine. Deprioritise. |
| **Perception Encoder PE-Core** (Meta, Apr 2025) | `facebook/PE-Core-B16-224`, `-L14-336`, `-G14-448` | 0.09B / 0.32B / 1.88B (vision) | per-frame image; video-aligned via synthetic video captions | CLIP dim 1024 (B, L), 1280 (G); width 768/1024/1536 | Apache-2.0 | Best CLIP-style encoder: K400 zero-shot 73.4 (L) / 76.9 (G). **Primary semantic/per-frame pick** with text prompts. Intermediate layers carry richer features (paper title: "best visual embeddings are not at the output"). |
| **PE-Spatial / PE-Lang** | `facebook/PE-Spatial-L14-448`, `facebook/PE-Lang-L14-448` | 0.3B-1.9B | image | 1024 / 1536 | Apache-2.0 | Spatial = dense features (tracking/segmentation); Lang = MLLM-aligned. Not needed unless we do crop-level features. |
| **SigLIP 2** (Google, Feb 2025) | `google/siglip2-so400m-patch14-384`, `-so400m-patch16-naflex`, `-giant-opt-patch16-384` | 0.4B / 1B | image | 1152 (so400m) | Apache-2.0 | Near-equal alternative to PE-Core (frozen avg 81.1 versus 82.3 for PE-G). Strong zero-shot text prompts, multilingual. |
| **CLIP ViT-L/14** (OpenAI) | `openai/clip-vit-large-patch14-336` | 0.3B | image | 768 | MIT | Superseded by PE-Core and SigLIP 2. Keep only for comparison with old results. |
| **DINOv3** (Meta, Aug 2025) | `facebook/dinov3-vitl16-pretrain-lvd1689m`, ViT-S/B/H+/7B, ConvNeXt | 0.02-7B | image | 1024 (L) | DINOv3 License (custom; commercial allowed with conditions) | Best dense/spatial self-supervised features, but no text tower and no temporal modelling. Its predecessor DINOv2 does poorly on SSv2 (50.7). Use only for crop-level (person/face) embeddings. |
| **VideoPrism** (Google, ICML 2024; weights Jun 2025) | `google/videoprism-base-f16r288`, `-large-f8r288`, LvT video-text variants | 114M / 354M (+text) | 16 frames, 288 px | 768 / 1024; tokens T x 16 x 16 | Apache-2.0 | Good frozen video encoder (lit. SSv2 68.5, K400 87.6) but **JAX/Flax only**. Not worth the friction versus V-JEPA 2. |
| **UMT** (ICCV 2023) | github `OpenGVLab/unmasked_teacher` | 0.3B | 8 frames | 1024 | MIT/Apache | Superseded by InternVideo2 (same group). Skip. |
| **X-CLIP** (2022) | `microsoft/xclip-base-patch32`, `-large-patch14` | 0.2-0.4B | 8-16 frames | 512-768 | MIT | Old video-text CLIP. Superseded by PE-Core and PE-AV. Skip. |

### 3.2 Frozen-probe evidence (same protocol)

These numbers come from the V-JEPA 2 paper, [arXiv 2506.09985](https://arxiv.org/abs/2506.09985), Table 4, with an
attentive probe on a frozen backbone. SSv2, Diving-48 and Jester are **motion-centric**. K400, COIN and IN1K are
**appearance-centric**.

| Encoder | Params | SSv2 | Diving-48 | Jester | K400 | COIN | IN1K | Avg |
|---|---|---|---|---|---|---|---|---|
| DINOv2 | 1.1B | 50.7 | 82.5 | 93.4 | 83.6 | 90.7 | 86.1 | 81.1 |
| SigLIP 2 | 1.2B | 49.9 | 75.3 | 91.0 | 87.3 | 95.1 | 88.0 | 81.1 |
| PE-Core-G | 1.9B | 55.4 | 76.9 | 90.0 | 88.5 | 95.3 | 87.6 | 82.3 |
| InternVideo2 s2-1B | 1B | 69.7 | 86.4 | 97.0 | 89.4 | 93.8 | 85.8 | 87.0 |
| V-JEPA 2 ViT-L | 0.3B | 73.7 | 89.0 | 97.6 | 85.1 | 86.8 | 83.5 | 86.0 |
| V-JEPA 2 ViT-g | 1B | 75.3 | 90.1 | 97.7 | 86.6 | 90.7 | 84.6 | 87.5 |
| V-JEPA 2 ViT-g384 | 1B | 77.3 | 90.2 | 97.8 | 87.3 | 91.1 | 85.1 | 88.2 |

EK100 action anticipation, frozen V-JEPA 2 + attentive probe, mean-class recall@5 for action:
ViT-L 32.7, ViT-H 36.5, ViT-g 38.0, ViT-g384 39.7. The previous best was PlausiVL (8B) at 27.6.
Anticipation quality **scales with encoder size**, which argues for ViT-g when the compute allows.

**Takeaway.** Image-text encoders (PE, SigLIP, DINO) are excellent at "what is in the scene" but weak at "how
people move". V-JEPA 2 and InternVideo2 are the reverse. The two families are **complementary**, so use one of each.

---

## 4. Audio backbones

| Model | HF id / source | Params | Input | Output | License | Notes |
|---|---|---|---|---|---|---|
| **PE-AV** (Meta, Dec 2025) | `facebook/pe-av-small`, `-base`, `-large` (+ `-16-frame` variants) | 0.8B / 1B / 2B (audio encoder L: 1.1B) | audio (48 kHz in HF example), video (all frames or 16), text | shared 1024-d embedding; audio tokens at 25 Hz; AV fused embedding | Apache-2.0 | New SOTA joint audio-video-text encoder. ESC-50 zero-shot 96.0, VGGSound 47-48, K400 zero-shot 78.9; ImageBind gets 55.0 on K400 and LanguageBind 64.3. Gives audio-only, video-only **and fused AV** embeddings plus text-prompt scores. `transformers` `PeAudioVideoModel`. |
| **PE-A-Frame** | `facebook/pe-a-frame-small`, `-base`, `-large` | 0.4B / 0.6B / 1B | audio + free-form text | per-frame (25 Hz) audio-text similarity | Apache-2.0 | **Open-vocabulary sound event detection.** AudioSet-Strong AUROC 0.96, DESED 0.97. Query "people shouting angrily", "crowd yelling", "argument", "slap", "scream" for an interpretable 25 Hz timeline. |
| **EAT** (IJCAI 2024) | `worstchan/EAT-large_epoch20_finetune_AS2M`, `-base_epoch30_finetune_AS2M` | 309M / 88M | 16 kHz log-mel, ~10 s (variable OK) | 1024 / 768 + 527 AudioSet logits | MIT | AS-2M mAP 49.5 (large). Loads in HF with `trust_remote_code`. Good fixed-taxonomy tagger (Shout, Yell, Screaming, Crowd, Slap, Glass). |
| **BEATs** iter3+ (Microsoft, ICML 2023) | `microsoft/unilm/beats` (OneDrive); community mirror `Bencr/beats-checkpoints` | 90M | 16 kHz fbank | 768 tokens (~50 Hz before pooling) + 527 logits | MIT code; weight license unstated | De-facto standard audio encoder in audio-LLMs (SALMONN etc.). AS-2M ~48.6 mAP (paper). No official HF transformers class. |
| **AST** | `MIT/ast-finetuned-audioset-10-10-0.4593` | 86.6M | 16 kHz, 10.24 s | 768 + 527 logits | BSD-3 | AS mAP 45.9. Easiest to use (`pipeline("audio-classification")`). Fine baseline, weaker than EAT/BEATs. |
| **Dasheng** (Xiaomi) | `mispeech/dasheng-base`, `-0.6B`, `-1.2B` | 86M-1.2B | 16 kHz, 64-mel | 768-1536 at ~25 Hz | Apache-2.0 | Trained on 272k h. Strong on both general audio **and** paralinguistics (CREMA-D emotion, LibriCount speaker counting). Good single-model alternative. |
| **CLAP** (LAION) | `laion/larger_clap_general` | ~0.2B | 48 kHz, 10 s | 512 | Apache-2.0 | Zero-shot audio-text. ESC-50 ~90.5 in MAEB. Superseded by PE-AV but cheap. |
| **MS-CLAP 2023** | `microsoft/CLAP` (pip `msclap`) | ~0.2B | 44.1 kHz | 1024 | MIT (repo) | ESC-50 ~93.9 (paper), UrbanSound8K 83.0 (MAEB). |
| **PANNs CNN14** | `qiuqiangkong/audioset_tagging_cnn` | 80M | 32 kHz | 2048 + 527 logits | MIT | 2020 baseline (mAP 43.1). Only for comparison. |
| **VGGish** | TF-Hub / torchvggish | 72M | 16 kHz, 0.96 s | 128 | Apache-2.0 | What XD-Violence papers use. Outdated. |
| **Whisper encoder** | `openai/whisper-large-v3` (encoder) | 635M (enc) | 16 kHz, 30 s | 1280 at 50 Hz | MIT | Rich speech and paralinguistic features (emotion, shouting, crowd). Turbo shares a large-v3-sized encoder. Mean-pool per second. |
| **WavLM-large** | `microsoft/wavlm-large` | 316M | 16 kHz | 1024 at 50 Hz | MIT | Best SSL speech encoder for paralinguistics (SUPERB). Use middle layers (about 6-12) for emotion. |

### 4.1 Paralinguistics (anger, arousal, shouting)

| Model | HF id | Output | License | Notes |
|---|---|---|---|---|
| **emotion2vec+ large** | `emotion2vec/emotion2vec_plus_large` (FunASR `iic/emotion2vec_plus_large`) | 9-class posteriors (angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown) + utterance or **50 Hz frame** embeddings | FunASR model license (custom; research OK) | ~300M params, about 40k h of emotion pseudo-labelled speech. **Top pick** for anger probability per second. |
| **WavLM-large MSP-Podcast (dimensional)** | `tiantiaf/wavlm-large-msp-podcast-emotion-dim` | arousal, valence, dominance in [0, 1] | Open RAIL, no commercial use | Continuous arousal and dominance are ideal escalation curves. |
| **audEERING wav2vec2 MSP-dim** | `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` | arousal, dominance, valence | research only (CC-BY-NC-SA) | Widely used, robust to noise. Alternative to the above. |
| DSP features (no model) | librosa / `opensmile` eGeMAPS | RMS dBFS, loudness, F0 mean/var, spectral flux, HNR, speech rate | MIT / research | Cheap, interpretable, domain-robust. **Always include.** |

All of these SER models were trained on mostly clean, single-speaker speech. On street audio with music and wind they degrade.
Gate them with VAD, use the speech-probability-weighted mean per second, and let the head learn how far to trust them.

---

## 5. Speech content: verbal escalation

| Component | Recommended | Alternatives | License | Notes |
|---|---|---|---|---|
| ASR | `openai/whisper-large-v3` via **faster-whisper / WhisperX** (batched, word timestamps, wav2vec2 alignment) | `openai/whisper-large-v3-turbo` (~4x faster, 1-2% WER worse, **more hallucinations**); `distil-whisper/distil-large-v3.5` (English only) | MIT | Run Silero/pyannote **VAD first**, because Whisper hallucinates on music and noise. In a PennSound evaluation turbo produced ~13x more extraneous tokens than large-v3. Set `condition_on_previous_text=False` and drop segments with high `no_speech_prob` or low `avg_logprob`. |
| Diarization / overlap | `pyannote/speaker-diarization-community-1` | `pyannote/speaker-diarization-3.1` | CC-BY-4.0 (gated) | Gives speaker count, **overlapped-speech ratio** and turn rate per window. These are escalation proxies that need no transcript. |
| Toxicity | **detoxify** `unbiased` (`unitary/unbiased-toxic-roberta`) | `unitary/toxic-bert` (original Jigsaw: toxic, severe_toxic, obscene, **threat**, insult, identity_hate); `unitary/multilingual-toxic-xlm-roberta` | Apache-2.0 | Per-utterance scores, then a per-second max over a rolling 10/30/60 s window. Use the pip `detoxify` weights; the HF copies differ slightly. |
| Escalation rating (LLM) | `Qwen/Qwen3-8B` (or the repo's existing `Qwen2.5-3B-Instruct`) on a rolling 30-60 s transcript | Llama-Guard-style classifiers (built for chatbot safety, not conflict) | Apache-2.0 | A prompted 0-4 rating for "verbal conflict escalation" (challenge, insult, threat, de-escalation). The **delta** over time is the key feature. Cheap with vLLM on one GPU. |
| Text embedding (optional) | `BAAI/bge-small-en-v1.5` / `sentence-transformers/all-MiniLM-L6-v2` | — | MIT / Apache | 384-d embedding of the rolling transcript, so the head can learn its own phrases. Risks overfitting on small data. |

Caveat: many YouTube fight videos have music overlays, non-English speech, or no intelligible speech. Treat speech features
as **sparse and high-precision**. Always provide a "speech present" mask so the head does not read silence as calm.

---

## 6. Pose and skeleton interaction features

| Model | Source | Type | Accuracy / speed | License | Notes |
|---|---|---|---|---|---|
| **RTMO-l** (CVPR 2024) | mmpose `projects/rtmo`; **`rtmlib`** (ONNX, no mmcv) | one-stage, multi-person | COCO 74.8 AP at 141 FPS (V100); **CrowdPose 73.2 AP** (81.7 with extra data); runtime independent of person count | Apache-2.0 | Best fit for crowded fight scenes. |
| **RTMPose / RTMW / DWPose** | `rtmlib` | top-down (needs detector) | RTMPose-m ~75 AP COCO, fast | Apache-2.0 | RTMW gives 133 whole-body keypoints (hands and face) if we want hand-raise detail. |
| **YOLO11-pose / YOLO26-pose** | `ultralytics` (`yolo11x-pose.pt`, `yolo26x-pose.pt`) | one-stage | YOLO26 reports +7.2 pose AP over YOLO11; YOLO26x ~71.6 mAP50-95 (third-party) | **AGPL-3.0** | Easiest drop-in from the repo's YOLOv8m-pose. Built-in BoT-SORT/ByteTrack tracking. |
| **ViTPose++** | `usyd-community/vitpose-plus-base`, `-huge` (HF transformers) | top-down | 81.1 AP COCO test-dev (huge) | Apache-2.0 | Most accurate, slow. Use for offline label QA, not bulk extraction. |
| Sapiens (Meta) | `facebook/sapiens-pose-*` | top-down, high-res | strong in the wild | CC-BY-NC | Overkill. |

**Tracking:** BoT-SORT or ByteTrack (in ultralytics) or OC-SORT. Use IDs to compute per-person and per-pair time series.
For very long videos, also consider re-identification.

### 6.1 Recommended engineered features

Compute at about 10 fps, then aggregate to 1 Hz by mean/max/min. Normalise by body scale, using shoulder-hip
torso length, so CCTV and phone footage are comparable.

- **Dyadic (for the closest or most-interacting pair, and the max over all pairs):** centroid distance / body height,
  approach velocity (d distance/dt), closing acceleration, relative torso orientation (cosine of facing vectors,
  i.e. mutual facing), head-to-head distance, wrist-to-other-person-head distance (min), contact flag (keypoint overlap).
- **Individual agitation:** wrist speed, acceleration and jerk; fraction of time wrists are above the shoulders; arm extension
  toward the other person; stance width; torso lean toward the other person; motion energy (sum of keypoint speeds).
- **Group / crowd:** number of tracked people, people within 3 body-heights of the dyad, **circle score** (angular
  coverage of bystanders around the dyad's midpoint), convergence rate (mean radial velocity of others toward the
  dyad), "phone-raised" pose count, and a "between" score (a third person on the line segment between the dyad).
- **Camera:** global motion and shake (from optical flow or frame differencing). This is a leading cue in phone footage
  but also a **shortcut risk**, so keep it as a separate ablatable feature.

These match the kinematic descriptors (speed, acceleration, jerk, dispersion, wrist speed, motion energy) used in the
only directly comparable pre-incident study (Penchala et al., 2026, section 8), and extend them with **interpersonal** geometry,
which that paper did not model.

Expected failure modes: low-resolution or overhead CCTV, night footage, heavy occlusion in crowds. Always provide
pose-confidence and number-of-people channels so the head can down-weight bad pose.

---

## 7. Multimodal joint encoders and video-LLMs

### 7.1 Joint embedding models

| Model | HF id | License | Verdict |
|---|---|---|---|
| **PE-AV** (Dec 2025) | `facebook/pe-av-large` | Apache-2.0 | **Use.** Beats ImageBind and LanguageBind by wide margins: AudioCaps V->A R@1 88.3 versus 51.3 (ImageBind) and 9.1 (LanguageBind); K400 zero-shot 78.9 versus 55.0 and 64.3. |
| ImageBind (2023) | `facebook/imagebind-huge` (mirrors e.g. `nielsr/imagebind-huge`) | **CC-BY-NC-4.0** | Skip; superseded. |
| LanguageBind (2023/24) | `LanguageBind/LanguageBind_Video_FT`, `LanguageBind_Audio_FT` | MIT (code/models) | Skip; superseded. |

### 7.2 Video-LLMs as zero-shot scorers or teachers

| Model | HF id | Inputs | License | Use |
|---|---|---|---|---|
| **Qwen3-VL-8B-Instruct** (and 2B/4B/32B, MoE) | `Qwen/Qwen3-VL-8B-Instruct` | video + text | Apache-2.0 | Upgrade from the repo's Qwen2.5-VL-7B. The tech report says 8B is competitive with Qwen2.5-VL-72B on video. |
| **Qwen3-Omni-30B-A3B** | `Qwen/Qwen3-Omni-30B-A3B-Instruct` / `-Thinking` / `-Captioner` | **video + audio** + text | Apache-2.0 | The only strong open model that hears the shouting *and* sees the face-off in one pass. 3B active params; fits on one GPU in bf16 (~70 GB weights). |
| InternVL3.5 / InternVideo2.5-Chat-8B | `OpenGVLab/InternVideo2_5_Chat_8B` | video | MIT-ish | Alternative teacher. |

How to use them, given small data and the need for a causal setup:

1. **Teacher ratings.** Every 4-8 s, show the model the past 16-32 s of clip (Omni: with audio). Ask a structured question:
   escalation 0-4, plus a checklist (arguing? squaring up? shoving? crowd forming? threats heard? someone restraining?).
   Use the outputs as extra input features and/or as **dense pseudo-labels** to pretrain the head on unlabeled footage
   (UCF-Crime and XD-Violence normals, Kinetics, YouTube).
2. **Hidden-state probing.** HiProbe-VAD ([arXiv 2507.17394](https://arxiv.org/abs/2507.17394), ACM MM 2025) shows that
   *intermediate-layer* MLLM hidden states are more linearly separable for anomalies than the outputs. Pool a middle
   layer's video-token states into a feature (about 3584-d for 7-8B) and train a linear or attentive probe on it.
3. **Do not use as the online scorer.** Zero-shot VLM VAD is still well below supervised methods. Training-free results
   on XD-Violence: LAVAD 62.0 AP, EventVAD 64.0 AP, PANDA 70.2 AP, Flashback 75.1 AP. Weakly-supervised AV methods
   reach about 85-88 AP. Latency is also seconds per call.

---

## 8. Related work: pre-incident violence anticipation

- **Penchala et al., "Can We Anticipate Violence? Multimodal Learning from Pre-Incident Behavioral Cues"**
  (Univ. of Alabama, [arXiv 2609.40014](https://arxiv.org/html/2609.40014), Sep 2026). 443 pre-incident clips cut from XD-Violence
  (the incident itself removed), labelled No/Low/Medium/High risk and binarised. Inputs are 32 frames at 8 fps,
  so the horizon is **only about 4 s**. Three modalities:
  - face crops through a frozen Swin-T, ViT-T or DeiT-T plus a temporal transformer;
  - log-spectrogram through a small CNN;
  - 12 kinematic descriptors from YOLO-pose COCO-17.

  Results: face only AUC 91.9-95.2; motion only 86.2; audio only 78.8 (precision 92, recall 56); all three (DeiT) acc 91.2, F1 93.7, AUC 96.4.
  Limitations: tiny test set (~91 clips), single run, subjective labels, no lead-time analysis, movie footage.

  **Lessons for us:** fusion helps; audio is high-precision but low-recall; pose kinematics are a solid, cheap signal.
  Face-crop appearance was the strongest single cue. That is worth an ablation (PE-Core or DINOv3 on head and upper-body crops of the dyad),
  but it is a likely shortcut on movie footage. Their backbones were tiny ImageNet models. Swapping in V-JEPA 2, PE-Core and PE-AV,
  and adding interpersonal pose geometry, are clear upgrades.
- **XD-Violence audio-visual detection**
  ([Wu et al. ECCV 2020](https://arxiv.org/abs/2007.04687)) found that audio alone is much weaker than visual,
  but audio fusion adds about 3 AP. Later works (MACIL-SD 83.4 AP, HyperVD 85.7 AP, ~86-88 AP fusion variants) all still use
  I3D + VGGish. The backbones in this report are 4-6 years newer than what those benchmarks use.
- **Older pre-fight work:** Blunsden and Fisher's pre-fight recognition used cuboid features and AdaBoost. On continuous video it confused
  pre-fight with normal activity, which shows how hard the pre-onset regime is.
- **Classic aggression-audio systems** (Sound Intelligence verbal aggression detection; CASSANDRA): shouting is the main
  anticipatory audio cue, and also the main false-alarm source (loud but benign shouting). Fuse audio with visual proximity cues to suppress that.

---

## 9. Recommended stack: best accuracy per compute on two data-centre GPUs

All streams are extracted **offline, once**, cached as per-second token sequences (fp16 `.npy` under `data/processed/<stream>/`),
and computed **causally**. The temporal head then trains in minutes.

### Tier 1: core, build first

| Stream | Model | Sampling | Stored per second | Approx. extraction cost (est.) |
|---|---|---|---|---|
| V-motion | V-JEPA 2 ViT-g/16 256 (`facebook/vjepa2-vitg-fpc64-256`); ablate with ViT-L | 16-frame clip at ~8 fps (2 s) ending at t, stride 1 s | 1408-d mean-pooled + optional 2x2 spatial pooled tokens; also cache a middle layer | ViT-g: ~5-10 GPU-min per hour of video; ViT-L ~4x faster |
| V-semantic | PE-Core-L14-336 (`facebook/PE-Core-L14-336`) | 2 fps | 1024-d CLIP embedding (+ a middle-layer CLS) + **~40-60 prompt similarity scores** | ~1 GPU-min per hour of video |
| Audio / AV | PE-AV large (`facebook/pe-av-large`) | 2-4 s windows, stride 1 s (16-frame video variant for AV) | 1024-d audio, 1024-d AV fused + prompt scores | a few GPU-min per hour of video |
| Audio SED | PE-A-Frame base (`facebook/pe-a-frame-base`) | 25 Hz, then max/mean per second | ~20-30 free-text event scores | cheap |
| Voice affect | emotion2vec+ large + WavLM MSP-dim + DSP (RMS, F0, flux) | 50 Hz, then per second, VAD-weighted | 9 posteriors + A/V/D + ~10 DSP + speech-prob | cheap |
| Pose | RTMO-l (rtmlib) or YOLO11x-pose + BoT-SORT | 10 fps | ~40-60 engineered dyadic/group features + confidences | the slowest CPU/GPU stage; ~real-time per GPU |
| Speech | WhisperX large-v3 + VAD; detoxify; pyannote community-1 | per utterance, aligned to seconds | toxicity/threat/insult (rolling max), overlap ratio, #speakers, words/s, speech mask | ~2-5 min per hour of audio per GPU |

With 1,000-3,000 videos of a few minutes each (~50-150 h), the entire Tier 1 extraction is **well under one day on 2x GPU**.
Video decoding (use `torchcodec` or `decord` with NVDEC) is likely the bottleneck, not the encoders.
Storage is about 1-3 GB per 100 h for pooled features, or about 15-30 GB if spatial tokens are kept.

### Tier 2: add if Tier 1 plateaus

- **Qwen3-Omni-30B-A3B** (or Qwen3-VL-8B) teacher ratings every 4-8 s, plus middle-layer hidden-state pooled features.
- **InternVideo2 s2-1B** as a second motion stream, or **V-JEPA 2.1** (384 px, dense features).
- **EAT-large AS2M** (or BEATs) 527-class AudioSet logits, if PE-A-Frame free-text prompts prove noisy.
- **Crop-level features**: PE-Core or DINOv3 embeddings of the two dyad members' upper-body and head crops (the Penchala-style face cue).

### Tier 3: skip unless there is a specific reason

ImageBind, LanguageBind, X-CLIP, UMT, VideoMAE V2, VideoPrism (JAX), CLIP ViT-B, PANNs, VGGish, Sapiens.

### 9.1 Temporal head guidance (brief)

This depends on the backbone choice, so it is noted here briefly. Full design belongs in the modelling doc.

- Per-stream LayerNorm, then Linear down to 128-256-d, then **modality dropout** (p ~0.2-0.3 per stream) and a "stream present" mask.
- Causal temporal model over 60-120 s of 1 Hz tokens: a small transformer (2-4 layers, d=256) or GRU.
  Add explicit slope features (Δ over 5, 15 and 30 s) for the interpretable streams.
- Multi-task outputs: (a) violence within the next 30 s / 60 s (or a discrete-time **hazard** head), (b) violence now (auxiliary),
  (c) optionally escalation level from the VLM teacher (distillation).
- Small-data hygiene: grouped CV by source video and domain; heavy dropout and weight decay; feature-space mixup;
  random temporal crops; class-balanced sampling; report AUROC/AP **per lead-time bucket** (0-10, 10-30, 30-60 s).
- Use **intermediate layers**. PE, HiProbe-VAD and WavLM all show the best frozen features are often not at the last
  layer. Cache 2-3 layers per encoder and let the probe choose, for example with learned layer weights.

---

## 10. Licensing summary (academic capstone use)

| Permissive (Apache/MIT/BSD) | Custom / gated | Non-commercial (research OK) |
|---|---|---|
| V-JEPA 2 (MIT), PE-Core/Spatial/Lang, PE-AV, PE-A-Frame, SigLIP 2, VideoPrism, EAT, BEATs code, AST, Dasheng, LAION/MS CLAP, PANNs, WavLM, Whisper, detoxify, rtmlib/RTMO/RTMPose, ViTPose++ (HF), LanguageBind, X-CLIP, Qwen3-VL, Qwen3-Omni | DINOv3 License; emotion2vec (FunASR model license); InternVideo2 (gated, Apache metadata); pyannote community-1 (CC-BY-4.0, gated); **Ultralytics YOLO (AGPL-3.0)** | VideoMAE V2 (CC-BY-NC), ImageBind (CC-BY-NC), audEERING wav2vec2-emotion, WavLM MSP-Podcast (Open RAIL, no commercial), Sapiens |

Everything here is fine for an academic capstone. If the system is ever commercialised, swap Ultralytics for RTMO
and the non-commercial SER models for emotion2vec plus DSP features.

---

## 11. Sources

**Video**
- V-JEPA 2: [paper arXiv 2506.09985](https://arxiv.org/abs/2506.09985), [GitHub facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2), [HF collection](https://huggingface.co/collections/facebook/v-jepa-2), [vitl card](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256), [vitg card](https://huggingface.co/facebook/vjepa2-vitg-fpc64-256), [transformers docs](https://huggingface.co/docs/transformers/model_doc/vjepa2)
- V-JEPA 2.1: [arXiv 2603.14482](https://arxiv.org/html/2603.14482v1), [community HF port (L)](https://huggingface.co/apiantonio/vjepa2.1-vit-large-384), [TechTalks summary](https://bdtechtalks.com/2026/03/23/v-jepa-2-1/)
- InternVideo2: [GitHub](https://github.com/OpenGVLab/InternVideo), [Stage2 1B](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4), [6B](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_6B), [distillation models](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models), [MODEL_ZOO](https://github.com/OpenGVLab/InternVideo/blob/main/InternVideo2/multi_modality/MODEL_ZOO.md); InternVideo2.5 [Chat-8B](https://huggingface.co/OpenGVLab/InternVideo2_5_Chat_8B); InternVideo3 [arXiv 2606.12195](https://arxiv.org/abs/2606.12195)
- VideoMAE V2: [Large](https://huggingface.co/OpenGVLab/VideoMAEv2-Large), [giant](https://huggingface.co/OpenGVLab/VideoMAEv2-giant)
- Perception Encoder: [paper arXiv 2504.13181](https://arxiv.org/abs/2504.13181), [PE README / model table](https://github.com/facebookresearch/perception_models/blob/main/apps/pe/README.md), [PE-Core-G14-448](https://huggingface.co/facebook/PE-Core-G14-448), [collection](https://huggingface.co/collections/facebook/perception-encoder)
- SigLIP 2: [HF blog](https://huggingface.co/blog/siglip2), [so400m-patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384)
- DINOv3: [HF ViT-L](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m), [Meta blog](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/), [license discussion](https://github.com/facebookresearch/dinov3/issues/28)
- VideoPrism: [HF base](https://huggingface.co/google/videoprism-base-f16r288), [GitHub](https://github.com/google-deepmind/videoprism)

**Audio / audio-visual**
- PE-AV: [paper arXiv 2512.19687](https://arxiv.org/abs/2512.19687), [pe-av-large](https://huggingface.co/facebook/pe-av-large), [collection](https://huggingface.co/collections/facebook/perception-encoder-audio-visual), [transformers PE Audio Video docs](https://huggingface.co/docs/transformers/model_doc/pe_audio_video)
- EAT: [GitHub](https://github.com/cwx-worst-one/EAT), [paper](https://arxiv.org/abs/2401.03497)
- BEATs: [README](https://github.com/microsoft/unilm/blob/master/beats/README.md), [paper](https://huggingface.co/papers/2212.09058), [license issue](https://github.com/microsoft/unilm/issues/1760)
- AST: [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
- Dasheng: [dasheng-1.2B](https://huggingface.co/mispeech/dasheng-1.2B), [collection](https://huggingface.co/collections/mispeech/dasheng)
- CLAP: [laion/larger_clap_general](https://huggingface.co/laion/larger_clap_general), [MAEB benchmark](https://arxiv.org/abs/2602.16008)
- emotion2vec: [emotion2vec_plus_large](https://huggingface.co/emotion2vec/emotion2vec_plus_large), [GitHub](https://github.com/ddlBoJack/emotion2vec)
- Dimensional SER: [tiantiaf/wavlm-large-msp-podcast-emotion-dim](https://huggingface.co/tiantiaf/wavlm-large-msp-podcast-emotion-dim), [audeering wav2vec2 msp-dim](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim)

**Speech**
- Whisper: [large-v3-turbo card](https://huggingface.co/openai/whisper-large-v3-turbo), [faster-whisper turbo benchmark](https://github.com/SYSTRAN/faster-whisper/issues/1030), [PennSound hallucination evaluation](https://arxiv.org/abs/2504.05702)
- detoxify: [GitHub](https://github.com/unitaryai/detoxify), [toxic-bert](https://huggingface.co/unitary/toxic-bert)
- pyannote: [community-1 blog](https://www.pyannote.ai/blog/community-1), [pyannote-audio](https://github.com/pyannote/pyannote-audio)

**Pose**
- RTMO: [paper arXiv 2312.07526](https://arxiv.org/abs/2312.07526), [mmpose project](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo); [rtmlib](https://github.com/Tau-J/rtmlib)
- YOLO26: [docs](https://docs.ultralytics.com/models/yolo26), [paper arXiv 2606.03748](https://arxiv.org/html/2606.03748v1)
- ViTPose++: [usyd-community/vitpose-plus-huge](https://huggingface.co/usyd-community/vitpose-plus-huge)
- Skeleton violence features: [DIFEM arXiv 2412.05386](https://arxiv.org/abs/2412.05386), [moderate violence / pushing, arXiv 2605.02659](https://arxiv.org/html/2605.02659v1), [MSTFDet fistfight localization](https://pmc.ncbi.nlm.nih.gov/articles/PMC12343968/)

**Multimodal / VLM / VAD**
- ImageBind / LanguageBind licences: [nielsr/imagebind-huge](https://huggingface.co/nielsr/imagebind-huge), [LanguageBind](https://huggingface.co/LanguageBind/LanguageBind_Image)
- Qwen3-VL: [tech report arXiv 2511.21631](https://arxiv.org/abs/2511.21631); Qwen3-Omni: [Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct), [tech report](https://huggingface.co/papers/2509.17765)
- HiProbe-VAD: [arXiv 2507.17394](https://arxiv.org/abs/2507.17394); LAVAD: [project](https://lucazanella.github.io/lavad/); QVAD comparison table: [arXiv 2604.03040](https://arxiv.org/html/2604.03040); holistic zero-shot VAD: [arXiv 2511.00962](https://arxiv.org/abs/2511.00962)
- XD-Violence: [Wu et al. 2020](https://arxiv.org/abs/2007.04687); MACIL-SD: [GitHub](https://github.com/JustinYuu/MACIL_SD); HyperVD: [arXiv 2305.18797](https://arxiv.org/abs/2305.18797); [cross-modal fusion 2024](https://arxiv.org/abs/2412.20455)
- Pre-incident anticipation: [Penchala et al. arXiv 2609.40014](https://arxiv.org/html/2609.40014), [Unite.AI summary](https://www.unite.ai/predicting-violence-in-advance-with-ai/)
- Aggression audio: [Verbal aggression detection in complex social environments](https://ieeexplore.ieee.org/document/4425279/), [Multi-modal human aggression detection](https://www.researchgate.net/publication/279456550_Multi-modal_Human_Aggression_Detection)
