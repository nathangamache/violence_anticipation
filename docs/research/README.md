# Research brief: datasets, backbones and literature for 30 s violence anticipation

Short summary of the three research reports in this folder (written 2026-10-14), with a prioritized
shortlist. Read the full reports before acting on a line here; they carry the sources and caveats.

| Report | Question it answers |
|---|---|
| [datasets.md](datasets.md) | Which public datasets have untrimmed real-world footage with onset labels and long lead-in? (lead-in counts measured from the annotations) |
| [backbones.md](backbones.md) | Which frozen pretrained encoders (video, audio, speech, pose, VLM) give the best features per GPU-hour on 2x GPU? |
| [literature.md](literature.md) | What has been published on anticipation, which objectives work, how to evaluate honestly, and what accuracy is realistic? |

Note: several cited papers are 2025 to 2026 preprints (for example arXiv 2609.40014, 2608.27879,
2607.03558) without code releases. Numbers marked "(secondary)" in the reports should be re-checked before
they appear in the capstone write-up.

---

## 1. Five things the research establishes

1. **No public dataset or published model targets 30 s violence anticipation.** Published anticipation
   horizons are 0.5 to 8 s; the only 20 s / 60 s video results (Wang et al., CVPR 2026) reach window AUC
   **0.69 on UCF-Crime and 0.62 on XD-Violence at 60 s**, and a strong detector used as an anticipator is near
   chance (0.51 to 0.54). On an honest split, 0.65 to 0.75 would be state of the art; above about 0.85 is a
   red flag for leakage.
2. **Lead-in is the scarce resource.** Median onset is 6 to 12 s in every source. Pooling all public
   sources gives about **370 interpersonal-violence videos with at least 30 s of pre-onset footage (about
   220 fight/assault/abuse only)**. The UCF-Crime test set alone has 4 such fight videos.
3. **Provenance shortcuts are real** (Ganesh 2026): pre-onset separability on UCF/XD is partly driven by
   title cards, watermarks and clip length, because violent videos come from YouTube/news and normal ones
   from CCTV. Length-matched, same-source negatives and blind baselines are mandatory.
4. **What is predictable 30 s ahead is escalation, not the punch:** approach and shrinking distance,
   face-offs, gesturing, crowd gathering and encircling, raised voices, overlapping speech, verbal threats.
   Most "violence" features (impacts, screams, grappling) fire only once violence has started.
5. **Evaluate like seizure prediction and accident anticipation:** event-level sensitivity at a fixed false
   alarm rate per hour, uncapped lead-time distribution, a Poisson random-predictor baseline, grouped splits
   and cluster-bootstrap confidence intervals. This is now the repo's VA30 protocol ([../BASELINE.md](../BASELINE.md) section 6).

---

## 2. Prioritized dataset shortlist

| Priority | Dataset | Why | Lead-in (at least 30 s) | Audio | Access | Action |
|---|---|---|---|---|---|---|
| 1 | **HIVAU-70k raw annotations** (Holmes-VAU, CVPR 2025) | Human event start/end times for the UCF-Crime and XD-Violence **training** splits; raises onset-labelled videos from about 640 to about 3,300 | enables the two rows below | n/a | GitHub (MIT), small JSON | Download now; write `setup/build_hivau_onsets.py`; filter on non-empty `label` (normal videos also have random "events") |
| 2 | **XD-Violence train split** (3,954 videos) | Largest pool of audio-visual positives with lead-in | 397 videos (172 interpersonal, 93 fight/abuse) via HIVAU | yes (some music-only) | open (OneDrive/Baidu) | Download; only the 798 test clips are local |
| 3 | **UCF-Crime train onsets** (already on disk) | 810 anomaly videos currently mislabelled as negatives get real onsets | 192 videos (73 interpersonal) via HIVAU | mostly silent | local | Merge HIVAU onsets; also use the 950 unused normal videos as clean negatives |
| 4 | **UBI-Fights** | Long real-world fight videos plus 784 long normal videos, frame-level CSV labels | 45 fight videos | none | direct zip, 8.1 GB (`curl -k`, expired TLS cert) | Download |
| 5 | **Curated YouTube set** (`data/anticipation_curation/`, 92 videos, on disk) | Best domain match (road rage, bodycam, CCTV confrontations), all with audio | about 28 (from pseudo-labels; unverified) | yes | local | **Hand-annotate onsets**; use as the real-world hold-out test |
| 6 | **NTU CCTV-Fights** | 1,000 fight videos, 2,414 instances, frame-level, CCTV up to 12 min | likely 100 to 200 (to be computed) | partial | request form, non-commercial | Request access |
| 7 | **UCA** (UCF-Crime dense captions) | Timestamped captions including pre-event behaviour; precursor-cue supervision and onset checks | n/a | n/a | GitHub, Apache-2.0 | Download (small) |
| 8 | **DVD** (2025) | Bodycam, doorbell, dashcam, phone; frame-level; long videos; event-linked sound | likely many | yes | "upon acceptance" | Watch for release |
| 9 | MSAD, CUVA/ECVA, NWPU Campus, AHB-F | Extra untrimmed anomaly video, cause text, anticipation protocols | some | mostly no | forms / GitHub | Secondary evaluation or protocol reference |
| 10 | AudioSet / VGGSound / FSD50K classes (shout, yell, scream, slap, smash, crowd), MuTox, SSPNet Conflict, CREMA-D | Train or validate audio and speech cue heads | n/a | yes | open / request | Only if a pretrained tagger is not enough |
| Skip for anticipation | RWF-2000, RLVS, Hockey, Movies, SCVD, Bus Violence, AIRTLab, Kinetics clips | 1 to 10 s trimmed clips, no lead-in | none | mixed | open | Use only for a "violence now" auxiliary head or hard negatives |

---

## 3. Prioritized backbone shortlist (all frozen, causal, cached at 1 Hz)

| Priority | Stream | Model | What it contributes | Notes |
|---|---|---|---|---|
| 1 | Pose / interaction | RTMO-l via `rtmlib` (Apache-2.0) or YOLO11x-pose (AGPL; `yolov8m-pose.pt` already local) + BoT-SORT tracking | About 50 engineered dyadic and crowd features: distance divided by body height, approach velocity, mutual facing, arm raises toward the other person, crowd convergence, confidences | Most interpretable leading cues; first ablation stream. Fails on low-res or overhead CCTV |
| 2 | Video motion | **V-JEPA 2** ViT-L (`facebook/vjepa2-vitl-fpc64-256`) for ablations, ViT-g for final | 16-frame clip ending at t, stride 1 s; best frozen motion and action-anticipation features | Replaces VideoMAE-base. Cache a middle layer too |
| 3 | Frame semantics | **PE-Core-L14-336** (`facebook/PE-Core-L14-336`) | 1024-d embedding plus 40 to 60 text-prompt concept scores ("two men arguing face to face", "crowd forming a circle") | Replaces CLIP ViT-B/16; SigLIP 2 is the alternative, not both |
| 4 | Audio / AV | **PE-AV large** + **PE-A-Frame base** | Audio and fused AV embeddings; 25 Hz free-text sound-event scores (shouting, crowd yelling, argument, slap) | Only for sources with real audio (XD, YouTube); needs a presence mask |
| 5 | Voice affect | emotion2vec+ large, WavLM arousal/dominance, DSP loudness and F0 | Rising vocal arousal and anger, a true leading cue | Gate with VAD; WavLM MSP-dim is non-commercial |
| 6 | Speech content | WhisperX large-v3 + detoxify + pyannote overlap ratio | Threat, insult and toxicity scores; overlapping speech | Use large-v3, not turbo (hallucinates); speech-present mask |
| 7 | VLM teacher (offline only) | Qwen3-VL-8B or Qwen3-Omni-30B-A3B every 4 to 8 s | Escalation ratings, cue checklists, pseudo-labels, middle-layer probes | Not an online scorer (zero-shot VLM VAD is 62 to 75 AP on XD and takes seconds per call) |
| Keep for comparison | Frame semantics (existing) | Qwen2.5-VL-7B embeddings (`data/processed/vlm_emb*`) | Already extracted; basis of baseline B0 | Irregular sampling; scene-dominated |
| Skip | various | ImageBind, LanguageBind, X-CLIP, UMT, VideoMAE V2, VideoPrism (JAX), CLIP ViT-B, PANNs, VGGish | Superseded | |

Estimated Tier-1 extraction for 50 to 150 h of video is under one day on 2x GPU; video decoding is the
likely bottleneck (backbones.md section 9).

---

## 4. Recommended modelling recipe (from literature.md section 9)

- **Target:** censored discrete-time hazard over bins (0,5], (5,10], (10,20], (20,30], (30,45], (45,60] s,
  scored every second from past inputs only; P(onset within H) = 1 - prod(1 - h_j). In-violence and
  aftermath ticks masked; normal-video ends right-censored. Fallback: independent sigmoid heads for
  H in {5, 10, 20, 30, 60} with a monotonicity penalty.
- **Auxiliary heads:** violence-now detection; censored time-to-onset regression; future-feature regression
  (predict the frozen embedding at t+5 s and t+10 s; usable on unlabeled footage).
- **Weighting:** down-weight far-from-onset positives (exponential or linear ramp) so the model is not
  pushed to find cues that are not there, which is what drives shortcut learning.
- **Architecture:** per-stream LayerNorm and projection to 128 to 256-d, modality dropout and presence
  masks, token-level fusion, causal transformer or GRU over 60 to 120 s of 1 Hz history plus explicit slope
  features over 5, 15 and 30 s.
- **Shortcut suppression:** same-source normals, masking burned-in text and borders, optional
  gradient-reversal source classifier, and the blind baselines required by VA30.
- **Ablation order:** pose only -> + V-JEPA 2-L -> + PE-Core prompts -> + audio -> + voice affect ->
  + speech -> V-JEPA 2-g -> + VLM teacher.
- **Report CCTV and phone footage separately:** phone filmers start recording because they expect a fight,
  so camera behaviour is a precursor there that will not transfer to fixed CCTV.
