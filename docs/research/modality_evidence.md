# Modality evidence: which signals detect or anticipate violence, how early, and on what footage

Written 2026-10-14 (Investigation 2 of the modality review). Audience: future agents choosing extractors, fusion
designs and ablations for the VA30 benchmark ([../VA30.md](../VA30.md)). The question is the owner's: **which
cues and modalities give the highest performance AND the longest anticipation time (30 s or more) on CCTV,
bodycam, dashcam and phone footage of confrontations?** The extractors in [../FEATURES.md](../FEATURES.md) were
picked from a backbone survey ([backbones.md](backbones.md)), not from this kind of evidence; this report supplies
the evidence and ranks the modalities with numbers.

How numbers are marked:
- **(paper)**: read from the paper's own text or tables during this investigation (PDF converted to text, or the
  arXiv HTML page).
- **(abstract)**: only the abstract or landing page was available.
- **(secondary)**: from a citing paper or summary; re-check before quoting in the capstone write-up.
- **(repo)**: measured in this repository (VA30 results in `benchmark/results/`, feature sanity checks in
  [../FEATURES.md](../FEATURES.md), or counts computed for this report from `benchmark/videos.csv` and
  `data/annotations/_cache/media_probe.json`).

Lead-time vocabulary used throughout: **detection** (fires at or after onset), **early recognition** (the act has
started but is partially observed), **short** (under 10 s before onset), **mid** (10 to 30 s), **long** (30 s or
more). Many papers in this area only measure detection; they are included because they bound what a modality can
see at all.

---

## 0. TL;DR

1. **No published study measures any modality at a 30 s lead on real violence footage except VLM-style scene
   models**, and those are weak: window AUC 0.62 to 0.69 at 60 s (STCH, CVPR 2026, fine-tuned), and in this repo
   the zero-shot Qwen2.5-VL "violent" score reaches Success@30 11.6 % at 0.54 false alarms per hour (position
   null 6.0 %). Everything else in the violence literature is detection or a lead of 4 s or less.
2. **The only computational evidence for signals 30 s or more ahead comes from what people say and how a
   conversation is going**, measured on non-violent outcomes: an officer's first 45 words (about the first 27 s)
   predict a later search, handcuffing or arrest with 70.8 % balanced accuracy (577 bodycam stops); text models
   warn of a personal attack on average 3 comments (hours) ahead at 61.6 to 71.4 % balanced accuracy, humans
   72 %; overlap and prosody predict the conflict level of the **next** 30 s clip above chance (UA 37.7 to 41.5 %
   vs 33.3 %), but much worse than the current clip (71.1 %). These are the strongest leads for the 30 s goal and
   none of them has been tested on fights.
3. **Near onset (under about 10 s), vision is stronger than audio.** In the only pre-incident tri-modal study
   (XD-Violence, windows of at most 4 s), the person-appearance crop scores AUC 0.95, pose motion 0.86, audio
   0.79. In real-world AV violence detection, video-only beats audio-only by 9 to 11 accuracy points (NTU-CCTV,
   DVD). Fighting is the exception: audio and RGB are tied on XD-Violence (AP 85.0 vs 86.0).
4. **Audio and speech plausibly carry earlier signal than vision, but nobody has measured it on the same violent
   footage.** Supporting evidence: verbal-escalation datasets, train-station actor scenarios where audio alone
   caught 7 of 13 aggression events vs 4 of 13 for video motion, keyword effects that last about 22 s, and in this
   repo a loudness rise from 30 s out in 81 % of 21 movie clips while pose and motion rise mostly in the last
   10 s. Counter-evidence: in abuse the scream follows the hit, and audio-only is the weakest stream in every
   detection benchmark. Treat "audio leads" as a hypothesis to test on VA30's audio-present subset and the
   YouTube hold-out, not as a fact.
5. **Observability decides more than accuracy.** CCTV is silent (UBI 0 of 1000 videos with audio; UCF 490 of 1879
   non-silent, mostly not scene audio); Axon body cameras record the 30 s pre-activation buffer **without audio by
   default**; dashcams often disable audio for consent law. 199 of VA30's 551 eligible events are silent CCTV, and
   143 of the 352 with audio are movie soundtracks. Faces at these resolutions are about 5 to 16 px tall, where
   expression recognition falls to 50 to 76 % even on posed benchmarks. Speech and face channels can only help the
   phone and post-activation bodycam subset; the model must work without them.
6. **Ranking for 30 s anticipation (evidence x lead x observability):** (1) whole-scene semantics from a VLM,
   read from hidden states or probability-weighted answers, with long memory; (2) speech content scored for
   escalation acts (commands, challenges, insults, threats), not generic toxicity; (3) conversational dynamics and
   vocal-arousal trends (loudness, overlap, shouting) over 30 to 120 s; (4) group geometry from tracks (gathering,
   distance, approach); (5) event history (earlier scuffles and alarms); then short-lead or detection-only streams:
   (6) sound events (shout, scream, crowd), (7) motion and camera behaviour, (8) facial expression (skip for CCTV).
   Section 1 has the numbers; section 14 has the concrete changes to the extraction plan.

---

## 1. Modality ranking with numbers

Ranked for the repo's goal: highest anticipation performance at 30 s or more on the target footage. "Best number"
is the strongest measured result for that modality anywhere near this task; check the footage column before
comparing rows.

| Rank | Modality | Best reported number (task, footage) | Lead of that number | How it degrades with lead | Observability in target footage | Evidence |
|---|---|---|---|---|---|---|
| 1 | Whole-scene semantics via VLM (risk judgement, scene context) | STCH window AUC 0.692 (UCF) / 0.622 (XD) at 60 s (paper, via [literature.md](literature.md) 2.1); repo B-teacher Success@30 11.6 % [7.6, 16.1] at 0.54 FA/h, far AUROC 0.681, time-matched 0.712 (repo) | 30 to 60 s | STCH UCF 0.718 / 0.696 / 0.692 at 2 / 20 / 60 s; without memory 0.601 at 60 s. Zero-shot VLMs: accuracy 45-50 % to 36-44 % as the cue-to-event gap grows from 1 to 20 s (RiskCueBench) | All footage incl. silent CCTV | moderate |
| 2 | Speech content (what is said) | Officer's first 45 words to stop escalation: 70.8 % balanced acc (Rho 2023); conversation derailment: 71.4 % acc (CGA-Wiki, generation-based), humans 72 % | about 27 s of speech, outcome later (minutes); online text: about 3 comments, hours | Not measured by lead; text-only escalation-level recognition is weak (UAR 37 to 47 %, chance 33 %) | Phone, bodycam after activation, some dashcam; never CCTV | moderate (non-violent outcomes) |
| 3 | Conversational dynamics and vocal arousal (loudness, F0, overlap, shouting) | 2-class conflict 80.8 to 83.9 % UAR (SSPNet, 30 s clips); 3-level escalation 59.8 % UAR cross-corpus (ComParE 2021) | current state; next 30 s clip | Conflict level UA 71.1 % now vs 37.7 % (41.5 % with deltas) for the next 30 s (chance 33.3 %); overlap correlation 0.77 now vs 0.37 next | As row 2, plus non-speech audio | moderate |
| 4 | Group geometry from tracks (count, gathering, distance, approach) | Box geometry full-video AUC 0.72 (UCF) / 0.83 (XD), keeps 39 to 84 % of its separation before onset (Ganesh 2026) | pre-onset window (seconds to tens of seconds, not resolved) | Pre-fight recall in continuous video 0.20 (BEHAVE), 0.00 (CAVIAR); repo: +0.5 persons at 10 to 30 s, +0.9 in the last 10 s | CCTV yes if people are large enough; bodycam partial; dashcam rarely | moderate |
| 5 | Event history (earlier incidents, earlier alarms) | Timing of past aggression onsets only: ROC-AUC 0.85 / 0.81 / 0.78 at 1 / 10 / 25 min (inpatient youths) | 1 to 25 min | Slow decay (0.85 to 0.78 over 24 min) | Any footage, through the model's own detection head | moderate (non-video) |
| 6 | Non-speech sound events (shout, scream, crowd, impact) | XD fighting AP 85.0 audio vs 86.0 RGB; NTU-CCTV audio-only 72.8 % vs video-only 84.0 % (detection) | detection | Repo: AST shout / crowd rise only in the last 10 s (+0.2 / +0.5 logits), 0.5 to 1.0 at onset | Phone, bodycam, XD; not CCTV | strong for detection, weak for anticipation |
| 7 | Motion, optical flow, camera behaviour | Crowd violence detection 81.3 % acc / AUC 0.85 (ViF); pose kinematics 0.86 AUC at 4 s or less (Penchala) | detection to 4 s | Repo: flow +0.009 at 10 to 30 s, +0.020 in the last 10 s, +0.040 at onset | All footage; camera motion is a provenance shortcut | moderate for detection, weak for anticipation |
| 8 | Face and facial expression | "Face" crop AUC 0.952 at 4 s or less (Penchala; actually the upper part of the largest person's box at 224 px) | 4 s or less | No lead-time data. Low-resolution FER: 12x12 px about 50 to 57 %, 14x14 px about 71 to 76 % vs 92 % at full resolution | Faces about 5 to 16 px in UCF / UBI / XD; close-ups only in phone and bodycam | weak for anticipation |
| n/a | Physiology (not observable in video) | AUROC 0.80 at 3 min (wearables, autistic inpatient youths) | 1 to 3 min | 0.71 (population) to 0.84 (personal) at 1 min | No | strong, for calibration only |

Reading the ranking:
- Rows 1 to 3 are the only modalities with any measured signal beyond about 10 s. Row 1 is the only one measured
  on violence; rows 2 and 3 were measured on escalation to non-violent outcomes (arrest, personal attacks, conflict
  level, aggression scenarios with actors).
- Rows 6 to 8 are excellent or decent **detectors**. Use them for the auxiliary "violence now" head, for event
  history, and for the last 10 s, not as the source of 30 s lead.
- Row 5 is free (derived from the model's own outputs) but VA30's headline metric deliberately excludes it
  (pre-violence ticks only, [../VA30.md](../VA30.md) section 8.3); it matters for deployment and for the 239 of 551
  eligible events that follow earlier violence in the same clip.

---

## 2. Whole-scene semantics via VLMs and frame encoders

### 2.1 What has been measured

| Study | Footage | Task and lead | Number |
|---|---|---|---|
| Repo B-teacher, zero-shot Qwen2.5-VL-7B `violent.score`, sample-and-hold (repo, [../VA30.md](../VA30.md) 7.1) | XD test + UCF `videos/` (198 eligible events, 15.2 h clean) | onset in 30 to 60 s, pre-violence ticks | far AUROC 0.681 [0.625, 0.735]; source x time matched 0.712 [0.657, 0.760]; AUROC@30 0.671 |
| same | same | event level, Success@30 | 11.6 % [7.6, 16.1] at 0.54 FA/h (Poisson 1.3 %, position null 6.0 %); 13.6 % at 1.82 FA/h (null 19.4 %, i.e. below the null) |
| same, breakdown at the 2 FA/h target | per source | Success@30 | XD movie 37.5 % (24 events), XD YouTube 13.3 % (30), UCF 6.9 % (144); official onsets 20.0 % vs HIVAU 6.5 % |
| STCH, Wang et al. CVPR 2026 (paper, via [literature.md](literature.md) 2.1) | UCF-Crime (train/test), XD cross-domain | window AUC at 2 / 20 / 60 s | UCF 0.718 / 0.696 / 0.692; XD 0.659 / 0.639 / 0.622. Memory bank removed: UCF 60 s 0.601 |
| same table, baselines | same | same | Flash-VStream (streaming VLM) UCF 0.652 / 0.606 / 0.597; GPT-4o sliding window 0.581 / 0.543 / 0.537; UR-DMU detector 0.603 / 0.581 / 0.535 (XD 0.506 at 60 s) |
| AssistPDA (paper, via [literature.md](literature.md) 2.4) | UCF + XD | anomaly prediction | weighted F1 64.7 %, average advance time 29.2 s, **no false-alarm rate reported** |
| RiskCueBench (Luo et al., arXiv 2601.03369, paper) | 986 videos: protests (crowds, throwing, shoving) and car crashes | binary risk from the early "risk signal" clip | Protest F1: Gemini-Flash-3 0.67, Keye-VL 0.59, Qwen3-VL-8B-Thinking 0.57, Qwen3-VL-8B 0.48, random 0.44; humans 0.97. Accuracy falls from 45-50 % to 36-44 % as the gap to the event grows from 1 to 20 s. Shuffling or reversing frames changes F1 by only 1 to 3 points |
| PaSBench-Video (arXiv 2606.02443, paper) | 740 streaming clips (driving, daily life incl. UCF-Crime, healthcare, industry) | first warning inside [risk start, accident - 1 s] | First-Hit at most 31 %; median remaining intervention time 1.7 s; 35 % of correct first warnings arrive less than 1 s before the event; Claude Opus 4.7 false-positive videos 46.7 %; recall and FP rate correlate (rho 0.64) |
| HiProbe-VAD (arXiv 2507.17394, paper) | UCF / XD detection | probe on MLLM hidden states, no fine-tuning | Qwen2.5-VL: UCF AUC 85.89, XD AP 80.94; InternVL2.5 86.72 / 82.15. Decoded-text pipelines: LAVAD 80.28 / 62.01. Last layer 3.5 AUC points below the best intermediate layer |
| Rank compression (Song & Lee, arXiv 2608.21244, abstract) | image and video anomaly benchmarks | decoded answer vs probability-weighted answer | probability weighting gains 7.66 to 19.95 points on average; most of the gap comes from ties among decoded answers |
| Ganesh 2026 (arXiv 2608.27879, paper) | UCF (165), XD (436) | separation kept before onset (length-matched) | whole-frame context (DINOv2) keeps 83 to 91 % (UCF) and 70 to 71 % (XD); person crops 50 to 82 %; box geometry 39 to 84 %. Watermarks and title cards found in pre-onset windows |
| VLEP (Lei et al., EMNLP 2020, paper) | TV shows and vlogs | which of two future events is more likely | future-text only 58.1 %, + video 59.0 %, + dialogue 66.6 %, + both 67.5 %; humans 76.3 % with dialogue only, 90.5 % with video and dialogue |

### 2.2 What it means
- **It is the only family with any measured 30 to 60 s signal on violence footage**, and in this repo it is the
  only control above the position null. It is also available on silent CCTV.
- **The signal is weak and fragile:** 0.62 to 0.69 AUC at 60 s after fine-tuning with a memory bank; zero-shot
  VLMs barely use temporal order (RiskCueBench shuffling result), lose accuracy as the gap grows, and warn about
  1.7 s before the event in streaming use. Part of the separability is provenance (Ganesh). The repo teacher is
  three times better on official onsets and on movies than on UCF CCTV, which is consistent with both a real
  effect and a shortcut.
- **How it is read matters as much as which model:** decoded yes/no or 0-1 answers throw away ranking information
  (7.7 to 20 points); intermediate hidden states beat the last layer and beat caption-then-LLM pipelines by a wide
  margin on XD. The repo's teacher uses decoded scores (`b_teacher` scores are discrete: 0, 0.1, 0.5, 0.8, 0.9).
- **Models do not yet extract the visual future signal that humans use:** in VLEP video adds 0.9 points for the
  model and 14 points for humans. In RiskCueBench humans score 0.97 F1 and the best VLM 0.67. There is headroom in
  this modality, but it needs supervision (fine-tuning or probes), not more prompting.
- **Long memory helps at long horizons** (STCH 0.601 to 0.692 at 60 s with the memory bank).

---

## 3. Speech content (what is said)

### 3.1 What has been measured

| Study | Data | Task and lead | Number |
|---|---|---|---|
| Rho et al., PNAS 2023 (paper, PMC10266001) | 577 bodycam traffic stops of Black drivers (81 escalated: 76 searches, 56 handcuffings, 24 arrests) | escalation predicted from the officer's first 45 words (about the first 27 s) | DeBERTa-v3 70.83 % accuracy (95 % CI 68.90 to 72.60) on a balanced test set; escalated stops open with commands and without a stated reason; listeners predicted escalation for 84.8 % of clips with orders and no reason vs 36.6 % with a reason and no orders |
| Zhang et al., ACL 2018, Conversations Gone Awry (paper) | Wikipedia talk pages, balanced pairs | will this civil-looking start end in a personal attack | pragmatic features 61.6 %, toxicity + pragmatic 64.9 %, bag of words at chance, **humans 72 %** |
| CRAFT, Chang and Danescu-Niculescu-Mizil, EMNLP 2019 (paper) | CGA-Wiki, Reddit CMV | online forecast as the conversation unfolds | accuracy 66.5 % / F1 69.8 (Wiki), 63.4 % / 67.9 (CMV); first warning on average 3 comments before the attack (4 on CMV); at least 3 h of warning for over 50 % of conversations, at least 12 h for 39 % |
| Forecasting through generation (arXiv 2504.08905, paper) | CGA-Wiki | same | Mistral + sampled futures 71.4 % acc / 73.6 F1; GPT-4o few-shot 65.1 % / 71.8 (recall 88.8, over-predicts) |
| LLM derailment prediction (Findings EMNLP 2025, abstract) | CGA, CMV | zero-shot prompting | state of the art and earlier forecasts than fine-tuned baselines; misses "slow-burn" derailments that start politely |
| Lefter et al. 2012 (paper, TNO PDF) | 43 min of acted train scenarios, 3-level aggression | prosody vs prosody + aggression keywords (6 classes: cursing, negative emotion, actions, context, nonverbal, positive) | keywords add +10 points weighted accuracy, +16 points for the high-aggression class; best "impact length" of a keyword is 11 segments of 2 s (about 22 s) |
| ComParE 2021 Escalation Sub-Challenge (paper, arXiv 2102.13468) | Dutch actors: train with service desk, test on trains; 3 levels | escalation level of a 5 s segment | Dutch BERT linguistic features 36.8 to 47.2 % UAR vs acoustic openSMILE / bag-of-audio-words / DeepSpectrum 49.4 to 59.8 % (auDeep 35 to 52 %; chance 33.3 %); low vs high almost never confused |
| DeToxy, Interspeech 2022 (paper) | 2 M English utterances (DeToxy-B subset) | toxic or not | text classifier trained on Civil Comments: macro F1 0.801 on gold transcripts, 0.674 to 0.682 on ASR transcripts (ASR WER 27 to 43 %); trigger-term-free test set 0.26 to 0.28; end-to-end wav2vec 2.0 0.877 |
| Speech toxicity with paralinguistics (arXiv 2605.15984, paper) | 60.8 h English, 7 categories incl. threats | audio-only toxicity | best model 80.0 % acc / 69.7 macro F1; Gemini-2.5-Flash 70.8 / 57.6; GPT-4o Audio 61.9 / 39.9; Qwen2-Audio 55.2 / 19.2 |
| Careless Whisper, FAccT 2024 (abstract) | 13,140 segments | Whisper hallucinations | about 1 % of transcriptions contain invented phrases; 38 % of those carry explicit harms (violence among them); more frequent with long non-vocal stretches |
| BWC ASR (arXiv 2607.24958, abstract; weak) | one 484-word bodycam sample | WhisperX WER | 32.4 %, mostly from noise, overlap and interruptions |

### 3.2 What it means
- **Speech content is the only channel with measured predictive value tens of seconds to hours ahead**, but the
  outcomes are arrests, personal attacks and rated aggression, not punches. The police result is the closest
  analogue to the target (bodycam, real encounters, about 27 s of speech, outcome later in the stop).
- **The useful content is dialogue acts, not slurs:** commands without reasons, challenges, refusals, threats,
  cursing aimed at a person. Generic toxicity classifiers trained on web text lose about 0.12 F1 on ASR transcripts
  and collapse to 0.26 to 0.28 F1 when toxicity has no trigger words. The repo's `speech_tox` (Whisper + detoxify)
  therefore measures the wrong thing for anticipation; keep it, but add an escalation-act scorer.
- **For the current escalation level, words are weaker than voice** (ESS: 37 to 47 % vs 60 % UAR). Content and
  prosody are complementary: content says where a conversation is heading, prosody says how heated it is now.
- **Failure modes on target audio:** ASR WER about 25 to 36 % on noisy far-field speech (Whisper on CHiME-6 / AMI,
  secondary) and about 32 % on a bodycam sample; shouted and distant speech degrade ASR further (BERSt, abstract);
  hallucinations are more likely in long non-speech stretches, which describes most CCTV-like audio. Gate every
  speech feature with VAD and a speech-present mask (FEATURES.md already does this for `speech_tox`).

---

## 4. Conversational dynamics and vocal arousal

### 4.1 What has been measured

| Study | Data | Task and lead | Number |
|---|---|---|---|
| Kim, Yella, Valente, Interspeech 2012 (paper, Idiap PDF) | 45 French TV debates, 3109 clips of 30 s, 3 conflict levels | conflict level of the **current** clip vs the **next** clip from overlap, turn-taking and prosody | current clip WA 86.1 % / UA 71.1 % (chance 59.1 / 34.0); next clip WA 74.2 % / UA 37.7 %, with delta features UA 41.5 % (chance 58.4 / 33.3); escalation vs de-escalation vs constant UA 49.5 %. Total overlap correlates 0.77 with the current level and 0.37 with the next; minimum pitch during overlap -0.65 and -0.35 |
| ComParE 2013 Conflict Sub-Challenge (abstract, ISCA) | SSPNet Conflict Corpus, 1430 clips of 30 s | high vs low conflict | baseline 80.8 % UAR, winner 83.9 % |
| ComParE 2021 Escalation (paper) | see section 3 | 3 levels, cross-corpus | best test UAR 59.8 % (bag of audio words); openSMILE functionals 58.6 % |
| Lefter et al., Frontiers 2022 (paper, via WebFetch) | TR 534, SD 413, NAA 2417 samples (Dutch) | 3-level escalation, cross-corpus | up to 63.8 % UAR; non-actor test 50.3 %; F0 rises with escalation in every corpus, intensity in two of three (distant microphone in the third) |
| Kooij et al., CVIU 2016, CASSANDRA (paper, gavrila.net) | 25 acted runs at Amsterdam-Amstel station (1 to 2 min each), 13 aggression events | event detected within 10 s after its start; DBN fusion | audio alone 7 / 13 events with 2 false positives; video limb motion alone 4 / 13 with 5; motion + person interaction 10 / 13 with 5; all cues 11 / 13 with 5. Frame-level RMSE vs annotated level: audio 0.223, video 0.248, all 0.188 |
| Zajdel et al., AVSS 2007, CASSANDRA (paper) | 13 clips of 100 to 150 s | same | audio "scream-like" detector + video articulation; fusion raised detection without extra false alarms; borderline scenarios (lively argument, happy supporters) caused the errors |
| Wojda, Baucom et al. 2021 (abstract, PubMed 34472937) | 149 couples, two problem-solving discussions | vocal F0 trajectories by IPV level | men show linear increases in F0 within the conversation at higher levels of intimate partner violence |
| Shouted-speech detection (Pohjalainen et al. JASA 2013, abstract; Interspeech 2011, secondary) | read speech in noise | shouted vs normal | machine matches humans even in the noisiest conditions; EER about 3.0 to 4.4 % in babble at 0 to -10 dB (secondary) |
| Speech emotion recognition (Wagner et al., TPAMI 2023, secondary) | MSP-Podcast | arousal regression | CCC about 0.745 |
| van Hengel and Andringa, AVSS 2007 (abstract; vendor-affiliated) | Groningen city centre deployment, about 1400 police-annotated recordings | verbal aggression (shouting) alarms | "only a few false alarms per microphone per year" and no missed incidents |
| Repo audio sanity check ([../FEATURES.md](../FEATURES.md) audio section) | 21 XD B1/B5 clips with first onset at least 30 s in, change vs own [-60, -30) s baseline | trend before onset | loudness +2.3 dB at [-30, -10), +2.7 dB at [-10, 0), up in 81 % of clips; +7.5 dB after onset. F0 flat until onset (+35 Hz after). Voice arousal +0.02, emotion2vec anger does not rise |

### 4.2 What it means
- **Prosody and overlap are reliable for the current state of a conversation (71 to 84 % UAR) and weak but above
  chance for the next 30 s (38 to 42 % UA vs 33 %).** That is the only direct lead-time measurement for a vocal
  channel, and it shows the typical shape: most of the signal is "now", a small part leaks forward, and change
  (delta) features carry the forward part.
- **The forward part lives in trends:** the next-clip gain from delta features, the linear F0 increase in violent
  couples, the 30 s loudness drift in the repo clips. A 1 Hz model needs explicit slope features over 15, 30 and
  60 s and a memory of 60 to 120 s, not just the current window.
- **Shouting is easy to detect but not specific:** the train-station false alarms came from happy supporters,
  lively arguments and shouting from outside the scene. Most raised voices never lead to violence (only 130 of
  3109 debate clips were high conflict). This is a risk modulator, not an alarm.
- **Off-the-shelf emotion models are the weakest part of this channel** on movie and soundtrack audio (repo:
  emotion2vec anger flat). Loudness, F0 and overlap trends are cheaper and moved more.

---

## 5. Non-speech sound events (shout, yell, scream, crowd, impact, gunshot)

| Study | Data | Number |
|---|---|---|
| Wu et al., ECCV 2020, XD-Violence (paper, arXiv 2007.04687) | XD test, per-class AP | Fighting: audio 85.04, RGB 85.97, audio + RGB 88.02. Shooting 71.53 / 83.51 / 90.30. Riot 65.07 / 70.54 / 76.42. Abuse 76.48 / 90.10 / 83.43. Audio + RGB beats RGB by 3.2 AP overall; audio + flow beats flow by 3.5 AP |
| MSBT (arXiv 2405.05130, paper) | XD detection | RGB + audio 82.54, RGB + flow 80.68, audio + flow (no RGB) 77.47, all three 84.32 AP |
| AViS-Mamba (arXiv 2604.03329, paper) | NTU CCTV-Fights, DVD (audio-valid splits) | NTU-CCTV: audio-only 72.82 % acc, video-only 83.99 %, AV 88.59 %. DVD: 62.61 / 71.82 / 75.74 % |
| Penchala et al. (arXiv 2609.40014, paper) | 443 XD pre-incident clips, windows of at most 4 s, 91 test clips | audio-only (STFT CNN) AUC 0.788 |
| MACIL-SD (arXiv 2207.05500, paper) | XD | modality asynchrony: in abuse "the scream occurs afterward" |
| MIVIA audio events (secondary) | screams, gunshots, glass, 5 to 30 dB SNR | recognition 86.5 to 96.5 % |
| Repo sanity check (n = 21, [../FEATURES.md](../FEATURES.md)) | AST logits vs own baseline | Shout -0.12 at [-30, -10), +0.20 at [-10, 0), +0.50 after onset; Crowd -0.06 / +0.49 / +0.99; Screaming flat until onset |

Verdict: strong for detection (fighting is as audible as it is visible), a short-lead cue (last 10 s) at best.
Its main value for anticipation is indirect: an earlier shout or scuffle sets the risk for the next minutes
(section 8), and an auxiliary "violence now" head trained on these streams stabilizes the encoder.

---

## 6. Pose, tracks and interaction geometry

| Study | Data | Task and lead | Number |
|---|---|---|---|
| Penchala et al. (paper) | XD pre-incident, at most 4 s | 12 kinematic descriptors from YOLO-pose -> temporal transformer | AUC 0.862, balanced acc 76.1 % |
| Ganesh (arXiv 2608.27879, paper) | UCF 165, XD 436 | full-video separation by representation | UCF probe: box geometry G7 0.724; pose P7 0.580, P11 0.624, PR4 0.643, learned joints 0.631 (P7 significantly worse than G7). Retained before onset (length-matched): G7 79 / 84 % (UCF probe / MIL), 46 / 39 % (XD) |
| Blunsden and Fisher, VISAPP 2009 (paper) | BEHAVE, CAVIAR (acted) | 4 classes incl. "pre-fight" | pre-segmented: pre-fight recall 0.88. Continuous video (window around the frame): BEHAVE pre-fight recall 0.20 (0.42 called normal, 0.38 called fight); CAVIAR 0.00 |
| Kooij et al. 2016 (paper) | train station | add a person-interaction cue to limb motion | 4 / 13 -> 10 / 13 events detected |
| Ke et al., IJCAI 2017 (paper, arXiv 1705.09467) | UT-Interaction #2 (punch, kick, push, ...; chance 16.7 %) | early recognition with 10 % of the act observed | 36.7 % (RGB) / 58.3 % (flow) vs 76.7 / 91.3 % for the full act |
| Repo sanity check ([../FEATURES.md](../FEATURES.md) pose section, n = 19) | XD B1/B5 | change vs own [-60, -30) s baseline | persons +0.53 at [-30, -10), +0.89 at [-10, 0); close pairs +2.4 / +2.5; arm-raise rate +0.035 / +0.062; wrist speed -0.02 / +0.08; mutual facing and face-off flat (the version-1 "face-off" signal was an artefact) |
| Troscianko et al., Perception 2004 (secondary, via Blunsden 2009) | 100 real UK CCTV clips, 18 led to crime | humans predict from silent footage | 80 % of criminal incidents and 65 % of normal-but-similar incidents judged correctly; gestures and body position were the cues |

Verdict: the geometry of a group (how many people, whether they gather, how close, how fast they close) carries
pre-onset signal and works on silent CCTV; fine-grained joint descriptors add little over boxes (Ganesh), and
"pre-fight" posture classifiers that look good on segmented clips fall apart on continuous video (Blunsden). Lead
is mostly under 10 s; gathering starts earlier (repo: 10 to 30 s). Humans predict incidents from silent CCTV, so
the visual channel has signal that current features do not capture. Keep `pose_interact`, weight box-level and
group features over joint angles, and add 30 to 120 s trend features.

---

## 7. Motion, optical flow and camera behaviour

| Study | Data | Number |
|---|---|---|
| Hassner et al., Violent Flows, CVPRW 2012 (paper) | real crowd-violence clips (average 3.6 s) | ViF 81.30 % acc, AUC 0.85 (LTP 71.5 %, HOG 57.4 %); a detector of outbreaks, not a predictor |
| MSBT / Wu 2020 (paper) | XD | flow adds less than audio to RGB (80.68 vs 82.54 AP) |
| TeSTra / LSTR (secondary) | THUMOS'14 action anticipation | mAP 66.2 at 0.25 s -> 48.9 at 2.0 s (TeSTra); 60.4 -> 45.7 (LSTR): steep decay within 2 s |
| NWPU Campus anomaly anticipation (via [literature.md](literature.md) 3.1) | campus CCTV | frame AUC 71.3 at 0.5 s -> 69.1 at 3 s; humans 90.4 at 3 s |
| Repo sanity check ([../FEATURES.md](../FEATURES.md) motion section) | 21 XD B1/B5 | flow +0.009 at [-30, -10), +0.020 at [-10, 0), +0.040 after onset; camera shake up in 79 % of clips in the last 10 s; cuts become rarer before onset (movie editing) |

Verdict: detection-strong, anticipation-weak, and the most shortcut-prone stream (camera shake, cut rate and
letterboxing identify phone, CCTV and movie footage). Phone filmers start shaking because they expect a fight: a
real precursor in phone video that will not transfer to fixed CCTV. Report it per source and include it in
shortcut audits.

---

## 8. Event history (earlier incidents in the same scene)

| Study | Data | Number |
|---|---|---|
| Hawkes point process (arXiv 2503.15821, paper) | aggression onsets of psychiatric inpatient youths with autism; **only the timestamps of earlier onsets** | ROC-AUC 0.85 / 0.83 / 0.81 / 0.80 / 0.78 / 0.78 at 1 / 5 / 10 / 15 / 20 / 25 min; Poisson baselines about 0.50 |
| Wearable physiology (Imbiriba et al., JAMA Netw Open 2023; Goodwin 2019, via [literature.md](literature.md) 2.8) | autistic psychiatric inpatient youths (70 per press coverage; check the paper) | AUROC 0.80 at 3 min (Imbiriba); 0.71 (population) / 0.84 (person-specific) at 1 min (Goodwin) |
| VA30 structure (repo) | 551 eligible events | 239 follow earlier violence in the same clip; `b_memory` (perfect detector with memory) scores 0.72 on the all-segment far AUROC and 0.50 on the pre-violence headline |

Verdict: aggression clusters in time, and the timing of earlier incidents alone predicts the next one minutes
ahead about as well as physiology does. In video, "an earlier scuffle, shove or shouting burst in this scene" is a
legitimate long-lead feature that costs nothing (time since the last "violence now" or "shouting" alarm, count of
alarms in the last 5 and 15 min). VA30's headline masks it on purpose; report it separately as an operational
metric so it is neither hidden nor mistaken for anticipation.

---

## 9. Face and facial expression

| Evidence | Number |
|---|---|
| Penchala et al. (paper): "face" stream | AUC 0.952 (Swin-T), the strongest single stream at 4 s or less. **The crop is the upper part of the largest person's box (or face keypoints), resized to 224 x 224; no face detection rate is reported.** It is a person-appearance and posture stream more than an expression stream. Labels (No / Low / Medium / High risk) were assigned from the same visual, motion and audio cues |
| Low-resolution FER (arXiv 2511.05938, secondary summary of its tables) | FER2013 at 12 x 12 px: ResNet-50 about 50 %, best about 57 %; RAF-DB at 14 x 14 px: about 71 to 76 % vs about 92 % at full resolution (POSTER-Var, secondary) |
| Target resolution (repo, `media_probe.json`) | UCF-Crime: all 2950 files 320 x 240. XD: 640 x 266 to 640 x 360. UBI: 640 x 360. YouTube hold-out: 80 of 92 at 640 x 360 |
| Face size estimate (assumption: head = body height / 7.5, face = 2/3 of head) | a person half the frame tall: about 11 px face in UCF, about 16 px at 360p; a person a quarter of the frame tall: about 5 to 8 px |
| Facial precursors of violence | no computational study found that measures facial anger or contempt 10 s or more before physical violence; Matsumoto et al. 2015 (abstract) found rising contempt and disgust in **speeches** months before group violence |

Verdict: skip facial expression for CCTV (faces sit in the 5 to 16 px regime where posed-benchmark accuracy already
drops 20 to 40 points). For phone and bodycam close-ups, a face-size-gated FER stream is optional and should be
tested only after the streams above. Do not read the Penchala "face" result as evidence for expressions.

---

## 10. Crowd behaviour

- Detection: ViF 81.3 % / AUC 0.85 on crowd violence (section 7); protest risk in RiskCueBench is hard for VLMs
  (F1 0.44 to 0.67 vs humans 0.97).
- Precursors: the repo pose check shows gathering (+0.5 to +0.9 persons) before onset; Kooij's false alarms came
  from happy supporters and crowds ("crowds yield other proximity features").
- No study found that measures crowd agitation at a 30 s lead. Treat crowd convergence and encirclement as part of
  the group-geometry stream (section 6) and expect high false-alarm risk at markets, sports and nightlife.

---

## 11. Multimodal fusion results

| Setting | Best unimodal | Fusion | Gain |
|---|---|---|---|
| XD-Violence detection (Wu 2020) | RGB | RGB + audio | +3.2 AP; fighting 85.97 -> 88.02 |
| XD-Violence detection (MSBT) | RGB + flow 80.68 | RGB + audio + flow 84.32 | +3.6 AP; dropping RGB costs 5 to 7 AP |
| NTU-CCTV / DVD detection (AViS-Mamba) | video 83.99 / 71.82 % | AV 88.59 / 75.74 % | +4.6 / +3.9 points; visual-steered fusion is robust when audio is missing |
| Pre-incident, at most 4 s (Penchala) | face crop 0.952 | audio + face + motion 0.964 | +0.012 AUC (tiny test set; audio + face ties the triple at 0.961) |
| Train station aggression events (Kooij) | audio 7 / 13 | all cues 11 / 13 | +4 events at +3 false positives |
| Train aggression labels (Lefter 2012) | prosody | prosody + keywords | +10 points WA |
| Human annotation (Lefter 2012) | | | audio-only, video-only and audio-visual labels disagree on 46 % of segments; "semantics" (what the scene means) explained 61 % of disagreements |

Fusion gives consistent 3 to 5 point gains for detection; video carries most of the weight. The interesting result
for anticipation is Lefter's: the multimodal judgement often differs from either unimodal one, and the reason is
scene semantics, which is what VLM features and speech content try to capture.

---

## 12. Does audio or speech carry earlier signal than vision?

Evidence that audio or speech leads:
1. Train-station scenarios (Kooij): audio alone detected more aggression events (7 / 13, 2 false positives) than
   video motion alone (4 / 13, 5 false positives); in the build-up scenarios supporters "intimidate each other"
   (shouting) before the fight.
2. Lefter 2012: an aggression keyword keeps predicting aggression for about 22 s.
3. Repo (n = 21 movie and compilation clips, no significance test): loudness rises from 30 s out in 81 % of clips,
   while person count, arm raises, wrist speed and flow rise mostly in the last 10 s.
4. Police stops: the first 27 s of officer speech predicts a later escalation.
5. VLEP: models extract far more future signal from dialogue (+8.5 points) than from video (+0.9).
6. XD-Violence: for fighting, audio alone is as good as RGB (85.0 vs 86.0 AP), so audio is not a weak channel
   when people fight in earshot.

Evidence that vision leads or dominates:
1. The only pre-incident tri-modal comparison (Penchala, at most 4 s) ranks person appearance 0.95 > pose motion
   0.86 > audio 0.79.
2. Real-world AV detection: video-only beats audio-only by 11 points (NTU-CCTV) and 9 points (DVD).
3. Abuse: the hit comes first and the scream follows (MACIL-SD).
4. Humans predict incidents from silent CCTV from gestures and body position (Troscianko 2004).
5. The only control in VA30 that anticipates at 30 s is visual (the VLM teacher), although no audio or speech
   control has been scored on VA30 yet.

Verdict: **plausible but unproven.** The verbal channel is where escalation becomes observable first in settings
that record speech (stops, trains, debates, online threads), and its predictive window is the longest measured
(tens of seconds to hours). The visual channel is stronger near onset and is the only one available on CCTV. The
decisive test is cheap once the audio extractors finish: per-lead-bucket AUROC of loudness, overlap and
speech-act features vs visual features on VA30's audio-present real-world subset (XD YouTube 162 events, UCF with
audio 47 events) and on the 92-video YouTube hold-out.

---

## 13. Observability in the target footage

| Footage | Audio | Speech | Faces | People in view | Notes |
|---|---|---|---|---|---|
| Fixed CCTV (stores, bars, streets) | almost never (UBI 0 / 1000 videos; UCF 490 / 1879 non-silent, much of it not scene audio) | no | 5 to 16 px | yes, small, often overhead | the modality set is vision only: scene semantics, group geometry, motion |
| Police bodycam | yes after activation; **the pre-event buffer is video-only by default** (Axon Body 3/4: 30 s default, 0 to 120 s configurable; NYPD 60 s video-only) | yes, the officer's voice dominates | close-ups of the subject | the wearer is never visible | the 30 s before activation, which is exactly the lead window, often has no audio |
| Dashcam / road rage | often disabled (all-party consent states), or replaced by music in compilations | in-cabin only | rarely | the other driver, through glass | vehicle motion (tailgating, cut-ins) is the visual precursor |
| Phone video of confrontations | yes, but often music overlays, wind, or muted | yes, shouted and overlapping | close-ups possible | yes | filming starts because a fight is expected: a selection effect and a camera-behaviour shortcut |

VA30 composition by audio (repo, `benchmark/videos.csv`, eligible30 events): silent CCTV 199 (UBI 102, UCF 97);
UCF with a non-silent track 47; XD movie 143 (soundtrack, foley, edited dialogue); XD YouTube 162 (3 of them
tagged as CCTV). Clean negatives: 132.5 h silent (UBI 68.4, UCF 64.1), 171.0 h with audio (UCF 28.5, XD movie
44.2, XD YouTube 98.3). So an audio or speech stream can improve at most about 210 real-world events, and movie audio must be audited for soundtrack shortcuts (music swells
before fights). The YouTube hold-out has audio in 90 of 92 videos and is the right place to test speech channels.

---

## 14. Recommendations for this repository

Ordered by expected gain at 30 s per GPU-hour, given that extraction of the nine current extractors is already
running.

1. **Upgrade the VLM channel before adding new backbones** (rank 1). Replace decoded `violent.score` with
   (a) probability-weighted scores over a short checklist of escalation questions ("two people arguing face to
   face", "someone pointing or squaring up", "a crowd gathering around two people", "someone being held back",
   "aggressive gesture toward another person", "people running away"), and (b) a linear or small-MLP probe on an
   intermediate hidden layer (HiProbe's layer analysis peaked around layer 20 on XD, and its last layer was 3.5
   AUC points worse; sweep layers on a validation fold), sampled every 2 to 4 s and held. Qwen2.5-VL-7B is already local. Train the probe on VA30 folds with the hazard targets; compare against
   B-teacher on the time-matched far AUROC and Success@30 vs the position null.
2. **Add an escalation-act scorer on transcripts** (rank 2). Keep `speech_tox`, but add per-tick scores from a
   local text LLM over the last 60 to 120 s of WhisperX transcript (VAD-gated, hallucination-filtered): commands,
   refusals, challenges and insults directed at a person, threats, de-escalation attempts, and "who talks to whom"
   when diarization allows. Detoxify-style toxicity is the wrong target (section 3.2).
3. **Add conversational-dynamics features** (rank 3): overlapped-speech ratio (pyannote overlap detection),
   speech rate, turn length, shout probability, and slopes of loudness, F0 and overlap over 15, 30 and 60 s.
   `audio_dsp` already has loudness and F0; the trend features are cheap CPU work on existing outputs.
4. **Make trends and memory first-class** in the model: explicit 15 / 30 / 60 s slope features for every stream
   and a 60 to 120 s causal memory. Every lead-time result above (Kim deltas, STCH memory, F0 trajectories)
   says the forward signal is in change, not level.
5. **Add event-history features** (rank 5) from the model's own detection head and sound events: time since the
   last "violence now" or shouting alarm, number of such alarms in the last 5 and 15 minutes. Report their effect
   on the all-segment metrics and deployment metrics, not on the pre-violence headline.
6. **Keep pose, but favour group geometry** (rank 4): people count, gathering rate, minimum distance, closing
   velocity and encirclement, plus their trends, over joint-angle descriptors (Ganesh: boxes >= pose).
7. **Use sound events, motion and V-JEPA 2 for the "violence now" auxiliary head and the last 10 s**, not as the
   source of 30 s lead. Keep camera-motion columns out of the anticipation head until a per-source audit passes.
8. **Deprioritize facial expression and emotion2vec anger** for this capstone; if added, gate FER on a face size of
   at least 48 px and evaluate only on phone and bodycam footage.
9. **Measure the lead-time profile of every stream before fusing** (CPU, cheap, as soon as features exist): for
   each feature or single-stream model, AUROC per lead bucket (0-10, 10-20, 20-30, 30-60, 60-120 s) against clean
   negatives, source x time matched, per source, and on the audio-present subset. This answers the
   audio-vs-vision question with the repo's own data and tells which streams deserve a place in the 30 s head.
10. **Evaluate speech and audio streams where they can matter:** XD YouTube, UCF-with-audio, and the hand-annotated
    YouTube hold-out; report the silent-CCTV subset separately so a speech gain is not diluted or overstated.

---

## 15. Caveats and open questions

- **No study measures audio vs video lead time on the same real violence footage.** Every "audio leads" item is
  either acted (trains), a non-violent outcome (stops, debates, online threads), or the repo's 21-clip check.
- **Speech-content evidence is mostly English and Dutch, mostly non-violent outcomes,** and the police result is
  limited to Black drivers in one city (under 1 % of escalated stops involved other drivers).
- **Several key numbers come from 2026 preprints without code** (Penchala, Ganesh, RiskCueBench, PaSBench,
  rank compression, HiProbe). STCH and AssistPDA numbers were read for [literature.md](literature.md), not
  re-read here.
- **Penchala's labels** were assigned by humans looking at the same appearance, motion and audio cues the models
  use, on 443 XD clips (91 test); its numbers are an upper bound for 4 s windows, not an anticipation result.
- **Movie audio** (XD movie clips) has soundtrack cues that rise before fights; any audio gain on `xd_movie` must
  be checked against XD YouTube and the hold-out.
- Open: does the VLM teacher's 30 s signal survive the HIVAU onset spot-check (official onsets 20.0 % vs HIVAU
  6.5 %)? Do loudness and overlap trends separate 30 to 60 s pre-onset ticks from clean negatives on real-world
  audio? Can a transcript LLM score escalation at 1 Hz on noisy phone audio without hallucination-driven alarms?

---

## 16. References

Repository
- VA30 benchmark and controls: [../VA30.md](../VA30.md); baseline: [../BASELINE.md](../BASELINE.md); feature
  sanity checks: [../FEATURES.md](../FEATURES.md); earlier research: [literature.md](literature.md),
  [backbones.md](backbones.md), [datasets.md](datasets.md).

Scene semantics, VLMs, anticipation
- Wang et al., Streaming Video Crime Anticipation with Spatio-Temporal Causal Reasoning (STCH), CVPR 2026: https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Streaming_Video_Crime_Anticipation_with_Spatio-Temporal_Causal_Reasoning_CVPR_2026_paper.html
- AssistPDA, arXiv 2503.21904: https://arxiv.org/abs/2503.21904
- Luo et al., RiskCueBench, arXiv 2601.03369: https://arxiv.org/abs/2601.03369 (HTML v2: https://arxiv.org/html/2601.03369v2)
- PaSBench-Video, arXiv 2606.02443: https://arxiv.org/html/2606.02443
- HiProbe-VAD, arXiv 2507.17394: https://arxiv.org/html/2507.17394v1
- Song and Lee, A VLM Answer Is Not an Anomaly Score: Rank Compression, arXiv 2608.21244: https://arxiv.org/abs/2608.21244
- Ganesh, Pre-Event Separability in Weakly-Supervised Violence Detection, arXiv 2608.27879: https://arxiv.org/html/2608.27879
- Lei et al., What is More Likely to Happen Next? (VLEP), EMNLP 2020: https://aclanthology.org/2020.emnlp-main.706.pdf
- Zero-shot VAD comparison (LAVAD, VERA), NeurIPS 2025, arXiv 2511.00962: https://arxiv.org/pdf/2511.00962
- Benschop et al., Evaluation of Vision-LLMs in Surveillance Video, arXiv 2510.23190: https://arxiv.org/abs/2510.23190

Speech content and conversations
- Rho et al., Escalated police stops of Black men are linguistically and psychologically distinct in their earliest moments, PNAS 2023: https://pmc.ncbi.nlm.nih.gov/articles/PMC10266001/
- Zhang et al., Conversations Gone Awry, ACL 2018: https://aclanthology.org/P18-1125/
- Chang and Danescu-Niculescu-Mizil, Trouble on the Horizon (CRAFT), EMNLP 2019: https://aclanthology.org/D19-1481/
- Forecasting Conversation Derailments Through Generation, arXiv 2504.08905: https://arxiv.org/html/2504.08905
- Olpadkar et al., Can LLMs Be Efficient Predictors of Conversational Derailment?, Findings EMNLP 2025: https://aclanthology.org/2025.findings-emnlp.816/
- Ghosh et al., DeToxy, Interspeech 2022: https://www.isca-archive.org/interspeech_2022/ghosh22b_interspeech.pdf
- Beyond Content: speech toxicity with paralinguistic cues, arXiv 2605.15984: https://arxiv.org/html/2605.15984v1
- Koenecke et al., Careless Whisper, FAccT 2024: https://arxiv.org/abs/2402.08021
- Speech intelligence on bodycam audio (WhisperX WER), arXiv 2607.24958: https://arxiv.org/pdf/2607.24958
- Automatic conflict detection in police body-worn audio, arXiv 1711.05355: https://arxiv.org/abs/1711.05355

Voice, conflict and aggression
- Kim, Yella, Valente, Automatic detection of conflict escalation in spoken conversations, Interspeech 2012: https://publications.idiap.ch/downloads/papers/2014/Kim_INTERSPEECH_2012.pdf
- Kim, Valente, Vinciarelli, Annotation and detection of conflict escalation in political debates, Interspeech 2013: https://www.isca-archive.org/interspeech_2013/kim13e_interspeech.pdf
- Schuller et al., INTERSPEECH 2013 ComParE (Conflict sub-challenge): https://www.isca-archive.org/interspeech_2013/schuller13_interspeech.pdf
- Schuller et al., INTERSPEECH 2021 ComParE (Escalation sub-challenge), arXiv 2102.13468: https://arxiv.org/abs/2102.13468
- Lefter, Baird, Stappen, Schuller, A cross-corpus speech-based analysis of escalating negative interactions, Frontiers 2022: https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2022.749804/full
- Lefter, Rothkrantz, Burghouts, Aggression detection in speech using sensor and semantic information (TSD 2012): https://publications.tno.nl/publication/100144/GI8ACg/lefter-2012-aggression.pdf
- Lefter, Rothkrantz, Burghouts, A comparative study on automatic audio-visual fusion for aggression detection using meta-information, Pattern Recognition Letters 2013: https://www.sciencedirect.com/science/article/abs/pii/S0167865513000068
- Kooij, Liem, Krijnders, Andringa, Gavrila, Multi-modal human aggression detection, CVIU 2016: http://www.gavrila.net/Publications/cviu16.pdf
- Zajdel, Krijnders, Andringa, Gavrila, CASSANDRA: audio-video sensor fusion for aggression detection, AVSS 2007: http://gavrila.net/avss2007_cassandra.pdf
- van Hengel and Andringa, Verbal aggression detection in complex social environments, AVSS 2007: https://ieeexplore.ieee.org/document/4425279/
- Wojda, Baucom et al., IPV and couples' vocally encoded emotional arousal, 2021: https://pubmed.ncbi.nlm.nih.gov/34472937/
- Pohjalainen et al., Detection of shouted speech in noise: human and machine, JASA 2013: https://pubmed.ncbi.nlm.nih.gov/23556603
- Wagner et al., Dawn of the transformer era in speech emotion recognition, arXiv 2203.07378: https://arxiv.org/abs/2203.07378
- BERSt: distanced, emotional and shouted speech benchmark, arXiv 2505.00059: https://arxiv.org/abs/2505.00059

Audio-visual violence detection
- Wu et al., Not only Look, but also Listen (XD-Violence), ECCV 2020: https://arxiv.org/abs/2007.04687
- MACIL-SD, ACM MM 2022: https://arxiv.org/abs/2207.05500
- Multi-scale Bottleneck Transformer (MSBT), arXiv 2405.05130: https://arxiv.org/abs/2405.05130
- Aligning First, Then Fusing, arXiv 2501.07496: https://arxiv.org/abs/2501.07496
- AViS-Mamba, arXiv 2604.03329: https://arxiv.org/html/2604.03329v2
- Penchala et al., Can We Anticipate Violence?, arXiv 2609.40014: https://arxiv.org/html/2609.40014
- MIVIA audio events: https://mivia.unisa.it/datasets/audio-analysis/mivia-audio-events/ ; SNR study: http://doi.org/10.3390/electronics9101593

Pose, interaction, motion, crowds
- Blunsden and Fisher, Pre-fight detection, VISAPP 2009: https://homepages.inf.ed.ac.uk/rbf/PAPERS/sbvisapp09.pdf
- Ke et al., Predicting Human Interaction via Relative Attention Model, IJCAI 2017: https://arxiv.org/abs/1705.09467
- Hassner, Itcher, Kliper-Gross, Violent Flows, CVPRW 2012: https://talhassner.github.io/home/projects/violentflows/violent_flows.pdf
- Troscianko et al., What happens next? The predictability of natural behaviour viewed through CCTV cameras, Perception 2004: https://pubmed.ncbi.nlm.nih.gov/15035331/
- TeSTra, arXiv 2209.09236: https://arxiv.org/abs/2209.09236 ; LSTR, arXiv 2107.03377: https://arxiv.org/abs/2107.03377

Faces
- Low-resolution FER (GME-Net), arXiv 2511.05938: https://arxiv.org/pdf/2511.05938
- Matsumoto, Frank, Hwang, The role of intergroup emotions in political violence, CDPS 2015: http://davidmatsumoto.com/content/2015%20MF&H%20%20CDPS.pdf

Event history and physiology
- Temporal point process modeling of aggressive behavior onset (Hawkes), arXiv 2503.15821: https://arxiv.org/html/2503.15821 ; Scientific Reports 2026: https://www.nature.com/articles/s41598-026-46996-8
- Imbiriba et al., Wearable biosensing to predict imminent aggressive behavior, JAMA Netw Open 2023: https://pmc.ncbi.nlm.nih.gov/articles/PMC10739066/
- Timmons et al., Using multimodal wearable technology to detect conflict among couples, IEEE Computer 2017: https://sail.usc.edu/publications/files/timmons_chaspari_han_perrone_narayanan_margolin_2017.pdf

Observability
- Axon Body 4 operation (pre-event buffer without audio by default): https://www.axon.com/help/axon-body-4/cameras-and-sensors/body/operation.htm
- NYPD body-worn camera impact and use policy (60 s video-only buffer): https://www.nyc.gov/assets/nypd/downloads/pdf/public_information/post-final/body-worn-cameras-nypd-Impact-and-use-policy_2.4.26_final.pdf
- Dashcam audio consent overview (vendor blog, check statutes): https://www.getnexar.com/blog/dash-cam-audio-recording-laws
