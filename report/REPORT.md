# Anticipating Interpersonal Violence in Real-World Video at Least 30 Seconds Before Onset

**Lawrence Technological University (LTU) — Senior Capstone Report**

**Author(s): [PLACEHOLDER: student names]**

**Program, course, and advisor: [PLACEHOLDER]**

**Submission date: [PLACEHOLDER]**

Draft scope: the related work, problem definition, data, historical analysis, and evaluation protocol are drafted below. The new model, experiments, and conclusions about its performance remain explicitly marked placeholders. Repository facts use the October 14, 2026 snapshot at commit `9f14b92`; the implemented VA30 v2 protocol takes precedence over older planning notes. External references were opened and checked on October 14, 2026; verification details and corrections are in [CITATION_AUDIT.md](CITATION_AUDIT.md).

## Abstract

**[PLACEHOLDER — write after the new experiments are complete.]** State the problem of warning at least 30 seconds before physical violence, the evaluated data and onset sources, the causal model, and the evaluation protocol. Insert measured Success@30, actual test false alarms per hour, confidence intervals, and comparison with position controls. End with the supported conclusion and principal limitation. Do not substitute detection accuracy for anticipation or describe an unrun model as successful.

## 1. Introduction and Motivation

Recognizing a fight after physical contact begins can support incident review, but a useful warning must arrive early enough for a person to assess the situation and consider de-escalation. This capstone investigates whether real-world video contains evidence that can support a warning at least 30 seconds before interpersonal violence begins. The intended settings include fixed surveillance cameras, body-worn cameras, dashcams, and phone footage of confrontations. The 30-second interval is an engineering target for investigation; this report does not establish that it is sufficient for a safe intervention.

The central difficulty is separating evidence of an approaching incident from evidence that a video belongs to a violence dataset. A punch, impact sound, or person falling may support detection while offering no advance warning. A watermark, cinematic soundtrack, or short clip duration may correlate with violent videos while revealing little about escalation. A valid anticipator must use only past observations and generalize to unseen sources under a realistic false-alarm burden.

The repository's experimental history, covering approximately November 2025 through April 2026, motivates this distinction. Its strongest source-grouped historical MLP achieved test AUROC 0.622 for onset within 30 seconds, but its median accepted warning lead was 8.9 seconds. Other high scores were compromised by label misalignment, source leakage, future information, or permissive alarm metrics. These findings support a careful benchmark and a new research attempt, rather than a claim that the capstone goal has already been achieved. Historical evidence is documented in [BASELINE.md](../docs/BASELINE.md) and [EXPERIMENT_HISTORY.md](../docs/EXPERIMENT_HISTORY.md).

The report addresses three questions:

1. Can a causal model warn about eligible violent events at least 30 seconds before onset at validation-selected false-alarm targets?
2. Does its performance exceed controls that know only the footage source and elapsed clip time, and remain informative after matching those factors?
3. How do results change across real-world and movie footage, interpersonal classes, audio availability, and annotation sources?

The report contributes a synthesis of relevant research, an accountable account of prior failures, and a reproducible evaluation plan using the repository's implemented VA30 v2 benchmark. **[PLACEHOLDER: add the new model's actual contribution after implementation and evaluation.]**

## 2. Related Work

### 2.1. Violence Detection and Video Anomaly Detection

Violence detection asks whether an incident is occurring in observed footage. Early recognition uses an initial portion of an event that has already begun. Anticipation instead predicts a future onset from observations available before it. These tasks have different information constraints, even when they use the same videos.

Sultani, Chen, and Shah introduced UCF-Crime and a weakly supervised anomaly-detection approach using video-level labels [1]. Multiple-instance learning treats a video as a collection of snippets, enabling training without labeling every training frame. This setting is attractive when temporal annotation is expensive, but a model can succeed by recognizing an incident's most distinctive moments. It need not learn anything about the preceding non-violent interaction. Wu et al. extended this line with XD-Violence, which contains untrimmed video and audio, and a network modeling local and long-range snippet relationships [2]. Their contribution establishes the value of audiovisual evidence for detection, rather than a guaranteed warning interval.

Later work improves semantic and temporal representations. Pu et al.'s prompt-enhanced context features combine temporal context aggregation with semantic prompts for anomaly classes [3]. Such methods suggest useful encoders or auxiliary detection objectives. However, improved localization of visible anomalies cannot be treated as evidence of 30-second anticipation. Evaluation must exclude incident and aftermath observations from anticipation metrics and measure how early alarm episodes actually begin.

### 2.2. Anomaly and Crime Anticipation

Cao et al. introduced NWPU Campus with an explicit video-anomaly-anticipation task, alongside scene-dependent anomaly detection [4]. Their work emphasizes collecting the full development of events and uses a forward-backward scene-conditioned model. The benchmark shows why forecasting requires suitable footage: an anomaly defined merely by an unexpected object appearing may have no observable precursor. Its scope includes diverse anomalies, so performance is not automatically specific to interpersonal violence.

Majhi et al. address forecasting abnormal human behavior from normal interactions through the AHB-F dataset and Scene-Interaction-aware Transformer, SIaT [5]. AHB-F combines 1,470 untrimmed videos from existing sources. Its short and long indicators cover future seconds 1–3 and 4–8, respectively. The model incorporates scene and object interactions and uncertainty about future outcomes. These ideas are relevant to escalation modeling, while the evaluation interval is substantially shorter than the capstone target. AHB-F's reuse of existing datasets also makes source overlap relevant if it is later added to VA30.

Wang et al. study streaming crime anticipation with a spatio-temporal causal hypergraph and a memory bank [6]. Their Table 3 reports 60-second future-window AUC of 0.692 on UCF-Crime and 0.622 on XD-Violence in cross-domain evaluation; the UR-DMU detector comparison obtains 0.535 and 0.506. The primary paper's evaluation section specifies that AUC uses full videos, although its temporal-regression and classification tasks use pre-crime windows. Consequently, these values provide methodological context, not a directly comparable pre-onset-only VA30 score or a ceiling on achievable anticipation. The paper's causal terminology also does not by itself establish intervention-level causality.

AssistPDA unifies streaming anomaly prediction, detection, and analysis using a vision-language model and spatio-temporal relation distillation [7]. It introduces event-level prediction and an average advance-time measure. This supports investigating long context and explicit prediction supervision, but an advance-time average alone does not identify the fraction of events warned about at least 30 seconds ahead under a fixed false-alarm burden. VA30 therefore retains event sensitivity, warning lead, and false alarms per hour together.

### 2.3. General Action Anticipation and Temporal Representation Learning

Action anticipation provides useful architectural ideas without establishing that violence is predictable at the same horizons. Furnari and Farinella's rolling-unrolling LSTM separates summarizing observed history from modeling future actions, and uses attention to combine modalities [8]. Girdhar and Grauman's Anticipative Video Transformer jointly learns next-action prediction and future-feature prediction [9]. Future-feature supervision is particularly relevant when labeled violent lead-ins are scarce: it can use otherwise unlabeled training footage, provided the target future is used only during training and never enters inference inputs.

MeMViT retains and compresses historical representations to extend temporal support efficiently [10]. AFFT brings modalities together before final score prediction and evaluates audiovisual features for action anticipation [11]. These approaches motivate ablations of long memory and feature-level fusion. They do not guarantee that a longer observation window will help: irrelevant past activity, editing, and viewpoint changes can obscure an escalation sequence.

Recent pretrained encoders offer candidate representations. V-JEPA 2 learns predictive video representations and reports action-anticipation performance on EPIC-Kitchens-100 [12]. Perception Encoder demonstrates that intermediate visual features can support tasks beyond the final embedding's original objective [13]. PE-AV extends contrastive representation learning to audio, video, and text, with PE-A-Frame supporting finer temporal audio-text alignment [14]. Together, these works motivate testing motion, scene semantics, and sound as separate streams. Their published benchmark results do not establish performance on this report's violence-anticipation task, and no extractor is declared part of the new model until its implementation is documented.

### 2.4. Behavioral Science of Escalation

The behavioral motivation is that violence may be preceded by an evolving interaction state rather than an immediately recognizable physical action. Lefter et al. analyze escalating negative interactions across three Dutch speech corpora [15]. Their study relates higher escalation to changes in vocal pitch and intensity, while identifying dependence on corpus and recording conditions. This supports measuring vocal arousal and its trajectory. It does not establish that loudness or anger reliably predicts physical violence, or that such a signal is available 30 seconds in advance in street footage.

Kim, Valente, and Vinciarelli study annotations and automatic detection of conflict escalation in political debates using conversational and prosodic features [16]. Their work treats escalation as a change across interaction segments, helping distinguish a trajectory from a static emotion label. Political disagreement and role-played aggression differ substantially from actual assaults; transfer must therefore be tested rather than assumed.

Based on this literature and the repository's feature research, candidate visual hypotheses include closing interpersonal distance, sustained face-offs, raised arms, crowd convergence, and third-party intervention. These are proposed measurable cues, not validated causal indicators or established 30-second predictors. Proximity, gesturing, and excited speech also occur in benign social activity. An escalation model requires negative examples of arguments that resolve peacefully, celebrations, sports, and ordinary crowd behavior. It must estimate uncertain event risk rather than label an individual as inherently dangerous.

### 2.5. Multimodal Aggression and Pre-Incident Risk Recognition

DeepSafety combines acoustic and transcript-derived features to identify verbal abuse in conversations [17]. Its reported multimodal F1 of approximately 0.85 concerns conversational violence detection, not future physical contact. MuTox provides multilingual audio toxicity annotations and a detector [18], offering a possible source of speech-related features. Toxicity, perceived aggression, and physical violence are different targets: a toxicity posterior cannot be interpreted directly as a probability of a fight.

Penchala et al. examine pre-incident Normal-versus-Risky recognition using facial appearance, aligned audio, and keypoint motion [19]. The preprint uses 443 samples with source-level separation and reports 96.38% ROC-AUC for its best three-modality configuration. Its input construction samples 32 frames at 8 fps, giving an observation interval of at most approximately four seconds. **Observation duration is not warning lead:** the result does not establish a 30-second gap between the final observed frame and incident onset. The task's risk labels and clip-level metrics also differ from VA30 event recall at fixed false alarms per hour. Nevertheless, the controlled modality comparisons support testing complementary cues under common splits.

### 2.6. Provenance Shortcuts, Annotation Resources, and the Research Gap

Ganesh tests what interaction representations measure by comparing geometry, pose, person appearance, and whole-frame context under common pipelines [20]. Substantial video-level separation remains before annotated onset, even after controlling sequence length. Inspected examples include editorial title cards and platform watermarks absent from normal surveillance footage. This is evidence that pre-event separability can mix event information with source information; it is not evidence that every pre-event signal is spurious. The study motivates source-matched negatives, grouped splits, and explicit shortcut controls.

Holmes-VAU's HIVAU-70k provides hierarchical annotations based on manual temporal segmentation and human or model-generated descriptions [21]. UCA supplies timestamped surveillance-video descriptions [22]. These resources can extend supervision beyond the official detection test annotations. Their availability does not independently validate the exact first physical-contact timestamp required here; the repository's train-onset audit remains necessary.

Survival analysis supplies a way to represent incomplete future observation. DeepHit learns an event-time distribution while accounting for censoring [23]. This motivates a possible time-to-event formulation for clips that end without an observed incident, without implying that DeepHit itself is a video-violence method. Accident anticipation work by Chan et al. likewise motivates reporting warning time alongside recognition quality [24].

The gap addressed by this capstone is therefore operational and empirical: establish whether causal evidence of escalation improves warnings at least 30 seconds ahead on unseen footage, beyond position and provenance, at a stated false-alarm rate. The reviewed studies use different populations, targets, horizons, and metrics. Their numbers cannot be pooled into a single expected accuracy or used to assert that this goal has already been solved.

## 3. Problem Definition

### 3.1. Inputs, Onset, and Prediction Target

Let `X[0:t]` denote video, available audio, and any derived features observed up to video time `t`. Let `t0(t)` be the start of the next annotated violent segment strictly after `t`. A causal model produces a risk score `r(t)` using only `X[0:t]`. The benchmark scores integer-second ticks `t = 1, ..., floor(duration)`.

For each horizon `H` in `{5, 10, 20, 30, 60}` seconds, the supervised label is:

```text
y_H(t) = 1 if 0 < t0(t) - t <= H; otherwise 0,
         subject to masking and right-censoring.
```

Predicting onset **within** the next 30 seconds differs from issuing a warning **at least** 30 seconds before onset. A model firing two seconds before a fight can classify a positive `y_30` tick correctly while failing the capstone's objective. Operational success requires an alarm episode whose start `a` satisfies `t0 - a >= 30` and lies in the event's permitted pre-onset acceptance window. Section 7 defines that window and its denominator.

The intended physical onset is the first observable physical act in an incident. Current benchmark boundaries are inherited from annotations and may not all satisfy this interpretation. The evaluation must preserve onset provenance and report uncertainty rather than silently equate every anomaly boundary with a verified first-contact frame.

### 3.2. Target Classes and Scope

VA30's broad violent class set is fighting, shooting, riot, abuse, assault, robbery, and arrest. Its `interpersonal` flag covers fighting, abuse, assault, riot, robbery, and arrest; shooting is reported as other violence under the implemented definition. These are dataset categories, not legal judgments or proof that every frame depicts harm. The broad benchmark supports comparison, while the interpersonal and real-world source breakdowns are essential to the capstone claim. Movie performance alone cannot substantiate real-world anticipation.

### 3.3. Why Thirty Seconds Is Difficult

Long non-violent lead-ins are scarce. The historical official annotation pool contained 112 videos whose first annotated anomaly began at least 30 seconds into the clip, including non-violence classes; it supplied only about 2.35 hours of pre-onset footage under the historical audit. The XD test median first onset was 5.9 seconds. VA30 v2 expands the available pool, but only 551 events meet its operational eligibility rule, and 374 use HIVAU train boundaries awaiting human validation. These counts refer to different snapshots and eligibility definitions, not contradictory measurements. See [VA30.md](../docs/VA30.md), sections 2–3.

The problem is also underdetermined: similar confrontations can escalate, remain verbal, or de-escalate. Camera operators may begin filming because they already expect an incident, and movie clips may contain scripted buildup. Titles, music, source, recording duration, and elapsed clip time can predict labels without localizing escalating behavior. Even a causal model can exploit these correlations. Finally, a model that remembers an earlier fight may rank later pre-event periods highly while never anticipating the first incident. The protocol must test these alternatives explicitly.

## 4. Data and VA30 Benchmark Construction

### 4.1. Datasets and Annotation Sources

The following table describes the committed VA30 v2 catalog, rather than the nominal sizes of the original public releases. Counts and durations come from [catalog_summary.json](../benchmark/catalog_summary.json) and [VA30.md](../docs/VA30.md). A catalog row is distinct from an evaluated video.

| Source kind | Catalog videos | Catalog hours | Evaluated violent videos | Clean-negative videos | Clean-negative hours | Eligible30 videos / events |
|---|---:|---:|---:|---:|---:|---:|
| UCF-Crime (`ucf`) | 1,879 | 126.62 | 378 | 915 | 92.55 | 127 / 144 |
| XD movie (`xd_movie`) | 2,194 | 64.62 | 870 | 1,131 | 44.23 | 135 / 143 |
| XD YouTube (`xd_youtube`) | 2,558 | 151.89 | 666 | 1,215 | 98.31 | 137 / 162 |
| UBI-Fights (`ubi`) | 973 | 76.06 | 215 | 749 | 68.44 | 69 / 102 |
| Curated YouTube hold-out (`yt`) | 92 | 9.96 | Excluded | Excluded | Excluded | Not established |
| **Total** | **7,696** | **429.15** | **2,129** | **4,010** | **303.53** | **468 / 551** |

Rounded source hours may not sum exactly to the rounded total. Clean-negative hours are catalog durations; masked portions are excluded from the evaluator's exposure denominator. All 92 curated videos are excluded from the evaluated columns. The remaining unevaluated catalog roles are 1,388 other anomalies, 48 unlabelled videos, 28 near-identical duplicates, and one contaminated clip.

UCF-Crime is an untrimmed surveillance anomaly dataset [1]; XD-Violence includes movie and diverse online footage with audio [2]. UBI-Fights supplies frame-level fight labels and long normal videos; its authors' repository describes 1,000 original videos, comprising 216 fight and 784 normal videos [25]. VA30's smaller UBI catalog and evaluated counts reflect deduplication and role assignment, including a nominal fight clip with no positive frames. The benchmark combines these releases for new grouped cross-validation; its folds are not the original public train/test splits, so its results must be labeled as VA30 results.

| Onset source | Temporal evidence and conversion | Violent videos | Eligible30 events |
|---|---|---:|---:|
| `official` | XD test `annotations.txt`: clip-relative frame indices divided by 24 fps, with `.mp4` suffixes normalized; UCF test `Temporal_Anomaly_Annotation_for_Testing_Videos.txt`: frame indices divided by video fps | 362 | 75 |
| `hivau` | HIVAU-70k train JSON event boundaries in seconds for UCF and XD training footage [21] | 1,552 | 374 |
| `ubi` | UBI per-frame 0/1 CSVs converted to intervals at 30 fps; annotation row counts match video frame counts for all 1,000 original files | 215 | 102 |
| `pseudo` | Legacy LSTM-generated curated YouTube boundaries | 0 in default evaluation | Excluded |

The primary ground-truth artifacts are [videos.csv](../benchmark/videos.csv) and [events.json](../benchmark/events.json). The latter preserves violent segments, other segments, excluded spans, and onset provenance. Normal HIVAU entries contain randomly sampled segments with empty class labels; those segments are not violent events. UCA captions [22] are available as possible supervision but are not used as onset ground truth. Trimmed Kinetics and short violence-classification clips cannot provide 30-second lead-ins and do not form VA30 anticipation events.

### 4.2. Role Assignment, Deduplication, and Contamination Masks

Default evaluation includes only videos assigned `violent` with supported temporal onsets and `clean_negative` from explicit normal classes. An unannotated anomalous video is not a negative. Non-violence anomalies, such as crashes or explosions, are excluded from this binary evaluation rather than silently grouped with ordinary activity. Duplicate, contaminated, and curated hold-out roles are also excluded.

The catalog first consolidates repeated filenames across storage folders, then removes 49 byte-identical copies under different names, leaving 7,696 rows. A thumbnail-based search identifies 93 near-duplicate pairs in 73 clusters. Of these, 28 redundant near-identical video rows receive the `duplicate` role; related or overlapping footage is grouped for splitting. This search includes cross-dataset copies. It is a lower bound because differently cropped copies and overlaps beginning late in both videos can be missed.

Mixed-label XD clips require segment-level treatment. VA30 uses HIVAU event judgments to distinguish violence from other anomalies, including when matching judgments to official test intervals. The current catalog contains 114 excluded other-anomaly segments in 58 violent clips; 27 mixed-label clips with no retained violent segment become unlabelled. Thus, even an official temporal boundary can have a separate, model-assisted class-selection decision requiring audit.

XD clip names can encode positions in a source film. VA30 maps annotated violence and aftermath onto overlapping clips from that source, and masks clean clips' genuine pre-onset tails. In the catalog, 121 clean clips carry masks totaling approximately 1.735 hours. One nominally normal clip overlapping violence is marked contaminated. These masks prevent known precursors or visible violence from being counted as ordinary negative exposure. Trainers must use the provided label builder, which applies all masks, rather than extracting only the onset list.

### 4.3. Event Eligibility and Available Lead-In

An event needs at least 30 seconds of observed, unmasked non-violent lead-in for the primary event analysis. Let `clear_start` be zero for an initial event, or the end of previous violence plus its 10-second aftermath, advanced further by any blocking excluded span. The default causal two-of-three alarm rule cannot start an alarm before tick 2. VA30 therefore additionally requires:

```text
leadin = onset - clear_start >= 30
max_lead = onset - max(2, ceil(clear_start)) >= 30
```

This avoids counting events that cannot possibly achieve a 30-second warning on the scoring clock. VA30 v2 contains 551 eligible30 events in 468 videos, with a default Success@30 ceiling of 100%. The eligible60 subset has 218 events in 187 videos. There are 393 eligible30 videos carrying the implemented interpersonal flag; this is a video count, not an event count. Clear lead-in capped at 120 seconds per eligible event totals approximately 9.93 hours.

Of the 551 eligible30 events, 312 are first segments and 239 follow earlier violence in the same clip. A later lead-in may follow an edit or depict a different scene. Event metrics preserve these events with appropriate masks, but results must also be reported for first segments separately. The official-only analysis contains 75 eligible30 events in 72 violent videos, including 53 eligible interpersonal videos. Restricting to official onsets therefore changes both sample size and source composition.

### 4.4. Frozen Grouped Folds

[folds_v2.json](../benchmark/folds_v2.json) assigns 7,604 non-hold-out videos across 4,630 split groups to five test folds. XD groups begin with the source movie or YouTube video identifier; UCF and UBI begin with video identifiers. The split merges known near-duplicate relations and 16 hand-listed XD film franchises so related footage stays together. Groups are stratified by source kind and eligibility/role, then balanced by video count and duration. All roles receive fold assignments, but only the defined evaluation roles enter default metrics.

For each test fold, approximately one eighth of the remaining groups form a group-disjoint validation slice. Validation selection uses a fold-specific deterministic hash. Model fitting uses the remaining training groups; checkpoint choice, preprocessing statistics, calibration, and alarm tuning use training/validation data only. Validation groups from different cross-validation runs may overlap, but none overlaps its own run's training or test groups. The v2 audit reports zero known near-duplicate pairs crossing test-fold assignments; the superseded v1 file had 41 such pairs.

| Test fold | All assigned videos | Violent videos | Eligible30 videos / events | Clean-negative videos | Clean-negative hours |
|---|---:|---:|---:|---:|---:|
| 0 | 1,522 | 418 | 97 / 130 | 797 | 55.6 |
| 1 | 1,525 | 433 | 89 / 105 | 790 | 61.2 |
| 2 | 1,528 | 425 | 92 / 101 | 813 | 58.0 |
| 3 | 1,518 | 432 | 95 / 109 | 805 | 70.8 |
| 4 | 1,511 | 421 | 95 / 106 | 805 | 57.9 |

The frozen fold content SHA-256 is `8ebdf6ec2e549f4a7c41710c045a66af2823c8498cb60b4377363ccbd03965e6`. Predictions are pooled out of fold: every test video is scored by a model that did not train on its group. The planned learned-model comparison uses three seeds; existing seed-zero controls do not satisfy that learned-model requirement by themselves. Future catalog changes require an explicitly versioned fold file rather than silently changing membership.

### 4.5. Audio Availability, Hold-Out Footage, and Onset Quality

Having an audio stream does not imply usable sound. The catalog checks decoded amplitude for digital silence. Non-silent audio is present in 490 of 1,879 UCF videos, 2,194 of 2,194 XD movie videos, 2,554 of 2,558 XD YouTube videos, zero UBI videos, and 90 of the 92 curated videos. Non-silence still does not establish that speech or sound is related to the incident. Audio availability and quality require explicit masks and separate reporting.

The curated YouTube collection contains approximately ten hours of road-rage, bodycam, CCTV, and confrontation footage, but its onsets were generated by a legacy LSTM. It remains outside every fold and the default evaluation universe. The first 60 seconds were forced into a buildup stage during pseudo-label generation, producing circular and sometimes visibly incorrect labels. This collection can become a separately reported real-world hold-out only after independent human annotation and a provenance-overlap check.

HIVAU's agreement with official test annotations validates parsing, not the independent accuracy of its training onsets: the test entries largely copy official boundaries. One near-duplicate comparison between an official UCF onset and a HIVAU train onset differs by approximately 10.9 seconds. Human review of the prepared 50 HIVAU and 30 official-control contact sheets is pending. Full-universe and official-onset results must therefore be co-primary, with UBI reported separately. No completed human validation is claimed here.

## 5. Prior Approaches and Why They Failed

The evidence below is reconstructed in [EXPERIMENT_HISTORY.md](../docs/EXPERIMENT_HISTORY.md). A model may have a useful component while its reported experiment remains invalid. No compromised headline score is adopted as an anticipation result.

| Approach | Evidence that can be reported with its scope | Failure or unresolved limitation |
|---|---|---|
| Graph/temporal transformer | No trustworthy held-out anticipation result. Audit ablations found frame AUROC 0.469 after zeroing the scene vector and causal clip AUROC 0.659 on potentially seen data; neither is a clean generalization estimate. | Clip-mean scene features included future frames; temporal attention was non-causal; audio was zero; split reconstruction may have included test clips. |
| Text LLM on graph summaries | Video AUROC 0.588; accuracy 0.727 against majority accuracy 0.826 in the historical evaluation. | Sparse person-count summaries, label-revealing prompt fields, and performance below the majority baseline. |
| Zero-shot Qwen2.5-VL teacher | Historical XD frame AUROC 0.776 for current violence, with correctly aligned observations and no task training. | This is a detection reference. A current-violence score does not establish a 30-second warning. |
| Pre-onset-only and random-clip student models | No valid 30-second event-level result. | Some early label sets lacked negative videos; random splits placed clips from the same source in training and test. |
| Source-grouped multi-horizon StudentMLP | Test AUROC 0.562–0.622 and AP 0.096–0.158 across four historical runs; see the exact values below. | Best preserved historical reference, but noisy negatives, few eligible events, irregular sampling, and an alarm window that could not measure the capstone target. |
| Focal-loss MLP, BiLSTM, and contrastive heads | No corrected held-out VA30 result available. | Embeddings paired with the wrong label rows; horizon labels collapsed; time steps treated as seconds; clip leakage and threshold selection contaminated comparisons. |
| LoRA tuning and VQA features | LoRA validation accuracy stayed at 0.4925, the negative-class rate. The VQA probe's recorded AUROC 0.486 also used misaligned frames. | Degenerate prediction, incorrect temporal targets, out-of-range frame fetching, and invalid generation/evaluation inputs. These are failed implementation diagnostics. |
| Flow, audio, pose, and fusion | No reliable held-out anticipation improvement established. | About 71.6% of legacy audio rows were zero and 70% of pose rows repeated a clamped frame because of clock errors. Fusion and simulations compounded alignment and leakage problems. |
| Stitched streams, smoothing, ensembles, and calibration probes | No valid deployment sensitivity or FA/h estimate from the legacy simulations. | Many streams reused training clips; thresholds were tuned on reported data; event metrics credited ongoing violence and persistent alarms; time and lead definitions were wrong. |
| RTSP demo and curated-video pseudo-labeling | A working inference demonstration; no independently labeled live evaluation. | Training/inference differences and absent ground truth. Model-generated curation labels cannot validate the model that generated them. |

The four historical source-grouped MLP runs are reproduced below from [BASELINE.md](../docs/BASELINE.md), section 3. They use a single saved split and differing validation threshold rules. Their frame metrics concern onset within 30 seconds after masking in-event labels, not VA30 v2 Success@30.

| Historical run | Test AUROC | Test AP | XD-only test AUROC | Median accepted alarm lead |
|---|---:|---:|---:|---:|
| `outputs/student_mlp` | 0.622 | 0.151 | 0.564 | 8.9 s |
| `runs/student_fix1` | 0.578 | 0.158 | 0.521 | 7.8 s |
| `runs/student_fix2` | 0.592 | 0.149 | 0.497 | 8.4 s |
| `results/anticipation_prboost_v3` | 0.562 | 0.096 | 0.495 | 8.9 s |

The historical test split had 158 videos and a positive sampled-step rate of approximately 0.081. Of 118 videos treated as negative, 84 were unannotated UCF anomalies; only 34 were explicit normal videos. Only ten test videos had an onset at least 30 seconds into the recording. Its acceptance window, `[onset - 30, onset + 1]`, rejected earlier alarms and permitted alarms after onset. It therefore could not establish Success@30; absence of a valid measurement should not be rewritten as a measured zero. The historical MLP must be retrained and scored on VA30 before serving as the new protocol's B0 comparison.

The recurring lesson is that more modalities, a temporal head, or additional smoothing cannot repair an invalid time axis or evaluation target. Future work must first preserve causal timing, ground-truth provenance, source separation, and independent operating-point selection.

## 6. Methods: New Model

**[PLACEHOLDER — proposed design space only; no final architecture, completed training, or performance is asserted.]**

### 6.1. Causal Feature Extraction

**[PLACEHOLDER: identify the actual video, frame, pose/tracking, audio, and speech extractors used; checkpoint revisions; frozen versus trainable parameters; preprocessing; output dimensions; and presence/confidence masks.]**

Each feature timestamp must describe media ending at or before the scored tick. Whole-video normalization, future-filled missing values, centered windows, and transcripts containing future speech would violate this constraint. Document available history and handling of absent, silent, noisy, or unintelligible modalities. Candidate streams from Section 2 require empirical ablation rather than assumed inclusion.

### 6.2. Temporal Model and Fusion

**[PLACEHOLDER: give the implemented architecture, history length in seconds, causal attention or recurrence, modality projections, fusion rule, and parameter count. Explain which mechanism is intended to represent change in interaction state.]**

### 6.3. Objectives and Optimization

**[PLACEHOLDER: specify the actual target and loss, sampling/weighting, optimizer, learning-rate schedule, regularization, seeds, and checkpoint-selection rule.]**

A candidate discrete-time hazard head would use bins `(0,5]`, `(5,10]`, `(10,20]`, `(20,30]`, `(30,45]`, and `(45,60]` seconds, with horizon risk `1 - product(1 - hazard_j)` over the relevant bins. Alternatively, independent horizon heads require an explicit consistency rule. Record which formulation was actually used and how masked/censored ticks contribute. Auxiliary detection or future-feature objectives require separate descriptions and causal inference checks.

### 6.4. Ablations and Implementation Record

**[PLACEHOLDER: record the baseline model and the controlled order of adding motion, interaction, audio, affect, and speech; history-length and objective ablations; any artifact masking; and all training commands/configurations.]** Compare each variant on common coverage, folds, seeds, and validation policy. Extraction throughput and training time must be measured if reported; estimates from research notes are not capstone measurements.

## 7. Evaluation Protocol

This section describes the implemented [VA30 v2 evaluator](../anticipation/eval/va30.py), detailed in [VA30.md](../docs/VA30.md), rather than the less restrictive legacy simulators. All final comparisons must state benchmark version, score coverage, onset selection, and whether they use the full evaluation universe.

### 7.1. Clock, Masks, and Censoring

Every evaluated video receives one score per integer video second. Sparse causal observations use the last available score, never a future observation. Ticks during violence, the ten seconds after its end, and designated excluded spans do not enter anticipation metrics. Headline frame metrics additionally use only ticks before the first annotated segment of any kind, avoiding the shortcut of remembering an earlier incident. All-segment variants are secondary diagnostics.

Right-censoring means that the future was not observed for long enough to assign a negative label. For example, a calm clip ending ten seconds after a tick cannot establish that no incident would occur within the next 30 seconds. VA30 masks a horizon-negative tick if the required future extends beyond the recording. An observed positive onset within the horizon remains usable. This is different from treating every recording end as evidence of safety.

### 7.2. Frame-Level Ranking and Calibration

**Far-window AUROC** evaluates whether scores 30–60 seconds before onset are higher than scores on clean-negative footage. More precisely, positive ticks satisfy `30 < time_to_onset <= 60`, are unmasked, and precede any observed segment. Primary negatives are unmasked clean-negative ticks with at least 60 seconds of remaining footage. A secondary comparison uses ticks with no onset within 120 seconds across the evaluation universe. AUROC is the probability that a randomly selected positive receives a higher score than a negative, with tied scores receiving half credit; 0.5 indicates chance ranking. It does not measure alarm lead or practical usefulness at a chosen threshold.

Every far-window result is paired with **source × elapsed-time matched AUROC**. Clean-negative weights match positive ticks' source-kind and elapsed-time-bin distribution; positives without comparable negatives are dropped and coverage is reported. Scores determined only by those matched strata should rank at chance. VA30 also reports an uncensored-clean-negative comparison, because excluding the ends of short normal clips changes the time distribution, and secondary all-segment metrics that expose detection-memory effects.

**Per-horizon AUROC and average precision (AP)** are reported for 5, 10, 20, 30, and 60 seconds. Positives use `y_H`; negatives have no onset within 120 seconds and enough observable future for that horizon. Intermediate ticks whose onset is between `H` and 120 seconds are excluded so plausible earlier warnings are not automatically negatives. AP summarizes precision across recall levels and depends on prevalence; the positive tick rate must accompany it. Report pooled, per-source, and matched AUROC results, rather than comparing AP across differently sampled populations as if it were invariant.

For probability-valued scores, **Brier score** measures mean squared error between predicted 30-second risk and the observed label; lower is better. Compare it with the constant-prevalence reference. A reliability plot and expected calibration error describe whether, for example, ticks assigned a given probability have a corresponding event frequency. Benchmark calibration does not establish deployment calibration under a different event rate.

### 7.3. Alarm Episodes and Success@30

At threshold `theta`, a tick is a raw alarm if its score is at least `theta`. The default policy activates an alarm when two of the most recent three ticks cross threshold. Active runs separated by fewer than ten seconds of silence merge into one episode. Starts less than 30 seconds after the preceding counted start are absorbed by a refractory period. All smoothing is causal.

For an eligible event with onset `t0`, the acceptance window is:

```text
[max(t0 - 120, clear_start), t0)
```

Take the first alarm episode that **starts** in this window. Its warning lead is `t0 - alarm_start` in seconds. An episode already running when the window opens earns no free credit, and an alarm raised during violence cannot count as anticipation. Episode merging and refractory timing reset at each new clear lead-in; a persistent alarm is not restarted merely by that reset.

**Success@30** is the fraction of all eligible30 events with an accepted alarm lead of at least 30 seconds; missed events remain in the denominator. Success@10 and Success@0 use the same eligible30 denominator with weaker lead requirements. Report the eligible60 subset, including Success@60, and the `eligible30_late` subset of 254 events whose acceptance window begins at or after second 30. A start-of-clip alarm cannot anticipate those late-window events.

Report median lead, interquartile range, and mean time to anticipation over successfully alarmed events, alongside event recall and its denominator. Leads are not artificially capped at 30 seconds; the protocol's 120-second acceptance bound must nevertheless be stated. An alarm earlier than that bound is outside the event's accepted prediction interval. A long mean lead among a few successes does not imply high event sensitivity.

As an illustration of the policy, not an experimental result, consider an event at second 100 with `clear_start = 0`: an episode beginning at second 60 has 40 seconds of lead and can satisfy Success@30; one beginning at second 85 has 15 seconds of lead and cannot. An episode that began before the permitted window and simply stayed active is not a new warning.

### 7.4. False Alarms and Validation-Selected Operating Points

**False alarms per hour (FA/h)** count alarm episode starts during eligible negative exposure, divided by its duration in hours. The primary denominator is unmasked clean-negative footage. The secondary rate uses all evaluated footage outside event windows `[t0 - 120, end + 10]` and excluded spans. Also report the rate restricted to clean videos shorter than one hour, because a few very long recordings can dominate exposure.

FA/h measures interruptions, while **alarm-time fraction** measures the fraction of clean time under an active alarm. Both are needed: a nearly continuous alarm may create few episode starts while remaining unusable. Reporting only a low episode rate can conceal this failure.

Each fold chooses thresholds on its validation slice at targets of 1, 2, and 5 FA/h, then freezes them for test scoring. The evaluator starts at a threshold that never alarms and lowers it through unique validation scores until the false-alarm target is first exceeded, retaining the lowest threshold before that crossing. A bottom-up search is inappropriate because merging can make FA/h fall again near the always-on regime. Some discrete score sets cannot achieve the target. The report must show the actual test FA/h beside the validation target; a target of 2 FA/h does not imply that the test rate equals two.

An AMOC-style plot displays the tradeoff between Success@30 and FA/h. Its sweep over pooled test scores is descriptive; it does not authorize selecting the reported operating point on test data. Checkpoint choice, smoothing, fusion weights, and calibration must likewise be fixed using training/validation data.

### 7.5. Chance References and Shortcut Controls

A Poisson reference asks how often independent random alarms would land in a success window at the same observed test alarm rate:

```text
P_random = 1 - exp(-FAR * W_hours)
```

For a full Success@30 window of 90 seconds, two alarms per hour give approximately 4.9% random success. The evaluator also uses each event's actual available window. This is an analytic reference, not a measured model result. Independence from clip position is a strong assumption and is violated by edited videos with early onsets.

VA30 v2 therefore reports a **position null** at the same test FA/h: an optimistic envelope over policies that use only source and elapsed clip time, including start-of-clip alarms. Its oracle fitting on evaluated labels is deliberate diagnostic construction, not a deployable trained-model comparison. The upper concave hull permits mixtures of those policies. Success@30 must be interpreted against this stronger reference, including its late-event version, rather than against Poisson alone.

Required controls include:

| Control | What it uses | What it tests / current status |
|---|---|---|
| `b_prior` | Constant score | Evaluator sanity; implemented. |
| `b_time`, `b_time_fit` | Elapsed clip time, directly or with training-fitted bins | Clip-position shortcuts; implemented. |
| `b_time_source_fit` | Training-fitted source × elapsed-time bins | Provenance plus position; implemented. |
| `b_memory` | Oracle indicator that an annotated incident has already begun | Past-detection memory; diagnostic, implemented. |
| `b_teacher` | Zero-shot Qwen current-violence scores held on the seconds clock | Detection-only reference; implemented on restricted video coverage. |
| B-scene0 | Classifier on the first two seconds, broadcast over the clip | Static source/scene shortcut; pending implementation/evaluation. |
| B0 | Historical StudentMLP retrained with VA30 labels and folds | Comparable learned baseline; pending retraining/evaluation. |

The committed control analysis illustrates why this is necessary: `b_time_source_fit` reaches far-window AUROC 0.858 but matched AUROC 0.496 without reading pixels. These are existing benchmark diagnostics, not results of the proposed model. Their exact source is [the committed control report](../benchmark/results/b_time_source_fit/report.md). The teacher covers only 1,031 evaluation-universe videos, so it must be compared with controls and models scored on that same subset, with coverage stated. Restricted-universe numbers cannot stand in for full-universe performance.

### 7.6. Detection, Uncertainty, and Reporting Requirements

Detection is reported separately. The implemented event-detection measure credits an active alarm tick in `[max(t0 - 120, clear_start), event_end]` for any violent event, eligible or not; thus it also credits accepted early alarms and is broader than post-onset-only detection. Detection AUROC compares in-violence ticks against clean-negative ticks. Neither measure replaces Success@30.

Use five-fold pooled out-of-fold predictions and three training seeds. Report 95% cluster-bootstrap intervals using 1,000 video resamples; grouping by source group is an available alternative and should be declared. Do not resample individual frames as independent observations. Default bootstrap intervals hold thresholds fixed and omit validation threshold-selection uncertainty. Add the evaluator's threshold-aware bootstrap, with 200 replicates as used by the controls, which resamples validation and test videos and reselects thresholds. Report seed means and standard deviations without treating seeds as independent datasets.

Break down results by XD movie, XD YouTube, UCF, and UBI; interpersonal versus other violence; audio present versus absent; official/HIVAU/UBI onset source; and first versus later events. The official-onset breakdown is co-primary while HIVAU validation is pending. The independently annotated curated hold-out, once available, must have its own results and denominator. Do not infer demographic fairness from source or audio breakdowns.

Every run must save scores with `video_id`, `t_sec`, `score`, `fold`, and `seed`, including the validation predictions needed for threshold selection. Duplicate keys and non-finite values are errors; incomplete coverage requires an explicit, reported evaluator option and is not directly comparable with complete coverage. Save the source commit, exact commands, fold/catalog/event/score hashes, feature and model configuration, seeds, threshold rule, coverage, and `metrics.json`. Preserve event-level audit output such as `events_op2.csv`.

The final interpretation must require consistent evidence from matched frame ranking, event success above the position null, actual test FA/h, alarm-time fraction, and onset/source breakdowns. Compare the new model with B0 and B-scene0 once available; the existing protocol calls for non-overlapping bootstrap intervals on the far-window comparison. Marginal overlap or failure to exceed a control must be reported without claiming improvement. Strong pooled AUROC alone is insufficient.

## 8. Results

**[PLACEHOLDER — no new model results are available in this report. Every PENDING entry is reserved for the final comparison and is not zero, chance performance, or an estimate.]** Import existing control results with their exact coverage and benchmark snapshot. Historical diagnostics belong to Section 5 and are not interchangeable with these VA30 v2 results.

### 8.1. Main VA30 Comparison

Each cell must include the metric and its 95% confidence interval where applicable. Use identical coverage for compared rows; report official-onset co-primary results in a companion table. Learned-model rows additionally need the mean and standard deviation over three seeds.

| Model / control | Coverage and onset set | Far AUROC | Matched far AUROC | AP@30 / tick prevalence | Success@30 at validation target 2 FA/h | Actual test FA/h | Clean alarm-time fraction |
|---|---|---|---|---|---|---|---|
| B-prior | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| B-time-source | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| B-scene0 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| B-teacher, matched subset | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| B0, retrained StudentMLP | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| New model | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

### 8.2. Operating Points and Warning Lead

| Validation FA/h target | Actual test FA/h | Eligible event count | Success@30 | Late-event Success@30 | Position null / late null | Poisson reference | Median lead / IQR / mean lead |
|---:|---|---|---|---|---|---|---|
| 1 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| 2 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| 5 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

**[PLACEHOLDER: include Success@10, Success@0, eligible60/Success@60, separately labeled detection, horizon AUROC/AP, Brier score, reliability, fixed-threshold and threshold-aware intervals, and AMOC/lead-distribution figures from the final evaluator outputs.]**

### 8.3. Ablations and Breakdowns

| Experiment or breakdown | Videos / eligible events / clean hours | Matched far AUROC | Success@30 / actual test FA/h | Interpretation |
|---|---|---|---|---|
| Visual-only versus interaction features | PENDING | PENDING | PENDING | PENDING |
| Addition of usable audio and speech | PENDING | PENDING | PENDING | PENDING |
| History length and training objective | PENDING | PENDING | PENDING | PENDING |
| Official versus HIVAU versus UBI onsets | PENDING | PENDING | PENDING | PENDING |
| Interpersonal classes; first versus later events | PENDING | PENDING | PENDING | PENDING |
| Real-world sources versus movie footage | PENDING | PENDING | PENDING | PENDING |
| Audio present versus absent | PENDING | PENDING | PENDING | PENDING |
| Independent curated hold-out after human annotation | PENDING | PENDING | PENDING | PENDING |

## 9. Discussion

**[PLACEHOLDER — complete from measured results.]** Determine whether any improvement persists after source/time matching, exceeds the position null at the same observed FA/h, and is supported on official-onset and real-world interpersonal subsets. Explain uncertainty, missing coverage, annotation sensitivity, and disagreement between frame-level and event-level metrics. Inspect false alarms, misses, earliest accepted alarms, and high-scoring artifact frames using representative examples selected by a stated rule. Report a negative finding if the new model fails to support the 30-second target.

## 10. Limitations and Ethics

### 10.1. Statistical, Annotation, and Domain Limits

VA30 combines datasets designed largely for detection. Its long-lead event population is selected and may differ from spontaneous violence with no visible buildup. Onset boundaries can be late, broad, or inconsistent across annotation sources, changing the meaning of a claimed warning. Many eligible events rely on unvalidated HIVAU train labels; official-only estimates are smaller and source-confounded. Multiple events in edited videos need not describe one continuous escalation process. Residual duplicates and unobserved uploader/camera relations may remain despite grouping.

Movie editing, filming selection, source duration, compression, and missing audio create domain differences. The `domain` metadata is best effort, not independently verified camera provenance. A long normal recording can dominate exposure while representing little environmental diversity. Cluster intervals and source breakdowns make these limits visible but do not eliminate them. Retrospective curated footage cannot establish performance or safety on continuous operational surveillance.

### 10.2. Surveillance and Privacy

Anticipation expands observation from visible incidents to ordinary behavior before an incident. This raises privacy concerns for bystanders and people engaged in lawful disagreement. Face imagery, speech, transcripts, and interaction tracks can reveal sensitive personal information even without identity recognition. Public availability of footage does not imply informed consent for behavioral profiling. Any future data collection or user study should obtain applicable LTU ethics review, limit retention and access, and avoid redistributing identifiable clips without the required rights and permissions. Release annotations, evaluation artifacts, and provenance records only in forms permitted by their source terms.

The system's appropriate research output is an uncertain event-risk estimate for review. It must not be presented as proof of criminal intent, an identification tool, or justification for an automated punitive response. No deployment safety claim follows from this capstone benchmark.

### 10.3. Bias and Unequal Error Burdens

Datasets reflect what gets recorded, uploaded, labeled, and retained. Apparent aggression can be interpreted differently across cultures, languages, disabilities, and interaction styles. Pose estimation, audio transcription, and emotion or toxicity models introduce additional unequal measurement errors. Loudness, proximity, and expressive gestures are ambiguous; treating them as danger indicators can penalize ordinary behavior. Source matching addresses one confound, not demographic fairness.

The present catalog does not establish representative demographic coverage or provide an adequate basis for demographic fairness claims. Future audits need appropriately governed annotations, meaningful subgroup denominators, and error analysis that does not infer protected characteristics from appearance. The arrest category also requires careful review because a category label does not settle who initiates harm or whether force is justified.

### 10.4. False Alarms, Misses, and Human Use

False alarms can stigmatize people, divert attention, and contribute to alarm fatigue or unnecessary intervention. Missed incidents can create false confidence. These costs depend on context and cannot be resolved by AUROC alone. Report both episode frequency and time under alarm, document uncertainty, and avoid converting a retrospective score threshold into an operational intervention policy. Any later pilot requires independent validation, human oversight, a clear response protocol, and examination of whether warnings reduce harm rather than escalate encounters.

## 11. Future Work

First, complete the pending onset review and document an annotation definition for the first physical act, reviewer agreement, ambiguous cases, and any corrections. Preserve the original boundaries and version the revised benchmark. Human-annotate the curated collection independently of legacy scores, add provenance-matched benign confrontations and de-escalations, and retain the collection as an external hold-out.

Second, implement and evaluate the B0 and B-scene0 controls before attributing gains to a complex architecture. Test a small causal model with explicit interaction trajectories, then add motion, audio, and speech through controlled ablations. Evaluate censor-aware objectives, calibrated horizon risks, and auxiliary future-feature training under the same folds and coverage. Validate temporal alignment and causal availability before expensive extraction or training.

Third, collect genuinely continuous footage with long benign exposure and sufficient scene, camera, and social diversity. Evaluate cross-domain transfer, latency, missing modalities, robustness to editing and overlays, and sensitivity to onset shifts. Only after these steps should a separately reviewed prospective study investigate whether warnings help trained human reviewers make safer decisions.

## 12. Conclusion

**[PLACEHOLDER — finalize after Section 8.]** State the measured answer to the research questions, whether the 30-second target was supported at the stated alarm burden, and the limits of that conclusion. The current defensible statement is that the repository has an implemented evaluation framework and documented historical failures; it has not yet established reliable real-world interpersonal-violence anticipation 30 seconds before onset.

## References

All entries below were verified by opening the linked primary paper, author-hosted manuscript, or official author repository on October 14, 2026. An arXiv entry is identified as a preprint unless the cited source confirms a publication venue. Local benchmark statistics and historical experiments are attributed through repository links in the relevant sections.

1. Sultani, W., Chen, C., and Shah, M. (2018). *Real-World Anomaly Detection in Surveillance Videos*. CVPR. [Official author repository and citation](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018).
2. Wu, P., et al. (2020). *Not only Look, but also Listen: Learning Multimodal Violence Detection under Weak Supervision*. ECCV. [Paper](https://arxiv.org/abs/2007.04687).
3. Pu, Y., Wu, X., Yang, L., and Wang, S. (2024 revision). *Learning Prompt-Enhanced Context Features for Weakly-Supervised Video Anomaly Detection*. [Author manuscript](https://arxiv.org/abs/2306.14451).
4. Cao, C., Lu, Y., Wang, P., and Zhang, Y. (2023). *A New Comprehensive Benchmark for Semi-supervised Video Anomaly Detection and Anticipation*. CVPR. [Paper](https://arxiv.org/abs/2305.13611).
5. Majhi, S., et al. (2025). *Guess Future Anomalies from Normalcy: Forecasting Abnormal Behavior in Real-World Videos*. WACV. [Author-hosted paper](https://www-sop.inria.fr/members/Francois.Bremond/Postscript/Snehashi_WACV2025.pdf).
6. Wang, Y., et al. (2026). *Streaming Video Crime Anticipation with Spatio-Temporal Causal Reasoning*. CVPR, pp. 16933–16943. [Primary proceedings paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_Streaming_Video_Crime_Anticipation_with_Spatio-Temporal_Causal_Reasoning_CVPR_2026_paper.pdf).
7. Yang, Z., et al. (2025). *AssistPDA: An Online Video Surveillance Assistant for Video Anomaly Prediction, Detection, and Analysis*. Preprint. [Paper](https://arxiv.org/abs/2503.21904).
8. Furnari, A., and Farinella, G. M. (2020). *Rolling-Unrolling LSTMs for Action Anticipation from First-Person Video*. IEEE TPAMI. [Author manuscript](https://arxiv.org/abs/2005.02190).
9. Girdhar, R., and Grauman, K. (2021). *Anticipative Video Transformer*. ICCV. [Paper](https://arxiv.org/abs/2106.02036).
10. Wu, C.-Y., et al. (2022). *MeMViT: Memory-Augmented Multiscale Vision Transformer for Efficient Long-Term Video Recognition*. [Author manuscript](https://arxiv.org/abs/2201.08383).
11. Zhong, Z., et al. (2023). *Anticipative Feature Fusion Transformer for Multi-Modal Action Anticipation*. WACV. [Paper](https://arxiv.org/abs/2210.12649).
12. Assran, M., et al. (2025). *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*. Preprint. [Paper](https://arxiv.org/abs/2506.09985).
13. Bolya, D., et al. (2025). *Perception Encoder: The best visual embeddings are not at the output of the network*. Preprint. [Paper](https://arxiv.org/abs/2504.13181).
14. Vyas, A., et al. (2025). *Pushing the Frontier of Audiovisual Perception with Large-Scale Multimodal Correspondence Learning*. Preprint. [Paper](https://arxiv.org/abs/2512.19687).
15. Lefter, I., Baird, A., Stappen, L., and Schuller, B. W. (2022). *A Cross-Corpus Speech-Based Analysis of Escalating Negative Interactions*. Frontiers in Computer Science, 4, 749804. [Article](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2022.749804/full).
16. Kim, S., Valente, F., and Vinciarelli, A. (2013). *Annotation and Detection of Conflict Escalation in Political Debates*. Interspeech. [Author-hosted paper](https://www.dcs.gla.ac.uk/~vincia/papers/escalation.pdf).
17. Anwar, A., Kanjo, E., and Ortega Anderez, D. (2022). *DeepSafety: Multi-level Audio-Text Feature Extraction and Fusion Approach for Violence Detection in Conversations*. Preprint. [Paper](https://arxiv.org/abs/2206.11822).
18. Costa-jussà, M. R., et al. (2024). *MuTox: Universal MUltilingual Audio-based TOXicity Dataset and Zero-shot Detector*. [Author manuscript](https://arxiv.org/abs/2401.05060).
19. Penchala, S., et al. (2026). *Can We Anticipate Violence? Multimodal Learning from Pre-Incident Behavioral Cues*. Preprint. [Paper](https://arxiv.org/abs/2609.40014).
20. Ganesh, P. (2026). *What Do Interaction Representations Actually Measure? Pre-Event Separability in Weakly-Supervised Violence Detection*. Preprint. [Paper](https://arxiv.org/abs/2608.27879).
21. Zhang, H., et al. (2025). *Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity*. CVPR. [Paper](https://arxiv.org/abs/2412.06171).
22. Yuan, T., et al. (2024). *Towards Surveillance Video-and-Language Understanding: New Dataset, Baselines, and Challenges*. CVPR. [Author manuscript](https://arxiv.org/abs/2309.13925).
23. Lee, C., Zame, W. R., Yoon, J., and van der Schaar, M. (2018). *DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks*. AAAI. [Paper](https://cdn.aaai.org/ojs/11842/11842-13-15370-1-2-20201228.pdf).
24. Chan, F.-H., Chen, Y.-T., Xiang, Y., and Sun, M. (2016). *Anticipating Accidents in Dashcam Videos*. ACCV. [Author-hosted paper](https://yuxng.github.io/Papers/2016/chan_accv16.pdf).
25. Degardin, B., and Proença, H. (2020). *Human Activity Analysis: Iterative Weak/Self-Supervised Learning Frameworks for Detecting Abnormal Events*. IJCB. [Official author repository documenting UBI-Fights](https://github.com/DegardinBruno/human-self-learning-anomaly).
