# Public datasets for 30-second violence anticipation

_Research note, written 2026-10-14. Scope: public datasets that can help train or evaluate a model that flags a fight or assault **30 seconds or more before it starts**, in real-world footage (CCTV, bodycam, phone/YouTube street video, bars, buses, schools). Audio-visual data gets priority._

---

## 0. Summary

1. **No public dataset targets 30-second violence anticipation.** Every violence or anomaly benchmark is built for *detection*. The few anticipation papers (2023–2026) only test horizons of **≤5 s**, or use clips of at most about 4 s. A 30-second benchmark has to be **assembled** from datasets that (a) are untrimmed, (b) have onset timestamps, and (c) include long non-violent lead-ins.
2. **The repo can only evaluate on videos with a long lead-in.** We measured lead-in lengths directly from the onset annotations (method in §2). Counting all public sources together, only about **370 interpersonal-violence videos** have ≥30 s of pre-onset footage, and about **220** if we keep only fighting/assault/abuse. The **UCF-Crime test set alone has just 4 fighting/assault/abuse videos with ≥30 s of lead-in**, so any "30 s anticipation accuracy" measured only on the UCF-Crime test split is statistically meaningless.
3. **The two biggest gains are free and need no new videos:**
   - **HIVAU-70k raw annotations** (Holmes-VAU, CVPR 2025) give *human-annotated event start/end timestamps for the UCF-Crime and XD-Violence **training** sets*. These datasets officially ship only video-level labels for training. With these annotations, the number of onset-labelled videos goes from about 640 to about 3,300.
   - **UBI-Fights** can be downloaded directly with no form, has frame-level labels, and 45 of its 215 fight videos have ≥30 s of lead-in. The repo does not use it yet.
4. **Next tier, needing a request or a wait:** NTU **CCTV-Fights** (1,000 videos, 2,414 fight instances, frame-level, CCTV videos average about 2 min, needs an access request) and **DVD** (2025, 500 long videos up to 20 min including bodycam, doorbell and phone footage, frame-level, about 63% of scenes have event-linked sound, "released upon acceptance").
5. **Audio:** **XD-Violence is the only large violence dataset with real audio.** UCF-Crime files usually *have* an audio track, but most of it is digital silence: in our sample, 10 of 16 UCF Fighting/Assault files sat at −91 dB. UBI-Fights videos have **no audio stream** (verified with ffprobe). For audio cues, use AudioSet/VGGSound event classes (Shout, Yell, Screaming, Slap, Smash, Gunshot) and speech resources (MuTox toxicity, SSPNet conflict, CREMA-D/MSP-Podcast anger), and transfer the resulting models.
6. **Trimmed clip datasets** (RWF-2000, RLVS, Hockey, Movies, SCVD, Bus Violence, AIRTLab, Violent-Flows, Surveillance Fight) contain **no lead-in** (clips are 1–7 s). They only help train an *"is violence happening now"* recognizer (for onset labelling and auxiliary heads) and supply hard negatives.

---

## 1. What the repo already has (as of 2026-10-14)

Data lives outside the repo at `data -> data/`.

| Source | Local path | Local size | What is there | How it is used |
|---|---|---|---|---|
| UCF-Crime | `data/raw_videos/ucf_crime/` | 123 GB, ~2,950 mp4 | Full release (Anomaly-Videos, Normal videos, `Temporal_Anomaly_Annotation_for_Testing_Videos.txt`) | `setup/build_ucf_crime_metadata.py` → `data/metadata/ucf_crime.json` (1,000 entries, 140 positives = test anomalies). Onset = first annotated frame. **Train-set anomalies have no onsets, so they are effectively unused as positives.** |
| XD-Violence | `data/raw_videos/xd_violence/` | 11 GB, 798 videos | **Test split only** (800 videos, `annotations.txt` with 500 violent) | `setup/build_xdviolence_metadata.py` → `data/metadata/xd_violence.json` (798 entries, 226 marked positive). The 3,954-video train split is **not downloaded**. |
| Kinetics-700 | `data/raw_videos/kinetics/` | 863 MB, 700 clips | 10 s clips from violence-adjacent classes (e.g. slapping, wrestling) | `data/metadata/kinetics.json`. `event_frame` is set to mid-clip (synthetic onset), so these clips cannot teach anticipation. |
| YouTube curation | `data/anticipation_curation/` | 92 videos | Built by `setup/curate_youtube.py` (yt-dlp search on road rage, bodycam, confrontations, then VLM onset labelling) | `candidates.json`: 97 candidates, 92 downloaded, 47 VLM-flagged violent, 34 with an anticipation window, **28 with ≥30 s** (median window 70 s). Labels come from the VLM and **are not human-verified**. |

`experiments/xd_violence_investigation.py:333` (removed in the 2026-10 cleanup; `git show pre-cleanup:experiments/xd_violence_investigation.py`) suggests Hockey Fight, VIRAT or CCTV-Fight "which have more pre-event frames". **Hockey Fight does not**: its clips are about 1–2 s. Of those three, only CCTV-Fights is a valid suggestion.

---

## 2. The critical number: how much lead-in each dataset has

**Method.** Onset is the start of the **first** annotated violent/anomalous interval. Lead-in is the onset time in seconds.

| Source of onsets | How lead-in was computed |
|---|---|
| UCF-Crime test | Temporal annotation file, frame ÷ 30 fps |
| XD-Violence test | `annotations.txt`, frame ÷ 24 fps |
| UCF/XD train | HIVAU-70k `*_database_train.json`, `events[0][0]` in seconds |
| UBI-Fights | Per-frame CSVs, read straight from the remote zip with HTTP range requests (no full download), ÷ 30 fps |
| Repo YouTube set | `candidates.json` `anticipation_window_seconds` |

"Interpersonal" means UCF {Abuse, Arrest, Assault, Fighting, Robbery, Shooting} and XD {Fighting, Abuse, Riot, Shooting}. "Fight-only" means {Fighting, Abuse, Assault}.

| Source | Anomalous / violent videos with onset | Median onset (s) | ≥30 s lead-in (all classes) | ≥60 s | ≥30 s interpersonal | ≥30 s fight-only |
|---|---|---|---|---|---|---|
| UCF-Crime **test** (official) | 140 | ~20.7 | 59 | 31 | 22 | **4** |
| UCF-Crime **train** (via HIVAU-70k) | 758 | 11.8 | 192 | 85 | 73 | 32 |
| XD-Violence **test** (official) | 500 | 5.9 | 53 (34 from YouTube, 19 from movies) | 18 | 34 | 19 |
| XD-Violence **train** (via HIVAU-70k) | 1,904 | 9.6 | 397 | 146 | 172 | 93 |
| UBI-Fights | 215 | 6.8 | 45 | 16 | 45 | 45 |
| Repo YouTube curation (VLM labels) | 34 | 69.9 | 28 | — | 28 | ~28 |
| **Total (before dedup / QC)** | | | | | **~374** | **~221** |

Not countable yet:
- **CCTV-Fights:** 1,000 videos and 2,414 fight instances. CCTV videos average 2 min and run up to 12 min. Annotations sit behind an access request, and the yield is likely another 100–200.
- **DVD:** 500 videos of 15–1,200 s, not released yet.

**What the numbers imply:**
- The **median** onset is about 6–12 s in every source. Most real-world violence clips are cut close to the action. A 30 s anticipation dataset will therefore be a long-tail subset (≈10–25% of positives), and the class prior inside the "≥30 s" subset changes: UCF test ≥30 s is dominated by Shooting (12), Shoplifting (12) and Explosion (10), not fights.
- XD-Violence "Riot" has median onset 0 s (violence from frame 1), so it is useless for anticipation.
- XD movie clips contain cinematic build-up (music cues, cuts). Keep them out of evaluation, or report them separately.
- **Annotation quality caveat.** A 2026 diagnostic paper (Ganesh, arXiv 2608.27879) hand-inspected tight-margin pre-onset windows and found 3 of 5 UCF-Crime windows already showed visible violence, meaning the annotated onset was late. It also found that pre-onset windows carry **provenance cues** (title cards, watermarks) that separate classes without being about the event. **Re-verify the onset of every evaluation video (±2 s)** and mask or crop title cards.

---

## 3. Ranked dataset list (by usefulness for ≥30 s anticipation)

Legend: **Lead-in** = are there long non-violent stretches before onset? **Onset** = are there frame- or second-level onset labels? **Audio** = is there real diegetic audio?

| Rank | Dataset | Size | Lead-in | Onset | Audio | Access | Repo uses? |
|---|---|---|---|---|---|---|---|
| 1 | **XD-Violence** + HIVAU-70k train onsets | 4,754 videos / 217 h | Yes (397 train + 53 test with ≥30 s) | Test: frame. Train: via HIVAU | **Yes** (some silent or music-only) | Open (OneDrive/Baidu) | Test only |
| 2 | **UCF-Crime** + UCA + HIVAU-70k | 1,900 videos / 128 h, avg ~4 min | Yes (192 train + 59 test with ≥30 s) | Test: frame. Train: via HIVAU/UCA | Mostly silent tracks | Open | Test onsets only |
| 3 | **HIVAU-70k** (annotation layer) | 5,443 videos, 11,076 events, 55,806 clips | — | **Event start/end (s) for train sets** | — | GitHub (MIT repo) | No |
| 4 | **UBI-Fights** | 1,000 videos / 80 h (216 fight) | Yes (45 with ≥30 s; max 20 min) | Frame-level CSV | **No audio stream** | Direct zip, 8.1 GB | No |
| 5 | **NTU CCTV-Fights** | 1,000 videos / 17.7 h, 2,414 fights | Likely (CCTV mean 2 min, ≤12 min) | Frame-level start/end | Partial (non-CCTV) | Request form, non-commercial | No |
| 6 | **UCA** (UCF-Crime Annotation) | 1,854 videos, 23,542 timestamped sentences | Captions cover ~90% of each video, **including pre-event behaviour** | 0.1 s caption timestamps | — | GitHub | No |
| 7 | **DVD** (2025) | 500 videos, 2.7 M frames, 15–1,200 s | Likely (long, untrimmed) | Frame-level, ≥3/4 annotator agreement | ~63% of scenes have event-linked sound | "Upon acceptance" | No |
| 8 | Repo YouTube curation | 92 videos | 28 with ≥30 s | VLM-estimated | Yes | Local | Partly |
| 9 | **MSAD** (NeurIPS 2024) | 720 videos, 14 scenarios, ~500 cameras | Some (untrimmed) | Frame-level test | No | Request form; I3D/Swin features open | No |
| 10 | **CUVA / ECVA** | 1,000 / 2,240 videos, avg ~117–141 s | Some | Start/end + **cause ("why")** text | Unknown | GitHub / ModelScope | No |
| 11 | **NWPU Campus** | 547 videos / 16 h, 43 scenes | Yes (anticipation protocol) | Frame-level | No | Google Drive/Baidu, research only | No |
| 12 | **CamNuvem** | 972 (486 robbery + 486 UCF normal) | Some | Weak (test intervals) | Unknown | GitHub | No |
| 13 | Kinetics-700 escalation classes | ~600–970 clips/class | No (10 s clips) | Clip label | Yes | YouTube IDs | Yes (700 clips) |
| 14 | AudioSet / VGGSound violence-sound classes | see §5 | — | 10 s clip labels | **Yes** | YouTube IDs / HF mirrors | No |
| 15 | MuTox, SSPNet Conflict, CREMA-D/MSP-Podcast | speech | — | utterance/clip | **Yes** | Open / request | No |
| 16 | RWF-2000, RLVS, SCVD, Bus Violence, Hockey, Movies, Violent-Flows, AIRTLab, Surveillance Fight, VioPeru | 200–2,000 trimmed clips | **No** | Clip label | Mixed | Mostly open | No |
| 17 | NTU RGB+D 120 (mutual actions), UT-Interaction | 114k / 120 videos | No (staged) | Clip / segment | NTU: no | Request / open | No |
| 18 | ShanghaiTech, TAD, UCF-Crime-DVS, VSD2014, GTA-Crime, Hawk, RiskCueBench, DeEscalWild, DAD/CCD/DADA | various | various | various | various | various | No |

---

## 4. Per-dataset details

### 4.1 Tier A: untrimmed real-world video with onset labels (use for training and evaluating anticipation)

#### XD-Violence (Wu et al., ECCV 2020): already used, test split only
- **Size:** 4,754 untrimmed videos, 217 h. Train 3,954 videos (video-level weak labels). Test 800 videos (500 violent, 300 normal) with frame-level labels.
- **Classes:** Abuse (B5), Car accident (B6), Explosion (G), Fighting (B1), Riot (B4), Shooting (B2). Normal is A. Violent videos can carry 1–3 labels.
- **Sources:** Movies plus "in-the-wild" YouTube footage (sports, games, hand-held, CCTV, car cameras, news). In the test set, 279 of 500 violent videos come from YouTube (`v=` prefix) and 221 from movies.
- **Audio:** Yes. This is the main audio-visual violence benchmark. The project page notes "quite a few videos are silent or only contain background music". Our ffprobe sample: 25 of 25 files have audio, mean volume −16 to −43 dB.
- **Lead-in:** Test median 5.9 s, 53 with ≥30 s. Train (via HIVAU) median 9.6 s, 397 with ≥30 s.
- **Pre-extracted features:** I3D RGB/flow and VGGish audio are on the project page.
- **Access:** https://roc-ng.github.io/XD-Violence/ (OneDrive, Baidu, AliyunDrive). No explicit license; research use.
- **Watch out:** Movie clips carry editing and score cues. "Riot" starts at frame 0. Normal videos are background-matched to violent ones, which is good for reducing scene bias.

#### UCF-Crime (Sultani et al., CVPR 2018): already used
- **Size:** 1,900 untrimmed surveillance videos, about 128 h, 13 anomaly classes (Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism) plus Normal. Train 1,610 (810 anomalous / 800 normal, video-level labels). Test 290 (140 anomalous / 150 normal, frame-level labels). Average about 4 min per video.
- **Audio:** Tracks exist in most files but are mostly **silent or non-diegetic**. Our sample: 10 of 16 Fighting/Assault files at −91 dB (digital silence), 1 with no stream. Treat UCF-Crime as video-only.
- **Lead-in:** Test: 59 of 140 have ≥30 s, but only 4 are Fighting/Assault/Abuse. Train via HIVAU-70k: 192 of 758 have ≥30 s, 32 fight-only.
- **Access:** https://www.crcv.ucf.edu/projects/real-world/ (Dropbox link on that page). Research use.
- **Extensions:**
  - **UCA** for dense captions.
  - **HIVAU-70k** for train onsets.
  - **FS-UCF-Crime** (Univ. Salerno, Zenodo record 21336651, July 2026): interval labels for *train* plus a corrected test set and a validation split. **Placeholder only. "Released after paper acceptance", CC BY 4.0.** Check back: https://zenodo.org/records/21336651
  - **UCF-Crime-DVS** (AAAI 2025): event-camera re-recording, not useful here.

#### HIVAU-70k (Holmes-VAU, CVPR 2025): annotation layer, **highest value per unit of effort**
- **What it is:** Hierarchical annotations over UCF-Crime and XD-Violence **training** videos: 5,443 videos, 11,076 events, 55,806 clips, 70k+ instruction samples.
- **How the onsets were made:** The paper states that event temporal boundaries were **manually** annotated. Clip captions come from UCA (UCF) or uniform splits (XD), with LLM-generated summaries on top.
- **Files:** `HIVAU-70k/raw_annotations/{ucf,xd}_database_{train,test}.json`. Schema per video:
  ```json
  {"n_frames", "fps", "label": [...], "events": [[start_s, end_s], ...], "clips": [...], "clips_caption": [...]}
  ```
  For XD there is also `"events(frame)"`. **Normal videos also have "events"** (random crops) with `label == []`. Filter on a non-empty `label`.
- **Counts:** UCF train has 758 anomalous and 735 normal. XD train has 1,904 anomalous and 2,046 normal.
- **Access:** https://github.com/pipixin321/HolmesVAU (repo MIT). Raw JSON on GitHub. Preprocessed Hugging Face copies: `backseollgi/HIVAU-70k_UCF-Crime`, `backseollgi/HIVAU-70k_XD-Violence`.
- **Use here:** Turns about 2,660 weakly labelled training anomalies into onset-labelled positives, including about 570 with ≥30 s lead-in. Captions also describe what happened *before* the event (e.g. *"A man … walked towards the short-haired woman … When the woman was not paying attention …"*), which gives precursor text supervision.

#### UBI-Fights (Degardin & Proença, IJCB 2020): not used yet, **download it**
- **Size:** 1,000 videos, 80 h, 216 fights and 784 normal. 640×360 at 30 fps. Filename flags give indoor/outdoor, fixed/rotating/moving camera, and RGB/grayscale.
- **Annotations:** Per-frame 0/1 CSV for every video. "Unnecessary segments (intros, news) removed."
- **Measured:** Fight videos have median length 80 s, mean 125 s, max 1,200 s. Onset median 6.8 s. **≥30 s: 45, ≥60 s: 16, ≥120 s: 5.** About 65% of frames in a fight video are non-violent (median).
- **Audio:** **None.** Sampled mp4s contain only an h264 video stream.
- **Access:** Direct download, no form. https://socia-lab.di.ubi.pt/EventDetection/ → `UBI_FIGHTS.zip` (8.1 GB). The site's TLS certificate is expired, so use `curl -k`. No license stated; cite the IJCB 2020 paper.
- **Also useful:** The 784 normal videos are long real-world negatives.

#### NTU CCTV-Fights (Perez, Kot & Rocha, ICASSP 2019): not used, **request access**
- **Size:** 1,000 real fight videos, 17.68 h. 280 are CCTV (8.54 h, 5 s–12 min, mean 2 min). 720 are non-CCTV: phone, dash-cam, drone (9.13 h, 3 s–7 min, mean 45 s). 2,414 fight instances (about 2.4 per video). 7.2 GB.
- **Annotations:** Frame-level start and end of each fight instance. 50/25/25 train/val/test split.
- **Lead-in:** Not published. The CCTV subset's long durations and multiple instances per video suggest a meaningful ≥30 s subset. Compute it once the annotations arrive.
- **Audio:** Not stated by NTU. Many non-CCTV clips are YouTube re-uploads with audio, and AVadCLIP (2025) evaluates audio-visual on it. Verify per file.
- **Access:** https://rose1.ntu.edu.sg/dataset/cctvFights/. Register, submit the request form, accept the release agreement. Academic non-commercial only. **No redistribution or derived datasets without permission**, so we cannot publish clips we re-cut from it.

#### DVD (Kollias et al., arXiv 2506.05372, 2025): not released yet, watch it
- **Size:** 500 long videos, 2.7 M frames, 15–1,200 s each, up to 4K. Collected from YouTube with multilingual keywords.
- **Footage types:** **phone, CCTV, body-cam, dash-cam, doorbell, in-vehicle**. This is the closest public match to this project's target domain.
- **Annotations:** Frame-level binary violence labels from 4 trained annotators (kept only with ≥3 agreeing) plus a reviewer. "Extended de-escalation" is labelled non-violent, and renewed violence counts as a new instance. Per-scene metadata: number of people, event type (e.g. bar fight, protest), indoor/outdoor, footage type, and **whether sound is linked to the event** (1,263 scenes yes, 737 no). Split 55/15/30.
- **Collection:** Explicitly targets "actions leading to violence, such as arguments that escalate into fights".
- **Access:** "Will be publicly released upon acceptance". A subset was used in the ABAW-10 competition (2026). Track: https://arxiv.org/abs/2506.05372

#### Repo's own YouTube curation (`data/anticipation_curation/`)
- 28 videos with ≥30 s windows, median 70 s. This is the only source built for anticipation, but the labels come from the VLM.
- **Action:** Human-verify onset times. Extend the search to real bodycam, bar and school queries. Keep a per-video provenance list (URL plus timestamps) so the set can be shared as a list of links.

### 4.2 Tier B: annotation, text and reasoning layers (precursor cues, labelling engines)

| Dataset | What it adds | Size | Access |
|---|---|---|---|
| **UCA** – UCF-Crime Annotation (Yuan et al., CVPR 2024) | Dense, timestamped (0.1 s) English captions over **the whole video**. Captions cover ~90% of each video's duration, including normal activity *before* the anomaly (calm captions precede the violent ones). Usable for precursor-event mining, text-supervised cue heads, and checking onsets. | 1,854 videos (46 removed), 23,542 sentences, 110.7 h. UCA's own split: 1,165 train / 379 val / 310 test. | https://github.com/Xuange923/Surveillance-Video-Understanding (Apache-2.0 for annotations; academic use) |
| **CUVA** (CVPR 2024) / **ECVA** (IJCV 2026) | Anomaly type with start/end, **cause ("why")**, and effect text. ECVA adds an "importance curve" over key evidence segments. Cause text is an explicit precursor description. | CUVA 1,000 videos (avg 117 s, 10 categories / 42 sub-categories). ECVA 2,240 videos, 88 h, 6,720 QA pairs. | https://github.com/fesvhtr/CUVA, https://github.com/Dulpy/ECVA (videos on ModelScope `gouchenyi/ECVA`) |
| **Hawk** (NeurIPS 2024) | Open-world anomaly descriptions and QA over ~8k videos from several VAD sets (UCF-Crime, DoTA, Avenue, Ped, …). Excludes XD-Violence. | ~8k videos | https://github.com/jqtangust/hawk |
| **MA-CBP** data (arXiv 2508.06189) | Splits UCF-Crime and CamNuvem robbery/theft videos into **pre-crime suspicious segments** and crime segments. The only public precedent for labelling "suspicious before the act". | 711 abnormal + 950 normal (MSVD) | https://ma-cbp.github.io/ (data release unclear) |
| **SurveillanceVQA-589K** (2025) | Large surveillance video QA set, including causal/temporal questions. | 589k QA | arXiv 2505.12589 |

### 4.3 Tier C: other untrimmed anomaly datasets (some fights, mostly no audio)

- **MSAD** (Zhu et al., NeurIPS 2024 D&B): 720 videos from about 500 camera views and 14 scenarios (street, shop, restaurant, park, …). 35 human-related anomaly types, including Fighting (6 sub-types: street, restaurant, shop, doorway, indoor, other) and Assault (street, office, other). Weakly supervised split: 360 normal + 120 abnormal for training, 120 + 120 for testing (frame-level). RGB only. Videos by request form; I3D and Video-Swin features are open. https://msad-dataset.github.io/
- **NWPU Campus** (Cao et al., CVPR 2023 / TPAMI 2025): the **first dataset with an explicit video-anomaly-anticipation task**. 43 scenes, 28 classes, 547 videos, 16.3 h at 25 fps. Violence-related classes include Scuffle, Battering, Group conflict, Snatching bag and Chasing. The collection deliberately captured "the complete process of the events" so anomalies are predictable. Anticipation horizons are short (seconds). Semi-supervised (normal-only training). Research-only terms, no redistribution. https://campusvad.github.io/ . **Mainly useful as a protocol reference and an extra evaluation set for pre-onset scoring.**
- **CamNuvem** (de Paula et al., Sensors 2022): 486 real robbery CCTV videos from social media plus 486 UCF normal videos, 320×240 at 30 fps. Robberies often have a long approach phase. https://github.com/daviduarte/camnuvem-dataset
- **ShanghaiTech Campus:** 437 videos from 13 campus scenes (Zhong et al. weakly supervised re-split ~238/199). Anomalies include fighting and chasing among pedestrians. Videos are short and lead-ins are limited. No audio. Low value.
- **TAD** (Lv et al. 2021): 500 traffic videos, about 25 h. Off-domain (traffic).
- **UCF-Crime-DVS** (AAAI 2025): UCF-Crime re-recorded off a monitor with an event camera. Not useful for this project.
- **VSD2014 / MediaEval Violent Scenes:** 32 Hollywood movies (not distributed; you buy the DVDs) plus 86 YouTube clips. Annotations include audio concepts (gunshots, explosions, screams). Movie domain, mostly superseded by XD-Violence. https://www.interdigital.com/data_sets/violent-scenes-dataset
- **GTA-Crime** (arXiv 2509.08232, 2025): synthetic GTA-V CCTV-view shootings and stabbings (262 normal, 124 stabbing, 146 shooting) with a **generation framework** and synthetic-to-real domain adaptation. The framework could script **escalation sequences with controlled lead times**, which is a possible way to augment the scarce ≥30 s data. https://github.com/ta-ho/GTA-Crime

### 4.4 Tier D: trimmed violence-classification clips (no lead-in; detector training and hard negatives only)

| Dataset | Size / length | Domain | Audio | Access |
|---|---|---|---|---|
| **RWF-2000** (Cheng et al., ICPR 2020) | 2,000 clips × 5 s at 30 fps, 1,000 fight / 1,000 non-fight | Real CCTV from YouTube | Some | https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection (signed agreement); Kaggle mirrors |
| **RLVS** – Real Life Violence Situations (Soliman et al., 2019) | 2,000 clips (1,000 / 1,000), avg ~5 s | Street fights; non-violent sports, eating, talking | Some | https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset |
| **SCVD** – Smart-City CCTV Violence (Aremu et al., 2022) | ~484 clips: 248 non-violence, ~112–124 violence, ~112–124 **weaponized** violence (counts differ by source) | CCTV-like | No | https://www.kaggle.com/datasets/toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd |
| **Bus Violence** (Ciampi et al., Sensors 2022) | 1,400 clips, acted, multiple cameras in a moving bus | Public transport | No | https://zenodo.org/record/7044203 (CC BY-NC 4.0) |
| **Hockey Fights** (Bermejo Nievas et al., CAIP 2011) | 1,000 clips (500 / 500), ~1–2 s (≈41–50 frames) | NHL broadcast | No | Academic Torrents mirror |
| **Movies Fights** (same paper) | 200 clips (100 / 100) | Action movies | — | same |
| **Violent-Flows** (Hassner et al., 2012) | 246 clips, avg 3.6 s | Crowd violence, YouTube | — | https://www.openu.ac.il/home/hassner/data/violentflows/ |
| **AIRTLab** (Bianculli et al., 2020) | 350 clips (230 violent / 120 non), avg 5.6 s, acted | Indoor. **Hard negatives: hugs, claps** | — | https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos |
| **Surveillance Camera Fight** (Aktı et al., 2019) | 300 clips (150 / 150), ~2 s | YouTube CCTV | — | GitHub (seymanurakti) |
| **VioPeru** (Huillcen Baca et al., 2023) | ~280 real municipal CCTV videos from Apurímac, Peru, incl. 87 non-violent "false-positive validation" | Real municipal CCTV, never posted online | — | https://github.com/hhuillcen/VioPeru |

**Role in this project:** These sets can fine-tune a "violence happening now" detector, used to (1) auto-propose onsets on new long videos (replacing or checking the VLM labeller in `setup/curate_youtube.py`) and (2) serve as an auxiliary head. They **cannot** supply ≥30 s pre-onset context.

### 4.5 Pose and interaction datasets (for a pose-based precursor branch)
- **NTU RGB+D 120** (ROSE Lab): 114k samples, 120 classes, with **26 two-person mutual actions**, including A50 punch/slap, A51 kicking, A52 pushing, A54 point finger, A59 walking towards, A60 walking apart, and NTU-120 additions such as A106 hit with object, A107 wield knife, A109 grab stuff, A110 shoot with gun and A111 step on foot. 3D skeletons (5.8 + 4.5 GB). Good for pre-training a two-person interaction encoder on skeletons such as YOLO-pose (YOLO11x-pose is used in `anticipation/features/pose.py`; YOLOv8-pose was used in the removed `scripts/pose_feature_extraction.py`). Request access: https://rose1.ntu.edu.sg/dataset/actionRecognition/
- **UT-Interaction:** 120 videos (2 sets), 6 two-person interactions (push, kick, punch, handshake, hug, point). A **classic early-prediction benchmark** (accuracy at 10–100% of the observed sequence). Staged and short.
- **Kinetics-700 escalation/violence classes** (train counts from the local `train.csv`): arguing (641), shouting (644), slapping (968), headbutting (819), punching person (boxing) (607), wrestling (850), tackling (701), drop kicking (779), side kick (955), high kick (966), sword fighting (972), arresting. **Hard negatives:** pillow fight (949), arm wrestling (888), punching bag (913). **"arguing" and "shouting" are the only large public labelled sources of verbal-escalation precursors**, so add them to the Kinetics subset.

### 4.6 Audio datasets (aggression, shouting, screaming, impacts, verbal conflict)

**Sound-event classes in AudioSet and VGGSound.** Clip counts below were computed from the official segment CSVs on 2026-10-14. They are useful for training or fine-tuning an audio tagger (PANNs, AST, BEATs, CLAP) whose class posteriors become anticipation features.

| Class (AudioSet id) | Balanced train | Eval | Unbalanced train |
|---|---|---|---|
| Shout `/m/07p6fty` | 60 | 60 | 1,248 |
| Yell `/m/07sr1lc` | 60 | 60 | 674 |
| Screaming `/m/03qc9zr` | 59 | 61 | 1,126 |
| Bellow `/m/07q4ntr` | 60 | 60 | 391 |
| Battle cry `/m/04gy_2` | 60 | 60 | 327 |
| Children shouting `/t/dd00135` | 60 | 60 | 553 |
| Slap, smack `/m/07ryjzk` | 59 | 60 | 751 |
| Whack, thwack `/m/07rdhzs` | 62 | 60 | 1,441 |
| Thump, thud `/m/07qnq_y` | 60 | 60 | 1,680 |
| Smash, crash `/m/07pjjrj` | 60 | 60 | 1,355 |
| Shatter `/m/07rn7sz` | 60 | 61 | 247 |
| Breaking `/m/07pc8lb` | 60 | 60 | 346 |
| Gunshot, gunfire `/m/032s66` | 178 | 174 | 3,869 |
| Crying, sobbing `/m/0463cq4` | 60 | 62 | 1,340 |
| Groan / Gasp / Grunt / Whimper | ~60 each | ~60 each | 582 / 209 / 214 / 1,136 |
| Hubbub, speech babble `/m/07qfr4h` | 60 | 60 | 1,360 |
| Crowd `/m/03qtwd` | 147 | 140 | 10,116 |
| Police car (siren) `/m/04qvtq` | 61 | 60 | 3,538 |

- **VGGSound** (309 classes, ~200k clips): people screaming (731), people slapping (211), children shouting (201), people battle cry (291), people sobbing (481), people crowd (1,050), machine gun shooting (986), cap gun shooting (904), police car (siren) (1,050), police radio chatter (958). https://www.robots.ox.ac.uk/~vgg/data/vggsound/
- **FSD50K:** 51k Freesound clips, 200 AudioSet-ontology classes, CC licenses. Includes Screaming, Shout, Yell and Slap/smack. Cleaner labels than AudioSet. https://zenodo.org/record/4060432
- **MIVIA Audio Events:** 6,000 events (glass breaking, gunshots, screams) mixed into background noise at 5–30 dB SNR. 32 kHz, registration required. https://mivia.unisa.it/datasets/audio-analysis/mivia-audio-events/
- **Audio-based Violence Detection Dataset** (Kaggle, FangFang Zhu-Zhou): about 14,743 segments labelled violent / non-violent (shouts, screams, verbal confrontations, physical fight sounds), mostly phone-quality YouTube audio. This is the closest *audio* match to our footage. https://www.kaggle.com/datasets/fangfangz/audio-based-violence-detection-dataset

**Speech and verbal aggression or escalation:**
- **MuTox** (Meta, ACL 2024): about 116k audio utterances labelled toxic / non-toxic. 20k each for English and Spanish, 4k each for 19 more languages (21 total). Ships with a zero-shot audio toxicity classifier. Released through `facebookresearch/seamless_communication`. Can serve as a **verbal-threat/insult detector** directly on audio.
- **ADIMA** (ICASSP 2022): 11,775 clips, 65 h, abusive vs non-abusive, 10 Indic languages. HF: `callmesan/ADIMA`.
- **SSPNet Conflict Corpus:** 1,430 × 30 s clips from Swiss French political debates (Canal9), continuous conflict score from −10 to +10. Interspeech 2013 Conflict Challenge. Models *conversational* conflict escalation (overlap, interruptions, loudness), a plausible minutes-scale precursor.
- **NAA** – Negative Affect & Aggression (Lefter et al., ACII 2017): acted dyadic aggression with audio, video, mocap and physiology, rated aggression levels, **no physical violence**. Request from TU Delft.
- **Lefter et al. train-aggression recordings** (TNO/TU Delft): audio-visual scenarios of escalating aggression in trains, labelled on a 3-point scale per modality. Historically not public; contact the authors.
- **Anger and arousal in speech:** CREMA-D (7,442 clips, 91 actors), RAVDESS, IEMOCAP, MSP-Podcast (naturalistic, large). Use these for a raised-voice / anger head.
- **BERSting at the Screams** (2025): distanced, emotional and **shouted** speech recognition benchmark. Useful for making ASR robust to shouting.

**Practical note:** Whisper (or WhisperX) transcripts plus a toxicity or threat text classifier, combined with a shout/scream/slap audio tagger, is the cheapest audio stack to build. Only XD-Violence, CCTV-Fights (partially), DVD and our YouTube set have usable audio for *in-domain* fine-tuning.

### 4.7 Anticipation-adjacent benchmarks and protocols (for evaluation design)
- **"Can We Anticipate Violence?"** (Univ. Alabama, arXiv 2609.40014, Sep 2026): 443 pre-incident clips curated from XD-Violence. The incident is removed and clips are rated No / Low / Medium / High risk. Clips are ≤ ~4 s (32 frames at 8 fps). Face + pose kinematics + audio spectrogram gives AUC 0.964 and accuracy 91.2%. Face alone is strongest; audio alone gives AUC 0.79. Test set is only 91 clips, and no data release was found. **This is the closest published work, and its horizon is ≤4 s.**
- **PULS / "Latent Clarity"** (arXiv 2607.03558, 2026): world-model anomaly anticipation on UCF-Crime with zero-shot transfer to XD-Violence. Lead-time windows go up to T−5 s, and the anticipatory gain disappears by T−3 s. Shows that **current methods have no demonstrated signal beyond about 2–3 s**.
- **Ganesh, "Pre-Event Separability"** (arXiv 2608.27879, 2026): truncates anomalous videos at onset and compares them with **length-matched** normal videos. 39–91% of above-chance AUC survives before onset, but much of it is **scene or provenance bias**. Gives a reusable evaluation protocol (see §6).
- **Chang et al., Neurocomputing 2026:** Siamese "onset branch" that models build-up before violence. Visual, audio and BLIP-caption fusion.
- **RiskCueBench** (arXiv 2601.03369): 986 protest and car-crash videos with **risk-signal start/end timestamps** before the incident. Mean cue length is 1–18 s. VLMs perform near chance as the cue-to-incident gap grows to 20 s. Not interpersonal, but a template for annotation.
- **DeEscalWild** (arXiv 2604.13075): 1,500 in-the-wild police–civilian interactions (~18 min average) from YouTube, TikTok and Facebook. **Only diarized transcripts are released** (Harvard Dataverse doi:10.7910/DVN/CWMCZI, non-commercial). Useful for a *text* escalation model trained on real police encounters.
- **OpenBWC** (RIT + Rochester PD, 2026): body-worn-camera research with 10 s window labels. **Access-controlled, no public release.** We found no public, annotated bodycam use-of-force dataset.
- **Dashcam accident anticipation** (DAD, CCD, A3D, DADA-2000): accidents placed at a fixed frame. The metrics are AP and **mTTA (mean time-to-accident)**. Their evaluation protocol (score at every frame, alarm when it crosses a threshold, report the lead time) is the standard to copy.

---

## 5. Recommended dataset plan for this repo

1. **Pull the free onset layers now** (small text downloads):
   - HIVAU-70k raw annotations (`ucf_database_*.json`, `xd_database_*.json`).
   - UCA JSON.
   - Write `setup/build_hivau_onsets.py` to merge them into `data/metadata/*.json` with `event_frame` = first event start, for **train** videos.
2. **Download the XD-Violence train split** (3,954 videos, audio included). Today only the 800 test videos are local. This is the largest source of ≥30 s audio-visual positives (397 anomalous, 93 fight/abuse).
3. **Download UBI-Fights** (8.1 GB, direct). It adds 45 fight videos with ≥30 s lead-in and 784 long real-world negatives. Video-only.
4. **Request NTU CCTV-Fights** (and MSAD). Compute the ≥30 s subset once access is granted.
5. **Watch for DVD and FS-UCF-Crime** releases. DVD is the best domain match (bodycam, doorbell, phone, with sound).
6. **Build a named benchmark ("VA30")** with these rules:
   - Positives: interpersonal-violence videos with **≥30 s verified pre-onset footage**, about 220 fight-only or about 370 interpersonal across sources. **Hand-verify every test onset.**
   - Negatives: **length-matched** windows from normal videos of the *same source* (UCF normal, XD normal, UBI normal, MSAD normal) plus hard negatives (sports, horseplay, crowds, hugs).
   - Splits: by **source video** (and by YouTube uploader where known), never by window.
   - Report results separately per source (CCTV vs phone vs movie) and for audio-present vs audio-absent videos.
7. **Use trimmed datasets** (RWF-2000, RLVS, SCVD, Bus Violence) only for the "violence-now" auxiliary head and as an onset proposer that replaces or checks the VLM labeller. **Use Kinetics "arguing"/"shouting"** and the AudioSet/VGGSound shout, scream and slap classes plus MuTox to train precursor-cue heads.

## 6. Evaluation pitfalls to design against
- **Small n:** With about 4 UCF test fights at ≥30 s, use pooled multi-source test sets, **video-level bootstrap CIs**, and multiple seeds.
- **Scene and provenance leakage:** Title cards, watermarks, news overlays and movie grading all separate the classes without being about the event (Ganesh 2026). Crop overlays, length-match negatives, and run a "scene-only" baseline (e.g. a CLIP embedding of the first frame) to measure the leak.
- **Late onset labels:** UCF-Crime onsets can trail visible violence. Add a ±2 s exclusion zone and re-label the test onsets.
- **Movie domain:** About 44% of XD-Violence test violent videos come from movies. Report them separately.
- **Metric:** Copy the accident-anticipation protocol. Per-frame risk score, then threshold, then report **AP, mTTA (mean lead time of correct alarms), recall at a fixed lead time (≥30 s), and false alarms per hour on long normal video**. Frame-level AUC alone penalises early alarms (noted in arXiv 2607.03558).

---

## 7. Links (primary sources)
- XD-Violence: https://roc-ng.github.io/XD-Violence/
- UCF-Crime: https://www.crcv.ucf.edu/projects/real-world/
- UCA (UCF-Crime Annotation): https://github.com/Xuange923/Surveillance-Video-Understanding · paper: https://openaccess.thecvf.com/content/CVPR2024/html/Yuan_Towards_Surveillance_Video-and-Language_Understanding_New_Dataset_Baselines_and_Challenges_CVPR_2024_paper.html
- HIVAU-70k / Holmes-VAU: https://github.com/pipixin321/HolmesVAU · https://arxiv.org/abs/2412.06171
- FS-UCF-Crime (pending): https://zenodo.org/records/21336651
- UBI-Fights: https://socia-lab.di.ubi.pt/EventDetection/ · code: https://github.com/DegardinBruno/human-self-learning-anomaly
- NTU CCTV-Fights: https://rose1.ntu.edu.sg/dataset/cctvFights/
- DVD: https://arxiv.org/abs/2506.05372
- MSAD: https://msad-dataset.github.io/ · https://github.com/Tom-roujiang/MSAD
- NWPU Campus: https://campusvad.github.io/ · https://arxiv.org/abs/2305.13611
- CUVA: https://github.com/fesvhtr/CUVA · ECVA: https://github.com/Dulpy/ECVA
- Hawk: https://github.com/jqtangust/hawk
- CamNuvem: https://github.com/daviduarte/camnuvem-dataset
- MA-CBP: https://arxiv.org/abs/2508.06189
- UCF-Crime-DVS: https://github.com/YBQian-Roy/UCF-Crime-DVS
- GTA-Crime: https://github.com/ta-ho/GTA-Crime
- RWF-2000: https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection
- RLVS: https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset
- SCVD: https://www.kaggle.com/datasets/toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd
- Bus Violence: https://ciampluca.github.io/bus_violence_dataset/ · https://zenodo.org/record/7044203
- Hockey Fights: https://academictorrents.com/details/38d9ed996a5a75a039b84cf8a137be794e7cee89
- Violent-Flows: https://www.openu.ac.il/home/hassner/data/violentflows/
- AIRTLab: https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos
- VioPeru: https://github.com/hhuillcen/VioPeru
- VSD2014: https://www.interdigital.com/data_sets/violent-scenes-dataset
- ShanghaiTech (CVF COVE): https://cove.thecvf.com/datasets/562
- NTU RGB+D 120: https://rose1.ntu.edu.sg/dataset/actionRecognition/
- AudioSet ontology: https://github.com/audioset/ontology · VGGSound: https://www.robots.ox.ac.uk/~vgg/data/vggsound/ · FSD50K: https://arxiv.org/abs/2010.00475
- MIVIA audio events: https://mivia.unisa.it/datasets/audio-analysis/mivia-audio-events/
- Audio-based Violence Detection (Kaggle): https://www.kaggle.com/datasets/fangfangz/audio-based-violence-detection-dataset
- MuTox: https://arxiv.org/abs/2401.05060 · ADIMA: https://huggingface.co/datasets/callmesan/ADIMA
- SSPNet Conflict / escalation: https://www.dcs.gla.ac.uk/~vincia/papers/escalation.pdf
- NAA: https://dl.acm.org/doi/10.1145/3242969.3243013
- BERSting: https://arxiv.org/abs/2505.00059
- Can We Anticipate Violence?: https://arxiv.org/abs/2609.40014
- Latent Clarity / PULS: https://arxiv.org/abs/2607.03558
- Pre-Event Separability: https://arxiv.org/abs/2608.27879
- Chang et al. 2026 (Siamese onset branch): https://www.sciencedirect.com/science/article/abs/pii/S092523122600929X
- RiskCueBench: https://arxiv.org/abs/2601.03369
- DeEscalWild: https://arxiv.org/abs/2604.13075
- OpenBWC: https://arxiv.org/abs/2605.17095
- Accident anticipation survey (DAD/CCD/DADA): https://arxiv.org/abs/2308.15985

_Reproducibility: the lead-in statistics in §2 were computed by scratch scripts that are not committed. The inputs were HIVAU-70k `raw_annotations/*_train.json`, the local UCF/XD test annotation files, UBI-Fights `annotation/F_*.csv` (fetched with HTTP range reads from the remote zip), and `data/anticipation_curation/candidates.json`. To reproduce: first event start ≥ 30 s, filtered by label set as described in §2._
