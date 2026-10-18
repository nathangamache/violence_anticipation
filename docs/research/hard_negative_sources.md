# Hard-negative and near-miss video sources

Research note, written 2026-10-18. Goal: footage of crowded or multi-person real-world scenes (streets, bars, transit, stores, parking lots, schools, protests, sports crowds, bodycam, road incidents) that does NOT turn violent, to measure false alarms per hour (kind 1, "same setting, never violent") and to stress the model on escalation that de-escalates (kind 2, "near misses"). Complements [datasets.md](datasets.md) (positives and lead-in) and [../DATA_ACQUISITION.md](../DATA_ACQUISITION.md).

Verification key: V = read on an official page or registry in this session; S = from a search snippet or paper listing only; U = unverified, taken from prior repo notes or memory. Nothing was downloaded.

Scores are 1-5. (a) = same-setting hard negatives, (b) = near-miss de-escalation.

## 1. Headline finding

No public dataset provides labelled near-miss (heated but non-violent) real-world video with long continuous lead-in. The nearest are DeEscalWild (police interactions with de-escalation signals, but released as transcripts, video unclear) and EgoPolice (bodycam, access unclear). Near misses must be curated from YouTube (section 4) and hand-labelled, using the same pipeline as `setup/curate_youtube.py` but keeping the non-violent outcomes. Public datasets only supply kind 1, and mostly in surveillance settings with few interpersonal interactions.

## 2. Candidate sources: same-setting never-violent footage (kind 1)

| Source | Contents | Violent? | Videos / length / hours | Res / audio | License and access | Size | Official URL | (a) | (b) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| MEVA (Kitware, DIVA) | 29 overlapping and non-overlapping fixed cameras, scripted and unscripted activity by about 100 actors, urban sites, long continuous recordings | Mostly non-violent (people interactions, vehicles, carrying, talking); no fights listed (S) | 4,259 clips, about 328 h public (9,300 h collected), annotations about 184 h; UAV 4.6 h (V) | Res not stated on registry (paper says mixed visible and IR, S); no audio stated | AWS Open Data, no account: `aws s3 ls --no-sign-request s3://mevadata-public-01/`; license text at mevadata.org (paper says CC-BY-4.0, S) | 516 GB video + 26 GB UAV (V); a subset of cameras is enough | https://registry.opendata.aws/mevadata/ ; https://mevadata.org | 4 | 1 | Best source of long (minutes to hours) continuous, fixed-camera multi-person footage for FA/h. Actors are staged but half unscripted. Few crowds, no verbal aggression. Download only a few sites/hours |
| VIRAT Ground 2.0 | 11 outdoor scenes (parking lots, building entrances, streets), stationary ground cameras | Non-violent (people-vehicle, carrying, gesturing) (V) | about 8.5 h HD (V); video count not stated | HD, person height 10-200 px, no audio | VIRAT Video Dataset Protection Agreement (PDF on site); data on data.kitware.com | Not listed | https://viratdata.org/ | 3 | 1 | Parking-lot setting matches target but low density and small people. 8.5 h is small |
| ShanghaiTech Campus | 13 campus scenes, 330 train + 107 test videos, 130 anomalous events (cycling, chasing, brawl-like) | Training split normal; test has a few fights/chasing (S) | 437 videos, short (sub-minute to minutes); about 270k train frames | 856x480, no audio (S) | Research use; page `sviplab.github.io/dataset/campus_dataset.html` returned 404 in this session; mirror via CVF COVE https://cove.thecvf.com/datasets/562 (U) | about 4-6 GB (U) | https://cove.thecvf.com/datasets/562 | 3 | 1 | Campus walkways with moderate crowds; train videos are clean normals, mostly under 1 min. Use train split only. Clips of 30 s+ are a minority (U) |
| UCSD Ped1/Ped2 | Fixed camera over a walkway, sparse to crowded; anomalies are bikes, carts, skaters | Non-violent | Ped1 34 train + 36 test clips; Ped2 16 train + 12 test clips (S); each clip about 200 frames, i.e. about 7-20 s | 238x158 (Ped1), 360x240 (Ped2), no audio | Free download, UCSD SVCL page (U) | under 1 GB | http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm (U) | 1 | 1 | Too short (under 30 s per clip) and low resolution; not useful for 30 s anticipation |
| CUHK Avenue | Single campus-walkway camera, 16 train + 21 test clips, 47 abnormal events (throwing, running, loitering) | Non-violent | about 30,652 frames in total, about 30 min (S) | 640x360, no audio | CUHK page, free (U) | about 1 GB | http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html (U) | 1 | 1 | Clips 1-2 min, one scene; small |
| UMN crowd | 11 clips, 3 scenes (field, courtyard, indoor foyer) of people milling, ending in staged panic running | Staged panic, no violence (S) | 7,740 frames total, 320x240, about 5 min total (S) | 320x240, no audio | Free, mha.cs.umn.edu (U) | tiny | http://mha.cs.umn.edu/proj_events.shtml (U) | 1 | 1 | Too short and tiny; skip |
| Street Scene (MERL) | One USB camera on a two-lane city street with sidewalks and bike lane; 46 train + 35 test sequences, 205 anomalies in 17 types (jaywalking, loitering, illegal turns) | Non-violent | 203,257 frames at 1280x720 (about 5-6 h at 10-15 fps) (S) | 1280x720, no audio | MERL license form (non-commercial); official page https://www.merl.com/research/downloads/StreetScene (S) | about 80 GB (third-party mirror listed 34 GB test + 46 GB train, S) | https://www.merl.com/research/downloads/StreetScene | 3 | 1 | Genuinely long continuous street-corner footage with many pedestrians, good for FA/h. Single scene, no interpersonal conflict. Large download |
| NWPU Campus | 43 scenes, 547 videos, 16.3 h, 25 fps; training set is normal-only; test includes scuffle and group conflict | Mixed: train normal, test has anomalies (U) | 16.3 h | Mixed, no audio | Research-only, Google Drive/Baidu form at https://campusvad.github.io/ (U) | large (U) | https://campusvad.github.io/ | 4 | 2 | Normal-only training split gives many hours of campus and street normals with 43 scene variety. Its test set has scuffle events with anticipation protocol |
| MSAD normal split | 14 scenarios (street, shop, restaurant, park), about 500 cameras; 360 normal train + 120 normal test videos | Normal splits clean | 720 videos total (S); lengths untrimmed | RGB, no audio | Project site https://msad-dataset.github.io/ ; mirror https://time.griffith.edu.au/paper-sites/msad (S) | about tens of GB (U) | https://msad-dataset.github.io/ | 4 | 1 | Best "same setting" match (shops, restaurants, streets). Request/form status not confirmed here (S) |
| UCF-Crime normal | 950 normal videos (already on disk, unused) + 150 train normals | Normal | Already local | Mostly silent | Already have | local | n/a | 4 | 1 | Use first; free |
| UBI-Fights normal | 784 normal videos (already planned) | Normal | Already planned | None | Direct zip | 8.1 GB | n/a | 3 | 1 | Already in plan |
| XD-Violence train non-violent | Already on disk under `data/raw_videos/xd_violence_train` | Normal | local | Audio yes | Have | local | n/a | 4 | 1 | Same source as positives, so shortcut-controlled negatives |
| NTU CCTV-Fights (non-fight regions) | Received 2026-10-15; 1,000 videos with fight spans labelled | Mixed; non-fight spans of each video are clean negatives | 17.7 h | Partial audio | Have | local | n/a | 4 | 3 | Pre-fight and between-fight stretches in the same CCTV feed are the best same-source negatives; some contain scuffles that did not escalate (U, needs viewing) |
| RWF-2000 NonFight | 1,000 five-second non-fight clips from YouTube surveillance | Non-violent | 1,000 clips x 5 s | various, some audio | GitHub with signed agreement; Kaggle mirrors (U) | about 7 GB (U) | https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection | 2 | 1 | Clips of 5 s: TOO SHORT for 30 s anticipation. Scene-diverse, but no context |
| RLVS non-violence | 1,000 non-violent clips (sports, eating, talking, walking) | Non-violent | 1,000 clips, avg about 5 s | mixed, some audio | Kaggle (login), license unverified: https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset | about 4 GB (U) | same | 1 | 1 | TOO SHORT (about 5 s) and not crowd-in-context; Kaggle page did not render in this session |
| SCVD non-violence | 248 non-violent CCTV-like clips | Non-violent | short | no audio | Kaggle (U) | small | https://www.kaggle.com/datasets/toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd | 1 | 1 | Short clips; skip |
| Hockey / Movies fights | Hockey 500 fight + 500 non-fight (1-2 s); Movies 100+100 | Mixed | 1-2 s clips | no | open | small | U | 1 | 1 | TOO SHORT (1-2 s); movie clips staged |
| AIRTLab hard negatives | 120 non-violent indoor acted clips (hugs, claps), avg 5.6 s | Non-violent, acted | 120 | no | GitHub | small | https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos | 1 | 2 | Acted and short (about 5 s) |
| Kinetics-700 arguing / shouting | Large numbers of YouTube 10 s clips (641 / 644 train per local csv) | Verbal only, no physical violence | 10 s each | with audio | Already partially local (700 clips) | local csv | https://github.com/cvdfoundation/kinetics-dataset | 1 | 3 | 10 s clips: usable as verbal-escalation cue head training or near-miss content, but too short for 30 s build-up |
| NTU RGB+D pushing/shoving | Lab-recorded, Kinect | Staged | 56k clips | depth | Request form | Large | U | 1 | 1 | Staged, plain background, not a real-world setting. Skip |
| Caltech Pedestrian, MIT Traffic, USVD | Street/crossroads/school-gate/mall scenes with 90 min (MIT) to 10 h (Caltech, vehicle-mounted) | Non-violent | MIT: 90 min, 20 clips; Caltech about 10 h (S) | 640x480 | Free / varied | Medium | U | 2 | 1 | Pedestrian counts high; no interpersonal behaviour. MIT Traffic is one fixed scene (S) |

## 3. Candidate sources: near-miss and conflict without violence (kind 2)

| Source | Contents | Violent? | Size | Access | Official URL | (a) | (b) | Notes |
|---|---|---|---|---|---|---|---|---|
| DeEscalWild (2026) | 1,500 scenarios of police-civilian interactions mined from YouTube, TikTok, Facebook; filter keeps videos with escalation and de-escalation signals; Whisper transcripts + Gemini diarization; 285,887 dialogue turns (V) | Mixed; includes physical and verbal conflict, de-escalation attempts (V) | hours not stated (V) | Code MIT on GitHub; data at Harvard Dataverse DOI 10.7910/DVN/CWMCZI. Dataverse page did not load; whether it contains video, or only transcripts and source links, is UNVERIFIED | https://arxiv.org/abs/2604.13075 ; https://github.com/Hasebul/DeEscalWild-Benchmark-Framework ; https://doi.org/10.7910/DVN/CWMCZI | 2 | 4 | Best lead for near misses: the selection pipeline is reproducible and the candidate source URLs/channels can seed our own YouTube search. Check Dataverse for video links before relying |
| EgoPolice (2026) | Over 180 h of police body-worn-camera footage, Chicago COPA, Pasadena, Dallas, DC, LA, San Antonio; per-second labels of 9 action classes (V) | Skewed to high-stakes (shootings, injuries); includes routine conversations (V) | 180+ h | Annotation code on the project site; dataset URL and license not given in the text read; paper warns about content (V) | https://huggingface.co/papers/2607.06468 | 2 | 3 | Long untrimmed bodycam with per-second Physical Interaction labels; seconds before Physical Interaction and non-violent stretches give both onset and near-miss material. Access terms UNVERIFIED |
| FALEBaction (2025, EURECOM) | Bodycam multimodal action recognition | Mixed | unknown | On request from authors (S) | https://www.eurecom.edu/en/node/4900128 | 2 | 2 | Request-only; content unconfirmed |
| OpenBWC (RIT + Rochester PD) | Body-worn camera, 10 s window labels | Mixed | unknown | Access-controlled, no public release (U, from datasets.md) | n/a | 1 | 2 | Not obtainable |
| CONFER (Georgakis et al., Image and Vision Computing 2017) | Greek televised political debates, 120 episodes, 142 min, frame-level real-valued conflict intensity, audio and video (S) | Verbal conflict only, TV studio | about 2.4 h | Request via Imperial iBUG (U); official page returned 500 error here | https://ibug.doc.ic.ac.uk/resources/conflict-escalation-resolution-database/ | 1 | 3 | Only dataset with graded verbal conflict intensity over time, but studio talk shows, not the fight-prone settings; useful for training an argument-intensity audio-visual cue only |
| NAA - Negative Affect and Aggression (TU Delft) | Acted dyadic aggression, audio-video-mocap, no physical violence | Non-violent, acted | unknown | Request (U, from datasets.md) | n/a | 1 | 3 | Acted; cue-head training only |
| AVABOS (ITMO, 2024) | Multimodal corpus of verbal and physical aggression from online video streams, Russian speakers (S) | Mixed | unknown | Unknown | https://ntv.ifmo.ru/en/article/23093/ | 1 | 2 | Moderate agreement (kappa 0.48) on verbal aggression; availability unverified |
| Violent Interactions project (video analysis of near-violent situations in public space) | Coding of internet footage of violent and near-violent public incidents, with de-escalation behaviour | Mixed | unknown | Research project, not a dataset (S) | https://violent-interactions.org/current-projects/video-analysis | 1 | 3 | Contact the authors (Dhattiwala, van der Duin, Weenink) for the footage list; rare lead on labelled de-escalation in public space |
| MSAD anomaly types | 35 human anomaly types incl. arguing-adjacent categories | Mixed | see above | see above | https://msad-dataset.github.io/ | 2 | 1 | Scene-specific; no de-escalation label |

## 4. YouTube sources and queries (near misses)

All footage must be real (not staged, not movie). Run through `setup/curate_youtube.py`-style download, but label manually with a four-way outcome: stays verbal, de-escalates, escalates later, unclear. Provenance list (URL plus timestamps) only; do not redistribute.

Search queries (try with and without "full video" and "raw footage"):

1. `road rage confrontation no fight walks away dashcam`
2. `road rage standoff drivers get out argue drive off`
3. `bodycam de-escalation calm arrest no force full video`
4. `police bodycam heated argument turns peaceful`
5. `first amendment auditor police confrontation argument no arrest`
6. `bar argument bouncer breaks up before fight cctv`
7. `bar security de-escalates drunk customer`
8. `bus driver passenger argument cctv no violence`
9. `subway platform argument shouting security stops`
10. `parking lot confrontation caught on camera argument ends`
11. `protest counter-protester face to face shouting no violence`
12. `supermarket customer argument manager calms down`
13. `school hallway argument teacher intervenes surveillance`
14. `neighbors yelling at each other ring doorbell camera`
15. `crowd tense standoff police line no clashes live stream`

Known channels and compilations (names from general knowledge; UNVERIFIED, check each before use and respect ToS):

| Type | Examples | Why useful |
|---|---|---|
| Police bodycam archive channels | Official police department channels and city transparency portals (e.g. Chicago COPA releases, Pasadena PD) are the sources behind EgoPolice (V) | Long, minimally edited, includes routine contacts |
| Bodycam compilation channels | PoliceActivity, Audit the Audit, LA Video, Lemonade Stand (U) | Many tense-but-nonviolent encounters; compilations are edited, prefer full-length uploads |
| Road-rage and dashcam channels | Dashcam compilation channels such as Dash Cam Owners Australia, Road Rage USA (U) | Standoffs where drivers do not fight; edited compilations contain violence too, so filter |
| Live webcams and streams | EarthCam, Explore.org, Times Square and Shibuya crossing 24/7 streams, YouTube "live city cam" (U) | Hours of crowded never-violent footage with no editing; ideal for FA/h; check terms before recording a stream |
| Nightlife venue cams | Bourbon Street and Beale Street live cams (U) | Matches bar/nightlife setting; occasional fights occur, review needed |
| Security-camera-style uploads | Surveillance Archive style channels and 'caught on CCTV' channels (U) | Mixed; have escalations but also many non-escalations |

Note on curation: the local curated set was built from LSTM pseudo-labels and is circular for VLM features; near-miss material should be labelled by human review.

## 5. Cost and licensing summary

| Source | Access effort | Disk | License risk |
|---|---|---|---|
| MEVA | one command, no account | choose 20-50 h, 30-80 GB | low (CC-BY-4.0 S) |
| Street Scene | MERL form | about 80 GB | non-commercial |
| MSAD | project site / form | tens of GB | research |
| NWPU Campus | form / Drive | large | research only |
| VIRAT | signed agreement | small | restricted |
| DeEscalWild | Dataverse (check) | small if transcripts | unknown |
| EgoPolice | unknown | large | unknown, graphic content |
| YouTube near misses | own curation | tens of GB | fair-use research; no redistribution |
| Live webcam recordings | own recording | hours at low bitrate | check each stream's terms |

## 6. Recommended order (see also summary in chat)

1. Already local: UCF-Crime unused normals, XD train non-violent, UBI-Fights normals, CCTV-Fights non-fight spans. Build the FA/h measurement set from these first (same-source rule).
2. MEVA subset via S3 (no form): the only large freely downloadable long continuous multi-person set.
3. MSAD normal splits and NWPU Campus normal train split: scene-diverse, request needed.
4. DeEscalWild: inspect Dataverse for video links, reuse its channel list for YouTube near-miss curation.
5. Hand-curated YouTube near misses and 24/7 city webcam recordings: the only practical source for kind 2.
6. Skip as too short for 30 s anticipation: UCSD, UMN, RWF-2000, RLVS, SCVD, Hockey, Movies, AIRTLab, Kinetics (10 s).
