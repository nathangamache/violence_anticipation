# Data, Datasets, Labels and Splits

> Paths in this file describe the tree before the 2026-10 cleanup. Surviving code is under `legacy/<same path>`;
> the honest runs are under `baselines/reference_runs/`; removed tracked files: `git show pre-cleanup:<path>`;
> ignored weights and arrays: `data/archive/repo_legacy/<same path>`
> until 2027-01-12. See [../REPO_CLEANUP_PLAN.md](../REPO_CLEANUP_PLAN.md).

Audience: future AI agents and developers who need to reuse the data in this repo to build a new
violence-anticipation model (goal: predict violence 30+ s before it starts).

Everything below was verified by reading the code and by cheap read-only inspection of the files on
disk (Oct 2026). Anything not directly verified is marked **(inferred)**.

---

## 0. TL;DR (read this first)

* **Where the data is:** `data/` in the repo is a symlink to `data/`
  (raw videos 134 GB, `processed/` 11 GB, `windowed/` 8.7 GB). Pretrained VLM: `models/` ->
  `models/Qwen2.5-VL-7B-Instruct` (only model in that dir).
* **Sources on disk:** UCF-Crime (1000 videos used + 950 extra normal videos never used),
  XD-Violence **test split only** (798 clips: 500 violent, 298 non-violent), a 700-clip Kinetics-700
  subset (`wrestling`, `slapping`, 10 s clips), and 92 self-curated YouTube escalation videos
  (road rage / bodycam / CCTV confrontations).
* **How violence onset is known:**
  * XD-Violence: official, manual, frame-level annotations (`raw_videos/xd_violence/annotations.txt`,
    24 fps, clip-relative frame indices) for all 500 violent clips. Reliable.
  * UCF-Crime: official, manual, frame-level annotations **only for the 140 test anomaly videos**
    (`Temporal_Anomaly_Annotation_for_Testing_Videos.txt`, 30 fps). The other 810 anomaly videos have
    **no onset** and the pipeline treats them as all-negative.
  * Curated YouTube: **no human labels.** "violence_start" was produced by the project's own LSTM
    (trained on XD) via `experiments/fast_analyze_curation.py`; every start is >= 59 s because the
    first 60 s are force-labelled `ramp_up`. Treat as unreliable pseudo-labels.
  * Kinetics: synthetic `event_frame = total_frames // 2`; no real onset, no pre-violence context.
  * VLM (Qwen2.5-VL) per-frame JSON labels exist (`processed/vlm_labels*`) but are features/soft
    labels, not used as onset ground truth.
* **Videos with a usable >= 30 s pre-onset segment (from the official annotations): 112 total**
  (XD 53, UCF 59). Only **49** of those are interpersonal violence (fight/shooting/abuse/assault/
  arrest/robbery: XD 27, UCF 22); the rest are explosions, riots, car accidents, shoplifting, arson,
  burglary, etc. >= 60 s: 49 total (18 interpersonal). Total pre-onset footage in the >=30 s set:
  ~2.35 hours. This is the core data bottleneck.
* **Main label set:** `processed/labels_multih/*.npy` (`[T, 7]`, horizons 2,5,10,20,30,60,120 s,
  `-1` = inside violence). Built from the official annotations, so it is correct in principle, but
  the time axis `T` is inherited from the YOLO graph pipeline and is irregular (see section 7).
* **Critical bugs found (details in section 11):**
  1. `setup/build_xdviolence_metadata.py` subtracts the clip's movie timestamp from already
     clip-relative annotations, so **274 of 500 violent XD clips are labelled non-violent** in
     `metadata/xd_violence.json` (and 11 more get a wrong onset). Everything derived from the
     metadata (`processed/labels`, `labels_future_h*`, `windowed/`, the graph pipeline) inherits this.
  2. **Embedding/label misalignment:** `vlm_emb/*.npz` stores one embedding every 5th label step
     (`t = 0,5,10,...`), but `training/train_temporal_model.py`, `training/train_student.py`,
     `scripts/build_fused_features.py`, `scripts/multimodal_fusion.py`, `sim/build_long_simulation.py`,
     `sim/create_ucf_crime_simulation.py`, `tuning/contrastive_projection.py` etc. pair embedding `i`
     with label row `i` (`labels[:min_len]`) instead of label row `t[i]`. 15% of XD embedding steps
     get the wrong label and the number of positive steps nearly doubles. Only
     `baselines/train_student_original.py` indexes labels by `t` correctly.
  3. Those same loaders collapse `[T,7]` with `max(axis=1) > 0`, i.e. they train on the **120 s**
     horizon (not 30 s) and turn the `-1` "inside violence" steps into **negatives**.
  4. 357 UCF videos in violent categories (Abuse/Arrest/Assault/Fighting/Robbery/Shooting) without
     temporal annotation are used as **all-negative** videos.
  5. Root `train/val/test_index.json` split by clip name only; 46 val and 44 test clips share a
     source movie / YouTube video with train (leakage). `eval_index.json` == `test_index.json`.
     Test contains only 12 videos with onset >= 30 s (3 XD, 9 UCF); with the default
     `--source_filter xd_violence` the 30 s-anticipation test set is effectively **3 videos**.
  6. Inter-embedding spacing in `vlm_emb` is median 2.7 s but 90th percentile 18 s and max 756 s,
     so many videos have far too coarse sampling for 30 s anticipation.

---

## 1. Storage layout

```
data/        (= <repo>/data)
  raw_videos/
    ucf_crime/          123 GB  (see 2.1)
    xd_violence/         11 GB  annotations.txt + videos/ (798 mp4)
    kinetics/           863 MB  K700 csvs + videos/ (700 mp4)
  anticipation_curation/ 1.9 GB candidates.json + videos/ (92 mp4)
  metadata/             ucf_crime.json, xd_violence.json, kinetics.json, kinetics_labels.txt
  windowed/             8.7 GB  ucf_crime/ (759 mp4), xd_violence/ (690 mp4)  trimmed clips
  processed/            11 GB   features + label sets (see sections 6 and 8)
  plots/                6 risk-curve PNGs from scripts/inspect_risk_curves.py
```

Repo-root data-ish files: `train_index.json` (1438 items), `val_index.json` (180), `test_index.json`
(180), `eval_index.json` (identical to test), `good_video_indices.npy` (433 sorted ints, max 766,
**no code references it**; provenance unknown). `test_video.mp4` at repo root was byte-identical to
`anticipation_curation/videos/holdout_025.mp4` and was deleted as a duplicate in the 2026-10 cleanup; the same clip is
`data/anticipation_curation/videos/holdout_025.mp4`. `.gitignore` excludes `*.npy`, `*.npz`, `*.mp4`, so
`good_video_indices.npy` and all features/videos are not in git.

Hard-coded paths: almost all setup/preprocessing scripts use
a hard-coded home-relative path to the project's `data` directory; this works only if that path resolves to
the repo's `data` symlink. `scripts/quick_inspect.py`
hard-codes `data`. Newer scripts use relative
`data/...` defaults and must be run from the repo root.

---

## 2. Raw datasets

### 2.1 UCF-Crime (`raw_videos/ucf_crime/`)

| Folder | Count | Used by pipeline? |
|---|---|---|
| `videos/` | 1000 mp4 = 950 anomaly + 50 normal | YES (this is the only folder the scripts read) |
| `Anomaly-Videos/<Category>/` | 950 mp4 (byte-copies of the anomaly files in `videos/`, separate inodes, 24 GB duplicated) | no (but `rglob` in some scripts can resolve to it) |
| `Normal_Videos_for_Event_Recognition/` | 50 mp4 (same 50 normals as in `videos/`) | indirectly |
| `Training-Normal-Videos/` | 800 mp4, 69 GB | **NO - unused** |
| `Testing_Normal_Videos/` | 150 mp4, 4.5 GB | **NO - unused** |
| `UCF_Crimes-Train-Test-Split/` | official `Anomaly_Detection_splits/{Anomaly_Train,Anomaly_Test}.txt` and `Action_Regnition_splits/` | no |
| `Anomaly_Train.txt` | 1610 lines (810 anomaly + 800 normal) | no |
| `Temporal_Anomaly_Annotation_for_Testing_Videos.txt` | 290 lines (140 anomaly + 150 normal) | YES - the only UCF onset source |
| `ReadMe-Anomaly-Detection.txt` | official readme | - |

`videos/` composition: Robbery 150, RoadAccidents 150, Burglary 100, Stealing 100, and 50 each of
Abuse, Arrest, Arson, Assault, Explosion, Fighting, Shooting, Shoplifting, Vandalism, Normal.
FPS: 999 videos at 30.0, one at 29.97. Total duration 35.1 h, median 70 s, max 4730 s.

Annotation format (from the official readme): `video  category  start1 end1  start2 end2` in frames
at 30 fps, `-1` = no second instance. Annotated anomaly videos per category: RoadAccidents 23,
Shooting 23, Explosion 21, Shoplifting 21, Burglary 13, Arson 9, Arrest 5, Fighting 5, Robbery 5,
Stealing 5, Vandalism 5, Assault 3, Abuse 2. 16 of the 140 have a second segment. 5 annotations
end beyond the cv2 frame count of the file.

Audio: 862/1000 files have an audio stream, but a random sample of 16 showed 14 at digital silence
(-91 dB mean/max); UCF audio is effectively absent. Do not count on UCF for the audio modality.

### 2.2 XD-Violence (`raw_videos/xd_violence/`)

* 798 mp4 clips = the XD-Violence **test** set (500 violent + 298 non-violent; official test set is
  800, 2 non-violent clips missing). The XD **training** set (~3954 videos, video-level labels only)
  is not on disk.
* 422 YouTube clips (`v=<id>__#...`), 376 movie clips (`Movie.Name.Year__#HH-MM-SS_HH-MM-SS_...`).
* Filename encodes the label after `_label_`: `A` = non-violent, `B1` fighting, `B2` shooting,
  `B4` riot, `B5` abuse, `B6` car accident, `G` explosion; up to 3 labels joined by `-`, `0` = none.
  Primary-label counts among violent clips: B1 120, B4 101, B6 96, G 91, B2 84, B5 8.
* `#HH-MM-SS_HH-MM-SS` in a filename is the position of the clip inside the **source movie/video**,
  not inside the clip. `#1` means "whole video, no offset".
* `annotations.txt`: 500 lines, `vid s1 e1 s2 e2 ...` (even count, frame indices **relative to the
  clip**, 24 fps). 14 lines have a trailing `.mp4` on the id (strip it). 260 clips have more than
  one violent segment. All annotation frames fall inside the clip.
* All clips are 24.0 fps. Total 26.96 h (violent clips 12.5 h). Violent clip duration median 67 s.
* Audio: 798/798 have audio streams with real content (sampled loudness -17 to -36 dB mean).
  XD is the only labelled source with usable audio.
* Same source movie/YouTube video often contributes several clips (e.g. `Bad.Boys.1995`,
  `Sin.City.2005` x8, `v=wVey5JDRf_g` x3). Group by the part before `__#` when splitting.

### 2.3 Kinetics-700 subset (`raw_videos/kinetics/`)

* `train.csv`, `val.csv`, `test.csv`, `k700_2020_*_path.txt`, `K700_2020_readme.txt` (official K700
  lists) and `videos/` with 700 clips named `<youtube_id>_<start>_<end>.mp4`.
* Classes (recovered from the csvs; the metadata lost them): `wrestling` 395, `slapping` 305. Most
  of the `TARGET_CLASSES` in `setup/kinetics_download_relevant.py` (e.g. "punching person",
  "kicking") do not exist under those names in K700, so only two classes were downloaded.
* ~10 s clips (median 9.25 s), 695/700 with audio. No pre-violence context, so **useless for
  anticipation**; at best a short-clip "violent action" auxiliary. Only used by
  `processed/kinetics_actions` and `setup/videomae_finetune_kinetics.py`.

### 2.4 Curated YouTube escalation set (`anticipation_curation/`)

* Produced by `setup/curate_youtube.py search` + `download` (YouTube Data API key read from the
  environment or a local `.env`, or yt-dlp search fallback), then analyzed by
  `experiments/fast_analyze_curation.py` (NOT by `setup/curate_youtube.py analyze`; the stored
  timeline schema `{frame,timestamp,stage,score,raw_score}` and stage names `ramp_up`/`aftermath`
  only exist in `fast_analyze_curation.py`).
* `candidates.json`: 97 candidates (created 2026-03-22), 92 downloaded, 5 failed. Duration 241-586 s
  (median 373 s), 9.98 h total, 92/92 have audio. Queries: road rage, bodycam, street/bar/club fights,
  CCTV confrontations - this is the **closest domain match** to the project's target.
* Labels are LSTM pseudo-labels at 0.5 fps: `has_violence` 47/92, `violence_start_second` always
  >= 59.06 s (artifact: `--ramp_up_frames 30` at 0.5 fps forces the first 60 s to `ramp_up`), so
  `escalation_start_second` is ~59-60 s for almost every video. Several "violent" hits are clearly
  wrong (e.g. "How to de-escalate someone", a music video, a news segment about Gaza) and several
  real fights are marked non-violent. **Do not use these as ground truth; hand-annotate them.**
* `processed/vlm_emb/curated/<video_id>.npz` (92 files): `emb [N,3584] f16`, `t`, `raw_frame_idx`
  (both just `arange(N)`, not real frame indices), `timestamps [N]` seconds (0.5 fps), `scores [N]`
  LSTM output.

### 2.5 Audio availability summary (ffprobe over every file)

| Source | Files | With audio stream | Real audio content |
|---|---|---|---|
| XD-Violence | 798 | 798 | yes |
| UCF-Crime `videos/` | 1000 | 862 | mostly digital silence (sampled) |
| Kinetics subset | 700 | 695 | yes (not checked in depth) |
| Curated YouTube | 92 | 92 | yes |

`processed/audio/*.npy` (1798 files) are **all-zero 64-d placeholders** from
`core.extractors.extract_audio_placeholder`. Real audio features exist only in
`processed/audio_features/xd_violence/` (91-d, see section 6).

---

## 3. How the violence onset is known (per source)

| Source | Onset source | Type | Videos with onset | Notes |
|---|---|---|---|---|
| XD-Violence | `annotations.txt` | official manual, frame-level, multi-segment | 500 violent | 88 start at frame 0 (no pre-onset at all) |
| UCF-Crime | `Temporal_Anomaly_Annotation_for_Testing_Videos.txt` | official manual, frame-level, <=2 segments | 140 test anomalies | the other 810 anomaly videos have no onset |
| Curated YouTube | `candidates.json` | LSTM pseudo-label (model trained on XD) | 47 "violent" | unreliable, >= 59 s artifact |
| Kinetics | `metadata/kinetics.json` | synthetic midpoint | 625 | meaningless for anticipation |
| (VLM) | `processed/vlm_labels*/` | Qwen2.5-VL per-frame JSON (`violent.label/score`, actions, weapons, ...) | 1721 | soft features, not used as GT |

Pre-onset availability from the official annotations (onset = first annotated frame / fps):

| Onset >= | Total | XD | UCF | Interpersonal total (XD / UCF) |
|---|---|---|---|---|
| 10 s | 275 | 185 | 90 | 127 (94 / 33) |
| 20 s | 169 | 97 | 72 | 76 (51 / 25) |
| **30 s** | **112** | **53** | **59** | **49 (27 / 22)** |
| 45 s | 68 | 29 | 39 | 27 (15 / 12) |
| 60 s | 49 | 18 | 31 | 18 (10 / 8) |

"Interpersonal" = XD labels containing B1/B2/B5; UCF categories Abuse/Arrest/Assault/Fighting/
Robbery/Shooting. Breakdown for onset >= 30 s: XD `G` 16, `B1` 16, `B4` 7, `B2` 4, `B6` 3, `B2+G` 2,
`B5` 2, others 1 each; UCF Shooting 12, Shoplifting 12, Explosion 10, Arrest 5, Burglary 5, Arson 4,
Stealing 3, Assault 2, Vandalism 2, Abuse/Fighting/RoadAccidents/Robbery 1 each.
XD onset distribution: median 5.9 s, 75th pct 14.4 s, 90th pct 32.4 s, max 201 s. UCF annotated:
median 20.7 s, 75th pct 51.2 s, max 378 s. Caveat (inferred): many XD movie/YouTube clips are edited
(scene cuts, compilations), so the "pre-onset" footage is not always causally related to the violence.

Correct way to get onsets (copy-paste; do NOT use `metadata/xd_violence.json` for XD onsets):

```python
import json
D = "data"

def xd_segments_sec():
    """vid_id -> [(start_s, end_s), ...]; frames are clip-relative at 24 fps."""
    out = {}
    for line in open(f"{D}/raw_videos/xd_violence/annotations.txt"):
        p = line.split()
        vid = p[0].removesuffix(".mp4")          # 14 ids carry a .mp4 suffix
        fr = list(map(int, p[1:]))
        out[vid] = [(fr[i] / 24.0, fr[i + 1] / 24.0) for i in range(0, len(fr), 2)]
    return out

def ucf_segments_sec():
    """vid_id -> [(start_s, end_s), ...] for the 140 annotated test anomalies."""
    meta = json.load(open(f"{D}/metadata/ucf_crime.json"))   # fps/total_frames are fine for UCF
    out = {}
    for line in open(f"{D}/raw_videos/ucf_crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"):
        p = line.split()
        if len(p) < 6 or p[1] == "Normal":
            continue
        vid, fps = p[0].removesuffix(".mp4"), meta[p[0].removesuffix(".mp4")]["fps"]
        a = list(map(int, p[2:6]))
        out[vid] = [(a[0] / fps, a[1] / fps)] + ([(a[2] / fps, a[3] / fps)] if a[2] >= 0 else [])
    return out
```

---

## 4. Negatives

* XD-Violence: 298 `_label_A` clips (movies like "Before.Sunset", "About.Time", and YouTube) -
  clean negatives, but movie/dialogue domain, not CCTV. Non-violent clip durations: median 141 s,
  14.4 h total.
* UCF-Crime: the 50 normal videos in `videos/`. Plus, in every pipeline here, **all 810
  unannotated anomaly videos are treated as negatives**, including 357 in violent categories
  (Abuse 48, Arrest 45, Assault 47, Fighting 45, Robbery 145, Shooting 27). These are label noise.
  The 950 extra normal CCTV videos (`Training-Normal-Videos/`, `Testing_Normal_Videos/`) are clean
  negatives that were never used.
* Pre-onset frames more than H seconds before onset and frames after a segment ends (aftermath)
  are negatives in `labels_multih` (aftermath frames are visually violent-looking negatives).

---

## 5. Metadata JSONs (`metadata/`)

All three are `dict[vid_id -> {event_frame, total_frames, label, fps, source}]`.

| File | Entries | label=1 | Producer | Problems |
|---|---|---|---|---|
| `ucf_crime.json` | 1000 | 140 | `setup/build_ucf_crime_metadata.py` | label=1 only for the 140 annotated test anomalies; `event_frame` = earliest of the two segment starts; all other anomalies label 0 |
| `xd_violence.json` | 798 | 226 | `setup/build_xdviolence_metadata.py` | **wrong**: 274 violent clips labelled 0 (260 due to timestamp offset bug, 14 due to `.mp4` key mismatch), 11 label-1 clips with wrong `event_frame`; only 215 correct |
| `kinetics.json` | 629 | 625 (+4 with string labels) | `setup/kinetics_download_relevant.py` then `setup/enrich_kinetics_metadata.py` | enrich overwrote class names with `1` and set `event_frame = total_frames//2`; 4 entries (source `kinetics700`, label `slapping`/`wrestling`, no fps) look like they were added by a download re-run after enrichment (inferred); 71 downloaded videos are not in the json |
| `kinetics_labels.txt` | 700 lines | - | `setup/kinetics_list_labels.py` | all K700 class names |

`total_frames`/`fps` are cv2 values of the full raw video (also used by later label builders).

---

## 6. Processed feature catalog (`processed/`)

Two families exist. **Legacy graph pipeline** (Nov 2025) works on *windowed* clips; the **VLM
pipeline** (Jan-Mar 2026) samples frames from the *full* raw video. Counts: 1798 = 1000 UCF + 798 XD;
1721 = the 1798 minus 77 videos whose graph sequence is empty (51 UCF, 26 XD; 22 of them positives).

| Dir | Count | Format / shape | Time axis | Producer |
|---|---|---|---|---|
| `tracks/<vid>.json` | 1798 | list over frames (stride 4 of the windowed clip) of `[{track_id, bbox[x1,y1,x2,y2]}]` (YOLOv8n persons, ids not tracked across frames) | windowed clip, every 4th frame | `scripts/preprocess_anticipation.py` + `core/extractors.extract_tracks` |
| `actions/<vid>.npy` | 1798 (+1 stray `v=pFamvR9CpYw__#00-06-36_00-15-20_label_B4-0-0.npy.tmp.npy` from the old np.save `.tmp` bug) | object array, per tracked frame `{track_id: 768-d VideoMAE crop emb}` | same as tracks | same, `extract_actions` |
| `scenes/<vid>.npy` | 1798 | `(512,)` CLIP mean over the window (stride 16) | global | same, `extract_scene` |
| `audio/<vid>.npy` | 1798 | `(64,)` **all zeros** | global | same, `extract_audio_placeholder` |
| `graphs/<vid>.npy` | 1798 | object array of `T` dicts `{frame_idx, nodes (N, 772 = 4 bbox + 768), adj (N,N), scene (512,), audio (64,)}`; frames with no person are **skipped** | irregular: only tracked frames with >=1 person | same, `core/graph.build_graph_sequence` |
| `labels/<vid>.npy` | 1798 | `(T,)` float risk curve | graph steps | same, `scripts/generate_labels.generate_risk_curve` (older version, see 8.1) |
| `kinetics_actions/<clip>.npy` | 625 | `(768,)` VideoMAE-base mean token, 16 uniformly sampled frames | clip | `scripts/preprocess_kinetics_for_videomae.py` |
| `vlm_emb/{ucf_crime,xd_violence}/<vid>.npz` | 949 + 772 | `t [n] int32`, `raw_frame_idx [n] int32`, `emb [n,3584] f16` (Qwen2.5-VL-7B hidden state) | `t = 0,5,10,...<T`; `raw_frame_idx = round((t+0.5)/T * total_frames)` over the **full** raw video | `scripts/vlm_label_dataset.py --every_t 5` |
| `vlm_emb_meanpool/`, `vlm_emb_old/`, `vlm_emb_violence/` | 949 + 772 each | same schema | same | `vlm_label_dataset.py` with other prompt/pooling modes (`violence` = `--prompt_mode violence_embedding`) (inferred from names + jsonl fields) |
| `vlm_emb/curated/` | 92 | see 2.4 | 0.5 fps | `experiments/fast_analyze_curation.py --save_embeddings` |
| `vlm_labels*/{ucf_crime,xd_violence}/<vid>.jsonl` | 949 + 772 each | one JSON per sampled `t`: `{vid_id, source, t, T, raw_frame_idx, violent:{label,score}, people_count, actions, weapons, scene, notes, ...}` | same as vlm_emb | `scripts/vlm_label_dataset.py` |
| `audio_features/xd_violence/<vid>.npz` | 798 | `features [T, 91] f32`, `feature_dim` | one row per **label step** (`T` = labels_multih length) | `scripts/audio_feature_extraction.py` (inferred) |
| `pose_features/xd_violence/<vid>.npz` | 798 | `features [n, 540]`, `frame_indices [n]` | ~same frames as vlm_emb (indices 0,120,240,... vs vlm 12,131,250,...) | `scripts/pose_feature_extraction.py` (YOLOv8m-pose) |
| `optical_flow/xd_violence/<vid>.npz` | 772 | `features [F, 22]`, `frame_indices`, `feature_names`, `video_info` | every raw frame | `tuning/optical_flow_features.py` (inferred) |
| `fused_features/xd_violence/` | 772 | `emb [n, 4124] f16` = vlm 3584 + pose 540, `concat_norm` | vlm steps | `scripts/build_fused_features.py` (`fusion_metadata.json`) |
| `vlm_audio_features/xd_violence/` | 772 | `emb [n, 3675] f16` = vlm 3584 + audio 91 | vlm steps | `scripts/multimodal_fusion.py` (inferred) |
| `fused_vlm_flow/` | 772 | `emb [n, 3606]` = vlm 3584 + flow 22 | vlm steps | `tuning/fuse_features.py` (inferred) |
| `fused_finetuned_flow/` | 772 | `emb [F, 22]` - **only the 22 flow dims**, no finetuned-VLM part | raw frames | `tuning/fuse_features.py` (inferred; looks broken) |

`windowed/<source>/<vid>.mp4` (759 UCF + 690 XD): ffmpeg `-c copy` trims made by
`preprocess_anticipation.py`. Positives: `[event - 30 s, event + 10 s]`; negatives (including the 274
mislabelled violent XD clips): the middle 40 s. Videos shorter than the window use the raw file.
`-c copy` cuts at keyframes, so the actual start can differ from the requested one (inferred).

---

## 7. Time axis and the alignment contract (important)

* `T` for a video = `len(processed/graphs/<vid>.npy)` = number of stride-4 frames of the
  **windowed** clip that contain at least one person. It has no fixed relation to seconds.
* All newer label builders (`build_multihorizon_labels.py`, `build_labels_temporal_only.py`) and the
  VLM extractor map step `t` to a raw-video frame with
  `raw = round((t + 0.5) / T * total_frames)` over the **full** raw video (not the window).
  So `labels_multih`, `labels_preonly_*`, `labels_future_h*_inc`, `labels_temporal_only`, `vlm_emb`,
  `vlm_labels`, `audio_features` share one consistent time axis; the legacy `graphs`, `labels`,
  `labels_future_h{5,10,30}` use the windowed-clip axis and are **not** aligned with them.
* Seconds per label step: median 0.54 s (range 0.04 s to 343 s). Seconds between consecutive
  `vlm_emb` samples: median 2.67 s, 75th pct 6.4 s, 90th pct 18.4 s, max 756 s.
* Contract: embedding row `i` of `vlm_emb/<src>/<vid>.npz` corresponds to label row
  `labels_multih/<vid>.npy[t[i]]` (and `audio_features` row `t[i]`). Correct loader:

```python
z = np.load(f"{D}/processed/vlm_emb/{src}/{vid}.npz")
y = np.load(f"{D}/processed/labels_multih/{vid}.npy")      # [T, 7]
X, Y = z["emb"].astype(np.float32), y[z["t"]]               # Y[:, 4] is the 30 s horizon
keep = Y[:, 4] >= 0                                          # drop "inside violence" (-1)
```

Measured impact of the common `labels[:min_len]` bug (labels collapsed with `max(axis=1) > 0`):
XD 47,310 embedding steps -> 15% get a different label; positive steps 10,280 (wrong) vs 5,880
(correct). UCF 30,924 steps -> 5.4% differ; positives 2,944 vs 1,620.

With correct alignment, 30 s horizon (`Y[:,4]`) on `vlm_emb` steps: XD 5,144 positive / 32,869
negative / 9,297 ignored steps, 406 videos with >=1 positive step; UCF 1,095 / 29,214 / 615, 121
videos. Steps 30-120 s before an onset (the hard "far pre-onset" region): XD 736 steps in 84 videos,
UCF 525 steps in 57 videos.

---

## 8. Label sets (`processed/labels*`)

| Dir | Files | Values / shape | Semantics | Producer (and inferred command) |
|---|---|---|---|---|
| `labels/` | 1798 | `(T,)` float in [0,1] | risk ramp 0->1 over `[event-30 s, event]` then plateau 1.0 (positives); all 0 (negatives) | `preprocess_anticipation.py` -> `generate_risk_curve` (older version) |
| `labels_future_h30/`, `_h10/`, `_h5/` | 1721 each | `(T,)` {0,1} | `y_bin = labels >= 0.5`, then `y[t] = max(y_bin[t : t+H_steps])` | `scripts/build_anticipation_labels.py --horizon_sec 30` (default out dir); h5/h10 by changing args (inferred) |
| `labels_future/` | 0 | - | empty | - |
| `labels_multih/` | 1721 + `labels_info.json` | `(T, 7)` in {-1,0,1}; horizons [2,5,10,20,30,60,120] s | `1` if next segment start is within h seconds, `-1` inside a segment, else 0 | `scripts/build_multihorizon_labels.py --graphs_dir data/processed/graphs --out_dir data/processed/labels_multih --ignore_inside_segments` (confirmed by `labels_info.json`) |
| `labels_preonly_h{1,5,10,30}_ignore/` | 610 each | `(T,)` {-1,0,1} | 1 in `[start-H, start)`, -1 inside segment | `scripts/build_labels_temporal_only.py --task anticipation --horizon_sec H --ignore_during_event` (inferred) |
| `labels_future_h{1,5,10,30}_inc/` | 610 each | `(T,)` {0,1} | 1 in `[start-H, end]` (anticipation + early detection) | same script with `--include_during_event` (inferred) |
| `labels_temporal_only/` | 901 | `(T,)` {0,1} | 1 inside a segment (detection) | same script `--task detection` (inferred); includes 291 all-zero XD non-violent clips, which the current "Option A" code would skip, so it came from an earlier version (inferred) |

Per-source composition (videos with at least one positive):

* `labels_multih`: UCF 949 (124 with positives, 823 all-zero, 2 with only ignore); XD 772 (413 with
  positives, 292 all-zero non-violent, 67 violent clips with no pre-onset step because onset ~0).
* `labels_preonly_*` / `_inc`: only the 610 annotated videos (129 UCF + 481 XD). **No negative
  videos at all** - a model trained on these only learns within-violent-video localization. The
  `results/anticipation_preonly_h30_grace1_ignore*` runs on them got test ROC-AUC ~0.63 and F1 equal
  to the always-positive baseline.
* `labels_future_h*`: positives only for the 344 metadata-positive videos (129 UCF + 215 XD); the
  266 mislabelled violent XD clips are all-zero. For 259 of 344 positive videos `labels_future_h30`
  is all ones (the whole 40 s window is within 30 s + ramp of the event), so it is effectively a
  video-level label.

### 8.1 Note on `labels/` and `generate_labels.py`

The current `scripts/generate_labels.generate_risk_curve(meta, num_frames=None, anticipation_window_sec=30, post_event_window_sec=10)`
ramps 0->1 over 30 s before the event and decays 1->0 over 10 s after it, in **raw frames**.
`preprocess_anticipation.py` calls it as `generate_risk_curve(local_meta, graph_seq)`, passing a
list as `num_frames` -> `int(list)` raises `TypeError`, so the current code cannot regenerate
`labels/`. The stored files plateau at 1.0 after the event (no decay) and the ramp covers ~75% of `T`
(30/(30+10) of the window), so they came from an older version that worked in graph steps
(inferred). 22 positive videos have empty (`T=0`) labels.

---

## 9. Splits

### 9.1 Root index files (used by the main VLM trainer and sim)

`train_index.json` / `val_index.json` / `test_index.json` / `eval_index.json`: JSON lists of
`{"graph_path": "data/processed/graphs/<vid>.npy", "label_path": "data/processed/labels/<vid>.npy"}`.
Consumers extract `vid` from the basename and look up their own embeddings/labels
(`training/train_temporal_model.py --train_index/--val_index/--test_index`, `sim/live_simulator.py`,
`sim/build_long_simulation.py`, `scripts/build_llm_dataset.py`, `training/evaluate.py`).

Produced by `tools/make_nonoverlap_splits.py` (inferred from the 80/10/10 ratio, the
`eval_index.json` alias, and file dates): shuffle unique clip ids (seed 1337), split 80/10/10.
1438 / 180 / 180 (eval == test). No clip id appears in two splits, but grouping is by clip name, not
by source movie/video:

* 36 source groups shared train/val -> 46 val clips share a source with train.
* 32 shared train/test -> 44 test clips share a source with train.
* 7 shared val/test -> 11 test clips share a source with val.

Composition (onset counts from the official annotations):

| Split | Clips | XD (violent / non-violent / onset>=30 s) | UCF (annotated / onset>=30 s / unannotated anomaly / normal) | metadata label=1 |
|---|---|---|---|---|
| train | 1438 | 645 (405 / 240 / 47) | 793 (111 / 45 / 644 / 38) | 305 |
| val | 180 | 70 (46 / 24 / 3) | 110 (13 / 5 / 91 / 6) | 31 |
| test | 180 | 83 (49 / 34 / 3) | 97 (16 / 9 / 75 / 6) | 30 |

So almost all >= 30 s examples are in train; val/test can not measure 30 s anticipation reliably.
`train_temporal_model.py` defaults to `--source_filter xd_violence`, leaving 83 test clips of which
3 have onset >= 30 s.

### 9.2 Other split mechanisms

* `scripts/split_train_val.py` - older 80/20 random split that overwrites `train_index.json`
  in place (seed 42). Superseded.
* `tools/check_split_overlap.py` - checks clip-id overlap of train/val/eval (exact names only;
  passes, but does not detect source-level leakage).
* `baselines/train_student_original.py` builds its own split with `--split_group source_movie`
  (default) - groups by `source/<part before __#>`, which **does** prevent movie-level leakage.
  It also indexes labels by the npz `t` array (correct alignment). `results/anticipation_preonly_*`
  (447/56/56 videos) appear to come from it (inferred from the metrics.json fields).

---

## 10. Script reference

### setup/

| File | Purpose | Inputs -> Outputs | Run | Issues |
|---|---|---|---|---|
| `build_ucf_crime_metadata.py` | UCF metadata | `raw_videos/ucf_crime/videos/*.mp4` + temporal annotation txt -> `metadata/ucf_crime.json` | `python setup/build_ucf_crime_metadata.py` | label=1 only for annotated test videos; uses first segment start only |
| `build_xdviolence_metadata.py` | XD metadata | `raw_videos/xd_violence/{videos,annotations.txt}` -> `metadata/xd_violence.json` | same | **bug**: subtracts `HH-MM-SS` movie offset from clip-relative frames; does not strip `.mp4` from annotation ids (274 violent clips lost) |
| `kinetics_download_relevant.py` | download K700 violent classes with yt-dlp (optional cookies file) | `kinetics/{train,val}.csv` -> `kinetics/videos/`, `metadata/kinetics.json` | needs network | most class names do not exist in K700 |
| `kinetics_list_labels.py` | list K700 classes | `train.csv` -> `metadata/kinetics_labels.txt` | - | - |
| `enrich_kinetics_metadata.py` | add fps/frames, synthetic midpoint event, label=1 | `metadata/kinetics.json` (in place) | - | destroys class names |
| `preprocess_kinetics_actions.py` | YOLO+VideoMAE per-person pooled clip features | -> `processed/kinetics_actions/{features,labels}.npy, info.json` | `python -m setup.preprocess_kinetics_actions` | outputs not on disk (never completed or overwritten); labels would all be `1` |
| `videomae_finetune_kinetics.py` | fine-tune VideoMAE on Kinetics classes | -> `models/videomae_violence_finetuned.pt` | - | after enrichment `sorted({1,'slapping','wrestling'})` raises `TypeError`; output model does not exist |

### Repo root

* `setup/curate_youtube.py` - YouTube search/download/analyze/filter/export pipeline for
  escalation videos. Subcommands `search` (`--api_key` or `YOUTUBE_API_KEY`), `download`, `analyze`
  (slow Qwen text generation, `prompt_mode=anticipation_simple`), `filter --min_anticipation 10`
  (writes `filtered_candidates.json`), `export` (writes `annotations.json/csv`), `status`.
  `--output_dir` is a top-level flag (before the subcommand) default `data/anticipation_curation`;
  the docstring's `export --output_dir` usage is wrong. Only `search`/`download` results are on
  disk; `filtered_candidates.json` and an exported dataset do not exist. Analysis was done by
  `experiments/fast_analyze_curation.py` (Qwen embedding + LSTM `outputs/outputs_temporal/best_temporal_model.pt`,
  thresholds violence 0.75 / tension 0.3).

### scripts/ (data/label related)

| File | Purpose | Notes |
|---|---|---|
| `preprocess_anticipation.py` | legacy pipeline: window clips (`windowed/`), YOLO tracks, VideoMAE actions, CLIP scene, zero audio, graphs, risk labels for XD+UCF | run `python -m scripts.preprocess_anticipation` from repo root; skips existing outputs; uses buggy XD metadata; current `generate_risk_curve` call crashes |
| `generate_labels.py` | `generate_risk_curve()` | see 8.1 |
| `build_anticipation_labels.py` | `labels/` -> `labels_future_h{H}` via threshold 0.5 + future-OR | `--labels_dir --meta_dir --out_dir --horizon_sec --gt_thresh`; inherits windowed axis and XD bug |
| `build_multihorizon_labels.py` | GT annotations -> `labels_multih` `[T,K]` | `--graphs_dir` (only for `T`), `--horizons_sec 2,5,10,20,30,60,120`, `--ignore_inside_segments`, `--xd_units auto` (auto picks "frames" for all 500 XD clips - verified), supports `--rank/--world_size` sharding |
| `build_labels_temporal_only.py` | GT annotations -> detection or anticipation `(T,)` labels, only for annotated videos ("Option A") | `--task detection|anticipation --horizon_sec --include_during_event --ignore_during_event --overwrite`; kinetics fallback uses the synthetic midpoint |
| `split_train_val.py` | old 80/20 split | overwrites in place |
| `preprocess_kinetics_for_videomae.py` | 768-d VideoMAE features per Kinetics clip | loads model at import time |
| `check_audio_availability.py` | ffprobe + librosa audio report | `--videos_dir ... --output report.json`; no report was saved in the repo |
| `analyze_youtube.py` | end-to-end inference on a URL/local file: frames at `--fps 1.0` -> QwenTeacher `violence_embedding` -> LSTM -> plot/alerts | training embeddings were sampled at irregular per-video rates (section 7) while this samples at fixed 1 fps (train/inference mismatch); pads the first window with zeros (fast_analyze_curation pads with copies); if `vlm.qwen_teacher` fails to import it falls back (with a printed warning) to Qwen2.5-VL-3B and a different embedding recipe that does not match training. `probe_clip_01_{embeddings,scores}.npy` (197 x 3584, scores 0.51-0.77) at repo root are outputs of it (inferred) |
| `quick_inspect.py` | plot GT vs predicted risk for the legacy `ViolenceTransformer` | hard-coded absolute data path |
| `vlm_label_dataset.py` (not in original scope, read for alignment) | produces `vlm_emb*` / `vlm_labels*`; `T` from `graphs`, sample every `--every_t 5`, frame = center-of-bin over the full raw video | multi-GPU via file locks in `.locks/` |

### tools/

| File | Purpose |
|---|---|
| `make_nonoverlap_splits.py` | union index files, group by clip basename (optional `--strip_regex`), shuffle (seed 1337), 80/10/10, writes train/val/test + `eval_index.json` alias. Use `--strip_regex '__#.*$'` to group XD clips by source movie |
| `check_split_overlap.py` | exact-id overlap check (train/val/eval), run from repo root |
| `debug_embeddings.py` | prints npz keys/shapes in an embedding dir |
| `debug_sources.py` | counts sources in `train_index.json` by name pattern |
| `diagnose_gap.py` | per-split stats for an embeddings dir + labels dir; same `[:min_len]` misalignment as the trainers |

---

## 11. Known issues (data side), most severe first

1. **XD metadata onset bug** (`setup/build_xdviolence_metadata.py` lines 56-86): 260 clips with a
   `#HH-MM-SS` timestamp lose all annotated frames after `f - clip_start_frame`; 14 annotation ids
   end with `.mp4` and never match. Result: 274/500 violent clips labelled 0, 11 with a wrong onset.
   Affects `metadata/xd_violence.json`, `windowed/xd_violence`, `processed/{tracks,actions,scenes,graphs,labels}`
   window choice, `labels_future_h{5,10,30}`, and the legacy graph model. `labels_multih` and the
   `build_labels_temporal_only` sets are NOT affected (they read `annotations.txt` directly).
2. **`labels[:min_len]` misalignment** in most VLM-pipeline consumers (list in TL;DR). Index labels
   by the npz `t` array.
3. **Horizon collapse and ignore-to-negative**: `labels.max(axis=1) > 0` trains the 120 s horizon
   and turns in-violence (-1) steps into negatives. Use one column (index 4 = 30 s) and mask -1.
4. **UCF unannotated anomalies as negatives** (810 videos, 357 in violent categories).
5. **Source-level split leakage** in the root index files; tiny >= 30 s val/test sets.
6. **Irregular, sometimes extremely coarse temporal sampling** (`T` from person-detection graphs);
   time between VLM samples up to 756 s. Several videos have fewer than a handful of embeddings.
7. 77 videos missing from all VLM features (empty graphs), including 22 positives.
8. Curated YouTube labels are model-generated with a 60 s artifact.
9. Legacy windowing: positives keep only 30 s before / 10 s after the first onset, negatives the
   middle 40 s; the legacy graph features never see more than 30 s of pre-onset context.
10. Kinetics metadata corrupted (class names lost, synthetic event); Kinetics useless for anticipation.
11. `processed/audio` is all zeros; UCF audio is silent; only XD (and curated YouTube) have real audio.
12. `fused_finetuned_flow` contains only flow features (inferred broken).
13. Dead/unreproducible code: `setup/preprocess_kinetics_actions.py` outputs and
    `setup/videomae_finetune_kinetics.py` model never produced; `preprocess_anticipation.py` crashes
    on the current `generate_risk_curve` signature; `labels_future/` empty; `good_video_indices.npy`
    unreferenced.

---

## 12. Recommendations for a new model (data side)

1. Build ground truth directly from `annotations.txt` and the UCF temporal file (section 3 snippet);
   regenerate metadata with a fixed XD script before reusing anything metadata-based.
2. Re-extract features on a **fixed time grid over the full raw video** (e.g. 1-2 fps clips with
   per-clip video features + audio windows) instead of reusing the graph-derived `T`. Store a
   `timestamps` array in every feature file and build labels from seconds.
3. Split by **source group** (XD part before `__#`; UCF video id), and stratify so val/test each
   contain a meaningful number of onset >= 30 s videos (there are only 112 in total; consider
   cross-validation over groups instead of a single small test set).
4. Negatives: XD `_label_A`, UCF normals including the unused 950 `Training-Normal-Videos` /
   `Testing_Normal_Videos`, and pre-onset steps far from the onset. Exclude unannotated UCF anomaly
   videos (or obtain annotations for them) rather than calling them negative.
5. The 92 curated YouTube videos are the best domain match (bodycam/road rage/CCTV, all with audio,
   4-10 min long, 10 h total). Manually annotating the first physical-violence time in each is likely
   the highest-value data work available; the existing pseudo-labels should be discarded.
6. External sources worth checking (not on disk, not verified here): UBI-Fights (long surveillance
   videos with frame-level fight annotations), CCTV-Fights (real-world fight videos with temporal
   annotations), the XD-Violence training split (video-level labels only), and UCA, the UCF-Crime
   dense timestamped-caption annotations, which could provide event times for the 810 unannotated
   UCF anomaly videos.

---

## 13. Reproducing the counts in this document

All numbers above came from short read-only scripts run with
`.venv/bin/python` against the files listed.
Key recipes:

* Onset table: section 3 loaders, then count `segments[0][0] >= threshold`.
* Split composition: read `*_index.json`, `vid = basename(graph_path)`, group = `vid.split("__#")[0]`.
* Misalignment impact: compare `(y.max(1) > 0)[:len(t)]` vs `(y.max(1) > 0)[t]` per video.
* Audio presence: `ffprobe -v error -select_streams a -show_entries stream=codec_name -of csv=p=0 <file>`;
  loudness: `ffmpeg -t 30 -i <file> -af volumedetect -vn -f null -`.

## Notes moved from code comments

* XD segment units with `--xd_units auto` (`build_labels_temporal_only.py::segments_to_frame_pairs`,
  `build_multihorizon_labels.py::xd_segments_to_seconds`): if the largest segment end is <= 1.0 the
  values are treated as fractions of the clip; else if `max_end * fps <= 1.25 * total_frames` they are
  seconds; otherwise raw frame indices. Non-positive fps falls back to 25.
* `build_labels_temporal_only.py --ignore_during_event` applies only to `--task anticipation` on UCF/XD
  and overrides `--include_during_event`: in-event steps get `--ignore_value` (-1) even if they were positive.
