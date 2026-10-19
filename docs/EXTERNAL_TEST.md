# External test set: NTU CCTV-Fights

Written 2026-10-15. Source, licence and ingest details: docs/DATA_ACQUISITION.md ("NTU CCTV-Fights (received 2026-10-15)").
Code: `setup/ingest_cctv_fights.py`, `anticipation/data/external_cctv.py`. Artifacts: `benchmark/external/cctv_fights/`.

## 1. What it is and the rule

NTU CCTV-Fights (Perez, Kot, Rocha, ICASSP 2019): 1,000 real fight videos (17.7 h) taken from YouTube, with frame-level fight
segments. 280 are CCTV footage, 707 mobile-phone, 9 "Other" and 4 car (dashcam) videos; we call the first `cctv` and the rest
`non_cctv` (`domain`; the original label is `gt_source`). Every video contains at least one fight; there are no clean negatives.

**It is an external test set.** It is never used for training, fold construction, threshold selection, fusion weights, the
modality screen or any model selection. It is kept in its own files and is not part of `benchmark/videos.csv`,
`events.json` or `folds_*.json` (their hashes are pinned by the screen's pre-registration). The groundtruth's own
training / validation / testing subsets are kept in the `official_split` column for reference only; all 1,000 videos are
used as test data because no model has seen any of them.

## 2. Files (`benchmark/external/cctv_fights/`)

| File | Content |
|---|---|
| `videos.csv` | Main catalog columns (`video_id = cctv_fights/<stem>`, `source = cctv_fights`, `onset_source = official`, `role = violent`, `class = fighting`, `eligible30/60`, `n_eligible30/60`, `has_audio`, `domain`, ...) plus `official_split`, `gt_source`, `gt_duration_sec`, `decode_ok`, `label_reliable`, `overlap_main`, `overlap_with` |
| `events.json` | `{"meta", "videos": {video_id: {segments, events, ...}}}`; segments in seconds, sorted, merged, clipped to the decoded duration; `events` from `labels.event_table` (10 s aftermath, `EARLIEST_ALARM_TICK = 2`, no blockers) |
| `summary.json` | Counts per subset (all, label_reliable, evaluation_set, cctv / non_cctv, official splits) |
| `overlap_with_main.json` | Near-duplicate pairs against the main catalog and within the set |
| `transcoded.json` | Per-file record of the 242 webm re-encodes |
| `extract_manifest.csv`, `extract_manifest_audio.csv` | Extraction manifests (all videos; videos with real audio) in the runner's format |

`evaluation_set` = `label_reliable` and not `overlap_main`. Use it as the default population; report the full set only as a sensitivity.

## 3. Counts

Eligibility follows docs/VA30.md: an onset is eligible for lead L when its lead-in is at least L and `onset - max(2, ceil(clear_start)) >= L`.
"First" is the first fight of a video (clear lead-in from the start of the clip); "later" are second and further fights, whose
lead-in starts 10 s after the previous fight ended.

| Subset | Videos | Hours | Fights | Elig30 first | Elig30 later | Elig60 first | Elig60 later | Real audio (videos / h) |
|---|---|---|---|---|---|---|---|---|
| All | 1000 | 17.69 | 2412 | 95 | 53 | 23 | 19 | 868 / 13.61 |
| CCTV | 280 | 8.56 | 747 | 53 | 40 | 14 | 14 | 156 / 4.58 |
| non-CCTV | 720 | 9.13 | 1665 | 42 | 13 | 9 | 5 | 712 / 9.03 |
| Evaluation set (no overlap, label reliable) | 985 | 17.20 | 2385 | 93 | 52 | 23 | 18 | 858 / 13.29 |
| Evaluation set, CCTV | 268 | 8.14 | 724 | 51 | 39 | 14 | 13 | 149 / 4.33 |
| Evaluation set, non-CCTV | 717 | 9.06 | 1661 | 42 | 13 | 9 | 5 | 709 / 8.96 |

(2,412 merged fight instances against the paper's 2,414: overlapping annotations are merged.) 122 videos (119 in the
evaluation set) have at least one eligible30 event, 39 (38) have an eligible60 event. Median video length 44 s (3 s to 704 s);
median first onset 4.7 s, only 109 videos have a first fight at 30 s or later. The eligible30 events are therefore few:
**145 in the evaluation set (93 + 52)**, which is fewer than VA30's own 254 late events and gives wide confidence intervals;
report bootstrap intervals over videos and never per-domain precision beyond what these counts support (CCTV 90 events,
non-CCTV 55). Eligible60: 41.

## 4. Data quality

- **webm clock problem.** Decord opened all files, but 226 of 242 webm gave wrong durations (variable frame rate, container average
  rate 1000 fps). All webm were re-encoded to constant-frame-rate mp4 (crf 16, timestamps preserved); the originals are in
  `raw_videos/cctv_fights/videos_original/`. After that, decord duration equals the groundtruth duration (max(1 s, 2 %)) for
  997 of 1,000 videos.
- **Three label-unreliable videos** (`label_reliable = False`): three videos; the annotations do not
  belong to the file we have (durations 351.5 vs 344.2 s, 121.5 vs 108.0 s, 43.7 vs 80.4 s). Excluded from the evaluation set.
- **Audio.** 868 videos have a non-silent audio stream (`io.is_digital_silence`); the 132 without it are mostly CCTV. Audio is a YouTube re-encode of unknown content (speech, crowd noise, music).
- **Overlap.** See section 5.

## 5. Overlap with the main catalog

`overlap_with_main.json` comes from the VA30 near-duplicate fingerprints (`anticipation/data/near_dups.py`, read-only; the
committed `benchmark/near_duplicates.json` was not touched; 1,000 new fingerprints were cached next to the existing ones).
Matching ran over the 1,000 CCTV-Fights videos together with the 7,696 catalog videos.

Result: **12 CCTV-Fights videos overlap with the main catalog** (15 pairs, against 15 catalog videos: 7 UCF-Crime,
7 XD-Violence YouTube clips and 1 UBI-Fights video). By kind: 7 `identical` re-encodes, 3 `similar` (same scene with a colour shift)
and 5 time-shifted `overlap`. They are flagged `overlap_main = True` (`overlap_with` lists the catalog videos) and excluded from `evaluation_set`.
Five pairs are inside CCTV-Fights itself (two identical, three overlapping): both members are kept (they are the same
YouTube source cut twice); treat them as one unit in the bootstrap (`split_group` is per video, so resample by `overlap` clusters if
precision matters). The detector only finds overlaps that start in the first minute and have the same framing; the true overlap
may be larger. Because the main models never saw CCTV-Fights, overlap matters only through catalog videos used in training.

## 6. How it will be used

1. **Frozen model.** The final fused model, trained on VA30 folds only, is applied unchanged. Alarm thresholds, smoothing and fusion
   weights are the ones chosen on VA30 validation data (the operating points at the target FA/h); nothing is tuned on CCTV-Fights.
2. **Features.** Same causal 1 Hz extractors as the screen (section 7), computed on the full videos (no cut).
3. **Event metrics** (VA30 section 6): Success@30 and Success@10 on eligible30 events (and Success@60 on eligible60) at the transferred
   operating points; lead-time distribution (lead of the first alarm that starts in the acceptance window); bootstrap CIs over videos.
4. **Pre-onset band AUROC** (tick level, unmasked pre-violence ticks only): positives = ticks with `0 < tte <= 30` (and the 10 to 30 s
   band separately), negatives (a) far-from-onset ticks of the same videos (`tte > 120` or no onset ahead, unmasked) and (b) VA30 clean
   negatives, with the VA30 evaluation code and the same censoring rules. (a) controls for domain, (b) matches the main benchmark but
   mixes domains; report both, and CCTV vs non-CCTV separately.
5. **False alarms.** There are no clean-negative videos, so FA/h cannot be measured the VA30 way. Report alarms per hour in the
   far-from-onset unmasked ticks of the same videos as an upper-bound-style proxy (those ticks are close to fights in the same clip), and
   the VA30 clean-negative FA/h as the operating-point definition.
6. **Reporting.** One report with the evaluation set as headline, the full set and the CCTV / non-CCTV split as breakdowns, and the
   position null (onsets are early, median 4.7 s): the eligible events are late-onset ones, so quote the late-event numbers only.

## 7. Extraction

Manifest `benchmark/external/cctv_fights/extract_manifest.csv` (1,000 videos, 17.69 h; audio lanes read
`extract_manifest_audio.csv`, 868 videos, 13.61 h). Run by `data/logs/extract_cctv/run.sh` (cuda:0, at most 4 processes, `nice -n 10`,
`taskset -c 0-11`, resumable: re-run the script). Logs: `data/logs/extract_cctv/` (`status.log`, `<extractor>_shard<i>.log`); runner
records in `data/logs/extract/<extractor>/`. Outputs: `features/<stream>/cctv_fights/<stem>.npz`.

Status (finished 2026-10-15 19:15, wall 3 h 38 min, cuda:0 only, 4 processes at a time; speeds 6 to 7x realtime for visual and pose,
11x for speech_tox, 12x for motion):

| Stream (output) | Videos | Failures |
|---|---|---|
| vjepa2_l, siglip2, siglip2_prompts (`visual`) | 1000 | 0 |
| siglip2_prompts_v2 (CPU rescore) | 1000 | 0 |
| pose_raw, pose_interact (`pose`) | 1000 | 1, retried (below) |
| pose_state, pose_state_pairs | 1000 | 0 (after the pose retry) |
| motion | 1000 | 0 |
| siglip2_scene0p | 1000 | 0 (after the pose retry) |
| ast_events, ast_emb, clap_prompts | 868 | 0 |
| voice_affect, e2v_emb | 868 | 0 |
| audio_dsp | 868 | 0 |
| speech_tox | 868 | 0 |

The one failure: `pose` on one of the three label-unreliable videos raised a decord EOF error ("takes too long to retrieve
last few frames", `DECORD_EOF_RETRY_MAX=10240`); the other lanes decoded it. It was re-run with `DECORD_EOF_RETRY_MAX=1000000` and
succeeded; pose_state and scene0p were then completed for it. The 132 videos without real audio have no audio-lane files by design (a
missing audio file means "no real audio", as in the main catalog). The number of ticks equals floor(decoded duration) in every
checked stream for all videos (`siglip2`, `vjepa2_l`, `siglip2_prompts(_v2)`, `pose_interact`, `pose_state`, `motion`, and the audio lanes
on the 868).

## 8. Caveats

- Small: 145 eligible30 events (41 eligible60); very wide intervals. Do not cut by many strata.
- Every video is violent, so there is no negative-video population: a model that fires on "any fight-like content early in a clip"
  looks good at event level. Compare against the position null and the far-from-onset tick negatives.
- The median first onset is 4.7 s, so the eligible events are the late-onset minority and probably differ from typical fights (not checked in detail).
- YouTube originals of unknown provenance: letterboxing, watermarks, mixed frame rates (before transcoding), compilation cuts. A
  cut inside a clip resets the scene; motion `cut_1s` handles it as elsewhere.
- Annotation convention (frame-level "Fight") may place onsets earlier or later than HIVAU / UBI onsets used in training; there is no
  reconciliation, and annotators' boundaries are not audited here (compare with the onset label audit, docs/LABEL_AUDIT.md).
- Licence: academic, non-commercial, no redistribution. Only the derived annotation tables are in git; videos, transcodes and features
  are not. Acknowledgement and citation text in docs/DATA_ACQUISITION.md section 5.
