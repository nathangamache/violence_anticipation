# Experiment history: what was tried, what it scored, and why it failed

Audience: future agents and developers deciding what to build next. This file reconstructs the project's
history from file dates, result folders and code (audit date 2026-10-14). Each approach lists its result
folder, the reported number, the number that survives scrutiny, and the reason it failed or plateaued.
Details and evidence live in the per-area docs in [codebase/](codebase/); this file links to them.

Conventions:
- "Reported" = what the folder's metrics file says. "Honest" = after removing leakage, misalignment or
  test-tuned thresholds, as measured or recomputed by the audit. Where no honest number exists, it says so.
- The recurring defects are named once in section 0 and referenced by code (D1 to D8) below.
- Dates are file modification dates (inferred to be close to when each run happened).

---

## 0. Recurring defects (referenced throughout)

| Code | Defect | Where it bites | Doc |
|---|---|---|---|
| D1 | **Label/embedding misalignment.** `vlm_emb` row i was sampled at label row `t[i] = 5i`, but most loaders use `labels[:len(emb)]`. On XD, 15 % of steps get the wrong label and positives nearly double (5,880 to 10,280). | `train_temporal_model.py`, `train_student.py`, contrastive scripts, all sim builders, fusion builders, experiment loaders | [data_and_labels.md](codebase/data_and_labels.md) section 7 |
| D2 | **Horizon collapse and violence-as-negative.** `labels_multih.max(axis=1) > 0` means "onset within 120 s", and the `-1` in-violence rows become negatives. | same scripts as D1, LoRA | [simulation_and_evaluation.md](codebase/simulation_and_evaluation.md) section 0 |
| D3 | **No real time base.** "1 step = 1 s" is assumed everywhere, but a VLM step is median about 3 s (0.2 s to 756 s). "30 s horizon", "lead time" and "FA/hour" are in steps. Audio and pose were extracted on yet another wrong clock. | every trainer except `train_student_original.py`; every simulator | [audio_pose_flow_fusion.md](codebase/audio_pose_flow_fusion.md) section 1 |
| D4 | **Split leakage.** The root `train/val/test_index.json` split is by clip, so 44 of 83 XD test clips share a source movie or YouTube video with training. Older student runs used a random per-clip split. Most `simulation_data/*.npz` files are 81 % training videos. | `outputs/*`, `results/anticipation_*` (except prboost_v3), `final/*` | [results_inventory.md](codebase/results_inventory.md) C3 |
| D5 | **Test-tuned thresholds and oracle metrics.** Best-F1, FA-target and smoothing grids chosen on the reported set. | `final/*`, `outputs/*/final_metrics.json`, `experiments/output/*`, late fusion | [results_inventory.md](codebase/results_inventory.md) C4 |
| D6 | **Lenient event metrics.** Detection credits alarms during violence; lead is anchored at the anticipation-window start and capped at 30; an always-on alarm gets 100 % detection at about 1 FA. | `sim/comprehensive_eval.py`, simulators, late fusion | [simulation_and_evaluation.md](codebase/simulation_and_evaluation.md) section 3 |
| D7 | **Label noise.** 810 unannotated UCF anomaly videos (357 in violent categories) are treated as all-negative. The XD metadata builder drops 274 of 500 violent clips to label 0 (affects the gen-1 pipeline and `labels_future_h*`). | everything trained on UCF or on `metadata/xd_violence.json` | [data_and_labels.md](codebase/data_and_labels.md) section 11 |
| D8 | **Not enough pre-onset footage.** Only 112 of 640 onset-annotated videos have 30 s or more before onset (49 interpersonal; XD median onset 5.9 s; 88 XD clips start at frame 0). Test splits hold 10 to 12 such videos. | every anticipation claim | [data_and_labels.md](codebase/data_and_labels.md) section 3 |

---

## 1. Timeline at a glance

| # | Dates | Approach | Best reported | Honest | Verdict |
|---|---|---|---|---|---|
| 1 | Nov 2025 | Data setup: metadata, windowing, Kinetics subset | n/a | n/a | XD metadata bug (D7); Kinetics useless |
| 2 | Nov to Dec 2025 | Gen-1 graph features + `ViolenceTransformer` | none saved (re-eval: frame AUC 0.975) | causal clip AUC 0.66 on probably-seen data | Leaky, non-causal, no audio |
| 3 | Dec 2025 | Text-LLM head on graph summaries | video AUC 0.588 | same; below majority accuracy | Failed |
| 4 | Jan to Feb 2026 | Zero-shot Qwen2.5-VL-7B teacher | AUC 0.682 (XD 0.776) | same (no training) | Detector, not anticipator |
| 5 | Feb 2 to 10 | Single-horizon student MLP on pre-onset-only labels | test AUC up to 0.73 | val AUC 0.48 to 0.55 | Chance |
| 6 | Feb 12 to 17 | Multi-horizon student MLP, random split | test AUC 0.90 to 0.94 | 0.56 to 0.62 after grouping | Movie leakage (D4) |
| 7 | Feb 17 to 23 | **Multi-horizon student MLP, movie-grouped split** | test AUC 0.58 to 0.62 | **same; current honest baseline** | Plateau |
| 8 | Feb 18 | Focal-loss frame MLP (`train_student.py`) | test AUC 0.925 (XD) | val 0.44 to 0.68 | D1 to D5 |
| 9 | Feb 19 to 24 | BiLSTM temporal head (`train_temporal_model.py`) | test AUC 0.991 | val 0.53 to 0.67 | D1 to D5; immediate overfit |
| 10 | Feb 19 to 24 | Contrastive projectors (SupCon) | test AUC 0.74 to 0.90 | val 0.60 to 0.67 | No gain |
| 11 | Feb 23 | LoRA finetune of Qwen2.5-VL; VQA question features | val acc 0.4925; QA AUC 0.486 | same | Collapsed / chance |
| 12 | Feb 23 to 24 | Optical flow, audio, pose; early and late fusion | sim AUC 0.87 to 0.89; "100 % det, 0.08 FA/h" | val 0.64 to 0.72, misaligned | Clock bugs, leakage |
| 13 | Feb 19 to Mar 23 | "8-hour shift" simulations, smoothing and threshold sweeps | AUC 0.985, 0.0 FA/h, 29 s lead | none valid | D1 to D6 |
| 14 | Feb 26 to Mar 2 | Live RTSP demo and YouTube probe | 1 alert per session | YouTube frame AUC 0.72, max score after the fight | No ground truth; train/serve skew |
| 15 | Mar 16 | `experiments/` suite (EMA, algorithms, calibration, UCF) | LDA AUC 1.0; sliding window 0.844 | per-video AUC from one video | Loader bugs |
| 16 | Mar 22 to Apr 6 | YouTube curation + LSTM pseudo-labels; exp1 to exp4 | n/a | person filter hurt; cross-domain chance | Circular labels |

---

## 2. Phase by phase

### 2.1 Data setup (Nov 2025)

- **What:** `setup/build_ucf_crime_metadata.py`, `setup/build_xdviolence_metadata.py` -> `data/metadata/*.json`;
  Kinetics-700 download (`setup/kinetics_download_relevant.py`); 40 s windowing in
  `scripts/preprocess_anticipation.py` -> `data/windowed/`.
- **Outcome:** UCF metadata marks only the 140 temporally annotated test anomalies as positive (D7). The
  XD builder subtracts the movie timestamp from clip-relative frames and misses 14 `.mp4` ids, so only 215 of
  500 violent XD clips get a correct onset. Only two Kinetics classes (wrestling, slapping) exist under the
  requested names; their onset is a synthetic midpoint, so they carry no anticipation signal.
- **Why it matters:** every gen-1 artifact and `labels_future_h*` inherits these labels. `labels_multih`
  does **not** (it reads `annotations.txt` directly).

### 2.2 Gen-1 graph transformer (Nov 26 to Dec 18, 2025)

- **What:** YOLOv8n person boxes (no tracking) + VideoMAE-base embedding of single-frame person crops +
  one clip-mean CLIP ViT-B/16 vector + a zero audio vector, packed into per-step "graphs";
  `core/model.py::ViolenceTransformer` (mean node projection + scene + audio, 2-layer **bidirectional**
  transformer). Trained twice: `best_checkpoints/` (40 epochs, Dec 15) and `checkpoints/` (60 epochs, Dec 18).
- **Reported:** nothing. `training/train.py` has no validation and saved no metrics.
- **Re-evaluated in the audit (CPU, `test_index.json`):** epoch 60 frame AUC 0.975, clip AUC 0.959. Zeroing
  the CLIP scene vector drops frame AUC to **0.469**; causal prefix-only scoring drops clip AUC to **0.659**;
  a position-only baseline scores 0.613.
- **Why it failed:**
  1. `train.py` rebuilds `train_index.json` from every clip, so the test clips were probably trained on (inferred).
  2. The clip-mean CLIP vector includes post-event frames, so it is a label oracle.
  3. The encoder sees the future (no causal mask) and learns the event's position (event placed at about
     75 % of each positive window, labels are a deterministic ramp).
  4. Audio was all zeros; VideoMAE saw single static crops; adjacency was never used.
  5. Labels and windows inherit the XD metadata bug (D7).
- **Result folders:** `checkpoints/`, `best_checkpoints/` (weights only). Doc: [core_and_training.md](codebase/core_and_training.md).

### 2.3 Text-LLM decision head (Dec 2025)

- **What:** `scripts/build_llm_dataset.py` turned graphs into text ("about K people visible...") and asked a
  Qwen text LLM for a label or a risk score (`llm/llm_head_infer_seq.py`, `llm/llm_head_infer.py`).
- **Reported:** video-level AUC 0.588, accuracy 0.727 against a 0.826 majority baseline
  (`llm_data/val_seq_preds_qwen.jsonl`, n = 344); per-sample head AUC 0.341 (`llm_data/val_preds_qwen.jsonl`).
- **Why it failed:** the summaries contain only person counts; the prompts also leak the label (the
  "annotated event frame" line alone gives AUC 0.942), yet the LLM still did not beat the majority class.
- Doc: [vlm_and_llm.md](codebase/vlm_and_llm.md) section 7.

### 2.4 Zero-shot Qwen2.5-VL-7B teacher (Jan 29 to Feb 20, 2026)

- **What:** `scripts/vlm_label_dataset.py` sampled every 5th graph step of the full raw video, generated a
  JSON verdict per frame and saved a 3584-d image-token embedding. Four extraction passes:
  `vlm_*_old` (Jan 29), `_meanpool` (Feb 9), default context-both (Feb 16), `_violence` (Feb 20). About 78k
  frames per pass.
- **Reported (`results/vlm_eval*`, correct `t` alignment, no training):** frame AUC 0.682 overall,
  **XD 0.776**, UCF 0.497 for current violence; 0.662 overall / 0.772 XD for "violence now or within 30 s".
- **Why it plateaued:**
  - It is a **detector**: AUC falls as the horizon grows (h5 0.677, h10 0.672, h30 0.662).
  - UCF looks like chance largely because UCF ground truth is broken (D7): Fighting videos without temporal
    annotations are "negative" but the VLM scores them 0.38 on average versus 0.000 on real Normal videos.
  - The pooled image-token embedding is scene-dominated (within-video cosine 0.95 to 0.97). The prompt
    follows the image in a causal decoder, so `prompt_mode` cannot change the embedding; the
    "violence-focused prompt" premise was wrong. `vlm_emb_old` is effectively constant (cosine 0.999).
- Doc: [vlm_and_llm.md](codebase/vlm_and_llm.md) sections 2 to 4.

### 2.5 Single-horizon student MLP on pre-onset-only labels (Feb 2 to 10)

- **What:** `training/train_student_original.py` (older version) on `labels_preonly_h30_ignore`
  (positive in the 30 s before onset, ignore during violence).
- **Reported:** `results/anticipation_preonly_h30_grace1_ignore{,_fp1..4,_meanpool}`: val AUC 0.48 to 0.55,
  test AUC 0.51 to 0.73; the selected threshold fires on everything (F1 equals the always-positive baseline).
- **Why it failed:** those label sets contain only the 610 annotated violent videos, so there were no
  negative videos at all; the task reduces to locating the pre-onset window inside short violent clips,
  which mostly start within seconds of the violence (D8). Not reproducible with today's code (1-D labels
  are rejected).

### 2.6 Multi-horizon student MLP with a random split (Feb 12 to 17)

- **What:** `scripts/build_multihorizon_labels.py` built `labels_multih` (Feb 12: `[T,7]`, horizons
  2 to 120 s, `-1` inside violence). `train_student_original.py` trained a `StudentMLP` (LayerNorm -> 256 ->
  256 -> 7) on VLM embeddings plus hashed VLM tags, masking `-1`.
- **Reported:** `results/anticipation_multih_hset{,_v2,_v3,_v4*}`, `results/anticipation_prboost_v1|v2`:
  test AUC 0.83 to 0.94, XD test AUC up to 0.967.
- **Why it is invalid:** the split was random per clip; in `prboost_v1`, 46 of 77 test XD clips share a
  movie or YouTube source with training (D4). `hset_v3` picked threshold 1.0 (zero alerts).

### 2.7 Multi-horizon student MLP with a movie-grouped split: the honest baseline (Feb 17 to 23)

- **What:** the same script with `--split_group source_movie` (groups 1072/134/134, videos 1356/156/158),
  thresholds chosen on val, video-level alarm statistics in seconds.
- **Result:** `runs/student_fix1`, `runs/student_fix2`, `outputs/student_mlp`, `results/anticipation_prboost_v3`:
  **test AUC 0.56 to 0.62, AP 0.10 to 0.16 (base rate 0.081), XD-only test AUC 0.47 to 0.56**, 30 to 60 s
  bucket AUC about 0.61 (about 0.50 on XD); 50 to 60 % of violent test videos first alarm in
  [onset - 30 s, onset + 1 s] with median lead 7.8 to 8.9 s; 15 to 25 % of normal test videos false-alarm.
- **Why it plateaued:** clip-level "this video looks violent" signal exists (XD video-level max-prob AUC
  0.71 to 0.76), but almost no ability to locate the 30 s pre-onset window on unseen movies. Val and test
  differ by about 0.2 AUC because each split holds only about 32 to 39 violent XD clips and 9 to 10 videos
  with onset at least 30 s. Single-frame features, D7 noise, and D8 scarcity cap it.
- Full numbers: [BASELINE.md](BASELINE.md).

### 2.8 Focal-loss frame MLP, `train_student.py` (Feb 18)

- **Reported:** `runs/outputs_xd` val 0.680 / test 0.925; `runs/outputs` (XD + UCF) val 0.617 / test 0.723;
  `runs/outputs_ucf` val 0.444 / test 0.558.
- **Why it is invalid:** D1, D2, D3, D4, oracle thresholds (D5), and 59 % of XD test samples come from one
  14-minute non-violent clip. The `--ucf_weight` sampler silently drops about 27 % of each epoch. UCF-only is chance.

### 2.9 BiLSTM temporal head, `train_temporal_model.py` (Feb 19 to 24)

- **What:** 2-layer BiLSTM + attention over 30 VLM steps (stride 5), focal loss, XD only. Variants on every
  embedding flavour and fused features.
- **Reported (`outputs/outputs_temporal*/final_metrics.json`):** test AUC 0.76 to 0.991.
- **Honest view:** val AUC 0.53 to 0.67 for VLM-only inputs (0.577 for the default checkpoint, selected at
  epoch 1). Train AUC reaches 0.99 to 1.0 within a few epochs. 78 % of test windows come from one
  non-violent clip and 72 % of val windows from one violent clip; 44 % of training clips are too short to
  produce a single window.
- **Why it failed:** D1 to D5, no seed (identical args gave 0.577 and 0.600), and "30 steps" spans 6 s to hours.
  Despite this, `outputs/outputs_temporal/best_temporal_model.pt` became the default checkpoint of the realtime
  demo, YouTube analysis, curation pseudo-labelling and most evaluators.
- Doc: [temporal_and_student_models.md](codebase/temporal_and_student_models.md) section 2.3.

### 2.10 Contrastive projectors (Feb 19 to 24)

- **What:** SupCon + focal projectors on VLM embeddings (`tuning/contrastive_finetune.py` V1,
  `tuning/contrastive_projection.py` V3, plus V2 and an "attention" variant whose code is gone).
- **Reported:** `outputs/outputs_contrastive` test AUC 0.737; `outputs/contrastive_v3` test 0.903;
  `outputs/outputs_projected` (LSTM on projected features) test 0.945.
- **Honest view:** best val AUC 0.60 to 0.67, no better than raw embeddings. Labels misaligned (D1, D2).

### 2.11 LoRA finetune and VQA features (Feb 23)

- **LoRA** (`tuning/lora_finetune_vlm.py`, `lora_finetuned_v2/`): val accuracy 0.4925 in every epoch, which
  equals the negative rate. The eval loop feeds prompt + answer into `generate`, the labels mean "violence
  within 120 s" with in-violence frames as "no" (D2), and 71 % of XD frames were fetched past the end of the
  video (timestep treated as seconds). Nothing loads the adapters.
- **VQA features** (`outputs/qa_feature_test`): 25 yes/no questions on 50 + 50 frames, logistic AUC 0.486,
  frames misaligned with labels. Never scaled up.

### 2.12 Optical flow, audio, pose and fusion (Feb 23 to 24)

- **What:** Farneback flow (22 stats per frame), 91-d hand-crafted audio, YOLOv8m-pose 540-d vectors; early
  fusion by concatenation (`fused_vlm_flow`, `fused_features`, `vlm_audio_features`); late fusion of LSTM scores.
- **Reported:** fused LSTM val AUC: VLM + flow 0.721 (best of any temporal run), VLM + audio 0.693,
  VLM + pose 0.674, audio-only 0.642. On the 772-video sims: VLM + pose 0.887, VLM + audio 0.868. Late
  fusion: "100 % detection, 0.08 FA/h" (`final/late_fusion_*`).
- **Why it does not hold up:**
  - Audio and pose used graph steps as seconds: 71.6 % of audio rows are zero and 70 % of pose rows repeat
    the last frame. Fusion then paired rows by truncation. Only `fused_vlm_flow` is time-aligned.
  - The 772-video sims are 81 % training videos (D4); the VLM-only and fused sims use different label
    definitions, so the "gains" are not comparable.
  - The flow simulation pasted flow from 11 unrelated videos next to the VLM stream (AUC 0.557).
  - Late fusion: the audio stream was stretched from a different video order; the audio score never drops
    below 0.252, so threshold 0.1 is always on and counts as 1 FA (D6). Recomputed AUC: VLM 0.811,
    audio 0.480, fused 0.743, so fusion hurt.
  - Per-video z-scoring (non-causal) probably explains much of the val "fusion gain".
  - With correct alignment, the best single hand-crafted feature has pooled AUC 0.60 to 0.67 but only
    0.54 to 0.58 within a video: these features separate videos, not pre-onset moments.
- Doc: [audio_pose_flow_fusion.md](codebase/audio_pose_flow_fusion.md).

### 2.13 "8-hour shift" simulations and operating-point sweeps (Feb 19 to Mar 23)

- **What:** per-video embeddings stitched into long streams (`sim/build_long_simulation.py` and builders not
  in the repo), scored by `sim/comprehensive_eval.py`, `sim/*_simulator.py`, `scripts/evaluate_with_smoothing.py`.
- **Reported headlines:** `final/clean_smoothing_sweep` AUC 0.985, 199/200 events, **0.0 FA/h**, 29.2 s lead;
  `final/clean_t02_eval` AUC 0.982; `final/detection_fa_tradeoff.png` "recommended" theta 0.3, w 10:
  91.9 % detection at 4.87 FA/h, 23.9 s lead.
- **Why none is valid:** the clean/lowfa/matched sims are about 81 % training videos with most normal clips
  removed (63 to 70 % positives; lowfa is circularly selected for low false alarms); labels have D1 and D2;
  steps are not seconds (D3); thresholds were swept on the reported stream (D5); lead is capped at 30 and
  an always-on alarm scores perfectly (D6). The least-contaminated set, `xd_violence_8hr_sim.npz` (test-only
  XD, 13 events, padding repeated about 5.7x, 12 of 18 onsets with no in-clip context), gives LSTM AUC
  0.834 and VLM-student AUC 0.853 under flawed labels, with 6 to 43 "FA/h".
- **Ensembles** (`simulation_results_ensemble/*`): 10 to 13 of 13 events at 61 to 87 false alarms.
- **Cross-domain:** XD-trained LSTM on the UCF sim AUC 0.459 (`final/eval_ucf_crime`).
- `continuation/time_to_violence.py` anchors its ETA on 26.9 s, which is the capped lead from the leaky
  `final/high_sensitivity` run; it was never integrated.
- Doc: [simulation_and_evaluation.md](codebase/simulation_and_evaluation.md).

### 2.14 Live RTSP demo and YouTube probe (Feb 26 to Mar 2)

- **What:** `realtime/rtsp_inference.py` (Qwen embedding at 1 fps on cuda:0, BiLSTM on cuda:1) ran twice
  on a MacBook stream; `scripts/analyze_youtube.py` scored YouTube video `probe_clip_01`.
- **Outcome:** real-time capable (0.35 s per VLM embedding, 2 ms per LSTM call). Each live session fired one
  borderline alert (score 0.504 and 0.540); no ground truth exists. On `probe_clip_01` (hand labels: violence
  37 to 45 s), frame AUC is 0.718, every score is at least 0.515, and the global maximum (0.767) is at
  196 s, after the fight.
- **Why it underperforms:** train/serve skew (ReLU instead of GELU, different prompt and pooling, 1 s
  sampling versus about 3 to 6.6 s training steps, window 32 versus 30) on top of a model with val AUC 0.58.
- The root `*_mlp.pt` files (Mar 2) have no producer script and no metrics; they are orphaned.
- Doc: [realtime_and_misc.md](codebase/realtime_and_misc.md).

### 2.15 `experiments/` suite (Mar 16)

- **What:** EMA smoothing, algorithm variations (LR, transformer, sliding window), prompt-engineering
  statistics, XD to UCF transfer, XD investigation, LDA calibration on the YouTube clip.
- **Reported:** `results/algorithm_comparison.csv` sliding window AUC 0.844; `results/prompt_engineering.csv`
  LDA AUC 1.0; `results/ema_results.csv` EMA never helps; `results/ucf_crime_results.csv` all NaN;
  `results/lda_calibration_results.txt` no calibration ranks violence highest on the YouTube clip.
- **Why it misled:** the loaders take the first N files of a sorted `rglob`, which are UCF videos, while
  claiming XD; "per-video AUC" is the AUC of exactly one validation video; LDA AUC 1.0 is in-sample.
- **Useful finding:** `results/xd_investigation` concluded the clips barely contain pre-violence context,
  which the audit confirms with correct labels (D8).

### 2.16 YouTube curation and pseudo-labelling; final experiments (Mar 22 to Apr 6)

- **What:** `curate_anticipation_dataset.py` downloaded 92 escalation videos (road rage, bodycam, CCTV
  confrontations; 10 h, all with audio). `experiments/fast_analyze_curation.py` labelled them with the
  default BiLSTM (`violence_start_second`, stage timelines) and saved `vlm_emb/curated/` (Apr 6).
  `experiments/run_experiments.py` exp1 to exp4 (Apr 4 to 5).
- **Outcome:** the curated labels are model output, not ground truth: every `violence_start` is at least
  59 s because the first 60 s are forced to `ramp_up`, and obvious false positives exist (a de-escalation
  tutorial, a music video). Training a VLM-embedding model on them is circular. exp1: YOLO person filtering
  made AUC worse (0.832 to 0.812) and false alarms worse (8.2 to 10.3 per hour). exp3: cross-dataset LSTMs
  overfit (train 0.99, val 0.49 to 0.61) and transfer at chance (XD to UCF 0.538).
- **Why it matters for the future:** the 92 curated videos are the closest domain match to the target and
  should become a hand-annotated real-world test set.

---

## 3. Why everything plateaued (synthesis)

1. **The data cannot express the task.** XD-Violence and UCF-Crime clips are cut close to the action
   (XD median onset 5.9 s). Only 112 videos (2.35 h of pre-onset footage) have 30 s or more of lead-in, and
   only 49 of those are interpersonal violence. Grouped test splits contain 10 such videos. Any model trained
   here mostly learns "violent-looking clip" or "violence happening now".
2. **The labels and the clock were broken in most pipelines** (D1, D2, D3, D7). Models were trained to
   predict a 120 s window with violence frames as negatives, on a misaligned and irregular step axis, with
   real UCF crimes as negatives.
3. **The evaluation gave false positive feedback** (D4, D5, D6, single-clip-dominated splits). Leaky numbers
   of 0.9 to 0.99 made each new variant look like progress, so effort went into smoothing, thresholds,
   ensembles and simulations instead of fixing data and labels.
4. **The features do not measure leading cues.** Single-frame VLM embeddings are scene-dominated and static;
   hand-crafted audio, pose and flow carry between-video signal only; the gen-1 audio was zeros and its CLIP
   vector leaked the future. Nothing measured approach, face-offs, crowd formation, raised voices or threats.
5. **Model and objective design invited shortcuts.** Detection-style targets, immediate overfitting
   (train AUC 1.0 at epoch 1 to 2), short or ill-defined context windows, non-causal normalization, and a
   train/serve skew in every deployment path. Domain shift (movie and YouTube clips versus CCTV and bodycam)
   was never controlled; cross-domain results are at chance.

---

## 4. Do-not-repeat list

- Do not pair embeddings with `labels[:len(emb)]`; index by `t` or, better, rebuild features on a 1 Hz clock.
- Do not collapse `labels_multih` with `max(axis=1)`; pick one column and mask `-1`.
- Do not split by clip name; group by source movie / YouTube id / camera.
- Do not pick thresholds, smoothing or fusion weights on the reported set.
- Do not stitch "N-hour" streams from short clips and call the result a benchmark.
- Do not count alarms during violence as anticipation, and do not cap lead time.
- Do not treat unannotated UCF anomaly videos or `metadata/xd_violence.json` labels as ground truth.
- Do not train or evaluate VLM-embedding models on the curated YouTube pseudo-labels.
- Do not run `python -m training.train` unmodified (it overwrites `train_index.json`).
- Do not copy model classes into inference scripts; import them (ReLU/GELU drift already happened twice).
