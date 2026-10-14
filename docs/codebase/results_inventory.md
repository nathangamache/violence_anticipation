# Results inventory: every metric the repo has produced

**Audience:** future AI agents and developers who need to know which numbers in this repo can be trusted and what a new model has to beat.
**Scope:** `final/`, `results/`, `simulation_results*/`, `outputs/`, `outputs_fused/`, `runs/`, `experiments/output/`, `logs/`, `snapshots/`, the root `inference_*.log` files, the root `*.pt`/`*.npy` artifacts, `best_checkpoints/` and `checkpoints/`.
**Method:** every metrics JSON/CSV was parsed with the venv Python. The producing scripts were read to learn what each metric means. Shapes, label statistics and split overlap were checked with cheap read-only Python. Anything not verified directly is marked **(inferred)**.
**Date of audit:** 2026-10-14. File timestamps fall between Dec 2025 and Apr 2026.

---

## 0. TL;DR

| | |
|---|---|
| **Strongest reported numbers** | `final/clean_smoothing_sweep` and `final/clean_t02_eval` (Temporal LSTM on the curated "clean" sim): ROC-AUC 0.982–0.985, AP 0.99, 199/200 events detected, **0.0 FA/h**, mean lead 29.2 s. Also `outputs/outputs_temporal` window-level **test AUC 0.991**. The "recommended" operating point in `final/detection_fa_tradeoff.png` is θ=0.3, w=10: 91.9 % detection, 4.87 FA/h, 23.9 s lead. **None of these numbers can be trusted** (see §3). |
| **Most trustworthy result** | The student MLP from `training/train_student_original.py` on the **movie-grouped split** (`--split_group source_movie`) with the threshold chosen on validation: `runs/student_fix1`, `runs/student_fix2`, `outputs/student_mlp`, `results/anticipation_prboost_v3`. **Test frame-level ROC-AUC is 0.58–0.62 and AP is 0.10–0.16 against a base rate of 0.081.** Val AUC is 0.76–0.82. **XD-only test AUC is 0.50–0.56, which is chance level.** |
| **Honest summary** | Under leak-free splits, no model in this repo reliably anticipates violence. Under the 30 s definition, the best honest models fire on about 60 % of violent test videos, with a mean lead of about 10 s (p90 about 21 s) and false alarms on about 15 % of normal videos. No artifact measures detection at ≥30 s lead on a correct time base. |
| **Why the headline numbers are wrong** | (1) Labels are misaligned with the VLM embeddings: they are truncated, not indexed by `t`. (2) "1 step = 1 second" is assumed everywhere, but a VLM step is really 0.2–750 s (median 3.2 s). (3) The 772-video "matched/fused/clean/lowfa" simulations are 81 % **training** videos. (4) Random video splits leak clips from the same movie (44 of the 83 XD test clips share a movie or YouTube source with a training clip). (5) Thresholds were tuned on the test sets. (6) Lead time is capped at 30 s by construction, and an alarm that is always on scores perfectly on the event metrics. |

---

## 1. Shared caveats that affect almost every result

Read this section before quoting any number. The section and row IDs below refer back to these items.

### C1. Label and embedding time misalignment (truncation bug)
- `data/processed/vlm_emb/<src>/<vid>.npz` holds `emb [E,3584] float16`, `t [E] int32`, and `raw_frame_idx [E]`. `t` steps by 5 over the graph/label index (`t = 0,5,10,…`).
- `data/processed/labels_multih/<vid>.npy` is `[T,7]` over the **graph** time base. The 7 horizons are 2, 5, 10, 20, 30, 60 and 120 s; a value of −1 means "inside a violence segment" (`labels_info.json`, `ignore_inside_segments: true`).
- `sim/build_long_simulation.py`, `training/train_temporal_model.py`, `training/train_student.py`, `scripts/build_fused_features.py`, `experiments/run_experiments.py` and `experiments/algorithm_variations.py` all do `labels.max(axis=1); labels = labels > 0; min_len = min(len(emb), len(labels)); labels = labels[:min_len]`.
  - Embedding step *i* is therefore paired with label row *i*, but it should be paired with label row `t[i] = 5i`. **This is true for 99 % of the 772 XD videos (E < T).**
- Taking the max over the 7 horizons means `frame_label = 1` when violence starts within the next **120 s**. Steps inside violence (−1) become 0, so they count as *negative*.
- The only pipelines that index labels correctly are `training/train_student_original.py` (`gt_vec = y_arr[t]`) and `scripts/eval_vlm_teacher.py` (`label_arr[t]`).

### C2. "1 step = 1 second" is false
- Every simulator, trainer and evaluator hard-codes `fps=1.0`, so a "30 s horizon" means 30 embedding steps and every "lead time in seconds" is really a step count.
- Measured over the 772 XD videos, one VLM step is median **3.17 s** (p10 0.95 s, p90 20.7 s, min 0.21 s, max 756 s). One label/graph step is median 0.64 s.
- The real-time demo (`realtime/rtsp_inference.py`) samples at 1 fps with `--sequence_length 32`, which matches neither the training time base nor the training sequence length (30).

### C3. Data leakage through splits
- `train_index.json`, `val_index.json` and `test_index.json` (dated 2026-02-17) are a **random per-clip** split (`scripts/split_train_val.py`). There are no duplicate clip names across splits, but **44 of the 83 XD test clips come from a movie or YouTube source that also has clips in train** (e.g. two `Braveheart.1995__#…` clips). XD-Violence clips from one movie share actors, scenes and grading.
- `train_student_original.py` runs without a group split (all `results/anticipation_*` folders except `prboost_v3`) use a random video split. With the identical code and features, switching to `--split_group source_movie` drops test AUC from about 0.92–0.94 to about 0.58–0.62, and XD test AUC from about 0.95–0.97 to about 0.50–0.56. This is strong evidence of movie-level leakage **(inferred cause; the size of the drop is measured)**. Validation AUC stays around 0.75–0.85 in both settings. The grouped validation and test sets still differ by about 0.2 AUC, so a single grouped split is high-variance; use grouped K-fold.
- The "matched / fused / vlm_audio / audio_only / filtered / clean / clean_t02 / lowfa" simulations contain all 772 XD videos or subsets of them. **626/772 (81 %) are training videos**, 67 are val and 79 are test (checked against the index files). Every evaluation on these files is contaminated.

### C4. Test-set threshold tuning
- θ = 0.388, used throughout `final/`, appears to come from `sim/temporal_simulator.py --target_fa_per_hour` auto-selection, which runs **on the test simulation itself** **(inferred)**.
- `experiments/run_experiments.py:evaluate_simulation` picks the best-F1 threshold on the evaluated set.
- `outputs/*/final_metrics.json["test"]["best_threshold"]` and `results/*["test_metrics_oracle"]` are oracle thresholds picked on test.
- `scripts/late_fusion.py` picks the "best" threshold on the evaluated set by maximizing `det_rate − 0.1·FA/h`.
- The only honest threshold reporting is `test_metrics_at_val_threshold` and `anticipation_video_stats_at_val_threshold` in the student-MLP metrics.

### C5. Event-metric definitions that inflate numbers
`sim/comprehensive_eval.py` (the source of most of `final/`):
- An event is a run of `anticipation_labels == 1`, which already starts 30 steps before `frame_label` onset. The detection window is `[start−30, end]`, so it **includes the event itself**: detecting during violence counts as "detected", with lead 0. Lead is `max(0, start−i)`, which is **capped at 30**.
- An alarm that was already on before the window gets lead = 30.
- In the padded 8 h sim, the 30-step look-back window reaches into the **previous, unrelated padding video**, so a "30 s lead" is often a coincidental false alarm.
- `event_details[*].lead_time_sec` is `null` whenever the lead is 0, because `float(x) if x else None` treats 0.0 as falsy. This is a cosmetic bug.
- The alarm counting in `_find_false_alarm_times` means **an always-on alarm produces about 1 FA in total**, 100 % detection and a 30 s lead. See `final_evaluation_lstm_2`, `late_fusion_*`, and θ=0.05 in `final/final_evaluation_simple`.
- `--smoothing_window` uses a centered `np.convolve(..., mode='same')`, which is **non-causal** because it looks into the future. It is applied only to the main evaluation, not to the threshold sweep.

Other evaluators:
- `scripts/evaluate_with_smoothing.py` uses `frame_labels` runs as events (so its event counts differ from comprehensive_eval) and counts FA on `anticipation_labels == 0`.
  - Its `apply_cooldown` turns an alarm into isolated one-step pulses, and each pulse counts as a new FA. This is why every `cd=30/60` row has *higher* FA/h.
  - Its `TemporalViolenceModel` uses **ReLU where training used GELU** (`training/train_temporal_model.py`). The state-dict loads anyway, but the outputs differ: AUC 0.8112 here versus 0.8127 from comprehensive_eval on the same model and sim.
- `scripts/late_fusion.py` has the same ReLU/GELU mismatch.
- `sim/temporal_simulator.py` and `sim/live_simulator.py` average only leads > 0, so detections with lead 0 are dropped from the mean.

### C6. Weak dataset for anticipation
- `results/xd_investigation/investigation_stats.json` (1000 videos) finds that violence starts at mean relative position 0.019 of the clip, median pre-violence fraction 0. Only **4 videos** have at least 5 pre-violence steps under that script's alignment. The XD-Violence and UCF-Crime clips mostly start with violence already in progress.
- UCF-Crime anomaly videos without official temporal annotations (the UCF train set) get all-zero labels in `labels_multih`, which is label noise **(inferred from `build_multihorizon_labels.py`: no segments means all-negative)**.
- `data/processed/labels` has positive labels in only 215 of 500 violence-named XD clips.

### C7. Overwritten checkpoints
- `outputs/outputs_temporal/best_temporal_model.pt` was rewritten on 2026-02-23 10:22. The `simulation_results_temporal` runs from Feb 19 (AUC 0.8338) and the run from Feb 23 (AUC 0.8502) therefore used different weights.
- No `final/*` comprehensive-eval JSON records the model path; only `final/final_evaluation_simple` stores `meta.model_path`. The model-to-result mapping below is **(inferred)** from timestamps, input dimensions and identical AUC values.

---

## 2. Evaluation sets (all in `simulation_data/`; stats measured)

| Sim file | Steps × dim | Videos (train/val/test) | Pos rate (antic) | Antic runs | Builder | Problems |
|---|---|---|---|---|---|---|
| `xd_violence_8hr_sim.npz` | 32,092 × 3584 (treated as 8.91 h) | XD **test** clips only (inferred: 13 violent clips padded with cycled non-violent test clips) | 1.74 % (557) | 13 (len 30–70) | `sim/build_long_simulation.py` (default `--test_index test_index.json`) | C1, C2, C3 (movie overlap), padding reuses the same normal clips many times. Keys: `embeddings, frame_labels(161 pos, 18 runs), anticipation_labels, fps, horizon_sec`. **The least-contaminated sim; this is the legacy "headline" set.** |
| `xd_violence_8hr_sim.meta.json` | — | — | — | — | same | `n_violent_events: 13, n_violent_frames: 161, n_anticipation_frames: 557` |
| `xd_violence_8hr_sim_violence.npz` (+`.meta.json`) | 32,092 × 3584 | same layout, embeddings from `vlm_emb_violence` (inferred) | 1.74 % | 13 | same | same as above |
| `xd_violence_8hr_flow_sim.npz` | 32,092 × 22 | same | 1.74 % | 13 | `sim/build_flow_sim.py` | 22-d optical-flow stats aligned to the 8 h sim |
| `xd_violence_8hr_flow_sim_fused.npz` | 32,092 × 3606 | same | 1.74 % | 13 | `sim/build_flow_sim.py` | VLM + flow concat (`vlm_dim`, `flow_dim` keys) |
| `xd_violence_vlm_matched.npz` | 47,310 × 3584 (13.14 h) | 772 (626/67/79) | 36.0 % | 183 (min 30) | builder **not in repo** | C3, **81 % training videos**. Anticipation computed across video boundaries (runs spill into the previous clip), so labels differ from the fused/vlm_audio sims. |
| `xd_violence_8hr_fused.npz` | 47,310 × 4124 (VLM + 540 pose, `concat_norm`) | 772 (626/67/79) | 23.45 % | 280 | `scripts/build_fused_features.py` (per-video anticipation) | Name says 8 h but it is 13.1 h and unpadded. C3. |
| `xd_violence_vlm_audio.npz` | 47,310 × 3675 (VLM + 91 audio) | 772 | 23.45 % | 269 | `scripts/multimodal_fusion.py` (inferred) | C3 |
| `xd_violence_audio_only.npz` | 234,751 × 91 | 772 | 14.6 % | 476 | `scripts/multimodal_fusion.py` (inferred) | Audio on the label time base, about 5× more steps than VLM. C3. |
| `xd_violence_vlm_filtered.npz` | 41,737 × 3584 | 752 (610/65/77) | 36.5 % | 174 | not in repo | C3 |
| `xd_violence_vlm_clean.npz` | 16,070 × 3584 | 431 (351/36/44) | 63.4 % | 39 | not in repo | C3; **most normal videos removed** (364/431 are violence-named) |
| `xd_violence_vlm_clean_t02.npz` | 14,052 × 3584 | 394 (323/34/37) | 69.5 % | 19 | not in repo | C3; 356/394 violence-named |
| `xd_violence_vlm_lowfa.npz` | 14,845 × 3584 | 433 (357/36/40) | 69.9 % | 49 | not in repo | C3. "lowfa" = selected for low false alarms (inferred; overlaps 396/433 with `good_video_indices.npy`). Keeps 384/481 violent and only 49/291 normal clips, so the selection is circular. |
| `xd_diluted_5pct_8hr.npz` | 28,790 × 3584 | from `clean_t02` | 4.97 % | 5 (160–445 steps) | `sim/create_diluted_simulation.py` | Negatives are all label-0 frames of `clean_t02`, stitched across videos, **tiled about 6× plus N(0, 0.01) noise** (the code's tile branch, because only about 4.3k negatives exist). The positive runs are cut out and inserted between negative chunks, so the 30 steps before each event come from unrelated footage, which makes lead 0 inevitable. Each "event" merges several videos. C3. |
| `ucf_crime_8hr_sim.npz` | 28,800 × 3584 | 949 UCF clips (`metadata`: 116 violent / 833 normal) | 10.2 % | 117 | `sim/create_ucf_crime_simulation.py` | Cross-domain for XD-trained models; includes UCF train videos with all-zero labels (C6) |

Other eval sets:
- **Repo index split** (`train/val/test_index.json`): 1438/180/180 clips. XD clips: 645/70/83; UCF: 793/110/97. `eval_index.json` is identical to `test_index.json`.
  - With `seq_len=30, stride=5` and C1 labels, the XD test split gives only **45 clips with ≥1 window, 1,020 windows, 109 positive windows from 13 videos**. Val: 37 clips, 608 windows, 147 positive windows from 14 videos. Recomputed here.
- **Student-MLP grouped split** (`runs/student_fix1/splits_manifest.json`; identical in `fix2`, `outputs/student_mlp`, `results/anticipation_prboost_v3`): `source_movie` groups 1072/134/134, videos 1356/156/158.
  - Samples are VLM steps outside violence, labelled with the 30 s column of `labels_multih` (correct `t` indexing).
  - Test: 8,632 samples, base rate 0.081, 39 positive videos. Test XD: 61 videos, 5,468 samples.
- **Student-MLP random split** (all other `results/anticipation_*`): 1336/167/167 or 1317/165/165 videos, no grouping (C3).

---

## 3. Master comparison table

Abbreviations:
- **AUC** = ROC-AUC. **AP** = average precision / PR-AUC (base rate in brackets where known).
- **Det** = events detected / events. **FA/h** = false-alarm episodes per hour, with 1 step = 1 s (C2). **Lead** = mean lead in "s" (= steps). **≥30** = number of detections with the capped lead of 30.
- **Thr** = threshold, and how it was chosen: *test* = tuned on the evaluated set, *val* = chosen on validation, *fixed* = hand-set, often after looking at test.
- **Trust** levels:
  - INVALID = invalid or leaky.
  - FLAWED = protocol flaws, comparable only within its own family.
  - TRUSTED = most trustworthy available.

### 3A. Legacy 8-hour XD simulation (`xd_violence_8hr_sim*.npz`, 32,092 steps, 13 events, base rate 0.0174)

| Folder (producer) | Model (inferred if not recorded) | Thr | AUC | AP | P / R / F1 / Acc | Det | FA/h | Lead | ≥30 | Trust / notes |
|---|---|---|---|---|---|---|---|---|---|---|
| `final/evaluate_original` (comprehensive_eval) | Temporal LSTM (`outputs/outputs_temporal`, inferred) | 0.388 test | 0.834 | 0.131 | .071/.431/.121/.892 | 11/13 | 8.64 | 11.4 | 4 | FLAWED C1–C5. **Legacy reference row.** |
| `final/final_evaluation_lstm_violence` (older comprehensive_eval schema) | same weights as above (identical AUC 0.83442520) | 0.373 | 0.834 | 0.131 | .068/.449/.119/.884 | 11/13 | 8.97 | 25.0* | 4 | FLAWED *older code averaged only leads > 0 |
| `final/final_evaluation_lstm` (older schema, Feb 23 10:16) | Temporal LSTM (pre-retrain, inferred) | 0.388 | 0.834 | 0.166 | .115/.442/.183/.931 | 13/13 | 6.51 | 20.7* | 3 | FLAWED |
| `final/test_original` | Temporal LSTM, unknown checkpoint | 0.60 | 0.854 | 0.138 | .095/.801/.170/.864 | 12/13 | 10.10 | 19.1 | 4 | FLAWED |
| `final/final_evaluation_simple` (schema with `meta`) | **VLM student MLP** `runs/outputs_xd/best_model.pt` (recorded) | 0.52 | **0.853** | **0.242** | .167/.720/.272/.933 | 13/13 | 43.4 | 13.3 | 1 | FLAWED. The θ=0.05 sweep row reports 100 % det at 0.11 FA/h, which is the always-on artifact (C5). |
| `final/final_evaluation_vlm_student` | same weights (identical AUC 0.8529) | 0.60 | 0.853 | 0.241 | .293/.684/.410/.966 | 13/13 | 26.4 | 9.6 | 1 | FLAWED |
| `simulation_results/final` (`sim/live_simulator.py`) | same VLM student (AUC 0.8529) | 0.60 | 0.853 | 0.242 | .379/.612/.468/– | 13/13 | 9.76 (87 FA) | 13.8* | – | FLAWED *mean of leads > 0 only |
| `final/final_evaluation_lstm_2` | unknown LSTM | 0.55 | 0.633 | 0.039 | .017/**1.000**/.034/**.018** | 13/13 | 1.46 | **30.0** | 13 | INVALID **Always-on alarm** (98 % of steps positive), yet "perfect" event metrics (C5) |
| `final/final_evaluation_optical_flow_fused` | `outputs_fused/best_temporal_model.pt` (VLM + 22-d flow, inferred from input 3606) | 0.20 | 0.557 | 0.024 | .028/.354/.051/.773 | 6/13 | 9.20 | 25.0 | 5 | FLAWED near chance |
| `simulation_results_temporal/thresh_0.388` (`sim/temporal_simulator.py`, Feb 23) | Temporal LSTM (new weights) | 0.388 | 0.850 | 0.227 | .357/.160/.221 | 5/13 | 0.79 (7 FA) | 0* | – | FLAWED |
| `simulation_results_temporal/thresh_0.45` (Feb 19) | Temporal LSTM (old weights) | 0.45 | 0.834 | 0.166 | .112/.363/.171 | 11/13 | 5.38 | 20.2* | – | FLAWED |
| `simulation_results_temporal/thresh_0.55` | same | 0.55 | 0.834 | 0.166 | .143/.325/.198 | 9/13 | 4.60 | 20.2* | – | FLAWED |
| `simulation_results_temporal/thresh_0.6` | same | 0.60 | 0.834 | 0.166 | .159/.316/.212 | 9/13 | 3.03 | 17.5* | – | FLAWED |
| `simulation_results_contrastive` (`tuning/contrastive_simulator.py`) | contrastive projection head (`outputs/outputs_contrastive/contrastive_model.pt`, inferred) | 0.50 | 0.730 | 0.120 | .120/.253/.162 | 13/13 | 25.0 | 23.0 | – | FLAWED |
| `simulation_results_contrastive/thresh_0.7` | same | 0.70 | 0.730 | 0.120 | .194/.201/.197 | 13/13 | 16.2 | 18.6 | – | FLAWED |
| `simulation_results_ensemble/average_strategy` (`tuning/ensemble_simulator.py`) | VLM student + LSTM average | 0.6 / 0.388 | 0.853 | 0.204 | .162/.483/.243 | 13/13 | 9.76 | 21.0 | – | FLAWED |
| `simulation_results_ensemble/weighted_0.7` | weighted | same | 0.848 | 0.190 | .135/.452/.207 | 13/13 | 6.84 | 20.7 | – | FLAWED |
| `simulation_results_ensemble/and_strategy` | AND | same | 0.840 | 0.176 | .198/.314/.243 | 10/13 | 9.09 | 17.6 | – | FLAWED |
| `final/late_fusion_max` (`scripts/late_fusion.py`) | VLM LSTM + audio LSTM, max fusion | 0.1 test | VLM 0.814 / audio **0.540** / fused 0.664 | 0.125 / 0.023 / 0.113 | – | 19/19 | **0.11** | 30.0 | – | INVALID The audio predictions come from a different 234k-step stream **stretched** onto this 32k-step timeline (not aligned, so audio is at chance). θ=0.1 is always on. ReLU/GELU mismatch. |

### 3B. 772-video concatenations (13.14 h, 81 % training videos), all INVALID for generalization

| Folder | Model | Eval set | Thr | AUC | AP (base) | P / R / F1 | Det | FA/h | Lead | ≥30 |
|---|---|---|---|---|---|---|---|---|---|---|
| `final/evaluate_vlm_matched` | Temporal LSTM, VLM only | vlm_matched | 0.50 | 0.813 | 0.694 (.360) | .626/.663/.644 | 172/183 | 16.6 | 18.6 | 92 |
| `final/high_sensitivity_eval` | same | vlm_matched | 0.20 | 0.813 | 0.694 | .523/.895/.661 | 183/183 | 20.4 | 26.5 | 144 |
| `final/temporal_t0.3` (`temporal_simulator`) | same | vlm_matched | 0.30 | 0.813 | 0.694 | .563/.828/.670 | 182/183 | 13.3 (175 FA) | 27.5* | – |
| `final/high_sensitivity` (`evaluate_with_smoothing`, ReLU) | same | vlm_matched; events = `frame_labels` runs | 0.20, w1 | 0.811 | 0.694 | .539/.863/.664 | 352/356 | 6.47 | 26.9 | 292 |
| `final/smoothing_sweep` (48 configs) + `final/detection_fa_tradeoff.png` | same | same | grid | 0.811 | 0.694 | – | **"recommended" θ=.3, w=10: 327/356 = 91.9 %** | **4.87** | 23.9 | 224 |
| `final/smoothing_w15` / `_w20` / `_w10_t0.5` / `_w10_t0.6` / `_w10_t0.75` / `_w15_t0.4` / `_w15_t0.7` / `_w5_t0.6` / `_w5_t0.7` / `_w10_cd60` | same | same | see name | 0.811 | 0.694 | – | 316 / 308 / 268 / 239 / 167 / 292 / 184 / 258 / 214 / 246 of 356 | 4.87 / 4.26 / 5.02 / 5.25 / 2.44 / 4.95 / 2.97 / 5.33 / 3.58 / 12.56 | 23.6 / 23.0 / 19.6 / 17.3 / 12.7 / 21.1 / 13.6 / 17.7 / 15.5 / 8.2 | – |
| `final/evaluate_fused` | `outputs/outputs_temporal_fused` (VLM + pose, 4124-d) | 8hr_fused | 0.388 | 0.887 | 0.706 (.235) | .615/.741/.672 | 218/280 | 20.9 | 14.8 | 86 |
| `final/evaluate_vlm_audio` | `outputs/outputs_temporal_vlm_audio` (3675-d) | vlm_audio | 0.388 | 0.868 | 0.670 (.235) | .585/.721/.646 | 199/269 | 22.8 | 14.8 | 77 |
| `final/evaluate_audio_only` | `outputs/outputs_temporal_audio_only` (91-d) | audio_only (65.2 h) | 0.53 | **0.438** | 0.153 (.146) | .304/.058/.097 | 94/476 | 3.33 | 6.2 | 16 |
| `final/high_sensitivity_filtered` | VLM LSTM | vlm_filtered | 0.50 | 0.854 | 0.773 (.365) | .730/.645/.685 | 162/174 | 16.6 | 18.1 | 82 |
| `final/late_fusion_matched` | VLM + audio, max | vlm_matched (old version with −1 labels) | 0.1 test | **0.5 / 0.0** (exception fallback) | – | – | 391/391 | 0.08 | 30.0 | – |

Note: `evaluate_vlm_matched` (AUC 0.813) and `evaluate_fused` / `evaluate_vlm_audio` (0.887 / 0.868) use **different anticipation labels** (36 % vs 23.45 % positives), so the "+pose" and "+audio" gains are not apples-to-apples.
`late_fusion_matched` is INVALID: its labels contained −1, sklearn raised, and the code silently substituted AUC 0.5 / AP 0.0.

### 3C. Curated subsets, all INVALID (training videos, normal clips removed, circular selection)

| Folder | Set | Thr | AUC | AP (base) | P / R / F1 / Acc | Det | FA/h | Lead | ≥30 |
|---|---|---|---|---|---|---|---|---|---|
| `final/clean_subset_eval` | vlm_clean (4.46 h) | 0.20 | 0.961 | 0.975 (.634) | .896/.955/.925/.901 | 39/39 | 10.5 | 22.2 | 26 |
| `final/clean_t02_eval` = `final/clean_thresh_0.2` | clean_t02 (3.90 h) | 0.20 | **0.982** | **0.991** (.695) | .965/.966/.966/.952 | 19/19 | 4.87 | 23.4 | 14 |
| `final/clean_thresh_0.3` / `_0.4` / `_0.5` | clean_t02 | .3/.4/.5 | 0.982 | 0.991 | F1 .960/.952/.928 | 19/19 | 4.61/4.61/4.10 | 23.2/18.3/17.4 | 14/11/11 |
| `final/clean_smoothing_sweep` (48 configs, `evaluate_with_smoothing`) | clean_t02; 200 `frame_labels` events | grid | 0.985 | 0.992 | best: θ=.2 w1 P .966 R .963 | **199/200** | **0.00** | 29.2 | 188 |
| `final/lowfa_eval` | vlm_lowfa (4.12 h) | 0.20 | 0.974 | 0.985 (.699) | .953/.968/.961/.945 | 48/49 | 11.9 | 26.3 | 41 |
| `final/eval_diluted_5pct` (Mar 23) | xd_diluted_5pct (8.0 h) | 0.388 | 0.945 | 0.851 (.050) | .925/.820/.869/.988 | 5/5 | 1.00 | **0.0** | 0 (pre-event context is unrelated tiled footage, so this measures detection only) |

### 3D. Cross-domain (UCF-Crime), honest about failure

| Folder | Model | Set | Thr | AUC | AP (base) | Det | FA/h | Lead |
|---|---|---|---|---|---|---|---|---|
| `final/eval_ucf_crime` (Mar 23) | XD-trained Temporal LSTM | ucf_crime_8hr_sim | 0.388 | **0.459** | 0.090 (.102) | 112/117 | 30.5 | 26.6 (Acc .282, fires almost constantly) |
| `experiments/output/exp4_ucf_crime_eval/exp4_ucf_crime_metrics.json` (`run_experiments.py --exp4`) | `outputs/outputs_temporal` | UCF test clips, padded sim (8.1 h) | 0.642 test | 0.479 | 0.035 | 11/15 | 24.3 | 16.5 |
| `experiments/output/exp4_ucf_crime_eval/exp4_xd_violence_metrics.json` | same | XD test clips, padded sim (23.0 h, 39 events) | 0.632 test | 0.826 | 0.150 | 18/39 | 1.91 | 8.3 |
| `runs/outputs_ucf/results.json` (`training/train_student.py`) | VLM student trained on UCF | UCF test frames (3,296) | – | val 0.444 / test **0.558** | test 0.175 (.161) | – | – | – |
| `results/ucf_crime_results.csv` (`experiments/ucf_crime_testing.py`) | LR / LSTM / LDA, XD→UCF | – | – | **NaN** | NaN | – | – | – |

### 3E. Window/frame-level held-out "test" from training scripts (repo index split, XD only unless noted; oracle threshold on test)

| Folder (producer) | Features (input dim) | Val AUC / AP | **Test AUC / AP** | Test oracle F1 (P/R) | Trust |
|---|---|---|---|---|---|
| `outputs/outputs_temporal` (`train_temporal_model.py`, Feb 23 weights) | vlm_emb (3584) | 0.577 / 0.365 | **0.991 / 0.907** | .888 (.868/.908) | INVALID The val/test gap is +0.41 AUC. Only 13 positive test videos (109 positive windows of 1,020). C1, C3. |
| `outputs/outputs_temporal_old` | vlm_emb | 0.600 / 0.338 | 0.957 / 0.753 | .755 | INVALID |
| `outputs/outputs_temporal_old_emb` | vlm_emb_old | 0.665 / 0.357 | 0.947 / 0.675 | .781 | INVALID |
| `outputs/outputs_temporal_vlm_emb` | vlm_emb | 0.575 / 0.356 | 0.893 / 0.720 | .759 | INVALID (best val thr 0.002) |
| `outputs/outputs_temporal_violence` | vlm_emb_violence | 0.533 / 0.240 | 0.912 / 0.785 | .758 | INVALID |
| `outputs/outputs_projected` | contrastive projected (256) | 0.598 / 0.281 | 0.945 / 0.756 | .795 | INVALID |
| `outputs/outputs_temporal_fused` | VLM + pose (4124) | 0.674 / 0.333 | 0.897 / 0.445 | .574 | INVALID |
| `outputs_fused` | VLM + flow (3606) | **0.721** / 0.426 | 0.920 / 0.698 | .615 | INVALID |
| `outputs/outputs_temporal_vlm_audio` | VLM + audio (3675) | 0.693 / 0.367 | 0.773 / 0.469 | .516 | INVALID |
| `outputs/outputs_temporal_audio_only` | audio (91) | 0.642 / 0.337 | 0.759 / 0.345 | .477 | INVALID (different window count: audio is on the label time base) |
| `outputs/contrastive_v3/metrics.json` (`tuning/contrastive_projection.py`) | vlm_emb, frame-level | best val 0.656 | 0.903 / 0.637 | .639 | INVALID |
| `outputs/contrastive_v2_large/history.json` | vlm_emb | max val 0.666 / 0.449 | – | – | FLAWED val only |
| `runs/outputs_xd/results.json` (`train_student.py`, XD) | vlm_emb frame-level | 0.680 / 0.439 | 0.925 / 0.632 (.164) | .674 | INVALID |
| `runs/outputs/results.json` (XD + UCF) | vlm_emb | 0.617 / 0.347 | 0.723 / 0.430 (.163) | .453 | FLAWED |
| `results/transformer_model.pt` (see `algorithm_comparison.csv`) | — | — | — | — | weights only |

**Pattern:** validation AUC is 0.53–0.72 for every temporal model, but test AUC is 0.77–0.99. A tiny test set (13 positive videos), movie leakage and oracle thresholds explain the gap (inferred). **Validation is the more honest of the two numbers here.**

### 3F. Student MLP (`training/train_student_original.py`): the only protocol with correct `t` alignment and seconds-based lead times

Features are VLM embedding plus hashed VLM-teacher tags (actions 64 + weapons 32 + scene 64), 3744-d. `prboost_v1/v2` use 3746-d and `prboost_v3`/`fix2` use 4034-d. The label is the 30 s column of `labels_multih`; in-violence steps are ignored. Per-source and bucket AUCs below were **recomputed from `per_sample.csv`**.

| Folder | Split | Val AUC (XD) | **Test AUC / AP** (base .081–.109) | Test XD AUC | Test AUC, 30–60 s-before bucket vs no violence within 120 s | Test @ val thr: P / R / F1 | Video-level @ val thr (test) | Trust |
|---|---|---|---|---|---|---|---|---|
| `runs/student_fix1` (fbeta thr .709) | **source_movie** | 0.805 (0.824) | **0.582 / 0.159** | 0.521 | 0.605 (XD 0.501) | .109/.270/.156 | 24/40 pos videos alerted in window (60 %), early-false 10 %, **18/118 neg videos FA (15.3 %)**, lead mean 10.0 s / median 7.8 / p90 20.7 | TRUSTED |
| `runs/student_fix2` (fbeta, 4034-d, min_consec 2) | source_movie | 0.772 (0.813) | 0.599 / 0.142 | 0.497 | 0.584 (0.458) | .085/.266/.128 | 20/40 (50 %), 29/118 neg FA (24.6 %), lead 9.7 s | TRUSTED |
| `outputs/student_mlp` (f1, meanpool emb) | source_movie | 0.763 (0.809) | **0.622** / 0.151 | 0.564 | 0.611 (0.498) | .126/.423/.195 | 24/40 (60 %), 28/118 neg FA (23.7 %), lead 10.3 s | TRUSTED |
| `results/anticipation_prboost_v3` (precision_floor) | source_movie | 0.801 (0.828) | 0.587 / 0.095 | 0.495 | 0.623 (0.506) | .034/.088/.050 | 12/40 (30 %), 9/118 neg FA, lead 10.8 s | TRUSTED |
| `results/anticipation_prboost_v1` | random | 0.829 | 0.942 / 0.763 | 0.967 | 0.851 (0.932) | .597/.766/.671 | 42/59 (71 %), 29/108 neg FA, lead 8.5 s | INVALID C3 |
| `results/anticipation_prboost_v2` | random | 0.853 | 0.944 / 0.714 | 0.951 | – | .425/.857/.569 | 40/59, 32/108 neg FA | INVALID C3 |
| `results/anticipation_multih_hset_v2` | random | 0.839 | 0.930 / 0.750 | 0.950 | – | .531/.757/.624 | 26/59, 8/108 neg FA | INVALID C3 |
| `results/anticipation_multih_hset_v4` | random (1317/165/165) | 0.750 | 0.833 / 0.520 | 0.818 | 0.656 (0.624) | .787/.118/.206 | 11/64, 1/101 neg FA | INVALID C3 |
| `results/anticipation_multih_hset_v4_ema085_k3_gap2_fap7` | random | 0.840 | 0.913 / 0.720 | – | – | .815/.511/.628 | 28/59, 8/108 neg FA | INVALID |
| `results/anticipation_multih_hset_v4_ema085_k2_gap5_fap7` | random | 0.840 | 0.913 / 0.720 | – | – | .815/.511/.628 | 22/59, 6/108 | INVALID |
| `results/anticipation_multih_hset_v4_ema085_k2_gap5_fap7_late3` | random | 0.840 | 0.913 / 0.720 (recomputed 0.895 / 0.702) | 0.934 | – | .810/.529/.640 | 23/59, 7/108 | INVALID |
| `results/anticipation_multih_hset` | random, meanpool | 0.818 | 0.904 / 0.710 | – | – | .136/.942/.238 (thr .08) | 37/59 | INVALID |
| `results/anticipation_multih_hset_v3` | random, meanpool | 0.818 | 0.904 / 0.710 | – | – | thr 1.0 → **0 alerts** | 0/59 | INVALID broken threshold |
| `results/anticipation_preonly_h30_grace1_ignore` (+`_fp1..4`, `_meanpool`) | random, only 610 videos with pre-violence labels (`labels_preonly_h30_ignore`) | **0.48–0.55 (chance)** | 0.51–0.73 / 0.55–0.80 (base .516) | – | – | fpr 0.69–1.0 (always positive) | – | INVALID |

### 3G. Zero-shot VLM teacher (`scripts/eval_vlm_teacher.py`; Qwen2.5-VL-7B JSONL violence score; no training, so no split leakage)

| Folder | GT labels | All AUC / AP (base) | XD AUC / AP | UCF AUC | Notes |
|---|---|---|---|---|---|
| `results/vlm_eval` | `data/processed/labels` (current violence) | 0.682 / 0.194 (.106) | **0.776** / 0.308 | 0.497 | 62,580 (score, gt) pairs, `label_arr[t]` alignment is correct. C6: these labels are positive in only 215/500 violent XD clips. |
| `results/vlm_eval_future_h5` | violence now or in next 5 s | 0.677 / 0.202 | 0.776 / 0.320 | 0.490 | |
| `results/vlm_eval_future_h10` | now or in next 10 s | 0.672 / 0.206 | 0.776 / 0.327 | 0.484 | |
| `results/vlm_eval_future_h30` | now or in next 30 s (`scripts/build_anticipation_labels.py`) | 0.662 / 0.216 (.131) | **0.772** / 0.341 | 0.476 | Mostly a detection metric, because the label includes current violence |

### 3H. Other experiment artifacts

| Artifact (producer) | Result | Trust / notes |
|---|---|---|
| `results/algorithm_comparison.csv` (`experiments/algorithm_variations.py`, Mar 16) | LR 0.631/0.775; Transformer 0.807/0.524; LR + temporal 0.594/0.637; best SlidingWindow_mean_w5 0.844/0.914 (AUC/AP) | INVALID The table mixes metric types: LR and sliding-window rows are per-video mean AUC over videos with both classes, the Transformer row is pooled. Validation = last 20 % of the first 300 alphabetically sorted `vlm_emb` files. With the current data that is **208 UCF videos → 41 UCF val videos** (inferred). C1. AP > AUC signals a high base rate. |
| `results/ema_results.csv` (`experiments/ema_smoothing.py`) | score-EMA raw 0.6305, best α=0.9 0.619, α=0.1 **0.372**; embedding-EMA raw 0.465, best α=0.9 0.592 | FLAWED EMA never helps |
| `results/prompt_engineering.csv` (`experiments/prompt_engineering_stats.py --mode offline`) | `lda_auc` **1.0**, Cohen d 0.233, 358 violent / 6631 non-violent | INVALID In-sample LDA in 3584-d is a guaranteed overfit |
| `results/lda_calibration_results.txt` (`continuation/lda_calibration_complete.py`) | Every calibration (raw, z-score, percentile, CORAL, relative) has `violence_highest: False, anticipation_rises: False` on the YouTube clip `probe_clip_01` | TRUSTED qualitative: LDA does not transfer to real footage |
| `probe_clip_01_scores.npy` / `_embeddings.npy` (197 steps; `scripts/analyze_youtube.py --save_scores/--save_embeddings`) + `analysis_probe_clip_01.png` | Mean score: safe (0–6 s) 0.539, anticipation (7–36 s) 0.678, violence (37–45 s) 0.738, post (46 s+) 0.658, post max **0.767** (higher than during violence) | FLAWED n = 1 real video. The ranking is right on average but the scores are not discriminative and a threshold of 0.5 fires everywhere. |
| `experiments/output/exp1_person_detection/exp1_results.json` (`run_experiments.py --exp1`) | Baseline AUC 0.832 / AP 0.481, FA/h 8.2. With YOLO person masking: AUC 0.812 / AP 0.479, FA/h 10.3. | FLAWED Person filtering made false alarms worse (23 → 29 FA events). 173 test clips. |
| `experiments/output/exp2_clip_analysis/exp2_unified_metrics.json` (`--exp2`) | All test clips (XD + UCF): AUC 0.699, AP 0.136, det 33/54, FA/h 9.99, lead 11.0 | FLAWED test-tuned threshold, C1/C2 |
| `experiments/output/exp3_cross_dataset/exp3_all_results.json` (`--exp3`, retrains LSTMs; weights in `*/best_temporal_model.pt`) | xd_only → XD 0.768, UCF 0.538, both 0.726. ucf_only → XD 0.666, UCF 0.531, both 0.616. combined → XD 0.634, UCF 0.463, both 0.587. | FLAWED Train AUC → 0.99 while val AUC stays 0.49–0.61, which is heavy overfitting. Thresholds are test-tuned. |
| `outputs/qa_feature_test/summary.json` | Question-answer VLM features, 50/50 clips: logistic test AUC **0.486**, weighted-score AUC 0.596 | FLAWED chance |
| `logs/lora_train_v2.log` / `lora_train_v2_full.log` (`tuning/lora_finetune_vlm.py`, Feb 23) | Qwen2.5-VL LoRA (23.8 M trainable params): val acc **0.4925 (197/400) in every epoch** while loss falls 10.6 → 3.5. Only "valid=225" of 3600 train samples per epoch. | INVALID Collapsed to a constant answer. Outputs are in `lora_finetuned*/` (outside scope). |
| `results/xd_investigation/` (`experiments/xd_violence_investigation.py`, Mar 16) | 1000 videos; 136 violent; **4 usable for anticipation**; violence starts at clip position 0.019 on average | TRUSTED dataset diagnosis (C6) |

---

## 4. Folder-by-folder inventory

### `final/` (Feb 23 – Mar 23 2026; 44 subfolders plus `detection_fa_tradeoff.png`)
- **comprehensive_eval schema** (`model_config`, `classification_metrics`, `operational_metrics`, `confusion_matrix`, `lead_times`, `event_details`, `threshold_sweep` with θ ∈ {0.1, …, 0.9}). Each folder also has 10 PNGs: roc, pr, score_distribution, confusion_matrix, timeline (first hour), lead_time_histogram, threshold_tradeoff, event_analysis, false_alarm_distribution, summary.
  - Folders: `clean_subset_eval, clean_t02_eval, clean_thresh_0.{2,3,4,5}, eval_diluted_5pct, eval_ucf_crime, evaluate_audio_only, evaluate_fused, evaluate_original, evaluate_vlm_audio, evaluate_vlm_matched, final_evaluation_lstm_2, final_evaluation_optical_flow_fused, final_evaluation_vlm_student, high_sensitivity_eval, high_sensitivity_filtered, lowfa_eval, test_original`.
  - Producer: `sim/comprehensive_eval.py --model_path … --prebuilt_sim simulation_data/<sim>.npz --threshold θ --output_dir final/<name>`. The exact commands are not recorded.
- **Older schema** (`false_alarms_per_hour`, `true_negatives`, …): `final_evaluation_lstm`, `final_evaluation_lstm_violence`. The producer is an earlier version of comprehensive_eval that is **no longer in the repo** (inferred).
- **`meta` schema** (`meta.model_path`, `per_stream_summary`, `false_alarm_episodes`): `final_evaluation_simple`. The producer script is **not in the repo**; no `.py` contains `per_stream_summary`.
- **`results.json` with `config` and `metrics`** (plus `scores.npy` [47,310] for `high_sensitivity`): `high_sensitivity, smoothing_w10_cd60, smoothing_w10_t0.{5,6,75}, smoothing_w15, smoothing_w15_t0.{4,7}, smoothing_w20, smoothing_w5_t0.{6,7}`. Producer: `scripts/evaluate_with_smoothing.py`.
- **Sweep lists** (`sweep_results.json`, 48 dicts over θ ∈ {.2, .3, .4, .5} × w ∈ {1, 3, 5, 10} × cd ∈ {0, 30, 60}, plus `scores.npy`): `smoothing_sweep` (vlm_matched, 47,310 scores) and `clean_smoothing_sweep` (clean_t02, 14,052 scores). Producer: `evaluate_with_smoothing.py --sweep`.
- **`late_fusion_results.json` + `predictions.npz`** (keys `vlm_preds, audio_preds, fused_preds, frame_labels, anticipation_labels, video_boundaries, fps`): `late_fusion_max` (32,092 steps; 132 boundaries borrowed from the audio sim) and `late_fusion_matched` (47,310 steps; 772 boundaries; labels contain −1). Producer: `scripts/late_fusion.py --mode evaluate --fusion_strategy max`.
- `temporal_t0.3/results.json`: produced by `sim/temporal_simulator.py --threshold 0.3`.
- **Empty folders (failed or abandoned runs):** `final_evaluation_student_mlp/`, `late_fusion_optimization/`, `smoothing_w3_t0.3/`.
- `detection_fa_tradeoff.png`: plots the `smoothing_sweep` points (cd=0) and marks θ=0.3, w=10 as "Recommended". Producer script not found **(inferred: ad hoc)**.

### `results/`
- `anticipation_*` (15 folders; Feb 2–23): `train_student_original.py --out_dir results/<name>`. Each contains `best.pt` (StudentMLP state-dict, `LayerNorm → 256 → 256 → K`), `metrics.json`, `per_sample.csv`, `per_video_{val,test}.csv`, ROC/PR/calibration/confusion/threshold-sweep PNGs, and sometimes `video_utility_sweep.csv` and `splits_manifest.json`.
  - `per_sample.csv` columns are `split, vid_id, source, t, gt, pred_prob_raw, pred_prob_ema` (val + test, 30 s column). It is the best raw material for re-scoring.
  - Key `metrics.json` fields: `val/test_metrics_oracle`, `test_metrics_at_val_threshold`, `anticipation_video_stats_at_val_threshold.{val,test}` (first alert in [onset−horizon, onset+grace], early-false, late-or-miss, negative-video FA, lead mean/median/p10/p90 in seconds from metadata), `history`, `split_videos`.
- `vlm_eval*` (4 folders, Feb 2): `scripts/eval_vlm_teacher.py`. Contains `summary.json`, `per_video.csv` (vid_id, source, n_samples, best_threshold_global, f1/precision/recall_at_best_th, roc_auc, pr_auc, mean_score, pos_rate_gt, …) and PNGs.
- `contrastive_results/` (Feb 23): producer **not in the repo**. No `.py` contains `bottleneck_dim`, `dimension_importance` or `projection_model.pt`; it is probably an earlier version of `tuning/contrastive_projection.py` **(inferred)**. args.json: attention architecture, 256-d projection, temperature 0.07, XD only. Contains `projection_model.pt`, `dimension_importance.json` (keys `violence_dims`, `non_violence_dims`, `all_attention_diff`), PNGs, and `projected_emb/xd_violence/<vid>.npz` (**772 files, `emb [E,256] float16`, 23 MB**).
- `xd_investigation/` (Mar 16): `experiments/xd_violence_investigation.py`. Contains `investigation_stats.json` and `best_anticipation_videos.txt` (ranked list, minimum 5 pre-violence frames; top entries Burglary005, Black.Hawk.Down, Explosion043, Shooting046).
- `algorithm_comparison.csv`, `ema_results.csv`, `prompt_engineering.csv`, `ucf_crime_results.csv` (Mar 16): `experiments/run_all_experiments.sh` steps 1–4. `baseline_comparison.csv` (step 6) **was never produced**.
- `lda_calibration_results.txt` (Mar 16): `continuation/lda_calibration_complete.py`.
- `transformer_model.pt` (3.0 MB, Mar 16): `experiments/algorithm_variations.py` saves `{"model": <whole nn.Module>, "scaler_mean", "scaler_scale"}`. Unpickling requires the class definition from that script.

### `simulation_results*/` (Feb 19–23)
- `simulation_results/final/{results.json, simulation_results.png}`: `sim/live_simulator.py` (VLM student, θ=0.6).
- `simulation_results_temporal/thresh_{0.388,0.45,0.55,0.6}/results.json`: `sim/temporal_simulator.py`. The top-level folder has no files of its own.
- `simulation_results_contrastive/{results.json, thresh_0.7/results.json}`: `tuning/contrastive_simulator.py`.
- `simulation_results_ensemble/{and_strategy, average_strategy, weighted_0.7}/results.json`: `tuning/ensemble_simulator.py` (simple student θ=0.6 + temporal θ=0.388).
- All of these ran on `xd_violence_8hr_sim.npz` (inferred from the 32,092 / 13-event counts).

### `outputs/` (Feb 19–24)
- `outputs_temporal{,_old,_old_emb,_violence,_vlm_emb,_fused,_vlm_audio,_audio_only}/` and `outputs_projected/`: `training/train_temporal_model.py`. Each contains `args.json` (embeddings_dir, labels_dir, input_dim, seq 30, stride 5, horizon 30, focal α 0.75 γ 2, source_filter xd_violence, `output_dir: ./outputs_temporal`, etc.), `best_temporal_model.pt` (checkpoint dict with `model_state_dict` and `args`; 4.6–8.6 MB), `final_metrics.json` (`validation`, `test`, `training_history`) and `graphs/` (13 PNGs).
  - **`outputs/outputs_temporal/best_temporal_model.pt` is the default checkpoint for almost every downstream script**: realtime, experiments, curation.
- `student_mlp/`: `train_student_original.py` with the source_movie split.
- `contrastive_v1/`, `outputs_contrastive/`: `contrastive_model.pt` (15 MB; `{model_state_dict, args}`) + `test_projected.npz` (`embeddings [6781,256]`, `labels [6781]`, `scores [6781]`). Producer: `tuning/contrastive_finetune.py` (it saves exactly these two files).
- `contrastive_v2_large/`: `args.json` (mixup, dim_dropout, subsample_negative), `history.json`, `model.pt` (122 MB), curves. No test metrics. Producer: an older version of `tuning/contrastive_projection.py` whose args are no longer present **(inferred)**.
- `contrastive_v3/`: `args.json` (`model_size large`, `contrastive_weight 0.1`), `history.json`, `metrics.json`, `model.pt`, `projected_emb/xd_violence/*.npz` (**772 files, [E,256] float16, 22 MB**). Producer: `tuning/contrastive_projection.py` (current version; saves `model.pt` and `history.json`).
- `qa_feature_test/`: `features.npz` (`violence [50,25]`, `nonviolence [50,25]`), `summary.json`, `analysis.png`. Producer: `tuning/question_analysis/test_qa_features.py` (25 yes/no VLM question features).

### `outputs_fused/` (Feb 23)
`train_temporal_model.py` on `data/processed/fused_vlm_flow` (3606-d). Contains `args.json`, `best_temporal_model.pt`, `final_metrics.json` and `graphs/`.

### `runs/` (Feb 17–18)
- `outputs/`, `outputs_xd/`, `outputs_ucf/`: `training/train_student.py` (VLMStudentModel MLP 512–256–128, focal loss). Each contains `results.json` (val/test, with thresholds chosen by f1/youden/precision_floor/mcc on each split) and `best_model.pt`.
- `student_fix1/`, `student_fix2/`: `train_student_original.py` with the source_movie split. These are the TRUSTED rows.

### `experiments/output/` (Apr 4–5 2026, the latest artifacts)
`experiments/run_experiments.py` with `--exp1` through `--exp4`, using the default model `outputs/outputs_temporal/best_temporal_model.pt`.
- exp1: JSON (+ `per_video` with 173 entries) and 2 PNGs.
- exp2: unified metrics, per-video violent (54) and non-violent (119) score summaries, 8 PNGs.
- exp3: `exp3_all_results.json`, comparison PNG, 3 retrained `best_temporal_model.pt` (7.2 MB each) and 3 × 3 per-target PNG sets.
- exp4: 2 JSONs and 9 PNGs.

`experiments/README.md` describes these, plus `run_all_experiments.sh` (which writes to `results/`).

### `logs/`
- `gpu0_final_scan.log`: `scripts/vlm_label_dataset.py` final pass. It indexed 1798 graphs (kinetics 699 / ucf 3800 / xd 798 raw videos), skipped 1721 existing, processed 0, 0 errors. **It confirms that VLM labels and embeddings exist for 1721 videos.**
- `audio_gpu_{0..7}.log`: `scripts/audio_feature_extraction.py` (inferred from the "Feature dimension: 91" output), 8 shards of 100 videos out of 798, with 83–84 processed and 16–17 skipped per shard.
- `lora_train_v2.log` / `lora_train_v2_full.log`: LoRA failure (see §3H).
- `inference_20260226_151941.log`: `realtime/rtsp_inference.py` could not open the RTSP stream; 0 alerts.

### Root-level logs and files
- `inference_20260226_225311.log`: `rtsp_inference.py` on `rtsp://127.0.0.1:8554/macbook` for about 15 min. 825 LSTM inferences, scores mostly 0.17–0.21, **1 alert** (score 0.5044). No ground truth.
- `inference_20260302_185137.log`: about 2 min, 75 inferences, scores 0.21–0.54, 1 alert (0.5059). No ground truth.
- These are **live demos without labels and carry no accuracy information**. Both used `sequence_length 32`, 1 fps, θ 0.5 and a 30 s cooldown.
- `snapshots/`: empty (alert snapshot directory of `rtsp_inference.py --snapshot_dir`).
- `good_video_indices.npy`: int64 [433], indices into the 772-video list (Feb 25). Related to the lowfa/clean subsets (inferred).
- `probe_clip_01_*`: see §3H.

### Root `*.pt` (checked by listing pickle keys only; no tensors loaded)
None of these has a producer script in the repo, recorded metrics, or metadata. They are bare `state_dict`s of a `nn.Sequential` called `net`, with parameterized layers at indices 0/3/6(/9). File sizes imply Linear → activation → Dropout blocks starting with a Linear from 3584-d **(inferred)**. This layout is **not** the `StudentMLP` of `train_student_original.py` or `comprehensive_eval.py`, which starts with a LayerNorm at `net.0`. They were saved 2026-03-02 between 10:33 and 13:35, probably from an interactive session **(inferred)**.

| File | Size | Layers | Likely hidden width (inferred from size, assuming 3584-d input) |
|---|---|---|---|
| `simple_mlp_model.pt` | 3.74 MB | 3 Linear | 256 |
| `regularized_mlp.pt` | 0.92 MB | 2 Linear | 64 |
| `full_dataset_mlp.pt` | 1.86 MB | 3 Linear | 128 |
| `ucf_crime_mlp.pt` | 3.74 MB | 3 Linear | 256 |
| `anticipation_mlp.pt` | 3.74 MB | 3 Linear | 256 |
| `full_anticipation_mlp.pt` | 1.86 MB | 3 Linear | 128 |
| `backward_anticipation_mlp.pt` | 1.86 MB | 3 Linear | 128 |
| `filtered_anticipation_mlp.pt` | 3.74 MB | 3 Linear | 256 |
| `synthetic_anticipation_mlp.pt` | 3.74 MB | 3 Linear | 256 |
| `best_synthetic_anticipation.pt` | 7.63 MB | 4 Linear | 512 |
| `yolov8n.pt`, `yolov8m-pose.pt` | 6.5 / 53 MB | — | Ultralytics pretrained weights (detector and pose); used by the pose and person-detection scripts |

**Treat these as orphaned. They cannot be evaluated or reproduced from the repo.**

### `best_checkpoints/` (Dec 15 2025) and `checkpoints/` (Dec 18 2025)
`model_epoch{1..40}.pt` and `model_epoch{1..60}.pt`, 11.9 MB each. Written by `training/train.py` (`checkpoints/model_epoch{N}.pt`): the original Graph Transformer + Temporal Transformer from `README.md` / `configs/training.yaml` (60 epochs). **No metrics files survive for these runs.** `training/evaluate.py --checkpoint` can score them (not run here).

---

## 5. Strongest, most trustworthy and suspicious, in one place

- **Strongest reported:**
  - `final/clean_smoothing_sweep`: AUC 0.985 / AP 0.992 / 199 of 200 / 0.0 FA/h / 29.2 s.
  - `outputs/outputs_temporal`: test AUC 0.991.
  - `results/anticipation_prboost_v1`: test AUC 0.942, XD 0.967.
  - The project's own "recommended" point (`final/detection_fa_tradeoff.png`): 91.9 % detection, 4.87 FA/h, about 24 s lead.
  - **All of these are invalid** (C1–C5).
- **Most trustworthy:** `runs/student_fix1` and its siblings on the source_movie split. Test AUC 0.58–0.62 and AP ≤ 0.16 (base 0.081); XD-only test is at chance. Video-level: about 60 % of violent test videos alert within 30 s, with 15–25 % of normal videos false-alarming.
- **Suspicious numbers, with reasons:**
  1. `final/clean_*`, `final/lowfa_eval`: 81 % training videos, 63–70 % positives, normal clips removed.
  2. `outputs/*/final_metrics.json`: test AUC 0.89–0.99 against val 0.53–0.72.
  3. `final/final_evaluation_lstm_2`: always-on alarm (R = 1.0, Acc = 0.018) that still scores 13/13, 1.46 FA/h and 30 s lead.
  4. `final/late_fusion_*`: audio not time-aligned; always-on fused alarm at 0.08–0.11 FA/h; AUC 0.5 / AP 0.0 fallbacks.
  5. Any "30 s lead": capped by construction and often inherited from an alarm that was already on.
  6. `final/eval_diluted_5pct`: AUC 0.945 with lead 0 on every event; negatives are tiled, noised frames, and pre-event context comes from unrelated footage.
  7. `results/prompt_engineering.csv` LDA AUC 1.0.
  8. `results/anticipation_preonly_*`: val at chance, threshold always positive.
  9. `results/anticipation_multih_hset_v3`: threshold 1.0, so no alerts.
  10. `results/algorithm_comparison.csv`: mixed metric definitions, UCF-only validation.
  11. `cd=30/60` rows in the smoothing sweeps raise FA/h, a counting artifact.
  12. `simulation_results_temporal/thresh_0.388` versus the other thresholds: different weights (C7).

---

## 6. Baseline a new model must beat, and how to measure it

**Primary honest baseline (beat all of these):**
- **Protocol:** `training/train_student_original.py --task anticipation --labels_dir data/processed/labels_multih --emb_root data/processed/vlm_emb --split_group source_movie --seed 0 --val_frac 0.1 --test_frac 0.1`. The split is the one recorded in `runs/student_fix1/splits_manifest.json` (groups 1072/134/134, videos 1356/156/158). Target = "violence starts within 30 s" (column 4 of `labels_multih`); in-violence steps are excluded.
- **Frame-level test:** ROC-AUC **0.622** (best, `outputs/student_mlp`), AP **0.159** (best, `runs/student_fix1`), base rate 0.081. XD-only test AUC **0.56** (best).
- **≥30 s anticipation proxy:** AUC for steps 30–60 s before onset versus steps with no violence within 120 s is **0.62** on test (best, `prboost_v3`) and about 0.50 on XD only.
- **Video-level at val threshold** (`runs/student_fix1`): 24/40 = **60 %** of violent test videos alerted within [onset − 30 s, onset + 1 s]; **15.3 %** of normal test videos with any false alarm; lead mean **10.0 s**, median 7.8 s, p90 20.7 s. Lead is ≤ 31 s by construction, so **detection at ≥30 s lead is effectively 0 %**.

**No-training reference:** the zero-shot Qwen2.5-VL-7B teacher score reaches XD frame AUC **0.77–0.78** (`results/vlm_eval*`). Its target is detection-ish (current violence or within 30 s), but no training is involved. A trained anticipation model that cannot beat this on a leak-free XD split adds nothing.

**Legacy comparison only:** on `simulation_data/xd_violence_8hr_sim.npz` with `sim/comprehensive_eval.py`, the bar is:
- AUC **0.853** / AP **0.242** (VLM student `runs/outputs_xd/best_model.pt`).
- For the Temporal LSTM, AUC 0.834 / AP 0.131; at θ = 0.388: 11/13 events, 8.6 FA/h, 11.4 "s" mean lead, 4/13 events at the capped 30 lead.

This set has C1/C2/C3/C5 problems, so use it only to show continuity with old reports.

**Recommended protocol for the next model** (none of the current artifacts satisfies it):
1. Labels indexed by true timestamps: use `t` / `raw_frame_idx` and the metadata fps, never `labels[:len(emb)]`.
2. Grouped K-fold over movie or YouTube source, with mean ± std reported.
3. A threshold chosen on validation only.
4. Report AUC and AP per time-to-onset bucket (0–30 s, 30–60 s, 60–120 s) against true negatives.
5. Event-level metrics on **untrimmed long normal footage** that is not stitched from training clips, with a causal-only alarm. Report detection at ≥30 s lead (lead uncapped, measured in real seconds), false alarms per hour, and the share of events first detected only after onset.

---

## 7. Reusable assets

| Asset | Path | Shape / size | Notes |
|---|---|---|---|
| Re-scorable predictions with GT (val + test) | `runs/student_fix1/per_sample.csv` (12,997 rows), `runs/student_fix2/per_sample.csv`, `outputs/student_mlp/per_sample.csv`, `results/anticipation_*/per_sample.csv` | CSV `split, vid_id, source, t, gt, pred_prob_raw, pred_prob_ema` | `t` indexes `labels_multih` rows, so it can be joined with any horizon column. This is how the bucket AUCs above were computed. |
| Leak-free split manifest | `runs/student_fix1/splits_manifest.json` | JSON with `groups.{train,val,test}` (≤ 5000 names each) | `source_movie` grouping; copy it for new experiments |
| Student MLP weights | `runs/student_fix1/best.pt`, `outputs/student_mlp/best.pt`, `results/anticipation_*/best.pt` | StudentMLP (`LayerNorm(in) → Linear 256 → GELU → Dropout → Linear 256 → GELU → Dropout → Linear K`); K = 7 for multi-horizon runs, 1 for `preonly` runs | `in_dim` 3744 / 3746 / 4034 |
| Temporal LSTM checkpoints | `outputs/outputs_temporal*/best_temporal_model.pt`, `outputs_fused/best_temporal_model.pt`, `experiments/output/exp3_cross_dataset/*/best_temporal_model.pt` | dict `{model_state_dict, args, …}` | Trained with C1 labels; reuse with caution |
| VLM student | `runs/outputs_xd/best_model.pt` etc. | VLMStudentModel | C1 labels |
| Contrastive projected embeddings | `results/contrastive_results/projected_emb/xd_violence/*.npz`, `outputs/contrastive_v3/projected_emb/xd_violence/*.npz` | 772 files each, `emb [E,256] float16`, about 22 MB | Projection heads were trained on the train index only, but with C1 labels |
| Test-set projected embeddings | `outputs/contrastive_v1/test_projected.npz` | `embeddings [6781,256]`, `labels`, `scores` | |
| Saved score traces | `final/high_sensitivity/scores.npy` [47,310], `final/smoothing_sweep/scores.npy` [47,310], `final/clean_smoothing_sweep/scores.npy` [14,052], `final/late_fusion_*/predictions.npz` | float32 | Align with the corresponding `simulation_data/*.npz` |
| Real-world probe clip | `probe_clip_01_embeddings.npy` [197,3584] float32, `probe_clip_01_scores.npy` [197] | | Hand ground truth in `YT_LABELS` of `continuation/lda_calibration_complete.py`: 0–6 s safe, 7–36 s anticipation, 37–45 s violence, 46 s+ post |
| Zero-shot teacher per-video scores | `results/vlm_eval*/per_video.csv` | CSV | Baseline reference |
| Pretrained YOLO | `yolov8n.pt`, `yolov8m-pose.pt` | 6.5 / 53 MB | For person and pose features |

---

## 8. Regenerating the comparison numbers

All metrics in §3 can be re-derived with the venv Python and no GPU. Two snippets:

```python
# 1) Normalize every final/*/metrics.json (3 schemas)
import json, glob
for f in sorted(glob.glob('final/*/metrics.json')):
    m = json.load(open(f)); c = m['classification_metrics']; o = m['operational_metrics']
    cfg = m.get('model_config') or m.get('meta'); cm = m['confusion_matrix']
    fa = o.get('fa_per_hour', o.get('false_alarms_per_hour'))
    lt = m.get('lead_times', m.get('lead_times_sec', []))
    print(f, cfg.get('threshold'), round(c['roc_auc'], 4), round(c['pr_auc'], 4),
          o.get('detected_events'), o.get('total_violence_events', o.get('total_events')),
          round(fa, 2), round(o['avg_lead_time_sec'], 1), sum(x >= 30 for x in lt))

# 2) Per-source / per-bucket AUC from a student run
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
df = pd.read_csv('runs/student_fix1/per_sample.csv'); s = df[df['split'] == 'test']
Y = np.stack([np.load(f'data/processed/labels_multih/{v}.npy')[t] for v, t in zip(s['vid_id'], s['t'])])
pos = (Y[:, 5] > 0) & (Y[:, 4] <= 0)   # 30-60 s before onset
neg = (Y[:, 6] == 0)                   # no violence within 120 s
print(roc_auc_score(np.r_[np.ones(pos.sum()), np.zeros(neg.sum())],
                    np.r_[s['pred_prob_raw'][pos], s['pred_prob_raw'][neg]]))
```
