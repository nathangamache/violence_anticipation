# Simulation and Evaluation Protocol

> Paths in this file describe the tree before the 2026-10 cleanup. Surviving code is under `legacy/<same path>`;
> the honest runs are under `baselines/reference_runs/`; removed tracked files: `git show pre-cleanup:<path>`;
> ignored weights and arrays: `data/archive/repo_legacy/<same path>`
> until 2027-01-12. See [../REPO_CLEANUP_PLAN.md](../REPO_CLEANUP_PLAN.md).

Scope: `sim/*.py`, `continuation/*.py`, `scripts/evaluate_with_smoothing.py`, `experiments/*.py`,
`experiments/README.md`, `experiments/run_all_experiments.sh`.

Audience: future agents and developers who need to (a) understand how every past number in this repo
was produced and (b) build a **correct** benchmark for a new violence-anticipation model.

All paths are relative to the repo root `.`
(`data -> data/`). Claims were checked by reading the code and
by cheap read-only inspection of the data (header reads, label counts, and exact replication of how two
simulation files were built). Anything not directly verified is marked **(inferred)**.

---

## 0. TL;DR

- A "simulation" here means: concatenate per-video, per-step embedding sequences (Qwen2.5-VL 3584-d
  embeddings) into one long stream (sold as an "8-hour security shift"), attach a per-step label, run a
  model over the stream, threshold the scores, and count (a) "detected events", (b) "false alarms per
  hour", and (c) "lead time". It also reports frame-level ROC-AUC, PR-AUC (AP), precision, recall and F1.
- **None of the historical numbers measure 30-second violence anticipation.** Five independent defects
  apply to all of them:
  1. **The label transform is wrong.** `data/processed/labels_multih/*.npy` holds multi-horizon
     *anticipation* labels with shape `[T, 7]` (horizons 2/5/10/20/30/60/120 s). Rows *inside* violence
     are `-1`. Every script here does `labels.max(axis=1) > 0`. That makes "positive" mean "violence
     starts within **120 s** and is not happening yet", and it marks **actual violence as negative**.
  2. **Embeddings and labels are misaligned in time.** Embedding `i` was sampled at label row
     `t[i] = 5*i` (the `t` key in each embedding `.npz`), but every script pairs it with label row `i`
     (`labels[:min(len(emb), len(labels))]`).
  3. **"1 step = 1 second" is false.** The real gap between embeddings varies by video. The median is
     about 1.5 s for XD and 3.3 s for UCF, with a range of about 0.2 s to 66 s. So "30 frames", "FA/hour"
     and "lead time (s)" are all in embedding-step units, not seconds.
  4. **Leakage and tuning on test.** Most simulation files contain **training** videos. For example,
     `xd_violence_vlm_matched.npz` has 626 of its 772 videos from `train_index.json`, and
     `ucf_crime_8hr_sim.npz` has 703 of 884. Thresholds, smoothing and cooldown were swept and chosen on
     the same stream they are reported on.
  5. **The event and lead-time scoring is lenient and mis-anchored.** An alarm *during* violence counts
     as a detection. A constantly-on alarm gets 100 % detection at about 1 FA per negative gap. Lead time
     is measured from the start of the label window, not from the violence onset, and is capped at 30.
- Because XD-Violence clips are short, violence onset is almost always within seconds of the clip start
  (median onset 5.9 s in XD test). With correct labels, **only 12 test videos** (4 XD + 8 UCF) have at
  least 30 s of in-video context before violence. Across all splits there are 117 such videos (XD 63,
  UCF 54). In the concatenated simulations, the 30 "anticipation" steps before most events come from an
  **unrelated** preceding clip.
- **What to reuse for a new benchmark:** the event logic in `scripts/evaluate_with_smoothing.py`
  (`apply_smoothing`, `apply_cooldown`, `find_violence_events`, `evaluate`). It anchors events on the
  violence onset and uses causal k-consecutive smoothing. It must be fixed as described in §7. Do **not**
  reuse any existing `simulation_data/*.npz` file as a benchmark. Rebuild the test set from per-video
  files with correct labels and real timestamps (§7).

---

## 1. Data contracts used by this area

### 1.1 Per-video embeddings: `data/processed/vlm_emb/{xd_violence,ucf_crime,curated}/<vid>.npz`
- Keys: `emb` `[N, 3584] float16`, `t` `[N] int32`, `raw_frame_idx` `[N] int32`.
  - Example: `A.Beautiful.Mind.2001__#00-25-20_00-29-20_label_A` has `N=37` and `t = 0,5,10,…,180`.
- `t` indexes the label timeline (graph steps). `raw_frame_idx` is the video frame number that was
  embedded (`scripts/vlm_label_dataset.py`, `every_t=5`,
  `raw = round((t+0.5)/T * total_frames)`).
- **The real timestamp of embedding i is `raw_frame_idx[i] / fps`**, with `fps` taken from
  `data/metadata/{xd_violence,ucf_crime}.json`. Example: frames 16, 175, 334, … at 24 fps are
  0.67 s, 7.29 s, 13.92 s, … apart, so 6.6 s per step for that clip.
- Counts: `ucf_crime` 949, `xd_violence` 772, `curated` 92 (curated has no labels in `labels_multih`).

### 1.2 Labels: `data/processed/labels_multih/<vid>.npy`
- Shape `[T, 7] float32`. Built by `scripts/build_multihorizon_labels.py` with
  `--ignore_inside_segments` (see `labels_multih/labels_info.json`).
- `T` = graph length. Row `r` maps to time `round((r+0.5)/T*total_frames)/fps`.
- Column `k` = 1 if the **next violence start** is within `horizons_sec[k]` ∈
  {2, 5, 10, 20, 30, 60, 120} s, and 0 otherwise.
- **Row = -1 (all columns) when the row lies inside a violence segment.**
- Example row-max pattern for a test clip (`p` = 1, `V` = -1, `.` = 0):
  `Braveheart…label_B1-0-0: ppppppppppppppppppppppppppp VVVVV…VVVV .........`
- Correct per-row signals, which no script in this area uses:
  - violence active: `(L == -1).all(1)`
  - "onset within 30 s": `L[:, 4] == 1`
- Coverage caveats, computed from the data:
  - UCF-Crime: only videos with official *temporal* annotations get positives.
    775 UCF **anomaly** videos (Abuse, Fighting, …) have all-zero labels. Every simulation treats them
    as "normal" footage.
  - Only 48 UCF `Normal_*` videos are labelled.
  - XD: 772 clips. Violent clips usually start at or within seconds of the violence (median onset
    5.9 s on test).

### 1.3 Split files
- `train_index.json` (1438), `val_index.json` (180), `test_index.json` (180). `eval_index.json` is
  identical to `test_index.json`.
- Items are `{"graph_path": ..., "label_path": ...}`. The video id is `Path(graph_path).stem`.
- There is no overlap between splits.
- Mix: train is 793 UCF + 645 XD, val is 110 UCF + 70 XD, test is 97 UCF + 83 XD.

### 1.4 Simulation NPZ format (`simulation_data/*.npz`)
- Common keys:
  - `embeddings [N, D] float32`
  - `anticipation_labels [N]`
  - `fps` (scalar, or `[1]` array)
  - optional `frame_labels [N]`, `horizon_sec`
  - newer files also have `video_boundaries [V, 2]` and `video_names [V]`
- Readers (`comprehensive_eval`, `live_simulator`, `temporal_simulator`) need `embeddings` and
  `anticipation_labels`. `evaluate_with_smoothing.py` **also needs `frame_labels`**, so it crashes on
  `ucf_crime_8hr_sim.npz` and `xd_diluted_5pct_8hr.npz`, which lack that key.

### 1.5 Inventory of `simulation_data/` (3.5 GB), with provenance and split membership

Split membership is computed from `video_names` against the index files. For files without
`video_names`, the build was replicated exactly in Python.

| File | N steps | D | Producer | Videos / split membership | Notes |
|---|---|---|---|---|---|
| `xd_violence_8hr_sim.npz` (+`.meta.json`) | 32,092 | 3584 | `sim/build_long_simulation.py` defaults (test_index, XD only, `vlm_emb`, `labels_multih`, seed 42); **replicated bit-exactly** | test only (XD) | 13 of 39 "violent" test clips used. The 40 "non-violent" clips (5,526 unique steps) are **cycled about 5.7×** as padding, and 7 of them contain real violence. 18 positive onsets; 12 have 0 in-clip context. Anticipation positives: 557 (1.74 %). |
| `xd_violence_8hr_sim_violence.npz` | 32,092 | 3584 | same builder, (inferred) `--embeddings_dir data/processed/vlm_emb_violence` | same as above | identical labels |
| `xd_violence_8hr_flow_sim.npz` / `_fused.npz` | 32,092 | 22 / 3606 | `sim/build_flow_sim.py` | n/a | **flow rows are not aligned to the VLM rows** (§4.5) |
| `xd_violence_8hr_fused.npz` | 47,310 | 4124 | `scripts/build_fused_features.py` (VLM+pose) | 772 XD: **626 train**, 67 val, 79 test | per-video anticipation labels |
| `xd_violence_vlm_matched.npz` | 47,310 | 3584 | not in this area; (inferred) `scripts/multimodal_fusion.py` (all labelled videos) | **626 train**, 67 val, 79 test | default sim for `evaluate_with_smoothing.py` |
| `xd_violence_vlm_audio.npz` | 47,310 | 3675 | (inferred) `scripts/multimodal_fusion.py` | 626/67/79 | |
| `xd_violence_audio_only.npz` | 234,751 | 91 | (inferred) `scripts/multimodal_fusion.py` | 626/67/79 | different step rate |
| `xd_violence_vlm_filtered.npz` | 41,737 | 3584 | unknown script (not committed) | 752 videos: 610 train | |
| `xd_violence_vlm_clean.npz` | 16,070 | 3584 | unknown | 431 videos: 351 train | |
| `xd_violence_vlm_clean_t02.npz` | 14,052 | 3584 | unknown | 394 videos: **323 train** | 69.5 % positive. `anticipation_labels` ≠ max over a 30-step window in 182 steps. |
| `xd_violence_vlm_lowfa.npz` | 14,845 | 3584 | unknown | 433 videos: 357 train | |
| `ucf_crime_8hr_sim.npz` | 28,800 | 3584 | `sim/create_ucf_crime_simulation.py` over **all** `vlm_emb/ucf_crime` (replicated bit-exactly) | 884 videos in the first 8 h: **703 train**, 96 val, 85 test | `anticipation_labels` = flawed frame label (no 30-step extension); 10.2 % positive |
| `xd_diluted_5pct_8hr.npz` | 28,790 | 3584 | `sim/create_diluted_simulation.py` from `xd_violence_vlm_clean_t02.npz` | inherits clean_t02 (mostly train) | only 5 positive segments (split bug, §4.4); negatives tiled about 6× plus noise |

---

## 2. What a "simulation" is: the builders

| Builder | Stream construction | Label construction | Cross-video label leakage? |
|---|---|---|---|
| `sim/build_long_simulation.py` | Alternates `uniform(0.5,1.5) × avg_padding` steps of whole "non-violent" clips with one "violent" clip. Stops when ≥ target steps are reached **or** violent clips run out. Non-violent clips are cycled and reshuffled when exhausted. | Per-clip `labels.max(1)>0` (flawed). `anticipation = max(frame_labels[t:t+30])` computed **over the whole concatenated stream**. | **Yes.** The 29 steps before a clip's first positive come from the previous unrelated clip. |
| `sim/live_simulator.py --synthetic` | Per-video anticipation labels computed first. Videos shuffled at video level. Non-violent videos sampled or oversampled (with replacement) until `dilute_ratio × (frames of violent videos)`. **Unseeded RNG.** | per-video, flawed base labels | No |
| `sim/create_ucf_crime_simulation.py` | 2–3 normal videos then 1 "violent" video, repeated. Trimmed (or tiled) to 8 h. | flawed frame label stored as `anticipation_labels` (no horizon extension) | n/a (no extension) |
| `sim/create_diluted_simulation.py` | Takes the positive steps of an existing sim (proportional prefix of each run) and interleaves them with negative steps that are randomly sampled (frame order destroyed) or tiled with N(0, 0.01) noise. | binary copy of the source labels | positives cut out of context |
| `experiments/run_experiments.py::build_simulation` | Like `build_long_simulation` but loops over **all** violent clips (no stop at target), so exp2 was 15.9 h. Non-violent clips cycled with `nv_idx % len`. | stream-level 30-step max (flawed base) | **Yes** |
| `sim/build_flow_sim.py` | Concatenates **every** optical-flow `.npz` under `--flow_dir` in sorted path order. Truncates or zero-pads to the VLM sim length. | copies the VLM sim labels | features misaligned |

Interpretation problems shared by all builders:

- Each "frame" is one embedding step (§1.1), not one second.
- A "violent event" is a run of the flawed positive label, i.e. the up-to-120-s window before violence,
  shifted by the 5× index misalignment. It is not the violence itself.
- Under the flawed rule, XD clips that start inside violence (like `v=2al4t5inObA__#1_label_B4-0-0`,
  whose rows are all `V`) have **no positive label**. They are classified as non-violent and used as
  padding negatives.

---

## 3. Metrics as implemented

### 3.1 Frame-level (all evaluators)
- `roc_auc_score(anticipation_labels, scores)` and `average_precision_score(...)` pooled over the
  stream.
- Precision, recall and F1 from the confusion matrix at one threshold. This is the "per-window F1". It is
  frame-level against the (flawed) anticipation labels.
- `comprehensive_eval` also reports accuracy.
- The temporal (LSTM) model's first `seq_len-1=29` scores are handled differently in each script:
  - `comprehensive_eval.py` sets them to 0.
  - `temporal_simulator.py` skips them, so its first prediction is at step 30.
  - `run_experiments.py`, `exp1_person_detection.py` and `fast_analyze_curation.py`
    **back-fill them with the first full-window score**, which comes from later frames.
  - `evaluate_with_smoothing.py` zero-pads the inputs.

### 3.2 Events, detection and lead time

| Script/function | Event definition | Detection window | Lead time | Notes |
|---|---|---|---|---|
| `sim/live_simulator.py::_compute_results`, `sim/temporal_simulator.py::_compute_results` | runs of `anticipation_labels==1` | `[start-30, end]` (steps) | `(start - first_alarm)/fps`, appended **only if > 0** | Lead is measured **before the anticipation window**, not before the onset. Alarms inside the window get lead 0 and are excluded from the mean, so the mean can be 0 (e.g. `simulation_results_temporal/thresh_0.388`). |
| `sim/comprehensive_eval.py::_analyze_events`, `experiments/run_experiments.py::_analyze_events` | runs of `anticipation_labels==1` | `[start-30, end]` | `max(0, (start - i)/fps)` | Same mis-anchoring, but zeros are kept. Max is 30 by construction. |
| `scripts/evaluate_with_smoothing.py::evaluate` | runs of **`frame_labels`** (`find_violence_events`) | `[start-horizon, end]` | `max(0, start - first_alarm)/fps` | Anchored on the "onset" of `frame_labels`. Closest to correct (§7). |

All of them give **detection credit for alarms that fire only during the event** (lead 0), so "detection
rate" is not anticipation success.

### 3.3 False alarms and FA/hour
- `comprehensive_eval._find_false_alarm_times` and `run_experiments._find_false_alarms`:
  - Count a new FA when `pred==1 & label==0` and not already in an FA run.
  - The run resets on `pred==0` **or `label==1`**.
  - Alarms in `[start-30, start)` count as **both** an FA and a detection.
  - `fa_per_hour = n_fa / (N_steps / fps / 3600)`.
- `live_simulator` / `temporal_simulator`: same, but the run resets only on `pred==0`.
- `evaluate_with_smoothing.evaluate`: counts rising edges of the alarm that start where
  `anticipation_labels==0`.
- `experiments/baseline_comparisons.py`: `fa_per_10min` = above-0.5 runs in all-negative videos per 10
  minutes, with a fixed threshold of 0.5.
- `experiments/ucf_crime_testing.py::false_alarm_analysis`: **broken**. It compares *per-video AUC* to
  0.5, and normal videos have AUC = NaN, so the FA count is always empty or 0.
- `experiments/exp1_person_detection.py`: `fa_per_hour = n_fa / (len(labels)/3600)`, assuming 1 fps.

### 3.4 Thresholds

| Where | How the threshold is chosen | On what data |
|---|---|---|
| `live_simulator --synthetic` without `--threshold` | max recall at precision ≥ `--target_precision` (0.5), else best F1 | **the test videos it reports on** |
| `temporal_simulator` without `--threshold` | threshold so that `target_fa_per_hour × hours` *negative windows* (stride 5) exceed it. This is a frame count, not an event count. | **the evaluation sim** |
| `comprehensive_eval --threshold` | fixed, default 0.388 (origin unrecorded; the main LSTM's val best-F1 threshold was 0.373). Also runs a sweep over 0.1…0.9. | sweep is on the evaluation sim |
| `evaluate_with_smoothing --sweep` | grid θ∈{0.2,0.3,0.4,0.5} × window∈{1,3,5,10} × cooldown∈{0,30,60}. Prints the best config under FA<5/h, FA<1/h and det>90 %. | **the evaluation sim** |
| `run_experiments.evaluate_simulation(threshold=None)`, `exp1_person_detection` | best F1 on the PR curve | **the test sim** |
| `sim/sweep_thresholds.py` | runs `live_simulator` at θ∈{0.3…0.9} | test |

### 3.5 Smoothing variants
- k-consecutive. Causal, and the correct form:
  - `live_simulator`/`temporal_simulator` `--smoothing_window N`: all of the last N scores ≥ θ.
  - `evaluate_with_smoothing.apply_smoothing`: the same as a counter.
- Cooldown:
  - `live_simulator --alert_cooldown`.
  - `evaluate_with_smoothing.apply_cooldown`: turns sustained alarms into one-step pulses every
    `cooldown+1` steps. With rising-edge FA counting, one long false alarm then counts as several FAs.
    It can also suppress the alarm inside an event window.
- `comprehensive_eval --smoothing_window`: `np.convolve(..., mode='same')`. This is a **centered,
  non-causal** moving average that uses future scores.
- EMA:
  - causal `ema_1d` in `experiments/ema_smoothing.py` and `baseline_comparisons.py`
  - `continuation/time_to_violence.py` (α=0.3)
  - (cross-reference) `results/anticipation_multih_hset_v4_ema085_k2_gap5_fap7*` come from another
    pipeline, `training/eval_anticipation.py`. That pipeline is documented elsewhere.

---

## 4. File-by-file reference

### 4.1 `sim/build_long_simulation.py` (443 lines)
- **Purpose:** builds the "8-hour shift" stream from `--test_index` clips (§2).
- **Run:**

  ```
  python sim/build_long_simulation.py --embeddings_dir data/processed/vlm_emb \
    --labels_dir data/processed/labels_multih --test_index test_index.json \
    --source_filter xd_violence --duration_hours 8 --violence_pct 1 \
    --horizon_sec 30 --fps 1 --seed 42 --output simulation_data/xd_violence_8hr_sim.npz
  ```

  Default `--output` is `./simulation_data/long_form_sim.npz`.
- **Output:**
  - `.npz` with `embeddings, frame_labels, anticipation_labels, fps, horizon_sec`
  - `.meta.json` with counts
  - Per-frame metadata (`video_id`, `frame_idx`) is built in memory but **not saved**. The produced
    file therefore has no `video_boundaries`, and boundaries cannot be recovered without replaying the
    build.
- **Helpers duplicated in many files:** `classify_video_source` (`_x264` → UCF; `v=` or `__#` → XD),
  `extract_video_name`, `find_embedding_file`, `find_label_file`, `load_embedding` (first of
  `emb|embedding|embeddings|features|arr_0`), `load_labels`.
- **Bugs:**
  - flawed label transform and index misalignment (§0)
  - anticipation labels cross video boundaries
  - padding added in whole clips, so durations overshoot
  - non-violent clips heavily duplicated
  - loop stops at the target, so only 13 of 39 violent clips are used

### 4.2 `sim/live_simulator.py` (952 lines)
- **Purpose:** frame-by-frame "live" replay for a **single-frame** `VLMStudentModel` (MLP
  3584→512→256→128→64→1, BatchNorm+GELU). Checkpoint keys: `model_state_dict`, `args`.
- **Modes:**
  - `--prebuilt_sim <npz>`: replay a built stream.
  - `--synthetic`: build a diluted stream in memory from `--test_index` (§2). If `--threshold` is
    omitted, the threshold is auto-tuned on the same test data.
- **Run:**

  ```
  python sim/live_simulator.py --model_path runs/outputs_xd/best_model.pt \
    --prebuilt_sim simulation_data/xd_violence_8hr_sim.npz --threshold 0.6 --output_dir simulation_results/final
  ```

  Default device is `cuda:1`.
- **Output:** `results.json` (threshold, roc_auc, pr_auc, precision, recall, f1, confusion_matrix,
  n_violence_events, n_detected_events, n_false_alarms, avg_lead_time_sec, total_duration_sec) and
  `simulation_results.png`.
- **Notes:**
  - It always builds `VLMStudentModel` with `use_temporal=False`, so it cannot load the LSTM.
  - `threshold=args.threshold or 0.5` turns an explicit 0.0 into 0.5.
  - One-sample-at-a-time GPU inference is slow.
  - `total_duration_sec = timestamps[-1]`.

### 4.3 `sim/temporal_simulator.py` (490 lines)
- **Purpose:** the same replay for `TemporalViolenceModel`:
  - frame encoder Linear→LN→GELU→Dropout ×2
  - 2-layer BiLSTM(128)
  - attention pooling
  - MLP head
- Input is a sliding window of 30 steps, labelled with `anticipation_labels[end-1]`.
- **Run:**

  ```
  python sim/temporal_simulator.py --model_path outputs/outputs_temporal/best_temporal_model.pt \
    --prebuilt_sim simulation_data/xd_violence_8hr_sim.npz [--threshold 0.388 | --target_fa_per_hour 0.5] \
    --output_dir simulation_results_temporal/thresh_0.388
  ```

- `sequence_length` is hard-coded to 30 and not read from the checkpoint.
- The horizon in the event logic is hard-coded to 30.

### 4.4 `sim/create_diluted_simulation.py` (321 lines)
- **Purpose:** re-mix an existing sim to a target positive rate (default 5 %, 8 h).
- **Run:**

  ```
  python sim/create_diluted_simulation.py --violence_sim simulation_data/xd_violence_vlm_clean_t02.npz \
    --output simulation_data/xd_diluted_5pct_8hr.npz --target_anomaly_pct 5 --total_duration_hours 8 \
    [--normal_emb_dir DIR]
  ```

- **Bugs:**
  - `split_into_events(violent_emb, violent_labels, violent_indices)` passes the **full** positive
    index list instead of `selected_violent`. The selected 1,440 positive steps are therefore cut using
    the original segment lengths, which leaves only 5 non-empty "events" (verified: 5 positive segments,
    mean 286 steps).
  - Negatives are sampled frame-by-frame without replacement (temporal order destroyed), or, as here,
    **tiled about 6× with N(0, 0.01) noise**. The result is trivially repetitive negatives.
  - The positives are the source's anticipation-positive steps, removed from their context.
  - `fps` is saved as a `[1]` array. `metadata` is a JSON string.

### 4.5 `sim/build_flow_sim.py` (158 lines)
- **Purpose:** make an optical-flow stream "matching" a VLM sim, plus a pre-fused copy (VLM ‖ z-scored
  flow).
- **Fatal bug:** it ignores `video_info`/`video_names`, which it reads but never uses. It concatenates
  all flow files in sorted order and truncates or zero-pads them to the VLM length. Rows do not
  correspond to the same video or time.
- The resulting `final/final_evaluation_optical_flow_fused` score (AUC 0.557) is consistent with noise
  features **(inferred)**.

### 4.6 `sim/create_ucf_crime_simulation.py` (226 lines)
- **Run:**

  ```
  python sim/create_ucf_crime_simulation.py --vlm_emb_dir data/processed/vlm_emb/ucf_crime \
    --labels_dir data/processed/labels_multih --output simulation_data/ucf_crime_8hr_sim.npz --duration_hours 8
  ```

- Uses every video in the directory. There is **no split filtering**: 703 of the 884 videos in the
  saved 8 h are train videos.
- "Violent" means it has a flawed-rule positive, which applies to only 116 of the 949 videos.
- The 775 unannotated anomaly videos act as negatives.

### 4.7 `sim/comprehensive_eval.py` (789 lines): the main plotting evaluator behind `final/*`
- Loads one of three architectures, auto-detected from state-dict keys (`lstm` → temporal,
  `feature_extractor` → vlm_student, `net.` → student_mlp):
  - `TemporalViolenceModel`
  - `VLMStudentModel`
  - `StudentMLP`
- Scores in batches (temporal: sliding 30-window).
- Optional `--optical_flow_sim` fusion: z-scores the flow with the whole stream's statistics
  (transductive).
- Computes the metrics of §3, then a threshold sweep over 0.1…0.9.
- **Run:**

  ```
  python sim/comprehensive_eval.py --model_path outputs/outputs_temporal/best_temporal_model.pt \
    --prebuilt_sim simulation_data/xd_violence_8hr_sim.npz --threshold 0.388 --output_dir final/xxx [--smoothing_window k]
  ```

  Default `--output_dir` is `./final_evaluation` and default device is `cuda:1`.
- **Output:**
  - PNGs: `roc_curve, pr_curve, score_distribution, confusion_matrix, timeline` (first 3600 steps),
    `lead_time_histogram, threshold_tradeoff, event_analysis, false_alarm_distribution, summary`
  - `metrics.json` with keys `model_config`, `classification_metrics`, `operational_metrics`,
    `confusion_matrix`, `lead_times`, `event_details`, `threshold_sweep`
- **Reusable part:** the `plot_*` functions and `save_metrics_json`, once fed correct inputs.
- **Bugs:**
  - non-causal smoothing
  - mis-anchored lead time
  - alarms in the pre-window are double counted
  - a vlm_student with `use_temporal` would be auto-detected as temporal

### 4.8 `sim/sweep_thresholds.py` (40 lines)
- Subprocess loop over `live_simulator.py --synthetic` at 7 thresholds.
- Paths are hard-coded: `runs/outputs_xd/best_model.pt`, `cuda:1`, `./simulation_sweep/`.

### 4.9 `sim/visualize_predictions.py` (593 lines)
- Renders an MP4 with a score bar, alert state, ground truth and a 60-s score graph. Also writes
  `<out>_predictions.json`.
- **Run:**

  ```
  python sim/visualize_predictions.py --model_path <temporal ckpt> --video_path <mp4> \
    --embeddings_path <vlm_emb npz> --labels_path <labels_multih npy> --output_path vis/x.mp4 --threshold 0.388
  ```

- It does index labels by `t` (correct row alignment), but:
  - `emb_fps = 1/mean(diff(t)) = 0.2` is in *label-row* units.
  - `timestamp = t/emb_fps = 5t` s is wrong.
  - `horizon_frames = int(30*0.2) = 6` rows, which is about 2 s for XD (median 0.30 s per row).
  - Video time (s) is matched to the nearest `t` (row index), which is wrong unless 1 row = 1 s.
  - "VIOLENCE ACTIVE" actually shows the flawed pre-violence flag.
  - Overlays are therefore mistimed **(verified from code; magnitude inferred)**.

### 4.10 `scripts/evaluate_with_smoothing.py` (476 lines): the best existing protocol skeleton
- **Pipeline:**
  - `load_model`: infers dims from the state dict.
  - `run_inference`: causal, zero-padded 30-window.
  - `apply_smoothing`: k consecutive.
  - `apply_cooldown`.
  - `evaluate`: §3.2/§3.3.
  - `sweep_parameters`.
- **Run:**

  ```
  python scripts/evaluate_with_smoothing.py --model_path outputs/outputs_temporal/best_temporal_model.pt \
    --sim_path simulation_data/xd_violence_vlm_matched.npz --threshold 0.3 --smoothing_window 3 \
    --alert_cooldown 30 --output_dir final/smoothing_test [--sweep]
  ```

  The sim must contain `frame_labels`.
- **Output:** `scores.npy`, plus `results.json` (`config` + `metrics`) or `sweep_results.json`.
- **Bugs:**
  - **The model class uses `nn.ReLU`, but training (`training/train_temporal_model.py`) uses
    `nn.GELU`.** The state dict loads silently, but scores differ. For example, on the vlm_matched sim at
    θ=0.2, `final/high_sensitivity` (this script) gives AUC 0.811 and `final/high_sensitivity_eval`
    (`comprehensive_eval`) gives 0.813. That gap also includes different warm-up handling, and the two
    runs are only assumed to use the same checkpoint **(inferred)**.
  - The `horizon_sec * fps` 1-fps assumption.
  - Events on flawed `frame_labels`.
  - Sweep chosen on the evaluation sim.

### 4.11 `continuation/time_to_violence.py` (425 lines)
- `TimeToViolenceEstimator`:
  - Applies EMA (α=0.3) to raw scores.
  - Computes velocity as the linear-regression slope over a 10-s window.
  - Assigns a tier: CLEAR <0.30 ≤ WATCH <0.45 ≤ ALERT <0.65 ≤ IMMINENT <0.75 ≤ ACTIVE.
  - Uses a heuristic ETA anchored on `BASELINE_LEAD_TIME_SECONDS = 26.9`.
  - Computes a stability-based confidence.
- `python continuation/time_to_violence.py` runs a synthetic 70-s demo, which prints emojis.
- It is **not integrated**: `realtime/rtsp_inference.py` does not import it. The patch is only a string.
- The 26.9 s anchor matches `final/high_sensitivity/results.json` (avg_lead_time 26.918). That number
  came from `evaluate_with_smoothing` with θ=0.2 on the leaky `xd_violence_vlm_matched.npz`, and its
  lead is capped at 30 by construction **(inferred match)**. The ETA is therefore uncalibrated.

### 4.12 `continuation/lda_calibration_complete.py` (409 lines)
- Fits LDA on `simulation_data/xd_violence_vlm_clean_t02.npz` (`frame_labels`, which are flawed).
- Scores the YouTube clip `probe_clip_01_embeddings.npy` (`[197, 3584]`; `probe_clip_01_scores.npy`
  `[197]`) with 5 calibrations: raw LDA, z-score, percentile matching, CORAL (256 PCA dimensions), and
  within-video normalisation.
- Hand labels: safe 0–6 s, anticipation 7–36 s, violence 37–45 s, post 46+ s. `period_means` slices with
  exclusive ends, so the last second of each period is dropped.
- **Result:** `results/lda_calibration_results.txt`. **No method** scores violence above safe (all
  `violence_highest: False`).
- Single video, n=1. Diagnostic only.

### 4.13 `continuation/domain_adaptation.py` (561 lines)
- Logistic regression on source, scored on target, under five settings: no adaptation, TTA per-dimension
  re-standardisation, CORAL, simplified "TCA" (actually per-domain z-scoring plus PCA), and low-shift
  subspace (Cohen's d ≤ 0.3 or top-512).
- Also has a `--mode youtube` period-mean variant.
- No output exists (`results/domain_adaptation.csv` is absent), so it was apparently **not run**
  **(inferred)**.
- **Bug:** the "XD source" is `sorted(data/processed/vlm_emb.rglob('*.npz'))[:300]`, which contains
  only UCF videos (see §4.14). The UCF target is the first 200 of `vlm_emb/ucf_crime`. Source and target
  are therefore largely the **same UCF videos**.

### 4.14 `experiments/` (README "six scripts" + runner + three newer scripts)

**Loader bug shared by `ema_smoothing.py`, `algorithm_variations.py`, `baseline_comparisons.py`,
`prompt_engineering_stats.py`, `xd_violence_investigation.py` and `ucf_crime_testing.py`:**

- Their loaders take `sorted(vlm_dir.rglob("*.npz"))[:max_videos]` over `data/processed/vlm_emb`.
- Sorted order is `curated/` (no labels, skipped), then `ucf_crime/…`, then `xd_violence/…`.
- **So the "XD-Violence" experiments load only UCF-Crime videos:**

  | max_videos | Videos loaded | Videos with any positive |
  |---|---|---|
  | 200 | 108 UCF | 8 |
  | 300 | 208 UCF | 16 |
  | 1000 | 908 UCF | 112 |

- Train/val is the last 20 % of that sorted list, ignoring the official split.
- Per-video AUC is averaged only over val videos that contain both classes. **There is exactly one such
  video**: `Burglary005_x264` at max 300, `Arson011_x264` at max 200.
- Every per-video AUC in `results/algorithm_comparison.csv` and `results/ema_results.csv` is therefore
  the AUC of **one UCF video**.

| Script | What it does | Output (if produced) | Additional issues |
|---|---|---|---|
| `ema_smoothing.py` | score-level EMA, embedding-level EMA (LR), hidden-state EMA wrapper | `results/ema_results.csv`: score-EMA never beats raw (0.6305); embedding-EMA goes 0.465 → 0.592 at α=0.9. Both are measured on one video. | scaler fit on all pairs incl. val. The hidden-state level needs `ckpt['model']` to be a full `nn.Module`; project checkpoints store `model_state_dict`, so it is always skipped. |
| `algorithm_variations.py` | Transformer (CLS, d=128) vs LR (+isotonic calibration) vs LR + temporal features vs sliding-window score pooling | `results/algorithm_comparison.csv` (LR 0.63, Transformer 0.81 pooled, SlidingWindow_mean_w5 0.844), `results/transformer_model.pt` (3 MB, pickled module + scaler stats) | The Transformer is trained on the label at `i` with window `[i-30, i)`. The sliding-window "AUC" is on one video. |
| `baseline_comparisons.py` | random, majority, mean-pool LR, LR, SVM, LSTM (+EMA), LR+EMA, LR + temporal features, sliding window, LDA | `results/baseline_comparison.csv` **absent** (never completed, inferred) | `run_lr_temporal_features` passes the already-expanded features into `score_fn`, which expands them again (4D→16D), so it **crashes** unless `--skip_slow`. LSTM always skipped (same `ckpt['model']` issue). The FA metric ignores EMA. |
| `prompt_engineering_stats.py` | 15-prompt library; offline: LDA separability on current embeddings; online: runs each prompt through Qwen2.5-VL on probe frames | `results/prompt_engineering.csv`: `lda_auc = 1.0` | **In-sample** LDA AUC (fit and scored on the same 6,989 frames in 3584 dimensions), so meaningless. Online mode labels pre-violent frames as 1. `data/probe_frames` does not exist **(inferred)**. |
| `xd_violence_investigation.py` | pre-violence window inventory, position bias, Cohen's d between neutral, pre-violent and violent | `results/xd_investigation/{investigation_stats.json,best_anticipation_videos.txt}`: 1000 videos (actually about 908 UCF), 136 "violent", 4 "usable" | Despite the name, these are UCF stats under the flawed labels. See §7.4 for the corrected inventory. |
| `ucf_crime_testing.py` | XD→UCF transfer: LR, LSTM, per-video LDA z-score calibration, shift stats | `results/ucf_crime_results.csv`: all NaN | Without `--ucf_labels`, labels come from filenames. The first 200 sorted UCF files are all anomalies (Abuse…Burglary), so there is one class and AUC is NaN. The "XD" source is UCF (loader bug). The FA analysis is broken (§3.3). The category list includes non-violent crimes (Shoplifting, Stealing, …). |
| `run_all_experiments.sh` | runs the six scripts above with `EMB_DIR=data/processed/vlm_emb`, `LABELS_DIR=data/processed/labels_multih`, `LSTM_CKPT=outputs/outputs_temporal/best_temporal_model.pt` | as above | `set -e`, so the `baseline_comparisons` crash aborts the run **(inferred)**. |
| `run_experiments.py` (1198 lines) | exp1 prints the command for `exp1_person_detection.py`. exp2: score test clips + unified sim. exp3: train XD-only, UCF-only and combined LSTMs (train/val index) and test on test-only 4-h sims. exp4: UCF test sim plus XD comparison. | `experiments/output/exp{2,3,4}_*` (JSON + PNGs); exp3 checkpoints `experiments/output/exp3_cross_dataset/{xd_only,ucf_only,combined}/best_temporal_model.pt` (7 MB each) | Best-F1 threshold on test. Stream-level labels cross clip boundaries. exp4 falls back to **all** splits if no UCF test video loads. Back-filled first-window scores. |
| `exp1_person_detection.py` | YOLOv8n person mask (holdover 5) zeroes scores on person-free steps; compares metrics | `experiments/output/exp1_person_detection/exp1_results.json`: θ=0.639 (best F1 on test), AUC 0.832→0.812, FA/h 8.2→10.4 (filtering **hurt**), 173 videos | Labels are the flawed frame label without a horizon. Assumes 1 fps. |
| `fast_analyze_curation.py` | for YouTube curation: Qwen embeddings at `--fps`, LSTM scoring, heuristic stage segmentation (calm/tension/escalation/violence/aftermath). Writes back to `data/anticipation_curation/candidates.json`, optionally `data/processed/vlm_emb/curated/*.npz`. | `vlm_emb/curated` (92 files) | A heuristic **pseudo-labeler**, not an evaluator. Centered smoothing. `t` saved as `arange` (not frame indices). Labels derived from the model's own scores are circular if reused as ground truth. |

---

## 5. Results produced, and what they really are

The simulation file behind each `final/*/metrics.json` is identified by its step count and duration
**(inferred)**:

- 8.91 h / 13 events → `xd_violence_8hr_sim*.npz`
- 13.14 h / 183 events → `vlm_matched`/`vlm_audio`/`8hr_fused`
- 3.90 h → `vlm_clean_t02`
- 4.46 h → `vlm_clean`
- 4.12 h → `vlm_lowfa`
- 11.59 h → `vlm_filtered`
- 8.0 h / 117 events → `ucf_crime_8hr_sim`
- 8.0 h / 5 events → `xd_diluted_5pct_8hr`

| Run (dir) | Sim (inferred) | Leaky? | θ | AUC | AP | Det | FA/h | Lead* |
|---|---|---|---|---|---|---|---|---|
| `final/clean_t02_eval`, `final/clean_thresh_0.2` | clean_t02 | **yes (323/394 train)** | 0.2 | 0.982 | 0.991 | 19/19 | 4.9 | 23.4 |
| `final/lowfa_eval` | lowfa | yes | 0.2 | 0.974 | 0.985 | 48/49 | 11.9 | 26.3 |
| `final/clean_subset_eval` | vlm_clean | yes | 0.2 | 0.961 | 0.975 | 39/39 | 10.5 | 22.2 |
| `final/eval_diluted_5pct` | diluted | yes | 0.388 | 0.945 | 0.851 | 5/5 | 1.0 | 0.0 |
| `final/evaluate_fused` | 8hr_fused | yes | 0.388 | 0.887 | 0.706 | 218/280 | 20.8 | 14.8 |
| `final/evaluate_vlm_audio` | vlm_audio | yes | 0.388 | 0.868 | 0.670 | 199/269 | 22.8 | 14.8 |
| `final/high_sensitivity_eval` | vlm_matched | yes | 0.2 | 0.813 | 0.694 | 183/183 | 20.4 | 26.5 |
| `final/high_sensitivity/results.json` (evaluate_with_smoothing) | vlm_matched | yes | 0.2 | 0.811 | 0.694 | 98.9 % of 356 | 6.5 | 26.9 |
| `final/evaluate_original`, `final/test_original` | 8hr_sim (test only) | no, but flawed labels | 0.388 / 0.6 | 0.834 / 0.854 | 0.131 / 0.138 | 11/13, 12/13 | 8.6 / 10.1 | 11.4 / 19.1 |
| `final/final_evaluation_lstm` | 8hr_sim | no | 0.388 | 0.834 | 0.166 | 13/13 | 0.0 | 20.7 |
| `final/final_evaluation_lstm_2` | 8hr_sim | no | 0.55 | 0.633 | 0.039 | 13/13 | 1.5 | 30.0 |
| `final/final_evaluation_vlm_student` | 8hr_sim | no | 0.6 | 0.853 | 0.241 | 13/13 | 26.4 | 9.6 |
| `final/final_evaluation_optical_flow_fused` | 8hr_flow_sim_fused | misaligned | 0.2 | 0.557 | 0.024 | 6/13 | 9.2 | 25.0 |
| `final/eval_ucf_crime` | ucf_crime_8hr_sim | **yes (703/884 train)** | 0.388 | 0.459 | 0.090 | 112/117 | 30.5 | 26.6 |
| `final/evaluate_audio_only` | audio_only | yes | 0.53 | 0.438 | 0.153 | 94/476 | 3.3 | 6.2 |
| `simulation_results/final/results.json` (live_simulator, VLM student) | 8hr_sim | no | 0.6 | 0.853 | 0.242 | 13/13 | 87 FAs (about 9.8/h) | 13.8 (pre-window only) |
| `simulation_results_temporal/thresh_0.388` | 8hr_sim | no | 0.388 | 0.850 | 0.227 | 5/13 | 7 FAs | 0 |
| exp3 `xd_only` → XD test (`run_experiments`) | fresh test-only 4 h sim | no (flawed labels, θ tuned on test) | 0.363 | 0.768 | 0.178 | 76.9 % | 5.8 | 3.9 |
| exp3 `xd_only` → UCF test | test-only | no | 0.617 | 0.538 | 0.073 | 93.3 % | 20.3 | 17.6 |
| exp4 UCF test / XD test (default `--model_path`, i.e. the main LSTM; inferred) | test-only | no | 0.642 / 0.632 | 0.479 / 0.826 | 0.035 / 0.150 | 73 % / 46 % | 24.3 / 1.9 | 16.5 / 8.3 |

\*Lead is measured in embedding steps and is mis-anchored (§3.2). It is at most 30 by construction.

Takeaways:

- The 0.96–0.98 AUC results exist only on simulations dominated by **training videos**.
- On test-only XD streams the LSTM's frame AUC is about 0.77–0.85 under the flawed labels.
- Cross-domain to UCF is chance level (0.46–0.54).
- No run measures anticipation of real violence onset in seconds.
- `final/final_evaluation_lstm` (FA/h 0.0 with 13/13 detection) and `final/final_evaluation_simple`
  report a different `model_type` string. They were presumably produced by an older or other version of
  the evaluator (`tuning/*_simulator.py`?) **(inferred)**.

The main checkpoint `outputs/outputs_temporal/best_temporal_model.pt` (XD-only, `labels_multih`,
seq 30, stride 5) has `final_metrics.json` val AUC 0.577 vs test AUC 0.991 at sequence level. That gap
is itself a red flag (owned by the training docs).

---

## 6. Known problems, ranked

1. **Label semantics inverted or shifted** (`labels.max(axis=1) > 0` on multi-horizon labels with -1
   inside violence). Affects every builder and evaluator here, and also `training/train_temporal_model.py`.
2. **5× embedding/label index misalignment** (`labels[:len(emb)]` instead of `labels[t]`). Per-step
   agreement between the two alignments is only 0.898 on XD test, so ~10 % of steps change label.
3. **No real time base.** "fps = 1" is assumed everywhere. Real step spacing is 0.2–66 s (median about
   1.5 s XD, 3.3 s UCF). Lead times and FA/hour are in steps.
4. **Train-video leakage** in every `simulation_data` file except `xd_violence_8hr_sim*.npz` and the
   (broken) flow sims (§1.5).
5. **Operating points tuned on the reported data** (§3.4).
6. **Cross-clip anticipation labels.** The 30-step look-back reaches into unrelated padding clips, and 12
   of 18 onsets in the 8 h sim have zero in-clip context. Anticipating them is impossible. "Detection"
   there is the alarm firing at the clip cut or during violence.
7. **Lenient or mis-anchored event scoring** (§3.2). Detection credits alarms during violence, a
   constant alarm gets 100 % detection, lead is measured from the wrong anchor, and alarms in the
   pre-window count as both FA and detection.
8. **Trivial or duplicated negatives.**
   - Padding clips are repeated about 5.7×.
   - The diluted sim tiles negatives with noise.
   - UCF anomaly videos without temporal annotation are used as "normal".
   - XD clips that start in violence become "non-violent" padding.
   - Only 48 real UCF Normal videos are labelled.
9. **Tiny event counts.** 13 events in the "8 h" XD test sim, so one event is 7.7 % detection. No
   confidence intervals are reported anywhere.
10. **Model-definition drift.** `TemporalViolenceModel` is copy-pasted into 10 files.
    `evaluate_with_smoothing.py` uses ReLU, and the others use GELU.
11. **Experiment loaders read UCF while claiming XD.** Per-video AUCs come from one video. In-sample
    LDA AUC is 1.0. The UCF test gives all NaN. `baseline_comparisons` crashes.
12. **Misc:**
    - `build_flow_sim` misalignment.
    - `create_diluted_simulation` segment-split bug.
    - Non-causal smoothing in `comprehensive_eval`.
    - First-window score back-fill (`run_experiments`, `exp1`, `fast_analyze_curation`).
    - `live_simulator` has an unseeded RNG.
    - `--threshold 0` becomes 0.5.
    - `visualize_predictions` timing.
    - `time_to_violence` is uncalibrated and unused.
    - Hard-coded `cuda:1` and relative paths.
    - Pickled-module checkpoints (`results/transformer_model.pt`) need the class on import.

---

## 7. Recommended corrected benchmark (reuse plan)

### 7.1 What to reuse

| Need | Reuse | Fix required |
|---|---|---|
| Causal alarm logic | `scripts/evaluate_with_smoothing.py::apply_smoothing` (k-consecutive), optionally `apply_cooldown` | Count FAs on merged alarm *episodes*, not cooldown pulses. Optionally add a causal EMA (`experiments/ema_smoothing.py::ema_1d`). |
| Event scoring skeleton | `scripts/evaluate_with_smoothing.py::find_violence_events` + `evaluate` | Inputs must be **true violence segments in seconds**, per video. Use the window `[onset - W, onset)` for anticipation. Report detection-during-violence separately. Lead = `onset - first_alarm_time` in seconds. Drop the `fps` and step assumptions. |
| Plots | `sim/comprehensive_eval.py::plot_*`, `save_metrics_json` | Feed them corrected arrays. Do not use its `evaluate`, `_analyze_events` or `_find_false_alarm_times`. |
| Scoring a temporal model | `scripts/evaluate_with_smoothing.py::run_inference` (zero-padded causal window) | Import the model class from the training module, so there is no ReLU/GELU drift. Reset the window per video. |
| Do **not** reuse | any `simulation_data/*.npz`, `build_long_simulation`, `create_diluted_simulation`, `create_ucf_crime_simulation`, `build_flow_sim`, `live_simulator --synthetic` auto-threshold, the experiment loaders in `experiments/*.py` | |

### 7.2 Protocol specification

**Splits.** Use `train_index.json` / `val_index.json` / `test_index.json`. All thresholds,
EMA/k-of-n settings and cooldowns are chosen on **val**, then frozen and applied once to **test**.
Because test has only 12 videos with ≥30 s of pre-onset context, also report **video-level k-fold
cross-validation** over all 117 such videos, grouping folds by source movie or camera where possible.
Alternatively, add datasets with long untrimmed context (§7.4).

**Per-video timeline.** For embedding `i` of video `v`:

```python
ts_i = raw_frame_idx[i] / meta[v]["fps"]           # seconds; data/metadata/*.json
row_i = t[i]                                        # label row for this embedding
L = np.load(f"data/processed/labels_multih/{v}.npy")    # [T,7]
active_i = (L[row_i] == -1).all()                   # violence happening now
```

Violence segments are maximal runs of `active` rows. Convert row → seconds with
`round((r+0.5)/T*total_frames)/fps`. Better still, parse the raw annotations directly:
`data/raw_videos/xd_violence/annotations.txt` and
`data/raw_videos/ucf_crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt`.

**Frame-level metrics** (per horizon H ∈ {5, 10, 30, 60} s):
- `y_H(i) = 1` if `0 < next_onset - ts_i ≤ H`.
- Ignore steps where `active_i` (report detection AUC on those separately).
- Report ROC-AUC and AP pooled over test videos, plus the mean per-video AUC over videos that have both
  classes.

**Event-level metrics** (the primary metric for "30 s in advance"):
- **Eligible events:** each onset τ whose video has ≥ `C` seconds of observed footage before τ
  (C = H + model warm-up, e.g. 30 s + the context window).
- **Alarm signal:** `a(t)` is causal: threshold, then k-of-n or EMA, then optional refractory merge.
- **Anticipation@H:**
  - For each eligible event, find the first alarm in `[τ - W, τ)` (W = 60 or 120 s, the "useful
    early-warning window").
  - Success@H if `τ - t_first ≥ H`.
  - Report Success@{0, 5, 10, 30} s and the lead-time distribution (median, IQR).
- **Detection:** any alarm in `[τ - W, τ_end]`. This is reported separately and never called
  anticipation.
- **False alarms:**
  - Merge alarm steps into episodes (gap < g s).
  - An episode is false if its start is not inside `[τ - W, τ_end]` of any event in the *same* video.
  - FA/h = false episodes / hours of footage outside those windows. Use real seconds, all test videos,
    no duplication.
  - Prefer clean negatives: XD `*_label_A` clips and UCF `Normal_*`. **Exclude UCF anomaly videos that
    lack temporal annotations** (they are unlabeled, not negative).
- **Curve:** sweep θ on val to produce Success@30 vs FA/h, analogous to a FROC curve. Freeze θ at val
  FA/h ∈ {0.5, 1, 2}, and report test Success@30, detection and FA/h at each.
- **Uncertainty:** bootstrap over videos, giving 95 % CIs. With 12–117 events this is mandatory.

**Optional long-stream realism.** Concatenating clips is fine for latency or demo purposes if:
- labels and eligibility are computed **per clip before concatenation**;
- no clip repeats;
- only test (or held-out fold) clips are used;
- clips without enough in-clip context are scored for detection only.

### 7.3 Minimal reference sketch (to implement in a new module, e.g. `eval/benchmark.py`)

```python
def event_metrics(videos, theta, k=1, W=60.0, gap=10.0, H_list=(0,5,10,30), min_ctx=30.0):
    # videos: list of dict(ts=[N] seconds, score=[N] causal, segs=[(on,off),...] seconds, dur=seconds)
    succ = {H: 0 for H in H_list}; n_elig = 0; n_det = 0; n_ev = 0; fa = 0; neg_sec = 0.0; leads = []
    for v in videos:
        alarm = apply_smoothing(v["score"], theta, k)              # reuse (causal k-consecutive)
        eps = merge_episodes(v["ts"], alarm, gap)                  # [(t_start, t_end), ...]
        windows = [(on - W, off) for on, off in v["segs"]]
        for on, off in v["segs"]:
            n_ev += 1
            pre = [s for s, _ in eps if on - W <= s < on]
            n_det += int(bool(pre) or any(on <= s <= off for s, _ in eps))
            if on >= min_ctx:                                      # enough in-clip context
                n_elig += 1
                if pre:
                    lead = on - min(pre); leads.append(lead)
                    for H in H_list: succ[H] += int(lead >= H)
        fa += sum(not any(a <= s <= b for a, b in windows) for s, _ in eps)
        neg_sec += v["dur"] - sum(min(b, v["dur"]) - max(a, 0) for a, b in windows)
    return dict(success={H: succ[H] / max(n_elig, 1) for H in H_list}, n_eligible=n_elig,
                detection=n_det / max(n_ev, 1), fa_per_hour=fa / max(neg_sec / 3600, 1e-9),
                lead_median=float(np.median(leads)) if leads else None)
```

(Overlapping windows in `neg_sec` should be unioned in a real implementation. Alarms inside violence
are not false alarms.)

### 7.4 Data reality check, computed from `labels_multih` and metadata

| Source / split | Labelled videos | With violence | Onset ≥ 30 s after clip start | Onset ≥ 60 s | Median onset | Total hours |
|---|---|---|---|---|---|---|
| XD train | 626 | 379 | 56 | 21 | 6.9 s | 21.3 |
| XD val | 67 | 42 | 3 | 1 | 8.3 s | 2.1 |
| XD test | 79 | 46 | **4** | 2 | 5.9 s | 2.5 |
| UCF train | 751 | 98 | 41 | 24 | 20.6 s | 27.6 |
| UCF val | 104 | 12 | 5 | 3 | 21.0 s | 3.2 |
| UCF test | 94 | 15 | **8** | 2 | 34.1 s | 2.65 |

Clean negatives:

| Split | XD `label_A` | UCF Normal | Unannotated UCF anomaly videos (labels all 0, **not** valid negatives) |
|---|---|---|---|
| test | 34 clips, 1.65 h | 6 videos, 0.25 h | 73 videos, 2.0 h |
| val | 24, 1.11 h | 6, 0.19 h | 87, 2.6 h |
| train | 240, 11.7 h | 38, 0.91 h | 615, 23.5 h |

Implication: XD-Violence (trimmed movie/YouTube clips) cannot support a 30-s anticipation benchmark on
its own. Any new benchmark needs untrimmed footage with long pre-event context. Options:
- the 117 eligible videos with cross-validation;
- re-annotating the unannotated UCF anomaly videos;
- **(inferred, not verified here)** external long-form CCTV fight datasets with temporal annotations,
  such as UBI-Fights or CCTV-Fights.

---

## 8. Quick command reference (historical; all affected by §6)

```bash
PY=.venv/bin/python
# build test-only XD stream (flawed labels)
$PY sim/build_long_simulation.py --test_index test_index.json --source_filter xd_violence --output simulation_data/xd_violence_8hr_sim.npz
# evaluate a temporal LSTM with plots
$PY sim/comprehensive_eval.py --model_path outputs/outputs_temporal/best_temporal_model.pt --prebuilt_sim simulation_data/xd_violence_8hr_sim.npz --threshold 0.388 --output_dir final/my_eval --device cuda:0
# smoothing/cooldown sweep (needs frame_labels; fix ReLU->GELU first)
$PY scripts/evaluate_with_smoothing.py --model_path outputs/outputs_temporal/best_temporal_model.pt --sim_path simulation_data/xd_violence_8hr_sim.npz --sweep --output_dir final/sweep --device cuda:0
# experiments runner (exp2-4 train/eval LSTMs; GPU)
$PY experiments/run_experiments.py --exp2 --exp4 --model_path outputs/outputs_temporal/best_temporal_model.pt --device cuda:0
# time-to-violence demo (CPU)
$PY continuation/time_to_violence.py
```

Ownership notes:
- Label building (`scripts/build_multihorizon_labels.py`), embedding extraction
  (`scripts/vlm_label_dataset.py`), training (`training/train_temporal_model.py`) and the
  graph-transformer evaluator (`training/eval_anticipation.py`, `results/anticipation_*`) are documented
  in their own `docs/codebase/*.md` files.
- The label-transform bug in §6.1 must also be fixed in training. Otherwise the model learns
  "pre-violence within 120 s, violence = negative".

## Notes moved from code comments

- `experiments/baseline_comparisons.py`: the default `--ema_alpha 0.5` was chosen as the best alpha from
  the `ema_smoothing.py` results, and the sliding window `w=15` as the best config from
  `algorithm_variations.py`. The old docstring's model list ([B1] to [C6]) was stale. It listed a
  Transformer encoder that the script never implements, and its numbering differs from the printed labels.
- `experiments/baseline_comparisons.py::run_majority_class` scores every frame with the training positive
  rate (a constant in (0, 1)), not with a 0/1 majority label.
- `experiments/fast_analyze_curation.py::score_with_lstm`: a clip shorter than `seq_len` gets one score
  from a single window that is front-padded with **zeros**. A removed comment wrongly said the first frame
  was repeated.
- `experiments/fast_analyze_curation.py::classify_stages`: `stage_timeline` has one entry per frame. It
  does not group consecutive frames of the same stage, though a removed comment said it did.
