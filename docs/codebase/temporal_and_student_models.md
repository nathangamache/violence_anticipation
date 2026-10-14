# Temporal and student models (the later training scripts)

Scope: the main line of modeling work from Feb 2026. These are small heads trained on frozen
Qwen2.5-VL-7B frame embeddings (plus optional flow, audio, or pose features). The goal is to predict
violence about 30 s before it starts.

| File | Lines | Role |
|---|---|---|
| `training/train_student_original.py` | 2135 | Multi-horizon frame MLP. It aligns labels correctly, splits by movie, and reports video-level alarm stats. **This is the only script in this area whose numbers can be trusted.** |
| `training/train_student.py` | 1077 | "Improved" single-frame MLP with focal loss. Labels are misaligned and it uses the leaky index splits. |
| `training/train_temporal_model.py` | 1119 | BiLSTM + attention over 30-step windows. It is the de-facto "production" model, but labels are misaligned and the evaluation is broken. |
| `tuning/fuse_features.py` | 346 | Builds the VLM + optical-flow (+QA) concatenated feature files (`data/processed/fused_vlm_flow`). |
| `tuning/contrastive_finetune.py` | 424 | SupCon + focal projector (V1). |
| `tuning/contrastive_projection.py` | 753 | Classification-first projector with auxiliary SupCon (V3). It can export projected embeddings. |
| `tuning/contrastive_simulator.py` | 298 | Runs the V1 contrastive projector over the "8-hour" stitched simulation. |
| `tuning/ensemble_simulator.py` | 519 | AND/OR/average/weighted ensemble of the `train_student.py` MLP and the temporal LSTM on the 8-hour simulation. |

Run every script from the repo root (`.`)
with `.venv/bin/python`. Defaults are relative paths such as `data/processed/...` and
`train_index.json`. Most scripts default to `--device cuda:1`. `nvidia-smi` is broken on this box, but
torch CUDA works.

---

## 0. Summary for the next agent

1. **None of the existing models anticipates violence 30 s ahead in any meaningful way.** On a
   leakage-free movie-grouped split, the best model is the `StudentMLP` from
   `train_student_original.py` (`outputs/student_mlp`, `runs/student_fix1`). It gets **test frame
   ROC-AUC 0.58–0.62** and PR-AUC 0.15–0.16, against a positive rate of 0.081. *Within* a source the
   test ROC is about chance (XD 0.51–0.56, UCF 0.51–0.59). The video-level alarm result is that 60% of
   violent test videos get a first alert inside [onset−30 s, onset+1 s], with a **median lead of about
   9 s**, and 24% of non-violent test videos get a false alarm.
2. **The headline numbers in `outputs/` (temporal test ROC 0.76–0.99) are artifacts.** Three things cause them.
   - **Label/feature misalignment.** Most scripts ignore the `t` index stored in the embedding npz
     (see §1.3).
   - **Domination by one video.** One non-violent 14-minute YouTube clip contributes 78% of the test
     sequences, and one violent clip contributes 72% of the val sequences (§1.5).
   - **Oracle thresholds.** Test F1 is picked on the test set itself.
3. The earlier `results/anticipation_prboost_v1|v2|multih_hset*` runs report test ROC 0.90–0.94. They
   used a **per-clip split with movie leakage**: 46 of 77 test XD clips share a movie or YouTube ID with
   training. When the script moved to `--split_group source_movie`, test ROC fell to 0.56–0.62 (§3.3).
4. The `fps=1.0` assumption used by every script except `train_student_original.py` is wrong. One
   embedding step is **about 3 s on average**, ranging from 0.2 s to 6 min depending on the clip (§1.2).
   "horizon_sec=30" and "sequence_length=30" therefore mean wildly different real durations per video.
5. Reusable pieces: the embedding npz files (aligned via `t`), `labels_multih`, `audio_features`
   (row-aligned to labels), `optical_flow`, the metadata JSONs, and these functions from
   `train_student_original.py`: grouped splitting, video-level anticipation stats, and causal EMA.

---

## 1. Shared data conventions (read this first)

### 1.1 Files consumed

| Path (under `data/processed/`) | Format | Notes |
|---|---|---|
| `vlm_emb/{xd_violence,ucf_crime,curated}/<vid>.npz` | keys `t` [N] int32, `raw_frame_idx` [N] int32, `emb` [N,3584] fp16 | Qwen2.5-VL hidden states. 772 XD + 949 UCF + 92 `curated` (the curated set was added 2026-04-06 and has no `labels_multih`). |
| `vlm_emb_old/`, `vlm_emb_violence/`, `vlm_emb_meanpool/` | same layout | Alternative extractions (old prompt / violence prompt / mean-pooled tokens). They are documented in the VLM-extraction doc. |
| `labels_multih/<vid>.npy` | float32 [T,7] in {−1,0,1} | One column per horizon: **[2,5,10,20,30,60,120] s** (the `--horizons_sec` default in `train_student_original.py`). 1 means an onset falls within h seconds. 0 means no. −1 means ignore (at or after onset). Example: `Bad.Boys.1995__#01-33-51_01-34-37_label_B2-0-0` has rows 0–7 = 1 and row 8 onward = −1 in every column. 1721 files. |
| `labels_preonly_h30_ignore/<vid>.npy` | float32 [T] | Single-horizon labels for 610 videos. Used only by the old Feb-02 runs; the current `train_student_original.py` rejects 1-D labels. |
| `vlm_labels/<source>/<vid>.jsonl` | one JSON per sampled `t` | `violent{label,score}`, `people_count`, `actions[]`, `weapons[]`, `scene[]`, `temporal_change`, `notes`, `context_mode`. The context comes from **previous** frames only (`vlm/qwen_teacher.py` `ContextWindow`), so it is causal. |
| `audio_features/xd_violence/<vid>.npz` | `features` [T,91] float32 + metadata | **Row-aligned to the label timeline**: rows == T for 146/146 val+test XD clips checked. The `output_fps` field says 1.0, which is misleading. |
| `optical_flow/xd_violence/<vid>.npz` | `features` [n_raw_frames,22], `frame_indices`, `video_info` | One row per raw video frame (24 fps). |
| `fused_vlm_flow/<vid>.npz` | `emb` [N,3606] fp16 | Built by `tuning/fuse_features.py`: VLM 3584 + flow 22. 772 XD only. |
| `fused_features/xd_violence/<vid>.npz` | `emb` [N,4124] | VLM + pose (540). Built by `scripts/build_fused_features.py` (not in this area). |
| `vlm_audio_features/xd_violence/<vid>.npz` | `emb` [N,3675] | VLM + audio (91). Built by `scripts/multimodal_fusion.py` (not in this area). |
| `data/metadata/{xd_violence,ucf_crime}.json` | `{vid: {fps,total_frames,label,event_frame,source}}` | Used for seconds conversion in `train_student_original.py`. |
| `data/raw_videos/xd_violence/annotations.txt`, `data/raw_videos/ucf_crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt` | raw temporal annotations | Used only by `train_student_original.py` video stats. |
| `train_index.json` / `val_index.json` / `test_index.json` (repo root) | list of `{graph_path,label_path}` | 1438 / 180 / 180 clips (XD 645/70/83, UCF 793/110/97). There is no clip overlap. **There is movie overlap**: 32/71 test and 36/60 val XD movies or YouTube IDs also appear in train. `eval_index.json` is identical to `test_index.json`. |

### 1.2 Time scale: the "label timeline" is not seconds

- T, the number of label rows per clip, is a **graph timeline**. Its rows-per-second varies by clip:
  XD median 1.56 (p10 0.23, p90 5.19) and UCF median 2.01. Only 26/772 XD clips have T equal to
  `total_frames`, which is 24 rows per second.
- Embeddings are sampled at `t = 0, 5, 10, …` (every 5 label rows). `raw_frame_idx` equals
  `round((t+0.5)/T * total_frames)`, the same formula as `scripts/vlm_label_dataset.py`.
- So **one embedding step is a median of about 3.1 s (train XD), ranging from 0.21 s to 362 s**. For
  example, `A.Beautiful.Mind…label_A` has 37 embeddings spread over 240 s, about 6.6 s per step.
  `v=vHnAmx0UYbM__#1_label_A` has 4009 embeddings over 835 s, 0.21 s per step.
- `train_temporal_model.py`, `train_student.py`, `contrastive_*.py`, and the simulators all assume
  `fps=1.0`. Their "30-second horizon" and "30-frame sequence" really mean "30 rows". On embedding rows
  that is about 90 s on average and anywhere from 6 s to 3 h. On audio rows it is about 19 s on average.
- `train_student_original.py` is the only script that converts to real seconds, using
  `map_t_to_raw_frame` with metadata fps. Its horizons come from `labels_multih` columns, whose seconds
  semantics were set by the label builder (inferred to be correct; see the labels doc).

### 1.3 The misalignment bug (affects 4 of the 8 scripts)

`train_temporal_model.load_video_data` (lines 380–386), `train_student.ViolenceAnticipationDataset`
(lines 343–356), `contrastive_finetune.load_video_data` together with `ContrastiveDataset`, and
`contrastive_projection.SimpleDataset` (lines 288–294) all do this:

```python
if labels.ndim > 1: labels = labels.max(axis=1)
labels = (labels > 0).astype(np.float32)
min_len = min(len(embeddings), len(labels))
embeddings = embeddings[:min_len]; labels = labels[:min_len]   # ignores npz['t']
```

Embedding row i actually corresponds to label row `t[i] = 5*i`. The code pairs it with label row i, so
**only the first ~20% of each clip's label timeline is used, and it is matched against embeddings that
span the whole clip**.

- On the test clip `Young.And.Dangerous.I.1996__#0-20-50_0-22-15_label_B1-0-0`, the misaligned
  positive rate is 1.00 versus 0.21 correct; the two agree on only 20.8% of rows.
- Over the full test set, the misaligned sequence positive rate is 0.107 versus 0.049 correct.
- The same misalignment carries into every derived feature built from `vlm_emb` rows: `fused_vlm_flow`,
  `fused_features`, `vlm_audio_features`, and the contrastive `projected_emb`.
- `audio_features` alone has T rows, so the audio-only temporal run is aligned.

Label semantics after binarization: `max over the 7 horizons > 0` means "within 120 s before an onset".
The −1 rows (during or after violence) become **0**, so frames *showing* violence are trained as
negatives.

### 1.4 Label definition per script

| Script | Target |
|---|---|
| `train_student_original.py` | Per sampled `t`: the 7-vector `labels_multih[t]`, with −1 masked out of the loss. Evaluation and thresholds use one column (`--select_horizon_sec`, default 30 → column 4). This is true pre-onset anticipation. |
| `train_student.py` | Per embedding row i: `max(bin_labels[i : i+30])`. The current row is included. Labels are misaligned. |
| `train_temporal_model.py` | Per window [s, s+30): `max(bin_labels[s:s+30] ∪ bin_labels[s+30 : s+60])`. This mixes "in-window" (detection) with the next 30 rows (anticipation). Labels are misaligned. |
| `contrastive_finetune.py`, `contrastive_projection.py` | Per row: `max(bin_labels[t : t+30])`. Labels are misaligned. |
| Simulators | `anticipation_labels` from the stitched npz (built by `sim/build_long_simulation.py` as `max(stream_labels[t:t+30])`). |

### 1.5 Split composition (why the temporal test ROC is about 0.99)

These numbers apply to `train_temporal_model.py` with seq_len 30, stride 5, and `source_filter xd_violence`:

| Split | Sequences | Dominant clip(s) |
|---|---|---|
| train | 4597 | Top 5 clips give 2003 sequences. 278/626 clips (44%) have fewer than 30 embeddings and give **0 sequences**. |
| val | 608 | **438 (72%)** from `v=pFamvR9CpYw__#00-06-36_00-15-20_label_B4-0-0` (violent, 2217 embeddings). |
| test | 1020 | **796 (78%)** from `v=vHnAmx0UYbM__#1_label_A` (non-violent, 4009 embeddings) and **76 (7%)** from `Young.And.Dangerous.I…B1`. That second clip is all-positive under the misalignment. |

As a result, the test ROC mostly measures whether the model can tell these two clips apart. Most
short clips contribute 1–4 windows, and each window covers nearly the whole clip, so the window label
is effectively a *clip-level* label.

The audio-only run uses label-timeline rows, which makes this even more skewed: 4004 of 6331 test
windows come from `vHnAmx0UYbM`.

The XD-only `train_student.py` run (`runs/outputs_xd`) has the same problem: 4009 of 6768 test samples
come from that one clip.

---

## 2. Per-script reference

### 2.1 `training/train_student_original.py` (the best-engineered script)

**Purpose.** Frame-level multi-horizon anticipation head. It uses VLM embeddings plus hashed
structured VLM outputs, with grouped splits, causal smoothing, and video-level alarm evaluation.

**Inputs.**
- `--emb_root` (default `data/processed/vlm_emb`). It scans `rglob('*.npz')` and takes the source from
  the parent folder name. It reads `t` and `emb`.
- `--jsonl_root` (`data/processed/vlm_labels`).
- `--labels_dir` (required; must hold [T,K] arrays with K equal to the number of `--horizons_sec`
  values).
- `--meta_dir` (`data/metadata`).
- `--ucf_temporal_file` and `--xd_temporal_file`.
- Samples are taken at each npz `t`, with `y = labels[t]`, which is **correctly aligned**. Rows where
  every column is −1 are dropped.

**Features (per t).** `emb` (3584), then optional `violent.score` (`--include_vlm_score`, 1 dim) and
optional `people_count/10` (`--include_people_count`, 1 dim). Then MD5-hashed, normalized multi-hot
vectors: `actions` (64), `weapons` (32), `scene` (64), and `notes` tokens (`--hash_dim_notes`,
default 0).

| Run | `feature_dim` | Configuration |
|---|---|---|
| default | 3744 | as above, no optional flags |
| `prboost_v1/v2` | 3746 | both flags |
| `prboost_v3`, `student_fix2` | 4034 | (inferred) notes hashing enabled |

**Model.** `StudentMLP` is LayerNorm(D) → Linear(D,256) → GELU → Dropout → Linear(256,256) → GELU →
Dropout → Linear(256,K=7). That is about 1.0 M parameters.

**Loss and tricks.**
- `BCEWithLogitsLoss(pos_weight)` per horizon, with pos_weight = (n_neg/n_pos)^0.5 capped at 20.
- Per-horizon loss weights `inv_sqrt_pos_weight`, normalized to mean 1.
- −1 targets are masked.
- Optional label smoothing, a `--video_balance` WeightedRandomSampler, and train-only z-score
  normalization (`--normalize_inputs`).
- Gradient clipping at 1.0. AdamW (lr 1e-3, weight decay 1e-4). ReduceLROnPlateau on val PR-AUC.
- `set_seed` makes runs deterministic.

**Split.** It does *not* use the index JSONs. `make_splits` shuffles **groups** with
`random.Random(seed)` using val_frac 0.1 and test_frac 0.1. `--split_group` defaults to
`source_movie` (`source/<prefix before "__#">`), which is the safe option. It writes
`splits_manifest.json`.

**Evaluation.**
- **Checkpoint selection.** Best val PR-AUC on the selected horizon column goes to `best.pt`.
- **Smoothing.** Optional causal per-video EMA (`--ema_alpha`) on probabilities.
- **Operating threshold.** Chosen on val only (`--threshold_select f1|youden|mcc|precision_floor|fbeta|video_utility`)
  and then applied to test. The result is `test_metrics_at_val_threshold`, which is leakage-free.
  `*_metrics_oracle` reports per-split oracle numbers for reference.
- **`compute_anticipation_video_stats`.** For each video:
  - It takes segment starts from the annotations. If those are missing, it estimates start = first
    positive row + horizon.
  - It turns probabilities into alert times. `--min_consecutive` K requires K consecutive points within
    `consecutive_t_gap`, and the alert fires at the Kth point.
  - It classifies the **first** alert as *in window* [s−H, s+grace], *early false* (before s−H), or
    *late/miss*.
  - It also reports lead-time quantiles and the false-alarm rate on negative videos.
- `video_utility` sweeps 101 thresholds on val, with
  `utility = any_in_window − early_pen·early − late_pen·(1−any_in_window) − fa_pen·neg_FA`.
  If the winning threshold produces no alerts at all, it falls back to the val F1 threshold.

**Outputs (`--out_dir`).**
- `metrics.json`. It contains everything except the raw argv. Hidden size, dropout, lr, normalization
  stats, and so on are **not saved**.
- `best.pt` (`{"model": state_dict, "in_dim": D}` only).
- `per_sample.csv` (split, vid, source, t, gt, raw/EMA probability).
- `per_video_{val,test}.csv`, `splits_manifest.json`, and `video_utility_sweep.csv` (video_utility
  mode only).
- About 15 PNGs: train curve, PR, ROC, threshold sweeps, histograms, calibration, confusion, and
  per-video scatter.

**How to run (reconstructed; full argv was never saved).**
```bash
.venv/bin/python training/train_student_original.py --task anticipation \
  --emb_root data/processed/vlm_emb_meanpool --labels_dir data/processed/labels_multih \
  --out_dir outputs/student_mlp --select_horizon_sec 30 --threshold_select f1 --device cuda:0
```

**Known issues.**
- `best.pt` cannot be reloaded faithfully elsewhere. It lacks `hidden`, `K`, horizons, and the x_mu/x_sig
  values used by `--normalize_inputs`.
- `--horizon_sec` is only echoed. `--task detection` still requires [T,K] labels, despite what the
  docstring says.
- The older single-horizon runs (`results/anticipation_preonly_*`, which use 1-D labels) **cannot be
  reproduced** with the current code, because it raises `Expected multi-horizon labels [T,K]`.
- `iter_npz_files` treats `vlm_emb/curated` as a new source. Those clips are skipped because they have
  no labels, but they will join the split if labels are ever added.
- The fallback onset estimate (first positive + horizon) is wrong when the onset falls less than H
  seconds into the clip.
- **UCF anomaly training videos** (for example `Abuse001_x264`) have all-zero labels, because UCF
  provides temporal annotations only for its test videos. Real crimes are therefore negatives, and
  alerts on them count as false alarms (inferred from labels; see the labels doc).
- The video-stats function reloads label npy files for every threshold in the sweep, which is slow
  but correct.
- The utility counts "any alert in window" even when the first alert was early, so it is not
  consistent with the first-alert classification. `hset_v3` chose threshold 1.0, which means zero
  alerts. (Inferred: that run predates the current grid clamp below 1.0 and the
  degenerate-threshold fallback.)

### 2.2 `training/train_student.py`

**Model.** `VLMStudentModel` is [Linear 3584→512 → BN → GELU → Dropout(0.3)] → [512→256 …] →
[256→128 …], followed by a 128→64→1 classifier. That is about 2.0 M parameters. An LSTM branch
(`use_temporal`) exists but `main()` never enables it.

**Data.**
- Uses the index JSONs, with an optional `--source_filter`.
- One sample per embedding row, with label `max(labels[i:i+30])`. The labels are misaligned (§1.3).
- `min_frames=5`.

**Loss and training.**
- `FocalLoss(alpha 0.75, gamma 2)` is always used. `--use_focal_loss` is `store_true` with
  `default=True`, so it cannot be turned off. `class_weight` is computed but unused.
- AdamW (lr 1e-4), CosineAnnealingWarmRestarts (T_0=10, T_mult=2), gradient clipping 1.0.
- Early stopping on val ROC with patience 10. No seed is set.
- Optional `SourceBalancedSampler`.

**Evaluation.** ROC and PR-AUC, plus F1, Youden, precision≥0.3, and MCC thresholds. Each split picks
its own thresholds, so **the test thresholds are oracle**. It also prints a per-source breakdown to
stdout only.

**Outputs.** `results.json` (val and test metrics) and, with `--save_model`, `best_model.pt`
(state_dict + args + results).

**Known issues.**
- With `--ucf_weight`, the weight dict includes the keys `'Movies'` and `'unknown'`, which have no
  samples. They still take weight mass, so ucf gets 3/5.5 and xd 1/5.5, and **about 27% of each
  epoch's samples are silently dropped**.
- History is not recorded for the epoch on which early stopping triggers.
- The labels include the current frame, so the target is part detection and part anticipation.

**Runs (`runs/`, 2026-02-18).**

| Run | Configuration | Val ROC / PR-AUC | Test ROC / PR-AUC (oracle-F1) |
|---|---|---|---|
| `runs/outputs` | all sources, source_balanced, ucf_weight 3 | 0.617 / 0.347 | 0.723 / 0.430 (F1 0.45) |
| `runs/outputs_xd` | XD only | 0.680 / 0.439 | 0.925 / 0.632 (F1 0.67). Inflated: 59% of test samples come from one negative clip. |
| `runs/outputs_ucf` | UCF only | 0.444 / 0.070 | 0.558 / 0.175. Chance. |

`tuning/ensemble_simulator.py` loads a `VLMStudentModel` checkpoint (`feature_extractor.{0,4,8}`
keys) as its "simple model". (Inferred: this is one of the `runs/outputs*/best_model.pt` files.)

### 2.3 `training/train_temporal_model.py`

**Model.** `TemporalViolenceModel`. A per-frame encoder (Linear D→256 → LN → GELU → Dropout → 256→256 →
LN → GELU → Dropout) feeds a 2-layer **bidirectional** LSTM (hidden 128). Additive attention pooling
over the window feeds a 256→64→1 classifier. That is about 1.8 M parameters for D=3584.

Bidirectionality is not leakage here, because the window ends at the current step. The `lengths`
packing path and `predict_with_attention` are unused.

**Data.**
- `load_video_data` reads the index JSON, filters to `--source_filter` (default `xd_violence`), and
  loads embeddings from `emb`, `embedding`, `embeddings`, `features`, or `arr_0`, together with
  `labels_multih`. Labels are misaligned (§1.3).
- `TemporalViolenceDataset` builds windows of `sequence_length=30` with stride 5. The label is
  described in §1.4.
- Augmentation adds Gaussian noise with std 0.01 to about 15% of windows. That is negligible.

**Loss and training.** FocalLoss(0.75, 2), AdamW (lr 1e-4, weight decay 1e-4),
CosineAnnealingWarmRestarts(10, 2), gradient clipping 1.0, batch 64, early stopping on val ROC with
patience 15. **No seed**: `outputs_temporal` and `outputs_temporal_old` have identical args, and their
val ROC is 0.577 versus 0.600.

**Evaluation.** Window-level ROC, PR-AUC, and best-F1 per split. The best-F1 threshold is chosen *on
the split itself*, so the reported test F1 is oracle.

**Outputs.**
- `best_temporal_model.pt`: `{model_state_dict, args, val_metrics, test_metrics}`.
- `args.json`.
- `final_metrics.json`: validation, test, and training history.
- `graphs/`: 13 PNGs, including `summary.png`.

**How to run** (this example reproduces `outputs_fused`):
```bash
.venv/bin/python training/train_temporal_model.py --embeddings_dir data/processed/fused_vlm_flow \
  --input_dim 3606 --labels_dir data/processed/labels_multih --source_filter xd_violence \
  --output_dir outputs_fused --device cuda:0
```

**Results** (all XD-only, horizon "30 rows", seq 30, stride 5, from `final_metrics.json`). "Best ep" is
the epoch at which the checkpoint was selected; the train ROC column is the train ROC at that epoch.

| Output dir (date) | Features (dim) | Val ROC | Val PR | Test ROC | Test PR | Test F1 (oracle) | Best ep / total | Train ROC at best |
|---|---|---|---|---|---|---|---|---|
| `outputs/outputs_temporal` (02-23) | `vlm_emb` (3584) | 0.577 | 0.365 | 0.991 | 0.907 | 0.888 | **1** / 16 | 0.88 |
| `outputs/outputs_temporal_old` (02-23) | `vlm_emb` (3584), same args | 0.600 | 0.338 | 0.957 | 0.753 | 0.755 | 2 / 17 | 0.97 |
| `outputs/outputs_temporal_vlm_emb` (02-24) | `vlm_emb` (3584) | 0.575 | 0.356 | 0.893 | 0.720 | 0.759 | 5 / 20 | 0.99 |
| `outputs/outputs_temporal_old_emb` (02-23) | `vlm_emb_old` (3584) | 0.665 | 0.357 | 0.947 | 0.675 | 0.781 | 6 / 21 | 0.94 |
| `outputs/outputs_temporal_violence` (02-23) | `vlm_emb_violence` (3584), batch 32 | 0.533 | 0.240 | 0.912 | 0.785 | 0.758 | 2 / 17 | 0.98 |
| `outputs/outputs_projected` (02-23) | `results/contrastive_results/projected_emb` (256) | 0.598 | 0.281 | 0.945 | 0.756 | 0.795 | 1 / 16 | 0.99 |
| `outputs_fused` (02-23) | `fused_vlm_flow` VLM+flow (3606) | **0.721** | **0.426** | 0.920 | 0.698 | 0.615 | 13 / 28 | 1.00 |
| `outputs/outputs_temporal_fused` (02-24) | `fused_features` VLM+pose (4124) | 0.674 | 0.333 | 0.897 | 0.445 | 0.574 | 11 / 26 | 1.00 |
| `outputs/outputs_temporal_vlm_audio` (02-24) | `vlm_audio_features` (3675) | 0.693 | 0.367 | 0.773 | 0.469 | 0.516 | 11 / 26 | 1.00 |
| `outputs/outputs_temporal_audio_only` (02-24) | `audio_features` (91), **aligned** | 0.642 | 0.337 | 0.759 | 0.345 | 0.477 | 3 / 18 | 0.69 |

How to read this table:
- Train ROC is about 1.0 while val ROC is 0.53–0.72, so every run overfits immediately. The selected
  checkpoint often comes from epoch 1–2.
- The val/test disagreement comes from the split composition (§1.5), not from generalization.
- By val ROC, VLM + optical flow (`outputs_fused`, 0.72) is the "best" temporal run, and adding
  audio or pose also beats VLM alone. Those val sets are dominated by a single clip and use misaligned
  labels, so this ranking is weak evidence.
- Note that `args.json` in `outputs/outputs_temporal*` still says `./outputs_temporal`. The folders
  were moved after the runs.

**Downstream consumers.** `outputs/outputs_temporal/best_temporal_model.pt`, the run whose checkpoint
was selected at epoch 1 with val ROC 0.577, is the default `--model_path` in:
- `scripts/analyze_youtube.py`
- `scripts/evaluate_with_smoothing.py`
- `experiments/run_experiments.py`
- `experiments/exp1_person_detection.py`
- `experiments/fast_analyze_curation.py`
- `sim/comprehensive_eval.py`

`scripts/late_fusion.py` re-invokes this trainer. Each of these files copies the
`TemporalViolenceModel` class, so architecture changes must be mirrored in all of them.

### 2.4 `tuning/fuse_features.py`

**Purpose.** Builds per-clip fused features in five steps:
1. Load VLM (`emb`), flow (`features`), and QA npz files, searching flat, `xd_violence/`, and
   `ucf_crime/`.
2. Subsample every stream to the shortest length with `np.linspace`.
3. Normalize each stream **per video**: standard (default), minmax, or l2.
4. Concatenate the streams. In `weighted` mode, each stream is first multiplied by its weight.
5. Save `emb` as fp16 to `<output_dir>/<vid>.npz`, plus `fusion_config.json`.

**Produced.** `data/processed/fused_vlm_flow` (772 XD, dim 3606, config `{normalize: standard, fusion_method: concat}`).

**Run.**
```bash
.venv/bin/python tuning/fuse_features.py --vlm_dir data/processed/vlm_emb \
  --flow_dir data/processed/optical_flow --output_dir data/processed/fused_vlm_flow --source_filter xd_violence
```

**Issues.**
- Per-video z-scoring uses whole-clip statistics. That is non-causal for online use, and it erases
  between-clip level differences. A single-row clip becomes all zeros.
- The linspace alignment of raw-frame flow (about 24 Hz) to VLM rows is approximate: both span the full
  clip, but the sampling is nonuniform.
- **When a modality is missing for a clip, the clip is silently written with fewer dims.** For example,
  without `--source_filter`, UCF clips would be written as 3584-d.
- Existing outputs are skipped and never refreshed.
- Inherits the VLM-row misalignment if the result is used with `labels_multih`.
- `--qa_dir data/processed/vlm_qa_features` does not exist.

### 2.5 `tuning/contrastive_finetune.py` (V1)

**Model.** `ContrastiveProjector` is 3584→512 (BN, GELU, Dropout) →512 →256, then L2-normalized. A
256→64→1 classifier sits on the normalized projection.

**Loss.** `w·SupCon(τ=0.07) + (1−w)·focal(γ=2, no alpha)`.

**Data.** The train set balances classes by duplicating positives; it divides by zero if a split has
no positives. Labels are `max(labels[t:t+30])` and misaligned. XD only by default.

**Training.** AdamW (lr 1e-3), CosineAnnealingWarmRestarts(10), batch 512 with drop_last, early
stopping on val ROC with patience 15. No seed.

**Outputs.** `contrastive_model.pt` (state + args) and `test_projected.npz` (embeddings [N,256],
labels, scores). Val metrics are only printed.

| Run | Contrastive weight | Test ROC | Test AP (pos 0.164, N=6781) |
|---|---|---|---|
| `outputs/outputs_contrastive` (02-19) | 0.5 | **0.737** | 0.483 |
| `outputs/contrastive_v1` (02-24) | 0.7 | 0.660 | 0.413 |

These numbers were recomputed from `test_projected.npz`. As with the other XD runs, 59% of the test rows
come from one negative clip.

### 2.6 `tuning/contrastive_projection.py` (V3; the docstring calls it `contrastive_projection_v3.py`)

**Model.** `ProjectionNetwork` has three encoder sizes, all using Linear + LN + GELU + Dropout layers:

| Size | Encoder layers | Dropout |
|---|---|---|
| small | [1024, 512, 256] | 0.2 |
| medium | [1536, 768, 384] | 0.25 |
| large | [2048, 1024, 512, 256] | 0.3 |

A linear projector (→128, L2-normalized) and a classifier (→128 → LN → GELU → 1) sit on top.

**Loss.** `focal(α 0.75, γ 2) + 0.1·SupCon(τ 0.1)`, clamped. Batches whose loss is NaN or above 100 are
skipped.

**Training.** CosineAnnealingLR, early stopping on val ROC.

**Data.** `SimpleDataset` uses natural class balance and misaligned labels.

**Outputs.**
- `model.pt`, `history.json`, `metrics.json`, `training_curves.png`, `args.json`.
- With `--export_embeddings`, it writes `projected_emb/<source>/<vid>.npz`. These are 256-d *encoder*
  features (not the 128-d projection), and they are written for **all** clips under `--embeddings_dir`.

**Runs.**
- `outputs/contrastive_v3` (02-23, large). Best val AUC 0.656 at epoch 5, when train AUC was 0.975.
  Test AUC 0.903, AP 0.637, F1 0.639 (oracle). `projected_emb/xd_violence` holds 772 files.
- `outputs/contrastive_v2_large` (02-23). Produced by an **older V2 version that is not in the repo**
  (args include mixup, warmup, subsample_negative). Best val ROC 0.667 at epoch 4. It has no test
  metrics.
- `results/contrastive_results`. Produced by a third script that is also not in the repo
  ("attention" architecture, bottleneck 64). Its `projected_emb` was the input for
  `outputs/outputs_projected` (§2.3).

**Conclusion.** Contrastive projection did not improve val ROC over raw embeddings (0.60–0.67 versus
0.58–0.67).

### 2.7 `tuning/contrastive_simulator.py` and `tuning/ensemble_simulator.py`

**Input.** `--prebuilt_sim` points to an npz with keys `embeddings` [N,3584], `anticipation_labels`
[N], `fps`, `frame_labels`, and `horizon_sec`. The runs used
`simulation_data/xd_violence_8hr_sim.npz`:
- N = 32,092 steps, treated as "1 fps" (so a nominal 8.9 h).
- 13 violent events, 161 violent steps, 557 anticipation-positive steps (1.7%).
- Built by `sim/build_long_simulation.py` from **test_index XD clips**.
- The real duration differs from the nominal one, because steps are not seconds (§1.2) (inferred).

**contrastive_simulator.** Rebuilds the V1 `ContrastiveProjector` (`contrastive_finetune.py`
architecture, with BatchNorm) and scores frames in batches. An alarm fires when the last
`smoothing_window` scores (default 3) are all at or above the threshold.

**ensemble_simulator.**
- Loads a `train_student.py` MLP (dimensions auto-detected) and a temporal LSTM (args taken from the
  checkpoint).
- Per step, it scores the frame with the MLP and scores the causal window [i−29, i] with the LSTM. The
  first 29 steps reuse the MLP score.
- Combines the two scores with `and`, `or`, `average`, or `weighted`. The defaults are thresholds 0.6
  and 0.388, strategy `and`.
- Scoring runs one frame at a time (slow).

**Metric definitions (both files) and their problems.**
- An "event" is a run of `anticipation_labels == 1`, so it starts about 30 steps **before** the onset
  and lasts until the violence ends.
- An event counts as **detected** if any alarm fires between event start − 30 steps and event end. Alarms
  *during* violence therefore count as detection, so this is not anticipation.
- `avg_lead_time_sec` uses `start − alarm_time`, where `start` is the start of the anticipation window,
  not the onset. Only alarms strictly *before* the window count. **So the reported lead times
  (17–23 s) describe alarms about 47–53 s before onset, which fall outside the horizon.** Alarms inside
  the 30-s window are not averaged at all.
- False alarms are counted as rising edges where pred=1 and label=0.
- FA/hour is printed but not saved.
- In `and` mode, smoothing tests `min(score)` against the *average* of the two thresholds, which does
  not match the per-model thresholds. In `average` and `weighted` modes, `smoothing_window` is ignored.

**Results** (`results.json`; 13 events over 32,092 steps; FA/h assumes 1 step = 1 s):

| Run | ROC | PR-AUC | P / R / F1 (frame) | Detected | False alarms (≈/h) | "Lead" |
|---|---|---|---|---|---|---|
| `simulation_results_contrastive` (th 0.5) | 0.730 | 0.120 | 0.12 / 0.25 / 0.16 | 13/13 | 223 (25) | 23.0 |
| `simulation_results_contrastive/thresh_0.7` | 0.730 | 0.120 | 0.19 / 0.20 / 0.20 | 13/13 | 144 (16) | 18.6 |
| `simulation_results_ensemble/and_strategy` | 0.840 | 0.176 | 0.20 / 0.31 / 0.24 | 10/13 | 81 (9.1) | 17.6 |
| `simulation_results_ensemble/average_strategy` | 0.853 | 0.204 | 0.16 / 0.48 / 0.24 | 13/13 | 87 (9.8) | 21.0 |
| `simulation_results_ensemble/weighted_0.7` | 0.848 | 0.190 | 0.13 / 0.45 / 0.21 | 13/13 | 61 (6.8) | 20.7 |

For comparison, the LSTM alone (`simulation_results_temporal/*`, produced by `sim/temporal_simulator.py`,
which is outside this area):

| Threshold | ROC | Detected | False alarms |
|---|---|---|---|
| 0.388 | 0.850 | 5/13 | 7 |
| 0.6 | 0.834 | 9/13 | 27 |

Even on this lenient metric, at most 13 events are caught at the cost of 60–220 false alarms.

---

## 3. Results inventory for `train_student_original.py` runs

Results live in `results/anticipation_*`, `outputs/student_mlp`, and `runs/student_fix*`. "Test @ val
threshold" is leakage-free. For video-level stats, "FirstIn" means the first alert fell in
[onset−30 s, onset+1 s], "Early" means it fell before that, "NegFA" is the share of negative videos
with any alert, and "Lead" is the median lead time.

### 3.1 Leakage-free (movie-grouped split, `source_movie`, 1356/156/158 clips)

| Run (date) | Embeddings | Threshold mode | Val ROC / PR | Test ROC / PR | Test P / R / F1 @ val threshold | Test video: FirstIn / Early / NegFA / Lead |
|---|---|---|---|---|---|---|
| `outputs/student_mlp` (02-23) | `vlm_emb_meanpool` (3744) | f1 | 0.763 / 0.410 | **0.622 / 0.151** | 0.13 / 0.42 / 0.19 | 24/40 = 0.60 / 0.125 / 0.237 / 8.9 s |
| `runs/student_fix1` (02-17) | `vlm_emb` (3744) | fbeta | 0.823 / 0.465 | 0.578 / 0.158 | 0.11 / 0.27 / 0.16 | 0.60 / 0.10 / 0.153 / 7.8 s |
| `runs/student_fix2` (02-18) | `vlm_emb` (4034) | fbeta | 0.804 / 0.459 | 0.593 / 0.149 | 0.08 / 0.27 / 0.13 | 0.50 / 0.125 / 0.246 / 8.4 s |
| `results/anticipation_prboost_v3` (02-17) | `vlm_emb` (4034) | precision_floor 0.75, K=2 | 0.818 / 0.456 | 0.562 / 0.096 | 0.03 / 0.09 / 0.05 | 0.30 / 0.05 / 0.076 / 8.9 s |

Test positive rate is 0.081, which gives an always-positive F1 of 0.151. So **test F1 at the val
threshold barely beats predicting "positive" for everything**.

Within-source ROC, computed from `per_sample.csv` with EMA probabilities:

| Run | Test XD | Test UCF | Val XD | Val UCF | XD video-level max-prob ROC (test) |
|---|---|---|---|---|---|
| `student_mlp` | 0.564 | 0.506 | 0.809 | 0.461 | 0.71 |
| `student_fix1` | 0.513 | 0.587 | 0.847 | 0.587 | 0.76 |
| `prboost_v3` | 0.470 | 0.528 | 0.846 | 0.604 | 0.72 |

The overall ROC is partly a source-mixture effect: XD has both higher scores and a higher positive
rate than UCF. The model shows **some clip-level "this clip will turn violent" signal** but almost
**no ability to locate the 30-s pre-onset window** on unseen movies. The large val/test gap
(0.81–0.85 versus 0.47–0.56 within XD) reflects the small number of positive clips: 39 violent XD
clips in val and 32 in test.

### 3.2 Older runs with movie leakage (per-clip split, 1336/167/167; no `splits_manifest.json`)

| Run (date) | Notes | Val ROC / PR | Test ROC / PR | Test F1 @ val threshold | Test video: FirstIn / NegFA / Lead |
|---|---|---|---|---|---|
| `anticipation_prboost_v1` (02-17) | 3746 dims, f1, 50 epochs | 0.829 / 0.518 | 0.942 / 0.763 | 0.672 | 0.712 / 0.269 / 6.4 s |
| `anticipation_prboost_v2` (02-17) | 3746 dims, f1, 60 epochs | 0.853 / 0.620 | 0.944 / 0.714 | 0.569 | 0.678 / 0.296 / 6.6 s |
| `anticipation_multih_hset_v4_ema085_k2_gap5_fap7` (02-17) | EMA 0.85, video_utility | 0.840 / 0.498 | 0.913 / 0.720 | 0.628 | 0.373 / 0.056 / 6.9 s |
| `…_late3` / `…_k3_gap2_fap7` | utility variants | 0.840 / 0.498 | 0.913 / 0.720 | 0.640 / 0.628 | 0.39 / 0.065 ; 0.475 / 0.074 |
| `anticipation_multih_hset_v2` (02-16) | meanpool, video_utility | 0.839 / 0.555 | 0.931 / 0.750 | 0.624 | 0.441 / 0.074 / 6.5 s |
| `anticipation_multih_hset` / `_v3` (02-12) | meanpool | 0.818 / 0.471 | 0.904 / 0.710 | 0.238 / **0.0** | v3 chose threshold 1.0 (no alerts) |
| `anticipation_multih_hset_v4` (02-16) | `vlm_emb`, 1697 npz | 0.750 / 0.493 | 0.833 / 0.520 | 0.206 | 0.172 / 0.01 / 6.8 s |

In `prboost_v1`, 46/77 test XD clips share a movie or YouTube ID with training (computed from
`per_video_test.csv`), and its within-XD test ROC is 0.967. With grouping (§3.1) the same pipeline gets
about 0.5. **Do not cite these numbers.**

### 3.3 Old single-horizon runs (02-02 to 02-10, `labels_preonly_h30_ignore`, 610 violent clips only)

`anticipation_preonly_h30_grace1_ignore{,_fp1..4,_meanpool}`: val ROC 0.48–0.55, test ROC 0.51–0.73.
At the selected threshold, F1 equals the always-positive baseline (0.68–0.70) in every run, so these
models are no better than chance. They were produced by an older version of the script that accepted
1-D labels.

### 3.4 Other

`outputs/qa_feature_test` (from `tuning/question_analysis/test_qa_features.py`, outside this area):
25 VLM yes/no question features on 50 violent and 50 non-violent frames. Logistic test AUC is 0.486;
the best single question ("violence_occurring") has AUC 0.59.

---

## 4. Which model is "best", and how good is it

- **Honest best:** `train_student_original.py` → `outputs/student_mlp/best.pt`. It is a frame MLP on
  mean-pooled VLM embeddings plus hashed VLM tags, with K=7 horizons. On unseen movies its 30-s
  horizon test ROC is 0.62 and PR-AUC is 0.15 (chance is 0.08). It alerts in-window for 60% of violent
  clips with a median lead of about 9 s, and false-alarms on 24% of non-violent clips. The `runs/student_fix1`
  checkpoint is comparable (0.58).
- **Model actually wired into the tools:** `outputs/outputs_temporal/best_temporal_model.pt`. It was
  selected at epoch 1, with val ROC 0.577. Its test ROC of 0.991 is an artifact (§1.5).
- **Highest val among the temporal runs:** VLM + optical flow (`outputs_fused`, val ROC 0.72). This
  hints that motion features help, but it uses misaligned labels and a val set dominated by one clip.
- **Overall:** after several months of variants (focal loss, contrastive, ensembles, fusion, EMA,
  K-consecutive, utility thresholds), there is **no evidence of real 30-s anticipation**. The models
  mostly learn clip-level "violent-movie" appearance, and the measured lead times are 5–10 s.

## 5. Guidance for a new model

1. **Align features and labels by `t`** (`labels[npz['t']]`). Compute horizons in **seconds** from
   `raw_frame_idx / fps`, not from row counts. `audio_features` rows correspond to label rows, so
   index audio by `t` as well when fusing with VLM.
2. **Split by movie or YouTube ID** (`split_group_id(..., 'source_movie')` + `make_splits` from
   `train_student_original.py`), or define a clean group split once and save it. The index JSONs leak
   at the movie level.
3. **Evaluate per video and per source**, and cap or weight each clip's contribution. Pick thresholds
   on val only. Report event-level metrics: first alert in [onset−H, onset], lead-time distribution,
   and false alarms per hour of *non-violent* footage.
4. Keep the −1 "ignore" rows masked, as `train_student_original.py` does, rather than turning them into
   negatives. Fix the UCF-train all-zero labels: either exclude UCF anomaly-train clips or label them as
   video-level positives.
5. Reusable code: `compute_anticipation_video_stats`, `apply_ema_by_video` (causal),
   `metrics_at_threshold`, `sweep_thresholds`, `select_threshold_from_sweep`, `split_group_id`, and
   `make_splits` (all in `training/train_student_original.py`). `TemporalViolenceModel` is a reasonable
   causal-window head *once the data is fixed*.
6. Remember that XD-Violence clips are short (median about 90 s) and often start near the violence, so
   few clips have 30+ s of non-violent lead-in. A 30-s anticipation target needs long untrimmed
   footage, or a specific selection of clips with a long lead-in, such as `curate_anticipation_dataset.py`
   / `vlm_emb/curated` (see the curation doc).
