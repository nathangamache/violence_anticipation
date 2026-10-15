# Baseline benchmark: the honest numbers to beat, and how new models must be scored

Audience: anyone training a new violence-anticipation model in this repo. This file states (1) the exact
evaluation sets and protocol behind the only trustworthy existing result, (2) its numbers, (3) every known
flaw of that evaluation, and (4) the corrected benchmark ("VA30") that new models must use.
Audit date 2026-10-14. Numbers were read from the metrics files and recomputed from `per_sample.csv`
with the venv Python; anything not verified is marked **(inferred)**.

---

## 1. TL;DR: the numbers to beat

| Metric (test split, movie-grouped, threshold chosen on val) | Best existing value | Run |
|---|---|---|
| Frame ROC-AUC, "violence starts within 30 s" | **0.622** (range over 4 runs 0.562 to 0.622) | `baselines/reference_runs/student_mlp` |
| Frame average precision (base rate 0.081) | **0.158** (range 0.096 to 0.158) | `baselines/reference_runs/student_fix1` |
| Frame ROC-AUC, XD-Violence test videos only | **0.56** (range 0.47 to 0.56; chance) | `baselines/reference_runs/student_mlp` |
| ROC-AUC of steps 30 to 60 s before onset vs steps with no onset within 120 s | **0.61** (149 positive steps from 15 videos; about 0.50 on XD only) | `baselines/reference_runs/student_mlp`, `baselines/reference_runs/student_fix1` |
| Violent test videos whose first alarm falls in [onset - 30 s, onset + 1 s] | **24 / 40 = 60 %** | `baselines/reference_runs/student_fix1`, `baselines/reference_runs/student_mlp` |
| "Negative" test videos with any alarm | 15.3 % (`student_fix1`) to 23.7 % (`student_mlp`) | |
| Alarm lead time before onset | mean 10.0 s, median 7.8 s, p90 20.7 s (`student_fix1`) | |
| Violent videos alarmed 30 s or more ahead | **effectively 0 %** (not measurable: the success window ends at 30 s) | |

Reference points (not trained anticipation models):
- **Zero-shot Qwen2.5-VL-7B violent score** (`baselines/reference_runs/vlm_eval*`, no training): XD frame AUC **0.776** for
  current violence and 0.772 for "violence now or within 30 s". It is a detector, but a trained model that
  cannot beat it on XD adds nothing.
- **Published ceiling** (Wang et al., CVPR 2026, see [research/literature.md](research/literature.md)):
  window AUC 0.69 (UCF-Crime) and 0.62 (XD-Violence, cross-domain) at a 60 s horizon. On an honest split,
  0.65 to 0.75 for "onset within 30 to 60 s" would match the state of the art; above about 0.85 means audit
  for leakage first.

Everything else in the removed `final/`, `outputs/`, `simulation_results*/` and random-split `results/anticipation_*`
folders is invalid as a benchmark (section 5 and [EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md)).

---

## 2. The existing honest baseline: exact evaluation set and protocol

### 2.1 Model and inputs
- Trainer: `baselines/train_student_original.py --task anticipation` (the only trainer that indexes labels
  by the embedding `t` key and reports seconds-based alarm statistics).
- Model: `StudentMLP` = LayerNorm(D) -> Linear 256 -> GELU -> Dropout -> Linear 256 -> GELU -> Dropout ->
  Linear 7 (one logit per horizon 2/5/10/20/30/60/120 s), about 1.0 M parameters.
- Features per sampled step: Qwen2.5-VL-7B frame embedding (3584-d) + MD5-hashed multi-hot VLM-teacher tags
  (actions 64, weapons 32, scene 64) = 3744-d. `student_fix2` and `prboost_v3` add notes hashing (4034-d).
- Steps: one per `vlm_emb` row, i.e. every 5th graph step of the full raw video. Spacing is irregular:
  median about 3 s, range 0.2 s to 756 s.

### 2.2 Labels
- `data/processed/labels_multih/<vid>.npy` `[T,7]`, row `t[i]` for embedding i. Column 4 (30 s) is the
  evaluated target: 1 if the next annotated onset is within 30 s. Rows inside violence (`-1`) are dropped.
- Onsets come from `data/raw_videos/xd_violence/annotations.txt` and
  `data/raw_videos/ucf_crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt`.
- Unannotated UCF anomaly videos have all-zero rows and therefore act as negatives.

### 2.3 Split
- `--split_group source_movie --seed 0 --val_frac 0.1 --test_frac 0.1`: groups are `source/<name before "__#">`
  (XD source movie or YouTube id; UCF video id). Groups 1072/134/134, videos 1356/156/158.
- Manifest: `baselines/reference_runs/student_fix1/splits_manifest.json` (identical copies in `baselines/reference_runs/student_fix2`,
  `baselines/reference_runs/student_mlp`, `baselines/reference_runs/anticipation_prboost_v3`). Reuse the saved manifest rather than re-deriving
  it: `vlm_emb/` now also contains `curated/`, which a re-run may pick up **(inferred)**.
- Test composition (verified): 158 videos = 32 annotated XD violent + 8 annotated UCF anomalies (40 videos
  with onsets) + 29 XD normal + 5 UCF Normal + **84 unannotated UCF anomaly videos**. Test samples: about
  8,600, positive rate 0.081. Only **10 test videos have an onset at least 30 s into the video** (6 XD).

### 2.4 Metrics as computed by the trainer
- Frame metrics: pooled ROC-AUC and PR-AUC over test steps (`test_metrics_oracle` holds the threshold-free
  AUCs), precision/recall/F1 at the val-selected threshold (`test_metrics_at_val_threshold`).
- Threshold: selected on val only (`--threshold_select f1 | fbeta | precision_floor | video_utility`).
- Video statistics (`anticipation_video_stats_at_val_threshold`): for each video with onsets, the **first**
  alarm is classified as in-window `[onset - 30 s, onset + 1 s]`, early-false (before `onset - 30 s`), or
  late/miss. Negative videos count as false-alarmed if any step crosses the threshold. Lead = onset minus
  first in-window alarm, in seconds (`raw_frame_idx / fps`).

### 2.5 Reproduce
```bash
# Re-train (GPU, minutes). Use the saved manifest's split; flags reconstructed from metrics.json
# (the trainer never saved its argv).
.venv/bin/python baselines/train_student_original.py --task anticipation \
  --emb_root data/processed/vlm_emb_meanpool --jsonl_root data/processed/vlm_labels_meanpool \
  --labels_dir data/processed/labels_multih --split_group source_movie --seed 0 \
  --select_horizon_sec 30 --threshold_select f1 --out_dir runs/b0_repro --device cuda:0
```
Re-scoring without training (CPU), from any run's `per_sample.csv` (columns `split, vid_id, source, t, gt,
pred_prob_raw, pred_prob_ema`; `t` indexes `labels_multih` rows):
```python
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
s = pd.read_csv("baselines/reference_runs/student_mlp/per_sample.csv"); s = s[s.split == "test"]
Y = np.stack([np.load(f"data/processed/labels_multih/{v}.npy")[t] for v, t in zip(s.vid_id, s.t)])
pos = (Y[:, 5] > 0) & (Y[:, 4] <= 0)      # 30 to 60 s before an onset
neg = (Y[:, 6] == 0)                       # no onset within 120 s
p = s.pred_prob_raw.values
print(roc_auc_score(np.r_[np.ones(pos.sum()), np.zeros(neg.sum())], np.r_[p[pos], p[neg]]))  # 0.611
```

---

## 3. Results of the four leak-free runs

| Run | Features | Threshold mode | Val AUC / AP | **Test AUC / AP** | Test XD AUC | 30 to 60 s bucket AUC (XD) | Test P / R / F1 at val threshold | First alarm in window | Neg. videos alarmed | Lead median |
|---|---|---|---|---|---|---|---|---|---|---|
| `baselines/reference_runs/student_mlp` | `vlm_emb_meanpool` + tags (3744) | f1 | 0.763 / 0.410 | **0.622** / 0.151 | 0.564 | 0.611 (0.498) | 0.126 / 0.423 / 0.195 | 24/40 (60 %) | 28/118 (23.7 %) | 8.9 s |
| `baselines/reference_runs/student_fix1` | `vlm_emb` + tags (3744) | fbeta | 0.823 / 0.465 | 0.578 / **0.158** | 0.521 | 0.605 (0.501) | 0.109 / 0.270 / 0.156 | 24/40 (60 %) | 18/118 (15.3 %) | 7.8 s |
| `baselines/reference_runs/student_fix2` | `vlm_emb` + tags + notes (4034), 2 consecutive | fbeta | 0.804 / 0.459 | 0.592 / 0.149 | 0.497 | 0.584 (0.458) | 0.085 / 0.266 / 0.128 | 20/40 (50 %) | 29/118 (24.6 %) | 8.4 s |
| `baselines/reference_runs/anticipation_prboost_v3` | as fix2 | precision_floor 0.75 | 0.818 / 0.456 | 0.562 / 0.096 | 0.495 | 0.623 (0.506) | 0.034 / 0.088 / 0.050 | 12/40 (30 %) | 9/118 (7.6 %) | 8.9 s |

How to read it:
- Always-positive F1 on test is 0.151, so F1 at the val threshold barely beats predicting "positive" everywhere.
- The pooled AUC is partly a source-mixture effect (XD has higher scores and a higher positive rate than UCF).
  Within XD the model is at chance at locating the pre-onset window, although its per-video max score
  separates violent from non-violent XD clips with AUC 0.71 to 0.76: it learned "this clip looks violent".
- Val XD AUC is 0.81 to 0.85 while test XD AUC is 0.50 to 0.56. With about 32 to 39 violent XD clips per
  split, a single grouped split has a variance of about 0.2 AUC.

### 3.1 Legacy comparison set (for continuity with old reports only)
`simulation_data/xd_violence_8hr_sim.npz` (test-only XD clips stitched into 32,092 steps, 13 events) scored by
`sim/comprehensive_eval.py`: VLM student `runs/outputs_xd/best_model.pt` AUC 0.853 / AP 0.242; BiLSTM
`outputs/outputs_temporal` AUC 0.834 / AP 0.131, 11/13 events, 8.6 "FA/h" at a test-picked threshold 0.388.
Labels are misaligned, steps are not seconds and the event metric is lenient. Never cite these as accuracy.
Since the 2026-10 cleanup the sim, both models and the score traces live under `data/archive/repo_legacy/`
(same relative paths, until 2027-01-12); `sim/comprehensive_eval.py` is recoverable with
`git show pre-cleanup:sim/comprehensive_eval.py`.

---

## 4. Known flaws of the existing baseline evaluation

1. **Too few eligible events.** The grouped test split has 10 videos with at least 30 s of pre-onset footage
   (6 XD); the 30 to 60 s bucket is 149 steps from 15 videos. Any 30 s claim on it is noise.
2. **Most "negative" videos are not negatives.** 84 of the 118 test videos without onsets are unannotated
   UCF-Crime anomaly videos (real crimes, many violent). Only 34 are clean negatives (29 XD normal,
   5 UCF Normal). The false-alarm rate and the pooled AUC are both contaminated by this label noise.
3. **Lead time is capped by construction.** Success is a first alarm in `[onset - 30 s, onset + 1 s]`; an
   earlier first alarm is "early false". The protocol cannot report detection at 30 s or more ahead, and it
   credits alarms up to 1 s after onset.
4. **Irregular, coarse clock.** Labels live on the graph-row grid of `labels_multih`; embeddings are sampled
   every 5th row (median about 3 s, up to 756 s apart), so horizons in seconds are approximate and lead
   times are quantized. 77 videos (22 with violence) have no embeddings at all.
5. **No false alarms per hour.** Negative-video false alarms are a per-video yes/no, not a rate on long
   footage, so operating points cannot be compared with deployment needs.
6. **Single split, single seed, no confidence intervals**, and the threshold-selection mode differs between
   runs (f1, fbeta, precision_floor), so the four runs are not one protocol.
7. **Pooled AUC across sources** rewards telling XD from UCF apart. Per-source numbers are at chance.
8. **No shortcut controls.** XD contains 376 movie clips (soundtrack, cuts, letterboxing) and YouTube
   compilations; there is no scene-only, time-only or first-seconds-only control.
9. **Domain mismatch with the target.** Neither XD movie clips nor UCF CCTV clips resemble the target
   bodycam / road-rage / phone footage; the only domain-matched set (92 curated YouTube videos) has no human
   labels yet.

Flaws of the legacy protocols (`final/`, `outputs/`, simulators) are listed in
[EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md) section 0 (D1 to D8).

---

## 5. Evaluation sets in the repo and their status

| Set | Where | Status |
|---|---|---|
| Movie-grouped split (1356/156/158) | `baselines/reference_runs/student_fix1/splits_manifest.json` | Leak-free by source; basis of section 3. Too few eligible events; noisy negatives. |
| Gen-1 clip split (1438/180/180) | `legacy/train_index.json`, `legacy/val_index.json`, `legacy/test_index.json` (`eval_index.json` = test, removed) | **Leaky**: 44 of 83 XD test clips share a source with train. Test has 12 eligible videos (3 XD, 9 UCF). Do not use. |
| Random video split (1336/167/167) | older `results/anticipation_*` (removed; `git checkout pre-cleanup -- results/<name>`) | **Leaky**. Do not use. |
| Stitched simulations | `simulation_data/*.npz`, archived to `data/archive/repo_legacy/simulation_data/` | Misaligned labels; most contain 81 % training videos. Legacy continuity only. |
| Zero-shot teacher eval | `baselines/reference_runs/vlm_eval*` | Valid reference (no training), detection-oriented. |
| YouTube probe `probe_clip_01` | `probe_clip_01_*.npy`, archived to `data/archive/repo_legacy/` | n = 1, smoke test only. |

---

## 6. The corrected benchmark: VA30 (required for every new model)

The goal is a protocol where "X % of fights were flagged 30 s or more ahead at Y false alarms per hour"
is a true statement with a confidence interval. It follows the recommendations in
[research/literature.md](research/literature.md) section 9.4 and
[codebase/simulation_and_evaluation.md](codebase/simulation_and_evaluation.md) section 7.
Implement it once as a standalone module (suggested `eval/va30.py`) that takes per-video score arrays with
timestamps and returns every metric below; never re-implement metrics inside trainers.

### 6.1 Ground truth and clock
- **Onsets in seconds from the raw annotation files**, never from `metadata/xd_violence.json` or
  `labels_multih` rows (loader snippet in [codebase/data_and_labels.md](codebase/data_and_labels.md) section 3).
  Every violent segment `[s, e]` counts; the next onset `t0(t)` is the first segment start after `t`.
- **Scoring clock: 1 Hz ticks in video seconds.** A score at tick `t` may only use media from `[0, t]`.
  Models with sparser features hold their last score (sample-and-hold) so all models are scored on the
  same ticks.
- Labels per tick, per horizon H in {5, 10, 20, 30, 60}: `y_H(t) = 1` if `0 < t0(t) - t <= H`, else 0.
- **Masked ticks:** inside a violent segment, and the 10 s after a segment ends (aftermath). These ticks go
  to a separate detection report, never to anticipation metrics.
- **Right-censoring:** in videos without violence, ticks with less than H seconds of remaining footage are
  masked for horizon H.

### 6.2 Videos
- **Eligible anticipation events:** onsets with at least 30 s of observed, non-violent footage before them
  (also report the at-least-60 s subset). Locally that is 112 videos (XD 53, UCF 59; 49 interpersonal).
- **Clean negatives only:** XD `_label_A` clips, UCF `Normal_*` videos (including the 950 unused ones in
  `Training-Normal-Videos/` and `Testing_Normal_Videos/`), and later UBI-Fights normals. **Exclude
  unannotated UCF anomaly videos** unless HIVAU-70k supplies their onsets.
- Annotated videos with onset under 30 s still contribute frame-level ticks and detection metrics.
- **VA30-local** = the sets above, available today. **VA30-full** adds HIVAU-70k onsets for the XD and UCF
  training splits, UBI-Fights, and (when available) CCTV-Fights; see [research/README.md](research/README.md).
- **Real-world hold-out:** the 92 curated YouTube videos in `data/anticipation_curation/`, after human onset
  annotation. Never used for training or tuning; reported separately.

### 6.3 Splits
- Group key: XD = text before `__#` (source movie or YouTube id); UCF = video id; UBI = video id;
  YouTube = channel when known, else video id.
- **5-fold grouped cross-validation**, stratified by source and by "has an eligible event". Commit the fold
  file to the repo (suggested `splits/va30_folds.json`) and never regenerate it silently.
- Inside each training fold, hold out one group-disjoint validation slice for checkpoint and threshold
  selection. Report out-of-fold predictions pooled over the 5 test folds, for 3 seeds.

### 6.4 Metrics (all reported, in this order)
1. **Far-window AUROC (headline frame metric):** positives = ticks with `30 < t0 - t <= 60`;
   negatives = ticks in clean negative videos. Secondary: negatives = all ticks with no onset within 120 s.
2. **Per-horizon AUROC and AP** for H in {5, 10, 20, 30, 60}: positives `y_H = 1`, negatives = ticks with no
   onset within 120 s (ticks with an onset between H and 120 s are excluded, so legitimate early alarms are
   not penalized). Report pooled and per source.
3. **Event-level anticipation (primary operational metric).** Alarm policy, fixed on validation:
   threshold theta, k-of-n smoothing (default 2 of 3 ticks), episodes merged across gaps under 10 s,
   30 s refractory period. For each eligible event, take the first alarm episode that starts in
   `[t0 - 120 s, t0)`; lead = `t0 - start`. Report **Success@30** (lead at least 30 s), Success@10,
   Success@0, and the lead distribution (median, IQR, mTTA). Uncapped, in seconds.
4. **False alarms per hour:** episodes that start outside every `[t0 - 120 s, e + 10 s]` window, divided by
   the hours of footage outside those windows. Primary: clean negative videos only. Secondary: all footage.
5. **Operating points:** choose theta on validation so that val FA/h is 1, 2 and 5; report test Success@30,
   detection and FA/h at each. Also plot the AMOC-style curve (Success@30 vs FA/h).
6. **Random-predictor reference:** at alarm rate FAR (per hour) and acceptance window W (hours),
   `P_random = 1 - exp(-FAR * W)`. For Success@30 the window is `[t0 - 120, t0 - 30]` (W = 90 s), so at
   2 FA/h the chance level is about 4.9 %.
7. **Detection (reported separately, never called anticipation):** any alarm in `[t0 - 120 s, e]`, and
   detection AUROC on in-violence ticks versus clean negative ticks.
8. **Calibration:** Brier score and a reliability plot for P(onset within 30 s).
9. **Uncertainty:** 95 % cluster-bootstrap confidence intervals over videos (1000 resamples) for every
   number, plus mean and standard deviation over seeds.
10. **Breakdowns:** per source (XD movie, XD YouTube, UCF CCTV, UBI CCTV, curated real-world),
    interpersonal versus other classes, audio present versus absent.

### 6.5 Required control baselines (scored with the same code)
| Name | Definition | Purpose |
|---|---|---|
| B-prior | constant score | sanity check (AUROC 0.5) |
| B-time | score = seconds since video start | detects position shortcuts (gen-1 position-only baseline scored 0.61) |
| B-scene0 | logistic regression on a frame embedding of the first 2 s only, broadcast to all ticks | detects provenance shortcuts (title cards, watermarks, movie grading) |
| B-teacher | zero-shot Qwen2.5-VL `violent.score` from `processed/vlm_labels*` (sample-and-hold) | strong detection-only reference |
| **B0** | `StudentMLP` retrained on VA30 folds with the existing `vlm_emb_meanpool` + tags, labels rebuilt in seconds from the annotations | the gen-2 honest baseline under the new protocol; every new model must beat it |

A new model is an improvement only if it beats B0 **and** B-scene0 on the far-window AUROC with
non-overlapping bootstrap intervals, and its Success@30 at 2 FA/h is clearly above the random-predictor
level. A far-window AUROC above about 0.85 triggers a leakage and shortcut audit (inspect the
highest-scoring far-from-onset ticks for overlays, watermarks, letterboxing and music) before it is reported.

### 6.6 Reporting template
Every experiment writes `metrics.json` with: git commit, exact command line, fold file hash, feature set and
extraction config, model config, seeds, threshold-selection rule, and all metrics in 6.4 with CIs. Save
per-tick out-of-fold scores (`video_id, t_sec, score, fold, seed`) so any metric can be recomputed later.
