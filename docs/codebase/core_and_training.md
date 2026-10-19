# Core model + original training pipeline (ViolenceTransformer)

> Paths in this file describe the tree before the 2026-10 cleanup. Surviving code is under `legacy/<same path>`;
> the honest runs are under `baselines/reference_runs/`; removed tracked files: `git show pre-cleanup:<path>`;
> ignored weights and arrays: `data/archive/repo_legacy/<same path>`
> until 2027-01-12. See [../REPO_CLEANUP_PLAN.md](../REPO_CLEANUP_PLAN.md).

Scope: `core/*.py`, `configs/*.yaml`, `training/{dataset,train,evaluate,eval_anticipation}.py`,
`tests/*.py`, `scripts/{benchmark,export_trace_graph,visualize_graphs,inspect_risk_curves}.py`, the producer
script `scripts/preprocess_anticipation.py` + `scripts/generate_labels.py` (read because they define the data this
pipeline consumes), and an audit of `README.md`.

This is the **first-generation pipeline** of the project (Nov–Dec 2025). Later generations (VLM embeddings,
temporal LSTM/students, LoRA, fusion, simulation) are documented in the sibling docs in `docs/codebase/`.

Every claim below was checked against the code or the files on disk. Anything marked **(inferred)** is a
deduction I could not prove directly.

---

## 0. TL;DR for the next agent

* **The README describes a system that was never built.** There is no graph transformer, no DeepSORT, no real
  audio, no sliding-window training, no working realtime/ONNX/benchmark path. What was built and trained is a
  **2-layer temporal `nn.TransformerEncoder` over per-frame vectors** = `Linear(mean of person nodes) +
  Linear(global CLIP vector) + Linear(zero audio vector)`.
* **It was trained to completion twice**: `checkpoints/model_epoch{1..60}.pt` (60 epochs, 2025-12-18) and
  `best_checkpoints/model_epoch{1..40}.pt` (a separate 40-epoch run, 2025-12-15). **No metrics were ever saved**
  (train.py only prints to stdout, and there is no validation loop).
* I re-scored the checkpoints on CPU (Section 10). Epoch 60 gets **frame AUC 0.975 / clip AUC 0.959** on
  `test_index.json`. **That number is not trustworthy**:
  1. the checkpoints were almost certainly trained on *every* clip, including today's test clips (inferred:
     `train.py` rebuilds `train_index.json` from every graph file, and the split files are dated 2026-02-17,
     two months after training);
  2. ablations show the model relies almost entirely on the **global CLIP "scene" vector**. That vector is one
     constant per clip, averaged over the whole 40 s window, *including the violent frames after the event*.
     Zeroing it drops frame AUC to **0.47**, which is chance. Zeroing all person/action features only drops it to 0.90;
  3. the encoder is **bidirectional**, so every timestep sees the future. Re-running it causally (prefix-only)
     drops clip AUC from 0.96 to **0.66**.
  The pipeline is effectively a clip-level classifier that leaks future information. It is **not** a 30 s
  anticipation model.
* **Danger:** running `python -m training.train` today **overwrites `train_index.json`** in the repo root with
  all 1798 clips. Later pipelines use that file as their curated 1438-clip training split, so this would destroy
  the split and leak test clips into training. Do not run it unmodified.
* Reusable: `data/windowed/{ucf_crime,xd_violence}/*.mp4` (1449 trimmed 40 s clips centred on events),
  `data/processed/tracks/*.json` (YOLOv8n person boxes), metadata JSONs. The graphs, actions and labels are
  low-value (see Sections 3 and 12).

---

## 1. File map

| File | Lines | What it is | Status |
|---|---|---|---|
| `core/model.py` | 113 | `ViolenceTransformer` (temporal transformer over frame vectors) | Used by train/eval; works |
| `core/graph.py` | 251 | `build_graph_sequence()` (per-frame node matrix + distance adjacency), `graph_to_text()` | Builder used by preprocessing; `graph_to_text` is trivial |
| `core/extractors.py` | 370 | YOLOv8n detection, VideoMAE crop embeddings, CLIP scene embedding, audio stubs, `live_*` versions | Used by preprocessing; **loads 3 models onto the GPU at import time** |
| `core/utils.py` | 23 | `save_json`, `load_json`, `center_of_bbox`, `get_device()` | `get_device` defaults to **cuda:1** |
| `core/__init__.py` | 0 | empty | |
| `configs/training.yaml` | 25 | training hyperparameters | Only some keys are read (Section 9) |
| `configs/paths.yaml` | 12 | relative data paths | Read by train/evaluate |
| `configs/model.yaml` | 20 | model dims | **Never read by any code** |
| `configs/features.yaml` | 6 | modality toggles | **Never read by any code** |
| `training/dataset.py` | 250 | `GraphSequenceDataset` (one whole clip per item) | Works; slow init |
| `training/train.py` | 271 | training loop (weighted MSE + "spam" penalty) | Works; **overwrites `train_index.json`**; no validation |
| `training/evaluate.py` | 272 | `evaluate_model()` (frame MSE + ROC-AUC), CLI | MSE is computed on logits (bug) |
| `training/eval_anticipation.py` | 384 | TTD / false-alarm / per-class AUC evaluator | **Metadata misaligned** with clips and wrong time base (bugs) |
| `scripts/preprocess_anticipation.py` | 309 | producer of `data/processed/{tracks,actions,scenes,audio,graphs,labels}` | Would crash today at label generation |
| `scripts/generate_labels.py` | 78 | `generate_risk_curve()` | Its current version **did not** produce the label files on disk |
| `scripts/inspect_risk_curves.py` | 161 | prints/plots GT risk curves | Works; wrong time axis for most clips |
| `scripts/benchmark.py` | 82 | FPS benchmark | **Broken** |
| `scripts/export_trace_graph.py` | 32 | ONNX export | **Broken** |
| `scripts/visualize_graphs.py` | 22 | matplotlib graph plot function | No `main`; expects a data format nothing produces |
| `scripts/quick_inspect.py` | 101 | plot predicted vs GT curve for one clip | Works (hard-coded absolute path); not in my original scope |
| `scripts/split_train_val.py` | 50 | 80/20 random split of `train_index.json` | Superseded by `tools/make_nonoverlap_splits.py` (inferred) |
| `tests/*.py` | 235 | 4 test files | **All stale; none pass** (Section 11) |
| `realtime/pipeline.py` | ~150 | "realtime" loop the README points to | **Broken** (Section 13) |

Checkpoints and other artifacts:

| Path | Content |
|---|---|
| `checkpoints/model_epoch{1..60}.pt` | 60 × 11.9 MB `state_dict`s of `ViolenceTransformer` (2025-12-18) |
| `best_checkpoints/model_epoch{1..40}.pt` | 40 × 11.9 MB, a **different earlier run** (2025-12-15). Hashes differ from `checkpoints/` for the same epoch numbers. Despite the folder name, this is every epoch of that run, not a selection of the best ones (inferred) |
| `data/plots/*_risk_curve.png` | 6 plots from `inspect_risk_curves.py --auto_samples` (2025-12-14) |
| `train_index.json` / `val_index.json` / `test_index.json` / `eval_index.json` | 1438 / 180 / 180 / 180 `{graph_path,label_path}` items (2026-02-17). `eval_index.json` is identical to `test_index.json` |

`runs/`, `results/`, `logs/`, `experiments/output/` contain **no** outputs of this pipeline. They belong to the
later VLM/student/LSTM pipelines. `grep` finds no saved AUC/TTD from `training/evaluate.py` or `eval_anticipation.py`
anywhere in the repo.

---

## 2. End-to-end data flow (as it actually ran)

```
data/metadata/{xd_violence,ucf_crime}.json         (event_frame, total_frames, label, fps, source)
data/raw_videos/{xd_violence,ucf_crime}/videos/*.mp4
        │  scripts/preprocess_anticipation.py
        │   1. compute window: positives [event-30s, event+10s], negatives = middle 40s
        │   2. ffmpeg -ss … -c copy  → data/windowed/<source>/<vid>.mp4  (if not the whole video)
        │   3. core.extractors.extract_tracks   (YOLOv8n, every 4th frame)   → processed/tracks/<vid>.json
        │   4. core.extractors.extract_actions  (VideoMAE per person crop)    → processed/actions/<vid>.npy
        │   5. core.extractors.extract_scene    (CLIP ViT-B/16, every 16th frame, MEAN) → processed/scenes/<vid>.npy
        │   6. core.extractors.extract_audio_placeholder() (zeros(64))       → processed/audio/<vid>.npy
        │   7. core.graph.build_graph_sequence(...)                          → processed/graphs/<vid>.npy
        │   8. generate_risk_curve(...)  (older version, see §6)             → processed/labels/<vid>.npy
        ▼
train_index.json  ── training/train.py (rebuilds it from ALL graphs) ──► checkpoints/model_epoch{N}.pt
val_index.json / test_index.json ── training/evaluate.py, training/eval_anticipation.py ──► stdout only
```

Run commands: all scripts use package imports (`from core...`, `from training...`) and relative paths
(`configs/…`, `data/processed/…`, `yolov8n.pt`). They **must be run from the repo root as modules**:
`python -m training.train`, **not** `python training/train.py`. I checked that the script form fails with
`ModuleNotFoundError: No module named 'training'`. No `.pth` or `PYTHONPATH` setup exists in `.venv`.

---

## 3. Feature extractors — `core/extractors.py`

Import-time side effects: it prints `[Load] YOLO + VideoMAE + CLIP (no DeepSORT)`, then loads `YOLO("yolov8n.pt")`
(path relative to the CWD; the file sits in the repo root), `MCG-NJU/videomae-base` (HF hub, fp16) and
`clip.load("ViT-B/16")` onto `get_device()` (default **cuda:1**, set with env `VA_GPU`). It also calls
**`torch.set_grad_enabled(False)` globally**, so any training process that imports this module silently loses gradients.

Constants: `FRAME_STRIDE = 4` (detection/actions), `SCENE_STRIDE = 16` (CLIP).

| Function | Input | Output | Notes |
|---|---|---|---|
| `extract_tracks(video_path)` | mp4 | `list[T_s]` of `list[{"track_id": int, "bbox": [x1,y1,x2,y2]}]`, one entry per 4th frame | YOLOv8n with default conf, class 0 only. `track_id` is the **index of the box in that frame's detections**, not a track. **No tracking, and no DeepSORT** (although `deep_sort_realtime` is installed) |
| `extract_actions(video_path, tracks)` | mp4 + tracks | `list[T_s]` of `{track_id: float32[768]}` | Seeks each strided frame with `cap.set(POS_FRAMES)`. Each person crop is **one frame replicated 16 times** and fed to VideoMAE, then mean-pooled over the last hidden state. The model is `videomae-base`, which is the **self-supervised pretrain, not Kinetics-finetuned**. So the "action embedding" carries **no motion information** and is mostly an appearance embedding |
| `extract_scene(video_path)` | mp4 | `float32[512]` | CLIP image embedding of every 16th frame, **averaged over the whole clip**. It is one vector per video, so it includes post-event frames (see leakage, §12) |
| `extract_audio_placeholder(dim=64)` | – | `zeros(64)` | **This is what was used.** All 1798 `processed/audio/*.npy` files are zero vectors of shape (64,); I verified every file |
| `extract_audio_from_file(video_path, sr=16000)` | mp4 | `float32[128]` (time-mean of the power mel spectrogram, librosa default `n_mels=128`, no log) | Never called. Writes to a fixed `/tmp/va_temp_audio.wav` (race condition if run in parallel). Its 128-d output does not match the model's `audio_dim=64` |
| `live_track(frame)` / `live_actions(frame, tracks)` / `live_scene(frame)` | one BGR frame | same as above, per frame | Same YOLO/VideoMAE/CLIP logic |
| `live_audio(seconds=0.3)` | microphone via `sounddevice` | `float32[128]` mel mean, or `zeros(64)` if PortAudio is missing | In this venv **PortAudio is missing**, so it returns zeros. The returned dimension also changes between 64 and 128 depending on the path |
| `crop_bbox(frame, bbox)` | | clipped crop or `None` | |

---

## 4. Graph builder — `core/graph.py`

`build_graph_sequence(tracks, actions, scene, audio, max_nodes=32) -> list[dict]`

* `tracks`: list-of-frames of person dicts (new format), or a list of `{"frame_idx", "tracks"}` dicts (legacy).
* **Frames with no valid person are skipped** (`continue`). The output sequence is therefore **not uniformly
  sampled in time**. 1416 of the 1695 non-empty graph files have fewer steps than their tracks file.
* Node feature = `_bbox_to_feature(bbox)` (raw pixel `[x1,y1,x2,y2]`, **not normalized**) concatenated with the
  768-d action vector, giving **772-d**. When a track has no action vector, the node is **4-d** and the dataset
  zero-pads it later.
* Adjacency `adj[i,j] = 1/(1+‖c_i − c_j‖)`, diagonal 0. Despite the name `centers`, `c` is `(x1, y1)` (the **top-left
  corner**), not the centre. `adj` is stored under key **`"adj"`**, but the dataset reads `"edges"`, so it is
  **never used** (§7).
* At most 32 nodes per frame (the first 32 detections).
* Output per step: `{"frame_idx": int (index into the strided list), "nodes": float32[N,772], "adj": float32[N,N],
  "scene": float32[512] (same vector every step), "audio": float32[64] (zeros)}`.

`graph_to_text(graph_seq, max_frames=8, max_people=5)` returns strings like "Frame k: about N people visible…".
It contains no semantic content and expects a **list**. `realtime/pipeline.py` passes a single dict, which raises
`TypeError: unhashable type: 'slice'`.

**On-disk graph files** (`data/processed/graphs/*.npy`, 1798 files, 3.9 GB, pickled object arrays; I loaded all of them):

| Format | Count | Description |
|---|---|---|
| new (`frame_idx,nodes,adj,scene,audio`) | 1695 | as above; median length about 194 steps, max 20045 |
| empty (`len == 0`) | 77 | 51 UCF and 26 XD; no person was ever detected |
| legacy (`audio,nodes,scene,scene_text,timestamp`) | 26 (all XD) | `nodes` is a **list of dicts** `{track_id,bbox,position,velocity,action_logits,action_names}`, one step per **raw** frame, un-windowed. `GraphSequenceDataset._coerce_nodes` cannot convert dicts, so **every node in these 26 clips becomes a single zero vector** |

Mean persons per step is about 3.8 and the max is 25 (from an 80-file sample).

---

## 5. The model — `core/model.py::ViolenceTransformer`

```
for each step t:  v_t = mean_i(node_proj(nodes_t[i]))  + scene_proj(scene_t) + audio_proj(audio_t)    # [H=256]
seq = [v_1..v_T] + pos_embed(0..T-1)       # learned absolute positions, max_len=4096 (longer clips are truncated)
h   = nn.TransformerEncoder(2 layers, d_model=256, nhead=4, ff=1024, dropout=0.1, batch_first)(seq)   # NO mask
logit_t = out_head(h_t)                    # [T]; forward(return_logits=False) applies sigmoid
```

* There is **no graph attention**. Adjacency and edges are ignored. Because `node_proj` is affine,
  `mean(node_proj(x)) == node_proj(mean(x))`, so the "graph" reduces to the **mean person feature per frame**.
* **Non-causal**: full bidirectional self-attention over the whole clip. Each prediction sees future frames,
  including the violent event itself. This is invalid for anticipation and for real-time use.
* Default constructor dims (`node_dim=772, scene_dim=512, audio_dim=64`) differ from the README, `configs/model.yaml`
  (770/512/128, 4 graph layers, 4 temporal layers with 8 heads) and the tests (770/128).
* The trained checkpoints have 2,974,209 parameters (`node_proj 256×772`, `scene_proj 256×512`, `audio_proj 256×64`,
  `pos_embed 4096×256`, 2 encoder layers, `out_head 1×256`). The weights don't reveal `num_heads`; train.py
  uses 4.
* Batch size is effectively 1 (a Python loop over frames builds the sequence).

---

## 6. Label definition (what is on disk vs what the code says)

**On disk** (`data/processed/labels/<vid>.npy`, float32, **length equals the graph sequence length**; I checked
all 1798 files):

* Negatives (`meta.label == 0`) are all zeros.
* Positives are a **linear ramp 0 → 1 that ends at the event step, followed by a constant 1.0 until the end of the
  clip**. All 344 non-zero label files end at 1.0, so there is **no post-event decay**. The ramp starts at step
  0/1 in practically every clip, because positive windows were cut at most 30 s before the event.
* The event step is placed **proportionally**: `e ≈ round(local_event_frame / local_total_frames × T_graph)`
  (inferred from argmax positions, e.g. `v=CcD-YNlkMPY…`: 720/961 × 193 ≈ 144 = argmax). This assumes graph steps
  are evenly spaced in time. They are not, because steps without people were dropped (§4). In extreme cases a
  clip has 241 tracked steps but 11 graph steps.
* Binarized target `label ≥ 0.5` means "within the last ~15 s before the event, or any time after it".
* Event position inside positive clips: median at 0.51 of the clip; 42 % of clips have it in [0.6, 0.9] (≈0.75 by
  construction of the 30 s + 10 s window); **55 positives have the event at step 0**, so anticipation is impossible for them.

**Current code** (`scripts/generate_labels.py::generate_risk_curve(meta, num_frames=None, anticipation_window_sec=30,
post_event_window_sec=10)`) builds a ramp over `30·fps` raw frames, then a **linear decay to 0 over 10 s**.
`preprocess_anticipation.py` calls it as `generate_risk_curve(local_meta, graph_seq)`, passing a list as
`num_frames`. I confirmed this raises `TypeError: int() argument must be … not 'list'`. **The label files on disk
were therefore produced by an earlier, uncommitted version** that rescaled to graph steps and held 1.0 after the
event (inferred). Rerunning preprocessing for any clip without an existing label file will crash.

**Upstream label quality (metadata)**: these numbers come from `data/metadata/*.json`; how those files were built
is covered in the data docs.

* UCF-Crime: only 140 / 1000 videos have `label = 1`. **810 anomaly videos are labelled 0** and are treated as
  all-negative. These include 45 Fighting, 47 Assault, 48 Abuse, 27 Shooting and 145 Robbery videos (only the
  temporally annotated videos got `label = 1`, inferred).
* XD-Violence: 226 / 798 have `label = 1` (170 with `event_frame > 0`, 56 with `event_frame = 0`). **274 clips whose
  file name carries a violent tag (B1/B2/B4/B5/B6/G) are labelled 0**.
* Usable after the dataset filter (≥16 steps): **1584 clips, 295 with a non-zero curve, 386,727 steps, 11.4 % of
  steps ≥ 0.5**.

---

## 7. Dataset — `training/dataset.py::GraphSequenceDataset(index_json, min_frames=16)`

* `index_json` is a list of `{"graph_path", "label_path"}` (relative paths such as `data/processed/graphs/<vid>.npy`).
* `__init__` **fully loads every graph file** to validate it. The full set is 3.9 GB, so expect about 2 minutes. It
  silently drops missing, corrupt, empty or short (<16 step) items and infers `node_dim/scene_dim/audio_dim` from
  the first item (fallback 772/512/64).
* `__getitem__` returns `(frames, labels)`:
  * `frames`: a list of length `T = min(len(graph), len(labels))` of dicts with `nodes [N,node_dim]` (padded or
    truncated; zero `[1,node_dim]` if empty or unparseable), `edges [N,N,4]` (**always zeros**, because the graph key
    is `adj`, not `edges`), `scene [512]` and `audio [64]`.
  * `labels`: float32 `[T]`.
* `infer_dims(dataset)` simply returns the stored dims. `train.py`, `evaluate.py` and `eval_anticipation.py` each
  define their own duplicate `infer_dims` that iterate items.

---

## 8. Training — `training/train.py`

Command (from the repo root): `python -m training.train`. **Read the warning in §0 first.**

1. Reads `configs/training.yaml` and `configs/paths.yaml`.
2. `build_index_file(graphs_dir, labels_dir, "train_index.json")` **lists every graph file that has a label and
   overwrites `./train_index.json`**. There is no train/val split, so training uses all usable clips (1584).
3. `GraphSequenceDataset`, `DataLoader(batch_size=1, shuffle=True, collate_fn=lambda b: b)`. Values >1 for
   batch_size are forced back to 1.
4. `ViolenceTransformer(node_dim, scene_dim, audio_dim, hidden_dim=cfg.get("hidden_dim",256),
   num_heads=cfg.get("num_heads",4), num_layers=cfg.get("num_layers",2), dropout=0.1)` on `get_device()` (cuda:1).
5. `AdamW(lr=1e-4, weight_decay=0.005)`, `epochs=60`. No scheduler, no gradient clipping, no AMP.
6. **Loss** (on `p = sigmoid(logits)`, after aligning lengths):
   * `mse = mean((p − y)² · (1 + 1.5·y))` (anticipation-weighted MSE regression to the ramp);
   * `spam = mean(relu(p − 0.35)²)` over steps with `y ≤ 0.02` ("anti-false-alarm" penalty);
   * `loss = mse + 0.8 · spam`.
7. **There is no validation, early stopping or best-model selection.** After every epoch it prints average losses
   and writes `checkpoints/model_epoch{N}.pt` (a `state_dict`).

`anticipation_window_sec` is only printed. Labels are precomputed (§6).

---

## 9. Configs — which keys are actually used

| Key | Used by | Value |
|---|---|---|
| `training.epochs, lr, weight_decay, batch_size(→1)` | train.py | 60, 1e-4, 0.005, 1 |
| `training.anticipation_weight_factor` | train.py loss | 1.5 |
| `training.spam_penalty_lambda / _margin, low_target_eps` | train.py loss | 0.8 / 0.35, 0.02 |
| `training.anticipation_window_sec` | printed only | 30.0 |
| `training.bin_thresh, lambda_mse, lambda_bce, pos_class_weight, early_stopping.*, window` | **nobody** | dead |
| `training.hidden_dim/num_heads/num_layers/dropout/num_workers` | train.py (absent, so defaults 256/4/2/0.1/0) | – |
| `paths.graphs, paths.labels` | train.py, evaluate.py | `data/processed/{graphs,labels}` (relative) |
| other `paths.*` | nobody | |
| `configs/model.yaml` (node_dim 770, audio 128, graph_layers 4, temporal 4×8 heads) | **nobody**; it also contradicts the checkpoints | dead |
| `configs/features.yaml` (use_* toggles) | **nobody** | dead |

---

## 10. Was it trained? What did it score?

**Evidence of training:** 60 complete epochs in `checkpoints/` (matching `epochs: 60`) and 40 in
`best_checkpoints/` (earlier run). Both have the architecture above (node 772 / scene 512 / audio 64 / 2 layers).
**No stdout logs, metrics JSON or eval outputs survive.**
Run timing (file mtimes, recorded before the 2026-10 cleanup): `checkpoints/` epoch 1 was saved 2025-12-18 14:41
and epoch 60 at 18:05, about 3.5 min per epoch and about 3.5 GPU-h in total. `best_checkpoints/` epoch 1 was saved
2025-12-15 13:06 and epoch 40 at 15:14, about 3.3 min per epoch and about 2.2 GPU-h. The three epochs in the
table below are at `data/archive/repo_legacy/checkpoints/model_epoch{10,60}.pt`
and `.../best_checkpoints/model_epoch40.pt`. The other 97 epochs (1.16 GB, never evaluated) were archived next to
them, in the same two folders.

**My re-evaluation** (2026-10-14, CPU only, read-only, scratch script not in repo; the code is reproduced below).
Inputs: `test_index.json` → 168 usable clips, 34,094 steps, 4,231 positive steps (y ≥ 0.5), 28 positive clips.
"clip AUC" ranks clips by max predicted probability against "clip has any non-zero label".

| Checkpoint | frame ROC-AUC | frame AP | MSE (prob) | clip AUC | mean p (neg / pos steps) |
|---|---|---|---|---|---|
| `checkpoints/model_epoch60.pt` | **0.9745** | 0.847 | 0.030 | 0.959 | 0.086 / 0.755 |
| `best_checkpoints/model_epoch40.pt` | 0.940 | 0.638 | 0.308 | 0.900 | 0.590 / 0.668 (badly calibrated) |
| `checkpoints/model_epoch10.pt` | 0.753 | 0.335 | 0.088 | 0.783 | 0.177 / 0.282 |
| position-only baseline (score = t/T) | 0.613 | – | – | – | – |

Ablations of epoch 60 (same clips):

| Variant | frame AUC | clip AUC |
|---|---|---|
| full | 0.9745 | 0.959 |
| scene (CLIP) zeroed | **0.469** | 0.688 |
| person nodes zeroed | 0.899 | 0.871 |
| both zeroed (position + bias only) | 0.569 | 0.634 |
| **causal**: score at t uses only steps 0..t (every 4th step) | 0.838 | **0.659** |

How to read this:

* The test clips were almost certainly **seen during training** (inferred from `train.py` §8 step 2 and the file
  dates). These are **training-set numbers**, and the steady rise from epoch 10 to epoch 60 looks like memorization.
* Performance comes from the **per-clip constant CLIP mean**, which includes the violent part of the window, plus
  positional ramp fitting. Person/VideoMAE features add little.
* Run causally, as it would have to be in deployment, clip-level discrimination falls to 0.66 even on (probably)
  seen data.
* **Conclusion: no valid generalization or anticipation score exists for this pipeline.** Treat it as a failed
  baseline, not a benchmark.

Reproduction (CPU, about 5 s per checkpoint plus about 2 min for the causal loop):

```python
import os, sys, json, numpy as np, torch; os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, REPO + "/legacy"); os.chdir(REPO)   # index entries are data/... paths relative to the repo root
ARCH = "data/archive/repo_legacy"
from core.model import ViolenceTransformer
m = ViolenceTransformer(node_dim=772, scene_dim=512, audio_dim=64, hidden_dim=256, num_heads=4, num_layers=2)
m.load_state_dict(torch.load(f"{ARCH}/checkpoints/model_epoch60.pt", map_location="cpu")); m.eval()
for it in json.load(open("legacy/test_index.json")):
    G = np.load(it["graph_path"], allow_pickle=True); L = np.load(it["label_path"]).reshape(-1)
    if min(len(G), len(L)) < 16: continue                       # same filter as GraphSequenceDataset
    frames = [{"nodes": torch.from_numpy(g["nodes"] if isinstance(g["nodes"], np.ndarray) and g["nodes"].ndim == 2
                                         else np.zeros((1, 772), np.float32)),
               "scene": torch.from_numpy(np.asarray(g["scene"], np.float32)),
               "audio": torch.zeros(64)} for g in G[:len(L)]]
    p = torch.sigmoid(m(frames, return_logits=True)).detach().numpy()   # compare with L (y>=0.5) via roc_auc_score
```

---

## 11. Evaluation code — `training/evaluate.py`, `training/eval_anticipation.py`

**`evaluate.py`**: `cd legacy && ../.venv/bin/python -m training.evaluate --ckpt $ARCH/checkpoints/model_epoch60.pt [--bin_thresh 0.5]`
(needs the `legacy/data -> ../data` symlink described in `legacy/README.md`)

* Uses `./val_index.json` if present; otherwise it builds and **writes `./eval_index.json`** from all graphs.
* Builds the model with hard-coded `hidden 256, heads 4, layers 2`.
* `evaluate_model()` returns `{"mse", "auc", "num_frames"}`, a pooled frame-level ROC-AUC over `y ≥ bin_thresh`.
* **Bug:** it calls `model(frames)`, which defaults to `return_logits=True`, so the **reported MSE compares logits
  with [0,1] targets** and is meaningless. AUC is unaffected because sigmoid is monotonic.
* The README says train.py calls this every epoch. **It never does.**

**`eval_anticipation.py`**: `python -m training.eval_anticipation --ckpt … --index val_index.json --thresholds 0.3,0.5,0.7 [--device cpu] [--max_videos N]`

Prints the global frame MSE/AUC (on probabilities, which is correct here), score percentiles for negative and
positive steps, and for each threshold θ: early-detected positive clips, mean TTD (s) and mean false-alarm seconds
per negative clip. It also prints per-class frame AUC (UCF class from the file name; `xd_violence`).
Metadata comes from `data/metadata/{xd_violence,ucf_crime,kinetics}.json`.

Bugs that make its TTD and false-alarm numbers wrong:

1. **Index misalignment**: it iterates `index_items[idx]` (the raw JSON list) but `dataset[idx]` (the *filtered*
   list). After the first dropped item, every clip is matched to **another video's metadata** (class, label, event,
   fps). In `test_index.json` the first drop is at position 5, so 163 of 168 clips are mismatched. In
   `val_index.json` it is position 22. The fix is to use `dataset.items[idx]["graph_path"]`.
2. **Wrong time base**: it uses the **full-video** `event_frame/total_frames` from metadata. Graphs were built on
   the **windowed** clip, so the mapped event step is wrong. `dt_sec = (total_frames/fps)/T` also assumes the full
   video and uniform steps.
3. "Early detection" counts any alarm at or before the event, including one at step 0. An always-on detector
   gets the maximum TTD, and only the false-alarm metric penalizes it.
4. `--max_videos` truncates `index_items` but not the dataset.

---

## 12. Data leakage and validity problems (prioritized)

1. **Training index = everything** (`train.py`): no held-out data was used when the checkpoints were trained (inferred).
2. **Future leakage in features**: the scene vector is the mean over the whole window, post-event frames
   included, and the same vector is fed at every step. It acts as a clip-label oracle (ablation: AUC 0.97 → 0.47 without it).
3. **Future leakage in the model**: the bidirectional encoder has no causal mask.
4. **Positional shortcut**: positive windows are cut so the event sits at about 75 % of the clip, and labels are a
   deterministic ramp in relative position. A learned absolute position embedding plus the "is this a violent clip"
   signal from (2) reproduces the target without anticipating anything.
5. **Label/time misalignment**: proportional event placement over non-uniformly sampled steps (§4, §6). Also,
   `ffmpeg -ss … -c copy` trimming cuts at keyframes, so the windowed clip can start earlier than requested and
   the event offset in window coordinates can be off by up to one GOP (inferred, standard ffmpeg behaviour).
6. **Label noise**: about 1084 violent or anomalous videos labelled as negatives (§6). 26 legacy-format clips have all-zero node features.
7. **Split leakage (later splits)**: `train/val/test_index.json` are disjoint by clip, but XD-Violence clips from the
   **same movie** fall into different splits. Grouping on the part of the name before `__#` gives 32 movies shared
   by train and test and 36 shared by train and val.
8. **No real audio**: every audio vector is zero, so the "multimodal" claim does not hold.

---

## 13. Tests, scripts and the README-referenced realtime path

`pytest` is **not installed** in `.venv`, so `pytest -q` from the README cannot run. I ran the CPU-safe functions directly:

| Test | Result | Why |
|---|---|---|
| `tests/test_extractors.py::test_center_bbox` | PASS | |
| `tests/test_extractors.py::test_import_extractors` | FAIL (by reading) | imports `detect_and_track` and `extract_audio`, which don't exist. Importing also loads YOLO/VideoMAE/CLIP onto the GPU |
| `tests/test_graph_builder.py::test_build_graph_sequence_basic` | FAIL (ran it) | expects the legacy API (`persons` key, a numpy object array, `timestamp`, `velocity` …) |
| `tests/test_graph_builder.py::test_graph_to_text_basic` | FAIL (ran it) | `TypeError: unhashable type: 'slice'` (passes a dict) |
| `tests/test_realtime_graph.py` | FAIL (by reading) | `build_live_graph` returns nodes as a list of dicts with no `edges`. Importing `realtime.pipeline` also loads the untrained model and **Qwen/Qwen2.5-3B-Instruct** at import time |
| `tests/tests_model_forward.py` | not collected (name doesn't match `test_*.py`) and would FAIL | 770-d nodes vs 772, 128-d audio vs 64; expects output shape `(1,)` but the model returns `[T]` |

Scripts:

* `scripts/benchmark.py`: broken. The default `sample.mp4` doesn't exist; it calls `build_live_graph(...)["edges"]`
  (KeyError) and `torch.tensor` on a list of dicts; the model is untrained. **The FPS numbers in the README were
  never produced by this code** (inferred).
* `scripts/export_trace_graph.py`: broken. It feeds 770-d nodes and 128-d audio to a 772/64 model; the input is a
  list of dicts; `dynamic_axes` are invalid for that input. No `violence_transformer.onnx` exists.
* `scripts/visualize_graphs.py`: defines `visualize_graph(graph)` only. It expects `graph["edges"]` as `(i, j, w)`
  triples, which no producer creates (graphs store `adj` as N×N). Running the file does nothing.
* `scripts/inspect_risk_curves.py`: works. Run `python -m scripts.inspect_risk_curves --data_root data
  --auto_samples` or `--video_id <vid> [--labels_dir …]`; it writes `data/plots/<vid>_risk_curve.png`. Caveat: the
  x-axis is `index / fps`, which is only correct for the 26 legacy per-frame clips. For normal graph-step labels
  (stride 4, dropped steps) and windowed clips, the time axis and the red `event_frame` line (full-video
  coordinates) are wrong.
* `realtime/pipeline.py` (the README's realtime entry point): there is no `__main__` and no argparse, so
  `--source` does not exist. It never loads a checkpoint. `build_live_graph` output crashes `ViolenceTransformer`,
  because `nodes` is a list of dicts. `model(buffer).item()` fails for T>1. `graph_to_text(dict)` raises
  TypeError. The working realtime code is `realtime/rtsp_inference.py` (a different LSTM-on-VLM-embeddings model,
  documented in `realtime_and_misc.md`).

---

## 14. README audit

| README claim | Verdict |
|---|---|
| YOLOv8 human detection | **True** (yolov8n, every 4th frame) |
| DeepSORT tracking | **False**: not used; IDs are per-frame indices |
| VideoMAE action embeddings | **Partly true**: self-supervised base model, a single crop replicated 16×, no motion |
| CLIP scene embeddings | **True, but** one clip-global mean (leaky), not per frame |
| Audio mel-spectrogram features | **False** for all training data (zeros(64)); the real extractor exists but was never called |
| Graph construction per frame | True (nodes + adjacency), but the adjacency is never consumed |
| Graph Transformer + Temporal Transformer | **False**: mean pooling plus a 2-layer temporal encoder |
| Sliding-window anticipation training | **False**: whole clips, batch 1 (`window: 60` unused) |
| Real-time inference with live cameras (`python realtime/pipeline.py --source …`) | **False/broken** (§13) |
| `scripts/preprocess_all.py` | **Doesn't exist**; the real script is `scripts/preprocess_anticipation.py` (run as `python -m scripts.preprocess_anticipation`) |
| Node = `[cx, cy] + 768 = 770` | **False**: `[x1, y1, x2, y2] + 768 = 772` (raw pixels) |
| frame = avg(nodes) + scene + audio | True (after the linear projections) |
| `python training/train.py` trains and "evaluates per epoch" | Fails as written (needs `-m`); **no evaluation** inside training |
| Metrics AUC + TTD | AUC in `evaluate.py`; TTD only in `eval_anticipation.py`, and it is buggy (§11) |
| Benchmark FPS output | Not reproducible; the script is broken |
| ONNX export | Broken; no ONNX file exists |
| `pytest -q` passes | pytest is not installed; every test is stale |
| `configs/model.yaml` controls the model | Not read by any code |
| Metadata format `{event_frame, total_frames}` | Real files also have `label, fps, source` |

---

## 15. Hard-coded paths and environment assumptions

* Relative paths everywhere (`configs/*.yaml`, `data/processed/...`, `yolov8n.pt`, `train_index.json`, `checkpoints/`),
  so you must run from the repo root.
* `scripts/preprocess_anticipation.py`: `BASE` is a hard-coded home-relative path that
  resolves to the repo's `data` symlink.
* `scripts/quick_inspect.py`: `data`.
* `core/utils.get_device()`: `VA_GPU` env, default `"1"` (a removed code comment wrongly said GPU 0).
* `core/extractors.extract_audio_from_file`: `/tmp/va_temp_audio.wav`.
* HF hub downloads at import: `MCG-NJU/videomae-base` and CLIP `ViT-B/16` (both appear cached; importing `core.extractors` succeeded in this venv). `realtime/pipeline.py` also pulls `Qwen/Qwen2.5-3B-Instruct`.
* The stray file `data/processed/actions/v=pFamvR9CpYw__#00-06-36_00-15-20_label_B4-0-0.npy.tmp.npy` is a leftover
  from the `.npy.tmp` naming bug that `atomic_save_npy` was written to fix. That is why actions has 1799 files and
  the other folders have 1798.

---

## 16. Reusable assets for a new model

| Asset | Path | Size / shape | Worth reusing? |
|---|---|---|---|
| Event-centred trimmed clips | `data/windowed/ucf_crime/*.mp4` (759), `data/windowed/xd_violence/*.mp4` (690) | 6.2 GB + 2.6 GB. Positives span [event−30 s, event+10 s]; negatives are the middle 40 s | **Yes**: ready-made anticipation windows for new extractors (pose, VideoMAE-finetuned, audio). Clips not listed (whole video ≤ window) are used directly from `data/raw_videos/<src>/videos/` |
| Raw videos | `data/raw_videos/{xd_violence (798), ucf_crime (1000), kinetics (700)}/videos` | | Yes (re-window with proper, re-encoded trimming) |
| Metadata | `data/metadata/{xd_violence,ucf_crime,kinetics}.json` | `{vid: {event_frame, total_frames, label, fps, source}}` | Yes, but fix the label noise (§6) |
| Person boxes | `data/processed/tracks/<vid>.json` (1798, 143 MB) | `list[T_s]` (every 4th frame of the window) of `[{track_id (per-frame index), bbox [x1,y1,x2,y2] px}]` | Maybe (no identities; yolov8n is weak). Re-running a pose model (`yolov8m-pose.pt` is in the repo root) is better |
| VideoMAE crop embeddings | `data/processed/actions/<vid>.npy` (3.8 GB) | object array `[T_s]` of `{track_id: float32[768]}` | Low (static single-frame, non-finetuned) |
| CLIP clip-mean | `data/processed/scenes/<vid>.npy` | float32[512] | **No** for anticipation (contains post-event frames) |
| Audio | `data/processed/audio/<vid>.npy` | zeros(64) | **No** (all zeros) |
| Graph sequences | `data/processed/graphs/<vid>.npy` (3.9 GB) | §4 | Low |
| Risk labels | `data/processed/labels/<vid>.npy` | float32[T_graph] ramp→1 | Low (misaligned time base); regenerate from metadata in seconds |
| Checkpoints | `checkpoints/`, `best_checkpoints/` | 2.97 M params each | Only as a negative-result baseline |

---

## 17. Lessons and recommendations for a new model

* Labels must live in **seconds**, aligned to real timestamps of the sampled frames, never proportional indices.
* All features at time t must be computed from frames ≤ t only. No clip-level means. Use a causal model
  (masked attention, GRU, or sliding windows that end at t).
* Evaluate by **video/movie-grouped** splits, with per-clip anticipation metrics (TTD at a fixed false-alarm rate
  per hour on negatives). Don't use pooled frame AUC alone, which a positional ramp can game. Include a
  position-only baseline (0.61 here) and a shuffled-time control.
* Fix metadata label noise before training: drop un-annotated UCF anomaly videos and the XD clips that carry a
  violent tag but `label = 0`, rather than calling them negative.
* If you reuse `train_index.json` / `val_index.json` / `test_index.json`, regroup by movie first, and **never run
  `training/train.py` unmodified** (it overwrites `train_index.json`).
