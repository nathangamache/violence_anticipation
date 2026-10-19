# VLM / LLM approaches (Qwen2.5-VL teacher, embeddings, QA features, LoRA, LLM text head)

> Paths in this file describe the tree before the 2026-10 cleanup. Surviving code is under `legacy/<same path>`;
> the honest runs are under `baselines/reference_runs/`; removed tracked files: `git show pre-cleanup:<path>`;
> ignored weights and arrays: `data/archive/repo_legacy/<same path>`
> until 2027-01-12. See [../REPO_CLEANUP_PLAN.md](../REPO_CLEANUP_PLAN.md).

Audience: future AI agents and developers who need to understand or reuse the vision-language-model (VLM)
and text-LLM parts of this repo. Everything below was checked against the code and the files on disk on
2026-10-14 unless it is marked **(inferred)**.

Repo root: `.` (paths below are relative to it).
`data/` is a symlink to `data/` and `models/` to `models/`.

---

## 0. TL;DR

* **Qwen2.5-VL-7B-Instruct** (local copy: `models/Qwen2.5-VL-7B-Instruct`, 16 GB, bf16; also in the local Hugging Face cache)
  is used in four ways:
  1. **Teacher labeller**: it reads one frame at a time and writes a JSON verdict (`violent.label/score`, people count, actions, weapons, scene and so on).
     Script: `scripts/vlm_label_dataset.py`. Output: `data/processed/vlm_labels*/<source>/<vid>.jsonl`.
  2. **Frozen feature extractor** (the most used role): each frame becomes one 3584-d embedding, the mean of the last-layer hidden states
     at the image-token positions, L2-normalised. Output: `data/processed/vlm_emb*/<source>/<vid>.npz`.
     These embeddings feed almost every later "student", LSTM, fusion and simulation model (see the training docs).
  3. **Visual question answering features**: 25 yes/no questions per frame give a 25-d vector (`tuning/question_analysis/`).
     This was only tried on 100 frames. Result: logistic-regression test AUC **0.49**, so no useful signal.
  4. **LoRA finetune** to answer "Is there violence in this image? yes/no" (`tuning/lora_finetune_vlm.py`).
     Its validation accuracy is stuck at 0.4925 because the eval loop has a bug, and its labels are semantically wrong. **The adapters are unusable as-is.**
* **Qwen2.5-3B-Instruct (text-only LLM)** was used as a "decision head" on text summaries of the scene graph
  (`llm/`, `scripts/build_llm_dataset.py`, `llm_data/`). The summaries carry almost no information, and the prompts leak the label.
  Results: video-level AUC **0.588** with accuracy 0.727, below the 0.826 you get by always answering "non-violent". On the frame-head variant the AUC is **0.341**.
* **Zero-shot teacher quality.** I recomputed this from the JSONL against the ground truth.
  Frame-level ROC-AUC of `violent.score` vs current-violence labels is about **0.67 to 0.68** (XD-Violence 0.75 to 0.78, UCF-Crime about 0.50).
  For the 30 s anticipation labels it is about **0.65 to 0.66**. UCF is at chance mostly because the **UCF ground truth is broken**:
  UCF *training* videos, including "Fighting", are labelled all-zero (see §7).
* **Was a "student" distilled?** There is a true distillation dataset builder (`scripts/vlm_build_student_dataset.py`:
  graph features in, VLM JSON out), but **its output was never generated and nothing consumes it**. That is dead code.
  The "students" that were actually trained are MLP and LSTM heads on frozen VLM embeddings, trained against ground-truth labels.
* **VLM outputs used as ground truth:** yes, in one place that matters. The 92 "curated" YouTube videos
  (`data/anticipation_curation/candidates.json`, embeddings in `data/processed/vlm_emb/curated/`)
  had their violence and escalation timestamps produced **by a model**, not by a human (§8).
  Training or evaluating VLM-embedding models on those labels is circular.

---

## 1. Model inventory

| Model | Where | Used by | Notes |
|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | `models/Qwen2.5-VL-7B-Instruct` (local) and HF cache `models--Qwen--Qwen2.5-VL-7B-Instruct` | `vlm/qwen_teacher.py`, `tuning/lora_finetune_vlm.py`, `tuning/question_analysis/*`, `realtime/rtsp_inference.py`, `scripts/analyze_youtube.py`, `setup/curate_youtube.py`, `experiments/fast_analyze_curation.py`, `experiments/prompt_engineering_stats.py` | hidden_size 3584, vision tower hidden 1280 → merger out 3584. `image_token_id` 151655, `vision_start_token_id` 151652. Preprocessor `max_pixels` 12,845,056, so full-res frames become thousands of visual tokens. |
| Qwen2.5-VL-3B-Instruct | HF cache only | none found in code | (inferred) downloaded for experiments |
| Qwen2.5-7B-Instruct (text) | HF cache only | none hard-coded | (inferred) may have produced `llm_data/val_seq_preds_qwen.jsonl`, provenance unknown |
| Qwen2.5-3B-Instruct (text) | **not present locally** | `llm/llm_embedder.py`, `llm/llm_head_infer*.py`, `realtime/pipeline.py` | `llm_head_infer_seq.py` defaults to `models/Qwen2.5-3B-Instruct`, which does not exist, so it would fail |
| LoRA adapters | `lora_finetuned/`, `lora_finetuned_v2/` (gitignored) | nothing loads them | see §5 |

Library versions in `.venv`: transformers 4.57.5, peft 0.18.1, bitsandbytes 0.49.2. `requirements.txt` does **not** list peft, bitsandbytes, scikit-learn, matplotlib or ultralytics.

Most scripts default to the HF id `Qwen/Qwen2.5-VL-7B-Instruct`. That resolves from the HF cache. Pass `--model_id models/Qwen2.5-VL-7B-Instruct` (or `--model`/`--model_name`, depending on the script) to force the local copy.

---

## 2. `vlm/qwen_teacher.py` (core wrapper, about 970 lines)

### Classes and functions
* `_best_effort_json_parse(text)` tries three things in order: a ```` ```json {…} ``` ```` fence, then the first balanced `{…}`, then light repairs (`'`→`"`, trailing commas). It returns a dict or `None`.
* `FrameContext` / `ContextWindow` form a sliding memory of previous parsed frames (and optionally their PIL images) used for "temporal context".
  `to_text_summary()` only knows the *default* schema keys (`violent`, `people_count`, `actions`, `weapons`, `scene`).
  With the `anticipation_*` prompt modes the injected context is therefore nearly empty.
* `ViolencePrompts` is a library of prompt templates. Each one asks for "ONLY valid JSON":
  * `embedding_focused` (mode `violence_embedding`): violent{label,score}, aggression_level, threat_indicators, body_language, actions, weapons, people_count, scene
  * `anticipation_focused` (`violence_anticipation`): violence_imminent, violence_active, threat_level and more
  * `detailed_violence` (`violence_detailed`), `minimal_violence` (`violence_minimal`), `temporal_violence` (`violence_temporal`, injects context text)
  * `anticipation_multicue` (`anticipation_multicue`): 30 cue scores in 6 groups plus `overall_threat_score`
  * `anticipation_simple` (`anticipation_simple`): 7 scores plus `stage` ∈ calm/tension/escalation/violence/aftermath
  * `anticipation_behavioral`, `anticipation_temporal(prev_scores)`, `scene_context`
  * `QwenTeacher.default_prompt()` / `context_aware_prompt(ctx)` (mode `default`): violent, people_count, actions, weapons, scene, notes, plus `temporal_change` when context is given
* `QwenTeacherConfig` holds: `model_id`, `max_new_tokens=256`, `temperature=0.0`, `dtype="bfloat16"`, `attn_implementation="sdpa"`,
  `context_mode="both"`, `context_window_size=5`, `store_context_images=False`, `prompt_mode="violence_embedding"`.
* `QwenTeacher`
  * loads with `AutoModelForVision2Seq` + `AutoProcessor` and moves the model to a single device.
  * `_build_inputs(image, prompt, context_images)`: the message list is a system prompt ("You are a security camera analyst …"),
    then user content `[ctx images + "[Previous frame i]" …, current image, "[Current frame to analyze]", prompt text]`.
    **The image comes before the prompt text.**
  * `_extract_embedding(inputs)` runs a separate forward pass with `output_hidden_states=True`, takes `hidden_states[-1]`, and builds a mask from candidate image token ids
    (the tokenizer/config `image_token_id`, plus every special token whose name contains image/vision/img, i.e. `<|vision_start|>`, `<|vision_end|>`, `<|image_pad|>`, `<|vision_pad|>`).
    It mean-pools the masked positions and L2-normalises, giving a `float32` array of shape `(3584,)`.
    If no image tokens are found it falls back to the mean over all attended tokens.
  * `infer_one(image, prompt=None, return_embedding=False, t=None, raw_frame_idx=None, use_context=None)`
    calls `generate` (greedy), decodes, takes `decoded.split("assistant")[-1]`, and parses the JSON.
    It returns `(parsed, raw_text)` or `(parsed, raw_text, emb)`. If `t` and `raw_frame_idx` are given, it pushes the result into the context window.
  * `infer_embedding_only(image, prompt)`: a forward pass only, no generation. This is the fast path used by realtime and curation.
  * `infer_batch_sequential(images, …)` and `extract_embeddings_batch(images, …)` are sequential loops, not real batching.
* `multicue_to_vector(parsed)` returns a `(31,)` float32 vector in the order confrontation(4), aggressive_posture(5), movement(5), crowd(5), preparation(4), active_violence(7), overall_threat_score(1).
* `simple_to_vector(parsed)` returns a `(7,)` vector: confrontation, aggressive_posture, crowd_tension, physical_contact, weapon_present, fleeing_panic, overall_threat.

### Important behaviours and gotchas
1. **`prompt_mode` cannot change the embedding.** The LM decoder is causal and the prompt text comes *after* the image tokens, so the image-token hidden states
   depend only on the system prompt, any context images and the image itself. All the "violence-focused prompts for better embeddings" logic only changes the generated JSON.
   (This is inferred from the architecture: causal attention, with M-RoPE positions for earlier tokens unaffected by later text. The ViT does not see the text either.)
2. With `context_mode` `visual` or `both` and `store_context_images=True`, the image-token mask covers **all images in the prompt**.
   The "frame embedding" is then the mean over the current frame and the previous context frame(s).
3. `infer_one(return_embedding=True)` runs a second full forward pass after `generate`, so the prefill cost is roughly doubled.
4. `decoded.split("assistant")` breaks the parse if the model's JSON itself contains the word "assistant" (for example "shop assistant") (inferred, edge case).
5. No resizing is done. Full-resolution frames go to the processor, and a 1080p frame is about 2.6k visual tokens. This drives the cost (§9).

---

## 3. `scripts/vlm_label_dataset.py` (teacher labelling and embedding extraction)

**Purpose:** for every video that has a graph file, sample every `every_t`-th graph timestep, read the matching raw frame,
run `QwenTeacher.infer_one(..., return_embedding=True)`, and write the JSON label plus the embedding.

**Run** (from the repo root so `vlm` is importable):
```bash
.venv/bin/python -m scripts.vlm_label_dataset --graphs_dir data/processed/graphs \
    --prompt_mode violence_embedding --context_mode none --every_t 5 --device cuda:0 --skip_existing
# 2 GPUs: add --rank 0 --world_size 2 / --rank 1 --world_size 2 (plus lock files in .locks/)
```
Key args: `--meta_dir data/metadata`, `--out_jsonl_root data/processed/vlm_labels`, `--out_npz_root data/processed/vlm_emb`,
`--out_features_root data/processed/vlm_features`, `--kinetics_root/--ucf_root/--xd_root data/raw_videos/...`,
`--every_t 5`, `--max_new_tokens 512`, `--context_mode none|text|visual|both`, `--context_window 1`,
`--prompt_mode` (10 choices, default `default`), `--save_features`, `--skip_existing`, `--verify_complete`, `--overwrite`, `--lock_dir .locks`, `--lock_timeout 3600`.

**Frame mapping:** `raw_frame_idx = round((t + 0.5) / T * total_frames)`, where `T = len(graph sequence)` and `total_frames` is taken from metadata first, then cv2.
This matches how the label files are indexed, since labels are per *graph timestep* and not per second.

**Outputs, per video:**
* `<out_jsonl_root>/<source>/<vid>.jsonl`, one line per sampled `t`:
  `{"vid_id","source","t","T","raw_frame_idx","context_mode","prompt_mode", **parsed_json}`.
  Error lines look like `{"error":"parse_failed"|"frame_read_failed"|"missing_video"|...}`.
* `<out_npz_root>/<source>/<vid>.npz`: `t` int32[N], `raw_frame_idx` int32[N], `emb` float16[N, 3584], sorted by `t`.
* (anticipation modes or `--save_features`) `<out_features_root>/<source>/<vid>.npz`: `t`, `raw_frame_idx`, `features` [N, 31 or 7], `prompt_mode`.
  **This directory does not exist on disk.** The multicue/simple feature modes were never run at scale.

**Known problems**
* On resume, a `t` is skipped only if it appears in *both* the JSONL and the NPZ. Otherwise the JSONL line is appended again.
  `vlm_labels_violence` contains **6,365 duplicate-`t` records** (84,601 lines for 78,236 unique frames). The cause is inferred to be resumes.
  Deduplicate by `(vid_id, t)` when reading.
* If the JSON parse fails, the embedding for that `t` is discarded, so the NPZ `t` set can be a subset of the JSONL `t` set.
* On resume, the `features` list is reloaded but then indexed with `order`, which is computed from `ts`. If the lengths differ this misaligns or raises (inferred).
* `--overwrite` deletes files before taking the lock, so it is unsafe with several ranks (inferred).
* The defaults disagree: the CLI `--prompt_mode` default is `default`, while the `QwenTeacherConfig` default is `violence_embedding`.

### On-disk outputs (all under `data/processed/`)

| Dir | Files | Content / provenance |
|---|---|---|
| `vlm_labels_old/` + `vlm_emb_old/` (2026-01-29) | 949 ucf + 772 xd | 78,236 records, 44 parse failures, no context/prompt fields. Embeddings are **unnormalised** (norm about 293) and nearly identical across frames (within-video cosine 0.999). Inferred to be the earliest version, which mean-pooled over the whole sequence so the prompt tokens dominated. That is exactly what the docstring of `_extract_embedding` warns against. **Do not use these embeddings.** |
| `vlm_labels_meanpool/` + `vlm_emb_meanpool/` (2026-02-09) | 949 + 772 | The JSONL is 195 of 200 files byte-identical to `_old`. The embeddings are L2-normalised and different. Inferred: mean over all attended tokens. Used by `baselines/reference_runs/student_mlp` (training area). |
| `vlm_labels/` + `vlm_emb/{ucf_crime,xd_violence}` (2026-02-16) | 949 + 772 | 78,236 records, `context_mode="both"`, no `prompt_mode` field (so the default/context-aware prompt from an older script version; inferred), 2 parse failures. Keys: `violent, people_count, actions, weapons, scene, temporal_change, notes`. Embeddings `[N,3584]` fp16, norm 1. Inferred to pool the current frame plus one previous frame's image tokens (see §2.2). **This is the default `vlm_emb` used by most later scripts.** |
| `vlm_labels_violence/` + `vlm_emb_violence/` (2026-02-20) | 949 + 772 | `prompt_mode=violence_embedding`, `context_mode=none`. Keys: `violent, aggression_level, threat_indicators, body_language, actions, weapons, people_count, scene`. Contains duplicate records (see above). Log: `logs/gpu0_final_scan.log`. |
| `vlm_emb/curated/` | 92 | **Different schema:** `emb` [N,3584] fp16, `t` int64, `raw_frame_idx` int64, `timestamps` float64, `scores` float32 (LSTM scores). Written by `experiments/fast_analyze_curation.py` (§8). Careful: any `rglob("*.npz")` over `vlm_emb/` also picks these up. |
| `vlm_audio_features/xd_violence/` | 772 | VLM (3584) and audio (91) concat-normalised, `total_dim` 3675 (`fusion_metadata.json`). Fusion area. |
| `fused_vlm_flow/`, `fused_finetuned_flow/` | 773 each | VLM + optical-flow fusion. `fused_finetuned_flow` came from `data/processed/vlm_emb_finetuned`, which no longer exists. It is a 22-d contrastive projection and is **not** related to the LoRA. Tuning area. |

Similarity between variants (mean cosine on matching frames, 8 videos): `vlm_emb`~`violence` 0.93, `vlm_emb`~`meanpool` 0.92, `meanpool`~`violence` 0.97, anything~`old` 0.12 to 0.19.
Within-video cosine between *different* frames is about 0.95 to 0.97 for every usable variant. The embeddings are dominated by scene and generic content, so the violence signal is a small residual.

---

## 4. `scripts/eval_vlm_teacher.py` (how good is the zero-shot teacher?)

**Run:** `.venv/bin/python scripts/eval_vlm_teacher.py --jsonl_root data/processed/vlm_labels --labels_dir data/processed/labels --out_dir baselines/reference_runs/vlm_eval [--skip_parse_failed]`

It pairs each JSONL record's score (`violent.score`, falling back to label yes→1, no→0, otherwise 0.5) with `labels/<vid>.npy[t] >= gt_thresh`.
It reports global ROC-AUC and PR-AUC, sweeps the F1 threshold (the best threshold is chosen *on the same data*, which is optimistic), compares against an always-positive baseline,
and writes per-source and per-video CSVs plus plots. Records without a numeric/label score are skipped (`skip_no_score`).

**Stored results** (`baselines/reference_runs/vlm_eval*/summary.json`). They were computed when `data/processed/vlm_labels` held what is now `vlm_labels_old`
(the counts match: 44 parse_failed, 15,612 no-score, 62,580 pairs):

| labels_dir | ROC-AUC | PR-AUC | best F1 (thr) | always-pos F1 | UCF AUC | XD AUC |
|---|---|---|---|---|---|---|
| `labels` (current violence) | 0.682 | 0.194 | 0.313 (0.51) | 0.191 | 0.497 | 0.776 |
| `labels_future_h5` | 0.677 | 0.202 | 0.320 | 0.203 | 0.490 | 0.776 |
| `labels_future_h10` | 0.672 | 0.206 | 0.324 | 0.212 | 0.484 | 0.776 |
| `labels_future_h30` | 0.662 | 0.216 | 0.332 | 0.231 | 0.476 | 0.772 |

**Recomputed on 2026-10-14** for all label sets. This was read-only and deduplicated by (vid,t). For `labels_future_h30`, labels <0 were skipped and multi-horizon arrays used column 4 (30 s).

| JSONL set | vs `labels` AUC / AP | vs `labels_future_h30` AUC / AP | UCF / XD AUC (vs labels) |
|---|---|---|---|
| `vlm_labels` (context both) | 0.670 / 0.197 | 0.654 / 0.216 | 0.492 / 0.750 |
| `vlm_labels_old` | 0.683 / 0.194 | 0.662 / 0.216 | 0.497 / 0.776 |
| `vlm_labels_violence` (violence prompt, no ctx) | 0.630 / 0.167 | 0.615 / 0.189 | 0.500 / 0.680 |

Takeaways: the zero-shot frame verdict is a weak *detector* (it reacts to violence that is visible now) and an even weaker *anticipator*.
The AUC drops as the horizon grows. The "violence-focused" prompt was **worse** than the plain default prompt.

---

## 5. `tuning/lora_finetune_vlm.py` (LoRA on Qwen2.5-VL, "V3 - Stability Fixed")

**What it trains:** a yes/no VQA. The user turn is `[image, "Is there violence in this image? Answer yes or no."]` and the target is the single token `"Yes"` or `"No"`.
No `<|im_end|>` is appended, and the loss covers only that token.

**Data pipeline (`prepare_dataset`)**
1. It enumerates `--embeddings_dir/**/*.npz`, but only to get video names; the embeddings themselves are not used. It filters by name: XD if `'__#' in name or name.startswith('v=')`, UCF if `'_x264' in name`.
2. It finds the video in `--videos_dir` and loads `--labels_dir/<vid>.npy`. If 2-D it takes `.max(axis=1)`, then `labels > 0`.
3. Per video it randomly picks up to 20 positive and 20 negative timestep indices (unseeded `np.random`), caps at `max_samples_per_class` per class, shuffles, and puts **10% into val at random**.
4. `extract_frame(path, idx)` seeks to frame `int(idx * fps)` and resizes to **336×336** (about 144 visual tokens).

**Config actually used for v2** (from `logs/lora_train_v2.log` / `logs/lora_train_v2_full.log`, 2026-02-23).
Command per the docstring: `--videos_dir data/raw_videos/xd_violence/videos --labels_dir data/processed/labels_multih --embeddings_dir data/processed/vlm_emb --source_filter xd_violence`.
I confirmed this by reproducing the logged "6388 violence, 13979 non-violence" exactly with `labels_multih`.
LR 1e-4 (seen in the progress bar), epochs 5, grad-accum 16 (3600 samples / 225 optimiser steps), LoRA r=8, alpha=16, dropout 0.05,
targets `q,k,v,o,gate,up,down_proj`. These target names also match the **vision tower MLPs**: 192 of the 584 adapter tensors are `model.visual.blocks.*.mlp.*`.
Trainable params 23.8 M of 8.3 B (0.29%). 8-bit AdamW, bf16, cosine LR to lr/10, **no warmup** (`warmup_steps` is computed but never used). 3600 train / 400 val frames, 772 XD videos.
Speed was about 2.7 it/s (22 min per epoch on one GPU) and evaluation took about 1 to 3 minutes for 400 frames.

**Results**

| Run | Adapter | history.json train_loss | val_acc |
|---|---|---|---|
| `lora_finetuned/` (v1, older script version; no log kept) | r=8, alpha=16, `q_proj,v_proj` only (112 tensors, LM only) | 113.9 → 96.8 → 96.7 | 0.1025 → 0.485 → 0.485 |
| `lora_finetuned/training_history.json` (an even earlier attempt) | ? | NaN, NaN, NaN | 0.508 ×3 |
| `lora_finetuned_v2/` | r=8, alpha=16, all 7 projections incl. vision MLP (584 tensors, 95 MB) | 10.60 → 8.14 → 6.23 → 4.63 → 3.53 | **0.4925 ×5 (197/400)** |

The `worker_0/` and `worker_1/` folders are empty, so a 2-worker attempt never finished. Each folder has `best_model/`, `final_model/` (adapter plus processor files) and `checkpoint_epoch*/`.
The `README.md` files in the adapter folders are the empty PEFT model-card template (39 "More Information Needed" fields) and carry no information.
`lora_finetuned_v2/best_model`'s adapter files are byte-identical to `checkpoint_epoch1` (`best_model` also carries the tokenizer/processor files), consistent with val accuracy never moving.
In the 2026-10 cleanup every adapter (`best_model`, `final_model`, all `checkpoint_epoch*`) and the history files
were moved to `archive/repo_legacy/lora_finetuned*/`; the empty `worker_*` folders were deleted.

**Why the LoRA results are meaningless (verified by reading the code and data):**
1. **The eval feeds the answer to the model.** `val_loader` uses the same `collate_fn_simple`, which builds `input_ids` from `prompt + target`.
   `evaluate()` calls `generate` on those ids and checks whether the *continuation after the answer* contains "yes".
   The continuation is essentially empty, so the prediction is always 0 and the accuracy is the negative rate: 197/400 = 0.4925 in every epoch.
2. **The labels contradict the question.** In `labels_multih` (`[T,7]`, horizons 2/5/10/20/30/60/120 s, `ignore_inside_segments=true`), `max(axis=1) > 0`
   means "violence **starts within the next 120 s**". Frames **inside** violence are `-1`, so they become label 0, which reads as "No, there is no violence".
   For XD, 46,235 of the 206,104 negative-pool steps (**22.4%**) are frames during actual violence.
3. **The frames are misaligned.** Label index `t` is a graph timestep, not a second, but the code seeks to `t*fps`.
   Using metadata `total_frames/fps`, **71.3% of all XD label steps (52.4% of positive steps) map past the end of the video**.
   Those are clamped to the last frame, so many training images are the same final frame carrying different labels.
4. The val split is frame-level and random, so the same videos appear in train and val (leakage, although it cannot matter given bug 1).
5. The reported `train_loss` is inflated ×`gradient_accumulation_steps`: `total_loss += accumulated_loss*G` over `valid_steps`.
   The true per-sample CE for v2 is about 0.66 → 0.22.
6. `--mode extract` and `--checkpoint` are parsed but never implemented. No code anywhere loads the adapters.
   (`fused_finetuned_flow` is not LoRA-based; see §3.)

If you revive LoRA: evaluate with prompt-only `input_ids` (or compare the logits of "Yes" vs "No"), map `t` to frames with `round((t+0.5)/T*total_frames)`,
use labels whose meaning matches the question, split by video (or by movie for XD), and consider a classification head on pooled vision tokens instead of text generation.

---

## 6. `tuning/question_analysis/` (VQA features)

### `vlm_question_features.py`
* `VIOLENCE_QUESTIONS`: 25 yes/no questions with hand-set weights. 19 are positive indicators (contact, hitting, pushing, person on ground, restraining, stances, fleeing, weapons, crowd, distress, anger, damage, blood, responders, violence_occurring, violence_imminent). 4 are negative (calm, sports, performance/acting, playful).
* `VLMQuestionExtractor` loads `Qwen2_5_VLForConditionalGeneration` in **fp16** (not bf16), `device_map=device`.
  * `extract_features(img)`: 25 separate `generate` calls (max 10 new tokens), giving `[25]` values in {0, .2, .25, .5, .75, .8, 1}.
  * `extract_features_batch_questions(img)` (default): one prompt listing all 25 questions, `max_new_tokens=200`, parsed line by line. Lines like "1. yes" are not stripped of their numbering in this file, so they score 0.8 or 0.2 instead of 1 or 0. `_parse_yes_no` also matches "no" inside words such as "not" or "know".
  * `get_weighted_score(features)` normalises the positive weights to sum to 1 and the negatives to -1, takes the dot product, and maps to [0,1].
* `process_single_video` iterates **every** decoded frame (`--sample_rate 1` default). That is extremely slow for long videos.
  It writes `<output_dir>/<vid>.npz` with `features [N,25]`, `scores [N]`, `frame_indices [N]` (raw frame numbers), `feature_names`, `questions`, `weights`, `video_info`, `sample_rate`.
* Modes: `--video_path` (one video), `--videos_dir --embeddings_dir` (videos named after embedding files), `--videos_dir` (all videos), `--analyze --features_dir --labels_dir`, `--list_questions`.
* `analyze_features` truncates features (raw-frame indexed) and labels (graph-timestep indexed) to the same length and compares them element-wise. **They are misaligned** unless `sample_rate` happens to equal the graph stride.
* **The default output `data/processed/vlm_qa_features` does not exist**, so full extraction was never run. `tuning/fuse_features.py` accepts a `qa_dir`, but the stored fusion used `qa_dir: null`.

### `test_qa_features.py` (quick probe)
* It samples 50 "violence" and 50 "non-violence" timesteps from `--labels_dir`. Per the docstring that is `labels_multih`, so "violence" means "violence starts within 120 s" and inside-violence frames count as negative (same problem as LoRA).
* `extract_frame_from_video(path, idx)` uses `idx` **as a raw frame number** whenever `idx < total_frames`, which is almost always. A label at timestep `t` is paired with frame `t` (0.2 to 10 s into the video). **The frames are misaligned.**
* It runs the batch-question prompt, then per-feature AUC, logistic regression on a random 70/30 split, and the weighted-score AUC.
* **Stored result:** `outputs/qa_feature_test/summary.json` (+ `features.npz` violence[50,25]/nonviolence[50,25], `analysis.png`).
  Logistic test AUC **0.486**, test accuracy 0.467, weighted-score AUC 0.596. The best single questions are `violence_occurring` (0.59) and `violence_imminent` (0.587).
  Almost every answer is "no" for both classes. Given the misalignment and the label semantics, this result says nothing either way about how useful VQA features are.

---

## 7. Text-LLM decision head (`llm/`, `scripts/build_llm_dataset.py`, `llm_data/`)

### `scripts/build_llm_dataset.py`
* Reads `train_index.json` / `val_index.json` (entries with `graph_path` and `label_path`), metadata `data/metadata/{xd_violence,ucf_crime,kinetics}.json`, and optionally `--pred_dir` `<vid>.npz` holding `pred`/`gt` curves.
* Builds **one example per video**: `{id, source, label, event_frame, total_frames, fps, prompt, response}`, written to `llm_data/{train,val}.jsonl`.
* The text summary comes from `core.graph.graph_to_text(graph_seq, meta)`. That call raises `TypeError`, because `meta` is passed as `max_frames`, so the code falls back to `graph_to_text(graph_seq)`.
  That function describes only the **last 8 frames** with a fixed template: "Frame N: about K people visible, with several pairwise distances indicating possible interactions."
  The only information in it is person counts. If that also fails, `fallback_graph_summary` is used, and it writes **"annotated as violent/non-violent"** into the prompt.
* Run: `.venv/bin/python -m scripts.build_llm_dataset --train_index train_index.json --val_index val_index.json --out_dir llm_data`
  (it reads `configs/paths.yaml` for the metadata directory).

### `llm_data/` (committed, 3.1 MB, 2025-12-14)

| File | Rows | Content |
|---|---|---|
| `train.jsonl` | 1376 videos (746 UCF, 630 XD) | 284 positive / 1092 negative |
| `val.jsonl` | 345 (203 UCF, 142 XD) | 60 positive / 285 negative. There is no id overlap with train. |
| `val_preds_qwen.jsonl` | 256 | Output of `llm/llm_head_infer.py` (the frame-level script run on these video-level rows; `frame_idx`/`t_sec` are null). Keys: id, video_id, frame_idx, t_sec, source, prompt, target_risk, pred_risk, bin_thresh, raw_output, rationale. |
| `val_seq_preds_qwen.jsonl` | 344 (1 parse failure) | Output of `llm/llm_head_infer_seq.py`. Keys: id, source, gt_label, pred_label, pred_risk, raw_output. |

**Label leakage in the prompts (verified):**
* The prompts contain `Annotated event frame (violence onset): N`. Every example with N > 0 has label 1 (train 237/284 positives, val 53/60).
  The rule "N>0 → violent" alone gives **val AUC 0.942, accuracy 0.980**.
* `Video ID:` is included. XD names encode the class (`label_A` means normal, `B1/B2/…` are violent types) and UCF names encode the category.
  The XD-filename rule alone gives val AUC 0.727.
* 19 train and 7 val prompts contain "annotated as <label>" from the fallback summary.

**Label noise (verified):** `label` is 1 only when the metadata says so, which for UCF effectively means the video is in the temporal-annotation test list.
Many violent videos are therefore labelled 0, for example UCF `Fighting` 45 of 50 and `Assault` 47 of 50 in the metadata. In `llm_data`, 266 XD videos (211 train + 55 val) whose filename tag is not `A` (normal) are also label 0.
The same metadata drives `data/processed/labels`, so this affects every model in the repo (inferred). The VLM teacher gives UCF `Normal_Videos_` frames a mean score of 0.000, but gives the label-0 `Fighting` videos a mean of 0.38, with 49% of frames ≥0.5.
In other words the teacher "disagrees" with the ground truth where the ground truth is wrong. This is the main reason the UCF teacher AUC is about 0.5.

### `llm/llm_head_infer.py` (per-sample "risk" head)
* Loads any causal LM (default `Qwen/Qwen2.5-3B-Instruct`) in bf16, applies the chat template with the system message
  "…respond with JSON {"risk": <float>, "rationale": …}", and parses the risk in three steps (JSON, then regex after "risk", then the first number). It computes MSE and ROC-AUC with targets binarised at 0.5.
* The **user prompt contradicts the system prompt**: the dataset prompt says "Answer with a single word: 'violent' or 'non-violent'".
* Defaults: `--max_examples 256`, `--data_path llm_data/val.jsonl`.
* Stored result (recomputed from `val_preds_qwen.jsonl`): n=256 (49 positive), **AUC 0.341**, MSE 0.432, accuracy@0.5 0.281.
  It predicts high risk (0.6 to 0.85) for most videos.

### `llm/llm_head_infer_seq.py` (video-level JSON head)
* Appends strict-JSON instructions to `prompt`, tokenises **without the chat template**, generates greedily (max 128 tokens), and parses the first `{…}` for `label` and `risk`.
* Run: `.venv/bin/python -m llm.llm_head_infer_seq --split val --data_dir llm_data --model_path <local or HF path> --device cuda:0`.
  The default `models/Qwen2.5-3B-Instruct` does not exist. It writes `llm_data/{split}_seq_preds_qwen.jsonl`.
* Stored result (recomputed): n=344, **AUC 0.588**, accuracy 0.727 (majority baseline 0.826). Risk outputs are mostly 0.0 (252) or ≥0.9 (62).
  The LLM did **not** exploit the leaked event-frame line. Which model produced this file is unknown.

### `llm/llm_embedder.py` and `llm/llm_head.py` (used only by `realtime/pipeline.py`)
* `LLMEmbedder.predict_risk(summary)` uses a plain prompt (no chat template) and `max_new_tokens=8`.
  `_extract_number` searches the **full decoded text including the prompt** and takes the *last* number.
  If the model emits no number, the last number in the prompt is "1.0" from "(0.0–1.0)", so it **returns risk 1.0** (bug).
* `LLMFusionHead`: `weighted` mode returns `w*base + (1-w)*llm`. `linear` mode is `Linear(2,1)`, initialised to [0.7, 0.3] with bias 0, and **never trained**.
  The output is therefore `sigmoid(0.7*p+0.3*s)` ∈ [0.5, 0.73] and can never fall below 0.5. It also applies `sigmoid` to `base_logits`, so if the base model already outputs a probability, the sigmoid is applied twice.
  `fuse_scores_weighted` is a stateless helper.
* `realtime/pipeline.py` calls `graph_to_text(latest_graph)` with a single dict, but `graph_to_text` expects a list (`graph_seq[-8:]`). This would raise `TypeError` (inferred; the realtime area owns this, see `docs/codebase/realtime_and_misc.md`).

**Verdict:** the graph-to-text plus LLM path never carried real visual information. Treat it as abandoned.

---

## 8. Where VLM or model outputs were used as ground truth (circularity audit)

| Place | Teacher output used as | Status / risk |
|---|---|---|
| `scripts/vlm_build_student_dataset.py` | **Targets**: `y_violent` (from `violent` label/score, with "yes"→max(score,.5), "no"→min(score,.5), else .5), `y_people` (clipped 0..10), multi-hot `y_actions/y_weapons/y_scenes` (vocabularies built from train+val records, min_count 10). **Inputs**: graph frame features `[1 + 2*node_dim + scene(512) + audio(64)]` (node mean and max). | This is true distillation (graph features → VLM). `data/processed/student/` **does not exist** and nothing reads `*_student.npz` or `vocabs.json`. **Dead code.** It is also slow, because it reloads the full graph `.npy` for every record. |
| `baselines/train_student_original.py` (training area) | **Inputs only**: the embedding plus, optionally, `violent.score`, people_count, hashed multi-hot actions/weapons/scene/notes. Targets are the GT `labels_dir`. | No circularity. Example output: `baselines/reference_runs/student_mlp/metrics.json` (`vlm_emb_meanpool` + `vlm_labels_meanpool`, `labels_multih`, feature_dim 3744, split by movie/source). Test ROC-AUC 0.622 and PR-AUC 0.151 at the 30 s horizon (val AUC 0.763). |
| `setup/curate_youtube.py analyze` | **Labels**: the first frame with `stage=="violence"` from the `anticipation_simple` prompt becomes `violence_start_second`, and the first `tension/escalation` becomes `escalation_start_second`. | Superseded (see next row). |
| `experiments/fast_analyze_curation.py` | **Labels**: VLM embeddings (`infer_embedding_only`, prompt `violence_embedding`) are scored by the LSTM `outputs/outputs_temporal/best_temporal_model.pt`, then thresholded (violence ≥0.75, tension ≥0.3) to give the stage timeline and violence/escalation start times. | **This produced all 92 analysed candidates** in `data/anticipation_curation/candidates.json` (each `stage_timeline` entry has `raw_score/score/stage`; notes like `max_score=…, analyzed_at_fps=0.5`). There is no human-verification field. 47 are flagged `has_violence`, and 34 have an anticipation window ≥10 s. **Any model trained or evaluated on these timestamps with VLM-embedding features learns to reproduce the old LSTM. That is circular, and the labels are pseudo-labels, not ground truth.** |
| `scripts/eval_vlm_teacher.py` | The teacher is compared *against* GT | Fine, but the GT itself is noisy (§7). |

---

## 9. Inference cost (measured where logged, otherwise estimated)

* **Memory:** 8.3 B params in bf16 is about 16.6 GB of weights. It fits on one GPU-80GB with room for full-resolution prefill. LoRA training at 336 px with batch 1 also fits on one GPU.
* **Teacher labelling (`infer_one` with an embedding):** greedy generation of up to 256 to 512 JSON tokens, plus a second full forward for the embedding, at native resolution.
  The docstring in `experiments/fast_analyze_curation.py` quotes about **6 s/frame** for text-generation analysis versus about **0.5 s/frame** for `infer_embedding_only`.
  (Those numbers are not re-measured. 6 s/frame is consistent with ~300 generated tokens on a 7B model.)
  One full pass over the dataset is 78,236 sampled frames (every 5th graph timestep, ~45 per video). At 6 s/frame that is about **130 GPU-hours** (inferred), or about 11 GPU-hours embeddings-only. It was done 4 times (old, meanpool, context-both, violence).
* **QA features:** batch mode is one generate of up to 200 tokens per frame. Individual mode is 25 generates per frame. With `--sample_rate 1` (every raw frame), a single 2-minute 24-fps video is about 2,900 frames, which makes full extraction impractical.
* **LoRA training:** about 2.7 it/s at 336×336, about 22 min per epoch for 3,600 frames, about 2 h 17 min for 5 epochs (from the log).
* **Realtime:** `realtime/rtsp_inference.py` runs embeddings on GPU0 and an LSTM on GPU1 at 1 fps sampling with sequence length 32 (`inference_*.log`).
  **Its embedding differs from the training embedding.** It uses the prompt "Describe the scene.", no system prompt, a resized image, a **mean over the whole sequence including text tokens, and no L2 normalisation**.
  That resembles `vlm_emb_old`, not the image-token-pooled, normalised `vlm_emb` the LSTMs were trained on.
  `scripts/analyze_youtube.py` is different: it uses `QwenTeacher.infer_embedding_only` (`violence_embedding`, context none), which matches `vlm_emb_violence`. See the realtime doc.

---

## 10. Issues checklist (most severe first)

1. LoRA evaluation leaks the target into `generate`, so its validation accuracy is constant and meaningless (§5.1).
2. LoRA and QA-probe labels: `labels_multih` max-over-horizons >0 means "violence within 120 s", and inside-violence frames (−1) are counted as "no violence" (§5.2, §6).
3. LoRA and QA-probe frames: timestep indices are treated as seconds (LoRA) or raw frames (QA probe). For LoRA, 71% of XD steps clamp to the final frame (§5.3, §6).
4. LLM dataset prompts leak the label (event-frame line, video id, fallback "annotated as") (§7).
5. UCF-Crime metadata labels: un-annotated anomaly videos, including Fighting, are labelled 0. The labels are noisy for the whole repo and push UCF teacher AUC to about 0.5 (§7).
6. The curated YouTube set's labels are LSTM-on-VLM pseudo-labels, so using them is circular (§8).
7. The realtime embedding extraction (`rtsp_inference.py`) does not match the training embeddings (§9).
8. `prompt_mode` has no effect on image-token embeddings, so the "violence-focused embedding prompts" premise is wrong (§2.1).
9. `vlm_labels_violence` has duplicate `(vid,t)` lines. Embeddings are dropped on parse failure, and the resume logic can misalign the structured features (§3).
10. `LLMEmbedder` returns 1.0 when parsing fails, and `LLMFusionHead(linear)` is untrained, so its output is ≥0.5 (§7).
11. Dead or unimplemented code: `vlm_build_student_dataset.py` (output never built or consumed), LoRA `--mode extract`/`--checkpoint`, `warmup_steps`, the multicue/simple `vlm_features` (never generated), and `infer_batch_sequential`/`extract_embeddings_batch` (not really batched).
12. Optimistic evaluation: `eval_vlm_teacher.py` picks the best F1 threshold on the evaluation data. `experiments/prompt_engineering_stats.py` reports LDA AUC = 1.0 by fitting and scoring on the same 3584-d samples (`results/prompt_engineering.csv`: Cohen's d 0.23, 336 dims with d>0.5, n=358 violent / 6631 non-violent).
13. Hard-coded or missing paths: `models/Qwen2.5-3B-Instruct` is missing. Video roots default to `data/raw_videos/{kinetics,ucf_crime,xd_violence}`. Scripts must run from the repo root (`python -m scripts.…`) so `vlm` and `core` import.
14. `requirements.txt` is incomplete (peft, bitsandbytes, scikit-learn, matplotlib, …).

---

## 11. Reusable assets for a new model

| Asset | Path | Shape / size | Reuse notes |
|---|---|---|---|
| VLM frame embeddings (current default) | `data/processed/vlm_emb/{ucf_crime,xd_violence}/<vid>.npz` | 1721 videos, `emb` fp16 [N≈T/5, 3584], `t`, `raw_frame_idx`; about 487 MB | L2-normalised image-token pool (inferred to include the previous frame). Index labels with `t` (graph timestep). |
| VLM embeddings, single frame | `data/processed/vlm_emb_violence/...` | same shapes | Pure current-frame pool (context none). This is the cleanest to match at inference via `QwenTeacher.infer_embedding_only`. |
| VLM embeddings, mean-pool | `data/processed/vlm_emb_meanpool/...` | same shapes | Used by `baselines/reference_runs/student_mlp`. |
| Teacher JSON per frame | `data/processed/vlm_labels*/.../<vid>.jsonl` | ~78k frames per variant | `violent.score` is a free zero-shot detector feature (AUC ~0.67). Actions, weapons, scene and notes are usable as weak text features. Deduplicate by `(vid,t)`. |
| VLM + audio concat | `data/processed/vlm_audio_features/xd_violence/` | 772 videos, 3675-d | XD only. |
| Curated YouTube embeddings | `data/processed/vlm_emb/curated/*.npz` | 92 videos, includes `timestamps`, `scores` | Labels are pseudo-labels. Use them only after human annotation. |
| QA probe features | `outputs/qa_feature_test/features.npz` | 2×[50,25] | Too small and misaligned to be useful. |
| LLM text dataset | `llm_data/*.jsonl` | 1376 / 345 videos | Labels are usable as video-level labels (noisy). The prompts are leaky and low-information. |
| LoRA adapters | `lora_finetuned_v2/best_model` (95 MB), `lora_finetuned/best_model` (10 MB) | PEFT r=8 | Trained on misaligned and contradictory labels. **Do not reuse.** |
| Teacher code | `vlm/qwen_teacher.py` | n/a | A sound wrapper for JSON prompting and image-token pooling. Reuse `_extract_embedding` / `infer_embedding_only` for consistent features. |

### Suggestions for the next attempt (inferred, not tested)
* Fix the labels first: use XD `annotations.txt` / UCF temporal annotations, and treat un-annotated UCF anomaly videos as *unknown* rather than negative.
* For anticipation, single-frame VLM embeddings are scene-dominated (within-video cosine ~0.95+). Use temporal deltas of embeddings, multi-frame or clip-level video inputs (Qwen2.5-VL supports video input), or a dedicated video backbone, and fuse with audio and pose.
* If a VLM is finetuned, train a light classification or regression head on pooled vision tokens with video-level splits. Keep the train-time and inference-time embedding code identical (one function).

## Notes moved from code comments

* `experiments/fast_analyze_curation.py` (old docstring estimate, not re-measured): embedding-only analysis is about 10 to 20x faster than the text-generation `analyze` command. The 92 curated videos at 0.2 fps take about 45 min, versus 60+ h with text generation. The script's `est_per_frame = 0.6` reflects about 0.5 to 0.7 s/frame for embedding extraction.

* `llm/llm_head_infer.py` was written for frame-level rows of the form `{id: "<video>|t=<sec>", video_id, frame_idx, t_sec, source, prompt, target_risk, risk_bin}`. No script in the repo produces that format, so it has only been run on the video-level `llm_data` rows. On those rows the target comes from `label`, because it reads `target_risk`, then `risk`, then `label`. Any `risk` field is assumed to be ground truth, not an earlier prediction.

* `scripts/vlm_label_dataset.py` (old docstring, author's intent, not validated): `--prompt_mode violence_embedding` was the recommended mode for violence-detection embeddings and `anticipation_multicue` for the anticipation task. `scene_context` was meant to run once per video as an environment baseline, not per frame.

* `scripts/eval_vlm_teacher.py`: global metrics and plots always use every usable (score, gt) pair, and `per_video.csv` lists every video. Only `video_aggregates_valid_only` in `summary.json` is filtered, to videos with at least `--min_samples_per_video` (5) samples and both GT classes. `warning_best_is_all_positive_baseline` is set when the best-F1 threshold is ~0 and its F1 is within 1e-3 of the always-positive baseline.

* `tuning/question_analysis/vlm_question_features.py` (old docstring, author's intent, not validated): the VQA approach was motivated by the claim that raw VLM embeddings are about 95% generic scene features, so explicit violence-indicator questions should be more discriminative. The negative-weight questions exist because sports, acting and play often look like violence. Batch-question mode was described as much faster but less accurate than per-question mode.
