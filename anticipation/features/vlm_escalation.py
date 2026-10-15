"""No-context conflict-phase probabilities and intermediate Qwen visual/prompt states."""

import argparse
import functools
import hashlib
import json
import os
import time
from contextlib import contextmanager

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "4")

import numpy as np

from anticipation.features import io

EXTRACTOR = "vlm_escalation"
HIDDEN_EXTRACTOR = "vlm_escalation_hidden"
VERSION = 3
PROMPT_VERSION = "conflict-phases-v3"
OUTPUT_VERSIONS = {EXTRACTOR: VERSION, HIDDEN_EXTRACTOR: VERSION}
MODEL_25 = "models/Qwen2.5-VL-7B-Instruct"
MODEL_3 = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_SMALL = "Qwen/Qwen2.5-VL-3B-Instruct"
STAGES = ("calm", "tension", "argument", "face_off", "contact", "violence")
CUES = (
    ("arguing_face_to_face", "People visibly argue or appear to shout at one another, judged only from mouth and body movement."),
    ("aggressive_gestures", "Someone points at another person or gestures aggressively."),
    ("approaching_following", "Someone approaches or follows another person."),
    ("personal_space_intrusion", "Someone intrudes into another person's personal space."),
    ("squaring_up", "People square up or adopt a confrontational fighting stance."),
    ("pushing_grabbing", "Someone pushes, shoves, or grabs another person's clothing or body."),
    ("intervention", "A third party blocks, separates, or physically holds someone back."),
    ("crowd_circle", "Bystanders gather around or form a circle around a confrontation."),
    ("bystanders_filming", "Bystanders raise phones to film the interaction."),
    ("weapon_visible", "A weapon is visibly present."),
    ("backing_away", "Someone retreats, turns away, or backs away from another person."),
    ("person_on_ground", "A person is on the ground during the interaction."),
    ("violence_now", "Active assault, punching, kicking, wrestling, or beating is visible."),
    ("conciliatory_gestures", "Someone makes calm open-palm, conciliatory, or apology-like gestures."),
    ("separation", "Antagonists separate without pursuit, or leave in different directions."),
    ("intervention_succeeds", "Intervention visibly settles the conflict and antagonists stop re-engaging."),
)
TRAJECTORY_CUES = (
    ("repeated_approach", "The same people repeatedly approach after retreat or separation."),
    ("persistent_conflict", "The visible conflict persists for more than a minute in this clip."),
    ("restrained_still_agitated", "Someone held back remains visibly agitated and tries to re-engage."),
    ("left_and_returned", "An antagonist leaves and returns to the confrontation."),
    ("intervention_failed", "An intervention fails and the confrontation resumes or continues."),
)
CONTEXT = (
    "Judge only this clip, in chronological order; the last frame is the present. No earlier verdicts "
    "or future frames are available. The clip is silent: do not infer spoken words or audible shouting. "
    "Ordinary walking, conversation, play, sport, or mere proximity are not hostile interaction. "
    "Stages: 0 calm, ordinary activity or resolved interaction; 1 tension, mutual hostile attention or "
    "agitation without a clear argument; 2 argument, visibly heated exchange or aggressive gesturing "
    "at a distance; 3 face-off, intimidation, squaring up, menacing pursuit, weapon threat or being held back; "
    "4 contact, hostile pushing, shoving, grabbing or closing into imminent physical confrontation; "
    "5 violence, active assault, striking, kicking, wrestling or beating. "
    "Rate the underlying interaction at the end. Intervention and retreat can mark a hot conflict; "
    "do not automatically lower the stage for either. Lower it when intervention succeeds, separation "
    "persists without pursuit, or conciliatory gestures visibly settle the interaction. A retreat followed "
    "by pursuit is not resolution. Conflict can persist without becoming violent. "
)
STAGE_QUESTION = "Which stage is currently supported? Answer exactly one digit, 0 through 5."
FORECAST_QUESTION = "From visible evidence only, will a physical fight start within 60 seconds? Answer 1 for yes or 0 for no."
STAGE_WEIGHTS = np.linspace(0, 1, len(STAGES), dtype=np.float32)
DEFAULTS = dict(model_id=MODEL_3, arm="B", stride_sec=None, max_pixels=448 * 448, max_edge=448,
                decode_threads=2, temperature=1.0, calibration="none", hidden_layers=(18, 27),
                save_hidden=True, include_early_baseline=True, rationale_samples=0, rationale_max_tokens=96, pose_gating=False)


def arm_config(arm, stride_sec=None):
    if arm not in ("A", "B", "C"):
        raise ValueError("arm must be A, B or C")
    window = {"A": 0, "B": 7, "C": 124}[arm]
    stride = stride_sec if stride_sec is not None else {"A": 2, "B": 2, "C": 8}[arm]
    if stride < 1:
        raise ValueError("stride_sec must be >= 1")
    return window, stride


def clip_times(t, arm="B"):
    if arm == "A":
        times = [t]
    elif arm == "B":
        times = np.arange(t - 7, t + 1)
    elif arm == "C":
        times = np.sort(np.r_[np.arange(t - 124, t + 1, 4), np.arange(t - 3, t + 1)])
    else:
        raise ValueError("Unknown arm")
    times = np.asarray(times, dtype=np.float64)
    return times[times >= 0] if arm == "C" else np.maximum(0, times)


def questions_and_columns(arm):
    cues = CUES + (TRAJECTORY_CUES if arm == "C" else ())
    questions = [STAGE_QUESTION] + [f"Is this visible: {q} Answer 1 or 0." for _, q in cues]
    columns = ["stage_" + s for s in STAGES] + ["cue_" + key for key, _ in cues]
    if arm == "B":
        questions.append(FORECAST_QUESTION)
        columns.append("forecast_fight_within60")
    return questions, columns + ["expected_risk"]


def hold(samples, values, sample_valid, ticks):
    out = np.zeros((len(ticks), values.shape[1]), dtype=values.dtype)
    valid = np.zeros(len(ticks), bool)
    idx = np.searchsorted(samples, ticks, side="right") - 1
    found = idx >= 0
    out[found], valid[found] = values[idx[found]], sample_valid[idx[found]]
    return out, valid


def pose_gate_samples(pose_path, samples):
    with np.load(pose_path, allow_pickle=False) as z:
        pt, ptr, tid, pose_ok = z["frame_t"], z["frame_ptr"], z["det_track_id"], z["frame_ok"]
        meta = json.loads(str(z["meta"])) if "meta" in z else {}
    gates = np.zeros(len(samples), bool)
    fps = meta.get("config", meta.get("cfg", {})).get("sample_fps")
    for i, t in enumerate(samples):
        left = max(0, t - 7)
        seen = np.flatnonzero((pt >= left) & (pt <= t))
        if not len(seen):
            raise ValueError("Pose gating has no causal coverage for a query")
        gaps = np.diff(pt[seen].astype(float))
        step = 1 / fps if fps else (float(np.median(gaps)) if len(gaps) else .2)
        if pt[seen[0]] > left + step + 1e-4 or pt[seen[-1]] < t - step - 1e-4 or np.any(gaps > 1.5 * step + 1e-4):
            raise ValueError("Pose gating has incomplete causal window coverage")
        gates[i] = bool(pose_ok[seen].all() and not any(np.any(tid[ptr[k]:ptr[k+1]] >= 0) for k in seen))
    return gates


def hidden_layout(layers, dim):
    columns, groups = [], []
    for layer in layers:
        for kind in ("visual", "stage_prompt"):
            start = len(columns)
            columns.extend(f"l{layer}_{kind}_{i}" for i in range(dim))
            groups.append(dict(layer=layer, kind=kind, start=start, stop=len(columns)))
    return columns, groups


class Scorer:
    def __init__(self, model_id, device, max_pixels, max_edge, layers):
        import torch
        from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration

        torch.set_num_threads(4)
        config = AutoConfig.from_pretrained(model_id, local_files_only=True)
        self.is_qwen3 = config.model_type == "qwen3_vl"
        self.device, self.max_pixels, self.max_edge = device, max_pixels, max_edge
        self.layers = tuple(layers)
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        self.processor.tokenizer.padding_side = "left"
        cls = Qwen3VLForConditionalGeneration if self.is_qwen3 else Qwen2_5_VLForConditionalGeneration
        self.model = cls.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                         attn_implementation="sdpa", config=config, revision=getattr(config, "_commit_hash", None),
                                         local_files_only=True).to(device).eval()
        self.attach_hooks()
        self.priors = {}

    def attach_hooks(self):
        text = self.model.model.language_model
        self.hidden_size = text.config.hidden_size
        self._mode, self._visual_mask, self._captured = None, None, {}
        self._hooks = []
        for layer in self.layers:
            if layer < 1 or layer > len(text.layers):
                raise ValueError(f"Layer {layer} outside 1..{len(text.layers)}")
            self._hooks.append(text.layers[layer - 1].register_forward_hook(self._hook(layer)))

    def _hook(self, layer):
        def capture(module, args, output):
            if self._mode is None:
                return
            import torch.nn.functional as F

            h = output[0] if isinstance(output, tuple) else output
            if self._mode == "visual":
                selected = h[0, self._visual_mask].float()
                value = F.layer_norm(selected.mean(0), (selected.shape[-1],))
            else:
                selected = h[0, -1].float()
                value = F.layer_norm(selected, (selected.shape[-1],))
            self._captured[(layer, self._mode)] = value.detach()
        return capture

    @contextmanager
    def capture(self, mode, mask=None):
        self._mode, self._visual_mask = mode, mask
        try:
            yield
        finally:
            self._mode, self._visual_mask = None, None

    def texts(self, questions, arm):
        kind = "image" if arm == "A" else "video"
        return [self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": kind}, {"type": "text", "text": CONTEXT + q}]}],
            tokenize=False, add_generation_prompt=True) for q in questions]

    def process(self, frames, texts, times, arm):
        from PIL import Image

        h, w = frames.shape[1:3]
        if max(h, w) > self.max_edge:
            scale = self.max_edge / max(h, w)
            size = (max(1, round(w * scale)), max(1, round(h * scale)))
            frames = np.stack([np.asarray(Image.fromarray(f).resize(size, Image.Resampling.BICUBIC)) for f in frames])
        size = {"shortest_edge": 4 * 32 * 32, "longest_edge": self.max_pixels}
        if arm == "A":
            return self.processor(text=texts, images=[frames[0]] * len(texts), size=size,
                                  padding=True, return_tensors="pt").to(self.device)
        kwargs = dict(do_sample_frames=False, size=size)
        if self.is_qwen3:
            # Qwen3's video size budget covers all frames, unlike Qwen2.5's per-frame budget.
            size = {k: v * len(frames) for k, v in size.items()}
            rel = np.round((times - times[0]) * 1000).astype(int).tolist()
            kwargs.update(size=size, video_metadata=[dict(fps=1000, total_num_frames=max(rel) + 1,
                                                         frames_indices=rel)] * len(texts))
        else:
            if arm == "C":
                raise ValueError("Arm C requires Qwen3 timestamp-aware video input")
            kwargs["fps"] = 1.0
        return self.processor(text=texts, videos=[frames] * len(texts), padding=True,
                              return_tensors="pt", **kwargs).to(self.device)

    def logits(self, frames, times, arm, questions, capture_hidden=True, reuse_prefix=True):
        import torch

        texts = self.texts(questions, arm)
        self._captured = {}
        with torch.inference_mode():
            if not reuse_prefix:
                inputs = self.process(frames, texts, times, arm)
                output = self.model(**inputs, use_cache=False, logits_to_keep=1)
            else:
                prefix = os.path.commonprefix(texts).rstrip()
                inputs = self.process(frames, [prefix], times, arm)
                visual = (inputs.input_ids[0] == self.model.config.image_token_id) | (inputs.input_ids[0] == self.model.config.video_token_id)
                if not visual.any():
                    raise ValueError("No visual tokens in prefix")
                with self.capture("visual" if capture_hidden else None, visual):
                    output = self.model(**inputs, use_cache=True, logits_to_keep=1)
                cache = output.past_key_values
                delta = self.model.model.rope_deltas.clone()
                cache.batch_repeat_interleave(len(texts))
                suffix = self.processor.tokenizer([t[len(prefix):] for t in texts], padding=True,
                                                  add_special_tokens=False, return_tensors="pt").to(self.device)
                length, mask = inputs.input_ids.shape[1], suffix.attention_mask
                # Ignore the left-padding gap between the cached prefix and each continuation in RoPE.
                positions = (mask.cumsum(-1) - 1 + length + delta[0, 0]).unsqueeze(0).expand(3, -1, -1)
                attention = torch.cat([inputs.attention_mask.repeat(len(texts), 1), mask], dim=1)
                with self.capture("stage_prompt" if capture_hidden else None):
                    output = self.model(input_ids=suffix.input_ids, attention_mask=attention,
                                        past_key_values=cache, position_ids=positions,
                                        cache_position=torch.arange(length, length + mask.shape[1], device=self.device),
                                        use_cache=False, logits_to_keep=1)
        logits = output.logits[:, -1].float()
        options = [list("012345")] + [["1", "0"]] * (len(questions) - 1)
        selected = []
        for i, words in enumerate(options):
            ids = [self.processor.tokenizer.encode(w, add_special_tokens=False) for w in words]
            if any(len(x) != 1 for x in ids):
                raise ValueError("Every answer option must be exactly one token")
            selected.append(logits[i, [x[0] for x in ids]].cpu().numpy())
        hidden = None
        if capture_hidden and reuse_prefix and self.layers:
            hidden = np.concatenate([self._captured[(layer, kind)].cpu().numpy()
                                     for layer in self.layers for kind in ("visual", "stage_prompt")])
        self._captured = {}
        return selected, hidden

    def score(self, frames, times, arm, temperature=1, calibration="none", save_hidden=True):
        questions, _ = questions_and_columns(arm)
        key = (arm, frames.shape, tuple(times - times[0]))
        if calibration == "blank" and key not in self.priors:
            self.priors[key] = self.logits(np.full_like(frames, 127), times, arm, questions, False)[0]
        logits, hidden = self.logits(frames, times, arm, questions, save_hidden)
        probabilities = []
        for i, x in enumerate(logits):
            if calibration == "blank":
                x = x - self.priors[key][i]
            x = np.exp((x - x.max()) / temperature)
            probabilities.append(x / x.sum())
        stage = probabilities[0]
        rating = np.asarray([*stage, *[p[0] for p in probabilities[1:]], stage @ STAGE_WEIGHTS], np.float32)
        return rating, hidden

    def rationale(self, frames, times, arm, max_tokens):
        import torch

        question = "Briefly explain the visible conflict phase, intervention or separation and uncertainty."
        inputs = self.process(frames, self.texts([question], arm), times, arm)
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        return self.processor.tokenizer.decode(ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)


@functools.lru_cache(maxsize=8)
def _hidden_size(model_id):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_id, local_files_only=True).text_config.hidden_size


@functools.lru_cache(maxsize=1)
def _scorer(model_id, device, max_pixels, max_edge, layers):
    return Scorer(model_id, device, max_pixels, max_edge, layers)


def extract_all(video_path, device="cuda:1", **cfg):
    import torch

    if device != "cpu" and device != "cuda:1":
        raise ValueError("Only cpu or physical cuda:1 is permitted")
    if device == "cuda:1" and os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Unset CUDA_VISIBLE_DEVICES to preserve physical cuda:1")
    allowed = set(DEFAULTS) | {"max_duration_sec", "sidecar_path", "targets_path", "pose_path", "query_start_sec", "deadline_monotonic", "checkpoint_path"}
    if set(cfg) - allowed:
        raise ValueError(f"Unknown config keys: {sorted(set(cfg) - allowed)}")
    c = {**DEFAULTS, **{k: v for k, v in cfg.items() if k in DEFAULTS}}
    window, stride = arm_config(c["arm"], c["stride_sec"])
    if c["decode_threads"] not in range(1, 5) or c["temperature"] <= 0 or c["calibration"] not in ("none", "blank"):
        raise ValueError("Invalid threads, temperature or calibration")
    layers = tuple(c["hidden_layers"]) if c["save_hidden"] else ()
    torch.set_num_threads(4)
    video = io.Video(video_path, num_threads=c["decode_threads"])
    duration = min(video.info.duration_sec, cfg.get("max_duration_sec", float("inf")))
    ticks = io.tick_times(duration)
    start = max(1, 1 + np.floor((cfg.get("query_start_sec", 1) - 1) / stride) * stride)
    samples = np.arange(start, len(ticks) + 1, stride, dtype=np.float32)
    if c["include_early_baseline"] and start > 1:
        samples = np.unique(np.r_[np.arange(1, min(start, 11, len(ticks) + 1), stride), samples]).astype(np.float32)
    questions, columns = questions_and_columns(c["arm"])
    values = np.zeros((len(samples), len(columns)), np.float32)
    valid_samples = np.zeros(len(samples), bool)
    scorer = _scorer(c["model_id"], device, c["max_pixels"], c["max_edge"], layers) if len(samples) else None
    hidden_columns, groups = hidden_layout(layers, scorer.hidden_size if scorer else _hidden_size(c["model_id"]))
    hidden_values = np.zeros((len(samples), len(hidden_columns)), np.float16)
    histories = np.minimum(samples, window).tolist()
    errors, rationales = [], []
    gated = np.zeros(len(samples), bool)
    if c["pose_gating"]:
        if not cfg.get("pose_path"):
            raise ValueError("pose_gating requires an explicit pose_raw NPZ")
        gated = pose_gate_samples(cfg["pose_path"], samples)
    checkpoint = cfg.get("checkpoint_path")
    identity = dict(version=VERSION, cfg=c, duration_sec=duration, start=float(start),
                    model_revision=getattr(scorer.model.config, "_commit_hash", None) if scorer else None,
                    media_path=os.path.realpath(video_path), media_size=os.stat(video_path).st_size,
                    media_mtime_ns=os.stat(video_path).st_mtime_ns,
                    prompt=hashlib.sha256(json.dumps([CONTEXT, questions]).encode()).hexdigest())
    if cfg.get("pose_path"):
        stat = os.stat(cfg["pose_path"])
        identity["pose"] = [os.path.realpath(cfg["pose_path"]), stat.st_size, stat.st_mtime_ns]
    identity = json.dumps(identity, sort_keys=True)
    progress, score_sec = 0, 0.0
    if checkpoint and os.path.isfile(checkpoint):
        with np.load(checkpoint, allow_pickle=False) as z:
            if str(z["identity"]) != identity or not np.array_equal(z["samples"], samples):
                raise ValueError("Checkpoint inputs or prompt changed")
            values[:], valid_samples[:], hidden_values[:] = z["values"], z["valid"], z["hidden"]
            saved = json.loads(str(z["state"]))
            progress, score_sec = saved["progress"], saved["score_sec"]
            errors, rationales = saved["errors"], saved["rationales"]
    started = time.perf_counter()
    try:
        for i in range(progress, len(samples)):
            t = samples[i]
            if time.perf_counter() >= cfg.get("deadline_monotonic", float("inf")):
                raise TimeoutError("Bounded GPU chunk expired")
            if gated[i]:
                values[i, 0] = 1
                valid_samples[i] = True
                progress = i + 1
                continue
            times = clip_times(float(t), c["arm"])
            try:
                frames = video.frames_at(times)
            except TimeoutError:
                raise
            except Exception as exc:
                errors.append(dict(t_sec=float(t), error=str(exc)))
                progress = i + 1
                continue
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            begin = time.perf_counter()
            values[i], hidden = scorer.score(frames, times, c["arm"], c["temperature"], c["calibration"], bool(layers))
            if layers:
                hidden_values[i] = hidden
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            score_sec += time.perf_counter() - begin
            valid_samples[i] = True
            if len(rationales) < c["rationale_samples"] and cfg.get("sidecar_path"):
                rationales.append(dict(t_sec=float(t), frame_times_sec=times.tolist(),
                                       frame_indices=video.frame_index(times).tolist(),
                                       rationale=scorer.rationale(frames, times, c["arm"], c["rationale_max_tokens"])))
            progress = i + 1
    except TimeoutError:
        if checkpoint:
            os.makedirs(os.path.dirname(os.path.abspath(checkpoint)), exist_ok=True)
            state = dict(progress=progress, score_sec=score_sec, errors=errors, rationales=rationales)
            with open(checkpoint + ".tmp", "wb") as f:
                np.savez_compressed(f, identity=identity, samples=samples, values=values,
                                    valid=valid_samples, hidden=hidden_values, state=json.dumps(state))
            os.replace(checkpoint + ".tmp", checkpoint)
        raise
    feat, valid = hold(samples, values.astype(np.float16), valid_samples, ticks)
    hidden_feat, hidden_valid = hold(samples, hidden_values, valid_samples & ~gated, ticks)
    gap = (ticks > 10) & (ticks < start) if c["include_early_baseline"] else ticks < start
    feat[gap], hidden_feat[gap], valid[gap], hidden_valid[gap] = 0, 0, False, False
    prompt_hash = hashlib.sha256(json.dumps([PROMPT_VERSION, CONTEXT, questions]).encode()).hexdigest()
    meta = dict(extractor=EXTRACTOR, version=VERSION, model_id=c["model_id"],
                model_revision=getattr(scorer.model.config, "_commit_hash", None) if scorer else None,
                window_sec=max(window, 7 if c["pose_gating"] else 0) + stride, clip_window_sec=window, D=len(columns), columns=columns,
                prompt_version=PROMPT_VERSION, prompt_sha256=prompt_hash, context=CONTEXT, questions=questions,
                forecast_prompt_sha256=hashlib.sha256(FORECAST_QUESTION.encode()).hexdigest() if c["arm"] == "B" else None,
                cfg=c, duration_sec=duration, no_context=True, pose_provenance=json.loads(identity).get("pose"),
                calibration="restricted single-token option softmax; no empirical forecast calibration", kv_cache_scope="one update only",
                update_t_sec=samples.tolist(), query_start_sec=float(start), available_history_sec=histories, gated=gated.tolist(),
                stage_weights=STAGE_WEIGHTS.tolist(), sample_ticks=len(samples),
                valid_samples=int(valid_samples.sum()), score_gpu_wall_sec=score_sec,
                extraction_wall_sec=time.perf_counter() - started, decode_errors=errors)
    hmeta = {**meta, "extractor": HIDDEN_EXTRACTOR, "D": len(hidden_columns), "columns": hidden_columns,
             "column_groups": groups, "normalization": "visual-token mean then non-affine LayerNorm; stage-prompt last token LayerNorm",
             "layer_indexing": "one-based decoder-block output, before final model norm"}
    if cfg.get("targets_path"):
        import pandas as pd

        path = cfg["targets_path"]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        target = pd.DataFrame(values, columns=columns)
        target.insert(0, "gated", gated)
        target.insert(0, "available_history_sec", histories)
        target.insert(0, "valid", valid_samples)
        target.insert(0, "t_sec", samples)
        target.to_parquet(path + ".tmp", index=False)
        os.replace(path + ".tmp", path)
    if cfg.get("sidecar_path"):
        path = cfg["sidecar_path"]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path + ".tmp", "w") as f:
            json.dump(dict(prompt_version=PROMPT_VERSION, prompt_sha256=prompt_hash, model_id=c["model_id"], samples=rationales), f, indent=2)
        os.replace(path + ".tmp", path)
    outputs = {EXTRACTOR: (ticks, feat, valid, meta)}
    if layers:
        outputs[HIDDEN_EXTRACTOR] = (ticks, hidden_feat, hidden_valid, hmeta)
    if checkpoint and os.path.isfile(checkpoint):
        os.unlink(checkpoint)
    return outputs


def extract(video_path, device="cuda:1", **cfg):
    return extract_all(video_path, device, **cfg)[EXTRACTOR]


def extract_and_save(video_path, video_id, device="cuda:1", output_name=EXTRACTOR, **cfg):
    base = io.feature_path(output_name, video_id)
    cfg.setdefault("checkpoint_path", base.replace(".npz", ".checkpoint"))
    cfg.setdefault("targets_path", base.replace(".npz", ".targets.parquet"))
    cfg.setdefault("sidecar_path", base.replace(".npz", ".rationales.json"))
    outputs = extract_all(video_path, device, **cfg)
    paths = {}
    for name, (t, f, v, meta) in outputs.items():
        name = output_name + ("_hidden" if name == HIDDEN_EXTRACTOR else "")
        meta["extractor"] = name
        paths[name] = io.save_features(name, video_id, t, f, v, meta)
    return dict(paths=paths, duration_sec=outputs[EXTRACTOR][3]["duration_sec"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video_path")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--model-id", default=DEFAULTS["model_id"])
    p.add_argument("--video-id")
    p.add_argument("--output-name", default=EXTRACTOR)
    p.add_argument("--feature-root")
    p.add_argument("--check-causal", type=float)
    p.add_argument("--cfg", default="{}")
    a = p.parse_args()
    cfg = {**json.loads(a.cfg), "model_id": a.model_id}
    if a.feature_root:
        io.FEATURE_ROOT = a.feature_root
    if a.check_causal is not None:
        result = io.check_causal(lambda path: extract(path, a.device, **cfg)[:3], a.video_path, a.check_causal)
        print(json.dumps(result))
        raise SystemExit(0 if result["causal"] else 1)
    if a.video_id:
        print(json.dumps(extract_and_save(a.video_path, a.video_id, a.device, a.output_name, **cfg)))
    else:
        outputs = extract_all(a.video_path, a.device, **cfg)
        print(json.dumps({k: dict(shape=v[1].shape, valid=int(v[2].sum()), meta=v[3]) for k, v in outputs.items()}))


if __name__ == "__main__":
    main()
