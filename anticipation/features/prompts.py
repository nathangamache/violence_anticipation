"""Text-prompt concept scores per 1 Hz tick (extractor ``siglip2_prompts``) and the versioned prompt list.

What
    ``PROMPTS`` is a list of ``(key, group, text)``. For every tick, ``feat[t, j]`` is the cosine similarity
    between the L2-normalized SigLIP 2 image embedding of the frame at t (extractor ``siglip2``, see
    siglip.py) and the L2-normalized text embedding of ``PROMPTS[j]``. window_sec = 0 (one frame at t), so the
    scores are causal by construction. D = len(PROMPTS).

    Groups:
      escalation    pre-violence cues that can lead an attack by 5 to 120 s (arguing, pointing, squaring up,
                    approaching, shoving, crowd forming, people being held back, weapon shown)
      violence_now  concurrent cues (punching, kicking, brawl, wrestling on the ground). Expected to rise only
                    at or after onset; useful as a detection auxiliary and for onset QA, not as a lead cue.
      calm          contrasts (calm street, people talking calmly, empty store, traffic)
      context       camera type and provenance (CCTV, dashcam, bodycam, phone, movie scene, news, captions).
                    These are scene priors and shortcut detectors: a model that leans on them is probably
                    learning provenance, not escalation (BASELINE.md section 6.5, B-scene0).

    SigLIP scores are sigmoid-trained, so a probability-like score is ``sigmoid(logit_scale * cos +
    logit_bias)``; both constants are stored in ``meta`` as the float32 checkpoint values (``logit_scale``
    109.894, ``logit_bias`` -15.9326; version 1 files stored bf16-rounded 108.5 / -15.9375). Most prompts stay
    far below 0.5 on real frames; use cosines (or z-scores per prompt) as model inputs.

Editing the prompts (versioning)
    Edit ``PROMPTS`` and bump ``PROMPT_SET_VERSION``. Every feature file stores the version, the sha1 of the
    prompt list (``prompt_set_hash``) and the full list in ``meta["prompts"]``; a mismatch between the code and
    the cache (prompt hash, this extractor's ``VERSION``, or the ``siglip2`` version the scores were computed
    from, ``meta["backbone_version"]``) is detected by ``--rescore`` and by ``check_cache``. Keys are stable
    column names: do not reuse a key for a different meaning. Re-scoring does not need the video: ``--rescore`` recomputes every
    ``siglip2_prompts`` file from the cached ``siglip2`` embeddings in seconds.

One-decode entry points (recommended for bulk extraction)
    ``extract_multi(path, device)``  -> {"siglip2": ..., "siglip2_prompts": ...}
    ``extract_visual(path, device)`` -> {"vjepa2_l": ..., "siglip2": ..., "siglip2_prompts": ...}; SigLIP reuses
    the frames V-JEPA already decoded. Each value is ``(t_sec, feat, valid, meta)``. Outputs are bitwise
    identical to the single-extractor functions.
    ``extract_and_save(path, video_id, device, **cfg)`` (registry ``visual``) also writes the three files.
    Config routing for these two (``split_visual_cfg``): keys only V-JEPA knows go to V-JEPA; the decode keys
    ``decode_threads`` and ``chunk_ticks`` go to V-JEPA, which does the only decode; ``dtype`` and ``batch_size``
    go to both models; ``vjepa_cfg={...}`` / ``siglip_cfg={...}`` set a key for one model only and win (from the
    runner: ``--cfg 'vjepa_cfg={"batch_size": 32}'``). Unknown keys are listed in every meta's ``ignored_cfg``;
    the effective per-model config is in ``meta["cfg"]``.

Version 2 (``siglip2_prompts_v2``, ``PROMPTS_V2``, ``PROMPT_SET_VERSION_V2``)
    22 concepts, each with 3 phrasings whose normalized text embeddings are averaged; rescored on CPU from cached
    ``siglip2`` files only (``--rescore --v2``). The v1 constants and outputs are unchanged (docs/FEATURES.md).

Usage
    .venv/bin/python -m anticipation.features.prompts --list
    .venv/bin/python -m anticipation.features.prompts VIDEO.mp4 --device cuda:1 [--video-id xd/<stem>] [--check-causal 60]
    .venv/bin/python -m anticipation.features.prompts VIDEO.mp4 --device cuda:1 --all-visual --video-id xd/<stem>
    .venv/bin/python -m anticipation.features.prompts --rescore --device cuda:1    # all cached siglip2 files

    from anticipation.features import prompts
    t_sec, feat, valid, meta = prompts.extract(path, device="cuda:1")
    outs = prompts.extract_visual(path, device="cuda:1")
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

from anticipation.features import io, siglip, vjepa

BACKBONE = siglip.EXTRACTOR
EXTRACTOR = f"{BACKBONE}_prompts"
VERSION = 2  # 2: float32 logit_scale / logit_bias in meta, computed from siglip2 version 2 embeddings
PROMPT_SET_VERSION = "v1"
DEFAULTS = dict(siglip.DEFAULTS)

# (key, group, text). Keep texts short, concrete and visual: SigLIP sees one frame, not motion or sound.
PROMPTS = [
    # ---- escalation: pre-violence interpersonal cues ----
    ("argue_face_to_face", "escalation", "two people arguing face to face"),
    ("argue_street", "escalation", "two men arguing angrily on the street"),
    ("point_aggressive", "escalation", "a man pointing aggressively at another man"),
    ("shout_at_person", "escalation", "a person shouting angrily at another person"),
    ("chest_to_chest", "escalation", "two men standing chest to chest in a confrontation"),
    ("in_face", "escalation", "a person getting in someone's face"),
    ("fists_clenched", "escalation", "a man with clenched fists squaring up to fight"),
    ("raise_fist", "escalation", "a man raising his fist threateningly"),
    ("approach_aggressive", "escalation", "a man walking aggressively toward another person"),
    ("chase_follow", "escalation", "an angry man following another person"),
    ("shove", "escalation", "people shoving each other"),
    ("push_away", "escalation", "a man pushing another man away with both hands"),
    ("grab_clothes", "escalation", "a person grabbing someone by the shirt"),
    ("held_back", "escalation", "a person being held back by friends"),
    ("separate", "escalation", "someone stepping between two people to separate them"),
    ("standoff_groups", "escalation", "a tense standoff between two groups of people"),
    ("crowd_circle", "escalation", "a crowd gathering in a circle around two people"),
    ("surround_person", "escalation", "a group of people surrounding a man"),
    ("filming_bystanders", "escalation", "bystanders filming a confrontation with their phones"),
    ("road_rage_argue", "escalation", "a driver arguing with another driver next to their cars"),
    ("driver_exits", "escalation", "an angry driver getting out of his car"),
    ("customer_yell", "escalation", "an angry customer yelling at an employee behind the counter"),
    ("police_confront", "escalation", "a police officer confronting an agitated man"),
    ("remove_jacket", "escalation", "a man taking off his shirt to fight"),
    ("knife_shown", "escalation", "a person holding a knife"),
    ("gun_shown", "escalation", "a person pointing a gun"),
    ("bat_shown", "escalation", "a person holding a baseball bat as a weapon"),
    # ---- violence_now: concurrent cues (detection, onset QA) ----
    ("punching", "violence_now", "people punching each other"),
    ("kick_ground", "violence_now", "a person kicking someone lying on the ground"),
    ("wrestle_ground", "violence_now", "two people wrestling on the ground"),
    ("brawl", "violence_now", "a brawl with many people fighting"),
    ("beaten", "violence_now", "a person being beaten up"),
    ("knocked_down", "violence_now", "a person knocked down after being hit"),
    ("police_tackle", "violence_now", "police officers tackling and restraining a person"),
    ("panic_run", "violence_now", "people running away in panic"),
    # ---- calm: contrasts ----
    ("calm_street", "calm", "a calm street with people walking"),
    ("talk_calm", "calm", "people talking calmly and laughing"),
    ("handshake", "calm", "two people shaking hands"),
    ("hug", "calm", "two people hugging"),
    ("queue_calm", "calm", "people waiting calmly in line"),
    ("restaurant_calm", "calm", "people sitting at tables in a restaurant"),
    ("store_empty", "calm", "an empty store aisle"),
    ("parking_empty", "calm", "an empty parking lot"),
    ("traffic", "calm", "cars driving on a road"),
    ("office_work", "calm", "people working quietly in an office"),
    ("sports_play", "calm", "people playing a sport together"),
    ("party_dance", "calm", "people dancing at a party"),
    # ---- context: camera, setting and provenance ----
    ("cctv_store", "context", "a security camera view of a store"),
    ("cctv_grainy", "context", "grainy black and white surveillance camera footage"),
    ("dashcam", "context", "a dashcam view from inside a car"),
    ("bodycam", "context", "a police body camera video"),
    ("phone_filming", "context", "a person filming on a phone"),
    ("shaky_phone", "context", "a shaky handheld phone video"),
    ("movie_scene", "context", "a scene from a hollywood movie"),
    ("news_broadcast", "context", "a television news broadcast with a reporter"),
    ("text_overlay", "context", "a video frame with large text captions"),
    ("night_outside_bar", "context", "a street outside a bar at night"),
    ("nightclub", "context", "a crowded nightclub"),
]


def prompt_keys():
    return [k for k, _, _ in PROMPTS]


def prompt_texts():
    return [t for _, _, t in PROMPTS]


EXTRACTOR_V2 = f"{EXTRACTOR}_v2"
VERSION_V2 = 2
PROMPT_SET_VERSION_V2 = "v2"

# (key, group, [phrasings]); the concept embedding is the L2-normalized mean of its phrasings' normalized embeddings.
# Context concepts exist only as shortcut detectors (BASELINE.md section 6.5).
PROMPTS_V2 = [
    ("phase_calm", "conflict_phase", ["people calmly going about their day", "a relaxed peaceful scene with people talking",
                                       "people standing around casually and chatting"]),
    ("phase_tension", "conflict_phase", ["a tense moment between people", "people looking angry and uneasy at each other",
                                          "an uncomfortable confrontation about to start"]),
    ("phase_argument", "conflict_phase", ["two people arguing heatedly", "a person yelling at another person",
                                           "an angry verbal dispute between people"]),
    ("phase_faceoff", "conflict_phase", ["two people standing face to face glaring at each other",
                                          "two men squaring off close together", "a tense standoff between two people"]),
    ("phase_contact", "conflict_phase", ["people physically bumping and grabbing each other",
                                          "the first physical contact in a confrontation", "a man shoving another man"]),
    ("phase_intervention", "conflict_phase", ["a bystander stepping in between two people",
                                               "friends pulling two people apart", "someone intervening to stop a fight"]),
    ("phase_separation", "conflict_phase", ["two people being separated and pulled apart",
                                             "people backing away from each other after a confrontation",
                                             "a group splitting two people who were arguing"]),
    ("phase_violence", "conflict_phase", ["people fighting and hitting each other", "a violent physical fight",
                                           "a person punching and kicking another person"]),
    ("phase_deescalation", "conflict_phase", ["two angry people calming down", "people shaking hands after an argument",
                                               "a tense situation ending peacefully"]),
    ("push", "contact_aggression", ["people pushing each other", "a man pushing another man in the chest",
                                     "someone shoving a person backward"]),
    ("grab", "contact_aggression", ["a man grabbing another man", "a person grabbing someone by the collar",
                                     "one man gripping another man's arm aggressively"]),
    ("fight_stance", "contact_aggression", ["a man in a fighting stance", "a person getting ready to fight with fists up",
                                             "two men squaring up in fighting stances"]),
    ("raised_fists", "contact_aggression", ["raised fists", "a man raising his fist to hit someone",
                                             "a person with a clenched fist ready to punch"]),
    ("held_back", "contact_aggression", ["someone being held back", "a man restrained by friends from attacking another",
                                          "a person being pulled away from a confrontation"]),
    ("walk_away", "deescalation", ["people walking away from each other", "a man turning and walking away from an argument",
                                    "two people leaving in opposite directions"]),
    ("apologize", "deescalation", ["a person apologizing with open hands", "a man holding up his palms to calm things down",
                                    "someone gesturing peacefully with open hands"]),
    ("calm_two", "deescalation", ["a bystander calming two people", "a person talking two angry people down",
                                   "someone mediating between two arguing people"]),
    ("ctx_cctv", "context", ["a security camera view", "grainy surveillance camera footage from above",
                              "a fixed CCTV camera recording"]),
    ("ctx_movie", "context", ["a scene from a movie", "a cinematic film shot with professional lighting",
                               "a dramatic scene from a hollywood film"]),
    ("ctx_news", "context", ["a TV news broadcast", "a news reporter on television with a lower third banner",
                              "footage in a television news report"]),
    ("ctx_captions", "context", ["a video frame with captions or text overlay", "large subtitles over a video",
                                  "a screenshot with a text watermark"]),
    ("ctx_phone", "context", ["a video filmed on a phone", "shaky handheld smartphone footage",
                               "a vertical video recorded by a bystander"]),
]


def prompt_keys_v2():
    return [k for k, _, _ in PROMPTS_V2]


def prompt_set_hash_v2():
    return hashlib.sha1(json.dumps([[k, g, t] for k, g, t in PROMPTS_V2]).encode()).hexdigest()[:12]


_TEXT_CACHE_V2 = {}


def text_embeddings_v2(device="cpu", dtype="fp32"):
    """Concept embeddings [C, D] (float32 CPU): mean of the L2-normalized phrasing embeddings, re-normalized."""
    key = (device, dtype, prompt_set_hash_v2())
    if key not in _TEXT_CACHE_V2:
        import torch

        flat = [t for _, _, ts in PROMPTS_V2 for t in ts]
        emb = siglip.embed_texts(flat, device, dtype)
        out, j = [], 0
        for _, _, ts in PROMPTS_V2:
            out.append(torch.nn.functional.normalize(emb[j:j + len(ts)].mean(0), dim=-1))
            j += len(ts)
        _TEXT_CACHE_V2[key] = torch.stack(out)
    return _TEXT_CACHE_V2[key]


def _meta_v2(backbone_meta, device, dtype):
    scale, bias = siglip.logit_params(device, dtype)
    return dict(
        extractor=EXTRACTOR_V2, version=VERSION_V2, backbone=BACKBONE, model_id=siglip.MODEL_ID,
        model_revision=backbone_meta.get("model_revision"), window_sec=0.0, D=len(PROMPTS_V2),
        columns=prompt_keys_v2(), prompt_set_version=PROMPT_SET_VERSION_V2, prompt_set_hash=prompt_set_hash_v2(),
        prompts=[{"key": k, "group": g, "texts": t} for k, g, t in PROMPTS_V2],
        groups=sorted({g for _, g, _ in PROMPTS_V2}), backbone_version=backbone_meta.get("version"),
        score="cosine(image_emb, concept_emb); concept_emb = normalized mean of normalized phrasing embeddings; "
              "prob = sigmoid(logit_scale * cos + logit_bias)",
        logit_scale=scale, logit_bias=bias, text_preprocessing="lowercase, pad to 64 tokens",
        text_device=device, text_dtype=dtype, v1_text_dtype="bf16 on cuda", video_fps=backbone_meta.get("video_fps"),
        duration_sec=backbone_meta.get("duration_sec"), n_decode_errors=backbone_meta.get("n_decode_errors"),
    )


def from_backbone_v2(t_sec, image_feat, valid, backbone_meta, device="cpu", dtype="fp32"):
    """v2 concept scores from already computed ``siglip2`` outputs; invalid ticks are zero."""
    feat = score(image_feat, text_embeddings_v2(device, dtype))
    feat[~np.asarray(valid, dtype=bool)] = 0.0
    return t_sec, feat.astype(np.float16), valid, _meta_v2(backbone_meta, device, dtype)


def check_cache_v2(meta, backbone_meta=None):
    ok = meta.get("prompt_set_hash") == prompt_set_hash_v2() and meta.get("version") == VERSION_V2
    if backbone_meta is not None:
        ok = ok and meta.get("backbone_version") == backbone_meta.get("version")
    return ok


def rescore_v2_and_save(video_path, video_id, device="cpu", **cfg):
    """Registry saver ``siglip2_prompts_v2``: rescore one video from its cached siglip2 file (no decoding)."""
    t_sec, img, valid, bmeta = io.load_features(BACKBONE, video_id)
    path = io.save_features(EXTRACTOR_V2, video_id, *from_backbone_v2(t_sec, img, valid, bmeta, device))
    return dict(paths={EXTRACTOR_V2: path}, duration_sec=bmeta.get("duration_sec"))


def coverage_v2():
    """(n siglip2 files, n current siglip2_prompts_v2 files) so a stale snapshot is visible."""
    root = os.path.join(io.FEATURE_ROOT, BACKBONE)
    n_src = n_ok = 0
    for path in glob.glob(os.path.join(root, "**", "*.npz"), recursive=True):
        if path.endswith(".tmp.npz"):
            continue
        n_src += 1
        out = io.feature_path(EXTRACTOR_V2, os.path.relpath(path, root)[:-len(".npz")])
        if os.path.exists(out) and check_cache_v2(_read_meta(out), _read_meta(path)):
            n_ok += 1
    return n_src, n_ok


def rescore_all_v2(device="cpu", dtype="fp32", force=False):
    """Recompute ``siglip2_prompts_v2`` for every cached ``siglip2`` file. Returns the number written."""
    root = os.path.join(io.FEATURE_ROOT, BACKBONE)
    n = 0
    for path in sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True)):
        if path.endswith(".tmp.npz"):
            continue
        video_id = os.path.relpath(path, root)[:-len(".npz")]
        bmeta = _read_meta(path)
        out_path = io.feature_path(EXTRACTOR_V2, video_id)
        if not force and os.path.exists(out_path) and check_cache_v2(_read_meta(out_path), bmeta):
            continue
        t_sec, img, valid, bmeta = io.load_features(BACKBONE, video_id)
        io.save_features(EXTRACTOR_V2, video_id, *from_backbone_v2(t_sec, img, valid, bmeta, device, dtype))
        n += 1
    return n


def prompt_set_hash(prompts=None):
    """sha1 of the (key, group, text) list; changes whenever any prompt changes."""
    prompts = PROMPTS if prompts is None else prompts
    return hashlib.sha1(json.dumps([list(p) for p in prompts]).encode()).hexdigest()[:12]


_TEXT_CACHE = {}


def text_embeddings(device="cuda:1", dtype="bf16"):
    """L2-normalized prompt embeddings [P, D] (float32, CPU), cached per process and prompt-set hash."""
    key = (device, dtype, prompt_set_hash())
    if key not in _TEXT_CACHE:
        _TEXT_CACHE[key] = siglip.embed_texts(prompt_texts(), device, dtype)
    return _TEXT_CACHE[key]


def score(image_feat, text_emb):
    """Cosine similarity [T, P] between L2-normalized image features [T, D] and text embeddings [P, D]."""
    return np.asarray(image_feat, dtype=np.float32) @ np.asarray(text_emb, dtype=np.float32).T


def _meta(backbone_meta, device, dtype):
    scale, bias = siglip.logit_params(device, dtype)
    return dict(
        extractor=EXTRACTOR,
        version=VERSION,
        backbone=BACKBONE,
        model_id=siglip.MODEL_ID,
        model_revision=backbone_meta.get("model_revision"),
        window_sec=0.0,
        D=len(PROMPTS),
        columns=prompt_keys(),
        prompt_set_version=PROMPT_SET_VERSION,
        prompt_set_hash=prompt_set_hash(),
        prompts=[{"key": k, "group": g, "text": t} for k, g, t in PROMPTS],
        backbone_version=backbone_meta.get("version"),
        score="cosine(image_emb, text_emb); prob = sigmoid(logit_scale * cos + logit_bias)",
        logit_scale=scale,
        logit_bias=bias,
        text_preprocessing="lowercase, pad to 64 tokens",
        video_fps=backbone_meta.get("video_fps"),
        duration_sec=backbone_meta.get("duration_sec"),
        n_decode_errors=backbone_meta.get("n_decode_errors"),
    )


def from_backbone(t_sec, image_feat, valid, backbone_meta, device="cuda:1", dtype="bf16"):
    """Prompt scores from already computed ``siglip2`` outputs; invalid ticks are zero."""
    feat = score(image_feat, text_embeddings(device, dtype))
    feat[~np.asarray(valid, dtype=bool)] = 0.0
    return t_sec, feat.astype(np.float16), valid, _meta(backbone_meta, device, dtype)


def extract_multi(video_path, device="cuda:1", **cfg):
    """One decode, both outputs: ``{"siglip2": (t, feat, valid, meta), "siglip2_prompts": (...)}``."""
    t0 = time.time()
    t_sec, img, valid, bmeta = siglip.extract(video_path, device, **cfg)
    out = from_backbone(t_sec, img, valid, bmeta, device, cfg.get("dtype", DEFAULTS["dtype"]))
    out[3]["wall_sec"] = round(time.time() - t0, 3)
    return {BACKBONE: (t_sec, img, valid, bmeta), EXTRACTOR: out}


def extract(video_path, device="cuda:1", **cfg):
    """Extract ``siglip2_prompts`` for one video: ``(t_sec [T], feat [T, P] float16, valid [T], meta)``."""
    return extract_multi(video_path, device, **cfg)[EXTRACTOR]


_DECODE_KEYS = ("decode_threads", "chunk_ticks")   # used by the model that decodes (V-JEPA in extract_visual)
_SHARED_KEYS = ("dtype", "batch_size")             # applied to both models


def split_visual_cfg(cfg, vjepa_cfg=None, siglip_cfg=None):
    """Route ``extract_visual`` config keys to the two models: ``(vjepa_cfg, siglip_cfg, ignored)``.

    Flat ``cfg`` keys: V-JEPA-only keys (``window_sec``, ``num_frames``, ``mid_layer``, ``crop``) and the decode
    keys (``decode_threads``, ``chunk_ticks``) go to V-JEPA, which does the only decode; ``dtype`` and
    ``batch_size`` go to both. ``vjepa_cfg`` / ``siglip_cfg`` set a key for one model and win over flat keys.
    ``ignored`` lists keys no model uses here, including SigLIP decode keys (SigLIP does not decode).
    """
    vjepa_cfg, siglip_cfg = dict(vjepa_cfg or {}), dict(siglip_cfg or {})
    vkeys, skeys = set(vjepa.DEFAULTS), set(_SHARED_KEYS)
    v = {k: x for k, x in cfg.items() if k in vkeys}
    s = {k: x for k, x in cfg.items() if k in skeys}
    ignored = {k for k in cfg if k not in vkeys and k not in skeys}
    ignored |= {f"vjepa_cfg.{k}" for k in vjepa_cfg if k not in vkeys}
    ignored |= {f"siglip_cfg.{k}" for k in siglip_cfg if k not in skeys}
    v.update({k: x for k, x in vjepa_cfg.items() if k in vkeys})
    s.update({k: x for k, x in siglip_cfg.items() if k in skeys})
    return v, s, sorted(ignored)


def extract_visual(video_path, device="cuda:1", vjepa_cfg=None, siglip_cfg=None, **cfg):
    """All three visual extractors from ONE decode: ``{"vjepa2_l": ..., "siglip2": ..., "siglip2_prompts": ...}``.

    The V-JEPA clip for tick t ends with the frame at t, so SigLIP 2 embeds those already decoded frames
    (``vjepa.extract(tick_frame_fn=...)``; the SigLIP resize runs in the decode thread). Outputs are bitwise
    identical to running ``vjepa.extract``, ``siglip.extract`` and ``extract`` separately with the same configs
    (fixed batch shapes), but the video is decoded once. See ``split_visual_cfg`` for how ``cfg``,
    ``vjepa_cfg`` and ``siglip_cfg`` reach each model.
    """
    t_start = time.time()
    vcfg, scfg, ignored = split_visual_cfg(cfg, vjepa_cfg, siglip_cfg)
    scfg = {**{k: siglip.DEFAULTS[k] for k in _SHARED_KEYS}, **scfg}
    model, _ = siglip.load_model(device, scfg["dtype"])
    sdtype = siglip._DTYPES[scfg["dtype"]]
    chunks = {}

    def sink(c0, c1, x, ok):
        emb = siglip.embed_images(model, siglip.normalize(x, device, sdtype), scfg["batch_size"])
        chunks[c0] = (emb.cpu().numpy().astype(np.float16), ok)

    t_sec, vfeat, vvalid, vmeta = vjepa.extract(video_path, device, tick_frame_fn=sink,
                                                tick_frame_prep=siglip.resize_frames, **vcfg)
    vmeta["ignored_cfg"] = ignored
    dim = model.config.vision_config.hidden_size
    sfeat = np.zeros((len(t_sec), dim), dtype=np.float16)
    svalid = np.zeros(len(t_sec), dtype=bool)
    for c0, (emb, ok) in chunks.items():
        sfeat[c0:c0 + len(emb)] = emb
        svalid[c0:c0 + len(emb)] = ok
    sfeat[~svalid] = 0
    smeta = siglip.make_meta(vmeta["video_fps"], vmeta["duration_sec"], dim, scfg,
                             int((~svalid).sum()), ignored, time.time() - t_start)
    smeta["shared_decode_with"] = vjepa.EXTRACTOR
    pout = from_backbone(t_sec, sfeat, svalid, smeta, device, scfg["dtype"])
    pout[3]["wall_sec"] = smeta["wall_sec"]
    pout[3]["ignored_cfg"] = ignored
    return {vjepa.EXTRACTOR: (t_sec, vfeat, vvalid, vmeta), BACKBONE: (t_sec, sfeat, svalid, smeta),
            EXTRACTOR: pout}


VISUAL_EXTRACTORS = (vjepa.EXTRACTOR, BACKBONE, EXTRACTOR)


def extract_and_save(video_path, video_id, device="cuda:1", vjepa_cfg=None, siglip_cfg=None, **cfg):
    """Registry ``visual``: one decode, writes vjepa2_l, siglip2 and siglip2_prompts for ``video_id``.

    Same return convention as ``pose.extract_and_save``: ``{"paths": {extractor: path}, "duration_sec": s}``.
    Config routing as in ``extract_visual`` / ``split_visual_cfg``.
    """
    outs = extract_visual(video_path, device, vjepa_cfg=vjepa_cfg, siglip_cfg=siglip_cfg, **cfg)
    paths = {name: io.save_features(name, video_id, *out) for name, out in outs.items()}
    return dict(paths=paths, duration_sec=outs[EXTRACTOR][3]["duration_sec"])


def check_cache(meta, backbone_meta=None):
    """True if a saved prompts file is current: same prompt list and ``VERSION``, and (if ``backbone_meta`` is
    given) computed from that ``siglip2`` file's version."""
    ok = meta.get("prompt_set_hash") == prompt_set_hash() and meta.get("version") == VERSION
    if backbone_meta is not None:
        ok = ok and meta.get("backbone_version") == backbone_meta.get("version")
    return ok


def _read_meta(path):
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["meta"]))


def rescore_all(device="cuda:1", dtype="bf16", force=False):
    """Recompute ``siglip2_prompts`` for every cached ``siglip2`` file (no video decoding). Returns count.

    A prompts file is rewritten when it is missing or stale (``check_cache``) or with ``force``. ``siglip2``
    files older than ``siglip.VERSION`` are rescored too (the result records ``backbone_version``) but listed
    in a warning: they need re-extraction from the video.
    """
    root = os.path.join(io.FEATURE_ROOT, BACKBONE)
    n, old_backbone = 0, []
    for path in sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True)):
        if path.endswith(".tmp.npz"):
            continue
        video_id = os.path.relpath(path, root)[:-len(".npz")]
        bmeta = _read_meta(path)
        if bmeta.get("version") != siglip.VERSION:
            old_backbone.append(video_id)
        out_path = io.feature_path(EXTRACTOR, video_id)
        if not force and os.path.exists(out_path) and check_cache(_read_meta(out_path), bmeta):
            continue
        t_sec, img, valid, bmeta = io.load_features(BACKBONE, video_id)
        io.save_features(EXTRACTOR, video_id, *from_backbone(t_sec, img, valid, bmeta, device, dtype))
        n += 1
    if old_backbone:
        print(f"warning: {len(old_backbone)} siglip2 files are older than version {siglip.VERSION} and need "
              f"re-extraction, e.g. {old_backbone[:3]}")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video", nargs="?")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--video-id", help="save every computed extractor for this <source>/<stem>")
    ap.add_argument("--all-visual", action="store_true",
                    help="also compute vjepa2_l from the same decode (prompts.extract_visual)")
    ap.add_argument("--check-causal", type=float, metavar="CUT_SEC")
    ap.add_argument("--list", action="store_true", help="print the prompt list and exit")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute prompt scores for all cached siglip2 files whose prompt hash is stale")
    ap.add_argument("--force", action="store_true", help="with --rescore: rewrite even up-to-date files")
    ap.add_argument("--v2", action="store_true", help="with --list or --rescore: use the v2 concept set (CPU)")
    args = ap.parse_args()

    if args.v2 and (args.list or args.rescore):
        if args.list:
            print(f"prompt set {PROMPT_SET_VERSION_V2} hash {prompt_set_hash_v2()} ({len(PROMPTS_V2)} concepts)")
            for k, g, t in PROMPTS_V2:
                print(f"{g:<18s} {k:<20s} {len(t)} phrasings: {t[0]}")
        else:
            if args.device.startswith("cuda") and "--device" in sys.argv:
                raise SystemExit("--v2 --rescore is CPU-only (fp32 text tower); omit --device or pass --device cpu")
            print("rescored", rescore_all_v2("cpu", force=args.force), "files")
            n_src, n_ok = coverage_v2()
            print(f"coverage siglip2_prompts_v2 {n_ok}/{n_src}", "OK" if n_src == n_ok else "INCOMPLETE")
        return
    if args.list:
        print(f"prompt set {PROMPT_SET_VERSION} hash {prompt_set_hash()} ({len(PROMPTS)} prompts)")
        for j, (k, g, t) in enumerate(PROMPTS):
            print(f"{j:3d}  {g:<13s} {k:<20s} {t}")
        return
    if args.rescore:
        print("rescored", rescore_all(args.device, force=args.force), "files")
        return
    if not args.video:
        ap.error("VIDEO is required unless --list or --rescore is given")

    siglip.load_model(args.device)
    text_embeddings(args.device)
    if args.all_visual:
        vjepa.load_model(args.device)
        outs = extract_visual(args.video, args.device)
    else:
        outs = extract_multi(args.video, args.device)
    for name, (t_sec, feat, valid, meta) in outs.items():
        print(f"{name}: T={len(t_sec)} D={feat.shape[1]} valid={int(valid.sum())}/{len(valid)}")
    meta = outs[EXTRACTOR][3]
    print(f"wall={meta['wall_sec']:.1f}s speed={meta['duration_sec'] / meta['wall_sec']:.1f} video-s/s")
    if args.video_id:
        for name, (t, f, v, m) in outs.items():
            print("saved", io.save_features(name, args.video_id, t, f, v, m))
    if args.check_causal:
        res = io.check_causal(lambda p: extract(p, args.device)[:3], args.video, args.check_causal)
        print("check_causal", json.dumps(res))


if __name__ == "__main__":
    main()
