"""Causal voice-affect features at 1 Hz ticks: VAD-gated arousal / dominance / valence and emotion posteriors.

Extractors (both follow ``anticipation/features/io.py``):

* ``voice_affect`` (28 columns, see ``COLUMNS``):
    - ``vad_mean``, ``vad_max``, ``speech_frac`` (fraction of 32 ms Silero-VAD chunks with p > 0.5),
      ``speech_gate`` (1 if ``speech_frac >= 0.1``);
    - ``arousal``, ``dominance``, ``valence``: audEERING ``wav2vec2-large-robust-12-ft-emotion-msp-dim``
      (MSP-Podcast dimensional SER, outputs roughly in [0, 1]) on the uniformly mean-pooled window;
    - ``arousal_sp``, ``dominance_sp``, ``valence_sp``: the same regression head applied to the
      speech-probability-weighted mean of the hidden states (weights ``p_vad + 0.01``, so a window without
      speech falls back smoothly to the uniform mean);
    - ``e2v_<class>`` and ``e2v_sp_<class>``: emotion2vec+ large (``emotion2vec/emotion2vec_plus_large``, via
      FunASR) 9-class softmax posteriors (angry, disgusted, fearful, happy, neutral, other, sad, surprised,
      unknown), uniform and speech-weighted pooling.
* ``e2v_emb`` (1024 columns): the uniformly mean-pooled emotion2vec+ frame embedding of the window
  (the model's "utterance" embedding).

Window: tick t uses only the audio in ``[max(0, t - W), t]`` (default W = 4 s). Every model, including the
VAD (whose recurrent state is reset at the start of each window), sees only that window, so every column is a
pure function of the window audio. Each window is normalised on its own (zero mean, unit variance) as the
models' feature extractors do per utterance.

Audio presence: ``valid`` is False on every tick when the file has no audio stream or the track is digital
silence (UCF-Crime), and on ticks whose window is past the end of the audio, digitally silent or near-silent
(RMS < -65 dBFS, ``audio_dsp.tick_valid``). The near-silence rule matters here: the per-window normalisation
would otherwise scale encoder noise up to unit variance and the SER heads would emit confident-looking
arousal on silence. All SER models were trained on clean single-speaker speech; on street audio, music and
crowd noise they are noisy, which is why the VAD columns are stored next to them (let the downstream head
decide how far to trust them).

Batch-shape invariance: batches of full-length windows are padded to ``batch_size`` rows, so a tick's
features do not depend on how long the video is (``io.check_causal`` max |diff| 0.0; ``--selftest``).

Usage:

    .venv/bin/python -m anticipation.features.voice <video.mp4> [--device cuda:0] [--window 4] [--causal-cut 60]
    .venv/bin/python -m anticipation.features.voice --selftest [--device cuda:0]     # truncation invariance

From Python:

    from anticipation.features import voice
    t_sec, feat, valid, meta = voice.extract(path, device="cuda:0")          # voice_affect
    outs = voice.extract_all(path, device="cuda:0")                         # {"voice_affect": ..., "e2v_emb": ...}
    voice.extract_and_save(path, "xd/<stem>", device="cuda:0")              # writes both under io.FEATURE_ROOT

Suggested runner registry line (``anticipation/features/run_extract.py``):
``"voice": ExtractorSpec("anticipation.features.voice", outputs=OUTPUTS, saver="extract_and_save")``.

Licences: the audEERING model is CC-BY-NC-SA-4.0 (research only); emotion2vec+ uses the FunASR model licence;
Silero VAD is MIT. Fine for the capstone, not for a commercial product.

Gotchas: ``import silero_vad`` calls ``torch.set_num_threads(1)`` (restored here); FunASR logs every loaded
parameter at INFO (silenced during loading); emotion2vec+'s ``model.pt`` is a pickle loaded by FunASR.
"""

import argparse
import contextlib
import json
import logging
import time
from io import StringIO

import numpy as np
import torch
import torch.nn as nn

from anticipation.features import io
from anticipation.features.audio_dsp import base_meta, empty_result, load_track, pad_indices, tick_valid, window_bounds

EXTRACTOR = "voice_affect"                 # what extract() returns by default
OUTPUTS = ("voice_affect", "e2v_emb")
VERSION = "1.1"                       # 1.1: fixed batch shapes, audio_dsp 1.1 track rules (A/V alignment, near-silence)
SR = 16000
VAD_CHUNK = 512                       # Silero chunk at 16 kHz (32 ms)
FRAME_HOP, FRAME_RF = 320, 400        # wav2vec2 / emotion2vec conv encoder: 20 ms hop, 25 ms receptive field
SPEECH_EPS = 0.01                     # weight floor in speech-weighted pooling
SPEECH_GATE_FRAC = 0.1

W2V_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
E2V_ID = "emotion2vec/emotion2vec_plus_large"
E2V_CLASSES = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]

COLUMNS = (["vad_mean", "vad_max", "speech_frac", "speech_gate",
            "arousal", "dominance", "valence", "arousal_sp", "dominance_sp", "valence_sp"]
           + [f"e2v_{c}" for c in E2V_CLASSES] + [f"e2v_sp_{c}" for c in E2V_CLASSES])

_CACHE = {}


# ----------------------------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------------------------

def _w2v_classes():
    """The custom regression head from the audEERING model card (not a stock transformers class)."""
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class RegressionHead(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, x):
            x = self.dropout(x)
            x = torch.tanh(self.dense(x))
            x = self.dropout(x)
            return self.out_proj(x)

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.post_init()

    return EmotionModel


def _w2v(device):
    key = ("w2v", str(device))
    if key not in _CACHE:
        EmotionModel = _w2v_classes()
        model, info = EmotionModel.from_pretrained(W2V_ID, output_loading_info=True)
        if info["missing_keys"]:
            raise RuntimeError(f"{W2V_ID}: missing keys {info['missing_keys'][:5]}")
        order = [model.config.id2label[i] for i in range(model.config.num_labels)]
        _CACHE[key] = (model.to(device).eval(), order)
    return _CACHE[key]


def _e2v(device):
    key = ("e2v", str(device))
    if key not in _CACHE:
        n_threads = torch.get_num_threads()
        root = logging.getLogger()
        level = root.level
        try:
            with contextlib.redirect_stdout(StringIO()):
                from funasr import AutoModel  # importing funasr configures root logging at INFO

                root.setLevel(logging.WARNING)  # so silence it after the import, while loading
                am = AutoModel(model=E2V_ID, hub="hf", device=str(device), disable_update=True, log_level="ERROR")
        finally:
            root.setLevel(level)
            torch.set_num_threads(n_threads)
        tokens = am.kwargs["tokenizer"].token_list
        if [t.split("/")[-1].strip("<>").replace("unk", "unknown") for t in tokens] != E2V_CLASSES:
            raise RuntimeError(f"unexpected emotion2vec label order: {tokens}")
        _CACHE[key] = am.model.to(device).eval()
    return _CACHE[key]


def _vad(device):
    key = ("vad", str(device))
    if key not in _CACHE:
        n_threads = torch.get_num_threads()
        import silero_vad

        torch.set_num_threads(n_threads)  # silero_vad sets it to 1 at import
        _CACHE[key] = silero_vad.load_silero_vad(onnx=False).to(device)
    return _CACHE[key]


# ----------------------------------------------------------------------------------------------------------
# Batched window processing
# ----------------------------------------------------------------------------------------------------------

@torch.no_grad()
def vad_probs(x, device):
    """Silero speech probability per 32 ms chunk, state reset at the start of each window. x: [B, n] -> [B, n//512]."""
    vad = _vad(device)
    n_chunks = x.shape[1] // VAD_CHUNK
    vad.reset_states()
    out = [vad(x[:, i * VAD_CHUNK:(i + 1) * VAD_CHUNK].float(), SR) for i in range(n_chunks)]
    return torch.cat(out, dim=1)


def frame_weights(probs, n_frames):
    """Map chunk probabilities [B, C] to the 20 ms SER frame grid [B, n_frames] (chunk containing frame centre)."""
    centres = torch.arange(n_frames, device=probs.device) * FRAME_HOP + FRAME_RF // 2
    idx = torch.clamp(centres // VAD_CHUNK, max=probs.shape[1] - 1)
    return probs[:, idx]


def _weighted_mean(h, w):
    w = w.unsqueeze(-1)
    return (h * w).sum(1) / w.sum(1)


@torch.no_grad()
def run_windows(x, device, fp16=True):
    """All voice outputs for a batch of equal-length windows x [B, n] (raw 16 kHz audio on ``device``)."""
    w2v, w2v_order = _w2v(device)
    e2v = _e2v(device)
    probs = vad_probs(x, device)                                                   # [B, C]

    xn_w2v = (x - x.mean(1, keepdim=True)) / torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-7)
    xn_e2v = torch.nn.functional.layer_norm(x, x.shape[-1:])
    with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and x.is_cuda):
        hw = w2v.wav2vec2(xn_w2v).last_hidden_state.float()                        # [B, F, 1024]
        he = e2v.extract_features(xn_e2v, padding_mask=None)["x"].float()          # [B, F', 1024]
    w_w = frame_weights(probs, hw.shape[1]) + SPEECH_EPS
    w_e = frame_weights(probs, he.shape[1]) + SPEECH_EPS
    with torch.autocast("cuda", enabled=False):
        avd_u = w2v.classifier(hw.mean(1))
        avd_s = w2v.classifier(_weighted_mean(hw, w_w))
        pe_u_emb = he.mean(1)
        p_u = torch.softmax(e2v.proj(pe_u_emb), -1)
        p_s = torch.softmax(e2v.proj(_weighted_mean(he, w_e)), -1)
    # reorder the regression outputs to (arousal, dominance, valence)
    order = [w2v_order.index(k) for k in ("arousal", "dominance", "valence")]
    speech_frac = (probs > 0.5).float().mean(1, keepdim=True)
    feat = torch.cat([probs.mean(1, keepdim=True), probs.max(1, keepdim=True).values, speech_frac,
                      (speech_frac >= SPEECH_GATE_FRAC).float(), avd_u[:, order], avd_s[:, order], p_u, p_s], dim=1)
    return feat.cpu().numpy(), pe_u_emb.cpu().numpy()


def run_voice(track, device, window_sec=4.0, batch_size=64, fp16=True):
    """voice_affect features [T, 28] and e2v embeddings [T, 1024] for all valid ticks (zeros elsewhere).

    Windows are batched by length (each early tick with a shorter window is its own group of one in every run);
    batches of full-length windows are padded to ``batch_size`` rows (``audio_dsp.pad_indices``).
    """
    T = len(track.t_sec)
    valid = tick_valid(track, window_sec)
    feat = np.zeros((T, len(COLUMNS)), dtype=np.float32)
    emb = np.zeros((T, 1024), dtype=np.float32)
    if not valid.any():
        return feat, emb, valid
    starts, ends = window_bounds(track.t_sec, window_sec, SR)
    wav = torch.from_numpy(np.ascontiguousarray(track.wav)).to(device)
    idx = np.flatnonzero(valid)
    lengths = ends[idx] - starts[idx]
    full = int(np.floor(window_sec * SR))
    for n in np.unique(lengths):                                       # early ticks have shorter windows
        group = idx[lengths == n]
        for a in range(0, len(group), batch_size):
            sel = group[a:a + batch_size]
            rows = pad_indices(sel, batch_size) if n >= full else sel
            x = torch.stack([wav[s:e] for s, e in zip(starts[rows], ends[rows])])
            f, e_ = run_windows(x, device, fp16)
            feat[sel] = f[: len(sel)]
            emb[sel] = e_[: len(sel)]
    return feat, emb, valid


# ----------------------------------------------------------------------------------------------------------
# Extractors
# ----------------------------------------------------------------------------------------------------------

def extract_all(video_path, device="cuda:0", which=OUTPUTS, window_sec=4.0, batch_size=64, fp16=True, track=None,
                **unused):
    """Run the requested extractors on one video; returns ``{name: (t_sec, feat, valid, meta)}``.

    Unknown keyword arguments are ignored and listed in ``meta["ignored_cfg"]``. ``track`` lets a caller reuse
    an already decoded 16 kHz ``audio_dsp.AudioTrack``. Changing ``window_sec`` changes the features.
    """
    t0 = time.time()
    which = (which,) if isinstance(which, str) else tuple(which)
    bad = [w for w in which if w not in OUTPUTS]
    if bad:
        raise ValueError(f"unknown voice output(s) {bad}; choose from {OUTPUTS}")
    if track is None:
        track = load_track(video_path, sr=SR)
    common = dict(models={"vad": "silero-vad 6 (pip silero-vad, JIT)", "avd": W2V_ID, "emotion": E2V_ID},
                  speech_weight="p_vad + %.2f" % SPEECH_EPS, speech_gate_frac=SPEECH_GATE_FRAC)
    m_aff = base_meta("voice_affect", VERSION, track, window_sec, COLUMNS, **common)
    m_emb = base_meta("e2v_emb", VERSION, track, window_sec, [f"e2v_emb_{i}" for i in range(1024)],
                      output="uniform mean of emotion2vec+ large frame features", **common)
    if track.wav is None:
        res = {"voice_affect": empty_result(track, len(COLUMNS), m_aff), "e2v_emb": empty_result(track, 1024, m_emb)}
    else:
        feat, emb, valid = run_voice(track, device, window_sec, batch_size, fp16)
        vf = float(valid.mean())
        res = {"voice_affect": (track.t_sec, feat.astype(np.float16), valid, dict(m_aff, valid_frac=vf)),
               "e2v_emb": (track.t_sec, emb.astype(np.float16), valid, dict(m_emb, valid_frac=vf))}
    elapsed = round(time.time() - t0, 3)
    out = {k: res[k] for k in which}
    for name in out:
        out[name][3]["elapsed_sec_all"] = elapsed
        if unused:
            out[name][3]["ignored_cfg"] = sorted(unused)
    return out


def extract(video_path, device="cuda:0", which=EXTRACTOR, **cfg):
    """Single-extractor entry point: ``which`` in {"voice_affect", "e2v_emb"}."""
    return extract_all(video_path, device, which=(which,), **cfg)[which]


def extract_and_save(video_path, video_id, device="cuda:0", which=OUTPUTS, **cfg):
    """Runner ``saver``: extract the requested outputs in one pass and write them with ``io.save_features``.

    Returns ``{"paths": {name: path}, "duration_sec": s}`` (the ``run_extract.py`` saver convention).
    """
    outs = extract_all(video_path, device, which=which, **cfg)
    paths = {name: io.save_features(name, video_id, *res) for name, res in outs.items()}
    return {"paths": paths, "duration_sec": next(iter(outs.values()))[3]["duration_sec"]}


def extract_e2v_emb(video_path, device="cuda:0", **cfg):
    return extract(video_path, device, which="e2v_emb", **cfg)


EXTRACTORS = {"voice_affect": extract, "e2v_emb": extract_e2v_emb}


def selftest(device="cuda:0", verbose=True):
    """Truncation invariance of ``run_voice`` (fp16): a 200 s synthetic track and its first 71 s must agree
    exactly on ticks 1..70 (ticks 68..70 sit in a 3-row last batch of the truncated run). Raises otherwise."""
    from anticipation.features.audio_events import synthetic_track, synthetic_wav

    x = synthetic_wav(SR, 200, seed=1)
    ff, ef, _ = run_voice(synthetic_track(x, SR, 200), device)
    fc, ec, _ = run_voice(synthetic_track(x, SR, 71), device)
    res = {"voice_affect_max_abs_diff": float(np.abs(ff[:70] - fc[:70]).max()),
           "e2v_emb_max_abs_diff": float(np.abs(ef[:70] - ec[:70]).max())}
    if verbose:
        print("voice selftest", json.dumps(res))
    assert max(res.values()) == 0.0, res
    return res


def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video", nargs="?")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--window", type=float, default=4.0)
    ap.add_argument("--causal-cut", type=float, default=0.0, help="also run io.check_causal at this cut (s)")
    ap.add_argument("--selftest", action="store_true", help="truncation-invariance check on synthetic audio")
    args = ap.parse_args()
    if args.selftest:
        selftest(args.device)
        if not args.video:
            return
    if not args.video:
        ap.error("video is required unless --selftest is given")
    t, f, v, m = extract(args.video, args.device, window_sec=args.window)
    print(json.dumps({k: m[k] for k in ("audio_status", "n_ticks", "valid_frac", "elapsed_sec_all")}))
    cols = ["speech_frac", "arousal", "arousal_sp", "dominance_sp", "valence_sp", "e2v_sp_angry", "e2v_sp_neutral"]
    ci = [COLUMNS.index(c) for c in cols]
    print("   t " + " ".join("%12s" % c for c in cols))
    for i in range(0, len(t), 5):
        vals = " ".join("%12.3f" % x for x in f[i, ci].astype(np.float32))
        print("%4d " % t[i] + vals + ("" if v[i] else "  invalid"))
    if args.causal_cut > 0:
        print(io.check_causal(lambda p: extract(p, args.device, window_sec=args.window)[:3], args.video,
                              args.causal_cut))


if __name__ == "__main__":
    _main()
