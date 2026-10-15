"""Causal AudioSet sound-event logits, AST embeddings and CLAP text-prompt scores at 1 Hz ticks.

Extractors (all follow ``anticipation/features/io.py``):

* ``ast_events``  : AST (``MIT/ast-finetuned-audioset-10-10-0.4593``, AudioSet mAP 0.459) logits of a curated
                    list of escalation-relevant AudioSet classes (``CURATED_CLASSES``, exact AudioSet names, stored
                    in ``meta["columns"]``). Pass ``classes="all"`` to keep all 527 logits instead.
* ``ast_emb``     : the 768-d AST pooled embedding (mean of the CLS and distillation tokens) of the same window.
* ``clap_prompts``: LAION CLAP (``laion/larger_clap_general``) cosine similarity between the window audio and
                    the 20 text prompts in ``CLAP_PROMPTS`` (columns ``0..19``), followed by the 512-d normalised
                    CLAP audio embedding (columns ``clap_emb_*``).

Windows: tick t uses only the audio in ``[max(0, t - W), t]`` (default W = 3 s for AST, 5 s for CLAP). AST
gets the Kaldi fbank frames lying fully inside the window, zero-padded to its fixed 10.24 s input exactly
as ``ASTFeatureExtractor`` pads short clips. CLAP gets the window repeat-padded to 10 s, exactly as
``ClapFeatureExtractor`` (``padding="repeatpad"``) does for short clips. The fbank is computed once per video
on the GPU and sliced per window; since Kaldi fbank frames are independent, this equals per-window extraction.

Audio presence: ``valid`` is False on every tick when there is no audio stream or the track is digital
silence (UCF-Crime), and on ticks whose window is past the end of the audio, digitally silent or near-silent
(RMS < -65 dBFS); see ``audio_dsp.tick_valid``. The audio is aligned to the video clock by ``load_track``.

Batch-shape invariance: the fbank is computed in fixed-size chunks and every GPU batch of full-length windows
is padded to ``batch_size`` rows, so a tick's features do not depend on how long the video is.
``io.check_causal`` gives max |diff| 0.0 (fp16 included); ``--selftest`` checks this on synthetic audio.

Usage:

    .venv/bin/python -m anticipation.features.audio_events <video.mp4> [--device cuda:0]
        [--which ast_events ast_emb clap_prompts] [--causal-cut 60] [--top 5]
    .venv/bin/python -m anticipation.features.audio_events --selftest [--device cuda:0]   # about 30 s

From Python:

    from anticipation.features import audio_events
    t_sec, feat, valid, meta = audio_events.extract(path, device="cuda:0")                 # ast_events
    outs = audio_events.extract_all(path, device="cuda:0")   # {"ast_events": ..., "ast_emb": ..., "clap_prompts": ...}
    audio_events.extract_and_save(path, "xd/<stem>", device="cuda:0")   # writes all three under io.FEATURE_ROOT

``extract_all`` decodes once and shares the AST forward pass between ``ast_events`` and ``ast_emb``; use it
(or the saver ``extract_and_save``, the ``run_extract.py`` multi-output convention) in bulk runs. Models are
loaded once per process and per device and cached. Suggested runner registry line:
``"audio_events": ExtractorSpec("anticipation.features.audio_events", outputs=OUTPUTS, saver="extract_and_save")``.

Weights: AST from the HF hub (safetensors). CLAP ``larger_clap_general`` only ships a pickle on ``main``,
which transformers 4.57 refuses to load with torch 2.5 (needs torch >= 2.6), so we pin the safetensors
conversion PR revision ``refs/pr/2`` (commit 16c8cc3159a3c8a31e8ff5ef1f66b0d9ab3667db, HF SFconvertbot).
"""

import argparse
import json
import time

import numpy as np
import torch

from anticipation.features import io
from anticipation.features.audio_dsp import (AudioTrack, base_meta, empty_result, load_track, pad_indices, tick_valid,
                                             window_bounds)

EXTRACTOR = "ast_events"                       # what extract() returns by default
OUTPUTS = ("ast_events", "ast_emb", "clap_prompts")
VERSION = "1.1"                                # 1.1: fixed-shape fbank chunks and batches, audio_dsp 1.1 track rules
SR = 16000
CLAP_SR = 48000

AST_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_MEAN, AST_STD = -4.2677393, 4.5689974     # ASTFeatureExtractor defaults (AudioSet statistics)
AST_FRAMES = 1024                              # fixed AST input length (10.24 s of 10 ms frames)
AST_WIN, AST_HOP = 400, 160                    # Kaldi fbank: 25 ms frames, 10 ms hop at 16 kHz

CLAP_ID = "laion/larger_clap_general"
CLAP_REVISION = "refs/pr/2"                    # safetensors conversion of main (see module docstring)
CLAP_MAX_SAMPLES = 480000                      # 10 s at 48 kHz

# Exact AudioSet display names (as in the AST config id2label), grouped by role.
CURATED_CLASSES = [
    # speech
    "Speech", "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking",
    "Conversation", "Narration, monologue", "Babbling", "Whispering",
    # raised voice and distress
    "Shout", "Bellow", "Yell", "Battle cry", "Children shouting", "Screaming", "Crying, sobbing", "Whimper",
    "Wail, moan", "Groan", "Grunt", "Gasp", "Pant", "Laughter",
    # crowd
    "Crowd", "Hubbub, speech noise, speech babble", "Chatter", "Cheering", "Applause", "Chant", "Children playing",
    # bodies, impacts, breakage
    "Run", "Slap, smack", "Whack, thwack", "Thump, thud", "Smash, crash", "Breaking", "Glass", "Shatter", "Bang",
    # weapons and emergency
    "Gunshot, gunfire", "Explosion", "Siren", "Police car (siren)", "Emergency vehicle", "Alarm",
    # context and confounders
    "Vehicle", "Car", "Vehicle horn, car horn, honking", "Traffic noise, roadway noise", "Music", "Television",
    "Dog", "Wind noise (microphone)", "Silence",
]

CLAP_PROMPTS = [
    "people arguing loudly",
    "a heated argument between two men",
    "a man shouting angrily",
    "a woman screaming",
    "a crowd yelling",
    "people yelling threats and insults",
    "an angry confrontation on the street",
    "people cheering and shouting at a fight",
    "punches and slaps hitting a person",
    "people scuffling and grunting",
    "glass breaking",
    "a gunshot",
    "a police siren",
    "a car horn honking",
    "calm conversation",
    "people talking quietly",
    "people laughing",
    "music playing",
    "traffic noise on a quiet street",
    "silence",
]

_CACHE = {}


# ----------------------------------------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------------------------------------

def _ast(device):
    key = ("ast", str(device))
    if key not in _CACHE:
        from transformers import ASTForAudioClassification

        model = ASTForAudioClassification.from_pretrained(AST_ID, attn_implementation="sdpa")
        model = model.to(device).eval()
        id2label = model.config.id2label
        labels = [id2label[i] for i in range(len(id2label))]
        _CACHE[key] = (model, labels)
    return _CACHE[key]


def _clap(device):
    key = ("clap", str(device))
    if key not in _CACHE:
        from transformers import ClapFeatureExtractor, ClapModel, ClapProcessor

        model = ClapModel.from_pretrained(CLAP_ID, revision=CLAP_REVISION).to(device).eval()
        proc = ClapProcessor.from_pretrained(CLAP_ID, revision=CLAP_REVISION)
        fe: ClapFeatureExtractor = proc.feature_extractor
        with torch.no_grad():
            tok = proc.tokenizer(CLAP_PROMPTS, padding=True, return_tensors="pt").to(device)
            text = model.get_text_features(**tok)
            text = torch.nn.functional.normalize(text.float(), dim=-1)
        mel_fb = torch.tensor(fe.mel_filters_slaney, dtype=torch.float32, device=device)  # [513, 64]
        _CACHE[key] = (model, text, mel_fb, fe)
    return _CACHE[key]


# ----------------------------------------------------------------------------------------------------------
# AST
# ----------------------------------------------------------------------------------------------------------

def ast_fbank(wav, device, chunk_sec=300):
    """Kaldi log-mel fbank (128 bins, 25 ms / 10 ms, ``snip_edges``) of the whole waveform, on ``device``.

    Identical to ``ASTFeatureExtractor`` before padding/normalisation. Computed in chunks aligned to the
    10 ms frame grid; frame j covers samples [160 j, 160 j + 400), so there are ``1 + (n - 400) // 160``
    frames. Every chunk (the last one zero-padded) has exactly ``L + 240`` samples and yields ``L / 160``
    frames, so the GPU FFT and matmul always see the same shapes and a frame's value does not depend on the
    track length. (With a shorter last chunk, cuFFT changed frames by about 1e-6, which the fp16 AST pass
    amplified to 0.011 between a video and its truncated copy.) Frames are independent, so the zero padding
    only creates extra frames, which are cut off.
    """
    import torchaudio.compliance.kaldi as ta_kaldi

    L = int(chunk_sec * SR)  # multiple of 160
    seg_len = L + (AST_WIN - AST_HOP)
    x = torch.from_numpy(np.ascontiguousarray(wav)).to(device)
    n_frames = 1 + (x.numel() - AST_WIN) // AST_HOP if x.numel() >= AST_WIN else 0
    out = []
    for s in range(0, n_frames * AST_HOP, L):
        seg = x[s: s + seg_len]
        if seg.numel() < seg_len:
            seg = torch.nn.functional.pad(seg, (0, seg_len - seg.numel()))
        out.append(ta_kaldi.fbank(seg.unsqueeze(0), sample_frequency=SR, window_type="hanning", num_mel_bins=128,
                                  dither=0.0))
    return torch.cat(out)[:n_frames] if out else torch.zeros(0, 128, device=device)


def _ast_batch(fbank, starts, ends):
    """Normalised AST inputs [B, 1024, 128] for windows given in samples (frames fully inside each window)."""
    B = len(starts)
    x = torch.zeros(B, AST_FRAMES, 128, device=fbank.device, dtype=torch.float32)
    for i, (s, e) in enumerate(zip(starts, ends)):
        a = -(-int(s) // AST_HOP)
        b = min((int(e) - AST_WIN) // AST_HOP + 1, fbank.shape[0])
        n = max(0, min(b - a, AST_FRAMES))
        if n > 0:
            x[i, :n] = fbank[a:a + n]
    return (x - AST_MEAN) / (AST_STD * 2)


@torch.no_grad()
def run_ast(track, device, window_sec=3.0, batch_size=64, fp16=True):
    """AST logits [T, 527] and pooled embeddings [T, 768] for every valid tick (zeros elsewhere).

    Every batch is padded to ``batch_size`` rows (``pad_indices``) so a tick never lands in a smaller batch
    in a truncated video: in fp16 a batch of 1 and a batch of 64 differ by up to 0.011 on the same input.
    """
    model, labels = _ast(device)
    T = len(track.t_sec)
    valid = tick_valid(track, window_sec)
    logits = np.zeros((T, len(labels)), dtype=np.float32)
    emb = np.zeros((T, model.config.hidden_size), dtype=np.float32)
    if not valid.any():
        return logits, emb, valid, labels
    fbank = ast_fbank(track.wav, device)
    starts, ends = window_bounds(track.t_sec, window_sec, SR)
    idx = np.flatnonzero(valid)
    for a in range(0, len(idx), batch_size):
        sel = idx[a:a + batch_size]
        rows = pad_indices(sel, batch_size)
        x = _ast_batch(fbank, starts[rows], ends[rows])
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and x.is_cuda):
            out = model.audio_spectrogram_transformer(x)
            pooled = out.pooler_output
            lg = model.classifier(pooled)
        logits[sel] = lg.float().cpu().numpy()[: len(sel)]
        emb[sel] = pooled.float().cpu().numpy()[: len(sel)]
    return logits, emb, valid, labels


# ----------------------------------------------------------------------------------------------------------
# CLAP
# ----------------------------------------------------------------------------------------------------------

def clap_mel(x, mel_fb):
    """Log-mel input of ``ClapFeatureExtractor`` (rand_trunc path, slaney mel, dB) for repeat-padded windows.

    ``x`` is [B, n] at 48 kHz with all rows the same length n <= 480000. Returns [B, 1, 1001, 64].
    """
    B, n = x.shape
    if n < CLAP_MAX_SAMPLES:
        x = x.repeat(1, CLAP_MAX_SAMPLES // n)
        x = torch.nn.functional.pad(x, (0, CLAP_MAX_SAMPLES - x.shape[1]))
    else:
        x = x[:, :CLAP_MAX_SAMPLES]
    win = torch.hann_window(1024, periodic=True, device=x.device, dtype=torch.float32)
    spec = torch.stft(x, n_fft=1024, hop_length=480, win_length=1024, window=win, center=True, pad_mode="reflect",
                      return_complex=True)
    power = spec.real ** 2 + spec.imag ** 2                       # [B, 513, F]
    mel = torch.einsum("bft,fm->btm", power, mel_fb)              # [B, F, 64]
    return (10.0 * torch.log10(torch.clamp(mel, min=1e-10))).unsqueeze(1)


@torch.no_grad()
def run_clap(track48, device, window_sec=5.0, batch_size=32, fp16=True):
    """CLAP prompt similarities [T, K] and audio embeddings [T, 512] for valid ticks of a 48 kHz track.

    Windows are batched by length (early ticks have shorter windows; each of those is its own group of one in
    every run). Batches of full-length windows are padded to ``batch_size`` rows so their shape does not
    depend on the video length (see ``run_ast``).
    """
    model, text, mel_fb, _ = _clap(device)
    T = len(track48.t_sec)
    valid = tick_valid(track48, window_sec)
    sims = np.zeros((T, text.shape[0]), dtype=np.float32)
    emb = np.zeros((T, text.shape[1]), dtype=np.float32)
    if not valid.any():
        return sims, emb, valid
    starts, ends = window_bounds(track48.t_sec, window_sec, CLAP_SR)
    wav = torch.from_numpy(np.ascontiguousarray(track48.wav)).to(device)
    idx = np.flatnonzero(valid)
    lengths = ends[idx] - starts[idx]
    full = int(np.floor(window_sec * CLAP_SR))
    for n in np.unique(lengths):                                   # early ticks have shorter windows
        group = idx[lengths == n]
        for a in range(0, len(group), batch_size):
            sel = group[a:a + batch_size]
            rows = pad_indices(sel, batch_size) if n >= full else sel
            x = torch.stack([wav[s:e] for s, e in zip(starts[rows], ends[rows])])
            feats = clap_mel(x, mel_fb)
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and x.is_cuda):
                is_longer = torch.zeros(len(rows), 1, dtype=torch.bool, device=x.device)
                e_ = model.get_audio_features(input_features=feats, is_longer=is_longer)
            e_ = torch.nn.functional.normalize(e_.float(), dim=-1)
            sims[sel] = (e_ @ text.T).cpu().numpy()[: len(sel)]       # slice after the matmul (fixed shape)
            emb[sel] = e_.cpu().numpy()[: len(sel)]
    return sims, emb, valid


# ----------------------------------------------------------------------------------------------------------
# Extractors
# ----------------------------------------------------------------------------------------------------------

def extract_all(video_path, device="cuda:0", which=OUTPUTS, ast_window_sec=3.0, clap_window_sec=5.0,
                classes="curated", batch_size=64, fp16=True, **unused):
    """Run the requested extractors on one video; returns ``{name: (t_sec, feat, valid, meta)}``.

    Unknown keyword arguments are ignored and listed in ``meta["ignored_cfg"]`` (so a runner can pass
    generic config). Changing a window or ``classes`` changes the features: save under a new name then.
    """
    t0 = time.time()
    which = (which,) if isinstance(which, str) else tuple(which)
    bad = [w for w in which if w not in OUTPUTS]
    if bad:
        raise ValueError(f"unknown audio_events output(s) {bad}; choose from {OUTPUTS}")
    out = {}
    duration = None
    if "ast_events" in which or "ast_emb" in which:
        track = load_track(video_path, sr=SR)
        duration = track.duration_sec
        _, labels = _ast(device)
        cols = labels if classes == "all" else CURATED_CLASSES
        col_idx = [labels.index(c) for c in cols]
        common = dict(model=AST_ID, window_pad="zero-pad fbank to 1024 frames (ASTFeatureExtractor)")
        m_ev = base_meta("ast_events", VERSION, track, ast_window_sec, cols, output="logits", **common)
        m_em = base_meta("ast_emb", VERSION, track, ast_window_sec, [f"ast_emb_{i}" for i in range(768)],
                         output="pooled embedding (mean of CLS and distillation tokens)", **common)
        if track.wav is None:
            res_ev, res_em = empty_result(track, len(cols), m_ev), empty_result(track, 768, m_em)
        else:
            logits, emb, valid, _ = run_ast(track, device, ast_window_sec, batch_size, fp16)
            vf = float(valid.mean())
            res_ev = (track.t_sec, logits[:, col_idx].astype(np.float16), valid, dict(m_ev, valid_frac=vf))
            res_em = (track.t_sec, emb.astype(np.float16), valid, dict(m_em, valid_frac=vf))
        if "ast_events" in which:
            out["ast_events"] = res_ev
        if "ast_emb" in which:
            out["ast_emb"] = res_em
    if "clap_prompts" in which:
        track48 = load_track(video_path, sr=CLAP_SR, duration_sec=duration)
        cols = list(CLAP_PROMPTS) + [f"clap_emb_{i}" for i in range(512)]
        meta = base_meta("clap_prompts", VERSION, track48, clap_window_sec, cols, model=CLAP_ID,
                         revision=CLAP_REVISION, n_prompts=len(CLAP_PROMPTS),
                         output="cosine(audio, prompt) for each prompt, then normalised 512-d audio embedding",
                         window_pad="repeatpad to 10 s (ClapFeatureExtractor)")
        if track48.wav is None:
            out["clap_prompts"] = empty_result(track48, len(cols), meta)
        else:
            sims, emb, valid = run_clap(track48, device, clap_window_sec, max(1, batch_size // 2), fp16)
            feat = np.concatenate([sims, emb], axis=1).astype(np.float16)
            out["clap_prompts"] = (track48.t_sec, feat, valid, dict(meta, valid_frac=float(valid.mean())))
    elapsed = round(time.time() - t0, 3)
    for name in out:
        out[name][3]["elapsed_sec_all"] = elapsed
        if unused:
            out[name][3]["ignored_cfg"] = sorted(unused)
    return out


def extract(video_path, device="cuda:0", which=EXTRACTOR, **cfg):
    """Single-extractor entry point: ``which`` in {"ast_events", "ast_emb", "clap_prompts"}."""
    return extract_all(video_path, device, which=(which,), **cfg)[which]


def extract_and_save(video_path, video_id, device="cuda:0", which=OUTPUTS, **cfg):
    """Runner ``saver``: extract the requested outputs from one decode and write them with ``io.save_features``.

    Returns ``{"paths": {name: path}, "duration_sec": s}`` (the ``run_extract.py`` saver convention).
    """
    outs = extract_all(video_path, device, which=which, **cfg)
    paths = {name: io.save_features(name, video_id, *res) for name, res in outs.items()}
    return {"paths": paths, "duration_sec": next(iter(outs.values()))[3]["duration_sec"]}


def extract_events(video_path, device="cuda:0", **cfg):
    return extract(video_path, device, which="ast_events", **cfg)


def extract_emb(video_path, device="cuda:0", **cfg):
    return extract(video_path, device, which="ast_emb", **cfg)


def extract_clap(video_path, device="cuda:0", **cfg):
    return extract(video_path, device, which="clap_prompts", **cfg)


EXTRACTORS = {"ast_events": extract_events, "ast_emb": extract_emb, "clap_prompts": extract_clap}


# ----------------------------------------------------------------------------------------------------------
# Self-test (synthetic audio, GPU)
# ----------------------------------------------------------------------------------------------------------

def synthetic_wav(sr, seconds, seed=0):
    """Deterministic test signal: amplitude-modulated noise plus a wandering tone (float32, ``seconds * sr``)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    env = 0.02 + 0.1 * (1.0 + np.sin(2 * np.pi * t / 37.0))
    tone = 0.05 * np.sin(2 * np.pi * (300.0 * t + 200.0 * 11.0 / (2 * np.pi) * (1 - np.cos(2 * np.pi * t / 11.0))))
    return (env * rng.standard_normal(t.size) + tone).astype(np.float32)


def synthetic_track(wav, sr, seconds):
    """An ``AudioTrack`` holding the first ``seconds`` of ``wav`` (stands in for a video truncated there)."""
    seconds = int(seconds)
    return AudioTrack("synthetic", sr, io.tick_times(seconds), float(seconds), wav[: seconds * sr].copy(),
                      seconds * sr, "ok")


def selftest(device="cuda:0", verbose=True):
    """Truncation invariance of the AST fbank, AST outputs and CLAP outputs (fp16); raises on failure.

    The fbank of a prefix must equal the prefix of the fbank bit for bit (cuts inside the first 300 s chunk
    and just past it). AST on a 200 s track and on its first 66 s (ticks 65..66 in a 2-row last batch) must
    agree exactly on ticks 1..65; CLAP likewise on its first 71 s (ticks 69..71 in a 3-row batch), ticks 1..70.
    """
    res = {}
    x = synthetic_wav(SR, 320)
    fb = ast_fbank(x, device)
    for cut in (61.5, 70.2, 150.0, 305.25):                       # the old code differed at 70.2 and 150
        fc = ast_fbank(x[: int(cut * SR)], device)
        res[f"fbank_prefix_equal_cut{cut}"] = bool(torch.equal(fb[: len(fc)], fc))
    full, cut = synthetic_track(x, SR, 200), synthetic_track(x, SR, 66)
    lf, ef, _, _ = run_ast(full, device)
    lc, ec, _, _ = run_ast(cut, device)
    res["ast_logits_max_abs_diff"] = float(np.abs(lf[:65] - lc[:65]).max())
    res["ast_emb_max_abs_diff"] = float(np.abs(ef[:65] - ec[:65]).max())
    x48 = synthetic_wav(CLAP_SR, 200)
    sf, mf, _ = run_clap(synthetic_track(x48, CLAP_SR, 200), device)
    sc, mc, _ = run_clap(synthetic_track(x48, CLAP_SR, 71), device)
    res["clap_max_abs_diff"] = float(max(np.abs(sf[:70] - sc[:70]).max(), np.abs(mf[:70] - mc[:70]).max()))
    if verbose:
        print("audio_events selftest", json.dumps(res))
    assert all(v for k, v in res.items() if k.startswith("fbank")), res
    assert max(v for k, v in res.items() if k.endswith("diff")) == 0.0, res
    return res


def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video", nargs="?")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--which", nargs="+", default=list(OUTPUTS))
    ap.add_argument("--causal-cut", type=float, default=0.0, help="also run io.check_causal at this cut (s)")
    ap.add_argument("--top", type=int, default=5, help="print the top classes/prompts every 10 ticks")
    ap.add_argument("--selftest", action="store_true", help="truncation-invariance checks on synthetic audio")
    args = ap.parse_args()
    if args.selftest:
        selftest(args.device)
        if not args.video:
            return
    if not args.video:
        ap.error("video is required unless --selftest is given")
    outs = extract_all(args.video, args.device, which=args.which)
    for name, (t, f, v, m) in outs.items():
        print(name, f.shape, "valid", int(v.sum()), "/", len(v), json.dumps({"status": m["audio_status"],
                                                                         "elapsed": m["elapsed_sec_all"]}))
        if args.top and name in ("ast_events", "clap_prompts") and v.any():
            cols = m["columns"][: (m.get("n_prompts") or len(m["columns"]))]
            for i in range(0, len(t), 10):
                if v[i]:
                    row = f[i, : len(cols)].astype(np.float32)
                    top = np.argsort(-row)[: args.top]
                    print("  t=%4d " % t[i] + ", ".join("%s %.2f" % (cols[j], row[j]) for j in top))
        if args.causal_cut > 0:
            print(" ", name, io.check_causal(lambda p, n=name: extract(p, args.device, which=n)[:3], args.video,
                                             args.causal_cut))


if __name__ == "__main__":
    _main()
