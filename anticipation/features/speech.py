"""Speech-content features (verbal escalation): causal sliding-window Whisper ASR plus toxicity scores.

Extractor name: ``speech_tox`` (full documentation: ``docs/FEATURES.md``, section "Speech").

Pipeline per video (strictly causal: the row at tick t only uses audio up to t; the ASR text, toxicity,
rolling and validity columns only use audio in [t - 42 s, t]; the VAD recurrent state runs from the start of
the track, declared in ``meta["history"]``):

1. Decode the first audio track to 16 kHz mono (``io.load_audio``). No track or digital silence -> every
   tick ``valid=False`` and all-zero features. Otherwise tick t is valid when the 12 s before it, (t-12, t],
   hold audible audio (peak >= 1e-4 and RMS >= -65 dBFS, the audio track's rule), so validity is causal.
2. Silero VAD v6 (ONNX, one CPU thread, exact streaming over the whole track) gives a speech probability per
   32 ms frame.
3. ASR windows end every ``stride_sec`` seconds (default 2) and span the previous ``window_sec`` seconds
   (default 12, at most 30; shorter at the video start). Gate: a window is transcribed if it has at least
   ``min_speech_sec`` of VAD speech (prob > 0.5) ("vad" gate), or if its maximum VAD probability reaches
   ``weak_gate_prob`` ("weak" gate, stricter acceptance). Silero badly under-detects shouting in noisy
   footage, which is exactly the speech that matters here; the weak gate recovers it while still skipping
   music-only and silent windows, where Whisper hallucinates.
4. Whisper large-v3 (bf16; encoder in fixed micro-batches; decoder replayed as a CUDA graph with a static KV
   cache; greedy; timestamp-mode decoding with OpenAI's timestamp rules; no conditioning on previous text):
   - language id and ``no_speech_prob`` from the logits at <|startoftranscript|>; the decoding language is
     the current window's argmax, and a causal prior over earlier accepted windows only breaks near-ties
     (current argmax prob < 0.6, prior language >= 0.5 of the prior and >= 0.1 in this window);
   - hallucination filter: sound captions ("(upbeat music)", "[MUSIC PLAYING]", "*sighs*", "Музыка") and
     sung lyrics between music symbols are removed; repeated n-grams are collapsed to at most 2 copies
     (shouted repetitions are kept, the count goes to the ``repeats`` column) and only then the compression
     ratio > 2.4 rule applies; OpenAI's no_speech AND logprob rule; an avg_logprob floor; subtitle-credit /
     outro sentences in many languages; stock phrases ("Thank you.", "Merci.", "谢谢大家", "Bye-bye.") on
     windows with little speech; weak-gate windows need stricter scores and at least 3 words (or a max VAD
     prob >= 0.9).
5. Non-English transcripts are translated to English by a text MT model (Qwen2.5-7B-Instruct as a plain
   few-shot completion, greedy, fixed shapes), grounded in the native transcript; the translation is checked
   (refusal, untranslated script, length blow-up, repetition, trivial output) and dropped if it fails.
6. Text features per window: detoxify ``unbiased`` (RoBERTa) on the English text (native or translated),
   detoxify ``multilingual`` (XLM-R) on the native text only for its 7 languages (en/fr/es/it/pt/tr/ru;
   ``mtox_lang_ok`` = 0 and zeros otherwise), regex keyword counts (threat, command, challenge, profanity,
   de-escalation; heuristic, low precision), second-person pronoun rate, '!' and '?' rates.
7. Tick t takes the features of the latest window ending at or before t (lag 0 to stride-1 s), VAD
   fractions over (t-1, t], (t-12, t] and (t-30, t], and 30 s rolling maxima over windows ending in (t-30, t].

Every GPU batch has a fixed size (padded), so a window's result never depends on later windows and a
truncated copy of the video reproduces the same features and validity (``io.check_causal`` plus a ``valid``
comparison).

Raw transcripts (window start/end, gate, VAD stats, language, avg logprob, no_speech_prob, compression ratio,
rejection reason, text, English text, timestamped segments, toxicity, keyword counts) are written as JSON next
to the feature file (``<FEATURE_ROOT>/speech_tox/<source>/<stem>.transcripts.json``) for later LLM use, or
embedded in ``meta["transcripts"]`` when no ``transcript_path`` is given. A consumer at tick t must only use
windows with ``end <= t`` (``speech.windows_until``).

Usage:
    # shared runner (registered as "speech_tox"; passes transcript_path per video); run every source: about a
    # quarter of UCF-Crime has real audio with speech
    .venv/bin/python -m anticipation.features.run_extract --extractor speech_tox \
        --manifest benchmark/videos.csv --sources xd,yt,ucf --device cuda:1

    # Python API
    from anticipation.features import speech
    t_sec, feat, valid, meta = speech.extract(video_path, "cuda:1",
                                              transcript_path=speech.transcript_path("xd/<stem>"))

    # standalone (small batches; resumable, skips existing outputs)
    .venv/bin/python -m anticipation.features.speech --device cuda:1 --source xd \
        --videos data/raw_videos/xd_violence/videos/A.mp4 data/raw_videos/xd_violence/videos/B.mp4

    # model-free self-test of the text filters, repeat collapsing, keyword patterns and translation checks
    .venv/bin/python -m anticipation.features.speech --selftest
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import zlib

import numpy as np
import torch  # import before decord (io.Video): importing decord first and then initialising CUDA segfaults

from anticipation.features import io

EXTRACTOR = "speech_tox"
VERSION = 2
OUTPUT_VERSIONS = {EXTRACTOR: VERSION}  # run_extract recomputes existing files with another meta["version"]
WHISPER_ID = "openai/whisper-large-v3"
MT_ID = "Qwen/Qwen2.5-7B-Instruct"
SR = 16000
VAD_FRAME = 512  # samples per Silero frame at 16 kHz (32 ms)
N_SAMPLES = 30 * SR  # Whisper input is always 30 s (zero padded)
SILENCE_PEAK = 1e-4  # same digital-silence threshold as io.is_digital_silence

DEFAULT_CFG = dict(
    window_sec=12,          # ASR window length (s, at most 30); windows end at multiples of stride_sec
    stride_sec=2,           # ASR window stride (s)
    roll_sec=30,            # horizon of the rolling-max columns and of the language prior (s)
    batch_size=16,          # Whisper decoder batch (windows); fixed CUDA-graph shape, padded (16 is as fast as 64)
    enc_batch_size=16,      # Whisper encoder micro-batch (fixed shape, padded)
    vad_threshold=0.5,      # Silero speech probability threshold
    min_speech_sec=0.5,     # VAD speech (prob > vad_threshold) in a window for the normal gate
    weak_gate_prob=0.15,    # else: run Whisper if the window's max VAD prob reaches this (shouting, noisy speech)
    weak_max_no_speech=0.4,       # weak-gate windows: stricter acceptance (no_speech_prob at most this ...
    weak_min_logprob=-0.8,        # ... avg_logprob at least this ...
    weak_min_words=3,             # ... and at least this many words; CJK characters count 1/2 word) ...
    weak_short_vad_prob=0.9,      # ... unless the window's max VAD prob reaches this (short real shouts)
    max_new_tokens=176,     # decode budget per window (12 s of fast speech is about 70 tokens + timestamps)
    timestamps=True,        # timestamp-mode decoding (OpenAI rules); no-timestamp mode often truncates
    no_speech_threshold=0.6,      # OpenAI rule: reject if no_speech_prob > this AND avg_logprob < logprob_threshold
    logprob_threshold=-1.0,
    logprob_floor=-1.5,           # reject any window with avg_logprob below this
    compression_ratio_threshold=2.4,  # applied after collapsing repeated n-grams
    max_repeats=2,          # repeated n-grams (shouted "Stop! Stop! Stop!", decoder loops) keep this many copies
    lang_keep_prob=0.6,     # keep the window's own language argmax when its probability reaches this ...
    lang_prior_min=0.5,     # ... else switch to the prior's language if it has at least this share of the prior ...
    lang_switch_min_prob=0.1,     # ... and at least this probability in the current window
    lang_hist_max_no_speech=0.5,  # windows feeding the prior: vad gate, no_speech_prob below this, not rejected
    max_speech_rate=8.0,    # cap of the speech_rate column (words per VAD-speech second)
    translate=True,         # text MT of non-English transcripts to English (for detoxify-unbiased / keywords)
    mt_batch_size=32,       # MT batch (fixed shape, padded)
    mt_max_text_tokens=160,  # native text longer than this keeps its last tokens (most recent speech)
    mt_max_new_tokens=128,
    min_rms_db=-65.0,       # tick valid only if (t - window_sec, t] has RMS >= this (and peak >= 1e-4)
    tox_batch_size=64,
    tox_max_tokens=128,
)

TOX_CLASSES = ["toxicity", "severe_toxicity", "obscene", "identity_attack", "insult", "threat", "sexual_explicit"]
# detoxify "multilingual" (XLM-R, Jigsaw multilingual) was trained on these languages only.
MTOX_LANGS = {"en", "fr", "es", "it", "pt", "tr", "ru"}

COLUMNS = (
    # voice activity (Silero VAD)
    ["vad_frac_1s", "vad_frac_win", "vad_frac_30s"]
    # latest ASR window ending at or before t
    + ["asr_ran", "has_text", "no_speech_prob", "asr_conf", "lang_en_prob", "non_english", "en_text",
       "words_per_sec", "speech_rate", "repeats"]
    # detoxify unbiased on English text (native or machine translation); 0 when en_text = 0
    + ["tox_" + c for c in TOX_CLASSES]
    # detoxify multilingual on the native transcript, only for its 7 training languages
    + ["mtox_lang_ok", "mtox_toxicity", "mtox_insult", "mtox_threat"]
    # lexical cues on English text
    + ["kw_threat", "kw_command", "kw_challenge", "kw_profanity", "kw_deescalate",
       "you_rate", "excl_rate", "question_rate"]
    # rolling maxima / means over windows ending in (t - roll_sec, t]
    + ["roll_tox_max", "roll_insult_max", "roll_threat_max", "roll_mtox_max", "roll_kw_hostile_max",
       "roll_words_per_sec"]
)
COL = {c: i for i, c in enumerate(COLUMNS)}
MAX_REPEATS_COL = 10.0  # cap of the repeats column

# Languages written without spaces: word counts come from the English translation instead.
NO_SPACE_LANGS = {"zh", "yue", "ja", "th", "lo", "my", "km", "bo"}
# CJK ideographs and kana (each character is a token for repeat collapsing and counts 1/2 word)
_CJK = "぀-ヿ㐀-䶿一-鿿豈-﫿"
_CJK_RE = re.compile(f"[{_CJK}]")

# ---------------------------------------------------------------------------------------------------------
# Lexical cue lists (English, lower-case, applied to the English text). Counts of regex matches per window.
# These are heuristics with modest precision (see selftest); the planned LLM escalation rating over the
# transcript sidecar should supersede them. Utterance-level commands ("Stop!", "Get back!") require the word
# to stand alone (sentence start, then punctuation or end) so narration ("trying to stop her") does not fire.
# ---------------------------------------------------------------------------------------------------------
_YOU = r"(?:you|ya|u|yo)"
_OBJ = r"(?:you|ya|u|yo|your|him|her|them|his|y'?all)"
_UTT = r"(?:^|(?<=[.!?,] )|(?<=[.!?,]))"  # utterance start: text start or right after . ! ? ,
_END = r"(?= *[!.,?]|$)"                  # utterance end: punctuation or end of text
_VIOLENT = (r"(?:kill|murder|shoot|stab|cut|hurt|beat|smash|punch|hit|slap|knock|bury|choke|strangle|destroy|"
            r"end|break|fuck)")
KEYWORDS = {
    "threat": [
        rf"\b(?:i'?ll|i will|i'?m (?:gonna|going to)|i am (?:gonna|going to)|we'?ll|we will|"
        rf"we'?re (?:gonna|going to)|gonna)\s+(?:fucking\s+)?{_VIOLENT}\s+{_OBJ}\b(?! up\b| back\b| to it\b)",
        rf"\bkill {_YOU}\b", rf"\b(?:shoot|stab|choke|smoke|pop|cap) {_YOU}\b(?! (?:an?|the|some|my|this) )",
        rf"\byou'?re (?:so |fucking )?(?:dead|done|finished){_END}", r"\byou'?re (?:gonna|going to) die\b",
        r"\byou (?:are|will be) dead\b",
        r"\b(?:beat|kick|whoop|whup|bust|kill|smash) (?:your|yo|ya) (?:fucking )?(?:ass|face|head|teeth)\b",
        rf"\bfuck {_YOU} up\b", rf"\b(?:knock|lay) {_YOU} (?:out|down)\b",
        r"\bwatch your back\b", r"\byou(?:'?ll| will| gonna| are gonna| are going to) regret\b",
        r"\bi'?ll find you\b", r"\bi know where you live\b", rf"\byou want some{_END}", r"\bwant to die\b",
        r"\bwant some of (?:this|me)\b", r"\bcatch (?:these|this) hands\b", r"\bi(?:'?ll| will) (?:get|end) you\b",
        r"\bi(?:'?ve| have) got a (?:gun|knife)\b", rf"\bor else{_END}", rf"\bor i'?ll (?:fucking )?{_VIOLENT}\b",
        r"\b(?:touch|hit|look at) (?:her|him|me|my \w+) again\b", r"\bone more (?:word|step|move)\b",
        r"\byou (?:won'?t|will not|ain'?t gonna|are not gonna|'re not gonna) (?:walk|leave|get|make it) out\b",
        r"\bi swear (?:to god )?i'?ll\b",
    ],
    "command": [
        r"\bback off\b", rf"\bback (?:the fuck )?(?:up|away){_END}",
        rf"\bget (?:the fuck )?(?:back|away|out|off|down){_END}", r"\bget (?:the fuck )?out of (?:here|my \w+)\b",
        r"\b(?:get|lie|lay) (?:down )?on the (?:ground|floor)\b", r"\bon your knees\b",
        r"\b(?:hands|put your hands) up\b", r"\bshow me (?:your )?hands\b", r"\bhands behind your back\b",
        r"\bdrop (?:it|that|the (?:gun|knife|weapon|bat)|your (?:gun|knife|weapon))\b(?! off)",
        r"\bput (?:it|that|the (?:gun|knife|weapon|bat)|your (?:gun|knife|weapon)) down\b",
        r"\bdon'?t (?:fucking )?(?:move|touch (?:me|him|her|it|that)|come (?:near|closer|any closer)|you dare)\b",
        r"\bdo not (?:move|touch (?:me|him|her))\b", rf"\bfreeze{_END}",
        r"\bshut (?:the fuck )?(?:up|your (?:fucking )?mouth)\b",
        r"\bleave (?:me|us|him|her|them) (?:the fuck )?alone\b",
        rf"\bgo away{_END}", rf"\bstep (?:back|away){_END}", r"\bstep out of the (?:car|vehicle)\b",
        r"\bstay (?:back|away from)\b", rf"\blet (?:me|him|her|us|them) go{_END}",
        rf"{_UTT}stop{_END}", r"\bstop (?:it|that|right there|resisting|moving|fighting)\b",
        rf"\bmove (?:back|away){_END}", r"\bout of (?:the|my) way\b", r"\bsit (?:the fuck )?down(?= *!)",
        rf"{_UTT}turn around{_END}", rf"\bcome here{_END}",
        r"\bgive me (?:the money|your (?:money|wallet|phone|keys|bag)|that|it back|my \w+ back)\b",
        rf"\bpull over{_END}", r"\bopen (?:the|this) (?:fucking )?door(?= *!)",
        r"\b(?:get|take) your (?:fucking )?hands off\b", r"\bout of my face\b",
    ],
    "challenge": [
        r"\bwhat (?:are |the fuck are )?you (?:gonna|going to) do(?: about it)?(?= *[!.?]|$)",
        r"\btry me\b", r"\bfight me\b", rf"\bhit me{_END}", r"\bi dare you\b", r"\b(?:wanna|want to) fight\b",
        rf"\bstep outside{_END}", rf"\bsay (?:that|it) again{_END}", r"\bwhat did you (?:just )?(?:fucking )?say\b",
        r"\bwho (?:the fuck )?do you think you are\b", r"\bwhat'?s your (?:fucking )?problem\b",
        r"\byou got a problem\b", r"\bwhat (?:are |the fuck are )?you looking at\b", r"\bsquare up\b",
        rf"\bbring it(?: on)?{_END}", rf"\bmake me{_END}", r"\b(?:you|ya) scared\b", r"\bdo something(?= *!)",
        r"\bcoward\b", r"\bman up\b", r"\byou wanna go\b", r"\bcome at me\b",
    ],
    "profanity": [
        r"\b(?:fuck\w*|motherf\w*|shit(?:s|ty|head|hole)?\b|bullshit|bitch\w*|asshole\w*|bastard\w*|cunt\w*|"
        r"dickhead\w*|prick(?:s)?\b|goddamn\w*|damn(?:ed|it)?\b|piss(?:ed)?\b|wanker\w*|twat\w*)",
        r"\b\w\*+\w*",  # Whisper sometimes censors swear words as "f***"
    ],
    "deescalate": [
        r"\bcalm down\b", r"\brelax\b", r"\bchill(?: out)?\b", r"\bit'?s (?:ok|okay|fine|alright|all right)\b",
        r"\bsorry\b", r"\bmy bad\b", r"\btake it easy\b", r"\blet'?s talk\b", r"\bno problem\b",
        r"\bwalk away\b", r"\bnot worth it\b", r"\bhold on\b", r"\bplease\b",
        r"\beasy,? (?:now|man|bro|buddy|there)\b",
    ],
}
_KW_RE = {k: [re.compile(p) for p in v] for k, v in KEYWORDS.items()}
_YOU_RE = re.compile(r"\b(?:you|your|yours|yourself|yourselves|ya|yall|y'all|u)\b")

# ---------------------------------------------------------------------------------------------------------
# Hallucination filter
# ---------------------------------------------------------------------------------------------------------
# Sentences containing these are subtitle credits / outro phrases from Whisper's training data; never real
# speech in this domain. They are removed wherever they appear.
_CREDIT_RE = re.compile(
    r"(amara\.org|subtit|subt[ií]tul|sous-titr|untertitel|субтитр|字幕|castingwords|transcribed by|translated by|"
    r"thanks? (?:you )?for watching|thank you so much for watching|please subscribe|subscribe to|"
    r"like and subscribe|продолжение следует|за просмотр|ご視聴|시청해 ?주셔서|감사합니다 시청|구독과 좋아요|"
    r"感谢观看|謝謝觀看|感谢收看|謝謝收看|多谢观看|请不吝点赞|點贊|订阅|訂閱|merci d'avoir regard|gracias por ver|"
    r"obrigad[oa] por assistir|danke f[üu]rs zuschauen|grazie per la visione|suscr[ií]bete|inscreva-se|"
    r"legendas? (?:por|pela)|napisy (?:wykonane|stworzone)|altyaz[ıi]|www\.|\.com\b|mbc 뉴스|copyright|"
    r"all rights reserved)",
    re.IGNORECASE,
)
# Whisper sound captions without brackets ("Музыка", "СПОКОЙНАЯ МУЗЫКА", "Música", "音乐"): a sentence of at most
# 3 words that contains one of these words is a caption, not speech.
_CAPTION_WORD_RE = re.compile(
    r"(music|musique|musik|m[uú]sica|muziek|muzyka|музык|müzik|音乐|音樂|음악|音楽|песн|canci[oó]n|applause|"
    r"applaudissements|"
    r"aplausos|applausi|beifall|applaus|аплодисмент|掌声|拍手|laughter|laughs|laughing|rires|risas|risos|"
    r"gelächter|смех|笑声|blank_audio|inaudible|unintelligible|no audio)",
    re.IGNORECASE,
)
# Bracketed / parenthesised / starred sound descriptions and sung lyrics between music symbols.
# Starred descriptions must stand alone ("*sighs*"), so censored words such as "f***ing" are kept.
_BRACKET_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)|(?:^|(?<=\s))\*[^*\n]{0,40}\*(?=$|[\s.,!?])|【[^】]*】|（[^）]*）|"
                         r"\uff3b[^\uff3d]*\uff3d")
_MUSIC_SYMBOLS = "\u00b6\u266a\u266b\u266c\u2669"
_LYRICS_RE = re.compile(f"[{_MUSIC_SYMBOLS}][^{_MUSIC_SYMBOLS}]*[{_MUSIC_SYMBOLS}]")
_MUSIC_RE = re.compile(f"[{_MUSIC_SYMBOLS}]+")
# Stock outputs on near-silence (compared without spaces/punctuation); a window whose every sentence or clause is
# one of these is dropped when it has little VAD speech, a raised no_speech_prob or came through the weak gate.
_STOCK_PHRASES = {
    "thankyou", "thankyouverymuch", "thanks", "thankyousomuch", "okaythankyou", "thanksalot", "спасибо",
    "спасибобольшое", "большоеспасибо", "merci", "mercibeaucoup", "obrigado", "obrigada", "muitoobrigado",
    "muitoobrigada", "gracias", "muchasgracias", "danke", "dankeschön", "vielendank", "grazie", "grazietante",
    "graziemille", "谢谢", "謝謝", "谢谢大家", "謝謝大家", "ありがとう", "ありがとうございました", "ありがとうございます",
    "감사합니다", "고맙습니다", "teşekkürler", "teşekkürederim", "dziękuję", "dankje", "bedankt", "terimakasih",
    "cảmơn", "شكرا", "bye", "byebye", "goodbye", "tchau", "adiós", "adios", "aurevoir", "пока", "досвидания",
    "再见", "再見", "tschüss", "ciao", "hola", "you", "oh", "uh", "um", "hmm", "so", "ah", "theend", "silence",
    "music", "applause", "okay", "嗯", "啊", "えー", "ん",
}
_STOCK_MAX_SPEECH_SEC = 2.0
# English words that are never a translation on their own ("Музыка" -> "The" was seen with Whisper translate)
_FUNCTION_WORDS = {"the", "a", "an", "and", "or", "so", "of", "to", "in", "on", "it", "i", "you", "is", "uh", "um",
                   "oh", "hmm", "ah", "music", "applause"}
# Text MT outputs that are commentary or refusals instead of a translation
_MT_REFUSAL_RE = re.compile(
    r"(\bi'?m sorry,? but\b|\bi (?:cannot|can'?t|am unable to|'m unable to) (?:assist|help|translate|provide|comply)\b|"
    r"\bas an ai\b|\bcould you (?:please )?(?:provide|clarify)\b|\btranscript:|\btranslation:|"
    r"\bthe (?:text|transcript|original) (?:is|appears|seems)\b)",
    re.IGNORECASE,
)


def _norm(text):
    return re.sub(r"\s+", " ", text.replace("’", "'").replace("‘", "'")).strip()


def _key(text):
    """Comparison key: lower case, hyphens as spaces, punctuation and spaces removed."""
    t = re.sub(r"[^\w\s']", " ", text.lower().replace("-", " "))
    return re.sub(r"[\s']+", "", t)


def _split_sents(text):
    """Sentences, split on .!?… followed by a space, or after CJK stops (punctuation kept)."""
    return [s.strip() for s in re.split("(?<=[.!?…])\\s+|(?<=[。！？])\\s*", text) if s.strip()]


def _strip_credits(text):
    """Drop sentences that contain a subtitle-credit / outro phrase."""
    return " ".join(s for s in _split_sents(text) if not _CREDIT_RE.search(s)).strip()


def _n_words(s):
    return len([w for w in _CJK_RE.sub(" ", s).split() if re.search(r"\w", w)])


def _n_units(text):
    """Word count; CJK ideographs / kana count 1/2 word each (no spaces in those scripts)."""
    return _n_words(text) + len(_CJK_RE.findall(text)) / 2.0


def _strip_captions(text):
    """Remove bracketed sound descriptions, sung lyrics, music symbols and short caption sentences."""
    text = _LYRICS_RE.sub(" ", text)
    text = _BRACKET_RE.sub(" ", text)
    text = _norm(_MUSIC_RE.sub(" ", text))
    kept = [s for s in _split_sents(text)
            if not (_CAPTION_WORD_RE.search(s) and _n_words(s) <= 3 and len(s) <= 30)]
    return " ".join(kept).strip()


def _compression_ratio(text):
    b = text.encode("utf-8")
    return len(b) / max(1, len(zlib.compress(b))) if b else 0.0


_TOKEN_RE = re.compile(f"[{_CJK}]|[^\\s{_CJK}]+")


def collapse_repeats(text, max_repeats=2, max_n=16):
    """Collapse consecutive repeats of any 1..max_n-token n-gram to at most ``max_repeats`` copies.

    Tokens are whitespace words, with each CJK character its own token; comparison ignores case and punctuation.
    Returns (text, n_removed) where n_removed counts removed copies ("Stop! Stop! Stop! Stop!" -> "Stop! Stop!", 2).
    Shouted repetitions are an escalation cue and Whisper decoder loops repeat a phrase until the token budget
    ends; collapsing keeps the real text of both instead of rejecting the whole window on compression ratio.
    """
    spans = [(m.group(0), text[m.end():m.end() + 1] == " ") for m in _TOKEN_RE.finditer(text)]
    keys = [re.sub(r"[^\w]", "", s.lower()) for s, _ in spans]
    out, i, removed, n_tok = [], 0, 0, len(spans)
    while i < n_tok:
        hit = None
        for n in range(1, max_n + 1):
            if i + n * (max_repeats + 1) > n_tok:
                break
            unit = keys[i:i + n]
            if not any(unit):
                continue
            r = 1
            while keys[i + r * n:i + (r + 1) * n] == unit:
                r += 1
            if r > max_repeats:
                hit = (n, r)
                break
        if hit:
            n, r = hit  # keep the first copies and the last one (it carries the run's final punctuation)
            out.extend(spans[i:i + n * (max_repeats - 1)])
            out.extend(spans[i + n * (r - 1):i + n * r])
            removed += r - max_repeats
            i += n * r
        else:
            out.append(spans[i])
            i += 1
    res = "".join(s + (" " if sp else "") for s, sp in out).strip()
    return res, removed


def filter_text(text, avg_logprob, no_speech_prob, speech_sec, weak_gate, cfg, vad_max_prob=0.0):
    """Return (clean_text, reject_reason or None, n_repeats_removed) for one Whisper window transcript.

    Weak-gate windows with fewer than ``weak_min_words`` words (counted before repeat collapsing, so
    "No! No! No! No!" counts 4) are kept only if the window's max VAD probability reaches
    ``weak_short_vad_prob``: short real shouts ("Hold it.", "过来") had max VAD prob >= 0.94 in the smoke set,
    short music/noise hallucinations ("nine", "like", "ПЕСНЯ", "Музыка") at most 0.47.

    Note: there is deliberately no words-per-VAD-second check. Silero VAD badly underestimates shouted and
    noisy speech (bodycam shouting scored 0.6 s of speech for 20 clearly spoken words), so such a check
    rejects exactly the speech this extractor is for. For the same reason a high compression ratio no longer
    rejects outright: repeated n-grams are collapsed first.
    """
    text = _norm(text)
    had = bool(re.search(r"\w", text))
    text = _strip_captions(text)
    if not re.search(r"\w", text):
        return "", ("sound_caption" if had else "empty"), 0
    n_units_raw = _n_units(_strip_credits(text))
    text, n_rep = collapse_repeats(text, cfg["max_repeats"])
    if _compression_ratio(text) > cfg["compression_ratio_threshold"]:
        return "", "compression_ratio", n_rep
    if no_speech_prob > cfg["no_speech_threshold"] and avg_logprob < cfg["logprob_threshold"]:
        return "", "no_speech", n_rep
    if avg_logprob < cfg["logprob_floor"]:
        return "", "low_logprob", n_rep
    if weak_gate and (no_speech_prob > cfg["weak_max_no_speech"] or avg_logprob < cfg["weak_min_logprob"]):
        return "", "weak_gate", n_rep
    clean = _strip_credits(text)
    if not re.search(r"\w", clean):
        return "", "credit_phrase", n_rep
    pieces = [_key(p) for p in re.split(r"[.!?。！？,，、…;]+", clean)]
    pieces = [p for p in pieces if p]
    if (pieces and all(p in _STOCK_PHRASES for p in pieces)
            and (speech_sec < _STOCK_MAX_SPEECH_SEC or no_speech_prob > 0.2 or weak_gate)):
        return "", "stock_phrase", n_rep
    if weak_gate and n_units_raw < cfg["weak_min_words"] and vad_max_prob < cfg["weak_short_vad_prob"]:
        return "", "weak_short", n_rep
    return clean, None, n_rep


def check_translation(native, text_en, cfg):
    """Validate a machine translation of ``native``. Returns (clean English text, reject_reason or None)."""
    en = _norm(text_en.split("\n")[0])
    en = re.sub(r"^(?:english|translation)\s*:\s*", "", en, flags=re.IGNORECASE).strip().strip('"').strip()
    en = _strip_captions(en)
    if not re.search(r"\w", en):
        return "", "mt_empty"
    if _MT_REFUSAL_RE.search(en):
        return "", "mt_refusal"
    letters = re.findall(r"[^\W\d_]", en)
    n_foreign = sum(ord(c) > 0x24F for c in letters)
    if letters and n_foreign / len(letters) > 0.2:
        # partly untranslated ("Get the fuck out,整理好 走"): keep the English if it is the clear majority
        latin = _norm(re.sub(r"[^\W\d_]*[^\x00-ɏ\s]+[^\W\d_]*", " ", en))
        if n_foreign / len(letters) > 0.5 or len(latin.split()) < 2:
            return "", "mt_untranslated"
        en = latin
    en, _ = collapse_repeats(en, cfg["max_repeats"])
    if _compression_ratio(en) > cfg["compression_ratio_threshold"]:
        return "", "mt_compression"
    en = _strip_credits(en)
    if not re.search(r"\w", en):
        return "", "mt_credit"
    n_en = len(en.split())
    if n_en > 2.5 * _n_units(native) + 5:
        return "", "mt_length"
    if n_en == 1 and _key(en) in _FUNCTION_WORDS:
        return "", "mt_trivial"
    return en, None


def text_cues(text_en):
    """Keyword counts, second-person rate, '!' and '?' rates of an English text."""
    low = _norm(text_en).lower()
    out = {k: float(sum(len(r.findall(low)) for r in regs)) for k, regs in _KW_RE.items()}
    words = low.split()
    nw = max(1, len(words))
    sents = max(1, len([s for s in re.split(r"[.!?]+", low) if s.strip()]))
    out["you_rate"] = len(_YOU_RE.findall(low)) / nw if words else 0.0
    out["excl_rate"] = low.count("!") / sents if words else 0.0
    out["question_rate"] = low.count("?") / sents if words else 0.0
    return out


# ---------------------------------------------------------------------------------------------------------
# Models (loaded once per device and cached)
# ---------------------------------------------------------------------------------------------------------
_MODELS = {}


class _GraphDecoder:
    """Single-token Whisper decoder step with a static KV cache, replayed as a CUDA graph.

    The HF decoder costs about 60 ms of CPU launch overhead per token on a loaded machine; the graph replays the
    same math (pre-LN layers, q scaling, bias-free k_proj, exact GELU, final LayerNorm, tied output projection)
    in 8 to 20 ms of GPU time. Shapes are fixed per (device, batch size), so a row's result never depends on
    the other rows of the batch.
    """

    def __init__(self, model, batch_size, max_len, device):
        self.torch = torch
        self.model, self.dec = model, model.model.decoder
        cfg = model.config
        self.B, self.max_len = batch_size, max_len
        self.H = cfg.decoder_attention_heads
        self.Dh = cfg.d_model // self.H
        L, B, H, Dh = cfg.decoder_layers, batch_size, self.H, self.Dh
        dt = model.dtype
        self.tok = torch.zeros(B, dtype=torch.long, device=device)
        self.pos = torch.zeros(1, dtype=torch.long, device=device)
        self.ar = torch.arange(max_len, device=device)
        self.k_self = torch.zeros(L, B, H, max_len, Dh, dtype=dt, device=device)
        self.v_self = torch.zeros_like(self.k_self)
        self.k_cross = torch.zeros(L, B, H, cfg.max_source_positions, Dh, dtype=dt, device=device)
        self.v_cross = torch.zeros_like(self.k_cross)
        self.device = device
        self.graph = None
        self.logits = None

    def set_encoder(self, enc):
        """Project the encoder output [B, 1500, d] into the static cross-attention K/V buffers."""
        B, H, Dh = self.B, self.H, self.Dh
        with torch.no_grad():
            for i, layer in enumerate(self.dec.layers):
                ca = layer.encoder_attn
                self.k_cross[i].copy_(ca.k_proj(enc).view(B, -1, H, Dh).transpose(1, 2))
                self.v_cross[i].copy_(ca.v_proj(enc).view(B, -1, H, Dh).transpose(1, 2))

    def _step(self):
        F = torch.nn.functional
        B, H, Dh = self.B, self.H, self.Dh
        d = H * Dh
        dec = self.dec
        x = dec.embed_tokens(self.tok[:, None]) + dec.embed_positions.weight[self.pos][None]
        mask = torch.zeros(1, 1, 1, self.max_len, dtype=x.dtype, device=self.device)
        mask = mask.masked_fill(self.ar[None, None, None, :] > self.pos, float("-inf"))
        for i, layer in enumerate(dec.layers):
            sa, ca = layer.self_attn, layer.encoder_attn
            h = layer.self_attn_layer_norm(x)
            q = (sa.q_proj(h) * sa.scaling).view(B, 1, H, Dh).transpose(1, 2)
            self.k_self[i].index_copy_(2, self.pos, sa.k_proj(h).view(B, 1, H, Dh).transpose(1, 2))
            self.v_self[i].index_copy_(2, self.pos, sa.v_proj(h).view(B, 1, H, Dh).transpose(1, 2))
            a = F.scaled_dot_product_attention(q, self.k_self[i], self.v_self[i], attn_mask=mask, scale=1.0)
            x = x + sa.out_proj(a.transpose(1, 2).reshape(B, 1, d))
            h = layer.encoder_attn_layer_norm(x)
            q = (ca.q_proj(h) * ca.scaling).view(B, 1, H, Dh).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, self.k_cross[i], self.v_cross[i], scale=1.0)
            x = x + ca.out_proj(a.transpose(1, 2).reshape(B, 1, d))
            h = layer.final_layer_norm(x)
            x = x + layer.fc2(layer.activation_fn(layer.fc1(h)))
        x = dec.layer_norm(x)
        return self.model.proj_out(x[:, 0]).float()

    def _capture(self):
        with torch.no_grad(), torch.cuda.device(self.device):
            s = torch.cuda.Stream(device=self.device)
            s.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._step()
            torch.cuda.current_stream(self.device).wait_stream(s)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.logits = self._step()

    def step(self, tok, pos):
        """Feed token ``tok`` [B] at position ``pos``; returns a fresh copy of the logits [B, vocab] (float32)."""
        if self.graph is None:
            self._capture()
        with torch.cuda.device(self.device):  # replay on the graph's device even if cuda:0 is current
            self.tok.copy_(tok)
            self.pos.fill_(int(pos))
            self.graph.replay()
            return self.logits.clone()


class _Whisper:
    """Batched greedy Whisper large-v3 with explicit language id, no_speech_prob and timestamp rules."""

    def __init__(self, device):
        from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer

        self.device = device
        self.model = WhisperForConditionalGeneration.from_pretrained(
            WHISPER_ID, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        self.tok = WhisperTokenizer.from_pretrained(WHISPER_ID)
        fe = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)
        self.mel_filters = torch.from_numpy(fe.mel_filters).to(device, torch.float32)  # [201, 128]
        self.window = torch.hann_window(fe.n_fft, device=device)
        self.n_fft, self.hop = fe.n_fft, fe.hop_length
        gc = self.model.generation_config
        self.lang_ids = {k.strip("<|>"): v for k, v in gc.lang_to_id.items()}
        self.lang_codes = sorted(self.lang_ids, key=self.lang_ids.get)
        self.lang_tok = torch.tensor([self.lang_ids[c] for c in self.lang_codes], device=device)
        self.sot = gc.decoder_start_token_id
        self.eot = gc.eos_token_id
        self.transcribe, self.translate = gc.task_to_id["transcribe"], gc.task_to_id["translate"]
        self.no_ts = gc.no_timestamps_token_id
        self.ts_begin = self.no_ts + 1
        self.max_initial_ts = gc.max_initial_timestamp_index
        self.no_speech = self.tok.convert_tokens_to_ids("<|nospeech|>")
        vocab = self.model.config.vocab_size
        sup = torch.zeros(vocab, dtype=torch.bool, device=device)
        sup[list(gc.suppress_tokens)] = True
        sup[self.eot + 1:] = True  # every special, language and timestamp token
        self.suppress = sup
        sup_ts = sup.clone()
        sup_ts[self.ts_begin:] = False  # timestamp mode: timestamps allowed (rules applied in decode)
        self.suppress_ts = sup_ts
        self.begin_suppress = torch.zeros(vocab, dtype=torch.bool, device=device)
        self.begin_suppress[list(gc.begin_suppress_tokens)] = True
        self.cols = torch.arange(vocab, device=device)[None, :]
        self._graphs = {}

    def graph(self, batch_size, max_new):
        key = (batch_size, max_new)
        if key not in self._graphs:
            self._graphs[key] = _GraphDecoder(self.model, batch_size, max_new + 5, self.device)
        return self._graphs[key]

    def log_mel(self, wavs):
        """[B, 480000] float32 (zero padded) -> [B, 128, 3000] log-mel, per-sample normalised like Whisper."""
        x = torch.as_tensor(wavs, device=self.device, dtype=torch.float32)
        stft = torch.stft(x, self.n_fft, self.hop, window=self.window, return_complex=True)
        mag = stft[..., :-1].abs() ** 2
        mel = self.mel_filters.T @ mag
        log = torch.clamp(mel, min=1e-10).log10()
        log = torch.maximum(log, log.amax(dim=(1, 2), keepdim=True) - 8.0)
        return (log + 4.0) / 4.0

    def encode(self, wavs, micro_batch):
        """Encoder output [N, 1500, d] for N zero-padded 30 s waveforms, in fixed-size micro-batches."""
        outs = []
        with torch.no_grad():
            for i in range(0, len(wavs), micro_batch):
                chunk = wavs[i:i + micro_batch]
                n = len(chunk)
                if n < micro_batch:
                    chunk = np.concatenate([chunk, np.zeros((micro_batch - n,) + chunk.shape[1:], chunk.dtype)])
                mel = self.log_mel(chunk).to(torch.bfloat16)
                outs.append(self.model.model.encoder(input_features=mel).last_hidden_state[:n])
        return torch.cat(outs)

    def decode(self, enc, n_real, task, max_new, timestamps=True, lang_codes=None, choose_lang=None):
        """Greedy decoding of a fixed-size batch of encoder outputs (rows >= n_real are padding).

        Either ``lang_codes`` (one per real row) is given, or ``choose_lang(lang_p [n, n_lang], no_speech [n])``
        is called after the <|startoftranscript|> step and returns the codes (language detection and
        no_speech_prob come from those logits, as in OpenAI Whisper). With ``timestamps=True`` the OpenAI
        timestamp rules are applied (paired, monotonic timestamp tokens; first token a timestamp <= 1 s; a
        timestamp is forced when the total timestamp probability beats every text token); this mode is much
        less prone than <|notimestamps|> mode to stopping early with only a fragment.
        Returns (texts, avg_logprobs, segments, lang_p, no_speech); segments[i] = [(t0, t1, text), ...].
        """
        B = enc.shape[0]
        ninf = float("-inf")
        tb, cols = self.ts_begin, self.cols
        G = self.graph(B, max_new)
        G.set_encoder(enc)
        logits = G.step(torch.full((B,), self.sot, dtype=torch.long, device=self.device), 0)
        lang_p = logits[:n_real, self.lang_tok].softmax(-1).cpu().numpy()
        no_speech = logits[:n_real].softmax(-1)[:, self.no_speech].cpu().numpy()
        if lang_codes is None:
            lang_codes = choose_lang(lang_p, no_speech)
        lang_codes = list(lang_codes) + ["en"] * (B - n_real)
        task_id = self.transcribe if task == "transcribe" else self.translate
        prompt = [[self.lang_ids[c] for c in lang_codes], [task_id] * B] + ([] if timestamps else [[self.no_ts] * B])
        pos = 1
        for col in prompt:
            logits = G.step(torch.tensor(col, dtype=torch.long, device=self.device), pos)
            pos += 1
        suppress = (self.suppress_ts if timestamps else self.suppress)[None, :]
        done = torch.arange(B, device=self.device) >= n_real
        sum_lp = torch.zeros(B, device=self.device)
        n_tok = torch.zeros(B, device=self.device)
        last_ts = torch.full((B,), -1, dtype=torch.long, device=self.device)  # largest timestamp token so far
        toks = []
        with torch.no_grad():
            for step in range(max_new):
                logits.masked_fill_(suppress, ninf)
                if step == 0:
                    logits.masked_fill_(self.begin_suppress[None, :], ninf)
                if timestamps:
                    if step == 0:
                        logits.masked_fill_((cols < tb) | (cols > tb + self.max_initial_ts), ninf)
                    else:
                        prev_ts = toks[-1] >= tb
                        pen_ts = toks[-2] >= tb if step >= 2 else torch.ones_like(prev_ts)
                        logits.masked_fill_((prev_ts & pen_ts)[:, None] & (cols >= tb), ninf)  # pair done: text/eot
                        logits.masked_fill_((prev_ts & ~pen_ts)[:, None] & (cols < self.eot), ninf)  # close pair
                        lower = torch.where(prev_ts & ~pen_ts, last_ts, last_ts + 1)
                        logits.masked_fill_((last_ts >= 0)[:, None] & (cols >= tb) & (cols < lower[:, None]), ninf)
                    lp0 = logits.log_softmax(-1)
                    force_ts = lp0[:, tb:].logsumexp(-1) > lp0[:, :tb].max(-1).values
                    logits.masked_fill_(force_ts[:, None] & (cols < tb), ninf)
                lp = logits.log_softmax(-1)
                nxt = torch.where(done, torch.full((B,), self.eot, device=self.device), logits.argmax(-1))
                sum_lp += torch.where(done, torch.zeros_like(sum_lp), lp.gather(1, nxt[:, None])[:, 0])
                n_tok += (~done).float()
                toks.append(nxt)
                last_ts = torch.where(nxt >= tb, torch.maximum(last_ts, nxt), last_ts)
                done = done | (nxt == self.eot)
                if bool(done.all()) or pos >= G.max_len:
                    break
                logits = G.step(nxt, pos)
                pos += 1
        toks = torch.stack(toks, 1)[:n_real].cpu().numpy()
        avg_lp = (sum_lp / n_tok.clamp(min=1))[:n_real].cpu().numpy()
        texts, segments = [], []
        for row in toks:
            ids = row.tolist()
            if self.eot in ids:
                ids = ids[:ids.index(self.eot)]
            segs, cur, t_open = [], [], None
            for x in ids:
                if x >= tb:
                    t = (x - tb) * 0.02
                    if cur:
                        segs.append((t_open if t_open is not None else t, t, self.tok.decode(cur).strip()))
                        cur, t_open = [], None
                    else:
                        t_open = t
                else:
                    cur.append(x)
            if cur:  # unterminated final segment (no-timestamp mode or budget exhausted)
                segs.append((t_open if t_open is not None else 0.0, None, self.tok.decode(cur).strip()))
            segs = [s for s in segs if s[2]]
            segments.append(segs)
            texts.append(" ".join(s[2] for s in segs).strip())
        return texts, avg_lp, segments, lang_p, no_speech


class _Tox:
    """detoxify checkpoint (pip ``detoxify`` weights) with fixed-shape batching for batch invariance."""

    def __init__(self, model_type, device):
        from detoxify import Detoxify

        d = Detoxify(model_type, device=device)
        self.model, self.tokenizer, self.class_names = d.model.eval(), d.tokenizer, d.class_names
        self.idx = [self.class_names.index(c) for c in TOX_CLASSES]
        self.device = device

    def __call__(self, texts, batch_size, max_tokens):
        out = np.zeros((len(texts), len(TOX_CLASSES)), np.float32)
        for i in range(0, len(texts), batch_size):
            chunk = list(texts[i:i + batch_size])
            n = len(chunk)
            chunk += [""] * (batch_size - n)
            enc = self.tokenizer(chunk, return_tensors="pt", truncation=True, padding="max_length",
                                 max_length=max_tokens).to(self.device)
            with torch.no_grad():
                p = torch.sigmoid(self.model(**enc)[0].float()).cpu().numpy()
            out[i:i + n] = p[:n, self.idx]
        return out


class _MT:
    """Text machine translation of native transcripts into English with Qwen2.5-7B-Instruct.

    Used as a plain few-shot completion ("Transcript: ...\\nEnglish:"), not through the chat template: in chat
    mode the model refused or lectured on profane input ("I'm sorry, but I cannot translate that type of
    language") and echoed instructions; in completion mode it translated all test lines faithfully. Greedy,
    left-padded to a fixed prompt length and a fixed batch size, so a text's translation never depends on the
    other texts of the batch (row-invariant; verified). Stops at the first newline.
    """

    HEAD = ("Translate noisy speech transcripts (movie dialogue, street videos, CCTV) into English. "
            "Translate faithfully and literally, keeping insults and profanity; add nothing.\n\n")
    SHOTS = [("Oye, ¿dónde está mi teléfono? No lo encuentro.", "Hey, where is my phone? I can't find it."),
             ("我们走吧，快点！", "Let's go, hurry up!"),
             ("Je sais pas, merde.", "I don't know, shit.")]

    def __init__(self, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.tok = AutoTokenizer.from_pretrained(MT_ID)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            MT_ID, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        gc = self.model.generation_config
        gc.temperature = gc.top_p = gc.top_k = gc.repetition_penalty = None  # greedy; silences sampling warnings
        vocab = self.tok.convert_ids_to_tokens(list(range(len(self.tok))))
        newline = "Ċ"  # byte-level BPE newline
        self.stop = [i for i, t in enumerate(vocab) if t and newline in t and set(t) <= set(newline + "Ġĉ")]
        self.stop += [self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|endoftext|>")]
        self.prefix = self.HEAD + "".join(f"Transcript: {a}\nEnglish: {b}\n\n" for a, b in self.SHOTS)
        self.n_prefix = len(self.tok(self.prefix + "Transcript: ")["input_ids"])

    def __call__(self, texts, batch_size, max_text_tokens, max_new):
        """English translations of ``texts``. Each row gets its own token budget,
        min(max_new, 1.5 x its native tokens + 16), enforced by a per-row stopping criterion, so a row's output
        never depends on the other rows (a batch no longer runs to 128 steps because one row is long)."""
        from transformers import StoppingCriteria, StoppingCriteriaList

        class RowBudget(StoppingCriteria):
            def __init__(self, start, budget):
                self.start, self.budget = start, budget

            def __call__(self, input_ids, scores, **kwargs):
                return (input_ids.shape[1] - self.start) >= self.budget

        max_in = self.n_prefix + max_text_tokens + 8
        prompts, budgets = [], []
        for t in texts:
            ids = self.tok(t, add_special_tokens=False)["input_ids"]
            if len(ids) > max_text_tokens:  # keep the most recent speech
                ids = ids[-max_text_tokens:]
                t = self.tok.decode(ids)
            prompts.append(f"{self.prefix}Transcript: {t}\nEnglish:")
            budgets.append(min(max_new, int(1.5 * len(ids)) + 16))
        pad_prompt = f"{self.prefix}Transcript: \nEnglish:"
        out = []
        for i in range(0, len(prompts), batch_size):
            chunk, bud = prompts[i:i + batch_size], budgets[i:i + batch_size]
            n = len(chunk)
            chunk, bud = chunk + [pad_prompt] * (batch_size - n), bud + [1] * (batch_size - n)
            enc = self.tok(chunk, return_tensors="pt", padding="max_length", truncation=True,
                           max_length=max_in).to(self.device)
            stop = StoppingCriteriaList([RowBudget(max_in, torch.tensor(bud, device=self.device))])
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=max(bud), do_sample=False, stopping_criteria=stop,
                                          pad_token_id=self.tok.pad_token_id, eos_token_id=self.stop)
            out += [self.tok.decode(g[max_in:max_in + b], skip_special_tokens=True).split("\n")[0].strip()
                    for g, b in zip(gen[:n], bud[:n])]
        return out


def get_models(device):
    """Load (once per process and device) Whisper large-v3, Silero VAD and the two detoxify models."""
    if device not in _MODELS:
        n_threads = torch.get_num_threads()
        from silero_vad import load_silero_vad  # its import calls torch.set_num_threads(1) process-wide

        torch.set_num_threads(n_threads)
        _MODELS[device] = dict(
            whisper=_Whisper(device),
            vad=load_silero_vad(onnx=True),  # CPU, single thread
            tox=_Tox("unbiased", device),
            mtox=_Tox("multilingual", device),
            torch=torch,
        )
    return _MODELS[device]


def get_mt(device):
    """The text MT model (loaded on first use: English-only processes never pay its 15 GB)."""
    models = get_models(device)
    if "mt" not in models:
        models["mt"] = _MT(device)
    return models["mt"]


# ---------------------------------------------------------------------------------------------------------
# VAD and validity
# ---------------------------------------------------------------------------------------------------------
def vad_probs(vad, wav):
    """Silero speech probability per 32 ms frame (frame i covers samples [512 i, 512 (i+1))).

    Exact streaming inference over the whole track (ONNX, one CPU thread, about 25 s per hour of audio on an
    idle machine, 3 to 4x slower under heavy CPU load): the recurrent state at frame i depends only on audio up
    to the end of frame i, so it is strictly causal and identical on truncated copies. Do not reset the state
    in segments: Silero adapts to the background over tens of seconds, and segment resets lost most of the
    shouting in a bodycam test (3 % vs 45 % speech).
    """
    return vad.audio_forward(torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)), SR)[0].numpy()


def tick_valid(wav, t_sec, window_sec, first_tick, min_rms_db):
    """Causal per-tick validity: audible audio in (t - window_sec, t] and t >= first_tick.

    Audible = peak >= 1e-4 (not digital silence) and RMS >= ``min_rms_db`` dBFS (not encoder noise), the same
    rule as the audio track (``audio_dsp.tick_valid``). Ticks past the end of the decoded audio are invalid.
    """
    T = len(t_sec)
    valid = np.zeros(T, bool)
    if wav is None or wav.size == 0:
        return valid
    hop = SR // 100
    n_hop = len(wav) // hop
    blocks = wav[:n_hop * hop].reshape(n_hop, hop)
    peak = np.abs(blocks).max(1) if n_hop else np.zeros(0)
    css = np.concatenate([[0.0], np.cumsum(np.square(blocks, dtype=np.float64).sum(1))])
    min_ms = 10.0 ** (min_rms_db / 10.0)
    for k, t in enumerate(np.asarray(t_sec, dtype=np.float64)):
        b = int(round(t * 100))
        a = max(0, int(round((t - window_sec) * 100)))
        if t < first_tick or b > n_hop or b <= a:
            continue
        valid[k] = peak[a:b].max() >= SILENCE_PEAK and (css[b] - css[a]) / ((b - a) * hop) >= min_ms
    return valid


# ---------------------------------------------------------------------------------------------------------
# Windowed ASR
# ---------------------------------------------------------------------------------------------------------
def _frame_range(a, b):
    """Indices [i0, i1) of the VAD frames that lie fully inside [a, b] seconds."""
    fs = VAD_FRAME / SR
    return int(np.ceil(a / fs - 1e-9)), int(np.floor(b / fs + 1e-9))


def _max_prob(vad_p, a, b):
    i0, i1 = _frame_range(a, b)
    i1 = min(i1, len(vad_p))
    return float(vad_p[i0:i1].max()) if i1 > i0 else 0.0


def _speech_seconds(vad_p, thr, a, b):
    """Seconds of VAD speech (prob > thr) in frames fully inside [a, b] (seconds)."""
    i0, i1 = _frame_range(a, b)
    return float((vad_p[i0:i1] > thr).sum()) * VAD_FRAME / SR if i1 > i0 else 0.0


def transcribe_windows(wav, vad_p, ends, models, cfg):
    """Run gated Whisper on windows [max(0, e - W), e] for each end e. Returns a list of window dicts.

    Windows that pass the VAD gate are processed in time order in fixed-size batches (padded), so the result
    for a window never depends on later windows: a truncated copy of the video gives identical transcripts.
    """
    W, thr, R = cfg["window_sec"], cfg["vad_threshold"], cfg["roll_sec"]
    wh = models["whisper"]
    B = cfg["batch_size"]
    wins = []
    for e in ends:
        a = max(0.0, float(e) - W)
        sp = _speech_seconds(vad_p, thr, a, float(e))
        pmax = _max_prob(vad_p, a, float(e))
        gate = "vad" if sp >= cfg["min_speech_sec"] else ("weak" if pmax >= cfg["weak_gate_prob"] else None)
        wins.append(dict(start=a, end=float(e), vad_speech_sec=round(sp, 3), vad_max_prob=round(pmax, 3),
                         gate=gate, asr_ran=gate is not None))
    todo = [i for i, w in enumerate(wins) if w["asr_ran"]]
    # (window index, end, weight, posterior) of earlier reliable ASR windows, for the causal language prior.
    # Windows of earlier batches whose transcript was rejected are skipped when the prior is formed.
    lang_hist = []
    en_idx = wh.lang_codes.index("en")

    def pad(enc):
        n = enc.shape[0]
        if n == B:
            return enc
        return torch.cat([enc, torch.zeros((B - n,) + tuple(enc.shape[1:]), dtype=enc.dtype, device=enc.device)])

    def choose_lang(idx, lang_p, no_sp):
        """Current-window argmax; the prior over earlier accepted windows only breaks near-ties."""
        codes = []
        for r, i in enumerate(idx):
            w = wins[i]
            p = lang_p[r]
            k_raw = k = int(np.argmax(p))
            used_prior = False
            if p[k_raw] < cfg["lang_keep_prob"]:
                hist = [(wt, pj) for (j, e_j, wt, pj) in lang_hist
                        if w["end"] - R < e_j < w["end"] and not wins[j].get("rejected")]
                if hist:
                    q = sum(wt * pj for wt, pj in hist) / max(1e-9, sum(wt for wt, _ in hist))
                    kp = int(np.argmax(q))
                    if kp != k_raw and q[kp] >= cfg["lang_prior_min"] and p[kp] >= cfg["lang_switch_min_prob"]:
                        k, used_prior = kp, True
            ns = float(no_sp[r])
            if w["gate"] == "vad" and ns < cfg["lang_hist_max_no_speech"]:
                lang_hist.append((i, w["end"], max(w["vad_speech_sec"], 0.5) * (1.0 - ns), p))
            lang_hist[:] = [h for h in lang_hist if h[1] > w["end"] - R]
            w.update(lang=wh.lang_codes[k], lang_raw=wh.lang_codes[k_raw], lang_prior=used_prior,
                     lang_prob=round(float(p[k]), 4), lang_en_prob=round(float(p[en_idx]), 4),
                     no_speech_prob=round(ns, 4))
            codes.append(wh.lang_codes[k])
        return codes

    def run_batch(idx):
        n = len(idx)
        audio = np.zeros((n, N_SAMPLES), np.float32)
        for r, i in enumerate(idx):
            seg = wav[int(round(wins[i]["start"] * SR)):int(round(wins[i]["end"] * SR))]
            audio[r, :len(seg)] = seg  # window_sec <= 30 is enforced in extract()
        enc = wh.encode(audio, cfg["enc_batch_size"])
        texts, avg_lp, segs, _, _ = wh.decode(pad(enc), n, "transcribe", cfg["max_new_tokens"], cfg["timestamps"],
                                              choose_lang=lambda lp, ns: choose_lang(idx, lp, ns))
        for r, i in enumerate(idx):
            w = wins[i]
            clean, why, n_rep = filter_text(texts[r], float(avg_lp[r]), w["no_speech_prob"], w["vad_speech_sec"],
                                            w["gate"] == "weak", cfg, w["vad_max_prob"])
            w.update(raw_text=texts[r], avg_logprob=round(float(avg_lp[r]), 4),
                     compression_ratio=round(_compression_ratio(_norm(texts[r])), 3), text=clean, rejected=why,
                     repeats=int(n_rep) if clean else 0)
            if clean and cfg["timestamps"]:
                w["segments"] = [[round(w["start"] + a, 2), None if b is None else round(w["start"] + b, 2), t]
                                 for a, b, t in segs[r]]
            w["text_en"] = clean if w["lang"] == "en" else ""

    for k in range(0, len(todo), B):
        run_batch(todo[k:k + B])
    return wins


def translate_windows(wins, device, cfg):
    """Machine-translate the accepted non-English transcripts (unique texts only) and validate the output."""
    idx = [i for i, w in enumerate(wins) if w.get("text") and w.get("lang") != "en"]
    if not idx or not cfg["translate"]:
        return
    uniq = list(dict.fromkeys(wins[i]["text"] for i in idx))
    mt = get_mt(device)
    out = dict(zip(uniq, mt(uniq, cfg["mt_batch_size"], cfg["mt_max_text_tokens"], cfg["mt_max_new_tokens"])))
    for i in idx:
        w = wins[i]
        raw = out[w["text"]]
        en, why = check_translation(w["text"], raw, cfg)
        w["text_en"] = en
        w["mt_rejected"] = why
        if raw != en:
            w["raw_text_en"] = raw


def _window_feature_rows(wins, models, cfg):
    """Per-window numeric features (rows aligned with ``wins``) and toxicity added into the dicts."""
    n = len(wins)
    rows = np.zeros((n, len(COLUMNS)), np.float32)
    en_idx = [i for i, w in enumerate(wins) if w.get("text") and w.get("text_en")]
    mt_idx = [i for i, w in enumerate(wins) if w.get("text") and w.get("lang") in MTOX_LANGS]
    tox = models["tox"]([wins[i]["text_en"] for i in en_idx], cfg["tox_batch_size"], cfg["tox_max_tokens"]) \
        if en_idx else np.zeros((0, 7), np.float32)
    mtox = models["mtox"]([wins[i]["text"] for i in mt_idx], cfg["tox_batch_size"], cfg["tox_max_tokens"]) \
        if mt_idx else np.zeros((0, 7), np.float32)
    tox_of = {i: tox[j] for j, i in enumerate(en_idx)}
    mtox_of = {i: mtox[j] for j, i in enumerate(mt_idx)}
    for i, w in enumerate(wins):
        r = rows[i]
        if not w["asr_ran"]:
            continue
        r[COL["asr_ran"]] = 1.0
        r[COL["no_speech_prob"]] = w["no_speech_prob"]
        r[COL["lang_en_prob"]] = w["lang_en_prob"]
        if not w.get("text"):
            continue
        text, text_en = w["text"], w.get("text_en", "")
        nw = len(text_en.split()) if (w["lang"] in NO_SPACE_LANGS and text_en) else _n_units(text)
        dur = max(1e-3, w["end"] - w["start"])
        r[COL["has_text"]] = 1.0
        r[COL["asr_conf"]] = float(np.exp(w["avg_logprob"]))
        r[COL["non_english"]] = float(w["lang"] != "en")
        r[COL["words_per_sec"]] = nw / dur
        sp = w["vad_speech_sec"]
        r[COL["speech_rate"]] = min(nw / sp, cfg["max_speech_rate"]) if sp >= 0.5 else 0.0
        r[COL["repeats"]] = min(float(w.get("repeats", 0)), MAX_REPEATS_COL)
        if i in tox_of:
            r[COL["en_text"]] = 1.0
            for c, v in zip(TOX_CLASSES, tox_of[i]):
                r[COL["tox_" + c]] = v
            w["tox"] = {c: round(float(v), 4) for c, v in zip(TOX_CLASSES, tox_of[i])}
            cues = text_cues(text_en)
            for k in ("threat", "command", "challenge", "profanity", "deescalate"):
                r[COL["kw_" + k]] = cues[k]
            for k in ("you_rate", "excl_rate", "question_rate"):
                r[COL[k]] = cues[k]
            w["kw"] = {k: cues[k] for k in ("threat", "command", "challenge", "profanity", "deescalate")}
        if i in mtox_of:
            m = mtox_of[i]
            r[COL["mtox_lang_ok"]] = 1.0
            r[COL["mtox_toxicity"]] = m[TOX_CLASSES.index("toxicity")]
            r[COL["mtox_insult"]] = m[TOX_CLASSES.index("insult")]
            r[COL["mtox_threat"]] = m[TOX_CLASSES.index("threat")]
            w["mtox"] = {c: round(float(v), 4) for c, v in zip(TOX_CLASSES, m)}
    return rows


def _tick_features(t_sec, wins, rows, vad_p, cfg):
    T = len(t_sec)
    feat = np.zeros((T, len(COLUMNS)), np.float32)
    W, R, thr = cfg["window_sec"], cfg["roll_sec"], cfg["vad_threshold"]
    ends = np.array([w["end"] for w in wins]) if wins else np.zeros(0)
    fs = VAD_FRAME / SR
    speech = (vad_p > thr).astype(np.float32)
    csum = np.concatenate([[0.0], np.cumsum(speech)])

    def frac(a, b):
        i0 = int(np.ceil(max(0.0, a) / fs - 1e-9))
        i1 = min(int(np.floor(b / fs + 1e-9)), len(speech))
        return float(csum[i1] - csum[i0]) / max(1, i1 - i0) if i1 > i0 else 0.0

    for k, t in enumerate(t_sec):
        t = float(t)
        feat[k, COL["vad_frac_1s"]] = frac(t - 1, t)
        feat[k, COL["vad_frac_win"]] = frac(t - W, t)
        feat[k, COL["vad_frac_30s"]] = frac(t - R, t)
        n_le = int(np.searchsorted(ends, t + 1e-6, side="right"))
        if n_le == 0:
            continue
        cur = n_le - 1
        keep = feat[k, :3].copy()
        feat[k] = rows[cur]
        feat[k, :3] = keep
        lo = int(np.searchsorted(ends, t - R + 1e-6, side="right"))
        rr = rows[lo:n_le]
        feat[k, COL["roll_tox_max"]] = rr[:, COL["tox_toxicity"]].max()
        feat[k, COL["roll_insult_max"]] = rr[:, COL["tox_insult"]].max()
        feat[k, COL["roll_threat_max"]] = rr[:, COL["tox_threat"]].max()
        feat[k, COL["roll_mtox_max"]] = rr[:, COL["mtox_toxicity"]].max()
        feat[k, COL["roll_kw_hostile_max"]] = (rr[:, COL["kw_threat"]] + rr[:, COL["kw_command"]]
                                               + rr[:, COL["kw_challenge"]]).max()
        feat[k, COL["roll_words_per_sec"]] = rr[:, COL["words_per_sec"]].mean()
    return feat


# ---------------------------------------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------------------------------------
def transcript_path(video_id):
    """Sidecar transcript JSON for ``video_id`` (``<source>/<stem>``), next to the feature npz."""
    return io.feature_path(EXTRACTOR, video_id)[:-len(".npz")] + ".transcripts.json"


def video_duration(video_path):
    """Duration as defined by the shared decoder (num_frames / avg_fps), so tick counts match other streams.

    Falls back to the ffprobe container duration if decord cannot open the file (same rule as audio_dsp.py).
    Returns None when neither works.
    """
    try:
        return io.Video(video_path, num_threads=1).info.duration_sec
    except Exception:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                              video_path], capture_output=True, text=True)
        try:
            return float(out.stdout.strip())
        except ValueError:
            return None


def _check_cfg(cfg):
    if not 0 < cfg["window_sec"] <= 30:
        raise ValueError(f"window_sec must be in (0, 30] (Whisper input is 30 s), got {cfg['window_sec']}")
    if cfg["stride_sec"] <= 0 or cfg["roll_sec"] <= 0:
        raise ValueError("stride_sec and roll_sec must be positive")


def _meta(cfg, D, extra):
    m = dict(extractor=EXTRACTOR, version=VERSION, model_id=WHISPER_ID, mt_model_id=MT_ID,
             toxicity_models=["detoxify/unbiased (unitary toxic_debiased-c7548aa0)",
                              "detoxify/multilingual (unitary multilingual_debiased-0b549669)"],
             vad="silero-vad 6.2.x (onnx, streaming)", window_sec=int(cfg["window_sec"] + cfg["roll_sec"]),
             history=("full: the Silero VAD recurrent state runs from the start of the audio track (it feeds the "
                      "vad_* columns and the ASR gate); every other column and valid use audio in "
                      "[t - window_sec, t]"),
             asr_window_sec=cfg["window_sec"], stride_sec=cfg["stride_sec"], valid_window_sec=cfg["window_sec"],
             D=D, columns=COLUMNS, cfg={k: v for k, v in cfg.items()})
    m.update(extra)
    m["video"] = dict(duration_sec=m.get("duration_sec"))  # read by run_extract.py for its speed log
    return m


def extract(video_path, device="cuda:1", transcript_path=None, duration_sec=None, **cfg):
    """Speech-content features for one video.

    Returns ``(t_sec [T] float32, feat [T, D] float16, valid [T] bool, meta dict)`` following
    ``anticipation/features/io.py``. If ``transcript_path`` is given the window transcripts are written there
    as JSON (atomically); otherwise they are embedded in ``meta["transcripts"]``.
    ``duration_sec`` overrides the decord-based duration (only for tests).
    Raises if the file is missing, its duration cannot be read, or its audio stream cannot be decoded, so the
    runner logs a failure and retries instead of saving an empty result.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    _check_cfg(cfg)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)
    timing = {}
    t0 = time.time()
    dur = video_duration(video_path) if duration_sec is None else float(duration_sec)
    if dur is None or not np.isfinite(dur) or dur <= 0:
        raise RuntimeError(f"cannot determine the duration of {video_path}")
    t_sec = io.tick_times(dur)
    T, D = len(t_sec), len(COLUMNS)
    wav = io.load_audio(video_path, sr=SR)
    if wav is None and io.has_audio(video_path):
        raise RuntimeError(f"audio stream present but ffmpeg could not decode it: {video_path}")
    timing["load_audio"] = round(time.time() - t0, 3)
    audio_sec = 0.0 if wav is None else len(wav) / SR
    S = cfg["stride_sec"]
    silent = io.is_digital_silence(wav)
    valid = np.zeros(T, bool) if silent else tick_valid(wav, t_sec, cfg["window_sec"], S, cfg["min_rms_db"])
    if not valid.any():
        # identical to the full pipeline's output (every row is zeroed when invalid), without loading models
        reason = ("no_audio" if wav is None or wav.size == 0 else "digital_silence" if silent
                  else "near_silence" if T else "too_short")
        meta = _meta(cfg, D, dict(audio=reason, audio_sec=round(audio_sec, 3), duration_sec=round(dur, 3),
                                  timing=timing, summary_noncausal=dict(n_windows=0, n_asr=0, n_text=0,
                                                                        n_translated=0, languages={})))
        _emit_transcripts(dict(video_path=video_path, audio=reason, windows=[]), meta, transcript_path)
        return t_sec, np.zeros((T, D), np.float16), valid, meta

    models = get_models(device)
    t1 = time.time()
    vad_p = vad_probs(models["vad"], wav)
    timing["vad"] = round(time.time() - t1, 3)

    last = min(dur, audio_sec)
    ends = np.arange(S, np.floor(last + 1e-6) + 1e-6, S)
    t1 = time.time()
    wins = transcribe_windows(wav, vad_p, ends, models, cfg)
    timing["asr"] = round(time.time() - t1, 3)
    t1 = time.time()
    translate_windows(wins, device, cfg)
    timing["mt"] = round(time.time() - t1, 3)
    t1 = time.time()
    rows = _window_feature_rows(wins, models, cfg)
    timing["text"] = round(time.time() - t1, 3)
    feat = _tick_features(t_sec, wins, rows, vad_p, cfg)
    feat[~valid] = 0.0
    timing["total"] = round(time.time() - t0, 3)

    langs = {}
    for w in wins:
        if w.get("text"):
            langs[w["lang"]] = langs.get(w["lang"], 0) + 1
    summary = dict(  # whole-video aggregates: diagnostics only, NOT causal, never use them as features
        n_windows=len(wins), n_asr=int(sum(w["asr_ran"] for w in wins)),
        n_text=int(sum(bool(w.get("text")) for w in wins)),
        n_translated=int(sum(bool(w.get("text_en")) and w.get("lang") != "en" for w in wins)),
        n_mt_rejected=int(sum(bool(w.get("mt_rejected")) for w in wins)),
        languages=langs, vad_speech_frac=round(float((vad_p > cfg["vad_threshold"]).mean()), 4),
        n_valid=int(valid.sum()))
    meta = _meta(cfg, D, dict(audio="ok", audio_sec=round(audio_sec, 3), duration_sec=round(dur, 3), timing=timing,
                              summary_noncausal=summary))
    doc = dict(video_path=video_path, audio="ok", windows=[_window_record(w) for w in wins if w["asr_ran"]])
    _emit_transcripts(doc, meta, transcript_path)
    return t_sec, feat.astype(np.float16), valid, meta


def _window_record(w):
    keys = ["start", "end", "vad_speech_sec", "vad_max_prob", "gate", "lang", "lang_raw", "lang_prior", "lang_prob",
            "no_speech_prob", "avg_logprob", "compression_ratio", "text", "text_en", "segments", "rejected",
            "raw_text", "repeats", "mt_rejected", "raw_text_en", "tox", "mtox", "kw"]
    rec = {k: w[k] for k in keys if k in w}
    if rec.get("raw_text") == rec.get("text"):
        rec.pop("raw_text", None)
    if not rec.get("mt_rejected"):
        rec.pop("mt_rejected", None)
    return rec


def _emit_transcripts(doc, meta, path):
    doc = dict(extractor=EXTRACTOR, version=VERSION, model_id=WHISPER_ID, mt_model_id=MT_ID,
               asr_window_sec=meta["asr_window_sec"], stride_sec=meta["stride_sec"],
               duration_sec=meta.get("duration_sec"), audio_sec=meta.get("audio_sec"), **doc)
    if path is None:
        meta["transcripts"] = doc["windows"]
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    meta["transcript_path"] = path


def load_transcripts(video_id):
    with open(transcript_path(video_id), encoding="utf-8") as f:
        return json.load(f)


def windows_until(doc_or_windows, t):
    """Transcript windows a model may use at tick ``t`` (causal): those with ``end <= t``.

    Accepts the sidecar dict (``load_transcripts``) or a window list (``meta["transcripts"]``). Every prompt or
    feature built from transcripts at tick t must go through this filter; the sidecar holds the whole video.
    """
    wins = doc_or_windows["windows"] if isinstance(doc_or_windows, dict) else doc_or_windows
    return [w for w in wins if w["end"] <= t + 1e-6]


# ---------------------------------------------------------------------------------------------------------
# Self-test (no models): text filters, repeat collapsing, translation checks, keyword precision
# ---------------------------------------------------------------------------------------------------------
_HOSTILE = [
    "I'm going to kill you, you hear me?", "Get the fuck out of here or I'll break your face.",
    "You're dead. You're fucking dead.", "Say that again and I'll stab you.", "I know where you live. Watch your back.",
    "Put the gun down or I swear I'll shoot him.", "Come on then, hit me! Hit me, you coward!",
    "Back up! Back the fuck up!", "Touch her again and you won't walk out of here.", "Stop! Stop resisting!",
    "Get on the ground! Show me your hands!", "What are you looking at?", "You want some?", "Drop the gun!",
    "Shut up! Get out of my face!", "Don't touch me!", "I'll fucking kill him.", "Who do you think you are?",
    "Give me the money!", "Get back! Get back!",
]
_CALM = [
    "Would you like some coffee? I just made a fresh pot.", "Hit me up when you get home, okay?",
    "We're going to hit the road around noon.", "I'm going to break the news to her tomorrow.",
    "Come on, let's go get lunch.", "The meeting is going to end at five.",
    "Stop by the store and pick up some milk, please.", "Drop the kids off at school, I'll pick them up.",
    "You want some cake? There's plenty left.", "I love punk music, it's my favorite.",
    "Shoot me an email with the details.", "This movie is killing me, it's so funny.",
    "Dick Cheney gave a speech today.", "Can you turn around and face the camera?",
    "She was trying to stop her from leaving.", "I'll get back to you tomorrow.", "Let's sit down and talk.",
    "You're done with the form, thanks.", "I'll beat you to it.", "I'll hit you up later.",
    "Police say the suspect tried to stop the car before the crash.", "Can you open the door for me?",
    "What are you going to do this weekend?", "Bring it to the table.", "We need to do something about the leak.",
]


def selftest(verbose=True):
    """Model-free checks of the fixes in version 2. Returns True if all pass (prints failures)."""
    cfg = DEFAULT_CFG
    fails = []

    def chk(cond, msg):
        if not cond:
            fails.append(msg)

    # hallucinations / captions must be rejected under both gates (VAD gate with 1 s of speech, weak gate)
    halluc = ["Thank you.", "Thank you. Thank you.", "Thank you for watching!", "(upbeat music)", "[MUSIC PLAYING]",
              "(dramatic music)", "[Music]", "[BLANK_AUDIO]", "(applause)", "*sighs*", "Музыка", "СПОКОЙНАЯ МУЗЫКА",
              "ДИНАМИЧНАЯ МУЗЫКА", "Спасибо за просмотр!", "Продолжение следует...", "Merci.", "Obrigado.",
              "Gracias.", "謝謝大家", "字幕由Amara.org社区提供", "Bye-bye.", "\u266a I'm gonna kill you \u266a", "Música",
              "¡Hola!", "ПЕЧАЛЬНАЯ МУЗЫКА", "Субтитры сделал DimaTorzok", "ご視聴ありがとうございました", "¶¶"]
    for h in halluc:
        for weak, sp in ((False, 1.0), (True, 0.0)):
            clean, why, _ = filter_text(h, -0.4, 0.05, sp, weak, cfg)
            chk(why is not None, f"hallucination accepted (weak={weak}): {h!r} -> {clean!r}")
    # real speech must be accepted, including shouted repetitions (collapsed, repeats counted)
    for txt, n_min in [("Stop resisting! " * 6, 4), ("Get on the ground! " * 5, 3), ("Stop! " * 8, 6),
                       ("Get out! " * 6, 4), ("Put the gun down! " * 4, 2), ("I'm sorry. Oh, my God.", 0),
                       ("你去死吧 你大吉小好 你去死吧 你去死吧 你去死吧 你去死吧", 2),
                       ("I don't see them. They're all over the place. " + "Go, go, go! " * 12, 10)]:
        clean, why, n_rep = filter_text(txt, -0.3, 0.05, 3.0, False, cfg)
        chk(why is None and n_rep >= n_min, f"real speech rejected or not collapsed: {txt[:40]!r} -> {why}, {n_rep}")
    clean, why, _ = filter_text("You all get out! Get out of here!", -0.3, 0.1, 0.3, True, cfg)
    chk(why is None, f"weak-gate shouting rejected: {why}")
    clean, why, _ = filter_text("*sighs* [speech removed]", -0.3, 0.05, 3.0, False, cfg)
    chk(clean == "[speech removed]", f"censored word or starred caption mishandled: {clean!r}")
    txt = "[speech removed] [speech removed] *laughs*"
    clean2, _, _ = filter_text(txt, -0.3, 0.05, 3.0, False, cfg)
    chk(clean2 == txt[:-9], f"censored words mangled: {clean2!r}")
    chk(text_cues(clean)["profanity"] == 1, "censored profanity not counted")
    clean, why, _ = filter_text("請 請 嗯 嗯", -0.3, 0.1, 0.0, True, cfg, 0.82)
    chk(why is not None, "weak-gate 2-word output accepted")
    for txt, pmax, ok in [("Hold it.", 0.976, True), ("nine", 0.33, False), ("all right", 0.61, False),
                          ("No! No! No! No! No! No!", 0.68, True), ("过来", 0.96, True)]:
        clean, why, _ = filter_text(txt, -0.3, 0.05, 0.4, True, cfg, pmax)
        chk((why is None) == ok, f"weak-gate short rule: {txt!r} (max VAD {pmax}) -> {why}")
    chk(collapse_repeats("Stop! Stop! Stop! Stop!")[0] == "Stop! Stop!", "collapse_repeats text")
    # translation checks
    for native, en, ok in [("O que é isso? A polícia!", "What is that? The police!", True),
                           ("我操 你去死", "I'm sorry, but I cannot translate that type of language.", False),
                           ("你还敢打我 来吧 来吧", "你还敢打我 来吧 来吧", False), ("Музыка", "The", False),
                           ("滚 都整理起来 走", "Get the fuck out,整理好 走", True),
                           ("니 꾸이니", "Ni guini maigreنت", True), ("Olha o gás!", "", False),
                           ("Olha o gás!", "Look at the gas! I'll kill you! I'll kill you! Give me the money! "
                                           "Give me all of it now or you die, motherfucker!", False)]:
        _, why = check_translation(native, en, cfg)
        chk((why is None) == ok, f"check_translation({en[:40]!r}) -> {why}")
    # keyword precision / recall on the review's sentences
    hostile = lambda s: sum(text_cues(s)[k] for k in ("threat", "command", "challenge"))  # noqa: E731
    fp = [s for s in _CALM if hostile(s) > 0]
    fn = [s for s in _HOSTILE if hostile(s) == 0]
    chk(not fp, f"keyword false positives: {fp}")
    chk(len(fn) <= 1, f"keyword misses: {fn}")
    chk(text_cues("Dick Cheney gave a speech.")["profanity"] == 0, "profanity on a name")
    if verbose:
        print(f"selftest: {len(fails)} failure(s)")
        for f in fails:
            print("  FAIL", f)
    return not fails


# ---------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--videos", nargs="*", default=[], help="video files")
    ap.add_argument("--list", help="text file with one video path per line")
    ap.add_argument("--source", help="source tag of the video id: xd, ucf, yt (curated), ubi, ...")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--window-sec", type=float, default=DEFAULT_CFG["window_sec"])
    ap.add_argument("--stride-sec", type=float, default=DEFAULT_CFG["stride_sec"])
    ap.add_argument("--batch-size", type=int, default=DEFAULT_CFG["batch_size"])
    ap.add_argument("--feature-root", default="", help="write under this root instead of io.FEATURE_ROOT (tests)")
    ap.add_argument("--selftest", action="store_true", help="run the model-free self-test and exit")
    args = ap.parse_args(argv)
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.source:
        ap.error("--source is required")
    if args.feature_root:
        io.FEATURE_ROOT = os.path.abspath(args.feature_root)  # io.feature_path reads the module global
    paths = list(args.videos)
    if args.list:
        with open(args.list) as f:
            paths += [ln.strip() for ln in f if ln.strip()]
    cfg = dict(window_sec=args.window_sec, stride_sec=args.stride_sec, batch_size=args.batch_size)
    tot_audio = tot_time = 0.0
    for p in paths:
        vid = f"{args.source}/{os.path.splitext(os.path.basename(p))[0]}"
        if not args.overwrite and os.path.exists(io.feature_path(EXTRACTOR, vid)):
            print(f"skip {vid}")
            continue
        t0 = time.time()
        try:
            t_sec, feat, valid, meta = extract(p, args.device, transcript_path=transcript_path(vid), **cfg)
        except Exception as e:  # keep going on bad files; they are retried on the next run
            print(f"FAIL {vid}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        io.save_features(EXTRACTOR, vid, t_sec, feat, valid, meta)
        dt = time.time() - t0
        tot_audio += meta.get("audio_sec", 0.0)
        tot_time += dt
        s = meta["summary_noncausal"]
        print(f"OK {vid} T={len(t_sec)} valid={int(valid.sum())} audio={meta.get('audio')} "
              f"asr={s.get('n_asr')}/{s.get('n_windows')} text={s.get('n_text')} "
              f"langs={s.get('languages')} {dt:.1f}s", flush=True)
    if tot_time:
        print(f"throughput: {tot_audio / tot_time:.1f} s audio per wall-second ({tot_audio:.0f} s in {tot_time:.0f} s)")


if __name__ == "__main__":
    main()
