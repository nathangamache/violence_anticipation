"""Raw annotation parsers for every source in the VA30 catalog.

All parsers return plain Python structures and never touch video files. Units are stated per function:
the official XD-Violence and UCF-Crime files store FRAME numbers (clip-relative, at the clip fps), HIVAU-70k
stores SECONDS. ``anticipation.data.catalog`` converts everything to seconds with the probed fps of each file.

Sources and where their ground truth comes from:

    xd  XD-Violence. Test clips: ``raw_videos/xd_violence/annotations.txt`` (official, frames at 24 fps, 14 ids
        carry a stray ``.mp4`` suffix). Train clips (when downloaded to ``raw_videos/xd_violence_train/``):
        HIVAU-70k ``xd_database_train.json``. Video-level classes come from the ``_label_`` codes in the name.
    ucf UCF-Crime. Test anomalies: ``Temporal_Anomaly_Annotation_for_Testing_Videos.txt`` (official, frames at
        30 fps, at most 2 segments, ``-1`` = none). Train anomalies: HIVAU-70k ``ucf_database_train.json``.
    ubi UBI-Fights (when downloaded to ``raw_videos/ubi_fights/``): per-video CSV with one 0/1 value per frame.
    yt  Curated YouTube set (``anticipation_curation/candidates.json``): LSTM pseudo-labels only, hold-out.

HIVAU-70k gotcha: normal videos also carry "events" (random clips used for captioning). Only entries with a
non-empty ``label`` list describe real anomalies; ``hivau_entry_events`` enforces this.
"""

import glob
import json
import os
import re

DATA_ROOT = os.environ.get("VA_DATA_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"))
RAW_ROOT = os.path.join(DATA_ROOT, "raw_videos")
ANNOT_ROOT = os.path.join(DATA_ROOT, "annotations")
CACHE_DIR = os.path.join(ANNOT_ROOT, "_cache")

XD_TEST_DIR = os.path.join(RAW_ROOT, "xd_violence", "videos")
XD_TEST_ANN = os.path.join(RAW_ROOT, "xd_violence", "annotations.txt")
XD_TRAIN_DIR = os.path.join(RAW_ROOT, "xd_violence_train")
XD_TRAIN_VERIFIED = os.path.join(XD_TRAIN_DIR, "verified.jsonl")
UCF_DIR = os.path.join(RAW_ROOT, "ucf_crime")
UCF_TEST_ANN = os.path.join(UCF_DIR, "Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
UCF_TRAIN_LIST = os.path.join(UCF_DIR, "Anomaly_Train.txt")
UBI_DIR = os.path.join(RAW_ROOT, "ubi_fights")  # layout: videos/{fight,normal}/*.mp4, annotation/*.csv
CURATED_DIR = os.path.join(DATA_ROOT, "anticipation_curation")
CURATED_JSON = os.path.join(CURATED_DIR, "candidates.json")

HIVAU_DIR = os.path.join(ANNOT_ROOT, "hivau")
HIVAU_RAW_DIR = os.path.join(HIVAU_DIR, "raw_annotations")
HIVAU_ONSETS = os.path.join(HIVAU_DIR, "hivau_onsets.json")
HIVAU_RAW_FILES = ("ucf_database_train.json", "ucf_database_test.json",
                   "xd_database_train.json", "xd_database_test.json")
UCA_DIR = os.path.join(ANNOT_ROOT, "uca")

VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".webm", ".mov", ".m4v")

NORMAL = "normal"

# XD-Violence label codes (``..._label_B1-B2-0``). There is no B3.
XD_CODES = {
    "A": NORMAL, "B1": "fighting", "B2": "shooting", "B4": "riot", "B5": "abuse",
    "B6": "car_accident", "G": "explosion",
}

# UCF-Crime folder / annotation category -> normalized class name.
UCF_CLASSES = {
    "Abuse": "abuse", "Arrest": "arrest", "Arson": "arson", "Assault": "assault", "Burglary": "burglary",
    "Explosion": "explosion", "Fighting": "fighting", "RoadAccidents": "road_accident", "Robbery": "robbery",
    "Shooting": "shooting", "Shoplifting": "shoplifting", "Stealing": "stealing", "Vandalism": "vandalism",
    "Normal": NORMAL,
}

# HIVAU-70k ``label`` strings -> normalized class name (XD uses words, UCF uses the folder names).
HIVAU_LABELS = {
    "Fighting": "fighting", "Shooting": "shooting", "Riot": "riot", "Abuse": "abuse",
    "Car accident": "car_accident", "Explosion": "explosion",
    **{k: v for k, v in UCF_CLASSES.items() if k != "Normal"},
}

# Default anticipation targets. Everything else that is an anomaly (car accidents, explosions, arson, theft,
# shoplifting, burglary, vandalism, road accidents) is neither a violent event nor a clean negative.
VIOLENT_CLASSES = frozenset({"fighting", "shooting", "riot", "abuse", "assault", "robbery", "arrest"})
INTERPERSONAL_CLASSES = frozenset({"fighting", "abuse", "assault", "riot", "robbery", "arrest"})

# XD-Violence movie franchises / series (group names = text before ``__#``). Films of one franchise share actors,
# grading and style, so folds_v2 keeps them in one split group. Hand-made from the 102 XD movie groups.
XD_FRANCHISES = {
    "bad_boys": ("Bad.Boys.1995", "Bad.Boys.II.2003"),
    "before_trilogy": ("Before.Sunrise.1995", "Before.Sunset.2004"),
    "city_of_god": ("City.of.God.2002", "City.Of.Men.2007"),
    "deadpool": ("Deadpool.2016", "Deadpool.2.2018"),
    "fast_furious": ("The.Fast.and.the.Furious.2001", "Fast.Furious.2009", "Fast.Five.2011", "Fast.Furious.6.2013"),
    "ip_man": ("Ip.Man.2008", "IP.Man.2.2010", "Ip.Man.3.2015"),
    "bourne": ("The.Bourne.Identity.2002", "The.Bourne.Supremacy.2004", "The.Bourne.Ultimatum.2007",
               "The.Bourne.Legacy.2012", "Jason.Bourne.2016"),
    "james_bond": ("GoldenEye.1995", "The.World.Is.Not.Enough.1999", "Casino.Royale.2006",
                   "Quantum.Of.Solace.2008", "Skyfall.2012", "Spectre.2015"),
    "kill_bill": ("Kill.Bill.Vol.1.2003", "Kill.Bill.Vol.2.2004"),
    "kingsman": ("Kingsman.The.Secret.Service.2014", "Kingsman.The.Golden.Circle.2017"),
    "love_death_robots": tuple(f"Love.Death.and.Robots.S01E{e:02d}" for e in range(1, 19)),
    "mission_impossible": ("Mission.Impossible.II.2000", "Mission.Impossible.III.2006",
                           "Mission.Impossible.Ghost.Protocol.2011", "Mission.Impossible.V.Rogue.Nation.2015",
                           "Mission.Impossible.Fallout.2018"),
    "rush_hour": ("Rush.Hour.1998.BluRay", "Rush.Hour.3.2007.BluRay"),
    "taken": ("Taken.Extended.Cut.2008", "Taken.2.UNRATED.EXTENDED.2012", "Taken.3.2014"),
    "tropa_de_elite": ("Tropa.de.Elite.2007", "Tropa.de.Elite.2.2010"),
    "young_and_dangerous": ("Young.And.Dangerous.I.1996", "Young.And.Dangerous.II.1996",
                            "Young.And.Dangerous.III.1996", "Young.And.Dangerous.IV.1997"),
}
XD_FRANCHISE_OF = {g: k for k, gs in XD_FRANCHISES.items() for g in gs}


# HIVAU per-event judgements are free LLM text ("The anomaly exists, and the specific name of the anomaly is
# Riot.", "There is no anomaly event (Car accident) in the video."). Keyword classes, checked in this order.
_JUDGE_NONE = re.compile(r"\b(no anomal|there is no anomal|not an anomal|no abnormal|no specific anomal)", re.I)
_JUDGE_VIOLENT = re.compile(
    r"fight|riot|shoot|shot|gun|firearm|assault|robber|arrest|abuse|violen|altercation|attack|brawl|stab|"
    r"beat|punch|kick|combat|melee|skirmish|hostage|kidnap|murder|kill|clash|aggress|struggl|protest", re.I)
_JUDGE_OTHER = re.compile(
    r"explo|blast|accident|crash|collision|fire\b|arson|burn|flame|burglar|steal|theft|shoplift|vandal|"
    r"vehicle|disaster", re.I)


def classify_judgement(judgement, description=""):
    """Class of one HIVAU event from its LLM judgement, with the description as a fallback.

    Returns ``"violent"`` (a violent keyword in a positive judgement), ``"other"`` (only non-violent anomaly
    keywords in a positive judgement, e.g. "Explosion and Car Accident"), ``"none"`` (the judgement denies an
    anomaly and the description names nothing violent), or ``"unknown"``. When the judgement denies the anomaly or
    names no known class, the description decides: a violent keyword there gives ``"violent"``, a non-violent one
    ``"other"``. Used only to split the segments of mixed-label XD clips (``B2-G``, ``B6-B1`` ...), whose clip
    label says that both kinds occur but not which segment is which; see ``catalog.build_rows``.
    """
    j = str(judgement or "")
    if not _JUDGE_NONE.search(j):
        if _JUDGE_VIOLENT.search(j):
            return "violent"
        if _JUDGE_OTHER.search(j):
            return "other"
    d = str(description or "")
    if _JUDGE_VIOLENT.search(d):
        return "violent"
    if _JUDGE_OTHER.search(d):
        return "other"
    return "none" if _JUDGE_NONE.search(j) else "unknown"


# ---------------------------------------------------------------------------------------------------------
# XD-Violence
# ---------------------------------------------------------------------------------------------------------

_XD_LABEL_RE = re.compile(r"_label_([A-Za-z0-9\-]+)$")


def parse_xd_name(stem):
    """Split an XD-Violence clip name into its group, label codes, classes and kind.

    ``Bad.Boys.1995__#00-26-51_00-27-53_label_B2-0-0`` -> group ``Bad.Boys.1995``, codes ``['B2']``,
    classes ``['shooting']``, kind ``movie``. YouTube clips start with ``v=`` (kind ``youtube``).
    The ``#HH-MM-SS`` part is the position inside the source video, NOT inside the clip.
    """
    group = stem.split("__#")[0]
    m = _XD_LABEL_RE.search(stem)
    codes = []
    if m:
        codes = [c for c in m.group(1).split("-") if c and c != "0"]
    classes = sorted({XD_CODES.get(c, "unknown_" + c) for c in codes}) or ["unknown"]
    kind = "youtube" if stem.startswith("v=") else "movie"
    return {"group": group, "codes": codes, "category": "-".join(codes) if codes else "unknown",
            "classes": classes, "kind": kind}


_XD_SPAN_RE = re.compile(r"__#(\d{1,2})-(\d{2})-(\d{2})_(\d{1,2})-(\d{2})-(\d{2})")


def parse_xd_source_span(stem):
    """``(start_sec, end_sec)`` of an XD movie clip inside its source film from ``__#HH-MM-SS_HH-MM-SS``, or None
    (YouTube clips carry ``__#1`` instead). Clip tick ``t`` is at ``start_sec + t`` in the source."""
    m = _XD_SPAN_RE.search(stem)
    if not m:
        return None
    h1, m1, s1, h2, m2, s2 = (int(x) for x in m.groups())
    return float(3600 * h1 + 60 * m1 + s1), float(3600 * h2 + 60 * m2 + s2)


def read_xd_official(path=XD_TEST_ANN):
    """Official XD-Violence test annotations: ``{stem: [[start_frame, end_frame], ...]}``.

    Frames are clip-relative at the clip fps (24 for every clip). Lines hold an even number of frame
    numbers after the id; 14 ids end with ``.mp4`` which is stripped here.
    """
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            vid = p[0][:-4] if p[0].endswith(".mp4") else p[0]
            fr = [int(x) for x in p[1:]]
            if len(fr) % 2:
                raise ValueError(f"odd number of frame indices for {vid} in {path}")
            out[vid] = [[fr[i], fr[i + 1]] for i in range(0, len(fr), 2)]
    return out


def read_verified_list(path):
    """``{file name: {"size", "sha256", ...}}`` from a downloader's verified.jsonl, or None if absent."""
    if not os.path.exists(path):
        return None
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line being written right now
            if rec.get("name"):
                out[rec["name"]] = rec
    return out


# ---------------------------------------------------------------------------------------------------------
# UCF-Crime
# ---------------------------------------------------------------------------------------------------------

def read_ucf_official(path=UCF_TEST_ANN):
    """Official UCF-Crime temporal annotations of the TEST split.

    Returns ``{stem: {"category": "Abuse", "frames": [[s, e], ...]}}`` for all 290 test videos; normal videos
    get ``frames == []``. Frames are at the video fps (30). ``-1`` marks a missing second instance.
    """
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 4:
                continue
            vid = p[0][:-4] if p[0].endswith(".mp4") else p[0]
            nums = [int(x) for x in p[2:]]
            frames = []
            for i in range(0, len(nums) - 1, 2):
                if nums[i] >= 0 and nums[i + 1] >= 0:
                    frames.append([nums[i], nums[i + 1]])
            out[vid] = {"category": p[1], "frames": frames}
    return out


def read_ucf_train_list(path=UCF_TRAIN_LIST):
    """Stems in the official UCF-Crime anomaly-detection TRAIN split (810 anomaly + 800 normal videos)."""
    out = set()
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(os.path.splitext(os.path.basename(line))[0])
    return out


# ---------------------------------------------------------------------------------------------------------
# HIVAU-70k (Holmes-VAU, CVPR 2025)
# ---------------------------------------------------------------------------------------------------------

def read_hivau_raw(path):
    with open(path) as f:
        return json.load(f)


def hivau_entry_events(entry):
    """``(classes, events_sec)`` of one HIVAU raw entry, or ``(None, [])`` for normal videos.

    Normal videos have an empty ``label`` list but still carry random "events"; those are discarded.
    ``events`` are ``[[start_sec, end_sec], ...]`` computed by HIVAU from their frame annotations.
    """
    labels = entry.get("label") or []
    if not labels:
        return None, []
    classes = sorted({HIVAU_LABELS.get(lab, "unknown_" + str(lab)) for lab in labels})
    events = [[float(s), float(e)] for s, e in (entry.get("events") or [])]
    return classes, events


def hivau_event_kinds(entry):
    """Per-event ``classify_judgement`` result aligned 1:1 with ``entry["events"]`` (``"unknown"`` when HIVAU
    has no per-event summary for an event)."""
    ev = entry.get("events") or []
    split = entry.get("events_summary_split") or []
    out = []
    for i in range(len(ev)):
        s = split[i] if i < len(split) else None
        if isinstance(s, dict):
            out.append(classify_judgement(s.get("judgement"), s.get("description")))
        elif isinstance(s, str):
            out.append(classify_judgement(s))
        else:
            out.append("unknown")
    return out


def load_hivau_onsets(path=HIVAU_ONSETS):
    """Processed HIVAU onsets written by ``setup/build_hivau_onsets.py``; ``None`` if not built yet.

    Layout: ``{"meta", "xd": {stem: rec}, "ucf": {stem: rec}, "normals": {"xd": [...], "ucf": [...]},
    "split_of": {"xd": {stem: "train"|"test"}, "ucf": {...}}, "xd_cctv_hits": {stem: int}}``
    with ``rec = {"split", "classes", "labels", "events_sec", "event_kinds", "fps", "n_frames"}``; ``event_kinds``
    is ``hivau_event_kinds`` (violent / other / none / unknown per event, from the LLM judgement).
    """
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


HIVAU_CCTV_WORDS = ("surveillance", "cctv", "security camera", "monitoring footage", "monitor screen")


def hivau_cctv_hits(entry):
    """How often HIVAU's captions of a clip mention surveillance / CCTV (domain hint for XD YouTube clips)."""
    txt = " ".join([str(entry.get("video_summary") or "")]
                   + [str(c) for caps in (entry.get("clips_caption") or []) for c in (caps or [])]).lower()
    return sum(txt.count(w) for w in HIVAU_CCTV_WORDS)


def build_hivau_onsets_from_raw(raw_dir=HIVAU_RAW_DIR):
    """Parse the four HIVAU raw files into the processed layout (see ``load_hivau_onsets``)."""
    out = {"xd": {}, "ucf": {}, "normals": {"xd": [], "ucf": []}, "xd_cctv_hits": {},
           "split_of": {"xd": {}, "ucf": {}}, "meta": {"raw_dir": raw_dir, "files": {}}}
    for fname in HIVAU_RAW_FILES:
        path = os.path.join(raw_dir, fname)
        if not os.path.exists(path):
            continue
        src = "xd" if fname.startswith("xd") else "ucf"
        split = "train" if "train" in fname else "test"
        raw = read_hivau_raw(path)
        n_anom = 0
        for vid, entry in raw.items():
            out["split_of"][src][vid] = split
            if src == "xd" and vid.startswith("v="):
                hits = hivau_cctv_hits(entry)
                if hits:
                    out["xd_cctv_hits"][vid] = hits
            classes, events = hivau_entry_events(entry)
            if classes is None:
                out["normals"][src].append(vid)
                continue
            n_anom += 1
            out[src][vid] = {
                "split": split, "classes": classes, "labels": list(entry.get("label") or []),
                "events_sec": events, "event_kinds": hivau_event_kinds(entry),
                "fps": float(entry.get("fps") or 0.0), "n_frames": int(float(entry.get("n_frames") or 0)),
            }
        out["meta"]["files"][fname] = {"entries": len(raw), "anomalous": n_anom, "normal": len(raw) - n_anom}
    out["normals"] = {k: sorted(v) for k, v in out["normals"].items()}
    return out


# ---------------------------------------------------------------------------------------------------------
# UBI-Fights (not downloaded yet; parser written against the documented layout, verify on arrival)
# ---------------------------------------------------------------------------------------------------------

def read_ubi_annotations(root=UBI_DIR):
    """Per-frame UBI-Fights labels: ``{stem: [0/1 per frame]}`` from every ``*.csv`` under an ``annotation*``
    folder whose name matches a video stem.

    The documented UBI-Fights layout is ``annotation/<stem>.csv`` with one 0/1 value per frame, videos under
    ``videos/fight/F_*.mp4`` and ``videos/normal/N_*.mp4``. If the CSVs instead hold frame pairs, the catalog
    falls back to treating them as ``[start, end]`` frame pairs (see ``ubi_segments_frames``).
    """
    out = {}
    if not os.path.isdir(root):
        return out
    for path in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):
        if "annotation" not in os.path.dirname(path).lower():
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            toks = re.split(r"[\s,;]+", f.read().strip())
        vals = []
        for t in toks:
            if not t:
                continue
            try:
                vals.append(float(t))
            except ValueError:
                continue  # header
        out[stem] = vals
    return out


def ubi_segments_frames(values):
    """Convert UBI per-frame 0/1 values into ``[[start_frame, end_frame], ...]`` (inclusive ends)."""
    if not values:
        return []
    if set(values) <= {0.0, 1.0}:
        segs, start = [], None
        for i, v in enumerate(values):
            if v >= 0.5 and start is None:
                start = i
            elif v < 0.5 and start is not None:
                segs.append([start, i - 1])
                start = None
        if start is not None:
            segs.append([start, len(values) - 1])
        return segs
    if len(values) % 2 == 0:  # fallback: frame pairs
        return [[int(values[i]), int(values[i + 1])] for i in range(0, len(values), 2)]
    raise ValueError("unrecognized UBI-Fights annotation format")


def read_ubi_extracted(root=UBI_DIR):
    """Relative paths of UBI files whose extraction finished (``.extracted.jsonl`` written by
    setup/download_ubi_fights.py, CRC-verified), or None if the log does not exist."""
    names = read_verified_list(os.path.join(root, ".extracted.jsonl"))
    return None if names is None else set(names)


def read_ubi_splits(root=UBI_DIR):
    """``{stem: "train"|"test"}`` from any split list under the UBI root whose name contains train/test.

    UBI-Fights ships only ``test_videos.csv`` (one stem per line); callers treat unlisted videos as train when
    a test list exists (see ``catalog.build_rows``).
    """
    out = {}
    if not os.path.isdir(root):
        return out
    for path in glob.glob(os.path.join(root, "**", "*.*"), recursive=True):
        base = os.path.basename(path).lower()
        if not base.endswith((".csv", ".txt")) or "annotation" in os.path.dirname(path).lower():
            continue
        split = "test" if "test" in base else ("train" if "train" in base else None)
        if split is None:
            continue
        with open(path, errors="replace") as f:
            for tok in re.split(r"[\s,;]+", f.read()):
                tok = os.path.splitext(os.path.basename(tok.strip()))[0]
                if tok:
                    out[tok] = split
    return out


# ---------------------------------------------------------------------------------------------------------
# Curated YouTube hold-out (pseudo-labels only)
# ---------------------------------------------------------------------------------------------------------

def read_curated(path=CURATED_JSON):
    """``{video_id: {"title", "pseudo_onset_sec" or None, "url"}}`` for downloaded curated videos.

    The onsets are LSTM pseudo-labels (always >= 59 s by construction), NOT ground truth.
    """
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        cands = json.load(f).get("candidates", [])
    for c in cands:
        vid = c.get("video_id")
        if not vid:
            continue
        onset = None
        if str(c.get("has_violence")) == "True":
            try:
                onset = float(c.get("violence_start_second"))
            except (TypeError, ValueError):
                onset = None
        out[vid] = {"title": c.get("title", ""), "url": c.get("url", ""), "pseudo_onset_sec": onset,
                    "download_status": c.get("download_status")}
    return out
