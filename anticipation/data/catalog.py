"""Build the canonical VA30 video catalog (benchmark/videos.csv) and ground truth (benchmark/events.json).

Usage (from the repo root; re-runnable, probes are cached so re-runs only touch new files):

    .venv/bin/python -m anticipation.data.catalog                 # full build, 16 workers
    .venv/bin/python -m anticipation.data.catalog --workers 8 --no_audio   # skip the audio-silence scan

What it does:
  1. Discovers every video of every source: XD-Violence test (and train, once it appears in
     raw_videos/xd_violence_train/), UCF-Crime (all five folders), UBI-Fights (raw_videos/ubi_fights/, once it
     appears), and the curated YouTube set.
  2. Probes each file once (decord fps / frame count / duration, exactly as ``anticipation.features.io.Video``
     does; a size + head/tail content hash; the audio track decoded with ``io.load_audio`` and tested with
     ``io.is_digital_silence``). Results are cached in annotations/_cache/media_probe.json keyed by path,
     size and mtime.
  3. Deduplicates: same stem in several UCF folders, and byte-identical content under different names
     (UCF-Crime ships 10 such normal-video pairs). One canonical path per unique video; the others are listed
     in the ``aliases`` column.
  4. Attaches ground truth in seconds: official onsets for XD test and UCF test, HIVAU-70k for UCF train and
     XD train (processed file from setup/build_hivau_onsets.py), UBI CSVs, pseudo-labels for curated YouTube.
  5. Writes videos.csv (one row per unique video, sorted by video_id), events.json and catalog_summary.json.

  6. Version 2 additions: mixed-label XD clips split into violent segments and excluded ``other_segments`` with
     HIVAU's per-event judgement; ``mask_spans`` for XD clips adjacent to violence of another clip of the same
     source film (``source_masks``); near-duplicates from ``benchmark/near_duplicates.json`` (identical re-encodes
     -> role ``duplicate``); ``split_group`` = catalog groups merged by XD franchise and near-duplicate pairs;
     eligibility on the 1 Hz clock (``labels.event_table``).

Roles (column ``role``):
    violent         video-level class intersects VIOLENT_CLASSES and trusted onsets exist (official/hivau/ubi)
    clean_negative  XD ``_label_A``, UCF Normal videos, UBI normal videos
    other_anomaly   anomaly classes that are not violence (car accident, explosion, arson, theft, ...)
    unlabelled      violent class but no trusted violent onsets (UCF train anomalies HIVAU does not cover, mixed
                    clips whose segments HIVAU judges all non-violent, UBI fight clips without a positive frame)
    duplicate       identical re-encode of another catalog video (``duplicate_of``); excluded from evaluation
    contaminated    clean clip whose source span overlaps violence annotated in another clip; excluded
    holdout         curated YouTube (pseudo-labels only; future human-labelled hold-out)
Only ``violent`` and ``clean_negative`` enter the default VA30 evaluation.
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from anticipation.data import annotations as A
from anticipation.data.labels import blocker_spans, event_table, normalize_segments

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCHMARK_DIR = os.path.join(REPO_ROOT, "benchmark")
VIDEOS_CSV = os.path.join(BENCHMARK_DIR, "videos.csv")
EVENTS_JSON = os.path.join(BENCHMARK_DIR, "events.json")
SUMMARY_JSON = os.path.join(BENCHMARK_DIR, "catalog_summary.json")
PROBE_CACHE = os.path.join(A.CACHE_DIR, "media_probe.json")


def _log(*args):
    print(*args, flush=True)


CATALOG_VERSION = 2
TRUSTED_ONSET_SOURCES = ("official", "hivau", "ubi")

COLUMNS = [
    "video_id", "source", "path", "duration_sec", "fps", "num_frames", "has_audio", "group", "official_split",
    "category", "is_violent_video", "is_clean_negative", "onset_source", "first_onset_sec", "eligible30",
    "eligible60", "domain",
    # extra columns
    "role", "classes", "interpersonal", "n_segments", "n_eligible30", "n_eligible60", "source_kind",
    "audio_stream", "audio_peak", "width", "height", "aliases", "content_key", "notes",
    # v2 columns
    "n_other_segments", "masked_sec", "franchise", "split_group", "duplicate_of", "first_anomaly_sec",
    "official_annotated",
]

# ---------------------------------------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------------------------------------

UCF_FOLDERS = [  # (folder, priority): lower priority wins when the same video exists in several folders
    ("videos", 0), ("Anomaly-Videos", 1), ("Testing_Normal_Videos", 2),
    ("Normal_Videos_for_Event_Recognition", 3), ("Training-Normal-Videos", 4),
]


def _list_videos(root, recursive=True):
    out = []
    if not os.path.isdir(root):
        return out
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(A.VIDEO_EXTS) and not f.startswith("."):
                out.append(os.path.join(dirpath, f))
        if not recursive:
            break
    return sorted(out)


def discover_files():
    """All candidate video files as dicts ``{source, stem, path, folder, priority}``."""
    files = []

    def add(source, path, folder, prio):
        stem = os.path.splitext(os.path.basename(path))[0]
        files.append({"source": source, "stem": stem, "path": os.path.abspath(path), "folder": folder,
                      "priority": prio})

    for p in _list_videos(A.XD_TEST_DIR, recursive=False):
        add("xd", p, "xd_test", 0)
    verified = A.read_verified_list(A.XD_TRAIN_VERIFIED)
    for p in _list_videos(A.XD_TRAIN_DIR):
        # The XD train downloader writes verified.jsonl (name, size, sha256) when a file is complete; while
        # that list exists, unlisted or size-mismatched files are in-flight downloads and are skipped.
        if verified is not None:
            v = verified.get(os.path.basename(p))
            if v is None or (v.get("size") and os.path.getsize(p) != v["size"]):
                continue
        folder = "xd_test_extra" if "missing_test" in os.path.dirname(p) else "xd_train"
        add("xd", p, folder, 1)
    for folder, prio in UCF_FOLDERS:
        for p in _list_videos(os.path.join(A.UCF_DIR, folder)):
            add("ucf", p, folder, prio)
    ubi_done = A.read_ubi_extracted()
    for p in _list_videos(A.UBI_DIR):
        # Only members the UBI extractor has CRC-verified (``.extracted.jsonl``); others are in flight.
        if ubi_done is not None and os.path.relpath(p, A.UBI_DIR) not in ubi_done:
            continue
        add("ubi", p, "ubi", 0)
    for p in _list_videos(os.path.join(A.CURATED_DIR, "videos"), recursive=False):
        add("yt", p, "curated", 0)
    return files


# ---------------------------------------------------------------------------------------------------------
# Probing (cached)
# ---------------------------------------------------------------------------------------------------------

def content_key(path, chunk=1 << 20):
    """sha1 over size + first and last MiB. Catches byte-identical copies cheaply."""
    size = os.path.getsize(path)
    h = hashlib.sha1(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if size > chunk:
            f.seek(max(chunk, size - chunk))
            h.update(f.read(chunk))
    return h.hexdigest()


def _ffprobe_stream(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=avg_frame_rate,nb_frames,width,height:format=duration", "-of", "json", path],
        capture_output=True, text=True)
    d = json.loads(out.stdout or "{}")
    st = (d.get("streams") or [{}])[0]
    num, _, den = (st.get("avg_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    dur = float((d.get("format") or {}).get("duration") or 0.0)
    nb = int(st.get("nb_frames") or round(dur * fps))
    return fps, nb, st.get("width", 0), st.get("height", 0)


def probe_file(path, audio=True):
    """Media facts of one file (see module docstring). Never raises; errors go to ``error``."""
    from anticipation.features import io as fio

    st = os.stat(path)
    rec = {"path": path, "size": st.st_size, "mtime_ns": st.st_mtime_ns, "error": ""}
    try:
        rec["content_key"] = content_key(path)
    except OSError as e:
        rec["content_key"] = ""
        rec["error"] += f"hash:{e};"
    try:
        import decord

        vr = decord.VideoReader(path, num_threads=1)
        fps = float(vr.get_avg_fps()) or 25.0
        n = len(vr)
        h, w = vr[0].shape[:2] if n else (0, 0)
        rec.update(fps=fps, num_frames=int(n), duration_sec=n / fps, width=int(w), height=int(h),
                   probe="decord")
        del vr
    except Exception as e:  # decord failure: fall back to container metadata
        rec["error"] += f"decord:{type(e).__name__}:{e};"
        try:
            fps, n, w, h = _ffprobe_stream(path)
            fps = fps or 25.0
            rec.update(fps=fps, num_frames=int(n), duration_sec=n / fps, width=int(w), height=int(h),
                       probe="ffprobe")
        except Exception as e2:
            rec.update(fps=0.0, num_frames=0, duration_sec=0.0, width=0, height=0, probe="failed")
            rec["error"] += f"ffprobe:{e2};"
    if audio:
        try:
            stream = fio.has_audio(path)
            peak = 0.0
            silent = True
            if stream:
                wav = fio.load_audio(path)
                silent = fio.is_digital_silence(wav)
                peak = 0.0 if wav is None or wav.size == 0 else float(np.abs(wav).max())
            rec.update(audio_stream=bool(stream), audio_peak=peak, has_audio=bool(stream and not silent),
                       audio_checked=True)
        except Exception as e:
            rec.update(audio_stream=False, audio_peak=0.0, has_audio=False, audio_checked=False)
            rec["error"] += f"audio:{e};"
    else:
        rec.update(audio_checked=False)
    return rec


def _probe_worker(args):
    path, audio = args
    return probe_file(path, audio=audio)


def load_probe_cache(path=PROBE_CACHE):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_probe_cache(cache, path=PROBE_CACHE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, path)


def probe_all(paths, workers=16, audio=True, cache_path=PROBE_CACHE, log=_log):
    """Probe ``paths`` with a process pool, reusing cache entries whose size and mtime still match."""
    cache = load_probe_cache(cache_path)
    todo = []
    for p in paths:
        c = cache.get(p)
        st = os.stat(p)
        stale = (c is None or c.get("size") != st.st_size or c.get("mtime_ns") != st.st_mtime_ns
                 or (audio and not c.get("audio_checked")))
        if stale:
            todo.append(p)
    log(f"[catalog] probe: {len(paths)} files, {len(paths) - len(todo)} cached, {len(todo)} to probe")
    if todo:
        t0 = time.time()
        with Pool(max(1, workers)) as pool:
            for i, rec in enumerate(pool.imap_unordered(_probe_worker, [(p, audio) for p in todo],
                                                        chunksize=4)):
                cache[rec["path"]] = rec
                if (i + 1) % 200 == 0 or i + 1 == len(todo):
                    save_probe_cache(cache, cache_path)
                    log(f"[catalog]   probed {i + 1}/{len(todo)} ({time.time() - t0:.0f} s)")
    return {p: cache[p] for p in paths}


# ---------------------------------------------------------------------------------------------------------
# Dedupe
# ---------------------------------------------------------------------------------------------------------

def dedupe(files, probes, log=_log):
    """Pick one canonical file per unique video.

    Pass 1: same (source, stem) -> keep the lowest folder priority. Pass 2: same (source, content_key)
    under different stems -> keep the lowest (priority, stem). Returns canonical file dicts with ``aliases``.
    """
    by_name = {}
    for f in sorted(files, key=lambda x: (x["source"], x["stem"], x["priority"], x["path"])):
        key = (f["source"], f["stem"])
        if key in by_name:
            keep = by_name[key]
            ck_a, ck_b = probes[keep["path"]].get("content_key"), probes[f["path"]].get("content_key")
            keep["aliases"].append(f["path"])
            if ck_a != ck_b:
                keep["notes"].append(f"same-name file with different content: {f['path']}")
            continue
        by_name[key] = dict(f, aliases=[], notes=[], alias_ids=[])
    by_content = {}
    out = []

    def normal_named(f):
        return int(f["stem"].startswith("Normal_Videos") or f["stem"].endswith("_label_A"))

    # Anomaly-labelled copies win over normal-labelled ones: a clip that is also published as violent must not
    # become a clean negative (XD ships v=qQCTvJq83_s__#1 as both _label_A and _label_B1-0-0).
    for f in sorted(by_name.values(), key=lambda x: (x["source"], normal_named(x), x["priority"], x["stem"])):
        ck = probes[f["path"]].get("content_key")
        key = (f["source"], ck)
        if ck and key in by_content:
            keep = by_content[key]
            keep["aliases"].append(f["path"])
            keep["aliases"].extend(f["aliases"])
            keep["alias_ids"].append(f"{f['source']}/{f['stem']}")
            if normal_named(f) != normal_named(keep):
                keep["notes"].append(f"label conflict: byte-identical to normal-labelled {f['stem']}")
            continue
        by_content[key] = f
        out.append(f)
    n_dup = len(by_name) - len(out)
    log(f"[catalog] dedupe: {len(files)} files -> {len(by_name)} unique names -> {len(out)} unique videos "
        f"({n_dup} content duplicates under other names)")
    return out


# ---------------------------------------------------------------------------------------------------------
# Domain (best effort)
# ---------------------------------------------------------------------------------------------------------

_KW = [
    ("movie", (" scene", "(19", "(20", "official music video", "snl", "season ", "netflix", "dance moms",
               "love & hip hop", "five nights")),
    ("cctv", ("cctv", "surveillance", "security cam", "cámaras de seguridad", "security footage")),
    ("phone_bodycam", ("bodycam", "body cam", "body camera", "dashcam", "dash cam", "road rage", "fight",
                       "brawl", "altercation", "caught on", "attack", "argument", "confrontation", "punch")),
    ("youtube_news", ("news", "gma", "tmz", "investigates", "speaks out", "testifies", "reacts", "documentary",
                      "crime alert", "police say", "verdict", "sued", "expert", "trial")),
]

def guess_domain(source, kind, title="", hivau_cctv_hits=0):
    """Best-effort footage domain. XD YouTube clips whose HIVAU captions mention surveillance / CCTV at least
    twice (``hivau_cctv_hits``, see setup/build_hivau_onsets.py) are called cctv; curated titles use keywords."""
    if source in ("ucf", "ubi"):
        return "cctv"
    if source == "xd":
        if kind == "movie":
            return "movie"
        if hivau_cctv_hits >= 2:
            return "cctv"
        return "youtube_news"
    if source == "yt":
        t = " " + title.lower()
        for dom, kws in _KW:
            if any(k in t for k in kws):
                return dom
        return "unknown"
    return "unknown"


# ---------------------------------------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------------------------------------

def _frames_to_sec(frames, fps):
    return [[s / fps, e / fps] for s, e in frames]


def is_mixed_xd(classes):
    """A violent XD clip whose label also carries a non-violent anomaly code (e.g. ``B2-G`` shooting + explosion)."""
    cs = set(classes)
    return bool(cs & A.VIOLENT_CLASSES) and bool(cs - A.VIOLENT_CLASSES - {A.NORMAL})


def segment_kinds(segs, hv_rec, tol=1.0):
    """Per-segment HIVAU kind (``A.hivau_event_kinds``) for ``segs`` (seconds): the HIVAU event whose start is
    nearest (within ``tol`` s) gives the kind; ``"unmatched"`` when HIVAU has no such event."""
    if not hv_rec:
        return ["unmatched"] * len(segs)
    hv_ev = hv_rec.get("events_sec") or []
    kinds = hv_rec.get("event_kinds") or ["unknown"] * len(hv_ev)
    out = []
    for s, _ in segs:
        d = [abs(float(h[0]) - s) for h in hv_ev]
        j = int(np.argmin(d)) if d else -1
        out.append(kinds[j] if j >= 0 and d[j] <= tol else "unmatched")
    return out


def build_rows(canon, probes, hivau, near_dups=None, log=_log):
    """Assemble catalog rows and events.json entries for the canonical files.

    Pass 1 attaches ground truth per file; ``source_masks`` adds cross-clip masks for XD clips cut from the same
    source; ``apply_near_dups`` marks identical re-encodes as role ``duplicate`` and builds ``split_group``;
    ``finalize_eligibility`` computes the eligible events last, after every exclusion is known.
    """
    xd_off = A.read_xd_official()
    ucf_off = A.read_ucf_official()
    ucf_train = A.read_ucf_train_list()
    ubi_ann = A.read_ubi_annotations()
    ubi_split = A.read_ubi_splits()
    curated = A.read_curated()
    hv_xd = (hivau or {}).get("xd", {})
    hv_ucf = (hivau or {}).get("ucf", {})
    hv_cctv = (hivau or {}).get("xd_cctv_hits", {})

    recs = []
    for f in canon:
        src, stem, path = f["source"], f["stem"], f["path"]
        pr = probes[path]
        vid = f"{src}/{stem}"
        fps = float(pr.get("fps") or 0.0)
        dur = float(pr.get("duration_sec") or 0.0)
        notes = list(f["notes"])
        if pr.get("error"):
            notes.append("probe:" + pr["error"])
        segs_raw, onset_source, classes, category, kind, split, group = [], "none", ["unknown"], "", "", "", vid
        hv_segs = None
        pseudo = None
        other_raw = []
        span = None

        if src == "xd":
            info = A.parse_xd_name(stem)
            classes, category, kind = info["classes"], info["category"], info["kind"]
            group = "xd:" + info["group"]
            span = A.parse_xd_source_span(stem)
            split = "test" if f["folder"] in ("xd_test", "xd_test_extra") else "train"
            hv_split = (hivau or {}).get("split_of", {}).get("xd", {}).get(stem)
            if hv_split and hv_split != split:
                notes.append(f"folder split {split} but HIVAU lists it under {hv_split}; using HIVAU")
                split = hv_split
            if stem in hv_xd:
                hv_segs = hv_xd[stem]["events_sec"]
            if stem in xd_off:
                segs_raw = _frames_to_sec(xd_off[stem], fps)
                onset_source = "official"
            elif stem in hv_xd:
                segs_raw = list(hv_xd[stem]["events_sec"])
                onset_source = "hivau"
            if classes == [A.NORMAL] and (segs_raw or stem in hv_xd):
                notes.append("normal-labelled clip with annotated events; events ignored")
                segs_raw, onset_source = [], "none"
            if segs_raw and is_mixed_xd(classes):
                # Mixed-label clip: HIVAU's per-event judgement says which segment is violent. Explosions,
                # crashes and "no anomaly" segments become excluded other-anomaly time, not violent onsets.
                kinds = segment_kinds(segs_raw, hv_xd.get(stem))
                keep = [k in ("violent", "unmatched") for k in kinds]
                other_raw = [sg for sg, k in zip(segs_raw, keep) if not k]
                segs_raw = [sg for sg, k in zip(segs_raw, keep) if k]
                if other_raw:
                    notes.append(f"mixed-label clip: {len(other_raw)} segment(s) judged non-violent by HIVAU "
                                 f"({','.join(k for k in kinds if k not in ('violent', 'unmatched'))}) -> excluded")
        elif src == "ucf":
            folder = f["folder"]
            if stem.startswith("Normal_Videos"):
                category = "Normal"
            elif folder == "Anomaly-Videos":
                category = os.path.basename(os.path.dirname(path))
            else:
                category = next((c for c in A.UCF_CLASSES if stem.startswith(c)), "unknown")
            classes = [A.UCF_CLASSES.get(category, "unknown")]
            kind = "cctv"
            group = "ucf:" + stem
            if stem in ucf_off:
                split = "test"
            elif stem in ucf_train:
                split = "train"
            elif folder == "Training-Normal-Videos":
                split = "train"
            elif folder in ("Testing_Normal_Videos",):
                split = "test"
            else:
                split = "unknown"
            if stem in hv_ucf:
                hv_segs = hv_ucf[stem]["events_sec"]
            if category != "Normal":
                if stem in ucf_off and ucf_off[stem]["frames"]:
                    segs_raw = _frames_to_sec(ucf_off[stem]["frames"], fps)
                    onset_source = "official"
                elif stem in hv_ucf:
                    segs_raw = list(hv_ucf[stem]["events_sec"])
                    onset_source = "hivau"
                    hv_cls = hv_ucf[stem]["classes"]
                    if classes[0] not in hv_cls:
                        notes.append(f"hivau label {hv_cls} differs from folder {category}")
            elif stem in hv_ucf:
                notes.append("normal video with HIVAU anomaly label; ignored")
        elif src == "ubi":
            kind = "cctv"
            group = "ubi:" + stem
            is_fight = stem.upper().startswith("F") or "fight" in path.lower().split(os.sep)[-2]
            classes = ["fighting"] if is_fight else [A.NORMAL]
            category = "fight" if is_fight else "normal"
            split = ubi_split.get(stem, "train" if "test" in ubi_split.values() else "unknown")
            if is_fight and stem in ubi_ann:
                try:
                    segs_raw = _frames_to_sec(A.ubi_segments_frames(ubi_ann[stem]), fps)
                    onset_source = "ubi" if segs_raw else "none"
                except ValueError as e:
                    notes.append(f"ubi annotation unreadable: {e}")
        elif src == "yt":
            kind = "curated"
            group = "yt:" + stem
            split = "holdout"
            category = "curated"
            classes = ["unknown"]
            cinfo = curated.get(stem, {})
            if cinfo.get("pseudo_onset_sec") is not None:
                pseudo = float(cinfo["pseudo_onset_sec"])
                onset_source = "pseudo"

        segs = normalize_segments(segs_raw, dur) if dur > 0 else []
        others = normalize_segments(other_raw, dur) if dur > 0 else []
        if segs_raw and len(segs) < len(segs_raw):
            notes.append(f"{len(segs_raw)} raw segments -> {len(segs)} after clip/merge")
        if segs_raw and not segs:
            onset_source = "none"
            notes.append("all segments outside the video")
        if other_raw and not segs:
            onset_source = "none"
            notes.append("no violent segment left after the mixed-label split")

        cls_set = set(classes)
        violent = bool(cls_set & A.VIOLENT_CLASSES)
        interpersonal = bool(cls_set & A.INTERPERSONAL_CLASSES)
        if src == "yt":
            role = "holdout"
        elif cls_set == {A.NORMAL}:
            role = "clean_negative"
        elif violent:
            role = "violent" if (segs and onset_source in TRUSTED_ONSET_SOURCES) else "unlabelled"
        elif cls_set & set(A.UCF_CLASSES.values()) | cls_set & set(A.XD_CODES.values()):
            role = "other_anomaly"
        else:
            role = "unlabelled"
        if dur < 1.0:
            notes.append("shorter than 1 s: no ticks")

        title = curated.get(stem, {}).get("title", "") if src == "yt" else ""
        domain = guess_domain(src, kind, title=title, hivau_cctv_hits=int(hv_cctv.get(stem, 0)))
        source_kind = {"xd": "xd_" + (kind or "unknown"), "ucf": "ucf", "ubi": "ubi", "yt": "yt"}[src]
        franchise = ""
        if src == "xd" and kind == "movie":
            franchise = A.XD_FRANCHISE_OF.get(group[3:], "")
        recs.append({
            "vid": vid, "src": src, "path": path, "dur": dur, "fps": fps, "pr": pr, "notes": notes,
            "segs": segs if onset_source in TRUSTED_ONSET_SOURCES else [], "segs_any": segs, "others": others,
            "masks": [], "soft_masks": [], "onset_source": onset_source, "classes": classes, "category": category,
            "kind": kind,
            "split": split, "group": group, "role": role, "violent": violent, "interpersonal": interpersonal,
            "hv_segs": hv_segs, "pseudo": pseudo, "domain": domain, "source_kind": source_kind, "span": span,
            "franchise": franchise, "aliases": f["aliases"], "alias_ids": f["alias_ids"], "duplicate_of": "",
            "split_group": group,
            "official_annotated": bool((src == "xd" and stem in xd_off)
                                       or (src == "ucf" and bool(ucf_off.get(stem, {}).get("frames")))),
        })
    source_masks(recs, log=log)
    apply_near_dups(recs, near_dups, log=log)
    return finalize_rows(recs)


def _subtract(spans, cuts):
    """``spans`` minus the union of ``cuts`` (both lists of [s, e])."""
    out = []
    for s, e in spans:
        pieces = [[s, e]]
        for cs, ce in cuts:
            nxt = []
            for a, b in pieces:
                if ce <= a or cs >= b:
                    nxt.append([a, b])
                    continue
                if cs > a:
                    nxt.append([a, cs])
                if ce < b:
                    nxt.append([ce, b])
            pieces = nxt
        out += [p for p in pieces if p[1] - p[0] > 1e-6]
    return out


def source_masks(recs, window=120.0, aftermath=10.0, log=_log):
    """Cross-clip masks for XD clips cut from the same source film (``#HH-MM-SS_HH-MM-SS`` in the name).

    For a violent segment [s, e] of clip A (source time ``span_start + s``) and another clip B of the same group
    that does not annotate the same violence itself (none of B's segments overlaps it):
      * ``[s, e + 10]`` (violence and aftermath) on B's clock -> B ``mask_spans``: excluded AND it ends any clear
        lead-in that starts before it (B's footage there is not clear);
      * ``[s - 120, s)`` (A's pre-onset window) on B's clock -> excluded ticks of B that B itself labels as far from
        violence: for a clean B the whole window (``mask_spans``); for a violent B only the part outside B's own
        pre-onset windows [t0 - 120, t0) and violence + aftermath (``soft_mask_spans``, which never ends a
        lead-in: those ticks are pre-onset for B's own events anyway).
    A clean clip whose span overlaps A's violence becomes role ``contaminated`` (excluded from evaluation).
    """
    by_group = {}
    for r in recs:
        if r["src"] == "xd" and r["span"] is not None and r["dur"] > 0:
            by_group.setdefault(r["group"], []).append(r)
    n_masked = n_cont = 0
    for g, rs in by_group.items():
        viol = [(r["vid"], r["span"][0] + s, r["span"][0] + e) for r in rs if r["role"] == "violent"
                for s, e in r["segs"]]
        if not viol:
            continue
        for r in rs:
            if r["role"] == "holdout":
                continue
            a0 = r["span"][0]
            a1 = a0 + r["dur"]
            own = [(a0 + s, a0 + e) for s, e in r["segs"]]
            own_windows = [[s - a0 - window, e - a0 + aftermath] for s, e in own]
            hard, soft = [], []
            for v, s_abs, e_abs in viol:
                if v == r["vid"]:
                    continue
                if any(s_abs <= oe + 1.0 and os_ <= e_abs + 1.0 for os_, oe in own):
                    continue   # the same violence is annotated in this clip
                lo, hi = max(a0, s_abs), min(a1, e_abs + aftermath)
                if hi > lo:
                    hard.append([lo - a0, hi - a0])
                plo, phi = max(a0, s_abs - window), min(a1, s_abs)
                if phi > plo:
                    if r["role"] == "violent":
                        soft += _subtract([[plo - a0, phi - a0]], own_windows)
                    else:
                        hard.append([plo - a0, phi - a0])
                if r["role"] == "clean_negative" and min(a1, e_abs) > max(a0, s_abs):
                    r["role"] = "contaminated"
                    r["notes"].append(f"clean clip overlaps violence of {v} in the source film -> contaminated")
            r["masks"] = [[round(s, 3), round(e, 3)] for s, e in normalize_segments(hard, r["dur"])]
            r["soft_masks"] = [[round(s, 3), round(e, 3)] for s, e in normalize_segments(soft, r["dur"])]
            if r["masks"] or r["soft_masks"]:
                n_masked += 1
                tot = sum(e - s for s, e in normalize_segments(r["masks"] + r["soft_masks"], r["dur"]))
                r["notes"].append(f"{tot:.1f} s masked: violence of an adjacent clip of the same source "
                                  f"(within 120 s before it, or its violence / aftermath)")
            n_cont += r["role"] == "contaminated"
    log(f"[catalog] source masks: {n_masked} XD clips get mask spans, {n_cont} clean clips contaminated")


ROLE_RANK = {"violent": 0, "clean_negative": 1, "other_anomaly": 2, "unlabelled": 3, "contaminated": 4,
             "holdout": 5}
ONSET_RANK = {"official": 0, "ubi": 1, "hivau": 2}


def _union_find(edges, nodes=()):
    parent = {n: n for n in nodes}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {n: find(n) for n in list(parent)}


def apply_near_dups(recs, nd, log=_log):
    """Near-duplicates (``anticipation.data.near_dups``): identical re-encodes keep one canonical copy (violent
    with official onsets first, then other roles, then the official test split, then the id) and the others become
    role ``duplicate``; every near-duplicate pair, every XD franchise and every catalog group are merged into one
    ``split_group`` (the fold grouping key of folds_v2)."""
    by_vid = {r["vid"]: r for r in recs}
    edges = []
    # Catalog groups and XD franchises.
    first_of_group, first_of_fr = {}, {}
    for r in recs:
        edges.append((r["vid"], first_of_group.setdefault(r["group"], r["vid"])))
        if r["franchise"]:
            edges.append((r["vid"], first_of_fr.setdefault(r["franchise"], r["vid"])))
    n_ident = 0
    if nd:
        pairs = [p for p in nd.get("pairs", []) if p["a"] in by_vid and p["b"] in by_vid]
        edges += [(p["a"], p["b"]) for p in pairs]
        ident = _union_find([(p["a"], p["b"]) for p in pairs if p["kind"] == "identical"])
        comps = {}
        for v, root in ident.items():
            comps.setdefault(root, []).append(v)
        for members in comps.values():
            if len(members) < 2:
                continue
            ranked = sorted(members, key=lambda v: (ROLE_RANK.get(by_vid[v]["role"], 9),
                                                    ONSET_RANK.get(by_vid[v]["onset_source"], 9),
                                                    by_vid[v]["split"] != "test", v))
            keep = ranked[0]
            for v in ranked[1:]:
                r = by_vid[v]
                if r["role"] == "holdout":
                    continue
                note = f"identical re-encode of {keep} (near_duplicates.json)"
                if r["role"] != by_vid[keep]["role"] or set(r["classes"]) != set(by_vid[keep]["classes"]):
                    note += f"; label conflict: {r['role']}/{'|'.join(r['classes'])} vs " \
                            f"{by_vid[keep]['role']}/{'|'.join(by_vid[keep]['classes'])}"
                r["notes"].append(note)
                r["role"] = "duplicate"
                r["duplicate_of"] = keep
                n_ident += 1
    root = _union_find(edges, [r["vid"] for r in recs])
    comp_group = {}
    for r in recs:
        comp_group.setdefault(root[r["vid"]], set()).add(r["group"])
    for r in recs:
        r["split_group"] = min(comp_group[root[r["vid"]]])
    n_sg = len({r["split_group"] for r in recs})
    n_g = len({r["group"] for r in recs})
    log(f"[catalog] near-duplicates: {n_ident} identical copies -> role duplicate; {n_g} groups -> {n_sg} split "
        f"groups (franchises + near-duplicate pairs merged)")


def finalize_rows(recs):
    """Eligibility (after every exclusion), catalog rows and events.json entries."""
    rows, events = [], {}
    for r in recs:
        pr, segs, role = r["pr"], r["segs"], r["role"]
        blockers = blocker_spans(r["others"], r["masks"])
        ev = event_table(segs, blockers=blockers) if r["onset_source"] in TRUSTED_ONSET_SOURCES else []
        is_v_role = role == "violent"
        n30 = sum(e["eligible30"] for e in ev) if is_v_role else 0
        n60 = sum(e["eligible60"] for e in ev) if is_v_role else 0
        first_onset = r["segs_any"][0][0] if r["segs_any"] else (r["pseudo"] if r["pseudo"] is not None else np.nan)
        starts = [s for s, _ in r["segs_any"]] + [s for s, _ in r["others"]]
        first_anomaly = min(starts) if starts else np.nan
        rows.append({
            "video_id": r["vid"], "source": r["src"], "path": r["path"], "duration_sec": round(r["dur"], 4),
            "fps": r["fps"], "num_frames": int(pr.get("num_frames") or 0),
            "has_audio": bool(pr.get("has_audio", False)),
            "group": r["group"], "official_split": r["split"], "category": r["category"],
            "is_violent_video": r["violent"], "is_clean_negative": role == "clean_negative",
            "onset_source": r["onset_source"], "first_onset_sec": first_onset, "first_anomaly_sec": first_anomaly,
            "eligible30": bool(n30 > 0), "eligible60": bool(n60 > 0),
            "domain": r["domain"], "role": role, "classes": "|".join(r["classes"]),
            "interpersonal": r["interpersonal"],
            "n_segments": len(segs), "n_eligible30": n30, "n_eligible60": n60,
            "source_kind": r["source_kind"], "audio_stream": bool(pr.get("audio_stream", False)),
            "audio_peak": float(pr.get("audio_peak", 0.0)), "width": int(pr.get("width") or 0),
            "height": int(pr.get("height") or 0), "aliases": ";".join(r["aliases"]),
            "content_key": pr.get("content_key", ""), "notes": " | ".join(r["notes"]),
            "n_other_segments": len(r["others"]),
            "masked_sec": round(sum(e - s for s, e in normalize_segments(r["masks"] + r["soft_masks"], r["dur"])), 3),
            "franchise": r["franchise"], "split_group": r["split_group"], "duplicate_of": r["duplicate_of"],
            "official_annotated": r["official_annotated"],
        })
        entry = {
            "segments": [[round(s, 4), round(e, 4)] for s, e in segs],
            "onset_source": r["onset_source"], "classes": r["classes"], "category": r["category"], "role": role,
            "is_violent_video": r["violent"], "interpersonal": r["interpersonal"], "duration_sec": round(r["dur"], 4),
        }
        if r["others"]:
            entry["other_segments"] = [[round(s, 4), round(e, 4)] for s, e in r["others"]]
        if r["masks"]:
            entry["mask_spans"] = [[round(s, 4), round(e, 4)] for s, e in r["masks"]]
        if r["soft_masks"]:
            entry["soft_mask_spans"] = [[round(s, 4), round(e, 4)] for s, e in r["soft_masks"]]
        if r["hv_segs"] is not None and r["onset_source"] != "hivau":
            entry["hivau_segments"] = [[round(float(s), 4), round(float(e), 4)] for s, e in r["hv_segs"]]
        if r["pseudo"] is not None:
            entry["pseudo_onsets_sec"] = [r["pseudo"]]
        if r["alias_ids"]:
            entry["alias_ids"] = r["alias_ids"]
        if r["duplicate_of"]:
            entry["duplicate_of"] = r["duplicate_of"]
        if ev and is_v_role:
            entry["events"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items()} for e in ev]
        events[r["vid"]] = entry
    df = pd.DataFrame(rows, columns=COLUMNS).sort_values("video_id").reset_index(drop=True)
    return df, dict(sorted(events.items()))


# ---------------------------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------------------------

def summarize(df, events):
    """Counts per source / role / eligibility used by docs/VA30.md."""
    out = {"n_videos": int(len(df)), "hours": round(float(df.duration_sec.sum()) / 3600, 2)}
    by = {}
    for src, g in df.groupby("source_kind"):
        roles = g.role.value_counts().to_dict()
        v = g[g.role == "violent"]
        by[src] = {
            "videos": int(len(g)), "hours": round(float(g.duration_sec.sum()) / 3600, 2),
            "roles": {k: int(x) for k, x in roles.items()},
            "violent_onset_sources": {k: int(x) for k, x in v.onset_source.value_counts().items()},
            "segments": int(v.n_segments.sum()),
            "eligible30_videos": int(v.eligible30.sum()), "eligible60_videos": int(v.eligible60.sum()),
            "eligible30_events": int(v.n_eligible30.sum()), "eligible60_events": int(v.n_eligible60.sum()),
            "eligible30_interpersonal_videos": int((v.eligible30 & v.interpersonal).sum()),
            "first_onset_ge30_videos": int((v.first_onset_sec >= 30).sum()),
            "has_audio": int(g.has_audio.sum()),
            "clean_negative_hours": round(float(g[g.role == "clean_negative"].duration_sec.sum()) / 3600, 2),
        }
    out["by_source_kind"] = by
    v = df[df.role == "violent"]
    ev30 = [e for vid in v.video_id if vid in events for e in events[vid].get("events", []) if e["eligible30"]]
    out["violent"] = {
        "videos": int(len(v)), "segments": int(v.n_segments.sum()),
        "eligible30_videos": int(v.eligible30.sum()), "eligible30_events": int(v.n_eligible30.sum()),
        "eligible30_first_segment_events": int(sum(e["index"] == 0 for e in ev30)),
        "eligible60_videos": int(v.eligible60.sum()), "eligible60_events": int(v.n_eligible60.sum()),
        "eligible30_interpersonal_videos": int((v.eligible30 & v.interpersonal).sum()),
        "eligible30_by_onset_source": {k: int(x) for k, x in v[v.eligible30].onset_source.value_counts().items()},
        "eligible30_events_by_onset_source": {k: int(x) for k, x in
                                              v.groupby("onset_source").n_eligible30.sum().items()},
        "eligible30_leadin_hours": round(float(sum(min(e["leadin"], 120.0) for e in ev30)) / 3600, 2),
        "mixed_label_clips_with_excluded_segments": int(((df.role == "violent") & (df.n_other_segments > 0)).sum()),
        "excluded_other_segments": int(df[df.role.isin(["violent", "unlabelled"])].n_other_segments.sum()),
    }
    off = v[v.onset_source == "official"]
    out["official_only"] = {
        "violent_videos": int(len(off)), "eligible30_videos": int(off.eligible30.sum()),
        "eligible30_events": int(off.n_eligible30.sum()),
        "eligible30_interpersonal_videos": int((off.eligible30 & off.interpersonal).sum()),
    }
    # Annotated videos (any class) whose first annotated segment (violent or not) starts >= 30 s, from official
    # files: the docs' "112" number (XD 53, UCF 59).
    ann = df[df.official_annotated]
    out["official_any_class_first_onset_ge30"] = {
        k: int(((ann.source == k) & (ann.first_anomaly_sec >= 30)).sum()) for k in ("xd", "ucf")}
    cn = df[df.role == "clean_negative"]
    out["clean_negatives"] = {"videos": int(len(cn)), "hours": round(float(cn.duration_sec.sum()) / 3600, 2),
                              "videos_with_mask_spans": int((cn.masked_sec > 0).sum()),
                              "masked_hours": round(float(cn.masked_sec.sum()) / 3600, 3)}
    out["roles"] = {k: int(x) for k, x in df.role.value_counts().items()}
    out["domains"] = {k: int(x) for k, x in df.domain.value_counts().items()}
    out["duplicates_removed_byte_identical"] = int(sum(len(e.get("alias_ids", [])) for e in events.values()))
    out["duplicates_near_identical"] = int((df.role == "duplicate").sum())
    out["groups"] = int(df.group.nunique())
    out["split_groups"] = int(df.split_group.nunique())
    return out


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------
# Loading (for downstream code)
# ---------------------------------------------------------------------------------------------------------

def load_catalog(path=VIDEOS_CSV):
    """videos.csv as a DataFrame indexed by video_id (booleans parsed)."""
    df = pd.read_csv(path, keep_default_na=True, na_values=[""])
    for c in ("has_audio", "is_violent_video", "is_clean_negative", "eligible30", "eligible60",
              "interpersonal", "audio_stream", "official_annotated"):
        if c in df and df[c].dtype != bool:
            df[c] = df[c].astype(str).str.lower().eq("true")
    for c in ("aliases", "notes", "category", "classes", "content_key", "franchise", "split_group",
              "duplicate_of"):
        if c in df:
            df[c] = df[c].fillna("")
    return df.set_index("video_id", drop=False)


def load_events(path=EVENTS_JSON):
    """events.json ``videos`` dict: ``{video_id: {"segments": [[s, e], ...], ...}}`` (seconds)."""
    with open(path) as f:
        return json.load(f)["videos"]


# ---------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------

def build(workers=16, audio=True, out_dir=BENCHMARK_DIR, log=_log):
    files = discover_files()
    log(f"[catalog] discovered {len(files)} files: "
        + ", ".join(f"{s}={sum(1 for f in files if f['source'] == s)}" for s in ("xd", "ucf", "ubi", "yt")))
    probes = probe_all(sorted({f["path"] for f in files}), workers=workers, audio=audio, log=log)
    canon = dedupe(files, probes, log=log)
    hivau = A.load_hivau_onsets()
    if hivau is None:
        if all(os.path.exists(os.path.join(A.HIVAU_RAW_DIR, f)) for f in A.HIVAU_RAW_FILES):
            log("[catalog] WARNING: hivau_onsets.json missing; parsing raw HIVAU files directly "
                "(run setup/build_hivau_onsets.py for the cross-check report)")
            hivau = A.build_hivau_onsets_from_raw()
        else:
            log("[catalog] WARNING: no HIVAU annotations; UCF/XD train anomalies will be 'unlabelled'. "
                "Run setup/build_hivau_onsets.py")
    from anticipation.data.near_dups import load_near_dups

    nd = load_near_dups()
    if nd is None:
        log("[catalog] WARNING: benchmark/near_duplicates.json missing; re-encoded copies are not grouped "
            "(run python -m anticipation.data.near_dups all, then re-run the catalog)")
    df, events = build_rows(canon, probes, hivau, near_dups=nd, log=log)
    os.makedirs(out_dir, exist_ok=True)
    vpath = os.path.join(out_dir, "videos.csv")
    epath = os.path.join(out_dir, "events.json")
    df.to_csv(vpath, index=False, float_format="%.6g")
    meta = {
        "version": CATALOG_VERSION, "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "units": "seconds", "violent_classes": sorted(A.VIOLENT_CLASSES),
        "interpersonal_classes": sorted(A.INTERPERSONAL_CLASSES),
        "trusted_onset_sources": list(TRUSTED_ONSET_SOURCES),
        "notes": "segments = sorted, clipped, merged violent (or other-anomaly) segments in seconds; empty for "
                 "clean negatives, unlabelled and hold-out videos. other_segments = non-violent anomaly segments "
                 "of mixed-label XD clips (excluded time). mask_spans = violence / aftermath of another clip of the same "
                 "XD source film, or (clean clips) its 120 s pre-onset window: excluded, and it ends a clear lead-in. "
                 "soft_mask_spans = the pre-onset window of such violence inside a violent clip, outside the clip's "
                 "own pre-onset windows: excluded, does not end a lead-in. events = per-segment "
                 "eligibility (labels.event_table) for violent videos. hivau_segments are given for "
                 "cross-reference where official onsets were used. pseudo_onsets_sec are LSTM pseudo-labels "
                 "(not ground truth). duplicate_of = canonical copy of an identical re-encode.",
        "near_duplicates": (nd or {}).get("meta"),
        "hivau": (hivau or {}).get("meta", {}),
    }
    with open(epath, "w") as f:
        json.dump({"meta": meta, "videos": events}, f, indent=1)
    summ = summarize(df, events)
    summ["videos_csv_sha256"] = file_sha256(vpath)
    summ["events_json_sha256"] = file_sha256(epath)
    summ["created"] = meta["created"]
    with open(os.path.join(out_dir, "catalog_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    log(f"[catalog] wrote {vpath} ({len(df)} videos), {epath}")
    log(json.dumps({k: summ[k] for k in ("n_videos", "hours", "roles", "violent", "official_only",
                                         "official_any_class_first_onset_ge30", "clean_negatives")}, indent=1))
    return df, events, summ


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no_audio", action="store_true", help="skip the audio decode (has_audio left False)")
    ap.add_argument("--out_dir", default=BENCHMARK_DIR)
    args = ap.parse_args(argv)
    build(workers=args.workers, audio=not args.no_audio, out_dir=args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
