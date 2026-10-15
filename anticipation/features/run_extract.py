"""Shared batch runner for every per-video feature extractor in ``anticipation/features``.

What
    Resolves a list of videos (from the benchmark manifest or explicit paths), keeps this shard's share, skips
    videos whose outputs already exist, runs the extractor on the rest one by one, and writes the
    ``io.feature_path(<output>, <video_id>)`` files. A failing video is logged and skipped; only
    ``--max_consecutive_fail`` failures in a row stop the run (exit status 3).
    Re-running the same command resumes where it stopped (outputs are written atomically, so a killed run never
    leaves a half-written file).

Registry
    ``REGISTRY`` maps an extractor name to an ``ExtractorSpec``. The default protocol is a module with

        extract(video_path, device="cuda:0", **cfg) -> (t_sec, feat, valid, meta)

    whose result the runner saves with ``io.save_features(<name>, video_id, ...)``. Modules that write several
    outputs from one pass (``pose`` writes ``pose_raw`` + ``pose_interact``) or need extra arrays name a
    ``saver`` instead: ``saver(video_path, video_id, device=..., **cfg) -> {"paths": {...}, "duration_sec": s}``.
    To add an extractor, add one line to ``REGISTRY``. An unregistered dotted module path also works
    (``--extractor anticipation.features.foo``; output name = ``module.EXTRACTOR``).

Video ids and sources
    ``video_id`` = ``<source>/<stem>``, the key used by ``io.feature_path``. The manifest's ``video_id`` column
    wins. For ``--video_paths`` the source is inferred from the path: ``xd_violence`` -> ``xd``, ``ucf_crime`` ->
    ``ucf``, ``anticipation_curation`` -> ``yt`` (as in benchmark/videos.csv), ``kinetics``, ``ubi``,
    otherwise ``other`` (override with ``--source``).

Manifest
    CSV with a header. Recognized columns (first match wins): id = ``video_id|vid|id``, path =
    ``path|video_path|abs_path|file_path|filepath``, source = ``source|dataset``. Relative paths are resolved
    against the data root (data), then the repo root. An id without
    "/" is prefixed with its source.

Sharding
    A video belongs to shard ``crc32(video_id) % num_shards``: stable when the manifest grows or is reordered,
    so shards never overlap. Run one process per (GPU, shard), e.g. 2 GPUs x 3 processes = 6 shards.

Skip, versions and resume
    A video is skipped when all its outputs exist. A module that declares ``OUTPUT_VERSIONS = {output: version}``
    (pose, motion) also has each existing output's ``meta["version"]`` compared with it: a mismatching or
    unreadable file counts as missing and is recomputed (``stale`` in the header line). So "bump VERSION, re-run
    the same command" recomputes exactly the stale files and resumes after a crash, unlike ``--overwrite``,
    which recomputes everything again on every restart. ``--no_version_check`` restores existence-only skipping.
    Modules without ``OUTPUT_VERSIONS`` are only checked for existence.

Errors
    A missing manifest / video list, no input at all, or a selection that is empty after ``--sources`` /
    ``--video_ids`` exits with status 1 and a message (a detached bulk launch with a typo must not look like a
    successful no-op). An empty shard of a non-empty selection is fine (exit 0).

Logs (``data/logs/extract/<extractor>/``, or
``<feature_root>/_logs/<extractor>/`` when ``--feature_root`` is given)
    ``shard<i>of<n>.jsonl``: one record per video (status ok / skip / fail, wall seconds, video seconds).
    ``failures.jsonl``: failures from all shards with the traceback (file-locked appends).
    ``--skip_failed`` skips videos already listed in failures.jsonl (avoids re-crashing on corrupt files).
    Videos with a ``crash`` record (the run died inside them) are always skipped unless ``--retry_crashed``.

Usage
    python -m anticipation.features.run_extract --extractor pose \
        --manifest benchmark/videos.csv [--sources xd,ucf] [--video_ids xd/a ucf/b] \
        --shard 0 --num_shards 2 --device cuda:0
    python -m anticipation.features.run_extract --extractor motion --video_paths a.mp4 b.mp4 --device cpu
    python -m anticipation.features.run_extract --extractor pose --video_list paths.txt --cfg weights=yolo11l-pose.pt
    python -m anticipation.features.run_extract --list
    python -m anticipation.features.run_extract --selftest   # selection errors + version skipping (CPU, 1 s)
    # long runs: detach and log (resumable; re-run the same command after a crash)
    setsid nohup .venv/bin/python -m anticipation.features.run_extract --extractor pose --manifest ... \
        --shard 0 --num_shards 6 --device cuda:0 > data/logs/pose_0.log 2>&1 &
"""

import argparse
import csv
import fcntl
import importlib
import json
import os
import signal
import socket
import sys
import time
import traceback
import zlib
from dataclasses import dataclass

import numpy as np

from anticipation.features import io

DATA_ROOT = "data"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_ROOT = os.path.join(DATA_ROOT, "logs", "extract")
VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".webm", ".mov", ".m4v", ".mpg", ".mpeg", ".flv")


@dataclass(frozen=True)
class ExtractorSpec:
    module: str                 # import path of the extractor module
    outputs: tuple = ()         # feature directories written; default: (module.EXTRACTOR or the registry name,)
    saver: str = ""             # optional function that writes its own outputs (see module docstring)
    default_device: str = "cuda:0"
    video_kwargs: tuple = ()    # ((kwarg, module_function), ...): kwarg = module_function(video_id) per video


REGISTRY = {
    # pose + motion track
    "motion": ExtractorSpec("anticipation.features.motion", default_device="cpu"),
    "pose": ExtractorSpec("anticipation.features.pose", outputs=("pose_raw", "pose_interact"),
                          saver="extract_and_save"),
    # re-derive pose_interact from existing pose_raw files (CPU); pass --overwrite after changing feature code
    "pose_interact": ExtractorSpec("anticipation.features.pose", outputs=("pose_interact",),
                                   saver="derive_and_save", default_device="cpu"),
    # visual-backbone track ("visual" = all three from one decode; recommended for bulk runs)
    "vjepa2_l": ExtractorSpec("anticipation.features.vjepa"),
    "siglip2": ExtractorSpec("anticipation.features.siglip"),
    "siglip2_prompts": ExtractorSpec("anticipation.features.prompts"),
    "visual": ExtractorSpec("anticipation.features.prompts", outputs=("vjepa2_l", "siglip2", "siglip2_prompts"),
                            saver="extract_and_save"),
    # audio track
    "audio_events": ExtractorSpec("anticipation.features.audio_events",
                                  outputs=("ast_events", "ast_emb", "clap_prompts"), saver="extract_and_save"),
    "voice": ExtractorSpec("anticipation.features.voice", outputs=("voice_affect", "e2v_emb"),
                           saver="extract_and_save"),
    "audio_dsp": ExtractorSpec("anticipation.features.audio_dsp", default_device="cpu"),
    # speech track (raw transcripts go next to the feature file)
    "speech_tox": ExtractorSpec("anticipation.features.speech",
                                video_kwargs=(("transcript_path", "transcript_path"),)),
}

# Path substring -> source name. Matches benchmark/videos.csv (xd, ucf, yt); the manifest's video_id always wins.
SOURCE_ALIASES = [("xd_violence", "xd"), ("ucf_crime", "ucf"), ("anticipation_curation", "yt"),
                  ("kinetics", "kinetics"), ("ubi", "ubi")]


def resolve_spec(name):
    """``(spec, module, outputs)`` for a registry name or a dotted module path."""
    spec = REGISTRY.get(name)
    if spec is None:
        if "." not in name:
            raise SystemExit(f"unknown extractor {name!r}; registered: {sorted(REGISTRY)}")
        spec = ExtractorSpec(name)
    mod = importlib.import_module(spec.module)
    outputs = spec.outputs or (getattr(mod, "EXTRACTOR", name),)
    if not spec.saver and not hasattr(mod, "extract"):
        raise SystemExit(f"{spec.module} has no extract()")
    return spec, mod, tuple(outputs)


# ----------------------------------------------------------------------------------------------------------------
# Video lists
# ----------------------------------------------------------------------------------------------------------------

def infer_source(path):
    low = path.lower()
    for key, src in SOURCE_ALIASES:
        if key in low:
            return src
    return "other"


def video_id_for(path, source=None):
    stem = os.path.basename(path)
    for ext in VIDEO_EXTS:
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    return f"{source or infer_source(path)}/{stem}"


def _resolve_path(p):
    if os.path.isabs(p):
        return p
    for root in (DATA_ROOT, REPO_ROOT):
        cand = os.path.join(root, p)
        if os.path.exists(cand):
            return cand
    return os.path.join(DATA_ROOT, p)


def _existing_file(path, what):
    """``path`` as given, else relative to the repo root; SystemExit with a clear message if neither exists."""
    for cand in (path, os.path.join(REPO_ROOT, path)):
        if os.path.isfile(cand):
            return cand
    raise SystemExit(f"[run_extract] {what} not found: {path} (cwd {os.getcwd()}, also tried {REPO_ROOT})")


def read_manifest(path):
    """List of ``{"video_id", "path", "source"}`` from a manifest CSV (see module docstring for columns)."""
    path = _existing_file(path, "manifest")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    cols = rows[0].keys()

    def pick(cands):
        return next((c for c in cands if c in cols), None)

    c_id, c_src = pick(["video_id", "vid", "id"]), pick(["source", "dataset"])
    c_path = pick(["path", "video_path", "abs_path", "file_path", "filepath"])
    if c_path is None:
        raise SystemExit(f"manifest {path} has no path column (columns: {list(cols)})")
    out = []
    for r in rows:
        p = _resolve_path(r[c_path])
        src = (r.get(c_src) or "").strip() if c_src else ""
        vid = (r.get(c_id) or "").strip() if c_id else ""
        if not vid:
            vid = video_id_for(p, src or None)
        elif "/" not in vid:
            vid = f"{src or infer_source(p)}/{vid}"
        out.append(dict(video_id=vid, path=p, source=src or vid.split("/", 1)[0]))
    return out


def collect_videos(args):
    """Videos selected by the arguments, de-duplicated by video_id and sorted by it (input order with --keep_order).

    Raises SystemExit when there is no input, a list file is missing, or the filters leave nothing.
    """
    if not (args.manifest or args.video_paths or args.video_list):
        raise SystemExit("[run_extract] no input: give --manifest, --video_paths or --video_list")
    videos = []
    if args.manifest:
        videos += read_manifest(args.manifest)
    paths = list(args.video_paths or [])
    if args.video_list:
        with open(_existing_file(args.video_list, "video list")) as f:
            paths += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    for p in paths:
        p = os.path.abspath(p)
        vid = video_id_for(p, args.source)
        videos.append(dict(video_id=vid, path=p, source=vid.split("/", 1)[0]))
    if not videos:
        raise SystemExit("[run_extract] the input lists no videos (empty manifest or video list)")
    all_sources = sorted({v["source"] for v in videos})
    if args.sources:
        keep = set(s.strip() for s in args.sources.split(",") if s.strip())
        unknown = sorted(keep - set(all_sources))
        if unknown:
            print(f"[run_extract] warning: --sources {unknown} match no video (available: {all_sources})",
                  file=sys.stderr, flush=True)
        videos = [v for v in videos if v["source"] in keep]
    if args.video_ids:
        keep = set(args.video_ids)
        matched = {v["video_id"] for v in videos} | {v["video_id"].split("/", 1)[-1] for v in videos}
        unknown = sorted(keep - matched)
        if unknown:
            print(f"[run_extract] warning: {len(unknown)} --video_ids match no video: {unknown[:10]}",
                  file=sys.stderr, flush=True)
        videos = [v for v in videos if v["video_id"] in keep or v["video_id"].split("/", 1)[-1] in keep]
    if not videos:
        raise SystemExit(f"[run_extract] the selection is empty after --sources {args.sources!r} / --video_ids "
                         f"(sources in the input: {all_sources})")
    seen, uniq = set(), []
    for v in videos if args.keep_order else sorted(videos, key=lambda v: v["video_id"]):
        if v["video_id"] not in seen:
            seen.add(v["video_id"])
            uniq.append(v)
    return uniq


def shard_of(video_id, num_shards):
    return zlib.crc32(video_id.encode()) % num_shards


_UNREADABLE = object()


def file_version(path):
    """``meta["version"]`` of an existing feature file; ``_UNREADABLE`` if it cannot be read (corrupt file)."""
    try:
        with np.load(path, allow_pickle=False) as z:
            return json.loads(str(z["meta"])).get("version")
    except Exception:  # noqa: BLE001  (truncated zip, missing key, bad JSON)
        return _UNREADABLE


def output_state(output, video_id, expected_version=None):
    """``"missing"``, ``"stale"`` (version differs from ``expected_version`` or unreadable) or ``"ok"``."""
    path = io.feature_path(output, video_id)
    if not os.path.exists(path):
        return "missing"
    if expected_version is None:
        return "ok"
    return "ok" if file_version(path) == expected_version else "stale"


# ----------------------------------------------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------------------------------------------

def _append_jsonl(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(rec) + "\n"
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _failed_ids(path, status=None):
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path) as f:
        for ln in f:
            try:
                r = json.loads(ln)
                if status is None or r.get("status") == status:
                    out.add(r["video_id"])
            except (ValueError, KeyError):
                pass
    return out


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def parse_cfg(items):
    """``["k=v", ...]`` -> dict; values are parsed as JSON when possible (numbers, lists, booleans)."""
    cfg = {}
    for it in items or []:
        k, _, v = it.partition("=")
        try:
            cfg[k] = json.loads(v)
        except ValueError:
            cfg[k] = v
    return cfg


# ----------------------------------------------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------------------------------------------

def run_one(spec, mod, name, outputs, video, device, cfg):
    """Extract one video. Returns ``(paths, duration_sec)``."""
    cfg = dict(cfg)
    for kwarg, fn in spec.video_kwargs:
        cfg[kwarg] = getattr(mod, fn)(video["video_id"])
    if spec.saver:
        res = getattr(mod, spec.saver)(video["path"], video["video_id"], device=device, **cfg)
        return res.get("paths", {}), res.get("duration_sec")
    t_sec, feat, valid, meta = mod.extract(video["path"], device=device, **cfg)
    path = io.save_features(outputs[0], video["video_id"], t_sec, feat, valid, meta)
    dur = None
    if isinstance(meta, dict):
        dur = meta.get("duration_sec", (meta.get("video") or {}).get("duration_sec"))
    return {outputs[0]: path}, dur if dur is not None else float(len(t_sec))


def selftest():
    """Selection errors exit non-zero; version-aware skipping. Uses a temp feature root; returns the results."""
    import contextlib
    import tempfile

    res = {}
    old_root = io.FEATURE_ROOT
    with tempfile.TemporaryDirectory() as d:
        man = os.path.join(d, "m.csv")
        with open(man, "w") as f:
            f.write("video_id,path,source\nxd/a,/nonexistent/a.mp4,xd\nxd/b,/nonexistent/b.mp4,xd\n"
                    "ucf/c,/nonexistent/c.mp4,ucf\n")
        empty = os.path.join(d, "empty.csv")
        with open(empty, "w") as f:
            f.write("video_id,path,source\n")

        def exit_code(argv):
            try:
                with contextlib.redirect_stdout(open(os.devnull, "w")), \
                        contextlib.redirect_stderr(open(os.devnull, "w")):
                    rc = main(argv)
            except SystemExit as e:
                return 0 if e.code in (0, None) else 1
            return rc

        base = ["--extractor", "motion", "--feature_root", d, "--dry_run"]
        res["typo_sources_exit"] = exit_code(base + ["--manifest", man, "--sources", "XD"])
        res["missing_manifest_exit"] = exit_code(base + ["--manifest", os.path.join(d, "nope.csv")])
        res["empty_manifest_exit"] = exit_code(base + ["--manifest", empty])
        res["no_input_exit"] = exit_code(base)
        res["unknown_ids_exit"] = exit_code(base + ["--manifest", man, "--video_ids", "nope"])
        res["valid_selection_exit"] = exit_code(base + ["--manifest", man, "--sources", "xd"])
        assert res["typo_sources_exit"] == res["missing_manifest_exit"] == res["empty_manifest_exit"] == 1, res
        assert res["no_input_exit"] == res["unknown_ids_exit"] == 1 and res["valid_selection_exit"] == 0, res

        # version-aware skipping: motion declares OUTPUT_VERSIONS; old, current and corrupt files
        from anticipation.features import motion

        io.FEATURE_ROOT = d
        try:
            cur = motion.OUTPUT_VERSIONS[motion.EXTRACTOR]
            io.save_features("motion", "xd/a", np.ones(1), np.zeros((1, 1)), np.ones(1, bool), {"version": cur})
            io.save_features("motion", "xd/b", np.ones(1), np.zeros((1, 1)), np.ones(1, bool), {"version": -1})
            os.makedirs(os.path.dirname(io.feature_path("motion", "ucf/c")), exist_ok=True)
            with open(io.feature_path("motion", "ucf/c"), "w") as f:  # truncated / corrupt file
                f.write("not a zip")
            res["states"] = [output_state("motion", v, cur) for v in ("xd/a", "xd/b", "ucf/c", "xd/zz")]
            res["states_no_check"] = [output_state("motion", v) for v in ("xd/a", "xd/b", "ucf/c")]
        finally:
            io.FEATURE_ROOT = old_root
        assert res["states"] == ["ok", "stale", "stale", "missing"], res["states"]
        assert res["states_no_check"] == ["ok", "ok", "ok"], res["states_no_check"]
    io.FEATURE_ROOT = old_root
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extractor", help="registry name or dotted module path")
    ap.add_argument("--manifest", default="", help="benchmark manifest CSV (e.g. benchmark/videos.csv)")
    ap.add_argument("--video_paths", nargs="*", default=[], help="explicit video files")
    ap.add_argument("--video_list", default="", help="text file with one video path per line")
    ap.add_argument("--source", default=None, help="source name for --video_paths (default: inferred)")
    ap.add_argument("--sources", default="", help="comma-separated sources to keep (xd,ucf,yt,...)")
    ap.add_argument("--video_ids", nargs="*", default=[], help="keep only these ids (<source>/<stem> or <stem>)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--device", default=None, help="cuda:k or cpu (default: the extractor's default)")
    ap.add_argument("--cfg", nargs="*", default=[], help="extractor config overrides key=value")
    ap.add_argument("--overwrite", action="store_true", help="recompute even if outputs exist")
    ap.add_argument("--no_version_check", action="store_true",
                    help="skip existing outputs even when their meta version differs from OUTPUT_VERSIONS")
    ap.add_argument("--skip_failed", action="store_true", help="skip videos listed in failures.jsonl")
    ap.add_argument("--retry_crashed", action="store_true",
                    help="retry videos with a crash record in failures.jsonl (skipped by default)")
    ap.add_argument("--max_consecutive_fail", type=int, default=20,
                    help="exit with status 3 after this many failures in a row (0 = never)")
    ap.add_argument("--keep_order", action="store_true", help="process videos in input order, not sorted by id")
    ap.add_argument("--limit", type=int, default=0, help="process at most this many videos (smoke tests)")
    ap.add_argument("--cpu_threads", type=int, default=0, help="torch / OpenCV thread cap per process (0 = default)")
    ap.add_argument("--feature_root", default="", help="write under this root instead of io.FEATURE_ROOT (tests)")
    ap.add_argument("--dry_run", action="store_true", help="print the shard's work list and exit")
    ap.add_argument("--list", action="store_true", help="list registered extractors and exit")
    ap.add_argument("--selftest", action="store_true", help="test selection errors and version skipping, exit")
    args = ap.parse_args(argv)

    if args.selftest:
        print("run_extract selftest ok", json.dumps(selftest()))
        return 0
    if args.list:
        for k, s in REGISTRY.items():
            print(f"{k:16s} module={s.module} outputs={s.outputs or '(module.EXTRACTOR)'} saver={s.saver or '-'}")
        return 0
    if not args.extractor:
        ap.error("--extractor is required")
    if not (0 <= args.shard < args.num_shards):
        ap.error("need 0 <= shard < num_shards")
    if (args.device or "").startswith("cuda"):
        import torch  # noqa: F401  (import before any decord use, otherwise CUDA init segfaults)
    if args.cpu_threads:
        import cv2
        import torch

        cv2.setNumThreads(args.cpu_threads)
        torch.set_num_threads(args.cpu_threads)

    if args.feature_root:
        io.FEATURE_ROOT = os.path.abspath(args.feature_root)  # io.feature_path reads the module global
    spec, mod, outputs = resolve_spec(args.extractor)
    device = args.device or spec.default_device
    cfg = parse_cfg(args.cfg)
    log_root = os.path.join(io.FEATURE_ROOT, "_logs") if args.feature_root else LOG_ROOT
    log_dir = os.path.join(log_root, args.extractor.replace("/", "_"))
    shard_log = os.path.join(log_dir, f"shard{args.shard}of{args.num_shards}.jsonl")
    fail_log = os.path.join(log_dir, "failures.jsonl")
    current = os.path.join(log_dir, f"shard{args.shard}of{args.num_shards}.current")

    selected = collect_videos(args)
    videos = [v for v in selected if shard_of(v["video_id"], args.num_shards) == args.shard]
    skip_failed = _failed_ids(fail_log) if args.skip_failed else set()
    if not args.retry_crashed:
        skip_failed |= _failed_ids(fail_log, "crash")
    if not args.dry_run and os.path.exists(current):
        with open(current) as f:
            crashed = f.read().strip()
        rec = dict(video_id=crashed, extractor=args.extractor, shard=args.shard, num_shards=args.num_shards,
                   status="crash", time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   error="the previous run of this shard died on this video (native crash, kill or stall); "
                         "skipped until a run with --retry_crashed")
        _append_jsonl(fail_log, rec)
        _append_jsonl(shard_log, rec)
        skip_failed.add(crashed)
        os.remove(current)
    declared = {} if args.no_version_check else dict(getattr(mod, "OUTPUT_VERSIONS", None) or {})
    versions = {o: declared[o] for o in outputs if o in declared}
    todo, n_skip, n_stale = [], 0, 0
    for v in videos:
        if v["video_id"] in skip_failed:
            n_skip += 1
            continue
        if not args.overwrite:
            states = [output_state(o, v["video_id"], versions.get(o)) for o in outputs]
            if all(s == "ok" for s in states):
                n_skip += 1
                continue
            n_stale += "stale" in states
        todo.append(v)
    if args.limit:
        todo = todo[: args.limit]
    print(f"[run_extract] extractor={args.extractor} outputs={list(outputs)} root={io.FEATURE_ROOT} "
          f"device={device} shard={args.shard}/{args.num_shards} selected={len(selected)} videos={len(videos)} "
          f"skip={n_skip} stale={n_stale} todo={len(todo)} versions={versions or '-'} cfg={cfg}", flush=True)
    if args.dry_run:
        for v in todo:
            print(v["video_id"], v["path"])
        return 0

    n_ok = n_fail = n_consec = 0
    vid_sec = wall_sec = 0.0
    t_start = time.time()
    os.makedirs(log_dir, exist_ok=True)
    signal.signal(signal.SIGTERM, _interrupt)
    for i, v in enumerate(todo):
        rec = dict(video_id=v["video_id"], path=v["path"], extractor=args.extractor, shard=args.shard,
                   num_shards=args.num_shards, device=device, host=socket.gethostname(), pid=os.getpid(),
                   time=time.strftime("%Y-%m-%dT%H:%M:%S"))
        t0 = time.time()
        with open(current, "w") as f:
            f.write(v["video_id"])
        try:
            if not os.path.exists(v["path"]):
                raise FileNotFoundError(v["path"])
            paths, dur = run_one(spec, mod, args.extractor, outputs, v, device, cfg)
            rec.update(status="ok", wall_sec=round(time.time() - t0, 3), duration_sec=dur, paths=paths)
            n_ok += 1
            vid_sec += float(dur or 0.0)
            wall_sec += time.time() - t0
            n_consec = 0
        except KeyboardInterrupt:
            os.remove(current)
            raise
        except Exception as e:  # one bad video must never stop the run
            rec.update(status="fail", wall_sec=round(time.time() - t0, 3), error=f"{type(e).__name__}: {e}",
                       traceback=traceback.format_exc(limit=8))
            n_fail += 1
            n_consec += 1
            _append_jsonl(fail_log, rec)
            if "out of memory" in str(e).lower():
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
        os.remove(current)
        _append_jsonl(shard_log, rec)
        speed = vid_sec / max(wall_sec, 1e-9)
        print(f"[{i + 1}/{len(todo)}] {rec['status']:4s} {v['video_id']} {rec['wall_sec']:.1f}s"
              f"{'' if rec['status'] == 'ok' else ' ' + rec['error'][:200]} | ok={n_ok} fail={n_fail} "
              f"speed={speed:.1f}x realtime", flush=True)
        if args.max_consecutive_fail and n_consec >= args.max_consecutive_fail:
            print(f"[run_extract] abort: {n_consec} failures in a row (systemic, e.g. model load or GPU memory); "
                  f"ok={n_ok} fail={n_fail}", flush=True)
            return 3
    total = time.time() - t_start
    print(f"[run_extract] done ok={n_ok} fail={n_fail} skip={n_skip} video_sec={vid_sec:.0f} wall={total:.0f}s "
          f"speed={vid_sec / max(total, 1e-9):.1f}x realtime (failures: {fail_log})", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
