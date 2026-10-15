#!/usr/bin/env python3
"""Download the XD-Violence TRAIN split from the jherng mirror (3,950 of the 3,954 official videos, about 74.6 GB).

Source
    Hugging Face mirror ``jherng/xd-violence``, pinned to commit ``8eba22ca666b5061cf4f3e1b1907c05af520dc07``.
    Provenance: the mirror's card says only ``license: mit``; the text "downloaded from the authors' website"
    is in its loader script ``xd-violence.py`` (``_DESCRIPTION``). Files live under
    ``data/video/<range>/<name>.mp4`` with the same five range folders as the official OneDrive release
    (1-1004, 1005-2004, 2005-2804, 2805-3319, 3320-3954). The mirror's 798 test files that we already have
    locally are byte-identical (name and size for all 798, sha256 spot-checked), so the mirror is the original
    release, not a re-encode. The official OneDrive links answer 403 to scripted requests, the AliyunDrive share
    is taken down and Baidu is disabled for training videos.

Completeness: the mirror is NOT the full official train split
    The authors' own lists (github.com/Roc-Ng/XDVioDet ``list/rgb.list`` and ``list/audio.list``) name 3,954
    train videos; the mirror and its ``train_list.txt`` have 3,950. The 4 absent ones (one violent):
        NewAdd.NBA-2017.12.25_CLE@GSW__#01-08-34_01-40-09_label_A     normal
        Saving.Private.Ryan.1998__#02-29-31_02-30-55_label_B2-G-0     VIOLENT
        v=8cTqh9tMz_I__#1_label_A                                     normal
        v=9eME1y6V-T4__#01-12-00_01-18-00_label_A                     normal
    ``setup/download_xd_recover.py`` recovers what it can from the HIVAU-70k video tar into
    ``recovered_videos/``. The sha256 checks of this script prove the files match the MIRROR's manifest; they
    cannot detect files the mirror never had. ``--verify-only`` therefore also reports completeness against
    ``meta/official_train_list.txt`` when that file exists (written by ``download_xd_recover.py official``).

What it writes (under ``--dest``, default ``data/raw_videos/xd_violence_train``)
    videos/<name>.mp4           flat folder, same naming as ``raw_videos/xd_violence/videos``
    meta/train_list.txt         the mirror's train list (``<range>/<name>`` per line; 3,950 lines, i.e. the
                                official list minus the 4 clips above; it is not the authors' full list)
    meta/test_list.txt          mirror copy of the official test list
    meta/test_annotations.txt   mirror copy of the official test annotations (same as our annotations.txt)
    meta/remote_manifest.json   [{name, subdir, size, sha256, repo_path}] for every train file, from the HF tree API
    meta/test_manifest.json     the same for the mirror's test_videos/ (cached, so reruns need no tree API call)
    train_labels.csv            name, subdir, group, is_youtube, binary, codes (video-level labels parsed from
                                the filename; ``group`` is the text before ``__#``, a movie title or YouTube id;
                                XD train has no frame-level labels, use HIVAU-70k); also lists recovered clips
    verified.jsonl              one JSON line per file whose size and sha256 matched the manifest
    missing_test_videos/        (with --include-missing-test) the 2 official test clips absent from
                                raw_videos/xd_violence/videos; move them there by hand if wanted

Resumability
    Each file is streamed to ``<name>.mp4.part`` and renamed only after its size and LFS sha256 match. A rerun
    skips files listed in ``verified.jsonl`` whose size still matches (or re-hashes a final file that is not
    listed), resumes ``.part`` files with an HTTP Range request, and retries transient errors with backoff.
    Safe to kill at any time. ``--rehash`` also re-hashes listed files and re-downloads any that no longer
    match (the repair for what ``--verify-only`` reports).

Disk guard
    Before starting, before every file and every 64 MB written, aborts if free space on the destination
    filesystem minus the bytes that ALL queued and in-flight files still have to write would drop below
    ``--min-free-gb`` (default 60). The owed-bytes total is a shared counter (``DiskBudget``), so with 8 workers
    the guard still holds if another process fills the disk during the run.

--verify-only
    Read-only: re-hashes every file in ``videos/`` against ``remote_manifest.json``, every file of
    ``missing_test_videos/`` against the cached ``test_manifest.json``, and ``recovered_videos/`` against
    ``recovered_manifest.json``; writes nothing (no labels, no verified.jsonl lines), needs no network, and
    exits 1 if any file is missing or bad. It also prints completeness against the official list.

Usage (from the repo root)
    # foreground smoke test: 3 smallest files
    .venv/bin/python setup/download_xd_train.py --limit 3 --order smallest
    # full download, detached (preferred): see setup/download_xd_train.sh
    bash setup/download_xd_train.sh
    # verify everything already on disk against the manifests (re-hash), no downloads, no writes
    .venv/bin/python setup/download_xd_train.py --verify-only
"""

import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import urllib.parse

import requests

REPO = "jherng/xd-violence"
REVISION = "8eba22ca666b5061cf4f3e1b1907c05af520dc07"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"
TRAIN_SUBDIRS = ["1-1004", "1005-2004", "2005-2804", "2805-3319", "3320-3954"]
META_FILES = ["data/train_list.txt", "data/test_list.txt", "data/test_annotations.txt"]
DATA_ROOT = "data"
DEFAULT_DEST = os.path.join(DATA_ROOT, "raw_videos", "xd_violence_train")
LOCAL_TEST_DIR = os.path.join(DATA_ROOT, "raw_videos", "xd_violence", "videos")
CHUNK = 1 << 20

_log_lock = threading.Lock()


def log(msg):
    with _log_lock:
        print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def file_url(repo_path):
    return BASE_URL + urllib.parse.quote(repo_path, safe="/")


def list_remote(subdirs):
    """Return [{name, subdir, size, sha256, repo_path}] for every mp4 under data/video/<subdir>/."""
    from huggingface_hub import HfApi

    api = HfApi()
    out = []
    for sub in subdirs:
        for f in api.list_repo_tree(REPO, repo_type="dataset", revision=REVISION,
                                    path_in_repo=f"data/video/{sub}", recursive=True):
            if not f.path.endswith(".mp4") or getattr(f, "lfs", None) is None:
                continue
            out.append({"name": os.path.basename(f.path), "subdir": sub, "size": int(f.size),
                        "sha256": f.lfs.sha256, "repo_path": f.path})
    return out


def _cached_listing(path, subdirs, refresh, what):
    """Return the cached HF tree listing at ``path``; list the remote (and cache it) if absent or ``refresh``."""
    if os.path.exists(path) and not refresh:
        with open(path) as f:
            return json.load(f)
    log(f"listing remote {what} files via the HF tree API ...")
    files = list_remote(subdirs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(files, f, indent=0)
    os.replace(tmp, path)
    return files


def load_manifest(dest, refresh):
    """Cached remote listing for the train split (the tree API is slow-ish and paginated)."""
    return _cached_listing(os.path.join(dest, "meta", "remote_manifest.json"), TRAIN_SUBDIRS, refresh, "train")


def load_test_manifest(dest, refresh, allow_fetch=True):
    """Cached remote listing of the mirror's ``test_videos/`` (800 files). Returns [] if absent and not fetchable."""
    path = os.path.join(dest, "meta", "test_manifest.json")
    if not os.path.exists(path) and not allow_fetch:
        return []
    return _cached_listing(path, ["test_videos"], refresh, "test")


def load_recovered(dest):
    """Entries of ``meta/recovered_manifest.json`` (written by download_xd_recover.py), or []."""
    path = os.path.join(dest, "meta", "recovered_manifest.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f).get("files", [])


def load_official_list(dest):
    """The authors' 3,954 train stems (``meta/official_train_list.txt``), or None if not fetched yet."""
    path = os.path.join(dest, "meta", "official_train_list.txt")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def free_bytes(path):
    return shutil.disk_usage(path).free


def sha256_file(path, h=None):
    h = h or hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * CHUNK), b""):
            h.update(block)
    return h


class Verified:
    """Append-only log of verified files; survives restarts."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.names = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        self.names[rec["name"]] = rec

    def add(self, rec):
        with self.lock:
            self.names[rec["name"]] = rec
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")


class DiskBudget:
    """Bytes that every queued and in-flight download still has to write, for a run-wide disk guard.

    ``check`` raises ``RuntimeError("disk guard")`` when free space minus the total still owed would drop below
    ``min_free``. Because our own writes lower free space and the owed total by the same amount, only other
    processes can make the check fail mid-run, and it then fails before the floor is crossed by more than the
    bytes written since the last check (at most 64 MB per worker).
    """

    def __init__(self, path, min_free, owed):
        self.path, self.min_free = path, min_free
        self.owed = {k: max(0, int(v)) for k, v in owed.items()}
        self.total = sum(self.owed.values())
        self.lock = threading.Lock()

    def set(self, key, remaining):
        with self.lock:
            remaining = max(0, int(remaining))
            self.total += remaining - self.owed.get(key, 0)
            self.owed[key] = remaining

    def wrote(self, key, n):
        with self.lock:
            take = min(n, self.owed.get(key, 0))
            self.owed[key] = self.owed.get(key, 0) - take
            self.total -= take

    def finish(self, key):
        self.set(key, 0)

    def check(self):
        with self.lock:
            total = self.total
        free = free_bytes(self.path)
        if free - total < self.min_free:
            raise RuntimeError("disk guard")
        return free - total


class Progress:
    def __init__(self, total_bytes, total_files):
        self.total_bytes, self.total_files = total_bytes, total_files
        self.bytes = 0
        self.files = 0
        self.failed = 0
        self.t0 = time.time()
        self.lock = threading.Lock()
        self.last = 0.0

    def add_bytes(self, n):
        with self.lock:
            self.bytes += n
            now = time.time()
            if now - self.last >= 30:
                self.last = now
                self._report(now)

    def done_file(self, ok):
        with self.lock:
            if ok:
                self.files += 1
            else:
                self.failed += 1

    def _report(self, now):
        el = max(now - self.t0, 1e-6)
        rate = self.bytes / el
        eta = (self.total_bytes - self.bytes) / rate if rate > 0 else float("inf")
        log(f"PROGRESS {self.files}/{self.total_files} files, {self.bytes / 1e9:.2f}/{self.total_bytes / 1e9:.2f} GB "
            f"this run, {rate / 1e6:.1f} MB/s, ETA {eta / 60:.1f} min, failed {self.failed}")


GUARD_EVERY = 64  # blocks of CHUNK bytes between mid-file disk guard checks


def download_one(rec, out_dir, prog, verified, budget, retries=6, timeout=60):
    """Download one file with Range resume and sha256 check. Returns True on success.

    ``budget`` is the run-wide ``DiskBudget``; it raises ``RuntimeError("disk guard")``, which aborts the run.
    """
    final = os.path.join(out_dir, rec["name"])
    part = final + ".part"
    key = final
    try:
        for attempt in range(1, retries + 1):
            try:
                have = os.path.getsize(part) if os.path.exists(part) else 0
                if have > rec["size"]:
                    os.remove(part)
                    have = 0
                budget.set(key, rec["size"] - have)
                budget.check()
                h = sha256_file(part) if have else hashlib.sha256()
                if have < rec["size"]:
                    headers = {"Range": f"bytes={have}-"} if have else {}
                    with requests.get(file_url(rec["repo_path"]), headers=headers, stream=True,
                                      timeout=timeout) as r:
                        if have and r.status_code != 206:
                            # Server ignored the range: restart from scratch.
                            have = 0
                            h = hashlib.sha256()
                            mode = "wb"
                            budget.set(key, rec["size"])
                        else:
                            mode = "ab" if have else "wb"
                        r.raise_for_status()
                        with open(part, mode) as f:
                            for k, block in enumerate(r.iter_content(CHUNK), 1):
                                if block:
                                    f.write(block)
                                    h.update(block)
                                    prog.add_bytes(len(block))
                                    budget.wrote(key, len(block))
                                    if k % GUARD_EVERY == 0:
                                        budget.check()
                size = os.path.getsize(part)
                if size != rec["size"] or h.hexdigest() != rec["sha256"]:
                    log(f"WARNING checksum/size mismatch for {rec['name']} (size {size} vs {rec['size']}); restarting")
                    os.remove(part)
                    raise RuntimeError("checksum mismatch")
                os.replace(part, final)
                verified.add({"name": rec["name"], "subdir": rec["subdir"], "size": rec["size"],
                              "sha256": rec["sha256"]})
                return True
            except RuntimeError as e:
                if str(e) == "disk guard":
                    log(f"ABORT disk guard: free space minus bytes still owed by all queued files would drop "
                        f"below {budget.min_free / 1e9:.0f} GB")
                    raise
                wait = min(60, 2 ** attempt)
                log(f"retry {attempt}/{retries} {rec['name']}: {e}; sleeping {wait}s")
                time.sleep(wait)
            except (requests.RequestException, OSError) as e:
                wait = min(60, 2 ** attempt)
                log(f"retry {attempt}/{retries} {rec['name']}: {type(e).__name__}: {e}; sleeping {wait}s")
                time.sleep(wait)
        log(f"FAILED {rec['name']} after {retries} attempts")
        return False
    finally:
        budget.finish(key)  # done, failed or aborted: this file no longer owes bytes to the guard


def fetch_small(repo_path, out_path):
    r = requests.get(file_url(repo_path), timeout=60)
    r.raise_for_status()
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(r.content)
    os.replace(tmp, out_path)


def parse_name(name):
    """Video-level labels from an XD filename, e.g. ``Bad.Boys.1995__#01-11-55_01-12-40_label_G-B2-B6.mp4``."""
    stem = name[:-4] if name.endswith(".mp4") else name
    group = stem.split("__#")[0] if "__#" in stem else stem.split("__")[0]
    codes = [c for c in stem.rsplit("_label_", 1)[-1].split("-") if c and c != "0"]
    binary = 0 if codes == ["A"] else 1
    return {"group": group, "is_youtube": int(group.startswith("v=")), "binary": binary, "codes": "-".join(codes)}


def write_labels(dest, manifest, extra=()):
    """Write train_labels.csv for the mirror manifest plus ``extra`` records (recovered clips)."""
    path = os.path.join(dest, "train_labels.csv")
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "subdir", "group", "is_youtube", "binary", "codes"])
        for rec in sorted(list(manifest) + list(extra), key=lambda r: r["name"]):
            p = parse_name(rec["name"])
            w.writerow([rec["name"][:-4], rec["subdir"], p["group"], p["is_youtube"], p["binary"], p["codes"]])
    os.replace(path + ".tmp", path)
    return path


def missing_test_records(test_manifest):
    """Official test clips the mirror has but ``raw_videos/xd_violence/videos`` lacks (2 as of 2026-10-14)."""
    local = set(os.listdir(LOCAL_TEST_DIR)) if os.path.isdir(LOCAL_TEST_DIR) else set()
    return [r for r in test_manifest if r["name"] not in local]


def classify(records, folder, verified, rehash, record_new):
    """Check ``records`` against the files in ``folder``.

    A file is OK if it has the manifest size and either is listed in ``verified.jsonl`` with that size (skipped
    unless ``rehash``) or its sha256 matches. With ``record_new``, newly hashed OK files are appended to
    ``verified.jsonl`` (never in verify-only mode). Returns ``(n_ok, todo, problems)`` where ``todo`` are the
    records that need (re)downloading and ``problems`` are ``(name, reason)`` pairs.
    """
    n_ok, todo, problems = 0, [], []
    for rec in records:
        final = os.path.join(folder, rec["name"])
        if not os.path.exists(final):
            todo.append(rec)
            problems.append((rec["name"], "missing"))
            continue
        size = os.path.getsize(final)
        if size != rec["size"]:
            todo.append(rec)
            problems.append((rec["name"], f"size {size} != {rec['size']}"))
            continue
        listed = verified.names.get(rec["name"])
        if listed is not None and listed.get("size") == rec["size"] and not rehash:
            n_ok += 1
            continue
        if sha256_file(final).hexdigest() == rec["sha256"]:
            n_ok += 1
            if record_new and listed is None:
                verified.add({"name": rec["name"], "subdir": rec["subdir"], "size": rec["size"],
                              "sha256": rec["sha256"]})
            continue
        todo.append(rec)
        problems.append((rec["name"], "sha256 differs"))
    return n_ok, todo, problems


def verify_only(dest, manifest, verified):
    """Re-hash everything on disk against the cached manifests; write nothing. Returns the exit code."""
    out_dir = os.path.join(dest, "videos")
    report = []
    n_ok, _, problems = classify(manifest, out_dir, verified, rehash=True, record_new=False)
    report.append(("videos/ vs remote_manifest.json", len(manifest), n_ok, problems))

    extra_dir = os.path.join(dest, "missing_test_videos")
    test_manifest = load_test_manifest(dest, refresh=False, allow_fetch=False)
    if test_manifest:
        by_name = {r["name"]: r for r in test_manifest}
        present = set(os.listdir(extra_dir)) if os.path.isdir(extra_dir) else set()
        expect = {r["name"] for r in missing_test_records(test_manifest)} | {n for n in present if n in by_name}
        recs = [by_name[n] for n in sorted(expect)]
        n_ok, _, problems = classify(recs, extra_dir, verified, rehash=True, record_new=False)
        problems += [(n, "not an official test clip") for n in sorted(present) if n.endswith(".mp4")
                     and n not in by_name]
        report.append(("missing_test_videos/ vs test_manifest.json", len(recs), n_ok, problems))
    elif os.path.isdir(extra_dir):
        log("NOTE meta/test_manifest.json not cached; missing_test_videos/ not checked "
            "(run once without --verify-only, or with --include-missing-test, to cache it)")

    recovered = load_recovered(dest)
    if recovered:
        n_ok, _, problems = classify(recovered, os.path.join(dest, "recovered_videos"), verified, rehash=True,
                                     record_new=False)
        report.append(("recovered_videos/ vs recovered_manifest.json (sha256 computed locally)", len(recovered),
                       n_ok, problems))

    bad = 0
    for label, n, ok, problems in report:
        log(f"verify-only {label}: {ok}/{n} OK, {len(problems)} missing or bad")
        for name, why in problems:
            log(f"  BAD {name}: {why}")
        bad += len(problems)
    if bad:
        log("to repair: rerun without --verify-only and with --rehash (add --include-missing-test for "
            "missing_test_videos/; setup/download_xd_recover.py fetch for recovered_videos/)")

    official = load_official_list(dest)
    if official is None:
        log("completeness: meta/official_train_list.txt absent (run setup/download_xd_recover.py official); "
            "the checks above only prove the files match the MIRROR, which lacks 4 official train videos")
    else:
        have = {r["name"][:-4] for r in manifest} | {r["name"][:-4] for r in recovered}
        absent = [n for n in official if n not in have]
        log(f"completeness vs the authors' official train list: {len(official) - len(absent)}/{len(official)} "
            f"official train videos are covered by the manifests checked above (mirror + recovered); "
            f"absent: {absent}")
    return 0 if bad == 0 else 1


def run(args):
    dest = os.path.abspath(args.dest)
    out_dir = os.path.join(dest, "videos")
    min_free = int(args.min_free_gb * 1e9)

    if args.verify_only:
        # Read-only: no network, no labels, no verified.jsonl lines.
        manifest_path = os.path.join(dest, "meta", "remote_manifest.json")
        if not os.path.exists(manifest_path):
            log(f"verify-only: {manifest_path} does not exist; nothing to verify")
            return 1
        manifest = load_manifest(dest, refresh=False)
        log(f"manifest: {len(manifest)} train files, {sum(r['size'] for r in manifest) / 1e9:.2f} GB")
        return verify_only(dest, manifest, Verified(os.path.join(dest, "verified.jsonl")))

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(dest, "meta"), exist_ok=True)
    for repo_path in META_FILES:
        out = os.path.join(dest, "meta", os.path.basename(repo_path))
        if not os.path.exists(out):
            fetch_small(repo_path, out)
    manifest = load_manifest(dest, args.refresh_manifest)
    log(f"manifest: {len(manifest)} train files, {sum(r['size'] for r in manifest) / 1e9:.2f} GB")
    labels = write_labels(dest, manifest, load_recovered(dest))
    log(f"wrote {labels}")

    verified = Verified(os.path.join(dest, "verified.jsonl"))
    _, todo, problems = classify(manifest, out_dir, verified, rehash=args.rehash, record_new=True)
    for name, why in problems:
        if why != "missing":
            log(f"WARNING {name} on disk is bad ({why}); downloading it again to .part")
    jobs = [(rec, out_dir) for rec in todo]
    if args.include_missing_test:
        extra_dir = os.path.join(dest, "missing_test_videos")
        os.makedirs(extra_dir, exist_ok=True)
        test_manifest = load_test_manifest(dest, args.refresh_manifest)
        _, extra_todo, problems = classify(missing_test_records(test_manifest), extra_dir, verified,
                                           rehash=args.rehash, record_new=True)
        for name, why in problems:
            if why != "missing":
                log(f"WARNING missing_test_videos/{name} is bad ({why}); downloading it again to .part")
        jobs = [(rec, extra_dir) for rec in extra_todo] + jobs

    if args.order == "smallest":
        jobs.sort(key=lambda j: j[0]["size"])
    elif args.order == "largest":
        jobs.sort(key=lambda j: -j[0]["size"])
    if args.limit:
        jobs = jobs[: args.limit]

    def part_size(rec, d):
        p = os.path.join(d, rec["name"] + ".part")
        return min(os.path.getsize(p), rec["size"]) if os.path.exists(p) else 0

    owed = {os.path.join(d, r["name"]): r["size"] - part_size(r, d) for r, d in jobs}
    budget = DiskBudget(out_dir, min_free, owed)
    need = budget.total
    free = free_bytes(out_dir)
    log(f"to download: {len(jobs)} files, {need / 1e9:.2f} GB; free {free / 1e9:.1f} GB; "
        f"guard {args.min_free_gb:.0f} GB; workers {args.workers}")
    if free - need < min_free:
        log(f"ABORT: free space after download would be {(free - need) / 1e9:.1f} GB < {args.min_free_gb:.0f} GB")
        return 2
    if not jobs:
        log("nothing to do; all files verified")
        return 0

    prog = Progress(need, len(jobs))
    abort = threading.Event()

    def worker(job):
        if abort.is_set():
            budget.finish(os.path.join(job[1], job[0]["name"]))
            return False
        rec, d = job
        try:
            ok = download_one(rec, d, prog, verified, budget)
        except RuntimeError:
            abort.set()
            ok = False
        prog.done_file(ok)
        return ok

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(worker, jobs))
    prog._report(time.time())
    n_ok = sum(results)
    log(f"DONE this run: {n_ok}/{len(jobs)} files OK, {len(jobs) - n_ok} failed"
        + (" (aborted by disk guard)" if abort.is_set() else ""))
    total_ok = sum(1 for r in manifest if r["name"] in verified.names)
    log(f"verified on disk: {total_ok}/{len(manifest)} mirror train files (the official list has 3,954; "
        f"see setup/download_xd_recover.py)")
    return 0 if n_ok == len(jobs) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-free-gb", type=float, default=60.0)
    ap.add_argument("--limit", type=int, default=0, help="download at most N files (smoke tests)")
    ap.add_argument("--order", choices=["manifest", "smallest", "largest"], default="manifest")
    ap.add_argument("--include-missing-test", action="store_true",
                    help="also fetch the official test clips missing from raw_videos/xd_violence/videos")
    ap.add_argument("--verify-only", action="store_true",
                    help="re-hash files on disk against the cached manifests; read-only, no network")
    ap.add_argument("--refresh-manifest", action="store_true", help="re-list the remote repo")
    ap.add_argument("--rehash", action="store_true",
                    help="re-hash files already in verified.jsonl and re-download any that no longer match "
                         "(repairs what --verify-only reports as BAD)")
    args = ap.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
