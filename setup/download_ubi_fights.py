#!/usr/bin/env python3
"""Download, verify and extract UBI-Fights (Degardin and Proenca, IJCB 2020): 1,000 videos, 80 h, 8.1 GB zip.

Source
    https://socia-lab.di.ubi.pt/EventDetection/UBI_FIGHTS.zip (direct link, no form). The server's TLS
    certificate is expired, so certificate verification is disabled for this one host (equivalent of
    ``curl -k``); every zip member's CRC-32 is checked during extraction instead.
    The server caps a single connection at about 2 MB/s, so the file is fetched as fixed-size byte-range
    chunks over ``--workers`` parallel connections (8 connections measured at about 4.8 MB/s total).

Integrity versus authenticity
    The CRC-32 values come from the zip's own central directory, which arrived over the same unverified TLS
    connection. A CRC match therefore proves TRANSFER CONSISTENCY (no corrupted or truncated bytes), not
    AUTHENTICITY: a man-in-the-middle who substitutes the zip also supplies matching CRCs, and CRC-32 is not a
    cryptographic hash. The publisher offers no checksum. To make later comparisons strong, ``manifest.json``
    also records the sha256 of every extracted file (computed locally at extraction, or backfilled with
    ``--backfill-sha256``, which first re-checks each file against its CRC-32); ``--verify-only`` re-hashes
    the extracted tree against it, and any other copy of UBI-Fights can be compared file by file.

Zip layout (2,006 members, 8.27 GB uncompressed)
    UBI_FIGHTS/videos/fight/F_<id>_<flags>.mp4    216 fight videos (640x360, 30 fps, no audio stream)
    UBI_FIGHTS/videos/normal/N_<id>_<flags>.mp4   784 normal videos
    UBI_FIGHTS/annotation/<stem>.csv               1,000 per-frame label files, one 0/1 value per line
    UBI_FIGHTS/test_videos.csv                     the authors' test split (one stem per line)

What it writes
    downloads/ubi_fights/UBI_FIGHTS.zip (+ .state.json, .chunks)   the archive while it is being fetched
    raw_videos/ubi_fights/videos/{fight,normal}/*.mp4               extracted, ``UBI_FIGHTS/`` prefix stripped
    raw_videos/ubi_fights/annotation/*.csv
    raw_videos/ubi_fights/test_videos.csv
    raw_videos/ubi_fights/manifest.json      [{name, size, crc32, sha256}] of every extracted file
    raw_videos/ubi_fights/segments.csv       per video: label, flags, split, n_label_frames, violent segments
                                             (frame ranges, end exclusive), first onset in frames and in
                                             seconds at 30 fps, plus (ffprobe header) video_frames,
                                             video_duration_s, has_audio, frames_match
    The archive is deleted after every member was extracted and CRC-verified, unless ``--keep-archive``.

Resumability
    The archive is preallocated and each chunk is written at its offset with ``os.pwrite`` and flushed to disk
    (``fdatasync``) before its index is appended to ``UBI_FIGHTS.zip.chunks``, so a chunk logged as done is
    durable even after a power loss. A rerun downloads only missing chunks. If the remote size or ETag
    changes, the script stops and asks for ``--restart``. Extraction skips members already present with the
    right size and listed in ``.extracted.jsonl``. If a member fails its CRC, the chunks covering it are
    re-downloaded once and extraction is retried. If the zip cannot even be opened (corrupt end-of-central-
    directory or central directory), the chunks covering the central directory (located from the EOCD /
    zip64 records when readable, else the last two chunks) are re-downloaded, up to three rounds, before
    giving up and asking for ``--restart``.

Disk guard
    Aborts (before and during the download) if free space minus the bytes still needed (remaining archive
    bytes plus the uncompressed size) would drop below ``--min-free-gb`` (default 60).

Usage (from the repo root)
    bash setup/download_ubi_fights.sh                       # detached full run (preferred)
    .venv/bin/python setup/download_ubi_fights.py           # foreground full run
    .venv/bin/python setup/download_ubi_fights.py --max-chunks 2 --no-extract   # smoke test
    .venv/bin/python setup/download_ubi_fights.py --summarize-only               # rebuild segments.csv
    .venv/bin/python setup/download_ubi_fights.py --verify-only    # re-hash extracted files (read-only)
    .venv/bin/python setup/download_ubi_fights.py --backfill-sha256   # add sha256 to an old manifest.json
    .venv/bin/python setup/download_selftest.py                    # offline tests of the recovery logic
"""

import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
import zipfile
import zlib

import requests
import urllib3

URL = "https://socia-lab.di.ubi.pt/EventDetection/UBI_FIGHTS.zip"
DATA_ROOT = "data"
DEFAULT_ARCHIVE_DIR = os.path.join(DATA_ROOT, "downloads", "ubi_fights")
DEFAULT_DEST = os.path.join(DATA_ROOT, "raw_videos", "ubi_fights")
ZIP_PREFIX = "UBI_FIGHTS/"
UNCOMPRESSED_EST = 8_269_152_825  # sum of member sizes, read from the central directory on 2026-10-14
FPS = 30.0

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
_log_lock = threading.Lock()


def log(msg):
    with _log_lock:
        print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


_datasync = getattr(os, "fdatasync", os.fsync)


def remote_info(url):
    r = requests.head(url, verify=False, timeout=60, allow_redirects=True)
    r.raise_for_status()
    return {"url": url, "size": int(r.headers["Content-Length"]), "etag": r.headers.get("ETag"),
            "last_modified": r.headers.get("Last-Modified"),
            "accept_ranges": r.headers.get("Accept-Ranges")}


class ChunkedDownload:
    """Parallel byte-range download into a preallocated file with a persistent chunk log."""

    def __init__(self, url, path, chunk_mb, workers, min_free, restart=False):
        self.url, self.path, self.workers, self.min_free = url, path, workers, min_free
        self.state_path = path + ".state.json"
        self.chunks_path = path + ".chunks"
        self.lock = threading.Lock()
        info = remote_info(url)
        if info["accept_ranges"] != "bytes":
            raise SystemExit("server does not advertise byte ranges; cannot resume safely")
        if os.path.exists(self.state_path) and not restart:
            with open(self.state_path) as f:
                state = json.load(f)
            if state["size"] != info["size"] or state["etag"] != info["etag"]:
                raise SystemExit(f"remote file changed (size/etag {state['size']}/{state['etag']} -> "
                                 f"{info['size']}/{info['etag']}); rerun with --restart")
            self.chunk = state["chunk_size"]
        else:
            for p in (self.chunks_path, self.path):
                if os.path.exists(p):
                    os.remove(p)
            self.chunk = int(chunk_mb * (1 << 20))
            state = dict(info, chunk_size=self.chunk, created=time.strftime("%Y-%m-%d %H:%M:%S"))
            with open(self.state_path + ".tmp", "w") as f:
                json.dump(state, f, indent=1)
            os.replace(self.state_path + ".tmp", self.state_path)
        self.size = info["size"]
        self.n_chunks = (self.size + self.chunk - 1) // self.chunk
        self.done = set()
        if os.path.exists(self.chunks_path):
            with open(self.chunks_path) as f:
                self.done = {int(x) for x in f.read().split() if x.strip()}
        if not os.path.exists(self.path):
            with open(self.path, "wb") as f:
                f.truncate(self.size)
        elif os.path.getsize(self.path) != self.size:
            with open(self.path, "r+b") as f:
                f.truncate(self.size)

    def chunk_range(self, i):
        start = i * self.chunk
        return start, min(start + self.chunk, self.size) - 1

    def remaining_bytes(self):
        with self.lock:
            done = set(self.done)
        return sum(self.chunk_range(i)[1] - self.chunk_range(i)[0] + 1
                   for i in range(self.n_chunks) if i not in done)

    def is_complete(self):
        return len(self.done) == self.n_chunks

    def _mark(self, i):
        with self.lock:
            self.done.add(i)
            with open(self.chunks_path, "a") as f:
                f.write(f"{i}\n")

    def forget(self, indices):
        """Mark chunks as not done (used after a CRC failure) and rewrite the chunk log."""
        with self.lock:
            self.done -= set(indices)
            with open(self.chunks_path + ".tmp", "w") as f:
                f.write("".join(f"{i}\n" for i in sorted(self.done)))
            os.replace(self.chunks_path + ".tmp", self.chunks_path)

    def chunks_for_range(self, start, end):
        return list(range(start // self.chunk, min(end // self.chunk, self.n_chunks - 1) + 1))

    def _fetch(self, i, fd, retries=8):
        start, end = self.chunk_range(i)
        want = end - start + 1
        for attempt in range(1, retries + 1):
            try:
                r = requests.get(self.url, headers={"Range": f"bytes={start}-{end}"}, verify=False,
                                 timeout=(30, 120))
                if r.status_code != 206:
                    raise IOError(f"HTTP {r.status_code} for range request")
                data = r.content
                if len(data) != want:
                    raise IOError(f"short read {len(data)}/{want}")
                os.pwrite(fd, data, start)
                _datasync(fd)  # the chunk must be on disk before the log says it is done
                self._mark(i)
                return want
            except (requests.RequestException, IOError, OSError) as e:
                wait = min(120, 2 ** attempt)
                log(f"retry {attempt}/{retries} chunk {i}: {type(e).__name__}: {e}; sleeping {wait}s")
                time.sleep(wait)
        log(f"FAILED chunk {i}")
        return 0

    def run(self, max_chunks=0, extra_needed=0):
        todo = [i for i in range(self.n_chunks) if i not in self.done]
        if max_chunks:
            todo = todo[:max_chunks]
        need = sum(self.chunk_range(i)[1] - self.chunk_range(i)[0] + 1 for i in todo)
        log(f"archive {self.size / 1e9:.3f} GB in {self.n_chunks} chunks of {self.chunk >> 20} MB; "
            f"{len(self.done)} done, {len(todo)} to fetch ({need / 1e9:.2f} GB) with {self.workers} workers")
        if not todo:
            return True
        t0 = time.time()
        got = [0]
        abort = threading.Event()
        fd = os.open(self.path, os.O_WRONLY)
        last = [t0]

        def work(i):
            if abort.is_set():
                return 0
            free = shutil.disk_usage(os.path.dirname(self.path)).free
            if free - self.remaining_bytes() - extra_needed < self.min_free:
                log(f"ABORT disk guard: free {free / 1e9:.1f} GB would drop below {self.min_free / 1e9:.0f} GB")
                abort.set()
                return 0
            n = self._fetch(i, fd)
            with self.lock:
                got[0] += n
                now = time.time()
                if now - last[0] >= 30:
                    last[0] = now
                    rate = got[0] / max(now - t0, 1e-6)
                    eta = (need - got[0]) / rate if rate > 0 else float("inf")
                    log(f"PROGRESS {len(self.done)}/{self.n_chunks} chunks, {got[0] / 1e9:.2f}/{need / 1e9:.2f} GB "
                        f"this run, {rate / 1e6:.2f} MB/s, ETA {eta / 60:.1f} min")
            return n

        try:
            with cf.ThreadPoolExecutor(max_workers=self.workers) as ex:
                list(ex.map(work, todo))
        finally:
            os.fsync(fd)
            os.close(fd)
        rate = got[0] / max(time.time() - t0, 1e-6)
        log(f"fetched {got[0] / 1e9:.2f} GB at {rate / 1e6:.2f} MB/s; chunks done {len(self.done)}/{self.n_chunks}")
        return not abort.is_set()


def safe_target(dest, name):
    """Map a zip member to a path under ``dest``; refuse absolute paths and ``..`` components."""
    rel = name[len(ZIP_PREFIX):] if name.startswith(ZIP_PREFIX) else name
    parts = [p for p in rel.split("/") if p]
    if not parts or name.startswith("/") or any(p in ("..", ".") for p in parts) or "\\" in rel:
        return None
    return os.path.join(dest, *parts)


def extract_member(z, info, target):
    """Extract one member through ``.part``; checks size and CRC-32. Returns the sha256 hex of the content."""
    tmp = target + ".part"
    os.makedirs(os.path.dirname(target), exist_ok=True)
    crc = 0
    sha = hashlib.sha256()
    with z.open(info) as src, open(tmp, "wb") as dst:  # ZipExtFile raises BadZipFile on CRC mismatch
        for block in iter(lambda: src.read(1 << 22), b""):
            dst.write(block)
            crc = zlib.crc32(block, crc)
            sha.update(block)
    if os.path.getsize(tmp) != info.file_size or crc != info.CRC:
        os.remove(tmp)
        raise zipfile.BadZipFile(f"size/CRC mismatch for {info.filename}")
    os.replace(tmp, target)
    return sha.hexdigest()


def hash_file(path):
    """``(crc32 as 8 hex chars, sha256 hex)`` of a file, read once."""
    crc, sha = 0, hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            crc = zlib.crc32(block, crc)
            sha.update(block)
    return f"{crc:08x}", sha.hexdigest()


def central_directory_span(path, size):
    """Byte offset where the central directory starts, read from the EOCD (and zip64) records at the tail.

    Returns None when the records are unreadable or point outside the file (e.g. a zero-filled tail).
    """
    tail_len = min(size, (1 << 16) + 22)
    with open(path, "rb") as f:
        f.seek(size - tail_len)
        tail = f.read(tail_len)
        i = tail.rfind(b"PK\x05\x06")
        if i < 0 or len(tail) - i < 22:
            return None
        cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
        if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
            j = i - 20
            if j < 0 or tail[j:j + 4] != b"PK\x06\x07":
                return None
            z64 = struct.unpack("<Q", tail[j + 8:j + 16])[0]
            if z64 >= size:
                return None
            f.seek(z64)
            rec = f.read(56)
            if len(rec) < 56 or rec[:4] != b"PK\x06\x06":
                return None
            cd_size, cd_off = struct.unpack("<QQ", rec[40:56])
    if not 0 <= cd_off < size or cd_off + cd_size > size:
        return None
    return cd_off


def open_archive(archive, dl=None, rounds=3):
    """``zipfile.ZipFile(archive)``; if the central directory is corrupt, re-fetch the chunks that hold it.

    A chunk can be logged as done but hold bad bytes (e.g. a disk fault, or a log written by an older version
    of this script before a power loss). Member-level CRC failures are repaired in ``extract_all``; this
    handles the case where the zip cannot be opened at all, which would otherwise fail identically on every
    rerun until ``--restart`` re-fetched the whole archive.
    """
    for attempt in range(1, rounds + 1):
        try:
            return zipfile.ZipFile(archive)
        except (zipfile.BadZipFile, zipfile.LargeZipFile, struct.error, ValueError, OSError) as e:
            if dl is None or attempt == rounds:
                raise zipfile.BadZipFile(f"cannot open {archive} after {attempt} attempt(s): {e}; "
                                         f"rerun with --restart") from e
            start = central_directory_span(archive, dl.size)
            if start is None:
                start = max(0, dl.size - 2 * dl.chunk)
            idx = dl.chunks_for_range(start, dl.size - 1)
            log(f"cannot open the zip ({type(e).__name__}: {e}); re-downloading central-directory chunks "
                f"{idx[0]}..{idx[-1]} (round {attempt})")
            dl.forget(idx)
            if not dl.run():
                raise zipfile.BadZipFile("re-download of the central directory aborted (disk guard)") from e


def member_byte_range(z, info):
    """Byte span of a member (local header + data + optional descriptor) inside the archive."""
    start = info.header_offset
    end = start + 30 + len(info.filename.encode()) + len(info.extra) + info.compress_size + 24 + 1024
    return start, end


def extract_all(archive, dest, dl=None):
    """Extract every member with CRC verification; resumable via ``.extracted.jsonl``. Returns list of failures."""
    os.makedirs(dest, exist_ok=True)
    log_path = os.path.join(dest, ".extracted.jsonl")
    done = {}
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done[rec["name"]] = rec
    failures = []
    with open_archive(archive, dl) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        log(f"extracting {len(infos)} members to {dest} ({sum(i.file_size for i in infos) / 1e9:.2f} GB)")
        t0 = time.time()
        for k, info in enumerate(infos, 1):
            target = safe_target(dest, info.filename)
            if target is None:
                log(f"WARNING skipping unsafe member name {info.filename!r}")
                continue
            rel = os.path.relpath(target, dest)
            if rel in done and os.path.exists(target) and os.path.getsize(target) == info.file_size:
                if done[rel].get("sha256"):
                    continue
                crc, sha = hash_file(target)  # record from an older run without sha256: add it now
                if crc == f"{info.CRC:08x}":
                    done[rel] = dict(done[rel], sha256=sha)
                    with open(log_path, "a") as f:
                        f.write(json.dumps(done[rel]) + "\n")
                    continue
                log(f"WARNING {rel} on disk fails its CRC; extracting it again")
            ok = False
            sha = None
            for attempt in (1, 2):
                try:
                    sha = extract_member(z, info, target)
                    ok = True
                    break
                except (zipfile.BadZipFile, zlib.error, EOFError) as e:
                    if attempt == 2 or dl is None:
                        log(f"FAILED {info.filename}: {e}")
                        failures.append(info.filename)
                        break
                    s, e_ = member_byte_range(z, info)
                    idx = dl.chunks_for_range(s, min(e_, dl.size - 1))
                    log(f"CRC failure in {info.filename}; re-downloading chunks {idx[0]}..{idx[-1]}")
                    dl.forget(idx)
                    dl.run()
            if not ok:
                continue
            rec = {"name": rel, "size": info.file_size, "crc32": f"{info.CRC:08x}", "sha256": sha}
            done[rel] = rec
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if k % 100 == 0:
                log(f"extracted {k}/{len(infos)} members ({time.time() - t0:.0f}s)")
        expected = {os.path.relpath(safe_target(dest, i.filename), dest) for i in infos
                    if safe_target(dest, i.filename)}
    manifest = [done[n] for n in sorted(expected) if n in done]
    write_manifest(dest, manifest)
    log(f"extraction: {len(manifest)}/{len(expected)} members verified, {len(failures)} failures")
    return failures if len(manifest) == len(expected) else failures + sorted(expected - set(done))


def write_manifest(dest, manifest):
    path = os.path.join(dest, "manifest.json")
    with open(path + ".tmp", "w") as f:
        json.dump(manifest, f, indent=0)
    os.replace(path + ".tmp", path)


def load_manifest(dest):
    with open(os.path.join(dest, "manifest.json")) as f:
        return json.load(f)


def verify_extracted(dest):
    """Re-hash every file in ``manifest.json`` (size, CRC-32 and sha256 when recorded). Read-only.

    Returns ``(n_ok, problems)`` with ``problems`` a list of ``(name, reason)``.
    """
    manifest = load_manifest(dest)
    n_ok, problems = 0, []
    for rec in manifest:
        path = os.path.join(dest, rec["name"])
        if not os.path.exists(path):
            problems.append((rec["name"], "missing"))
            continue
        if os.path.getsize(path) != rec["size"]:
            problems.append((rec["name"], f"size {os.path.getsize(path)} != {rec['size']}"))
            continue
        crc, sha = hash_file(path)
        if crc != rec["crc32"]:
            problems.append((rec["name"], f"crc32 {crc} != {rec['crc32']}"))
        elif rec.get("sha256") and sha != rec["sha256"]:
            problems.append((rec["name"], "sha256 differs"))
        else:
            n_ok += 1
    n_sha = sum(bool(r.get("sha256")) for r in manifest)
    log(f"verify-only: {n_ok}/{len(manifest)} extracted files OK ({n_sha} with a recorded sha256), "
        f"{len(problems)} problems")
    for name, why in problems:
        log(f"  BAD {name}: {why}")
    return n_ok, problems


def backfill_sha256(dest):
    """Add ``sha256`` to every ``manifest.json`` entry from the extracted file, after re-checking its CRC-32.

    Refuses (exit code 1, manifest untouched) if any file is missing or fails its size or CRC check, because a
    sha256 recorded for a file that no longer matches the zip would anchor the wrong content.
    """
    manifest = load_manifest(dest)
    problems, out = [], []
    for rec in manifest:
        path = os.path.join(dest, rec["name"])
        if not os.path.exists(path) or os.path.getsize(path) != rec["size"]:
            problems.append((rec["name"], "missing or wrong size"))
            continue
        crc, sha = hash_file(path)
        if crc != rec["crc32"]:
            problems.append((rec["name"], f"crc32 {crc} != {rec['crc32']}"))
            continue
        if rec.get("sha256") and rec["sha256"] != sha:
            problems.append((rec["name"], "sha256 differs from the recorded one"))
            continue
        out.append(dict(rec, sha256=sha))
    if problems:
        for name, why in problems:
            log(f"  BAD {name}: {why}")
        log(f"backfill refused: {len(problems)} problems; manifest.json unchanged")
        return 1
    write_manifest(dest, out)
    log(f"backfill: sha256 recorded for {len(out)}/{len(manifest)} files in {dest}/manifest.json")
    return 0


def segments_from_labels(vals):
    """Contiguous runs of 1 as (start, end_exclusive) frame pairs."""
    segs, start = [], None
    for i, v in enumerate(vals):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(vals)))
    return segs


def probe_video(path):
    """Header-only ffprobe (no decoding): video frame count, video-stream duration, audio presence."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,nb_frames,duration",
                          "-of", "json", path], capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        return {"video_frames": -1, "video_duration_s": -1, "has_audio": -1}
    streams = json.loads(out.stdout).get("streams", [])
    v = next((x for x in streams if x.get("codec_type") == "video"), {})
    frames = int(v["nb_frames"]) if str(v.get("nb_frames", "")).isdigit() else -1
    dur = round(float(v["duration"]), 3) if v.get("duration") not in (None, "N/A") else -1
    return {"video_frames": frames, "video_duration_s": dur,
            "has_audio": int(any(x.get("codec_type") == "audio" for x in streams))}


def summarize(dest, probe=True):
    """Write ``segments.csv``: per-video violent segments and first onset from the per-frame CSVs.

    With ``probe`` the video header is read with ffprobe to add the frame count, duration and audio presence,
    and ``frames_match`` flags whether the label CSV has the same number of rows as the video has frames
    (within 1).
    """
    ann_dir = os.path.join(dest, "annotation")
    test = set()
    tpath = os.path.join(dest, "test_videos.csv")
    if os.path.exists(tpath):
        with open(tpath) as f:
            test = {line.strip() for line in f if line.strip()}
    rows = []
    for fn in sorted(os.listdir(ann_dir)):
        if not fn.endswith(".csv"):
            continue
        stem = fn[:-4]
        with open(os.path.join(ann_dir, fn)) as f:
            vals = [int(float(x)) for x in f.read().split() if x.strip()]
        segs = segments_from_labels(vals)
        parts = stem.split("_")
        onset = segs[0][0] if segs else -1
        rows.append({
            "name": stem, "label": parts[0], "id": parts[1], "flags": "_".join(parts[2:]),
            "split": "test" if stem in test else "train", "n_label_frames": len(vals),
            "label_duration_s": round(len(vals) / FPS, 2), "n_violent_frames": int(sum(vals)),
            "segments_frames": ";".join(f"{s}-{e}" for s, e in segs),
            "onset_frame": onset, "onset_s": round(onset / FPS, 3) if onset >= 0 else -1,
        })
        if probe:
            sub = "fight" if parts[0] == "F" else "normal"
            vpath = os.path.join(dest, "videos", sub, stem + ".mp4")
            info = probe_video(vpath) if os.path.exists(vpath) else {"video_frames": -1, "video_duration_s": -1,
                                                                      "has_audio": -1}
            info["frames_match"] = int(info["video_frames"] >= 0 and abs(info["video_frames"] - len(vals)) <= 1)
            rows[-1].update(info)
    path = os.path.join(dest, "segments.csv")
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    os.replace(path + ".tmp", path)
    fights = [r for r in rows if r["label"] == "F" and r["onset_frame"] >= 0]
    log(f"wrote {path}: {len(rows)} videos, {len(fights)} fights with an onset, "
        f"{sum(r['onset_s'] >= 30 for r in fights)} with onset >= 30 s, {len(test)} in test split")
    if probe:
        log(f"label rows == video frames (+-1): {sum(r['frames_match'] for r in rows)}/{len(rows)}; "
            f"videos with audio: {sum(r['has_audio'] == 1 for r in rows)}; "
            f"total video hours: {sum(max(r['video_duration_s'], 0) for r in rows) / 3600:.1f}")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--archive-dir", default=DEFAULT_ARCHIVE_DIR)
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk-mb", type=float, default=32.0)
    ap.add_argument("--min-free-gb", type=float, default=60.0)
    ap.add_argument("--max-chunks", type=int, default=0, help="fetch at most N chunks (smoke tests)")
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--keep-archive", action="store_true")
    ap.add_argument("--restart", action="store_true", help="discard partial archive state and start over")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--no-probe", action="store_true", help="skip the ffprobe columns in segments.csv")
    ap.add_argument("--verify-only", action="store_true",
                    help="re-hash the extracted files against manifest.json (size, crc32, sha256); read-only")
    ap.add_argument("--backfill-sha256", action="store_true",
                    help="add sha256 to manifest.json from the extracted files after re-checking each CRC-32")
    args = ap.parse_args()

    if args.verify_only:
        _, problems = verify_extracted(args.dest)
        return 0 if not problems else 1
    if args.backfill_sha256:
        return backfill_sha256(args.dest)
    if args.summarize_only:
        summarize(args.dest, probe=not args.no_probe)
        return 0

    os.makedirs(args.archive_dir, exist_ok=True)
    archive = os.path.join(args.archive_dir, os.path.basename(args.url))
    min_free = int(args.min_free_gb * 1e9)
    already = os.path.exists(os.path.join(args.dest, "manifest.json")) and not os.path.exists(archive)
    if already:
        log(f"{args.dest}/manifest.json exists and no archive is pending; nothing to download")
        summarize(args.dest, probe=not args.no_probe)
        return 0

    dl = ChunkedDownload(args.url, archive, args.chunk_mb, args.workers, min_free, restart=args.restart)
    free = shutil.disk_usage(args.archive_dir).free
    need = dl.remaining_bytes() + UNCOMPRESSED_EST
    log(f"free {free / 1e9:.1f} GB; still needed about {need / 1e9:.1f} GB (archive rest + extracted); "
        f"guard {args.min_free_gb:.0f} GB")
    if free - need < min_free:
        log("ABORT: not enough free space")
        return 2
    for attempt in range(1 if args.max_chunks else 3):  # extra passes re-fetch chunks that ran out of retries
        if attempt:
            log(f"pass {attempt + 1}: {dl.n_chunks - len(dl.done)} chunks still missing")
        if not dl.run(max_chunks=args.max_chunks, extra_needed=UNCOMPRESSED_EST):
            return 2
        if dl.is_complete():
            break
    if not dl.is_complete():
        log(f"archive incomplete ({len(dl.done)}/{dl.n_chunks} chunks); rerun to continue")
        return 0 if args.max_chunks else 1
    log("archive complete")
    if args.no_extract:
        return 0
    try:
        failures = extract_all(archive, args.dest, dl)
    except zipfile.BadZipFile as e:
        log(f"ERROR {e}")
        return 1
    if failures:
        log(f"extraction had {len(failures)} failures; keeping the archive. First: {failures[:5]}")
        return 1
    summarize(args.dest, probe=not args.no_probe)
    if not args.keep_archive:
        for p in (archive, dl.state_path, dl.chunks_path):
            if os.path.exists(p):
                os.remove(p)
        log(f"removed verified archive {archive}")
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
