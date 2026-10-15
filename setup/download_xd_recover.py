#!/usr/bin/env python3
"""Recover official XD-Violence train videos that the jherng mirror lacks, from the HIVAU-70k video tar.

Why
    The official train split has **3,954** videos: the authors' own lists in github.com/Roc-Ng/XDVioDet
    (``list/rgb.list`` = 19,770 lines = 3,954 videos x 5 crops; ``list/audio.list`` = the same 3,954 names).
    The Hugging Face mirror ``jherng/xd-violence`` used by ``setup/download_xd_train.py`` has 3,950. The 4 it
    lacks (checked 2026-10-14; the mirror has nothing the official list lacks):

        NewAdd.NBA-2017.12.25_CLE@GSW__#01-08-34_01-40-09_label_A     normal
        Saving.Private.Ryan.1998__#02-29-31_02-30-55_label_B2-G-0     VIOLENT (shooting, explosion)
        v=8cTqh9tMz_I__#1_label_A                                     normal
        v=9eME1y6V-T4__#01-12-00_01-18-00_label_A                     normal

    The official routes do not work for scripts: OneDrive answers HTTP 403, the AliyunDrive share answers
    ``ShareLink.Forbidden`` (taken down), and Baidu marks the training videos "Disabled". The HIVAU-70k
    repo ``backseollgi/HIVAU-70k_XD-Violence`` stores the XD videos it annotates as one GNU tar cut into
    5 parts (``xd-violence_videos.tar.part_00..04``, 85.97 GB). Rather than downloading 86 GB, this script
    reads only the 512-byte tar headers with HTTP Range requests (one request per member), writes a member
    index, and then fetches just the wanted members' bytes.

Steps (subcommands; ``all`` runs them in order)
    official   fetch the authors' lists, write ``meta/official_train_list.txt`` (3,954 names) and
               ``meta/official_vs_mirror.json`` (official minus mirror, mirror minus official)
    index      walk the tar headers (resumable; about 0.15 s per member, ~4,750 members, ~12 min) and write
               ``downloads/hivau_xd_tar/index.jsonl`` (one line per member: name, data offset, size, next)
    check      provenance: for every member also in the jherng mirror (train manifest + cached test
               manifest) compare byte sizes, then sha256 straight from the tar the ``--n-hash`` smallest shared
               members plus the shared members just before and after each wanted clip in tar order, and
               compare with the mirror's LFS sha256. Writes ``meta/hivau_tar_check.json``.
    fetch      download the members in (official minus mirror) that the tar has into ``recovered_videos/``,
               streamed through ``.part`` with Range resume; checks the size against the tar header, computes
               sha256, walks every video packet with ``ffprobe -count_packets`` (rejects truncated or unreadable
               files) and compares the packet count with HIVAU ``n_frames`` when HIVAU annotates the clip,
               then appends to ``verified.jsonl`` (so ``anticipation/data/catalog.py`` picks it up) and writes
               ``meta/recovered_manifest.json``. A member that fails is moved to
               ``downloads/hivau_xd_tar/rejected/`` (outside raw_videos/) and listed under ``rejected``.
               ``download_xd_train.py --verify-only`` re-hashes the accepted files.

    Result on 2026-10-14: the tar holds all 4, but its copy of ``v=8cTqh9tMz_I__#1_label_A`` is TRUNCATED
    (24,838,144 bytes, an exact multiple of 64 KiB; the mdat box declares 29,688,659 bytes and there is no
    moov atom), so 3 are recovered and that one is rejected. Members the tar lacks would be listed under
    ``still_missing``; see docs/DATA_ACQUISITION.md section 3.2 for the manual routes.

Trust
    The tar has no per-member hash. A recovered file is accepted because (1) its size equals the tar header,
    (2) the same tar is byte-identical to the jherng mirror on every shared member that ``check`` compared
    (sizes on all 4,750 shared members, sha256 on a sample that includes the tar neighbours of every recovered
    clip), (3) every video packet is readable, and (4) its frame count matches HIVAU's annotation when HIVAU
    has the clip (only the NBA clip). The sha256 we
    record is computed locally (``sha256_source: "local"``), not supplied by the publisher.

Usage (from the repo root; the index step takes ~12 min, so run it detached)
    .venv/bin/python setup/download_xd_recover.py official
    nohup setsid .venv/bin/python -u setup/download_xd_recover.py index \\
        >> data/logs/xd_recover.log 2>&1 < /dev/null &
    .venv/bin/python setup/download_xd_recover.py check
    .venv/bin/python setup/download_xd_recover.py fetch
    .venv/bin/python setup/download_xd_recover.py all          # every step, resumable
"""

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import download_xd_train as xdt  # noqa: E402  (same folder; shares paths, logging, manifests)

TAR_REPO = "backseollgi/HIVAU-70k_XD-Violence"
TAR_REVISION = "f92e63ab00465b6bad08bd772d706493043b04ad"
TAR_PREFIX = "xd-violence_videos.tar.part_"
OFFICIAL_LISTS = {
    "rgb": "https://raw.githubusercontent.com/Roc-Ng/XDVioDet/master/list/rgb.list",
    "audio": "https://raw.githubusercontent.com/Roc-Ng/XDVioDet/master/list/audio.list",
}
DEFAULT_TAR_DIR = os.path.join(xdt.DATA_ROOT, "downloads", "hivau_xd_tar")
HIVAU_TRAIN_JSON = os.path.join(xdt.DATA_ROOT, "annotations", "hivau", "raw_annotations", "xd_database_train.json")
RECOVERED_SUBDIR = "recovered_videos"
BLOCK = 512

log = xdt.log


# ---------------------------------------------------------------------------------------------------------
# Official list
# ---------------------------------------------------------------------------------------------------------
def official_names_from_lists(rgb_text, audio_text):
    """Video stems from the authors' feature lists (``<stem>__<crop>.npy`` and ``<stem>__vggish.npy``)."""
    rgb = {re.sub(r"__\d+\.npy$", "", os.path.basename(x.strip())) for x in rgb_text.splitlines() if x.strip()}
    aud = {re.sub(r"__vggish\.npy$", "", os.path.basename(x.strip())) for x in audio_text.splitlines() if x.strip()}
    if rgb != aud:
        raise SystemExit(f"official rgb.list ({len(rgb)}) and audio.list ({len(aud)}) disagree")
    return sorted(rgb)


def step_official(dest):
    """Write meta/official_train_list.txt and meta/official_vs_mirror.json. Returns the comparison dict."""
    meta = os.path.join(dest, "meta")
    os.makedirs(meta, exist_ok=True)
    texts = {}
    for key, url in OFFICIAL_LISTS.items():
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        texts[key] = r.text
    names = official_names_from_lists(texts["rgb"], texts["audio"])
    _write_text(os.path.join(meta, "official_train_list.txt"), "".join(n + "\n" for n in names))
    mirror = {r["name"][:-4] for r in xdt.load_manifest(dest, refresh=False)}
    cmp_ = {
        "source": OFFICIAL_LISTS, "fetched": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_official": len(names), "n_mirror": len(mirror),
        "official_minus_mirror": sorted(set(names) - mirror),
        "mirror_minus_official": sorted(mirror - set(names)),
    }
    _write_json(os.path.join(meta, "official_vs_mirror.json"), cmp_)
    log(f"official train list: {len(names)} videos; mirror {len(mirror)}; "
        f"official minus mirror {len(cmp_['official_minus_mirror'])}: {cmp_['official_minus_mirror']}; "
        f"mirror minus official {len(cmp_['mirror_minus_official'])}")
    return cmp_


# ---------------------------------------------------------------------------------------------------------
# Virtual concatenated tar over HTTP Range requests
# ---------------------------------------------------------------------------------------------------------
class RemoteSplitFile:
    """Read byte ranges of ``part_00 + part_01 + ...`` (an HF LFS file split into parts) without downloading it.

    The CDN URL each ``resolve`` redirect points to is signed and expires; it is cached per part and
    re-resolved on a 4xx answer.
    """

    def __init__(self, repo, revision, prefix, cache_path):
        self.repo, self.revision = repo, revision
        self.session = requests.Session()
        self.cdn = {}
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                parts = json.load(f)
        else:
            from huggingface_hub import HfApi

            info = HfApi().dataset_info(repo, revision=revision, files_metadata=True)
            parts = sorted(({"path": s.rfilename, "size": int(s.size),
                             "sha256": s.lfs.sha256 if s.lfs else None}
                            for s in info.siblings if s.rfilename.startswith(prefix)), key=lambda p: p["path"])
            if not parts:
                raise SystemExit(f"no {prefix}* files in {repo}@{revision}")
            _write_json(cache_path, parts)
        self.parts = parts
        self.starts = []
        off = 0
        for p in parts:
            self.starts.append(off)
            off += p["size"]
        self.size = off

    def _url(self, path):
        return (f"https://huggingface.co/datasets/{self.repo}/resolve/{self.revision}/"
                f"{requests.utils.quote(path, safe='/')}")

    def _get(self, k, start, end, stream=False, retries=6):
        """GET bytes [start, end] (inclusive) of part ``k``; returns the response (206 checked)."""
        path = self.parts[k]["path"]
        for attempt in range(1, retries + 1):
            url = self.cdn.get(path) or self._url(path)
            try:
                r = self.session.get(url, headers={"Range": f"bytes={start}-{end}"}, allow_redirects=True,
                                     timeout=(30, 120), stream=stream)
                if r.status_code == 206:
                    if r.url != url:
                        self.cdn[path] = r.url
                    return r
                r.close()
                if 400 <= r.status_code < 500:
                    self.cdn.pop(path, None)  # signed CDN URL expired: resolve again
                raise IOError(f"HTTP {r.status_code}")
            except (requests.RequestException, IOError) as e:
                wait = min(60, 2 ** attempt)
                log(f"retry {attempt}/{retries} {path} bytes {start}-{end}: {type(e).__name__}: {e}; sleep {wait}s")
                time.sleep(wait)
        raise IOError(f"giving up on {path} bytes {start}-{end}")

    def segments(self, offset, length):
        """Split a global byte range into ``(part index, start, end_inclusive)`` pieces."""
        out = []
        end = min(offset + length, self.size)
        pos = offset
        while pos < end:
            k = max(i for i, s in enumerate(self.starts) if s <= pos)
            local = pos - self.starts[k]
            take = min(end - pos, self.parts[k]["size"] - local)
            out.append((k, local, local + take - 1))
            pos += take
        return out

    def read(self, offset, length):
        return b"".join(self._get(k, s, e).content for k, s, e in self.segments(offset, length))

    def iter_range(self, offset, length, chunk=1 << 20):
        """Stream a global byte range in blocks (for large members that may span parts)."""
        for k, s, e in self.segments(offset, length):
            with self._get(k, s, e, stream=True) as r:
                got = 0
                for block in r.iter_content(chunk):
                    if block:
                        got += len(block)
                        yield block
                if got != e - s + 1:
                    raise IOError(f"short read {got}/{e - s + 1} from {self.parts[k]['path']}")


# ---------------------------------------------------------------------------------------------------------
# Tar header parsing (ustar / GNU, with GNU long names and pax path records)
# ---------------------------------------------------------------------------------------------------------
def _octal(field):
    if field and field[0] & 0x80:  # GNU base-256 for sizes >= 8 GiB
        return int.from_bytes(bytes([field[0] & 0x7F]) + field[1:], "big")
    s = field.split(b"\0", 1)[0].strip()
    return int(s, 8) if s else 0


def parse_header(block):
    """Decode one 512-byte tar header. Returns None for an all-zero block (end of archive)."""
    if len(block) < BLOCK:
        raise ValueError("short tar header")
    block = block[:BLOCK]
    if block == b"\0" * BLOCK:
        return None
    stored = _octal(block[148:156])
    if stored != sum(block[:148]) + 32 * 8 + sum(block[156:]):
        raise ValueError("tar header checksum mismatch")
    name = block[:100].split(b"\0", 1)[0]
    if block[257:262] == b"ustar" and block[257:263] != b"ustar ":
        prefix = block[345:500].split(b"\0", 1)[0]
        if prefix:
            name = prefix + b"/" + name
    return {"name": name.decode("utf-8", "replace"), "size": _octal(block[124:136]),
            "type": chr(block[156]) if block[156] else "0"}


def padded(n):
    return (n + BLOCK - 1) // BLOCK * BLOCK


def walk_tar(remote, start=0, read_ahead=4 * BLOCK):
    """Yield ``{path, name, data_offset, size, next}`` for every regular file, starting at header ``start``.

    GNU ``L`` (long name) and pax ``x`` (``path=``) records are applied to the following header. One HTTP
    request per member: ``read_ahead`` bytes are read at each header so a long-name record and the real
    header usually arrive together.
    """
    off = start
    pending_name = None
    buf_off, buf = -1, b""
    while off + BLOCK <= remote.size:
        if not (buf_off <= off and off + BLOCK <= buf_off + len(buf)):
            buf_off, buf = off, remote.read(off, min(read_ahead, remote.size - off))
        h = parse_header(buf[off - buf_off:off - buf_off + BLOCK])
        if h is None:
            return
        data = off + BLOCK
        nxt = data + padded(h["size"])
        if h["type"] in ("L", "x"):
            if buf_off <= data and data + h["size"] <= buf_off + len(buf):
                payload = buf[data - buf_off:data - buf_off + h["size"]]
            else:
                payload = remote.read(data, h["size"])
            if h["type"] == "L":
                pending_name = payload.split(b"\0", 1)[0].decode("utf-8", "replace")
            else:
                m = re.search(rb"\d+ path=([^\n]*)\n", payload)
                pending_name = m.group(1).decode("utf-8", "replace") if m else pending_name
        elif h["type"] in ("0", "7"):
            path = pending_name or h["name"]
            pending_name = None
            yield {"path": path, "name": os.path.basename(path), "data_offset": data, "size": h["size"],
                   "next": nxt}
        else:
            pending_name = None  # directories, links, global pax: skip
        off = nxt


def load_index(path):
    """Records of ``index.jsonl``; a line cut short by a kill is skipped (the walk resumes after the last good one)."""
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def step_index(tar_dir):
    os.makedirs(tar_dir, exist_ok=True)
    remote = RemoteSplitFile(TAR_REPO, TAR_REVISION, TAR_PREFIX, os.path.join(tar_dir, "parts.json"))
    idx_path = os.path.join(tar_dir, "index.jsonl")
    done_path = idx_path + ".complete"
    if os.path.exists(done_path):
        log(f"tar index already complete: {idx_path}")
        return load_index(idx_path)
    have = load_index(idx_path)
    start = have[-1]["next"] if have else 0
    log(f"indexing {TAR_REPO}@{TAR_REVISION[:8]} {TAR_PREFIX}* ({remote.size / 1e9:.2f} GB in "
        f"{len(remote.parts)} parts) from offset {start} ({len(have)} members already indexed)")
    t0, n = time.time(), 0
    with open(idx_path, "a") as f:
        if have and f.tell() and not _ends_with_newline(idx_path):
            f.write("\n")  # isolate a line cut short by a kill
        for rec in walk_tar(remote, start):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n += 1
            if n % 250 == 0:
                log(f"indexed {len(have) + n} members, offset {rec['next'] / 1e9:.2f}/{remote.size / 1e9:.2f} GB, "
                    f"{n / (time.time() - t0):.1f} members/s")
    _write_text(done_path, time.strftime("%Y-%m-%d %H:%M:%S\n"))
    idx = load_index(idx_path)
    log(f"tar index complete: {len(idx)} members, {sum(r['size'] for r in idx) / 1e9:.2f} GB of payload")
    return idx


# ---------------------------------------------------------------------------------------------------------
# Provenance check and fetch
# ---------------------------------------------------------------------------------------------------------
def mirror_records(dest):
    """``{name.mp4: {size, sha256}}`` for the jherng train manifest plus the cached test manifest."""
    recs = {r["name"]: r for r in xdt.load_manifest(dest, refresh=False)}
    for r in xdt.load_test_manifest(dest, refresh=False):
        recs.setdefault(r["name"], r)
    return recs


def sha256_member(remote, rec):
    h = hashlib.sha256()
    for block in remote.iter_range(rec["data_offset"], rec["size"]):
        h.update(block)
    return h.hexdigest()


def hash_sample(idx, mirror, dest, n_smallest, max_bytes=600_000_000):
    """Shared members to sha256: the ``n_smallest`` smallest, plus the nearest shared neighbour before and after
    every wanted (official minus mirror) member in tar order, so the bytes right around each recovered clip
    are proven identical to the mirror."""
    shared = [r for r in idx if r["name"] in mirror]
    picks = [(r["name"], "smallest") for r in sorted(shared, key=lambda r: r["size"])[:n_smallest]]
    path = os.path.join(dest, "meta", "official_vs_mirror.json")
    wanted = set()
    if os.path.exists(path):
        with open(path) as f:
            wanted = {n + ".mp4" for n in json.load(f)["official_minus_mirror"]}
    for i, r in enumerate(idx):
        if r["name"] not in wanted:
            continue
        for step, label in ((-1, "before"), (1, "after")):
            j = i + step
            while 0 <= j < len(idx) and (idx[j]["name"] not in mirror or idx[j]["size"] > max_bytes):
                j += step
            if 0 <= j < len(idx):
                picks.append((idx[j]["name"], f"tar neighbour {label} {r['name']}"))
    seen, out = set(), []
    for n, why in picks:
        if n not in seen:
            seen.add(n)
            out.append((n, why))
    return out


def step_check(dest, tar_dir, n_hash):
    idx = load_index(os.path.join(tar_dir, "index.jsonl"))
    if not os.path.exists(os.path.join(tar_dir, "index.jsonl.complete")):
        raise SystemExit("tar index incomplete; run the index step first")
    remote = RemoteSplitFile(TAR_REPO, TAR_REVISION, TAR_PREFIX, os.path.join(tar_dir, "parts.json"))
    mirror = mirror_records(dest)
    by_name = {r["name"]: r for r in idx}
    shared = sorted(set(by_name) & set(mirror))
    size_bad = [n for n in shared if by_name[n]["size"] != mirror[n]["size"]]
    hashed = []
    for n, why in hash_sample(idx, mirror, dest, n_hash):
        got = sha256_member(remote, by_name[n])
        hashed.append({"name": n, "size": by_name[n]["size"], "why": why, "ok": got == mirror[n]["sha256"]})
        log(f"sha256 {'OK' if hashed[-1]['ok'] else 'MISMATCH'} {n} ({by_name[n]['size'] / 1e6:.1f} MB, {why})")
    result = {
        "tar": f"{TAR_REPO}@{TAR_REVISION}", "checked": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_tar_members": len(idx), "n_mirror_files": len(mirror), "n_shared": len(shared),
        "size_mismatches": size_bad, "sha256_sample": hashed,
        "tar_only": sorted(set(by_name) - set(mirror)), "mirror_only_count": len(set(mirror) - set(by_name)),
    }
    _write_json(os.path.join(dest, "meta", "hivau_tar_check.json"), result)
    log(f"tar vs mirror: {len(shared)} shared members, {len(size_bad)} size mismatches, sha256 sample "
        f"{sum(h['ok'] for h in hashed)}/{len(hashed)} identical; tar-only {result['tar_only']}; "
        f"mirror-only {result['mirror_only_count']}")
    return result


def hivau_frames():
    if not os.path.exists(HIVAU_TRAIN_JSON):
        return {}
    with open(HIVAU_TRAIN_JSON) as f:
        return {k: v.get("n_frames") for k, v in json.load(f).items()}


def probe_packets(path):
    """Container check without decoding: ``{ok, frames, read_packets, duration_s, error}``.

    ``ffprobe -count_packets`` walks every video packet, so a file whose moov atom is missing (truncated
    download) or whose index points past the end of the file fails here. ``ok`` needs a video stream with
    at least one packet and, when the header states a frame count, packets == frames (+-1).
    """
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=nb_frames,nb_read_packets,duration", "-of", "json", path],
                         capture_output=True, text=True, timeout=600)
    try:
        st = (json.loads(out.stdout or "{}").get("streams") or [{}])[0]
    except json.JSONDecodeError:
        st = {}

    def num(key, cast):
        try:
            return cast(st[key])
        except (KeyError, TypeError, ValueError):
            return -1

    frames, packets, dur = num("nb_frames", int), num("nb_read_packets", int), num("duration", float)
    ok = packets > 0 and (frames < 0 or abs(frames - packets) <= 1)
    return {"ok": ok, "frames": frames, "read_packets": packets, "duration_s": round(dur, 3),
            "error": out.stderr.strip().splitlines()[-1][:300] if out.stderr.strip() else ""}


def mp4_box_note(path):
    """Describe the top-level MP4 boxes when a file is truncated or lacks ``moov`` ("" if it looks complete)."""
    size = os.path.getsize(path)
    notes, off, seen = [], 0, set()
    with open(path, "rb") as f:
        while off + 8 <= size:
            f.seek(off)
            head = f.read(16)
            n, kind = struct.unpack(">I4s", head[:8])
            if n == 1 and len(head) >= 16:
                n = struct.unpack(">Q", head[8:16])[0]
            elif n == 0:
                n = size - off
            if n < 8:
                notes.append(f"bad box size {n} at offset {off}")
                break
            seen.add(kind)
            if off + n > size:
                notes.append(f"truncated: '{kind.decode('latin-1')}' box at offset {off} declares {n:,} bytes, "
                             f"file has {size:,}")
                break
            off += n
    if b"moov" not in seen:
        notes.append("no moov atom")
    return "; ".join(notes)


def reject_reason(path, info, want):
    """Why a fetched member is unusable, or None."""
    if not info["ok"]:
        note = mp4_box_note(path)
        return f"unreadable container ({note or info['error'] or 'no video packets'})"
    if want is not None and abs(info["read_packets"] - int(want)) > 1:
        return f"{info['read_packets']} video packets != HIVAU n_frames {want}"
    return None


def step_fetch(dest, tar_dir, min_free_gb, names=None):
    check_path = os.path.join(dest, "meta", "hivau_tar_check.json")
    if not os.path.exists(check_path):
        raise SystemExit("run the check step first (provenance of the tar)")
    with open(check_path) as f:
        check = json.load(f)
    if check["size_mismatches"] or not check["sha256_sample"] or not all(h["ok"] for h in check["sha256_sample"]):
        raise SystemExit(f"tar failed the provenance check ({check_path}); refusing to recover from it")
    if names is None:
        with open(os.path.join(dest, "meta", "official_vs_mirror.json")) as f:
            names = json.load(f)["official_minus_mirror"]
    idx = {r["name"]: r for r in load_index(os.path.join(tar_dir, "index.jsonl"))}
    remote = RemoteSplitFile(TAR_REPO, TAR_REVISION, TAR_PREFIX, os.path.join(tar_dir, "parts.json"))
    out_dir = os.path.join(dest, RECOVERED_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    verified = xdt.Verified(os.path.join(dest, "verified.jsonl"))
    frames = hivau_frames()
    man_path = os.path.join(dest, "meta", "recovered_manifest.json")
    prev, prev_rej = {}, {}
    if os.path.exists(man_path):
        with open(man_path) as f:
            old = json.load(f)
        prev = {r["name"]: r for r in old.get("files", [])}
        prev_rej = {r["name"]: r for r in old.get("rejected", [])}
    rejected_dir = os.path.join(tar_dir, "rejected")
    files, missing, rejected = [], [], []
    for stem in names:
        name = stem + ".mp4"
        rec = idx.get(name)
        if rec is None:
            missing.append(stem)
            continue
        final = os.path.join(out_dir, name)
        bad_copy = os.path.join(rejected_dir, name + ".rejected")
        if os.path.exists(bad_copy) and os.path.getsize(bad_copy) == rec["size"] and not os.path.exists(final):
            info = probe_packets(bad_copy)
            reason = reject_reason(bad_copy, info, frames.get(stem)) or prev_rej.get(name, {}).get("reason")
            rejected.append({"name": name, "size": rec["size"], "sha256": xdt.sha256_file(bad_copy).hexdigest(),
                             "kept_at": bad_copy, "reason": reason, "probe": info,
                             "source": f"{TAR_REPO}@{TAR_REVISION}:{rec['path']}"})
            log(f"{name}: tar copy already rejected earlier ({bad_copy}); skipping")
            continue
        if (name in prev and os.path.exists(final) and os.path.getsize(final) == rec["size"]
                and xdt.sha256_file(final).hexdigest() == prev[name]["sha256"]):
            files.append(prev[name])
            continue
        if os.path.exists(final):
            log(f"WARNING {final} does not match recovered_manifest.json; fetching it again")
        if xdt.free_bytes(out_dir) - rec["size"] < min_free_gb * 1e9:
            raise SystemExit(f"disk guard: fetching {name} would leave under {min_free_gb:.0f} GB free")
        part = final + ".part"
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > rec["size"]:
            os.remove(part)
            have = 0
        log(f"fetching {name} ({rec['size'] / 1e6:.1f} MB) from tar offset {rec['data_offset']}, resume at {have}")
        with open(part, "ab") as f:
            for block in remote.iter_range(rec["data_offset"] + have, rec["size"] - have):
                f.write(block)
        if os.path.getsize(part) != rec["size"]:
            raise SystemExit(f"size mismatch for {name}: {os.path.getsize(part)} vs tar header {rec['size']}")
        digest = xdt.sha256_file(part).hexdigest()
        info = probe_packets(part)
        want = frames.get(stem)
        reason = reject_reason(part, info, want)
        if reason:
            # Keep the bytes (outside raw_videos/, so no catalog or extractor sees them) for a possible salvage.
            os.makedirs(rejected_dir, exist_ok=True)
            os.replace(part, bad_copy)
            rejected.append({"name": name, "size": rec["size"], "sha256": digest, "kept_at": bad_copy,
                             "reason": reason, "probe": info, "source": f"{TAR_REPO}@{TAR_REVISION}:{rec['path']}"})
            log(f"REJECTED {name}: {reason}; kept at {bad_copy}")
            continue
        os.replace(part, final)
        entry = {"name": name, "subdir": RECOVERED_SUBDIR, "size": rec["size"], "sha256": digest,
                 "sha256_source": "local", "source": f"{TAR_REPO}@{TAR_REVISION}:{rec['path']}",
                 "tar_data_offset": rec["data_offset"], "video_frames": info["read_packets"],
                 "video_duration_s": info["duration_s"], "hivau_n_frames": want}
        vrec = {k: entry[k] for k in ("name", "subdir", "size", "sha256")}
        if verified.names.get(name) != vrec:
            verified.add(vrec)
        files.append(entry)
        log(f"recovered {name}: {rec['size']} bytes, sha256 {digest[:16]}..., {info['read_packets']} video packets "
            f"= frames, {info['duration_s']} s (HIVAU n_frames {want})")
    _write_json(man_path, {"note": "official XD train videos absent from jherng/xd-violence, recovered from the "
                                   "HIVAU-70k tar; sha256 computed locally",
                           "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "files": files, "rejected": rejected, "still_missing": missing})
    labels = xdt.write_labels(dest, xdt.load_manifest(dest, refresh=False), files)
    log(f"recovered {len(files)}/{len(names)}; rejected (tar copy unusable) {[r['name'] for r in rejected]}; "
        f"not in the tar {missing}; updated {labels}")
    return files, missing


# ---------------------------------------------------------------------------------------------------------
def _ends_with_newline(path):
    with open(path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        return f.read(1) == b"\n"


def _write_text(path, text):
    with open(path + ".tmp", "w") as f:
        f.write(text)
    os.replace(path + ".tmp", path)


def _write_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["official", "index", "check", "fetch", "all"])
    ap.add_argument("--dest", default=xdt.DEFAULT_DEST)
    ap.add_argument("--tar-dir", default=DEFAULT_TAR_DIR)
    ap.add_argument("--n-hash", type=int, default=3,
                    help="smallest shared members to sha256 from the tar in 'check' (plus the tar neighbours "
                         "of every wanted clip)")
    ap.add_argument("--min-free-gb", type=float, default=60.0)
    ap.add_argument("--names", nargs="*", default=None, help="fetch these stems instead of official minus mirror")
    args = ap.parse_args()
    dest = os.path.abspath(args.dest)
    if args.step in ("official", "all"):
        step_official(dest)
    if args.step in ("index", "all"):
        step_index(args.tar_dir)
    if args.step in ("check", "all"):
        step_check(dest, args.tar_dir, args.n_hash)
    if args.step in ("fetch", "all"):
        step_fetch(dest, args.tar_dir, args.min_free_gb, args.names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
