#!/usr/bin/env python3
"""Offline self-tests for the dataset downloaders (no network, no real data touched; temp dirs only).

Covers the failure modes found in review of the download track:

    xd_disk_budget        download_xd_train.DiskBudget counts the bytes owed by ALL queued and in-flight files,
                          and the guard also fires mid-file when another process fills the disk
    xd_verify_only        --verify-only is read-only (no train_labels.csv, no verified.jsonl lines, no network),
                          re-hashes stale verified.jsonl entries, and checks missing_test_videos/ against the
                          cached test manifest; a normal --include-missing-test rerun uses that cache and
                          re-downloads a same-size corrupted test clip; --rehash repairs a stale entry
    ubi_cd_recovery       download_ubi_fights: a zip whose end-of-central-directory / central directory chunks
                          are zero-filled although logged as done is repaired by re-fetching those chunks
                          (a plain zipfile.ZipFile open fails on the same file)
    ubi_sha256            manifest.json records sha256; --verify-only catches a tampered file; --backfill-sha256
                          refuses on a CRC failure and otherwise restores identical hashes
    tar_walk              download_xd_recover.walk_tar / RemoteSplitFile on a GNU tar with >100-char names cut
                          into parts at arbitrary boundaries: offsets and bytes match tarfile's own reading
    recover_reject        download_xd_recover.probe_packets / reject_reason accept a valid mp4 (ffmpeg testsrc)
                          and reject a truncated one (mdat past EOF, no moov), as found in the HIVAU tar
    official_list         download_xd_recover.official_names_from_lists parses the authors' list formats

Usage (from the repo root)
    .venv/bin/python setup/download_selftest.py            # all tests; exit code 0 = all passed
    .venv/bin/python setup/download_selftest.py ubi_cd_recovery xd_verify_only
"""

import argparse
import hashlib
import http.server
import io
import json
import os
import random
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import threading
import types
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import download_ubi_fights as ubi  # noqa: E402
import download_xd_recover as rec  # noqa: E402
import download_xd_train as xdt  # noqa: E402

LOGS = []


def _capture(msg):
    LOGS.append(str(msg))


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _snapshot(root):
    """{relative path: sha256} of every file under root."""
    out = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(dp, f)
            with open(p, "rb") as fh:
                out[os.path.relpath(p, root)] = _sha(fh.read())
    return out


class Patch:
    """Minimal attribute patcher: ``with Patch(obj, name=value, ...)`` restores the originals on exit."""

    def __init__(self, obj, **attrs):
        self.obj, self.attrs, self.saved = obj, attrs, {}

    def __enter__(self):
        for k, v in self.attrs.items():
            self.saved[k] = getattr(self.obj, k)
            setattr(self.obj, k, v)
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(self.obj, k, v)


# ---------------------------------------------------------------------------------------------------------
# Fake HTTP for download_xd_train (requests.get replacement)
# ---------------------------------------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, data, status, on_block=None):
        self.data, self.status_code, self.on_block = data, status, on_block

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise xdt.requests.HTTPError(f"HTTP {self.status_code}")

    @property
    def content(self):
        return self.data

    def iter_content(self, n):
        for i in range(0, len(self.data), n):
            if self.on_block:
                self.on_block(i // n)
            yield self.data[i:i + n]


def fake_get_factory(files, calls, on_block=None):
    """``requests.get`` stand-in serving ``files`` = {url basename (unquoted): bytes} with Range support."""
    def fake_get(url, headers=None, stream=False, timeout=None):
        name = xdt.urllib.parse.unquote(url.rsplit("/", 1)[-1])
        calls.append(name)
        data = files[name]
        rng = (headers or {}).get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            return FakeResponse(data[start:], 206, on_block)
        return FakeResponse(data, 200, on_block)
    return fake_get


def no_network(*a, **k):
    raise AssertionError("network access attempted")


# ---------------------------------------------------------------------------------------------------------
def test_xd_disk_budget(tmp):
    gb = 10 ** 9
    free = {"v": 62.5 * gb}
    with Patch(xdt, free_bytes=lambda p: free["v"], log=_capture):
        b = xdt.DiskBudget(tmp, 60 * gb, {"a": gb, "b": gb, "c": gb})
        assert b.total == 3 * gb
        # The old per-file guard (free - this file's remaining) would pass: 62.5 - 1 = 61.5 >= 60.
        assert free["v"] - gb >= 60 * gb
        try:
            b.check()
            raise AssertionError("run-wide guard did not fire")
        except RuntimeError as e:
            assert str(e) == "disk guard"
        b.wrote("a", gb // 2)
        b.finish("b")
        assert b.total == gb // 2 + gb, b.total
        b.set("a", 10 * gb)  # a restart from scratch owes the full size again
        assert b.total == 11 * gb
        b.wrote("a", 20 * gb)  # never below zero
        assert b.total == gb and b.owed["a"] == 0

        # download_one refuses to start a file when the OTHER queued files make the total too large.
        data = os.urandom(5000)
        recd = {"name": "x.mp4", "subdir": "s", "size": len(data), "sha256": _sha(data), "repo_path": "d/x.mp4"}
        ver = xdt.Verified(os.path.join(tmp, "verified.jsonl"))
        prog = xdt.Progress(len(data), 1)
        calls = []
        free["v"] = 60 * gb + 10_000  # room for this file alone, not for the 1 MB queued behind it
        budget = xdt.DiskBudget(tmp, 60 * gb, {os.path.join(tmp, "x.mp4"): len(data), "other": 1_000_000})
        with Patch(xdt.requests, get=fake_get_factory({"x.mp4": data}, calls)):
            try:
                xdt.download_one(recd, tmp, prog, ver, budget)
                raise AssertionError("download_one ignored the queued bytes")
            except RuntimeError as e:
                assert str(e) == "disk guard"
        assert calls == [] and not os.path.exists(os.path.join(tmp, "x.mp4.part"))
        assert budget.total == 1_000_000  # the aborted file no longer owes bytes

        # Mid-file: another process eats the disk while we stream; the guard fires within GUARD_EVERY blocks.
        free["v"] = 100 * gb
        budget = xdt.DiskBudget(tmp, 60 * gb, {os.path.join(tmp, "x.mp4"): len(data)})

        def eat_disk(k):
            if k == 3:
                free["v"] = 60 * gb + 100  # someone else wrote ~40 GB

        with Patch(xdt, CHUNK=256, GUARD_EVERY=2), \
                Patch(xdt.requests, get=fake_get_factory({"x.mp4": data}, calls, on_block=eat_disk)):
            try:
                xdt.download_one(recd, tmp, prog, ver, budget)
                raise AssertionError("mid-file guard did not fire")
            except RuntimeError as e:
                assert str(e) == "disk guard"
        written = os.path.getsize(os.path.join(tmp, "x.mp4.part"))
        assert 0 < written <= 6 * 256, written  # stopped within GUARD_EVERY blocks of the change
        assert budget.total == 0
    return f"run-wide guard fires at 62.5 GB free with 3 GB owed; pre-start and mid-file aborts OK ({written} B)"


def _xd_fixture(tmp):
    dest = os.path.join(tmp, "xd")
    local_test = os.path.join(tmp, "xd_test_local")
    for d in ("videos", "meta", "missing_test_videos"):
        os.makedirs(os.path.join(dest, d))
    os.makedirs(local_test)
    blobs = {n: os.urandom(3000 + 100 * i) for i, n in enumerate(["a.mp4", "b.mp4", "c.mp4", "t1.mp4", "t2.mp4"])}
    train = [{"name": n, "subdir": "1-1004", "size": len(blobs[n]), "sha256": _sha(blobs[n]),
              "repo_path": f"data/video/1-1004/{n}"} for n in ("a.mp4", "b.mp4", "c.mp4")]
    test = [{"name": n, "subdir": "test_videos", "size": len(blobs[n]), "sha256": _sha(blobs[n]),
             "repo_path": f"data/video/test_videos/{n}"} for n in ("t1.mp4", "t2.mp4")]
    with open(os.path.join(dest, "meta", "remote_manifest.json"), "w") as f:
        json.dump(train, f)
    with open(os.path.join(dest, "meta", "test_manifest.json"), "w") as f:
        json.dump(test, f)
    for name in xdt.META_FILES:
        open(os.path.join(dest, "meta", os.path.basename(name)), "w").close()

    def corrupt(b):
        return bytes([b[0] ^ 0xFF]) + b[1:]

    files = {"videos/a.mp4": blobs["a.mp4"], "videos/b.mp4": corrupt(blobs["b.mp4"]),
             "missing_test_videos/t1.mp4": blobs["t1.mp4"], "missing_test_videos/t2.mp4": corrupt(blobs["t2.mp4"])}
    for rel, data in files.items():
        with open(os.path.join(dest, rel), "wb") as f:
            f.write(data)
    # Stale verified.jsonl: b is listed as verified although it is now corrupt (same size); t2 is corrupt
    # (same size) and unlisted; c is absent.
    with open(os.path.join(dest, "verified.jsonl"), "w") as f:
        for r in train[:2]:
            f.write(json.dumps({k: r[k] for k in ("name", "subdir", "size", "sha256")}) + "\n")
        f.write(json.dumps({k: test[0][k] for k in ("name", "subdir", "size", "sha256")}) + "\n")
    return dest, local_test, blobs


def _args(dest, **kw):
    a = dict(dest=dest, workers=2, min_free_gb=0.0, limit=0, order="manifest", include_missing_test=False,
             verify_only=False, refresh_manifest=False, rehash=False)
    a.update(kw)
    return types.SimpleNamespace(**a)


def test_xd_verify_only(tmp):
    dest, local_test, blobs = _xd_fixture(tmp)
    before = _snapshot(dest)
    LOGS.clear()
    with Patch(xdt, LOCAL_TEST_DIR=local_test, list_remote=no_network, log=_capture), \
            Patch(xdt.requests, get=no_network):
        rc = xdt.run(_args(dest, verify_only=True))
    assert rc == 1, rc
    assert _snapshot(dest) == before, "verify-only modified the destination"
    assert not os.path.exists(os.path.join(dest, "train_labels.csv"))
    text = "\n".join(LOGS)
    assert "BAD b.mp4: sha256 differs" in text, text
    assert "BAD c.mp4: missing" in text, text
    assert "BAD t2.mp4: sha256 differs" in text, text
    assert "re-downloading" not in text and "downloading it again" not in text, text
    assert "videos/ vs remote_manifest.json: 1/3 OK" in text, text
    assert "missing_test_videos/ vs test_manifest.json: 1/2 OK" in text, text

    # Normal rerun with --include-missing-test: no tree-API call (cached test manifest). t2 (corrupt, not in
    # verified.jsonl) is re-hashed and re-downloaded, c is downloaded; b is trusted from verified.jsonl.
    calls = []
    served = {n: blobs[n] for n in blobs}
    LOGS.clear()
    with Patch(xdt, LOCAL_TEST_DIR=local_test, list_remote=no_network, log=_capture), \
            Patch(xdt.requests, get=fake_get_factory(served, calls)):
        rc = xdt.run(_args(dest, include_missing_test=True))
    assert rc == 0, (rc, LOGS)
    assert sorted(calls) == ["c.mp4", "t2.mp4"], calls
    # --rehash repairs what verify-only reported for b.
    calls.clear()
    with Patch(xdt, LOCAL_TEST_DIR=local_test, list_remote=no_network, log=_capture), \
            Patch(xdt.requests, get=fake_get_factory(served, calls)):
        rc = xdt.run(_args(dest, include_missing_test=True, rehash=True))
    assert rc == 0, (rc, LOGS)
    assert calls == ["b.mp4"], calls
    for rel, name in (("videos/b.mp4", "b.mp4"), ("videos/c.mp4", "c.mp4"), ("missing_test_videos/t2.mp4", "t2.mp4")):
        with open(os.path.join(dest, rel), "rb") as f:
            assert f.read() == blobs[name], rel
    assert os.path.exists(os.path.join(dest, "train_labels.csv"))

    LOGS.clear()
    with Patch(xdt, LOCAL_TEST_DIR=local_test, list_remote=no_network, log=_capture), \
            Patch(xdt.requests, get=no_network):
        rc = xdt.run(_args(dest, verify_only=True))
    assert rc == 0, LOGS
    return ("verify-only read-only and offline, re-hashes stale verified.jsonl entries and missing_test_videos/; "
            "normal rerun uses the cached test manifest; --rehash repairs the stale entry")


# ---------------------------------------------------------------------------------------------------------
# Local HTTP server with Range support (for the UBI chunked downloader)
# ---------------------------------------------------------------------------------------------------------
class RangeHandler(http.server.BaseHTTPRequestHandler):
    payload = b""

    def log_message(self, *a):
        pass

    def _headers(self, status, length, extra=None):
        self.send_response(status)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("ETag", '"selftest"')
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()

    def do_HEAD(self):
        self._headers(200, len(self.payload))

    def do_GET(self):
        rng = self.headers.get("Range")
        if not rng:
            self._headers(200, len(self.payload))
            self.wfile.write(self.payload)
            return
        a, b = rng.split("=")[1].split("-")
        a, b = int(a), min(int(b), len(self.payload) - 1)
        body = self.payload[a:b + 1]
        self._headers(206, len(body), {"Content-Range": f"bytes {a}-{b}/{len(self.payload)}"})
        self.wfile.write(body)


def serve(payload):
    handler = type("H", (RangeHandler,), {"payload": payload})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/UBI_FIGHTS.zip"


def make_zip(n_small=400, seed=0):
    """Synthetic UBI-like zip: many small CSVs (long names, so the central directory spans several chunks)."""
    rnd = random.Random(seed)
    buf = io.BytesIO()
    members = {}
    with zipfile.ZipFile(buf, "w") as z:
        for i in range(n_small):
            name = f"UBI_FIGHTS/annotation/F_{i}_1_0_0_0_{'x' * 60}.csv"
            data = ("\r\n".join(str(rnd.randint(0, 1)) for _ in range(rnd.randint(50, 300)))).encode()
            z.writestr(zipfile.ZipInfo(name), data, compress_type=zipfile.ZIP_DEFLATED)
            members[name] = data
        for i, size in enumerate((150_000, 90_000)):
            name = f"UBI_FIGHTS/videos/fight/F_{900 + i}_0_0_0_0.mp4"
            data = bytes(rnd.getrandbits(8) for _ in range(size))
            z.writestr(zipfile.ZipInfo(name), data, compress_type=zipfile.ZIP_STORED)
            members[name] = data
    return buf.getvalue(), members


def _zero(path, start, end):
    with open(path, "r+b") as f:
        f.seek(start)
        f.write(b"\0" * (end - start + 1))


def test_ubi_cd_recovery(tmp):
    payload, members = make_zip()
    srv, url = serve(payload)
    try:
        notes = []
        for case in ("eocd_chunk", "cd_middle_and_eocd"):
            work = os.path.join(tmp, case)
            os.makedirs(work)
            archive = os.path.join(work, "UBI_FIGHTS.zip")
            dest = os.path.join(work, "out")
            LOGS.clear()
            with Patch(ubi, log=_capture):
                dl = ubi.ChunkedDownload(url, archive, chunk_mb=16 / 1024, workers=4, min_free=0)
                assert dl.run() and dl.is_complete()
                with open(archive, "rb") as f:
                    assert f.read() == payload
                cd_start = ubi.central_directory_span(archive, dl.size)
                with zipfile.ZipFile(io.BytesIO(payload)) as z:
                    assert cd_start == z.start_dir, (cd_start, z.start_dir)
                cd_chunks = dl.chunks_for_range(cd_start, dl.size - 1)
                assert len(cd_chunks) >= 3, cd_chunks  # the test must span several chunks
                # Simulate chunks logged as done whose bytes never reached the disk.
                last = dl.n_chunks - 1
                _zero(archive, *dl.chunk_range(last))
                if case == "cd_middle_and_eocd":
                    # The first central-directory chunk lies before the 2-chunk tail fallback, so a second
                    # round (EOCD readable again, span known) is needed.
                    assert cd_chunks[0] < dl.n_chunks - 2
                    _zero(archive, *dl.chunk_range(cd_chunks[0]))
                try:
                    zipfile.ZipFile(archive).close()
                    raise AssertionError("corruption not visible to a plain ZipFile open")
                except zipfile.BadZipFile:
                    pass  # this is what crashed every rerun before the fix
                failures = ubi.extract_all(archive, dest, dl)
            assert failures == [], failures
            for name, data in members.items():
                with open(ubi.safe_target(dest, name), "rb") as f:
                    assert f.read() == data, name
            rounds = sum("re-downloading central-directory chunks" in m for m in LOGS)
            notes.append(f"{case}: repaired in {rounds} round(s)")
        # Unrecoverable without a downloader: a clear BadZipFile, not a crash elsewhere.
        _zero(archive, max(0, len(payload) - 100), len(payload) - 1)
        try:
            ubi.open_archive(archive, None)
            raise AssertionError("expected BadZipFile")
        except zipfile.BadZipFile as e:
            assert "--restart" in str(e)
    finally:
        srv.shutdown()
    return "; ".join(notes)


def test_ubi_sha256(tmp):
    payload, members = make_zip(n_small=20, seed=1)
    srv, url = serve(payload)
    try:
        archive = os.path.join(tmp, "UBI_FIGHTS.zip")
        dest = os.path.join(tmp, "out")
        with Patch(ubi, log=_capture):
            dl = ubi.ChunkedDownload(url, archive, chunk_mb=16 / 1024, workers=2, min_free=0)
            assert dl.run()
            assert ubi.extract_all(archive, dest, dl) == []
            man = ubi.load_manifest(dest)
            assert len(man) == len(members) and all(r.get("sha256") for r in man)
            for r in man:
                with open(os.path.join(dest, r["name"]), "rb") as f:
                    assert _sha(f.read()) == r["sha256"], r["name"]
            assert ubi.verify_extracted(dest)[1] == []

            # Backfill on an old-style manifest (no sha256) restores identical hashes.
            ubi.write_manifest(dest, [{k: v for k, v in r.items() if k != "sha256"} for r in man])
            assert ubi.backfill_sha256(dest) == 0
            assert ubi.load_manifest(dest) == man

            # Tamper one file (same size): verify-only reports it; backfill refuses and leaves the manifest.
            victim = os.path.join(dest, man[0]["name"])
            with open(victim, "r+b") as f:
                b0 = f.read(1)
                f.seek(0)
                f.write(bytes([b0[0] ^ 0x01]))
            problems = ubi.verify_extracted(dest)[1]
            assert [p[0] for p in problems] == [man[0]["name"]], problems
            snap = _snapshot(dest)
            assert ubi.backfill_sha256(dest) == 1
            assert _snapshot(dest) == snap
    finally:
        srv.shutdown()
    return f"{len(members)} members with sha256; tamper detected; backfill refuses on CRC failure"


# ---------------------------------------------------------------------------------------------------------
class LocalSplitFile(rec.RemoteSplitFile):
    """RemoteSplitFile over local part files (same segment logic, no HTTP)."""

    def __init__(self, paths):  # noqa: super().__init__ is replaced on purpose
        self.parts = [{"path": p, "size": os.path.getsize(p)} for p in paths]
        self.starts, off = [], 0
        for p in self.parts:
            self.starts.append(off)
            off += p["size"]
        self.size = off

    def _get(self, k, start, end, stream=False, retries=1):
        with open(self.parts[k]["path"], "rb") as f:
            f.seek(start)
            data = f.read(end - start + 1)
        return FakeResponse(data, 206)


def test_tar_walk(tmp):
    rnd = random.Random(2)
    tar_path = os.path.join(tmp, "x.tar")
    want = {}
    with tarfile.open(tar_path, "w", format=tarfile.GNU_FORMAT) as t:
        for i in range(25):
            long = "videos/train/" + ("NewAdd.NBA-2017.12.25_CLE@GSW__#01-08-34_01-40-09_label_A" if i == 7 else
                                      f"Clip.{i}.{'y' * rnd.randint(10, 120)}__#00-00-0{i % 10}_label_A") + ".mp4"
            data = bytes(rnd.getrandbits(8) for _ in range(rnd.randint(0, 20_000)))
            ti = tarfile.TarInfo(long)
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
            want[os.path.basename(long)] = data
    # Reference offsets from tarfile itself.
    with tarfile.open(tar_path) as t:
        ref = {os.path.basename(m.name): (m.offset_data, m.size) for m in t.getmembers() if m.isfile()}
    with open(tar_path, "rb") as f:
        blob = f.read()
    cuts = sorted(rnd.sample(range(1, len(blob)), 3)) + [len(blob)]
    paths, prev = [], 0
    for i, c in enumerate(cuts):
        p = os.path.join(tmp, f"x.tar.part_{i:02d}")
        with open(p, "wb") as f:
            f.write(blob[prev:c])
        paths.append(p)
        prev = c
    remote = LocalSplitFile(paths)
    got = {r["name"]: r for r in rec.walk_tar(remote)}
    assert set(got) == set(want), set(want) ^ set(got)
    for name, r in got.items():
        assert (r["data_offset"], r["size"]) == ref[name], (name, r, ref[name])
        assert b"".join(remote.iter_range(r["data_offset"], r["size"], chunk=4096)) == want[name], name
    # Resume from the middle of the walk yields the same tail.
    mid = list(rec.walk_tar(remote))[12]
    tail = [r["name"] for r in rec.walk_tar(remote, start=mid["next"])]
    assert tail == [r["name"] for r in list(rec.walk_tar(remote))[13:]]
    longest = max(len(r["path"]) for r in got.values())
    return f"{len(got)} members (longest path {longest} chars) across {len(paths)} parts; offsets and bytes match"


def test_recover_reject(tmp):
    """The packet walk accepts a valid mp4 and rejects a truncated one (the HIVAU copy of v=8cTqh9tMz_I)."""
    good = os.path.join(tmp, "good.mp4")
    r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=24",
                        "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", good],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:  # some ffmpeg builds lack libx264
        r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=24",
                            "-t", "2", "-c:v", "mpeg4", good], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    info = rec.probe_packets(good)
    assert info["ok"] and info["read_packets"] == 48, info
    assert rec.reject_reason(good, info, 48) is None
    assert "packets != HIVAU" in rec.reject_reason(good, info, 100)
    # Truncated like the tar member: ftyp + free + an mdat that declares more bytes than the file has, no moov.
    bad = os.path.join(tmp, "bad.mp4")
    with open(bad, "wb") as f:
        f.write(struct.pack(">I4s", 32, b"ftyp") + b"isom" + b"\0\0\2\0" + b"isomiso2avc1mp41")
        f.write(struct.pack(">I4s", 8, b"free"))
        f.write(struct.pack(">I4s", 1_000_000, b"mdat") + os.urandom(65536 - 48))
    info = rec.probe_packets(bad)
    assert not info["ok"], info
    why = rec.reject_reason(bad, info, None)
    assert "truncated: 'mdat'" in why and "no moov atom" in why, why
    return f"valid mp4 accepted (48 packets); truncated mp4 rejected: {why}"


def test_official_list(tmp):
    rgb = "".join(f"/media/x/RGB/{s}__{c}.npy\n" for s in ("A__#1_label_A", "v=q__#1_label_B1-0-0") for c in range(5))
    aud = "/media/x/train/A__#1_label_A__vggish.npy\n/media/x/train/v=q__#1_label_B1-0-0__vggish.npy\n"
    assert rec.official_names_from_lists(rgb, aud) == ["A__#1_label_A", "v=q__#1_label_B1-0-0"]
    try:
        rec.official_names_from_lists(rgb, aud.splitlines()[0])
        raise AssertionError("mismatched lists accepted")
    except SystemExit:
        pass
    return "rgb.list / audio.list parsing OK"


TESTS = {
    "xd_disk_budget": test_xd_disk_budget,
    "xd_verify_only": test_xd_verify_only,
    "ubi_cd_recovery": test_ubi_cd_recovery,
    "ubi_sha256": test_ubi_sha256,
    "tar_walk": test_tar_walk,
    "recover_reject": test_recover_reject,
    "official_list": test_official_list,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tests", nargs="*", help=f"subset to run (default all): {', '.join(TESTS)}")
    ap.add_argument("--tmp", default=None, help="parent folder for the temp dirs (default: system temp)")
    args = ap.parse_args()
    unknown = [t for t in args.tests if t not in TESTS]
    if unknown:
        ap.error(f"unknown tests {unknown}; choose from {list(TESTS)}")
    names = args.tests or list(TESTS)
    failed = 0
    for name in names:
        tmp = tempfile.mkdtemp(prefix=f"dl_selftest_{name}_", dir=args.tmp)
        try:
            msg = TESTS[name](tmp)
            print(f"PASS {name}: {msg}")
        except Exception as e:  # report every test, then fail at the end
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            if LOGS:
                print("  last log lines:\n    " + "\n    ".join(LOGS[-10:]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"{len(names) - failed}/{len(names)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
