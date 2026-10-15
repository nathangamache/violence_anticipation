# Data acquisition: new raw datasets (XD-Violence train, UBI-Fights, access-gated sets)

Written 2026-10-14 by the dataset-download track; revised the same day after review (XD completeness vs the
authors' official list, recovery of the 4 clips the mirror lacks, UBI sha256 manifest, verify-only and
disk-guard fixes, self-tests). Covers where each new raw dataset comes from, how it was
verified, the exact commands, the on-disk layout, and the datasets that need a manual request. Background on
why these datasets matter is in [research/datasets.md](research/datasets.md) and
[research/README.md](research/README.md); how they enter the benchmark is in [BASELINE.md](BASELINE.md)
section 6.2 (VA30-full).

All paths below are relative to `data/` = `data/` unless absolute.

---

## 1. Status at a glance

| Dataset | Source used | Size | Local path | Status (2026-10-14 19:10 EDT) | License / terms |
|---|---|---|---|---|---|
| XD-Violence **train** (official list: **3,954** videos, audio) | HF mirror `jherng/xd-violence`, pinned commit `8eba22ca` (3,950 of them) | 74.64 GB | `raw_videos/xd_violence_train/videos/` | **3,950/3,950 mirror files** sha256-verified against the mirror's LFS hashes (re-hashed with `--verify-only`); 188.9 h; every file has an audio stream. Took 18 min at 40 to 67.6 MB/s. The mirror lacks **4 official train videos** (section 3.2) | No explicit license; research use; cite Wu et al. ECCV 2020. Mirror card says only `license: mit`; the videos are the authors' release |
| XD-Violence train, the 4 clips the mirror lacks | HIVAU-70k video tar `backseollgi/HIVAU-70k_XD-Violence` (pinned `f92e63ab`), members fetched with HTTP Range reads | 429 MB (3 files) | `raw_videos/xd_violence_train/recovered_videos/` | **3 of 4 recovered** (NBA, Saving Private Ryan (violent), `v=9eME1y6V-T4`), every video packet readable, NBA frame count = HIVAU's 45,482. **`v=8cTqh9tMz_I__#1_label_A` is truncated in the tar** (no moov atom), so it was rejected and no usable copy is known. Coverage of the official list: **3,953/3,954** | as above |
| XD-Violence test, 2 clips missing locally | same mirror (`data/video/test_videos/`) | 31 MB | `raw_videos/xd_violence_train/missing_test_videos/` | Done (sha256-verified) | as above |
| UBI-Fights (1,000 videos, no audio) | `https://socia-lab.di.ubi.pt/EventDetection/UBI_FIGHTS.zip` | 8.12 GB zip, 8.27 GB extracted | `raw_videos/ubi_fights/` | **Complete.** 2,001/2,001 zip members CRC-checked and extracted (216 fight + 784 normal videos, 1,000 label CSVs); 79.0 h; archive deleted afterwards. sha256 of every file now recorded in `manifest.json`. CRC = transfer consistency only, not authenticity (TLS unverified; section 4.1). Took 33 min (0.8 to 8.5 MB/s) | No license stated; cite Degardin and Proenca, IJCB 2020 |
| NTU CCTV-Fights | rose1.ntu.edu.sg | 7.2 GB | not downloaded | **Request form required** (section 5) | Academic, non-commercial, no redistribution or derived datasets without permission |
| DVD (2025) | not public | 500 videos | not downloaded | **Not released**; a subset is only available through the ABAW-10 challenge with a faculty-signed EULA (section 6) | Paper is CC BY-NC-ND 4.0; data terms set by the EULA |

Logs: `logs/xd_train_download.log`, `logs/ubi_fights_download.log` (both end with `DONE`),
`logs/xd_recover.log` (tar index and recovery), full XD ffprobe audit in `logs/xd_train_ffprobe.json` / `.log`. Disk use: XD train 70 GB (74.6 GB decimal) + 0.43 GB
recovered, UBI 7.8 GB; `/data2` had about 359 GB free afterwards.

---

## 2. Scripts (all in `setup/`, all resumable)

| Script | What it does |
|---|---|
| `setup/download_xd_train.py` | Lists the mirror with the HF tree API (sizes + LFS sha256; cached in `meta/remote_manifest.json` and `meta/test_manifest.json`), downloads every train mp4 with HTTP Range resume, verifies size and sha256, then renames `.part` to `.mp4`. Writes `meta/`, `train_labels.csv`, `verified.jsonl`. Disk guard: free space minus the bytes that **all** queued and in-flight files still owe must stay at or above 60 GB, checked before every file and every 64 MB written. `--verify-only` is read-only and offline; `--rehash` repairs what it reports. |
| `setup/download_xd_recover.py` | Completeness against the authors' official list and recovery of the clips the mirror lacks: `official` (fetch Roc-Ng/XDVioDet lists, write `meta/official_train_list.txt` + `meta/official_vs_mirror.json`), `index` (walk the HIVAU-70k video tar headers with Range requests, ~19 min, resumable), `check` (tar vs mirror: sizes of all shared members + sha256 of a sample), `fetch` (Range-read the wanted members into `recovered_videos/`, walk every video packet, frame count vs HIVAU, add to `verified.jsonl`; unreadable members go to `downloads/hivau_xd_tar/rejected/`). |
| `setup/download_selftest.py` | Offline tests (temp dirs, local HTTP server, no real data touched) for the disk guard, read-only verify, UBI central-directory repair, UBI sha256 manifest, tar walking, truncated-mp4 rejection and official-list parsing. |
| `setup/download_xd_train.sh` | Detached launcher (`nohup setsid flock`), log `logs/xd_train_download.log`, pid file `logs/xd_train_download.pid`. Extra args pass through. |
| `setup/download_ubi_fights.py` | Parallel byte-range download into a preallocated zip (each chunk is `fdatasync`ed before the chunk log marks it done), CRC-checked extraction (path-traversal safe; a zip whose central directory is corrupt is repaired by re-fetching those chunks), `manifest.json` with size, CRC-32 and sha256, `segments.csv`, then deletes the archive. `--verify-only` re-hashes the extracted tree (read-only); `--backfill-sha256` adds sha256 to an older manifest after re-checking each CRC. Disk guard as above, counting the extracted size too. |
| `setup/download_ubi_fights.sh` | Detached launcher, log `logs/ubi_fights_download.log`, pid file `logs/ubi_fights_download.pid`. |
| `setup/download_verify.py` | ffprobe a random sample (or `--all`) of a folder: codec, size, fps, duration, audio stream, optional mean volume (`--volume`). Read-only. |

Commands (from the repo root):

```bash
# start or resume (both are idempotent; a lock prevents double runs)
bash setup/download_xd_train.sh --include-missing-test --workers 8
bash setup/download_ubi_fights.sh --workers 8

# follow / stop
tail -f data/logs/xd_train_download.log
kill -- -"$(cat data/logs/xd_train_download.pid)"

# after the XD run finishes: re-hash every file against the manifests (read-only, offline; exit 0 = no file
# missing or bad). It also prints completeness against the authors' official 3,954-video list.
.venv/bin/python setup/download_xd_train.py --verify-only
# repair what --verify-only reported as BAD (re-hashes listed files, re-downloads mismatches)
.venv/bin/python setup/download_xd_train.py --rehash --include-missing-test

# the 4 official train clips the mirror lacks (section 3.2); index takes ~19 min, so detach it
.venv/bin/python setup/download_xd_recover.py official
nohup setsid .venv/bin/python -u setup/download_xd_recover.py index \
    >> data/logs/xd_recover.log 2>&1 < /dev/null &
.venv/bin/python setup/download_xd_recover.py check
.venv/bin/python setup/download_xd_recover.py fetch

# UBI: re-hash the extracted files against manifest.json (size, CRC-32, sha256); read-only
.venv/bin/python setup/download_ubi_fights.py --verify-only

# offline self-tests of the downloader logic (about 10 s)
.venv/bin/python setup/download_selftest.py

# ffprobe audit (-I: do not let the data folder shadow Python modules)
.venv/bin/python -I setup/download_verify.py data/raw_videos/xd_violence_train/videos --n 20 --volume
.venv/bin/python -I setup/download_verify.py data/raw_videos/ubi_fights/videos --recursive --n 20

# UBI: rebuild segments.csv from the per-frame CSVs without downloading anything
.venv/bin/python setup/download_ubi_fights.py --summarize-only
```

If a run dies (network, reboot), rerun the same launcher. XD resumes partial files with a Range request and
skips files already in `verified.jsonl`; UBI fetches only the chunks missing from `UBI_FIGHTS.zip.chunks`.

YouTube downloads: the account cookies live outside the repo at `<cookies file outside the repo>` (directory
mode 700, file mode 600; moved there from the repo root in the 2026-10 cleanup). No script reads them by default
(`setup/curate_youtube.py` calls `yt-dlp` without cookies); for age-restricted or rate-limited videos pass
`yt-dlp --cookies <cookies file outside the repo>`. Never copy them into the repo or `data/`. They expire around 2026-11/12; export fresh ones into the same file. `YOUTUBE_API_KEY` stays in
the repo-root `.env` (mode 600, gitignored).

---

## 3. XD-Violence train split

### 3.1 Why this source
- The official page (https://roc-ng.github.io/XD-Violence/) offers training videos on OneDrive (five zips:
  `0001-1004`, `1005-2004`, `2005-2804`, `2805-3319`, `3320-3954`), AliyunDrive, and Baidu (training videos
  marked "Disabled"). The OneDrive share links answer **HTTP 403** to scripted requests (2026-10-14), and the
  AliyunDrive share answers `{"code":"ShareLink.Forbidden"}` to its anonymous API (taken down).
- The Hugging Face mirror `jherng/xd-violence` holds the same five range folders as individual mp4 files
  (`data/video/<range>/<name>.mp4`) plus `test_videos/`, `train_list.txt`, `test_list.txt` and
  `test_annotations.txt`. Each file has an LFS sha256, so every download is verified exactly.
- **Verified original, not a re-encode:** all 798 test clips we already have in `raw_videos/xd_violence/videos/`
  match the mirror's `test_videos/` by name **and byte size** (798/798), and the sha256 of a sampled file is
  identical. All 3,950 train files are h264 + AAC at 24 fps, same as the local test clips.
- Provenance wording: the mirror's dataset card says only `license: mit`; the sentence "downloaded from the
  authors' website" is in its loader script `xd-violence.py` (`_DESCRIPTION`), not in the card.
- Speed: 40 to 67.6 MB/s from this server (peak PROGRESS line in the log). Other mirrors seen:
  `qiouzao/XD-Violence` (one 13 GB zip, test only), `bibistellar/xd-violence` (test videos + features),
  `YuFly0215/xd-violence` and `anumsaeed24725/xd-violence` (file-for-file copies of `jherng`, same 4 clips
  missing), `Arjun2k04/XD-Violence` (features only), and `backseollgi/HIVAU-70k_XD-Violence`
  (`xd-violence_videos.tar.part_00..04`, 85.97 GB; holds all 3,954 official train clips + the 800 test clips,
  section 3.2). The HIVAU tar is the fallback if `jherng` disappears; `download_xd_recover.py` can fetch any
  member from it without downloading the whole tar.

### 3.2 Counts: the official split has 3,954 train videos; the mirror has 3,950
- **Official count = 3,954.** The authors' own lists in https://github.com/Roc-Ng/XDVioDet name 3,954 train
  videos: `list/rgb.list` has 19,770 lines (3,954 videos x 5 crops) and `list/audio.list` has the same 3,954
  names. This matches the paper and the last range folder name (`3320-3954`). A local copy is
  `meta/official_train_list.txt`; `meta/official_vs_mirror.json` holds the set differences.
- **The `jherng` mirror has 3,950** (its `train_list.txt` too). Every range folder is full except `1005-2004`,
  which holds 996 files instead of 1,000, so all 4 absent clips belong to the official `1005-2004` zip.
  Mirror minus official is empty. Official minus mirror is 4 clips:

  | Clip | Label | HIVAU-70k entry |
  |---|---|---|
  | `NewAdd.NBA-2017.12.25_CLE@GSW__#01-08-34_01-40-09_label_A` | normal | yes (45,482 frames at 24 fps) |
  | `Saving.Private.Ryan.1998__#02-29-31_02-30-55_label_B2-G-0` | **violent** (shooting, explosion) | **no**, so no onset |
  | `v=8cTqh9tMz_I__#1_label_A` | normal | no |
  | `v=9eME1y6V-T4__#01-12-00_01-18-00_label_A` | normal | no |

- An earlier version of this document said "3,950 is what the authors actually released" because HIVAU-70k's
  `xd_database_train.json` also has 3,950 entries. That was wrong: the HIVAU set is a different 3,950 (it has
  the NBA clip and lacks `Yellow.Sea.2010__#00-15-34_00-16-47_label_A` and the other 3 above). The equal
  counts are a coincidence. The sha256 checks of `download_xd_train.py` only prove that files match the
  **mirror's** manifest, so they could never catch the gap; `--verify-only` now also prints completeness
  against the official list.
- **Recovery (2026-10-14, `setup/download_xd_recover.py`).** The official OneDrive links answer 403 to scripts,
  the AliyunDrive share is taken down (`ShareLink.Forbidden`), Baidu marks training videos "Disabled", and every
  other HF mirror either copies `jherng` or holds only features. The HIVAU-70k repo
  `backseollgi/HIVAU-70k_XD-Violence` (pinned `f92e63ab`) stores the XD videos as one GNU tar cut into 5 parts
  (`xd-violence_videos.tar.part_00..04`, 85.97 GB). Rather than download it, the script walked its 512-byte
  member headers with HTTP Range requests (4,754 members = 3,954 train + 800 test; 19 min at 4.3 members/s;
  index in `downloads/hivau_xd_tar/index.jsonl`), then:
  - **Provenance check** (`meta/hivau_tar_check.json`): 4,750 members are shared with the `jherng` train and
    test manifests and all 4,750 have identical byte sizes; sha256 read straight from the tar matches the
    mirror's LFS sha256 on 11/11 sampled members (the 3 smallest, plus the shared members immediately before
    and after each of the 4 wanted clips in tar order). The members only the tar has are exactly the 4 clips.
  - **Fetched** each wanted member by Range read, checked the size against the tar header, walked every video
    packet with `ffprobe -count_packets`, and compared with HIVAU `n_frames` where HIVAU has the clip:

    | Clip | Bytes | Result |
    |---|---|---|
    | `NewAdd.NBA-2017.12.25_CLE@GSW__#01-08-34_01-40-09_label_A` | 362,513,722 | recovered; 45,482 frames = HIVAU 45,482; 1,895.1 s; AAC stereo |
    | `Saving.Private.Ryan.1998__#02-29-31_02-30-55_label_B2-G-0` | 16,656,520 | recovered; 2,016 frames, 84.0 s (= the 02-29-31 to 02-30-55 span); AAC 5.1 |
    | `v=9eME1y6V-T4__#01-12-00_01-18-00_label_A` | 50,219,722 | recovered; 8,642 frames, 360.1 s (= the 6 min span); AAC stereo |
    | `v=8cTqh9tMz_I__#1_label_A` | 24,838,144 | **rejected: truncated.** The `mdat` box declares 29,688,659 bytes, the file has 24,838,144 (exactly 379 x 64 KiB) and there is no `moov` atom, so nothing can open it. Kept for a possible salvage at `downloads/hivau_xd_tar/rejected/` (outside `raw_videos/`, so no catalog or extractor sees it) |

  - The 3 recovered files are in `recovered_videos/`, listed in `verified.jsonl` (so `anticipation/data/catalog.py`
    catalogues them as `xd_train`) and `train_labels.csv` (subdir `recovered_videos`), and re-hashed by
    `download_xd_train.py --verify-only` against `meta/recovered_manifest.json`. There is no publisher hash for
    them: their sha256 is computed locally (`sha256_source: "local"`); trust rests on the tar check above.
  - **Still missing: `v=8cTqh9tMz_I__#1_label_A`** (normal, so no onset is lost). Manual routes if it is ever
    needed: the official OneDrive `1005-2004` zip downloaded in a browser (scripts get 403; the official copy
    may be truncated too), or an email to the authors ([contact on the dataset page], on the official page).
    Re-downloading `v=8cTqh9tMz_I` from YouTube would give a different encode and cut, not the release file.
- Composition of the official 3,954 (from filenames): 1,905 violent and 2,049 normal; 2,137 from YouTube
  (`v=` prefix), 1,811 from movies and 6 `NewAdd.*` extras; 1,630 distinct groups (the text before `__#`).
  The mirror's 3,950: 1,904 violent and 2,046 normal; 2,135 YouTube, 1,810 movies, 5 `NewAdd.*`; 1,629
  groups. Label code counts (mirror): A 2,046, B1 Fighting 481, B6 Car accident 444, B2 Shooting 441,
  B4 Riot 384, G Explosion 381, B5 Abuse 50.
- File sizes (mirror): 0.1 MB to 871 MB, median 10 MB. Durations 1.4 s to 3 h (`v=WpxJJFcWM8s__#1_label_A` is
  10,816 s), median 110 s, 188.9 h in total.

### 3.3 Layout

```
raw_videos/xd_violence_train/
  videos/<name>.mp4               3,950 files from the mirror, flat, same naming as raw_videos/xd_violence/videos
  recovered_videos/<name>.mp4     3 of the 4 official train clips the mirror lacks, from the HIVAU-70k tar
  meta/train_list.txt             the MIRROR's train list, "<range>/<name>" per line (no .mp4), 3,950 lines:
                                  the official list minus the 4 clips of section 3.2, not the authors' full list
  meta/official_train_list.txt    the authors' 3,954 train stems (from Roc-Ng/XDVioDet list/rgb.list + audio.list)
  meta/official_vs_mirror.json    official minus mirror (4) and mirror minus official (0)
  meta/test_list.txt              official test list (mirror copy)
  meta/test_annotations.txt       official test frame annotations (identical content to xd_violence/annotations.txt)
  meta/remote_manifest.json       [{name, subdir, size, sha256, repo_path}] for the mirror's 3,950 train files
  meta/test_manifest.json         the same for the mirror's 800 test files (cached; reruns need no tree API call)
  meta/hivau_tar_check.json       provenance check of the HIVAU tar against the mirror (section 3.2)
  meta/recovered_manifest.json    files: the 3 recovered clips (size, locally computed sha256, tar source and
                                  offset, frames, duration, HIVAU n_frames); rejected: the truncated
                                  v=8cTqh9tMz_I (reason, probe); still_missing: clips not in the tar (none)
  train_labels.csv                name, subdir, group, is_youtube, binary, codes (mirror + recovered clips;
                                  subdir "recovered_videos" for the latter)
  verified.jsonl                  one line per verified file (name, subdir, size, sha256), all three folders
  missing_test_videos/            Taken.3.2014__#01-20-10_01-21-30_label_A.mp4,
                                  v=p2ggHwtb-Zg__#00-21-00_00-24-00_label_A.mp4
```

The two `missing_test_videos/` clips are official **test** clips that the local test folder lacks (both
normal, `label_A`). They are kept apart so this track does not write into `raw_videos/xd_violence/`. To use
them, move them into `raw_videos/xd_violence/videos/` by hand.

### 3.4 Labels and onsets
- XD train has **video-level labels only** (codes in the filename; see `train_labels.csv`). Frame or second
  onsets for train videos come from **HIVAU-70k** `xd_database_train.json` (`events` in seconds; filter on a
  non-empty `label`, because normal videos also carry random "events"). That annotation layer is fetched by a
  different track (`data/annotations/hivau/`). `Saving.Private.Ryan.1998__#02-29-31_02-30-55_label_B2-G-0`
  (violent, recovered from the HIVAU tar) has **no HIVAU entry**, so it has no onset; it can only serve as a
  video-level positive until someone annotates it. The other recovered clips are normal (`label_A`).
- Label code meanings (official ReadMe): A normal, B1 Fighting, B2 Shooting, B4 Riot, B5 Abuse, B6 Car
  accident, G Explosion; `0` is padding. Group key for splits (BASELINE 6.3): text before `__#`.

### 3.5 Gotchas
- Every filename contains `#` (all 3,954 official), many contain `=` (2,137 YouTube clips) and four contain
  `@` (the NBA segments; three in the mirror, one recovered); there are no spaces. URL-quote names when
  building URLs (an unquoted `#` truncates the URL to a fragment) and quote paths in shell commands.
- The mirror's loader script (`xd-violence.py`) uses `x.rstrip(".mp4")`, which strips trailing `m`, `p`, `4`
  and `.` characters, not the suffix. Do not reuse it; strip the suffix with `name[:-4]`.
- **Do not take durations from `ffprobe` `format.duration`.** 1,318 of the 3,950 train clips (movie clips;
  the local test clips have it too, so this is the original release) carry a third `data` stream (a text /
  chapter track, `SubtitleHandler`) whose duration is the **whole source movie** (for example 4,646 s for the
  48 s clip `Fast.Furious.2009__#00-29-26_00-30-14_label_B1-0-0`). Use the video stream duration or
  `frames / fps` (what `anticipation/features/io.py` `Video` does via decord). `setup/download_verify.py`
  reports the video-stream duration and counts the inflated containers.
- Movie clips mostly have 6-channel (5.1) 48 kHz AAC, a few 7 or 8 channels; YouTube clips mostly stereo
  44.1 kHz (full breakdown in section 7). `io.load_audio` downmixes to
  mono 16 kHz, so this needs no special handling.
- Resolution varies (640x266 to 640x360 for movies, down to 202x360 or 320x240 for some YouTube clips);
  letterboxing in movie clips is a provenance cue (BASELINE 6.5 B-scene0).
- Clip-relative time: XD names encode the source movie time span (`#hh-mm-ss_hh-mm-ss`), but all onsets and
  ticks in this project are **relative to the clip start**. Never use the span in the name as a clock.
- Some clips are silent or music-only (official note). Use `anticipation/features/io.py`
  `is_digital_silence` and set `valid=False` for audio streams on those.
- Movie clips carry editing and score cues; report them separately from YouTube clips (BASELINE 6.4 item 10).

---

## 4. UBI-Fights

### 4.1 Source and transfer
- Page: https://socia-lab.di.ubi.pt/EventDetection/ ; file `UBI_FIGHTS.zip`, 8,117,112,419 bytes, server
  ETag `W/"1e3d14e63-5be33cf447581"`, `Accept-Ranges: bytes`. No form, no login.
- The TLS certificate is expired (notAfter Aug 25 2026), so the script disables certificate verification for
  this host only (`curl -k` equivalent). Every zip member's CRC-32 is checked during extraction.
- **What the CRC check does and does not prove.** The CRC-32 values come from the zip's own central
  directory, delivered over the same unverified connection. A match proves **transfer consistency** (no
  corrupted, truncated or mis-assembled bytes across the 242 range chunks); it does **not** prove
  **authenticity**: a man-in-the-middle who substituted the zip would also supply matching CRCs, and CRC-32 is
  not a cryptographic hash. The publisher offers no checksum to compare against. To make any later
  comparison strong (a re-download, another lab's copy), `manifest.json` now records the **sha256** of every
  extracted file. It was backfilled on 2026-10-14 from the extracted files (archive already deleted), only
  after every file re-matched its CRC-32 (`--backfill-sha256`: 2,001/2,001). Fresh extractions record it
  directly. `--verify-only` re-hashes the tree (2,001/2,001 OK, 59 s).
- The server is slow: one connection measured 2.2 MB/s at first and 0.56 MB/s later. In the real run, 8 range
  connections got 0.8 to 2 MB/s while the XD download (about 60 MB/s from Hugging Face) shared our link, and
  8.5 MB/s once XD had finished, so run UBI alone if speed matters. The script uses 8 connections and 32 MB
  chunks (242 chunks). Do not raise the connection count much; it is an academic server.

### 4.2 Contents (read from the zip central directory)
- 2,006 members: 216 fight videos `videos/fight/F_*.mp4` (1.35 GB), 784 normal videos `videos/normal/N_*.mp4`
  (6.89 GB), 1,000 per-frame label files `annotation/<stem>.csv` (one `0`/`1` per line, one line per frame
  at 30 fps), and `test_videos.csv` (the authors' test split, 67 stems, one per line, no header).
- Filename flags: the site documents `F_<id>_<environment>_<camera>_<color>.mp4` (environment 0 indoor /
  1 outdoor; camera 0 fixed / 1 rotating / 2 movable; color 0 RGB / 1 grayscale), but the actual names carry
  **four** flags after the id (for example `F_0_1_0_0_0`). The meaning of the fourth flag is not documented;
  `segments.csv` keeps the raw flag string.
- All videos are 640x360 at 30 fps. **No audio stream in any of the 1,000 videos** (`has_audio` column of
  `segments.csv`). Audio extractors must write `valid=False` for every UBI tick.
- Measured on the full download (`segments.csv`): 79.0 h in total; fight videos 7.5 h, median 80 s, max
  1,200 s; normal videos 71.5 h, median 64 s, max 6,416 s. 215 fight videos have an onset (`F_113_1_2_0_0`
  has no positive frame), median onset 6.8 s, **45 with at least 30 s of lead-in**, 16 with at least 60 s,
  5 with at least 120 s; 170 fight videos have more than one violent segment. No normal video has a positive
  frame. The official test split holds 40 fight and 27 normal videos.

### 4.3 Layout after extraction (the `UBI_FIGHTS/` prefix is stripped)

```
raw_videos/ubi_fights/
  videos/fight/F_<id>_<f1>_<f2>_<f3>_<f4>.mp4     216
  videos/normal/N_<id>_<f1>_<f2>_<f3>_<f4>.mp4    784
  annotation/<stem>.csv                            1,000 per-frame 0/1 labels
  test_videos.csv                                  official test split (67 stems)
  manifest.json                                    [{name, size, crc32, sha256}] of every extracted file
  segments.csv                                     name, label (F/N), id, flags, split (train/test),
                                                   n_label_frames, label_duration_s, n_violent_frames,
                                                   segments_frames ("s-e;..." end exclusive),
                                                   onset_frame, onset_s (frame / 30), and from an
                                                   ffprobe header read: video_frames, video_duration_s,
                                                   has_audio, frames_match (label rows == frames +-1)
  .extracted.jsonl                                 extraction progress log (resume)
```

The archive and its `.state.json` / `.chunks` files in `downloads/ubi_fights/` are removed automatically after
every member was extracted and CRC-verified (`--keep-archive` keeps them).

### 4.4 Gotchas
- The number of rows in every annotation CSV equals the video frame count (`frames_match` = 1 for all
  1,000), so `onset_s = onset_frame / 30` is on the video clock. The CSVs use CRLF line endings.
- Onsets are first-frame-of-a-run in the per-frame CSV; several fight videos have multiple segments. The
  VA30 label builder must use every segment (BASELINE 6.1), not only the first.
- Group key for splits: video id (BASELINE 6.3). The official 67-video test split is listed in `split` but
  VA30 uses its own grouped folds.

---

## 5. NTU CCTV-Fights: request form required (not submitted)

Page: https://rose1.ntu.edu.sg/dataset/cctvFights/ . 1,000 real fight videos (280 CCTV, 720 non-CCTV), 17.68 h,
2,414 frame-level fight instances, 7.2 GB. The request page (`/dataset/cctvFights/request`) redirects to a
login. There is no mirror on Hugging Face or GitHub, and the terms forbid redistribution, so do not look for
one. Steps for the user:

1. Create an account at https://rose1.ntu.edu.sg/login/ ("Need An Account? Sign Up"). Use the university
   email address; requests are validated by the ROSE Lab.
2. Log in and open https://rose1.ntu.edu.sg/dataset/cctvFights/request . Fill in the request form (name,
   institution, purpose: academic, non-commercial capstone research) and accept the Release Agreement.
3. Wait for approval (manual review; no turnaround time is published). The approval grants a download link for the 7.2 GB dataset.
4. Terms to respect: academic non-commercial use only; **no redistribution and no derived dataset** without
   ROSE Lab permission (so re-cut CCTV-Fights clips cannot be published as part of a VA30 release); images of
   subjects only in academic publications; publications must include the acknowledgment "(Portions of) the
   research in this paper used the NTU CCTV-Fights Dataset made available by the ROSE Lab at the Nanyang
   Technological University, Singapore." and cite Perez, Kot and Rocha, ICASSP 2019.
5. Once the archive is local, place it under `downloads/cctv_fights/` and extract to
   `raw_videos/cctv_fights/`; then compute the at-least-30 s lead-in subset from the frame annotations.

---

## 6. DVD (Kollias et al., arXiv 2506.05372): not publicly released

- arXiv still has only v1 (2025-05-29). The paper states "DVD will be publicly released upon acceptance".
  No release was found in GitHub or Hugging Face searches on 2026-10-14.
- A **subset** of DVD is the data of the "Fine-Grained Violence Detection" challenge of the 9th (ICCV 2025)
  and 10th (CVPR 2026) ABAW competitions: https://affective-behavior-analysis-in-the-wild.github.io/10th/ .
  Access route, as written on that page:
  1. The **lead researcher** sends an email **from an official institutional address** (personal emails are
     rejected) to [contact on the dataset page] with subject `10th ABAW Competition: Team Registration`.
  2. Attach the academic-only EULA, filled in and signed:
     https://drive.google.com/file/d/1-z7JH5HtsuFeIlT9a6t8QlNbMPkBAI2_/view (the VD challenge requires a team
     of academics only).
  3. Include the lead researcher's official academic website; **the lead researcher cannot be a student**
     (UG, PG, PhD), so the capstone advisor must be the lead.
  4. List each team member's email on its own line, the team name, and the point of contact.
  5. The reply grants access to the challenge videos and annotations.
- The CVPR 2026 competition has ended, so a late registration may be refused. An alternative is to email
  the first author ([contact on the dataset page]) and ask about research access to the full DVD release. Nothing
  was submitted by this track.

---

## 7. Verification performed and how to repeat it

| Check | Result (2026-10-14) |
|---|---|
| Mirror test clips vs local `raw_videos/xd_violence/videos` | 798/798 name and size match; sha256 identical on the sampled file |
| XD smoke test (`--limit 3 --order smallest`, then the largest file with an interrupted run and Range resume) | 4/4 sha256-verified; resume after kill produced a correct file |
| ffprobe of downloaded XD train files (`download_verify.py --all --volume`) | 4/4 h264 + AAC 44.1 kHz, 24 fps; durations 1.4 s to 5,278 s; mean volume -14 to -61 dB |
| UBI byte-range chunks | bytes across a chunk boundary identical to a fresh `curl -k -r` fetch |
| UBI extraction logic (synthetic zip with a corrupted member and a `../` member) | CRC failure detected, affected chunks re-fetched, member recovered; unsafe name skipped |
| UBI real members pulled remotely before the full download (`F_37_1_1_0_0`, `N_185_0_0_1_1`) | CRC OK; h264 640x360 30 fps, no audio; label CSV rows = video frames |
| **XD train final** `download_xd_train.py --verify-only` | **3,950 OK, 0 missing or bad** against the mirror manifest (exit 0, 4.4 min to re-hash 74.6 GB). This proves the files match the mirror, not that the mirror is complete (section 3.2) |
| XD official list vs mirror (`download_xd_recover.py official`) | official 3,954, mirror 3,950; official minus mirror 4, mirror minus official 0 |
| XD HIVAU tar provenance (`download_xd_recover.py check`) | 4,754 tar members (3,954 train + 800 test); 4,750 shared with the mirror, **0 size mismatches**; sha256 **11/11** identical (3 smallest + the tar neighbours of each wanted clip); tar-only members = exactly the 4 clips |
| XD recovery (`download_xd_recover.py fetch`) + re-verify | 3 of 4 recovered (429 MB), `v=8cTqh9tMz_I` rejected as truncated. Then `download_xd_train.py --verify-only`: videos/ **3,950/3,950**, missing_test_videos/ **2/2**, recovered_videos/ **3/3** OK, exit 0 (2.5 min); completeness vs the official list **3,953/3,954** (absent: `v=8cTqh9tMz_I__#1_label_A`). md5 of every non-video file and the videos/ listing identical before and after (read-only confirmed). `catalog.discover_files()` now lists 3,953 `xd_train` files |
| UBI sha256 backfill and re-verify (`--backfill-sha256`, `--verify-only`) | 2,001/2,001 files re-matched their CRC-32 and got a sha256; re-hash 2,001/2,001 OK |
| XD normal-mode rerun after the fixes (`--limit 3 --order smallest --include-missing-test`) | 0 files to download, no tree-API call (both manifests cached), `train_labels.csv` byte-identical (3,953 rows) |
| UBI real 2-chunk smoke after the `fdatasync` change (`--max-chunks 2 --no-extract`, scratch dirs) | 2/242 chunks (64 MB) at 4.7 MB/s; archive begins with a valid zip local header for `UBI_FIGHTS/` |
| Downloader self-tests (`setup/download_selftest.py`) | **7/7 passed** in about 10 s (disk budget, read-only verify, UBI central-directory repair in 1 and 2 rounds, UBI sha256 manifest, tar walking across part boundaries, truncated-mp4 rejection, official-list parsing) |
| **XD train final** ffprobe `--all` (`logs/xd_train_ffprobe.json`) | 3,950/3,950 probed, 0 errors, **3,950 with an audio stream**; all 24 fps; 188.9 h total (YouTube 2,135 clips / 133.9 h, other 1,815 / 55.0 h); median 110 s, max 10,816 s (`v=WpxJJFcWM8s__#1_label_A`); 2,992 clips at least 60 s, 142 at least 10 min; 1,318 containers inflated by the data stream (section 3.5). Audio (all AAC): 2,029 44.1 kHz stereo, 1,710 48 kHz 5.1 (6 ch), 110 44.1 kHz mono, 71 48 kHz stereo, 16 48 kHz 7.1 (8 ch), 14 48 kHz 7 ch (sum 3,950) |
| **UBI final** extraction + `segments.csv` | 2,001/2,001 members CRC-verified; 216 fight + 784 normal videos + 1,000 CSVs + `test_videos.csv`; label rows equal video frames (+-1) for **1,000/1,000**; 0 videos with audio; 79.0 h (fight 7.5 h, median 80 s, max 1,200 s; normal 71.5 h, median 64 s, max 6,416 s) |
| UBI lead-in (from `segments.csv`) | 215 fight videos with an onset (`F_113_1_2_0_0` has no positive frame), median onset 6.8 s, **45 at least 30 s**, 16 at least 60 s, 5 at least 120 s; 170 fight videos have more than one violent segment; no normal video has a positive frame. Official test split: 40 fight + 27 normal. Matches research/datasets.md |

Re-run the checks at any time:

```bash
D=data
.venv/bin/python setup/download_xd_train.py --verify-only                      # re-hash all, read-only
.venv/bin/python setup/download_ubi_fights.py --verify-only                    # re-hash UBI, read-only
.venv/bin/python setup/download_selftest.py                                    # offline logic tests
.venv/bin/python -I setup/download_verify.py $D/raw_videos/xd_violence_train/videos --all --json /tmp/xd.json
.venv/bin/python -I setup/download_verify.py $D/raw_videos/ubi_fights/videos --recursive --all
.venv/bin/python setup/download_ubi_fights.py --summarize-only                 # rebuild segments.csv
```

---

## 8. Notes for the feature and benchmark tracks
- Feature files follow `anticipation/features/io.py`: `video_id` = `<source>/<stem>`. Suggested sources:
  `xd` (test and train share one namespace; no file name occurs in both splits), `ubi`.
- **Group leakage across the official XD splits:** 184 of the 1,629 mirror train groups (1,630 official)
  also appear among the 447 test groups: **95 movie titles** (such as `A.Beautiful.Mind.2001`) **and 89 YouTube
  ids**, i.e. the same YouTube source video was cut into clips that landed in both splits (156 test clips and
  549 official train clips belong to those 89 YouTube groups, for example `v=9eME1y6V-T4`). Any split that
  mixes XD train and test videos must group on the `__#` prefix over the pooled set (BASELINE 6.3), for
  YouTube clips as much as for movies, and never reuse the official train/test boundary as if it were
  group-disjoint.
- XD train has 6 `NewAdd.*` clips in the official list (4 NBA game segments, `NewAdd.Dream.Team.8`,
  `NewAdd.xidianuniversity`; the mirror has 5, the 4th NBA segment is recovered) that are neither movies nor
  `v=` YouTube ids; treat each prefix as its own group.
- XD train in the catalog: 3,953 files (3,950 mirror + 3 recovered, all in `verified.jsonl`). The truncated
  `v=8cTqh9tMz_I__#1_label_A` is not under `raw_videos/` and is not catalogued. `docs/VA30.md` still says 3,950.
- XD train clips are 24 fps like the test clips; UBI is 30 fps. Always sample by timestamp (`Video.frames_at`),
  never by frame index.
- UBI has no audio track; `io.load_audio` returns `None`, so audio features must be all-invalid there.
- `raw_videos/xd_violence/` (test) is untouched by this track.
