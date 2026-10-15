#!/usr/bin/env python3
"""Download HIVAU-70k raw annotations (and UCA captions), build onsets in seconds, cross-check vs official.

HIVAU-70k (Holmes-VAU, CVPR 2025, MIT licence, https://github.com/pipixin321/HolmesVAU) gives human event
start/end times for the UCF-Crime and XD-Violence TRAIN splits (plus the test splits). The four raw files are
small JSON (about 49 MB in total) and download from raw.githubusercontent.com in seconds.

Usage (from the repo root):

    .venv/bin/python setup/build_hivau_onsets.py                # download if missing, build, cross-check
    .venv/bin/python setup/build_hivau_onsets.py --no_download  # use files already on disk
    .venv/bin/python setup/build_hivau_onsets.py --no_uca       # skip the UCA caption download

Outputs (all under data/annotations/):

    hivau/raw_annotations/{ucf,xd}_database_{train,test}.json   raw files, untouched
    hivau/README.md, hivau/LICENSE                               provenance
    hivau/hivau_onsets.json      {"meta", "xd": {stem: rec}, "ucf": {stem: rec}, "normals", "xd_cctv_hits"}
                                 rec = {split, classes, labels, events_sec, fps, n_frames}; ONLY entries with
                                 a non-empty ``label`` (normal videos carry random "events" that are dropped)
    hivau/crosscheck_vs_official.json   agreement of HIVAU and official onsets on the XD / UCF test videos,
                                 annotation-style statistics per split, and independent checks
    uca/json/UCFCrime_{Train,Val,Test}.json, uca/README.md      UCA dense captions (Yuan et al., CVPR 2024)

What the cross-check does and does NOT show: HIVAU's TEST entries are copies of the official annotations (XD
``events(frame)`` equals annotations.txt for 500/500 clips), so agreement there only validates our parsing and
unit conversion. It says nothing about the TRAIN-split onsets, which HIVAU annotated itself in a different style
(longer segments, more segments per video; see ``style``). The only independent evidence available offline is
footage that appears twice: a test copy with official onsets and a train copy with HIVAU onsets
(``independent_checks``, from benchmark/near_duplicates.json). A human spot-check of HIVAU train onsets is
prepared by ``python -m anticipation.data.spotcheck``.

The catalog (``python -m anticipation.data.catalog``) reads hivau_onsets.json. Official onsets always win
for the test splits; HIVAU onsets are used for train-split videos only.
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anticipation.data import annotations as A  # noqa: E402
from anticipation.data.labels import event_table, normalize_segments  # noqa: E402

HIVAU_BASE = "https://raw.githubusercontent.com/pipixin321/HolmesVAU/master"
UCA_BASE = "https://raw.githubusercontent.com/Xuange923/Surveillance-Video-Understanding/main"
UCA_FILES = ("UCFCrime_Train.json", "UCFCrime_Val.json", "UCFCrime_Test.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest, timeout=600):
    """Download ``url`` to ``dest`` atomically unless it already exists (non-empty)."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return False
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)
    return True


def fetch_hivau():
    for fname in A.HIVAU_RAW_FILES:
        url = f"{HIVAU_BASE}/HIVAU-70k/raw_annotations/{fname}"
        if download(url, os.path.join(A.HIVAU_RAW_DIR, fname)):
            print(f"[hivau] downloaded {fname}")
    for fname in ("README.md", "LICENSE"):
        download(f"{HIVAU_BASE}/{fname}", os.path.join(A.HIVAU_DIR, fname))


def fetch_uca():
    for fname in UCA_FILES:
        url = f"{UCA_BASE}/UCF%20Annotation/json/{fname}"
        if download(url, os.path.join(A.UCA_DIR, "json", fname)):
            print(f"[uca] downloaded {fname}")
    download(f"{UCA_BASE}/README.md", os.path.join(A.UCA_DIR, "README.md"))


# ---------------------------------------------------------------------------------------------------------
# Cross-check
# ---------------------------------------------------------------------------------------------------------

def union_iou(a, b):
    """Temporal IoU of two segment lists (seconds)."""
    a, b = normalize_segments(a), normalize_segments(b)
    if not a and not b:
        return 1.0
    pts = sorted({x for s in a + b for x in s})
    inter = union = 0.0
    for lo, hi in zip(pts[:-1], pts[1:]):
        mid = 0.5 * (lo + hi)
        ina = any(s <= mid <= e for s, e in a)
        inb = any(s <= mid <= e for s, e in b)
        union += (hi - lo) * (ina or inb)
        inter += (hi - lo) * (ina and inb)
    return inter / union if union > 0 else 0.0


def compare(official, hivau_raw, name):
    """Agreement stats for videos annotated in ``official`` ({stem: segments_sec}) vs HIVAU raw entries."""
    rows = []
    missing, empty_label = [], []
    for vid, segs in sorted(official.items()):
        if vid not in hivau_raw:
            missing.append(vid)
            continue
        classes, hv = A.hivau_entry_events(hivau_raw[vid])
        if classes is None:
            empty_label.append(vid)
            continue
        so, sh = normalize_segments(segs), normalize_segments(hv)
        if not so or not sh:
            continue
        eo, eh = event_table(so), event_table(sh)
        rows.append({
            "vid": vid, "d_onset": sh[0][0] - so[0][0], "iou": union_iou(so, sh),
            "nseg_o": len(so), "nseg_h": len(sh),
            "e30_o": any(e["eligible30"] for e in eo), "e30_h": any(e["eligible30"] for e in eh),
            "f30_o": so[0][0] >= 30, "f30_h": sh[0][0] >= 30,
        })
    extra = [v for v, e in hivau_raw.items() if (e.get("label") or []) and v not in official]
    d = np.array([r["d_onset"] for r in rows])
    ad = np.abs(d)
    iou = np.array([r["iou"] for r in rows])

    def conf(ko, kh):
        return {"both": sum(r[ko] and r[kh] for r in rows), "official_only": sum(r[ko] and not r[kh] for r in rows),
                "hivau_only": sum(r[kh] and not r[ko] for r in rows),
                "neither": sum(not r[ko] and not r[kh] for r in rows)}

    out = {
        "name": name, "official_annotated": len(official), "compared": len(rows),
        "missing_in_hivau": missing, "official_annotated_but_hivau_normal": empty_label,
        "hivau_anomalous_not_in_official": len(extra),
        "first_onset_abs_diff_sec": {
            "median": float(np.median(ad)) if len(ad) else None, "mean": float(ad.mean()) if len(ad) else None,
            "p90": float(np.percentile(ad, 90)) if len(ad) else None, "max": float(ad.max()) if len(ad) else None,
            "frac_le_0.5s": float((ad <= 0.5).mean()) if len(ad) else None,
            "frac_le_1s": float((ad <= 1.0).mean()) if len(ad) else None,
            "frac_le_5s": float((ad <= 5.0).mean()) if len(ad) else None,
            "mean_signed_hivau_minus_official": float(d.mean()) if len(d) else None,
        },
        "segment_union_iou": {"mean": float(iou.mean()) if len(iou) else None,
                              "median": float(np.median(iou)) if len(iou) else None,
                              "frac_ge_0.9": float((iou >= 0.9).mean()) if len(iou) else None},
        "same_segment_count": float(np.mean([r["nseg_o"] == r["nseg_h"] for r in rows])) if rows else None,
        "eligible30_any_event": conf("e30_o", "e30_h"),
        "first_onset_ge30": conf("f30_o", "f30_h"),
        "largest_disagreements": sorted(
            [{"vid": r["vid"], "d_onset": round(r["d_onset"], 2), "iou": round(r["iou"], 3)} for r in rows],
            key=lambda r: -abs(r["d_onset"]))[:10],
    }
    return out


def train_stats(proc, src):
    """Counts of HIVAU-labelled train videos and how many give eligible events."""
    out = {}
    recs = {k: v for k, v in proc[src].items() if v["split"] == "train"}
    out["labelled_videos"] = len(recs)
    n30 = n60 = n30v = n30i = 0
    by_cls = {}
    for vid, r in recs.items():
        ev = event_table(normalize_segments(r["events_sec"]))
        e30 = any(e["eligible30"] for e in ev)
        n30 += sum(e["eligible30"] for e in ev)
        n60 += sum(e["eligible60"] for e in ev)
        v = bool(set(r["classes"]) & A.VIOLENT_CLASSES)
        n30v += e30 and v
        n30i += e30 and bool(set(r["classes"]) & A.INTERPERSONAL_CLASSES)
        for c in r["classes"]:
            by_cls.setdefault(c, [0, 0])
            by_cls[c][0] += 1
            by_cls[c][1] += e30
    out.update(eligible30_events=n30, eligible60_events=n60, eligible30_violent_videos=n30v,
               eligible30_interpersonal_videos=n30i,
               by_class={c: {"videos": a, "eligible30_videos": b} for c, (a, b) in sorted(by_cls.items())})
    return out


def style_stats(proc):
    """Annotation style per source and split: segment length, segments per video, later-segment share."""
    out = {}
    for src in ("xd", "ucf"):
        for split in ("train", "test"):
            recs = [r for r in proc[src].values() if r["split"] == split]
            if not recs:
                continue
            lens, nseg, later, e30 = [], [], 0, 0
            for r in recs:
                segs = normalize_segments(r["events_sec"])
                lens += [e - s for s, e in segs]
                nseg.append(len(segs))
                for e in event_table(segs):
                    if e["eligible30"]:
                        e30 += 1
                        later += e["index"] > 0
            nseg = np.array(nseg)
            out[f"{src}_{split}"] = {
                "videos": len(recs), "median_segment_sec": round(float(np.median(lens)), 2) if lens else None,
                "mean_segments_per_video": round(float(nseg.mean()), 3),
                "frac_videos_ge3_segments": round(float((nseg >= 3).mean()), 3),
                "eligible30_events": e30, "eligible30_later_segment_share": round(later / e30, 3) if e30 else None}
    return out


def independent_checks(proc):
    """Footage present twice (near-duplicate pairs): official onsets on one copy, HIVAU train onsets on the other."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark",
                        "near_duplicates.json")
    if not os.path.exists(path):
        return {"note": "benchmark/near_duplicates.json not built yet"}
    with open(path) as f:
        nd = json.load(f)
    ucf_off = {k: v for k, v in A.read_ucf_official().items() if v["frames"]}
    xd_off = A.read_xd_official()
    rows = []
    for p in nd.get("pairs", []):
        for a, b in ((p["a"], p["b"]), (p["b"], p["a"])):
            sa, stem_a = a.split("/", 1)
            sb, stem_b = b.split("/", 1)
            off = None
            if sa == "ucf":
                off = ucf_off.get(stem_a, {}).get("frames")
            elif sa == "xd":
                off = xd_off.get(stem_a)
            hv = proc.get(sb, {}).get(stem_b) if sb in ("ucf", "xd") else None
            if not off or not hv or hv["split"] != "train" or p.get("lag_sec", 0) != 0:
                continue
            fps = 30.0 if sa == "ucf" else 24.0
            o = normalize_segments([[s / fps, e / fps] for s, e in off])
            h = normalize_segments(hv["events_sec"])
            if o and h:
                rows.append({"official_copy": a, "hivau_train_copy": b, "official_segments": o, "hivau_segments": h,
                             "first_onset_diff_sec": round(h[0][0] - o[0][0], 3),
                             "segment_iou": round(union_iou(o, h), 3), "pair_kind": p["kind"]})
    return {"pairs": rows, "n": len(rows),
            "note": "same footage annotated twice: official (test) vs HIVAU (train); the only independent check of "
                    "HIVAU train onsets available without new human labels"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no_download", action="store_true")
    ap.add_argument("--no_uca", action="store_true")
    args = ap.parse_args(argv)

    if not args.no_download:
        fetch_hivau()
        if not args.no_uca:
            try:
                fetch_uca()
            except Exception as e:  # UCA is optional
                print(f"[uca] WARNING: download failed: {e}")

    proc = A.build_hivau_onsets_from_raw()
    proc["meta"].update({
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": f"{HIVAU_BASE}/HIVAU-70k/raw_annotations",
        "sha256": {f: sha256(os.path.join(A.HIVAU_RAW_DIR, f)) for f in A.HIVAU_RAW_FILES
                   if os.path.exists(os.path.join(A.HIVAU_RAW_DIR, f))},
        "note": "only entries with a non-empty label; events_sec in seconds as given by HIVAU",
    })
    os.makedirs(A.HIVAU_DIR, exist_ok=True)
    tmp = A.HIVAU_ONSETS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(proc, f)
    os.replace(tmp, A.HIVAU_ONSETS)
    print(f"[hivau] wrote {A.HIVAU_ONSETS}: xd {len(proc['xd'])} anomalous, ucf {len(proc['ucf'])} anomalous; "
          f"normals xd {len(proc['normals']['xd'])}, ucf {len(proc['normals']['ucf'])}")

    # Cross-check on the test splits where both exist.
    xd_raw = A.read_hivau_raw(os.path.join(A.HIVAU_RAW_DIR, "xd_database_test.json"))
    ucf_raw = A.read_hivau_raw(os.path.join(A.HIVAU_RAW_DIR, "ucf_database_test.json"))
    xd_off = {k: [[s / 24.0, e / 24.0] for s, e in v] for k, v in A.read_xd_official().items()}
    ucf_off = {}
    for k, v in A.read_ucf_official().items():
        if v["frames"]:
            fps = float(ucf_raw.get(k, {}).get("fps") or 30.0)
            ucf_off[k] = [[s / fps, e / fps] for s, e in v["frames"]]
    report = {
        "created": proc["meta"]["created"],
        "xd_test": compare(xd_off, xd_raw, "XD-Violence test (official annotations.txt vs HIVAU)"),
        "ucf_test": compare(ucf_off, ucf_raw, "UCF-Crime test (official temporal annotation vs HIVAU)"),
        "hivau_train_coverage": {"ucf": train_stats(proc, "ucf"), "xd": train_stats(proc, "xd")},
        "style": style_stats(proc),
        "independent_checks": independent_checks(proc),
        "interpretation": "HIVAU TEST entries are copies of the official annotations, so the xd_test / ucf_test "
                          "agreement validates parsing and unit conversion only. HIVAU TRAIN onsets (most VA30 "
                          "eligible events) are not validated by it; see style and independent_checks, and run the "
                          "human spot-check (python -m anticipation.data.spotcheck).",
        "files": proc["meta"]["files"],
    }
    path = os.path.join(A.HIVAU_DIR, "crosscheck_vs_official.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=1)
    for key in ("xd_test", "ucf_test"):
        r = report[key]
        print(f"[crosscheck] {r['name']}: compared {r['compared']}/{r['official_annotated']}, "
              f"HIVAU-normal {len(r['official_annotated_but_hivau_normal'])}, missing {len(r['missing_in_hivau'])}; "
              f"|d onset| median {r['first_onset_abs_diff_sec']['median']}, <=1s {r['first_onset_abs_diff_sec']['frac_le_1s']}, "
              f"IoU mean {r['segment_union_iou']['mean']}; eligible30 {r['eligible30_any_event']}")
    for src in ("ucf", "xd"):
        print(f"[hivau] train {src}: {json.dumps({k: v for k, v in report['hivau_train_coverage'][src].items() if k != 'by_class'})}")
    for k, v in report["style"].items():
        print(f"[hivau] style {k}: {json.dumps(v)}")
    ic = report["independent_checks"]
    for r in ic.get("pairs", []):
        print(f"[hivau] independent check {r['official_copy']} (official) vs {r['hivau_train_copy']} (HIVAU train): "
              f"first onset diff {r['first_onset_diff_sec']} s, IoU {r['segment_iou']}")
    print(f"[hivau] wrote {path}")


if __name__ == "__main__":
    main()
