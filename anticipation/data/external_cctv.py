"""NTU CCTV-Fights external test set: paths, loaders and catalog construction (docs/EXTERNAL_TEST.md).

Held out from training, folds and the modality screen. Files live in benchmark/external/cctv_fights/ and use the
column / key format of benchmark/videos.csv and benchmark/events.json with ``source = cctv_fights``.
"""

import csv
import json
import os

from anticipation.data import labels

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.environ.get("VA_DATA_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"))
RAW_DIR = os.path.join(DATA_ROOT, "raw_videos", "cctv_fights")
GT_JSON = os.path.join(DATA_ROOT, "downloads", "cctv_fights", "groundtruth.json")
PROBE_JSON = os.path.join(DATA_ROOT, "cache", "cctv_fights", "probe.json")
OUT_DIR = os.path.join(REPO, "benchmark", "external", "cctv_fights")
VIDEOS_CSV = os.path.join(OUT_DIR, "videos.csv")
EVENTS_JSON = os.path.join(OUT_DIR, "events.json")
OVERLAP_JSON = os.path.join(OUT_DIR, "overlap_with_main.json")
SOURCE = "cctv_fights"


def load_groundtruth():
    with open(GT_JSON) as f:
        return json.load(f)["database"]


def video_path(stem):
    for ext_ in (".mp4", ".webm"):
        p = os.path.join(RAW_DIR, "videos", stem + ext_)
        if os.path.exists(p):
            return p
    return os.path.join(RAW_DIR, "videos", stem + ".mp4")


def load_probes():
    with open(PROBE_JSON) as f:
        return json.load(f)


def build_catalog():
    gt = load_groundtruth()
    probes = load_probes()
    rows, evs = [], {}
    for stem in sorted(gt):
        g, p = gt[stem], probes[stem]
        ok = "fps" in p
        dur = float(p["duration_sec"]) if ok else 0.0
        segs = labels.normalize_segments([a["segment"] for a in g["annotations"]], dur if ok else None)
        events = labels.event_table(segs)
        vid = f"{SOURCE}/{stem}"
        cctv = g["source"] == "CCTV"
        reliable = ok and abs(dur - g["duration"]) <= max(1.0, 0.02 * g["duration"])
        n30, n60 = sum(e["eligible30"] for e in events), sum(e["eligible60"] for e in events)
        rows.append({
            "video_id": vid, "source": SOURCE, "path": p["path"], "duration_sec": round(dur, 4),
            "fps": round(p.get("fps", 0.0), 4), "num_frames": p.get("num_frames", 0),
            "has_audio": bool(p.get("has_audio", False)), "group": f"{SOURCE}:{stem}", "official_split": g["subset"],
            "category": "fight", "is_violent_video": True, "is_clean_negative": False, "onset_source": "official",
            "first_onset_sec": segs[0][0] if segs else "", "eligible30": n30 > 0, "eligible60": n60 > 0,
            "domain": "cctv" if cctv else "non_cctv", "role": "violent", "classes": "fighting", "interpersonal": True,
            "n_segments": len(segs), "n_eligible30": n30, "n_eligible60": n60, "source_kind": SOURCE,
            "audio_stream": bool(p.get("audio_stream", False)), "audio_peak": round(p.get("audio_peak", 0.0), 6),
            "width": p.get("width", 0), "height": p.get("height", 0), "gt_source": g["source"],
            "gt_duration_sec": round(g["duration"], 4), "gt_frame_rate": g["frame_rate"],
            "decode_ok": ok, "label_reliable": reliable, "split_group": f"{SOURCE}:{stem}", "overlap_main": False, "overlap_with": "",
        })
        evs[vid] = {"segments": [[round(s, 4), round(e, 4)] for s, e in segs], "onset_source": "official",
                    "classes": ["fighting"], "category": "fight", "role": "violent", "is_violent_video": True,
                    "interpersonal": True, "duration_sec": round(dur, 4),
                    "events": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items()}
                               for e in events]}
    os.makedirs(OUT_DIR, exist_ok=True)
    _write_csv(rows)
    meta = {"version": 1, "units": "seconds", "source": "NTU CCTV-Fights groundtruth.json (version 1.0)",
            "notes": "External test set. segments = sorted merged fight segments, clipped to the decoded duration; "
                     "events per labels.event_table (EARLIEST_ALARM_TICK=2, 10 s aftermath), no blockers."}
    with open(EVENTS_JSON, "w") as f:
        json.dump({"meta": meta, "videos": evs}, f, indent=1)
    write_summary()
    return rows


def _write_csv(rows):
    with open(VIDEOS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def load_rows():
    with open(VIDEOS_CSV) as f:
        return list(csv.DictReader(f))


def _b(x):
    return str(x).lower() == "true"


def write_summary():
    rows = load_rows()
    with open(EVENTS_JSON) as f:
        evs = json.load(f)["videos"]

    def block(sel):
        r = [x for x in sel]
        out = {"videos": len(r), "hours": round(sum(float(x["duration_sec"]) for x in r) / 3600, 3),
               "with_audio": sum(_b(x["has_audio"]) for x in r),
               "audio_hours": round(sum(float(x["duration_sec"]) for x in r if _b(x["has_audio"])) / 3600, 3),
               "fight_instances": 0, "eligible30_first": 0, "eligible30_later": 0, "eligible60_first": 0,
               "eligible60_later": 0, "videos_eligible30": 0, "videos_eligible60": 0}
        for x in r:
            es = evs[x["video_id"]]["events"]
            out["fight_instances"] += len(es)
            for e in es:
                key = "first" if e["index"] == 0 else "later"
                out[f"eligible30_{key}"] += e["eligible30"]
                out[f"eligible60_{key}"] += e["eligible60"]
            out["videos_eligible30"] += any(e["eligible30"] for e in es)
            out["videos_eligible60"] += any(e["eligible60"] for e in es)
        return out

    rel = [x for x in rows if _b(x["label_reliable"])]
    clean = [x for x in rel if not _b(x["overlap_main"])]
    summ = {"all": block(rows), "label_reliable": block(rel), "evaluation_set": block(clean),
            "label_unreliable": [x["video_id"] for x in rows if not _b(x["label_reliable"])]}
    for dom in ("cctv", "non_cctv"):
        summ[dom] = block([x for x in rows if x["domain"] == dom])
        summ[f"{dom}_evaluation_set"] = block([x for x in clean if x["domain"] == dom])
    for sp in ("training", "validation", "testing"):
        summ[f"official_{sp}"] = block([x for x in rows if x["official_split"] == sp])
    summ["decode_failures"] = [x["video_id"] for x in rows if not _b(x["decode_ok"])]
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    return summ


def run_overlap(workers=3):
    import pandas as pd

    from anticipation.data import near_dups as nd
    from anticipation.data.catalog import VIDEOS_CSV as MAIN_CSV

    main = pd.read_csv(MAIN_CSV)
    main = main[main.duration_sec > 0]
    rows = load_rows()
    ext_rows = [(r["video_id"], r["path"], SOURCE, r["group"], float(r["duration_sec"]), r["role"])
                for r in rows if _b(r["decode_ok"])]
    todo = [r[1] for r in ext_rows if nd._cached(r[1]) is None]
    todo += [p for p in main.path if nd._cached(p) is None]
    todo = list(dict.fromkeys(todo))
    print(f"fingerprints to compute: {len(todo)} (external {sum(nd._cached(r[1]) is None for r in ext_rows)})",
          flush=True)
    if todo:
        from multiprocessing import Pool

        os.makedirs(nd.FP_DIR, exist_ok=True)
        with Pool(workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(nd._work, todo, chunksize=1)):
                if (i + 1) % 250 == 0:
                    print(f"  {i + 1}/{len(todo)}", flush=True)
    main_rows = list(main[["video_id", "path", "source_kind", "group", "duration_sec", "role"]]
                     .itertuples(index=False, name=None))
    pairs, n = nd.find_pairs(ext_rows + main_rows)
    ext_ids = {r[0] for r in ext_rows}
    cross = [p for p in pairs if (p["a"] in ext_ids) != (p["b"] in ext_ids)]
    within = [p for p in pairs if p["a"] in ext_ids and p["b"] in ext_ids]
    hit = {}
    for p in cross:
        e, m = (p["a"], p["b"]) if p["a"] in ext_ids else (p["b"], p["a"])
        hit.setdefault(e, []).append(m)
    for r in rows:
        r["overlap_main"] = r["video_id"] in hit
        r["overlap_with"] = ";".join(sorted(hit.get(r["video_id"], [])))
    _write_csv(rows)
    out = {"meta": {"method": "anticipation.data.near_dups.find_pairs over external + main catalog, read-only",
                    "videos_fingerprinted": n, "external_checked": len(ext_rows)},
           "n_external_overlapping": len(hit), "external_videos": sorted(hit),
           "cross_pairs": cross, "within_external_pairs": within}
    with open(OVERLAP_JSON, "w") as f:
        json.dump(out, f, indent=1)
    write_summary()
    print(f"cross pairs {len(cross)}, overlapping external videos {len(hit)}, within-external pairs {len(within)}")


def write_manifests():
    rows = load_rows()
    cols = ["video_id", "source", "source_kind", "role", "priority", "eligible30", "has_audio", "path",
            "duration_sec", "fps", "num_frames", "domain", "overlap_main"]
    ok = [r for r in rows if _b(r["decode_ok"])]
    ok.sort(key=lambda r: (not _b(r["eligible30"]), r["video_id"]))
    out = []
    for r in ok:
        d = {c: r.get(c, "") for c in cols}
        d["priority"] = "external_eligible30" if _b(r["eligible30"]) else "external"
        out.append(d)
    for name, sel in (("extract_manifest.csv", out), ("extract_manifest_audio.csv", [d for d in out if _b(d["has_audio"])])):
        with open(os.path.join(OUT_DIR, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(sel)
        print(name, len(sel), "videos", round(sum(float(d["duration_sec"]) for d in sel) / 3600, 2), "h")
