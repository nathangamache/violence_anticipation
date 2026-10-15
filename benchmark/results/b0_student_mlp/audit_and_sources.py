"""Reproduce source breakdowns and the ranked B0 visual audit from saved scores."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from anticipation.eval import va30
from anticipation.eval.va30 import (Benchmark, EvalConfig, EventEval, ScoreTable, TickSet,
                                    bootstrap_weights, position_null, null_at, threshold_free)
from tools.contact_sheet import render_sheets

NAMES = ("b0_student_mlp", "b_scene0", "b0_student_mlp_teacher_subset")
ROOT = Path("benchmark/results")


def source_results(bench):
    va30.N_THREADS = 4
    for name in NAMES:
        out = ROOT / name
        result = json.loads((out / "metrics.json").read_text())
        table = ScoreTable(pd.read_parquet(out / "scores.parquet"))
        subsets = {s: [v for v in sorted(table.videos) if bench.videos.loc[v, "source_kind"] == s]
                   for s in ("xd_movie", "xd_youtube", "ucf")}
        subsets["xd"] = sorted(subsets["xd_movie"] + subsets["xd_youtube"])
        summary = {"definition": "Source-only test ticks and negatives; original pooled-validation thresholds held fixed. 1000 video-bootstrap CIs per seed; source-only position null at achieved source FA/h.",
                   "folds_content_sha256": bench.folds.sha256, "seeds": {}}
        for seed in table.seeds:
            by_seed = {}
            for source, vids in subsets.items():
                folds_of = {v: bench.folds.fold_of(v) for v in vids}
                scores = {v: table.aligned(seed, folds_of[v], v, bench.labels(v).t_sec)[0] for v in vids}
                ts = TickSet(bench, vids, scores, folds_of)
                W = bootstrap_weights(ts, 1000, 7919 * seed)
                cfg = EvalConfig(horizons=(30,), n_boot=1000)
                metrics = threshold_free(ts, W, cfg)
                ee = EventEval(ts, cfg.policy)
                null = position_null(ee)
                metrics["counts"] = {"videos": ts.V, "violent": int(ts.violent_v.sum()),
                                     "clean_negative": int(ts.clean_v.sum()), "clean_hours_unmasked": ee.clean_hours,
                                     "eligible30_events": int(ee.ev_e30.sum())}
                ops = {}
                for key, op in result["seeds"][str(seed)]["operating_points"].items():
                    thresholds = np.array([float(op["theta_per_fold"][str(f)]) for f in ts.fold_v])
                    stats = ee.summarize(ee.outcomes(thresholds), W)
                    stats["position_null_success30"] = null_at(null, stats["fa_per_hour_clean"]["value"])
                    stats["thresholds"] = "Held fixed at the pooled-validation operating point; no per-source retuning"
                    ops[key] = stats
                metrics["operating_points"] = ops
                by_seed[source] = metrics
            summary["seeds"][str(seed)] = by_seed
        (out / "source_metrics.json").write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")
        print(f"Sources: {name}", flush=True)


def ranked_ticks(bench):
    df = pd.read_parquet(ROOT / NAMES[0] / "scores.parquet")
    df = df[df.seed == 0]
    clean, pre = [], []
    for vid, group in df.groupby("video_id"):
        group = group[group.fold == bench.folds.fold_of(vid)]
        lab = bench.labels(vid)
        row = bench.videos.loc[vid]
        if row.role == "clean_negative":
            select = (~lab.masked) & (lab.tte > 120) & (lab.remaining >= 60)
            dest = clean
        else:
            select = (~lab.masked) & lab.pre_violence & np.isfinite(lab.tte)
            dest = pre
        selected = group[group.t_sec.isin(lab.t_sec[select])]
        for r in selected.sort_values(["score", "t_sec"], ascending=[False, True]).head(20).itertuples():
            tick = int(r.t_sec) - 1
            dest.append({"video_id": vid, "t_sec": float(r.t_sec), "score": float(r.score),
                         "time_to_onset": None if not np.isfinite(lab.tte[tick]) else float(lab.tte[tick]),
                         "source_kind": row.source_kind, "role": row.role, "path": row.path})
    order = lambda records: sorted(records, key=lambda r: (-r["score"], r["video_id"], r["t_sec"]))
    diverse = lambda records: list({r["video_id"]: r for r in reversed(order(records))}.values())
    sets = {"clean_top20_ticks": order(clean)[:20], "preonset_top20_ticks": order(pre)[:20],
            "clean_diverse20": order(diverse(clean))[:20], "preonset_diverse20": order(diverse(pre))[:20]}
    return sets


def contact_sheets(bench):
    out = ROOT / NAMES[0]
    sets = ranked_ticks(bench)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    for name, records in sets.items():
        canvas = Image.new("RGB", (1280, 5 * 255), "white")
        draw = ImageDraw.Draw(canvas)
        for rank, record in enumerate(records, 1):
            t = record["t_sec"]
            sheet = render_sheets(record["path"], str(out / "audit_tiles"), start=t, end=t + .01,
                                  step=1, cols=1, rows=1, tile_w=320, prefix=f"{name}_{rank:02d}")[0]
            tile = Image.open(sheet).convert("RGB")
            tile.thumbnail((320, 205))
            x, y = ((rank - 1) % 4) * 320, ((rank - 1) // 4) * 255
            canvas.paste(tile, (x, y))
            draw.text((x + 2, y + 207), f"#{rank} {record['source_kind']} p={record['score']:.4f}", font=font, fill="black")
            draw.text((x + 2, y + 222), record["video_id"].split("/", 1)[1][:42], font=font, fill="black")
            draw.text((x + 2, y + 237), f"tick={t:g}s; onset in {record['time_to_onset']}s", font=font, fill="black")
            record["rank"] = rank
            Path(sheet).unlink()
        path = out / f"audit_{name}.jpg"
        for quality in (82, 72, 62, 50):
            canvas.save(path, quality=quality, optimize=True)
            if path.stat().st_size <= 300000:
                break
        assert path.stat().st_size <= 300000
        print(f"Audit: {name}, {len(set(r['video_id'] for r in records))} videos, {path.stat().st_size} bytes", flush=True)
    (out / "audit_tiles").rmdir()
    previous_path = out / "shortcut_audit.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    annotations = {(r["video_id"], r["t_sec"]): {k: r[k] for k in
                   ("scene_type", "people_count_visual", "visible_cues", "escalation_assessment") if k in r}
                   for records in previous.get("sheets", {}).values() for r in records}
    for records in sets.values():
        for record in records:
            record.update(annotations.get((record["video_id"], record["t_sec"]), {}))
    report = {"selection": "Seed 0, test fold only. Literal top-20 tick rankings, descending score then video_id/tick; clean negatives unmasked with >=60 seconds remaining, pre-onset unmasked before the first annotated anomaly and with a future onset. Diverse companion sheets keep the highest tick per video. No labels changed.",
              "renderer": "tools/contact_sheet.py render_sheets, one sampled tick per tile, four decoder threads, 320px tiles",
              "audio": "Neither model consumes audio; no music or speech audit was performed.", "sheets": sets}
    for key in ("summary", "annotation_note"):
        if key in previous:
            report[key] = previous[key]
    (out / "shortcut_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources_only", action="store_true")
    ap.add_argument("--sheets_only", action="store_true")
    args = ap.parse_args()
    bench = Benchmark.load("benchmark/folds_v2.json")
    if not args.sheets_only:
        source_results(bench)
    if not args.sources_only:
        contact_sheets(bench)


if __name__ == "__main__":
    main()
