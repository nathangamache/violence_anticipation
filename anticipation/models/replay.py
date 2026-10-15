"""Re-evaluate frozen historical tabular models on corrected observed horizons and screening folds."""

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from anticipation.eval.va30 import Benchmark
from anticipation.features import io
from anticipation.models.data import FeatureStore, Normalizer
from anticipation.models.evaluation import SCREEN_FOLDS, evaluate_selected


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--horizons", default="benchmark/extract_manifest_v1.csv")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--threads", type=int, choices=range(1, 5), default=2)
    args = p.parse_args(argv)
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        raise ValueError("run_name must be a directory name")
    source = Path(args.run_dir)
    original = json.loads((source / "run_config.json").read_text())
    cfg = original["config"]
    bench = Benchmark.load(cfg["folds"], cfg["videos"], cfg["events"], horizons=args.horizons)
    allowed = {v for v, f in bench.folds.assignment.items() if f in SCREEN_FOLDS}
    saved = pd.read_parquet(source / "scores.parquet", filters=[("fold", "in", list(SCREEN_FOLDS))])
    saved = saved[saved.video_id.isin(allowed)].copy()
    io.FEATURE_ROOT = cfg["feature_root"]
    streams = cfg["streams"].split(",")
    negatives = {v for v in bench.eval_ids() if v in allowed and bench.videos.loc[v, "role"] == "clean_negative"
                 and all(Path(io.feature_path(s, v)).is_file() for s in streams)}
    out = Path("benchmark/results") / args.run_name
    out.mkdir(parents=True, exist_ok=False)
    catalog = bench.videos.copy()
    for vid in bench.video_horizons:
        catalog.loc[vid, "duration_sec"] = bench.duration(vid)
    with threadpool_limits(limits=args.threads):
        store = FeatureStore(out / "cache", catalog, bench.events, streams)
        coverage = store.snapshot(sorted(set(saved.video_id) | negatives))
        rows = [saved]
        for fold in SCREEN_FOLDS:
            ids = (set(bench.folds.test_ids(fold)) | set(bench.folds.val_ids(fold))) & negatives
            for seed in original["seeds"]:
                with (source / "checkpoints" / f"fold{fold}_seed{seed}.pkl").open("rb") as f:
                    model = pickle.load(f)
                norm = Normalizer.load(source / "checkpoints" / f"normalizer_fold{fold}.npz")
                for vid in sorted(ids):
                    if vid not in store.paths or ((saved.fold == fold) & (saved.seed == seed) & (saved.video_id == vid)).any():
                        continue
                    video = store[vid]
                    ix = np.flatnonzero(video.observed)
                    rows.append(pd.DataFrame({"video_id": vid, "t_sec": video.t_sec[ix],
                                              "score": model.predict(video, norm, indices=ix), "fold": fold, "seed": seed}))
                print(f"[replay] fold {fold} seed {seed}: frozen checkpoint; added negative coverage", flush=True)
        scores = pd.concat(rows, ignore_index=True)
        scores.to_parquet(out / "scores.parquet", index=False)
        meta = {"status": "scored", "source_run": str(source), "source_training_commit": original["git_commit"],
                "purpose": "historical diagnostic only; no refitting or hyperparameter selection",
                "historical_training_note": "models predate the fold-1 freeze; no fold-1 results are evaluated here",
                "fold_ids": list(SCREEN_FOLDS), "horizons": args.horizons, "benchmark": bench.provenance,
                "video_count": int(scores.video_id.nunique()), "score_rows": len(scores),
                "available_clean_negatives": len(negatives), "argv": [sys.executable, *sys.argv],
                "source_config": original}
        (out / "run_config.json").write_text(json.dumps(meta, indent=2) + "\n")
        (out / "feature_hashes.json").write_text(json.dumps({v: r["feature_sha256"] for v, r in coverage["videos"].items()}, indent=2) + "\n")
        try:
            evaluate_selected(scores, bench, SCREEN_FOLDS, out, args.run_name, args.n_boot,
                              threads=args.threads, meta=meta)
        except Exception as exc:
            meta.update(status="evaluation_unavailable", error=str(exc))
            (out / "run_config.json").write_text(json.dumps(meta, indent=2) + "\n")
            raise
        meta["status"] = "complete"
        (out / "run_config.json").write_text(json.dumps(meta, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
