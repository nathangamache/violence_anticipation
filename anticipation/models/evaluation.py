"""Observed-horizon evaluation without exposing the frozen confirmation fold."""

import numpy as np

from anticipation.data.folds import Folds, content_hash
from anticipation.eval import va30

SCREEN_FOLDS = (0, 2, 3, 4)


def evaluation_subset(bench, scores, fold_ids):
    mapping = {f: i for i, f in enumerate(fold_ids)}
    allowed = {v for v, f in bench.folds.assignment.items() if f in mapping}
    assignment = {v: mapping[f] for v, f in bench.folds.assignment.items() if v in allowed}
    val = {str(mapping[f]): [v for v in bench.folds.val_ids(f) if v in allowed] for f in fold_ids}
    data = {**bench.folds.d, "n_folds": len(mapping), "assignment": assignment, "val": val,
            "content_sha256": content_hash(assignment, val)}
    provenance = {**bench.provenance, "evaluated_original_fold_ids": list(fold_ids),
                  "internal_to_original_fold": {str(i): f for f, i in mapping.items()},
                  "evaluation_subset_sha256": data["content_sha256"]}
    videos = bench.videos.copy()
    if "split_group" in videos:
        videos["group"] = videos["split_group"]
    subset = va30.Benchmark(videos, bench.events, Folds(data), provenance)
    subset.video_horizons = dict(bench.video_horizons)
    selected = scores[scores.fold.isin(mapping) & scores.video_id.isin(allowed)].copy()
    selected["fold"] = selected.fold.map(mapping)
    return subset, selected


def check_score_coverage(scores, bench, fold_ids):
    allowed = {v for v, f in bench.folds.assignment.items() if f in fold_ids}
    bad, tolerated = [], []
    for (fold, seed, vid), rows in scores.groupby(["fold", "seed", "video_id"], sort=False):
        if fold not in fold_ids or vid not in allowed:
            raise ValueError("scores include a video or model from an excluded fold")
        expected = int(np.floor(bench.duration(vid)))
        ticks = np.sort(rows.t_sec.to_numpy())
        ticks = ticks[ticks <= expected]
        complete_prefix = np.array_equal(ticks, np.arange(1, len(ticks) + 1))
        if not complete_prefix or len(ticks) < expected - 1 or not len(ticks):
            bad.append({"video_id": vid, "fold": int(fold), "seed": int(seed),
                        "expected_ticks": expected, "scored_ticks": len(ticks)})
        elif len(ticks) == expected - 1:
            tolerated.append({"video_id": vid, "fold": int(fold), "seed": int(seed)})
    if bad:
        raise ValueError(f"incomplete joint feature coverage: {bad[:5]}; only one missing final tick is allowed")
    return {"tolerated_missing_final_tick": tolerated, "video_models_checked": int(scores.groupby(["fold", "seed", "video_id"]).ngroups)}


def restore_fold_ids(result, fold_ids):
    mapping = {str(i): str(f) for i, f in enumerate(fold_ids)}
    fields = {"theta_per_fold", "val_fa_per_hour_per_fold", "per_fold_far_window_auroc_clean",
              "val_videos_per_fold", "val_clean_negative_hours_per_fold"}
    visited = set()
    def visit(value):
        if not isinstance(value, (dict, list)) or id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, dict):
            for key, child in list(value.items()):
                if key in fields and isinstance(child, dict):
                    value[key] = {mapping.get(str(k), str(k)): v for k, v in child.items()}
                else:
                    visit(child)
        else:
            for child in value:
                visit(child)
    visit(result)
    return result


def evaluate_selected(scores, bench, fold_ids, out_dir, name=None, n_boot=1000,
                      threshold_bootstrap=0, threads=2, meta=None):
    coverage = check_score_coverage(scores, bench, fold_ids)
    subset, selected = evaluation_subset(bench, scores, fold_ids)
    va30.N_THREADS = threads
    cfg = va30.EvalConfig(restrict_to_scored=True, allow_missing=True, n_boot=n_boot, cluster="group",
                         threshold_bootstrap=threshold_bootstrap)
    result = va30.evaluate(selected, subset, cfg, name=name,
                           extra_meta={"model_meta": meta or {}, "trainer_tick_coverage": coverage})
    restore_fold_ids(result, fold_ids)
    va30.save_results(result, str(out_dir))
    return result


def main(argv=None):
    import argparse
    import json
    import pandas as pd
    from threadpoolctl import threadpool_limits
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--folds", required=True)
    p.add_argument("--videos", default="benchmark/videos.csv")
    p.add_argument("--events", default="benchmark/events.json")
    p.add_argument("--horizons", default="benchmark/extract_manifest_v1.csv")
    p.add_argument("--fold_ids", default="0,2,3,4")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--threshold_bootstrap", type=int, default=0)
    p.add_argument("--threads", type=int, choices=range(1, 5), default=2)
    p.add_argument("--meta")
    args = p.parse_args(argv)
    folds = [int(x) for x in args.fold_ids.split(",")]
    if 1 in folds:
        raise ValueError("fold 1 is frozen")
    bench = va30.Benchmark.load(args.folds, args.videos, args.events, horizons=args.horizons)
    with threadpool_limits(limits=args.threads):
        evaluate_selected(pd.read_parquet(args.scores), bench, folds, args.out_dir,
                          n_boot=args.n_boot, threshold_bootstrap=args.threshold_bootstrap,
                          threads=args.threads, meta=json.load(open(args.meta)) if args.meta else None)


if __name__ == "__main__":
    main()
