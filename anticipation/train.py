"""Train causal cached-feature models on the frozen VA30 folds."""

import os

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "4"

import argparse
import datetime
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from anticipation.data.folds import FOLDS_DEFAULT
from anticipation.eval import va30
from anticipation.features import io
from anticipation.models.data import FeatureStore, Normalizer, sample_ticks
from anticipation.models.stats import StatsModel, make_table
from anticipation.models.selection import MatchedValidation
from anticipation.models.objectives import TRAIN_HAZARD_EDGES, band_targets, matched_weights
from anticipation.models.controls import SceneProvider, NuisanceModel
from anticipation.models.evaluation import SCREEN_FOLDS, evaluate_selected


def write_json(path, data):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--streams", required=True, help="comma-separated required cached streams")
    p.add_argument("--model", required=True, choices=["stats_lgbm", "stats_logreg", "gru", "transformer", "time_only", "first_frame", "nuisance", "residual_logreg"])
    p.add_argument("--run_name", required=True)
    p.add_argument("--folds", default=FOLDS_DEFAULT)
    p.add_argument("--phase", choices=["diagnostic", "screen", "synthetic"], default="diagnostic")
    p.add_argument("--fold_ids", default=None, help="default 0,2,3,4; real fold 1 is frozen")
    p.add_argument("--registration", default="benchmark/screens/screen_set_v1.json")
    p.add_argument("--horizons", default="benchmark/extract_manifest_v1.csv")
    p.add_argument("--objective", choices=["band", "horizon30"], default="band")
    p.add_argument("--score_head", choices=["auto", "band", "hazard_band", "hazard30"], default="auto")
    p.add_argument("--far_positive_downweight", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--time_match_weights", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--band_weight", type=float, default=1.)
    p.add_argument("--event_weight", type=float, default=1.)
    p.add_argument("--event_batch_size", type=int, default=2)
    p.add_argument("--event_ticks", type=int, default=16)
    p.add_argument("--history_dropout", type=float, default=0.25)
    p.add_argument("--scene_stream", default="siglip2")
    p.add_argument("--scene_tick", type=int, default=2)
    p.add_argument("--first_frame_tick", type=int, default=1)
    p.add_argument("--videos", default=va30.VIDEOS_CSV)
    p.add_argument("--events", default=va30.EVENTS_JSON)
    p.add_argument("--feature_root", default=io.FEATURE_ROOT)
    p.add_argument("--cohort", default=None, help="JSON array of benchmark video IDs; freeze coverage for model/control comparisons")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--device", choices=["cpu", "cuda:1"], default="cpu")
    p.add_argument("--threads", type=int, choices=range(1, 5), default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--max_train_ticks", type=int, default=20000, help="0 = all usable training ticks")
    p.add_argument("--max_val_ticks", type=int, default=10000, help="checkpoint loss only; thresholds use every val tick")
    p.add_argument("--context", type=int, choices=range(60, 301), default=180, metavar="60..300")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--modality_dropout", type=float, default=0.15)
    p.add_argument("--auxiliary_weight", type=float, default=0.2)
    p.add_argument("--position_control", action="store_true", help="explicit elapsed-time control; off by default")
    p.add_argument("--time_matched_training", action="store_true", help="match training negatives to positives by source and tick-time bin")
    p.add_argument("--require_validation_metric", action="store_true", help="stop before fitting if any fold lacks matched validation positives/negatives")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--threshold_bootstrap", type=int, default=0)
    p.add_argument("--max_minutes", type=float, default=29, help="training/scoring wall budget, at most 30 minutes")
    p.add_argument("--horizons", default=str(Path(va30.REPO_ROOT) / "benchmark" / "extract_manifest_v1.csv"), help="per-video evaluation horizons passed to VA30")
    p.add_argument("--no_eval", action="store_true", help="save scores without calling VA30")
    return p


def split_coverage(bench, store, fold_ids=None):
    coverage = {}
    for f in (range(bench.folds.n_folds) if fold_ids is None else fold_ids):
        coverage[str(f)] = {}
        for split, vids in (("train", bench.folds.train_ids(f)), ("val", bench.folds.val_ids(f)),
                            ("test", bench.folds.test_ids(f))):
            ids = [v for v in vids if v in store.paths]
            rows = bench.videos.loc[ids]
            clean = [v for v in ids if bench.videos.loc[v, "role"] == "clean_negative"]
            coverage[str(f)][split] = {"videos": len(ids), "roles": rows.role.value_counts().to_dict(),
                                       "clean_hours": sum(store.records[v]["scored_ticks"] for v in clean) / 3600,
                                       "catalog_clean_hours": float(rows.loc[rows.role == "clean_negative", "duration_sec"].sum() / 3600),
                                       "video_ids": ids}
    return coverage


def train_run(args, bench, store, out, config, scene_provider=None):
    deadline = time.monotonic() + args.max_minutes * 60
    def check_time():
        if time.monotonic() >= deadline:
            raise TimeoutError("training/scoring wall budget reached; completed model scores are saved")
    files, histories = [], []
    for fold in config["fold_ids"]:
        ids = {s: config["split_coverage"][str(fold)][s]["video_ids"] for s in ("train", "val", "test")}
        if not ids["train"] or not ids["val"]:
            raise ValueError(f"fold {fold}: missing training or validation feature coverage")
        check_time()
        norm = Normalizer.fit(store, ids["train"])
        norm.save(out / "checkpoints" / f"normalizer_fold{fold}.npz")
        validation = MatchedValidation(store, ids["val"], args.max_val_ticks, 120 if args.objective == "band" else 60)
        for seed in config["seeds"]:
            check_time()
            random.seed(seed)
            np.random.seed(seed)
            rng = np.random.default_rng(seed)
            task = "temporal" if args.model in ("gru", "transformer") else "stats"
            selection = {s: sample_ticks(store, ids[s], getattr(args, f"max_{s}_ticks"), rng, task,
                                        time_matched=args.time_matched_training and s == "train", objective=args.objective)
                         for s in ("train", "val")}
            name = f"fold{fold}_seed{seed}"
            if task == "stats":
                nuisance = None
                if args.model in ("nuisance", "residual_logreg"):
                    vids, times, target = [], [], []
                    for vid, ix in selection["train"].items():
                        v = store[vid]
                        vids.extend([vid] * len(ix))
                        times.extend(v.t_sec[ix])
                        target.extend(band_targets(v, store.catalog.loc[vid, "role"])[0][ix] if args.objective == "band" else v.labels.y[30][ix])
                    weight = matched_weights(times, [store.catalog.loc[v, "source_kind"] for v in vids], target) if args.time_match_weights else np.ones(len(vids))
                    nuisance = NuisanceModel(scene_provider, {v: store.catalog.loc[v, "source_kind"] for v in store.paths}, seed)
                    nuisance.fit(ids["train"], vids, times, np.asarray(target), weight)
                if args.model == "nuisance":
                    model = nuisance
                    metric = validation.score(lambda vid, ix: model.predict(store[vid], indices=ix))
                    model.history = [{"epoch": 1, "val_matched_far_auroc": metric}]
                else:
                    model = StatsModel(args.model, seed, args.threads, args.position_control, args.first_frame_tick, nuisance)
                    tables = [make_table(store, ids[split], selection[split], norm, args.position_control,
                                         args.model == "time_only", args.objective, args.time_match_weights,
                                         args.model == "first_frame", args.first_frame_tick, nuisance)
                              for split in ("train", "val")]
                    model.fit(*tables, args.epochs, args.patience, check_time=check_time,
                              selection_score=lambda m: validation.score(lambda vid, ix: m.predict(store[vid], norm, indices=ix)))
                with open(out / "checkpoints" / (name + ".pkl"), "wb") as f:
                    pickle.dump(model, f)
                history, backend = model.history, model.backend
                predict = lambda v: model.predict(v, norm, indices=np.flatnonzero(v.observed))
            else:
                import torch
                from anticipation.models.temporal import CausalTemporalModel, WindowDataset, EventBagDataset, fit_temporal, predict_temporal
                torch.set_num_threads(args.threads)
                torch.manual_seed(seed)
                torch.backends.cudnn.benchmark = False
                torch.use_deterministic_algorithms(True)
                model = CausalTemporalModel(store.dims, args.model, args.hidden, args.layers, args.heads, args.context,
                                            args.modality_dropout, args.dropout, args.position_control, score_kind=config["score_head"])
                datasets = {s: WindowDataset(store, selection[s], norm, args.context, args.position_control,
                                             args.objective, args.far_positive_downweight,
                                             args.history_dropout if s == "train" else 0,
                                             args.time_match_weights and args.objective == "band") for s in ("train", "val")}
                event_dataset = EventBagDataset(datasets["train"], ids["train"], args.event_ticks, seed) if args.objective == "band" and args.event_weight else None
                history = fit_temporal(model, datasets["train"], datasets["val"], args.device, args.epochs,
                                       args.patience, args.batch_size, args.lr, args.auxiliary_weight, seed,
                                       check_time=check_time,
                                       selection_score=lambda m: validation.score(
                                           lambda vid, ix: predict_temporal(m, store[vid], norm, args.device, args.batch_size, ix)),
                                       band_weight=args.band_weight if args.objective == "band" else 0,
                                       event_weight=args.event_weight, event_dataset=event_dataset,
                                       event_batch_size=args.event_batch_size)
                torch.save({"state_dict": model.state_dict(), "config": vars(args), "dimensions": store.dims,
                            "best_epoch": model.best_epoch, "hazard_edges": list(TRAIN_HAZARD_EDGES),
                            "score_kind": config["score_head"]}, out / "checkpoints" / (name + ".pt"))
                backend = args.model
                predict = lambda v: predict_temporal(model, v, norm, args.device, args.batch_size, np.flatnonzero(v.observed))
            rows = []
            for split in ("test", "val"):
                for vid in ids[split]:
                    check_time()
                    v = store[vid]
                    score = predict(v)
                    if not np.isfinite(score).all():
                        raise ValueError(f"non-finite scores for {vid}")
                    rows.append(pd.DataFrame({"video_id": vid, "t_sec": v.t_sec[v.observed], "score": score,
                                              "fold": fold, "seed": seed}))
            path = out / (name + "_scores.parquet")
            pd.concat(rows, ignore_index=True).to_parquet(path, index=False)
            files.append(path)
            histories.append({"fold": fold, "seed": seed, "backend": backend, "best_epoch": model.best_epoch,
                              "checkpoint_selection": validation.info,
                              "event_bags": event_dataset.info if task == "temporal" and event_dataset is not None else None,
                              "train_ticks": sum(map(len, selection["train"].values())),
                              "val_ticks": sum(map(len, selection["val"].values())), "epochs": history})
            write_json(out / "training_history.json", histories)
            print(f"[train] {name}: {backend}, best epoch {model.best_epoch}, scored {sum(len(r) for r in rows)} ticks", flush=True)
    score_path = out / "scores.parquet"
    # Merge bounded per-model files without holding the whole score table in memory.
    import pyarrow.parquet as pq
    with pq.ParquetWriter(score_path, pq.read_schema(files[0])) as writer:
        for file in files:
            writer.write_table(pq.read_table(file))
    for file in files:
        file.unlink()
    config["status"] = "scored"
    write_json(out / "run_config.json", config)
    return score_path


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.run_name or Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        raise ValueError("run_name must be a single directory name")
    if not 0 < args.max_minutes <= 30 or min(args.epochs, args.patience, args.batch_size) < 1:
        raise ValueError("epochs/patience/batch_size must be positive; max_minutes must be in (0,30]")
    if min(args.max_train_ticks, args.max_val_ticks, args.n_boot, args.threshold_bootstrap) < 0:
        raise ValueError("tick limits and bootstrap counts cannot be negative")
    streams = [x.strip() for x in args.streams.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",")]
    if not streams or len(set(streams)) != len(streams) or not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("streams and nonnegative seeds must be nonempty and unique")
    if any(Path(s).name != s or s in (".", "..") for s in streams):
        raise ValueError("stream names cannot contain path components")
    if args.device == "cuda:1":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if args.phase == "screen":
        registration = Path(args.registration)
        if not registration.is_file() or not registration.with_name("summary.md").is_file():
            raise ValueError("Phase A is blocked until the evaluation agent's screen_set_v1.json and summary.md land")
        json.loads(registration.read_text())
    if args.phase != "synthetic" and args.context < 180:
        raise ValueError("real-data context must be 180..300 seconds")
    if not 0 <= args.history_dropout <= 1 or min(args.band_weight, args.event_weight) < 0 or min(args.event_ticks, args.event_batch_size) < 1:
        raise ValueError("invalid history-dropout or band/event loss settings")
    if args.phase == "synthetic" and args.horizons == "benchmark/extract_manifest_v1.csv" and not Path(args.horizons).exists():
        args.horizons = None
    out = Path("benchmark/results") / args.run_name
    out.mkdir(parents=True, exist_ok=False)
    (out / "checkpoints").mkdir()
    bench = va30.Benchmark.load(args.folds, args.videos, args.events, horizons=args.horizons)
    fold_ids = [int(x) for x in args.fold_ids.split(",")] if args.fold_ids else (list(range(bench.folds.n_folds)) if args.phase == "synthetic" else list(SCREEN_FOLDS))
    if len(set(fold_ids)) != len(fold_ids) or not fold_ids or any(f not in range(bench.folds.n_folds) for f in fold_ids):
        raise ValueError("invalid or duplicate fold IDs")
    if 1 in fold_ids and (args.phase != "synthetic" or bench.folds.d.get("name") in ("folds_v1", "folds_v2")):
        raise ValueError("fold 1 is frozen; screening and diagnostics cannot read it")
    packages = {}
    for package in ("numpy", "scipy", "scikit-learn", "torch", "pandas", "pyarrow", "lightgbm"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    config = {"config": vars(args), "packages": packages,
              "model_spec": {"statistics_windows_sec": [5, 15, 30, 60], "logreg_C": 0.1,
                             "logreg_solver": "newton-cg", "boosting_rounds_per_epoch": 10,
                             "tree_leaves": 15, "tree_learning_rate": 0.05,
                             "hazard_edges_sec": list(TRAIN_HAZARD_EDGES),
                             "positive_hazard_weight": "clip(30 / time_to_onset, 0.25, 1)" if args.far_positive_downweight else "1",
                             "optimizer": "AdamW", "weight_decay": 1e-4, "gradient_clip": 1.0,
                             "normalizer_scale_floor": 1e-4, "short_context": "masked padding; optional random history truncation",
                             "checkpoint_selection": "validation source_time_matched_band_auroc" if args.objective == "band" else "validation source_time_matched_far_auroc",
                             "selection_unavailable": "one fixed diagnostic epoch; no tuning"}, "command": [sys.executable, "-m", "anticipation.train", *(argv if argv is not None else sys.argv[1:])],
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "git_dirty": subprocess.check_output(["git", "status", "--porcelain"], text=True).splitlines(),
              "created": datetime.datetime.now(datetime.timezone.utc).isoformat(), "seeds": seeds, "fold_ids": fold_ids,
              "score_head": ("band" if args.objective == "band" else "hazard30") if args.score_head == "auto" else args.score_head,
              "benchmark": bench.provenance, "status": "snapshotting",
              "normalization": "per-dimension training-fold moments over valid ticks only",
              "score": "onset in (30,120] band" if args.objective == "band" else "legacy horizon-30 score",
              "restrict_to_scored": True}
    if args.phase == "screen":
        config["registration"] = {"file": args.registration, "sha256": va30.file_sha256(args.registration),
                                  "summary_sha256": va30.file_sha256(Path(args.registration).with_name("summary.md"))}
    write_json(out / "run_config.json", config)
    io.FEATURE_ROOT = args.feature_root
    try:
        with threadpool_limits(limits=args.threads):
            va30.N_THREADS = args.threads
            catalog = bench.videos.copy()
            for vid in bench.video_horizons:
                catalog.loc[vid, "duration_sec"] = bench.duration(vid)
            store = FeatureStore(out / "cache", catalog, bench.events, streams)
            video_ids = [v for v in bench.eval_ids() if bench.folds.fold_of(v) in fold_ids]
            if args.cohort:
                cohort = json.loads(Path(args.cohort).read_text())
                if not isinstance(cohort, list) or not all(isinstance(v, str) for v in cohort) or len(set(cohort)) != len(cohort):
                    raise ValueError("cohort must be a JSON array of unique video IDs")
                outside = set(cohort) - set(video_ids)
                if outside:
                    raise ValueError(f"cohort has IDs outside the benchmark universe: {sorted(outside)[:3]}")
                video_ids = sorted(cohort)
                config["cohort_sha256"] = va30.file_sha256(args.cohort)
            scene_provider = None
            if args.model in ("nuisance", "residual_logreg"):
                scene_provider = SceneProvider(args.scene_stream, video_ids, args.scene_tick)
                video_ids = [v for v in video_ids if v in scene_provider.anchors]
                config["scene_control"] = {"stream": args.scene_stream, "tick": args.scene_tick,
                                           "people_masked": scene_provider.people_masked,
                                           "feature_sha256": scene_provider.hashes, "missing": scene_provider.missing}
                if args.phase == "screen" and not scene_provider.people_masked:
                    raise ValueError("G3 requires people-masked scene embeddings; unmasked scenes are diagnostic only")
            coverage = store.snapshot(video_ids)
            write_json(out / "coverage.json", coverage)
            config["split_coverage"] = split_coverage(bench, store, fold_ids)
            config["features"] = {"streams": streams, "dimensions": store.dims,
                                  "manifest": str(out / "coverage.json"),
                                  "manifest_sha256": va30.file_sha256(out / "coverage.json")}
            missing_clean = [int(f) for f, splits in config["split_coverage"].items()
                             if splits["val"]["clean_hours"] <= 0]
            config["evaluation_readiness"] = {"validation_folds_without_clean_negatives": missing_clean}
            config["validation_selection"] = {f: MatchedValidation(store, splits["val"]["video_ids"], args.max_val_ticks, 120 if args.objective == "band" else 60).info
                                               for f, splits in config["split_coverage"].items()}
            config["status"] = "training"
            write_json(out / "run_config.json", config)
            print(f"[train] coverage: {coverage['available']}/{coverage['requested']} videos; "
                  f"{len(coverage['missing'])} missing, {len(coverage['invalid'])} invalid", flush=True)
            if missing_clean:
                print(f"[train] validation folds {missing_clean} have no covered clean negatives; "
                      "VA30 threshold evaluation will be unavailable", flush=True)
            if not coverage["available"]:
                raise ValueError("no videos have the required feature streams; see coverage.json")
            if args.require_validation_metric and not all(m["available"] for m in config["validation_selection"].values()):
                raise ValueError("matched validation AUROC unavailable in one or more folds; no models were fitted")
            scores = train_run(args, bench, store, out, config, scene_provider)
            if not args.no_eval:
                config["evaluation_command"] = [sys.executable, "-m", "anticipation.models.evaluation", "--scores", str(scores),
                    "--out_dir", str(out), "--folds", args.folds, "--videos", args.videos, "--events", args.events,
                    "--n_boot", str(args.n_boot), "--threshold_bootstrap", str(args.threshold_bootstrap),
                    "--fold_ids", ",".join(map(str, fold_ids)), "--threads", str(args.threads), "--meta", str(out / "run_config.json")]
                write_json(out / "run_config.json", config)
                if args.horizons:
                    config["evaluation_command"].extend(["--horizons", args.horizons])
                config["evaluation_original_fold_ids"] = fold_ids
                write_json(out / "run_config.json", config)
                evaluate_selected(pd.read_parquet(scores), bench, fold_ids, out, args.run_name, args.n_boot,
                                  args.threshold_bootstrap, args.threads, config)

            config["status"] = "complete" if not args.no_eval else "scored_without_evaluation"
    except Exception as exc:
        config["status"] = "evaluation_unavailable" if config.get("status") == "scored" else "failed"
        config["error"] = f"{type(exc).__name__}: {exc}"
        write_json(out / "run_config.json", config)
        print(f"[train] {config['error']}; artifacts saved in {out}", file=sys.stderr, flush=True)
        return 2
    write_json(out / "run_config.json", config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
