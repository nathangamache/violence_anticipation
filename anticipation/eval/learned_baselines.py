"""Train B0 and B-scene0 on frozen VA30 folds, then export causal tick scores."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from anticipation.data.catalog import EVENTS_JSON, VIDEOS_CSV, file_sha256
from anticipation.data.folds import FOLDS_DEFAULT
from anticipation.data.labels import labels_for_video
from anticipation.eval.baselines import read_teacher, sample_and_hold
from anticipation.eval.va30 import Benchmark, git_commit

SOURCES = {"xd_violence": "xd", "ucf_crime": "ucf"}
TAG_DIMS = {"actions": 64, "weapons": 32, "scene": 64}


class StudentMLP(nn.Module):
    def __init__(self, in_dim=3744, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 256), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(256, 256), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def tag_vector(rec):
    parts = []
    for key, dim in TAG_DIMS.items():
        tokens = rec.get(key)
        tokens = tokens[:8] if isinstance(tokens, list) else ([tokens] if isinstance(tokens, str) else [])
        vec = np.zeros(dim, np.float32)
        for token in tokens:
            token = str(token).strip().lower()
            if token:
                vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % dim] += 1
        vec /= max(float(vec.sum()), 1.0)
        parts.append(vec)
    return np.concatenate(parts)


def read_tags(path):
    records = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
                if "error" not in rec and rec.get("t") is not None and rec.get("raw_frame_idx") is not None:
                    records.setdefault(int(rec["t"]), rec)
            except (ValueError, TypeError):
                continue
    return records


def catalog_mapping(videos):
    mapping = {}
    for vid, row in videos.iterrows():
        source = row["source"]
        stems = {vid.split("/", 1)[1], Path(row["path"]).stem}
        stems.update(Path(p).stem for p in str(row.get("aliases", "")).split(";") if p)
        for stem in stems:
            key = (source, stem)
            if key in mapping and mapping[key] != vid:
                raise ValueError(f"Ambiguous catalogue alias: {key}")
            mapping[key] = vid
    return mapping


def read_features(path, tag_path, fps, duration):
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid catalogue fps for {path}: {fps}")
    with np.load(path) as z:
        emb = np.asarray(z["emb"], np.float32)
        ts = np.asarray(z["raw_frame_idx"], float) / fps
        steps = np.asarray(z["t"])
    if emb.shape != (len(ts), 3584) or steps.shape != ts.shape:
        raise ValueError(f"Malformed embedding: {path}")
    if not np.isfinite(emb).all() or not np.isfinite(ts).all():
        raise ValueError(f"Non-finite embedding: {path}")
    order = np.argsort(ts, kind="stable")
    order = order[(ts[order] >= 0) & (ts[order] <= duration)]
    order = order[np.r_[True, np.diff(ts[order]) > 0]] if len(order) else order
    emb, ts, steps = emb[order], ts[order], steps[order]
    records = read_tags(tag_path)
    tags = np.zeros((len(ts), 160), np.float32)
    matched = 0
    for i, (t, step) in enumerate(zip(ts, steps)):
        rec = records.get(int(step))
        # A mismatched JSONL row must never supply future information to an embedding.
        if rec is not None and abs(float(rec["raw_frame_idx"]) / fps - t) <= 1e-6:
            tags[i] = tag_vector(rec)
            matched += 1
    scene = emb[ts <= 2].mean(axis=0) if (ts <= 2).any() else np.zeros(3584, np.float32)
    return {"times": ts, "x": np.concatenate([emb, tags], axis=1), "scene": scene,
            "scene_rows": int((ts <= 2).sum()), "tag_matches": matched}


def load_features(bench, emb_root, tag_root):
    mapping = catalog_mapping(bench.videos)
    eligible = set(bench.eval_ids())
    features, skipped = {}, {}
    files = sorted(Path(emb_root).glob("*/*.npz"))
    files.sort(key=lambda p: (p.stem != str(mapping.get((SOURCES.get(p.parent.name), p.stem), "/")).split("/", 1)[1], str(p)))
    for path in files:
        source = SOURCES.get(path.parent.name)
        vid = mapping.get((source, path.stem))
        if vid not in eligible:
            reason = "unmapped" if vid is None else str(bench.videos.loc[vid, "role"])
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        if vid in features:
            skipped["duplicate_alias_feature"] = skipped.get("duplicate_alias_feature", 0) + 1
            continue
        row = bench.videos.loc[vid]
        feat = read_features(path, Path(tag_root) / path.parent.name / (path.stem + ".jsonl"),
                             float(row.fps), float(row.duration_sec))
        if not len(feat["times"]):
            skipped["empty"] = skipped.get("empty", 0) + 1
            continue
        lab = labels_for_video(vid, row, bench.events, t_sec=feat["times"])
        feat.update(y=lab.y[30].astype(np.float32), valid=lab.valid[30], path=str(path))
        features[vid] = feat
    if not features:
        raise ValueError("No usable features")
    ids = sorted(features)
    rows = bench.videos.loc[ids]
    coverage = {"embedding_files": len(files), "skipped_by_reason": skipped, "videos": len(ids),
                "video_ids": ids, "by_source": rows.source.value_counts().to_dict(),
                "by_role": rows.role.value_counts().to_dict(),
                "eligible30_events": sum(sum(e["eligible30"] for e in bench.event_rows(v)) for v in ids),
                "clean_negative_hours": sum((~bench.labels(v).masked).sum() / 3600 for v in ids
                                            if bench.videos.loc[v, "role"] == "clean_negative"),
                "feature_rows": sum(len(f["times"]) for f in features.values()),
                "valid_training_rows": sum(int(f["valid"].sum()) for f in features.values()),
                "tag_matches": sum(f["tag_matches"] for f in features.values()),
                "no_frame_in_first_2s": sum(f["scene_rows"] == 0 for f in features.values()),
                "ticks_before_first_embedding": sum(int((bench.labels(v).t_sec < f["times"][0]).sum())
                                                    for v, f in features.items())}
    return features, coverage


def training_arrays(features, ids):
    parts = [features[v] for v in ids if v in features]
    x = np.concatenate([f["x"][f["valid"]] for f in parts])
    y = np.concatenate([f["y"][f["valid"]] for f in parts])
    if len(np.unique(y)) != 2:
        raise ValueError("Training/validation slice needs both target classes")
    return x, y


def predict(model, x, device, batch_size):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(torch.as_tensor(x[i:i + batch_size], device=device)).sigmoid().cpu()
                          for i in range(0, len(x), batch_size)]).numpy()


def fit_student(train, val, args, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = StudentMLP(train[0].shape[1], args.dropout).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x = torch.as_tensor(train[0], device=args.device)
    y = torch.as_tensor(train[1], device=args.device)
    loss_fn = nn.BCEWithLogitsLoss()
    history, best, best_auc, best_epoch = [], None, -np.inf, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for indices in np.array_split(rng.permutation(len(y)), max(1, int(np.ceil(len(y) / args.batch_size)))):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x[indices]), y[indices])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            losses.append(float(loss.detach()))
        auc = float(roc_auc_score(val[1], predict(model, val[0], args.device, args.batch_size)))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_auroc30": auc})
        if auc > best_auc + 1e-5:
            best_auc, best_epoch = auc, epoch
            best = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best)
    return model, {"best_epoch": best_epoch, "val_auroc30": best_auc, "history": history,
                   "train_rows": len(y), "val_rows": len(val[1]), "train_positive_rate": float(train[1].mean())}


def fit_scene(features, ids, bench, seed):
    vids = [v for v in ids if v in features]
    x = np.stack([features[v]["scene"] for v in vids])
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)
    counts = []
    for v in vids:
        lab = bench.labels(v)
        valid = lab.valid[30] & (lab.t_sec >= 2)
        counts.append([int((valid & (lab.y[30] == k)).sum()) for k in (0, 1)])
    counts = np.asarray(counts)
    xx = np.repeat(x, 2, axis=0)
    yy = np.tile([0, 1], len(vids))
    weights = counts.ravel()
    keep = weights > 0
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=seed)
    model.fit(xx[keep], yy[keep], sample_weight=weights[keep])
    return scaler, model, {"training_videos": len(vids), "weighted_training_ticks": int(weights.sum()),
                           "iterations": model.n_iter_.tolist()}


def score_fold(features, ids, bench, fold, seed, scorer):
    parts = []
    for vid in sorted(set(ids) & features.keys()):
        f = features[vid]
        ticks = bench.labels(vid).t_sec
        scores = scorer(f, ticks)
        parts.append(pd.DataFrame({"video_id": vid, "t_sec": ticks, "score": scores,
                                   "fold": fold, "seed": seed}))
    return pd.concat(parts, ignore_index=True)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", default=FOLDS_DEFAULT)
    ap.add_argument("--videos", default=VIDEOS_CSV)
    ap.add_argument("--events", default=EVENTS_JSON)
    ap.add_argument("--emb_root", default="data/processed/vlm_emb_meanpool")
    ap.add_argument("--tag_root", default="data/processed/vlm_labels_meanpool")
    ap.add_argument("--teacher_root", default="data/processed/vlm_labels")
    ap.add_argument("--out_root", default="benchmark/results")
    ap.add_argument("--device", choices=["cpu", "cuda:1"], default="cpu")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    args = ap.parse_args(argv)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if args.device != "cpu":
        subprocess.run([sys.executable, "tools/server_load.py", "--gpu"], check=True)
    bench = Benchmark.load(args.folds, args.videos, args.events)
    features, coverage = load_features(bench, args.emb_root, args.tag_root)
    print(json.dumps({k: v for k, v in coverage.items() if k != "video_ids"}), flush=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    meta = {"git_commit": git_commit(), "command": sys.argv, "config": vars(args), "seeds": seeds,
            "folds_content_sha256": bench.folds.sha256, "coverage": coverage,
            "target": "P(onset within 30 seconds); valid_H30 at embedding timestamps for B0, tick weighted for scene",
            "loss": "unweighted binary cross entropy", "checkpoint_selection": "maximum validation AUROC30; patience 5",
            "architecture": "LayerNorm(3744), Linear256, GELU, Dropout0.2, Linear256, GELU, Dropout0.2, Linear1",
            "tags": "MD5 actions64/weapons32/scene64, count normalized per field; exact t AND raw frame match; first duplicate",
            "scene": "mean embedding at raw_frame_idx/catalogue_fps <=2s; zero vector if absent; no tags; train-only StandardScaler, logistic C=1",
            "causality": "B0 sample-and-hold; zero before first feature. Scene available at tick2; tick1 score zero.",
            "feature_manifest_sha256": hashlib.sha256("\n".join(
                f"{v} {file_sha256(f['path'])}" for v, f in sorted(features.items())).encode()).hexdigest()}
    outputs = {name: Path(args.out_root) / name for name in ("b0_student_mlp", "b_scene0")}
    for out in outputs.values():
        out.mkdir(parents=True, exist_ok=True)
        config_path = out / "run_config.json"
        if config_path.exists():
            previous = json.loads(config_path.read_text())
            for key in ("config", "folds_content_sha256", "feature_manifest_sha256"):
                if previous[key] != meta[key]:
                    raise ValueError(f"Refusing stale resume: {config_path} differs at {key}")
        write_json(config_path, meta)
    for name, out in outputs.items():
        runs, parts = [], []
        for seed in seeds:
            for fold in range(bench.folds.n_folds):
                cache = out / f"scores_seed{seed}_fold{fold}.parquet"
                info_path = out / f"training_seed{seed}_fold{fold}.json"
                if cache.exists() and info_path.exists():
                    parts.append(pd.read_parquet(cache))
                    runs.append(json.loads(info_path.read_text()))
                    continue
                train_ids, val_ids = bench.folds.train_ids(fold), bench.folds.val_ids(fold)
                ids = bench.folds.test_ids(fold) + val_ids
                if name == "b0_student_mlp":
                    model, info = fit_student(training_arrays(features, train_ids), training_arrays(features, val_ids), args, seed)
                    def scorer(f, ticks):
                        return sample_and_hold(f["times"], predict(model, f["x"], args.device, args.batch_size), ticks)
                else:
                    scaler, model, info = fit_scene(features, train_ids, bench, seed)
                    def scorer(f, ticks):
                        p = model.predict_proba(scaler.transform(f["scene"][None]))[0, 1]
                        return np.where(ticks >= 2, p, 0.0)
                scores = score_fold(features, ids, bench, fold, seed, scorer)
                info.update(seed=seed, fold=fold, train_video_ids=sorted(set(train_ids) & features.keys()),
                            val_video_ids=sorted(set(val_ids) & features.keys()))
                scores.to_parquet(cache, index=False, compression="zstd")
                write_json(info_path, info)
                parts.append(scores)
                runs.append(info)
                print(f"{name} seed={seed} fold={fold}: { {k:v for k,v in info.items() if k not in ('history','train_video_ids','val_video_ids')} }", flush=True)
        pd.concat(parts, ignore_index=True).to_parquet(out / "scores.parquet", index=False, compression="zstd")
        write_json(out / "training.json", runs)
    teacher_parts = []
    for vid in sorted(features):
        src, stem = vid.split("/", 1)
        legacy = next(k for k, v in SOURCES.items() if v == src)
        path = Path(args.teacher_root) / legacy / (stem + ".jsonl")
        if not path.exists():
            raise ValueError(f"Teacher missing on common subset: {vid}")
        ts, vals = read_teacher(path, float(bench.videos.loc[vid, "fps"]))
        if not len(ts):
            raise ValueError(f"Teacher has no valid records: {vid}")
        ticks = bench.labels(vid).t_sec
        teacher_parts.append(pd.DataFrame({"video_id": vid, "t_sec": ticks,
                                           "score": sample_and_hold(ts, vals, ticks), "fold": -1, "seed": 0}))
    out = Path(args.out_root) / "b0_student_mlp_teacher_subset"
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(teacher_parts, ignore_index=True).to_parquet(out / "scores.parquet", index=False, compression="zstd")
    write_json(out / "run_config.json", {**meta, "baseline": "b_teacher restricted to exact B0/B-scene0 video subset",
                                        "teacher_clock": "raw_frame_idx / benchmark/videos.csv fps"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
