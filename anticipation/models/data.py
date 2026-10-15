"""Timestamp joins, feature snapshots and training-only normalization."""

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from anticipation.data.catalog import file_sha256
from anticipation.data.labels import labels_for_video
from anticipation.features import io
from anticipation.models.objectives import TRAIN_HAZARD_EDGES, band_targets


@dataclass
class VideoFeatures:
    video_id: str
    t_sec: np.ndarray
    streams: list
    present: np.ndarray
    labels: object
    observed: np.ndarray = None


def join_streams(video_id, row, events, streams, loader=None):
    loader = loader or io.load_features
    loaded = []
    for stream in streams:
        t, x, valid, meta = loader(stream, video_id)
        t, x, valid = np.asarray(t), np.asarray(x), np.asarray(valid, bool)
        if t.ndim != 1 or x.ndim != 2 or valid.shape != t.shape or len(x) != len(t):
            raise ValueError(f"{video_id}/{stream}: invalid feature shapes")
        if not np.isfinite(t).all() or np.any(np.diff(t) <= 0) or np.any(abs(t - np.round(t)) > 1e-4):
            raise ValueError(f"{video_id}/{stream}: timestamps must be sorted unique integer seconds")
        if x.shape[1] == 0:
            raise ValueError(f"{video_id}/{stream}: empty feature dimension")
        loaded.append((t, x, valid, meta))
    last = min((float(t[-1]) if len(t) else 0) for t, _, _, _ in loaded)
    ticks = io.tick_times(min(float(row["duration_sec"]), last))
    labels = labels_for_video(video_id, row, events, t_sec=ticks, hazard_edges=TRAIN_HAZARD_EDGES)
    arrays, masks, metadata = [], [], []
    observed = np.ones(len(ticks), bool)
    for t, x, valid, meta in loaded:
        out = np.zeros((len(ticks), x.shape[1]), np.float32)
        present = np.zeros(len(ticks), bool)
        ix = np.searchsorted(t, ticks)
        found = ix < len(t)
        found[found] &= abs(t[ix[found]] - ticks[found]) < 1e-4
        observed &= found
        dest, src = np.flatnonzero(found), ix[found]
        good = valid[src] & np.isfinite(x[src]).all(axis=1)
        dest, src = dest[good], src[good]
        out[dest] = x[src]
        present[dest] = True
        arrays.append(out)
        masks.append(present)
        metadata.append(meta)
    return VideoFeatures(video_id, ticks, arrays, np.stack(masks, axis=1), labels, observed), metadata


class FeatureStore:
    def __init__(self, root, catalog, events, streams):
        self.root = Path(root)
        self.catalog, self.events, self.streams = catalog, events, streams
        self.paths, self.records, self.dims = {}, {}, None
        self._cache = OrderedDict()

    def snapshot(self, video_ids, loader=None):
        missing, invalid = {}, {}
        for vid in sorted(video_ids):
            absent = [s for s in self.streams if not Path(io.feature_path(s, vid)).is_file()] if loader is None else []
            if absent:
                missing[vid] = absent
                continue
            try:
                v, meta = join_streams(vid, self.catalog.loc[vid], self.events, self.streams, loader)
                dims = [a.shape[1] for a in v.streams]
                if self.dims is not None and dims != self.dims:
                    raise ValueError(f"stream dimensions changed: {dims} != {self.dims}")
                if not v.observed.any():
                    raise ValueError("no timestamps shared by all required streams")
            except (OSError, ValueError, KeyError, EOFError) as exc:
                invalid[vid] = str(exc)
                continue
            self.dims = dims
            directory = self.root / vid
            directory.mkdir(parents=True, exist_ok=True)
            for j, a in enumerate(v.streams):
                np.save(directory / f"stream{j}.npy", a)
            np.save(directory / "present.npy", v.present)
            np.save(directory / "observed.npy", v.observed)
            self.paths[vid] = directory
            self.records[vid] = {
                "ticks": int(np.floor(self.catalog.loc[vid, "duration_sec"])),
                "snapshot_ticks": len(v.t_sec), "scored_ticks": int(v.observed.sum()),
                "valid_ticks": v.present.sum(axis=0).tolist(), "metadata": meta,
                "feature_sha256": [file_sha256(io.feature_path(s, vid)) for s in self.streams] if loader is None else [],
            }
        return {"requested": len(video_ids), "available": len(self.paths), "missing": missing,
                "invalid": invalid, "streams": self.streams, "dimensions": self.dims,
                "videos": self.records}

    def __getitem__(self, vid):
        if vid in self._cache:
            self._cache.move_to_end(vid)
            return self._cache[vid]
        d = self.paths[vid]
        present = np.load(d / "present.npy", mmap_mode="r")
        ticks = np.arange(1, len(present) + 1, dtype=np.float32)
        lab = labels_for_video(vid, self.catalog.loc[vid], self.events, t_sec=ticks, hazard_edges=TRAIN_HAZARD_EDGES)
        v = VideoFeatures(vid, lab.t_sec,
                          [np.load(d / f"stream{j}.npy", mmap_mode="r") for j in range(len(self.streams))],
                          present, lab,
                          np.load(d / "observed.npy", mmap_mode="r"))
        self._cache[vid] = v
        if len(self._cache) > 8:
            self._cache.popitem(last=False)
        return v


class Normalizer:
    def __init__(self, means, scales, counts):
        self.means, self.scales, self.counts = means, scales, counts

    @classmethod
    def fit(cls, store, ids):
        sums = [np.zeros(d, np.float64) for d in store.dims]
        squares = [np.zeros(d, np.float64) for d in store.dims]
        counts = np.zeros(len(store.dims), np.int64)
        for vid in ids:
            v = store[vid]
            for j, a in enumerate(v.streams):
                for start in range(0, len(a), 4096):
                    x = a[start:start + 4096][v.present[start:start + 4096, j]].astype(np.float64)
                    sums[j] += x.sum(axis=0)
                    squares[j] += np.square(x).sum(axis=0)
                    counts[j] += len(x)
        means = [s / max(1, n) for s, n in zip(sums, counts)]
        scales = [np.sqrt(np.maximum(q / max(1, n) - m * m, 1e-8))
                  for q, n, m in zip(squares, counts, means)]
        return cls([x.astype(np.float32) for x in means], [x.astype(np.float32) for x in scales], counts)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            counts = z["counts"]
            return cls([z[f"mean{j}"] for j in range(len(counts))],
                       [z[f"scale{j}"] for j in range(len(counts))], counts)

    def transform(self, x, present, stream):
        return np.where(present[:, None], (x - self.means[stream]) / self.scales[stream], 0).astype(np.float32)

    def save(self, path):
        values = {f"mean{j}": x for j, x in enumerate(self.means)}
        values.update({f"scale{j}": x for j, x in enumerate(self.scales)})
        np.savez(path, counts=self.counts, **values)


def positive_weights(labels, floor=0.25):
    return np.where(np.isfinite(labels.tte), np.clip(30 / np.maximum(labels.tte, 30), floor, 1), 1).astype(np.float32)


def sample_ticks(store, ids, limit, rng, task, time_matched=False, objective="horizon30"):
    candidates = []
    for vid in ids:
        v = store[vid]
        usable = (v.labels.valid[30] & ((v.labels.y[30] > 0) | (v.labels.tte > 120))) if task == "stats" else (
            (v.labels.hazard_target >= 0).any(axis=1) | ~v.labels.excluded)
        if objective == "band" and task == "stats":
            _, usable = band_targets(v, store.catalog.loc[vid, "role"])
        if v.observed is not None:
            usable &= v.observed
        candidates.append(np.flatnonzero(usable))
    lengths = np.array([len(x) for x in candidates])
    offsets = np.r_[0, np.cumsum(lengths)]
    total = int(offsets[-1])
    eligible = np.arange(total)
    if time_matched and total:
        from anticipation.eval.va30 import time_strata
        times, sources, positive, auxiliary = [], [], [], []
        for vid, ix in zip(ids, candidates):
            v = store[vid]
            times.extend(v.t_sec[ix])
            sources.extend([str(store.catalog.loc[vid, "source_kind"])] * len(ix))
            target = band_targets(v, store.catalog.loc[vid, "role"])[0] if objective == "band" else v.labels.y[30]
            positive.extend((target[ix] > 0) if task == "stats" or objective == "band" else (v.labels.hazard_target[ix] == 1).any(axis=1))
            if objective == "band" and task != "stats":
                auxiliary.extend(~band_targets(v, store.catalog.loc[vid, "role"])[1][ix])
            else:
                auxiliary.extend(v.labels.masked[ix] if task != "stats" else np.zeros(len(ix), bool))
        strata = time_strata(np.asarray(times), np.asarray(sources))
        positive, auxiliary = np.asarray(positive), np.asarray(auxiliary)
        matched = [np.flatnonzero(auxiliary)]
        for st in np.unique(strata[positive]):
            pos = np.flatnonzero((strata == st) & positive)
            neg = np.flatnonzero((strata == st) & ~positive & ~auxiliary)
            n = min(len(pos), len(neg))
            if n:
                matched.extend([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])
        eligible = np.concatenate(matched)
    chosen = np.sort(rng.choice(eligible, min(len(eligible), limit), replace=False)) if limit else np.sort(eligible)
    return {vid: ticks[chosen[(chosen >= a) & (chosen < b)] - a]
            for vid, ticks, a, b in zip(ids, candidates, offsets[:-1], offsets[1:])}
