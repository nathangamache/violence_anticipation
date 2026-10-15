"""Causal first-frame and source/time/scene nuisance controls with an offset residual classifier."""

import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from anticipation.data.catalog import file_sha256
from anticipation.eval.va30 import time_strata
from anticipation.features import io


def first_frame_features(video, normalizer, indices, anchor_tick=1):
    ix = np.asarray(indices, int)
    values = []
    anchor = anchor_tick - 1
    for j, raw in enumerate(video.streams):
        valid = anchor < len(raw) and bool(video.present[anchor, j])
        row = normalizer.transform(raw[anchor:anchor + 1], np.array([valid]), j)[0] if anchor < len(raw) else np.zeros(raw.shape[1])
        values.extend([*row, float(valid)])
    result = np.broadcast_to(np.asarray(values, np.float32), (len(ix), len(values))).copy()
    result[video.t_sec[ix] < anchor_tick] = 0
    return result


class SceneProvider:
    def __init__(self, stream, ids, anchor_tick=2, loader=None):
        loader = loader or io.load_features
        self.stream, self.anchor_tick = stream, anchor_tick
        self.anchors, self.present, self.hashes, self.missing = {}, {}, {}, []
        self.people_masked = True
        dimension = None
        for vid in ids:
            try:
                if stream == "siglip2_scene0p" and loader is io.load_features:
                    with np.load(io.feature_path(stream, vid), allow_pickle=False) as z:
                        meta = json.loads(str(z["meta"]))
                        t = np.array([float(meta["t_sec"])])
                        x = z["emb"].astype(np.float32)[None]
                        valid = np.ones(1, bool)
                else:
                    t, x, valid, meta = loader(stream, vid)
            except (OSError, ValueError, KeyError, EOFError):
                self.missing.append(vid)
                continue
            if x.ndim != 2 or (dimension is not None and dimension != x.shape[1]):
                raise ValueError("scene embedding dimensions differ across videos")
            dimension = x.shape[1]
            index = int(np.searchsorted(t, anchor_tick))
            good = index < len(t) and abs(t[index] - anchor_tick) < 1e-4 and valid[index] and np.isfinite(x[index]).all()
            self.anchors[vid] = x[index].astype(np.float32) if good else np.zeros(dimension, np.float32)
            self.present[vid] = bool(good)
            self.people_masked &= bool(meta.get("people_masked") or meta.get("person_boxes_greyed") or
                                       (meta.get("extractor") == "siglip2_scene0p" and meta.get("box_source") and meta.get("grey") is not None))
            self.hashes[vid] = file_sha256(io.feature_path(stream, vid)) if loader is io.load_features else None


class NuisanceModel:
    def __init__(self, provider, sources, seed=0):
        self.provider, self.sources, self.seed = provider, dict(sources), seed
        self.scene_scaler, self.scene_pca = StandardScaler(), None
        self.categories = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.model = LogisticRegression(C=0.1, solver="newton-cg", max_iter=100, random_state=seed)
        self.best_epoch, self.backend, self.history = 1, "nuisance_logistic", []

    def _categories(self, vids, times):
        bins = time_strata(np.asarray(times))
        return np.array([f"{self.sources[v]}:{b}" for v, b in zip(vids, bins)], dtype=object)[:, None]

    def fit_encoder(self, train_ids):
        ids = sorted(set(train_ids))
        scene = np.stack([self.provider.anchors[v] for v in ids])
        scene = self.scene_scaler.fit_transform(scene)
        components = min(32, scene.shape[1], len(ids) - 1)
        if components < 1:
            raise ValueError("nuisance PCA requires at least two training videos")
        self.scene_pca = PCA(components, svd_solver="full", random_state=self.seed).fit(scene)
        self.encoder_train_ids = ids
        return self

    def features(self, vids, times, fit=False):
        categories = self._categories(vids, times)
        categorical = self.categories.fit_transform(categories) if fit else self.categories.transform(categories)
        unique = list(dict.fromkeys(vids))
        scenes = self.scene_pca.transform(self.scene_scaler.transform(np.stack([self.provider.anchors[v] for v in unique])))
        lookup = dict(zip(unique, scenes))
        projected = np.stack([lookup[v] for v in vids])
        present = np.asarray([self.provider.present[v] and t >= self.provider.anchor_tick for v, t in zip(vids, times)])
        return np.concatenate([categorical, projected * present[:, None], present[:, None]], axis=1)

    def fit(self, train_ids, vids, times, targets, weights):
        self.fit_encoder(train_ids)
        keep = weights > 0
        if len(np.unique(targets[keep])) < 2:
            raise ValueError("nuisance training requires matched positive and negative ticks")
        x = self.features(vids, times, fit=True)
        self.model.fit(x, targets, sample_weight=weights)
        return self

    def logits(self, video, indices):
        ix = np.asarray(indices, int)
        x = self.features([video.video_id] * len(ix), video.t_sec[ix])
        return self.model.decision_function(x)

    def predict(self, video, normalizer=None, batch_size=512, indices=None):
        ix = np.arange(len(video.t_sec)) if indices is None else np.asarray(indices, int)
        return expit(self.logits(video, ix))


class OffsetLogisticRegression:
    def __init__(self, C=0.1, max_iter=25):
        self.C, self.max_iter, self.coefficients = C, max_iter, None

    def fit(self, x, y, sample_weight=None, offset=None):
        weight = np.ones(len(y)) if sample_weight is None else sample_weight
        offset = np.zeros(len(y)) if offset is None else offset
        n = max(float(weight.sum()), 1)
        if self.coefficients is None:
            self.coefficients = np.zeros(x.shape[1] + 1, np.float64)
        def objective(beta):
            z = x @ beta[:-1] + beta[-1] + offset
            return np.dot(weight, np.logaddexp(0, z) - y * z) / n + np.dot(beta[:-1], beta[:-1]) / (2 * self.C * n)
        def gradient(beta):
            residual = weight * (expit(x @ beta[:-1] + beta[-1] + offset) - y) / n
            return np.r_[x.T @ residual + beta[:-1] / (self.C * n), residual.sum()]
        def hessian(beta, vector):
            probability = expit(x @ beta[:-1] + beta[-1] + offset)
            curvature = weight * probability * (1 - probability) * (x @ vector[:-1] + vector[-1]) / n
            return np.r_[x.T @ curvature + vector[:-1] / (self.C * n), curvature.sum()]
        result = minimize(objective, self.coefficients, method="Newton-CG", jac=gradient, hessp=hessian,
                          options={"maxiter": self.max_iter, "xtol": 1e-6})
        if not np.isfinite(result.x).all():
            raise ValueError("non-finite nuisance residual coefficients")
        self.coefficients = result.x
        return self

    def predict_proba(self, x, offset=None):
        offset = np.zeros(len(x)) if offset is None else offset
        score = expit(x @ self.coefficients[:-1] + self.coefficients[-1] + offset)
        return np.stack([1 - score, score], axis=1)
