"""Causal window statistics and validation-selected tabular classifiers."""

import copy
import warnings

import numpy as np
from scipy.ndimage import maximum_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning


WINDOWS = (5, 15, 30, 60)


def window_statistics(video, normalizer, indices=None, position_control=False):
    indices = np.arange(len(video.t_sec)) if indices is None else np.asarray(indices, int)
    T = len(video.t_sec)
    size = sum(16 * x.shape[1] + 4 for x in video.streams) + int(position_control)
    out = np.empty((len(indices), size), np.float32)
    offset = 0
    time = np.arange(T, dtype=np.float64)
    for j, raw in enumerate(video.streams):
        p = video.present[:, j].astype(np.float64)
        d = raw.shape[1]
        for left in range(0, d, 128):
            right = min(d, left + 128)
            x = np.where(p[:, None], (raw[:, left:right] - normalizer.means[j][left:right]) /
                         normalizer.scales[j][left:right], 0).astype(np.float64)
            def prefix(a):
                return np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)])
            sx, sx2, stx = prefix(x), prefix(x * x), prefix(time[:, None] * x)
            sn, st, st2 = prefix(p), prefix(p * time), prefix(p * time * time)
            for k, w in enumerate(WINDOWS):
                lo, hi = np.maximum(0, indices - w + 1), indices + 1
                n = sn[hi] - sn[lo]
                n_safe = np.maximum(n, 1)
                total = sx[hi] - sx[lo]
                mean = total / n_safe[:, None]
                std = np.sqrt(np.maximum((sx2[hi] - sx2[lo]) / n_safe[:, None] - mean * mean, 0))
                ts, ts2 = st[hi] - st[lo], st2[hi] - st2[lo]
                denom = ts2 - ts * ts / n_safe
                slope = ((stx[hi] - stx[lo]) - total * (ts / n_safe)[:, None]) / np.maximum(denom, 1e-8)[:, None]
                slope[denom <= 1e-8] = 0
                maximum = maximum_filter1d(np.where(p[:, None], x, -np.inf), size=w, axis=0,
                                           origin=(w - 1) // 2, mode="constant", cval=-np.inf)[indices]
                maximum[n == 0] = 0
                for stat, values in enumerate((mean, maximum, std, slope)):
                    out[:, offset + (4 * k + stat) * d + left:offset + (4 * k + stat) * d + right] = values
        for k, w in enumerate(WINDOWS):
            hi = indices + 1
            lo = np.maximum(0, hi - w)
            count = np.r_[0, p.cumsum()]
            out[:, offset + 16 * d + k] = (count[hi] - count[lo]) / (hi - lo)
        offset += 16 * d + 4
    if position_control:
        out[:, -1] = video.t_sec[indices]
    return out


def make_table(store, ids, selection, normalizer, position_control=False):
    rows, targets, weights = [], [], []
    for vid in ids:
        ix = selection[vid]
        if len(ix):
            v = store[vid]
            rows.append(window_statistics(v, normalizer, ix, position_control))
            targets.append(v.labels.y[30][ix])
            weights.append(np.ones(len(ix), np.float32))
    if not rows:
        raise ValueError("no uncensored training/validation ticks for horizon 30")
    return np.concatenate(rows), np.concatenate(targets), np.concatenate(weights)


class StatsModel:
    def __init__(self, kind, seed=0, threads=4, position_control=False):
        self.kind, self.seed, self.threads, self.position_control = kind, seed, threads, position_control
        self.scaler = StandardScaler(copy=False)
        self.backend = kind

    def fit(self, train, val, epochs=20, patience=4, check_time=lambda: None):
        x, y, w = train
        vx, vy, vw = val
        if len(np.unique(y)) < 2:
            raise ValueError("training horizon-30 labels need both classes")
        x = self.scaler.fit_transform(x)
        vx = self.scaler.transform(vx)
        if self.kind == "stats_logreg":
            # The supplied SciPy LBFGS native routine crashes on some float16-cache-derived tables.
            model = LogisticRegression(C=0.1, solver="newton-cg", max_iter=25, warm_start=True, random_state=self.seed)
        else:
            try:
                from lightgbm import LGBMClassifier
                model = LGBMClassifier(n_estimators=10, num_leaves=15, learning_rate=0.05,
                                       n_jobs=self.threads, random_state=self.seed, verbosity=-1)
                self.backend = "lightgbm"
            except ImportError:
                model = HistGradientBoostingClassifier(max_iter=10, max_leaf_nodes=15, learning_rate=0.05,
                                                       early_stopping=False, warm_start=True, random_state=self.seed)
                self.backend = "sklearn_hist_gradient_boosting"
        self.history, best, stale = [], float("inf"), 0
        for epoch in range(epochs):
            check_time()
            if self.backend == "sklearn_hist_gradient_boosting":
                model.set_params(max_iter=10 * (epoch + 1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                kwargs = {"init_model": model.booster_} if self.backend == "lightgbm" and epoch else {}
                model.fit(x, y, sample_weight=w, **kwargs)
            loss = log_loss(vy, model.predict_proba(vx)[:, 1], labels=[0, 1], sample_weight=vw)
            self.history.append({"epoch": epoch + 1, "val_loss": float(loss)})
            if loss < best - 1e-6:
                best, stale, self.best_epoch = loss, 0, epoch + 1
                self.model = copy.deepcopy(model)
            else:
                stale += 1
            if stale >= patience:
                break
        return self

    def predict(self, video, normalizer, batch_size=512):
        chunks = []
        for start in range(0, len(video.t_sec), batch_size):
            ix = np.arange(start, min(start + batch_size, len(video.t_sec)))
            # Statistics need at most 60 past ticks; this bounds memory on long videos.
            lo = max(0, start - 59)
            from anticipation.models.data import VideoFeatures
            part = VideoFeatures(video.video_id, video.t_sec[lo:ix[-1] + 1],
                                 [x[lo:ix[-1] + 1] for x in video.streams], video.present[lo:ix[-1] + 1], None)
            x = window_statistics(part, normalizer, ix - lo, self.position_control)
            chunks.append(self.model.predict_proba(self.scaler.transform(x))[:, 1])
        return np.concatenate(chunks)
