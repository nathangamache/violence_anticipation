"""Checkpoint selection on validation source-and-time-matched far-window AUROC."""

import numpy as np
from sklearn.metrics import roc_auc_score

from anticipation.eval.va30 import TIME_MATCH_EDGES, time_match_weights


class MatchedValidation:
    def __init__(self, store, ids, limit=0, band_hi=60):
        rows, times, sources, positives = [], [], [], []
        for vid in ids:
            v = store[vid]
            lab = v.labels
            observed = v.observed if v.observed is not None else np.ones(len(v.t_sec), bool)
            pos = ~lab.masked & lab.pre_violence & (lab.tte > 30) & (lab.tte <= band_hi) & observed
            neg = ~lab.masked & (lab.remaining >= 60) & observed & (store.catalog.loc[vid, "role"] == "clean_negative")
            ix = np.flatnonzero(pos | neg)
            rows.extend((vid, int(i)) for i in ix)
            times.extend(v.t_sec[ix])
            sources.extend([str(store.catalog.loc[vid, "source_kind"])] * len(ix))
            positives.extend(pos[ix])
        selected = np.arange(len(rows))
        if limit and len(selected) > limit:
            selected = np.sort(np.random.default_rng(0).choice(selected, limit, replace=False))
        self.rows = [rows[i] for i in selected]
        times, sources = np.asarray(times)[selected], np.asarray(sources)[selected]
        pos = np.asarray(positives, bool)[selected]
        weights, keep_pos = time_match_weights(times, pos, ~pos, TIME_MATCH_EDGES, sources)
        self.y, self.weights = pos, np.where(keep_pos, 1.0, weights)
        self.available = bool(keep_pos.any() and (weights > 0).any())
        self.info = {"metric": "source_time_matched_far_auroc" if band_hi == 60 else "source_time_matched_band_auroc",
                     "band_hi": band_hi, "available": self.available,
                     "candidate_ticks": len(rows), "sampled_ticks": len(self.rows),
                     "matched_positive_ticks": int(keep_pos.sum()),
                     "matched_negative_ticks": int((weights > 0).sum()), "sampling_seed": 0}
        self.selection = {}
        for vid, tick in self.rows:
            self.selection.setdefault(vid, []).append(tick)

    def score(self, predict):
        if not self.available:
            return None
        values = np.concatenate([predict(vid, np.asarray(ix)) for vid, ix in self.selection.items()])
        return float(roc_auc_score(self.y, values, sample_weight=self.weights))
