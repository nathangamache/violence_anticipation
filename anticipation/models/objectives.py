"""Band targets, extended censored hazards and event-level training bags."""

import numpy as np

from anticipation.data.labels import video_event_table
from anticipation.eval.va30 import TIME_MATCH_EDGES, time_match_weights

TRAIN_HAZARD_EDGES = (0., 5., 10., 20., 30., 45., 60., 90., 120., 180., 300.)


def band_targets(video, role):
    labels = video.labels
    observed = video.observed if video.observed is not None else np.ones(len(video.t_sec), bool)
    pos = observed & ~labels.masked & labels.pre_violence & (labels.tte > 30) & (labels.tte <= 120)
    neg = observed & ~labels.masked & (labels.remaining >= 60) & (role == "clean_negative")
    return pos.astype(np.float32), pos | neg


def matched_weights(times, sources, target):
    positive = np.asarray(target) > 0
    negative, keep = time_match_weights(np.asarray(times), positive, ~positive,
                                       TIME_MATCH_EDGES, np.asarray(sources))
    return np.where(keep, 1., negative).astype(np.float32)


def event_bags(store, ids, max_ticks=16, seed=0):
    rng = np.random.default_rng(seed)
    clean = {}
    for vid in ids:
        if store.catalog.loc[vid, "role"] == "clean_negative":
            v = store[vid]
            good = v.observed & ~v.labels.masked & (v.labels.remaining >= 60)
            clean.setdefault(str(store.catalog.loc[vid, "source_kind"]), []).append((vid, good))
    bags, missing = [], 0
    for vid in ids:
        if store.catalog.loc[vid, "role"] != "violent":
            continue
        video = store[vid]
        for event in video_event_table(store.events.get(vid, {})):
            if not event["eligible30"]:
                continue
            lo, hi = max(event["onset"] - 120, event["clear_start"]), event["onset"] - 30
            mask = ((video.t_sec >= lo) & (video.t_sec <= hi) & ~video.labels.masked & video.observed)
            ticks = np.flatnonzero(mask)
            if not len(ticks):
                continue
            ticks = np.sort(rng.choice(ticks, min(len(ticks), max_ticks), replace=False))
            matches = []
            for negative, good in clean.get(str(store.catalog.loc[vid, "source_kind"]), []):
                if ticks[-1] < len(good) and good[ticks].all():
                    matches.append(negative)
            if not matches:
                missing += 1
                continue
            negative = matches[int(rng.integers(len(matches)))]
            bags.extend([(vid, ticks, 1.), (negative, ticks.copy(), 0.)])
    return bags, {"positive_event_bags": len(bags) // 2, "matched_clean_bags": len(bags) // 2,
                  "events_without_matched_clean_window": missing, "max_ticks_per_bag": max_ticks}
