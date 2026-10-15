"""Synthetic tests that pin the VA30 semantics (labels, alarms, metrics, folds). Regression tests for the review
fixes live in tests/test_va30_fixes.py.

Run from the repo root:

    .venv/bin/python -m pytest tests/test_va30.py -q
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from anticipation.data import folds as F
from anticipation.data.labels import event_table, hazard_to_cdf, normalize_segments, tick_labels
from anticipation.eval.va30 import (AlarmPolicy, Benchmark, EvalConfig, EventEval, PairAUC, ScoreTable, TickSet,
                                    alarm_starts, average_precision, evaluate)


# ---------------------------------------------------------------------------------------------------------
# Synthetic benchmark helpers
# ---------------------------------------------------------------------------------------------------------

def make_bench(specs, n_folds=2):
    """``specs``: dicts with vid, duration, segments, role, fold, val (list of folds), source_kind, ..."""
    rows, events, assignment = [], {}, {}
    val = {str(f): [] for f in range(n_folds)}
    for s in specs:
        rows.append({"video_id": s["vid"], "duration_sec": float(s["duration"]), "role": s["role"],
                     "onset_source": "official" if s["role"] == "violent" else "none",
                     "source_kind": s.get("source_kind", "ucf"), "interpersonal": s.get("interpersonal", True),
                     "has_audio": s.get("has_audio", False), "group": s.get("group", s["vid"])})
        events[s["vid"]] = {"segments": s.get("segments", [])}
        assignment[s["vid"]] = s["fold"]
        for f in s.get("val", []):
            val[str(f)].append(s["vid"])
    df = pd.DataFrame(rows).set_index("video_id", drop=False)
    d = {"name": "synthetic", "n_folds": n_folds, "assignment": assignment, "val": val,
         "content_sha256": F.content_hash(assignment, val)}
    return Benchmark(df, events, F.Folds(d))


def standard_specs(n_violent=40, n_clean=40, onset_range=(70, 150), clean_dur=300, seed=0, post=30):
    rng = np.random.default_rng(seed)
    specs = []
    for i in range(n_violent):
        on = float(rng.uniform(*onset_range))
        fold = i % 2
        specs.append({"vid": f"v{i}", "duration": on + post, "segments": [[on, on + post - 5]], "role": "violent",
                      "fold": fold, "val": [1 - fold] if i % 4 < 2 else []})
    for i in range(n_clean):
        fold = i % 2
        specs.append({"vid": f"n{i}", "duration": clean_dur, "role": "clean_negative", "fold": fold,
                      "val": [1 - fold] if i % 4 < 2 else []})
    return specs


def score_table(bench, fn):
    """Fold-agnostic score table from ``fn(TickLabels) -> scores``."""
    parts = []
    for vid in bench.folds.video_ids:
        lab = bench.labels(vid)
        parts.append(pd.DataFrame({"video_id": vid, "t_sec": lab.t_sec, "score": fn(lab), "fold": -1, "seed": 0}))
    return pd.concat(parts, ignore_index=True)


def quick_cfg(**kw):
    kw.setdefault("n_boot", 50)
    kw.setdefault("breakdowns", False)
    kw.setdefault("amoc_points", 10)
    return EvalConfig(**kw)


# ---------------------------------------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------------------------------------

def test_labels_masks_censoring_and_horizons():
    lab = tick_labels(100.5, [[40.5, 50.0], [80.0, 85.0]])
    t = lab.t_sec
    assert len(t) == 100 and t[0] == 1 and t[-1] == 100
    iv = set(t[lab.in_violence].astype(int))
    assert iv == set(range(41, 51)) | set(range(80, 86))
    assert set(t[lab.aftermath].astype(int)) == set(range(51, 61)) | set(range(86, 96))
    i30 = 29  # t = 30
    assert lab.tte[i30] == pytest.approx(10.5)
    assert lab.y[10][i30] == 0 and lab.y[20][i30] == 1 and lab.y[5][i30] == 0
    # Ticks inside violence or aftermath are masked for every horizon.
    for H in (5, 10, 20, 30, 60):
        assert not lab.valid[H][lab.masked].any()
    # After the last aftermath there is no onset: censored when fewer than H seconds remain.
    i96 = 95  # t = 96, remaining 4.5
    assert lab.censored[5][i96] and not lab.valid[5][i96]
    # Next onset from inside the first segment is the second segment.
    assert lab.next_onset[44] == 80.0


def test_clean_negative_censoring():
    lab = tick_labels(50.0, [])
    t = lab.t_sec
    assert not lab.masked.any()
    assert set(t[lab.censored[30]].astype(int)) == set(range(21, 51))
    assert lab.valid[30][19]  # t = 20 has exactly 30 s of future
    assert np.isinf(lab.tte).all()


def test_hazard_targets():
    lab = tick_labels(200.0, [[100.0, 110.0]])
    h = lab.hazard_target
    assert list(h[92]) == [0, 1, -1, -1, -1, -1]          # t = 93, tte = 7 -> bin (5, 10]
    assert list(h[49]) == [0, 0, 0, 0, 0, 1]              # t = 50, tte = 50 -> bin (45, 60]
    assert list(h[29]) == [0, 0, 0, 0, 0, 0]              # t = 30, tte = 70: survived all bins
    assert (h[lab.masked] == -1).all()
    neg = tick_labels(300.0, [])
    assert list(neg.hazard_target[274]) == [0, 0, 0, -1, -1, -1]  # t = 275, 25 s remaining
    cdf = hazard_to_cdf(np.array([0.5, 0.5]))
    assert cdf == pytest.approx([0.5, 0.75])


def test_eligibility_and_segment_normalization():
    segs = normalize_segments([[80, 85], [40.5, 50], [49, 52], [300, 310]], duration=200)
    assert segs == [[40.5, 52.0], [80.0, 85.0]]
    ev = event_table(segs)
    assert ev[0]["eligible30"] and not ev[0]["eligible60"]
    assert ev[1]["clear_start"] == 62.0 and not ev[1]["eligible30"]


# ---------------------------------------------------------------------------------------------------------
# Alarm policy
# ---------------------------------------------------------------------------------------------------------

def test_alarm_policy_kofn_merge_refractory():
    T = 100
    t = np.arange(1, T + 1, dtype=float)
    s = np.zeros(T)
    for tick in (10, 11, 15, 16, 40, 41, 60, 61):
        s[tick - 1] = 1.0
    ev, et, active = alarm_starts(s, t, np.zeros(T, int), np.array([0, T]), 0.5, AlarmPolicy())
    # 10,11 -> active at 11; 15,16 resumes after 4 s silence (merged); 40,41 -> new alarm at 41 (30 s after 11);
    # 60,61 -> new episode but within 30 s of the alarm at 41 (refractory).
    assert list(et) == [11.0, 41.0]
    assert active[10] and not active[9]  # tick 11 active, tick 10 not (needs 2 of 3)
    # A single isolated tick never raises an alarm under 2-of-3.
    s2 = np.zeros(T)
    s2[[5, 20, 35]] = 1
    assert len(alarm_starts(s2, t, np.zeros(T, int), np.array([0, T]), 0.5, AlarmPolicy())[1]) == 0


# ---------------------------------------------------------------------------------------------------------
# Metric implementations vs sklearn
# ---------------------------------------------------------------------------------------------------------

def test_pair_auc_and_ap_match_sklearn():
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(1)
    N, V = 3000, 40
    vidx = rng.integers(0, V, N)
    y = rng.random(N) < 0.2
    score = np.round(rng.normal(size=N) + 0.7 * y, 1)  # rounding creates ties
    pa = PairAUC(score, vidx, y, ~y, V)
    assert pa.value() == pytest.approx(roc_auc_score(y, score), abs=1e-10)
    ap, _ = average_precision(score, y, vidx)
    assert ap == pytest.approx(average_precision_score(y, score), abs=1e-10)
    # Bootstrap weights of all ones reproduce the point estimate.
    W = np.ones((3, V))
    assert pa.boot(W) == pytest.approx([pa.value()] * 3)
    _, apb = average_precision(score, y, vidx, W)
    assert apb == pytest.approx([ap] * 3, abs=1e-6)


def test_weighted_pair_auc_and_time_matching():
    from sklearn.metrics import roc_auc_score

    from anticipation.eval.va30 import time_match_weights

    rng = np.random.default_rng(2)
    N, V = 2000, 30
    vidx = rng.integers(0, V, N)
    y = rng.random(N) < 0.3
    score = np.round(rng.normal(size=N) + y, 1)
    w = rng.random(N)
    for flip in (False, True):  # exercise both loop directions of PairAUC
        pos, neg = (y, ~y) if not flip else (y & (vidx < 3), ~y)
        pa = PairAUC(score, vidx, pos, neg, V, neg_w=w)
        sel = pos | neg
        sw = np.where(pos[sel], 1.0, w[sel])
        assert pa.value() == pytest.approx(roc_auc_score(pos[sel], score[sel], sample_weight=sw), abs=1e-10)
    # A score that is a function of time only gets AUROC 0.5 after time matching.
    t = np.r_[rng.uniform(1, 60, 500), rng.uniform(1, 3000, 5000)]
    pos = np.r_[np.ones(500, bool), np.zeros(5000, bool)]
    vid = np.arange(len(t)) % 50
    edges = (0, 10, 20, 30, 45, 60, np.inf)
    b = np.searchsorted(np.array(edges[1:-1]), t, side="right")
    tw, pos2 = time_match_weights(t, pos, ~pos, edges)
    assert PairAUC(-t, vid, pos, ~pos, 50).value() > 0.9
    assert PairAUC(-b.astype(float), vid, pos2, ~pos, 50, neg_w=tw).value() == pytest.approx(0.5)


# ---------------------------------------------------------------------------------------------------------
# End-to-end semantics
# ---------------------------------------------------------------------------------------------------------

def perfect(lab):
    return ((lab.tte <= 60) & ~lab.masked).astype(float)


def test_perfect_anticipator():
    bench = make_bench(standard_specs())
    res = evaluate(score_table(bench, perfect), bench, quick_cfg())
    m = res["seeds"]["0"]
    assert m["far_window_auroc"]["clean"]["value"] == pytest.approx(1.0)
    assert m["far_window_auroc"]["secondary"]["value"] == pytest.approx(1.0)
    for op in m["operating_points"].values():
        assert op["eligible30"]["success30"]["value"] == pytest.approx(1.0)
        assert op["fa_per_hour_clean"]["value"] == pytest.approx(0.0)
        assert op["fa_per_hour_all"]["value"] == pytest.approx(0.0)
        assert op["eligible30"]["lead_sec"]["median"]["value"] >= 30


def test_always_on_alarm_far_from_clip_start_is_not_rewarded():
    # Onsets >= 130 s so an alarm raised at the start of a video is outside [t0 - 120, t0). With real-like onsets
    # (many in [32, 122] s) an always-on alarm DOES earn Success@30 at a fixed threshold; that case is pinned in
    # tests/test_va30_fixes.py::test_always_on_with_real_like_onsets_is_caught_by_the_position_null.
    bench = make_bench(standard_specs(onset_range=(130, 200)))
    df = score_table(bench, lambda lab: np.ones(lab.T))
    res = evaluate(df, bench, quick_cfg())
    m = res["seeds"]["0"]
    assert m["far_window_auroc"]["clean"]["value"] == pytest.approx(0.5)
    for op in m["operating_points"].values():  # validation FA/h of always-on is far above 5 -> theta = inf
        assert op["eligible30"]["success30"]["value"] == 0.0
    # At a fixed threshold the always-on alarm fires once per video: high FA/h and no anticipation credit.
    ts = TickSet(bench, bench.folds.video_ids, {v: np.ones(bench.labels(v).T) for v in bench.folds.video_ids})
    ee = EventEval(ts, AlarmPolicy())
    s = ee.summarize(ee.outcomes(np.full(ts.V, 0.5)))
    n_clean, hours = ts.clean_v.sum(), ts.duration_v[ts.clean_v].sum() / 3600
    assert s["fa_per_hour_clean"]["value"] == pytest.approx(n_clean / hours)
    assert s["fa_per_hour_clean"]["value"] > 5
    assert s["eligible30"]["success30"]["value"] == 0.0
    assert s["random_reference_success30"]["nominal_W90"]["value"] > 0.25


def test_random_scores_match_random_reference():
    rng = np.random.default_rng(0)
    specs = []
    for i in range(1500):  # one eligible onset at 150 s: the Success@30 window [30, 120] is complete
        specs.append({"vid": f"v{i}", "duration": 200.0, "segments": [[150.0, 170.0]], "role": "violent",
                      "fold": i % 2, "val": [1 - i % 2] if i % 3 == 0 else []})
    for i in range(600):
        specs.append({"vid": f"n{i}", "duration": 600.0, "role": "clean_negative", "fold": i % 2,
                      "val": [1 - i % 2] if i % 3 == 0 else []})
    bench = make_bench(specs)
    df = score_table(bench, lambda lab: rng.random(lab.T))
    # The Poisson reference 1 - exp(-FAR * W) is accurate in the operating regime (1 to 5 FA/h). At high alarm
    # rates (about 20 FA/h) the 30 s refractory period makes alarms more regular than Poisson and the true chance
    # level is a few points higher than the formula (measured: 0.43 vs 0.37 at 18.7 FA/h).
    res = evaluate(df, bench, quick_cfg(n_boot=20, fa_targets=(2.0, 5.0)))
    m = res["seeds"]["0"]
    assert abs(m["far_window_auroc"]["clean"]["value"] - 0.5) < 0.02
    assert abs(m["horizon"]["30"]["auroc"]["value"] - 0.5) < 0.02
    for key in ("fa2", "fa5"):
        op = m["operating_points"][key]
        far = op["fa_per_hour_clean"]["value"]
        assert far > 0.5 * float(key[2:])  # the validation threshold transfers to test
        ref = op["random_reference_success30"]["nominal_W90"]["value"]
        got = op["eligible30"]["success30"]["value"]
        assert ref == pytest.approx(1 - np.exp(-far * 90 / 3600))
        assert abs(got - ref) < 0.03, (key, got, ref, far)


def test_alarms_inside_violence_never_count_as_anticipation():
    bench = make_bench(standard_specs())
    res = evaluate(score_table(bench, lambda lab: lab.in_violence.astype(float)), bench, quick_cfg())
    m = res["seeds"]["0"]
    for op in m["operating_points"].values():
        assert op["eligible30"]["success0"]["value"] == 0.0
        assert op["eligible30"]["success30"]["value"] == 0.0
        assert op["fa_per_hour_clean"]["value"] == 0.0
        assert op["fa_per_hour_all"]["value"] == 0.0  # alarms inside [t0 - 120, e + 10] are not false alarms
        assert op["detection"]["value"] == pytest.approx(1.0)  # ...but they are detections
    assert m["detection_auroc"]["value"] == pytest.approx(1.0)
    # Masked ticks are excluded from anticipation metrics: the detector's in-violence ticks do not raise AUROC.
    assert m["horizon"]["30"]["auroc"]["value"] == pytest.approx(0.5)


def test_aftermath_and_late_alarms_are_not_anticipation():
    # An alarm that starts in the aftermath of event 1 and continues does not anticipate event 2 if it starts
    # before event 2's clear lead-in window.
    specs = [{"vid": "v0", "duration": 400.0, "segments": [[100.0, 110.0], [250.0, 260.0]], "role": "violent",
              "fold": 0}, {"vid": "n0", "duration": 300.0, "role": "clean_negative", "fold": 1}]
    bench = make_bench(specs)
    lab = bench.labels("v0")
    s = ((lab.t_sec >= 112) & (lab.t_sec <= 125)).astype(float)  # inside the aftermath of event 1
    ts = TickSet(bench, ["v0", "n0"], {"v0": s, "n0": np.zeros(bench.labels("n0").T)})
    ee = EventEval(ts, AlarmPolicy())
    out = ee.outcomes(np.full(2, 0.5))
    # Event 2: clear_start = 120, window [130, 250); the alarm started at 113 -> no anticipation credit.
    assert np.isnan(out["lead"][1])
    assert out["fa"][0] == 0  # 113 lies inside [t0 - 120, e + 10] of event 1 -> not a false alarm either


def test_scores_must_cover_every_tick():
    bench = make_bench(standard_specs(n_violent=8, n_clean=8))
    df = score_table(bench, perfect)
    df = df[~((df.video_id == "v0") & (df.t_sec == 5))]
    with pytest.raises(ValueError):
        evaluate(df, bench, quick_cfg())
    res = evaluate(df, bench, quick_cfg(allow_missing=True))
    assert res["coverage"]["test"]["held_ticks"] == 1


def test_bootstrap_ci_brackets_point():
    bench = make_bench(standard_specs(n_violent=60, n_clean=60))
    rng = np.random.default_rng(3)
    df = score_table(bench, lambda lab: perfect(lab) * 0.3 + rng.random(lab.T))
    res = evaluate(df, bench, quick_cfg(n_boot=200))
    a = res["seeds"]["0"]["far_window_auroc"]["clean"]
    assert a["ci"][0] <= a["value"] <= a["ci"][1]
    assert 0.5 < a["value"] < 1.0


# ---------------------------------------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------------------------------------

def synthetic_catalog(n=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        src = ["xd_movie", "xd_youtube", "ucf"][i % 3]
        group = f"g{i // 4}" if src.startswith("xd") else f"ucf{i}"
        role = rng.choice(["violent", "clean_negative", "other_anomaly"], p=[0.5, 0.4, 0.1])
        e30 = role == "violent" and rng.random() < 0.2
        rows.append({"video_id": f"{src}/{i}", "source": src.split("_")[0], "path": f"/x/{i}.mp4",
                     "duration_sec": 100.0, "group": f"{src[:2]}:{group}", "role": role,
                     "source_kind": src, "eligible30": e30, "eligible60": False, "n_eligible30": int(e30)})
    rows.append({"video_id": "yt/a", "source": "yt", "path": "/x/a.mp4", "duration_sec": 100.0, "group": "yt:a",
                 "role": "holdout", "source_kind": "yt", "eligible30": False, "eligible60": False, "n_eligible30": 0})
    return pd.DataFrame(rows).set_index("video_id", drop=False)


def test_folds_grouped_deterministic_and_frozen(tmp_path):
    df = synthetic_catalog()
    a1, v1 = F.make_folds(df)
    a2, v2 = F.make_folds(df)
    assert a1 == a2 and v1 == v2
    assert "yt/a" not in a1
    g2f = {}
    for vid, f in a1.items():
        assert g2f.setdefault(df.loc[vid, "group"], f) == f  # groups never cross folds
    sizes = np.bincount(list(a1.values()))
    assert sizes.max() - sizes.min() <= 6
    for f in range(F.N_FOLDS):
        val = set(v1[str(f)])
        assert val and not any(a1[v] == f for v in val)
        train = {v for v, k in a1.items() if k != f and v not in val}
        assert not ({df.loc[v, "group"] for v in val} & {df.loc[v, "group"] for v in train})
    # Stratification: eligible30 videos spread over all folds.
    e30 = [a1[v] for v in a1 if df.loc[v, "eligible30"]]
    assert len(set(e30)) == F.N_FOLDS

    # CLI: writes once, refuses to overwrite without --force, verifies the hash.
    cat = tmp_path / "videos.csv"
    df.to_csv(cat, index=False)
    out = tmp_path / "folds.json"
    assert F.main(["--catalog", str(cat), "--events", str(tmp_path / "none.json"), "--out", str(out)]) == 0
    h = json.load(open(out))["content_sha256"]
    assert F.main(["--catalog", str(cat), "--events", str(tmp_path / "none.json"), "--out", str(out)]) == 2
    assert json.load(open(out))["content_sha256"] == h
    d = json.load(open(out))
    d["assignment"][next(iter(d["assignment"]))] = 4 if d["assignment"][next(iter(d["assignment"]))] != 4 else 3
    json.dump(d, open(tmp_path / "edited.json", "w"))
    with pytest.raises(ValueError):
        F.Folds.load(str(tmp_path / "edited.json"))

    # Extension: base assignments are kept, new videos in an existing group inherit its fold.
    base = json.load(open(out))
    extra = df.iloc[:2].copy()
    extra["video_id"] = [f"{v}_new" for v in extra.video_id]
    extra.index = extra.video_id
    df2 = pd.concat([df, extra])
    a3, v3 = F.make_folds(df2, base=base)
    assert all(a3[v] == f for v, f in base["assignment"].items())
    for v in extra.index:
        assert a3[v] == base["assignment"][v.removesuffix("_new")]
    for f in range(F.N_FOLDS):
        assert set(base["val"][str(f)]) <= set(v3[str(f)])
