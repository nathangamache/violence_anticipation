"""Regression tests for the VA30 review fixes (docs/VA30.md section 10): every test pins one reviewed defect.

Run from the repo root:

    .venv/bin/python -m pytest tests/test_va30_fixes.py -q
"""

import json

import numpy as np
import pandas as pd
import pytest

from anticipation.data import annotations as A
from anticipation.data import catalog as C
from anticipation.data import folds as F
from anticipation.data import near_dups as N
from anticipation.data.labels import blocker_spans, event_table, tick_labels
from anticipation.eval.va30 import (AlarmPolicy, Benchmark, EvalConfig, EventEval, PairAUC, ScoreTable, TickSet,
                                    TIME_EDGES, TIME_MATCH_EDGES, choose_threshold, evaluate, position_null,
                                    time_match_weights, time_strata)


# ---------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------

def make_bench(specs, n_folds=2):
    """``specs``: dicts with vid, duration, role, fold, optional segments, other_segments, mask_spans, val,
    source_kind, onset_source, group."""
    rows, events, assignment = [], {}, {}
    val = {str(f): [] for f in range(n_folds)}
    for s in specs:
        rows.append({"video_id": s["vid"], "duration_sec": float(s["duration"]), "role": s["role"],
                     "onset_source": s.get("onset_source", "official" if s["role"] == "violent" else "none"),
                     "source_kind": s.get("source_kind", "ucf"), "interpersonal": s.get("interpersonal", True),
                     "has_audio": s.get("has_audio", False), "group": s.get("group", s["vid"])})
        events[s["vid"]] = {"segments": s.get("segments", [])}
        for k in ("other_segments", "mask_spans"):
            if s.get(k):
                events[s["vid"]][k] = s[k]
        assignment[s["vid"]] = s["fold"]
        for f in s.get("val", []):
            val[str(f)].append(s["vid"])
    df = pd.DataFrame(rows).set_index("video_id", drop=False)
    d = {"name": "synthetic", "n_folds": n_folds, "assignment": assignment, "val": val,
         "content_sha256": F.content_hash(assignment, val)}
    return Benchmark(df, events, F.Folds(d))


def table(bench, fn, vids=None):
    parts = []
    for vid in vids or bench.folds.video_ids:
        lab = bench.labels(vid)
        parts.append(pd.DataFrame({"video_id": vid, "t_sec": lab.t_sec, "score": fn(lab, vid), "fold": -1,
                                   "seed": 0}))
    return pd.concat(parts, ignore_index=True)


def quick(**kw):
    kw.setdefault("n_boot", 30)
    kw.setdefault("breakdowns", False)
    kw.setdefault("amoc_points", 8)
    return EvalConfig(**kw)


def one_video_outcomes(duration, segments, raw_ticks):
    """Leads of every event of one violent video for raw alarm ticks ``raw_ticks`` (threshold 0.5)."""
    b = make_bench([{"vid": "v", "duration": duration, "segments": segments, "role": "violent", "fold": 0},
                    {"vid": "n", "duration": 3600.0, "role": "clean_negative", "fold": 0}])
    lab = b.labels("v")
    sc = {"v": np.isin(lab.t_sec, raw_ticks).astype(float), "n": np.zeros(b.labels("n").T)}
    ts = TickSet(b, ["v", "n"], sc)
    ee = EventEval(ts, AlarmPolicy())
    return ee, ee.outcomes(np.full(2, 0.5))


def real_like_specs(n_violent=120, n_clean=60, seed=0):
    """First onsets spread like VA30 (many in [32, 122] s), a few late ones, 300 s clean negatives."""
    rng = np.random.default_rng(seed)
    specs = []
    for i in range(n_violent):
        on = float(rng.uniform(35, 120)) if i % 4 else float(rng.uniform(160, 260))
        specs.append({"vid": f"v{i}", "duration": on + 40, "segments": [[on, on + 20]], "role": "violent",
                      "fold": i % 2, "val": [1 - i % 2] if i % 3 == 0 else []})
    for i in range(n_clean):
        specs.append({"vid": f"n{i}", "duration": 300.0, "role": "clean_negative", "fold": i % 2,
                      "val": [1 - i % 2] if i % 3 == 0 else []})
    return specs


# ---------------------------------------------------------------------------------------------------------
# Issue: detection memory (later segments) inflated the headline frame AUROC
# ---------------------------------------------------------------------------------------------------------

def test_memory_oracle_earns_nothing_on_headline_frame_metrics():
    specs = []
    for i in range(30):  # first fight at 10 s, a second one with a long clear lead-in
        specs.append({"vid": f"v{i}", "duration": 200.0, "segments": [[10.0, 20.0], [120.0, 140.0]],
                      "role": "violent", "fold": i % 2, "val": [1 - i % 2] if i % 4 < 2 else []})
    for i in range(30):
        specs.append({"vid": f"n{i}", "duration": 300.0, "role": "clean_negative", "fold": i % 2,
                      "val": [1 - i % 2] if i % 4 < 2 else []})
    b = make_bench(specs)
    res = evaluate(table(b, lambda lab, v: (~lab.pre_violence).astype(float)), b, quick())
    m = res["seeds"]["0"]
    fw = m["far_window_auroc"]
    assert fw["clean_all_segments"]["value"] == pytest.approx(1.0)       # the spec-literal metric is fooled
    # Headline: only pre-violence ticks; here every 30-60 s positive comes after the first fight -> none left.
    assert fw["clean"]["n_pos"] == 0
    for op in m["operating_points"].values():
        assert op["eligible30"]["success30"]["value"] == 0.0              # always-on after violence: no new start


def test_pre_violence_headline_equals_spec_when_single_segment():
    rng = np.random.default_rng(0)
    b = make_bench(real_like_specs())
    res = evaluate(table(b, lambda lab, v: ((lab.tte <= 60) & ~lab.masked) * 0.5 + rng.random(lab.T)), b, quick())
    fw = res["seeds"]["0"]["far_window_auroc"]
    assert fw["clean"]["value"] == pytest.approx(fw["clean_all_segments"]["value"])


# ---------------------------------------------------------------------------------------------------------
# Issue: event-level metrics and the position shortcut; position null and late-window Success@30
# ---------------------------------------------------------------------------------------------------------

def test_start_of_clip_alarm_beats_poisson_but_not_position_null():
    b = make_bench(real_like_specs())
    vids = b.folds.video_ids
    sc = {v: (b.labels(v).t_sec <= 3).astype(float) for v in vids}   # one alarm at tick 2 of every clip
    ts = TickSet(b, vids, sc)
    ee = EventEval(ts, AlarmPolicy())
    s = ee.summarize(ee.outcomes(np.full(ts.V, 0.5)))
    far = s["fa_per_hour_clean"]["value"]
    assert far == pytest.approx(12.0)                                   # one alarm per 300 s clean clip
    got = s["eligible30"]["success30"]["value"]
    poisson = s["random_reference_success30"]["nominal_W90"]["value"]
    assert got > 0.6 and poisson < 0.3                                   # the Poisson null is fooled
    assert s["eligible30_late"]["success30"]["value"] == 0.0             # late windows exclude the clip start
    curve = position_null(ee)
    from anticipation.eval.va30 import null_at
    assert null_at(curve, far) >= got - 1e-9                             # the position oracle is not fooled


def test_always_on_with_real_like_onsets_is_caught_by_the_position_null():
    b = make_bench(real_like_specs())
    res = evaluate(table(b, lambda lab, v: np.ones(lab.T)), b, quick(fa_targets=(2.0, 5.0)))
    m = res["seeds"]["0"]
    for op in m["operating_points"].values():   # validation FA/h of always-on is 12 > 5 -> theta = +inf
        assert op["eligible30"]["success30"]["value"] == 0.0
    top = [p for p in m["amoc_curve"] if p["theta"] <= 1.0][0]          # a fixed threshold: always on
    assert top["success30"] > 0.5 and top["success30_late"] == 0.0
    from anticipation.eval.va30 import null_at
    assert null_at(m["position_null_curve"], top["fa_per_hour_clean"]) >= top["success30"] - 1e-9


def test_position_null_is_a_concave_hull():
    from anticipation.eval.va30 import _concave_hull, null_at
    hull = _concave_hull([(1.0, 0.05, "a"), (2.0, 0.06, "b"), (4.0, 0.30, "c"), (3.0, 0.10, "d"), (8.0, 0.32, "e")])
    assert [h[2] for h in hull] == ["origin", "c", "e"]
    curve = {"success30": hull}
    assert null_at(curve, 2.0) == pytest.approx(0.15)       # half-way to "c": a 50/50 mixture of "off" and "c"
    assert null_at(curve, 20.0) == pytest.approx(0.32)


# ---------------------------------------------------------------------------------------------------------
# Issue: time matching ignored the source; source x time matching
# ---------------------------------------------------------------------------------------------------------

def test_source_by_time_matching_neutralizes_source_aware_position_scores():
    rng = np.random.default_rng(1)
    n = 40000
    src = np.where(rng.random(n) < 0.5, "movie", "cctv")
    pos = rng.random(n) < np.where(src == "movie", 0.2, 0.05)
    t = np.where(pos, rng.uniform(1, 60, n), rng.uniform(1, np.where(src == "movie", 120, 3000), n))
    vid = rng.integers(0, 200, n)
    score = time_strata(t, src, TIME_MATCH_EDGES).astype(float) * 0.37 % 1.0 + (src == "movie")  # any f(src, bin)
    w_blind, p_blind = time_match_weights(t, pos, ~pos, TIME_MATCH_EDGES)
    w_src, p_src = time_match_weights(t, pos, ~pos, TIME_MATCH_EDGES, source=src)
    blind = PairAUC(score, vid, p_blind, ~pos, 200, neg_w=w_blind).value()
    matched = PairAUC(score, vid, p_src, ~pos, 200, neg_w=w_src).value()
    assert abs(blind - 0.5) > 0.03            # source-blind matching leaves the provenance shortcut
    assert matched == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------------------------------------
# Issue: eligibility ignored the 1 Hz clock and k-of-n smoothing
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("onset,eligible", [(30.0, False), (31.0, False), (31.5, False), (32.0, True)])
def test_eligibility_is_tick_aware(onset, eligible):
    assert event_table([[onset, onset + 10]])[0]["eligible30"] is eligible
    ee, out = one_video_outcomes(300.0, [[onset, onset + 10]], [1, 2])   # the best possible alarm
    assert bool(out["lead"][0] >= 30) == eligible


def test_eligibility_fractional_clear_start():
    ev = event_table([[50.0, 60.2], [100.2, 110.0]])
    assert ev[1]["clear_start"] == pytest.approx(70.2) and ev[1]["max_lead"] == pytest.approx(29.2)
    assert not ev[1]["eligible30"]
    ev = event_table([[50.0, 60.2], [101.0, 110.0]])
    assert ev[1]["eligible30"]


def test_perfect_anticipator_reaches_100_percent_with_onsets_near_30s():
    specs = []
    for i, on in enumerate(np.linspace(30.0, 45.0, 32)):   # includes onsets in [30, 32): not eligible any more
        specs.append({"vid": f"v{i}", "duration": on + 30, "segments": [[on, on + 20]], "role": "violent",
                      "fold": i % 2, "val": [1 - i % 2] if i % 4 < 2 else []})
    for i in range(20):
        specs.append({"vid": f"n{i}", "duration": 300.0, "role": "clean_negative", "fold": i % 2,
                      "val": [1 - i % 2] if i % 4 < 2 else []})
    b = make_bench(specs)
    res = evaluate(table(b, lambda lab, v: ((lab.tte <= 60) & ~lab.masked).astype(float)), b, quick())
    m = res["seeds"]["0"]
    assert m["counts"]["eligible30_events"] == sum(on >= 32 for on in np.linspace(30.0, 45.0, 32))
    for op in m["operating_points"].values():
        assert op["eligible30"]["success30"]["value"] == pytest.approx(1.0)
        assert op["eligible30"]["success30_ceiling"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------------------------
# Issue: merging / refractory carried across event boundaries
# ---------------------------------------------------------------------------------------------------------

def test_detection_alarm_does_not_swallow_next_event_anticipation():
    # Event 1 [100, 105] detected from tick 96; fresh anticipation alarm for event 2 (onset 160) at tick 120.
    ee, out = one_video_outcomes(300.0, [[100, 105], [160, 170]], list(range(95, 105)) + list(range(119, 160)))
    assert out["lead"][0] == pytest.approx(4.0) and out["lead"][1] == pytest.approx(40.0)
    # Detection alarm runs into the aftermath until 108 and is re-raised at 117 (gap < merge gap): split at 115.
    ee, out = one_video_outcomes(300.0, [[100, 105], [160, 170]], list(range(95, 109)) + list(range(116, 160)))
    assert out["lead"][1] == pytest.approx(43.0)
    # An alarm that stays on across the clear_start is NOT restarted by it (no free anticipation credit).
    ee, out = one_video_outcomes(300.0, [[100, 105], [160, 170]], list(range(95, 160)))
    assert np.isnan(out["lead"][1])


# ---------------------------------------------------------------------------------------------------------
# Issue: threshold search used coarse quantiles in the 1-5 FA/h tail
# ---------------------------------------------------------------------------------------------------------

def test_threshold_search_matches_bruteforce_topdown_rule():
    rng = np.random.default_rng(0)
    specs = [{"vid": f"n{i}", "duration": float(rng.uniform(60, 500)), "role": "clean_negative", "fold": 0,
              "val": [1]} for i in range(160)]
    b = make_bench(specs)
    vids = b.folds.val_ids(1)
    sc = {}
    for v in vids:
        T = b.labels(v).T
        e, x = rng.normal(size=T), np.zeros(T)
        for k in range(1, T):
            x[k] = 0.8 * x[k - 1] + e[k]
        sc[v] = x
    ts = TickSet(b, vids, sc)
    ee = EventEval(ts, AlarmPolicy())
    u = np.unique(ts.score)[::-1]
    assert len(u) > 5000
    for target in (1.0, 2.0, 5.0):
        th, fa = choose_threshold(ee, target)
        best = np.inf
        for c in u:                       # brute force: lower theta until FA/h first exceeds the target
            if ee.fa_clean(np.full(ts.V, c)) <= target:
                best = c
            else:
                break
        assert th == best, (target, th, best)
        assert fa <= target


# ---------------------------------------------------------------------------------------------------------
# Issue: score-table contract had silent failure modes
# ---------------------------------------------------------------------------------------------------------

def test_score_table_rejects_silent_failure_modes():
    b = make_bench(real_like_specs(n_violent=8, n_clean=8))
    ref = table(b, lambda lab, v: ((lab.tte <= 60) & ~lab.masked).astype(float))
    ScoreTable(ref)                                                               # a clean table is fine
    dup = pd.concat([ref, ref[ref.video_id == "n0"].assign(score=50.0)])
    with pytest.raises(ValueError, match="share a"):
        ScoreTable(dup)
    with pytest.raises(ValueError, match="share a"):
        ScoreTable(pd.concat([ref[ref.video_id == "n0"].assign(score=50.0), ref]))
    with pytest.raises(ValueError, match="fold"):
        ScoreTable(ref.assign(fold=np.nan))
    with pytest.raises(ValueError, match="seed"):
        ScoreTable(ref.assign(seed=np.nan))
    with pytest.raises(ValueError, match="non-integer"):
        ScoreTable(ref.assign(fold=0.5))
    mixed = pd.concat([ref, ref[ref.video_id == "n0"].assign(fold=0)])
    with pytest.raises(ValueError, match="fold -1"):
        ScoreTable(mixed)
    bad_t = ref.copy()
    bad_t.loc[3, "t_sec"] = np.nan
    with pytest.raises(ValueError, match="t_sec"):
        ScoreTable(bad_t)


def test_nontest_threshold_set_only_for_fold_agnostic_tables():
    b = make_bench(real_like_specs(n_violent=16, n_clean=16))
    df = table(b, lambda lab, v: np.zeros(lab.T))
    res = evaluate(df, b, quick(threshold_set="nontest", fa_targets=(2.0,)))
    assert res["seeds"]["0"]["counts"]["threshold_set"] == "nontest"
    fold_specific = pd.concat([df.assign(fold=f) for f in (0, 1)])
    with pytest.raises(ValueError, match="fold-agnostic"):
        evaluate(fold_specific, b, quick(threshold_set="nontest", fa_targets=(2.0,)))


# ---------------------------------------------------------------------------------------------------------
# Issue: excluded time (non-violent segments of mixed clips, adjacent-clip masks) and uncensored far AUROC
# ---------------------------------------------------------------------------------------------------------

def test_excluded_spans_are_neither_onsets_nor_negatives_nor_false_alarms():
    lab = tick_labels(200.0, [[150.0, 170.0]], other_segments=[[40.0, 50.0]], mask_spans=[[100.0, 110.0]])
    t = lab.t_sec.astype(int)
    assert set(t[lab.excluded]) == set(range(40, 61)) | set(range(100, 111))   # other segment + 10 s aftermath
    assert not lab.pre_violence[t >= 40].any() and lab.pre_violence[t < 40].all()
    assert lab.next_onset[0] == 150.0                                           # other segments are not onsets
    ev = event_table([[150.0, 170.0]], blockers=blocker_spans([[40.0, 50.0]], [[100.0, 110.0]]))
    assert ev[0]["clear_start"] == 110.0 and ev[0]["leadin"] == 40.0
    # Clean negative with a mask span: alarms inside it are not false alarms and its time is not counted.
    b = make_bench([{"vid": "n", "duration": 3600.0, "role": "clean_negative", "fold": 0,
                     "mask_spans": [[1000.0, 1800.0]]},
                    {"vid": "v", "duration": 200.0, "segments": [[150, 160]], "role": "violent", "fold": 0}])
    lab_n = b.labels("n")
    s = ((lab_n.t_sec >= 1200) & (lab_n.t_sec <= 1210)).astype(float)
    ts = TickSet(b, ["n", "v"], {"n": s, "v": np.zeros(b.labels("v").T)})
    ee = EventEval(ts, AlarmPolicy())
    sm = ee.summarize(ee.outcomes(np.full(2, 0.5)))
    assert sm["fa_per_hour_clean"]["n_false_alarms"] == 0
    assert sm["fa_per_hour_clean"]["hours"] == pytest.approx(2800 / 3600, abs=1e-3)


def test_uncensored_far_auroc_counts_short_clean_clips():
    specs = real_like_specs(n_violent=20, n_clean=10)
    specs += [{"vid": f"s{i}", "duration": 45.0, "role": "clean_negative", "fold": i % 2,
               "val": [1 - i % 2] if i % 4 < 2 else []} for i in range(10)]
    b = make_bench(specs)
    res = evaluate(table(b, lambda lab, v: np.zeros(lab.T)), b, quick())
    fw = res["seeds"]["0"]["far_window_auroc"]
    assert fw["clean_uncensored"]["n_neg_videos"] == fw["clean"]["n_neg_videos"] + 10


def test_breakdowns_report_onset_source_and_event_index():
    specs = real_like_specs(n_violent=24, n_clean=12)
    for s in specs[:12]:
        s["onset_source"] = "hivau"
    b = make_bench(specs)
    rng = np.random.default_rng(2)
    res = evaluate(table(b, lambda lab, v: rng.random(lab.T)), b, quick(breakdowns=True))
    bd = res["seeds"]["0"]["breakdowns"]
    assert {"onset:official", "onset:hivau", "event:first_segment", "event:later_segment"} <= set(bd)


# ---------------------------------------------------------------------------------------------------------
# Folds: unknown videos, check(), validation slices
# ---------------------------------------------------------------------------------------------------------

def synthetic_catalog(n=600, seed=0):
    """Like the real catalog, footage hours are (nearly) unique per group, so a size-ordered assignment is the
    same in every fold (the v1 failure mode)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        src = ["xd_movie", "xd_youtube", "ucf"][i % 3]
        group = f"g{i // 4}" if src.startswith("xd") else f"ucf{i}"
        role = rng.choice(["violent", "clean_negative", "other_anomaly"], p=[0.5, 0.4, 0.1])
        e30 = role == "violent" and rng.random() < 0.2
        rows.append({"video_id": f"{src}/{i}", "source": src.split("_")[0], "path": f"/x/{i}.mp4",
                     "duration_sec": float(rng.lognormal(5.0, 1.0)), "group": f"{src[:2]}:{group}",
                     "split_group": f"{src[:2]}:{group}", "role": role, "source_kind": src, "eligible30": e30,
                     "eligible60": False, "n_eligible30": int(e30)})
    rows.append({"video_id": "yt/a", "source": "yt", "path": "/x/a.mp4", "duration_sec": 100.0, "group": "yt:a",
                 "split_group": "yt:a", "role": "holdout", "source_kind": "yt", "eligible30": False,
                 "eligible60": False, "n_eligible30": 0})
    return pd.DataFrame(rows).set_index("video_id", drop=False)


def test_role_in_fold_unknown_video_is_none():
    df = synthetic_catalog()
    a, v = F.make_folds(df)
    fo = F.Folds({"n_folds": 5, "assignment": a, "val": v, "content_sha256": F.content_hash(a, v)})
    assert fo.role_in_fold("yt/a", 0) is None and fo.role_in_fold("ubi/F_0", 3) is None
    vid = next(iter(a))
    assert fo.role_in_fold(vid, a[vid]) == "test"


def test_check_rejects_val_videos_outside_assignment_and_holdouts(tmp_path):
    df = synthetic_catalog()
    cat = tmp_path / "videos.csv"
    df.to_csv(cat, index=False)
    out = tmp_path / "folds.json"
    assert F.main(["--catalog", str(cat), "--events", str(tmp_path / "x.json"), "--out", str(out)]) == 0
    assert F.check(str(out), str(cat))["ok"]
    d = json.load(open(out))
    d["val"]["0"].append("yt/a")                                   # hold-out injected into a validation slice
    d["content_sha256"] = F.content_hash(d["assignment"], d["val"])
    json.dump(d, open(tmp_path / "bad.json", "w"))
    with pytest.raises(ValueError, match="not in the assignment"):
        F.check(str(tmp_path / "bad.json"), str(cat))
    d = json.load(open(out))
    d["assignment"]["yt/a"] = 0                                    # hold-out injected into a test fold
    d["content_sha256"] = F.content_hash(d["assignment"], d["val"])
    json.dump(d, open(tmp_path / "bad2.json", "w"))
    with pytest.raises(ValueError, match="hold-out"):
        F.check(str(tmp_path / "bad2.json"), str(cat))


def test_v2_validation_slices_are_representative_and_nearly_independent():
    df = synthetic_catalog(n=1200)

    def overlap_and_bias(rule):
        a, v = F.make_folds(df, val_rule=rule)
        sets = [set(v[str(f)]) for f in range(5)]
        ov = max(len(sets[i] & sets[j]) / min(len(sets[i]), len(sets[j])) for i in range(5) for j in range(i + 1, 5))
        dur_val = np.mean([df.loc[list(s), "duration_sec"].mean() for s in sets])
        return ov, dur_val

    ov1, d1 = overlap_and_bias("v1")
    ov2, d2 = overlap_and_bias("v2")
    assert ov2 < 0.5 * ov1
    assert ov2 < 0.35
    assert abs(d2 - df.duration_sec.mean()) < abs(d1 - df.duration_sec.mean())   # v1 favoured long videos


# ---------------------------------------------------------------------------------------------------------
# Catalog: HIVAU judgements, source spans, masks, near-duplicates
# ---------------------------------------------------------------------------------------------------------

def test_hivau_judgement_classifier():
    assert A.classify_judgement("The anomaly exists, specifically a Riot anomaly.") == "violent"
    assert A.classify_judgement("An anomaly exists, specifically an Explosion anomaly.") == "other"
    assert A.classify_judgement("The anomaly exists and the specific name is Explosion and Car Accident.") == "other"
    assert A.classify_judgement("There is no anomaly event (Car accident) in the video.") == "none"
    assert A.classify_judgement("No anomaly exists in this video.", "a man punches another man") == "violent"
    assert A.classify_judgement("The anomaly exists, specifically a Physical Altercation anomaly.") == "violent"


def test_xd_source_span():
    assert A.parse_xd_source_span("Taken.3.2014__#01-22-23_01-23-15_label_A") == (4943.0, 4995.0)
    assert A.parse_xd_source_span("Jason.Bourne.2016__#0-26-58_0-27-20_label_B2") == (1618.0, 1640.0)
    assert A.parse_xd_source_span("v=qQCTvJq83_s__#1_label_A") is None


def _rec(vid, role, span, dur, segs=(), group="xd:M", onset="official", classes=("fighting",), split="test"):
    return {"vid": vid, "src": "xd", "role": role, "span": span, "dur": dur, "segs": [list(s) for s in segs],
            "others": [], "masks": [], "soft_masks": [], "notes": [], "group": group, "franchise": "",
            "onset_source": onset,
            "classes": list(classes), "split": split, "split_group": group}


def test_source_masks_for_adjacent_clips():
    # Violent clip at source 100..250 s with an onset at 120 s (source 220 s); a clean clip ending at 215 s,
    # a clean clip overlapping the violence, and a clean clip far away.
    v = _rec("xd/M__#v", "violent", (100.0, 250.0), 150.0, segs=[(120.0, 140.0)])
    c_near = _rec("xd/M__#c1", "clean_negative", (150.0, 215.0), 65.0, classes=("normal",))
    c_over = _rec("xd/M__#c2", "clean_negative", (200.0, 260.0), 60.0, classes=("normal",))
    c_far = _rec("xd/M__#c3", "clean_negative", (1000.0, 1100.0), 100.0, classes=("normal",))
    C.source_masks([v, c_near, c_over, c_far], log=lambda *a: None)
    assert c_near["masks"] == [[0.0, 65.0]] and c_near["role"] == "clean_negative"   # all within 120 s before
    assert c_over["role"] == "contaminated"
    assert c_far["masks"] == [] and v["masks"] == []


def test_source_masks_for_violent_clips_do_not_break_their_own_lead_in():
    a = _rec("xd/M__#a", "violent", (0.0, 300.0), 300.0, segs=[(250.0, 260.0)])
    b = _rec("xd/M__#b", "violent", (100.0, 250.0), 150.0, segs=[(60.0, 70.0)])    # ends where A's fight starts
    c = _rec("xd/M__#c", "violent", (255.0, 400.0), 145.0, segs=[(60.0, 70.0)])    # starts in A's fight
    C.source_masks([a, b, c], log=lambda *x: None)
    # B: A's pre-onset window lies after B's own fight + aftermath -> soft mask only, B's event stays eligible.
    assert b["masks"] == [] and b["soft_masks"] == [[80.0, 150.0]]
    assert event_table(b["segs"], blockers=blocker_spans(b["others"], b["masks"]))[0]["eligible30"]
    # C: begins inside A's fight -> hard mask (violence + aftermath) that ends C's clear lead-in at 15 s.
    assert c["masks"] == [[0.0, 15.0]]
    ev = event_table(c["segs"], blockers=blocker_spans(c["others"], c["masks"]))[0]
    assert ev["clear_start"] == 15.0 and ev["eligible30"]


def test_near_duplicates_make_duplicate_role_and_split_groups():
    a = _rec("ucf/Shooting015_x264", "violent", None, 57.1, segs=[(28.5, 57.1)], group="ucf:Shooting015_x264")
    b = _rec("ucf/Arrest009_x264", "violent", None, 57.1, segs=[(17.6, 57.1)], group="ucf:Arrest009_x264",
             onset="hivau", classes=("arrest",), split="train")
    c = _rec("ucf/Normal_1", "clean_negative", None, 60.0, group="ucf:Normal_1", onset="none", classes=("normal",))
    d = _rec("ucf/Normal_2", "clean_negative", None, 80.0, group="ucf:Normal_2", onset="none", classes=("normal",))
    m1 = _rec("xd/Bad.Boys.1995__#a", "violent", (0, 60), 60.0, group="xd:Bad.Boys.1995")
    m2 = _rec("xd/Bad.Boys.II.2003__#b", "violent", (0, 60), 60.0, group="xd:Bad.Boys.II.2003")
    m1["franchise"] = m2["franchise"] = "bad_boys"
    nd = {"pairs": [{"a": "ucf/Arrest009_x264", "b": "ucf/Shooting015_x264", "kind": "identical"},
                    {"a": "ucf/Normal_1", "b": "ucf/Normal_2", "kind": "overlap"}]}
    recs = [a, b, c, d, m1, m2]
    C.apply_near_dups(recs, nd, log=lambda *x: None)
    assert a["role"] == "violent" and b["role"] == "duplicate" and b["duplicate_of"] == a["vid"]
    assert a["split_group"] == b["split_group"]
    assert c["split_group"] == d["split_group"] and c["role"] == d["role"] == "clean_negative"
    assert m1["split_group"] == m2["split_group"]


def test_near_dup_matching_on_synthetic_frames():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (80, 12, 16)).astype(np.uint8)
    reenc = np.clip(base.astype(int) + rng.integers(-2, 3, base.shape), 0, 255).astype(np.uint8)
    n_inf, corr, mad = N.compare_aligned(base[:9], reenc[:9])
    assert n_inf == 9 and corr > 0.99 and mad < 3
    lag, n_match, n_ov, c = N.best_lag(base[:60], reenc[17:77])      # b starts 17 s later in the footage
    assert lag == -17 and n_match >= 40
    other = rng.integers(0, 255, (60, 12, 16)).astype(np.uint8)
    assert N.best_lag(base[:60], other)[1] == 0
    assert N.compare_aligned(base[:9], other[:9])[1] < 0.5


# ---------------------------------------------------------------------------------------------------------
# Issue: bootstrap CIs ignored threshold-selection noise
# ---------------------------------------------------------------------------------------------------------

def test_threshold_aware_bootstrap_is_wider_than_fixed_threshold_ci():
    rng = np.random.default_rng(5)
    specs = real_like_specs(n_violent=160, n_clean=40)
    b = make_bench(specs)
    df = table(b, lambda lab, v: ((lab.tte <= 60) & ~lab.masked) * 0.8 + rng.random(lab.T))
    res = evaluate(df, b, quick(n_boot=60, fa_targets=(5.0,), threshold_bootstrap=60))
    op = res["seeds"]["0"]["operating_points"]["fa5"]
    fixed = op["eligible30"]["success30"]["ci"]
    aware = op["threshold_aware_ci"]["success30_ci"]
    assert op["threshold_aware_ci"]["n_boot"] == 60
    assert aware[0] <= op["eligible30"]["success30"]["value"] <= aware[1]
    assert (aware[1] - aware[0]) >= 0.9 * (fixed[1] - fixed[0])
