"""Synthetic checks of the modality screen: gates, classes, controls and the fold-1 freeze.

Run from the repo root:

    .venv/bin/python -m pytest tests/test_screen.py -q
"""

import argparse
import json
import zlib

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from anticipation.data import folds as FO
from anticipation.eval.va30 import Benchmark
from anticipation.screen import decision as D
from anticipation.screen import features as F
from anticipation.screen import metrics as M
from anticipation.screen import prereg as PR
from anticipation.screen import probes as P
from anticipation.screen import run as R
from anticipation.screen import sets as S

N_VIOL, N_CLEAN, N_H1 = 8, 20, 6
VAL_FROM = {0: 2, 1: 3, 2: 3, 3: 4, 4: 0}


@pytest.fixture(autouse=True)
def thread_cap():
    with threadpool_limits(limits=2):
        yield


def build_world(violent_signal=None, h1_signal=None, scene_shift=0.0, seed=0, all_signal=None, integer_onsets=False):
    """Synthetic benchmark + feature loader. ``*_signal(t, onset)`` returns the extra level added to column 0."""
    rng = np.random.default_rng(seed)
    rows, events, assign, info = [], {}, {}, {}
    for f in range(5):
        for i in range(N_VIOL):
            vid = f"synthetic/f{f}v{i}"
            onset = float(rng.uniform(150, 230))
            onset = float(round(onset)) if integer_onsets else onset
            src = "ucf" if i % 2 == 0 else "xd_youtube"
            rows.append(dict(video_id=vid, duration_sec=onset + 30, role="violent", onset_source="official",
                             source_kind=src, interpersonal=True, has_audio=True, group=vid, classes="fighting",
                             first_anomaly_sec=onset))
            events[vid] = {"segments": [[onset, onset + 20]]}
            info[vid] = ("violent", onset)
            assign[vid] = f
        for i in range(N_CLEAN):
            vid = f"synthetic/f{f}n{i}"
            src = "ucf" if i % 2 == 0 else "xd_youtube"
            rows.append(dict(video_id=vid, duration_sec=300.0, role="clean_negative", onset_source="none",
                             source_kind=src, interpersonal=False, has_audio=True, group=vid, classes="",
                             first_anomaly_sec=np.nan))
            events[vid] = {"segments": []}
            info[vid] = ("clean", None)
            assign[vid] = f
        for i in range(N_H1):
            vid = f"synthetic/f{f}h{i}"
            fa = float(rng.uniform(150, 230))
            rows.append(dict(video_id=vid, duration_sec=fa + 10, role="other_anomaly", onset_source="none",
                             source_kind="ucf", interpersonal=False, has_audio=True, group=vid, classes="stealing",
                             first_anomaly_sec=fa))
            events[vid] = {"segments": []}
            info[vid] = ("h1", fa)
            assign[vid] = f
    val = {str(f): [v for v, g in assign.items() if g == VAL_FROM[f] and int(v.split("/f")[1][1:].split("v")[-1].split("n")[-1].split("h")[-1]) % 2 == 0]
           for f in range(5)}
    df = pd.DataFrame(rows).set_index("video_id", drop=False)
    d = {"name": "synthetic", "n_folds": 5, "assignment": assign, "val": val,
         "content_sha256": FO.content_hash(assign, val)}
    bench = Benchmark(df, events, FO.Folds(d))
    h1 = df[df.role == "other_anomaly"][["video_id", "source_kind", "classes", "first_anomaly_sec", "duration_sec"]].copy()
    h1["split_group"] = h1["video_id"]
    ss = S.ScreenSet(bench, h1)
    violent_signal = violent_signal or (lambda t, onset: np.zeros_like(t))
    h1_signal = h1_signal or (lambda t, onset: np.zeros_like(t))

    def loader(stream, vid, probe=False, scene=False):
        if scene:
            kind, _ = info[vid]
            r = np.random.default_rng(zlib.crc32(vid.encode()))
            return (r.normal(0, 1, 16) + (scene_shift if kind == "violent" else 0.0)).astype(np.float32)
        if stream != "tstream":
            if probe:
                return False
            raise FileNotFoundError(stream)
        if probe:
            return True
        kind, onset = info[vid]
        T = int(np.floor(bench.videos.loc[vid, "duration_sec"]))
        t = np.arange(1, T + 1, dtype=np.float32)
        r = np.random.default_rng(zlib.crc32(vid.encode()) + 1)
        x = r.normal(0, 1, (T, 4)).astype(np.float32)
        if all_signal is not None:
            x[:, 0] += all_signal(t)
        if kind == "violent":
            x[:, 0] += violent_signal(t, onset)
        elif kind == "h1":
            x[:, 0] += h1_signal(t, onset)
        return t, x, np.ones(T, bool), {"columns": ["a", "b", "c", "d"]}

    return ss, F.FeatureSource(ss, loader=loader)


BLOCK = F.Block("tstream", "A", ("tstream",))


def run_world(tmp_path, **kw):
    ss, source = build_world(**kw)
    args = argparse.Namespace(n_boot=200, m2=False, m6=False, require_cut=False)
    run_block = R.run_block
    res = run_block(ss, source, BLOCK, str(tmp_path), args)
    metrics = json.loads((tmp_path / "tstream" / "metrics.json").read_text())
    metrics["tail"] = (metrics.get("events") or {}).get("tail")
    return res, metrics, D.decide_phase({"tstream": metrics})["tstream"]


BAND = lambda t, onset: 2.5 * ((onset - t > 30) & (onset - t <= 120))  # noqa: E731


def test_planted_band_signal_passes_g1_and_g3(tmp_path):
    res, m, dec = run_world(tmp_path, violent_signal=BAND)
    assert m["m1"]["value"] > 0.8
    assert dec["G1"] == "PASS"
    assert dec["G3"] == "PASS"
    assert dec["G2"] in ("PASS", "NA", "INCONCLUSIVE")
    assert dec["class"] in ("ANTICIPATION", "PROVISIONAL")
    assert m["m1_shuffled"]["value"] == pytest.approx(0.5, abs=0.06)
    assert m["m1_time_source"]["value"] == pytest.approx(0.5, abs=0.06)
    assert m["events"]["m8"]["value"] > 0.8


def test_scene_prior_signal_has_no_gain_over_nuisance(tmp_path):
    allt = lambda t, onset: 2.5 * np.ones_like(t)  # noqa: E731
    res, m, dec = run_world(tmp_path, violent_signal=allt, scene_shift=3.0)
    assert m["m1"]["value"] > 0.8
    assert m["m1_n"]["value"] > 0.8
    assert dec["G1"] == "PASS"
    assert dec["G3"] == "FAIL"
    assert dec["class"] == "CONTEXT_SHORTCUT"


def test_pure_position_signal_is_not_a_signal(tmp_path):
    res, m, dec = run_world(tmp_path, all_signal=lambda t: 0.02 * t)
    assert dec["G1"] != "PASS"
    assert dec["G3"] != "PASS"
    assert m["m1"]["value"] < 0.6


def test_signal_before_people_focused_anomalies_fails_g2(tmp_path):
    res, m, dec = run_world(tmp_path, violent_signal=BAND, h1_signal=BAND)
    assert m["m1"]["value"] > 0.8
    assert dec["G1"] == "PASS"
    assert dec["G2"] == "FAIL"
    assert dec["class"] == "CONTEXT_SHORTCUT"


def test_cue_only_in_last_10_seconds_is_short_horizon(tmp_path):
    last = lambda t, onset: 3.0 * ((onset - t > 0) & (onset - t <= 10))  # noqa: E731
    res, m, dec = run_world(tmp_path, violent_signal=last)
    assert dec["G1"] != "PASS"
    assert m["m1_near"]["ci"][0] > 0.6
    assert dec["class"] == "SHORT_HORIZON"


def test_fold_one_is_unreachable():
    ss, source = build_world()
    with pytest.raises(S.FrozenFoldError):
        ss.test_ids(1)
    with pytest.raises(S.FrozenFoldError):
        ss.train_ids(1)
    with pytest.raises(S.FrozenFoldError):
        ss.ticks("synthetic/f1v0")
    with pytest.raises(S.FrozenFoldError):
        source.load("tstream", "synthetic/f1v0")
    with pytest.raises(S.FrozenFoldError):
        source.has("tstream", "synthetic/f1n0")
    with pytest.raises(S.FrozenFoldError):
        source.scene("synthetic/f1v0")
    runner = P.BlockRun(ss, source, BLOCK)
    with pytest.raises(S.FrozenFoldError):
        runner.raw("synthetic/f1v0")
    bad = pd.DataFrame({"video_id": ["synthetic/f1v0"], "t_sec": [1.0], "score": [0.1], "fold": [0]})
    with pytest.raises(S.FrozenFoldError):
        ss.assert_no_frozen(bad)
    bad = pd.DataFrame({"video_id": ["synthetic/f0v0"], "t_sec": [1.0], "score": [0.1], "fold": [1]})
    with pytest.raises(S.FrozenFoldError):
        ss.assert_no_frozen(bad)
    for f in ss.screen_folds:
        ids = ss.train_ids(f) + ss.val_ids(f) + ss.test_ids(f, kinds=("violent", "clean", "h1"))
        assert not any(ss.fold_of(v) == 1 for v in ids)


def test_set_definition_counts():
    ss, _ = build_world()
    c = ss.counts()
    assert c["positive_videos"] == 4 * N_VIOL
    assert c["first_segment_events"] == 4 * N_VIOL
    assert c["clean_negative_videos"] == 4 * N_CLEAN
    assert c["h1_videos"] == 4 * N_H1 and c["h1p_videos"] == 4 * N_H1
    assert len(ss.h1_ids()) == 4 * N_H1


def test_soft_masks_remove_ticks():
    ss, _ = build_world()
    vid = "synthetic/f0v0"
    n = (~ss.ticks(vid).masked).sum()
    ss2, _ = build_world()
    ss2.soft_masks = {vid: [[10, 20, "test"]]}
    assert (~ss2.ticks(vid).masked).sum() == n - 11


def brute_window(x, cut, W, thr):
    T = len(x)
    out = np.zeros((T, 3, x.shape[1]))
    last = -1
    for i in range(T):
        if cut[i]:
            last = i
        lo = min(max(i - W + 1, last + 1, 0), i)
        w = x[lo:i + 1]
        out[i, 0], out[i, 1], out[i, 2] = w.mean(0), w.max(0), (w > thr).mean(0)
    return out


def test_window_features_match_brute_force_and_reset_at_cuts():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(120, 3)).astype(np.float32)
    cut = np.zeros(120, bool)
    cut[[30, 31, 80]] = True
    thr = np.array([0.3, 0.0, -0.2], np.float32)
    f = F.window_features(x, F.last_cut_index(cut), thr, windows=(10, 60, 180))
    for k, W in enumerate((10, 60, 180)):
        b = brute_window(x, cut, W, thr)
        for j, stat in enumerate(("mean", "max", "frac")):
            got = f[:, j * 9 + k * 3: j * 9 + k * 3 + 3]
            assert np.allclose(got, b[:, j], atol=1e-5), (W, stat)


def test_slopes_match_polyfit():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(100, 2)).astype(np.float32)
    f = F.window_features(x, F.last_cut_index(np.zeros(100, bool)), np.zeros(2, np.float32), windows=(10,), slopes=(15,))
    i = 60
    want = np.polyfit(np.arange(i - 14, i + 1), x[i - 14:i + 1], 1)[0]
    assert np.allclose(f[i, -2:], want, atol=1e-4)
    assert (f[:2, -2:] == 0).all()


def test_early_baseline_is_causal_and_masks_the_first_ticks():
    x = np.random.default_rng(5).normal(size=(60, 2)).astype(np.float32)
    a = F.early_baseline_subtract(x)
    b = F.early_baseline_subtract(x[:30])
    assert np.allclose(a[:30], b)
    assert (a[:F.BASELINE_SEC] == 0).all()
    assert np.allclose(a[F.BASELINE_SEC:], x[F.BASELINE_SEC:] - x[:F.BASELINE_SEC].mean(0), atol=1e-6)
    assert (F.early_baseline_subtract(x[:F.BASELINE_SEC]) == 0).all()


def test_pls_finds_the_planted_direction():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(4000, 300))
    X[:, :50] += 5 * rng.normal(size=(4000, 1))
    y = (X[:, 200] + 0.5 * rng.normal(size=4000) > 0).astype(float)
    pls = F.Pls.fit(X, y, np.ones(4000), 4)
    z = pls.transform(X)
    assert z.shape == (4000, 4)
    assert int(np.argmax(np.abs(pls.rotation[:, 0]))) == 200
    assert P.weighted_auc(z[:, 0], y > 0, np.ones(4000)) > 0.75


def test_weighted_auc_and_holm():
    assert P.weighted_auc(np.array([1., 2, 3, 4]), np.array([0, 0, 1, 1], bool), np.ones(4)) == 1.0
    assert P.weighted_auc(np.ones(4), np.array([0, 0, 1, 1], bool), np.ones(4)) == 0.5
    adj = M.holm({"a": 0.01, "b": 0.02, "c": 0.04, "d": None})
    assert adj["a"] == pytest.approx(0.03) and adj["b"] == pytest.approx(0.04) and adj["c"] == pytest.approx(0.04)
    assert adj["d"] is None


def stat(v, lo, hi, p=None):
    return {"value": v, "ci": [lo, hi], "p_le_null": p}


def test_decision_rule_classes_and_tags():
    good = {"m1": stat(0.65, 0.6, 0.7, 0.001), "m4n": stat(0.05, 0.02, 0.08), "m3p": stat(0.6, 0.55, 0.65),
            "m3p_ratio": stat(0.7, 0.5, 0.9), "m1_near": stat(0.6, 0.55, 0.65)}
    assert D.decide_phase({"x": good})["x"]["class"] == "ANTICIPATION"
    assert D.decide_phase({"x": {**good, "m4n": stat(0.0, -0.02, 0.01)}})["x"]["class"] == "CONTEXT_SHORTCUT"
    assert D.decide_phase({"x": {**good, "m4n": stat(0.03, -0.01, 0.07)}})["x"]["class"] == "PROVISIONAL"
    assert D.decide_phase({"x": {**good, "m3p": stat(0.5, 0.45, 0.54), "m3p_ratio": stat(0.0, -0.5, 0.5)}})["x"]["class"] == "CONTEXT_SHORTCUT"
    drop = {"m1": stat(0.5, 0.48, 0.52, 0.5), "m1_near": stat(0.5, 0.4, 0.6)}
    assert D.decide_phase({"x": drop})["x"]["class"] == "DROP"
    wide = {"m1": stat(0.55, 0.45, 0.65, 0.2), "m1_near": stat(0.5, 0.4, 0.6)}
    assert D.decide_phase({"x": wide})["x"]["class"] == "INCONCLUSIVE"
    short = {"m1": stat(0.5, 0.48, 0.52, 0.5), "m1_near": stat(0.7, 0.65, 0.75)}
    assert D.decide_phase({"x": short})["x"]["class"] == "SHORT_HORIZON"
    tags = D.decide_phase({"x": {**good, "m1_long": stat(0.6, 0.55, 0.65), "m5_ramp": {"wilcoxon_p": 0.01, "share_rising": 0.7}}})["x"]["tags"]
    assert tags == ["LONG_LEAD", "RAMP"]
    sparse = {"m1": stat(0.52, 0.45, 0.6, 0.3), "m1_sparse_pauc": stat(0.6, 0.55, 0.65, 0.001), "m1_near": stat(0.5, 0.4, 0.6),
              "m4n": stat(0.05, 0.02, 0.08)}
    assert D.decide_phase({"x": sparse})["x"]["G1"] == "PASS"


def test_holm_changes_g1_across_a_phase():
    blocks = {f"b{i}": {"m1": stat(0.6, 0.5, 0.7, 0.02), "m1_near": stat(0.5, 0.4, 0.6)} for i in range(4)}
    out = D.decide_phase(blocks)
    assert all(v["G1"] == "INCONCLUSIVE" for v in out.values())
    assert D.decide_phase({"only": blocks["b0"]})["only"]["G1"] == "PASS"


def test_sparse_pauc_prefers_a_cue_in_the_top_tail():
    ss, _ = build_world()
    vts = [ss.ticks(v) for f in ss.screen_folds for v in ss.test_ids(f, kinds=("violent", "clean"))]
    pool = M.Pool(ss, vts)
    rng = np.random.default_rng(0)
    score = rng.normal(size=len(pool.t))
    pos = pool.band_pos(30, 120)
    d = pool.pair_def("x", pos, pool.clean_neg())
    W = pool.boot_weights(50)
    base = M.sparse_pauc(pool, score, d, W).value
    score[pos] += 6
    better = M.sparse_pauc(pool, score, d, W).value
    assert base == pytest.approx(0.5, abs=0.12) and better > 0.95


def stat_g2(m3p, ratio):
    return D.gate_g2(m3p, ratio)


def test_frozen_gate_variants():
    assert D.gate_g1("FAIL", "INCONCLUSIVE") == "FAIL"
    assert D.gate_g1("FAIL", "PASS") == "PASS"
    assert D.gate_g1("INCONCLUSIVE", "NA") == "INCONCLUSIVE"
    assert stat_g2(stat(0.6, 0.55, 0.65), None) == "INCONCLUSIVE"
    assert stat_g2(stat(0.5, 0.45, 0.54), None) == "FAIL"
    assert stat_g2(stat(0.5, 0.45, 0.54), stat(0.6, 0.1, 1.0)) == "FAIL"
    ratio = M.ratio_stat(M.Stat(0.6, np.array([0.6, 0.6])), M.Stat(0.5, np.array([0.5, 0.5])))
    assert ratio.value is None


def test_p0_directions_and_p1b_target():
    blk = F.BLOCKS["siglip2_prompts_v2"]
    assert F.p0_sign("siglip2_prompts_v2:phase_calm", blk) == -1.0
    assert F.p0_sign("siglip2_prompts_v2:walk_away", blk) == -1.0
    assert F.p0_sign("siglip2_prompts_v2:phase_violence", blk) == 1.0
    assert F.p0_sign("calm_street", F.BLOCKS["pose_interact"]) == 1.0
    ss, _ = build_world()
    vt = ss.ticks("synthetic/f0v0")
    pos, neg = P.target_masks(vt, "band_b")
    pos0, _ = P.target_masks(vt, "band")
    assert not pos[vt.t <= F.BASELINE_SEC].any() and (pos == (pos0 & (vt.t > F.BASELINE_SEC))).all()


def test_sparse_pauc_ties_give_chance_level():
    ss, _ = build_world()
    vts = [ss.ticks(v) for f in ss.screen_folds for v in ss.test_ids(f, kinds=("violent", "clean"))]
    pool = M.Pool(ss, vts)
    pos = pool.band_pos(30, 120)
    d = pool.pair_def("ties", pos, pool.clean_neg())
    W = pool.boot_weights(100)
    rng = np.random.default_rng(1)
    assert M.sparse_pauc(pool, np.zeros(len(pool.t)), d, W).value == pytest.approx(0.5, abs=1e-6)
    sparse_noise = np.where(rng.random(len(pool.t)) < 0.002, rng.normal(size=len(pool.t)), 0.0)
    st = M.sparse_pauc(pool, sparse_noise, d, W)
    assert abs(st.value - 0.5) < 0.1 and st.lower < 0.55
    planted = np.where(pos, 3.0, 0.0) + sparse_noise
    assert M.sparse_pauc(pool, planted, d, W).value > 0.9


def test_sparse_pauc_matches_sklearn_on_continuous_scores():
    from sklearn.metrics import roc_auc_score

    ss, _ = build_world()
    vts = [ss.ticks(v) for f in ss.screen_folds for v in ss.test_ids(f, kinds=("violent", "clean"))]
    pool = M.Pool(ss, vts)
    rng = np.random.default_rng(2)
    score = rng.normal(size=len(pool.t))
    pos = pool.band_pos(30, 120)
    score[pos] += 1.5
    pos2, neg, w = pool.pair_def("skl", pos, pool.clean_neg())
    m = pos2 | (neg & (w > 0))
    want = roc_auc_score(pos2[m], score[m], sample_weight=np.where(pos2, 1.0, w)[m], max_fpr=0.01)
    assert M.sparse_pauc(pool, score, (pos2, neg, w), np.zeros((0, pool.V))).value == pytest.approx(want, abs=2e-3)


def test_sparse_route_is_holm_adjusted_with_the_main_route():
    def blk(p):
        return {"m1": stat(0.52, 0.45, 0.6, 0.5), "m1_sparse_pauc": stat(0.6, 0.55, 0.65, p), "m1_near": stat(0.5, 0.4, 0.6)}

    one = D.decide_phase({"a": blk(0.01)})
    assert one["a"]["G1_sparse"] == "PASS"
    many = D.decide_phase({f"b{i}": blk(0.01) for i in range(9)})
    assert all(v["G1_sparse"] == "INCONCLUSIVE" for v in many.values())
    tail = {"m1": stat(0.52, 0.45, 0.6, 0.5), "tail": stat(0.2, 0.05, 0.3, 0.2), "m1_near": stat(0.5, 0.4, 0.6)}
    assert D.decide_phase({"t": tail})["t"]["G1_sparse"] == "INCONCLUSIVE"
    tail["tail"] = stat(0.2, 0.05, 0.3, 0.004)
    assert D.decide_phase({"t": tail})["t"]["G1_sparse"] == "PASS"


def test_g2_fails_on_a_low_upper_bound_even_without_a_ratio():
    assert D.gate_g2(stat(0.5, 0.45, 0.54), stat(None, None, None)) == "FAIL"
    assert D.gate_g2(stat(0.52, 0.48, 0.57), stat(None, None, None)) == "INCONCLUSIVE"
    assert D.gate_g2(stat(0.7, 0.6, 0.8), stat(None, None, None)) == "INCONCLUSIVE"
    assert D.classify("PASS", D.gate_g2(stat(0.5, 0.45, 0.54), stat(None, None, None)), "PASS", False) == "CONTEXT_SHORTCUT"


def test_exempt_g2_never_reaches_anticipation_and_is_listed():
    assert D.classify("PASS", "EXEMPT", "PASS", False) == "PROVISIONAL"
    assert "context_cue_G2_exemption" in R.NOT_IMPLEMENTED
    assert not any(b.g2_exempt for b in F.BLOCKS.values())


def test_gate_edge_cases():
    assert D.gate_g1_main({"value": 0.7, "ci": None}, 0.001) == "INCONCLUSIVE"
    assert D.gate_g3(stat(0.7 - 0.68, 0.01, 0.03)) == "PASS"
    assert D.gate_g3(stat(0.0, -0.05, 0.019)) == "FAIL"
    assert D.soundtrack_flag(0.66, 0.58) is False
    assert D.soundtrack_flag(0.67, 0.58) is True
    row = D.decide_block({"m1": stat(0.7, 0.6, 0.8, 0.001), "m4n": None, "soundtrack_flag": True}, 0.001)
    assert row["unevaluated_gates"] == ["G2", "G3"] and row["soundtrack_flag"] is True and row["class"] == "PROVISIONAL"


def test_first_frame_broadcast_is_reported_and_flagged():
    base = {"m1": stat(0.7, 0.6, 0.8, 0.001), "m1_near": stat(0.5, 0.4, 0.6)}
    shortcut = {**base, "m4b_first_frame": stat(1.0, 0.98, 1.0), "m1_ramp_gain_over_m4b": stat(0.0, -0.01, 0.01)}
    honest = {**base, "m4b_first_frame": stat(0.5, 0.45, 0.55), "m1_ramp_gain_over_m4b": stat(0.2, 0.1, 0.3)}
    assert D.decide_phase({"x": shortcut})["x"]["first_frame_flag"] is True
    row = D.decide_phase({"x": honest})["x"]
    assert row["first_frame_flag"] is False and row["m4b"] == 0.5


def fake_metrics(root, name, p, sha="SHA", events_first_tail=None, events_tail=None):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    m = {"m1": stat(0.62, 0.5, 0.7, p), "m1_near": stat(0.5, 0.4, 0.6), "config": {"prereg_sha256": sha},
         "events": {"tail": events_tail}, "events_first": {"tail": events_first_tail}}
    (d / "metrics.json").write_text(json.dumps(m))


def test_decide_refuses_incomplete_or_stale_families(tmp_path):
    names = ["pose_interact", "siglip2", "vjepa2_l"]
    fake_metrics(tmp_path, "pose_interact", 0.02)
    fake_metrics(tmp_path, "siglip2", 0.03)
    with pytest.raises(R.FamilyError):
        R.decide(str(tmp_path), "A", None, family=names)
    with pytest.raises(R.FamilyError):
        R.decide(str(tmp_path), "A", None)
    pr = PR.load()
    fake_metrics(tmp_path, "pose_interact", 0.02, sha="STALE")
    with pytest.raises(R.FamilyError):
        R.decide(str(tmp_path), "A", pr)
    assert set(R.family_names("A", pr)) == {n for n, b in F.BLOCKS.items() if b.phase == "A"}
    assert R.family_names("B", pr) == []


def test_decide_uses_first_segment_tail_and_records_the_family(tmp_path):
    for n in ("pose_interact", "siglip2"):
        fake_metrics(tmp_path, n, 0.5, events_first_tail=stat(0.0, -0.1, 0.1, 0.6), events_tail=stat(0.3, 0.1, 0.4, 0.001))
    names = ["pose_interact", "siglip2"]
    out = R.decide(str(tmp_path), "A", None, family=names)
    assert out["pose_interact"]["G1_sparse"] == "INCONCLUSIVE"
    rec = json.loads((tmp_path / "decisions_phase_A.json").read_text())
    assert rec["holm_family_blocks"] == sorted(names) and rec["family_status"] == "COMPLETE"
    part = R.decide(str(tmp_path), "A", None, partial=True)
    assert json.loads((tmp_path / "decisions_phase_A.json").read_text())["family_status"] == "PARTIAL_FAMILY"
    assert part


def test_tail_threshold_comes_from_the_clean_validation_slice():
    ss, _ = build_world()
    pool = M.Pool(ss, [ss.ticks(v) for v in ss.test_ids(0, kinds=("violent", "clean"))])
    val = {f: np.arange(1001.0) for f in ss.screen_folds}
    assert set(M.tail_thresholds(pool, val)) == {995.0}
    assert "VALIDATION" in R.INTERPRETATIONS["tail_threshold_source"]


def test_m8_window_stops_before_tte_30_and_includes_tte_120():
    ss, _ = build_world(integer_onsets=True)
    pool = M.Pool(ss, [ss.ticks(v) for f in ss.screen_folds for v in ss.test_ids(f, kinds=("violent", "clean"))])
    assert len(pool.events)
    for e in range(len(pool.events)):
        sl = pool.sl(pool.ev_vi[e])
        tte = pool.tte[sl]
        assert tte[pool.ev_ib[e]] > 30 and tte[pool.ev_ib[e] + 1] <= 30
        assert tte[pool.ev_ia[e]] <= 120 + 1e-9 and tte[pool.ev_ia[e] - 1] > 120


def test_p1b_is_paired_with_p1_on_the_same_ticks(tmp_path):
    res, metrics, _ = run_world(tmp_path, violent_signal=BAND)
    assert "m1_p1_late" in metrics and "p1b_gain_over_p1" in metrics
    assert metrics["p1b_gain_over_p1"]["value"] == pytest.approx(metrics["m1_p1b"]["value"] - metrics["m1_p1_late"]["value"])
    assert metrics["m1_p1_late"]["n_pos"] == metrics["m1_p1b"]["n_pos"]


def test_p1b_windows_divide_by_post_baseline_history_only():
    ss, source = build_world()
    run = P.BlockRun(ss, source, BLOCK, require_cut=False)
    train = ss.train_ids(0)
    run.set_fold(train)
    v = train[0]
    x = F.early_baseline_subtract(run.xform(v))
    got = run.feats(v, "p1b")
    for i in (10, 11, 25, 69):
        want = x[F.BASELINE_SEC:i + 1][-60:].mean(0)
        assert np.allclose(got[i, 4:8], want, atol=1e-4), i


def test_missing_or_short_cut_stream_fails_the_block():
    ss, source = build_world()
    strict = P.BlockRun(ss, source, BLOCK, require_cut=True)
    with pytest.raises(P.CutStreamError):
        strict.cut_last(ss.train_ids(0)[0])
    lax = P.BlockRun(ss, source, BLOCK, require_cut=False)
    lax.cut_last(ss.train_ids(0)[0])
    assert lax.cut_incomplete == [ss.train_ids(0)[0]]
    inner = source.loader

    def short(stream, vid, **kw):
        if stream == F.CUT_STREAM and not kw:
            t, x, v, meta = inner("tstream", vid)
            return t[:5], x[:5], v[:5], {"columns": ["cut_1s", "b", "c", "d"]}
        return inner(stream, vid, **kw)

    source.loader = short
    with pytest.raises(P.CutStreamError):
        P.BlockRun(ss, source, BLOCK, require_cut=True).cut_last(ss.train_ids(0)[0])


def test_cut_tick_window_keeps_the_current_value():
    x = np.full((40, 1), 5.0, np.float32)
    cut = np.zeros(40, bool)
    cut[25] = True
    f = F.window_features(x, F.last_cut_index(cut), np.zeros(1, np.float32), windows=(10,))
    assert f[25, 0] == 5.0 and f[26, 0] == 5.0


def test_tick_one_carries_no_scene_information():
    ss, source = build_world()
    nuis = P.Nuisance(source, ss.train_ids(0))
    a, b = ss.ticks("synthetic/f0v0"), ss.ticks("synthetic/f0v1")
    fa, fb = nuis.features(a, "ucf"), nuis.features(b, "ucf")
    k = nuis.k
    assert (fa[0, -1 - k:-1] == 0).all() and (fb[0, -1 - k:-1] == 0).all()
    assert not np.allclose(fa[1, -1 - k:-1], fb[1, -1 - k:-1])


REAL_PREREG = PR.PREREG_PATH


def synthetic_prereg_data(ss):
    """A pre-registration dictionary for the synthetic world, built from the in-code rules."""
    ev = ss.events()
    row = lambda r: {"video_id": r.video_id, "source_kind": r.source_kind, "onset_source": r.onset_source,  # noqa: E731
                     "index": int(r.index), "onset": float(r.onset), "clear_start": float(r.clear_start),
                     "leadin": float(r.leadin), "max_lead": float(r.max_lead), "eligible60": bool(r.eligible60)}
    ev_rows = list(ev.itertuples())
    lists = ss.video_lists()
    sets = {"positive_videos": lists["positive_videos"], "eligible30_events": [row(r) for r in ev_rows],
            "first_events": [row(r) for r in ev_rows if r.first], "clean_cpu_videos": lists["clean_negative_videos"],
            "clean_vlm_videos": lists["clean_negative_videos"][:10], "h1_videos": lists["h1_videos"],
            "h1p_videos": lists["h1p_videos"]}
    for name, col in (("real_first_events", "s_real"), ("long60_first_events", "s_long60"), ("ramp90_first_events", "s_ramp"),
                      ("official_ubi_first_events", "s_offubi"), ("audio_first_event_candidates", "s_audio")):
        sets[name] = [row(r) for r in ev_rows if getattr(r, col)]
    allv = sets["positive_videos"] + sets["clean_cpu_videos"] + sets["h1_videos"]
    membership = {}
    for f in ss.screen_folds:
        member = lambda ids: sorted(v for v in ids if ss.fold_of(v) != 1 and v in set(allv))  # noqa: E731
        membership[str(f)] = {"train": member(ss.bench.folds.train_ids(f)), "val": member(ss.bench.folds.val_ids(f)),
                              "test": member(ss.bench.folds.test_ids(f))}
    return {"version": 1, "status": "PREREGISTERED_BEFORE_NEW_SCREEN_RESULTS", "screen_folds": list(ss.screen_folds),
            "confirmation_fold": 1, "sets": sets,
            "scored_intervals_sec": {v: [1.0, float(ss.ticks_computed(v))] for v in allv},
            "audio_gate": "Phase B blocked until audit", "label_policy": "frozen", "blocks": {"A": ["pose_interact"]},
            "controls": [], "probes": {}, "nuisance": {}, "metrics": {}, "gates": {}, "classes": {}, "tags": {},
            "forward_selection": "", "provenance": {}, "fold_membership": membership}


def write_prereg(tmp_path, data, name="synthetic_set.json", sidecar=True):
    import hashlib

    path = tmp_path / name
    text = json.dumps(data, indent=1) + "\n"
    path.write_text(text)
    if sidecar:
        (tmp_path / (name[:-5] + ".sha256")).write_text(f"{hashlib.sha256(text.encode()).hexdigest()}  {name}\n")
    return str(path)


def frozen_world(tmp_path, mutate=None, **kw):
    ss, source = build_world(**kw)
    data = synthetic_prereg_data(ss)
    if mutate:
        mutate(data)
    pr = PR.load(write_prereg(tmp_path, data), pinned={})
    ss_f = S.ScreenSet(ss.bench, ss.h1.reset_index(drop=True), prereg=pr)
    return ss_f, source, ss, pr


def test_frozen_set_matches_computation_on_the_synthetic_world(tmp_path):
    ss_f, _, ss, pr = frozen_world(tmp_path)
    assert ss_f.counts() == ss.counts()
    ag = PR.agreement(pr, ss_f, ss, ss.bench)
    assert [c["name"] for c in ag["disagreements"]] == []


def test_frozen_ids_replace_computed_ids_and_disagreement_is_reported(tmp_path):
    drop_clean = "synthetic/f0n0"
    drop_h1 = "synthetic/f2h0"

    def mutate(d):
        d["sets"]["clean_cpu_videos"].remove(drop_clean)
        d["sets"]["clean_vlm_videos"] = [v for v in d["sets"]["clean_vlm_videos"] if v != drop_clean]
        d["sets"]["h1_videos"].remove(drop_h1)
        d["sets"]["h1p_videos"].remove(drop_h1)
        for f in d["fold_membership"].values():
            for k in f:
                f[k] = [v for v in f[k] if v not in (drop_clean, drop_h1)]
        d["scored_intervals_sec"]["synthetic/f0n1"] = [1.0, 50.0]

    ss_f, _, ss, pr = frozen_world(tmp_path, mutate)
    assert drop_clean not in ss_f.test_ids(0) and drop_clean in ss.test_ids(0)
    assert drop_h1 not in ss_f.h1_ids() and drop_h1 in ss.h1_ids()
    assert ss_f.counts()["clean_negative_videos"] == ss.counts()["clean_negative_videos"] - 1
    assert ss_f.ticks("synthetic/f0n1").n == 50
    ag = PR.agreement(pr, ss_f, ss, ss.bench)
    bad = {c["name"]: c for c in ag["disagreements"]}
    assert bad["clean_cpu_videos"]["only_computed"] == [drop_clean]
    assert bad["h1_videos"]["only_computed"] == [drop_h1]
    assert bad["scored_ticks_per_video"]["n_differ"] == 1


def test_tampered_preregistration_is_refused(tmp_path):
    ss, _ = build_world()
    data = synthetic_prereg_data(ss)
    good = write_prereg(tmp_path, data)
    assert PR.load(good, pinned={}).sha256
    with pytest.raises(PR.PreregError, match="registered value"):
        PR.load(good, pinned={"synthetic_set.json": "0" * 64})
    path = tmp_path / "synthetic_set.json"
    path.write_text(path.read_text().replace('"confirmation_fold": 1', '"confirmation_fold": 2'))
    with pytest.raises(PR.PreregError, match="sha256 mismatch"):
        PR.load(good, pinned={})
    data2 = synthetic_prereg_data(ss)
    with pytest.raises(PR.PreregError, match="missing checksum"):
        PR.load(write_prereg(tmp_path, data2, "other_set.json", sidecar=False), pinned={})


def test_tampered_sidecar_and_pinned_hash(tmp_path):
    ss, _ = build_world()
    data = synthetic_prereg_data(ss)
    path = write_prereg(tmp_path, data)
    (tmp_path / "synthetic_set.sha256").write_text("1" * 64 + "  synthetic_set.json\n")
    with pytest.raises(PR.PreregError, match="sha256 mismatch"):
        PR.load(path, pinned={})
    if not __import__("os").path.isfile(REAL_PREREG):
        pytest.skip("committed pre-registration absent")
    import shutil

    for src in (REAL_PREREG, REAL_PREREG[:-5] + ".sha256"):
        shutil.copy(src, tmp_path / __import__("os").path.basename(src))
    real = tmp_path / "screen_set_v1.json"
    assert PR.load(str(real)).sha256 == PR.PINNED_SHA256["screen_set_v1.json"]
    txt = real.read_text()
    forged = txt.replace('"confirmation_fold": 1', '"confirmation_fold": 3', 1)
    assert forged != txt
    real.write_text(forged)
    import hashlib

    (tmp_path / "screen_set_v1.sha256").write_text(f"{hashlib.sha256(forged.encode()).hexdigest()}  screen_set_v1.json\n")
    with pytest.raises(PR.PreregError, match="registered value"):
        PR.load(str(real))


def test_run_refuses_a_tampered_preregistration_before_doing_anything(tmp_path):
    if not __import__("os").path.isfile(REAL_PREREG):
        pytest.skip("committed pre-registration absent")
    import shutil

    for src in (REAL_PREREG, REAL_PREREG[:-5] + ".sha256"):
        shutil.copy(src, tmp_path / __import__("os").path.basename(src))
    real = tmp_path / "screen_set_v1.json"
    real.write_text(real.read_text().replace("0.58", "0.50", 1))
    with pytest.raises(PR.PreregError):
        R.main(["--dry_run", "--prereg", str(real), "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_committed_preregistration_agrees_with_code_constants():
    if not __import__("os").path.isfile(REAL_PREREG):
        pytest.skip("committed pre-registration absent")
    pr = PR.load()
    assert R.check_frozen_constants(pr) == []
    status = {r["name"]: r["status"] for r in F.resolve_blocks(pr)}
    named = [n for names in pr.block_phases().values() for n in names]
    assert sorted(status) == sorted(named) and len(named) == 23
    assert status["camera_wearer_diagnostic"] == F.NOT_AVAILABLE
    assert status["dialogue_acts"] == F.NOT_AVAILABLE and status["vlm_forecast"] == F.NOT_AVAILABLE
    assert status["audio_dsp"] == F.BLOCKED_AUDIO_GATE
    assert all(status[n] == F.AVAILABLE for n in pr.block_phases()["A"] if n != "camera_wearer_diagnostic")
    assert set(F.BLOCKS) <= set(named)
    assert pr.screen_folds == (0, 2, 3, 4) and pr.confirmation_fold == 1
    assert len(pr.ids("positive_videos")) == 379 and len(pr.ids("clean_cpu_videos")) == 3220


def test_tampered_constants_are_detected():
    if not __import__("os").path.isfile(REAL_PREREG):
        pytest.skip("committed pre-registration absent")
    pr = PR.load()
    pr.data["gates"]["G1"]["PASS"] = "point>=0.60 and adjusted one-sided CI lower>0.50"
    assert any("G1_POINT" in b for b in R.check_frozen_constants(pr))


def test_frozen_run_never_touches_fold_one(tmp_path):
    ss_f, source, ss, pr = frozen_world(tmp_path, violent_signal=BAND)
    seen, label_calls = set(), set()
    inner = source.loader

    def spy(stream, vid, **kw):
        seen.add(vid)
        return inner(stream, vid, **kw)

    source.loader = spy
    source.ss = ss_f
    orig_labels = ss_f.bench.labels
    ss_f.bench.labels = lambda vid: (label_calls.add(vid), orig_labels(vid))[1]
    args = argparse.Namespace(n_boot=100, m2=False, m6=False, require_cut=False)
    res = R.run_block(ss_f, source, BLOCK, str(tmp_path / "out"), args, pr.sha256)
    ss_f.bench.labels = orig_labels
    listed = set(pr.all_listed_videos())
    assert seen and label_calls
    assert not any(ss_f.fold_of(v) == 1 for v in seen | label_calls)
    assert seen <= listed and label_calls <= listed
    assert res["m1"].value > 0.8
    scores = pd.read_parquet(tmp_path / "out" / "tstream" / "scores.parquet")
    assert not (scores["video_id"].map(ss_f.bench.folds.assignment) == 1).any() and not (scores["fold"] == 1).any()
    for f in ss_f.screen_folds:
        ids = ss_f.train_ids(f) + ss_f.val_ids(f) + ss_f.test_ids(f, kinds=("violent", "clean", "h1"))
        assert set(ids) <= listed and not any(ss_f.fold_of(v) == 1 for v in ids)
        assert not any(ss_f.kind(v) == "h1" for v in ss_f.train_ids(f) + ss_f.val_ids(f))
    with pytest.raises(S.FrozenFoldError):
        ss_f.test_ids(1)
    with pytest.raises(S.FrozenFoldError):
        ss_f.ticks("synthetic/f1v0")
    with pytest.raises(PR.PreregError):
        pr.membership(1, "test")


def test_membership_with_a_fold_one_video_is_fatal(tmp_path):
    def mutate(d):
        d["fold_membership"]["0"]["train"].append("synthetic/f1v0")

    ss_f, _, ss, pr = frozen_world(tmp_path, mutate)
    fatal, _ = PR.verify_inputs(pr, ss.bench)
    assert any("confirmation fold" in x for x in fatal)


def test_frozen_file_may_not_list_the_confirmation_fold(tmp_path):
    ss, _ = build_world()
    data = synthetic_prereg_data(ss)
    data["fold_membership"]["1"] = {"train": [], "val": [], "test": []}
    with pytest.raises(PR.PreregError, match="confirmation fold"):
        PR.load(write_prereg(tmp_path, data), pinned={})


def sens_masks_for(ss, frac_whole=0.5):
    """Mask the whole 30-120 s window of the first ``frac_whole`` of the violent videos (reason S1-kept) and a tail piece
    of one more, plus a weapon-only entry on another."""
    vids = sorted(v for v in ss.video_lists()["positive_videos"])
    out = {}
    for v in vids[: int(len(vids) * frac_whole)]:
        onset = float(ss.events().set_index("video_id").loc[v, "onset"])
        out[v] = [[onset - 125.0, onset - 25.0, "uca_strike"]]
    w = vids[-1]
    out[w] = [[0.0, 400.0, "uca_weapon_only"]]
    return out


def test_sensitivity_masks_load_variants_and_verify_sidecar(tmp_path):
    m1, i1 = S.load_sensitivity_masks("S1")
    m2, i2 = S.load_sensitivity_masks("S2")
    assert i2["entries"] == i2["entries_in_file"] and i1["entries"] < i2["entries"]
    assert all(e[2] != "uca_weapon_only" for es in m1.values() for e in es)
    assert any(e[2] == "uca_weapon_only" for es in m2.values() for e in es)
    assert i1["sha256"] == i2["sha256"] and i1["variant"] == "S1"
    with pytest.raises(S.SoftMaskError):
        S.load_sensitivity_masks("S3")
    import shutil

    shutil.copy(S.SOFT_MASKS_PATH, tmp_path / "m.json")
    (tmp_path / "m.json.sha256").write_text("0" * 64 + "\n")
    with pytest.raises(S.SoftMaskError, match="differs"):
        S.load_sensitivity_masks("S1", str(tmp_path / "m.json"))
    (tmp_path / "m.json.sha256").unlink()
    with pytest.raises(S.SoftMaskError, match="missing"):
        S.load_sensitivity_masks("S1", str(tmp_path / "m.json"))


def test_sensitivity_off_leaves_primary_output_unchanged(tmp_path):
    args = argparse.Namespace(n_boot=100, m2=False, m6=False, require_cut=False)
    ss, source = build_world(violent_signal=BAND)
    R.run_block(ss, source, BLOCK, str(tmp_path / "a"), args)
    ss2, source2 = build_world(violent_signal=BAND)
    R.run_block(ss2, source2, BLOCK, str(tmp_path / "b"), args)
    a = json.loads((tmp_path / "a" / "tstream" / "metrics.json").read_text())
    b = json.loads((tmp_path / "b" / "tstream" / "metrics.json").read_text())
    assert "sensitivity" not in a and "sensitivity" not in a["config"]
    for d in (a, b):
        d["config"].pop("seconds")
    assert a == b
    ss3, source3 = build_world(violent_signal=BAND)
    ss3.apply_sensitivity({"synthetic/not_a_video": [[0, 10, "uca_strike"]]}, {"applied": True, "variant": "S2"})
    R.run_block(ss3, source3, BLOCK, str(tmp_path / "c"), args)
    c = json.loads((tmp_path / "c" / "tstream" / "metrics.json").read_text())
    assert c["m1"] == a["m1"] and c["sensitivity"]["events_losing_whole_window"] == 0


def test_sensitivity_counts_lost_windows_and_report(tmp_path, monkeypatch):
    from types import SimpleNamespace

    ss, source = build_world(violent_signal=BAND)
    masks = sens_masks_for(ss)
    pr = SimpleNamespace(sha256="x")
    monkeypatch.setattr(R.F, "resolve_blocks", lambda p: [{"name": "tstream", "phase": "A", "status": F.AVAILABLE, "reason": ""}])
    args = argparse.Namespace(n_boot=100, m2=False, m6=False, require_cut=False)
    R.run_block(ss, source, BLOCK, str(tmp_path), args, "x")
    n_events = len(ss.events())
    ss_s, source_s = build_world(violent_signal=BAND)
    ss_s.apply_sensitivity(masks, {"applied": True, "variant": "S2", "path": "x", "sha256": "y", "videos": len(masks)})
    sens_root = tmp_path / "sensitivity_S2"
    R.run_block(ss_s, source_s, BLOCK, str(sens_root), args, "x")
    m = json.loads((sens_root / "tstream" / "metrics.json").read_text())
    sv = m["sensitivity"]
    n_masked_videos = len(masks)
    assert sv["variant"] == "S2" and sv["events"] == n_events
    assert sv["events_losing_whole_window"] == n_masked_videos
    assert sv["events_with_window_ticks"] == n_events
    assert sv["window_ticks_after"] < sv["window_ticks_before"]
    assert m["config"]["sensitivity"] == "S2" and m["config"]["soft_masks"]["variant"] == "S2"
    rep = R.sensitivity_report(str(sens_root), str(tmp_path), ["tstream"], "S2", pr)
    row = rep["blocks"]["tstream"]
    p = json.loads((tmp_path / "tstream" / "metrics.json").read_text())["m1"]["value"]
    assert row["primary_m1"]["value"] == p and abs(row["m1_minus_primary_m1"] - (row["m1"]["value"] - p)) < 1e-12
    assert rep["used_for_decision"] is False and (sens_root / "sensitivity_report.md").is_file()
    assert row["events"]["events_losing_whole_window"] == n_masked_videos
    assert row["g1"]["primary_G1"] and row["g1"]["sensitivity_G1"]
    assert row["g1"]["signal_explained_by_onset_label_noise"] is False

    with pytest.raises(R.FamilyError, match="another pre-registration"):
        R.decide(str(sens_root), "A", pr, family=["tstream"])


def test_sensitivity_option_is_refused_with_decide(tmp_path):
    with pytest.raises(SystemExit, match="sensitivity"):
        R.main(["--sensitivity", "S1", "--decide", "A", "--out", str(tmp_path)])


def test_folds_without_confirmation_drops_fold_one(tmp_path):
    from anticipation.screen import run as R
    path, fmap = R.folds_without_confirmation(str(tmp_path))
    from anticipation.data.folds import Folds
    fo = Folds.load(path)
    assert fmap == {0: 0, 2: 1, 3: 2, 4: 3}
    assert fo.n_folds == 4 and set(fo.assignment.values()) == {0, 1, 2, 3}
    assert all(len(fo.val_ids(f)) > 0 for f in range(4))
