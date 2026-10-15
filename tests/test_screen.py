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
from anticipation.screen import probes as P
from anticipation.screen import run as R
from anticipation.screen import sets as S

N_VIOL, N_CLEAN, N_H1 = 8, 20, 6
VAL_FROM = {0: 2, 1: 3, 2: 3, 3: 4, 4: 0}


@pytest.fixture(autouse=True)
def thread_cap():
    with threadpool_limits(limits=2):
        yield


def build_world(violent_signal=None, h1_signal=None, scene_shift=0.0, seed=0, all_signal=None):
    """Synthetic benchmark + feature loader. ``*_signal(t, onset)`` returns the extra level added to column 0."""
    rng = np.random.default_rng(seed)
    rows, events, assign, info = [], {}, {}, {}
    for f in range(5):
        for i in range(N_VIOL):
            vid = f"synthetic/f{f}v{i}"
            onset = float(rng.uniform(150, 230))
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
    args = argparse.Namespace(n_boot=200, m2=False, m6=False)
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
    assert dec["class"] == "CONTEXT/SHORTCUT"


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
    assert dec["class"] == "CONTEXT/SHORTCUT"


def test_cue_only_in_last_10_seconds_is_short_horizon(tmp_path):
    last = lambda t, onset: 3.0 * ((onset - t > 0) & (onset - t <= 10))  # noqa: E731
    res, m, dec = run_world(tmp_path, violent_signal=last)
    assert dec["G1"] != "PASS"
    assert m["m1_near"]["ci"][0] > 0.6
    assert dec["class"] == "SHORT-HORIZON"


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
        lo = max(i - W + 1, last + 1, 0)
        if lo > i:
            continue
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


def test_early_baseline_is_causal():
    x = np.random.default_rng(5).normal(size=(60, 2)).astype(np.float32)
    a = F.early_baseline_subtract(x)
    b = F.early_baseline_subtract(x[:30])
    assert np.allclose(a[:30], b)


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
    assert D.decide_phase({"x": {**good, "m4n": stat(0.0, -0.02, 0.01)}})["x"]["class"] == "CONTEXT/SHORTCUT"
    assert D.decide_phase({"x": {**good, "m4n": stat(0.03, -0.01, 0.07)}})["x"]["class"] == "PROVISIONAL"
    assert D.decide_phase({"x": {**good, "m3p": stat(0.5, 0.45, 0.54), "m3p_ratio": stat(0.0, -0.5, 0.5)}})["x"]["class"] == "CONTEXT/SHORTCUT"
    drop = {"m1": stat(0.5, 0.48, 0.52, 0.5), "m1_near": stat(0.5, 0.4, 0.6)}
    assert D.decide_phase({"x": drop})["x"]["class"] == "DROP"
    wide = {"m1": stat(0.55, 0.45, 0.65, 0.2), "m1_near": stat(0.5, 0.4, 0.6)}
    assert D.decide_phase({"x": wide})["x"]["class"] == "INCONCLUSIVE"
    short = {"m1": stat(0.5, 0.48, 0.52, 0.5), "m1_near": stat(0.7, 0.65, 0.75)}
    assert D.decide_phase({"x": short})["x"]["class"] == "SHORT-HORIZON"
    tags = D.decide_phase({"x": {**good, "m1_long": stat(0.6, 0.55, 0.65), "m5_ramp": {"wilcoxon_p": 0.01, "share_rising": 0.7}}})["x"]["tags"]
    assert tags == ["LONG-LEAD", "RAMP"]
    sparse = {"m1": stat(0.52, 0.45, 0.6, 0.3), "m1_sparse_pauc": stat(0.6, 0.55, 0.65), "m1_near": stat(0.5, 0.4, 0.6),
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


def test_prereg_detects_a_changed_set(tmp_path):
    from anticipation.screen import prereg

    ss, _ = build_world()
    path = str(tmp_path / "screen_set_v1.json")
    sha = prereg.write(ss, path)
    assert len(sha) == 64 and (tmp_path / "screen_set_v1.json.sha256").read_text().startswith(sha)
    assert prereg.verify(ss, path) == []
    ss2, _ = build_world(seed=1)
    ss2.h1 = ss2.h1.iloc[:-3]
    assert "h1_videos" in prereg.verify(ss2, path)
