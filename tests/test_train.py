"""Synthetic learning, leakage, censoring and VA30 integration checks."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from threadpoolctl import threadpool_limits

from anticipation.data.folds import Folds, content_hash
from anticipation.data.labels import tick_labels
from anticipation.eval.va30 import Benchmark, EvalConfig, EventEval, TickSet, choose_threshold, evaluate
from anticipation.models.data import FeatureStore, Normalizer, VideoFeatures, join_streams, sample_ticks
from anticipation.models.stats import StatsModel, make_table, window_statistics
from anticipation.models.temporal import (CausalTemporalModel, WindowDataset, context_window, fit_temporal,
                                          hazard_loss, predict_temporal)


@pytest.fixture(autouse=True)
def thread_cap():
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        yield


def synthetic_benchmark(root, precursor=True):
    rng = np.random.default_rng(7)
    rows, events, assignment, features = [], {}, {}, {}
    for fold in range(5):
        for i in range(4):
            vid = f"synthetic/f{fold}v{i}"
            violent = i < 2
            onset = 110 + 9 * fold + 11 * i
            rows.append({"video_id": vid, "duration_sec": 220.0, "role": "violent" if violent else "clean_negative",
                         "onset_source": "official" if violent else "none", "source_kind": "ucf",
                         "interpersonal": True, "has_audio": False, "group": vid})
            events[vid] = {"segments": [[onset, onset + 15]] if violent else []}
            assignment[vid] = fold
            t = np.arange(1, 221, dtype=np.float32)
            x = rng.normal(0, 0.05, (len(t), 4)).astype(np.float32) if precursor else np.zeros((len(t), 4), np.float32)
            if violent:
                signal = (t >= onset - 40) & (t <= onset + 15) if precursor else (t >= onset)
                x[signal, 0] += 4
                x[signal, 1] -= 4
            features[vid] = t, x, np.ones(len(t), bool), {"extractor": "synthetic", "version": 1, "D": 4}
    val = {str(f): [f"synthetic/f{(f + 1) % 5}v0", f"synthetic/f{(f + 1) % 5}v2"] for f in range(5)}
    fold_data = {"name": "synthetic", "n_folds": 5, "assignment": assignment, "val": val,
                 "content_sha256": content_hash(assignment, val)}
    catalog = pd.DataFrame(rows).set_index("video_id", drop=False)
    bench = Benchmark(catalog, events, Folds(fold_data))
    store = FeatureStore(root / "cache", catalog, events, ["synthetic"])
    store.snapshot(bench.eval_ids(), loader=lambda stream, vid: features[vid])
    return bench, store, features, fold_data


def fit_fold(bench, store, kind):
    train, val = bench.folds.train_ids(0), bench.folds.val_ids(0)
    norm = Normalizer.fit(store, train)
    task = "stats" if kind.startswith("stats") else "temporal"
    sel = [sample_ticks(store, ids, 1600, np.random.default_rng(0), task) for ids in (train, val)]
    if task == "stats":
        model = StatsModel(kind)
        model.fit(make_table(store, train, sel[0], norm), make_table(store, val, sel[1], norm), epochs=8, patience=3)
        predict = lambda v: model.predict(v, norm)
    else:
        torch.manual_seed(1)
        model = CausalTemporalModel([4], kind, hidden=16, layers=1, heads=2, context=60, dropout=0,
                                    modality_dropout=0)
        datasets = [WindowDataset(store, s, norm, 60) for s in sel]
        fit_temporal(model, *datasets, epochs=8, patience=3, batch_size=64, lr=0.005)
        predict = lambda v: predict_temporal(model, v, norm, batch_size=64)
    return model, norm, predict


def success_for_fold(bench, store, predict):
    val = bench.folds.val_ids(0)
    test = bench.folds.test_ids(0)
    vts = TickSet(bench, val, {v: predict(store[v]) for v in val})
    theta, _ = choose_threshold(EventEval(vts, EvalConfig().policy), 2)
    ts = TickSet(bench, test, {v: predict(store[v]) for v in test})
    ee = EventEval(ts, EvalConfig().policy)
    return ee.summarize(ee.outcomes(np.full(len(test), theta)))["eligible30"]["success30"]["value"]


@pytest.mark.parametrize("kind", ["stats_logreg", "stats_lgbm", "gru", "transformer"])
def test_planted_precursor_learns_success30(tmp_path, kind):
    bench, store, _, _ = synthetic_benchmark(tmp_path)
    _, _, predict = fit_fold(bench, store, kind)
    assert success_for_fold(bench, store, predict) >= 0.9


@pytest.mark.parametrize("kind", ["stats_logreg", "stats_lgbm", "gru", "transformer"])
def test_future_truncation_preserves_earlier_scores(tmp_path, kind):
    bench, store, _, _ = synthetic_benchmark(tmp_path)
    model, norm, predict = fit_fold(bench, store, kind)
    full = store[bench.folds.test_ids(0)[0]]
    cut = 105
    truncated = VideoFeatures(full.video_id, full.t_sec[:cut], [x[:cut] for x in full.streams],
                              full.present[:cut], None)
    np.testing.assert_allclose(predict(full)[:cut], predict(truncated), atol=2e-6, rtol=2e-6)
    changed = VideoFeatures(full.video_id, full.t_sec, [x.copy() for x in full.streams], full.present.copy(), None)
    changed.streams[0][cut:] = 1000
    changed.present[cut:] = False
    np.testing.assert_allclose(predict(full)[:cut], predict(changed)[:cut], atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("kind", ["stats_logreg", "stats_lgbm", "gru", "transformer"])
def test_post_onset_signal_has_no_anticipation_credit(tmp_path, kind):
    bench, store, _, _ = synthetic_benchmark(tmp_path, precursor=False)
    _, _, predict = fit_fold(bench, store, kind)
    assert success_for_fold(bench, store, predict) == 0


def test_timestamp_join_masks_and_training_only_normalizer(tmp_path):
    row = {"duration_sec": 6, "role": "clean_negative"}
    arrays = {"a": (np.array([1, 3, 5]), np.array([[2, 4], [100, 100], [6, 8]]), np.array([1, 0, 1]), {}),
              "b": (np.array([2, 3, 6]), np.array([[3], [np.nan], [9]]), np.ones(3, bool), {})}
    v, _ = join_streams("v", row, {}, ["a", "b"], lambda stream, vid: arrays[stream])
    np.testing.assert_array_equal(v.present[:, 0], [1, 0, 0, 0, 1, 0])
    np.testing.assert_array_equal(v.present[:, 1], [0, 1, 0, 0, 0, 1])
    class Store:
        dims = [2, 1]
        def __getitem__(self, vid):
            assert vid == "train"
            return v
    norm = Normalizer.fit(Store(), ["train"])
    np.testing.assert_allclose(norm.means[0], [4, 6])
    np.testing.assert_allclose(norm.scales[0], [2, 2])
    np.testing.assert_array_equal(norm.counts, [2, 2])
    np.testing.assert_array_equal(norm.transform(v.streams[0], v.present[:, 0], 0)[1], [0, 0])


def test_statistics_match_manual_causal_windows():
    x = np.arange(1, 101, dtype=np.float32)[:, None]
    present = np.ones((100, 1), bool)
    present[88] = False
    v = VideoFeatures("v", np.arange(1, 101), [x], present, None)
    norm = Normalizer([np.zeros(1)], [np.ones(1)], [99])
    stats = window_statistics(v, norm)
    for tick in (0, 3, 35, 90):
        for k, w in enumerate((5, 15, 30, 60)):
            ix = np.arange(max(0, tick - w + 1), tick + 1)
            ix = ix[present[ix, 0]]
            values = x[ix, 0]
            slope = np.polyfit(ix, values, 1)[0] if len(ix) > 1 else 0
            np.testing.assert_allclose(stats[tick, 4 * k:4 * k + 4],
                                       [values.mean(), values.max(), values.std(), slope], rtol=1e-5, atol=1e-5)


def test_censored_hazards_have_zero_gradient():
    target = torch.tensor(tick_labels(50, []).hazard_target[[0, 44]], dtype=torch.float32)
    logits = torch.zeros_like(target, requires_grad=True)
    loss = hazard_loss(logits, target, torch.ones(2))
    loss.backward()
    assert torch.count_nonzero(logits.grad[target < 0]) == 0
    assert torch.count_nonzero(logits.grad[target >= 0]) > 0
    h = torch.zeros((1, 6))
    assert CausalTemporalModel.score30(h).item() == pytest.approx(1 - 0.5 ** 4)


def test_missing_streams_are_skipped_and_reported(tmp_path, monkeypatch):
    from anticipation.features import io
    bench, _, features, _ = synthetic_benchmark(tmp_path / "fixture")
    monkeypatch.setattr(io, "FEATURE_ROOT", str(tmp_path / "features"))
    vid = bench.eval_ids()[0]
    t, x, valid, meta = features[vid]
    io.save_features("a", vid, t, x, valid, meta)
    store = FeatureStore(tmp_path / "snapshot", bench.videos, bench.events, ["a", "b"])
    coverage = store.snapshot(bench.eval_ids())
    assert coverage["available"] == 0
    assert coverage["missing"][vid] == ["b"]
    assert set(coverage["missing"]) == set(bench.eval_ids())


def test_cli_five_folds_three_seeds_and_va30(tmp_path, monkeypatch):
    from anticipation.features import io
    from anticipation.train import main
    monkeypatch.chdir(tmp_path)
    bench, _, features, fold_data = synthetic_benchmark(tmp_path)
    root = tmp_path / "features"
    monkeypatch.setattr(io, "FEATURE_ROOT", str(root))
    for vid, (t, x, valid, meta) in features.items():
        io.save_features("synthetic", vid, t, x, valid, meta)
    bench.videos.to_csv(tmp_path / "videos.csv", index=False)
    (tmp_path / "events.json").write_text(json.dumps({"videos": bench.events}))
    (tmp_path / "folds.json").write_text(json.dumps(fold_data))
    original_hash = (tmp_path / "folds.json").read_bytes()
    # Keep git provenance available while isolating every generated artifact.
    import anticipation.train as trainer
    actual = trainer.subprocess.check_output
    monkeypatch.setattr(trainer.subprocess, "check_output", lambda args, **kw: actual(args, cwd=Path(__file__).resolve().parents[1], **kw))
    assert main(["--streams", "synthetic", "--model", "stats_logreg", "--run_name", "synthetic_e2e",
                 "--videos", str(tmp_path / "videos.csv"), "--events", str(tmp_path / "events.json"),
                 "--folds", str(tmp_path / "folds.json"), "--feature_root", str(root),
                 "--epochs", "3", "--n_boot", "20", "--threads", "2"]) == 0
    out = tmp_path / "benchmark/results/synthetic_e2e"
    scores = pd.read_parquet(out / "scores.parquet")
    assert set(scores.columns) == {"video_id", "t_sec", "score", "fold", "seed"}
    assert set(scores.seed) == {0, 1, 2}
    assert set(scores.fold) == set(range(5))
    assert not scores.duplicated(["video_id", "t_sec", "fold", "seed"]).any()
    for f in range(5):
        assert set(scores[scores.fold == f].video_id) == set(bench.folds.test_ids(f) + bench.folds.val_ids(f))
    config = json.loads((out / "run_config.json").read_text())
    metrics = json.loads((out / "metrics.json").read_text())
    assert config["status"] == "complete"
    assert metrics["summary_over_seeds"]["success30_at_fa2"]["mean"] >= 0.9
    assert len(json.loads((out / "training_history.json").read_text())) == 15
    assert (tmp_path / "folds.json").read_bytes() == original_hash
