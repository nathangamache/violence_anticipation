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
from anticipation.models.selection import MatchedValidation
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
    metric = MatchedValidation(store, val)
    task = "stats" if kind.startswith("stats") else "temporal"
    sel = [sample_ticks(store, ids, 1600, np.random.default_rng(0), task) for ids in (train, val)]
    if task == "stats":
        model = StatsModel(kind)
        model.fit(make_table(store, train, sel[0], norm), make_table(store, val, sel[1], norm), epochs=8, patience=3,
                  selection_score=lambda m: metric.score(lambda vid, ix: m.predict(store[vid], norm, indices=ix)))
        predict = lambda v: model.predict(v, norm)
    else:
        torch.manual_seed(1)
        model = CausalTemporalModel([4], kind, hidden=16, layers=1, heads=2, context=60, dropout=0,
                                    modality_dropout=0)
        datasets = [WindowDataset(store, s, norm, 60) for s in sel]
        fit_temporal(model, *datasets, epochs=8, patience=3, batch_size=64, lr=0.005,
                     band_weight=0, selection_score=lambda m: metric.score(lambda vid, ix: predict_temporal(m, store[vid], norm, indices=ix)))
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
    np.testing.assert_array_equal(v.present[:, 0], [1, 0, 0, 0, 1])
    np.testing.assert_array_equal(v.present[:, 1], [0, 1, 0, 0, 0])
    np.testing.assert_array_equal(v.observed, [0, 0, 1, 0, 0])
    class Store:
        dims = [2, 1]
        def __getitem__(self, vid):
            assert vid == "train"
            return v
    norm = Normalizer.fit(Store(), ["train"])
    np.testing.assert_allclose(norm.means[0], [4, 6])
    np.testing.assert_allclose(norm.scales[0], [2, 2])
    np.testing.assert_array_equal(norm.counts, [2, 1])
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
    (tmp_path / "cohort.json").write_text(json.dumps(bench.eval_ids()))
    original_hash = (tmp_path / "folds.json").read_bytes()
    # Keep git provenance available while isolating every generated artifact.
    import anticipation.train as trainer
    actual = trainer.subprocess.check_output
    monkeypatch.setattr(trainer.subprocess, "check_output", lambda args, **kw: actual(args, cwd=Path(__file__).resolve().parents[1], **kw))
    assert main(["--phase", "synthetic", "--objective", "horizon30", "--no-time_match_weights", "--streams", "synthetic", "--model", "stats_logreg", "--run_name", "synthetic_e2e",
                 "--videos", str(tmp_path / "videos.csv"), "--events", str(tmp_path / "events.json"),
                 "--folds", str(tmp_path / "folds.json"), "--cohort", str(tmp_path / "cohort.json"), "--feature_root", str(root),
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


def test_time_matched_training_balances_source_time_bins(tmp_path):
    from anticipation.eval.va30 import time_strata
    bench, store, _, _ = synthetic_benchmark(tmp_path)
    selected = sample_ticks(store, bench.folds.train_ids(0), 0, np.random.default_rng(8), "stats", time_matched=True)
    times, pos = [], []
    for vid, ix in selected.items():
        v = store[vid]
        times.extend(v.t_sec[ix])
        pos.extend(v.labels.y[30][ix])
    pos = np.asarray(pos, bool)
    strata = time_strata(np.asarray(times))
    assert pos.any()
    np.testing.assert_array_equal(np.bincount(strata[pos]), np.bincount(strata[~pos]))


def test_matched_validation_uses_shared_matching_and_validation_only(tmp_path):
    from anticipation.eval.va30 import time_strata
    bench, store, _, _ = synthetic_benchmark(tmp_path)
    ids = bench.folds.val_ids(0)
    metric = MatchedValidation(store, ids)
    assert set(metric.selection) <= set(ids)
    assert metric.score(lambda vid, ix: time_strata(store[vid].t_sec[ix]).astype(float)) == pytest.approx(0.5)
    expected = np.concatenate([((store[vid].labels.tte[ix] > 30) & (store[vid].labels.tte[ix] <= 60))
                               for vid, ix in metric.selection.items()])
    assert metric.score(lambda vid, ix: ((store[vid].labels.tte[ix] > 30) &
                                        (store[vid].labels.tte[ix] <= 60)).astype(float)) == pytest.approx(1)
    assert expected.any()


def test_checkpoint_selection_uses_matched_auc_even_when_loss_improves(tmp_path):
    bench, store, _, _ = synthetic_benchmark(tmp_path)
    ids, val = bench.folds.train_ids(0), bench.folds.val_ids(0)
    norm = Normalizer.fit(store, ids)
    tables = [make_table(store, group, sample_ticks(store, group, 0, np.random.default_rng(0), "stats"), norm)
              for group in (ids, val)]
    metrics = iter([0.8, 0.7, 0.6])
    model = StatsModel("stats_logreg").fit(*tables, epochs=3, selection_score=lambda m: next(metrics))
    assert model.best_epoch == 1
    assert [h["val_matched_far_auroc"] for h in model.history] == [0.8, 0.7, 0.6]


def test_time_only_control_is_invariant_to_feature_content(tmp_path):
    bench, store, _, _ = synthetic_benchmark(tmp_path)
    ids, val = bench.folds.train_ids(0), bench.folds.val_ids(0)
    norm = Normalizer.fit(store, ids)
    tables = [make_table(store, group, sample_ticks(store, group, 0, np.random.default_rng(0), "stats"), norm,
                         time_only=True) for group in (ids, val)]
    metric = MatchedValidation(store, val)
    model = StatsModel("time_only").fit(*tables, epochs=2,
             selection_score=lambda m: metric.score(lambda vid, ix: m.predict(store[vid], norm, indices=ix)))
    original = store[bench.folds.test_ids(0)[0]]
    changed = VideoFeatures(original.video_id, original.t_sec, [np.full_like(x, 1000) for x in original.streams],
                            ~original.present, original.labels, original.observed)
    np.testing.assert_array_equal(model.predict(original, norm), model.predict(changed, norm))


def test_partial_timelines_emit_only_common_ticks_and_block_inflated_hours(tmp_path, monkeypatch):
    import anticipation.train as trainer
    from anticipation.features import io
    bench, _, features, fold_data = synthetic_benchmark(tmp_path)
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "features"
    monkeypatch.setattr(io, "FEATURE_ROOT", str(root))
    for vid, (t, x, valid, meta) in features.items():
        cut = 180 if bench.videos.loc[vid, "role"] == "clean_negative" else 220
        io.save_features("a", vid, t[:cut], x[:cut], valid[:cut], meta)
        io.save_features("b", vid, t[:cut - 1], x[:cut - 1], valid[:cut - 1], meta)
    bench.videos.to_csv(tmp_path / "videos.csv", index=False)
    (tmp_path / "events.json").write_text(json.dumps({"videos": bench.events}))
    (tmp_path / "folds.json").write_text(json.dumps(fold_data))
    (tmp_path / "cohort.json").write_text(json.dumps(bench.eval_ids()))
    actual = trainer.subprocess.check_output
    monkeypatch.setattr(trainer.subprocess, "check_output", lambda args, **kw: actual(args, cwd=Path(__file__).resolve().parents[1], **kw))
    monkeypatch.setattr(trainer.va30, "main", lambda args: pytest.fail("partial timelines must not reach the unfixed evaluator"))
    assert trainer.main(["--phase", "synthetic", "--objective", "horizon30", "--no-time_match_weights", "--streams", "a,b", "--model", "stats_logreg", "--run_name", "partial",
                         "--videos", str(tmp_path / "videos.csv"), "--events", str(tmp_path / "events.json"),
                         "--folds", str(tmp_path / "folds.json"), "--cohort", str(tmp_path / "cohort.json"), "--feature_root", str(root),
                         "--seeds", "0", "--epochs", "1", "--threads", "2"]) == 2
    out = tmp_path / "benchmark/results/partial"
    scores = pd.read_parquet(out / "scores.parquet")
    for vid, rows in scores.groupby("video_id"):
        assert rows.t_sec.max() == (179 if bench.videos.loc[vid, "role"] == "clean_negative" else 219)
    config = json.loads((out / "run_config.json").read_text())
    assert config["status"] == "evaluation_unavailable"
    assert "catalog durations" in config["error"]


@pytest.mark.parametrize("kind", ["gru", "transformer"])
def test_absent_modalities_cannot_change_scores(kind):
    torch.manual_seed(9)
    model = CausalTemporalModel([4, 3], kind, hidden=8, layers=1, heads=2, context=60, dropout=0)
    model.eval()
    xs = [torch.randn(2, 60, 4), torch.randn(2, 60, 3)]
    present = torch.ones(2, 60, 2)
    present[:, :, 1] = 0
    with torch.no_grad():
        first = model(*[xs, present])[0]
        xs[1] = torch.full_like(xs[1], 1000)
        second = model(*[xs, present])[0]
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_missing_matched_validation_allows_only_one_diagnostic_epoch(tmp_path):
    bench, store, _, _ = synthetic_benchmark(tmp_path)
    ids = bench.folds.train_ids(0)
    val = [v for v in bench.folds.val_ids(0) if bench.videos.loc[v, "role"] == "violent"]
    norm = Normalizer.fit(store, ids)
    tables = [make_table(store, group, sample_ticks(store, group, 0, np.random.default_rng(0), "stats"), norm)
              for group in (ids, val)]
    metric = MatchedValidation(store, val)
    assert not metric.available
    model = StatsModel("stats_logreg").fit(*tables, epochs=10, selection_score=lambda m: metric.score(None))
    assert model.best_epoch == 1 and len(model.history) == 1
    assert model.history[0]["val_matched_far_auroc"] is None


@pytest.mark.parametrize("kind", ["gru", "transformer"])
def test_scalar_stream_retains_a_learnable_signal(kind):
    torch.manual_seed(11)
    model = CausalTemporalModel([1], kind, hidden=8, layers=1, heads=2, context=60, dropout=0, modality_dropout=0)
    model.eval()
    x = torch.randn(2, 60, 1, requires_grad=True)
    h, _ = model([x], torch.ones(2, 60, 1))
    h.sum().backward()
    assert x.grad.abs().max().item() > 1e-6
