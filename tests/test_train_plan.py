"""Band supervision, nuisance controls and frozen-fold evaluation checks."""

import numpy as np
import pandas as pd
import pytest
import torch

from tests.test_train import synthetic_benchmark, thread_cap
from anticipation.models.controls import SceneProvider, NuisanceModel, first_frame_features, OffsetLogisticRegression
from anticipation.models.data import Normalizer, VideoFeatures, sample_ticks
from anticipation.models.evaluation import evaluation_subset, check_score_coverage, restore_fold_ids
from anticipation.models.objectives import band_targets, event_bags, TRAIN_HAZARD_EDGES
from anticipation.models.temporal import (CausalTemporalModel, WindowDataset, EventBagDataset,
                                          event_band_loss, fit_temporal, predict_temporal)


def test_band_boundaries_and_extended_censoring(tmp_path):
    bench, store, _, _ = synthetic_benchmark(tmp_path)
    v = store['synthetic/f0v0']
    y, valid = band_targets(v, 'violent')
    assert y[v.labels.tte == 31].all() and not y[v.labels.tte == 30].any()
    assert not valid[v.labels.in_violence].any()
    assert v.labels.hazard_target.shape == (220, 10)
    assert tuple(v.labels.hazard_edges) == TRAIN_HAZARD_EDGES
    clean = store['synthetic/f0v2']
    assert np.all(clean.labels.hazard_target[-1] == -1)
    cy, cv = band_targets(clean, 'clean_negative')
    assert not cy.any() and not cv[-59:].any()


def test_event_bags_pair_identical_clock_and_exclude_onset(tmp_path):
    bench, store, _, _ = synthetic_benchmark(tmp_path)
    bags, info = event_bags(store, bench.folds.train_ids(0), max_ticks=16)
    assert info['positive_event_bags'] > 0
    for positive, negative in zip(bags[::2], bags[1::2]):
        pv, ticks, y = positive
        nv, nticks, ny = negative
        np.testing.assert_array_equal(ticks, nticks)
        assert y == 1 and ny == 0
        assert np.all(store[pv].labels.tte[ticks] >= 30)
        assert np.all(store[pv].labels.tte[ticks] <= 120)
        assert store.catalog.loc[pv, 'source_kind'] == store.catalog.loc[nv, 'source_kind']
        assert store.catalog.loc[nv, 'role'] == 'clean_negative'
        assert not store[pv].labels.masked[ticks].any()


def test_sparse_event_loss_uses_only_unmasked_bag_ticks():
    logits = torch.tensor([[8., -8., 100.], [-8., -8., 100.]], requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, False]])
    loss = event_band_loss(logits, mask, torch.tensor([1., 0.]))
    assert loss < .01
    loss.backward()
    assert torch.equal(logits.grad[:, -1], torch.zeros(2))
    assert logits.grad[0, 0] < 0 and logits.grad[1, 0] > 0


@pytest.mark.parametrize('kind', ['gru', 'transformer'])
def test_band_model_padding_and_future_causality(tmp_path, kind):
    bench, store, _, _ = synthetic_benchmark(tmp_path)
    norm = Normalizer.fit(store, bench.folds.train_ids(0))
    selection = {'synthetic/f0v0': np.array([0, 90])}
    ds = WindowDataset(store, selection, norm, 180, objective='band')
    model = CausalTemporalModel([4], kind, hidden=8, layers=1, heads=2, context=180,
                                dropout=0, modality_dropout=0, score_kind='band').eval()
    first = ds[0]
    assert first['history'].sum() == 1
    assert not first['present'][:-1].any()
    assert np.all(first['streams'][0][:-1] == 0)
    a = predict_temporal(model, store['synthetic/f0v0'], norm, indices=np.array([0, 90]))
    assert np.isfinite(a).all()
    full = store['synthetic/f0v0']
    v = VideoFeatures(full.video_id, full.t_sec, [x.copy() for x in full.streams], full.present, full.labels, full.observed)
    v.streams[0][91:] = 123
    b = predict_temporal(model, v, norm, indices=np.array([0, 90]))
    np.testing.assert_allclose(a, b, atol=1e-7)
    augmented = WindowDataset(store, selection, norm, 180, history_dropout=1)
    assert 1 <= augmented[1]['history'].sum() <= 91


def test_distance_downweight_is_optional_and_off_by_default(tmp_path):
    bench, store, _, _ = synthetic_benchmark(tmp_path)
    norm = Normalizer.fit(store, bench.folds.train_ids(0))
    sel = {'synthetic/f0v0': np.array([0])}
    assert WindowDataset(store, sel, norm, 180)[0]['weight'] == 1
    assert WindowDataset(store, sel, norm, 180, far_positive_downweight=True)[0]['weight'] < 1


def test_first_frame_control_and_train_only_scene_encoder(tmp_path):
    bench, store, features, _ = synthetic_benchmark(tmp_path)
    ids = bench.folds.train_ids(0)
    norm = Normalizer.fit(store, ids)
    v = store['synthetic/f0v0']
    a = first_frame_features(v, norm, np.arange(100), anchor_tick=2)
    assert not a[0].any()
    np.testing.assert_array_equal(a[1], a[-1])
    provider = SceneProvider('synthetic', bench.eval_ids(), loader=lambda s, vid: features[vid])
    model = NuisanceModel(provider, bench.videos.source_kind.to_dict()).fit_encoder(ids)
    assert set(model.encoder_train_ids) == set(ids)
    expected = np.mean([features[vid][1][1] for vid in ids], axis=0)
    np.testing.assert_allclose(model.scene_scaler.mean_, expected, atol=1e-6)
    x = model.features([v.video_id] * 3, [1, 2, 3], fit=True)
    assert not x[0, -model.scene_pca.n_components_-1:].any()
    np.testing.assert_array_equal(x[1, -model.scene_pca.n_components_-1:], x[2, -model.scene_pca.n_components_-1:])


def test_residual_logistic_preserves_fixed_nuisance_offset():
    offset = np.array([-2., 2.] * 30)
    y = np.array([0., 1.] * 30)
    model = OffsetLogisticRegression().fit(np.zeros((60, 2)), y, offset=offset)
    a = model.predict_proba(np.zeros((2, 2)), offset=np.array([-2., 2.]))[:, 1]
    np.testing.assert_allclose(np.log(a / (1-a)), [-2, 2], atol=1e-5)


def test_frozen_fold_subset_and_horizon_prefix_guard(tmp_path):
    bench, _, _, _ = synthetic_benchmark(tmp_path)
    folds = [0, 2, 3, 4]
    ids = [v for v in bench.eval_ids() if bench.folds.fold_of(v) in folds]
    scores = pd.concat([pd.DataFrame(dict(video_id=v, t_sec=np.arange(1, 220), score=.1,
                                         fold=bench.folds.fold_of(v), seed=0)) for v in ids])
    check_score_coverage(scores, bench, folds)
    subset, selected = evaluation_subset(bench, scores, folds)
    assert not set(subset.folds.assignment) & set(bench.folds.test_ids(1))
    assert set(selected.fold) == {0, 1, 2, 3}
    assert subset.provenance['internal_to_original_fold'] == {'0': 0, '1': 2, '2': 3, '3': 4}
    with pytest.raises(ValueError, match='incomplete joint'):
        check_score_coverage(scores[scores.t_sec != 10], bench, folds)
    bench.video_horizons = {v: 180. for v in ids if bench.videos.loc[v, 'role'] == 'clean_negative'}
    capped = scores[[t <= bench.duration(v) for v, t in zip(scores.video_id, scores.t_sec)]]
    check_score_coverage(capped, bench, folds)


def test_band_gru_event_training_end_to_end(tmp_path):
    bench, store, _, _ = synthetic_benchmark(tmp_path)
    ids, val = bench.folds.train_ids(0), bench.folds.val_ids(0)
    norm = Normalizer.fit(store, ids)
    windows = []
    for group in [ids, val]:
        sel = sample_ticks(store, group, 64, np.random.default_rng(2), 'temporal', objective='band')
        windows.append(WindowDataset(store, sel, norm, 180, objective='band', time_match_band=True))
    bags = EventBagDataset(windows[0], ids, max_ticks=4)
    model = CausalTemporalModel([4], hidden=8, layers=1, heads=2, context=180, score_kind='band')
    history = fit_temporal(model, *windows, epochs=1, batch_size=32, event_dataset=bags,
                           selection_score=lambda m: .5)
    assert np.isfinite(history[0]['train_loss'])
    assert np.isfinite(predict_temporal(model, store['synthetic/f0v0'], norm, indices=np.array([0, 100]))).all()


def test_original_fold_ids_survive_event_report_remapping():
    result = {"seeds": {"0": {"_event_table_op2": [{"fold": 1}, {"fold": 3}],
                              "theta_per_fold": {"0": .1, "1": .2}}}}
    restore_fold_ids(result, [0, 2, 3, 4])
    assert result["seeds"]["0"]["_event_table_op2"] == [{"fold": 2}, {"fold": 4}]
    assert result["seeds"]["0"]["theta_per_fold"] == {"0": .1, "2": .2}


def test_person_masked_scene_file_contract(tmp_path, monkeypatch):
    import json
    from anticipation.features import io
    monkeypatch.setattr(io, 'FEATURE_ROOT', str(tmp_path))
    path = tmp_path / 'siglip2_scene0p' / 'synthetic' / 'v.npz'
    path.parent.mkdir(parents=True)
    np.savez(path, emb=np.array([1., 2., 3.]), meta=json.dumps(dict(
        extractor='siglip2_scene0p', t_sec=2., box_source='pose_raw', grey=128)))
    provider = SceneProvider('siglip2_scene0p', ['synthetic/v'])
    assert provider.people_masked and provider.present['synthetic/v']
    np.testing.assert_array_equal(provider.anchors['synthetic/v'], [1, 2, 3])


@pytest.mark.parametrize('kind', ['first_frame', 'nuisance', 'residual_logreg'])
def test_band_controls_fit_and_score(tmp_path, kind):
    from anticipation.models.stats import StatsModel, make_table
    bench, store, features, _ = synthetic_benchmark(tmp_path)
    ids, val = bench.folds.train_ids(0), bench.folds.val_ids(0)
    norm = Normalizer.fit(store, ids)
    selection = [sample_ticks(store, group, 256, np.random.default_rng(0), 'stats', objective='band')
                 for group in (ids, val)]
    nuisance = None
    if kind != 'first_frame':
        provider = SceneProvider('synthetic', bench.eval_ids(), loader=lambda s, v: features[v])
        nuisance = NuisanceModel(provider, bench.videos.source_kind.to_dict())
        vids = [v for v, ix in selection[0].items() for _ in ix]
        times = np.concatenate([store[v].t_sec[ix] for v, ix in selection[0].items()])
        target = np.concatenate([band_targets(store[v], store.catalog.loc[v, 'role'])[0][ix]
                                 for v, ix in selection[0].items()])
        nuisance.fit(ids, vids, times, target, np.ones(len(target)))
    if kind == 'nuisance':
        model = nuisance
    else:
        model = StatsModel(kind, nuisance=nuisance)
        tables = [make_table(store, group, selected, norm, objective='band', time_match=True,
                             first_frame=kind == 'first_frame', nuisance=nuisance)
                  for group, selected in zip([ids, val], selection)]
        model.fit(*tables, epochs=1, selection_score=lambda m: .5)
    score = model.predict(store['synthetic/f0v0'], norm)
    assert score.shape == (220,) and np.isfinite(score).all()


def test_temporal_time_sampling_matches_band_positives(tmp_path):
    from anticipation.eval.va30 import time_strata
    bench, store, _, _ = synthetic_benchmark(tmp_path)
    selected = sample_ticks(store, bench.folds.train_ids(0), 0, np.random.default_rng(0),
                            'temporal', objective='band', time_matched=True)
    positive, times = [], []
    for vid, ix in selected.items():
        y, valid = band_targets(store[vid], store.catalog.loc[vid, 'role'])
        ix = ix[valid[ix]]
        positive.extend(y[ix] > 0)
        times.extend(store[vid].t_sec[ix])
    positive = np.array(positive)
    bins = time_strata(np.array(times))
    assert positive.any()
    np.testing.assert_array_equal(np.bincount(bins[positive]), np.bincount(bins[~positive]))
