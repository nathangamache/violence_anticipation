"""Check Phase C fold isolation, validity, validation-only layer selection and paired imminent gates."""
import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from anticipation.screen import metrics as M
from anticipation.screen import phase_c as C
from anticipation.screen import probes as P
from anticipation.screen import sets as S
from tests.test_screen import build_world


def test_layer_selection_uses_validation_and_deterministic_ties():
    infos = {18: {'p1': {'val_auc': .7}, 'p1_near': {'val_auc': .8}},
             27: {'p1': {'val_auc': .7}, 'p1_near': {'val_auc': .9}}}
    assert C.select_layer(infos, 'band') == 18
    assert C.select_layer(infos, 'near') == 27
    infos[27]['test_auc'] = 1.0
    assert C.select_layer(infos, 'band') == 18
    with pytest.raises(ValueError):
        C.select_layer({18: {'p1': {'val_auc': np.nan}}}, 'band')


def test_pilot_guard_and_interval_masks():
    ss, source = build_world()
    vid = ss.test_ids(0)[0]
    base = copy.copy(ss)
    base.prereg = None
    pilot = C.PilotSet(base, [dict(video_id=vid, start_sec=120, cap_sec=180)])
    vt = pilot.ticks(vid)
    assert vt.n == 180 and vt.masked[vt.t < 120].all()
    assert not set(pilot.train_ids(0)) & {vid}
    forbidden = next(v for v in ss.bench.videos.index if ss.fold_of(v) == 1)
    with pytest.raises(S.FrozenFoldError):
        C.PilotSet(base, [dict(video_id=forbidden, start_sec=1, cap_sec=100)])


def test_invalid_rows_masked_and_reset_causal_windows():
    ss, source = build_world()
    vid = ss.test_ids(0)[0]
    old = source.loader

    def loader(stream, v, **kwargs):
        if stream == 'phase_c':
            result = old('tstream', v, **kwargs)
            if kwargs.get('probe'):
                return result
            t, x, valid, meta = result
            valid = valid.copy()
            valid[20:100] = False
            return t, x, valid, meta
        return old(stream, v, **kwargs)

    source.loader = loader
    runner = C.PilotRun(ss, source, require_cut=False)
    x = runner.raw(vid)
    assert runner.vt(vid).masked[20:100].all()
    assert runner.cut_last(vid)[100] >= 99
    runner.pls = None
    runner.thr = {'p1': np.zeros(x.shape[1])}
    feats = runner.feats(vid, 'p1')
    assert np.allclose(feats[100, :x.shape[1]], x[100])


def test_near_n3_is_paired_joint_gain():
    ss, source = build_world(violent_signal=lambda t, onset: 4 * ((onset - t > 0) & (onset - t <= 10)))
    vids = [v for f in ss.screen_folds for v in ss.test_ids(f, kinds=('violent', 'clean', 'h1'))]
    pool = M.Pool(ss, [ss.ticks(v) for v in vids])
    rng = np.random.default_rng(2)
    signal = pool.band_pos(0, 10).astype(float)
    scores = dict(p1_near=signal, n_near=rng.normal(size=len(pool.t)), n_plus_p1_near=signal)
    stats = C.near_statistics(pool, scores, pool.boot_weights(50))
    assert stats['gates']['N1'] and stats['gates']['N3']
    assert stats['m4n'].value == pytest.approx(stats['n_plus_p1'].value - stats['n'].value)


def test_serial_null_for_gain_is_zero():
    d = C.serial({'m4n': M.Stat(.1, np.full(20, .1)), 'm1': M.Stat(.6, np.full(20, .6))})
    assert d['m4n']['p_le_null'] == pytest.approx(1 / 21)
    assert d['m1']['p_le_null'] == pytest.approx(1 / 21)


def test_p1_matches_registered_runner_on_fully_valid_inputs():
    ss, source = build_world(violent_signal=lambda t, onset: ((onset - t > 30) & (onset - t <= 120)))
    old = source.loader
    source.loader = lambda stream, vid, **kw: old('tstream' if stream == 'phase_c' else stream, vid, **kw)
    source.layer, source.arm = None, 'synthetic'
    train, val, test = ss.train_ids(0), ss.val_ids(0), ss.test_ids(0)
    runner = C.PilotRun(ss, source, require_cut=False)
    actual, _ = C.fit_fold(runner, 0, train, val, test)
    other = C.PilotRun(ss, source, require_cut=False)
    expected, _ = P.BlockRun.run_fold(other, 0, train, val, test)
    for model in ('p1', 'p1_near'):
        for vid in test:
            assert np.allclose(actual[model]['test'][vid], expected[model]['test'][vid], atol=1e-8)


def test_event_denominator_uses_pilot_cap_and_removes_original_fold1(tmp_path):
    ss, source = build_world()
    vid = next(v for v in ss.test_ids(0) if ss.kind(v) == 'clean')
    pilot = C.PilotSet(ss, [dict(video_id=vid, start_sec=1, cap_sec=90)])
    (tmp_path / 'A').mkdir()
    pd.DataFrame(dict(video_id=vid, t_sec=np.arange(1, 91), fold=0, seed=0,
                      p1=0., p1_near=0., split='test')).to_parquet(tmp_path / 'A/scores.parquet')
    bench, scores = C.event_inputs(pilot, 'A', tmp_path)
    assert bench.duration(vid) == 90
    assert not any(ss.fold_of(v) == 1 for v in bench.folds.video_ids)
    ts = C.va30.TickSet(bench, [vid], {vid: np.zeros(90)})
    ee = C.va30.EventEval(ts, C.va30.AlarmPolicy())
    assert ee.clean_hours == pytest.approx(90 / 3600)


def test_holdout_timelines_reset_and_copies_share_original_cluster():
    ss, source = build_world()
    ids = [v for f in ss.screen_folds for v in ss.test_ids(f)]
    pilot = C.PilotSet(ss, [dict(video_id=v, start_sec=1, cap_sec=ss.ticks(v).n) for v in ids])
    original = next(v for v in ids if ss.kind(v) == 'violent')
    manifest = {'rows': [dict(video_id='yt_clip/example_c000', original_video_id=original,
                              start_sec=100, duration_sec=100, annotations=[
                                  dict(onset_sec=140, end_sec=144), dict(onset_sec=180, end_sec=184)])]}
    holdout = C.HoldoutSet(pilot, manifest)
    external = list(holdout.external)
    assert len(external) == 4
    assert len({holdout.split_group[v] for v in external}) == 1
    assert set(holdout.fold_of(v) for v in external) == {0, 2, 3, 4}
    assert all(holdout.ticks(v).tte[0] == 39 for v in external)
    assert len(holdout.events()) == 4
    assert holdout.events()['first'].all()


def test_position_null_cannot_fit_on_another_copy_of_the_same_incident():
    table = pd.DataFrame({'split_group': ['incident', 'incident', 'clean']}, index=['v0', 'v2', 'n'])
    ts = SimpleNamespace(fold_v=np.array([0, 2, 3]), vids=['v0', 'v2', 'n'], bench=SimpleNamespace(videos=table))
    folds = C.null_group_folds(ts)
    assert folds[0] == folds[1] and folds[2] == 3
    assert np.array_equal(ts.fold_v, [0, 2, 3])
