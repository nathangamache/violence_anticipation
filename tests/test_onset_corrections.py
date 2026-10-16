import copy

import pytest

from anticipation.data.apply_corrections import apply
from anticipation.data.onset_scan import flag_frames, scan_times


def fixture():
    videos = {'v': {'role': 'violent', 'onset_source': 'hivau', 'segments': [[40, 50], [100, 110]],
                    'duration_sec': 120, 'mask_spans': [[10, 20]]}}
    catalog = {'v': {'role': 'violent', 'is_clean_negative': 'False'}}
    return videos, catalog


def correction(index=0, verdict='earlier', onset=30):
    return {'video_id': 'v', 'event_index': index, 'onset_source': 'hivau',
            'original_onset_sec': [40, 100][index], 'verdict': verdict, 'corrected_onset_sec': onset,
            'evidence': 'sheets/a.jpg', 'passes': [
                {'blind_id': p, 'verdict': verdict, 'corrected_onset_sec': onset} for p in ['a13', 'b02']]}


def test_eligibility_blockers_and_original_untouched():
    videos, catalog = fixture()
    original = copy.deepcopy(videos)
    new, rows = apply(videos, catalog, [correction()])
    assert videos == original
    assert new['v']['events'][0]['clear_start'] == 20
    assert not new['v']['events'][0]['eligible30']
    assert new['v']['events'][1]['eligible30']
    assert rows['v']['n_eligible30'] == '1'


def test_removal_reindexes_and_never_creates_clean_negatives():
    videos, catalog = fixture()
    new, rows = apply(videos, catalog, [correction(verdict='no_violence', onset=None)])
    assert new['v']['events'][0]['index'] == 0
    assert new['v']['events'][0]['onset'] == 100
    new, rows = apply(videos, catalog, [correction(verdict='no_violence', onset=None),
                                      correction(1, 'no_violence', None)])
    assert new['v']['role'] == rows['v']['role'] == 'unlabelled'
    assert rows['v']['is_clean_negative'] == 'False'
    assert not new['v']['segments']
    assert rows['v']['n_eligible30'] == '0'


def test_separate_missed_episode_ends_clear_history_without_moving_target():
    videos, catalog = fixture()
    c = correction(verdict='keep', onset=40)
    c['additional_violence_spans'] = [[25, 28]]
    with pytest.raises(ValueError, match='visual reconciliation'):
        apply(videos, catalog, [c])
    c['reconciliation'] = {'verdict': 'keep', 'additional_violence_spans': [[25, 28]],
                           'evidence': 'Separate earlier fight at 25-28 on fine sheets'}
    new, rows = apply(videos, catalog, [c])
    assert new['v']['segments'] == [[25, 28], [40, 50], [100, 110]]
    assert new['v']['events'][1]['clear_start'] == 38
    assert not new['v']['events'][1]['eligible30']
    assert rows['v']['n_eligible30'] == '1'
    c['additional_violence_spans'] = [[25, 121]]
    c['reconciliation']['additional_violence_spans'] = [[25, 121]]
    with pytest.raises(ValueError, match='Invalid additional episode'):
        apply(videos, catalog, [c])


@pytest.mark.parametrize('change', ['duplicate', 'original', 'direction', 'passes', 'disagreement', 'bounds'])
def test_invalid_reviews_rejected(change):
    videos, catalog = fixture()
    c = correction()
    if change == 'duplicate':
        cs = [c, c]
    else:
        cs = [c]
        if change == 'original':
            c['original_onset_sec'] = 41
        elif change == 'direction':
            c['verdict'] = 'later'
        elif change == 'passes':
            c['passes'] = c['passes'][:1]
        elif change == 'disagreement':
            c['passes'][1]['verdict'] = 'keep'
        elif change == 'bounds':
            c['corrected_onset_sec'] = -1
    with pytest.raises(ValueError):
        apply(videos, catalog, cs)


def test_scan_rolling_and_missing_detection():
    times = list(range(12))
    scores = [0, .9, 0, .9, .9, 0, 0, 0, 0, 0, 0, 0]
    flags = flag_frames(times, scores, 8, .8)
    assert flags['early_violence']
    assert flags['candidate_onset_sec'] == 1
    assert flags['no_violence_at_onset']
    assert not flag_frames(times, [0] * 7 + [.9] * 5, 8, .8)['early_violence']
    assert not flag_frames(times, [.8] * 12, 8, .8)['early_violence']


def test_scan_bounds_fractional_and_video_end():
    times = scan_times({'clear_start': 23.5, 'onset': 60.2}, 70)
    assert times[0] == 23.5
    assert times[-1] < 70
    assert all(b - a == 1 for a, b in zip(times, times[1:]))


def test_registration_alias_keeps_v1_default():
    from anticipation.screen import prereg
    assert prereg.load('v1').sha256 == prereg.load().sha256
    assert prereg.resolve_path('v2').endswith('screen_set_v2.json')
    assert prereg.resolve_path('/tmp/custom.json') == '/tmp/custom.json'


def test_screen_set_uses_registered_catalog_paths(monkeypatch):
    from anticipation.screen import prereg, sets
    calls = []

    def capture(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError('captured')

    monkeypatch.setattr(sets.va30.Benchmark, 'load', capture)
    pr = prereg.Prereg({'screen_folds': [0, 2, 3, 4], 'confirmation_fold': 1, 'sets': {},
                       'provenance': {'videos_csv': 'benchmark/v3/videos.csv',
                                      'events_json': 'benchmark/v3/events.json'}}, '', '')
    with pytest.raises(RuntimeError, match='captured'):
        sets.load_screen_set(pr)
    assert calls[0]['videos_csv'].endswith('/benchmark/v3/videos.csv')
    assert calls[0]['events_json'].endswith('/benchmark/v3/events.json')


def test_rebuilt_registration_drops_ineligible_events_and_keeps_fold_freeze(monkeypatch):
    from pathlib import Path
    from anticipation.screen import prereg, register_v2
    from anticipation.data.catalog import REPO_ROOT
    from tests.test_screen import build_world, synthetic_prereg_data

    screen, _ = build_world()
    original = prereg.load().data
    synthetic = synthetic_prereg_data(screen)
    original['sets'] = synthetic['sets']
    original['sets']['teacher_videos'] = []
    original['fold_membership'] = synthetic['fold_membership']
    original['scored_intervals_sec'] = synthetic['scored_intervals_sec']
    h1 = original['sets']['h1_videos'][0]
    original['scored_intervals_sec'][h1] = [1, 9]
    target = screen.test_ids(0, kinds=('violent',))[0]
    screen.bench.events[target]['segments'] = [[10, 20]]
    screen.bench._events.clear()
    screen.bench._labels.clear()
    screen._events = None
    for vid in screen.bench.events:
        screen.bench.events[vid]['events'] = screen.bench.event_rows(vid)
    monkeypatch.setattr(register_v2, 'file_sha256', lambda path: 'synthetic')
    rebuilt = register_v2.build(original, screen, Path(REPO_ROOT) / 'benchmark/v3/videos.csv',
                                Path(REPO_ROOT) / 'benchmark/v3/events.json')
    assert target not in rebuilt['sets']['positive_videos']
    assert all(screen.fold_of(e['video_id']) != 1 for e in rebuilt['sets']['eligible30_events'])
    assert rebuilt['fold_membership'] == original['fold_membership']
    assert rebuilt['sets']['h1_videos'] == original['sets']['h1_videos']
    assert rebuilt['scored_intervals_sec'][h1] == [1, 9]
    assert rebuilt['blocks'] == original['blocks']
    assert rebuilt['gates'] == original['gates']
    assert all(e['onset'] - e['first_tick'] >= 30 for e in rebuilt['sets']['eligible30_events'])


def test_blind_reviews_require_reconciliation_and_preserve_unreviewed_status():
    from anticipation.data.onset_finalize import join_reviews
    scan = {'video_id': 'v', 'event_index': 0, 'original_onset_sec': 40, 'onset_source': 'hivau',
            'early_violence': True, 'no_violence_at_onset': False}
    a = {'blind_id': 'A0', 'verdict': 'earlier', 'corrected_onset_sec': 30, 'evidence': 'grab at 30'}
    b = {'blind_id': 'B4', 'verdict': 'keep', 'corrected_onset_sec': None, 'evidence': 'uncertain contact'}
    with pytest.raises(ValueError, match='Unreconciled'):
        join_reviews([scan], [{('v', 0): a}, {('v', 0): b}], {})
    reconciled = {'verdict': 'earlier', 'corrected_onset_sec': 30.5, 'evidence': 'fine sheet, grab at 30.5'}
    result = join_reviews([scan], [{('v', 0): a}, {('v', 0): b}], {('v', 0): reconciled})[0]
    assert result['corrected_onset_sec'] == 30.5
    assert result['visually_reviewed']
    scan['early_violence'] = False
    result = join_reviews([scan], [{}, {}], {})[0]
    assert result['verdict'] == 'keep'
    assert not result['visually_reviewed']
    assert result['evidence']['basis'] == 'scan_no_flag_not_visually_reviewed'


def test_additional_episode_proposals_cannot_be_silently_discarded():
    from anticipation.data.onset_finalize import join_reviews
    scan = {'video_id': 'v', 'event_index': 0, 'original_onset_sec': 40, 'onset_source': 'hivau',
            'early_violence': False, 'no_violence_at_onset': False}
    a = {'blind_id': 'A0', 'verdict': 'keep', 'evidence': 'separate fight',
         'additional_violence_spans': [[10, 20]]}
    b = {'blind_id': 'B4', 'verdict': 'keep', 'evidence': 'target correct'}
    with pytest.raises(ValueError, match='Unreconciled'):
        join_reviews([scan], [{('v', 0): a}, {('v', 0): b}], {})
    reconciliation = {'verdict': 'keep', 'additional_violence_spans': [[10, 20]],
                      'evidence': 'earlier separate fight verified on fine sheets'}
    result = join_reviews([scan], [{('v', 0): a}, {('v', 0): b}], {('v', 0): reconciliation})[0]
    assert result['additional_violence_spans'] == [[10, 20]]


def test_clean_calibration_excludes_catalog_masks():
    from anticipation.data.onset_scan import calibration_usable
    videos = {'v': {'mask_spans': [[10, 20]], 'soft_mask_spans': [[30, 40]],
                    'other_segments': [[50, 55]]}}
    for t in [10, 15, 20, 35, 60, 65]:
        assert not calibration_usable({'video_id': 'v', 'label': 'clean', 't_sec': t}, videos)
    assert calibration_usable({'video_id': 'v', 'label': 'clean', 't_sec': 66}, videos)
    assert calibration_usable({'video_id': 'v', 'label': 'violent', 't_sec': 15}, videos)
