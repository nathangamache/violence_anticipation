"""Rebuild the label-dependent screen registration from the reviewed v3 catalog."""

import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from anticipation.data.catalog import REPO_ROOT, file_sha256
from anticipation.screen import prereg, sets

SUBSETS = {'first_events': 'first', 'real_first_events': 's_real', 'long60_first_events': 's_long60',
           'ramp90_first_events': 's_ramp', 'official_ubi_first_events': 's_offubi',
           'audio_first_event_candidates': 's_audio'}


def build(original, screen, videos_csv, events_json):
    data = copy.deepcopy(original)
    data.update(version=2, status='PREREGISTERED_BEFORE_V3_SCREEN_RESULTS',
                created_local_date=datetime.now(ZoneInfo('America/Detroit')).date().isoformat())
    data['label_policy'] = ('Reviewed v3 onset corrections are primary. Amendment 03 proposed masks remain '
                            'secondary sensitivity analyses only; fold 1 remains frozen for model evaluation.')
    data['prior_knowledge'] += (' V1 Phase A results and its audit motivated independent label curation. '
                                'V3 labels were fixed using raw videos and a purpose-specific Qwen scan before '
                                'any v3 block outcome was computed. The lead retains the one-time confirmation.')
    data['provenance'].update(videos_csv=str(videos_csv.relative_to(REPO_ROOT)),
                              events_json=str(events_json.relative_to(REPO_ROOT)),
                              videos_csv_sha256=file_sha256(str(videos_csv)),
                              events_json_sha256=file_sha256(str(events_json)))
    events = []
    for row in screen.events().itertuples():
        entry = screen.bench.events[row.video_id]
        event = entry['events'][int(row.index)]
        events.append({'video_id': row.video_id, 'source_kind': row.source_kind,
                       'onset_source': row.onset_source, **event})
    events.sort(key=lambda e: (e['video_id'], e['index']))
    data['sets']['eligible30_events'] = events
    data['sets']['positive_videos'] = sorted({e['video_id'] for e in events})
    keyed = {(e['video_id'], e['index']): e for e in events}
    frame = screen.events()
    for name, column in SUBSETS.items():
        data['sets'][name] = [keyed[(r.video_id, int(r.index))]
                              for r in frame[frame[column]].itertuples()]
        data['sets'][name].sort(key=lambda e: (e['video_id'], e['index']))
    listed = set(data['sets']['positive_videos']) | set(data['sets']['clean_cpu_videos']) | set(data['sets']['h1_videos'])
    positives = set(data['sets']['positive_videos'])
    preserve_intervals = set(data['sets']['h1_videos']) | set(data['sets']['h1p_videos'])
    data['scored_intervals_sec'] = {
        v: [1, screen.ticks_computed(v)] if v in positives and v not in preserve_intervals
        else original['scored_intervals_sec'][v]
        for v in sorted(listed)}
    data['sets']['teacher_videos'] = sorted(set(original['sets']['teacher_videos']) & listed)
    for name in ('h1_videos', 'h1p_videos', 'clean_cpu_videos', 'clean_vlm_videos'):
        assert data['sets'][name] == original['sets'][name]
    folds = screen.bench.folds
    if folds.sha256 != original['provenance']['folds_content_sha256']:
        raise ValueError('The frozen folds_v2 assignment or validation memberships changed')
    screenable = {vid for vid in listed if folds.fold_of(vid) != data['confirmation_fold']}
    data['fold_membership'] = {}
    for fold in data['screen_folds']:
        data['fold_membership'][str(fold)] = {
            'train': sorted(set(folds.train_ids(fold)) & screenable),
            'val': sorted(set(folds.val_ids(fold)) & screenable),
            'test': sorted(set(folds.test_ids(fold)) & screenable),
        }
        for split, ids in data['fold_membership'][str(fold)].items():
            if any(folds.fold_of(vid) == data['confirmation_fold'] for vid in ids):
                raise ValueError(f'fold {fold} {split} contains a confirmation-fold video')
    if set(data['fold_membership']) != set(original['fold_membership']):
        raise ValueError('The registered screening folds changed')
    data['carried_amendments'] = []
    for number in range(2, 6):
        path = next((Path(REPO_ROOT) / 'benchmark/screens').glob(f'amendment_{number:02d}_*.md'))
        data['carried_amendments'].append({'path': str(path.relative_to(REPO_ROOT)), 'sha256': file_sha256(str(path))})
    return data


def summary_markdown(data, registration_sha256):
    sets = data['sets']
    rows = [
        '# Screen set v2 registration',
        '',
        'This summary describes the frozen registration only; it contains no screen outcomes.',
        '',
        f"- Registration SHA-256: `{registration_sha256}`",
        f"- V3 videos catalog SHA-256: `{data['provenance']['videos_csv_sha256']}`",
        f"- V3 events catalog SHA-256: `{data['provenance']['events_json_sha256']}`",
        f"- Fold assignment SHA-256: `{data['provenance']['folds_content_sha256']}`",
        f"- Screening folds: {', '.join(map(str, data['screen_folds']))}",
        f"- Frozen confirmation fold: {data['confirmation_fold']}",
        '',
        '| Registered set | Count |',
        '|---|---:|',
    ]
    for name in ('positive_videos', 'eligible30_events', 'first_events', 'real_first_events',
                 'long60_first_events', 'ramp90_first_events', 'official_ubi_first_events',
                 'audio_first_event_candidates', 'clean_cpu_videos', 'clean_vlm_videos',
                 'h1_videos', 'h1p_videos'):
        rows.append(f"| `{name}` | {len(sets[name])} |")
    rows.extend(['', '| Fold | Train videos | Validation videos | Test videos |',
                 '|---:|---:|---:|---:|'])
    for fold in data['screen_folds']:
        members = data['fold_membership'][str(fold)]
        rows.append(f"| {fold} | {len(members['train'])} | {len(members['val'])} | {len(members['test'])} |")
    rows.extend(['', 'H1 and H1p memberships and their scored intervals are preserved from v1. Fold 1 remains',
                 'frozen for the lead’s one-time confirmation.', '', 'Carried amendments:'])
    rows.extend(f"- `{row['path']}` (SHA-256 `{row['sha256']}`)" for row in data['carried_amendments'])
    return '\n'.join(rows) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='benchmark/screens/screen_set_v2.json')
    args = ap.parse_args()
    root = Path(REPO_ROOT)
    corrections = json.loads((root / 'benchmark/labels_v3/onset_corrections.json').read_text())
    if corrections['status'] != 'REVIEW_COMPLETE':
        raise ValueError('Visual review is incomplete')
    original = prereg.load().data
    videos_csv, events_json = root / 'benchmark/v3/videos.csv', root / 'benchmark/v3/events.json'
    screen = sets.load_screen_set(soft_masks_path=None, folds=original['provenance']['folds_path'],
                                  videos_csv=str(videos_csv), events_json=str(events_json))
    data = build(original, screen, videos_csv, events_json)
    data['provenance']['corrections_sha256'] = file_sha256(str(root / 'benchmark/labels_v3/onset_corrections.json'))
    path = Path(args.out)
    if path.exists():
        raise ValueError('Refusing to overwrite an existing registration')
    path.write_text(json.dumps(data, indent=2) + '\n')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix('.sha256').write_text(f'{sha}  {path.name}\n')
    summary = path.with_name('summary_v2.md')
    if summary.exists():
        raise ValueError('Refusing to overwrite an existing v2 summary')
    summary.write_text(summary_markdown(data, sha))
    print(f'{sha}  {path}')
    print(f'{summary}')
    print('Pin this hash in anticipation/screen/prereg.py and commit before running the screen.')


if __name__ == '__main__':
    main()
