"""Rebuild the label-dependent screen registration from the reviewed v3 catalog."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from anticipation.data.catalog import REPO_ROOT, file_sha256
from anticipation.screen import prereg, sets

SUBSETS = {'first_events': 'first', 'real_first_events': 's_real', 'long60_first_events': 's_long60',
           'ramp90_first_events': 's_ramp', 'official_ubi_first_events': 's_offubi',
           'audio_first_event_candidates': 's_audio'}


def build(original, screen, videos_csv, events_json):
    data = copy.deepcopy(original)
    data.update(version=2, status='PREREGISTERED_BEFORE_V3_SCREEN_RESULTS', created_local_date='2026-10-16')
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
    data['scored_intervals_sec'] = {v: [1, screen.ticks_computed(v)] for v in sorted(listed)}
    data['sets']['teacher_videos'] = sorted(set(original['sets']['teacher_videos']) & listed)
    for name in ('h1_videos', 'h1p_videos', 'clean_cpu_videos', 'clean_vlm_videos'):
        assert data['sets'][name] == original['sets'][name]
    assert data['fold_membership'] == original['fold_membership']
    data['carried_amendments'] = []
    for number in range(1, 6):
        path = next((Path(REPO_ROOT) / 'benchmark/screens').glob(f'amendment_{number:02d}_*.md'))
        data['carried_amendments'].append({'path': str(path.relative_to(REPO_ROOT)), 'sha256': file_sha256(str(path))})
    return data


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
    screen = sets.load_screen_set(soft_masks_path=None, videos_csv=str(videos_csv), events_json=str(events_json))
    data = build(original, screen, videos_csv, events_json)
    data['provenance']['corrections_sha256'] = file_sha256(str(root / 'benchmark/labels_v3/onset_corrections.json'))
    path = Path(args.out)
    if path.exists():
        raise ValueError('Refusing to overwrite an existing registration')
    path.write_text(json.dumps(data, indent=2) + '\n')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix('.sha256').write_text(f'{sha}  {path.name}\n')
    print(f'{sha}  {path}')
    print('Pin this hash in anticipation/screen/prereg.py and commit before running the screen.')


if __name__ == '__main__':
    main()
