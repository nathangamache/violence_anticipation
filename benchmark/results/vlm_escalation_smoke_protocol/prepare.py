"""Freeze screening and pilot manifests without opening confirmation-fold media or feature files."""
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

from anticipation.data.folds import Folds
from anticipation.screen.prereg import load
from protocol import FOLDS, ROOT, require_screen_ids, sha


def ordered(ids, salt):
    return sorted(ids, key=lambda v: hashlib.sha256((salt + v).encode()).hexdigest())


def write_manifest(name, rows):
    path = ROOT / (name + '.csv')
    fields = sorted(set().union(*(set(r) for r in rows)))
    import io
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    raw = stream.getvalue()
    if path.exists() and path.read_bytes() != raw.encode():
        raise ValueError(f'Refusing to alter frozen {path}')
    path.write_text(raw)
    path.with_suffix('.csv.sha256').write_text(sha(path) + '\n')
    return dict(path=str(path.relative_to(Path.cwd().resolve())), sha256=sha(path), n=len(rows))


def main():
    pr = load()
    folds = Folds.load('benchmark/folds_v2.json')
    catalog = pd.read_csv('benchmark/videos.csv').set_index('video_id')
    h1 = pd.read_csv('benchmark/extract_manifest_v2.csv').set_index('video_id')
    groups = {'positive': set(pr.ids('positive_videos')), 'clean': set(pr.ids('clean_vlm_videos')),
              'h1': set(pr.ids('h1_videos'))}
    ids = set.union(*groups.values())
    require_screen_ids(ids, folds)
    screen = {}
    for vid in sorted(ids):
        row = catalog.loc[vid]
        path = h1.loc[vid, 'path'] if vid in h1.index else row['path']
        lo, hi = pr.interval(vid)
        cap = min(float(row.duration_sec), hi)
        screen[vid] = dict(video_id=vid, path=path, start_sec=lo, cap_sec=cap,
                           source_kind=row.source_kind, group=next(k for k, vs in groups.items() if vid in vs),
                           fold=folds.fold_of(vid))
    events = pr.events('long60_first_events')
    pilot = {}
    for e in events:
        vid = e['video_id']
        pilot[vid] = {**screen[vid], 'start_sec': max(1., e['onset'] - 300),
                      'cap_sec': min(screen[vid]['cap_sec'], e['onset'] + 10)}
    counts = {}
    for source, n in [('ucf', 38), ('ubi', 38), ('xd_movie', 37), ('xd_youtube', 37)]:
        clean = [v for v in groups['clean'] if screen[v]['source_kind'] == source]
        selected = ordered(clean, 'vlm-pilot-v1-clean-')[:n]
        if len(selected) != n:
            raise ValueError('Frozen clean source has insufficient coverage')
        counts[source] = n
        for vid in selected:
            pilot[vid] = screen[vid]
    for vid in pr.ids('h1p_videos'):
        pilot[vid] = screen[vid]
    pilot_rows = [pilot[v] for v in sorted(pilot)]
    rest = [screen[v] for v in sorted(screen) if v not in pilot or screen[v] != pilot[v]]
    # Rest includes longer intervals for pilot positives, so the winning stream covers all S ticks.
    timing_ids = ordered(pilot, 'vlm-timing-v1-')[:200]
    timing = [{**pilot[v], 't_sec': min(130., pilot[v]['cap_sec'] - 1.)} for v in timing_ids]
    output = dict(screen=write_manifest('screen_S', list(screen.values())),
                  pilot=write_manifest('pilot_P', pilot_rows), rest=write_manifest('rest_S', rest),
                  timing=write_manifest('timing_queries', timing),
                  registration_sha256=pr.sha256, screening_folds=FOLDS, confirmation_fold=1,
                  selection='S_long60 first events + deterministic 150 clean (38 ucf,38 ubi,37 movie,37 youtube) + all H1p',
                  clean_counts=counts,
                  pilot_hours=sum(r['cap_sec'] - r['start_sec'] + 1 for r in pilot_rows) / 3600,
                  screen_hours=sum(r['cap_sec'] - r['start_sec'] + 1 for r in screen.values()) / 3600,
                  rest_hours=sum(r['cap_sec'] - r['start_sec'] + 1 for r in rest) / 3600)
    (ROOT / 'sets.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
