"""Join blind onset reviews and explicit reconciliations into validated v3 correction artifacts."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from anticipation.data.apply_corrections import apply, VERDICTS
from anticipation.data.onset_scan import dump


def join_reviews(flags, reviews, reconciliations):
    output = []
    reviewed = set().union(*(set(p) for p in reviews))
    for scan in flags:
        key = scan['video_id'], scan['event_index']
        original = scan['original_onset_sec']
        record = {k: scan[k] for k in ('video_id', 'event_index', 'onset_source', 'original_onset_sec')}
        passes = [p[key] for p in reviews if key in p]
        flagged = scan['early_violence'] or scan['no_violence_at_onset']
        if flagged and len(passes) != 2:
            raise ValueError(f'Flagged event requires both passes: {key}')
        if key not in reviewed:
            verdict, corrected = 'keep', original
            evidence = {'basis': 'scan_no_flag_not_visually_reviewed'}
        else:
            if len(passes) != 2:
                raise ValueError(f'Incomplete paired review: {key}')
            for p in passes:
                if p['verdict'] not in VERDICTS or not p.get('evidence') or p.get('needs_expansion'):
                    raise ValueError(f'Incomplete review: {key}')
            verdicts = {p['verdict'] for p in passes}
            changed = bool(verdicts & {'earlier', 'later', 'no_violence'})
            times = [p.get('corrected_onset_sec') for p in passes]
            disagree = len(verdicts) != 1 or (changed and 'no_violence' not in verdicts and (
                any(t is None for t in times) or abs(times[0] - times[1]) > 2))
            reconciliation = reconciliations.get(key)
            if disagree and not reconciliation:
                raise ValueError(f'Unreconciled review: {key}')
            if reconciliation:
                verdict = reconciliation['verdict']
                corrected = reconciliation.get('corrected_onset_sec')
                if not reconciliation.get('evidence'):
                    raise ValueError(f'Reconciliation lacks evidence: {key}')
                record['reconciliation'] = reconciliation
            else:
                verdict = passes[0]['verdict']
                corrected = float(np.round(np.mean(times) * 2) / 2) if verdict in {'earlier', 'later'} else None
            if verdict == 'keep':
                corrected = original
            evidence = {'basis': 'two_blind_visual_passes', 'passes': [p['evidence'] for p in passes]}
        record.setdefault('reconciliation', None)
        record.update(verdict=verdict, corrected_onset_sec=corrected, evidence=evidence, passes=passes,
                      scan_stats={k: v for k, v in scan.items() if k not in record},
                      visually_reviewed=bool(passes))
        output.append(record)
    return output


def load_pass(root, name):
    folder = root / 'blind' / name
    key = {r['blind_id']: (r['video_id'], r['event_index'])
           for r in json.loads((root / f'private_key_{name}.json').read_text())}
    rows = json.loads((folder / 'verdicts.json').read_text())
    if len(rows) != len(key) or {r['blind_id'] for r in rows} != set(key):
        raise ValueError(f'Pass {name} must review exactly the blinded queue')
    return {key[r['blind_id']]: r for r in rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('data/annotation/onset_v3'))
    ap.add_argument('--passes', nargs=2, default=['A', 'B'])
    ap.add_argument('--out', type=Path, default=Path('benchmark/labels_v3'))
    args = ap.parse_args()
    runtime = json.loads((args.root / 'runtime.json').read_text())
    if not runtime['complete']:
        raise ValueError('Scan incomplete')
    flags = json.loads((args.root / 'flags.json').read_text())
    queue = json.loads((args.root / 'review_queue.json').read_text())
    random = {(r['video_id'], r['event_index']) for r in queue if r['review_reason'] == 'random_nonflagged'}
    if len(random) != 40:
        raise ValueError('Exactly 40 non-flagged sample events required')
    reconciliation_path = args.root / 'reconciliations.json'
    reconciliations = json.loads(reconciliation_path.read_text()) if reconciliation_path.exists() else []
    reconciliations = {(r['video_id'], r['event_index']): r for r in reconciliations}
    reviews = [load_pass(args.root, name) for name in args.passes]
    corrections = join_reviews(flags, reviews, reconciliations)
    videos = json.loads(Path('benchmark/events.json').read_text())['videos']
    catalog = {r['video_id']: r for r in csv.DictReader(open('benchmark/videos.csv'))}
    expected = {(v, e['index']) for v, row in videos.items() for e in row.get('events', []) if e['eligible30']}
    actual = {(r['video_id'], r['event_index']) for r in corrections}
    if actual != expected:
        raise ValueError('Scan event universe differs from source eligible30 events')
    if set(reviews[0]) != set(reviews[1]) or set(reviews[0]) != {
            (r['video_id'], r['event_index']) for r in queue}:
        raise ValueError('Blind review coverage differs from planned queue')
    apply(videos, catalog, corrections)
    shifts = [r['corrected_onset_sec'] - r['original_onset_sec'] for r in corrections
              if r['verdict'] in {'earlier', 'later'}]
    sample = [r for r in corrections if (r['video_id'], r['event_index']) in random]
    counts = Counter((r['verdict'], r['onset_source']) for r in corrections)
    summary = {'n_events': len(corrections), 'n_visually_reviewed': len(reviews[0]),
               'counts_verdict_by_source': {v: {s: counts[v, s] for s in ('official', 'hivau', 'ubi')}
                                          for v in sorted(VERDICTS)},
               'median_shift_sec': float(np.median(shifts)) if shifts else None,
               'nonflagged_review_n': len(sample),
               'scan_misses': sum(r['verdict'] in {'earlier', 'later', 'no_violence'} for r in sample),
               'sample_unclear': sum(r['verdict'] == 'unclear' for r in sample)}
    summary['scan_miss_rate'] = summary['scan_misses'] / len(sample)
    result = {'version': 3, 'status': 'REVIEW_COMPLETE',
              'source_sha256': {n: hashlib.sha256((Path('benchmark') / n).read_bytes()).hexdigest()
                                for n in ('videos.csv', 'events.json')},
              'scan_manifest': json.loads((args.root / 'manifest.json').read_text()),
              'calibration': json.loads((args.root / 'calibration.json').read_text()),
              'summary': summary, 'corrections': corrections}
    args.out.mkdir(parents=True, exist_ok=True)
    dump(args.out / 'onset_corrections.json', result)
    with (args.out / 'onset_corrections.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(corrections[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()}
                         for r in corrections)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
