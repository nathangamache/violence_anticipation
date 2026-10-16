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
            additions_proposed = any(p.get('additional_violence_spans') for p in passes)
            if (disagree or additions_proposed) and not reconciliation:
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
        record['additional_violence_spans'] = (record['reconciliation'] or {}).get('additional_violence_spans', [])
        record.update(verdict=verdict, corrected_onset_sec=corrected, evidence=evidence, passes=passes,
                      scan_stats={k: v for k, v in scan.items() if k not in record},
                      visually_reviewed=bool(passes))
        output.append(record)
    return output


def load_pass(root, name, followups=False):
    folder = root / 'blind' / name
    suffix = '_followup' if followups else ''
    if followups:
        folder = folder / 'followup'
    key = {r['blind_id']: (r['video_id'], r['event_index'])
           for r in json.loads((root / f'private_key_{name}{suffix}.json').read_text())}
    rows = json.loads((folder / 'verdicts.json').read_text())
    if len(rows) != len(key) or {r['blind_id'] for r in rows} != set(key):
        raise ValueError(f'Pass {name} must review exactly the blinded queue')
    if any(r.get('scope_protocol') != 'episode_v2' for r in rows):
        raise ValueError(f'Pass {name} must complete the episode-scope audit')
    return {key[r['blind_id']]: r for r in rows}


def newly_eligible_unreviewed(original, corrected, reviewed):
    pending = []
    for vid, entry in corrected.items():
        old = {round(e['onset'], 4): e for e in original[vid].get('events', [])}
        for event in entry.get('events', []):
            source = old.get(round(event['onset'], 4))
            if event['eligible30'] and source and not source['eligible30'] and (vid, source['index']) not in reviewed:
                pending.append((vid, source['index']))
    return pending


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
    n_nonflagged = sum(not (r['early_violence'] or r['no_violence_at_onset']) for r in flags)
    if len(random) != min(40, n_nonflagged):
        raise ValueError('Non-flagged sample size differs from the planned 40-event sample or census')
    n_scanned = len(flags)
    followup_path = args.root / 'followup_queue.json'
    followups = json.loads(followup_path.read_text()) if followup_path.exists() else []
    if any(r.get('scanned') is not False or r['review_reason'] != 'newly_eligible_followup' for r in followups):
        raise ValueError('Follow-ups must explicitly distinguish direct visual review from Qwen scanning')
    reconciliation_path = args.root / 'reconciliations.json'
    reconciliations = json.loads(reconciliation_path.read_text()) if reconciliation_path.exists() else []
    reconciliations = {(r['video_id'], r['event_index']): r for r in reconciliations}
    reviews = [load_pass(args.root, name) for name in args.passes]
    videos = json.loads(Path('benchmark/events.json').read_text())['videos']
    catalog = {r['video_id']: r for r in csv.DictReader(open('benchmark/videos.csv'))}
    expected = {(v, e['index']) for v, row in videos.items() for e in row.get('events', []) if e['eligible30']}
    actual = {(r['video_id'], r['event_index']) for r in flags}
    if actual != expected:
        raise ValueError('Scan event universe differs from source eligible30 events')
    followup_keys = {(r['video_id'], r['event_index']) for r in followups}
    if len(followup_keys) != len(followups) or followup_keys & actual:
        raise ValueError('Duplicate or originally eligible follow-up event')
    if followups:
        for name, review in zip(args.passes, reviews):
            supplement = load_pass(args.root, name, followups=True)
            if set(supplement) != followup_keys:
                raise ValueError('Follow-up review coverage differs from its queue')
            review.update(supplement)
    flags += followups
    queue += followups
    if set(reviews[0]) != set(reviews[1]) or set(reviews[0]) != {
            (r['video_id'], r['event_index']) for r in queue}:
        raise ValueError('Blind review coverage differs from planned queue')
    corrections = join_reviews(flags, reviews, reconciliations)
    corrected, _ = apply(videos, catalog, corrections)
    pending = newly_eligible_unreviewed(videos, corrected, set(reviews[0]))
    if pending:
        raise ValueError(f'Newly eligible source events require blind follow-up review: {pending}')
    shifts = [r['corrected_onset_sec'] - r['original_onset_sec'] for r in corrections
              if r['verdict'] in {'earlier', 'later'}]
    sample = [r for r in corrections if (r['video_id'], r['event_index']) in random]
    counts = Counter((r['verdict'], r['onset_source']) for r in corrections)
    reviewed_counts = Counter((r['verdict'], r['onset_source']) for r in corrections if r['visually_reviewed'])
    summary = {'n_events': len(corrections), 'n_scanned_events': n_scanned,
               'n_followup_events': len(followups),
               'n_retained_without_visual_review': sum(not r['visually_reviewed'] for r in corrections),
               'n_visually_reviewed': len(reviews[0]),
               'counts_verdict_by_source': {v: {s: counts[v, s] for s in ('official', 'hivau', 'ubi')}
                                          for v in sorted(VERDICTS)},
               'visually_reviewed_counts_verdict_by_source': {
                   v: {s: reviewed_counts[v, s] for s in ('official', 'hivau', 'ubi')}
                   for v in sorted(VERDICTS)},
               'median_shift_sec': float(np.median(shifts)) if shifts else None,
               'nonflagged_review_n': len(sample),
               'additional_violence_spans': sum(len(r['additional_violence_spans']) for r in corrections),
               'scan_misses': sum(r['verdict'] in {'earlier', 'later', 'no_violence'} or
                                  bool(r['additional_violence_spans']) for r in sample),
               'nonflagged_census': n_nonflagged <= 40,
               'sample_unclear': sum(r['verdict'] == 'unclear' for r in sample)}
    summary['scan_miss_rate'] = summary['scan_misses'] / len(sample) if sample else None
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
