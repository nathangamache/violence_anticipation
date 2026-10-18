"""Join blind onset reviews and explicit reconciliations into validated v3 correction artifacts."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np

from anticipation.data.apply_corrections import apply, VERDICTS
from anticipation.data.onset_scan import dump


AUDIT_SEED = 301018
AUDIT_FRACTION = 0.10


def audit_keys(review_a, fraction=AUDIT_FRACTION, seed=AUDIT_SEED):
    """Choose a fixed, seeded sample of A's unchanged keep/unclear judgments."""
    eligible = sorted(key for key, row in review_a.items()
                      if row['verdict'] in {'keep', 'unclear'} and not row.get('additional_violence_spans'))
    count = math.ceil(len(eligible) * fraction) if eligible else 0
    return set(random.Random(seed).sample(eligible, count))


def audit_keep_stats(review_a, review_b, selected):
    keep_keys = sorted(key for key in selected if key in review_a and review_a[key]['verdict'] == 'keep')
    # Owner amendment 2026-10-18 (post hoc, documented): B 'unclear' does not contradict an A keep.
    disagreements = [key for key in keep_keys if key not in review_b
                     or review_b[key]['verdict'] in {'earlier', 'later', 'no_violence'}
                     or bool(review_b[key].get('additional_violence_spans'))]
    return {'seed': AUDIT_SEED, 'fraction': AUDIT_FRACTION, 'n_sampled_keep': len(keep_keys),
            'n_keep_disagreements': len(disagreements),
            'disagreement_rate': len(disagreements) / len(keep_keys) if keep_keys else None,
            'stop_threshold': 0.15,
            'stop': bool(keep_keys and len(disagreements) / len(keep_keys) > 0.15)}


def join_reviews(flags, review_a, review_b, reconciliations, required_b=()):
    output = []
    required_b = set(required_b)
    for scan in flags:
        key = scan['video_id'], scan['event_index']
        original = scan['original_onset_sec']
        record = {k: scan[k] for k in ('video_id', 'event_index', 'onset_source', 'original_onset_sec')}
        a = review_a.get(key)
        b = review_b.get(key)
        pass_records = {**({'A': a} if a else {}), **({'B': b} if b else {})}
        flagged = scan['early_violence'] or scan['no_violence_at_onset']
        if a is None:
            if key in required_b or flagged:
                raise ValueError(f'Flagged or selected event lacks Pass A: {key}')
            verdict, corrected = 'keep', original
            evidence = {'basis': 'scan_no_flag_not_visually_reviewed'}
            reconciled = None
        elif key in required_b and b is None:
            raise ValueError(f'Selected event requires Pass B: {key}')
        else:
            for name, p in pass_records.items():
                # Owner amendment 2026-10-18: needs_expansion only blocks label-changing, unreconciled reviews.
                expansion_blocks = (p.get('needs_expansion') and key not in reconciliations
                                    and p['verdict'] not in {'keep', 'unclear'})
                if p['verdict'] not in VERDICTS or not p.get('evidence') or expansion_blocks:
                    raise ValueError(f'Incomplete Pass {name} review: {key}')
            if b is not None and a['blind_id'] == b['blind_id']:
                raise ValueError(f'Pass A and B must use independent blind identifiers: {key}')
            changed_a = a['verdict'] in {'earlier', 'later', 'no_violence'} or bool(a.get('additional_violence_spans'))
            if changed_a and b is None:
                raise ValueError(f'Label-changing Pass A review requires Pass B: {key}')
            disagree = False
            if b is not None:
                disagree = a['verdict'] != b['verdict']
                if a['verdict'] in {'earlier', 'later'} and b['verdict'] == a['verdict']:
                    ta, tb = a.get('corrected_onset_sec'), b.get('corrected_onset_sec')
                    disagree = ta is None or tb is None or abs(ta - tb) > 2
                if a.get('additional_violence_spans', []) != b.get('additional_violence_spans', []):
                    disagree = True
            reconciliation = reconciliations.get(key)
            additions_proposed = any(p.get('additional_violence_spans') for p in pass_records.values())
            if (disagree or additions_proposed) and not reconciliation:
                raise ValueError(f'Unreconciled review: {key}')
            if reconciliation:
                verdict = reconciliation['verdict']
                corrected = reconciliation.get('corrected_onset_sec')
                if not reconciliation.get('evidence'):
                    raise ValueError(f'Reconciliation lacks evidence: {key}')
                record['reconciliation'] = reconciliation
            else:
                verdict = a['verdict']
                if verdict in {'earlier', 'later'}:
                    corrected = float(np.round(np.mean([a['corrected_onset_sec'], b['corrected_onset_sec']]) * 2) / 2)
                else:
                    corrected = None
            if verdict == 'keep':
                corrected = original
            evidence = {'basis': 'two_blind_visual_passes' if b is not None else 'single_blind_visual_pass',
                        'A': a['evidence'], **({'B': b['evidence']} if b else {})}
            reconciled = record.get('reconciliation')
        record.setdefault('reconciliation', None)
        record['additional_violence_spans'] = (reconciled or {}).get('additional_violence_spans', [])
        record.update(verdict=verdict, corrected_onset_sec=corrected, evidence=evidence,
                      passes=list(pass_records), pass_evidence=pass_records,
                      scan_stats={k: v for k, v in scan.items() if k not in record},
                      visually_reviewed=a is not None)
        output.append(record)
    return output


def load_pass(root, name, followups=False):
    folder = root / 'blind' / name
    suffix = '_followup' if followups else ''
    if followups:
        folder = folder / 'followup'
    key = {r['blind_id']: (r['video_id'], r['event_index'])
           for r in json.loads((root / f'private_key_{name}{suffix}.json').read_text())}
    verdict_path = folder / 'verdicts.json'
    rows = json.loads(verdict_path.read_text()) if verdict_path.exists() else []
    ids = [r['blind_id'] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) - set(key):
        raise ValueError(f'Pass {name} contains duplicate or unknown blind IDs')
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
    review_a = load_pass(args.root, 'A')
    review_b = load_pass(args.root, 'B')
    followup_a = load_pass(args.root, 'A', followups=True)
    followup_b = load_pass(args.root, 'B', followups=True)
    videos = json.loads(Path('benchmark/events.json').read_text())['videos']
    catalog = {r['video_id']: r for r in csv.DictReader(open('benchmark/videos.csv'))}
    expected = {(v, e['index']) for v, row in videos.items() for e in row.get('events', []) if e['eligible30']}
    actual = {(r['video_id'], r['event_index']) for r in flags}
    if actual != expected:
        raise ValueError('Scan event universe differs from source eligible30 events')
    followup_keys = {(r['video_id'], r['event_index']) for r in followups}
    if len(followup_keys) != len(followups) or followup_keys & actual:
        raise ValueError('Duplicate or originally eligible follow-up event')
    expected_a = {(r['video_id'], r['event_index']) for r in queue} | followup_keys
    if set(review_a) | set(followup_a) != expected_a or set(review_a) & set(followup_a):
        raise ValueError('Pass A review coverage differs from the planned queue')
    review_a.update(followup_a)
    review_b.update(followup_b)
    if set(followup_a) != followup_keys or set(followup_b) != followup_keys:
        raise ValueError('Every newly eligible follow-up requires both blind reviews')
    selected_audit = audit_keys({k: review_a[k] for k in expected_a if k in review_a and k not in followup_keys})
    changing = {key for key, row in review_a.items()
                if key in actual | followup_keys and
                (row['verdict'] in {'earlier', 'later', 'no_violence'} or row.get('additional_violence_spans'))}
    required_b = changing | selected_audit | followup_keys
    prior_path = args.root / 'b_prior_completed.json'
    prior_b = {(r['video_id'], r['event_index']) for r in json.loads(prior_path.read_text())} if prior_path.exists() else set()
    assignment_path = args.root / 'b_assignment_state.json'
    assigned_b = ({(r['video_id'], r['event_index']) for r in json.loads(assignment_path.read_text()).get('assigned', [])}
                  if assignment_path.exists() else set())
    if not required_b <= set(review_b):
        missing = sorted(required_b - set(review_b))
        raise ValueError(f'Pass B coverage incomplete; {len(missing)} selected reviews missing: {missing[:12]}')
    if set(review_b) - required_b - prior_b - assigned_b:
        raise ValueError('Pass B contains reviews outside the selection policy and recorded prior assignments')
    keep_audit = audit_keep_stats(review_a, review_b, selected_audit)
    dump(args.root / 'keep_audit_summary.json', keep_audit)
    print(json.dumps({'keep_audit': keep_audit}, indent=2))
    if keep_audit['stop']:
        raise ValueError('Keep audit disagreement exceeds 15%; label release stopped')
    flags += followups
    queue += followups
    corrections = join_reviews(flags, review_a, review_b, reconciliations, required_b)
    corrected, _ = apply(videos, catalog, corrections)
    pending = newly_eligible_unreviewed(videos, corrected, set(review_a))
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
               'n_visually_reviewed': len(review_a),
               'n_one_pass': sum(len(r['passes']) == 1 for r in corrections),
               'n_two_pass': sum(len(r['passes']) == 2 for r in corrections),
               'n_audit_sample': len(selected_audit),
               'n_keep_audit': keep_audit['n_sampled_keep'],
               'keep_audit_disagreement_rate': keep_audit['disagreement_rate'],
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
