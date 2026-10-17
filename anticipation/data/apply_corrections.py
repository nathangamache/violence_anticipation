"""Apply reviewed onset corrections to a separate VA30 catalog and recompute eligibility."""

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path

from anticipation.data.annotations import parse_xd_source_span
from anticipation.data.catalog import source_masks
from anticipation.data.labels import blocker_spans, event_table, normalize_segments

VERDICTS = {'keep', 'earlier', 'later', 'no_violence', 'unclear'}


def refresh_source_masks(videos, catalog):
    records = []
    for vid, row in catalog.items():
        if row.get('source') != 'xd':
            continue
        span = parse_xd_source_span(vid.split('/', 1)[1])
        if span is None:
            continue
        entry = videos[vid]
        records.append(dict(vid=vid, src='xd', span=span, dur=entry['duration_sec'], group=row['group'],
                            role=entry['role'], segs=entry['segments'], masks=[], soft_masks=[], notes=[]))
    source_masks(records, log=lambda *args: None)
    affected = set()
    for record in records:
        vid = record['vid']
        entry, row = videos[vid], catalog[vid]
        for field, key in [('mask_spans', 'masks'), ('soft_mask_spans', 'soft_masks')]:
            if entry.get(field, []) == record[key]:
                continue
            affected.add(vid)
            if record[key]:
                entry[field] = record[key]
            else:
                entry.pop(field, None)
        if entry['role'] != record['role']:
            affected.add(vid)
            entry['role'] = row['role'] = record['role']
            row['is_clean_negative'] = 'False'
        if vid in affected:
            spans = normalize_segments(record['masks'] + record['soft_masks'], entry['duration_sec'])
            row['masked_sec'] = str(round(sum(e - s for s, e in spans), 3))
            row['notes'] = row.get('notes', '') + ' | v3 source masks recomputed from corrected segments'
    return affected


def apply(videos, catalog, corrections):
    videos, catalog = copy.deepcopy(videos), copy.deepcopy(catalog)
    grouped, seen = {}, set()
    for correction in corrections:
        vid, index = correction['video_id'], correction['event_index']
        key = vid, index
        if key in seen or not isinstance(index, int) or index < 0:
            raise ValueError(f'Duplicate or invalid event key: {key}')
        seen.add(key)
        if vid not in videos or vid not in catalog or videos[vid]['role'] != 'violent':
            raise ValueError(f'Unknown violent video: {vid}')
        entry = videos[vid]
        if index >= len(entry['segments']):
            raise ValueError(f'Unknown event: {key}')
        original = float(correction['original_onset_sec'])
        if not math.isclose(original, entry['segments'][index][0], abs_tol=1e-4):
            raise ValueError(f'Original onset mismatch: {key}')
        if correction['onset_source'] != entry['onset_source']:
            raise ValueError(f'Onset source mismatch: {key}')
        verdict = correction['verdict']
        if verdict not in VERDICTS:
            raise ValueError(f'Unknown verdict: {verdict}')
        additions = correction.get('additional_violence_spans', [])
        if additions:
            reconciliation = correction.get('reconciliation')
            if not reconciliation or not reconciliation.get('evidence'):
                raise ValueError(f'Additional episodes require visual reconciliation: {key}')
            if additions != reconciliation.get('additional_violence_spans'):
                raise ValueError(f'Additional episodes differ from reconciliation: {key}')
            for start, end in additions:
                if not all(math.isfinite(t) for t in (start, end)) or not 0 <= start < end <= entry['duration_sec']:
                    raise ValueError(f'Invalid additional episode: {key}')
        passes = correction.get('passes', [])
        pass_evidence = correction.get('pass_evidence', {})
        if passes not in (['A'], ['A', 'B'], []):
            raise ValueError(f'Pass list must be A or A/B: {key}')
        if set(pass_evidence) != set(passes):
            raise ValueError(f'Pass evidence does not match pass list: {key}')
        if passes and not correction.get('evidence'):
            raise ValueError(f'Evidence required: {key}')
        if verdict in {'earlier', 'later', 'no_violence'} or additions:
            if passes != ['A', 'B']:
                raise ValueError(f'Two blind passes required for label changes: {key}')
        if len(passes) == 2:
            a, b = pass_evidence['A'], pass_evidence['B']
            if a['blind_id'] == b['blind_id']:
                raise ValueError(f'Pass A and B must use independent blind identifiers: {key}')
            disagree = a['verdict'] != b['verdict']
            if a['verdict'] in {'earlier', 'later'} and b['verdict'] == a['verdict']:
                ta, tb = a.get('corrected_onset_sec'), b.get('corrected_onset_sec')
                disagree = ta is None or tb is None or abs(ta - tb) > 2
            if a.get('additional_violence_spans', []) != b.get('additional_violence_spans', []):
                disagree = True
            if (disagree or additions) and not correction.get('reconciliation'):
                raise ValueError(f'Unreconciled disagreement: {key}')
        elif passes == ['A'] and pass_evidence['A']['verdict'] != verdict and not correction.get('reconciliation'):
            raise ValueError(f'Single-pass label differs from Pass A: {key}')
        if verdict in {'earlier', 'later'}:
            onset = float(correction['corrected_onset_sec'])
            if not math.isfinite(onset) or not 0 <= onset <= entry['segments'][index][1]:
                raise ValueError(f'Corrected onset outside segment: {key}')
            if (verdict == 'earlier' and onset >= original) or (verdict == 'later' and onset <= original):
                raise ValueError(f'Correction direction mismatch: {key}')
        elif correction.get('corrected_onset_sec') not in (None, original):
            raise ValueError(f'{verdict} cannot change the onset: {key}')
        grouped.setdefault(vid, {})[index] = correction
    for vid, changes in grouped.items():
        entry, row = videos[vid], catalog[vid]
        segments = []
        for index, (onset, end) in enumerate(entry['segments']):
            correction = changes.get(index, {})
            segments.extend(correction.get('additional_violence_spans', []))
            verdict = correction.get('verdict', 'keep')
            if verdict == 'no_violence':
                continue
            if verdict in {'earlier', 'later'}:
                onset = float(correction['corrected_onset_sec'])
            segments.append([onset, end])
        entry['segments'] = normalize_segments(segments, entry['duration_sec'])
        if not entry['segments']:
            entry['role'] = row['role'] = 'unlabelled'
            row['is_clean_negative'] = 'False'
    affected = set(grouped) | refresh_source_masks(videos, catalog)
    for vid in affected:
        entry, row = videos[vid], catalog[vid]
        events = event_table(entry['segments'], blockers=blocker_spans(
            entry.get('other_segments'), entry.get('mask_spans'))) if entry['role'] == 'violent' else []
        entry['events'] = events
        for lead in (30, 60):
            count = sum(e[f'eligible{lead}'] for e in events)
            row[f'n_eligible{lead}'] = str(count)
            row[f'eligible{lead}'] = str(bool(count))
        if vid in grouped:
            row['n_segments'] = str(len(entry['segments']))
            row['first_onset_sec'] = str(entry['segments'][0][0]) if entry['segments'] else ''
            starts = [s for s, _ in entry['segments'] + entry.get('other_segments', [])]
            row['first_anomaly_sec'] = str(min(starts)) if starts else ''
    return videos, catalog


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--corrections', default='benchmark/labels_v3/onset_corrections.json')
    ap.add_argument('--source', default='benchmark')
    ap.add_argument('--out', default='benchmark/v3')
    args = ap.parse_args(argv)
    source, out = Path(args.source).resolve(), Path(args.out).resolve()
    if out == source:
        raise ValueError('Output must differ from the pinned source catalog')
    raw = json.loads((source / 'events.json').read_text())
    with (source / 'videos.csv').open(newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        catalog = {r['video_id']: r for r in reader}
    correction_path = Path(args.corrections)
    corrections = json.loads(correction_path.read_text())
    if corrections.get('status') != 'REVIEW_COMPLETE':
        raise ValueError('Corrections are not marked REVIEW_COMPLETE')
    for name in ('events.json', 'videos.csv'):
        actual = hashlib.sha256((source / name).read_bytes()).hexdigest()
        if corrections['source_sha256'][name] != actual:
            raise ValueError(f'Source hash mismatch: {name}')
    videos, catalog = apply(raw['videos'], catalog, corrections['corrections'])
    raw['videos'] = videos
    raw['meta']['labels_version'] = 3
    raw['meta']['onset_corrections'] = {'sha256': hashlib.sha256(correction_path.read_bytes()).hexdigest(),
                                       'path': args.corrections, 'version': 3}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'events.json').write_text(json.dumps(raw, indent=1) + '\n')
    with (out / 'videos.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(catalog.values())


if __name__ == '__main__':
    main()
