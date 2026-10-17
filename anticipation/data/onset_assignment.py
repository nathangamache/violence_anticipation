"""Append Pass B assignments selected by the owner-approved onset review protocol."""

import argparse
import json
from pathlib import Path

from anticipation.data.onset_finalize import audit_keys
from anticipation.data.onset_scan import dump


def _private_map(path):
    return {row['blind_id']: (row['video_id'], row['event_index']) for row in json.loads(path.read_text())}


def assign(root, include_audit=False):
    root = Path(root)
    blind = root / 'blind'
    queue = json.loads((root / 'review_queue.json').read_text())
    followups = json.loads((root / 'followup_queue.json').read_text()) if (root / 'followup_queue.json').exists() else []
    source_keys = {(row['video_id'], row['event_index']) for row in queue}
    followup_keys = {(row['video_id'], row['event_index']) for row in followups}

    a_map = _private_map(root / 'private_key_A.json')
    a_rows = json.loads((blind / 'A' / 'verdicts.json').read_text())
    a_reviews = {a_map[row['blind_id']]: row for row in a_rows}
    a_follow_map_path = root / 'private_key_A_followup.json'
    a_follow_map = _private_map(a_follow_map_path) if a_follow_map_path.exists() else {}
    a_follow_path = blind / 'A' / 'followup' / 'verdicts.json'
    if a_follow_path.exists():
        a_reviews.update({a_follow_map[row['blind_id']]: row for row in json.loads(a_follow_path.read_text())})

    changing = {key for key, row in a_reviews.items() if key in source_keys | followup_keys and
                (row['verdict'] in {'earlier', 'later', 'no_violence'} or row.get('additional_violence_spans'))}
    selected = set(changing)
    if include_audit:
        if not source_keys <= set(a_reviews):
            raise ValueError('Pass A must finish all original review cases before assigning its 10% audit')
        selected |= audit_keys({key: a_reviews[key] for key in source_keys})
    selected |= followup_keys

    b_map = _private_map(root / 'private_key_B.json')
    b_follow_map_path = root / 'private_key_B_followup.json'
    b_follow_map = _private_map(b_follow_map_path) if b_follow_map_path.exists() else {}
    b_done_path = blind / 'B' / 'verdicts.json'
    b_done = json.loads(b_done_path.read_text()) if b_done_path.exists() else []
    b_follow_done_path = blind / 'B' / 'followup' / 'verdicts.json'
    b_follow_done = json.loads(b_follow_done_path.read_text()) if b_follow_done_path.exists() else []
    b_done_keys = {b_map[row['blind_id']] for row in b_done}
    b_done_keys |= {b_follow_map[row['blind_id']] for row in b_follow_done}

    b_templates = json.loads((blind / 'B' / 'review_template.json').read_text())
    b_template_by_key = {b_map[row['blind_id']]: row for row in b_templates if row['blind_id'] in b_map}
    follow_template_path = blind / 'B' / 'followup' / 'review_template.json'
    follow_templates = json.loads(follow_template_path.read_text()) if follow_template_path.exists() else []
    follow_template_by_key = {b_follow_map[row['blind_id']]: row for row in follow_templates
                              if row['blind_id'] in b_follow_map}

    state_path = root / 'b_assignment_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'version': 1, 'assigned': [], 'audit_seed': 301018, 'audit_fraction': 0.10}
    assigned_keys = {(row['video_id'], row['event_index']) for row in state['assigned']}
    new_main, new_followups = [], []
    for key in sorted(selected - b_done_keys - assigned_keys):
        if key in followup_keys:
            if key not in follow_template_by_key:
                raise ValueError(f'Pass B follow-up template is missing {key}; refresh it before assigning')
            template = follow_template_by_key[key]
            new_followups.append(template)
            state['assigned'].append({'video_id': key[0], 'event_index': key[1], 'reason': 'followup'})
        else:
            if key not in b_template_by_key:
                raise ValueError(f'Pass B original template is missing {key}')
            template = b_template_by_key[key]
            new_main.append(template)
            reason = 'label_change' if key in changing else 'audit_sample'
            state['assigned'].append({'video_id': key[0], 'event_index': key[1], 'reason': reason})

    def append_rows(path, rows):
        prior = json.loads(path.read_text()) if path.exists() else []
        ids = {row['blind_id'] for row in prior}
        prior.extend(row for row in rows if row['blind_id'] not in ids)
        dump(path, prior)

    append_rows(blind / 'B' / 'assigned_v3.json', new_main)
    append_rows(blind / 'B' / 'followup' / 'assigned_v3.json', new_followups)
    dump(state_path, state)
    return {'new_main_assignments': len(new_main), 'new_followup_assignments': len(new_followups),
            'selected_label_changes': len(changing), 'audit_added': include_audit,
            'already_selected': len(assigned_keys), 'completed_b': len(b_done_keys)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/annotation/onset_v3'))
    parser.add_argument('--include-audit', action='store_true')
    args = parser.parse_args()
    print(json.dumps(assign(args.root, args.include_audit), indent=2))


if __name__ == '__main__':
    main()
