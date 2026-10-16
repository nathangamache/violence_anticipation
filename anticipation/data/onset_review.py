"""Render blinded contact sheets and review templates for the purpose-specific onset scan."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from anticipation.data.onset_scan import dump


def prepare(root, pass_name, seed, cached_only=False, workers=1, followups=False):
    from tools.contact_sheet import render_sheets
    suffix = '_followup' if followups else ''
    queue_path = root / ('followup_queue.json' if followups else 'review_queue.json')
    queue = json.loads(queue_path.read_text())
    runtime = json.loads((root / 'runtime.json').read_text())
    if not runtime['complete']:
        raise ValueError('The complete scan is required before sampling and blinding review')
    catalog = {r['video_id']: r for r in csv.DictReader(open('benchmark/videos.csv'))}
    videos = json.loads(Path('benchmark/events.json').read_text())['videos']
    out = root / 'blind' / pass_name
    if followups:
        out = out / 'followup'
    out.mkdir(parents=True, exist_ok=True)
    metadata = {'seed': seed, 'queue_sha256': hashlib.sha256(queue_path.read_bytes()).hexdigest()}
    metadata_path = root / f'private_meta_{pass_name}{suffix}.json'
    private_path = root / f'private_key_{pass_name}{suffix}.json'
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous['seed'] != seed or (not followups and previous != metadata):
            raise ValueError('Existing blind pass has a different seed or review queue')
        if followups and private_path.exists():
            previous_keys = {(r['video_id'], r['event_index']) for r in json.loads(private_path.read_text())}
            if not previous_keys <= {(r['video_id'], r['event_index']) for r in queue}:
                raise ValueError('Follow-up queues may only grow')
    dump(metadata_path, metadata)
    order = np.random.default_rng(seed).permutation(len(queue))
    private, public = [], []
    def render_case(pair):
        i, j = pair
        item = queue[j]
        vid, index = item['video_id'], item['event_index']
        event = videos[vid]['events'][index]
        duration = videos[vid]['duration_sec']
        token = hashlib.sha256(f'{seed}:{vid}:{index}'.encode()).hexdigest()[:12]
        blind_id = f'{pass_name}F{token}' if followups else f'{pass_name}{i:04d}'
        start = max(0 if followups else event['clear_start'], event['onset'] - 120)
        end = min(duration, max(event['onset'] + 18, event['end'] + 3))
        candidate = item.get('candidate_onset_sec')
        center = event['onset'] if candidate is None else candidate
        folder = out / blind_id
        folder.mkdir(exist_ok=True)
        alias = folder / 'raw.mp4'
        if not alias.exists():
            alias.symlink_to(catalog[vid]['path'])
        cache_id = hashlib.sha256(f'{vid}:{index}'.encode()).hexdigest()[:20]
        cache = root / 'sheet_cache' / cache_id
        cache.mkdir(parents=True, exist_ok=True)
        done = cache / 'sheets.json'
        if cached_only and not done.exists():
            return None
        if done.exists():
            sheets = json.loads(done.read_text())
        else:
            coarse = render_sheets(catalog[vid]['path'], str(cache), start=start,
                                   end=end, step=1, tile_w=256, prefix='coarse', num_threads=1)
            fine = render_sheets(catalog[vid]['path'], str(cache), start=max(0, center - 4),
                                 end=min(duration, center + 6), step=.5, tile_w=320, prefix='fine', num_threads=1)
            sheets = {'coarse': coarse, 'fine': fine}
            dump(done, sheets)
        linked = {}
        for kind, paths in sheets.items():
            linked[kind] = []
            for path in paths:
                target = folder / Path(path).name
                if not target.exists():
                    os.link(path, target)
                linked[kind].append(str(target))
        public_row = {'blind_id': blind_id, 'video_path': str(alias), 'original_onset_sec': event['onset'],
                      'clear_start_sec': event['clear_start'], 'event_end_sec': event['end'],
                      'prior_annotated_segments': videos[vid]['segments'][:index],
                      'following_annotated_segments': videos[vid]['segments'][index + 1:],
                      'sheets': linked, 'verdict': None, 'corrected_onset_sec': None,
                      'evidence': None, 'needs_expansion': None}
        private_row = {'blind_id': blind_id, 'video_id': vid, 'event_index': index,
                       'review_reason': item['review_reason']}
        return public_row, private_row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(render_case, enumerate(order)):
            if result is None:
                continue
            public.append(result[0])
            private.append(result[1])
            dump(out / 'review_template.json', public)
            dump(private_path, private)
            print(f'{pass_name}: {len(public)}/{len(queue)}', flush=True)



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('data/annotation/onset_v3'))
    ap.add_argument('--pass-name', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--cached-only', action='store_true')
    ap.add_argument('--workers', type=int, default=1, choices=range(1, 5))
    ap.add_argument('--followups', action='store_true')
    args = ap.parse_args()
    prepare(args.root, args.pass_name, args.seed, args.cached_only, args.workers, args.followups)


if __name__ == '__main__':
    main()
