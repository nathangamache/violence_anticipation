"""Render blinded contact sheets and review templates for the purpose-specific onset scan."""

import argparse
import csv
import hashlib
import os
import json
from pathlib import Path

import numpy as np

from anticipation.data.onset_scan import dump


def prepare(root, pass_name, seed):
    from tools.contact_sheet import render_sheets
    queue = json.loads((root / 'review_queue.json').read_text())
    runtime = json.loads((root / 'runtime.json').read_text())
    if not runtime['complete']:
        raise ValueError('The complete scan is required before sampling and blinding review')
    catalog = {r['video_id']: r for r in csv.DictReader(open('benchmark/videos.csv'))}
    videos = json.loads(Path('benchmark/events.json').read_text())['videos']
    out = root / 'blind' / pass_name
    out.mkdir(parents=True, exist_ok=True)
    metadata = {'seed': seed, 'queue_sha256': hashlib.sha256((root / 'review_queue.json').read_bytes()).hexdigest()}
    metadata_path = root / f'private_meta_{pass_name}.json'
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError('Existing blind pass has a different seed or review queue')
    dump(metadata_path, metadata)
    order = np.random.default_rng(seed).permutation(len(queue))
    private, public = [], []
    for i, j in enumerate(order):
        item = queue[j]
        vid, index = item['video_id'], item['event_index']
        event = videos[vid]['events'][index]
        duration = videos[vid]['duration_sec']
        blind_id = f'{pass_name}{i:04d}'
        start = max(event['clear_start'], event['onset'] - 120)
        end = min(duration, max(event['onset'] + 18, event['end'] + 3))
        candidate = item['candidate_onset_sec']
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
        public.append({'blind_id': blind_id, 'video_path': str(alias), 'original_onset_sec': event['onset'],
                       'sheets': linked, 'verdict': None, 'corrected_onset_sec': None,
                       'evidence': None, 'needs_expansion': None})
        private.append({'blind_id': blind_id, 'video_id': vid, 'event_index': index,
                        'review_reason': item['review_reason']})
        dump(out / 'review_template.json', public)
        dump(root / f'private_key_{pass_name}.json', private)
        print(f'{pass_name}: {i + 1}/{len(queue)}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('data/annotation/onset_v3'))
    ap.add_argument('--pass-name', required=True)
    ap.add_argument('--seed', type=int, required=True)
    args = ap.parse_args()
    prepare(args.root, args.pass_name, args.seed)


if __name__ == '__main__':
    main()
