"""Render blind calm-first-frame audit sheets from screening-fold UCF videos."""
import argparse
import json
import subprocess
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw
import torch

from anticipation.data.folds import Folds
from prepare import ordered
from protocol import ROOT, require_screen_ids, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--render', action='store_true')
    a = p.parse_args()
    folds = Folds.load('benchmark/folds_v2.json')
    catalog = pd.read_csv('benchmark/videos.csv')
    catalog = catalog[catalog.video_id.map(folds.fold_of).isin([0, 2, 3, 4])]
    eligible = catalog[(catalog.source_kind == 'ucf') & catalog.role.isin(['violent', 'other_anomaly', 'clean_negative'])]
    ids = ordered(eligible.video_id.tolist(), 'contamination-blind-v1-')
    require_screen_ids(ids, folds)
    eligible = eligible.set_index('video_id')
    names = dict(abuse='Abuse', arrest='Arrest', arson='Arson', assault='Assault', burglary='Burglary',
                 explosion='Explosion', fighting='Fighting', normal='Normal', road_accident='RoadAccidents',
                 robbery='Robbery', shooting='Shooting', shoplifting='Shoplifting', stealing='Stealing', vandalism='Vandalism')
    rows = [dict(audit_id=f'c{i+1:04d}', video_id=v, path=eligible.loc[v, 'path'],
                 category=names[eligible.loc[v, 'classes']],
                 group={'violent': 'violent', 'other_anomaly': 'h1', 'clean_negative': 'clean'}[eligible.loc[v, 'role']],
                 calm_verified='', reason='') for i, v in enumerate(ids)]
    out = ROOT / 'contamination_candidates.csv'
    frame = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out, keep_default_na=False)
        if not old.drop(columns=['calm_verified', 'reason']).equals(frame.drop(columns=['calm_verified', 'reason'])):
            raise ValueError('Contamination candidate membership changed')
        frame = old
    else:
        frame.to_csv(out, index=False)
    if not a.render:
        print(json.dumps(dict(candidates=len(rows), groups=frame.groupby('group').size().to_dict(), sha256=sha(out))))
        return
    torch.set_num_threads(2)
    folder = ROOT / 'contamination_sheets'
    folder.mkdir(exist_ok=True)
    for page in range((len(rows) + 29) // 30):
        path = folder / f'{page:02d}.jpg'
        if path.exists():
            continue
        sheet = Image.new('RGB', (6 * 240, 5 * 160), 'white')
        draw = ImageDraw.Draw(sheet)
        for n, row in enumerate(rows[page*30:page*30+30]):
            x, y = (n % 6) * 240, (n // 6) * 160
            draw.text((x+4, y+2), row['audit_id'], fill='black')
            try:
                data = subprocess.run(['ffmpeg', '-v', 'error', '-threads', '1', '-i', row['path'],
                    '-frames:v', '1', '-vf', 'scale=240:140:force_original_aspect_ratio=decrease',
                    '-f', 'image2pipe', '-vcodec', 'mjpeg', '-threads', '1', 'pipe:1'],
                    capture_output=True, check=True, timeout=20)
                tile = Image.open(BytesIO(data.stdout)).convert('RGB')
                tile.thumbnail((240, 140))
                sheet.paste(tile, (x, y+20))
            except Exception as exc:
                draw.text((x+4, y+40), 'DECODE ERROR', fill='black')
                print(json.dumps(dict(audit_id=row['audit_id'], error=str(exc))), flush=True)
        sheet.save(path, quality=88)
        print(str(path), flush=True)


if __name__ == '__main__':
    main()
