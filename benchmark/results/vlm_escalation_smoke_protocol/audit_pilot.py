"""Audit frozen pilot caches and export score tables without running scientific gates."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from anticipation.features import io, vlm_escalation as vlm
from protocol import ROOT, read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['A', 'B', 'C'])
    args = parser.parse_args()
    plan = json.loads((ROOT / 'budget_plan.json').read_text())
    arms = [args.arm] if args.arm else plan['arms']
    if any(arm not in plan['arms'] for arm in arms):
        raise ValueError('Arm is outside the frozen cost plan')
    rows = read_manifest(plan['pilot']['path'], plan['pilot']['sha256'])
    result = dict(pilot=plan['pilot'], screening_folds=[0, 2, 3, 4],
                  confirmation_fold_read=False, registered_gates_run=False, arms={})
    for arm in arms:
        stream = 'vlm_' + arm.lower()
        report = dict(videos=0, valid_ticks=0, main_interval_valid_ticks=0, updates=0,
                      decode_errors=0, failures=[])
        for row in rows:
            try:
                outputs = []
                for name in (stream, stream + '_hidden'):
                    with np.load(io.feature_path(name, row['video_id']), allow_pickle=False) as z:
                        t, x, valid = z['t_sec'], z['feat'], z['valid']
                        meta = json.loads(str(z['meta']))
                    assert t.dtype == np.float32 and x.dtype == np.float16 and valid.dtype == bool
                    assert np.array_equal(t, io.tick_times(float(row['cap_sec'])))
                    assert x.shape == (len(t), meta['D']) and valid.shape == t.shape
                    assert np.isfinite(x).all() and len(meta['columns']) == x.shape[1]
                    assert meta['model_id'] == plan['model_id'] and meta['version'] == vlm.VERSION
                    assert meta['no_context'] and meta['cfg']['arm'] == arm
                    assert meta['prompt_version'] == vlm.PROMPT_VERSION
                    outputs.append((t, x, valid, meta))
                t, x, valid, meta = outputs[0]
                assert np.array_equal(valid, outputs[1][2])
                stage = x[:, :6].astype(np.float32)
                assert np.max(np.abs(stage[valid].sum(axis=1) - 1)) < .002
                assert np.all((x[valid] >= 0) & (x[valid] <= 1))
                risk = stage @ vlm.STAGE_WEIGHTS
                assert np.max(np.abs(risk[valid] - x[valid, -1])) < .001
                path = Path(io.feature_path(stream, row['video_id'])).with_suffix('.targets.parquet')
                targets = pd.read_parquet(path)
                samples = targets.t_sec.to_numpy()
                assert np.array_equal(samples, meta['update_t_sec'])
                index = np.searchsorted(samples, t, side='right') - 1
                assert np.all(index[valid] >= 0) and np.all(samples[index[valid]] <= t[valid])
                expected = targets[meta['columns']].to_numpy().astype(np.float16)
                assert np.array_equal(x[valid], expected[index[valid]])
                report['videos'] += 1
                report['valid_ticks'] += int(valid.sum())
                report['main_interval_valid_ticks'] += int((valid & (t >= float(row['start_sec']))).sum())
                report['updates'] += int(targets.valid.sum())
                report['decode_errors'] += len(meta['decode_errors'])
            except (AssertionError, OSError, ValueError, KeyError) as exc:
                report['failures'].append(dict(video_id=row['video_id'], error=type(exc).__name__ + ': ' + str(exc)))
        result['arms'][arm] = report
        if not report['failures']:
            subprocess.run([sys.executable, str(ROOT / 'export_scores.py'),
                            '--manifest', plan['pilot']['path'], '--manifest-sha256', plan['pilot']['sha256'],
                            '--extractor', stream, '--out', str(ROOT / f'scores_pilot_{arm}.parquet')], check=True)
    result['pass'] = all(not x['failures'] and x['videos'] == len(rows) for x in result['arms'].values())
    suffix = '_' + args.arm if args.arm else ''
    (ROOT / ('pilot_cache_audit' + suffix + '.json')).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
