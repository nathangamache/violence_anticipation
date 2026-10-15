"""Derive the all-UCF pilot pose-gating ablation from ungated ratings and states on CPU."""
import argparse
import json
from pathlib import Path

import numpy as np

from anticipation.features import io, vlm_escalation as vlm
from protocol import ROOT, read_manifest, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=['A', 'B', 'C'], required=True)
    a = p.parse_args()
    sets = json.load(open(ROOT / 'sets.json'))
    plan = json.load(open(ROOT / 'budget_plan.json'))
    rows = read_manifest(plan['pilot']['path'], plan['pilot']['sha256'])
    base = 'vlm_' + a.arm.lower()
    report = dict(arm=a.arm, total_ucf=0, completed=0, missing=[], gated_ticks=0)
    for row in rows:
        if row['source_kind'] != 'ucf':
            continue
        report['total_ucf'] += 1
        try:
            t, feat, valid, meta = io.load_features(base, row['video_id'])
            ht, hidden, hvalid, hmeta = io.load_features(base + '_hidden', row['video_id'])
            samples = np.asarray(meta['update_t_sec'])
            pose_path = io.feature_path('pose_raw', row['video_id'])
            gates = vlm.pose_gate_samples(pose_path, samples)
        except (FileNotFoundError, ValueError) as exc:
            report['missing'].append(dict(video_id=row['video_id'], reason=str(exc)))
            continue
        index = np.searchsorted(samples, t, side='right') - 1
        held = (index >= 0) & gates[np.maximum(index, 0)] & valid
        feat[held] = 0
        feat[held, 0] = 1
        hidden[held] = 0
        hvalid[held] = False
        for name, x, mask, md in [(base + '_posegate', feat, valid, meta),
                                  (base + '_posegate_hidden', hidden, hvalid, hmeta)]:
            md.update(extractor=name, ungated_parent=base, pose_gating_derived=True, pose_sha256=sha(pose_path),
                      gating_policy='No tracked person, complete valid pose window [t-7,t] at each update',
                      window_sec=max(md['clip_window_sec'], 7) + vlm.arm_config(a.arm)[1])
            md['cfg']['pose_gating'] = True
            io.save_features(name, row['video_id'], t, x, mask, md)
        report['completed'] += 1
        report['gated_ticks'] += int(held.sum())
    (ROOT / ('posegate_' + a.arm + '.json')).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
