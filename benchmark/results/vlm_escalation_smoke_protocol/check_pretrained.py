"""Run paired full/cut pretrained causality checks in a bounded, start-gated GPU-1 chunk."""
import json
from pathlib import Path
import time

import numpy as np

from anticipation.features import io, vlm_escalation as vlm
from protocol import ROOT, gpu_budget, ledger, preflight, read_manifest, start_gate


def main():
    evidence = preflight(ROOT / 'phase0_evidence.json')
    if evidence['status'] != 'READY':
        raise RuntimeError('Phase 0 not ready')
    sets = json.load(open(ROOT / 'sets.json'))
    rows = read_manifest(sets['pilot']['path'], sets['pilot']['sha256'])
    row = next(r for r in rows if float(r['cap_sec']) > 30)
    gate = start_gate()
    if gate['status'] != 'READY':
        raise RuntimeError('GPU start gate closed')
    report = dict(model_id=vlm.MODEL_3, video_id=row['video_id'], cut_sec=20, max_duration_sec=24,
                  screening_folds=[0, 2, 3, 4], confirmation_fold_read=False, checks={})
    with gpu_budget(ROOT / 'budget.json', vlm.MODEL_3 + '::causal', 1200) as deadline:
        import torch
        torch.set_num_threads(4)
        torch.cuda.set_device(1)
        torch.cuda.set_per_process_memory_fraction(.75, 1)
        for arm in 'ABC':
            cache = {}
            def outputs(path):
                key = 'full' if path == row['path'] else 'cut'
                if key not in cache:
                    cache[key] = vlm.extract_all(path, 'cuda:1', arm=arm, max_duration_sec=24,
                                                 deadline_monotonic=deadline - 30)
                return cache[key]
            for stream in (vlm.EXTRACTOR, vlm.HIDDEN_EXTRACTOR):
                check = io.check_causal(lambda path: outputs(path)[stream][:3], row['path'], 20)
                k = check['ticks_compared']
                check['valid_equal'] = bool(np.array_equal(cache['full'][stream][2][:k], cache['cut'][stream][2][:k]))
                report['checks'][arm + ':' + stream] = check
                if not check['causal'] or not check['valid_equal']:
                    (ROOT / 'causal_pretrained.json').write_text(json.dumps(report, indent=2))
                    raise RuntimeError('Pretrained causality failed; do not start pilot')
    report['pass'] = True
    (ROOT / 'causal_pretrained.json').write_text(json.dumps(report, indent=2))
    with ledger(ROOT / 'budget.json') as data:
        data['stages'][vlm.MODEL_3 + '::causal'] = dict(complete=True, model_id=vlm.MODEL_3)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
