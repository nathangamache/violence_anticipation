"""Extract one bounded Phase C pilot or lead-approved screening chunk without running registered gates."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from anticipation.features import io, vlm_escalation as vlm
from protocol import ROOT, FOLDS, gpu_budget, ledger, preflight, read_manifest, sha, start_gate


def current(name, row, model, arm):
    questions, _ = vlm.questions_and_columns(arm)
    prompt = hashlib.sha256(json.dumps([vlm.PROMPT_VERSION, vlm.CONTEXT, questions]).encode()).hexdigest()
    stride = vlm.arm_config(arm)[1]
    start = max(1, 1 + np.floor((float(row['start_sec']) - 1) / stride) * stride)
    for stream in (name, name + '_hidden'):
        path = io.feature_path(stream, row['video_id'])
        if not Path(path).exists():
            return False
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['meta']))
            if meta.get('model_id') != model or meta.get('prompt_sha256') != prompt or meta.get('version') != vlm.VERSION:
                raise ValueError('Existing feature stream has a different model or prompt')
            expected = {k: vlm.DEFAULTS[k] for k in ('max_pixels', 'max_edge', 'temperature', 'calibration', 'save_hidden', 'pose_gating', 'stride_sec', 'include_early_baseline')}
            if any(meta['cfg'].get(k) != value for k, value in expected.items()):
                raise ValueError('Existing stream has different inference settings')
            if tuple(meta['cfg'].get('hidden_layers', ())) != tuple(vlm.DEFAULTS['hidden_layers']):
                raise ValueError('Existing hidden-state layers differ from the pilot configuration')
            if not meta.get('valid_samples'):
                raise RuntimeError('All-invalid feature output needs inspection before continuing')
            if meta['cfg']['arm'] != arm or meta.get('query_start_sec') != start:
                return False
            if len(z['t_sec']) != int(np.floor(float(row['cap_sec']))):
                return False
    return True


def lead_go(path, arm, sets):
    decision = json.load(open(path))
    required = dict(decision='GO', winner=arm, registration_sha256=sets['registration_sha256'],
                    pilot_manifest_sha256=sets['pilot']['sha256'], screening_folds=list(FOLDS),
                    confirmation_fold_read=False)
    if any(decision.get(k) != v for k, v in required.items()):
        raise ValueError('No matching lead Phase C GO decision')
    report = decision.get('harness_report', {})
    if not report.get('path') or sha(report['path']) != report.get('sha256'):
        raise ValueError('GO must pin the lead registered-harness report')
    return decision


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['pilot', 'screen'])
    p.add_argument('--arm', choices=['A', 'B', 'C'], required=True)
    p.add_argument('--model', default=vlm.MODEL_3)
    p.add_argument('--chunk-sec', type=int, default=1200)
    p.add_argument('--go-decision', default=str(ROOT / 'go_decision.json'))
    a = p.parse_args()
    sets = json.load(open(ROOT / 'sets.json'))
    plan = json.load(open(ROOT / 'budget_plan.json'))
    if plan.get('decision') != 'RUN_PILOT' or a.arm not in plan['arms'] or plan['model_id'] != a.model:
        raise RuntimeError('Arm not authorized by the frozen cost plan')
    sets['pilot'] = plan['pilot']
    ready = preflight(ROOT / 'phase0_evidence.json')
    if ready['status'] != 'READY':
        raise RuntimeError('Phase 0 evidence mismatch')
    with ledger(ROOT / 'budget.json') as budget:
        stages = budget['stages'].copy()
    for arm in 'ABC':
        if not stages.get(a.model + '::timing_' + arm, {}).get('complete'):
            raise RuntimeError('Complete 200 timing queries per arm before pilot')
    if not stages.get(a.model + '::causal', {}).get('complete'):
        raise RuntimeError('Pretrained causality must pass before pilot')
    if not stages.get(a.model + '::contamination', {}).get('complete'):
        raise RuntimeError('Complete the contamination probe before pilot')
    if a.stage == 'screen':
        decision = lead_go(a.go_decision, a.arm, sets)
        if decision.get('model_id') != a.model:
            raise ValueError('GO model differs from extraction model')
    record = sets['pilot' if a.stage == 'pilot' else 'screen']
    rows = read_manifest(record['path'], record['sha256'])
    name = 'vlm_' + a.arm.lower()
    pending = [r for r in rows if not current(name, r, a.model, a.arm)]
    if not pending:
        report = dict(stage=a.stage, arm=a.arm, complete=True, total_videos=len(rows),
                      model_id=a.model, manifest_sha256=record['sha256'], remaining_videos=0)
        (ROOT / f'{a.stage}_{a.arm}.json').write_text(json.dumps(report, indent=2))
        with ledger(ROOT / 'budget.json') as budget:
            budget['stages'][a.model + '::' + a.stage + '_' + a.arm] = report
        print(json.dumps(report))
        return
    start = start_gate()
    (ROOT / 'start_gate.json').write_text(json.dumps(start, indent=2))
    if start['status'] != 'READY':
        raise RuntimeError('GPU 0/CCTV start gate remains closed')
    import os
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Unset CUDA_VISIBLE_DEVICES so cuda:1 is the physical GPU')
    completed = 0
    with gpu_budget(ROOT / 'budget.json', a.model + '::' + a.stage + '_' + a.arm, a.chunk_sec) as deadline:
        import torch
        torch.set_num_threads(4)
        torch.cuda.set_device(1)
        torch.cuda.set_per_process_memory_fraction(.75, 1)
        for row in pending:
            if time.perf_counter() >= deadline - 30:
                break
            try:
                result = vlm.extract_and_save(row['path'], row['video_id'], 'cuda:1', output_name=name,
                    model_id=a.model, arm=a.arm, query_start_sec=float(row['start_sec']),
                    max_duration_sec=float(row['cap_sec']), deadline_monotonic=deadline - 20,
                    rationale_samples=1 if row['video_id'] in {r['video_id'] for r in rows[:3]} else 0)
            except TimeoutError:
                break
            completed += 1
            with open(ROOT / 'extraction.jsonl', 'a') as f:
                f.write(json.dumps(dict(stage=a.stage, arm=a.arm, video_id=row['video_id'], result=result)) + '\n')
    remaining = len(pending) - completed
    report = dict(stage=a.stage, arm=a.arm, model_id=a.model, complete=remaining == 0,
                  completed_this_chunk=completed, remaining_videos=remaining, total_videos=len(rows),
                  manifest_sha256=record['sha256'])
    (ROOT / f'{a.stage}_{a.arm}.json').write_text(json.dumps(report, indent=2))
    if report['complete']:
        if not all(current(name, row, a.model, a.arm) for row in rows):
            raise RuntimeError('Feature coverage verification failed')
        with ledger(ROOT / 'budget.json') as budget:
            budget['stages'][a.model + '::' + a.stage + '_' + a.arm] = report
    print(json.dumps(report))


if __name__ == '__main__':
    main()
