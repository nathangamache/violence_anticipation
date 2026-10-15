"""After timing, apply the frozen cost rule and extract pilot arms; stop before the lead's registered gates."""
import json
import os
import subprocess
import sys
import time

from anticipation.features import vlm_escalation as vlm
from protocol import ROOT, ledger, start_gate
from extract_chunk import lead_go


def run(script, *args):
    command = [sys.executable, str(ROOT / script), *map(str, args)]
    print(json.dumps(dict(command=command)), flush=True)
    result = subprocess.run(command, cwd=ROOT.parents[2])
    if result.returncode:
        raise RuntimeError(f'{script} failed with status {result.returncode}; inspect before retry')


def wait_start():
    while True:
        gate = start_gate()
        if gate['status'] == 'READY':
            return
        print(json.dumps(dict(stage='waiting', start_gate=gate['status'], cctv_status=gate['cctv_status'])), flush=True)
        time.sleep(60)


def stages():
    with ledger(ROOT / 'budget.json') as data:
        return data['stages'].copy()


def main():
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '4'
    while not all(stages().get(vlm.MODEL_3 + '::timing_' + a, {}).get('complete') for a in 'ABC'):
        log = ROOT / 'queue_timing.log'
        if log.exists() and 'Traceback (most recent call last)' in log.read_text()[-8000:]:
            raise RuntimeError('Timing failed; inspect its log before continuing')
        time.sleep(60)
    if not (ROOT / 'budget_plan.json').exists():
        run('plan_cost.py')
    plan = json.load(open(ROOT / 'budget_plan.json'))
    if plan['decision'] != 'RUN_PILOT':
        print('NO_GO_COST: no contamination or pilot launched', flush=True)
        return
    contamination = json.load(open(ROOT / 'contamination_set.json'))['manifest']
    while not stages().get(vlm.MODEL_3 + '::contamination', {}).get('complete'):
        wait_start()
        run('protocol.py', 'contamination', '--phase0-manifest', ROOT / 'phase0_evidence.json',
            '--manifest', contamination['path'], '--manifest-sha256', contamination['sha256'],
            '--out-dir', ROOT / 'contamination_qwen3')
        time.sleep(15)
    if not stages().get(vlm.MODEL_3 + '::causal', {}).get('complete'):
        wait_start()
        run('check_pretrained.py')
        time.sleep(15)
    for arm in plan['arms']:
        while not stages().get(vlm.MODEL_3 + '::pilot_' + arm, {}).get('complete'):
            wait_start()
            run('extract_chunk.py', 'pilot', '--arm', arm)
            time.sleep(15)
        run('derive_posegate.py', '--arm', arm)
    report = dict(status='PILOT_READY_WAIT_LEAD_GATES', arms=plan['arms'], model_id=vlm.MODEL_3,
                  pilot=plan['pilot'], registration_sha256=plan['registration_sha256'],
                  registered_gates_run=False, confirmation_fold_read=False)
    (ROOT / 'pilot_ready.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    decision_path = ROOT / 'go_decision.json'
    while not decision_path.exists():
        time.sleep(60)
    decision = json.load(open(decision_path))
    if decision.get('decision') == 'NO_GO':
        print('NO_GO from lead; screening extraction not launched', flush=True)
        return
    winner = decision.get('winner')
    if winner not in plan['arms']:
        raise ValueError('Lead GO winner is outside the pilot cost plan')
    scope = json.load(open(ROOT / 'sets.json'))
    scope['pilot'] = plan['pilot']
    lead_go(decision_path, winner, scope)
    while not stages().get(vlm.MODEL_3 + '::screen_' + winner, {}).get('complete'):
        wait_start()
        run('extract_chunk.py', 'screen', '--arm', winner)
        time.sleep(15)
    (ROOT / 'screen_ready.json').write_text(json.dumps(dict(status='SCREEN_READY', arm=winner,
        registration_sha256=scope['registration_sha256'], confirmation_fold_read=False), indent=2))


if __name__ == '__main__':
    main()
