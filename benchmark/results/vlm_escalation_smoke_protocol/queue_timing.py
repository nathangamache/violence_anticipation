"""Wait for the lead's GPU-0 start condition, then run bounded timing chunks on GPU 1."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from anticipation.features import vlm_escalation as vlm
from protocol import ROOT, ledger, start_gate


def main():
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '4'
    sets = json.load(open(ROOT / 'sets.json'))
    record = sets['timing']
    out = ROOT / 'timing_qwen3'
    out.mkdir(exist_ok=True)
    for arm in 'ABC':
        while True:
            with ledger(ROOT / 'budget.json') as data:
                if data['stages'].get(vlm.MODEL_3 + '::timing_' + arm, {}).get('complete'):
                    break
            gate = start_gate()
            (ROOT / 'start_gate.json').write_text(json.dumps(gate, indent=2))
            print(json.dumps(dict(arm=arm, start_gate=gate['status'], gpu0_used_gib=gate['gpu0_used_gib'],
                                 cctv_status=gate['cctv_status'])), flush=True)
            if gate['status'] != 'READY':
                time.sleep(60)
                continue
            command = [sys.executable, str(ROOT / 'protocol.py'), 'timing', '--arm', arm,
                       '--phase0-manifest', str(ROOT / 'phase0_evidence.json'),
                       '--manifest', record['path'], '--manifest-sha256', record['sha256'],
                       '--out-dir', str(out)]
            result = subprocess.run(command)
            if result.returncode:
                raise RuntimeError(f'Timing child failed with status {result.returncode}; inspect before retry')
            time.sleep(15)
    print('TIMING_DONE: review costs before contamination or pilot', flush=True)


if __name__ == '__main__':
    main()
