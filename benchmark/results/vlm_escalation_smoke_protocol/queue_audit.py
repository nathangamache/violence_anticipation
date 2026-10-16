"""Wait for completed pilot extraction, then audit caches and export all score tables on CPU."""
import json
import os
import subprocess
import sys
import time

from protocol import ROOT


def main():
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    while not (ROOT / 'pilot_ready.json').exists():
        log = ROOT / 'queue_pilot.log'
        tail = log.read_text()[-8000:] if log.exists() else ''
        if 'Traceback (most recent call last)' in tail:
            raise RuntimeError('Pilot worker failed; inspect before auditing')
        time.sleep(60)
    subprocess.run([sys.executable, str(ROOT / 'audit_pilot.py')], check=True)
    print(json.dumps(dict(status='PILOT_CACHE_AUDIT_AND_SCORES_READY', registered_gates_run=False,
                          confirmation_fold_read=False)), flush=True)


if __name__ == '__main__':
    main()
