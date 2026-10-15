"""Exercise CPU-only fold isolation, budget accounting, provenance and permutation controls."""
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('protocol', Path(__file__).with_name('protocol.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)

class Folds:
    def fold_of(self, vid):
        return {'allowed': 0, 'frozen': 1}.get(vid)

p.require_screen_ids(['allowed'], Folds())
for vid in ['frozen', 'unknown']:
    try:
        p.require_screen_ids([vid], Folds())
        raise AssertionError('Fold isolation did not reject')
    except ValueError:
        pass
with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    with patch.object(sys, 'argv', ['protocol.py', 'timing', '--out-dir', str(root / 'blocked'),
            '--budget-path', str(root / 'blocked_budget.json')]), patch.object(p.vlm, '_scorer',
            side_effect=AssertionError('GPU model requested while blocked')), patch.object(p.subprocess, 'run',
            side_effect=AssertionError('GPU load check requested while blocked')):
        try:
            p.main()
            raise AssertionError('Phase 0 gate did not reject')
        except RuntimeError as exc:
            assert 'Phase 0 incomplete' in str(exc)
    assert not (root / 'blocked_budget.json').exists()
    budget = root / 'budget.json'
    with p.ledger(budget) as data:
        data['cap_sec'], data['prior_work_reserved_sec'] = 10, 0
    with p.gpu_budget(budget, 'cpu_simulation', 5):
        pass
    state = json.load(open(budget))
    assert state['spent_sec'] > 0 and not state['active']
    with p.ledger(budget) as data:
        data['active']['orphaned_reservation'] = 9
    try:
        with p.gpu_budget(budget, 'cpu_simulation', 2):
            raise AssertionError('Budget accepted over-reservation')
    except RuntimeError:
        pass
    p.provenance(root, 'timing_A', 'model', 'A', 'manifest_hash')
    p.provenance(root, 'timing_A', 'model', 'A', 'manifest_hash')
    try:
        p.provenance(root, 'timing_A', 'different_model', 'A', 'manifest_hash')
        raise AssertionError('Provenance mixing accepted')
    except ValueError:
        pass
    for perfect in (False, True):
        path = root / 'contamination.jsonl'
        records = [dict(video_id=f'{group}/{i}', group=group, category=str(i % 4),
                        prediction=str(i % 4) if perfect else '0') for group in ('violent', 'h1') for i in range(80)]
        path.write_text(''.join(json.dumps(r) + '\n' for r in records))
        result = p.contamination_stats(path, permutations=499)
        assert result['contamination_flag'] == perfect
    path = root / 'timing.jsonl'
    path.write_text(''.join(json.dumps(dict(query=i, wall_sec=.5)) + '\n' for i in range(200)))
    cost = p.timing_stats(path, 'B')
    assert cost['video_seconds_per_gpu_second'] == 4
    assert cost['projected_gpu_hours']['all_240h'] == 60
report = dict(phase0_blocks_before_gpu_or_budget=True, fold1_and_unknown_rejected=True, budget_reservation_and_recovery=True,
              provenance_mixing_rejected=True, permutation_controls=True, cost_units=True)
Path(__file__).with_name('protocol_checks.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
