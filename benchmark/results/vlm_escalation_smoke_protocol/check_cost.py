"""Check cap-respecting plan selection and the ordered pilot reductions with synthetic timing costs."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import plan_cost
import prepare
from protocol import ROOT

sets = json.load(open(ROOT / 'sets.json'))
reports = []
for realtime, expected in [(100., ('RUN_PILOT', 'ABC', 150)), (2.1, ('RUN_PILOT', 'BC', 150)),
                           (2., ('RUN_PILOT', 'BC', 100)), (.001, ('NO_GO_COST', '', 0))]:
    with tempfile.TemporaryDirectory(dir=ROOT) as temp:
        root = Path(temp)
        (root / 'sets.json').write_text(json.dumps(sets))
        (root / 'timing_qwen3').mkdir()
        for arm in 'ABC':
            (root / 'timing_qwen3' / f'timing_{arm}.cost.json').write_text(json.dumps(
                dict(video_seconds_per_gpu_second=realtime, mean_wall_sec=.1)))
        with patch.object(plan_cost, 'ROOT', root), patch.object(prepare, 'ROOT', root), contextlib.redirect_stdout(io.StringIO()):
            plan_cost.main()
        result = json.load(open(root / 'budget_plan.json'))
        actual = (result['decision'], ''.join(result.get('arms', [])), result.get('clean_videos', 0))
        assert actual == expected, (realtime, actual, expected)
        if result['decision'] == 'RUN_PILOT':
            assert result['projected_gpu_hours'] <= result['remaining_gpu_hours']
        reports.append(dict(realtime=realtime, decision=actual[0], arms=actual[1], clean_videos=actual[2]))
(ROOT / 'cost_checks.json').write_text(json.dumps(reports, indent=2))
print(json.dumps(reports, indent=2))
