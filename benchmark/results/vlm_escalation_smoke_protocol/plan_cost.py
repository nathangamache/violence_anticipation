"""Choose only the pre-authorized pilot cost reductions before any contamination or arm outcome."""
import csv
import json

from prepare import ordered, write_manifest
from protocol import ROOT, ledger, read_manifest, sha
from anticipation.features import vlm_escalation as vlm


def main():
    sets = json.load(open(ROOT / 'sets.json'))
    pilot = read_manifest(sets['pilot']['path'], sets['pilot']['sha256'])
    screen = read_manifest(sets['screen']['path'], sets['screen']['sha256'])
    timing = {a: json.load(open(ROOT / 'timing_qwen3' / f'timing_{a}.cost.json')) for a in 'ABC'}
    rates = {a: timing[a]['video_seconds_per_gpu_second'] for a in 'ABC'}
    with ledger(ROOT / 'budget.json') as data:
        remaining = (data['cap_sec'] - data['prior_work_reserved_sec'] - data['spent_sec'] - sum(data['active'].values())) / 3600
    reduced = [r for r in pilot if r['group'] != 'clean']
    for source in ('ucf', 'ubi', 'xd_movie', 'xd_youtube'):
        group = {r['video_id']: r for r in pilot if r['group'] == 'clean' and r['source_kind'] == source}
        reduced.extend(group[v] for v in ordered(group, 'vlm-pilot-v1-clean-')[:25])
    plans = []
    for arms, rows in [('ABC', pilot), ('BC', pilot), ('BC', sorted(reduced, key=lambda r: r['video_id']))]:
        by_id = {r['video_id']: r for r in rows}
        rest = [r for r in screen if r['video_id'] not in by_id or r != by_id[r['video_id']]]
        hours = lambda rr: sum(float(r['cap_sec']) - float(r['start_sec']) + 1 for r in rr) / 3600
        pilot_h, rest_h = hours(rows), hours(rest)
        overhead = 1. + 1000 * timing['A']['mean_wall_sec'] / 3600
        pilot_gpu_h = sum(pilot_h / rates[a] for a in arms)
        rest_gpu_h = max(rest_h / rates[a] for a in arms)
        estimate = 1.15 * (pilot_gpu_h + rest_gpu_h) + overhead
        plan = dict(arms=list(arms), clean_videos=sum(r['group'] == 'clean' for r in rows),
                    pilot_hours=pilot_h, rest_hours=rest_h, pilot_gpu_hours=pilot_gpu_h,
                    worst_winning_arm_rest_gpu_hours=rest_gpu_h, overhead_gpu_hours=overhead,
                    uncertainty_factor=1.15, remaining_gpu_hours=remaining, projected_gpu_hours=estimate,
                    fits=estimate <= remaining)
        plans.append(plan)
        if plan['fits']:
            record = sets['pilot'] if rows is pilot else write_manifest('pilot_P100', rows)
            plan.update(decision='RUN_PILOT', pilot=record, model_id=vlm.MODEL_3,
                        registration_sha256=sets['registration_sha256'])
            break
    else:
        plan = dict(decision='NO_GO_COST', remaining_gpu_hours=remaining, reason='Even the allowed B/C +100-clean pilot and worst winner screen exceed the cap')
    report = {**plan, 'candidates': plans, 'timing_sha256': {a: sha(ROOT / 'timing_qwen3' / f'timing_{a}.cost.json') for a in 'ABC'}}
    path = ROOT / 'budget_plan.json'
    if path.exists() and json.load(open(path)).get('decision') == 'RUN_PILOT':
        raise RuntimeError('A pilot plan is already frozen; do not reselect after outcomes')
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
