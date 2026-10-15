"""Enforce phase-0 readiness, fold isolation, ordered VLM stages and a shared GPU budget."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from contextlib import contextmanager

import numpy as np

from anticipation.data.folds import Folds
from anticipation.features import io, vlm_escalation as vlm

ROOT = Path(__file__).resolve().parent
FOLDS = (0, 2, 3, 4)
CAP = 35 * 3600
PRIOR_RESERVE = 3600


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_screen_ids(ids, folds):
    forbidden = [v for v in ids if folds.fold_of(v) not in FOLDS]
    if forbidden:
        raise ValueError(f"{len(forbidden)} ids outside screening folds; no media or features opened")


def completed_csv(path, column="verdict"):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return dict(rows=len(rows), completed=sum(bool(r.get(column, '').strip()) for r in rows))


def preflight(manifest=None):
    source = Path('anticipation/eval/va30.py').read_text()
    doc = Path('docs/VA30.md').read_text()
    checks = {k: k in source for k in ('band', 'sustained', 'window_max')}
    checks['capped_horizons'] = 'horizons_csv' in source and 'with_horizons' in source
    checks['no_context_reference_documented'] = 'aggression_level' in doc
    checks['frozen_registration'] = False
    try:
        from anticipation.screen.prereg import load
        load()
        checks['frozen_registration'] = True
    except (OSError, ValueError, RuntimeError):
        pass
    evidence = {}
    for name, path in [('hivau_spotcheck', 'data/annotations/hivau/spotcheck/spotcheck.csv'),
                       ('official_spotcheck', 'data/annotations/hivau/spotcheck_official_control/spotcheck.csv')]:
        if Path(path).exists():
            evidence[name] = completed_csv(path)
            checks[name] = evidence[name]['rows'] > 0 and evidence[name]['completed'] == evidence[name]['rows']
        else:
            checks[name] = False
    required = ('screen_set', 'screen_summary', 'aggression_reference', 'early_contact_audit',
                'ucf_soft_masks', 'audio_type_audit', 'h1_manifest')
    declared = json.load(open(manifest)) if manifest else {}
    for key in required:
        item = declared.get(key, {})
        path = Path(item.get('path', '__missing__'))
        checks[key] = path.is_file() and item.get('sha256') == sha(path)
        if checks[key]:
            evidence[key] = dict(path=str(path), sha256=sha(path))
    if checks['screen_set']:
        screen = json.load(open(declared['screen_set']['path']))
        checks['screen_set_scope'] = sorted(screen.get('screen_folds', [])) == list(FOLDS) and screen.get('confirmation_fold') == 1
    if checks['early_contact_audit']:
        audit = completed_csv(declared['early_contact_audit']['path'], 'classification')
        evidence['early_contact_audit'].update(audit)
        checks['early_contact_audit'] = audit['rows'] >= 53 and audit['completed'] == audit['rows']
    return dict(status='READY' if all(checks.values()) else 'BLOCKED_PHASE0', checks=checks,
                missing=[k for k, ok in checks.items() if not ok], evidence=evidence,
                instruction='docs/MODALITY_PLAN.md section 6.1: blocking; no new GPU job',
                note='False artifact checks mean missing or unpinned evidence, not necessarily absent files.',
                screening_folds=FOLDS, confirmation_fold=1)


@contextmanager
def ledger(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + '.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.load(open(path)) if path.exists() else dict(cap_sec=CAP, prior_work_reserved_sec=PRIOR_RESERVE,
                   spent_sec=0, active={}, history=[], stages={})
        yield data
        temp = Path(str(path) + '.tmp')
        temp.write_text(json.dumps(data, indent=2))
        os.replace(temp, path)


@contextmanager
def gpu_budget(path, stage, seconds):
    if not 0 < seconds <= 1500:
        raise ValueError('Chunks must reserve 1..1500 seconds')
    token = f'{os.getpid()}-{time.time_ns()}'
    with ledger(path) as data:
        occupied = data['prior_work_reserved_sec'] + data['spent_sec'] + sum(data['active'].values())
        if occupied + seconds > data['cap_sec']:
            raise RuntimeError('35 GPU-hour budget exhausted or reserved by another chunk')
        data['active'][token] = seconds
    start = time.perf_counter()
    previous = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise TimeoutError('Bounded GPU chunk expired')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield start + seconds
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        elapsed = time.perf_counter() - start
        with ledger(path) as data:
            data['active'].pop(token)
            data['spent_sec'] += elapsed
            data['history'].append(dict(stage=stage, seconds=elapsed, token=token))


def read_manifest(path, digest):
    if sha(path) != digest:
        raise ValueError('Frozen manifest SHA256 mismatch')
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows or len({r['video_id'] for r in rows}) != len(rows):
        raise ValueError('Manifest must contain unique video ids')
    folds = Folds.load('benchmark/folds_v2.json')
    require_screen_ids([r['video_id'] for r in rows], folds)
    return rows


def timing(rows, arm, deadline, model, out):
    scorer = vlm._scorer(model, 'cuda:1', 448*448, 448, (18, 27))
    n_done = sum(1 for _ in open(out)) if out.exists() else 0
    for query in range(n_done, 200):
        if time.perf_counter() >= deadline:
            return False
        row = rows[query % len(rows)]
        start = time.perf_counter()
        video = io.Video(row['path'], num_threads=2)
        t = min(float(row.get('t_sec') or 130), video.info.duration_sec - 1 / video.info.fps)
        times = vlm.clip_times(t, arm)
        frames = video.frames_at(times)
        score, hidden = scorer.score(frames, times, arm)
        import torch
        torch.cuda.synchronize('cuda:1')
        rec = dict(query=query, video_id=row['video_id'], arm=arm, wall_sec=time.perf_counter()-start,
                   hidden_dim=len(hidden), rating_dim=len(score), model_id=model)
        with open(out, 'a') as f:
            f.write(json.dumps(rec) + '\n')
    return True


def timing_stats(path, arm):
    records = [json.loads(line) for line in open(path)]
    if len(records) != 200 or [r['query'] for r in records] != list(range(200)):
        raise ValueError('Timing requires exactly 200 distinct ordered queries')
    wall = np.asarray([r['wall_sec'] for r in records])
    stride = vlm.arm_config(arm)[1]
    realtime = float(stride / wall.mean())
    return dict(queries=200, mean_wall_sec=float(wall.mean()), p95_wall_sec=float(np.quantile(wall, .95)),
                video_seconds_per_gpu_second=realtime,
                projected_gpu_hours={k: h / realtime for k, h in
                    dict(pilot_12_5h=12.5, rest_screen_30h=30, positive_plus_30h_negative_46_7h=46.7, all_240h=240).items()},
                measurement='decoder opening, frame retrieval, model and CPU transfer; model startup excluded')


def contamination(rows, deadline, model, out):
    classes = ('Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting',
               'Normal', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism')
    options = list('ABCDEFGHIJKLMN')
    question = 'Which UCF-Crime dataset category is this video? ' + '; '.join(f'{a}={b}' for a, b in zip(options, classes)) + '. Answer one letter only.'
    if any(r.get('calm_verified', '').lower() != 'true' or not r['video_id'].startswith('ucf/') or r.get('category') not in classes or r.get('group') not in ('violent', 'h1') for r in rows):
        raise ValueError('Contamination manifest requires verified calm UCF first frames and canonical categories')
    scorer = vlm._scorer(model, 'cuda:1', 448*448, 448, ())
    existing = {json.loads(line)['video_id'] for line in open(out)} if out.exists() else set()
    for row in rows:
        if row['video_id'] in existing:
            continue
        if time.perf_counter() >= deadline:
            return False
        frame = io.Video(row['path'], num_threads=2).frames_at([0])
        inputs = scorer.process(frame, scorer.texts([question], 'A'), np.asarray([0.]), 'A')
        import torch
        with torch.inference_mode():
            logits = scorer.model(**inputs, logits_to_keep=1, use_cache=False).logits[0, -1].float()
        ids = [scorer.processor.tokenizer.encode(o, add_special_tokens=False) for o in options]
        if any(len(x) != 1 for x in ids):
            raise ValueError('Contamination options are not single tokens')
        scores = logits[[x[0] for x in ids]].cpu().numpy()
        pred = classes[int(np.argmax(scores))]
        rec = dict(video_id=row['video_id'], category=row['category'], group=row['group'], prediction=pred,
                   model_id=model, prompt_sha256=hashlib.sha256(question.encode()).hexdigest())
        with open(out, 'a') as f:
            f.write(json.dumps(rec) + '\n')
    return len(rows) >= 1000


def provenance(out, stage, model, arm, digest):
    path = out / (stage + '.config.json')
    config = dict(stage=stage, model_id=model, arm=arm, manifest_sha256=digest,
                  prompt_version=vlm.PROMPT_VERSION, extractor_sha256=sha(vlm.__file__))
    if path.exists() and json.load(open(path)) != config:
        raise ValueError('Output provenance mismatch; use a separate output directory')
    path.write_text(json.dumps(config, indent=2))


def contamination_stats(path, permutations=10000):
    records = [json.loads(line) for line in open(path)]
    if len({r['video_id'] for r in records}) != len(records):
        raise ValueError('Duplicate contamination queries')
    groups = sorted({r['group'] for r in records})
    if groups != ['h1', 'violent']:
        raise ValueError('Both violent and H1 contamination strata are required')
    rng = np.random.default_rng(0)
    output = {}
    for group in groups:
        rows = [r for r in records if r['group'] == group]
        truth = np.asarray([r['category'] for r in rows])
        pred = np.asarray([r['prediction'] for r in rows])
        observed = float(np.mean(truth == pred))
        null = np.asarray([np.mean(rng.permutation(truth) == pred) for _ in range(permutations)])
        p = float((1 + np.sum(null >= observed)) / (permutations + 1))
        output[group] = dict(n=len(rows), accuracy=observed, permutation_mean=float(null.mean()),
                             p_value=p, flag=p < .025)
    return dict(strata=output, contamination_flag=any(x['flag'] for x in output.values()),
                permutations=permutations, seed=0, correction='Bonferroni over two strata, alpha=.05')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['preflight', 'timing', 'contamination'])
    p.add_argument('--phase0-manifest')
    p.add_argument('--manifest')
    p.add_argument('--manifest-sha256')
    p.add_argument('--arm', choices=['A', 'B', 'C'], default='B')
    p.add_argument('--model', default=vlm.MODEL_3)
    p.add_argument('--chunk-sec', type=int, default=1200)
    p.add_argument('--out-dir', default=str(ROOT))
    p.add_argument('--budget-path', default=str(ROOT / 'budget.json'))
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    readiness = preflight(a.phase0_manifest)
    (out / 'preflight.json').write_text(json.dumps(readiness, indent=2))
    if a.stage == 'preflight':
        with ledger(a.budget_path):
            pass
        print(json.dumps(readiness, indent=2))
        return
    if readiness['status'] != 'READY':
        raise RuntimeError('Phase 0 incomplete; see preflight.json. No GPU initialized.')
    if not a.manifest or not a.manifest_sha256:
        raise ValueError('A frozen query manifest and its SHA256 are required')
    rows = read_manifest(a.manifest, a.manifest_sha256)
    state_path = Path(a.budget_path)
    with ledger(state_path) as data:
        stages = data['stages'].copy()
    if a.stage == 'contamination' and not all(stages.get(a.model + '::timing_' + arm, {}).get('model_id') == a.model for arm in 'ABC'):
        raise RuntimeError('Complete 200 timing queries per arm with this model first')
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('Use unmasked device ordinals so cuda:1 identifies the shared pose GPU')
    key = 'timing_' + a.arm if a.stage == 'timing' else a.stage
    provenance(out, key, a.model, a.arm if a.stage == 'timing' else None, a.manifest_sha256)
    subprocess.run([sys.executable, 'tools/server_load.py', '--gpu'], check=True)
    with gpu_budget(state_path, a.model + '::' + key, a.chunk_sec) as deadline:
        import torch
        torch.set_num_threads(4)
        torch.cuda.set_device(1)
        if a.stage == 'timing':
            done = timing(rows, a.arm, deadline, a.model, out / (key + '.jsonl'))
        else:
            done = contamination(rows, deadline, a.model, out / 'contamination.jsonl')
    if done:
        if a.stage == 'timing':
            report = timing_stats(out / (key + '.jsonl'), a.arm)
            (out / (key + '.cost.json')).write_text(json.dumps(report, indent=2))
        if a.stage == 'contamination':
            report = contamination_stats(out / 'contamination.jsonl')
            (out / 'contamination_stats.json').write_text(json.dumps(report, indent=2))
        with ledger(state_path) as data:
            data['stages'][a.model + '::' + key] = dict(manifest_sha256=a.manifest_sha256, model_id=a.model, complete=True)
    print(json.dumps(dict(stage=key, complete=done)))


if __name__ == '__main__':
    main()
