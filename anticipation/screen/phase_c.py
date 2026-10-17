"""Evaluate frozen Phase C pilot caches without reading fold 1 or authorizing screening extraction."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from anticipation.data.folds import Folds, content_hash
from anticipation.eval import va30
from anticipation.eval.aggression_reference import load_records
from anticipation.eval.baselines import read_teacher, sample_and_hold
from anticipation.eval.learned_baselines import SOURCES, catalog_mapping
from anticipation.features import io
from anticipation.features import vlm_escalation as VLM
from anticipation.screen import decision as D
from anticipation.screen import features as F
from anticipation.screen import metrics as M
from anticipation.screen import prereg
from anticipation.screen import probes as P
from anticipation.screen import sets as S

PROTOCOL = Path('benchmark/results/vlm_escalation_smoke_protocol')
REGISTRATION = 'benchmark/screens/screen_set_v1.json'
REGISTRATION_SHA = '8831e16e664c895271b9cd77e0bab68733ca85a47655708998ab51d5e625797b'
ARMS = ('A', 'B', 'C', 'forecast', 'hidden_A', 'hidden_B', 'hidden_C', 'aggression')


def serial(value, key=''):
    if isinstance(value, M.Stat):
        return value.to_dict(0.0 if any(s in key for s in ('gain', 'm4n')) else .5)
    if isinstance(value, dict):
        return {str(k): serial(v, str(k)) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [serial(v, key) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(serial(data), indent=1, allow_nan=False) + '\n')
    temp.replace(path)


def pilot_rows():
    plan = json.loads((PROTOCOL / 'budget_plan.json').read_text())
    manifest = Path(plan['pilot']['path'])
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != plan['pilot']['sha256']:
        raise ValueError('Frozen pilot manifest changed')
    rows = list(csv.DictReader(manifest.open()))
    folds = Folds.load('benchmark/folds_v2.json')
    if any(folds.fold_of(r['video_id']) not in S.SCREEN_FOLDS for r in rows):
        raise S.FrozenFoldError('Pilot contains confirmation or unassigned videos')
    if len({r['video_id'] for r in rows}) != len(rows):
        raise ValueError('Pilot IDs are not unique')
    return rows, plan


class PilotSet(S.ScreenSet):
    def __init__(self, base, rows):
        super().__init__(base.bench, base.h1.reset_index(drop=True), prereg=base.prereg)
        self.intervals = {r['video_id']: (float(r['start_sec']), float(r['cap_sec'])) for r in rows}
        for vid in self.intervals:
            self.guard(vid)

    def ticks(self, vid):
        if vid not in self.intervals:
            raise ValueError('Video is outside the frozen pilot')
        vt = super().ticks(vid)
        lo, hi = self.intervals[vid]
        vt = copy.deepcopy(vt.cut(int(np.floor(hi))))
        vt.masked |= (vt.t < lo) | (vt.t > hi)
        return vt

    def train_ids(self, fold):
        return [v for v in super().train_ids(fold) if v in self.intervals]

    def val_ids(self, fold):
        return [v for v in super().val_ids(fold) if v in self.intervals]

    def test_ids(self, fold, kinds=('violent', 'clean', 'h1')):
        return [v for v in super().test_ids(fold, kinds) if v in self.intervals]

    def events(self):
        events = super().events()
        keep = [v in self.intervals and self.intervals[v][0] <= onset <= self.intervals[v][1]
                for v, onset in zip(events.video_id, events.onset)]
        return events.loc[keep].reset_index(drop=True)


class PilotSource(F.FeatureSource):
    def __init__(self, ss, arm, layer=None):
        super().__init__(ss)
        self.arm, self.layer, self.records = arm, layer, None
        if arm == 'aggression':
            bench = copy.copy(ss.bench)
            bench.videos = bench.videos.loc[sorted(ss.intervals)]
            self.records, self.reference_provenance = load_records(bench, 'data/processed/vlm_labels_violence')

    def has(self, stream, vid):
        self.ss.guard(vid)
        if stream == 'phase_c':
            return vid in self.records if self.arm == 'aggression' else super().has(self.parent_stream(), vid)
        return super().has(stream, vid)

    def parent_stream(self):
        arm = 'B' if self.arm == 'forecast' else self.arm.removeprefix('hidden_')
        return 'vlm_' + arm.lower() + ('_hidden' if self.arm.startswith('hidden_') else '')

    def load(self, stream, vid):
        self.ss.guard(vid)
        if stream != 'phase_c':
            return super().load(stream, vid)
        if self.arm == 'aggression':
            rows = self.records[vid]
            t = self.ss.ticks(vid).t
            x = sample_and_hold(np.array([r['time_sec'] for r in rows]), np.array([r['aggression'] for r in rows]), t)
            return t, x[:, None].astype(np.float32), np.ones(len(t), bool), {'columns': ['aggression_level']}
        return self.select_columns(*super().load(self.parent_stream(), vid))

    def select_columns(self, t, x, valid, meta):
        if self.arm.startswith('hidden_'):
            groups = [g for g in meta['column_groups'] if g['layer'] == self.layer]
            if len(groups) != 2:
                raise ValueError('Missing requested intermediate layer groups')
            idx = np.concatenate([np.arange(g['start'], g['stop']) for g in groups])
        elif self.arm == 'forecast':
            idx = np.array([meta['columns'].index('forecast_fight_within60')])
        else:
            idx = np.array([i for i, c in enumerate(meta['columns']) if c != 'forecast_fight_within60'])
        return t, x[:, idx], valid, {**meta, 'columns': [meta['columns'][i] for i in idx]}


class PilotRun(P.BlockRun):
    def __init__(self, ss, source, **kwargs):
        super().__init__(ss, source, F.Block('phase_c', 'C', ('phase_c',)), arms=('p1',), **kwargs)
        self.observed = {}

    def raw(self, vid):
        x = super().raw(vid)
        if vid not in self.observed:
            _, _, valid, _ = self.source.load('phase_c', vid)
            valid = valid[:len(x)].copy()
            self.observed[vid] = valid
            self._vt[vid].masked |= ~valid
        return x

    def cut_last(self, vid):
        last = super().cut_last(vid)
        observed = self.observed[vid]
        extra = np.where(~observed, np.arange(len(observed)), -1)
        return np.maximum(last, np.maximum.accumulate(extra))


def fit_fold(runner, fold, train, val, test, out=None):
    runner.set_fold(train)
    rng = np.random.default_rng(1000 * fold)
    tr = {t: P.Plan([runner.vt(v) for v in train], t, rng, P.MAX_TRAIN_POS, P.MAX_TRAIN_NEG, runner.src) for t in P.TARGETS}
    va = {t: P.Plan([runner.vt(v) for v in val], t, rng, None, P.MAX_VAL_NEG, runner.src) for t in P.TARGETS}
    nuisance = P.Nuisance(runner.source, train)
    specs = {'p1': ('band', 'p1'), 'p1_near': ('near', 'p1'), 'n': ('band', 'n'),
             'n_near': ('near', 'n'), 'n_plus_p1': ('band', 'joint'), 'n_plus_p1_near': ('near', 'joint')}
    cache = {}

    def matrix(vid, kind):
        if vid not in cache:
            vt = runner.vt(vid)
            x, n = runner.feats(vid, 'p1'), nuisance.features(vt, runner.src(vid))
            cache[vid] = {'p1': x, 'n': n, 'joint': np.concatenate([n, x], axis=1)}
        return cache[vid][kind]

    models, info = {}, {}
    for name, (target, kind) in specs.items():
        def gather(plan):
            if plan.empty:
                raise ValueError(f'No matched {target} rows in fold {fold}')
            return [np.concatenate(a) for a in zip(*[(matrix(v, kind)[ti], y, w) for v, (ti, y, w) in plan.by_video.items()])]
        x, y, w = gather(tr[target])
        if va[target].empty:
            xv, yv, wv = None, np.zeros(0), None
            callback = None
        else:
            xv, yv, wv = gather(va[target])
            callback = lambda p: P.weighted_auc(p.score(xv), yv, wv)
        model = P.fit_probe(x, y, w, callback)
        models[name] = model
        info[name] = dict(C=model.C, val_auc=model.val_auc, path=model.path, n_train=len(y), n_val=len(yv),
                          converged=model.converged, val_fallback=bool(getattr(model, 'fallback', False)))
    scores = {m: {'test': {}, 'val': {}} for m in specs}
    for split, vids in [('test', test), ('val', val)]:
        for vid in vids:
            for name, (_, kind) in specs.items():
                scores[name][split][vid] = models[name].score(matrix(vid, kind))
    if out:
        temp = Path(str(out) + '.tmp')
        joblib.dump(dict(models=models, pls=runner.pls, thr=runner.thr, nuisance=nuisance, info=info,
                         layer=runner.source.layer, arm=runner.source.arm, fold=fold), temp, compress=3)
        temp.replace(out)
    return scores, info


def select_layer(infos, target):
    key = 'p1_near' if target == 'near' else 'p1'
    usable = [(info[key]['val_auc'], -layer, layer) for layer, info in infos.items() if np.isfinite(info[key]['val_auc'])]
    if not usable:
        raise ValueError('No finite validation layer score')
    return max(usable)[2]


def near_statistics(pool, scores, W):
    pos, neg = pool.band_pos(0, 10), pool.clean_neg()
    d = pool.pair_def('near', pos, neg)
    out = {'m1': pool.auc(scores['p1_near'], d, W)}
    h1 = (pool.kind == M.KIND_CODE['h1']) & pool.h1p_v[pool.vidx] & ~pool.masked & (pool.tte > 0) & (pool.tte <= 10)
    out['m3p'] = pool.auc(scores['p1_near'], pool.pair_def('near_h1', pos & (pool.src == M.SRC_CODE['ucf']), h1, source=False), W)
    out['n'] = pool.auc(scores['n_near'], d, W)
    out['n_plus_p1'] = pool.auc(scores['n_plus_p1_near'], d, W)
    out['m4n'] = M.diff(out['n_plus_p1'], out['n'])
    near = copy.copy(pool)
    near.events = pool.events.copy()
    if len(near.events):
        near.events['win_lo'] = np.maximum(near.events.onset - 10, near.events.clear_start)
        near.events['win_hi'] = near.events.onset
        near.ev_ia = np.maximum(np.ceil(near.events.win_lo.to_numpy() - 1e-9).astype(int), 1) - 1
        near.ev_ib = np.ceil(near.events.win_hi.to_numpy() - 1e-9).astype(int) - 2
    out['events'] = M.event_metrics(near, scores['p1_near'], W)
    n1, n2, n3 = out['m1'], out['m3p'], out['m4n']
    out['gates'] = dict(N1=bool(n1.value is not None and n1.value >= .70 and n1.lower is not None and n1.lower > .60),
                        N2=bool(n2.lower is not None and n2.lower > .50),
                        N3=bool(n3.lower is not None and n3.lower > 0))
    return out


def all_statistics(pool, scores, val, W):
    out = M.evaluate_pool(pool, scores, val, W)
    first = np.empty(len(pool.t))
    for i in range(pool.V):
        sl = pool.sl(i)
        first[sl] = scores['p1'][min(sl.start + 1, sl.stop - 1)]
    d = pool.pair_def(('band', False), pool.band_pos(30, 120), pool.clean_neg())
    out['m4b_all_pilot'] = pool.auc(first, d, W)
    out['m1_gain_over_m4b_all_pilot'] = M.diff(out['m1'], out['m4b_all_pilot'])
    out['imminent'] = near_statistics(pool, scores, W)
    out['G2'] = D.gate_g2(serial(out.get('m3p')), serial(out.get('m3p_ratio')))
    out['G3'] = D.gate_g3(serial(out.get('m4n')))
    return out


def evaluate_arm(ss, arm, args):
    out = Path(args.out) / arm
    out.mkdir(parents=True, exist_ok=True)
    hidden = arm.startswith('hidden_')
    layers = (18, 27) if hidden else (None,)
    per_fold, choices, vts, main_runner = {}, {}, [], None
    for fold in ss.screen_folds:
        infos, runners, splits = {}, {}, {}
        for layer in layers:
            source = PilotSource(ss, arm, layer)
            runner = PilotRun(ss, source, require_cut=not args.allow_missing_cut)
            c = {k: [v for v in ids if runner.covered(v)] for k, ids in
                 [('train', ss.train_ids(fold)), ('val', ss.val_ids(fold)), ('test', ss.test_ids(fold))]}
            if not all(c.values()):
                raise ValueError(f'{arm}: fold {fold} lacks a train/val/test cohort')
            print(json.dumps(dict(arm=arm, fold=fold, layer=layer, counts={k: len(v) for k, v in c.items()})), flush=True)
            bundle_path = out / f'fold{fold}_layer{layer}.joblib'
            bundle = None
            if getattr(args, 'resume', False) and bundle_path.exists():
                try:
                    bundle = joblib.load(bundle_path)
                except (EOFError, ValueError, OSError):
                    bundle = None
            if bundle is not None:
                if bundle['arm'] != arm or bundle['fold'] != fold or bundle['layer'] != layer:
                    raise ValueError('Probe checkpoint does not match this candidate')
                runner.pls, runner.thr = bundle['pls'], bundle['thr']
                info = bundle.get('info') or {m: dict(C=p.C, val_auc=p.val_auc, path=p.path,
                            converged=p.converged, resumed_legacy_checkpoint=True) for m, p in bundle['models'].items()}
                sc = None
                if not hidden:
                    raise ValueError('Resume checkpoints are currently supported only for hidden probes')
            else:
                sc, info = fit_fold(runner, fold, c['train'], c['val'], [] if hidden else c['test'], bundle_path)
            infos[layer], runners[layer], splits[layer] = info, runner, c
            if not hidden:
                per_fold[fold] = sc
        selected = {head: select_layer(infos, head) if hidden else None for head in ('band', 'near')}
        choices[fold] = dict(selected_layers=selected, validation=infos)
        if hidden:
            combined = {}
            for layer in set(selected.values()):
                runner, c = runners[layer], splits[layer]
                bundle = joblib.load(out / f'fold{fold}_layer{layer}.joblib')
                sc = {m: {'test': {}, 'val': {}} for m in bundle['models']}
                for split in ('test', 'val'):
                    for vid in c[split]:
                        x = runner.feats(vid, 'p1')
                        n = bundle['nuisance'].features(runner.vt(vid), runner.src(vid))
                        for m, model in bundle['models'].items():
                            matrix = n if m in ('n', 'n_near') else np.concatenate([n, x], axis=1) if m.startswith('n_plus') else x
                            sc[m][split][vid] = model.score(matrix)
                for m in sc:
                    if selected['near' if 'near' in m else 'band'] == layer:
                        combined[m] = sc[m]
            per_fold[fold] = combined
        runner = runners[selected['band']]
        vts.extend(runner.vt(v) for v in splits[selected['band']]['test'])
        main_runner = runner
    pool = M.Pool(ss, vts)
    assembled = {m: pool.assemble({v: s for f in per_fold.values() for v, s in f[m]['test'].items()}) for m in per_fold[ss.screen_folds[0]]}
    val_scores = {}
    for model in assembled:
        val_scores[model] = {}
        for fold, sc in per_fold.items():
            parts = []
            for vid, score in sc[model]['val'].items():
                vt = main_runner.vt(vid)
                if vt.kind == 'clean':
                    parts.append(score[~vt.masked & (vt.remaining >= 60)])
            val_scores[model][fold] = np.concatenate(parts) if parts else np.zeros(0)
    W = pool.boot_weights(args.n_boot, seed=0)
    result = all_statistics(pool, assembled, val_scores, W)
    subsets = {'ubi': pool.src_name_v == 'ubi', 'official_ubi': np.array([
        ss.bench.videos.loc[v, 'onset_source'] in ('official', 'ubi') for v in pool.vids]),
        'hivau': np.array([ss.bench.videos.loc[v, 'onset_source'] == 'hivau' for v in pool.vids])}
    subsets.update({str(src): pool.src_name_v == src for src in np.unique(pool.src_name_v)})
    result['subsets'] = {}
    for name, selected in subsets.items():
        vids = [v for v, flag, kind in zip(pool.vids, selected, pool.kind_v) if flag or kind != M.KIND_CODE['violent']]
        sub = M.Pool(ss, [vt for vt in pool.vts if vt.vid in vids])
        idx = np.array([pool.index[v] for v in sub.vids])
        sc = {m: sub.assemble({v: score[pool.sl(pool.index[v])] for v in sub.vids}) for m, score in assembled.items()}
        result['subsets'][name] = all_statistics(sub, sc, val_scores, W[:, idx])
    frames = []
    for fold, scores in per_fold.items():
        for split in ('test', 'val'):
            for vid in scores['p1'][split]:
                vt = main_runner.vt(vid)
                keep = ~vt.masked
                frame = dict(video_id=vid, t_sec=vt.t[keep], fold=fold, seed=0, split=split)
                frame.update({m: s[split][vid][keep] for m, s in scores.items()})
                frames.append(pd.DataFrame(frame))
    df = pd.concat(frames, ignore_index=True)
    ss.assert_no_frozen(df)
    df.to_parquet(out / 'scores.parquet', index=False)
    result['config'] = dict(arm=arm, registration=REGISTRATION, registration_sha256=REGISTRATION_SHA,
                           interim_v1_labels=True, pilot_only=True, confirmation_fold_read=False,
                           bootstrap_cluster='split_group', n_boot=args.n_boot, layer_selection=choices,
                           invalid_ticks_excluded=True, go_decision='OWNER_ONLY')
    result['coverage'] = dict(videos=pool.V, events=len(pool.events), scored_rows=len(df),
                              by_kind={k: int((pool.kind_v == i).sum()) for k, i in M.KIND_CODE.items()})
    write_json(out / 'metrics.json', result)
    return result


def reference_comparison(ss, arm, out_root, n_boot):
    root = Path(out_root)
    source = PilotSource(ss, 'aggression')
    df = pd.read_parquet(root / arm / 'scores.parquet', filters=[('split', '==', 'test')])
    vids = sorted(set(df.video_id) & set(source.records) & set(ss.prereg.ids('teacher_videos')))
    pool = M.Pool(ss, [ss.ticks(v) for v in vids])
    W = pool.boot_weights(n_boot)
    scored = {v: g.set_index('t_sec') for v, g in df.groupby('video_id') if v in vids}
    vectors = {}
    for key in ('p1', 'p1_near'):
        vectors[key] = pool.assemble({v: scored[v][key].reindex(ss.ticks(v).t).fillna(-1e6).to_numpy() for v in vids})
    raw = pool.assemble({v: source.load('phase_c', v)[1][:, 0] for v in vids})
    out = dict(shared_videos=len(vids), shared_teacher_videos=len(set(vids) & set(ss.prereg.ids('teacher_videos'))),
               reference='no-context aggression_level, zero-shot probability', paired_cluster='split_group')
    for name, lo, hi, key in [('band', 30, 120, 'p1'), ('near', 0, 10, 'p1_near')]:
        d = pool.pair_def(name, pool.band_pos(lo, hi), pool.clean_neg())
        actual, baseline = pool.auc(vectors[key], d, W), pool.auc(raw, d, W)
        out[name] = dict(actual=actual, aggression=baseline, paired_gain=M.diff(actual, baseline))
        per_source = {}
        for src in np.unique(pool.src_name_v):
            pos = pool.band_pos(lo, hi, vsel=pool.src_name_v == src)
            sd = pool.pair_def((name, str(src)), pos, pool.clean_neg())
            a, b = pool.auc(vectors[key], sd, W), pool.auc(raw, sd, W)
            per_source[str(src)] = dict(actual=a, aggression=b, paired_gain=M.diff(a, b))
        out[name]['by_source'] = per_source
    mapping = catalog_mapping(ss.bench.videos.loc[vids])
    teacher = {}
    for path in sorted(Path('data/processed/vlm_labels').glob('*/*.jsonl')):
        vid = mapping.get((SOURCES.get(path.parent.name), path.stem))
        if vid is None or vid in teacher:
            continue
        times, values = read_teacher(path, float(ss.bench.videos.loc[vid, 'fps']))
        if len(times):
            teacher[vid] = sample_and_hold(times, values, ss.ticks(vid).t)
    shared = [v for v in vids if v in teacher]
    tp = M.Pool(ss, [ss.ticks(v) for v in shared])
    tw = tp.boot_weights(n_boot)
    baseline = tp.assemble(teacher)
    teacher_out = dict(shared_videos=len(shared), note='Legacy context-bearing b_teacher; control only, never a feature')
    for name, lo, hi, key in [('band', 30, 120, 'p1'), ('near', 0, 10, 'p1_near')]:
        actual = tp.assemble({v: vectors[key][pool.sl(pool.index[v])] for v in shared})
        d = tp.pair_def(name, tp.band_pos(lo, hi), tp.clean_neg())
        a, b = tp.auc(actual, d, tw), tp.auc(baseline, d, tw)
        teacher_out[name] = dict(actual=a, b_teacher=b, paired_gain=M.diff(a, b))
    out['b_teacher'] = teacher_out
    write_json(root / arm / 'reference_comparison.json', out)
    return out


def event_inputs(ss, arm, root):
    root = Path(root) / arm
    scores = pd.read_parquet(root / 'scores.parquet')
    ss.assert_no_frozen(scores)
    ids = sorted(v for v in set(scores.video_id) if ss.kind(v) != 'h1')
    folds = ss.bench.folds
    fmap = {f: i for i, f in enumerate(ss.screen_folds)}
    assignment = {v: fmap[folds.fold_of(v)] for v in ids}
    val = {str(fmap[f]): [v for v in ss.val_ids(f) if v in assignment] for f in ss.screen_folds}
    fd = dict(name='phase_c_pilot_without_original_fold1', version=2, n_folds=4,
              assignment=assignment, val=val, content_sha256=content_hash(assignment, val))
    write_json(root / 'folds_without_confirmation.json', fd)
    catalog = ss.bench.videos.loc[ids].copy()
    events = {}
    for vid in ids:
        lo, cap = ss.intervals[vid]
        cap = min(cap, ss.ticks(vid).n)
        catalog.loc[vid, 'duration_sec'] = min(float(catalog.loc[vid, 'duration_sec']), cap)
        e = copy.deepcopy(ss.bench.events.get(vid, {}))
        e['segments'] = [[s, min(end, cap)] for s, end in e.get('segments', []) if s < cap]
        if lo > 1:
            e.setdefault('soft_mask_spans', [])
            e['soft_mask_spans'] = (e['soft_mask_spans'] or []) + [[0, lo - 1e-6]]
        events[vid] = e
    bench = va30.Benchmark(catalog, events, Folds(fd), provenance={**ss.bench.provenance,
                           'phase_c_pilot_intervals': True, 'original_to_event_fold_map': fmap},
                           video_horizons=S.MANIFEST_V1)
    filtered = scores[scores.video_id.isin(ids)].copy()
    filtered['fold'] = filtered['fold'].map(fmap)
    return bench, filtered


class LeadNull:
    def __init__(self, ee, lead):
        self.ee, self.lead = ee, lead
        self.ts, self.ev_e30, self.ev_v, self.ev_lo = ee.ts, ee.ev_e30, ee.ev_v, ee.ev_lo
        self.ev_on = ee.ev_on + (30 - lead)

    def fast(self, threshold):
        ee = self.ee
        vi, t, _ = ee._starts(threshold)
        clean = ee.ts.clean_v[vi] if len(vi) else np.zeros(0, bool)
        n = int(ee._outside(vi[clean], t[clean]).sum()) if clean.any() else 0
        hit = np.nan_to_num(ee._leads(vi, t), nan=-1) >= self.lead
        eligible, late = ee.ev_e30, ee.ev_e30 & ee.ev_late
        return (n / ee.clean_hours if ee.clean_hours else np.nan,
                float(hit[eligible].mean()) if eligible.any() else np.nan,
                float(hit[late].mean()) if late.any() else np.nan)


def null_group_folds(ts):
    result = ts.fold_v.copy()
    groups = ts.bench.videos.loc[ts.vids, 'split_group'].astype(str).to_numpy()
    for group in np.unique(groups):
        selected = groups == group
        if len(np.unique(result[selected])) > 1:
            result[selected] = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 4
    return result


def lead_metrics(bench, df, lead, n_boot):
    cfg = va30.EvalConfig(n_boot=n_boot, cluster='split_group', restrict_to_scored=True, allow_missing=True)
    table = va30.ScoreTable(df, bench.video_horizons)
    ids = set(table.videos) & set(bench.eval_ids(tuple(va30.TRUSTED_ONSETS) + ('curated_visual',)))
    tests = {f: [v for v in bench.folds.test_ids(f) if v in ids] for f in range(4)}
    vals = {f: [v for v in bench.folds.val_ids(f) if v in ids] for f in range(4)}
    test_scores = va30._collect(bench, table, 0, tests, cfg, {})
    fold_of = {v: f for f, vids in tests.items() for v in vids}
    ts = va30.TickSet(bench, [v for f in tests for v in tests[f] if v in test_scores], test_scores, fold_of)
    W = va30.bootstrap_weights(ts, n_boot, 0, 'split_group')
    ee = va30.EventEval(ts, cfg.policy)
    validation = {}
    for f in vals:
        vs = va30._collect(bench, table, 0, {f: vals[f]}, cfg, {})
        validation[f] = va30.EventEval(va30.TickSet(bench, list(vs), vs), cfg.policy)
    null_ee = copy.copy(ee)
    null_ee.ts = copy.copy(ts)
    null_ee.ts.fold_v = null_group_folds(ts)
    null = va30.position_null(LeadNull(null_ee, lead))
    event_src = bench.videos.loc[np.array(ts.vids), 'source_kind'].values[ee.ev_v]
    onset_src = bench.videos.loc[np.array(ts.vids), 'onset_source'].values[ee.ev_v]

    def rate(hit, sel):
        sel = sel & ee.ev_e30
        if not sel.any():
            return M.Stat(None, n_events=0)
        w = W[:, ee.ev_v[sel]]
        with np.errstate(invalid='ignore', divide='ignore'):
            boot = (w * hit[sel]).sum(1) / w.sum(1)
        return M.Stat(float(hit[sel].mean()), boot, n_events=int(sel.sum()))

    ops = {}
    for far in (1, 2, 5):
        theta = {f: va30.choose_threshold(validation[f], far)[0] for f in vals}
        outcome = ee.outcomes(np.array([theta[f] for f in ts.fold_v]))
        hit = np.nan_to_num(outcome['lead'], nan=-1) >= lead
        achieved = float(outcome['fa'][ts.clean_v].sum() / ee.clean_hours) if ee.clean_hours else None
        ops[str(far)] = dict(target_fa_per_hour=far, achieved_fa_per_hour=achieved,
                            success=rate(hit, np.ones(len(hit), bool)), position_null=va30.null_at(null, achieved),
                            random=None if achieved is None else 1 - np.exp(-achieved * (120 - lead) / 3600),
                            by_source={str(s): rate(hit, event_src == s) for s in sorted(set(event_src))},
                            official_ubi=rate(hit, np.isin(onset_src, ['official', 'ubi'])))
    return dict(lead_sec=lead, ci='split_group, fixed validation-selected thresholds',
                position_null_fold_policy='Copies sharing split_group stay in the same position-null fold',
                clean_hours=ee.clean_hours, events=int(ee.ev_e30.sum()), operating_points=ops)


def evaluate_events(ss, arm, args):
    root = Path(args.out) / arm
    bench, scores = event_inputs(ss, arm, args.out)
    outputs = {}
    for head, column in [('early', 'p1'), ('imminent', 'p1_near')]:
        df = scores[['video_id', 't_sec', 'fold', 'seed', column]].rename(columns={column: 'score'})
        df.to_parquet(root / f'va30_{head}_scores.parquet', index=False)
        cfg = va30.EvalConfig(n_boot=args.n_boot, cluster='split_group', restrict_to_scored=True,
                             allow_missing=True, threshold_bootstrap=args.threshold_bootstrap,
                             onset_sources=tuple(va30.TRUSTED_ONSETS) + ('curated_visual',),
                             position_null=not arm.startswith('holdout_'))
        result = va30.evaluate(df, bench, cfg, name=f'phase_c_{arm}_{head}')
        va30.save_results(result, str(root / f'va30_{head}'))
        outputs[head] = {str(lead): lead_metrics(bench, df, lead, args.n_boot) for lead in (10, 30)}
    write_json(root / 'event_metrics.json', outputs)
    return outputs


def prepare_holdout(out_root):
    root = Path(out_root) / 'holdout'
    root.mkdir(parents=True, exist_ok=True)
    path = Path('benchmark/youtube_holdout/annotations_v2.json')
    annotations = json.loads(path.read_text())
    rows = []
    for vid, video in annotations['videos'].items():
        for index, clip in enumerate(video['clips']):
            selected = [o for o in clip['onsets'] if o['onset_sec'] - clip['start_sec'] >= 30] if clip['content'] == 'incident_raw' else []
            if not selected:
                continue
            duration = clip['end_sec'] - clip['start_sec']
            stem = vid.split('/', 1)[1]
            target = root / 'clips' / f'{stem}_c{index:03d}.mp4'
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                original = Path('data/anticipation_curation/videos') / (stem + '.mp4')
                subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-threads', '2', '-filter_threads', '1',
                                '-i', str(original), '-ss', str(clip['start_sec']),
                                '-t', str(duration), '-an', '-vf', 'scale=-2:448', '-c:v', 'libx264', '-preset', 'ultrafast',
                                '-crf', '18', '-threads', '2', '-y', str(target.with_suffix('.tmp.mp4'))], check=True)
                target.with_suffix('.tmp.mp4').replace(target)
            rows.append(dict(video_id=f'yt_clip/{stem}_c{index:03d}', original_video_id=vid, clip_index=index,
                             path=str(target.resolve()), start_sec=clip['start_sec'], end_sec=clip['end_sec'],
                             duration_sec=duration, camera=clip['camera'], annotations=clip['onsets'],
                             nominated_onsets=len(selected)))
    if len({r['original_video_id'] for r in rows}) != 14 or sum(r['nominated_onsets'] for r in rows) != 36:
        raise ValueError('Hold-out membership differs from the requested 14-video / 36-onset cohort')
    data = dict(annotation_path=str(path), annotation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                timeline='Independent decoded incident clips, timestamps relative to clip start',
                videos=14, clips=len(rows), onsets=36, rows=rows)
    write_json(root / 'clips.json', data)
    return data


def extract_holdout(arm, out_root, chunk_sec=1200):
    sys.path.insert(0, str(PROTOCOL.resolve()))
    from protocol import gpu_budget, start_gate
    manifest = json.loads((Path(out_root) / 'holdout' / 'clips.json').read_text())
    if arm not in ('A', 'B', 'C'):
        raise ValueError('Hold-out extraction needs an image/video arm')
    gate = start_gate()
    if gate['status'] != 'READY':
        raise RuntimeError('GPU 0/CCTV start gate is closed')
    name = 'vlm_holdout_' + arm.lower()
    pending = [r for r in manifest['rows'] if not Path(io.feature_path(name, r['video_id'])).exists()]
    if pending:
        with gpu_budget(PROTOCOL / 'budget.json', VLM.MODEL_3 + '::holdout_' + arm, chunk_sec) as deadline:
            import torch
            torch.set_num_threads(4)
            torch.cuda.set_device(1)
            torch.cuda.set_per_process_memory_fraction(.75, 1)
            for row in pending:
                if time.perf_counter() > deadline - 30:
                    break
                try:
                    VLM.extract_and_save(row['path'], row['video_id'], 'cuda:1', output_name=name,
                                         model_id=VLM.MODEL_3, arm=arm, max_duration_sec=row['duration_sec'],
                                         deadline_monotonic=deadline - 20, rationale_samples=0)
                except TimeoutError:
                    break
    remaining = [r['video_id'] for r in manifest['rows'] if not Path(io.feature_path(name, r['video_id'])).exists()]
    report = dict(arm=arm, stream=name, clips=len(manifest['rows']), remaining=remaining,
                  complete=not remaining, screening_set_S_started=False, go_decision='OWNER_ONLY')
    write_json(Path(out_root) / 'holdout' / ('extraction_' + arm + '.json'), report)
    print(json.dumps(report), flush=True)


def holdout_scene(out_root):
    sys.path.insert(0, str(PROTOCOL.resolve()))
    from protocol import gpu_budget, start_gate
    from anticipation.features import pose, scene0p, siglip
    if start_gate()['status'] != 'READY':
        raise RuntimeError('GPU 0/CCTV start gate is closed')
    manifest = json.loads((Path(out_root) / 'holdout/clips.json').read_text())
    with gpu_budget(PROTOCOL / 'budget.json', 'holdout_scene0p', 1200):
        import torch
        torch.set_num_threads(4)
        torch.cuda.set_device(1)
        detector = pose.load_model(pose.DEFAULTS['weights'], 'cuda:1')
        model, _ = siglip.load_model('cuda:1')
        for row in manifest['rows']:
            path = Path(io.feature_path('vlm_holdout_scene0p', row['video_id']))
            if path.exists():
                continue
            video = io.Video(row['path'], num_threads=1)
            frames = video.frames_at([2.0])
            detections = detector(frames)[0]
            masked = scene0p.grey_boxes(frames[0], detections[detections[:, 4] >= .25, :4])
            embedding = scene0p.embed([masked], model, 'cuda:1')[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, emb=embedding.astype(np.float16), meta=json.dumps(dict(
                extractor='vlm_holdout_scene0p', t_sec=2, original_video=row['original_video_id'],
                clip_start_sec=row['start_sec'], detector='YOLO11x-pose, standalone frame, conf>=.25',
                pipeline='scene0p.grey_boxes and scene0p.embed; clip-relative t=2; no earlier clip context')))


class HoldoutSet(S.ScreenSet):
    def __init__(self, pilot, manifest):
        registered_ids = sorted(pilot.intervals)
        cat = pilot.bench.videos.loc[registered_ids].copy()
        events = {v: pilot.bench.events.get(v, {}) for v in registered_ids}
        assignment = {v: pilot.fold_of(v) for v in registered_ids}
        self.external = {}
        self.intervals = pilot.intervals.copy()
        additions = []
        for fold in pilot.screen_folds:
            for row in manifest['rows']:
                vid = f'yt_external/f{fold}/' + row['video_id'].split('/', 1)[1]
                record = pilot.bench.videos.loc[row['original_video_id']].copy()
                record['video_id'], record['duration_sec'] = vid, row['duration_sec']
                record['role'], record['onset_source'], record['source_kind'] = 'violent', 'curated_visual', 'xd_youtube'
                record['group'] = record['split_group'] = row['original_video_id']
                record.name = vid
                additions.append(record)
                events[vid] = {'segments': [[o['onset_sec'] - row['start_sec'], o['end_sec'] - row['start_sec']]
                                            for o in row['annotations']]}
                assignment[vid] = fold
                self.external[vid] = row
                self.intervals[vid] = (1., row['duration_sec'])
        cat = pd.concat([cat, pd.DataFrame(additions)])
        val = {str(f): [v for v in pilot.val_ids(f) if v in assignment] for f in pilot.screen_folds}
        fd = dict(name='external_holdout_screen_models', version=2, n_folds=5, assignment=assignment,
                  val=val, content_sha256=content_hash(assignment, val))
        bench = va30.Benchmark(cat, events, Folds(fd), video_horizons=S.MANIFEST_V1)
        super().__init__(bench, pilot.h1.reset_index(drop=True))

    def events(self):
        if self._events is not None:
            return self._events
        rows = []
        for vid in self.external:
            for event in self.bench.event_rows(vid):
                if not event['eligible30']:
                    continue
                row = self._event_row(vid, self.fold_of(vid), event, 'xd_youtube', 'curated_visual', False, event['index'] == 0)
                row.update(s_real=row['first'], s_long60=row['first'] and event['eligible60'],
                           s_ramp=row['first'] and event['max_lead'] >= 90, s_offubi=False, s_audio=False)
                rows.append(row)
        self._events = pd.DataFrame(rows)
        return self._events


class HoldoutSource(PilotSource):
    def _path(self, stream, vid):
        if vid in self.ss.external:
            parent = self.ss.external[vid]['video_id']
            if stream in ('vlm_a', 'vlm_b', 'vlm_c', 'vlm_a_hidden', 'vlm_b_hidden', 'vlm_c_hidden'):
                return io.feature_path('vlm_holdout_' + stream.removeprefix('vlm_'), parent)
        return super()._path(stream, vid)

    def load(self, stream, vid):
        if vid in self.ss.external and stream == 'phase_c':
            self.ss.guard(vid)
            parent = self.ss.external[vid]['video_id']
            name = 'vlm_holdout_' + self.parent_stream().removeprefix('vlm_')
            return self.select_columns(*io.load_features(name, parent))
        if vid in self.ss.external and stream == F.CUT_STREAM:
            vt = self.ss.ticks(vid)
            return vt.t, np.zeros((vt.n, 1), np.float32), np.ones(vt.n, bool), {'columns': [F.CUT_COLUMN]}
        return super().load(stream, vid)

    def scene(self, vid):
        self.ss.guard(vid)
        if vid in self.ss.external:
            path = io.feature_path('vlm_holdout_scene0p', self.ss.external[vid]['video_id'])
            with np.load(path, allow_pickle=False) as z:
                return z['emb'].astype(np.float32)
        return super().scene(vid)


def evaluate_holdout(pilot, arm, args):
    manifest = json.loads((Path(args.out) / 'holdout/clips.json').read_text())
    ss = HoldoutSet(pilot, manifest)
    original_metrics = json.loads((Path(args.out) / arm / 'metrics.json').read_text())
    model_root = Path(args.out) / arm
    scores, vts, val_scores = {}, [], {}
    for fold in ss.screen_folds:
        chosen = original_metrics['config']['layer_selection'][str(fold)]['selected_layers']
        frame_vids = [v for v in ss.external if ss.fold_of(v) == fold]
        controls = [v for v in ss.test_ids(fold, kinds=('clean', 'h1'))]
        fold_scores = {}
        for head in ('band', 'near'):
            layer = chosen[head]
            bundle = joblib.load(model_root / f'fold{fold}_layer{layer}.joblib')
            source = HoldoutSource(ss, arm, layer)
            runner = PilotRun(ss, source)
            runner.pls, runner.thr = bundle['pls'], bundle['thr']
            nuisance = bundle['nuisance']
            nuisance.source = source
            models = [m for m in bundle['models'] if ('near' in m) == (head == 'near')]
            for vid in frame_vids + controls:
                x = runner.feats(vid, 'p1')
                n = nuisance.features(runner.vt(vid), runner.src(vid))
                if head == 'band':
                    vts.append(runner.vt(vid))
                for name in models:
                    matrix = n if name in ('n', 'n_near') else np.concatenate([n, x], axis=1) if name.startswith('n_plus') else x
                    fold_scores.setdefault(name, {})[vid] = bundle['models'][name].score(matrix)
        for name, d in fold_scores.items():
            scores.setdefault(name, {}).update(d)
    pool = M.Pool(ss, vts)
    assembled = {m: pool.assemble(d) for m, d in scores.items()}
    W = pool.boot_weights(args.n_boot)
    for name in assembled:
        val_scores[name] = {}
    result = all_statistics(pool, assembled, val_scores, W)
    pos, neg = pool.band_pos(30, 120), pool.clean_neg()
    h1 = (pool.kind == M.KIND_CODE['h1']) & pool.h1p_v[pool.vidx] & ~pool.masked & (pool.tte > 30) & (pool.tte <= 120)
    result['external_h1p_specificity'] = pool.auc(assembled['p1'], pool.pair_def('external_h1p', pos, h1, source=False), W)
    result['external_h1p_ratio'] = M.ratio_stat(result['external_h1p_specificity'], result['m1'])
    near_h1 = (pool.kind == M.KIND_CODE['h1']) & pool.h1p_v[pool.vidx] & ~pool.masked & (pool.tte > 0) & (pool.tte <= 10)
    result['external_h1p_near'] = pool.auc(assembled['p1_near'], pool.pair_def('external_h1p_near', pool.band_pos(0, 10), near_h1, source=False), W)
    result['config'] = dict(pilot_arm=arm, no_holdout_fit=True, no_holdout_selection=True,
                           annotations_sha256=manifest['annotation_sha256'], original_videos=14, incident_clips=21,
                           same_clip_lead30_onsets=36, unique_clear_eligible30_events=len(pool.events) // 4,
                           unique_first_events=int(pool.events['first'].sum()) // 4,
                           method='Four frozen fold models score each clip; copies share original-video split_group; clean/H1 controls are out-of-fold pilot controls',
                           source_matching='Curated real footage mapped to xd_youtube; no holdout-specific clean negatives available',
                           specificity='External versus UCF H1p is time-matched but crosses datasets; registered UCF-only N2 is NA')
    root = Path(args.out) / ('holdout_' + arm)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / 'metrics.json', result)
    frames = []
    for vid in pool.vids:
        vt = next(v for v in pool.vts if v.vid == vid)
        keep = ~vt.masked
        frame = dict(video_id=vid, t_sec=vt.t[keep], fold=ss.fold_of(vid), seed=0, split='test')
        frame.update({m: d[vid][:vt.n][keep] for m, d in scores.items()})
        frames.append(pd.DataFrame(frame))
    validation = pd.read_parquet(model_root / 'scores.parquet', filters=[('split', '==', 'val')])
    validation = validation[validation.video_id.isin(ss.intervals)]
    df = pd.concat(frames + [validation], ignore_index=True)
    df.to_parquet(root / 'scores.parquet', index=False)
    evaluate_events(ss, 'holdout_' + arm, args)
    return result


def report(out_root):
    root = Path(out_root)
    metrics = {a: json.loads((root / a / 'metrics.json').read_text()) for a in ARMS
               if (root / a / 'metrics.json').exists()}

    def stat(value):
        if not value or value.get('value') is None:
            return 'NA'
        ci = value.get('ci')
        return f"{value['value']:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ' [CI unavailable]')

    def point(value):
        return 'NA' if not value or value.get('value') is None else f"{value['value']:.3f}"

    lines = ['# Phase C pilot: interim v1 labels', '',
             '| arm | M1 band | M1 near | M8 near | M3p near | near gain over N | Success@10 / achieved FA/h | N1/N2/N3 | G2/G3 |',
             '|---|---|---|---|---|---|---|---|---|']
    for arm in ARMS:
        if arm not in metrics:
            lines.append(f'| {arm} | pending | | | | | | | |')
            continue
        m = metrics[arm]
        n = m['imminent']
        gates = '/'.join('PASS' if n['gates'][k] else 'FAIL' for k in ('N1', 'N2', 'N3'))
        events_path = root / arm / 'event_metrics.json'
        event = json.loads(events_path.read_text())['imminent']['10']['operating_points']['2'] if events_path.exists() else None
        success = f"{stat(event['success'])} / {event['achieved_fa_per_hour']:.2f}" if event else 'pending'
        lines.append(f"| {arm} | {stat(m['m1'])} | {stat(m['m1_near'])} | {stat(n['events'].get('m8'))} | "
                     f"{stat(n.get('m3p'))} | {stat(n.get('m4n'))} | {success} | {gates} | {m['G2']}/{m['G3']} |")
    lines += ['', 'Owner decides GO/NO-GO and the chosen arm. No decision artifact is written and no rest-of-S extraction is started.', '',
              'All CIs use 1,000 paired split_group cluster-bootstrap replicates; undefined / insufficient-replicate CIs remain NA. '
              'Models use the registered P1 summaries, train-only weighted PLS32 for wide states, validation C selection, and no H1 training. '
              'Hidden layers 18/27 are selected separately for each head on validation slices only. '
              'A/B/C use identical 352-video pilot labels and intervals (100 positive, 150 clean, 102 H1p). '
              'The aggression P1 reference has narrower cache coverage and must not be compared as an unrestricted row.', '',
              'Success@10 in the top table uses the nominal validation 2 FA/h operating point; achieved test rates are shown explicitly.', '',
              '## Early metrics and temporal control', '',
              '| arm | M1 long | M4b, all P | paired M1 minus M4b | registered S_ramp M4b | paired S_ramp gain | M4n | M3p | specificity ratio | M8 |',
              '|---|---|---|---|---|---|---|---|---|---|']
    for arm, m in metrics.items():
        keys = [m.get('m1_long'), m.get('m4b_all_pilot'), m.get('m1_gain_over_m4b_all_pilot'),
                m.get('m4b_first_frame'), m.get('m1_ramp_gain_over_m4b'), m.get('m4n'), m.get('m3p'),
                m.get('m3p_ratio'), m.get('events', {}).get('m8')]
        lines.append('| ' + arm + ' | ' + ' | '.join(stat(x) for x in keys) + ' |')
    lines += ['', '## Shared-reference and real-footage checks', '',
              '| arm | shared teacher videos | paired band gain vs raw aggression | criterion (i) | criterion (ii): G2 not FAIL | S_real band | criterion (iii) | UBI band | UBI near | official/UBI band |',
              '|---|---|---|---|---|---|---|---|---|---|']
    baseline_g3 = metrics.get('aggression', {}).get('G3')
    for arm, m in metrics.items():
        path = root / arm / 'reference_comparison.json'
        reference = json.loads(path.read_text()) if path.exists() else {}
        gain = reference.get('band', {}).get('paired_gain')
        ci = (gain or {}).get('ci')
        criterion_i = bool(ci and ci[0] > 0) or (m['G3'] == 'PASS' and baseline_g3 == 'FAIL')
        real = m.get('breakdowns', {}).get('s_real')
        criterion_iii = bool(real and real.get('value') is not None and real['value'] >= .60 and real.get('ci') and real['ci'][0] > .50)
        ubi = m.get('subsets', {}).get('ubi', {})
        off = m.get('subsets', {}).get('official_ubi', {})
        lines.append(f"| {arm} | {reference.get('shared_teacher_videos', 'pending')} | {stat(gain)} | "
                     f"{'PASS' if criterion_i else 'NOT MET'} | {'PASS' if m['G2'] != 'FAIL' else 'FAIL'} | {stat(real)} | "
                     f"{'PASS' if criterion_iii else 'NOT MET'} | {stat(ubi.get('m1'))} | {stat(ubi.get('m1_near'))} | {stat(off.get('m1'))} |")
    lines += ['', 'Criterion (iv): **UCF contamination flagged** in both violent and H1 strata. No automatic pass is assigned. '
              'UBI and the curated hold-out must inform the owner decision. Every metric by source / official-UBI / HIVAU is in each metrics.json `subsets`; '
              'source-specific UCF-vs-H1p specificity is undefined for UBI-only positives. Primary v1 labels remain unchanged.', '',
              '### Legacy b_teacher control on exactly shared videos', '',
              '| arm | shared videos | b_teacher band | actual band | paired gain |',
              '|---|---|---|---|---|']
    for arm in metrics:
        path = root / arm / 'reference_comparison.json'
        reference = json.loads(path.read_text()) if path.exists() else {}
        teacher = reference.get('b_teacher', {})
        band = teacher.get('band', {})
        lines.append(f"| {arm} | {teacher.get('shared_videos', 'pending')} | {stat(band.get('b_teacher'))} | "
                     f"{stat(band.get('actual'))} | {stat(band.get('paired_gain'))} |")
    lines += ['',
              '## Event operating points', '',
              'Thresholds are selected on each model fold’s clean validation slices at nominal 1/2/5 FA/h. '
              'Report achieved test FA/h alongside success: these can differ substantially on this small pilot. '
              'Success CIs below are paired split_group resamples at fixed selected thresholds; VA30 files additionally contain threshold-aware Success@30 CIs (200 replicates). '
              'The evaluator uses restrict_to_scored and benchmark/extract_manifest_v1.csv horizons; positives are additionally limited to frozen P intervals. '
              'Original fold 1 is removed before the event evaluator remaps the other folds.', '',
              '| arm / head / lead | nominal FA/h | achieved test FA/h | success | position null | random | UBI success | official/UBI success |',
              '|---|---|---|---|---|---|---|---|']
    for arm in metrics:
        path = root / arm / 'event_metrics.json'
        if not path.exists():
            continue
        events = json.loads(path.read_text())
        for head, lead in [('imminent', '10'), ('early', '30')]:
            for far, op in events[head][lead]['operating_points'].items():
                achieved = op['achieved_fa_per_hour']
                lines.append(f"| {arm} / {head} / {lead}s | {far} | {achieved:.3f} | {stat(op['success'])} | "
                             f"{op['position_null']:.3f} | {op['random']:.3f} | {stat(op['by_source'].get('ubi'))} | {stat(op['official_ubi'])} |")
    lines += ['', 'Both leads for both heads, all per-source event rows, and validation/test coverage are saved in each `event_metrics.json`.', '',
              '## Curated hold-out', '',
              'The frozen annotation cohort has **36 onsets in 14 videos / 21 raw incident clips** with at least 30 s since clip start. '
              'Accounting for prior violence, merged contact bouts, aftermath and the 2-of-3 alarm clock leaves **11 clear VA30-eligible events, 10 first events**. '
              'Features are re-extracted on per-clip cuts; history cannot cross an editorial boundary. Hold-out values are never used to fit or tune.', '']
    for path in sorted(root.glob('holdout_*/metrics.json')):
        m = json.loads(path.read_text())
        lines += [f"{path.parent.name}: M1 band {stat(m.get('m1'))}; M1 near {stat(m.get('m1_near'))}; "
                  f"M8 {stat(m.get('events', {}).get('m8'))}; M8-near {stat(m.get('imminent', {}).get('events', {}).get('m8'))}.", '',
                  'The four frozen fold models each score every held-out clip. Their copies share the original video’s bootstrap cluster; '
                  'controls are the corresponding out-of-fold pilot clean/H1 videos. Real hold-out footage is source-matched to XD YouTube clean controls. '
                  'External H1p specificity crosses collections and is descriptive; the registered UCF-only N2 is NA. '
                  'Clip-relative person-masked scene embeddings use the same SigLIP/grey-box pipeline, with standalone >=.25-confidence pose boxes at t=2 s.', '']
        lines += [f"M1-long {stat(m.get('m1_long'))}; M4b {stat(m.get('m4b_all_pilot'))}; paired temporal gain "
                  f"{stat(m.get('m1_gain_over_m4b_all_pilot'))}; M4n {stat(m.get('m4n'))}; external H1p specificity "
                  f"{stat(m.get('external_h1p_specificity'))}, ratio {stat(m.get('external_h1p_ratio'))}; external near specificity "
                  f"{stat(m.get('external_h1p_near'))}; near gain over N {stat(m['imminent'].get('m4n'))}.", '']
        event_path = path.parent / 'event_metrics.json'
        if event_path.exists():
            events = json.loads(event_path.read_text())
            lines += ['| head / lead | nominal FA/h | achieved FA/h | success | position null | random |',
                      '|---|---|---|---|---|---|']
            for head, lead in [('imminent', '10'), ('early', '30')]:
                for far, op in events[head][lead]['operating_points'].items():
                    lines.append(f"| {head} / {lead}s | {far} | {op['achieved_fa_per_hour']:.3f} | {stat(op['success'])} | "
                                 f"{op['position_null']:.3f} | {op['random']:.3f} |")
            lines += ['', 'The frozen pilot validation timelines, including their positive score candidates, are retained for exact threshold transfer. '
                      'Position-null folds keep all copies of an original incident together, so a copy cannot supply onset-position labels to another copy. '
                      'High near-window AUROC alone does not establish a usable >=10 s alarm lead; assess the event operating points separately.', '']
    if not list(root.glob('holdout_*/metrics.json')):
        lines += ['Hold-out extraction/evaluation pending.', '']
    budget = json.loads((PROTOCOL / 'budget.json').read_text())
    lines += ['## Cost and reproducibility', '',
              f"GPU use charged: {budget['spent_sec'] / 3600:.3f} h plus {budget['prior_work_reserved_sec'] / 3600:.3f} h reserved prior work; "
              f"active reservations {sum(budget['active'].values()) / 3600:.3f} h, hard total cap 35 h, cuda:1 only.", '',
              'Timing on 200 queries per Qwen3 arm: A 10.30x / B 6.45x / C 8.04x realtime; projected 240 h costs 23.31 / 37.20 / 29.84 GPU-h, excluding reloads. '
              'Pretrained cut-causality and all 352-video A/B/C cache audits pass exactly. '
              'Cache validity is applied before fitting and metrics; feature windows reset at invalid gaps. '
              'Model bundles remain local; score tables, provenance, metrics and tests are versioned. No fold-1 data or results were read.', '']
    (root / 'pilot_report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--out', default='benchmark/results/vlm_pilot')
    parser.add_argument('--n-boot', type=int, default=1000)
    parser.add_argument('--allow-missing-cut', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--postprocess', action='store_true')
    parser.add_argument('--reference-only', action='store_true')
    parser.add_argument('--threshold-bootstrap', type=int, default=200)
    parser.add_argument('--prepare-holdout', action='store_true')
    parser.add_argument('--extract-holdout', choices=['A', 'B', 'C'])
    parser.add_argument('--holdout-scene', action='store_true')
    parser.add_argument('--evaluate-holdout', choices=ARMS)
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    if args.report:
        report(args.out)
        return
    if args.prepare_holdout:
        prepare_holdout(args.out)
        return
    if args.extract_holdout:
        extract_holdout(args.extract_holdout, args.out)
        return
    if args.holdout_scene:
        holdout_scene(args.out)
        return
    rows, plan = pilot_rows()
    pr = prereg.load(REGISTRATION, pinned={Path(REGISTRATION).name: REGISTRATION_SHA})
    base = S.load_screen_set(prereg=pr, folds='benchmark/folds_v2.json')
    fatal, _ = prereg.verify_inputs(pr, base.bench)
    if fatal:
        raise ValueError('Frozen v1 inputs changed: ' + '; '.join(fatal))
    ss = PilotSet(base, rows)
    va30.N_THREADS = 1
    with threadpool_limits(limits=1):
        if args.evaluate_holdout:
            evaluate_holdout(ss, args.evaluate_holdout, args)
            return
        for arm in args.arms:
            if args.postprocess or args.reference_only:
                reference_comparison(ss, arm, args.out, args.n_boot)
                if not args.reference_only:
                    evaluate_events(ss, arm, args)
            else:
                evaluate_arm(ss, arm, args)


if __name__ == '__main__':
    main()
