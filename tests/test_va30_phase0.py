import json

import numpy as np
import pytest

from anticipation.eval import va30
from anticipation.eval.phase0 import (band_stat, group_weights, person_counts, sustained_outcomes,
    sustained_position_null, sustained_summary, time_fraction_threshold, warning_episodes, window_max_auc)
from anticipation.eval.va30 import AlarmPolicy, EventEval, TickSet, evaluate
from tests.test_va30 import make_bench, quick_cfg, score_table, standard_specs


def event_evaluator(scores, onset=200, duration=400, segments=None, clean_duration=None, policy=None):
    b = make_bench([dict(vid="v", duration=duration, segments=segments or [[onset, onset+10]], role="violent", fold=0),
                    dict(vid="n", duration=clean_duration or duration, role="clean_negative", fold=1)])
    values = {v: np.asarray(scores(v,b.labels(v)), float) for v in b.folds.video_ids}
    ts = TickSet(b, b.folds.video_ids, values)
    return EventEval(ts, policy or AlarmPolicy())


def test_band_matches_source_and_position_and_masks_detection_memory():
    b = make_bench(standard_specs(8,8))
    values = {v:(b.labels(v).t_sec<100).astype(float) for v in b.folds.video_ids}
    ts = TickSet(b,b.folds.video_ids,values)
    result = band_stat(ts,group_weights(ts,40,0))
    assert result['value']==pytest.approx(.5,abs=.03)
    assert result['interval_sec']==[30,120]
    for v in b.folds.video_ids:
        values[v]=(~b.labels(v).pre_violence).astype(float)
    ts=TickSet(b,b.folds.video_ids,values)
    assert band_stat(ts,group_weights(ts,20,0))['value']==pytest.approx(.5)


def test_new_bootstrap_uses_split_group_instead_of_old_group():
    ee=event_evaluator(lambda v,l:np.zeros(l.T))
    ee.ts.bench.videos['split_group']='shared'
    W=group_weights(ee.ts,50,1)
    np.testing.assert_array_equal(W[:,0],W[:,1])
    assert (W==1).all()
    np.testing.assert_array_equal(W,va30.bootstrap_weights(ee.ts,50,1,cluster="split_group"))


def test_long_running_warning_gets_secondary_credit_without_legacy_credit():
    ee=event_evaluator(lambda v,l:np.ones(l.T) if v=='v' else np.zeros(l.T))
    theta=np.full(ee.ts.V,.5)
    assert np.isnan(ee.outcomes(theta)['lead'][0])
    out=sustained_summary(ee,theta,group_weights(ee.ts,30,0))
    assert out['success_sus60']['value']==pytest.approx(1)
    assert out['success_sus90']['value']==pytest.approx(1)
    assert out['success_sus120']['value']==pytest.approx(1)
    assert out['alarm_time_fraction_clean']['value']==0


@pytest.mark.parametrize('start,end,success',[(130,175,1),(150,175,0),(2,100,0)])
def test_warning_must_start_early_enough_and_reach_onset_minus30(start,end,success):
    ee=event_evaluator(lambda v,l:((l.t_sec>=start)&(l.t_sec<=end)).astype(float))
    out=sustained_summary(ee,np.full(ee.ts.V,.5),np.zeros((0,ee.ts.V)))
    assert out['success_sus60']['value']==success


def test_warning_merge_gaps_are_charged_as_active_time():
    p=AlarmPolicy(k=1,n=1,merge_gap_sec=10)
    ee=event_evaluator(lambda v,l:(((l.t_sec>=2)&(l.t_sec<=90))|((l.t_sec>=100)&(l.t_sec<=180))).astype(float),policy=p)
    episodes,active=warning_episodes(ee.ts,np.full(ee.ts.V,.5),p)
    assert all(len(episodes[episodes[:,0]==i])==1 for i in range(ee.ts.V))
    assert active[ee.ts.t==95].all()
    assert sustained_summary(ee,np.full(ee.ts.V,.5),np.zeros((0,ee.ts.V)))['success_sus120']['value']==pytest.approx(1)


def test_warning_already_active_before_later_clear_start_cannot_restart():
    ee=event_evaluator(lambda v,l:np.ones(l.T),segments=[[100,110],[300,310]])
    hits=sustained_outcomes(ee,np.full(ee.ts.V,.5))[0]
    assert hits[60][1].tolist()==[True,False]


def test_time_fraction_threshold_equals_bruteforce_and_respects_ties():
    ee=event_evaluator(lambda v,l:np.sin(l.t_sec/20)+1)
    target=.05
    threshold,cost=time_fraction_threshold(ee,target)
    grid=np.r_[np.inf,np.unique(ee.ts.score)[::-1]]
    feasible=[]
    for value in grid:
        _,active,total=sustained_outcomes(ee,np.full(ee.ts.V,value))
        fraction=active.sum()/total.sum()
        if fraction<=target+1e-12:feasible.append(value)
    assert threshold==min(feasible)
    assert cost<=target+1e-12


def test_sustained_null_bounds_a_start_of_clip_position_policy():
    ee=event_evaluator(lambda v,l:np.ones(l.T))
    curve=sustained_position_null(ee,max_thresholds=3)
    out=sustained_summary(ee,np.full(ee.ts.V,.5),np.zeros((0,ee.ts.V)))
    for lead in (60,90,120):
        h=np.array([[x,y] for x,y,_ in curve[str(lead)]])
        assert np.interp(out['alarm_time_fraction_clean']['value'],h[:,0],h[:,1])>=out[f'success_sus{lead}']['value']-1e-6


def test_window_max_uses_same_observed_time_and_length_in_clean_video():
    ee=event_evaluator(lambda v,l:np.where(l.t_sec==20,.9 if v=='v' else .2,0),onset=70,duration=150,clean_duration=50)
    result=window_max_auc(ee,group_weights(ee.ts,20,0))
    assert result['value']==pytest.approx(1)
    assert result['events_matched']==1
    assert result['windows'][0]['end']==40
    assert result['windows'][0]['matched_clean_windows']==1


def test_window_max_drops_clean_windows_with_excluded_ticks():
    ee=event_evaluator(lambda v,l:np.zeros(l.T),onset=70)
    i=ee.ts.vids.index('n');ee.ts.masked[ee.ts.offsets[i]+19]=True
    result=window_max_auc(ee,np.zeros((0,ee.ts.V)))
    assert result['value'] is None and result['events_unmatched']==1


def test_person_counts_missing_invalid_and_future_samples_are_not_zero(tmp_path):
    ee=event_evaluator(lambda v,l:np.zeros(l.T))
    for vid in ee.ts.vids:
        path=tmp_path/(vid+'.npz')
        np.savez(path,t_sec=np.array([1.,2.,4.]),feat=np.array([[0],[2],[6]]),valid=[True,False,True],
                 meta=json.dumps({'columns':['n_persons_last']}))
    counts,coverage=person_counts(ee.ts,tmp_path)
    i=ee.ts.vids.index('v');a=ee.ts.offsets[i]
    assert counts[a]==0 and np.isnan(counts[a+1]) and np.isnan(counts[a+2]) and counts[a+3]==6
    assert coverage['valid_ticks']==4


def test_person_strata_remove_count_only_separation():
    ee=event_evaluator(lambda v,l:np.ones(l.T) if v=='v' else (l.t_sec%2).astype(float))
    ts=ee.ts
    people=np.where(ts.score>0,2,0)
    source=np.array([f'ucf:{n}' for n in people])
    W=group_weights(ts,40,0)
    assert band_stat(ts,W)['value']>.7
    assert band_stat(ts,W,strata=source)['value']==pytest.approx(.5)


def test_opt_in_preserves_all_legacy_results_and_default_config():
    b=make_bench(standard_specs(4,4))
    scores=score_table(b,lambda l:((l.tte>30)&(l.tte<=120)).astype(float))
    base=evaluate(scores,b,quick_cfg(n_boot=12,position_null=False))
    extended=evaluate(scores,b,quick_cfg(n_boot=12,position_null=False,phase0_metrics=True))
    assert 'phase0_metrics' not in base['config'] and 'person_counts_root' not in base['config']
    assert 'phase0' not in base['seeds']['0']
    phase0=extended['seeds']['0'].pop('phase0')
    assert phase0['band_auroc']['value']==pytest.approx(1)
    assert phase0['band_long_auroc']['value']==pytest.approx(1)
    assert phase0['band_long_auroc']['n_neg_videos']>0
    for result in [base,extended]:result['seeds']['0'].pop('_seconds')
    assert va30._clean_nans(base['seeds'])==va30._clean_nans(extended['seeds'])
    assert base['summary_over_seeds']==extended['summary_over_seeds']


def test_sustained_threshold_uncertainty_and_capped_exposure():
    b=make_bench(standard_specs(4,4,clean_dur=600)).with_horizons({f"n{i}":120 for i in range(4)})
    scores=score_table(b,lambda l:((l.t_sec<80)&(l.t_sec>10)).astype(float))
    result=evaluate(scores,b,quick_cfg(n_boot=12,threshold_bootstrap=12,position_null=False,phase0_metrics=True))
    phase=result['seeds']['0']['phase0']
    for op in phase['sustained_warning']['operating_points'].values():
        assert op['threshold_aware_ci']['n_boot']==12
        assert op['alarm_time_fraction_clean']['value']<=1
    assert result['counts']['clean_negative_hours']==pytest.approx(480/3600,abs=.01)


def test_phase0_report_writes_opt_in_metrics(tmp_path):
    b=make_bench(standard_specs(4,4))
    scores=score_table(b,lambda l:np.zeros(l.T))
    result=evaluate(scores,b,quick_cfg(n_boot=12,position_null=False,phase0_metrics=True))
    va30.write_report(result,str(tmp_path))
    text=(tmp_path/'report.md').read_text()
    assert 'Band source x time matched AUROC' in text
    assert 'Sustained@60' in text


def test_warning_start_is_reported_relative_to_conflict_when_available():
    ee=event_evaluator(lambda v,l:np.ones(l.T) if v=='v' else np.zeros(l.T))
    ee.ts.bench.events['v']['t_conflict']=1
    out=sustained_summary(ee,np.full(ee.ts.V,.5),np.zeros((0,ee.ts.V)))
    assert out['conflict_timing60'][0]['alarm_start_minus_conflict_sec']==1


def test_frozen_pose_manifest_excludes_new_files_and_detects_mutation(tmp_path):
    ee=event_evaluator(lambda v,l:np.zeros(l.T))
    for vid in ee.ts.vids:
        np.savez(tmp_path/(vid+'.npz'),t_sec=[1.],feat=[[2]],valid=[True],meta=json.dumps({'columns':['n_persons_last']}))
    path=tmp_path/'snapshot.json'
    path.write_text(json.dumps({'root':str(tmp_path),'files':{'v':va30.file_sha256(tmp_path/'v.npz')}}))
    _,coverage=person_counts(ee.ts,path)
    assert coverage['pose_files']==1
    np.savez(tmp_path/'v.npz',t_sec=[1.],feat=[[3]],valid=[True],meta=json.dumps({'columns':['n_persons_last']}))
    with pytest.raises(ValueError,match='changed after snapshot'):person_counts(ee.ts,path)


def test_full_phase0_position_null_is_numeric():
    b=make_bench(standard_specs(4,4))
    result=evaluate(score_table(b,lambda l:np.ones(l.T)),b,
                    quick_cfg(n_boot=12,position_null=True,phase0_metrics=True))
    ops=result['seeds']['0']['phase0']['sustained_warning']['operating_points']
    assert all(isinstance(op['position_null']['60'],float) for op in ops.values())


def test_world_movie_breakdowns_hold_pooled_thresholds_and_source_windows():
    specs=standard_specs(4,4)
    for i,r in enumerate(specs):r['source_kind']='xd_movie' if i%2 else 'ucf'
    b=make_bench(specs)
    result=evaluate(score_table(b,lambda l:((l.tte>30)&(l.tte<=120)).astype(float)),b,
                    quick_cfg(n_boot=12,position_null=False,phase0_metrics=True))
    phase=result['seeds']['0']['phase0']
    for source in ['real_world','movie']:
        block=phase['breakdowns'][source]
        assert block['window_max_auroc']['events_matched']==2
        assert block['event_operating_points']['fa2']['eligible30']['success30']['n']==2
