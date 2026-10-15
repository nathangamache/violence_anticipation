import json

import numpy as np
import pytest

from anticipation.eval.aggression_reference import export_scores, load_records, specificity
from tests.test_va30 import make_bench


def fixture():
    rows=[]
    for vid,role,category in [('ucf/v','violent','fighting'),('ucf/n','clean_negative','normal'),
                              ('ucf/shop','other_anomaly','shoplifting'),('ucf/car','other_anomaly','road_accident')]:
        rows.append(dict(vid=vid,duration=220,role=role,fold=0,segments=[[150,160]] if role=='violent' else []))
    bench=make_bench(rows)
    bench.videos['source']='ucf';bench.videos['fps']=10.
    bench.videos['path']=['/tmp/'+v.split('/')[1]+'.mp4' for v in bench.videos.index]
    bench.videos['aliases']='';bench.videos['classes']=['fighting','normal','shoplifting','road_accident']
    bench.videos['first_anomaly_sec']=[150,np.nan,150,150]
    return bench


def test_no_context_records_use_catalogue_clock_and_first_duplicate(tmp_path):
    b=fixture();root=tmp_path/'ucf_crime';root.mkdir()
    records=[dict(t=0,raw_frame_idx=25,aggression_level=.2,context_mode='none'),
             dict(t=0,raw_frame_idx=25,aggression_level=.8,context_mode='none'),
             dict(t=5,raw_frame_idx=105,aggression_level=.9,context_mode='none')]
    (root/'v.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
    loaded,coverage=load_records(b,tmp_path)
    assert [r['time_sec'] for r in loaded['ucf/v']]==[2.5,10.5]
    assert loaded['ucf/v'][0]['aggression']==.2
    df=export_scores(b,loaded)
    assert df[df.t_sec==2].score.item()==0
    assert df[df.t_sec==3].score.item()==.2
    assert df[df.t_sec==11].score.item()==.9
    assert coverage['records']==2


def test_first_frame_broadcast_waits_for_observation_and_ignores_later_rows():
    b=fixture();records={'ucf/v':[dict(time_sec=2.5,aggression=.2),dict(time_sec=10.5,aggression=.9)]}
    df=export_scores(b,records,broadcast=True)
    assert df[df.t_sec==2].score.item()==0
    assert df[df.t_sec>=3].score.eq(.2).all()


def test_context_bearing_record_is_rejected(tmp_path):
    b=fixture();root=tmp_path/'ucf_crime';root.mkdir()
    (root/'v.jsonl').write_text(json.dumps(dict(t=0,raw_frame_idx=0,aggression_level=.5,context_mode='both')))
    with pytest.raises(ValueError,match='Context-bearing'):load_records(b,tmp_path)


def test_people_focused_h1_does_not_reward_road_vs_people_separation():
    b=fixture()
    records={v:[dict(time_sec=0,aggression=score)] for v,score in
             [('ucf/v',.9),('ucf/n',0),('ucf/shop',.9),('ucf/car',0)]}
    out=specificity(b,records,n_boot=20)
    assert out['h1p_videos']==1 and out['h1_videos']==2
    assert out['M1_UCF']['value']==pytest.approx(1)
    assert out['M3p']['value']==pytest.approx(.5)
    assert out['specificity_ratio']['value']==pytest.approx(0)
    assert out['ucf_violent_vs_all_h1']['value']>.5
