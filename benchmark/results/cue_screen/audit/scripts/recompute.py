from lib import *
import json
pr,ss=load_ss()
res={}
for b in ['siglip2_prompts_v2','pose_interact']:
    s=pd.read_parquet(f'{R}/{b}/scores.parquet'); s=s[s.split=='test']
    assert not (s.fold==1).any()
    T=tick_table(ss,s); T.to_parquet(f'<tmp>/screen_audit/T_{b}.parquet')
    m=json.load(open(f'{R}/{b}/metrics.json'))
    r={}
    for name,kw,key in [('m1',{},'m1'),('m1_30_60',dict(hi=60),'m1_30_60')]:
        a,np_,nn,*_=band_auc(T,'score',**kw); r[name]=(a,m[key]['value'],np_,m[key]['n_pos'])
    for mod,key in [('n','m1_n'),('time_source','m1_time_source'),('shuffled','m1_shuffled'),('n_plus_p1','m1_n_plus_p1')]:
        a=band_auc(T,mod)[0]; r[key]=(a,m[key]['value'])
    res[b]=r
    print(b,json.dumps(r,indent=0))
json.dump({b:{k:{'recomputed':v[0],'metrics_json':v[1]} for k,v in r.items()} for b,r in res.items()},open(f'{R}/audit/recompute_check.json','w'),indent=1)
