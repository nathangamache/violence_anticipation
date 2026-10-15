from lib import *
import json
from scipy.stats import spearmanr
from anticipation.features.io import load_features
pr,ss=load_ss()
allowed=set()
for f in (0,2,3,4): allowed|=set(ss.test_ids(f,kinds=('violent','clean','h1')))
T2=pd.read_parquet('T_siglip2_prompts_v2.parquet'); T1=pd.read_parquet('T_siglip2.parquet') if os.path.exists('T_siglip2.parquet') else None
assert set(T2.video_id)<=allowed
cols=None; rows=[]
for vid,g in T2.groupby('video_id',sort=False):
    t,f,v,meta=load_features('siglip2_prompts_v2',vid)
    cols=meta['columns']; idx=np.searchsorted(t,g.t.values)
    assert np.allclose(t[idx],g.t.values)
    d=pd.DataFrame(f[idx],columns=cols); rows.append(d)
F=pd.concat(rows,ignore_index=True); assert len(F)==len(T2)
for c in cols: T2['f_'+c]=F[c].values
if T1 is not None:
    s2=T1.set_index(['video_id','t']).score
    T2['score_siglip2']=s2.reindex(pd.MultiIndex.from_frame(T2[['video_id','t']])).values
T2.to_parquet('T_v2_feats.parquet')
neg=((T2.kind=='clean')&~T2.masked&(T2.rem>=60)).values
ctx=['ctx_cctv','ctx_movie','ctx_news','ctx_captions','ctx_phone']
out={'corr':{}, 'ctx_band_auc':{}, 'subset':{}, 'n_neg':int(neg.sum())}
for sc in ['score','score_siglip2']:
    out['corr'][sc]={}
    for c in ctx:
        x=T2['f_'+c].values
        r=spearmanr(T2[sc].values[neg],x[neg])[0]
        per={s:float(spearmanr(T2[sc].values[neg&(T2.src==s).values],x[neg&(T2.src==s).values])[0]) for s in T2.src.unique() if (neg&(T2.src==s).values).sum()>1000}
        out['corr'][sc][c]={'all':float(r),**per}
for c in cols:
    out['ctx_band_auc'][c]=band_auc(T2,'f_'+c)[0]
# subset AUROCs
def sub(mask,label):
    r={}
    T=T2[mask]
    for sc in ['score','score_siglip2']:
        a,np_,nn,*_=band_auc(T.reset_index(drop=True),sc)
        r[sc]=(a,np_)
    out['subset'][label]=r
sub(np.ones(len(T2),bool),'all')
q75={c:np.percentile(T2['f_'+c].values[neg],75) for c in ctx}
q50={c:np.percentile(T2['f_'+c].values[neg],50) for c in ctx}
for qn,q in (('q75',q75),('q50',q50)):
    low=np.ones(len(T2),bool)
    for c in ('ctx_movie','ctx_news','ctx_captions','ctx_phone','ctx_cctv'): low&=T2['f_'+c].values<=q[c]
    sub(low,f'all_ctx_low_{qn}')
    for c in ctx: sub(T2['f_'+c].values<=q[c],f'{c}_low_{qn}')
pv=np.percentile(T2['f_phase_violence'].values[neg],[50,80,90])
for p,v in zip((50,80,90),pv): sub(T2['f_phase_violence'].values<=v,f'phase_violence_below_clean_p{p}')
for s in T2.src.unique(): sub((T2.src==s).values,f'src_{s}')
sub((T2.src!='xd_movie').values,'S_real_like_nonmovie')
sub(((T2.src!='xd_movie')&(T2.src!='xd_youtube')).values,'ucf_ubi_only')
json.dump(out,open('<tmp>/screen_audit/quant.json','w'),indent=1)
print(json.dumps(out,indent=1))
