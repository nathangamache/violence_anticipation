import sys, os, numpy as np, pandas as pd
sys.path.insert(0, '.')
os.environ.setdefault('OMP_NUM_THREADS','4')
from anticipation.screen import prereg, sets as S
from anticipation.eval.va30 import time_match_weights, TIME_MATCH_EDGES, SOURCE_KINDS
R='benchmark/results/cue_screen'
SRC={s:i for i,s in enumerate(SOURCE_KINDS)}

def load_ss():
    pr=prereg.load(prereg.PREREG_PATH)
    return pr, S.load_screen_set(pr)

def tick_table(ss, scores):
    """scores: test rows df with video_id,t_sec,score -> merged with labels"""
    ev=ss.events(); first=set(ev.loc[ev['first'],'video_id'])
    out=[]
    for vid,g in scores.groupby('video_id',sort=False):
        vt=ss.ticks(vid)
        n=min(len(g),vt.n)
        g=g.sort_values('t_sec').iloc[:n]
        d=pd.DataFrame({'video_id':vid,'t':vt.t[:n],'tte':vt.tte[:n],'pre':vt.pre[:n],'masked':vt.masked[:n],'rem':vt.remaining[:n],
            'kind':vt.kind,'src':ss.source_kind(vid),'first_video':vid in first})
        assert np.allclose(g['t_sec'].values,vt.t[:n])
        for c in g.columns:
            if c not in ('video_id','t_sec'): d[c]=g[c].values
        out.append(d)
    return pd.concat(out,ignore_index=True)

def wauc(s,pos,neg,w):
    s=np.asarray(s); sp=s[pos]; sn=s[neg]; wn=w[neg]
    o=np.argsort(sn); sn=sn[o]; wn=wn[o]; cw=np.cumsum(wn); tot=cw[-1]
    lo=np.searchsorted(sn,sp,'left'); hi=np.searchsorted(sn,sp,'right')
    cwz=np.r_[0,cw]
    below=cwz[lo]; ties=cwz[hi]-cwz[lo]
    return float(((below+0.5*ties)/tot).mean())

def band_auc(T,col,lo=30,hi=120,first=True,vmask=None,posmask=None):
    code=T['src'].map(SRC).values
    pos=((T.kind=='violent')&~T.masked&(T.tte>lo)&(T.tte<=hi)).values
    if first: pos&=(T.pre&T.first_video).values
    if vmask is not None: pos&=vmask
    neg=((T.kind=='clean')&~T.masked&(T.rem>=60)).values
    w,pos2=time_match_weights(T.t.values,pos,neg,TIME_MATCH_EDGES,source=code)
    return wauc(T[col].values,pos2,neg,w), int(pos2.sum()), int(neg.sum()), pos2, neg, w
