from sheets import *
T=pd.read_parquet('T_siglip2_prompts_v2.parquet')
ev=ss.events(); ramp=set(ev.loc[ev.s_ramp,'video_id'])
t2=T[T.t==2]
cl=t2[(t2.kind=='clean')]
th={q:np.percentile(cl.score,q) for q in (50,75,90)}
r=t2[t2.video_id.isin(ramp)].sort_values('score',ascending=False)
print(len(r), th, {q:int((r.score>v).sum()) for q,v in th.items()})
print(r.groupby('src').size())
r=r.assign(rank=range(1,len(r)+1))
r[['video_id','src','score']].to_csv(f'{OUT}/ramp_first_frame_scores_v2.csv',index=False)
top=r.head(30).copy()
top['tte']=top.tte
ps=sheet(top,'firstframe_v2',offs=(-1,0,4))
print(ps)
print(top[['video_id','src','score','tte']].to_string())
