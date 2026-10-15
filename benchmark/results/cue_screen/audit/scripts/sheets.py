from lib import *
from PIL import Image, ImageDraw, ImageFont
from anticipation.features.io import Video
import json
vids=ss=None
pr,ss=load_ss()
V=ss.bench.videos
FONT=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",12)
OUT=f'{R}/audit'
def pick(b):
    p=f'<tmp>/screen_audit/T_{b}.parquet'
    if os.path.exists(p): T=pd.read_parquet(p)
    else:
        s=pd.read_parquet(f'{R}/{b}/scores.parquet'); s=s[s.split=='test']; T=tick_table(ss,s); T.to_parquet(p)
    neg=T[(T.kind=='clean')&~T.masked&(T.rem>=60)]
    pos=T[(T.kind=='violent')&~T.masked&(T.pre)&T.first_video&(T.tte>30)&(T.tte<=120)]
    n=neg.sort_values('score',ascending=False).drop_duplicates('video_id').head(30)
    q=pos.sort_values('score',ascending=False).drop_duplicates('video_id').head(30)
    return T,n,q
def sheet(rows,name,offs=(-6,-3,0)):
    tiles=[]
    for r in rows.itertuples():
        path=V.loc[r.video_id,'path']
        try:
            v=Video(path,num_threads=2); ts=[max(0.0,r.t+o) for o in offs]
            fr=v.frames_at(ts)
        except Exception as e:
            fr=[np.zeros((108,192,3),np.uint8)]*len(offs)
        W=192; ims=[]
        for f in fr:
            im=Image.fromarray(f); h=int(round(W*im.height/im.width)); ims.append(im.resize((W,min(h,144))))
        row=Image.new('RGB',(W*len(offs),144+16),(0,0,0)); 
        for i,im in enumerate(ims): row.paste(im,(i*W,16))
        d=ImageDraw.Draw(row)
        lab=f'{r.video_id[:34]} t={r.t:.0f} tte={r.tte:.0f} s={r.score:.2f}'
        d.text((2,2),lab,fill=(255,255,0),font=FONT)
        tiles.append(row)
    per=10; paths=[]
    for k in range(0,len(tiles),per):
        ch=tiles[k:k+per]; w=ch[0].width
        S_=Image.new('RGB',(2*w,((len(ch)+1)//2)*160),(30,30,30))
        for i,t in enumerate(ch): S_.paste(t,((i%2)*w,(i//2)*160))
        p=f'{OUT}/{name}_{k//per+1}.jpg'; S_.save(p,quality=80); paths.append(p)
    return paths
if __name__=='__main__':
    allp={}
    for b in ['siglip2_prompts_v2','siglip2']:
        T,n,q=pick(b)
        n.to_csv(f'{OUT}/top_clean_{b}.csv',index=False); q.to_csv(f'{OUT}/top_posband_{b}.csv',index=False)
        allp[b]=sheet(n,f'clean_{b}')+sheet(q,f'posband_{b}')
    print(json.dumps(allp,indent=1))
