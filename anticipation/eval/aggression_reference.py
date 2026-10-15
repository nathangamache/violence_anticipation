"""Export and evaluate the frozen no-context Qwen aggression reference and specificity controls."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from anticipation.data.catalog import file_sha256
from anticipation.data.labels import labels_for_video
from anticipation.eval import va30
from anticipation.eval.baselines import sample_and_hold, teacher_score
from anticipation.eval.learned_baselines import SOURCES, catalog_mapping
from anticipation.eval.phase0 import group_weights
from anticipation.eval.va30 import Benchmark, EvalConfig, PairAUC, TickSet, time_match_weights, TIME_MATCH_EDGES

H1_PEOPLE = {"stealing", "shoplifting", "burglary", "arson", "vandalism"}


def load_records(bench, root):
    mapping = catalog_mapping(bench.videos)
    paths = list(Path(root).glob("*/*.jsonl"))
    paths.sort(key=lambda p:(p.stem != str(mapping.get((SOURCES.get(p.parent.name),p.stem),"/")).split("/",1)[1],str(p)))
    records, manifest, skipped = {}, [], {}
    for path in paths:
        vid = mapping.get((SOURCES.get(path.parent.name),path.stem))
        if vid is None or vid in records:
            continue
        fps = float(bench.videos.loc[vid,"fps"])
        if not np.isfinite(fps) or fps<=0:
            raise ValueError(f"Invalid catalogue FPS: {vid}")
        seen, rows = set(), []
        for line in path.read_text().splitlines():
            try:
                record = json.loads(line)
                if "error" in record or record.get("raw_frame_idx") is None or record.get("t") is None:
                    continue
                if record.get("context_mode") not in (None,"none"):
                    raise ValueError(f"Context-bearing record in no-context reference: {path}")
                key = int(record["t"])
                if key in seen:
                    continue
                aggression = float(record["aggression_level"])
                seconds = float(record["raw_frame_idx"])/fps
                if not np.isfinite(aggression) or not 0<=aggression<=1:
                    raise ValueError(f"Invalid aggression level: {path}")
                if not np.isfinite(seconds) or not 0<=seconds<=float(bench.videos.loc[vid,"duration_sec"]):
                    continue
                seen.add(key)
                rows.append({**record,"time_sec":seconds,"aggression":aggression,"violent_score":teacher_score(record)})
            except (json.JSONDecodeError,KeyError,TypeError):
                skipped["malformed_or_missing_field"] = skipped.get("malformed_or_missing_field",0)+1
        if rows:
            rows.sort(key=lambda r:r["time_sec"])
            records[vid] = rows
            manifest.append(f"{vid} {file_sha256(path)}")
    return records,{"files":len(manifest),"records":sum(map(len,records.values())),"skipped":skipped,
                    "input_sha256":hashlib.sha256("\n".join(manifest).encode()).hexdigest()}


def export_scores(bench, records, broadcast=False, video_ids=None):
    ids = sorted(set(video_ids if video_ids is not None else bench.eval_ids()) & records.keys())
    parts = []
    for vid in ids:
        rows = records[vid]
        times = np.array([r["time_sec"] for r in rows])
        values = np.array([r["aggression"] for r in rows])
        if broadcast:
            times,values = times[:1],values[:1]
        ticks = bench.labels(vid).t_sec
        parts.append(pd.DataFrame({"video_id":vid,"t_sec":ticks,"score":sample_and_hold(times,values,ticks),
                                   "fold":-1,"seed":0}))
    if not parts:
        raise ValueError("Reference has no covered videos")
    return pd.concat(parts,ignore_index=True)


def specificity(bench, records, n_boot=1000, broadcast=False):
    h1 = [v for v in records if bench.videos.loc[v,"role"]=="other_anomaly"
          and bench.videos.loc[v,"first_anomaly_sec"]>=30]
    ids = sorted(set(h1) | (set(bench.eval_ids()) & records.keys()))
    scores = {}
    for v in ids:
        rows=records[v];times=np.array([r['time_sec'] for r in rows]);vals=np.array([r['aggression'] for r in rows])
        if broadcast:times,vals=times[:1],vals[:1]
        scores[v]=sample_and_hold(times,vals,bench.labels(v).t_sec)
    ts=TickSet(bench,ids,scores)
    W=group_weights(ts,n_boot,0)
    row=bench.videos.loc[ids]
    h1p_v=np.array([v in h1 and row.loc[v,'source']=='ucf' and bool(set(str(row.loc[v,'classes']).split(';'))&H1_PEOPLE) for v in ids])
    anomaly=row.first_anomaly_sec.astype(float).values[ts.vidx]-ts.t
    h1_v=np.array([v in h1 for v in ids])
    h1_band=ts.video_mask(h1_v) & (anomaly>30)&(anomaly<=120)&~ts.masked
    ucf_band=(ts.source_v[ts.vidx]=='ucf') & ~ts.masked & ts.pre & (ts.tte>30)&(ts.tte<=120)
    clean=ts.clean & ~ts.masked & (ts.remaining>=60)
    source=ts.source_v[ts.vidx]
    def auc(pos,neg):
        weights,matched=time_match_weights(ts.t,pos,neg,TIME_MATCH_EDGES,source=source)
        obj=PairAUC(ts.score,ts.vidx,matched,neg,ts.V,neg_w=weights)
        return obj.stat(W),obj.boot(W)
    people= h1_band & ts.video_mask(h1p_v)
    m3p,m3p_boot=auc(ucf_band,people)
    m1,m1_boot=auc(ucf_band,clean & (source=='ucf'))
    ratio=None
    if m1['value'] is not None and m1['value']>.5 and m3p['value'] is not None:
        ratio=(m3p['value']-.5)/(m1['value']-.5)
    ratio_boot=np.divide(m3p_boot-.5,m1_boot-.5,out=np.full(n_boot,np.nan),where=m1_boot>.5)
    out={'status':'diagnostic only; H1 never used to fit or select thresholds','ci_cluster':'split_group',
         'h1_videos':int(h1_v.sum()),'h1p_videos':int(h1p_v.sum()),'M3p':m3p,'M1_UCF':m1,
         'specificity_ratio':va30.ci(ratio_boot,ratio),'classes':{},
         'all_h1_vs_clean':auc(h1_band,clean)[0],
         'ucf_violent_vs_all_h1':auc(ucf_band,h1_band & (source=='ucf'))[0]}
    for category in sorted(H1_PEOPLE):
        category_v=np.array([v in h1 and category in str(row.loc[v,'classes']).split(';') for v in ids])
        out['classes'][category]=auc(ucf_band,h1_band & ts.video_mask(category_v))[0]
    return out


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='data/processed/vlm_labels_violence')
    ap.add_argument('--out_root',default='benchmark/results')
    ap.add_argument('--n_boot',type=int,default=1000)
    ap.add_argument('--threshold_bootstrap',type=int,default=200)
    ap.add_argument('--pose_root',default='data/features/pose_interact')
    ap.add_argument('--export_only',action='store_true')
    args=ap.parse_args(argv)
    va30.N_THREADS=4
    bench=Benchmark.load('benchmark/folds_v2.json')
    records,coverage=load_records(bench,args.root)
    covered=sorted(set(bench.eval_ids()) & records.keys())
    print(json.dumps({**coverage,'evaluated_videos':len(covered)}),flush=True)
    manifest_path=Path(args.out_root)/'b_aggression_nocontext'/'pose_manifest.json'
    manifest_path.parent.mkdir(parents=True,exist_ok=True)
    pose_root=Path(args.pose_root)
    pose_files={v:file_sha256(pose_root/(v+'.npz')) for v in covered if (pose_root/(v+'.npz')).exists()}
    manifest_path.write_text(json.dumps({'root':str(pose_root),'files':pose_files},indent=2)+'\n')
    for broadcast,name in [(False,'b_aggression_nocontext'),(True,'b_aggression_nocontext_first_frame')]:
        out=Path(args.out_root)/name;out.mkdir(parents=True,exist_ok=True)
        scores=export_scores(bench,records,broadcast)
        scores.to_parquet(out/'scores.parquet',index=False,compression='zstd')
        meta={'model':'Qwen2.5-VL-7B violence_embedding, context none','field':'aggression_level','input_root':args.root,
              'coverage':coverage,'covered_video_ids':covered,'timestamps':'raw_frame_idx / benchmark/videos.csv fps',
              'broadcast':broadcast,'fill_before_first_record':0.,'folds_content_sha256':bench.folds.sha256,
              'implementation_sha256':file_sha256(__file__),'score_sha256':file_sha256(out/'scores.parquet')}
        (out/'run_config.json').write_text(json.dumps(meta,indent=2)+'\n')
        control=specificity(bench,records,args.n_boot,broadcast)
        (out/'h1_specificity.json').write_text(json.dumps(va30._clean_nans(control),indent=1)+'\n')
        if not args.export_only:
            cfg=EvalConfig(n_boot=args.n_boot,threshold_bootstrap=args.threshold_bootstrap,cluster='split_group',
                           restrict_to_scored=True,allow_missing=True,phase0_metrics=True,person_counts_root=str(manifest_path))
            result=va30.evaluate(scores,bench,cfg,name=name,extra_meta={'reference':meta,'scores_sha256':meta['score_sha256']})
            va30.save_results(result,str(out))
            print(name,result['seeds']['0']['phase0']['band_auroc'],flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
