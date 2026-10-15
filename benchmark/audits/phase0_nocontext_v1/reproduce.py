"""Prepare immutable-label audit proposals and sampled no-context contact sheets."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from anticipation.data.catalog import file_sha256
from anticipation.data.labels import labels_for_video
from anticipation.eval.aggression_reference import load_records
from anticipation.eval.va30 import Benchmark
from tools.contact_sheet import render_sheets

OUT=Path('benchmark/audits/phase0_nocontext_v1')
STRIKE=re.compile(r'\b(hit(?:s|ting)?|punch\w*|kick\w*|beat(?:s|ing|en)?|slap\w*|stab\w*|fight(?:s|ing)?|strik\w*|attack\w*|chok\w*|strangl\w*|drag\w*|wrestl\w*|push\w* down|shoot\w*|shot)\b',re.I)
WEAPON=re.compile(r'\b(gun\w*|pistol\w*|rifle\w*|knife|knives|weapon\w*|threaten\w*)\b',re.I)


def prepare(bench,records,coverage):
    rows=[]
    comparison=Counter()
    for vid,rr in records.items():
        if bench.videos.loc[vid,'role']!='violent':continue
        times=np.array([r['time_sec'] for r in rr])
        labels=labels_for_video(vid,bench.videos.loc[vid],bench.events,t_sec=times)
        scores=np.array([r['violent_score'] if r['violent_score'] is not None else -1 for r in rr])
        aggression=np.array([r['aggression'] for r in rr])
        selected=(~labels.masked)&labels.pre_violence&(labels.tte>30)&(labels.tte<=120)&(scores>=.8)
        if selected.any():
            keep=np.flatnonzero(selected)
            peak=int(keep[np.argmax(scores[keep])]);event=bench.event_rows(vid)[0]
            rows.append({'video_id':vid,'event_index':0,'onset_sec':event['onset'],'end_sec':event['end'],
                         'eligible30':event['eligible30'],'max_lead':event['max_lead'],
                         'source_kind':bench.videos.loc[vid,'source_kind'],'onset_source':bench.videos.loc[vid,'onset_source'],
                         'path':bench.videos.loc[vid,'path'],'peak_time_sec':float(times[peak]),
                         'peak_violent_score':float(scores[peak]),'aggression_at_peak':float(aggression[peak]),
                         'high_frame_times_sec':times[selected].tolist(),
                         'teacher_description_at_peak':{k:rr[peak].get(k) for k in ['actions','body_language','threat_indicators','weapons']},
                         'classification':'PENDING_VISUAL_REVIEW','status':'PROPOSAL_NOT_APPLIED'})
        for event in bench.event_rows(vid):
            if not event['eligible30']:continue
            window=(times>=event['onset']-120)&(times<=event['onset']-30)&(aggression>=.8)
            if window.any():
                comparison['aggression_all_eligible_events']+=1
                if (window&~labels.masked&(times>=event['clear_start'])).any():
                    comparison['aggression_events_with_clear_window_frame']+=1
                else:comparison['aggression_events_earlier_annotated_or_excluded']+=1
    rows.sort(key=lambda r:r['video_id'])
    for rank,r in enumerate(rows,1):r['audit_id']=f'V{rank:03d}'
    meta={'status':'PROPOSAL_NOT_APPLIED','selection':'No-context violent.score>=0.8 at raw_frame_idx/catalogue_fps; unmasked pre-violence records with 30<tte<=120; one first event per video. Eligibility reported rather than silently filtering record-time cases.',
          'documented_expected_events':53,'actual_events':len(rows),'reference_input':coverage,
          'catalog_sha256':file_sha256('benchmark/videos.csv'),'events_sha256':file_sha256('benchmark/events.json'),
          'comparison_to_existing_aggression_audit':dict(comparison),'events':rows}
    path=OUT/'early_frames_proposal.json'
    if path.exists():
        previous=json.loads(path.read_text())
        annotations={(r['video_id'],r['event_index']):r for r in previous['events']}
        for r in rows:
            old=annotations.get((r['video_id'],r['event_index']),{})
            for key in ['classification','visual_evidence','confidence','mask_start_proposal_sec','sheet','sheet_sha256','sheet_times_sec']:
                if key in old:r[key]=old[key]
    path.write_text(json.dumps(meta,indent=2)+'\n')
    print('Visual candidates:',len(rows),dict(comparison),flush=True)


def uca_proposals(bench):
    captions={};hashes={}
    for path in sorted(Path('data/annotations/uca/json').glob('*.json')):
        data=json.loads(path.read_text());hashes[str(path)]=file_sha256(path)
        for stem,entry in data.items():
            captions.setdefault(stem,[]).extend(zip(entry['timestamps'],entry['sentences']))
    candidates=[]
    for vid in sorted(bench.eval_ids()):
        if not vid.startswith('ucf/') or bench.videos.loc[vid,'role']!='violent':continue
        events=bench.event_rows(vid)
        if not events:continue
        event=events[0];onset=event['onset'];strike=[];weapon=[]
        for (start,end),text in captions.get(vid.split('/',1)[1],[]):
            if end>onset-10 or start<event['clear_start']:continue
            row={'start_sec':float(start),'end_sec':float(end),'text':text}
            if STRIKE.search(text):strike.append(row)
            elif WEAPON.search(text):weapon.append(row)
        if strike or weapon:
            candidates.append({'video_id':vid,'event_index':0,'onset_sec':onset,'onset_source':bench.videos.loc[vid,'onset_source'],
                 'eligible30':event['eligible30'],'strike_captions':strike,'weapon_only_captions':weapon,
                 'classification':'PENDING_CAPTION_REVIEW','status':'PROPOSAL_NOT_APPLIED'})
    (OUT/'uca_candidates.json').write_text(json.dumps({'status':'PROPOSAL_NOT_APPLIED','unit':'First violent segment per UCF video; caption end<=onset-10; start>=clear_start; full lead-in, not only120s.',
            'caption_files':hashes,'strike_regex':STRIKE.pattern,'weapon_regex':WEAPON.pattern,'candidates':candidates},indent=2)+'\n')
    print('UCA candidate videos:',len(candidates),'strike:',sum(bool(r['strike_captions']) for r in candidates),flush=True)


def render(bench):
    path=OUT/'early_frames_proposal.json';data=json.loads(path.read_text())
    sheets=OUT/'sheets';tiles=OUT/'tiles';sheets.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    for record in data['events']:
        target=sheets/(record['audit_id']+'.jpg')
        if target.exists():continue
        peak,onset=record['peak_time_sec'],record['onset_sec']
        times=np.clip([peak-3,peak-1,peak,peak+1,peak+3,onset-2,onset,onset+2,onset+5],0,max(0.,float(bench.videos.loc[record['video_id'],'duration_sec'])-1.))
        canvas=Image.new('RGB',(768,3*178+65),'white');draw=ImageDraw.Draw(canvas)
        draw.text((4,2),f"{record['audit_id']} {record['video_id'][:100]}",fill='black',font=font)
        draw.text((4,19),f"onset={onset:.3f}s peak={peak:.3f}s violent={record['peak_violent_score']} aggression={record['aggression_at_peak']} source={record['onset_source']}",fill='black',font=font)
        for j,t in enumerate(times):
            sheet=render_sheets(record['path'],str(tiles),start=float(t),end=float(t)+.005,step=1,cols=1,rows=1,tile_w=256,prefix=f"{record['audit_id']}_{j}")[0]
            im=Image.open(sheet).convert('RGB');im.thumbnail((256,160))
            x,y=(j%3)*256,(j//3)*178+42
            canvas.paste(im,(x,y));draw.text((x+2,y+160),f"t={t:.2f}s; lead={onset-t:.2f}s",fill='black',font=font)
            Path(sheet).unlink()
        for quality in (82,72,62,50):
            canvas.save(target,quality=quality,optimize=True)
            if target.stat().st_size<=300000:break
        assert target.stat().st_size<=300000
        record['sheet']=str(target)
        record['sheet_sha256']=file_sha256(target)
        record['sheet_times_sec']=times.tolist()
        print(record['audit_id'],record['video_id'],flush=True)
    for record in data['events']:
        target=sheets/(record['audit_id']+'.jpg')
        record.setdefault('sheet',str(target));record.setdefault('sheet_sha256',file_sha256(target))
    if tiles.exists():tiles.rmdir()
    path.write_text(json.dumps(data,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--render_only',action='store_true')
    ap.add_argument('--skip_render',action='store_true')
    args=ap.parse_args()
    bench=Benchmark.load('benchmark/folds_v2.json')
    if not args.render_only:
        records,coverage=load_records(bench,'data/processed/vlm_labels_violence')
        prepare(bench,records,coverage);uca_proposals(bench)
    if not args.skip_render:render(bench)


if __name__=='__main__':main()
