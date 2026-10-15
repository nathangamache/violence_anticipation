"""Apply recorded visual and caption reviews to proposal artifacts without editing labels."""

import csv,json,hashlib
from pathlib import Path
from collections import Counter
root=Path('benchmark/audits/phase0_nocontext_v1')
reviews=json.loads((root/'visual_reviews.json').read_text())['events']
p=root/'early_frames_proposal.json';d=json.loads(p.read_text())
for row in d['events']:
 row.update(reviews[row['audit_id']]);row['review_basis']='Sampled contact sheet; UCA corroboration explicitly stated where used. No labels changed.'
d['summary']={'all_record_time_candidates':dict(Counter(r['classification'] for r in d['events'])),
 'onset_at_least_32s_candidates':dict(Counter(r['classification'] for r in d['events'] if r['onset_sec']>=32)),
 'va30_eligible30_candidates':dict(Counter(r['classification'] for r in d['events'] if r['eligible30'])),
 'category_b_confidence':dict(Counter(r['confidence'] for r in d['events'] if r['classification']=='b')),
 'by_source':{s:dict(Counter(r['classification'] for r in d['events'] if r['source_kind']==s)) for s in sorted({r['source_kind'] for r in d['events']})},
 'clock_or_clear_history_ineligible':[r['audit_id'] for r in d['events'] if not r['eligible30']],
 'suspected_fiction_in_youtube_proxy':[r['audit_id'] for r in d['events'] if r.get('suspected_fiction_under_youtube_source')]}
d['count_reconciliation']='54 raw record-time cases reproduce the score/window rule. Of these, 53 have onset>=32s; Robbery105 has onset 31.333s and only 29.333s maximum alarm lead. Two further mixed-anomaly cases fail clear-history eligibility. All 54 are retained and reviewed; 51 are VA30 eligible30. This is distinct from the existing aggression_level>=.8 all-event audit (51 candidates, 27 clear-window cases).'
d['interpretation_limits']='Model-selected, nonrepresentative audit; sparse stills cannot exclude unseen earlier violence or establish intention. Non-strike category a includes armed coercion and ordinary holding, not necessarily genuine escalation. Several caption-supported or low-confidence b calls require lead adjudication. Low-confidence masks must not be applied automatically.'
p.write_text(json.dumps(d,indent=2)+'\n')
fields=['audit_id','video_id','event_index','onset_sec','peak_time_sec','peak_violent_score','aggression_at_peak','eligible30','source_kind','onset_source','classification','confidence','mask_start_proposal_sec','visual_evidence','sheet','status']
with (root/'early_frames_proposal.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(d['events'])
uca=json.loads((root/'uca_candidates.json').read_text());rejected={'ucf/Abuse018_x264':'Kicking a bag/object, not a person.','ucf/Arrest006_x264':'A car hits a flagpole; not interpersonal violence.','ucf/Assault041_x264':'Kicked out of the shop is an idiom, not evidence of a physical kick.'}
restraint={'ucf/Arrest001_x264':'Subdued and dragged describes restraint, not an explicit strike; retain only as an adjudication/sensitivity case.'}
primary=[];weapon=[];rejected_rows=[];uncertain=[]
for row in uca['candidates']:
 vid=row['video_id'];captions=row['strike_captions']
 if captions:
  item={'video_id':vid,'event_index':0,'start_sec':min(c['start_sec'] for c in captions),'end_sec':row['onset_sec'],
        'onset_source':row['onset_source'],'eligible30':row['eligible30'],'evidence_captions':captions,'status':'PROPOSAL_NOT_APPLIED'}
  if vid in rejected:item['review']=rejected[vid];rejected_rows.append(item)
  elif vid in restraint:item['review']=restraint[vid];uncertain.append(item)
  else:
   item['reason']='uca_explicit_physical_strike_or_discharge';item['review']='Caption describes physical force against a person or a firearm discharge; not visually adjudicated as an exact onset.';primary.append(item)
 if row['weapon_only_captions']:
  captions=row['weapon_only_captions']
  weapon.append({'video_id':vid,'event_index':0,'start_sec':min(c['start_sec'] for c in captions),'end_sec':row['onset_sec'],
                 'reason':'uca_weapon_only','onset_source':row['onset_source'],'eligible30':row['eligible30'],'evidence_captions':captions,
                 'default_action':'KEEP_IN_PRIMARY; optional exclusion only in separately labelled weapon sensitivity',
                 'status':'PROPOSAL_NOT_APPLIED'})
proposal={'status':'PROPOSAL_NOT_APPLIED','version':1,'catalog_sha256':d['catalog_sha256'],'events_sha256':d['events_sha256'],
 'rules':uca['unit'],'caption_files':uca['caption_files'],'semantics':'Proposed soft_mask_spans only, from earliest vetted caption start up to annotated first onset; no replacement onsets, no catalog writes, and no automatic changes to clear_start.',
 'primary_strike_proposals':primary,'weapon_only_sensitivity_flags':weapon,'restraint_only_adjudication':uncertain,
 'rejected_non_interpersonal_or_idiom_matches':rejected_rows,
 'counts':{'raw_strike_candidates':23,'reviewed_physical_strike_or_shot':len(primary),'weapon_only_flags':len(weapon),
           'restraint_only':len(uncertain),'rejected':len(rejected_rows),'eligible30_primary':sum(r['eligible30'] for r in primary)}}
(root/'uca_soft_masks_proposal.json').write_text(json.dumps(proposal,indent=2)+'\n')
visual_masks=[{'video_id':r['video_id'],'event_index':r['event_index'],'start_sec':r['mask_start_proposal_sec'],'end_sec':r['onset_sec'],
              'confidence':r['confidence'],'basis':r['visual_evidence'],'status':'PROPOSAL_NOT_APPLIED'} for r in d['events'] if r['classification']=='b']
(root/'visual_b_masks_proposal.json').write_text(json.dumps({'status':'PROPOSAL_NOT_APPLIED','masks':visual_masks},indent=2)+'\n')
for p in root.glob('*.json'):
 p.with_suffix(p.suffix+'.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n')
print(json.dumps(d['summary'],indent=2));print(proposal['counts'])
