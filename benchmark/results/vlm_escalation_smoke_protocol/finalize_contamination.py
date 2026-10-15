"""Freeze the visually reviewed first-frame contamination cohort before any VLM probe outcome."""
import json
from pathlib import Path

import pandas as pd

from prepare import ordered, write_manifest
from protocol import ROOT, sha


def main():
    candidates = pd.read_csv(ROOT / 'contamination_candidates.csv', keep_default_na=False)
    audit = json.load(open(ROOT / 'contamination_reviews.json'))
    reviews = audit['reviews']
    if set(candidates.audit_id) != set(reviews):
        raise ValueError('Review every blind first frame before freezing the cohort')
    candidates['calm_verified'] = candidates.audit_id.map(lambda k: reviews[k]['calm_verified'])
    candidates['reason'] = candidates.audit_id.map(lambda k: reviews[k]['reason'])
    calm = candidates[candidates.calm_verified].set_index('video_id', drop=False)
    nonclean = calm[calm.group != 'clean'].video_id.tolist()
    clean = ordered(calm[calm.group == 'clean'].video_id.tolist(), 'contamination-clean-fill-v1-')
    selected = sorted(nonclean + clean[:max(0, 1000 - len(nonclean))])
    if len(selected) != 1000:
        raise ValueError(f'Need 1000 verified distinct videos; available={len(calm)}')
    output = calm.loc[selected]
    record = write_manifest('contamination_queries', output.to_dict('records'))
    report = dict(manifest=record, candidates=len(candidates), reviewed=len(reviews), calm_available=len(calm),
                  groups=output.groupby('group').size().to_dict(),
                  categories=output.groupby(['group', 'category']).size().to_dict(),
                  review_sha256=sha(ROOT / 'contamination_reviews.json'),
                  candidate_sha256=sha(ROOT / 'contamination_candidates.csv'),
                  method=audit['method'], selection='All visually calm violent/H1; deterministic normal fill to 1000',
                  clean_group='Descriptive control; contamination flag tests violent and H1 strata only',
                  sheet_sha256={p.name: sha(p) for p in sorted((ROOT / 'contamination_sheets').glob('*.jpg'))})
    report['categories'] = {str(k): v for k, v in report['categories'].items()}
    (ROOT / 'contamination_set.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ('candidates', 'reviewed', 'calm_available', 'groups', 'manifest')}, indent=2))


if __name__ == '__main__':
    main()
