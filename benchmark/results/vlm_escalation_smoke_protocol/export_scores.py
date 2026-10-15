"""Export valid expected-risk ticks from one homogeneous stream, refusing confirmation-fold videos."""
import argparse
import json
from pathlib import Path

import pandas as pd

from anticipation.features import io
from protocol import read_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--extractor', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    rows = read_manifest(a.manifest, a.manifest_sha256)
    blocks, signature = [], None
    for row in rows:
        t, f, valid, meta = io.load_features(a.extractor, row['video_id'])
        cfg = {k: v for k, v in meta['cfg'].items() if k not in ('rationale_samples', 'rationale_max_tokens', 'decode_threads')}
        current = (meta['model_id'], meta['prompt_sha256'], json.dumps(cfg, sort_keys=True))
        if signature is not None and current != signature:
            raise ValueError('Refusing mixed models, prompts or configurations')
        signature = current
        selected = valid & (t >= float(row.get('start_sec') or 1)) & (t <= float(row['cap_sec']))
        index = meta['columns'].index('expected_risk')
        blocks.append(pd.DataFrame(dict(video_id=row['video_id'], t_sec=t[selected],
                                        score=f[selected, index], fold=-1, seed=0)))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pd.concat(blocks, ignore_index=True)
    table.to_parquet(str(out) + '.tmp', index=False)
    Path(str(out) + '.tmp').replace(out)
    out.with_suffix('.meta.json').write_text(json.dumps(dict(manifest_sha256=a.manifest_sha256,
        stream=a.extractor, model_prompt_cfg=signature, screening_folds=[0, 2, 3, 4]), indent=2))
    print(json.dumps(dict(rows=len(table), videos=table.video_id.nunique(), out=str(out))))


if __name__ == '__main__':
    main()
