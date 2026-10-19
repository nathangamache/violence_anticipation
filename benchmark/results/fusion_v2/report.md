# Fusion (early head): forward selection and final model

Pre-registration: `25bf720b688c5fe4475da25d2b4ba1de52a0a140678649073d08ba7586e52dbb`

Start candidates ['siglip2_prompts_v1', 'siglip2_prompts_v2'], further candidates []

## Forward selection (P1)

| step | candidate | M8 | M8 gain vs current (95% CI) | M1 band | selected |
|---|---|---|---|---|---|
| 0 start | siglip2_prompts_v1 | 0.875 |  | 0.844 | yes |
| 0 start | siglip2_prompts_v2 | 0.859 |  | 0.813 |  |
| 1 stop | siglip2_prompts_v2 | 0.874 | -0.001 [-0.009, +0.007] | 0.847 |  |

Selected, in order: ['siglip2_prompts_v1']

## Selected set

| model | M1 band | M1 30-60 | M8 (all) | M8 (first segment) |
|---|---|---|---|---|
| N (nuisance) | 0.771 [0.723, 0.816] | 0.774 [0.732, 0.817] | 0.757 [0.716, 0.800] | 0.736 [0.693, 0.782] |
| P1 | 0.844 [0.814, 0.873] | 0.838 [0.808, 0.867] | 0.875 [0.852, 0.898] | 0.849 [0.821, 0.878] |

P1 vs N: M4n 0.084 [0.045, 0.122], M8 gain over N 0.118 [0.079, 0.156]

Final model: **p1** on ['siglip2_prompts_v1']. Rule: P2 replaces P1 only if its seed-ensemble M8 gain over P1 has a paired-bootstrap CI lower bound > 0; otherwise P1 is kept.

## VA30 event metrics on the screening folds (post-selection)

| FA/h target | test FA/h | Success@30 | position null | random | Success@10 | median lead (s) |
|---|---|---|---|---|---|---|
| 1 | 1.16 | 0.206 [0.155, 0.260] | 0.073 | 0.029 [0.021, 0.036] | 0.308 [0.249, 0.368] | 34.771 [30.399, 43.522] |
| 2 | 2.12 | 0.314 [0.262, 0.365] | 0.095 | 0.052 [0.040, 0.063] | 0.424 [0.363, 0.485] | 42.563 [36.865, 51.256] |
| 5 | 4.99 | 0.390 [0.330, 0.443] | 0.161 | 0.117 [0.096, 0.137] | 0.515 [0.456, 0.571] | 42.604 [36.522, 47.679] |

Per-source Success@30 at each target:

| source | 1 FA/h | 2 FA/h | 5 FA/h |
|---|---|---|---|
| source:ubi | 0.196 [0.104, 0.314] | 0.339 [0.217, 0.471] | 0.339 [0.236, 0.450] |
| source:ucf | 0.044 [0.009, 0.089] | 0.165 [0.090, 0.245] | 0.209 [0.130, 0.299] |
| source:xd_movie | 0.286 [0.168, 0.403] | 0.407 [0.294, 0.526] | 0.527 [0.412, 0.653] |
| source:xd_youtube | 0.283 [0.189, 0.379] | 0.349 [0.250, 0.452] | 0.453 [0.343, 0.558] |
