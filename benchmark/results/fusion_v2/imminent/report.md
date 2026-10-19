# Fusion (imminent head): forward selection and final model

Pre-registration: `25bf720b688c5fe4475da25d2b4ba1de52a0a140678649073d08ba7586e52dbb`

Start candidates ['siglip2_prompts_v2'], further candidates []

## Forward selection (P1)

| step | candidate | M8 | M8 gain vs current (95% CI) | M1 band | selected |
|---|---|---|---|---|---|
| 0 start | siglip2_prompts_v2 | 0.894 |  | 0.853 | yes |

Selected, in order: ['siglip2_prompts_v2']

## Selected set

| model | M1-near | M3p-near | M8-near | M8-near official/UBI first |
|---|---|---|---|---|
| N (nuisance) | 0.792 [0.760, 0.827] | 0.497 [0.391, 0.602] | 0.801 [0.765, 0.837] | 0.789 [0.722, 0.852] |
| P1 | 0.853 [0.815, 0.885] | 0.688 [0.594, 0.776] | 0.894 [0.871, 0.914] | 0.874 [0.831, 0.912] |

P1 vs N: M4n 0.089 [0.055, 0.121], M8-near gain over N 0.093 [0.057, 0.127]
B-detect (violence-now probe, M1-near): 0.844 [0.805, 0.878]

Final model: **p1** on ['siglip2_prompts_v2']. Rule: P2 replaces P1 only if its seed-ensemble M8-near gain over P1 has a paired-bootstrap CI lower bound > 0; otherwise P1 is kept.

## Success@10 on the screening folds (post-selection)

| FA/h target | test FA/h | Success@10 | threshold-aware CI | position null | random | official/UBI onsets | median lead (s) |
|---|---|---|---|---|---|---|---|
| 1 | 1.20 | 0.259 [0.200, 0.319] | [0.173, 0.342] | 0.089 | 0.036 | 0.278 [0.179, 0.382] | 38.0 |
| 2 | 2.15 | 0.334 [0.276, 0.394] | [0.258, 0.416] | 0.112 | 0.064 | 0.333 [0.243, 0.434] | 38.6 |
| 5 | 4.71 | 0.483 [0.422, 0.548] | [0.389, 0.559] | 0.177 | 0.134 | 0.491 [0.403, 0.586] | 40.4 |

Per-source Success@10:

| source | 1 FA/h | 2 FA/h | 5 FA/h |
|---|---|---|---|
| ubi | 0.304 [0.159, 0.462] | 0.393 [0.271, 0.529] | 0.500 [0.388, 0.634] |
| ucf | 0.077 [0.030, 0.132] | 0.110 [0.052, 0.175] | 0.264 [0.179, 0.351] |
| xd_movie | 0.429 [0.289, 0.581] | 0.538 [0.388, 0.695] | 0.692 [0.544, 0.836] |
| xd_youtube | 0.245 [0.167, 0.337] | 0.321 [0.236, 0.417] | 0.481 [0.382, 0.593] |
