# Fusion (imminent head): forward selection and final model

Pre-registration: `8831e16e664c895271b9cd77e0bab68733ca85a47655708998ab51d5e625797b`

Start candidates ['siglip2', 'siglip2_prompts_v2'], further candidates []

## Forward selection (P1)

| step | candidate | M8 | M8 gain vs current (95% CI) | M1 band | selected |
|---|---|---|---|---|---|
| 0 start | siglip2 | 0.928 |  | 0.912 | yes |
| 0 start | siglip2_prompts_v2 | 0.904 |  | 0.860 |  |
| 1 stop | siglip2_prompts_v2 | 0.931 | +0.003 [-0.001, +0.006] | 0.913 |  |

Selected, in order: ['siglip2']

## Selected set

| model | M1-near | M3p-near | M8-near | M8-near official/UBI first |
|---|---|---|---|---|
| N (nuisance) | 0.794 [0.761, 0.827] | 0.521 [0.430, 0.617] | 0.805 [0.770, 0.837] | 0.820 [0.765, 0.874] |
| P1 | 0.912 [0.889, 0.934] | 0.626 [0.536, 0.722] | 0.928 [0.907, 0.945] | 0.930 [0.896, 0.961] |

P1 vs N: M4n 0.114 [0.083, 0.144], M8-near gain over N 0.123 [0.096, 0.152]
B-detect (violence-now probe, M1-near): 0.903 [0.876, 0.927]

Final model: **p1** on ['siglip2']. Rule: P2 replaces P1 only if its seed-ensemble M8-near gain over P1 has a paired-bootstrap CI lower bound > 0; otherwise P1 is kept.

## Success@10 on the screening folds (post-selection)

| FA/h target | test FA/h | Success@10 | threshold-aware CI | position null | random | official/UBI onsets | median lead (s) |
|---|---|---|---|---|---|---|---|
| 1 | 1.48 | 0.294 [0.239, 0.343] | [0.230, 0.392] | 0.079 | 0.044 | 0.239 [0.164, 0.322] | 33.6 |
| 2 | 2.44 | 0.408 [0.354, 0.461] | [0.346, 0.493] | 0.103 | 0.072 | 0.387 [0.307, 0.469] | 37.0 |
| 5 | 5.19 | 0.518 [0.461, 0.571] | [0.458, 0.591] | 0.170 | 0.147 | 0.500 [0.413, 0.582] | 40.2 |

Per-source Success@10:

| source | 1 FA/h | 2 FA/h | 5 FA/h |
|---|---|---|---|
| ubi | 0.151 [0.079, 0.256] | 0.302 [0.203, 0.402] | 0.430 [0.321, 0.547] |
| ucf | 0.191 [0.118, 0.261] | 0.252 [0.181, 0.331] | 0.313 [0.235, 0.398] |
| xd_movie | 0.390 [0.261, 0.533] | 0.559 [0.423, 0.688] | 0.763 [0.648, 0.867] |
| xd_youtube | 0.394 [0.299, 0.482] | 0.480 [0.389, 0.576] | 0.535 [0.451, 0.634] |
