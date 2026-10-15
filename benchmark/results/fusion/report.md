# Fusion (early head): forward selection and final model

Pre-registration: `8831e16e664c895271b9cd77e0bab68733ca85a47655708998ab51d5e625797b`

Start candidates ['siglip2_prompts_v1', 'siglip2_prompts_v2'], further candidates ['motion_camera_shortcut', 'motion_no_camera', 'pose_interact']

## Forward selection (P1)

| step | candidate | M8 | M8 gain vs current (95% CI) | M1 band | selected |
|---|---|---|---|---|---|
| 0 start | siglip2_prompts_v1 | 0.881 |  | 0.835 | yes |
| 0 start | siglip2_prompts_v2 | 0.872 |  | 0.829 |  |
| 1 stop | motion_camera_shortcut | 0.880 | -0.001 [-0.006, +0.003] | 0.837 |  |
| 1 stop | motion_no_camera | 0.886 | +0.005 [-0.003, +0.013] | 0.864 |  |
| 1 stop | pose_interact | 0.886 | +0.005 [-0.004, +0.015] | 0.839 |  |
| 1 stop | siglip2_prompts_v2 | 0.884 | +0.004 [-0.005, +0.011] | 0.841 |  |

Selected, in order: ['siglip2_prompts_v1']

## Selected set

| model | M1 band | M1 30-60 | M8 (all) | M8 (first segment) |
|---|---|---|---|---|
| N (nuisance) | 0.784 [0.743, 0.822] | 0.770 [0.733, 0.807] | 0.771 [0.735, 0.807] | 0.732 [0.693, 0.776] |
| P1 | 0.835 [0.799, 0.866] | 0.828 [0.795, 0.861] | 0.881 [0.859, 0.901] | 0.858 [0.829, 0.885] |

P1 vs N: M4n 0.066 [0.025, 0.105], M8 gain over N 0.109 [0.075, 0.143]

Final model: **p1** on ['siglip2_prompts_v1']. Rule: P2 replaces P1 only if its seed-ensemble M8 gain over P1 has a paired-bootstrap CI lower bound > 0; otherwise P1 is kept.

## VA30 event metrics on the screening folds (post-selection)

| FA/h target | test FA/h | Success@30 | position null | random | Success@10 | median lead (s) |
|---|---|---|---|---|---|---|
| 1 | 1.16 | 0.235 [0.191, 0.283] | 0.064 | 0.029 [0.021, 0.037] | 0.323 [0.269, 0.379] | 35.200 [32.416, 41.378] |
| 2 | 2.26 | 0.312 [0.258, 0.366] | 0.090 | 0.055 [0.043, 0.068] | 0.435 [0.379, 0.494] | 35.834 [33.220, 41.147] |
| 5 | 5.11 | 0.381 [0.333, 0.429] | 0.159 | 0.120 [0.100, 0.144] | 0.484 [0.434, 0.537] | 39.200 [34.792, 43.042] |

Per-source Success@30 at each target:

| source | 1 FA/h | 2 FA/h | 5 FA/h |
|---|---|---|---|
| source:ubi | 0.244 [0.160, 0.325] | 0.314 [0.211, 0.427] | 0.360 [0.260, 0.469] |
| source:ucf | 0.113 [0.056, 0.174] | 0.148 [0.082, 0.219] | 0.183 [0.118, 0.254] |
| source:xd_movie | 0.347 [0.222, 0.461] | 0.466 [0.330, 0.582] | 0.568 [0.463, 0.673] |
| source:xd_youtube | 0.236 [0.165, 0.319] | 0.315 [0.232, 0.412] | 0.402 [0.313, 0.495] |
