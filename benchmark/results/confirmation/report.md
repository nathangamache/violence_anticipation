# Fold-1 confirmation (single evaluation, both heads)

## early head

Model: p1 on ['siglip2_prompts_v1']; fold-1 test videos scored 1214 of 1214; events 75

| model | M1 band | M1 30-60 | M8 |
|---|---|---|---|
| frozen model | 0.905 [0.848, 0.948] | 0.893 [0.841, 0.933] | 0.906 [0.866, 0.939] |
| nuisance N | 0.841 [0.772, 0.897] | 0.832 [0.772, 0.886] | 0.751 [0.675, 0.826] |

M8 gain over N: 0.155 [0.081, 0.226]; M4n: 0.090 [0.032, 0.160]

Rule: fold 1 band M1 CI lower > 0.50 and positive M8 point gain over the nuisance model N
- m1_lower: 0.847836858665569
- m1_lower_gt_0.5: True
- m8_gain_point: 0.15475143844941774
- m8_gain_point_gt_0: True
- m8_gain_ci: [0.08121288623981734, 0.22565462978746278]
- confirmed: True

| target FA/h | test FA/h | Success@30 | position null | random |
|---|---|---|---|---|
| 1 | 0.33 | 0.160 [0.079, 0.250] | 0.034 | 0.008 [0.002, 0.017] |
| 2 | 0.72 | 0.227 [0.130, 0.318] | 0.043 | 0.018 [0.008, 0.028] |
| 5 | 2.64 | 0.360 [0.238, 0.480] | 0.086 | 0.064 [0.043, 0.088] |

## imminent head

Model: p1 on ['siglip2_prompts_v2']; fold-1 test videos scored 1214 of 1214; events 75

| model | M1-near | M3p-near | M8-near |
|---|---|---|---|
| frozen model | 0.893 [0.843, 0.938] | n/a | 0.919 [0.879, 0.950] |
| nuisance N | 0.799 [0.732, 0.865] | n/a | 0.783 [0.709, 0.858] |

M8 gain over N: 0.136 [0.057, 0.206]; M4n: 0.109 [0.062, 0.156]

Rule: fold 1 M1-near CI lower > 0.60 and Success@10 at the 2 FA/h validation operating point has a threshold-aware CI lower bound above the position null at the same test FA/h (amendment 05)
- m1_near_lower: 0.843076093087593
- m1_near_lower_gt_0.6: True
- success10_fa2: 0.2
- success10_threshold_aware_ci: [0.10168877425693934, 0.39221209610604807]
- position_null_at_test_fa: 0.053750062492344174
- success10_ci_lower_gt_position_null: True
- confirmed: True

| target FA/h | test FA/h | Success@10 | threshold-aware CI | position null |
|---|---|---|---|---|
| 1 | 0.42 | 0.107 [0.035, 0.185] | [0.016, 0.224] | 0.040 |
| 2 | 0.85 | 0.200 [0.117, 0.288] | [0.102, 0.392] | 0.054 |
| 5 | 3.23 | 0.453 [0.333, 0.564] | [0.275, 0.586] | 0.117 |

Confirmed: early True, imminent True
