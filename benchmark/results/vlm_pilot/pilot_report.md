# Phase C pilot: interim v1 labels

| arm | M1 band | M1 near | M8 near | M3p near | near gain over N | Success@10 / achieved FA/h | N1/N2/N3 | G2/G3 |
|---|---|---|---|---|---|---|---|---|
| A | 0.747 [0.687, 0.805] | 0.693 [0.619, 0.764] | 0.781 [0.709, 0.856] | 0.600 [0.460, 0.737] | 0.048 [-0.038, 0.124] | 0.570 [0.473, 0.670] / 8.09 | FAIL/FAIL/FAIL | PASS/PASS |
| B | 0.699 [0.637, 0.766] | 0.802 [0.734, 0.863] | 0.875 [0.811, 0.929] | 0.665 [0.519, 0.793] | 0.155 [0.069, 0.233] | 0.420 [0.318, 0.520] / 3.95 | PASS/PASS/PASS | PASS/INCONCLUSIVE |
| C | 0.747 [0.680, 0.818] | 0.781 [0.700, 0.856] | 0.835 [0.768, 0.906] | 0.649 [0.515, 0.769] | 0.174 [0.084, 0.252] | 0.300 [0.206, 0.400] / 1.88 | PASS/PASS/PASS | PASS/PASS |
| forecast | 0.728 [0.656, 0.799] | 0.744 [0.655, 0.828] | 0.820 [0.744, 0.888] | 0.647 [0.504, 0.778] | 0.054 [0.001, 0.107] | 0.240 [0.157, 0.326] / 4.14 | PASS/PASS/PASS | PASS/INCONCLUSIVE |
| hidden_A | 0.778 [0.699, 0.850] | 0.731 [0.634, 0.838] | 0.764 [0.675, 0.855] | 0.511 [0.384, 0.646] | 0.099 [0.001, 0.197] | 0.370 [0.280, 0.462] / 9.04 | PASS/FAIL/PASS | FAIL/INCONCLUSIVE |
| hidden_B | 0.783 [0.702, 0.863] | 0.836 [0.765, 0.902] | 0.863 [0.802, 0.923] | 0.593 [0.451, 0.743] | 0.151 [0.043, 0.261] | 0.390 [0.298, 0.494] / 4.52 | PASS/FAIL/PASS | INCONCLUSIVE/INCONCLUSIVE |
| hidden_C | 0.798 [0.721, 0.866] | 0.864 [0.803, 0.920] | 0.886 [0.828, 0.937] | 0.547 [0.407, 0.689] | 0.191 [0.080, 0.297] | 0.490 [0.393, 0.592] / 4.71 | PASS/FAIL/PASS | PASS/INCONCLUSIVE |
| aggression | 0.808 [0.582, 0.909] | 0.881 [0.688, 0.956] | 0.921 [0.754, 0.998] | 0.756 [0.621, 0.869] | 0.131 [0.029, 0.245] | 0.114 [0.024, 0.217] / 1.09 | PASS/PASS/PASS | PASS/INCONCLUSIVE |

Decision: **NO-GO**, recorded in [decision_phase_c.md](../../screens/decision_phase_c.md). No rest-of-S extraction was authorized or started.

All CIs use 1,000 paired split_group cluster-bootstrap replicates; undefined / insufficient-replicate CIs remain NA. Models use the registered P1 summaries, train-only weighted PLS32 for wide states, validation C selection, and no H1 training. Hidden layers 18/27 are selected separately for each head on validation slices only. A/B/C use identical 352-video pilot labels and intervals (100 positive, 150 clean, 102 H1p). The aggression P1 reference has narrower cache coverage and must not be compared as an unrestricted row.

Success@10 in the top table uses the nominal validation 2 FA/h operating point; achieved test rates are shown explicitly.

## Early metrics and temporal control

| arm | M1 long | M4b, all P | paired M1 minus M4b | registered S_ramp M4b | paired S_ramp gain | M4n | M3p | specificity ratio | M8 |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.752 [0.689, 0.815] | 0.624 [0.534, 0.716] | 0.122 [0.038, 0.203] | 0.596 [0.485, 0.709] | 0.146 [0.049, 0.253] | 0.081 [0.030, 0.134] | 0.705 [0.598, 0.811] | 1.030 [0.536, 2.478] | 0.828 [0.763, 0.896] |
| B | 0.683 [0.614, 0.759] | 0.585 [0.489, 0.678] | 0.114 [0.021, 0.209] | 0.557 [0.446, 0.662] | 0.131 [0.020, 0.245] | 0.023 [-0.046, 0.096] | 0.689 [0.578, 0.785] | 1.501 [0.749, 7.939] | 0.834 [0.770, 0.893] |
| C | 0.724 [0.644, 0.797] | 0.618 [0.537, 0.712] | 0.129 [0.049, 0.213] | 0.590 [0.490, 0.704] | 0.147 [0.040, 0.252] | 0.079 [0.008, 0.151] | 0.687 [0.569, 0.791] | 0.919 [0.445, 2.297] | 0.847 [0.780, 0.908] |
| forecast | 0.720 [0.644, 0.795] | 0.599 [0.500, 0.688] | 0.128 [0.063, 0.202] | 0.572 [0.457, 0.684] | 0.140 [0.066, 0.230] | 0.004 [-0.030, 0.041] | 0.656 [0.525, 0.783] | 0.745 [0.204, 1.537] | 0.800 [0.721, 0.873] |
| hidden_A | 0.781 [0.701, 0.856] | 0.780 [0.705, 0.849] | -0.002 [-0.101, 0.081] | 0.785 [0.694, 0.866] | 0.002 [-0.115, 0.099] | 0.053 [-0.038, 0.148] | 0.526 [0.396, 0.659] | 0.097 [-0.537, 0.635] | 0.808 [0.730, 0.882] |
| hidden_B | 0.789 [0.702, 0.869] | 0.743 [0.658, 0.824] | 0.041 [-0.042, 0.116] | 0.759 [0.662, 0.846] | 0.037 [-0.064, 0.129] | 0.064 [-0.042, 0.173] | 0.590 [0.444, 0.733] | 0.377 [-0.347, 1.577] | 0.819 [0.745, 0.890] |
| hidden_C | 0.791 [0.711, 0.861] | 0.694 [0.590, 0.790] | 0.104 [0.037, 0.166] | 0.677 [0.560, 0.784] | 0.115 [0.028, 0.189] | 0.070 [-0.033, 0.174] | 0.648 [0.508, 0.779] | 0.546 [0.069, 1.069] | 0.808 [0.734, 0.879] |
| aggression | 0.807 [0.566, 0.916] | 0.611 [0.384, 0.776] | 0.197 [0.066, 0.327] | 0.600 [0.324, 0.799] | 0.235 [0.050, 0.420] | -0.047 [-0.128, 0.035] | 0.743 [0.628, 0.846] | 0.688 [0.351, 2.168] | 0.841 [0.650, 0.953] |

## Shared-reference and real-footage checks

| arm | shared teacher videos | paired band gain vs raw aggression | criterion (i) | criterion (ii): G2 not FAIL | S_real band | criterion (iii) | UBI band | UBI near | official/UBI band |
|---|---|---|---|---|---|---|---|---|---|
| A | 112 | -0.108 [-0.225, 0.029] | NOT MET | PASS | 0.754 [0.679, 0.823] | PASS | 0.914 [0.802, 0.987] | 0.971 [0.882, 1.000] | 0.802 [0.705, 0.892] |
| B | 112 | -0.024 [-0.138, 0.097] | NOT MET | PASS | 0.688 [0.612, 0.763] | PASS | 0.774 [0.605, 0.921] | 0.879 [0.681, 1.000] | 0.759 [0.659, 0.847] |
| C | 112 | -0.016 [-0.112, 0.107] | NOT MET | PASS | 0.770 [0.690, 0.848] | PASS | 0.920 [0.820, 0.989] | 0.931 [0.803, 1.000] | 0.844 [0.761, 0.920] |
| forecast | 112 | -0.086 [-0.270, 0.151] | NOT MET | PASS | 0.759 [0.675, 0.836] | PASS | 0.942 [0.869, 0.994] | 0.976 [0.922, 1.000] | 0.767 [0.649, 0.867] |
| hidden_A | 112 | 0.097 [0.038, 0.190] | PASS | FAIL | 0.810 [0.716, 0.895] | PASS | 0.842 [0.644, 0.998] | 0.799 [0.468, 1.000] | 0.783 [0.670, 0.882] |
| hidden_B | 112 | 0.052 [-0.042, 0.184] | NOT MET | PASS | 0.801 [0.701, 0.892] | PASS | 0.830 [0.629, 0.977] | 0.824 [0.554, 1.000] | 0.779 [0.659, 0.882] |
| hidden_C | 112 | 0.065 [-0.030, 0.215] | NOT MET | PASS | 0.817 [0.735, 0.898] | PASS | 0.849 [0.620, 0.997] | 0.906 [0.748, 1.000] | 0.799 [0.668, 0.923] |
| aggression | 112 | 0.000 [-0.055, 0.053] | NOT MET | PASS | 0.845 [0.597, 0.943] | PASS | NA | NA | 0.754 [0.634, 0.865] |

Criterion (iv): **UCF contamination flagged** in both violent and H1 strata. No automatic pass is assigned. UBI and the curated hold-out must inform the owner decision. Every metric by source / official-UBI / HIVAU is in each metrics.json `subsets`; source-specific UCF-vs-H1p specificity is undefined for UBI-only positives. Primary v1 labels remain unchanged.

### Legacy b_teacher control on exactly shared videos

| arm | shared videos | b_teacher band | actual band | paired gain |
|---|---|---|---|---|
| A | 112 | 0.702 [0.608, 0.780] | 0.700 [0.542, 0.822] | -0.002 [-0.146, 0.125] |
| B | 112 | 0.702 [0.608, 0.780] | 0.784 [0.641, 0.879] | 0.082 [-0.035, 0.187] |
| C | 112 | 0.702 [0.608, 0.780] | 0.791 [0.652, 0.883] | 0.089 [-0.058, 0.222] |
| forecast | 112 | 0.702 [0.608, 0.780] | 0.721 [0.562, 0.851] | 0.019 [-0.144, 0.169] |
| hidden_A | 112 | 0.702 [0.608, 0.780] | 0.904 [0.758, 0.961] | 0.203 [0.061, 0.302] |
| hidden_B | 112 | 0.702 [0.608, 0.780] | 0.859 [0.739, 0.944] | 0.158 [0.033, 0.272] |
| hidden_C | 112 | 0.702 [0.608, 0.780] | 0.873 [0.746, 0.956] | 0.171 [0.052, 0.292] |
| aggression | 112 | 0.702 [0.608, 0.780] | 0.808 [0.589, 0.906] | 0.106 [-0.101, 0.231] |

## Event operating points

Thresholds are selected on each model fold’s clean validation slices at nominal 1/2/5 FA/h. Report achieved test FA/h alongside success: these can differ substantially on this small pilot. Success CIs below are paired split_group resamples at fixed selected thresholds; VA30 files additionally contain threshold-aware Success@30 CIs (200 replicates). The evaluator uses restrict_to_scored and benchmark/extract_manifest_v1.csv horizons; positives are additionally limited to frozen P intervals. Original fold 1 is removed before the event evaluator remaps the other folds.

| arm / head / lead | nominal FA/h | achieved test FA/h | success | position null | random | UBI success | official/UBI success |
|---|---|---|---|---|---|---|---|
| A / imminent / 10s | 1 | 7.523 | 0.560 [0.462, 0.656] | 0.336 | 0.205 | 0.750 [0.461, 1.000] | 0.667 [0.476, 0.840] |
| A / imminent / 10s | 2 | 8.088 | 0.570 [0.473, 0.670] | 0.357 | 0.219 | 0.750 [0.461, 1.000] | 0.667 [0.476, 0.840] |
| A / imminent / 10s | 5 | 8.276 | 0.560 [0.461, 0.660] | 0.363 | 0.223 | 0.750 [0.461, 1.000] | 0.667 [0.476, 0.840] |
| A / early / 30s | 1 | 6.395 | 0.460 [0.367, 0.558] | 0.247 | 0.148 | 0.833 [0.571, 1.000] | 0.704 [0.522, 0.862] |
| A / early / 30s | 2 | 6.959 | 0.470 [0.373, 0.570] | 0.263 | 0.160 | 0.833 [0.571, 1.000] | 0.741 [0.571, 0.889] |
| A / early / 30s | 5 | 10.345 | 0.530 [0.421, 0.629] | 0.358 | 0.228 | 0.667 [0.364, 0.929] | 0.667 [0.486, 0.833] |
| B / imminent / 10s | 1 | 3.762 | 0.340 [0.238, 0.434] | 0.195 | 0.109 | 0.417 [0.111, 0.714] | 0.444 [0.250, 0.645] |
| B / imminent / 10s | 2 | 3.950 | 0.420 [0.318, 0.520] | 0.202 | 0.114 | 0.500 [0.222, 0.818] | 0.519 [0.310, 0.724] |
| B / imminent / 10s | 5 | 5.454 | 0.450 [0.351, 0.545] | 0.259 | 0.154 | 0.583 [0.312, 0.857] | 0.519 [0.320, 0.720] |
| B / early / 30s | 1 | 1.128 | 0.280 [0.183, 0.385] | 0.069 | 0.028 | 0.167 [0.000, 0.429] | 0.333 [0.156, 0.552] |
| B / early / 30s | 2 | 1.128 | 0.290 [0.194, 0.388] | 0.069 | 0.028 | 0.250 [0.000, 0.533] | 0.370 [0.188, 0.583] |
| B / early / 30s | 5 | 4.138 | 0.340 [0.247, 0.439] | 0.174 | 0.098 | 0.417 [0.133, 0.750] | 0.407 [0.219, 0.607] |
| C / imminent / 10s | 1 | 1.881 | 0.290 [0.196, 0.390] | 0.117 | 0.056 | 0.500 [0.222, 0.818] | 0.370 [0.182, 0.579] |
| C / imminent / 10s | 2 | 1.881 | 0.300 [0.206, 0.400] | 0.117 | 0.056 | 0.500 [0.222, 0.818] | 0.370 [0.182, 0.579] |
| C / imminent / 10s | 5 | 4.890 | 0.460 [0.360, 0.557] | 0.238 | 0.139 | 0.833 [0.571, 1.000] | 0.630 [0.435, 0.808] |
| C / early / 30s | 1 | 3.009 | 0.180 [0.108, 0.252] | 0.135 | 0.072 | 0.250 [0.000, 0.500] | 0.148 [0.034, 0.286] |
| C / early / 30s | 2 | 3.009 | 0.180 [0.108, 0.252] | 0.135 | 0.072 | 0.250 [0.000, 0.500] | 0.148 [0.034, 0.286] |
| C / early / 30s | 5 | 5.454 | 0.340 [0.238, 0.442] | 0.220 | 0.127 | 0.417 [0.143, 0.714] | 0.407 [0.207, 0.600] |
| forecast / imminent / 10s | 1 | 3.950 | 0.240 [0.157, 0.326] | 0.202 | 0.114 | 0.417 [0.143, 0.714] | 0.259 [0.100, 0.417] |
| forecast / imminent / 10s | 2 | 4.138 | 0.240 [0.157, 0.326] | 0.210 | 0.119 | 0.417 [0.143, 0.714] | 0.259 [0.100, 0.417] |
| forecast / imminent / 10s | 5 | 5.078 | 0.280 [0.192, 0.374] | 0.245 | 0.144 | 0.417 [0.143, 0.714] | 0.370 [0.200, 0.542] |
| forecast / early / 30s | 1 | 2.633 | 0.200 [0.120, 0.283] | 0.122 | 0.064 | 0.417 [0.143, 0.727] | 0.259 [0.105, 0.441] |
| forecast / early / 30s | 2 | 2.633 | 0.200 [0.120, 0.283] | 0.122 | 0.064 | 0.417 [0.143, 0.727] | 0.259 [0.105, 0.441] |
| forecast / early / 30s | 5 | 4.890 | 0.260 [0.159, 0.357] | 0.200 | 0.115 | 0.333 [0.083, 0.625] | 0.259 [0.097, 0.438] |
| hidden_A / imminent / 10s | 1 | 8.476 | 0.360 [0.269, 0.450] | 0.370 | 0.228 | 0.167 [0.000, 0.429] | 0.370 [0.194, 0.565] |
| hidden_A / imminent / 10s | 2 | 9.041 | 0.370 [0.280, 0.462] | 0.391 | 0.241 | 0.167 [0.000, 0.429] | 0.370 [0.194, 0.565] |
| hidden_A / imminent / 10s | 5 | 11.489 | 0.480 [0.375, 0.587] | 0.482 | 0.296 | 0.167 [0.000, 0.429] | 0.444 [0.263, 0.643] |
| hidden_A / early / 30s | 1 | 7.534 | 0.280 [0.202, 0.369] | 0.279 | 0.172 | 0.083 [0.000, 0.286] | 0.222 [0.074, 0.400] |
| hidden_A / early / 30s | 2 | 7.722 | 0.320 [0.248, 0.411] | 0.284 | 0.176 | 0.083 [0.000, 0.286] | 0.222 [0.074, 0.400] |
| hidden_A / early / 30s | 5 | 10.359 | 0.390 [0.299, 0.485] | 0.358 | 0.228 | 0.250 [0.000, 0.501] | 0.370 [0.200, 0.538] |
| hidden_B / imminent / 10s | 1 | 3.955 | 0.290 [0.204, 0.385] | 0.202 | 0.114 | 0.167 [0.000, 0.417] | 0.296 [0.133, 0.480] |
| hidden_B / imminent / 10s | 2 | 4.520 | 0.390 [0.298, 0.494] | 0.224 | 0.129 | 0.333 [0.083, 0.637] | 0.407 [0.217, 0.588] |
| hidden_B / imminent / 10s | 5 | 8.287 | 0.580 [0.486, 0.680] | 0.363 | 0.224 | 0.417 [0.143, 0.714] | 0.556 [0.360, 0.741] |
| hidden_B / early / 30s | 1 | 4.709 | 0.260 [0.179, 0.353] | 0.194 | 0.111 | 0.167 [0.000, 0.400] | 0.259 [0.105, 0.417] |
| hidden_B / early / 30s | 2 | 4.709 | 0.260 [0.179, 0.353] | 0.194 | 0.111 | 0.167 [0.000, 0.400] | 0.259 [0.105, 0.417] |
| hidden_B / early / 30s | 5 | 6.969 | 0.340 [0.252, 0.442] | 0.263 | 0.160 | 0.167 [0.000, 0.438] | 0.296 [0.115, 0.500] |
| hidden_C / imminent / 10s | 1 | 4.520 | 0.500 [0.402, 0.598] | 0.224 | 0.129 | 0.417 [0.143, 0.728] | 0.444 [0.261, 0.632] |
| hidden_C / imminent / 10s | 2 | 4.709 | 0.490 [0.393, 0.592] | 0.231 | 0.134 | 0.417 [0.143, 0.728] | 0.444 [0.261, 0.632] |
| hidden_C / imminent / 10s | 5 | 5.650 | 0.540 [0.433, 0.648] | 0.266 | 0.159 | 0.417 [0.143, 0.728] | 0.519 [0.333, 0.704] |
| hidden_C / early / 30s | 1 | 2.260 | 0.240 [0.158, 0.333] | 0.109 | 0.055 | 0.250 [0.000, 0.500] | 0.333 [0.161, 0.500] |
| hidden_C / early / 30s | 2 | 2.260 | 0.240 [0.158, 0.333] | 0.109 | 0.055 | 0.250 [0.000, 0.500] | 0.333 [0.161, 0.500] |
| hidden_C / early / 30s | 5 | 4.332 | 0.400 [0.306, 0.495] | 0.181 | 0.103 | 0.250 [0.000, 0.500] | 0.370 [0.187, 0.538] |
| aggression / imminent / 10s | 1 | 1.091 | 0.114 [0.024, 0.217] | 0.398 | 0.033 | NA | 0.133 [0.000, 0.308] |
| aggression / imminent / 10s | 2 | 1.091 | 0.114 [0.024, 0.217] | 0.398 | 0.033 | NA | 0.133 [0.000, 0.308] |
| aggression / imminent / 10s | 5 | 2.182 | 0.250 [0.128, 0.386] | 0.591 | 0.065 | NA | 0.267 [0.056, 0.500] |
| aggression / early / 30s | 1 | 0.727 | 0.091 [0.021, 0.186] | 0.223 | 0.018 | NA | 0.133 [0.000, 0.333] |
| aggression / early / 30s | 2 | 0.727 | 0.091 [0.021, 0.186] | 0.223 | 0.018 | NA | 0.133 [0.000, 0.333] |
| aggression / early / 30s | 5 | 1.455 | 0.136 [0.040, 0.245] | 0.377 | 0.036 | NA | 0.200 [0.000, 0.429] |

Both leads for both heads, all per-source event rows, and validation/test coverage are saved in each `event_metrics.json`.

## Curated hold-out

The frozen annotation cohort has **36 onsets in 14 videos / 21 raw incident clips** with at least 30 s since clip start. Accounting for prior violence, merged contact bouts, aftermath and the 2-of-3 alarm clock leaves **11 clear VA30-eligible events, 10 first events**. Features are re-extracted on per-clip cuts; history cannot cross an editorial boundary. Hold-out values are never used to fit or tune.

holdout_B: M1 band 0.588 [0.471, 0.755]; M1 near 0.877 [0.735, 0.983]; M8 0.748 [0.630, 0.860]; M8-near 0.892 [0.746, 0.977].

The four frozen fold models each score every held-out clip. Their copies share the original video’s bootstrap cluster; controls are the corresponding out-of-fold pilot clean/H1 videos. Real hold-out footage is source-matched to XD YouTube clean controls. External H1p specificity crosses collections and is descriptive; the registered UCF-only N2 is NA. Clip-relative person-masked scene embeddings use the same SigLIP/grey-box pipeline, with standalone >=.25-confidence pose boxes at t=2 s.

M1-long 0.576 [0.395, 0.789]; M4b 0.511 [0.297, 0.784]; paired temporal gain 0.077 [-0.095, 0.241]; M4n 0.016 [-0.112, 0.117]; external H1p specificity 0.526 [0.407, 0.683], ratio 0.297 [-5.037, 1.278]; external near specificity 0.700 [0.505, 0.885]; near gain over N 0.067 [-0.000, 0.138].

| head / lead | nominal FA/h | achieved FA/h | success | position null | random |
|---|---|---|---|---|---|
| imminent / 10s | 1 | 3.767 | 0.114 [0.000, 0.271] | 0.455 | 0.109 |
| imminent / 10s | 2 | 3.955 | 0.114 [0.000, 0.271] | 0.477 | 0.114 |
| imminent / 10s | 5 | 5.462 | 0.227 [0.089, 0.361] | 0.636 | 0.154 |
| early / 30s | 1 | 1.130 | 0.023 [0.000, 0.083] | 0.118 | 0.028 |
| early / 30s | 2 | 1.130 | 0.023 [0.000, 0.083] | 0.118 | 0.028 |
| early / 30s | 5 | 4.144 | 0.045 [0.000, 0.167] | 0.432 | 0.098 |

The frozen pilot validation timelines, including their positive score candidates, are retained for exact threshold transfer. Position-null folds keep all copies of an original incident together, so a copy cannot supply onset-position labels to another copy. High near-window AUROC alone does not establish a usable >=10 s alarm lead; assess the event operating points separately.

## Cost and reproducibility

GPU use charged: 5.361 h plus 1.000 h reserved prior work; active reservations 0.000 h, hard total cap 35 h, cuda:1 only.

Timing on 200 queries per Qwen3 arm: A 10.30x / B 6.45x / C 8.04x realtime; projected 240 h costs 23.31 / 37.20 / 29.84 GPU-h, excluding reloads. Pretrained cut-causality and all 352-video A/B/C cache audits pass exactly. Cache validity is applied before fitting and metrics; feature windows reset at invalid gaps. Model bundles remain local; score tables, provenance, metrics and tests are versioned. No fold-1 data or results were read.
