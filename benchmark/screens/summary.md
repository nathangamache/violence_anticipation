# Frozen modality screen v1

Registered 2026-10-15 before new modality-screen outcomes. SHA256: `8831e16e664c895271b9cd77e0bab68733ca85a47655708998ab51d5e625797b`.

Screen folds: 0, 2, 3, 4. Confirmation fold: **1**, excluded from train, validation and test during screening. Previous published baseline results used all five folds; this is prospective protection for new blocks, not external confirmation.

Composition: 379 positive videos, 446 eligible30 events, 253 first events; 3220 CPU clean negatives, 600 VLM clean negatives (150/source, all teacher-covered screening clean videos included); 328 H1 and 102 people-focused H1p videos. Exact IDs, intervals and per-fold memberships are frozen in JSON. Audio remains blocked pending a diegetic audit. Proposed label masks are not applied to v2.

Blocks: Phase A = pose_interact, pose_state_identity_free, pose_state_pair_identity_ucf_ubi, siglip2_prompts_v1, siglip2_prompts_v2, siglip2, vjepa2_l, motion_no_camera, motion_camera_shortcut, camera_wearer_diagnostic. Phase B = audio_dsp, ast_events, clap_prompts, speech_tox, voice_affect, e2v_emb, overlap_and_duty_cycles, dialogue_acts. Phase C = vlm_single_frame_A, vlm_short_clip_B, vlm_long_window_C, vlm_forecast, vlm_hidden_state_probe. The controls, complete probe grids, training limits, metric thresholds and decision rules are serialized in JSON.

P1 is L2 logistic, C={0.001,0.01,0.1,1,10}, validation-selected band M1; causal mean/max/p95-duty summaries over 10/60/180s, slopes 15/30/60s as an ablation. Wide inputs use train-only PLS32. P2 is a 64-unit one-layer GRU only after G1 PASS, three seeds.

G1 PASS: M1>=0.58 and lower>0.50 after within-phase Holm adjustment; FAIL if upper<0.58. Sparse alternative: standardized pAUC>=0.55 with lower>0.50 or tail-hit-difference lower>0. G2 PASS: people-focused specificity lower>0.50 and ratio>=0.5; FAIL if upper<0.55 or ratio<0.25. G3 PASS: gain over nuisance N>=0.02 and lower>0; FAIL if upper<0.02. Other cases are INCONCLUSIVE. All CIs use 1000 split_group-cluster resamples, paired for differences; event threshold CIs use 200 validation/test resamples.

ANTICIPATION requires G1/G3 PASS and G2 not FAIL. PROVISIONAL requires G1 PASS and neither remaining gate FAIL. Otherwise classify CONTEXT/SHORTCUT, SHORT-HORIZON, INCONCLUSIVE or DROP as frozen in JSON. Forward selection adds only ANTICIPATION/PROVISIONAL blocks with positive paired M8 lower bound. Confirmation requires fold1 M1 lower>0.50 and positive M8 gain over N.

No block outcomes have been read for this registration. Do not read or execute the old trainer screen plan (which predates the new protocol) as this screen. Any change requires a dated hashed amendment before affected outcomes are read.
