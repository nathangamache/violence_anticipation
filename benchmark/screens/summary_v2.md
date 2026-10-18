# Screen set v2 registration

This summary describes the frozen registration only; it contains no screen outcomes.

- Registration SHA-256: `25bf720b688c5fe4475da25d2b4ba1de52a0a140678649073d08ba7586e52dbb`
- V3 videos catalog SHA-256: `eed1f922003df96996f2bbca381d3e395f404c972897fb60865116e5b930f842`
- V3 events catalog SHA-256: `5dfc0fd492cc5352d272f2336e00697377c1919a0f774d10253941f49f32158b`
- Fold assignment SHA-256: `8ebdf6ec2e549f4a7c41710c045a66af2823c8498cb60b4377363ccbd03965e6`
- Screening folds: 0, 2, 3, 4
- Frozen confirmation fold: 1

| Registered set | Count |
|---|---:|
| `positive_videos` | 295 |
| `eligible30_events` | 344 |
| `first_events` | 197 |
| `real_first_events` | 130 |
| `long60_first_events` | 93 |
| `ramp90_first_events` | 53 |
| `official_ubi_first_events` | 62 |
| `audio_first_event_candidates` | 68 |
| `clean_cpu_videos` | 3220 |
| `clean_vlm_videos` | 600 |
| `h1_videos` | 328 |
| `h1p_videos` | 102 |

| Fold | Train videos | Validation videos | Test videos |
|---:|---:|---:|---:|
| 0 | 2503 | 375 | 965 |
| 2 | 2539 | 332 | 972 |
| 3 | 2544 | 356 | 943 |
| 4 | 2542 | 338 | 963 |

H1 and H1p memberships and their scored intervals are preserved from v1. Fold 1 remains
frozen for the lead’s one-time confirmation.

Carried amendments:
- `benchmark/screens/amendment_02_harness_interpretations.md` (SHA-256 `33b32a859bc1d88b9fad5c2dea627a4a94e6c616791ca75f5783b30e036ebae9`)
- `benchmark/screens/amendment_03_label_mask_sensitivity.md` (SHA-256 `e355212e83a8e14190d536064ed8ca7f850f97e575a92b7e4d6a3c517c2dbb19`)
- `benchmark/screens/amendment_04_phase_b_descriptive.md` (SHA-256 `e72ef7423027534f94899b3c0494364836771b9c7cb57c8497188f3cf8fa501f`)
- `benchmark/screens/amendment_05_imminent_head.md` (SHA-256 `26e16450cabc9e55da5fefe72c48b38d5bc76b3c79db3297101cdc8542129dd7`)
