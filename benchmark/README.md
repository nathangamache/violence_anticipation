# benchmark/: VA30 ground truth, folds and control results

Full documentation: [../docs/VA30.md](../docs/VA30.md) (version 2; section 10 lists what changed after the v1 review).

| File | Produced by | Notes |
|---|---|---|
| `videos.csv` | `python -m anticipation.data.catalog` | one row per unique video (7696, incl. UBI-Fights); re-generated when new sources land |
| `events.json` | same | violent segments in seconds per video, plus excluded spans (`other_segments`, `mask_spans`, `soft_mask_spans`) and per-event eligibility (`events`) |
| `catalog_summary.json` | same | counts per source / role / eligibility and the hashes of the two files above |
| `near_duplicates.json` | `python -m anticipation.data.near_dups all` | re-encoded / overlapping copies (93 pairs); read by the catalog (role `duplicate`, `split_group`) |
| `folds_v2.json` | `python -m anticipation.data.folds --out benchmark/folds_v2.json --salt va30-v2 --group_key split_group` | **current, frozen** 5-fold grouped CV + validation slices (content sha256 `8ebdf6ec2e54...`); extend with `--base`, never overwrite |
| `folds_v1.json` | v1 | **superseded, frozen** (no UBI, near-duplicates across folds, overlapping validation slices); kept to reproduce the v1 report |
| `results/<baseline>/` | `python -m anticipation.eval.baselines` | `metrics.json`, `report.md`, `amoc.png`, `reliability.png`, `events_op2.csv`, `scores.parquet` (when under 5 MB) |

Score a model: `.venv/bin/python -m anticipation.eval.va30 --scores <scores.parquet> --out_dir benchmark/results/<name>`
(score table columns `video_id, t_sec, score, fold, seed`; see docs/VA30.md section 6). A model must beat
`b_time_source_fit` and the position null, not only the Poisson random reference (docs/VA30.md section 8).

The v1 versions of every file here (and the v1 results) are archived in
`data/annotations/_archive/benchmark_v1_2026-10-14/`.
