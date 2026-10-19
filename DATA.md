# Data

No video, audio, frame, face or extracted feature from any dataset is redistributed in this repository. To run
anything beyond the committed reports you need your own copy of each dataset, obtained from its authors under
their terms.

## What the repository does contain

- **Benchmark definition** (`benchmark/videos.csv`, `benchmark/events.json`, `benchmark/folds_*.json`,
  `benchmark/v3/`, `benchmark/labels_v3/`): dataset clip identifiers as the datasets name them, durations,
  violence onsets and ends in seconds, our derived labels (eligibility, roles, folds) and the v3 onset corrections.
  These are research annotations of public datasets, published for research use and reproducibility. Reviewer notes
  were replaced by a redaction marker before publication.
- **Aggregate results** under `benchmark/results/` (reports, metrics, per-video scores keyed by clip id).
- **The curated YouTube hold-out** appears only as anonymised ids (`yt/holdout_001` to `yt/holdout_092`). Video
  ids, titles and annotations are not published.
- [`benchmark/public_redaction.json`](benchmark/public_redaction.json) lists every file that differs from the
  version used in the experiments, with both hashes.

## Sources and terms

| Dataset | Used for | Where to get it | Terms (as stated by the source) |
|---|---|---|---|
| XD-Violence (Wu et al., ECCV 2020) | Train and test videos with audio; official test onsets | https://roc-ng.github.io/XD-Violence/ | No explicit license; research use; cite the paper |
| UCF-Crime (Sultani et al., CVPR 2018) | CCTV videos; test onsets | https://www.crcv.ucf.edu/projects/real-world/ | Research use; cite the paper |
| UBI-Fights (Degardin and Proenca, IJCB 2020) | CCTV fight and normal videos, frame labels | https://socia-lab.di.ubi.pt/EventDetection/ | No license stated; cite the paper |
| HIVAU-70k (Holmes-VAU, CVPR 2025) | Onset timestamps for the UCF-Crime and XD-Violence training sets | The authors' Hugging Face release | See the dataset card (not verified here) |
| UCA (Yuan et al., CVPR 2024) | Captions used to propose soft masks | https://github.com/Xuange923/Surveillance-Video-Understanding | Annotations Apache-2.0; videos are UCF-Crime |
| NTU CCTV-Fights (Perez et al., ICASSP 2019) | External test set only | https://rose1.ntu.edu.sg/dataset/cctvFights/ (request form) | Academic, non-commercial; no redistribution and no derived datasets without permission. Nothing derived from it is published here. |
| MEVA (Kitware) | Never-violent multi-person footage for the external false-alarm test | `s3://mevadata-public-01/` (public, no account) | See the MEVA site (CC BY 4.0 reported, not verified here) |
| Curated YouTube hold-out | Descriptive hold-out only | Not redistributed | YouTube terms; research use |

## Rebuilding locally

1. Put datasets under `./data` or set `VA_DATA_ROOT` to a directory with the layout described in
   [docs/DATA_ACQUISITION.md](docs/DATA_ACQUISITION.md) (`raw_videos/<dataset>/...`, `annotations/...`).
2. Download with the scripts in `setup/` (`download_xd_train.py`, `download_xd_recover.py`, `download_ubi_fights.py`,
   `build_hivau_onsets.py`); UCF-Crime and the XD-Violence test split come from the authors' links.
3. Build the catalog, events and folds with `anticipation.data.catalog` and `anticipation.data.folds`
   ([docs/VA30.md](docs/VA30.md)); apply the published v3 corrections with `anticipation.data.apply_corrections`.
4. Extract features with `anticipation.features.run_extract` ([docs/FEATURES.md](docs/FEATURES.md)).
5. NTU CCTV-Fights: after you receive access, `setup/ingest_cctv_fights.py` rebuilds the external-test tables
   from your local copy ([docs/EXTERNAL_TEST.md](docs/EXTERNAL_TEST.md)).

The curated YouTube hold-out cannot be rebuilt from this repository.
