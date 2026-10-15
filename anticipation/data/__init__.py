"""Ground truth, tick labels and folds for the VA30 benchmark.

Modules:
    annotations  raw annotation parsers (XD-Violence, UCF-Crime, HIVAU-70k incl. per-event judgements, UBI-Fights,
                 curated YouTube), XD franchises and source spans
    catalog      builds benchmark/videos.csv and benchmark/events.json (one row per unique video)
    near_dups    near-duplicate fingerprints and matching (benchmark/near_duplicates.json)
    labels       per-video 1 Hz tick labels (horizons, masks, excluded spans, censoring, hazard bins, eligibility)
    folds        5-fold grouped cross-validation files (benchmark/folds_v2.json; folds_v1.json superseded)
    spotcheck    contact sheets for a human check of onsets

See docs/VA30.md for the benchmark as implemented.
"""
