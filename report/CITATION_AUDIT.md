# Citation Verification for the Capstone Draft

Checked October 14, 2026 for [REPORT.md](REPORT.md), against repository snapshot `9f14b92`. Reference numbers refer to that report's References section. Every retained reference URL was opened; title, author identity, date/version, and the scope of the attributed claim were checked in primary material. This is a scoped verification of included claims, not an assertion that every reference in `docs/research/` was reviewed or retained.

| References | Primary evidence inspected | Scope retained |
|---|---|---|
| 1 | Official Sultani author repository, citation, and description of snippet features / test temporal annotations | UCF-Crime and weakly supervised detection; no detection leaderboard number copied. |
| 2 | Wu arXiv record and official XD-Violence project page | Dataset and audiovisual snippet-relation model; no anticipation claim. |
| 3 | Pu arXiv abstract and January 2024 revision | Temporal context aggregation and semantic prompts. |
| 4 | Cao arXiv record and full HTML | NWPU Campus, scene dependence, and anomaly anticipation. Secondary 2026 comparison tables omitted. |
| 5 | INRIA author-hosted full PDF, dataset section and evaluation indicators | 1,470 videos; future seconds 1–3 and 4–8. Official CVF PDF fetch failed, so the verified author copy is the reference URL. |
| 6 | Official CVPR PDF, evaluation section and Table 3 | Verified reported values and full-video AUC scope. Browser access failed; direct HTTPS returned a PDF (HTTP 200), inspected through Ghostscript text extraction. |
| 7 | AssistPDA arXiv record and full HTML, metric definition | Online prediction/detection/analysis and average advance time. The research notes' exact advance-time number is omitted. |
| 8–11 | RULSTM, AVT, MeMViT, AFFT author abstracts and bibliographic records | Past/future modeling, future-feature training, memory, and feature fusion; no action-benchmark accuracy extrapolated to violence. |
| 12–14 | V-JEPA 2, Perception Encoder, and PE-AV primary author abstracts | Candidate representations only; no unverified throughput, licensing, or violence-performance claims. |
| 15 | Full Frontiers article, acoustic analysis and discussion | Escalation, pitch/intensity, and corpus dependence; no claim of a verified 30-second physical-violence precursor. |
| 16 | Author-hosted full political-debate escalation PDF and Glasgow institutional publication record | Prosodic/conversational modeling and changing conflict; 2013 Interspeech publication; transfer to assaults remains unproven. |
| 17 | DeepSafety arXiv record and full HTML, data and results | Conversational verbal abuse and reported F1 around 0.85; no future-contact claim. |
| 18 | MuTox author abstract and record | Multilingual audio toxicity resource, distinct from physical violence. |
| 19 | Penchala arXiv record and full HTML, input construction and Table 1 | 443 samples, source separation, 96.38% AUC, and at-most-four-second observation interval; warning lead is not established. |
| 20 | Ganesh author abstract and full HTML, pre-onset controls and inspected examples | Provenance artifacts and limits of pre-event separation; no broad claim that all separation is invalid. |
| 21 | Holmes-VAU arXiv record, full HTML annotation procedure, official CVPR-labeled author repository | Manual temporal segmentation plus model-assisted text; local train-onset validation remains pending. |
| 22 | UCA author abstract and bibliographic record | Surveillance video/language annotation resource; not VA30 onset ground truth. |
| 23 | Full AAAI DeepHit PDF | Event-time modeling with censoring; a methodological analogy. |
| 24 | Full Chan author-hosted PDF | Accident anticipation and warning-time reporting; no transfer of accident results to violence. |
| 25 | Degardin official author repository, citation and dataset description | Original UBI 1,000-video release and frame annotations; VA30 counts attributed separately to the local catalog. |

## Corrections and Exclusions

- Penchala's four-second observation interval was conflated with warning lead in the research notes. The report distinguishes these quantities.
- Wang's full-video AUC is qualified and is not treated as a pre-onset-only ceiling or a directly comparable VA30 measurement.
- Ganesh's inspection count was reversed in the dataset note: three of five UCF windows were violence-free, with an altercation underway in one. The erroneous count is excluded.
- The research notes call the linked Frontiers seizure paper a review; the opened article is a specific multiresolution-CNN seizure-prediction study. It is not retained as a review citation. The Poisson formula in the report is attributed to the implemented VA30 protocol and explicitly qualified by its independence assumption.
- The UBI website failed to open through the browser; the retained UBI reference is the verified official author repository. No access, licensing, or release-status assertion depends on the unavailable page.
- The wearable-aggression PMC link did not yield the paper through the browser (access challenge). Its empirical numbers are omitted. Other secondary-only leaderboard values, estimates, and sources not needed for the retained discussion are also omitted.

## Repository Evidence

Benchmark counts were checked against the committed `benchmark/catalog_summary.json`; fold counts and content hash against `benchmark/folds_v2.json` and `docs/VA30.md`; historical numbers against `docs/BASELINE.md` and `docs/EXPERIMENT_HISTORY.md`. The report distinguishes documented historical diagnostics, current benchmark controls, proposed methods, and unmeasured new-model results. No model training, feature extraction, video decoding, or benchmark recomputation was performed to prepare this documentation.
