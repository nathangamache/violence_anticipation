# Onset spot-check: HIVAU train-split onsets vs official onsets

WARNING: the verdicts below are visual judgements by an AI model (one annotator, one pass), not human ground truth.
They are a coarse sanity check on onset quality and should be replaced or confirmed by a human pass before any
claim is made in a paper.

## Method

- Samples: 50 HIVAU-sourced eligible30 events (`annotations/hivau/spotcheck/`) and 30 events with official
  onsets (`annotations/hivau/spotcheck_official_control/`), rendered by `anticipation/data/spotcheck.py` as
  7-frame contact sheets at onset -10, -5, -2, 0, +2, +5, +10 s.
- Blind protocol: all 80 sheets were copied under seeded-shuffle names (`bNN.jpg`, seed 12345) with a private key
  kept outside the repo; the annotator judged each sheet without looking at the key, video id or onset source.
  The frame time stamps on the sheet (and the onset label) were visible, so the judge knew the marked time but not
  the source. Sheets are movie, CCTV, news and sports footage mixed in both sets, which also limits blinding.
- Verdicts: ok (violence visibly starts within about 2 s of the mark), early (mark precedes the violence),
  late (violence already under way at the mark), wrong (no violence in the segment), unclear.
- Ambiguous sheets were re-rendered with `tools/contact_sheet.py` (step 0.5 s, tile width 192) around the mark
  using the key only to locate the video; 6 events were refined this way (HIVAU: 2 early, 1 ok, 1 wrong; official: 1 ok, 1 early;
  the note column says "fine pass"). Corrected onsets are from those
  passes or read off the 7-frame sheet (accurate to a few seconds only).
- Results are written to the verdict, corrected_onset_sec and notes columns of the two `spotcheck.csv` files;
  summaries come from `python -m anticipation.data.spotcheck --summarize --out_dir <dir>`.

## Verdict counts

| Verdict | HIVAU (n=50) | Official (n=30) |
|---|---|---|
| ok | 12 (24%) | 18 (60%) |
| early | 3 | 3 |
| late | 9 | 2 |
| wrong | 14 | 3 |
| unclear | 12 | 4 |

By source kind (ok / early / late / wrong / unclear):

| Kind | HIVAU | Official |
|---|---|---|
| ucf | 1 / 1 / 4 / 7 / 3 | 6 / 0 / 1 / 0 / 1 |
| xd_movie | 4 / 1 / 4 / 1 / 6 | 4 / 1 / 1 / 2 / 2 |
| xd_youtube | 7 / 1 / 1 / 6 / 3 | 8 / 2 / 0 / 1 / 1 |

Among sheets with a decisive verdict (ok, early, late, wrong), ok is 12/38 (32%) for HIVAU and 18/26 (69%) for
official. Counting unclear as failures, the ok rates are 24% and 60%.

## Offsets

Corrected onsets (corrected minus marked, seconds; negative means the mark is after the true start, i.e. late):

- HIVAU early (n=3): +3.4, +6.3, +9.6.
- HIVAU late with an estimate (n=5 of 9): -3.6, -6.3, -8.0, -10.2, -14.9.
- Over all 8 early/late corrections: median -5.0 s, range -14.9 to +9.6 s. The summarize tool reports -3.6 s
  because it also includes one ok event (+2.0 s).
- Official early (n=3): +4.8, +5.1, +7.5 (the fourth correction is an ok event, +1.5 s). All early marks, no
  negative offsets.

## Examples

- Early (HIVAU, ice hockey): play continues for 6 s after the mark; the fight starts at 132.5 s vs mark 126.2 s.
- Early (HIVAU, CCTV arrest): officer works at a pole for about 10 s after the mark; the struggle starts near 100.5 s.
- Late (HIVAU, store robbery CCTV): robber is already over the counter attacking the clerk at -2 s.
- Late (HIVAU, street CCTV): vehicles already ramming police cars at -10 s.
- Wrong (HIVAU): pixel-art video game, a news talking head, shooting-range practice, calm shop and street CCTV.
- Early (official, basketball): mark falls during play; brawl starts about 5 s later.

## Implications for VA30

- HIVAU onsets are noticeably less reliable than official ones: about a quarter pass the visual check against
  60% for official onsets. Roughly 28% of HIVAU events (14/50) show no violence at the mark, but 3/30 official events
  also look wrong, so part of that is the limit of judging from 7 frames (movie cutaways, video-game footage,
  news inserts). The excess over official is real: wrong+late is 23/50 vs 5/30.
- There is no single systematic direction. Late marks outnumber early ones (9 vs 3), and late errors tend to be
  larger (median about 8 s vs about 6 s early). A late mark is the dangerous direction for anticipation: it makes
  the labelled lead-in contain violence, so a 30 s warning in training is actually a warning at or after the start.
  Even so, HIVAU also contains early marks of 3 to 10 s, which would inflate apparent lead time.
- The official-onset breakdown should therefore be co-primary for VA30 headline numbers, with the HIVAU-onset
  breakdown reported separately and labelled as noisier. Training on HIVAU onsets should be expected to carry
  label noise of the order of 5 to 10 s on roughly one third of events, plus some events with no violence.
- Action items, subject to human confirmation: re-check the `wrong` and `late` HIVAU rows in the CSV; consider a
  conservative filter that drops HIVAU events judged wrong or late from any evaluation-critical set; do not use
  the corrected onsets as ground truth without a second human pass.
- Sample sizes are small (50 and 30), so these rates have wide intervals (a 95% interval on 12/50 is about 14% to
  38%).
