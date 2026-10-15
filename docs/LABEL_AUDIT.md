# Label audit and soft masks (Phase 0 step 3)

Outputs: `benchmark/audits/label_audit_v1.csv`, `benchmark/audits/soft_masks_v1.json`. Contact sheets (not committed;
`data/annotation/` is outside the repo history) are under `data/annotation/label_audit/<NN>/` (task 1, `NN` = row
order of the audited rows in the CSV) and `data/annotation/label_audit/uca/<video>/` (task 2).
`benchmark/events.json`, `videos.csv` and the fold files are unchanged; the masks are a proposal for a later version.

## 1. Early-contact audit

### Method

- Candidates: every `events[]` entry with `eligible30` true whose no-context `aggression_level`
  (`data/processed/vlm_labels_violence`, deduplicated by `(video, t)`) has a frame >= 0.8 in `[onset - 120, onset - 30]`.
  Time of a record = `raw_frame_idx / fps` with fps from `benchmark/videos.csv`; ids map to `ucf/<stem>` and
  `xd/<stem>`. The scored set only covers UCF and the XD test videos (1720 videos), so UBI has no scores and
  198 of the 551 eligible30 events are scored.
- Result: 51 events (the plan counts 53). The difference is not a threshold or window effect (0.75 or 0.8, 120 or
  125 s windows all give 51); it is most likely that the plan counted before the dedup and id mapping, or counted
  videos with duplicate-`t` records. Not resolved further.
- 24 of the 51 have all >= 0.8 frames inside an earlier annotated segment or before the event's `clear_start`
  (earlier annotated violence or its aftermath, including the 120 s pre-onset span of a previous event). Those are
  category d: the teacher is right, there is nothing to audit, and `eligible30` already starts the lead-in after
  that violence. This is the main reason the plan's 53 overstates "early contact".
- The remaining 27 got one contact sheet each (`tools/contact_sheet.py`, `[max(0, onset - 60), onset + 5]`, 1.5 s
  step, tile 192 px) plus a second sheet around the peak frame when it lay before `onset - 60` (12 events).
  Classes: (a) non-violent contact or posturing, (b) violence already under way before the onset (label noise),
  (c) teacher false positive or unclear. Classification was made by eye from the sheets; (b) calls for
  `Fighting024` and `Kill.Bill` are low confidence.

### Counts (27 audited events)

| | official | hivau | ubi | total |
|---|---|---|---|---|
| (a) non-violent contact or posturing | 5 | 0 | 0 | 5 |
| (b) violence before the onset | 3 | 10 | 0 | 13 |
| (c) teacher false positive or unclear | 7 | 2 | 0 | 9 |
| (d) earlier annotated violence (not audited) | 13 | 11 | 0 | 24 |

By source kind: (a) 5 xd_movie; (b) 10 ucf, 1 xd_movie, 2 xd_youtube; (c) 4 ucf, 2 xd_movie, 3 xd_youtube;
(d) 11 ucf, 5 xd_movie, 8 xd_youtube. There are no ubi rows because UBI has no `vlm_labels_violence`.

### Examples

- (a): `City.of.God` boys posing with pistols before the shooting; `Shoot.Em.Up` and `The.Hurt.Locker` gun standoffs;
  `Death.Proof` menacing driver before the crash. All five are movie clips.
- (b): `Fighting036` (man beaten on the ground from 23 s, onset 82.8); `Fighting050` (shop fight from 605 s, onset
  717.2); `Robbery100` (choke and struggle at 29 s, onset 134.5); `Arrest049` (restraint on the ground from 163 s,
  onset 256.2); two news montages (`v=7zEFHKKBA0g`, `v=uprb6aBzymw`) that show real clash footage in the intro.
- (c): `Shooting047` (title card text mentions a shooting); `Fighting048` (empty street); `v=251___mEwZA` (hockey
  gloves and commentators after an earlier fight); Hunger Games aftermath of a death; `Death.Proof` crash aftermath.

### Reading

Of the 27 audited events only 5 (19 %) are the thing the contact-state hypothesis needs: non-violent escalation
that precedes the onset. 13 (48 %) are label noise and 9 (33 %) are teacher false positives. Adding the 24 that
fall inside earlier annotated violence, 5 of 51 (10 %) high early scores are genuine escalation. All 5 are movie
clips (posed or staged), none is real-world footage. The 10 HIVAU (b) rows are all UCF.

## 2. UCA soft masks

### Method

- UCA captions (Train, Val, Test merged) for UCF videos with an eligible30 event.
- Strike regex (hit, punch, kick, beat, slap, stab, push down, attack, fight, drag, choke, wrestle, ...) with a
  non-physical filter (car hit, fire fight, drag a bag). Weapon-only regex (gun, pistol, knife, shot, threaten)
  when no strike verb is present.
- Keep captions with `end <= onset - 10 s`, `start >= clear_start - 0.5` and not inside an annotated segment
  (captions earlier than `clear_start` describe violence that already ends the lead-in).
- Hand review of the 38 candidate strike events removed four false positives (`Arrest006` car hit a flagpole,
  `Abuse018` a kick at a bag, `Assault002` a man pulled away, `Assault041` "kicked out of the shop").
- Mask = `[max(first caption start, clear_start), onset]`, one `uca_strike` entry and, separately, one
  `uca_weapon_only` entry per event.
- Negation: 12 caption hits contain a negation word; those read were all genuine violence ("did not stop and hit").

### Counts

- 34 events with a `uca_strike` mask (24 whose mask overlaps the scored 30-120 s window; the plan expects about
  18 to 22, which fits the 24 better than the 34).
- 19 events with a `uca_weapon_only` mask, 15 of them with no strike caption.
- 13 `label_audit_b` entries; 4 events are in both a strike mask and the (b) list.
- Union: 55 events in 54 videos, 66 entries. Total masked seconds (union per video) 5785 s: 4169 uca_strike,
  1453 uca_weapon_only, 985 label_audit_b; 4846 s without the weapon-only entries. Inside the 30-120 s lead
  window that VA30 band metrics use, 1861 s. The strike masks include long spans (`Fighting004`, `Fighting050`),
  which dominate the total.

### Visual spot check (15 sheets, 1.5 s step)

Confirmed 6 (`Abuse004`, `Abuse012`, `Abuse043`, `Assault027`, `Robbery088`, `Robbery124` gun), plausible 1
(`Abuse040`), inconclusive or not visible 8 (`Arrest008`, `Arrest047`, `Fighting031`, `Fighting034`,
`Fighting043`, `Fighting048`, `Assault037`, `Robbery128`). None is contradicted outright, but in `Arrest047` and
`Robbery128` nothing is visible in the caption window, so UCA time spans are loose (a few seconds, sometimes more)
and some strikes are too small at 192 px. The sheets therefore support the masks as a conservative exclusion, not
as exact violence onsets.

## 3. Recommendation on the contact-state hypothesis

Downgrade ranks 2 and 3 (pose contact state, SigLIP contact prompts) as an evidence-backed lead. The plan's rule
("if (b) dominates, downgrade") is met in the audited set (b is the largest class, 13 of 27), and the stronger point
is that only 5 of 51 early high teacher scores are real non-violent escalation, all in movies. Early aggression
scores in the real-world far window are mostly violence that the onset annotation missed (UCF HIVAU onsets) or
teacher errors, so the 30-120 s separation attributed to contact is partly label noise.

Do not drop them: keep them as screening candidates with the masks applied. The practical consequences:

- Apply `soft_masks_v1.json` before scoring any contact or aggression block, and report with and without the
  weapon-only entries.
- Re-score `aggression_level` through VA30 with the masks to see how much of the 0.67 to 0.71 band AUROC survives.
- Prioritise the precursor timeline annotation (MODALITY_PLAN 4.2) for the 5 genuine (a) cases and a sample of
  HIVAU UCF onsets, since the HIVAU UCF onsets are the main source of (b).
