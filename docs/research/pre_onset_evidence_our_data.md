# Pre-onset evidence from our own data: what is visible 30 to 120 s before violence

Audience: future agents and the project owner choosing which cues, modalities and extractors to invest in for
30+ s violence anticipation. This is **investigation 4** of the modality research: it uses only data already on
disk (CPU only, no model training) to measure what actually happens 30 to 120 s before an annotated violence onset
in our datasets, and how much of it is specific to violence. Written 2026-10-14. All numbers were computed by the
scripts listed in section 9 and are read from their output files; nothing is quoted from papers.

Read with: [README.md](README.md) (research brief), [../VA30.md](../VA30.md) (benchmark, onsets, time matching,
shortcut controls), [../BASELINE.md](../BASELINE.md), [../codebase/vlm_and_llm.md](../codebase/vlm_and_llm.md)
(teacher formats).

---

## 0. TL;DR

1. **Lead-in is scarce and mostly fights.** The median first onset is 6.9 s; only 15.6 % of the 2,129 violent videos
   have at least 30 s before the first onset (5.8 % at least 60 s, 2.5 % at least 120 s). The 551 VA30 eligible30
   events are 57 % fight/assault/abuse, 16 % riot, 14 % shooting, 7 % robbery and 5 % arrest. 43 % of them are later
   segments, where violence resumes after a lull. Their clear lead-in totals 9.9 h (capped at 120 s per event),
   against 303.5 h of clean negatives (section 2).
2. **Human captions (UCA, UCF-Crime CCTV) describe ordinary activity 30 to 60 s before violence.**
   - **No behaviour cue is significant at 30 to 60 s.** At least one interpersonal-precursor cue (approach,
     face-off, argue, gesture, push, grab, follow, crowd) appears in 31 % of 30 to 60 s windows, against 25 % of
     time-matched windows in normal videos; the confidence intervals overlap. Confrontation behaviours (face-off,
     pointing, push, grab, running) become significantly over-represented only 10 to 30 s before onset, and even
     then each appears in at most 7 % of events.
   - **Context is significant at 30 to 60 s:** people arriving or entering (23 vs 11 %), a cashier or counter
     (19 vs 9 %), a weapon in view (8 vs 1 %), police (5 vs 0 %), and fewer vehicles (11 vs 26 %).
   - **Precursors depend on the kind of violence.** Fights: talking, arguing and crowding. Robbery: counter, weapon
     display, face covering, entry. Arrest: police, following. Shooting: nothing beyond arrivals (section 3).
3. **The zero-shot Qwen2.5-VL teacher's far-window signal is mostly a static scene prior, not anticipation.**
   - **Strong AUROC, but it is not anticipation.** Its context-carrying `violent.score` reaches time-matched AUROC
     0.75 [0.67, 0.82] at 60 to 120 s and 0.73 [0.65, 0.81] at 30 to 60 s.
   - **It is static.** The first frame's score, broadcast over the whole video, does as well (0.72). The score
     does not rise toward the onset (within-video change from 60-120 s to 0-30 s: -0.04 [-0.10, 0.01]).
   - **It is not specific to violence.** It separates the windows before non-violent anomalies (burglary, road
     accidents, arson) from clean negatives just as well (0.71 to 0.72). The first frame, judged with no context,
     already scores 0.5 or more in 43 % of UCF violent videos and 44 % of UCF non-violent anomaly videos, but in
     0 of 47 UCF normal videos.
   - **What it means for VA30.** b_teacher's Success@30 is above the position null, but that is evidence of a
     dataset-collection prior, not anticipation (section 5.3).
4. **The no-context teacher's `aggression_level` is specific to violence, but it is a plateau, not a ramp.**
   - **Performance.** Time-matched AUROC is 0.71 [0.65, 0.77] at 60 to 120 s and 0.67 [0.62, 0.72] at 30 to 60 s.
     It is only 0.52 to 0.53 for windows before non-violent anomalies.
   - **No ramp.** It is flat from 170 s before onset to the onset (mean 0.17 to 0.23, against 0.07 in negatives),
     with no within-video rise.
   - **Concentrated in a subset.** The signal sits in the 41 % of events that have a "violent-looking" frame 30 to
     120 s before onset, against 20 % of random 90 s windows in negatives. Without those events it drops to 0.56.
   - **The most specific early tags are low-level physical contact:** "grabbing" 8.0 vs 0.2 % of frames,
     "physical contact" 6.5 vs 0.1 %, "pushing" 6.2 vs 0.2 % (section 5.4).
5. **HIVAU-70k has no text before events.** XD clip captions cover only annotated events (0 s outside), so an XD
   caption analysis of the lead-in is impossible. HIVAU onsets often include the build-up: in the 21 % of XD
   events whose machine captions mention a strike at all, the first such clip starts a median 3.7 s after the
   HIVAU onset (30 % more than 10 s after). Conversely, UCA describes violence before the benchmark onset in 22 of
   335 HIVAU-onset UCF videos (about 19 of them genuine) and in 0 of 43 official-onset videos (sections 3.5 and 4).
6. **Real-world YouTube clips show short visible escalation.** Snapshot: 55 reconciled videos. In 30 raw incident
   clips where the first contact is on screen, the visible escalation before contact has a median of 8.8 s
   (IQR 3.9 to 20.9), and 13 % last at least 30 s. Where the start of the escalation is itself on screen
   (15 clips), the median is 18.6 s and 20 % last at least 30 s. Only 39 % of the 206 raw clips with a visible
   escalation contain violence. These are visual estimates; the annotators worked from frames without listening to
   the audio (section 6).
7. **No measured cue rises between 120 s and 30 s before onset.** In every source, the time-varying behavioural
   signal that is specific to violence sits in the last 10 to 30 s. Beyond 30 s the data show three things:
   - a sustained aggressive state (low-level contact, aggressive posture, arguing) in a minority of events;
   - context specific to the kind of violence (weapons, police, counters, arrivals);
   - static scene priors.

   Success@30 well above about 15 to 25 % is not supported by any visual cue measured here. Audio could not be
   measured (UCF is mostly silent, UBI has none, the YouTube annotations were made from frames). It is the main
   remaining candidate for longer leads in bodycam, phone and road-rage footage. It must be measured on the
   extracted audio features before fusion work (section 7).

---

## 1. Questions, data and shared definitions

| Part | Question | Source on disk |
|---|---|---|
| (e) | How much pre-onset footage exists per source and class? Which kinds of violence are the eligible events? | `benchmark/videos.csv`, `benchmark/events.json` |
| (a) | What do human dense captions describe 120 to 0 s before UCF-Crime violence, compared with normal videos and non-violent anomalies? | `data/annotations/uca/json/UCFCrime_{Train,Val,Test}.json` |
| (b) | Does HIVAU-70k describe pre-event behaviour? | `data/annotations/hivau/raw_annotations/*.json` |
| (c) | Which zero-shot Qwen2.5-VL teacher scores and tags rise 30 to 60 s and 60 to 120 s before onset? | `data/processed/vlm_labels/` (context "both"), `data/processed/vlm_labels_violence/` (violence prompt, no context) |
| (d) | How many seconds of visible escalation precede violence in real-world footage? | YouTube annotation journals (reconciled `rec:` records) |

Shared definitions:

- **Onset** = start of the first violent segment of a violent video (events.json, seconds), unless stated
  otherwise. "Pre-violence" = before the first annotated segment of any kind, as in the VA30 headline.
- **Lead windows** are relative to the onset t0: 60-120 s = [t0-120, t0-60), 30-60 s = [t0-60, t0-30),
  10-30 s = [t0-30, t0-10), 0-10 s = [t0-10, t0), and "in" = [t0, t0+10] (detection reference). A window must hold
  at least 5 s of footage after the clip start.
- **Time matching** follows VA30: negatives are weighted (or pseudo-onsets drawn) so their time since clip start,
  and for the teacher also their source kind, match the positives'. This removes the position and provenance
  shortcuts documented in [../VA30.md](../VA30.md) section 8.
- **Hard negatives** = `other_anomaly` videos (burglary, stealing, shoplifting, arson, explosion, vandalism, road or
  car accident), measured before their first annotated anomaly. They share the "crime footage" look of violent
  videos but contain no violence. VA30 does not score them; here they test whether a cue is specific to violence.
- **Confidence intervals** (95 %) come from cluster bootstrap over videos (positive and negative videos resampled
  independently).

---

## 2. (e) How much pre-onset footage exists, and before what kind of violence

### 2.1 First onset per source (all 2,129 violent videos)

| Source kind | Violent videos | Median first onset (s) | p75 | p90 | >= 10 s | >= 30 s | >= 60 s | >= 120 s | Pre-onset hours (cap 120 s) |
|---|---|---|---|---|---|---|---|---|---|
| ubi | 215 | 6.8 | 22.0 | 48.0 | 44.7 % | 20.9 % | 7.4 % | 2.3 % | 1.08 |
| ucf | 378 | 12.2 | 29.9 | 71.1 | 56.1 % | 25.1 % | 11.4 % | 6.6 % | 2.60 |
| xd_movie | 870 | 8.6 | 18.8 | 34.6 | 45.7 % | 13.1 % | 3.1 % | 0.6 % | 3.52 |
| xd_youtube | 666 | 2.0 | 12.8 | 37.4 | 29.1 % | 11.9 % | 5.6 % | 2.9 % | 2.33 |
| **all** | **2129** | **6.9** | 19.3 | 41.3 | 42.3 % | **15.6 %** | **5.8 %** | 2.5 % | 9.52 |

### 2.2 Eligible30 events by kind of violence

Groups are assigned with priority fight/assault/abuse > riot > arrest > robbery > shooting, so a `fighting|shooting`
clip counts as a fight.

| Group | Violent videos | Median first onset (s) | First onset >= 30 s | Eligible30 events | of which first segment | Eligible60 | Clear lead-in h (cap 120 s) | Eligible30 by source |
|---|---|---|---|---|---|---|---|---|
| fight/assault/abuse | 1010 | 10.0 | 19.1 % | **315 (57 %)** | 184 | 110 | 5.35 | ubi 102, xd_movie 80, xd_youtube 73, ucf 60 |
| riot | 475 | 0.0 | 6.5 % | 89 (16 %) | 30 | 49 | 1.94 | xd_youtube 86, xd_movie 3 |
| shooting | 457 | 8.1 | 14.0 % | 79 (14 %) | 56 | 31 | 1.37 | xd_movie 60, ucf 16, xd_youtube 3 |
| robbery | 140 | 13.0 | 20.7 % | 41 (7 %) | 26 | 18 | 0.76 | ucf 41 |
| arrest | 47 | 14.7 | 34.0 % | 27 (5 %) | 16 | 10 | 0.51 | ucf 27 |
| **all** | 2129 | 6.9 | 15.6 % | **551** | **312** | 218 | **9.93** | xd_youtube 162, ucf 144, xd_movie 143, ubi 102 |

Facts that matter for cue selection:

- **Interpersonal fights dominate**, but **one in three eligible events is a riot or a shooting**. Riots are
  mostly XD YouTube compilations whose eligible events are later segments: 59 of 89 riot events are not the first
  segment. Shootings are mostly movie clips. Neither has a dyadic "argument, then punch" structure.
- **43 % of eligible events (239 of 551) follow earlier violence in the same clip.** Their lead-in is a lull
  after violence (the aftermath plus at least 30 s), not a build-up from calm. VA30's headline frame metrics
  exclude them; event metrics include them.
- **Lead-in length.** Among eligible30 events the median lead-in is 52.8 s (p25 38.8, p75 87.1). 40.5 % have at
  least 60 s and 15.8 % at least 120 s. Per source, the median is 54.0 (ubi), 53.0 (ucf), 45.8 (xd_movie) and
  57.6 s (xd_youtube).
- **Modalities available.** 352 of the 551 events (64 %) have a real audio track. By domain: CCTV 249 (45 %),
  YouTube/news 159 (29 %), movie 143 (26 %). All UBI events and most UCF events are silent.
- **Once violence starts it is short.** The median segment lasts 10.9 s (fights), 7.4 s (shootings), 9.1 s
  (robberies), 18.2 s (arrests) and 33.9 s (riots).

---

## 3. (a) UCA dense captions: what humans describe before UCF-Crime violence

### 3.1 Coverage and method

- **Coverage.** UCA covers all 378 UCF violent videos, 882 clean UCF normals and 547 other-anomaly videos. Captions
  are timestamped sentences with a median of 17 words; durations match the catalog within 0.04 s (median). They
  cover about 80 % of the timeline in every role (violent 82 %, normal 83 %, other anomaly 83 %; inside pre-onset
  windows 84 to 90 % median). A missing description therefore means "nothing notable", not "not captioned".
- **Unit.** One unit is the first violent segment of each UCF violent video. A caption belongs to a window if its
  midpoint lies in the window and it ends by t0 + 1 s. Captions that straddle the onset are analysed separately.
- **"Clean" variant (main tables).** 25 events are dropped because a caption with a strike word (hit, punch, kick,
  beat, slap, stab, fight, ...) ends 5 s or more before the onset, within 120 s. These are events whose "lead-in"
  already contains violence (section 3.5); 353 events remain. The "all" variant (378 events) is in the output files.
- **Negatives.** For every clean normal video and window, 10 pseudo-onsets are drawn from the onsets of the positive
  events that contribute to that window, constrained to fit the video. This matches time since clip start; caption
  density matches too (3.1 to 5.2 captions per minute in every group and window).
- **Hard negatives.** Other-anomaly videos, before their first annotated segment.
- **Cue lexicon.** 18 regex families on the caption text (`textutil.py`); categories are not exclusive.
- **Word-level effect sizes.** Log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi and Quinn 2008).
  The prior is the whole UCA corpus scaled to a0 = 1000; terms must appear in at least 3 positive videos. The z
  ignores clustering by video, so treat it as a ranking rather than a test.

Sample sizes (positive events per window, clean variant): 31 (60-120 s), 62 (30-60 s), 142 (10-30 s),
246 (0-10 s), 353 (in). Negatives: 440 to 882 normal videos per window; hard negatives: 64 to 361.

### 3.2 Cue presence by lead window

Cells give the % of events whose window has at least one caption with the cue, as violent vs time-matched normal.
**Bold** means the 95 % cluster-bootstrap CI of the difference excludes 0.

| Cue | 60-120 s (n=31) | 30-60 s (n=62) | 10-30 s (n=142) | 0-10 s (n=246) | in 0-10 s (n=353) |
|---|---|---|---|---|---|
| approach / close in | 26 vs 20 | 18 vs 13 | 11 vs 12 | 7 vs 5 | 10 vs 10 |
| face-off / confront / block | 3 vs 2 | 0 vs 2 | **6 vs 2** | 2 vs 1 | **3 vs 1** |
| argue / shout | 10 vs 9 | 6 vs 5 | 8 vs 5 | 1 vs 2 | 4 vs 4 |
| gesture / point / raise hand | 10 vs 2 | 0 vs 2 | **6 vs 1** | 2 vs 1 | **6 vs 1** |
| push / shove | 10 vs 3 | 2 vs 2 | **5 vs 2** | 0 vs 1 | **7 vs 2** |
| grab / pull / drag | 10 vs 1 | 0 vs 1 | **4 vs 1** | **2 vs 0** | **5 vs 1** |
| group / crowd / surround | 16 vs 6 | 6 vs 4 | 3 vs 3 | 1 vs 1 | **8 vs 3** |
| follow / chase | 3 vs 4 | 2 vs 2 | 2 vs 2 | 2 vs 1 | **4 vs 1** |
| run / flee / rush | 10 vs 2 | 5 vs 1 | **7 vs 1** | **5 vs 1** | **14 vs 1** |
| loiter / wait / look around | **19 vs 5** | 3 vs 2 | 3 vs 2 | 1 vs 1 | 1 vs 2 |
| enter / arrive | 29 vs 18 | **23 vs 11** | **19 vs 10** | **8 vs 5** | **12 vs 8** |
| talk / converse (calm) | 23 vs 12 | 13 vs 7 | 10 vs 6 | 5 vs 3 | 4 vs 6 |
| weapon / shooting | 6 vs 1 | **8 vs 1** | **8 vs 1** | **7 vs 0** | **17 vs 1** |
| police / security | 10 vs 0 | 5 vs 0 | **6 vs 0** | **2 vs 0** | **10 vs 0** |
| cashier / counter / money | **39 vs 13** | **19 vs 9** | **20 vs 9** | **11 vs 5** | **16 vs 8** |
| hand in pocket / conceal / mask | 3 vs 1 | 3 vs 1 | **6 vs 1** | 2 vs 1 | **5 vs 1** |
| vehicle | **13 vs 36** (lower) | **11 vs 26** (lower) | **16 vs 25** (lower) | 9 vs 12 | **16 vs 23** (lower) |
| strike words ("all" variant) | 7.5 vs 0.2 | 11.2 vs 0.1 | 9.6 vs 0.1 | 4.4 vs 0.0 | 17.2 vs 0.1 |
| **any interpersonal precursor** | 48 [32, 65] vs 36 | 31 [19, 42] vs 25 | 32 [24, 39] vs 23 | 16 [12, 21] vs 11 | 35 [30, 41] vs 21 |
| same, before non-violent anomalies | 31 | 22 | 31 | 10 | n/a |

Reading:

- **30 to 60 s: no interpersonal behaviour cue is significant.** Approach, argue, face-off, gesture, push and grab
  are all within noise; the "any precursor" rate (31 %) sits inside the normal footage's base rate (25 %).
- **10 to 30 s: confrontation behaviours appear.** Face-off, pointing, push, grab and running each occur in 4 to 7 %
  of events against 1 to 2 % of normal windows. Behaviours that humans would call escalation are visible about
  10 to 30 s ahead in CCTV, and only in a small minority of events.
- **The cues significant at 30 to 120 s are contextual:** where the camera is (counters, homes; fewer roads), who
  is present (police, people arriving) and what is in view (weapons).
- **Low specificity.** Normal footage carries an interpersonal-precursor cue in 25 to 36 % of time-matched windows,
  and the windows before non-violent anomalies carry one at about the same rate as before violence at 10 to 30 s
  (31 vs 32 %).
- **Human captions inside violence are not reliably violent.** Even in the first 10 s of the violent segment, only
  17 % of events have a strike word. Many captions inside annotated segments describe bystanders, people pressing
  someone down, falls ("fell to the ground") or flight. Segment boundaries are coarse.

### 3.3 Words that distinguish pre-onset windows (clean variant, log-odds z)

| Window | Over-represented before violence (z) | Under-represented |
|---|---|---|
| 60-120 s vs normal | man (4.7), house (4.5), ground (3.8), police (3.8), come in (3.6), take out (3.2), wander (3.2), money (3.0), bag (2.9) | road, car, driving, park, customer |
| 30-60 s vs normal | three men (4.2), police (3.5), gun (3.3), touch (2.9), man (2.6), lean (2.5), door (2.5), counter (2.1) | "there are", many, customer, road |
| 10-30 s vs normal | old (5.0), run (4.8), gun (4.7), man (4.6), counter (3.6), point (3.5), man enter (3.4), suddenly (3.3), cashier (3.1), policeman (3.1), another man (3.0) | clothing words, walk, many |
| 0-10 s vs normal | man (4.1), man walk (3.6), gun (3.5), run (2.9), house (2.8), two men (2.8), old (2.7), ground (2.7), raise (2.6) | clothing words |
| 30-60 s vs non-violent anomalies | counter (3.5), three men (2.5), enter (2.4), sit (2.3), police (2.1), gun (1.8) | car, motorcycle, look |

The distinctive vocabulary is about **who and where**: men, groups of men, old people and children (care-home and
child-abuse scenes), counters, police, houses, guns. It is not about interaction verbs. "Argue", "approach",
"shout" and "push" never reach the top of any window beyond 10 s. Only "point" and "run" do, at 10 to 30 s.

### 3.4 Cues by kind of violence (windows 10-60 s merged, clean variant)

Violent vs time-matched normal, % of event windows. **Bold** = CI of the difference excludes 0.

| Cue | fight/assault/abuse (n=43) | robbery (n=56) | shooting (n=23) | arrest (n=20) |
|---|---|---|---|---|
| approach / close in | 18 vs 12 | 11 vs 12 | 12 vs 12 | 6 vs 12 |
| face-off / confront | 6 vs 2 | 3 vs 2 | 3 vs 2 | 3 vs 2 |
| argue / shout | 11 vs 5 | 6 vs 5 | 9 vs 5 | 0 vs 5 |
| push / shove | 6 vs 2 | 4 vs 2 | 3 vs 2 | 0 vs 2 |
| grab / pull / drag | 5 vs 1 | 1 vs 1 | 0 vs 1 | 3 vs 1 |
| group / crowd | 10 vs 4 | 1 vs 4 | 0 vs 4 | 3 vs 4 |
| talk / converse | **19 vs 7** | 3 vs 7 | 12 vs 7 | 13 vs 7 |
| run / flee / rush | **6 vs 1** | **8 vs 1** | 6 vs 1 | 3 vs 1 |
| enter / arrive | 11 vs 10 | **25 vs 10** | **34 vs 10** | 10 vs 10 |
| weapon | 0 vs 1 | **19 vs 1** | 6 vs 1 | 0 vs 1 |
| hand in pocket / conceal / mask | 2 vs 1 | **10 vs 1** | 3 vs 1 | 0 vs 1 |
| cashier / counter / money | 5 vs 9 | **42 vs 9** | 16 vs 9 | 0 vs 9 |
| police / security | 5 vs 0 | 3 vs 0 | 3 vs 0 | **19 vs 0** |
| follow / chase | 0 vs 2 | 0 vs 2 | 0 vs 2 | **13 vs 2** |
| vehicle | **11 vs 25** (lower) | **8 vs 25** (lower) | 31 vs 25 | 23 vs 25 |

- **Fights** show the expected pattern (people talking, then arguing, a crowd, pushing), but each cue appears in
  only 5 to 19 % of events, and only "talk" and "run" clear the CI.
- **Robberies** are announced by place and props: a counter, a weapon shown, a covered face, someone entering.
  Part of this is the robbery itself starting before the annotated onset (for example, "a man holding a gun
  threatened a man ... and walked to the counter" 53 s before onset).
- **Arrests** are announced by police presence and pursuit.
- **Shootings** have no interpersonal precursor in CCTV captions; only arrivals are elevated.

### 3.5 Violence described before the annotated onset (label quality)

- **Earlier violence before HIVAU onsets.** In 22 of 378 UCF violent videos, a UCA caption with a strike word ends
  at least 10 s before the benchmark onset. All 22 have HIVAU onsets; none of the 43 official-onset videos do.
  Reading them, about 19 describe real violence, in some cases more than 100 s before onset (the earliest are
  108 s and 265 s). The other 3 are false matches (a car hitting a pole, a smashed screen,
  and one unclear). These lead-ins contain unannotated violence: a detector would "anticipate" them by detecting.
- **Onset agreement with UCA.** Counting strike and weapon words, the first UCA violence caption starts within 5 s
  of the onset for 74 % of official-onset events (n = 31) but only 35 % of HIVAU-onset events (n = 258); it starts
  more than 10 s before the onset for 6.5 % (official) vs 22 % (HIVAU). 89 of the 378 videos have no such caption. This independently supports VA30's warning that HIVAU
  train onsets are noisier. UCA could be used to audit them, or to move them earlier.

---

## 4. (b) HIVAU-70k: there is no pre-event text

The HIVAU raw files store, per video, `events` (segments), `clips` and `clips_caption` (per event), `events_summary`
and `video_summary`.

- **Captions exist only inside annotated events.** XD clip time outside events is exactly 0 s (train 421,275 s,
  test 52,143 s of clips). For UCF the HIVAU clips are UCA captions clipped to the events: 8,796 s (train) and
  2,299 s (test) fall outside because of straddling captions. The footage before the first event of anomalous
  XD videos (11.8 h in train, 1.8 h in test) is never described. **The XD pre-onset caption analysis the task asked for cannot be done from HIVAU.** XD
  pre-onset evidence comes only from the teacher (section 5).
- **Normal videos have random "events" with captions.** They work as a negative caption corpus.
- **Machine captions are vague about violence.** The XD clip captions come from a video LLM: median clip 4.8 s,
  frame-only. Strike words appear in only 6.6 % of first clips of violent events, against 0.5 % of normal clips
  and 2.8 % of first clips of other anomalies; only 779 of 3,747 violent events (21 %) ever mention a strike.
  "Crowd / group" is the most distinctive first-clip cue (42 % vs 22 % in normal clips). The top distinctive terms
  of first clips vs normal clips are police, officer, crowd, gear, smoke, run, gun, gather, helmet: riot and police
  footage.
- **HIVAU onsets include build-up.** Where a strike word exists, the first clip with one starts after the HIVAU
  onset by a median of 3.7 s in XD (p75 12.9, p90 29.6; 30 % more than 10 s, 9.5 % more than 30 s). In UCF the
  median is 0.4 s (15 % more than 10 s). Some HIVAU XD onsets therefore mark the build-up rather than the first
  blow, while some UCF ones come after earlier violence (section 3.5). Both errors push in opposite directions on
  the "lead time" of a model, which is another reason to keep the `onset:official` breakdown co-primary.
- **Event summaries.** Words like argument, confrontation, tension or threat appear in 60 % of XD violent event
  summaries (train 1,862 of 3,103; test 392 of 644) and 38 % of UCF ones (286 of 745), against 5 to 12 % of
  non-violent anomaly summaries. These are LLM rewrites of the clip captions and describe the event itself, so
  they are not evidence of a pre-event precursor.

---

## 5. (c) The Qwen2.5-VL-7B teacher outputs

### 5.1 Setup

- **Two label sets.**
  - `vlm_labels` (context "both"): each frame sees the text verdicts of up to 5 previous frames of the same video.
    Fields: violent, people_count, actions, weapons, scene, temporal_change, notes. This is the set behind VA30
    `b_teacher`.
  - `vlm_labels_violence` (violence prompt, no context). Fields: violent, aggression_level, threat_indicators,
    body_language, actions, weapons, people_count, scene.
- **Coverage.** XD test plus UCF `videos/`: 1,031 universe videos, 60,385 records. Records are deduplicated by
  `t`; time = `raw_frame_idx / fps` (fps from `data/metadata/<source>.json`), as in `anticipation/eval/baselines.py`.
- **Labels at record times.** From `anticipation.data.labels.tick_labels`, applying the benchmark's segments,
  `other_segments`, `mask_spans` and `soft_mask_spans`.
- **Positives.** Unmasked pre-violence records in each lead window: 60-120 s has 433 records from 53 videos; 30-60 s
  886 from 129; 10-30 s 1,973 from 309; 0-10 s 2,679 from 550.
- **Negatives.** 30,184 unmasked records of 327 clean videos (14.3 h).
- **AUROC.** Mann-Whitney AUROC, raw, source-matched and (source kind x time since start) matched with the VA30
  `TIME_MATCH_EDGES`. The tables show the time-matched value with a 1,000-resample video-cluster CI.
- **Tag families.** 15 regex families over the text fields (for example "push / grab / restrain" =
  push|shove|grab|pull|drag|restrain|grapple|wrestle|struggle|physical contact).

### 5.2 Per-window AUROC (time-matched; "in" = in-violence vs clean, source-matched)

| Feature | 60-120 s | 30-60 s | 10-30 s | 0-10 s | in |
|---|---|---|---|---|---|
| **context set: violent.score** | **0.752 [0.67, 0.82]** | **0.729 [0.65, 0.81]** | 0.702 [0.66, 0.74] | 0.705 [0.67, 0.74] | 0.839 |
| context set: any "argue / shout" tag | 0.544 [0.51, 0.58] | 0.543 [0.52, 0.57] | 0.544 [0.52, 0.57] | 0.560 [0.54, 0.58] | 0.600 |
| context set: any "push / grab" tag | 0.549 [0.51, 0.59] | 0.508 [0.50, 0.52] | 0.512 [0.50, 0.52] | 0.525 | 0.584 |
| context set: "calm / ordinary" tag | 0.427 [0.34, 0.52] | 0.430 [0.35, 0.52] | 0.408 [0.36, 0.46] | 0.408 | 0.414 |
| context set: people_count | 0.453 | 0.422 | 0.427 | 0.425 | 0.588 |
| context set: temporal_change = escalating | 0.507 | 0.504 | 0.508 | 0.518 | 0.534 |
| context set: weapon tag | 0.517 | 0.510 | 0.524 | 0.529 | 0.542 |
| **no-context set: aggression_level** | **0.714 [0.65, 0.77]** | **0.670 [0.62, 0.72]** | 0.676 [0.63, 0.72] | 0.695 [0.66, 0.73] | 0.842 |
| no-context set: violent.score | 0.624 [0.57, 0.67] | 0.620 [0.57, 0.66] | 0.640 [0.61, 0.67] | 0.649 [0.63, 0.67] | 0.811 |
| no-context set: "aggressive / tense posture" | 0.579 [0.54, 0.63] | 0.559 [0.54, 0.58] | 0.575 [0.56, 0.59] | 0.591 [0.57, 0.61] | 0.657 |
| no-context set: "push / grab / restrain" | 0.576 [0.54, 0.62] | 0.551 [0.53, 0.57] | 0.552 [0.54, 0.57] | 0.556 | 0.663 |
| no-context set: "strike / fight" | 0.534 | 0.525 | 0.526 | 0.532 | 0.599 |
| no-context set: "crowd / group" | 0.540 [0.49, 0.59] | 0.522 | 0.518 | 0.504 | 0.610 |
| no-context set: people_count | 0.502 | 0.464 | 0.453 | 0.436 | 0.572 |

Breakdowns at 30-60 s (time-matched, within source; CIs in `c_vlm_*.json`):

- **Context violent.score by source:** ucf 0.72 (90 videos), xd_movie 0.74 (20), xd_youtube 0.74 (17).
- **Context violent.score by kind of violence:** fights 0.74, shootings 0.73, robberies 0.70, arrests 0.73.
- **No-context aggression by source:** ucf 0.67, xd_movie 0.70, xd_youtube 0.67.
- **No-context aggression by kind of violence:** fights 0.70 [0.63, 0.76], shootings 0.69, robberies 0.65,
  arrests 0.48, riots 0.63 (7 videos).
- **By onset source:** official 0.74 and hivau 0.71 (context violent.score); official 0.68 and hivau 0.67
  (aggression).

No tag reaches the far-window level of the two scores. The shift is roughly the same at every lead, from 120 s to
0 s, and does not grow toward the onset.

### 5.3 The context teacher's signal is a static scene prior that is not specific to violence

Three controls on `violent.score` from the context set (record-level, time-matched AUROC against clean negatives):

| Window | Violent pre-onset (actual score) | Same, first record's score broadcast to the whole video | **Non-violent anomaly** pre-onset (actual score) |
|---|---|---|---|
| 60-120 s | 0.752 [0.68, 0.84] | 0.722 [0.64, 0.81] | **0.710 [0.65, 0.78]** |
| 30-60 s | 0.729 [0.65, 0.81] | 0.719 [0.64, 0.80] | **0.716 [0.66, 0.77]** |
| 10-30 s | 0.702 [0.66, 0.74] | 0.713 [0.67, 0.76] | **0.713 [0.67, 0.75]** |
| 0-10 s | 0.705 [0.67, 0.73] | 0.707 [0.67, 0.74] | **0.692 [0.66, 0.72]** |

- **The first record already carries the signal.** The first record is judged with no context, yet it scores 0.5
  or more in 43 % of UCF violent and 44 % of UCF non-violent anomaly videos, against 0 of 47 UCF normal videos.
  XD: violent 69 / 62 % (movie / YouTube), non-violent 49 / 53 %, normal 31 / 24 %. The context window carries
  the first verdict forward ("actions remain stable with no new violent indicators" at score 0.5), so the score
  becomes a per-video constant.
- **No within-video rise.** For the 52 videos with records both 60-120 s and 0-30 s before onset, the mean score
  changes by -0.04 [-0.10, 0.01] from far to near (negatives at pseudo-onsets: -0.02).
- **Not explained by earlier violence.** Removing the videos whose first UCA violence or weapon caption starts
  more than 5 s before onset (40 of the 90 UCF videos in the 30-60 s window) raises the AUROC (30-60 s: 0.76;
  UCF only: 0.78).
- **What it means.** The teacher separates "a video from the UCF anomaly collection" from "a UCF normal video"
  equally well before burglaries and before fights. Its 30 to 120 s signal is a **collection and scene prior**,
  not anticipation, and it will not transfer to a fixed deployment camera, whose scene does not change before a
  fight. VA30's `b_teacher` beats the position null at 0.54 FA/h (11.6 % vs 6.0 %), but the position null does
  not model a per-video scene prior. The tags behave the same way: "arguing/shouting" appears in 9.0 % of
  violent pre-onset records and 8.8 % of non-violent anomaly pre-onset records, against 0.3 to 0.4 % of clean
  records.

### 5.4 The no-context aggression score is specific to violence, but it is a plateau

The same controls on `aggression_level` from `vlm_labels_violence` (each frame judged alone):

| Window | Violent pre-onset | First record broadcast | Non-violent anomaly pre-onset |
|---|---|---|---|
| 60-120 s | 0.714 [0.65, 0.77] | 0.627 [0.52, 0.74] | **0.527 [0.45, 0.60]** |
| 30-60 s | 0.670 [0.62, 0.72] | 0.556 [0.44, 0.68] | **0.516 [0.45, 0.57]** |
| 10-30 s | 0.676 [0.63, 0.72] | 0.629 [0.55, 0.69] | **0.518 [0.46, 0.58]** |
| 0-10 s | 0.695 [0.66, 0.73] | 0.653 [0.60, 0.70] | **0.495 [0.44, 0.55]** |

(The no-context `violent.score` behaves the same way at a lower level: 0.62 / 0.53 at 30 to 60 s, violent vs
non-violent.)

- **Specific to violence.** It does not fire before non-violent anomalies, and the first frame explains only part
  of it. This is the one measured far-window signal that passes both controls.
- **A plateau, not a ramp.** Mean aggression per 10 s lead bin, with the number of videos:

| Lead (s) | 0-10 | 10-20 | 20-30 | 30-40 | 50-60 | 70-80 | 90-100 | 110-120 | 130-140 | clean (time-matched) | in violence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no-context aggression | 0.23 (550) | 0.21 (304) | 0.19 (185) | 0.17 (120) | 0.17 (62) | 0.22 (43) | 0.23 (29) | 0.21 (24) | 0.27 (15) | 0.068 | 0.431 |
| no-context "push / grab" share | 0.12 | 0.11 | 0.11 | 0.11 | 0.10 | 0.14 | 0.18 | 0.17 | 0.21 | 0.009 | 0.334 |
| context violent.score | 0.26 | 0.25 | 0.22 | 0.20 | 0.32 | 0.33 | 0.29 | 0.25 | 0.25 | 0.041 | 0.455 |

  The within-video change from 60-120 s to 0-30 s is +0.02 [-0.06, 0.10]: 35 % of positive videos rise by more
  than 0.05 and 29 % fall, against 7 % and 10 % of negatives. Positives are more volatile, not trending.
- **Concentrated in a subset.** 53 of 129 events (41 %) have a frame scored 0.8 or more on the no-context violent
  score 30 to 120 s before onset, against 20 % of random 90 s windows in clean videos. At record level the rates
  are 14.9 % vs 1.1 %. Under the context teacher it is 17 of 129 events (13 %) vs 0.5 % of negative windows. That
  is close to `b_teacher`'s Success@30 of 11.6 %, which suggests its successes are these events. Removing every
  event with such a frame anywhere more than 30 s before onset drops aggression to 0.56 at 30 to 60 s (74 videos
  left) and 0.61 at 60 to 120 s (18 left).
- **Mostly not violence in the human captions.** In UCF, only 10 of the 34 events with an early violent-looking
  frame (no-context teacher) have a UCA caption describing violence or a weapon in the same span. The rest are
  frames the teacher reads as aggressive (posture, contact) that the human captioner described as ordinary
  activity, plus teacher false positives.
- **Most specific tags 30 to 120 s ahead** (prevalence violent vs time-matched clean, with positive videos):
  - "grabbing" 8.0 vs 0.2 % (33 videos);
  - "physical contact" 6.5 vs 0.1 % (26);
  - "pushing" 6.2 vs 0.2 % (28);
  - "crowd dynamics" 8.7 vs 4.1 % (29);
  - "aggressive postures" 4.0 vs 0.1 % (21);
  - "fighting stances" 3.6 vs 0.1 % (17);
  - "raised fists" 1.2 vs 0.0 % (9).

  Low-level physical contact and aggressive posture, seen in a minority of events, is the visible "state" that
  precedes annotated violence by more than 30 s.

---

## 6. (d) YouTube hold-out: visible escalation in real-world clips

- **Source.** Reconciled records (`rec:` labels) from the two annotation workflow journals; the later run wins on a
  repeated video id. Snapshot 2026-10-14 21:24: **55 videos** (13 news reports, 8 bodycam breakdowns,
  7 compilations, 6 raw single incidents, 7 staged, 14 other). 25 videos have at least one onset, in 84 raw
  incident clips.
- **Fields.** Each clip (a shot or continuous take) has `content`, `camera`, `escalation_start_sec` and `onsets`.
  Visible escalation = first onset minus `escalation_start_sec`.
- **Exclusions.** Staged clips are excluded. "First contact seen" also excludes replays, onsets described as
  "already in progress" or "continuation", and onsets within 0.6 s of the cut.
- **Left-censoring.** If the escalation start lies at the cut, it is left-censored, so the value is a lower bound.
- **Visual only.** The annotators worked from frame contact sheets and burned-in subtitles, not the audio (one note
  reads "without audio this is uncertain"). Audible escalation may start earlier.

| Subset | Clips (videos) | Median visible escalation (s) | IQR | p90 | >= 10 s | >= 30 s | >= 60 s |
|---|---|---|---|---|---|---|---|
| All raw violent clips (non-staged) | 71 (21) | 0.8 | 0.05-6.8 | 21.6 | 21 % | 6 % | 1 % |
| **First contact on screen** | **30 (20)** | **8.8** | 3.9-20.9 | 33.4 | 47 % | **13 %** | 3 % |
| First contact and escalation start both on screen | 15 (11) | **18.6** | 6.4-26.5 | 45.3 | 67 % | **20 %** | 7 % |
| Civilian (punch, shove, grab, other) | 13 (10) | 7.7 | 2.8-21.6 | 49.8 | 46 % | 23 % | 8 % |
| Police use of force | 17 (11) | 9.5 | 5.0-17.2 | 26.3 | 47 % | 6 % | 0 % |
| Staged / re-enactment | 10 (4) | 1.5 | 0.9-1.9 | 8.7 | 10 % | 0 % | 0 % |

Other facts:

- **The footage before violence is short because clips are edited.** With the first contact seen, the median
  pre-onset footage in the clip is 12.0 s (27 % have at least 30 s). 41 % of raw violent clips open with violence
  already in progress or at the cut.
- **Escalation without violence is common.** Of 206 raw clips with a visible escalation start, 81 (39 %) contain
  violence. The 125 clips with escalation but no violence show a median of 10.3 s of escalation (16.2 s and 25 %
  at least 30 s when not left-censored). Some may continue past a cut.
- **Longest visible build-ups are civilian road-rage and argument shoves.** The three longest are 174.5 s of
  arguing before a shove, 53.4 s, and 35.6 s before a punch. Police force follows a
  shorter visible build-up (at most 33 s).
- **Cues in the notes.** Over the 207 raw clips with an escalation start (56 records), notes mention vehicles in
  55 %, verbal conflict (argue, yell, swear, threat) in 36 %, approach in 29 %, gestures in 23 %, face-off in 18 %
  and crowd in 16 % (`d2_youtube_note_cues.py`). The verbal share is a floor, since the annotators did not hear the
  audio.

These numbers will change as annotation completes (25 of 92 videos at the time of the briefing; 55 reconciled at
this snapshot). Re-run `d_youtube.py` (section 9).

---

## 7. Synthesis: what exists 30+ s ahead, and what it means for modality choice

### 7.1 Cues, ordered by how specific and how early they are in our data

| Cue | Lead time observed in our data | Specific to violence? | Who can see it | Evidence |
|---|---|---|---|---|
| Low-level physical contact (pushing, grabbing, "physical contact") before the annotated onset | plateau from 170 s to 0 s in about 10 to 20 % of pre-onset frames; in a minority of events | yes: 6 to 8 % vs 0.1 to 0.2 % of clean frames (no-context teacher); UCA push/grab 10 vs 1 to 3 % at 60-120 s (n=31), 4 to 5 vs 1 to 2 % at 10-30 s | pose and interaction, contact prompts | 5.4, 3.2 |
| Aggressive posture, fighting stance, raised fists | plateau over 0-170 s | yes (no-context aggression 0.67 to 0.71; non-violent anomalies 0.52) | pose, frame semantics (no context) | 5.4 |
| Face-off, pointing / gesturing, confrontation | mainly 10-30 s (UCA 6 vs 1 to 2 %); YouTube visible escalation median 9 to 19 s | moderately (base rate 1 to 2 % in normals) | pose and interaction | 3.2, 6 |
| Verbal argument / shouting | visual tags flat over 0-170 s; YouTube civilian build-ups up to 175 s | not as a visual tag (teacher "arguing" equal before burglaries); audio untested | audio prosody, speech content | 3.2, 5.3, 6 |
| Weapon displayed, hand in pocket, face covered | 30-120 s for robberies (weapon 19 vs 1 %, conceal 10 vs 1 %) | specific to robbery and shooting; partly the crime already starting | object detection, frame prompts | 3.4 |
| Police presence, following / pursuit | 30-120 s for arrests (19 vs 0 %, 13 vs 2 %) | specific to arrests | object and scene | 3.4 |
| People arriving or entering | 30-60 s (23 vs 11 %); shootings 34 vs 10 % | weak (non-violent anomalies show it too) | tracking (new tracks), scene | 3.2, 3.4 |
| Crowd gathering / group | 60-120 s 16 vs 6 % (n=31, not significant); in violence 8 vs 3 % | weak | pose count, crowd density | 3.2 |
| Running / rushing | 0-30 s (7 vs 1 %), mostly in violence (14 vs 1 %) | yes, but late | motion / flow | 3.2 |
| Scene type (counters, homes, care settings, no traffic) | static | **no** (identical before non-violent anomalies) | any frame embedding | 3.3, 5.3 |
| Global "violent-looking video" score with context | static, constant per video | **no** (0.71 to 0.72 before non-violent anomalies) | VLM, CLIP-like embeddings | 5.3 |

### 7.2 Modality ranking supported by this evidence

1. **Person interaction and pose: proximity, contact, posture, gestures.** This is the most specific cue family in
   both the human captions and the teacher, and the only one that shows a far-window state (contact, aggressive
   posture) that does not also fire before non-violent anomalies. Typical lead is 10 to 30 s; it reaches 30 to
   120 s in the minority of events with sustained low-level contact. Extractor: the built YOLO11 pose and
   interaction stream, plus contact and aggression prompts.
2. **Frame semantics, judged per frame without context, aimed at aggression and contact.** The no-context
   aggression score reaches time-matched AUROC 0.67 to 0.71 at 30 to 120 s and is specific to violence. Use SigLIP
   2 prompts such as "people pushing each other", "a man grabbing another man", "fighting stance", "raised fists"
   and "aggressive posture" rather than scene prompts. Never feed context-carrying VLM verdicts as features.
3. **Audio prosody and speech content.** It could not be measured here: UCF is mostly digital silence, UBI has no
   audio, and the YouTube annotations are visual. It is still the only plausible source of 30+ s lead in the target
   footage, where civilian build-ups are verbal and long (up to 175 s visible, and audible earlier). Rank it high,
   but treat it as a hypothesis until the AST and voice features pass the same controls on XD and YouTube.
4. **Objects and context: weapons, police, counters, arrivals.** These give 30 to 120 s of lead for robberies,
   arrests and shootings in CCTV. They are specific to the kind of violence, and part of the signal is the incident
   itself starting before the annotated onset. Useful as auxiliary cue heads.
5. **Global scene and clip embeddings** (Qwen embeddings, V-JEPA 2, SigLIP image embedding). These will reproduce
   the 0.7-level static collection prior and add little timing. Use them only as context, with per-video
   normalisation, and check them with the controls in 7.3.
6. **Motion and optical flow.** Not tested here. Captions put running and rushing in the last 30 s and during
   violence, so expect a short-lead cue.

### 7.3 Recommendations

1. **Add two controls to every feature screen and every model report** (cheap; the code pattern is in
   `c3_scene_prior.py`):
   - a "first-frame broadcast" version of the same score, as a scene-prior control;
   - the AUROC of non-violent anomaly pre-onset windows vs clean negatives (`other_anomaly` videos are already in
     the catalog and folds).

   A far-window signal that is just as high on the first frame, or before burglaries, is not anticipation. The
   context teacher fails both controls; the no-context aggression score passes both.
2. **Use other_anomaly pre-onset footage as hard negatives in training.** It shares the anomaly-collection look and
   forces the model off the scene prior.
3. **Model a state, not a ramp.** No cue rises between 120 s and 30 s; signals are plateaus present in some
   events. Prefer features such as "fraction of the last 60 to 120 s with contact or aggressive posture", "time
   since first aggressive frame" and "max aggression over the window" to slope features. Normalise per video or
   camera against an early baseline, so that a constant scene prior cancels out.
4. **Set realistic targets.** Report Success@10 and Success@20 next to Success@30. In our data the share of events
   with a visible aggressive state 30+ s ahead is about 13 % (context-teacher frames scored 0.8 or more) to 41 %
   (no-context frames, with a 20 % false-positive window rate). YouTube visible escalation reaches 30 s in 13 to
   20 % of clips. A Success@30 far above those levels at low FA/h should trigger a leakage audit.
5. **Measure audio before building fusion.**
   - Re-run sections 5.2 to 5.4 with the extracted AST, CLAP, voice-arousal and Whisper-toxicity features on XD
     (real audio for 2,554 YouTube and 2,194 movie clips) and on the YouTube hold-out, with the same time matching
     and the two controls.
   - In the YouTube annotation, add an audio escalation start (first raised voice, insult or threat) next to the
     visual one; the difference is the direct measure of audio's extra lead.
   - Also re-run the screen with the pose and SigLIP features once they are extracted.
6. **Fix the labels the captions expose.** Flag the about 19 UCF HIVAU-onset videos where UCA describes violence
   10 s or more before the onset (`out/a_first_violence_caption_all.csv`; the list is in section 3.5's output).
   Either move their onsets or mark their lead-ins as excluded, and complete the HIVAU spot-check.
7. **Treat riots and shootings differently from fights.** One in three eligible events is a riot or shooting, with
   no dyadic precursor in our data. Report per-group Success@30, and do not expect pose-interaction cues to
   anticipate shootings.
8. **Use UCA as weak supervision for cue heads on UCF.** Pre-onset captions give per-second labels for approach,
   face-off, push, grab, police, weapon and counter. They are cheap auxiliary targets for the pose and frame-semantic
   streams, without hand annotation.

---

## 8. Caveats

- **Small far windows.** The 60-120 s and 30-60 s windows hold 31 and 62 UCA events and 53 and 129 teacher videos;
  per-class far numbers rest on 20 to 56 events. CIs are wide; treat class-level 60-120 s cells as anecdotal.
- **UCA is human, hindsight-written and visual.** Captioners knew what happens later, which may raise the mention
  rate of the eventual actors (an upper bound on precursor visibility, if anything). Captions say nothing about
  sound. The lexicon is regex-based: it has false matches (a car "hitting" a flagpole) and misses (people "pressed
  down"). UCA covers only UCF CCTV, and its normals differ in scene type from its anomalies.
- **Teacher limits.** The teacher covers only XD test and UCF `videos/`, with 47 UCF and about 290 XD clean
  negatives. Its scores are coarse (0, 0.1, 0.5, 0.8, 0.9). "Violent-looking" frames are not ground truth.
- **Time matching** removes the time-since-start and source shortcuts, but not scene type within a source (which is
  what section 5.3 exposes).
- **HIVAU onsets** (most UCF train and all XD train events) are unvalidated and differ in style; teacher results are
  similar for official and HIVAU onsets (5.2).
- **YouTube** is a curated set of escalation videos (selection bias toward visible build-ups), still being
  annotated, frame-based, and full of edits and replays. The 30-clip "first contact seen" subset is small.
- **Nothing here measures audio, optical flow, V-JEPA or pose directly.** The modality ranking extrapolates from
  which cues exist, not from measured extractor performance.

---

## 9. Reproduction

The scripts live in `<tmp>/our_data_analysis/` (scratch; copy them into the repo if they need to be
kept). Outputs go to `<tmp>/our_data_analysis/out/`. Run them with the repo venv, single-threaded and
niced:

```bash
cd <tmp>/our_data_analysis
PY="nice -n 10 env OMP_NUM_THREADS=1 .venv/bin/python -I"
$PY e_leadin.py                      # (e) -> out/e_leadin.json, out/e_events.csv (seconds)
$PY a_uca.py all; $PY a_uca.py clean # (a) -> out/a_uca_{all,clean}.json, a_logodds_*.csv, a_first_violence_caption_*.csv
$PY b_hivau.py                       # (b) -> out/b_hivau.json, b_hivau_clips.csv
$PY b2_hivau_strike_lag.py           # (b) strike-word lag (section 4 numbers) -> out/b_hivau_strike_lag.json
$PY c_vlm.py vlm_labels 1000         # (c) per-window AUROC, trajectories, tags (~10 min) -> out/c_vlm_vlm_labels.json, c_records_*.parquet
$PY c_vlm.py vlm_labels_violence 1000
$PY c2_followup.py                   # context violent.score: earlier-violence exclusion, within-video change
$PY c2b_followup_generic.py vlm_labels_violence aggression
$PY c3_scene_prior.py                # first-record broadcast + non-violent anomaly control (context score)
$PY c3b_scene_prior_generic.py vlm_labels_violence aggression
$PY c3b_scene_prior_generic.py vlm_labels_violence score
$PY c4_tags_specificity.py           # teacher action tags: violent vs non-violent anomaly vs clean
$PY d_youtube.py                     # (d) -> out/d_youtube.json, d_youtube_clips.csv (re-run as annotation progresses)
$PY d2_youtube_note_cues.py          # (d) cue families in escalation-clip notes
```

| File | Role |
|---|---|
| `common.py` | catalog and events loaders, class groups |
| `textutil.py` | tokenizer, light stemmer, the 18-family cue lexicon, Monroe log-odds |
| `e_leadin.py` | section 2 |
| `a_uca.py` | section 3 (argument `all` or `clean`) |
| `b_hivau.py`, `b2_hivau_strike_lag.py` | section 4 |
| `c_vlm.py` | section 5.2 (argument: label set, bootstrap count; `loadonly` saves the record table only) |
| `c2_followup.py`, `c2b_followup_generic.py` | 5.3 and 5.4 exclusions and within-video change |
| `c3_scene_prior.py`, `c3b_scene_prior_generic.py` | 5.3 and 5.4 controls |
| `c4_tags_specificity.py` | 5.3 tag specificity |
| `d_youtube.py`, `d2_youtube_note_cues.py` | section 6 (journal paths are at the top of each script) |

Total CPU time is about 30 min on one core; no GPU.
