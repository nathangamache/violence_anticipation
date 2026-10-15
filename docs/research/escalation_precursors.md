# Escalation precursors: what observable behaviour precedes interpersonal violence, how early, and how specific

Audience: future agents and developers choosing cues, modalities, labels and evaluation subsets for 30 s+
violence anticipation in this repo. Written 2026-10-14 from a web literature review (criminology,
micro-sociology, social psychology, policing, nightlife, psychiatry, speech and audio surveillance). It
complements [literature.md](literature.md) (ML anticipation papers), [backbones.md](backbones.md)
(which encoders to run) and [datasets.md](datasets.md). The question here is different: **which human
behaviours precede violence, by how many seconds, and how often they occur without violence following**.
Sibling reports written the same day: [modality_evidence.md](modality_evidence.md) (ML evidence per modality)
and [cue_extractors.md](cue_extractors.md) (which model measures each cue best). This file is the behavioural
science input to both (Investigation 1).

Reading guide:
- Section 0 is the summary. Section 9 ranks modalities for this project. Section 10 maps the evidence to the
  extractors already built in `anticipation/features/` ([../FEATURES.md](../FEATURES.md)) and lists the gaps.
  Section 11 is the recommended plan.
- Every numeric claim carries its source in brackets, for example [S6]. The reference list is in section 13.
- Each source is tagged by how it was read: **(full text)** = the PDF or HTML was read, **(abstract)** = the
  abstract or a publisher summary only, **(secondary)** = reported by another paper, digest or summary and not
  checked against the original. Re-check (abstract) and (secondary) numbers before quoting them in the capstone
  write-up.
- Evidence strength used in the tables: **strong** = several quantitative studies of real (not acted) incidents
  agree; **moderate** = one quantitative real-world study, or consistent qualitative video studies;
  **weak** = theory, lab or acted data, perception studies, practitioner lists, or secondary reports only.
- The web search budget for this run ran out near the end. A few questions remain open; they are listed in
  section 12 rather than filled in with guesses.

---

## 0. Summary

1. **Violence is rarely the first move. What precedes it, when anything does, is a visible conflict phase**
   that lasts tens of seconds to a few minutes: agitated talking and gesturing, pointing, standing close while
   gesturing, approach and retreat, insults, challenges and threats, then pushing or grabbing [S1, S4, S6, S10,
   S11, S24]. Public conflicts filmed by CCTV operators from the moment the operator noticed a potential for
   violence lasted **3.27 min on average (SD 4.20; n = 219)** [S5], and the first bystander intervention came
   **92 s after the conflict started on average** (n = 80) [S6]. Small-group violence hinges on thresholds of
   "a few minutes or less" [S3]. **This conflict phase is the only family of cues with a documented lead time
   above 30 s in ordinary interpersonal violence.**
2. **Most conflicts do not become violent, so conflict cues are sensitive but not specific.** Most pub
   quarrels "rarely get beyond the insulting and pushing stage" [S1]; fight-seekers usually fail to get a fight
   [S1, S26]; 90 % of UK drivers report road-rage incidents in a year but about 1 % report being assaulted [S44];
   only 2 of 588 recorded traffic stops involved force [S37]; in a video-monitored psychiatric ward, threatening
   or intrusive behaviour occurred in 10 % of random 5-min control periods without assault [S46]. **A detector
   of "conflict present" will raise many alarms on conflicts that end peacefully.** VA30's clean negatives are
   mostly calm footage, so they cannot measure this false-alarm source (section 11.4).
3. **A large share of violence has no visible escalation at all.** In 100 internet videos of one-on-one fights,
   **about half were brief unilateral assaults that began with high-severity strikes and showed little
   escalation; a quarter of those were sneak attacks** [S28]. In 127 Australian one-punch deaths, an argument
   preceded the punch in 76 (60 %) and 16 (13 %) were completely unprovoked [S45]. In video-recorded inpatient
   assaults, 60 % had threatening or intrusive antecedents within 5 min, so 40 % did not [S46]. **Any
   escalation-based model has a hard ceiling: only about half to 60 % of interpersonal violent events show a
   visible antecedent at all**, and only part of those show it 30 s or more ahead.
4. **The last steps before the first blow are short and the most specific.** A "working agreement" to fight
   (mutual challenges such as "touch me", chest-forward approach, grabbing, a push) typically comes seconds to
   tens of seconds before the blow [S11, S24, S26]. A push can trigger near-simultaneous group violence within
   2 s [S12]. These cues are high-precision but give little lead.
5. **Third parties are the strongest modulator, and they cut both ways.** At least one bystander intervened in
   about 9 of 10 CCTV-recorded public conflicts [S5] (85 % in 81 violent Copenhagen incidents [S13]).
   Intervention has about **19 times higher odds during or just after aggression** than without it [S6], so "being held
   back" or "someone stepping between" marks a conflict that is already hot. But once it happens, a physically
   forceful intervention **roughly halves the odds that the antagonist continues in the next 3 s** (OR about
   0.47 to 0.5), while verbal or gestural disapproval has no measurable effect [S7]. Aggressive third parties
   mirror and amplify aggression [S8]; third-party escalation occurred in 29 % of phone-filmed incidents [S10].
6. **Turning away is not automatically de-escalation.** Collins and Nassauer describe violence breaking out when
   one side suddenly shows weakness: turning one's back, falling, fleeing [S1]. In a filmed case the attack
   started when the reluctant party turned to walk away [S11]. Retreat lowers risk only when the other side
   lets it happen.
7. **Audio carries the verbal stage of escalation but is absent from most CCTV.** Raised voices precede physical
   assault in build-up scenarios, while visual cues dominate once the assault starts [S52]. Verbal aggression
   (shouting) is rare in real public spaces, which makes it specific: a deployed Groningen detector was tuned to
   flag under 10 s per month per microphone [S52, S51]. Speech content predicts later outcomes in police stops:
   the officer's first 45 words (about 27 s) separate stops that end in search, handcuffing or arrest from
   those that do not, with 70.8 % balanced accuracy [S37]. UCF-Crime is mostly silent and UBI-Fights has no
   audio, so these cues help only on XD, the curated YouTube set, bodycam, dashcam and phone footage.
8. **Human observers can anticipate fights from CCTV only modestly.** Naive and professional observers
   distinguished pre-crime CCTV scenes from matched scenes above chance, using gestures and body position, and
   experts were no better than novices [S33]; police officers were only marginally more accurate and much more
   prone to predict crime [S34]. Expect the same modest ceiling for models.
9. **For the modality question:** the evidence favours (1) long-window multi-person interaction geometry
   (pose and tracking), (2) semantic scene-state from vision-language models, (3) speech content and
   turn-taking where audio exists, (4) vocal prosody and shouting, then (5) context priors. Generic motion
   embeddings, facial expression, gaze and emotion classifiers have little or no support as 30 s leading cues
   (section 9).
10. **The most valuable next step is annotation, not another backbone:** mark, for each eligible event, the
   time the conflict became visible and code the precursor sequence. Its distribution is the empirical ceiling
   for Success@30 on this data, and the codes become auxiliary training targets (section 11).

---

## 1. Why this question matters for VA30

VA30 scores a model by Success@30 (first alarm at least 30 s before onset) at a fixed false-alarm rate per
hour ([../VA30.md](../VA30.md), [../BASELINE.md](../BASELINE.md) section 6). Two properties of a cue decide whether
it can help:

- **Lead time:** how long before the first blow the cue is observable. A cue that appears only in the last
  10 s cannot produce Success@30 no matter how specific it is.
- **Specificity:** how often the cue occurs in situations that end without violence. Lead time and specificity
  trade off. The longest-lead cues (setting, crowding, a conflict starting) are the least specific, and the most
  specific cues (a push, a raised fist, mutual "come on then") come last.

So the useful question is not "what predicts violence" but **"what is visible 30 to 120 s before onset, and
how much more often is it visible before violence than in comparable footage without violence"**. The control
results already in the repo fit this picture. The only above-chance control is a zero-shot VLM "violent" score
(Success@30 11.1 % at 0.66 FA/h versus 1.6 % for random alarms), and a position-only control reaches frame
AUROC 0.76 by exploiting where onsets sit in clips. A VLM rating of a frame mostly sees whether a confrontation
is visible, which matches the "conflict phase" signal described below.

---

## 2. Theory: how a confrontation becomes violent

### 2.1 Confrontational tension and fear (Collins)
Randall Collins' micro-sociology [S1, S2, S3] argues that face-to-face violence is hard. A threat of violence
produces confrontational tension and fear (CT/F) in both parties, and most confrontations abort into bluster,
standoffs or incompetent violence. Violence happens when one side finds a path around CT/F:
- **Emotional dominance**: one side establishes the rhythm and the other falls into passivity (robberies,
  police domination, bullying).
- **Attacking the weak**: a falling, fleeing, outnumbered or isolated target releases tension in a "forward
  panic": a three-part sequence of high tension on both sides, a sudden release when one side shows weakness,
  then an attack frenzy with piling-on [S1].
- **Audience support**: a supporting audience makes fights more likely and longer. Weenink found that the
  attackers' support group outnumbering the victims' supporters mattered even more than attackers outnumbering
  victims for prolonged "frenzied attacks" (159 interactions from Dutch court files) [S1, S27].
- **Time scales** [S3] (abstract): small-group violence turns on thresholds of "a few minutes or less", where
  boredom can make violence abort; organized crowds have tension-building "danger time-zones" of a few hours;
  riots run over days. A review of the follow-up book reports that face-to-face violent encounters usually last
  a few minutes, while organized crowd violence takes 1 to 3 hours to build [S3] (secondary).
- **Observable CT/F**: Collins notes CT/F is visible "in postures and rhythms as well as facial expressions",
  and in protests the dangerous moment is "a two-part sequence of heightened tension (visible in the shift from
  loose to tense body postures) followed by a sudden shift to emotional domination" [S1] (full text).

### 2.2 The interaction sequence (Luckenbill, Felson, Jackson-Jacobs, Weenink)
- **Luckenbill (1977)**, 70 homicide transactions [S24] (abstract and secondary): (1) an opening move the
  offender reads as an offense, (2) interpretation, (3) retaliation, a verbal or physical challenge, (4) a
  "working agreement" that violence will settle it, (5) the battle, (6) termination, shaped by the audience.
  Escalation happens because neither side will back down ("character contest").
- **Felson and Steadman (1983)** [S25] (secondary): homicides and assaults typically move from verbal conflict
  to threats and evasive action to physical attack; third parties and weapons shape the outcome.
- **Jackson-Jacobs (2013)**, 4 years of ethnography and 189 violent encounters [S26] (abstract): fights start
  when someone agrees to fight in response to a challenge to personal autonomy, then overcomes the ordinary
  fear of violence. Instigators need the opponent to appear as a voluntary combatant. Collins summarizes that
  in this group "bumping into someone, staring at them, all the usual marks of disrespect do not get automatic
  pushback" and "most of the time the belligerent moves remain abortive" [S1].
- **Weenink, Tuma and van Bruchem (2022)**, three fights analysed turn by turn from 131 phone clips [S11]
  (full text): antagonists build a "contested projection of violence", disputes over the conditions under
  which a fight will happen. The path is "not unidirectional" and has turning points where it could stop. Case
  timings: a subway fight about 30 s long in which one party stepped back at about 11 s ("the situation could
  have ended here") before the other re-engaged; a playground case in which collective violence began about
  4 min into the clip after the victim repeatedly said the fight was over and turned to leave. The recurring
  bodily pattern was **approach, touch, distance**, then approach, grab, distance, then the charge.

### 2.3 Ethology: assessment before escalation
Animal contest theory (the sequential assessment game of Enquist and Leimar, tested in fish, spiders, crabs
and deer) predicts that contests escalate step by step from cheap displays to costly fighting, that **evenly
matched opponents display longer and escalate further**, and that lopsided contests end early [S58]
(secondary). In 100 human street-fight videos, winners more often had a pre-fight size or vigour advantage, and
pre-fight verbal or gestural signals only "loosely" predicted who struck first [S28] (abstract). Implication:
a long, evenly matched face-off is a risk state; a clear mismatch either ends fast or turns into a unilateral
attack with little warning.

### 2.4 What this implies for anticipation
- The predictable unit is the **conflict state** (who is engaged with whom, for how long, how intensely), not
  the punch.
- Risk inside a conflict rises with: duration of mutual engagement without exit, repeated approach after
  retreat, physical contact (touch, grab, push), verbal challenges, a supporting audience, asymmetric
  numbers, and sudden weakness of one side.
- Risk falls with: sustained separation, physically forceful third-party intervention, several cooperating
  interveners, and boredom or dispersal of the audience.

---

## 3. Timing evidence

No study found reports the full distribution of "time from first visible conflict cue to first blow" in
real-world footage. The table collects the best proxies.

| Setting | Study | n | What was timed | Result | Read |
|---|---|---|---|---|---|
| Public space CCTV (Amsterdam, Lancaster, Cape Town), operator-recorded from first sign of potential violence | Philpot et al. 2020 [S5] | 219 conflicts (from 1,225 clips) | Conflict duration | **Mean 3.27 min, SD 4.20** | full text |
| Same program | Lindegaard et al. 2022 [S6] | 80 conflicts, 1,567 5-s intervals | Conflict start to first bystander intervention | **Mean 92 s**, "variation is large" | full text |
| Same program | Liebst et al. 2021 [S14] | 93 clips | Situation length | "on average around 3 minutes" | full text (postprint) |
| Amsterdam street CCTV | Ejbye-Ernst 2023 [S7] | 52 situations, 140 antagonists | Effect window of an intervention | Effect visible over 1 to 5 s segments; "takes a few seconds to take effect" | full text |
| Phone videos of group street fights | Bruggeman, Weenink, Mak [S12] | 42 videos, 59 groups | Clip length (left-truncated, includes the pre-fight phase) | Mean 101 s (SD 59; 30 s to about 5 min); a burst = at least half a group starting to fight within 2 s | full text |
| Phone videos | Weenink et al. 2022 [S11] | 3 cases | First antagonistic turn to violence | About 30 s (subway); about 4 min (playground) | full text |
| Police body-worn camera, one US agency | Willits & Makin 2018 [S38] | 95 use-of-force incidents | Contact to force (TtF) | Force at a mean of 5 min 18 s into the interaction; median TtF about **30 s for non-resisting and 120 s for resisting suspects** (resistance delays force); force lasted about 36 s | full text |
| Police BWC, Newark | Piza & Sytsma 2022 [S39] | use-of-force events | Contact to first force | Median **3.33 min when the officer explained the reason** for the detention, **0.83 min when not** | secondary (author summary) |
| Police traffic stops, BWC audio | Rho et al. 2023 [S37] | 577 stops | How early the outcome is predictable | First 45 officer words, about **27 s**, predict escalated outcome (search, handcuffing, arrest) | full text |
| Video-monitored psychiatric ward | Crowner et al. 2005 [S46] | 55 assaults | Antecedents in the 5 min before | Threatening or intrusive behaviour before **60 % of assaults vs 10 % of control periods** | abstract |
| Forensic inpatients | Klein Tuente et al. 2021 [S47] | 120 patients, 30 weeks | Weekly staff ratings | Two-thirds of physical incidents preceded by increased non-physical aggression (week scale) | abstract |
| Protest marches (US, Germany) | Nassauer 2019 [S30] | protests (12 violent vs 8 peaceful in a related paper, secondary) | Protest start to violence | Violence usually **1.5 to 2.5 h after the start**; then a seconds-scale trigger | abstract |
| School playground | Craig & Pepler 1998 via Hawkins et al. 2001 [S50] | 58 children | Bullying episodes | About one every 7 min, mean 38 s long | secondary |
| Psychiatric 24-h risk scales | Brøset Violence Checklist meta-analysis [S49] | many | Horizon 24 h | Pooled AUC 0.83 (range 0.69 to 0.98) | abstract |

Reading:
- In public conflicts that a camera operator noticed, the conflict lasts minutes, and physical aggression
  (which triggers intervention with odds ratios of 10 to 87 [S6]) typically appears within the first minute or
  two. A model that recognizes "a conflict is under way" can therefore plausibly gain 30 to 120 s on a large
  fraction of conflict-type events. This is an inference from the durations; no study reports the time to the
  first blow directly.
- Police encounters have their own clock: minutes from contact to force, and the earliest seconds of officer
  speech already carry predictive signal for escalated outcomes.
- Crowd and protest violence has an hours-long risk window and then a short trigger; the long window is a
  context prior, not an event predictor.
- Footage itself is left-truncated: phone filmers and CCTV operators start recording because they already sense
  trouble [S5, S12, S70]. The pre-onset footage in the datasets is therefore usually already inside the conflict
  phase, and the true conflict start is often not visible.

---

## 4. Specificity: how often precursors occur without violence

| Situation | Evidence | Implication | Read |
|---|---|---|---|
| Pub and nightlife quarrels | "The great majority of pub fights rarely get beyond the insulting and pushing stage"; third parties usually break them up [S1 on S4] | Insults and pushing are common and mostly do not lead to punches | full text (Collins) on abstract (Levine) |
| Operator-recorded public conflicts | Before the first intervention, medium-level aggression (punches, kicks, shoves) appeared in only 45 of 80 conflicts; high-level (weapons, attacking someone on the ground) in 7, low-level (pointing, light pushes) in 8; levels overlap within a conflict [S6] | Many conflicts that look like "potential violence" to trained operators stay non-physical or are interrupted first | full text |
| Fight-seeking young men | Fights rarely materialize despite looking for them; bumping and staring get no automatic pushback [S1, S26] | Staring, bumping, insults alone are weak cues | full text (Collins) |
| Bars, Toronto | 1,057 aggressive incidents in 1,334 observation visits to 118 large bars [S17]; 860 incidents over 503 nights in 87 bars, third parties involved in about one third [S18] | Aggression (much of it verbal or low-level) occurs most nights in large bars; physical violence is a minority | abstract / secondary |
| Bars, Sydney | Male drunkenness and rowdiness predicted non-physical aggression more strongly than physical violence [S22] | Rowdiness is a context cue, not an event cue | abstract |
| Road rage | 90 % of UK drivers had road-rage incidents in 12 months, 60 % lost their temper, about 1 % were physically assaulted; tailgating 62 %, headlight flashing 59 %, obscene gestures 48 %, verbal abuse 16 % [S44] | Horn, gestures and tailgating have very low specificity | full text (survey cited in report) |
| Police traffic stops | 2 of 588 BWC-recorded stops involved force; 15.3 % ended in search, handcuffing or arrest [S37] | Even "escalated" police stops are mostly non-violent | full text |
| Psychiatric ward | Threatening or intrusive behaviour in 10 % of control periods [S46]; Brøset checklist PPV 4.9 % at sensitivity 26 % in a forensic disability unit [S49] | Precursor checklists have good NPV but low PPV | abstract |
| Protests | Protests announcing violence can stay peaceful when no opportune moment arises [S1] | Context and intent are not enough | full text (Collins) |

The base rate is the crux. Even a cue with a likelihood ratio of 6 (60 % vs 10 %, [S46]) gives a low precision
when violent onsets are rare in the footage. VA30 negatives are mostly calm CCTV, movie or YouTube footage
without any confrontation, which makes conflict cues look far more specific on VA30 than they will be in a bar or
on a police bodycam (section 11.4).

---

## 5. Violence without visible escalation (the anticipation ceiling)

| Evidence | Share with no or little visible escalation | Read |
|---|---|---|
| 100 dyadic internet fight videos [S28] | About 50 % brief unilateral assaults starting with high-severity strikes, little escalation, fewer bystander interventions; a quarter of these were sneak attacks | abstract |
| 127 one-punch deaths, Australia 2000 to 2016 [S45] | Argument preceded the punch in 76 (60 %); 16 (13 %) completely unprovoked; the rest unclear | full text |
| 55 video-recorded inpatient assaults [S46] | 40 % without threatening or intrusive antecedent in the prior 5 min | abstract |
| Typology of 500 police-reported public assaults, Copenhagen [S65] | Five event types: night-time economy, excessively violent, revenge among acquaintances, weapon use with severe injury, disrespect encounters in everyday contexts | abstract |
| Weenink's "performing badness" [S1] | Attackers pick weak targets encountered by chance, provoke a slight, and attack; little audience support | full text (Collins) |
| Bursts in group fights [S12] | Group violence often starts within 2 s once agitation passes a threshold (13 of 35 small groups, 11 of 24 larger groups showed bursts) | full text |

Rough ceiling for this project (inference, to be replaced by the annotation in section 11.1): if about half of
interpersonal violent events have a visible antecedent conflict, and some fraction of those (unknown; the
durations in section 3 suggest a majority) have it in frame at least 30 s before onset, then **even a perfect
conflict detector would reach Success@30 on perhaps 25 to 50 % of interpersonal events**, at a false-alarm rate
set by how often non-violent conflicts appear in the negative footage. Classes differ: riots and arrests have
long visible build-ups; shootings, ambushes and sneak attacks mostly do not.

---

## 6. Cue catalogue

Columns: **Lead** = when the cue is typically visible before the first blow; **Specificity** = how often it
occurs without violence following; **Obs.** = observability in fixed CCTV (C, usually no audio, elevated angle,
low resolution), bodycam (B, egocentric, audio, heavy motion), dashcam (D, in-cabin audio, other vehicle in view)
and phone (P, filmer-framed, audio). Ratings: H = usually observable, M = sometimes, L = rarely, N = not.

### 6.1 Conflict-state and interaction geometry (pose, tracking, VLM)

| Cue | Lead | Specificity | Obs. C/B/D/P | Evidence | Key sources |
|---|---|---|---|---|---|
| **Sustained dyadic engagement**: two people (or groups) standing close, oriented to each other, for an extended time; "standing close to each other while gesturing aggressively" | 30 s to minutes (conflicts last 3.3 min on average) | Low to moderate: many such conflicts never turn physical | H/H/M/H | strong (operator recording criterion and coding in several CCTV studies) | [S5, S6, S7] |
| **Agitated gesturing**, pointing an index finger at someone, raised arms | 10 s to minutes | Low to moderate | M/H/M/H | moderate (used to define conflicts; "aggressive gesturing" coded as medium aggression in [S8]) | [S6, S8, S33] |
| **Pacing**, "walking back and forth in apparent frustration" | 30 s to minutes | Low | H/M/L/M | moderate (operator criterion [S6]; STAMP "pacing" in emergency departments [S48]) | [S6, S48] |
| **Approach toward another person**, closing distance, chest forward | Seconds to tens of seconds | Moderate | H/H/H/H | moderate (qualitative phone-video analysis; robbery dominance postures) | [S11, S31] |
| **Approach, touch, retreat cycles**, repeated re-engagement after stepping back | Tens of seconds to minutes | Moderate to high once repeated | M/M/M/H | moderate (qualitative, 3 cases) | [S11] |
| **Physical contact below violence**: chest-to-chest, flat-hand push, grabbing an arm or clothes | Seconds to tens of seconds | Moderate (pubs: most stop at pushing [S1]) to high | M/H/M/H | strong (coded in all CCTV studies; a push can start a burst within 2 s [S12]) | [S1, S6, S8, S12] |
| **Evenly matched face-off held for a long time** | Tens of seconds | Moderate | M/M/L/H | weak (ethology [S58]; loosely supported in human fights [S28]) | [S28, S58] |
| **One side suddenly weak**: falls, turns back, flees, is isolated or outnumbered | 0 to 5 s (trigger) | High inside a hot conflict | M/H/L/H | moderate (forward panic; protest and robbery video studies) | [S1, S11] |
| **Group size asymmetry and support groups** around the antagonists | Tens of seconds | Moderate | M/M/L/H | moderate (frenzied attacks; school fights where the winning side had an audience) | [S1, S27] |
| **Collective burst readiness**: groups in close face-to-face contact exchanging insults and threats | Seconds | High | M/M/L/H | moderate (42 videos, formal model) | [S12] |

### 6.2 Third parties and the audience

| Cue | Lead | Specificity / direction | Obs. C/B/D/P | Evidence | Key sources |
|---|---|---|---|---|---|
| **Audience gathering**, onlookers turning toward one spot, a line that curves into a **circle** around the antagonists | Tens of seconds to minutes | Marks a conflict others see as serious; circles came before intervention and doubled the number of interveners (IRR 2.15; circles in 27.5 % of 131 phone clips, lines in 52.7 %) | M/M/L/H | moderate | [S10, S35] |
| **Bystanders filming** with raised phones | Tens of seconds | Raises risk (filmers expect a fight); also the reason phone footage exists at all, so it is a provenance shortcut in phone data | M/M/L/H (filmer is usually off-camera) | weak to moderate | [S11, S12, S70] |
| **Cheering, encouraging audience** | Seconds to tens of seconds | Raises risk and severity | L/M/L/H | moderate (qualitative; court files) | [S1, S11, S27] |
| **Third party restrains, blocks or steps between** ("being held back") | Comes after aggression has started (intervention odds x19 after aggression) | Marks a hot conflict; then lowers continuation odds about 2x if physically forceful | H/H/M/H | strong | [S6, S7, S9] |
| **Calming gestures or touches only**, verbal disapproval | During the conflict | No measurable effect on continuation | M/H/M/H | moderate | [S7] |
| **Several cooperating interveners** (3 or more) | During the conflict | Lowers risk; conciliation by multiple third parties is more successful than by one | M/M/L/H | moderate | [S4, S10, S16] |
| **Aggressive third parties**, intoxicated third parties joining in | During the conflict | Raises risk; third parties mirror the preceding aggression level (OR 7.57 for high after high) | M/M/L/H | moderate | [S8, S18] |
| **Staff or bouncers forcibly removing someone** | Seconds to minutes | Perceived by patrons as the last event before violence outside the venue | M/L/N/M | weak (perception study) | [S23] |

### 6.3 Voice, prosody and non-speech vocal events (audio)

| Cue | Lead | Specificity | Obs. C/B/D/P | Evidence | Key sources |
|---|---|---|---|---|---|
| **Raised voices, shouting** | 10 s to minutes; voices are raised before physical assault in build-ups | High in ordinary public space (real verbal aggression is rare: target under 10 s per month per microphone in a Groningen deployment); low at parties, sports, music | N/H/H/H | moderate (acted train-station scenarios; deployed system claims) | [S51, S52] |
| **Rising pitch (F0) and intensity** | Tens of seconds | Moderate | N/H/H/H | moderate (role-play corpora, 3-level escalation UAR up to 63.8 %, chance 33.3 %) | [S53] |
| **Loudness rising over 30 s** | 30 s | Unknown | N/H/H/H | weak (repo sanity check: +2 to 3 dB from 30 s before onset in 81 % of 21 XD clips, movie audio) | [../FEATURES.md] |
| **Overlapping speech, interruptions** | Tens of seconds | Moderate (marks conflict, not violence) | N/M/M/H | weak to moderate (TV debates: overlap ratio is a strong conflict cue; detecting *changes* in conflict is near chance, 45.3 % vs 33 %) | [S55, S56] |
| **Screams, crowd roar, cheering** | 0 to 10 s (mostly at or after onset) | High for violence now | N/H/M/H | moderate (repo sanity: shout, yell, crowd rise only in the last 10 s) | [../FEATURES.md] |
| **Vocal anger or "emotion" classifier output** | Unclear | Low | N/M/M/M | weak (emotion2vec anger flat before onset in repo data) | [../FEATURES.md] |

### 6.4 Speech content (ASR and text)

| Cue | Lead | Specificity | Obs. C/B/D/P | Evidence | Key sources |
|---|---|---|---|---|---|
| **Insults, identity attacks** ("opening move", retaliation) | Tens of seconds to minutes | Low to moderate | N/H/H/H | moderate (homicide and assault sequence studies) | [S24, S25] |
| **Threats** ("I will cut you"), **challenges** ("touch me", "come on", "what are you gonna do"), countdowns | Seconds to tens of seconds (the "working agreement") | High | N/H/H/H | moderate (sequence studies; phone-video conversation analysis) | [S11, S24, S26] |
| **Commands at the start of a police encounter, no reason given** | The first ~27 s predict escalated outcomes minutes later | Moderate (outcome is search or arrest, not force) | N/H/L/M | strong for that outcome (577 stops; OR 1.29 per order, 0.48 for giving a reason; 70.8 % balanced accuracy) | [S37] |
| **Officer explains the reason for the detention** | Minutes | De-escalating: median time to force 3.33 vs 0.83 min | N/H/N/M | moderate (secondary) | [S39] |
| **Non-compliance** (refusing commands, walking away, hands hidden) | Seconds to minutes | Moderate; non-compliance preceded 95 % of officer shootings in a practitioner review | N/H/L/M | weak (secondary, practitioner) | [S68] |
| **De-escalating talk** ("calm down", "it's not worth it", apologies) | During the conflict | Lowers risk when delivered calmly; screamed appeals do not deter | N/H/M/H | weak (qualitative) | [S1] |
| **Conversation derailment markers** (text analog) | A few turns ahead | Moderate | n/a | moderate (online text: early politeness and directness predict later personal attacks) | [S57] |

### 6.5 Context and scene priors (long lead, low temporal specificity)

| Cue | Lead | Specificity | Obs. C/B/D/P | Evidence | Key sources |
|---|---|---|---|---|---|
| **Night-time economy**: late night, bar entrance, dance floor, near the serving bar, groups of male strangers, intoxication, crowding, boredom, aggressive staff | Hours | Low (base-rate multiplier) | H/M/L/H | strong (Toronto 1,057 incidents: dance floor 20.0 %, serving bar 15.7 %, tables 13.1 %; Sydney observation) | [S17, S19, S20, S22] |
| **Protest with police lines**: spatial incursions across agreed lines, police mismanagement, property damage, escalation signs, communication breakdown | 1.5 to 2.5 h after start, then seconds-scale trigger | Low to moderate | M/M/N/H | moderate | [S1, S30] |
| **Police contact**: subject resistance, intoxication, weapon possession | Minutes | Moderate; weapon possession shortens time to the highest force | N/H/M/M | moderate | [S38, S39, S42] |
| **Road incident**: tailgating, horn, gestures, cut-off, brake-check, vehicles stopping, a **driver getting out and walking toward the other car** | Seconds (exit and approach) to minutes (traffic dispute) | Very low for horn and gestures; higher once a driver exits and approaches | N/N/H/M | weak to moderate (10,037 violent incidents compiled from news and police; survey base rates) | [S44] |
| **Store robbery**: clerk away from the counter when the offender enters, script breaks (clerk resists or acts unexpectedly) | Seconds to a minute | Moderate for injury given robbery | H/N/N/L | moderate (about 200 robbery videos; 20 and 23 robbery videos qualitatively) | [S29, S31, S32] |
| **Prior violence in the same scene**, antagonists still present after separation | Minutes | High for further violence (violence comes in bursts; each prior violent act raises the odds of high-level third-party responses, OR 1.43) | H/H/M/H | moderate | [S1, S8, S12] |

### 6.6 Individual-level cues with weak support at video resolution

| Cue | Status | Notes |
|---|---|---|
| Staring, "hard stare", eye contact | weak | Stares push people to flee in field experiments, but effects depend on power and context [S59]; in fight-seeking groups staring does not get automatic pushback [S1]. Unobservable on CCTV. |
| Anger face, clenched jaw, clenched fists, "blading" | weak | Practitioner pre-attack lists and perception studies only; no peer-reviewed predictive validation found [S68]. The same cues also signal fear. |
| Personal-space intrusion sensitivity | weak | Violent prisoners have larger "body-buffer zones" (Kinzel 1970 and replications) [S60]; individual trait, not a situational cue. |
| Anxiety, mumbling, pacing, staring (STAMP); irritability, boisterousness, threats (Brøset) | moderate in clinical settings | Validated as 24-h or shift-level risk scales [S48, S49], not as seconds-scale video cues. Pacing and boisterousness are the parts observable in video. |
| Heart rate and arousal physiology | moderate (wearables, about 1 min lead) | Not observable in surveillance video; see [literature.md](literature.md) section 2.8. |

### 6.7 Domestic violence
Laboratory arguments of couples with a violent husband show more anger, belligerence and provocative anger
codes [S62], and stage models describe control attempts, challenges, verbal aggression and threats before
physical violence [S63]. Day-level diaries link IPV to within-person spikes in relationship tension [S64].
No timing data at the seconds-to-minutes scale were found, and the setting (indoors, private) is barely
represented in the target footage (doorbell and phone video at best). **Deprioritize** for this project.

### 6.8 Schools and youth
Peers were present in 88 % of observed playground bullying episodes and intervened in 19 %; 57 % of
interventions stopped the bullying [S50] (abstract). Collins reports from a three-year school ethnography that
small fights were usually "attacking the weak" with a supportive same-group audience, and that evenly matched
groups usually ended in standoffs [S1]. Phone videos of school fights share the left-truncation problem of all
phone footage.

---

## 7. De-escalation cues (should lower the predicted hazard)

| Cue | Effect | Caveat | Sources |
|---|---|---|---|
| Physically forceful third-party intervention (holding back, blocking, hauling off) | Odds of continued conflict behaviour in the next 3 s roughly halved (OR about 0.47 to 0.5), robust for 1 to 5 s windows | It appears only after aggression has started, so it also signals a hot conflict | [S7, S6] |
| Several interveners, situational group formation (circle) | More interveners (3.58 vs 1.40), conciliation more successful | The circle forms around an already serious incident | [S4, S10] |
| Sustained separation, antagonists leaving in different directions, audience dispersing | Collins: boredom makes violence abort | Re-engagement after separation is common (approach after retreat) | [S1, S3, S11] |
| Calm, strong, face-on verbal de-escalation ("we are peaceful, what about you?") | Lowers risk in protest confrontations | Shouted or panicked appeals do not | [S1] |
| Officer explains the reason for the stop | Delays force; less escalated outcomes | Bodycam only | [S37, S39] |
| "Fronting" etiquette: hands visible, nonchalance, brief eye contact, no looking back | Avoids violence between armed rivals | Qualitative, gang-specific | [S1] |
| Retreat, turning away | Can de-escalate | **Can also trigger the attack** (forward panic; filmed cases) | [S1, S11] |

Modelling consequence: treat intervention and retreat as **state variables with learned, conditional effects**,
not as fixed negative evidence. A model that sees "someone being held back" should raise the near-term hazard
if pushing just happened and lower it after sustained separation.

---

## 8. Human anticipation as a reference ceiling

| Study | Design | Result | Read |
|---|---|---|---|
| Troscianko et al. 2004 [S33] | 100 UK CCTV scenes: 18 leading to fights or vandalism, 18 matched without crime, 64 neutral; 50 naive and 50 professional observers | Both groups distinguished crime sequences above chance using gestures and body position; experts were no better | abstract |
| Crundall & Eyre-Jackson 2017 [S34] | 10 Nottingham CCTV crime clips cut before the act, 10 controls; 30 police vs 30 civilians | Police only marginally more accurate, much more likely to predict crime regardless | abstract |
| Grant & Williams 2011 [S35] | 12 operators vs 12 novices, 24 scenes, half leading to antisocial acts, eye tracking | Cues used: body language (looking around), large groups, provocation or competition for resources, social associations from spacing | abstract / secondary |
| Howard et al. 2013 [S36] | 33 novices, 11 operators, one-minute clips from 10 city-centre cameras | Operators more consistent with each other, not more suspicious on average | full text (summary) |

Humans watching CCTV without sound can anticipate antisocial incidents only modestly, using the same
interaction-level cues listed in 6.1 and 6.2. Expect the same of models trained on CCTV.

---

## 9. Modality ranking for this project

Ranked by expected contribution to Success@30 on the target footage (CCTV, bodycam, dashcam, phone), weighing
lead time, specificity, observability and evidence strength.

| Rank | Modality | Why | Expected lead | Extractor |
|---|---|---|---|---|
| 1 | **Multi-person interaction geometry over long windows** (pose, tracking, group layout) | Every quantitative CCTV study of real conflicts defines and codes precursors this way (proximity while gesturing, approach, pacing, contact, interposition, audience formation), and all of them worked **without audio**. It is the only cue family observable on all four footage types | 30 s to 3 min for the conflict state; 2 to 10 s for contact and trigger cues | Existing `pose_interact` plus new long-window integrators (section 10.2) |
| 2 | **Semantic scene state from image-text and VLM models** | The zero-shot VLM score is the only control above chance in VA30; a VLM can recognize "two men arguing face to face", "being held back", "a crowd forming a circle" without hand-built geometry, and tolerates poor pose quality | Same as rank 1 | `siglip2_prompts` (revise prompts to section 6), VLM teacher every 4 to 8 s with a phase questionnaire |
| 3 | **Speech content and turn-taking** (where audio exists) | Challenges, threats and insults precede blows in sequence studies; the first ~27 s of officer speech predict escalated police outcomes; on bodycam and phone footage this is likely the best single modality | 10 s to minutes | `speech_tox` plus overlap and speaker-turn features, dialog acts, longer rolling windows |
| 4 | **Vocal prosody and shouting** | Voices are raised before assault in build-ups; shouting is rare in ordinary public space; loudness rises from about 30 s out in the repo's XD sanity check | 10 to 60 s | `audio_dsp`, `ast_events` (shout, yell, bellow), with 60 to 180 s slopes and duty cycles |
| 5 | **Context priors** (setting, time, crowd density, police presence, traffic stop, protest line) | Real base-rate multipliers with hours of lead, but no temporal specificity and the main provenance shortcut risk | Hours | `siglip2_prompts` context group, `pose_interact` counts; use as hazard multipliers under the VA30 shortcut controls |
| 6 | **Generic motion and video embeddings** (V-JEPA 2, optical flow) | Respond strongly at onset and weakly in the last 10 to 30 s; no evidence as a 30 s leading cue; useful for the detection head and as complementary signal | 0 to 10 s | `vjepa2_l`, `motion` |
| 7 | **Face, gaze, micro-expression, emotion classifiers** | Unobservable at CCTV resolution; low specificity; support only from perception studies and practitioner lists | Unknown | Do not build |
| - | Physiology, wearables | Leads by about 1 min in other settings but not observable in video | about 1 min | Not applicable |

Footage-specific weighting:
- **Fixed CCTV (UCF, UBI, store and street cameras):** ranks 1, 2, 5 carry almost everything; no audio.
- **Bodycam:** speech content (commands, reasons, threats, non-compliance) and prosody move up to rank 1 and 2;
  the subject's approach toward the camera, hands and distance replace dyadic geometry (the officer is a party).
- **Dashcam:** in-cabin audio (yelling, horn duration), vehicles stopping, a door opening, a person walking
  toward the camera car; horn and gestures alone are near useless (section 4).
- **Phone:** audio and framing are good, but the act of filming and the filmer's zoom and movement are strong
  precursors **that will not transfer to CCTV**; report phone footage separately (already advised in
  [README.md](README.md) section 4).

---

## 10. Evidence versus the extractors already built

### 10.1 What is already covered

| Evidence-backed cue | Existing columns ([../FEATURES.md](../FEATURES.md)) | Gap |
|---|---|---|
| Proximity, approach, face-off | `pose_interact`: `min_pair_dist_mean`, `n_close_pairs`, `max_pair_closing_vel`, `faceoff_frac`, `mutual_facing_closest`, slopes over 5, 15, 30 s | Window is 3 s with slopes to 30 s; the conflict phase lasts minutes |
| Gesturing, arm raising, reach | `arm_raise_rate`, `arm_toward_other_max`, `wrist_speed_mean` | No pointing detector; no persistence measure |
| Audience gathering | `n_persons_mean`, `crowd_spread`, `crowd_convergence`, `new_tracks` | No orientation convergence (how many people face the same point), no ring or circle measure |
| Escalation states (zero-shot) | `siglip2_prompts`: arguing face to face, pointing aggressively, chest to chest, being held back, someone separating two people, crowd circle, bystanders filming, driver getting out, police confronting an agitated man | Single frame at t; no temporal integration; prompt set chosen before this review |
| Shouting, crowd noise | `ast_events` (Shout, Yell, Bellow, Screaming, Crowd, Cheering), `clap_prompts` | 3 to 5 s windows |
| Prosody | `audio_dsp` loudness and F0 with slopes to 30 s, `rel_loud_db` | No speaker overlap, no shouting duty cycle over minutes |
| Speech content | `speech_tox`: toxicity, insult, threat, `kw_threat`, `kw_command`, `kw_challenge`, `kw_deescalate`, rolling max over 30 s | No dialog acts (order vs reason vs question), no speaker turns or overlap, 30 s rolling only |

### 10.2 Proposed additions (highest expected value first)
1. **Conflict-state integrators on top of `pose_raw`** (CPU only, no new detector run). Per tracked pair and
   per scene, over 60, 120 and 300 s causal windows: cumulative seconds a pair spent within about 1.5 body
   heights while mutually oriented; number of approach-retreat cycles (distance minima separated by retreats);
   time since the pair first engaged; longest uninterrupted engagement; pacing index (back-and-forth
   displacement with small net displacement); bystander orientation convergence (fraction of non-engaged persons
   facing the pair's midpoint) and ring circularity around it; interposition (a third person's torso on the
   segment between the pair); sustained third-party hand-to-torso or hand-to-arm contact (restraint); a party
   leaving the scene or turning their back while the other approaches; group size on each side; people moving
   fast toward the pair.
2. **Audio turn-taking:** overlapped-speech ratio and number of simultaneous speakers per window (for example
   pyannote overlapped-speech detection), shouting duty cycle (fraction of the last 60 and 180 s with Shout or
   Yell above threshold), loudness and F0 slopes over 60 to 180 s. Gate by `has_audio`.
3. **Speech dialog acts** from the Whisper transcripts: order or command, explanation or reason, question,
   threat, challenge or dare, insult, de-escalation, apology; rolling counts over 30, 120 and 300 s. A small
   instruction-tuned LLM or zero-shot classifier on the existing transcript sidecars is enough; the
   [S37] six-act scheme (greeting, reason, documentation, details, order, legitimacy) is a ready template for
   bodycam.
4. **Prompt set v2 for SigLIP and the VLM teacher** organized by the phases in section 11.2 (calm, tension,
   argument, face-off, contact, intervention, separation, violence), plus de-escalation states (people walking
   apart, someone calming another, handshake after an argument). Keep the context prompts as shortcut
   detectors.
5. **Re-escalation state:** time since the last detected violent burst and whether the former antagonists are
   still in view (VA30 already shows 239 of 551 eligible events follow earlier violence in the same clip;
   report them separately because the headline metric uses pre-violence ticks only).

---

## 11. Recommended plan

### 11.1 Annotate the precursor timeline (highest value)
For every eligible event in the curated YouTube hold-out and a stratified sample of about 150 VA30 eligible
events (all sources, interpersonal classes), record:
- `t_conflict`: the first visible sign of the conflict (agitated gesturing or talking, pointing, pacing,
  standing close while gesturing, a verbal exchange audible as hostile), or "before clip start";
- `t_contact`: the first non-violent physical contact (touch, grab, chest-to-chest, push);
- `t_onset`: the annotated violence onset (as in VA30);
- codes: sudden attack yes or no, audience present, circle formed, filming visible, third-party intervention
  before onset, retreat or turn-away before onset, verbal challenge audible, weapons visible.
Use the published CCTV coding schemes as templates: the operator criteria of [S6], the three aggression levels
of [S8], the intervention behaviours of [S7] and [S9], and the circle and line formations of [S10].
The distribution of `t_onset - t_conflict` is the **empirical ceiling for Success@30 on this data**; the share
of sudden attacks is the floor of misses. A suggested location is `benchmark/precursors/` (CSV plus a README),
created by whoever runs the annotation.

### 11.2 Model the conflict state, then the hazard within it
- Stage A: **P(conflict active at t)** with multi-minute lead, trained on the phase annotations or on VLM
  pseudo-labels of the phases (calm, tension, argument, face-off, contact, intervention, separation, violence).
- Stage B: **hazard of onset within H given the conflict state**, using duration of engagement, contact,
  challenges, audience, asymmetry, intervention and separation history.
- This matches the evidence: lead time comes from Stage A, specificity from Stage B. It also gives
  interpretable alarms ("argument with physical contact for 70 s, crowd forming") that a capstone reviewer can
  check.

### 11.3 Training details implied by the literature
- History length: at least 180 s, ideally 300 s, of causal context (conflicts last minutes; the current
  recipe uses 60 to 120 s).
- Auxiliary targets: phase labels, time since `t_conflict`, and "contact happened" are all cheaper to label
  than onsets and far more frequent.
- Do not hard-code third-party intervention or retreat as negative evidence (section 7).
- Keep context features but run them through the VA30 shortcut controls (B-scene0, source x time matched AUROC).

### 11.4 Evaluation additions
- Report Success@30 separately for events **with** at least 30 s of visible conflict before onset and for
  **sudden** events (from 11.1). A model can only be judged fairly on the first group.
- **Add hard negatives:** footage of non-violent confrontations (arguments that de-escalate, road disputes
  without contact, police stops without force, bar quarrels broken up). The literature says these are the
  dominant false-alarm source in deployment (section 4), and VA30's clean negatives contain almost none. Report
  FA/h on them as a separate line. Candidate sources: YouTube searches for heated arguments and de-escalation
  videos, and DeEscalWild's channel list for police encounters (it releases transcripts, not video) [S43].
- Per-class reporting: riots, arrests and fights have build-ups; shootings and ambush assaults mostly do not.

### 11.5 What not to spend effort on
- More generic backbones before the conflict-state features exist.
- Face, gaze, micro-expression or voice-emotion classifiers as leading cues.
- Tuning on movie clips: films stage and edit their build-ups (the repo's motion check shows shots hold longer
  just before onset), so movie precursors are cinematic grammar rather than behaviour.

---

## 12. Open questions (not resolved in this review)

- No study found reports the distribution of time from first conflict cue to first blow in real CCTV or phone
  footage. Levine et al. 2011 [S4] and the Copenhagen and Amsterdam CCTV programs hold the data; their papers
  report durations up to interventions, not to blows. Section 11.1 fills this gap for our data.
- The proportion of operator-recorded public conflicts that never become physical is not reported directly
  (the panel analyses are censored at the first intervention).
- Graham's Toronto data (1,057 incidents) and Homel and Clark's Sydney data report physical versus verbal
  shares that were not reachable here; the 2004 and 2006 papers [S20, S21] should contain them.
- The Felson and Steadman (1983) and Luckenbill (1977) stage frequencies, Troscianko et al. (2004) cue-level
  results and Potegal et al. (2025) pre-contact durations were read only as abstracts or secondary summaries.
- Road-rage escalation has no peer-reviewed sequence analysis with timing; an 81-clip YouTube and Bilibili
  road-rage dataset exists (ACM, 2026) but was not read [S71].
- Piza et al. (2023, Criminology) analyse police force over micro-time intervals in BWC video and would give
  bodycam timing; only its title was seen [S40].

---

## 13. References

Tags: (full text), (abstract), (secondary) as defined at the top.

- [S1] Collins, R. (2019). Preventing violence: Insights from micro-sociology. *Contemporary Sociology* 48(5), 487-494. https://www.asanet.org/wp-content/uploads/attach/journals/sept19csfeature.pdf (full text). Summarizes Levine et al. 2011, Jackson-Jacobs 2013, Liebst et al. 2018, Sanchez-Jankowski 2016, Nassauer 2016 and 2018, Mosselman et al. 2018, Weenink 2014 and 2015, Krupnick and Winship 2015.
- [S2] Collins, R. (2008). *Violence: A Micro-sociological Theory*. Princeton University Press; and Collins, R. (2009). The micro-sociology of violence. *British Journal of Sociology*. https://onlinelibrary.wiley.com/doi/full/10.1111/j.1468-4446.2009.01256.x (secondary)
- [S3] Collins, R. (2020). Theorizing the time-dynamics of violence. *Journal of Violence*. https://journals.sagepub.com/doi/abs/10.1177/2633002420907768 (abstract); review of *Explosive Conflict* (Malesevic) https://www.researchgate.net/publication/364956571_Review_of_Randall_Collins_Explosive_Conflict_Time-Dynamics_of_Violence (secondary)
- [S4] Levine, M., Taylor, P. J., & Best, R. (2011). Third parties, violence, and conflict resolution: The role of group size and collective action in the microregulation of violence. *Psychological Science* 22, 406-412. https://pubmed.ncbi.nlm.nih.gov/21303991/ (abstract; details via [S15])
- [S5] Philpot, R., Liebst, L. S., Levine, M., Bernasco, W., & Lindegaard, M. R. (2020). Would I be helped? Cross-national CCTV footage shows that intervention is the norm in public conflicts. *American Psychologist* 75(1), 66-75. https://eprints.lancs.ac.uk/id/eprint/134891/1/Would_I_be_Helped_Cross_National_CCTV_Footage_Shows_That_Intervention_Is_the_Norm_in_Public_Conflicts_Postprint.pdf (full text)
- [S6] Lindegaard, M. R., Liebst, L. S., Philpot, R., Levine, M., & Bernasco, W. (2022). Does danger level affect bystander intervention in real-life conflicts? Evidence from CCTV footage. *Social Psychological and Personality Science* 13(4). https://pure.uva.nl/ws/files/99842199/19485506211042683.pdf (full text)
- [S7] Ejbye-Ernst, P. (2023). Does third-party intervention matter? A video-based analysis of the effect of third-party intervention on the continuation of interpersonal conflict behaviour. *British Journal of Criminology* 63(1), 78-96. https://academic.oup.com/bjc/article/63/1/78/6527657 (full text)
- [S8] Ejbye-Ernst, P., Lindegaard, M. R., & Bernasco, W. (2022). Third parties mirror the aggression of the antagonists: A video-based analysis of third-party aggression in interpersonal conflicts. *Journal of Interpersonal Violence* 37(17-18). https://pmc.ncbi.nlm.nih.gov/articles/PMC9682270/ (full text, summarized)
- [S9] Ejbye-Ernst, P., Lindegaard, M. R., & Bernasco, W. (2022). How to stop a fight: A qualitative video analysis of how third-parties de-escalate real-life interpersonal conflicts in public. *Psychology of Violence* 12(2), 84-94. https://research.vu.nl/en/publications/how-to-stop-a-fight-a-qualitative-video-analysis-of-how-third-par/ (abstract: 48 Amsterdam CCTV conflicts; phases objection, separation, placation)
- [S10] Weenink, D., Dhattiwala, R., & van der Duin, D. (2022). Circles of peace: A video analysis of situational group formation and collective third-party intervention in violent incidents. *British Journal of Criminology* 62(1), 18-36. https://academic.oup.com/bjc/article/62/1/18/6299950 (full text, summarized)
- [S11] Weenink, D., Tuma, R., & van Bruchem, S. (2022). How to start a fight: A qualitative video analysis of the trajectories toward violence based on phone-camera recorded fights. *Human Studies*. https://pmc.ncbi.nlm.nih.gov/articles/PMC9576657/ (full text, summarized)
- [S12] Bruggeman, J., Weenink, D., & Mak, B. Intergroup violence in bursts. arXiv:2112.05088. https://arxiv.org/abs/2112.05088 (full text)
- [S13] Liebst, L. S., Philpot, R., Bernasco, W., et al. (2019). Social relations and presence of others predict bystander intervention: Evidence from violent incidents captured on CCTV. *Aggressive Behavior*. https://onlinelibrary.wiley.com/doi/full/10.1002/ab.21853 (secondary: 81 Copenhagen incidents, 764 bystanders, intervention in 85.0 %, 2.7 interveners on average)
- [S14] Liebst, L. S., Philpot, R., Levine, M., & Lindegaard, M. R. (2021). Cross-national CCTV footage shows low victimization risk for bystander interveners in public conflicts. *Psychology of Violence* 11(1), 11-18. https://curis.ku.dk/ws/portalfiles/portal/240634303/Postprint_Final.pdf (full text, partly read: 93 clips, de-escalating interveners victimized 3.6 %, non-involved 0.2 %)
- [S15] Philpot, R., Liebst, L. S., Moller, K. K., Lindegaard, M. R., & Levine, M. (2019). Capturing violence in the night-time economy: A review of established and emerging methodologies. *Aggression and Violent Behavior* 46, 56-65. http://www.diva-portal.org/smash/get/diva2:1401044/FULLTEXT01.pdf (full text)
- [S16] Discover Society (2019). The helpful bystander: Current evidence from CCTV-captured public conflicts. https://archive.discoversociety.org/2019/06/05/the-helpful-bystander-current-evidence-from-cctv-captured-public-conflicts/ (secondary)
- [S17] Graham, K., et al. (2011). Behavioural indicators of motives for barroom aggression: Implications for preventing bar violence. *Drug and Alcohol Review*. https://pmc.ncbi.nlm.nih.gov/articles/PMC3175098/ (full text, summarized; cites Wells and Graham 1999: third parties in 49 % of incidents, peacekeeping in 40 % of those)
- [S18] Parks, M. J., Osgood, D. W., Felson, R. B., Wells, S., & Graham, K. (2013). Third party involvement in barroom conflicts. *Aggressive Behavior* 39, 257-268. https://pubmed.ncbi.nlm.nih.gov/23494773/ ; digest https://www.bps.org.uk/research-digest/when-do-bystanders-intervene-barroom-brawls (abstract, secondary)
- [S19] Graham, K., et al. (2012). 'Hotspots' for aggression in licensed drinking venues. *Drug and Alcohol Review*. https://onlinelibrary.wiley.com/doi/10.1111/j.1465-3362.2011.00377.x (abstract)
- [S20] Graham, K., Bernards, S., Osgood, D. W., & Wells, S. (2006). Bad nights or bad bars? Multi-level analysis of environmental predictors of aggression in late-night large-capacity bars and clubs. *Addiction*. https://onlinelibrary.wiley.com/doi/10.1111/j.1360-0443.2006.01608.x (title only)
- [S21] Graham, K., et al. (2004). The effect of the Safer Bars programme on physical aggression in bars: Results of a randomized controlled trial. *Drug and Alcohol Review*. https://onlinelibrary.wiley.com/doi/10.1080/09595230410001645538 (abstract)
- [S22] Homel, R., & Clark, J. (1994). The prediction and prevention of violence in pubs and clubs. *Crime Prevention Studies* 3, 1-46. https://www.researchgate.net/publication/29462011_The_Prediction_and_Prevention_of_Violence_in_Pubs_and_Clubs (abstract)
- [S23] Taylor, O., Keatley, D. A., & Clarke, D. D. (2020). A behavior sequence analysis of perceptions of alcohol-related violence surrounding drinking establishments. *Journal of Interpersonal Violence* 35(9-10). https://pubmed.ncbi.nlm.nih.gov/29294697/ (abstract)
- [S24] Luckenbill, D. F. (1977). Criminal homicide as a situated transaction. *Social Problems* 25(2), 176-186. https://academic.oup.com/socpro/article-abstract/25/2/176/1714335 (abstract, secondary)
- [S25] Felson, R. B., & Steadman, H. J. (1983). Situational factors in disputes leading to criminal violence. *Criminology* 21(1), 59-74. doi:10.1111/j.1745-9125.1983.tb00251.x (secondary)
- [S26] Jackson-Jacobs, C. (2013). Constructing physical fights: An interactionist analysis of violence among affluent, suburban youth. *Qualitative Sociology* 36, 23-52. https://link.springer.com/article/10.1007/s11133-012-9244-2 (abstract)
- [S27] Weenink, D. (2014). Frenzied attacks: A micro-sociological analysis of the emotional dynamics of extreme youth violence. *British Journal of Sociology* 65(3), 411-433. https://www.researchgate.net/publication/266025402_Frenzied_Attacks_A_Micro-Sociological_Analysis_of_the_Emotional_Dynamics_of_Extreme_Youth_Violence (abstract)
- [S28] Potegal, M., et al. (2025). What dyadic internet street fight videos can and cannot tell us about the ethological, game theoretic, and sex-differentiated phenomenology of human physical aggression. *Aggressive Behavior* 51(1), e70017. https://pubmed.ncbi.nlm.nih.gov/39710846/ (abstract)
- [S29] Nassauer, A. (2018). How robberies succeed or fail: Analyzing crime caught on CCTV. *Journal of Research in Crime and Delinquency* 55(1), 125-154. https://journals.sagepub.com/doi/abs/10.1177/0022427817715754 (abstract)
- [S30] Nassauer, A. (2019). *Situational Breakdowns: Understanding Protest Violence and Other Surprising Outcomes*. Oxford University Press; chapter "How to keep protests peaceful". https://academic.oup.com/book/34941/chapter/298554743 (abstract)
- [S31] Mosselman, F., Weenink, D., & Lindegaard, M. R. (2018). Weapons, body postures, and the quest for dominance in robberies: A qualitative analysis of video footage. *Journal of Research in Crime and Delinquency* 55(1), 3-26. https://journals.sagepub.com/doi/full/10.1177/0022427817706525 (abstract)
- [S32] DeCelles, K. A., Kouchaki, M., & Halevy, N. (2022). Unexpected employee location is associated with injury during robberies. *PNAS*. https://pubmed.ncbi.nlm.nih.gov/36122220 (secondary: about 200 convenience-store robbery videos)
- [S33] Troscianko, T., Holmes, A., Stillman, J., Mirmehdi, M., Wright, D., & Wilson, A. (2004). What happens next? The predictability of natural behaviour viewed through CCTV cameras. *Perception* 33(1), 87-101. https://pubmed.ncbi.nlm.nih.gov/15035331/ (abstract)
- [S34] Crundall, D., & Eyre-Jackson, L. (2017). Predicting criminal incidents on the basis of non-verbal behaviour: The role of experience. *Security Journal* 30, 703-716. https://link.springer.com/article/10.1057/sj.2015.33 (abstract)
- [S35] Grant, D., & Williams, D. (2011). The importance of perceiving social contexts when predicting crime and antisocial behaviour in CCTV images. *Legal and Criminological Psychology* 16(2), 307-322. https://researchprofiles.herts.ac.uk/en/publications/the-importance-of-perceiving-social-contexts-when-predicting-crim/ (abstract, secondary)
- [S36] Howard, C. J., et al. (2013). Suspiciousness perception in dynamic scenes: A comparison of CCTV operators and novices. *Frontiers in Human Neuroscience* 7, 441. https://pmc.ncbi.nlm.nih.gov/articles/PMC3749488/ (full text, summarized)
- [S37] Rho, E. H., Harrington, M., Zhong, Y., Pryzant, R., Camp, N. P., Jurafsky, D., & Eberhardt, J. L. (2023). Escalated police stops of Black men are linguistically and psychologically distinct in their earliest moments. *PNAS* 120(23). https://sparq.stanford.edu/sites/g/files/sbiybj19021/files/media/file/rho-et-al.-2023-escalated-police-stops-of-black-men.pdf (full text)
- [S38] Willits, D. W., & Makin, D. A. (2018). Show me what happened: Analyzing use of force through analysis of body-worn camera footage. *Journal of Research in Crime and Delinquency* 55(1), 51-77. https://s3.wp.wsu.edu/uploads/sites/1517/2018/03/Willits-Makin-2017-Show-Me-What-Happened-Analyzing-Use-of-Force-through-Analysis-of-Body-Worn-1.pdf (full text)
- [S39] Piza, E. L., & Sytsma, V. A. The impact of suspect resistance, informational justice, and interpersonal justice on time until police use of physical force: A survival analysis. *Crime & Delinquency* 70 (2024; online 2022). https://ericpiza.net/2022/07/04/uof-survival/ (secondary, author summary)
- [S40] Piza, E. L., et al. (2023). Situational factors and police use of force across micro-time intervals: A video systematic social observation and panel regression analysis. *Criminology*. https://onlinelibrary.wiley.com/doi/abs/10.1111/1745-9125.12323 (title only)
- [S41] Terrill, W. (2003). Police use of force and suspect resistance: The micro process of the police-suspect encounter. *Police Quarterly* 6(1), 51-83. https://asu.elsevierpure.com/en/publications/police-use-of-force-and-suspect-resistance-the-micro-process-of-t (abstract: 3,544 encounters; most force and resistance at the low end of the continuum)
- [S42] Bolger, P. C. (2015). Just following orders: A meta-analysis of the correlates of American police officer use of force decisions. *American Journal of Criminal Justice* 40, 466-492. https://link.springer.com/article/10.1007/s12103-014-9278-y (secondary: situational factors, especially resistance, are the strongest correlates)
- [S43] DeEscalWild: A real-world benchmark for automated de-escalation training with SLMs. arXiv:2604.13075. https://arxiv.org/abs/2604.13075 (abstract: 1,500 police-civilian scenarios from 23 social-media channels; transcripts released)
- [S44] Mizell, L., et al. (1997). *Aggressive Driving: Three Studies*. AAA Foundation for Traffic Safety. https://rosap.ntl.bts.gov/view/dot/40587 (full text: 10,037 violent incidents 1990-1996 compiled from news and police reports; firearm 37 %, vehicle 35 %; UK survey base rates)
- [S45] Victorian Institute of Forensic Medicine (2019). Australian deaths involving coward's punches: Research update. https://www.vifm.org/wp-content/uploads/Cowards-Punch-Research-Update-2019.pdf (full text)
- [S46] Crowner, M. L., et al. (2005). Assailant and victim behaviors immediately preceding inpatient assault. *Psychiatric Quarterly* 76(3), 243-256. https://pubmed.ncbi.nlm.nih.gov/16080420/ (abstract)
- [S47] Klein Tuente, S., Bogaerts, S., & Veling, W. (2021). Mapping aggressive behavior of forensic psychiatric inpatients with self-report and structured staff-monitoring. *Psychiatry Research* 301. https://pubmed.ncbi.nlm.nih.gov/34000510/ (abstract)
- [S48] Luck, L., Jackson, D., & Usher, K. (2007). STAMP: Components of observable behaviour that indicate potential for patient violence in emergency departments. *Journal of Advanced Nursing* 59, 11-19. https://onlinelibrary.wiley.com/doi/10.1111/j.1365-2648.2007.04308.x (abstract)
- [S49] Hvidhjelm, J., et al. (2023). Short-term risk assessment in the long term: A scoping review and meta-analysis of the Broset Violence Checklist. *Journal of Psychiatric and Mental Health Nursing*. https://onlinelibrary.wiley.com/doi/10.1111/jpm.12905 (abstract); forensic disability validation https://pmc.ncbi.nlm.nih.gov/articles/PMC12051222/ (abstract)
- [S50] Hawkins, D. L., Pepler, D. J., & Craig, W. M. (2001). Naturalistic observations of peer interventions in bullying. *Social Development* 10, 512-527. https://eric.ed.gov/?id=EJ647735 (abstract; Craig and Pepler 1998 timing secondary)
- [S51] van Hengel, P. W. J., & Andringa, T. C. (2007). Verbal aggression detection in complex social environments. *IEEE AVSS*. https://www.researchgate.net/publication/4308142_Verbal_aggression_detection_in_complex_social_environments (abstract, secondary)
- [S52] Kooij, J. F. P., Liem, M. C., Krijnders, J. D., Andringa, T. C., & Gavrila, D. M. (2016). Multi-modal human aggression detection. *Computer Vision and Image Understanding* 144, 106-120. https://pure.rug.nl/ws/files/79156655/Multi_modal_human_aggression_detection.pdf (full text)
- [S53] Lefter, I., Baird, A., Stappen, L., & Schuller, B. W. (2022). A cross-corpus speech-based analysis of escalating negative interactions. *Frontiers in Computer Science* 4. https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2022.749804/full (abstract)
- [S54] Lefter, I., Rothkrantz, L. J. M., & Burghouts, G. J. (2013). A comparative study on automatic audio-visual fusion for aggression detection using meta-information. *Pattern Recognition Letters*. https://www.sciencedirect.com/science/article/abs/pii/S0167865513000068 (abstract, secondary: audio-only, video-only and multimodal labels often disagree)
- [S55] Kim, S., Valente, F., & Vinciarelli, A. (2013). Annotation and detection of conflict escalation in political debates. *Interspeech 2013*. https://www.isca-archive.org/interspeech_2013/kim13e_interspeech.pdf (full text: 792 consecutive 30-s clip pairs; 45.3 % vs 39.7 % unweighted accuracy, 3 classes)
- [S56] Speaker-overlap work on the SSPNet Conflict Corpus (Kim, Valente, Vinciarelli and others), e.g. "Detecting speech interruptions for automatic conflict detection". https://link.springer.com/chapter/10.1007/978-3-319-14081-0_18 (secondary)
- [S57] Zhang, J., Chang, J. P., Danescu-Niculescu-Mizil, C., et al. (2018). Conversations gone awry: Detecting early signs of conversational failure. *ACL*. https://aclanthology.org/P18-1125/ (abstract)
- [S58] Sequential assessment game (Enquist & Leimar 1983) and tests, e.g. Enquist et al. 1990, a test in the cichlid *Nannacara anomala*. https://www.sciencedirect.com/science/article/abs/pii/S0003347205806608 (secondary)
- [S59] Ellsworth, P. C., Carlsmith, J. M., & Henson, A. (1972). The stare as a stimulus to flight in human subjects; and later qualifications. Overview: https://link.springer.com/chapter/10.1007/978-1-4612-5106-4_2 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC5497934/ (secondary)
- [S60] Kinzel, A. F. (1970). Body-buffer zone in violent prisoners. *American Journal of Psychiatry* 127, 59-64; Curran, Blatchley & Hanlon (1978) https://journals.sagepub.com/doi/10.1177/009385487800500104 (secondary)
- [S61] Dael, N., Mortillaro, M., & Scherer, K. R. (2012). The Body Action and Posture coding system (BAP). https://www.unige.ch/cisa/gemep/bap-coding (abstract; a coding scheme for body movement)
- [S62] Jacobson, N. S., Gottman, J. M., et al. (1994). Affect, verbal content, and psychophysiology in the arguments of couples with a violent husband. *Journal of Consulting and Clinical Psychology* 62, 982-988. https://umimpact.umt.edu/en/publications/affect-verbal-content-and-psychophysiology-in-the-arguments-of-co/ (abstract)
- [S63] Winstok, Z. (2007, 2008), interactional escalation model of intimate partner violence, as described in https://link.springer.com/article/10.1007/s10896-026-01070-9 (secondary)
- [S64] An event-level examination of intimate partner violence perpetration (daily diaries, n = 150). https://pmc.ncbi.nlm.nih.gov/articles/PMC12393831/ (abstract)
- [S65] Friis, C. B., Moller, K., Ejbye-Ernst, P., Philpot, R., Heinskou, M. B., Lindegaard, M. R., & Liebst, L. S. (2025). From a single punch to weapon use: An event typology of public place violence. *European Journal of Criminology*. https://journals.sagepub.com/doi/10.1177/14773708251374764 (abstract)
- [S66] Bernasco, W., Hoeben, E. M., Koelma, D., Liebst, L. S., Thomas, J., Appelman, J., Snoek, C. G. M., & Lindegaard, M. R. (2023). Promise into practice: Application of computer vision in empirical research on social distancing. *Sociological Methods & Research* 52(3). https://journals.sagepub.com/doi/10.1177/00491241221099554 (abstract: computer-vision pedestrian detection and metric inter-person distance on 56,429 Amsterdam CCTV clips, validated against human coders; shows that this research group's coding variables are machine-measurable)
- [S67] Can we anticipate violence? Multimodal learning from pre-incident behavioral cues. arXiv:2609.40014. https://arxiv.org/html/2609.40014 (full text, summarized; 443 XD-Violence clips, about 4-s windows, no horizon analysis; face crops scored highest, which suggests provenance or identity shortcuts)
- [S68] Practitioner pre-attack indicator sources, e.g. https://www.policemag.com/training/article/15346362/pre-attack-indicators and Pinizzotto, Davis & Miller (2006), *Violent Encounters* https://www.ojp.gov/ncjrs/virtual-library/abstracts/violent-encounters-study-felonious-assaults-our-nations-law (weak; no peer-reviewed validation found)
- [S69] Goodwin, M. S., et al. (2019). Predicting aggression to others in youth with autism using a wearable biosensor. *Autism Research*. See [literature.md](literature.md) section 2.8.
- [S70] Weenink, D. (2023). Turning Violent, ERC Advanced Grant proposal, part B1. https://www.universiteitleiden.nl/binaries/content/assets/governance-and-global-affairs/isga/b1-turning-violent-weenink-1643.pdf (full text, partly read: "recorders often begin filming when they perceive something" is happening, so clips are left-truncated)
- [S71] Unveiling road rage dynamics: Recreating and modeling road rage in audiovisual and simulating environments based on real-world footage. ACM, 2026. https://dl.acm.org/doi/10.1145/3772318.3791220 (not read; 81 YouTube and Bilibili road-rage clips with trigger annotations)
