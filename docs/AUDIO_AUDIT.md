# Audio audit v1 (diegetic versus non-diegetic audio)

Written 2026-10-15. Artifact: `benchmark/audits/audio_type_v1.json` (sha256 in `audio_type_v1.json.sha256`,
`cd9c455d33486cb0f2e85caad1ec6a6819b3f2145a572de059dae95a3070eb02`). Purpose: the pre-registered screen
(`benchmark/screens/screen_set_v1.json`, `audio_gate`; `docs/MODALITY_PLAN.md` 6.1 step 4 and 6.3) runs the audio
and speech blocks (Phase B) only on diegetic audio. This audit decides how many events that leaves.

## 1. Scope and method

- Scope (346 videos): every eligible30 video with real audio in folds 0, 2, 3, 4 (256: 112 xd_movie, 109 xd_youtube,
  35 ucf; this contains all 86 S_audio first-segment candidates), plus the 90 curated YouTube hold-out videos that
  have an audio track (2 have none). Fold-1 videos were only counted (57 eligible30 audio videos), never read.
- Window: the clip part that Phase B scores, from `clear_start` to the onset of the first eligible30 event (S_audio
  candidates: 0 to first onset). Hold-out: 0 to the first onset, or the whole clip if it has none.
- Signals (all cached): `ast_events` (Music probability), `voice_affect` (speech fraction), `audio_dsp` (RMS level
  and spread), `speech_tox` transcripts (accepted Whisper windows, English or MT text). `clap_prompts` was
  computed but did not add separation. AST "Narration" almost never fires, so it is unused.
- Types: `diegetic` (sound recorded at the scene: participants, bystanders, crowd, ambient), `narration`
  (reporter, anchor, commentator, YouTuber voice-over), `music`, `mixed`, `silent_absent`. A `subtype` marks
  diegetic videos as `speech` (intelligible transcript) or `ambient` (no intelligible speech). XD movies carry
  `scripted_movie` and can never be diegetic (a produced mix of dialogue, score and effects).

Rules (thresholds in the json meta):
1. `silent_absent`: under 5 valid seconds or valid fraction under 0.4, or level below -50 dB without speech.
2. Little speech (under 15 words or VAD fraction under 0.15): `music` if Music > 0.5 on at least half the
   seconds, `mixed` between 0.25 and 0.5, else `diegetic` ambient (confidence 0.4 when AST hears speech-like sound
   that Whisper could not transcribe, for example commentary over a loud crowd).
3. Movies: `music` if Music >= 0.5 and VAD < 0.4, else `mixed`.
4. Speech present: `diegetic` if profanity >= 1 per 100 words or first-person "I" rate >= 8 per 100 words;
   `narration` if "I" rate <= 3.5 and profanity < 0.5 (confidence 0.8 with news or sports play-by-play
   vocabulary, or VAD >= 0.6 with steady loudness); otherwise `mixed`.
The text rule is the core idea: commentary and reporting are third-person and nearly free of "I" and swearing,
participants' speech is not.

## 2. Agreement

- Calibration set (24 videos, transcripts read by hand): 20 agree. Rules were tuned here, so this is optimistic.
- Fresh set (12 videos labelled after the rules were frozen): 9 agree. Misses: a sports clip with music called music,
  a re-enactment with narration called narration, a streamer talking called mixed.
- Frame checks (6): a talk-show studio and a local news reporter confirmed as narration; bike camera, city
  council meeting and restaurant CCTV confirmed as diegetic; one vlogger talking head was labelled diegetic by text
  but it is the vlogger's own monologue about another event, not scene audio.
- Weakest boundary: diegetic versus mixed (streamer talk, reaction videos, scripted clips on YouTube). Expect the
  true diegetic count to be somewhat below the number reported here.

## 3. Counts (346 videos)

| Source kind | n | diegetic | narration | music | mixed | silent/absent |
|---|---|---|---|---|---|---|
| xd_movie | 112 | 0 | 0 | 16 | 96 | 0 |
| xd_youtube | 109 | 23 | 60 | 6 | 19 | 1 |
| ucf | 35 | 11 | 2 | 3 | 1 | 18 |
| yt (curated hold-out) | 90 | 19 | 47 | 0 | 24 | 0 |
| total | 346 | 53 | 109 | 25 | 140 | 19 |

Hold-out videos use pseudo-labels (not ground truth) and are not part of S_audio; they are listed for completeness.

## 4. S_audio first-segment candidates (86 events)

| Source kind | n | diegetic | narration | music | mixed | silent/absent |
|---|---|---|---|---|---|---|
| xd_youtube | 62 | 17 | 26 | 6 | 12 | 1 |
| ucf | 24 | 8 | 1 | 2 | 1 | 12 |
| total | 86 | 25 | 27 | 8 | 13 | 13 |

**Diegetic S_audio events: 25 at best, of which 12 have intelligible speech (Phase B sample size) and 13 are
ambient only** (little or no transcribed speech; the speech block is empty for them). After removing the
streamer and talk-show look-alikes found in the spot checks (for example the vlogger above), the realistic usable
number is about 8 to 10 speech events. 27 of 86 candidates are narration or commentary (mostly sports play-by-play
and news segments) and 13 more are silent in the pre-onset window (UCF digital silence).

Examples (see the json for evidence per video):
- Narration: `yt/holdout_086` (talk-show studio), `xd/v=7zEFHKKBA0g__#1_label_B4-0-0` (local news reporter),
  `xd/v=Rve00ZElOO8__#1_label_B1-0-0` (hockey play-by-play), `yt/holdout_027` (true-crime voice-over).
- Diegetic: `yt/holdout_059` (bike camera, participants), `xd/v=NYnAnri6KTs__#1_label_B4-0-0` (city council meeting),
  `ucf/Robbery125_x264` (Chinese conversation), `ucf/Fighting050_x264` (Russian conversation).
- Music: `ucf/Robbery059_x264`, `xd/v=VaQpmWUfmn0__#1_label_B1-0-0` (soundtrack over a clip).
- Mixed: `yt/holdout_011` (911 audio plus commentary), all XD movies.

## 5. Recommendation

Phase B is not worth running as a confirmatory test: 12 speech events (25 counting ambient-only ones, 8 to 10
after cleanup) cannot support a bootstrap CI, a gate, or a modality-contribution estimate, and most of the
audio that exists is narration, commentary or soundtrack, not scene sound. Running it on that material would
measure whether a narrator announces a fight, which is not anticipation from the scene. If Phase B is run
anyway, label it descriptive only, restrict it to the 25 diegetic events, report the 12 with speech separately,
and do not use it to accept or reject audio. Better use of effort: collect more diegetic-audio material
(CCTV-Fights, bodycam and dashcam with raw audio) before testing audio again.

Reproduction: scratch code lived in `<tmp>/audio_audit/` (not kept); the rules above are the full
specification and `benchmark/audits/audio_type_v1.json` holds every per-video decision.
