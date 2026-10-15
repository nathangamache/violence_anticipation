# Amendment 04 to screen_set_v1: Phase B (audio and speech) becomes descriptive only

Date: 2026-10-15. Registration amended: `benchmark/screens/screen_set_v1.json`
(sha256 `8831e16e664c895271b9cd77e0bab68733ca85a47655708998ab51d5e625797b`).

**No Phase B block outcome had been read when this amendment was written.** Audio blocks were gated
(BLOCKED_AUDIO_GATE) and never scored.

## Why

The audio-type audit required by the registration's audio gate (`docs/AUDIO_AUDIT.md`,
`benchmark/audits/audio_type_v1.json`, sha256 `cd9c455d33486cb0f2e85caad1ec6a6819b3f2145a572de059dae95a3070eb02`)
found that of the 86 first-segment real-world audio events (S_audio), only 25 have diegetic audio and only 12 of
those contain intelligible speech; the rest is news narration or commentary (27), mixed (13), silent (13) or music
(8). All XD movie audio is scripted soundtrack. A gate on 12-25 events cannot produce informative confidence
intervals, and on the non-diegetic remainder an audio block would mostly measure whether a narrator announces a
fight.

## What changes

- Phase B blocks (`audio_dsp`, `ast_events`, `clap_prompts`, `speech_tox`, `voice_affect`, `e2v_emb`,
  `overlap_and_duty_cycles`, `dialogue_acts`) are never gated and never classified. Their class is recorded as
  INCONCLUSIVE (insufficient diegetic sample), as the registration anticipates for audio.
- If Phase B is run, it is descriptive only: M1 band and M1-near with CIs, restricted to the 25 diegetic S_audio
  events (speech blocks on the 12 speech events, reported separately), against clean negatives with diegetic or
  absent audio. No forward selection may use Phase B blocks.
- Audio is revisited only with new diegetic data (for example NTU CCTV-Fights or bodycam/dashcam sources with raw
  audio), under a new registration.
- Phases A and C are unchanged.
