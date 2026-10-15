# Curated YouTube hold-out: annotation protocol and results

Audience: anyone evaluating on, or extending, the real-world hold-out set (92 curated YouTube videos in
`data/anticipation_curation/videos/`, catalog role `holdout`, video ids `yt/<youtube id>`).
Status 2026-10-15: all 92 videos annotated (`benchmark/youtube_holdout/annotations_v2.json`). V2 preserves the
76 v1 video records and adds the 16 previously listed in v1's `meta.missing`. Summary numbers:
`benchmark/youtube_holdout/summary_v2.json`. V1 remains unchanged.

## 1. Why

The old `candidates.json` onsets were LSTM pseudo-labels (every `violence_start` was at least 59 s, an artifact).
The set is the closest public match to the target footage (bodycam, phone, CCTV confrontations), but most videos
are not raw incident footage: news reports, compilations, bodycam "breakdown" commentary, staged scenes and
explainers. Anticipation needs continuous raw footage before the onset, so the annotation segments every video
into clips and labels onsets per clip.

## 2. Protocol

- Tools: `tools/contact_sheet.py` renders timestamped frame grids. Coarse sheets (one frame every 2 s, 30 tiles =
  60 s per sheet) for every video live in `data/annotation/youtube/sheets/<id>/coarse_*.jpg`; annotators render fine
  sheets (0.5 s) around candidate onsets and cuts.
- Two independent annotators (A, B) per video, blind to each other, then a reconciler who re-checks every
  disagreement on fine sheets (onset differences over 2 s, different onset counts, different video type).
- Clips tile the whole video at every hard cut or content switch. Clip `content`: incident_raw, studio_or_reporter,
  aftermath_or_interview, graphics_or_text, other. `camera`: cctv, dashcam, bodycam, phone_handheld, broadcast,
  staged_hidden_camera, other.
- Onset = first frame of physical aggression against a person: shove, grab, punch, slap, kick, strike with an
  object, firing at a person, throwing an object at a person, a vehicle used as a weapon, or police use of force.
  Verbal arguing, pointing and squaring up are escalation cues (`escalation_start_sec`), not onsets. Replays get
  their own clip with a note. Each onset has a type, a confidence in [0, 1] and a description.
- `video_type`: raw_single_incident, compilation, news_report, bodycam_breakdown_or_commentary,
  staged_or_reenactment, other.
- The annotators are AI models judging frames visually (Claude Opus 5.5 for 56 videos, Sonnet 5.5 for 20,
  GPT-6 Codex for the 16 v2 additions; see `annotator_model` per video). They see frames only, no audio.
  Treat these as model labels with a two-annotator
  check, not human ground truth; spot-check before citing per-video results.

## 3. Results (92 videos)

| Video type | Videos |
|---|---|
| news_report | 26 |
| other (explainers, music video, unrelated) | 21 |
| staged_or_reenactment | 12 |
| bodycam_breakdown_or_commentary | 13 |
| compilation | 10 |
| raw_single_incident | 10 |

- 199 onsets inside raw incident clips; median lead-in from the clip start is 3.25 s.
- **36 onsets in 14 videos have at least 30 s of continuous same-clip footage before them** (camera: phone 16,
  bodycam 7, CCTV 5, other 8). This is the usable real-world 30 s evaluation set.
- Time from the first visible confrontation cue (`escalation_start_sec`) to the onset: median 4.6 s, 75th
  percentile 17.1 s (161 onsets with a marked escalation). Visible escalation mostly precedes violence by seconds,
  which supports the conflict-state view in [MODALITY_PLAN.md](MODALITY_PLAN.md): a 30 s warning has to come from
  recognizing an ongoing conflict, not from a visible ramp.

The 16 v2 additions were reviewed by two independent, mutually blind Codex agents, then reconciled by a third
reviewer using 0.5 s sheets for disagreements. Each new record's `annotation_review` retains both passes,
the reconciliation decisions and sheet paths; newly rendered sheets are in `fine_A/`, `fine_B/` and `fine_R/`
under the existing video sheet directories. Cut/onset times are sampled estimates at 0.5 s resolution.

The additions contribute 16 labeled occurrences across three videos: one parking-lot police-force occurrence,
six arrest occurrences in edited bodycam footage, and nine CCTV shove/grab occurrences. These counts include
replays and continuations of the same events. Nine of these occurrences have `left_censored: true` because contact is
already underway when the clip becomes visible; their timestamps mark visible clip entry, not the original
beginning of aggression. None of the additions supplies a new 30 s lead-in. The other 13 added videos have no
visually established physical-aggression onset; narrated shootings, property damage and voluntary stunts alone
were excluded.

## 4. Using it

Evaluate a model on the 14 usable videos by scoring each incident clip as its own timeline (features computed on
the full video are causal, but history before a clip start belongs to another clip; prefer re-extracting features
on per-clip cuts). Report it separately from VA30 and never tune on it. For future extensions, follow section 2
exactly and write a new annotation version; keep v1 and v2 unchanged.
