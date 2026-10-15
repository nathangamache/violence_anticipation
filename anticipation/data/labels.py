"""Per-video tick labels on the shared 1 Hz clock (VA30 section 6.1). Used by trainers AND the evaluator.

Clock: ticks ``t = 1, 2, ..., floor(duration)`` seconds (``anticipation.features.io.tick_times``). A model's
score at tick ``t`` may only use media from ``[0, t]``.

Definitions (all in seconds; ``segments`` = sorted, merged ``[[s, e], ...]`` violent segments):

    next onset        t0(t) = first segment start s with s > t (inf if none)
    time_to_onset     tte(t) = t0(t) - t  (inf if none)
    in_violence       s <= t <= e for some segment
    aftermath         e < t <= e + 10 for some segment (and not in violence)
    masked            in_violence | aftermath  -> never used by anticipation metrics or losses
    y_H               1 if 0 < tte <= H, for H in (5, 10, 20, 30, 60)
    censored_H        no onset in (t, t + H] AND t + H > duration (we did not observe H seconds of future)
    valid_H           ~masked & ~censored_H  -> ticks usable for horizon H
    neg120            ~masked & tte > 120   -> "no onset within 120 s" (VA30 negatives for per-horizon AUROC)

Discrete hazard targets (bins ``(0,5], (5,10], (10,20], (20,30], (30,45], (45,60]``):

    hazard_target [T, 6] in {1, 0, -1}: 1 in the bin containing tte, 0 in every earlier bin, -1 = no target
    (later bins, censored bins, masked ticks). A tick with no onset within 60 s gets 0 in every bin whose upper
    edge is <= the remaining footage (duration - t) and -1 after that (right-censoring).
    Train with BCE on entries >= 0 for per-bin hazards h_j; P(onset within edge_j) = 1 - prod_{i<=j} (1 - h_i)
    (``hazard_to_cdf``).

Excluded spans (not violent onsets, never positives or negatives, never false alarms):

    other_segments    non-violent anomaly segments inside a violent clip (explosions / crashes in mixed-label XD
                      clips, from HIVAU's per-event judgement); excluded together with their 10 s aftermath
    mask_spans        clip time covered by violence (+ 10 s aftermath) of ANOTHER clip cut from the same XD source
                      film, or, for clean clips, also its 120 s pre-onset window (see catalog.source_masks);
                      ends a clear lead-in like a violent segment does
    soft_mask_spans   the pre-onset window of such violence inside a violent clip, outside the clip's own
                      pre-onset windows: excluded, but does not end a lead-in
    excluded          ticks in any of these; ``masked`` includes them

Eligible anticipation events (VA30 6.2), on the 1 Hz clock: onset s_k with at least 30 s (60 s) of observed,
clear footage before it AND at least one alarm start possible with a lead of 30 s (60 s):

    clear_start_0 = 0, clear_start_k = e_{k-1} + 10, raised to the end of any excluded span that starts before s_k
    first_tick_k  = max(EARLIEST_ALARM_TICK, ceil(clear_start_k))   earliest tick an alarm can start in the window
    eligible30    = s_k - clear_start_k >= 30 and s_k - first_tick_k >= 30

``EARLIEST_ALARM_TICK = 2`` because the default 2-of-3 smoothing needs two raw ticks (ticks 1 and 2) before an
alarm can be active, so a first onset in [30, 32) s could never be anticipated by 30 s. See ``event_table``.

``pre_violence`` marks ticks before the first annotated segment (violent or other) of the clip, i.e. before any
anomaly has been seen. The headline frame metrics of VA30 use only those ticks, so a detector that merely
remembers past violence earns nothing there (see docs/VA30.md section 5).
"""

from dataclasses import dataclass, field

import numpy as np

from anticipation.features.io import tick_times

HORIZONS = (5, 10, 20, 30, 60)
EARLIEST_ALARM_TICK = 2     # first tick at which the default 2-of-3 alarm can be active (ticks 1 and 2 raw)
HAZARD_EDGES = (0.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0)
AFTERMATH_SEC = 10.0
NEG_WINDOW_SEC = 120.0
ELIGIBLE_LEADS = (30.0, 60.0)


def normalize_segments(segments, duration=None):
    """Sort, clip to ``[0, duration]``, drop empty segments and merge overlapping or touching ones."""
    segs = []
    for s, e in segments or []:
        s, e = float(s), float(e)
        if e < s:
            s, e = e, s
        s = max(0.0, s)
        if duration is not None:
            if s >= duration:
                continue
            e = min(e, float(duration))
        segs.append([s, e])
    segs.sort()
    merged = []
    for s, e in segs:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def event_table(segments, aftermath_sec=AFTERMATH_SEC, blockers=None, earliest_tick=EARLIEST_ALARM_TICK):
    """One dict per (normalized) segment: onset, end, clear_start, leadin, first_tick, max_lead, eligible30/60.

    ``clear_start`` is where the clear lead-in before this onset begins: 0 for the first segment, the end of the
    previous segment's aftermath otherwise, raised to the end of any ``blockers`` span (excluded time, see
    ``blocker_spans``) that starts before the onset. ``leadin = onset - clear_start``. ``first_tick`` is the
    earliest tick at which an alarm can start inside the lead-in on the 1 Hz clock with k-of-n smoothing
    (``max(earliest_tick, ceil(clear_start))``) and ``max_lead = onset - first_tick`` the largest lead any alarm
    policy with that k can reach. An event is eligible for lead L if both ``leadin`` and ``max_lead`` are >= L.
    """
    rows = []
    prev_end = None
    blockers = [list(b) for b in (blockers or [])]
    for k, (s, e) in enumerate(segments):
        clear = 0.0 if prev_end is None else prev_end + aftermath_sec
        for bs, be in blockers:
            if bs < s:
                clear = max(clear, float(be))
        lead = s - clear
        first_tick = max(float(earliest_tick), float(np.ceil(clear - 1e-9)))
        max_lead = s - first_tick
        rows.append({"index": k, "onset": float(s), "end": float(e), "clear_start": float(clear),
                     "leadin": float(lead), "first_tick": first_tick, "max_lead": float(max_lead),
                     "eligible30": bool(lead >= ELIGIBLE_LEADS[0] and max_lead >= ELIGIBLE_LEADS[0]),
                     "eligible60": bool(lead >= ELIGIBLE_LEADS[1] and max_lead >= ELIGIBLE_LEADS[1])})
        prev_end = e
    return rows


def blocker_spans(other_segments=None, mask_spans=None, aftermath_sec=AFTERMATH_SEC):
    """Excluded spans that also end a clear lead-in: other-anomaly segments plus their aftermath, and mask spans."""
    out = [[float(s), float(e) + aftermath_sec] for s, e in (other_segments or [])]
    out += [[float(s), float(e)] for s, e in (mask_spans or [])]
    return sorted(out)


@dataclass
class TickLabels:
    """Labels of one video on the 1 Hz clock. Arrays have length T = floor(duration)."""

    duration: float
    segments: list
    t_sec: np.ndarray
    next_onset: np.ndarray        # absolute time of the next onset (inf if none)
    tte: np.ndarray               # next_onset - t (inf if none)
    next_event: np.ndarray        # index into ``segments`` of the next onset (-1 if none)
    in_violence: np.ndarray
    aftermath: np.ndarray
    masked: np.ndarray
    remaining: np.ndarray         # duration - t
    excluded: np.ndarray = None   # ticks in other-anomaly segments (+ aftermath) or mask spans (part of masked)
    pre_violence: np.ndarray = None  # t < first annotated segment start (violent or other); all True if none
    other_segments: list = field(default_factory=list)
    mask_spans: list = field(default_factory=list)
    soft_mask_spans: list = field(default_factory=list)
    y: dict = field(default_factory=dict)          # H -> int8 [T]
    censored: dict = field(default_factory=dict)   # H -> bool [T]
    valid: dict = field(default_factory=dict)      # H -> bool [T]
    hazard_target: np.ndarray = None               # int8 [T, n_bins] in {1, 0, -1}
    hazard_edges: tuple = HAZARD_EDGES

    @property
    def T(self):
        return len(self.t_sec)

    def neg_far(self, window=NEG_WINDOW_SEC):
        """Unmasked ticks with no onset within ``window`` seconds (censoring NOT applied; combine as needed)."""
        return ~self.masked & (self.tte > window)

    def censored_at(self, horizon):
        """Ticks whose future is unobserved for ``horizon`` seconds and that have no onset within it."""
        return (self.tte > horizon) & (self.t_sec + horizon > self.duration + 1e-9)


def tick_labels(duration, segments, horizons=HORIZONS, aftermath_sec=AFTERMATH_SEC,
                hazard_edges=HAZARD_EDGES, t_sec=None, other_segments=None, mask_spans=None, soft_mask_spans=None):
    """Build ``TickLabels`` for a video of ``duration`` seconds with violent ``segments`` (seconds).

    ``other_segments`` (non-violent anomalies, excluded with their aftermath), ``mask_spans`` and
    ``soft_mask_spans`` (excluded as given) are optional; see the module docstring.
    """
    duration = float(duration)
    segs = normalize_segments(segments, duration)
    others = normalize_segments(other_segments, duration)
    masks = normalize_segments(mask_spans, duration)
    soft = normalize_segments(soft_mask_spans, duration)
    t = tick_times(duration).astype(np.float64) if t_sec is None else np.asarray(t_sec, dtype=np.float64)
    T = len(t)
    starts = np.array([s for s, _ in segs], dtype=np.float64)
    ends = np.array([e for _, e in segs], dtype=np.float64)

    # Next onset strictly after t.
    idx = np.searchsorted(starts, t, side="right") if len(starts) else np.zeros(T, dtype=np.int64)
    has_next = idx < len(starts)
    next_onset = np.full(T, np.inf)
    if len(starts):
        next_onset[has_next] = starts[idx[has_next]]
    next_event = np.where(has_next, idx, -1).astype(np.int64)
    tte = next_onset - t

    in_v = np.zeros(T, dtype=bool)
    aft = np.zeros(T, dtype=bool)
    for s, e in segs:
        in_v |= (t >= s) & (t <= e)
        aft |= (t > e) & (t <= e + aftermath_sec)
    aft &= ~in_v
    excl = np.zeros(T, dtype=bool)
    for s, e in blocker_spans(others, masks, aftermath_sec) + soft:
        excl |= (t >= s) & (t <= e)
    masked = in_v | aft | excl
    remaining = duration - t
    starts_all = [s for s, _ in segs] + [s for s, _ in others]
    pre = t < (min(starts_all) if starts_all else np.inf)

    y, cens, valid = {}, {}, {}
    for H in horizons:
        H = int(H) if float(H).is_integer() else float(H)
        y[H] = ((tte > 0) & (tte <= H)).astype(np.int8)
        cens[H] = (tte > H) & (t + H > duration + 1e-9)
        valid[H] = ~masked & ~cens[H]

    edges = np.asarray(hazard_edges, dtype=np.float64)
    nb = len(edges) - 1
    haz = np.full((T, nb), -1, dtype=np.int8)
    upper = edges[1:]
    ev = tte <= edges[-1]
    bin_of = np.searchsorted(upper, tte, side="left")  # tte in (edges[j], edges[j+1]] -> j
    for j in range(nb):
        # Event ticks: 0 before the event bin, 1 in it.
        haz[ev & (bin_of > j), j] = 0
        haz[ev & (bin_of == j), j] = 1
        # No event within the last edge: survived bin j if its upper edge has been observed.
        haz[~ev & (upper[j] <= remaining + 1e-9), j] = 0
    haz[masked] = -1

    return TickLabels(duration=duration, segments=segs, t_sec=t, next_onset=next_onset, tte=tte,
                      next_event=next_event, in_violence=in_v, aftermath=aft, masked=masked,
                      remaining=remaining, excluded=excl, pre_violence=pre, other_segments=others,
                      mask_spans=masks, soft_mask_spans=soft, y=y, censored=cens, valid=valid, hazard_target=haz,
                      hazard_edges=tuple(float(x) for x in edges))


def hazard_to_cdf(h):
    """Per-bin hazards ``h [..., n_bins]`` -> P(onset within edge_j) ``[..., n_bins]`` = 1 - prod(1 - h)."""
    h = np.asarray(h, dtype=np.float64)
    return 1.0 - np.cumprod(1.0 - h, axis=-1)


def labels_for_video(video_id, catalog_row, events, **kw):
    """Convenience: ``TickLabels`` for a catalog row (``duration_sec``, ``role``) and the events.json dict.

    Violent segments count only for role ``violent`` (other roles keep their segments in events.json for reference
    only); ``other_segments`` and ``mask_spans`` are applied for every role.
    """
    ev = events.get(video_id, {}) if isinstance(events, dict) else {}
    role = catalog_row.get("role", ev.get("role")) if hasattr(catalog_row, "get") else ev.get("role")
    segs = ev.get("segments", []) if role == "violent" else []
    return tick_labels(float(catalog_row["duration_sec"]), segs, other_segments=ev.get("other_segments"),
                       mask_spans=ev.get("mask_spans"), soft_mask_spans=ev.get("soft_mask_spans"), **kw)


def video_event_table(events_entry, role="violent", **kw):
    """``event_table`` of one events.json entry with its excluded spans as blockers (empty unless violent)."""
    if role != "violent":
        return []
    return event_table(events_entry.get("segments", []),
                       blockers=blocker_spans(events_entry.get("other_segments"), events_entry.get("mask_spans")),
                       **kw)
