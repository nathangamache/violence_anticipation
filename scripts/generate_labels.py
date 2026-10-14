import numpy as np


def generate_risk_curve(
    meta: dict,
    num_frames: int | None = None,
    anticipation_window_sec: float = 30.0,
    post_event_window_sec: float = 10.0,
) -> np.ndarray:
    """
    Build a per-frame risk curve in [0, 1] with an explicit anticipation horizon.

    - For negative clips (meta['label'] == 0): risk is 0 for all frames.
    - For positive clips:
        * Let t_event be the event frame.
        * Let T_pre = anticipation_window_sec (in seconds).
        * Let T_post = post_event_window_sec (in seconds).

      Then:
        - risk[t] = 0 for t < t_event - T_pre * fps
        - risk[t] ramps from 0 → 1 between (t_event - T_pre * fps) and t_event
        - risk[t] decays from 1 → 0 over T_post seconds after t_event

      => The earliest non-zero risk is ~T_pre seconds before the event,
         so max possible TTD is ~anticipation_window_sec.
    """

    # Pull metadata
    label = int(meta.get("label", 0))
    fps = float(meta.get("fps", 25.0))

    # Decide total number of frames
    total_frames = meta.get("total_frames", None)
    if total_frames is None and num_frames is None:
        raise ValueError("generate_risk_curve: need either meta['total_frames'] or num_frames.")
    if num_frames is not None:
        total_frames = int(num_frames)
    else:
        total_frames = int(total_frames)

    if total_frames <= 0:
        return np.zeros(0, dtype=np.float32)

    # Non-violent clip → all zeros
    if label == 0:
        return np.zeros(total_frames, dtype=np.float32)

    # Violent clip
    event_frame = int(meta.get("event_frame", 0))
    event_frame = max(0, min(event_frame, total_frames - 1))

    risk = np.zeros(total_frames, dtype=np.float32)

    # Convert horizons to frames
    pre_frames = int(anticipation_window_sec * fps)
    post_frames = int(post_event_window_sec * fps)

    # Start of anticipation ramp
    start = max(0, event_frame - pre_frames)

    # 1) Ramp-up region: [start, event_frame]
    if event_frame >= start:
        ramp_len = event_frame - start + 1  # inclusive of both endpoints
        ramp_len = max(1, ramp_len)
        ramp = np.linspace(0.0, 1.0, ramp_len, dtype=np.float32)
        risk[start:event_frame + 1] = np.maximum(risk[start:event_frame + 1], ramp)

    # 2) Post-event decay region: (event_frame, end_decay]
    end_decay = min(total_frames - 1, event_frame + post_frames)
    if end_decay > event_frame:
        decay_len = end_decay - event_frame
        decay = np.linspace(1.0, 0.0, decay_len + 1, dtype=np.float32)[1:]  # drop the first 1.0
        risk[event_frame + 1:end_decay + 1] = np.maximum(
            risk[event_frame + 1:end_decay + 1],
            decay,
        )

    return risk
