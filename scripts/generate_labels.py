import numpy as np


def generate_risk_curve(
    meta: dict,
    num_frames: int | None = None,
    anticipation_window_sec: float = 30.0,
    post_event_window_sec: float = 10.0,
) -> np.ndarray:
    """Per-frame risk in [0, 1]: zeros for negatives; for positives a linear ramp to 1 over
    anticipation_window_sec ending at event_frame, then a linear decay to 0 over post_event_window_sec."""
    label = int(meta.get("label", 0))
    fps = float(meta.get("fps", 25.0))

    total_frames = meta.get("total_frames", None)
    if total_frames is None and num_frames is None:
        raise ValueError("generate_risk_curve: need either meta['total_frames'] or num_frames.")
    if num_frames is not None:
        total_frames = int(num_frames)
    else:
        total_frames = int(total_frames)

    if total_frames <= 0:
        return np.zeros(0, dtype=np.float32)

    if label == 0:
        return np.zeros(total_frames, dtype=np.float32)

    event_frame = int(meta.get("event_frame", 0))
    event_frame = max(0, min(event_frame, total_frames - 1))

    risk = np.zeros(total_frames, dtype=np.float32)

    pre_frames = int(anticipation_window_sec * fps)
    post_frames = int(post_event_window_sec * fps)

    start = max(0, event_frame - pre_frames)

    if event_frame >= start:
        ramp_len = event_frame - start + 1
        ramp_len = max(1, ramp_len)
        ramp = np.linspace(0.0, 1.0, ramp_len, dtype=np.float32)
        risk[start:event_frame + 1] = np.maximum(risk[start:event_frame + 1], ramp)

    end_decay = min(total_frames - 1, event_frame + post_frames)
    if end_decay > event_frame:
        decay_len = end_decay - event_frame
        decay = np.linspace(1.0, 0.0, decay_len + 1, dtype=np.float32)[1:]
        risk[event_frame + 1:end_decay + 1] = np.maximum(
            risk[event_frame + 1:end_decay + 1],
            decay,
        )

    return risk
