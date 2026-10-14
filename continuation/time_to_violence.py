"""Alert tiers and a heuristic time-to-violence estimate from per-second violence scores (run directly for a demo)."""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AlertState(Enum):
    CLEAR    = 0
    WATCH    = 1
    ALERT    = 2
    IMMINENT = 3
    ACTIVE   = 4


@dataclass
class AlertOutput:
    tier: AlertState
    score: float
    eta_seconds: Optional[float]
    eta_str: str                       # e.g. "~20-30s"
    confidence: float                  # 0-1
    score_velocity: float              # score units per second
    sustained_seconds: float           # seconds since first reaching ALERT
    timestamp: float


THRESHOLD_WATCH     = 0.30
THRESHOLD_ALERT     = 0.45
THRESHOLD_IMMINENT  = 0.65
THRESHOLD_ACTIVE    = 0.75

# Mean lead time at first alert in the XD-Violence evaluation (final/high_sensitivity/results.json)
BASELINE_LEAD_TIME_SECONDS = 26.9

VELOCITY_WINDOW_SECONDS = 10


class TimeToViolenceEstimator:
    """EMA-smooth per-frame violence scores and emit an AlertOutput (tier, ETA, confidence) on each update."""

    def __init__(
        self,
        fps: float = 1.0,
        ema_alpha: float = 0.3,
        thresholds: Optional[dict] = None,
        baseline_lead_time: float = BASELINE_LEAD_TIME_SECONDS,
    ):
        self.fps = fps
        self.ema_alpha = ema_alpha
        self.baseline_lead_time = baseline_lead_time

        self.thresh = {
            "watch":    THRESHOLD_WATCH,
            "alert":    THRESHOLD_ALERT,
            "imminent": THRESHOLD_IMMINENT,
            "active":   THRESHOLD_ACTIVE,
        }
        if thresholds:
            self.thresh.update(thresholds)

        self._ema_score: float = 0.0
        self._score_history: deque = deque(maxlen=int(VELOCITY_WINDOW_SECONDS * fps))
        self._time_history: deque = deque(maxlen=int(VELOCITY_WINDOW_SECONDS * fps))
        self._alert_start_time: Optional[float] = None
        self._prev_tier: AlertState = AlertState.CLEAR
        self._frame_count: int = 0

    def reset(self):
        """Reset all state (call when starting a new video or after long gap)."""
        self._ema_score = 0.0
        self._score_history.clear()
        self._time_history.clear()
        self._alert_start_time = None
        self._prev_tier = AlertState.CLEAR
        self._frame_count = 0

    def update(self, raw_score: float, timestamp: Optional[float] = None) -> AlertOutput:
        """Feed one raw model score in [0, 1] and return the current alert."""
        if timestamp is None:
            timestamp = time.time()

        self._frame_count += 1

        if self._frame_count == 1:
            self._ema_score = raw_score
        else:
            self._ema_score = (
                self.ema_alpha * raw_score + (1 - self.ema_alpha) * self._ema_score
            )

        score = self._ema_score
        self._score_history.append(score)
        self._time_history.append(timestamp)

        velocity = self._compute_velocity()

        tier = self._classify_tier(score, velocity)

        if tier.value >= AlertState.ALERT.value:
            if self._alert_start_time is None:
                self._alert_start_time = timestamp
            sustained = timestamp - self._alert_start_time
        else:
            if tier == AlertState.CLEAR:
                self._alert_start_time = None
            sustained = 0.0

        eta_seconds, eta_str = self._estimate_eta(score, velocity, tier, sustained)

        confidence = self._compute_confidence(score, velocity)

        self._prev_tier = tier

        return AlertOutput(
            tier=tier,
            score=score,
            eta_seconds=eta_seconds,
            eta_str=eta_str,
            confidence=confidence,
            score_velocity=velocity,
            sustained_seconds=sustained,
            timestamp=timestamp,
        )

    def _compute_velocity(self) -> float:
        """Least-squares slope of the smoothed score over the recent window (score units per second)."""
        if len(self._score_history) < 3:
            return 0.0
        scores = list(self._score_history)
        times = list(self._time_history)
        dt = times[-1] - times[0]
        if dt < 0.5:
            return 0.0
        n = len(scores)
        t_arr = [times[i] - times[0] for i in range(n)]
        s_arr = scores
        t_mean = sum(t_arr) / n
        s_mean = sum(s_arr) / n
        num = sum((t_arr[i] - t_mean) * (s_arr[i] - s_mean) for i in range(n))
        den = sum((t_arr[i] - t_mean) ** 2 for i in range(n)) + 1e-8
        return num / den

    def _classify_tier(self, score: float, velocity: float) -> AlertState:
        if score >= self.thresh["active"]:
            return AlertState.ACTIVE
        elif score >= self.thresh["imminent"]:
            return AlertState.IMMINENT
        elif score >= self.thresh["alert"]:
            if velocity > 0.03 and score > self.thresh["alert"] + 0.05:
                return AlertState.IMMINENT
            return AlertState.ALERT
        elif score >= self.thresh["watch"]:
            return AlertState.WATCH
        else:
            return AlertState.CLEAR

    def _estimate_eta(
        self,
        score: float,
        velocity: float,
        tier: AlertState,
        sustained: float,
    ):
        """Score-bracketed ETA: baseline lead time at ALERT down to 5 s at ACTIVE, adjusted for velocity and time in alert."""
        if tier == AlertState.CLEAR:
            return None, "N/A"

        if tier == AlertState.ACTIVE:
            return 0.0, "NOW"

        alert_range = self.thresh["active"] - self.thresh["alert"]
        if alert_range > 0:
            progress = max(0.0, min(1.0, (score - self.thresh["alert"]) / alert_range))
        else:
            progress = 0.0

        base_eta = self.baseline_lead_time * (1 - progress) + 5.0 * progress

        if velocity > 0.02:
            velocity_adjustment = min(8.0, velocity * 200.0)
            base_eta = max(5.0, base_eta - velocity_adjustment)
        elif velocity < -0.01:
            base_eta = min(base_eta + 5.0, self.baseline_lead_time * 1.5)

        eta = max(3.0, base_eta - sustained * 0.5)

        if tier == AlertState.WATCH:
            return None, ">30s"

        eta_low = max(5, int(eta / 5) * 5 - 5)
        eta_high = eta_low + 10
        eta_str = f"~{eta_low}-{eta_high}s"

        if tier == AlertState.IMMINENT:
            if eta < 10:
                eta_str = "<10s"
            elif eta < 20:
                eta_str = "10-20s"
            else:
                eta_str = "~20s"

        return eta, eta_str

    def _compute_confidence(self, score: float, velocity: float) -> float:
        """Confidence in [0, 1] from the margin above ALERT, recent score stability and whether the score is rising."""
        if score < self.thresh["watch"]:
            return 0.0

        margin = (score - self.thresh["alert"]) / (1.0 - self.thresh["alert"] + 1e-8)
        margin = max(0.0, min(1.0, margin))

        if len(self._score_history) >= 3:
            recent = list(self._score_history)[-5:]
            variance = sum((s - sum(recent) / len(recent)) ** 2 for s in recent) / len(recent)
            stability = max(0.0, 1.0 - variance * 20.0)
        else:
            stability = 0.3

        vel_factor = 1.0 if velocity >= 0 else 0.7

        return round(min(1.0, (0.5 * margin + 0.4 * stability) * vel_factor), 2)


RTSP_INTEGRATION_PATCH = '''
# ── Add near top of rtsp_inference.py ──────────────────────────────────────
from time_to_violence import TimeToViolenceEstimator, AlertState

# ── In the InferencePipeline or main inference class, add to __init__: ─────
self.ttv = TimeToViolenceEstimator(fps=1.0, ema_alpha=0.3)

# ── Replace the existing alert logic in the inference loop: ────────────────
# OLD:
#   if score > ALERT_THRESHOLD:
#       log.warning(f"ALERT: violence probability {score:.3f}")

# NEW:
alert = self.ttv.update(score, timestamp=pred.timestamp)

if alert.tier == AlertState.CLEAR:
    pass  # no output
elif alert.tier == AlertState.WATCH:
    log.info(f"[WATCH]    score={alert.score:.3f}  {alert.eta_str}  conf={alert.confidence:.2f}")
elif alert.tier == AlertState.ALERT:
    log.warning(f"[ALERT]    score={alert.score:.3f}  ETA={alert.eta_str}  conf={alert.confidence:.2f}  vel={alert.score_velocity:+.3f}/s")
elif alert.tier == AlertState.IMMINENT:
    log.error(f"[IMMINENT] score={alert.score:.3f}  ETA={alert.eta_str}  conf={alert.confidence:.2f}")
elif alert.tier == AlertState.ACTIVE:
    log.critical(f"[ACTIVE]   VIOLENCE DETECTED  score={alert.score:.3f}")

# Also store in the Prediction dataclass if you want to log or stream:
pred.alert_tier   = alert.tier.name
pred.eta_str      = alert.eta_str
pred.ttv_confidence = alert.confidence
'''


def demo():
    """Run the estimator on a synthetic 70 s clip: calm, rising tension, escalation, violence, de-escalation."""
    print("\n" + "=" * 65)
    print("TIME-TO-VIOLENCE ESTIMATOR — DEMO")
    print("=" * 65)
    print(f"\nBaseline lead time: {BASELINE_LEAD_TIME_SECONDS}s")
    print(f"\nThresholds: WATCH={THRESHOLD_WATCH}  ALERT={THRESHOLD_ALERT}  "
          f"IMMINENT={THRESHOLD_IMMINENT}  ACTIVE={THRESHOLD_ACTIVE}")

    ttv = TimeToViolenceEstimator(fps=1.0, ema_alpha=0.3)

    import random
    random.seed(42)

    def noisy(val, noise=0.03):
        return max(0.0, min(1.0, val + random.gauss(0, noise)))

    scores = (
        [noisy(0.08) for _ in range(10)] +
        [noisy(0.1 + 0.015 * i) for i in range(30)] +
        [noisy(0.5 + 0.03 * i) for i in range(10)] +
        [noisy(0.82) for _ in range(10)] +
        [noisy(0.82 - 0.06 * i) for i in range(10)]
    )

    print(f"\n{'Time':>4}  {'Raw':>6}  {'EMA':>6}  {'Tier':<10}  {'ETA':<12}  {'Conf':>5}  {'Vel':>8}")
    print("-" * 65)

    base_ts = time.time()
    for t, raw in enumerate(scores):
        ts = base_ts + t
        alert = ttv.update(raw, timestamp=ts)

        tier_icon = {
            AlertState.CLEAR:    "      ",
            AlertState.WATCH:    "WATCH",
            AlertState.ALERT:    "ALERT",
            AlertState.IMMINENT: "IMMNT",
            AlertState.ACTIVE:   "ACTIV",
        }[alert.tier]

        print(f"{t:3d}s  {raw:>6.3f}  {alert.score:>6.3f}  {tier_icon:<10}  "
              f"{alert.eta_str:<12}  {alert.confidence:>5.2f}  {alert.score_velocity:>+8.4f}")

    print()
    print("Integration patch for rtsp_inference.py:")
    print(RTSP_INTEGRATION_PATCH)


if __name__ == "__main__":
    demo()