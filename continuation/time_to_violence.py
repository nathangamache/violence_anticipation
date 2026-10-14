"""
time_to_violence.py
====================
Time-to-violence heuristic for the RTSP inference pipeline.

Adds a `TimeToViolenceEstimator` class that wraps the raw violence probability
score and produces:
  1. A human-readable lead-time estimate ("~20-30s")
  2. An alert tier (WATCH / ALERT / IMMINENT / ACTIVE)
  3. A confidence level based on score stability

Background:
  From evaluation on XD-Violence, the model's average lead time when it first
  triggers on a pre-violence window is 26.9 seconds. We use this as the
  calibration anchor.

  The key insight: instead of just thresholding (alert/no-alert), we can use
  the RATE OF CHANGE of the violence score to estimate how close the event is.
  A slowly rising score from a low base → longer lead time.
  A rapidly rising score already near threshold → shorter lead time.

Usage (drop-in addition to rtsp_inference.py):

    from time_to_violence import TimeToViolenceEstimator, AlertState

    ttv = TimeToViolenceEstimator(fps=1.0)

    # In the inference loop, replace:
    #   is_alert = score > threshold
    # with:
    alert = ttv.update(score, timestamp)
    if alert.tier != AlertState.CLEAR:
        print(f"[{alert.tier.name}] Score={score:.3f}  ETA={alert.eta_str}  Conf={alert.confidence:.2f}")
"""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AlertState(Enum):
    CLEAR    = 0   # score below watch threshold
    WATCH    = 1   # elevated score, monitoring
    ALERT    = 2   # score above alert threshold, estimating lead time
    IMMINENT = 3   # score high and rising fast, <15s estimated
    ACTIVE   = 4   # score above detection threshold, violence likely occurring


@dataclass
class AlertOutput:
    """Output from TimeToViolenceEstimator.update()"""
    tier: AlertState
    score: float
    eta_seconds: Optional[float]       # estimated seconds until violence
    eta_str: str                       # human-readable e.g. "~20-30s"
    confidence: float                  # 0-1, based on score stability
    score_velocity: float              # rate of change (score/second)
    sustained_seconds: float           # how long score has been elevated
    timestamp: float


# ──────────────────────────────────────────────
# Thresholds (tunable)
# ──────────────────────────────────────────────

THRESHOLD_WATCH     = 0.30   # start monitoring
THRESHOLD_ALERT     = 0.45   # raise alert, begin ETA estimate
THRESHOLD_IMMINENT  = 0.65   # high confidence, short lead time
THRESHOLD_ACTIVE    = 0.75   # violence likely occurring now

# From XD-Violence evaluation: average lead time at first alert trigger
BASELINE_LEAD_TIME_SECONDS = 26.9

# How many historical scores to keep for velocity computation
VELOCITY_WINDOW_SECONDS = 10


class TimeToViolenceEstimator:
    """
    Wraps a frame-by-frame violence probability score and produces
    structured alert outputs with time-to-violence estimates.

    Parameters
    ----------
    fps : float
        Frame rate at which scores are fed in (typically 1.0 for this system)
    ema_alpha : float
        EMA smoothing applied to incoming scores before analysis.
        Reduces jitter. 0.3 = moderate smoothing.
    thresholds : dict, optional
        Override default thresholds.
    baseline_lead_time : float
        Calibrated average lead time in seconds (from XD-Violence eval).
    """

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

        # State
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
        """
        Feed one new score. Returns an AlertOutput.

        Parameters
        ----------
        raw_score : float
            Violence probability in [0, 1] from the model
        timestamp : float, optional
            Unix timestamp (defaults to time.time())
        """
        if timestamp is None:
            timestamp = time.time()

        self._frame_count += 1

        # EMA smoothing
        if self._frame_count == 1:
            self._ema_score = raw_score
        else:
            self._ema_score = (
                self.ema_alpha * raw_score + (1 - self.ema_alpha) * self._ema_score
            )

        score = self._ema_score
        self._score_history.append(score)
        self._time_history.append(timestamp)

        # ── Score velocity ─────────────────────
        velocity = self._compute_velocity()

        # ── Determine tier ─────────────────────
        tier = self._classify_tier(score, velocity)

        # ── Track sustained alert time ─────────
        if tier.value >= AlertState.ALERT.value:
            if self._alert_start_time is None:
                self._alert_start_time = timestamp
            sustained = timestamp - self._alert_start_time
        else:
            if tier == AlertState.CLEAR:
                self._alert_start_time = None
            sustained = 0.0

        # ── ETA estimate ───────────────────────
        eta_seconds, eta_str = self._estimate_eta(score, velocity, tier, sustained)

        # ── Confidence ─────────────────────────
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
        """Score change per second over the recent window."""
        if len(self._score_history) < 3:
            return 0.0
        scores = list(self._score_history)
        times = list(self._time_history)
        dt = times[-1] - times[0]
        if dt < 0.5:
            return 0.0
        # Linear regression slope
        n = len(scores)
        t_arr = [times[i] - times[0] for i in range(n)]
        s_arr = scores
        t_mean = sum(t_arr) / n
        s_mean = sum(s_arr) / n
        num = sum((t_arr[i] - t_mean) * (s_arr[i] - s_mean) for i in range(n))
        den = sum((t_arr[i] - t_mean) ** 2 for i in range(n)) + 1e-8
        return num / den  # score units per second

    def _classify_tier(self, score: float, velocity: float) -> AlertState:
        """Map score + velocity to an alert tier."""
        if score >= self.thresh["active"]:
            return AlertState.ACTIVE
        elif score >= self.thresh["imminent"]:
            return AlertState.IMMINENT
        elif score >= self.thresh["alert"]:
            # Fast-rising score near threshold → bump to IMMINENT
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
        """
        Estimate time to violence in seconds using a score-bracketed heuristic.

        The approach:
          - Anchor: at ALERT threshold (~0.45), baseline lead time = 26.9s
          - As score rises toward ACTIVE threshold (~0.75), lead time shrinks
          - Velocity modifies the estimate: fast rise → shorter ETA
          - Sustained time past alert reduces ETA (we are already inside the window)
        """
        if tier == AlertState.CLEAR:
            return None, "N/A"

        if tier == AlertState.ACTIVE:
            return 0.0, "NOW"

        # Base ETA from score bracket
        alert_range = self.thresh["active"] - self.thresh["alert"]
        if alert_range > 0:
            progress = max(0.0, min(1.0, (score - self.thresh["alert"]) / alert_range))
        else:
            progress = 0.0

        # At progress=0 (just hit alert): full baseline lead time
        # At progress=1 (just below active): 5s
        base_eta = self.baseline_lead_time * (1 - progress) + 5.0 * progress

        # Velocity adjustment: fast rise means the event is accelerating toward us
        if velocity > 0.02:
            # Fast rise: subtract up to 8s
            velocity_adjustment = min(8.0, velocity * 200.0)
            base_eta = max(5.0, base_eta - velocity_adjustment)
        elif velocity < -0.01:
            # Falling score: add some time
            base_eta = min(base_eta + 5.0, self.baseline_lead_time * 1.5)

        # Subtract time already spent in alert state
        eta = max(3.0, base_eta - sustained * 0.5)

        # Watch tier: no reliable estimate yet
        if tier == AlertState.WATCH:
            return None, ">30s"

        # Round to nearest 5s and express as range
        eta_low = max(5, int(eta / 5) * 5 - 5)
        eta_high = eta_low + 10
        eta_str = f"~{eta_low}-{eta_high}s"

        # Imminent: tighter range
        if tier == AlertState.IMMINENT:
            if eta < 10:
                eta_str = "<10s"
            elif eta < 20:
                eta_str = "10-20s"
            else:
                eta_str = "~20s"

        return eta, eta_str

    def _compute_confidence(self, score: float, velocity: float) -> float:
        """
        Confidence in the estimate: 0-1.
        Higher when:
          - Score is well above threshold (not borderline)
          - Score has been elevated for multiple frames (not a spike)
          - Velocity is positive (rising, not falling)
        """
        if score < self.thresh["watch"]:
            return 0.0

        # Score margin above alert threshold
        margin = (score - self.thresh["alert"]) / (1.0 - self.thresh["alert"] + 1e-8)
        margin = max(0.0, min(1.0, margin))

        # History stability (variance of recent scores)
        if len(self._score_history) >= 3:
            recent = list(self._score_history)[-5:]
            variance = sum((s - sum(recent) / len(recent)) ** 2 for s in recent) / len(recent)
            stability = max(0.0, 1.0 - variance * 20.0)
        else:
            stability = 0.3

        # Velocity bonus (rising = more confident)
        vel_factor = 1.0 if velocity >= 0 else 0.7

        return round(min(1.0, (0.5 * margin + 0.4 * stability) * vel_factor), 2)


# ──────────────────────────────────────────────
# Integration patch for rtsp_inference.py
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
# Standalone demo / test
# ──────────────────────────────────────────────

def demo():
    """
    Simulates a 70-second surveillance clip:
      0-9s:   calm
      10-39s: tension building (score rises 0.1 → 0.5)
      40-50s: escalation (0.5 → 0.8)
      51-60s: active violence (0.8-0.9)
      61-70s: de-escalation
    """
    print("\n" + "=" * 65)
    print("TIME-TO-VIOLENCE ESTIMATOR — DEMO")
    print("=" * 65)
    print(f"\nBaseline lead time: {BASELINE_LEAD_TIME_SECONDS}s")
    print(f"\nThresholds: WATCH={THRESHOLD_WATCH}  ALERT={THRESHOLD_ALERT}  "
          f"IMMINENT={THRESHOLD_IMMINENT}  ACTIVE={THRESHOLD_ACTIVE}")

    ttv = TimeToViolenceEstimator(fps=1.0, ema_alpha=0.3)

    # Synthetic score sequence
    import random
    random.seed(42)

    def noisy(val, noise=0.03):
        return max(0.0, min(1.0, val + random.gauss(0, noise)))

    scores = (
        [noisy(0.08) for _ in range(10)] +   # calm
        [noisy(0.1 + 0.015 * i) for i in range(30)] +   # tension building
        [noisy(0.5 + 0.03 * i) for i in range(10)] +   # escalation
        [noisy(0.82) for _ in range(10)] +   # active
        [noisy(0.82 - 0.06 * i) for i in range(10)]    # de-escalation
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