"""Hysteresis + dwell + re-alert state machine. Pure: no I/O.

Three mechanisms, all required, because a raw threshold on a noisy signal
flaps and at 15 fps that is tens of alerts a second:
  hysteresis  - separate rise/fall thresholds (in levels.classify)
  dwell       - a candidate level must persist before it is adopted
  re-alert    - escalation always bypasses this interval (getting worse must
                always be heard immediately); de-escalation is gated by it,
                keyed on the level being stepped down from, so an
                oscillating signal cannot emit alternating alert / all-clear
                pairs forever

Dwell tracks divergence from the *adopted* level (self.level), not the exact
candidate value. A noisy signal straddling a boundary (e.g. flapping between
ELEVATED and HIGH while the adopted level is still NORMAL) must count as one
continuous divergence -- resetting the timer on every identity change would
mean dwell never completes and the anti-flapping property breaks down
exactly where it matters most.
"""
from __future__ import annotations

import math
from typing import Optional

from .events import RiskEvent
from .levels import RiskLevel, Thresholds, classify, severity


class RiskStateMachine:
    def __init__(
        self,
        thresholds: Thresholds,
        min_dwell_s: float = 2.0,
        min_realert_s: float = 60.0,
        blind_alert_s: float = 30.0,
    ) -> None:
        if min_dwell_s < 0 or min_realert_s < 0 or blind_alert_s < 0:
            raise ValueError("durations must be non-negative")
        self.thresholds = thresholds
        self.min_dwell_s = min_dwell_s
        self.min_realert_s = min_realert_s
        self.blind_alert_s = blind_alert_s

        self.level = RiskLevel.NORMAL
        self._candidate_since: Optional[float] = None
        self._last_alert_t: Optional[float] = None
        self._last_alerted_level: Optional[RiskLevel] = None
        self._last_t: Optional[float] = None

    def _reset_timers(self) -> None:
        self._candidate_since = None

    def update(
        self, timestamp: float, pressure, coverage: Optional[float] = None
    ) -> Optional[RiskEvent]:
        t = float(timestamp)

        # A clock that moves backwards means the source restarted. Restart the
        # timers rather than computing a negative dwell.
        if self._last_t is not None and t < self._last_t:
            self._reset_timers()
            self._last_alert_t = None
            self._last_alerted_level = None
        self._last_t = t

        candidate = classify(pressure, self.thresholds, current=self.level)

        if candidate is self.level:
            # Back at (or still at) the adopted level: nothing pending.
            self._reset_timers()
            return None

        # candidate differs from the adopted level -- track how long we have
        # been continuously away from self.level (see module docstring).
        if self._candidate_since is None:
            self._candidate_since = t

        required_dwell = (
            self.blind_alert_s if candidate is RiskLevel.UNKNOWN else self.min_dwell_s
        )
        held_for = t - self._candidate_since
        if held_for < required_dwell:
            return None

        previous = self.level
        self.level = candidate
        self._reset_timers()

        old, new = severity(previous), severity(candidate)
        if old is None or new is None:
            reason = "sensor-blind" if candidate is RiskLevel.UNKNOWN else "escalation"
        elif new > old:
            reason = "escalation"
        elif new < old:
            reason = "de-escalation"
        else:  # pragma: no cover - equal severities cannot differ in level
            reason = "sustained"

        # Re-alert gate: escalation ("getting worse") must always be heard
        # immediately, so it always bypasses the interval -- even back up to
        # a level that was already alerted. De-escalation (the all-clear) is
        # gated instead, keyed on the level being stepped down FROM: without
        # this, an oscillating signal emits alternating alert / all-clear
        # pairs forever, which defeats the anti-spam purpose of this gate.
        if reason == "de-escalation":
            if (
                self._last_alerted_level is previous
                and self._last_alert_t is not None
                and (t - self._last_alert_t) < self.min_realert_s
            ):
                return None

        self._last_alerted_level = candidate
        self._last_alert_t = t
        p = None if pressure is None or not math.isfinite(float(pressure)) else float(pressure)
        message = (
            f"{previous.value} -> {candidate.value}"
            + (f" (pressure {p:.4f} s^-2)" if p is not None else " (pressure unavailable)")
            + (f", coverage {coverage:.0%}" if coverage is not None else "")
        )
        return RiskEvent(
            timestamp=t,
            level=candidate,
            previous_level=previous,
            pressure=p,
            coverage=coverage,
            reason=reason,
            message=message,
        )
