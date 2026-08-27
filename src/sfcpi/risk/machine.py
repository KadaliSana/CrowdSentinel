"""Hysteresis + dwell + re-alert state machine. Pure: no I/O.

Three mechanisms, all required, because a raw threshold on a noisy signal
flaps and at 15 fps that is tens of alerts a second:
  hysteresis  - separate rise/fall thresholds (in levels.classify)
  dwell       - a candidate level must persist before it is adopted
  re-alert    - bypassed only when the candidate genuinely escalates PAST
                the last level actually alerted (severity(candidate) >
                severity(last_alerted_level)), or on the first-ever alert,
                or on the FIRST sensor-blind alert, or on a recovery from
                UNKNOWN straight to ELEVATED-or-worse (a sensor coming back
                already showing danger must be heard). Re-escalating back up
                to a level already alerted, any de-escalation, every
                *subsequent* sensor-blind, and every recovery to NORMAL are
                gated by time instead -- otherwise a slow full-swing
                oscillation (HIGH -> NORMAL -> HIGH -> ...) or a flapping
                camera (blind -> fine -> blind) would re-alert every dwell
                period forever, at any min_realert_s. Note that closing one
                of these bypasses tends to RELOCATE the spam onto whichever
                is still open, so each is gated on the clock alone rather
                than on a condition about the last alerted level.

Dwell is tracked on TWO independent timers, because evidence for a level and
absence of evidence are different things:

  known track   - `_known_since` / `_known_floor`. Runs while measured frames
                  disagree with the adopted level. `_known_floor` is the
                  LEAST severe candidate seen since the track started, so the
                  level adopted is the strongest claim that *every* frame in
                  the window supports. Adoption happens only on a measured
                  frame. A single spike therefore cannot be adopted, and an
                  unscorable frame can never act as evidence for a danger
                  level (it does not touch this track at all).
  unknown track - `_unknown_since`. Runs while frames are unscorable and is
                  cleared by any measured frame. Adoption of UNKNOWN is gated
                  by `blind_alert_s` rather than `min_dwell_s`.

The floor (rather than the last candidate) is what keeps the anti-flapping
property: a noisy signal straddling a boundary (e.g. flapping between
ELEVATED and HIGH while the adopted level is still NORMAL) is one continuous
divergence and settles at ELEVATED, instead of adopting whichever side the
frame that happened to complete the dwell landed on.

Consequence worth knowing: because blindness does not clear the known track,
a danger frame, a gap of unscorable frames, and another danger frame can
together satisfy dwell. That errs toward alerting (never toward silence) and
is still bounded by `min_realert_s`.
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
        min_coverage: float = 0.0,
    ) -> None:
        if min_dwell_s < 0 or min_realert_s < 0 or blind_alert_s < 0:
            raise ValueError("durations must be non-negative")
        if not math.isfinite(min_coverage) or not 0.0 <= min_coverage <= 1.0:
            raise ValueError(
                f"min_coverage must be a finite fraction in [0.0, 1.0], got {min_coverage!r}"
            )
        self.thresholds = thresholds
        self.min_dwell_s = min_dwell_s
        self.min_realert_s = min_realert_s
        self.blind_alert_s = blind_alert_s
        self.min_coverage = min_coverage

        self.level = RiskLevel.NORMAL
        self._known_since: Optional[float] = None
        self._known_floor: Optional[RiskLevel] = None
        self._unknown_since: Optional[float] = None
        self._last_alert_t: Optional[float] = None
        self._last_alerted_level: Optional[RiskLevel] = None
        self._blind_alerted = False
        self._last_t: Optional[float] = None

    def _reset_timers(self) -> None:
        self._known_since = None
        self._known_floor = None
        self._unknown_since = None

    def update(
        self, timestamp: float, pressure, coverage: Optional[float] = None
    ) -> Optional[RiskEvent]:
        t = float(timestamp)
        # A non-finite timestamp makes every comparison below False, which
        # would defeat the dwell gate AND the re-alert gate simultaneously.
        # Refuse it rather than silently alerting on every frame. Checked
        # before any state is touched so a rejected call mutates nothing.
        if not math.isfinite(t):
            raise ValueError(f"timestamp must be finite, got {timestamp!r}")

        # A clock that moves backwards means the source restarted. Restart the
        # timers rather than computing a negative dwell.
        if self._last_t is not None and t < self._last_t:
            self._reset_timers()
            self._last_alert_t = None
            self._last_alerted_level = None
            self._blind_alerted = False
        self._last_t = t

        # Too few faces to trust the field: the frame is unscorable, not calm.
        # A face detector dropping out is a legitimate reading of ~0 pressure,
        # so scoring it as NORMAL turns the dense-crowd signature into silence.
        effective_pressure = pressure
        if (
            self.min_coverage > 0.0
            and coverage is not None
            and float(coverage) < self.min_coverage
        ):
            effective_pressure = float("nan")

        candidate = classify(effective_pressure, self.thresholds, current=self.level)

        if candidate is self.level:
            # Back at (or still at) the adopted level: nothing pending.
            self._reset_timers()
            return None

        if candidate is RiskLevel.UNKNOWN:
            # Unscorable. Runs the blind timer only; it is not evidence for or
            # against any measured level, so the known track is left alone.
            if self._unknown_since is None:
                self._unknown_since = t
            if (t - self._unknown_since) < self.blind_alert_s:
                return None
            adopted = RiskLevel.UNKNOWN
        else:
            # Measured frame disagreeing with the adopted level: it ends any
            # run of blindness and extends/starts the known track.
            self._unknown_since = None
            if self._known_since is None:
                self._known_since = t
                self._known_floor = candidate
            elif severity(candidate) < severity(self._known_floor):
                self._known_floor = candidate
            if (t - self._known_since) < self.min_dwell_s:
                return None
            adopted = self._known_floor

        previous = self.level
        self.level = adopted
        self._reset_timers()

        if adopted is RiskLevel.UNKNOWN:
            reason = "sensor-blind"
        elif previous is RiskLevel.UNKNOWN:
            # UNKNOWN carries no ordered severity to compare against, so a
            # return from it is its own reason rather than a mislabelled
            # escalation/de-escalation.
            reason = "recovery"
        else:
            old, new = severity(previous), severity(adopted)
            if new > old:
                reason = "escalation"
            elif new < old:
                reason = "de-escalation"
            else:  # pragma: no cover - equal severities cannot differ in level
                reason = "sustained"

        # Re-alert gate.
        if reason == "sensor-blind":
            # A blind sensor is itself an incident, so the first one is heard
            # immediately -- but only the first. A camera flapping blind/fine
            # otherwise pages once per flap forever, because leaving UNKNOWN
            # gates the all-clear and _last_alerted_level never advances past
            # UNKNOWN, so the next blind period bypasses again.
            if (
                self._blind_alerted
                and self._last_alert_t is not None
                and (t - self._last_alert_t) < self.min_realert_s
            ):
                return None
        elif reason == "recovery" and adopted is RiskLevel.NORMAL:
            # Recovery to NORMAL is the resolution of a prior blind alert --
            # gate it the same way as any other all-clear, on time ALONE.
            #
            # This deliberately does NOT also require
            # `self._last_alerted_level is previous` (i.e. UNKNOWN). Gating
            # repeat sensor-blinds means the blind alert is often suppressed,
            # so `_last_alerted_level` sits at whatever was last actually
            # paged -- a danger level -- while `self.level` is UNKNOWN. That
            # identity check therefore failed and let the all-clear bypass
            # min_realert_s; firing then set `_last_alerted_level = NORMAL`,
            # which is equally not UNKNOWN, so the bypass re-armed itself and
            # a blind/fine flap paged forever via `recovery` instead of via
            # `sensor-blind`. Only the clock may open this gate.
            if (
                self._last_alert_t is not None
                and (t - self._last_alert_t) < self.min_realert_s
            ):
                return None
        elif reason == "recovery":
            pass  # recovery straight to danger (ELEVATED+) always heard immediately
        else:
            # escalation / de-escalation: bypass only when the candidate
            # genuinely escalates PAST the last level actually alerted.
            # Equal or lower severity (including re-escalating back up to
            # the same level) is gated by time instead.
            cand_sev = severity(adopted)
            if self._last_alerted_level is None:
                escalates_past_last_alert = True  # first-ever alert
            elif self._last_alerted_level is RiskLevel.UNKNOWN:
                # A stale UNKNOWN here means the last thing the operator was
                # told is "the sensor is blind" -- no danger level has been
                # communicated at all. Any danger level is therefore new
                # information and is heard; an all-clear is not, and waits.
                # (severity(UNKNOWN) is None, so this must NOT fall through to
                # the first-ever-alert branch, which would bypass the gate for
                # de-escalations too.)
                escalates_past_last_alert = cand_sev > 0
            else:
                escalates_past_last_alert = cand_sev > severity(self._last_alerted_level)
            if not escalates_past_last_alert:
                if (
                    self._last_alert_t is not None
                    and (t - self._last_alert_t) < self.min_realert_s
                ):
                    return None

        self._last_alerted_level = adopted
        self._last_alert_t = t
        if reason == "sensor-blind":
            self._blind_alerted = True
        p = (
            None
            if effective_pressure is None or not math.isfinite(float(effective_pressure))
            else float(effective_pressure)
        )
        message = (
            f"{previous.value} -> {adopted.value}"
            + (f" (pressure {p:.4f} s^-2)" if p is not None else " (pressure unavailable)")
            + (f", coverage {coverage:.0%}" if coverage is not None else "")
        )
        return RiskEvent(
            timestamp=t,
            level=adopted,
            previous_level=previous,
            pressure=p,
            coverage=coverage,
            reason=reason,
            message=message,
        )
