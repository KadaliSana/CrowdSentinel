"""Hysteresis + dwell + re-alert state machine. Pure: no I/O.

Three mechanisms, all required, because a raw threshold on a noisy signal
flaps and at 15 fps that is tens of alerts a second:
  hysteresis  - separate rise/fall thresholds (in levels.classify)
  dwell       - a candidate level must persist before it is adopted
  re-alert    - the invariant below.

THE RE-ALERT INVARIANT -- read this before touching the gate
------------------------------------------------------------
    An adopted level is emitted if and only if

        min_realert_s has elapsed since the last emission
        OR the adopted level strictly OUTRANKS `_last_alerted_level`
           (what the operator was last actually told).

That is ONE decision, implemented once, in `_outranks_last_alert`. It is
deliberately NOT a chain of per-reason cases. This machine has had five
separate unbounded-page defects, and every one of them was a special case
added for one `reason` that left a bypass open for another: closing a
bypass RELOCATES the spam onto whichever bypass is still open. A new case
here is a new hole. If you are about to add `elif reason == ...` to the
gate, you are reintroducing the defect.

`reason` is a DISPLAY LABEL for the operator and nothing else. It must
never appear in the gate. That separation is the property being defended.

How UNKNOWN compares (it has no severity, so this is a decision, stated
once, here and in `_outranks_last_alert`):

  * `_last_alerted_level is None` -- nothing has ever been alerted, so
    anything outranks it. The first event always fires.
  * adopted IS UNKNOWN -- "the sensor is blind" is not a point on the
    severity scale, it is a distinct incident. It outranks exactly once
    (`_blind_alerted`): the FIRST blindness is always heard immediately,
    even mid-window and even right after a danger page, because a sensor
    dying is exactly when its silence matters. Every subsequent blindness
    is time-gated, or a camera flapping blind/fine pages once per flap
    forever.
  * `_last_alerted_level` IS UNKNOWN -- a blind alert told the operator
    nothing about severity, so it compares as severity 0. Any danger level
    is therefore new information and is heard; an all-clear is not, and
    waits for the clock.

Consequence worth stating so it is not mistaken for a sixth leak: the ladder
re-arms every time the clock opens. A blind/danger flap therefore costs up to
one clock-opened page (the sensor-blind) plus one ladder ascent (the recovery
to a danger level, which outranks the severity-0 UNKNOWN just alerted) per
min_realert_s window -- at most `1 + ladder` pages per WINDOW, never per flap.
That count is flat in the flap rate and in the run length; only the number of
windows moves it, which is exactly what min_realert_s is for.

Two pieces of state, and the difference between them is load-bearing:
`self.level` is what the world IS (it advances on adoption, even when the
emission is suppressed -- otherwise a suppressed blind never enters
UNKNOWN and recovery is never detected); `_last_alerted_level` is what the
operator has been TOLD. Only the second one gates.

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

    # --- the re-alert invariant: exactly two predicates, no reason cases ---

    def _realert_elapsed(self, t: float) -> bool:
        """Has min_realert_s passed since the operator was last paged?"""
        return (
            self._last_alert_t is None
            or (t - self._last_alert_t) >= self.min_realert_s
        )

    def _outranks_last_alert(self, adopted: RiskLevel) -> bool:
        """Is `adopted` strictly worse news than what the operator was told?

        The ONLY clock-independent reason to page. The UNKNOWN rules here are
        the decisions described in the module docstring; keep the two in sync.
        """
        last = self._last_alerted_level
        if last is None:
            # Nothing has ever been alerted: the first event always fires.
            return True
        if adopted is RiskLevel.UNKNOWN:
            # Blindness is off the severity scale: news exactly once, then
            # time-gated, or a blind/fine flap pages once per flap forever.
            return not self._blind_alerted
        # A blind alert conveyed no severity, so it ranks as the floor: any
        # danger level outranks it, an all-clear does not. (severity(UNKNOWN)
        # is None and must never reach the comparison below.)
        last_severity = 0 if last is RiskLevel.UNKNOWN else severity(last)
        return severity(adopted) > last_severity

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

        # THE RE-ALERT GATE. One decision, no per-reason cases -- see the
        # module docstring. `reason` is a label and must not appear here.
        if not (self._realert_elapsed(t) or self._outranks_last_alert(adopted)):
            return None

        # Record what the operator has now been told. Keyed on the LEVEL, not
        # on `reason` -- the label must have no mechanical role anywhere, or it
        # starts drifting back into being the thing that decides.
        self._last_alerted_level = adopted
        self._last_alert_t = t
        if adopted is RiskLevel.UNKNOWN:
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
