# Cycle 3 — Risk Model and SNS Alerting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Turn the per-frame metric stream into a small number of trustworthy alerts and deliver them via AWS SNS.

**Architecture:** A pure `src/sfcpi/risk/` package (hysteresis + dwell + re-alert state machine, no I/O) feeding a list of interchangeable `AlertSink`s. `LogSink` always works; `SnsSink` wraps an INJECTED boto3 client so every test runs without AWS.

**Tech Stack:** Python 3.10, numpy, boto3 1.42 (already installed), pytest 6.2.

**Spec:** `docs/superpowers/specs/2026-08-28-cycle3-risk-alerting-design.md`

## Global Constraints

- `src/sfcpi/risk/` MUST be pure: no I/O, no boto3, no cv2. It is what makes the alerting logic testable without AWS.
- **NaN/absent pressure maps to `UNKNOWN`, NEVER to `NORMAL`.** Cycle 1 emits NaN when sensing fails; mapping that to NORMAL is the safety-inverted failure this project exists to avoid.
- Hysteresis: every fall threshold must be strictly BELOW its rise threshold. Validate at construction.
- The `0.02 s^-2` literature reference (Johansson/Helbing, turbulence onset) is a DEFAULT, never a hardcoded constant — it is R-dependent.
- `SnsSink` takes an injected client. It must NEVER construct a boto3 client itself in tests.
- Publish failures are caught, logged and counted — a dropped alert is bad, a crashed monitor is worse.
- Package root `src/sfcpi/`, tests `tests/sfcpi/`. Run `python3 -m pytest`.
- Stage only your own paths; never `git add -A` (an untracked `webrtc` symlink must never be committed).
- Commit after every task.

---

### Task 1: Risk levels, thresholds, and hysteresis classification

**Files:** Create `src/sfcpi/risk/__init__.py`, `src/sfcpi/risk/levels.py`; Test `tests/sfcpi/test_risk_levels.py`

**Interfaces produced:**
- `RiskLevel` enum: `UNKNOWN, NORMAL, ELEVATED, HIGH, CRITICAL`
- `severity(level) -> int | None` (None for UNKNOWN)
- `Thresholds` frozen dataclass with `elevated_rise/fall`, `high_rise/fall`, `critical_rise/fall`
- `classify(pressure, thresholds, current=RiskLevel.NORMAL) -> RiskLevel`

- [ ] **Step 1: Write the failing test**

```python
# tests/sfcpi/test_risk_levels.py
import math
import pytest
from sfcpi.risk.levels import RiskLevel, Thresholds, classify, severity

T = Thresholds()  # defaults

def test_nan_is_unknown_never_normal():
    """The whole point: a failed measurement must not read as safe."""
    assert classify(float("nan"), T) is RiskLevel.UNKNOWN
    assert classify(None, T) is RiskLevel.UNKNOWN
    assert classify(float("inf"), T) is RiskLevel.UNKNOWN

def test_unknown_has_no_severity():
    assert severity(RiskLevel.UNKNOWN) is None
    assert severity(RiskLevel.NORMAL) == 0
    assert severity(RiskLevel.CRITICAL) == 3

def test_ascending_thresholds():
    assert classify(0.0, T) is RiskLevel.NORMAL
    assert classify(T.elevated_rise, T) is RiskLevel.ELEVATED
    assert classify(T.high_rise, T) is RiskLevel.HIGH
    assert classify(T.critical_rise, T) is RiskLevel.CRITICAL

def test_hysteresis_holds_level_between_fall_and_rise():
    """Between fall and rise the level STAYS — this is what stops flapping."""
    mid = (T.high_fall + T.high_rise) / 2
    assert classify(mid, T, current=RiskLevel.HIGH) is RiskLevel.HIGH
    assert classify(mid, T, current=RiskLevel.NORMAL) is RiskLevel.ELEVATED

def test_drops_only_below_fall_threshold():
    assert classify(T.high_fall - 1e-6, T, current=RiskLevel.HIGH) is not RiskLevel.HIGH

def test_current_unknown_is_treated_as_normal_for_hysteresis():
    """Coming back from blind must not inherit a stale high level."""
    mid = (T.high_fall + T.high_rise) / 2
    assert classify(mid, T, current=RiskLevel.UNKNOWN) is RiskLevel.ELEVATED

def test_thresholds_reject_fall_above_rise():
    with pytest.raises(ValueError, match="fall"):
        Thresholds(high_rise=0.02, high_fall=0.03)

def test_default_high_rise_matches_literature_reference():
    assert T.high_rise == pytest.approx(0.02)
```

- [ ] **Step 2: Run test to verify it fails** — `python3 -m pytest tests/sfcpi/test_risk_levels.py -v` → `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/sfcpi/risk/__init__.py
from .levels import RiskLevel, Thresholds, classify, severity

__all__ = ["RiskLevel", "Thresholds", "classify", "severity"]
```

```python
# src/sfcpi/risk/levels.py
"""Risk levels and hysteresis classification. Pure: no I/O."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RiskLevel(Enum):
    UNKNOWN = "unknown"
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


_SEVERITY = {
    RiskLevel.NORMAL: 0,
    RiskLevel.ELEVATED: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


def severity(level: RiskLevel) -> Optional[int]:
    """Ordinal severity, or None for UNKNOWN.

    UNKNOWN is deliberately unordered: it is the absence of a measurement,
    not a position on the scale.
    """
    return _SEVERITY.get(level)


@dataclass(frozen=True)
class Thresholds:
    """Rise/fall pairs in s^-2. Defaults reference Johansson/Helbing's
    turbulence onset at 0.02 s^-2 -- a DEFAULT, not a constant (it is
    R-dependent). Every fall must sit strictly below its rise."""

    elevated_rise: float = 0.010
    elevated_fall: float = 0.008
    high_rise: float = 0.020
    high_fall: float = 0.016
    critical_rise: float = 0.040
    critical_fall: float = 0.032

    def __post_init__(self) -> None:
        for name in ("elevated", "high", "critical"):
            rise = getattr(self, f"{name}_rise")
            fall = getattr(self, f"{name}_fall")
            if not fall < rise:
                raise ValueError(
                    f"{name}_fall ({fall}) must be strictly below {name}_rise ({rise}); "
                    "equal or inverted thresholds defeat hysteresis"
                )


def classify(
    pressure, thresholds: Thresholds, current: RiskLevel = RiskLevel.NORMAL
) -> RiskLevel:
    """Map a pressure to a level, holding `current` inside the hysteresis band.

    A non-finite or absent pressure is UNKNOWN -- never NORMAL.
    """
    if pressure is None or not math.isfinite(float(pressure)):
        return RiskLevel.UNKNOWN

    p = float(pressure)
    if p >= thresholds.critical_rise:
        return RiskLevel.CRITICAL
    if p >= thresholds.high_rise:
        return RiskLevel.HIGH
    if p >= thresholds.elevated_rise:
        return RiskLevel.ELEVATED

    # Below every rise threshold: hold the current level until its fall threshold.
    # UNKNOWN carries no level to hold, so it decays to NORMAL rather than
    # inheriting a stale severity from before the sensor went blind.
    held = severity(current)
    if held is None:
        held = 0
    if held >= 3 and p >= thresholds.critical_fall:
        return RiskLevel.CRITICAL
    if held >= 2 and p >= thresholds.high_fall:
        return RiskLevel.HIGH
    if held >= 1 and p >= thresholds.elevated_fall:
        return RiskLevel.ELEVATED
    return RiskLevel.NORMAL
```

- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit** — `git add src/sfcpi/risk tests/sfcpi/test_risk_levels.py && git commit -m "feat(sfcpi): risk levels with hysteresis classification"`

---

### Task 2: RiskEvent and the state machine (dwell + transitions)

**Files:** Create `src/sfcpi/risk/events.py`, `src/sfcpi/risk/machine.py`; Test `tests/sfcpi/test_risk_machine.py`

**Interfaces consumed:** `RiskLevel`, `Thresholds`, `classify`, `severity` (Task 1)
**Interfaces produced:**
- `RiskEvent` frozen dataclass: `timestamp, level, previous_level, pressure, coverage, reason, message`
- `RiskStateMachine(thresholds, min_dwell_s=2.0, min_realert_s=60.0, blind_alert_s=30.0)`
- `.update(timestamp: float, pressure, coverage=None) -> RiskEvent | None`

- [ ] **Step 1: Write the failing test**

```python
# tests/sfcpi/test_risk_machine.py
import pytest
from sfcpi.risk.levels import RiskLevel, Thresholds
from sfcpi.risk.machine import RiskStateMachine
from sfcpi.risk.events import RiskEvent

T = Thresholds()

def _machine(**kw):
    kw.setdefault("min_dwell_s", 2.0)
    kw.setdefault("min_realert_s", 60.0)
    return RiskStateMachine(thresholds=T, **kw)

def test_starts_normal_and_silent():
    m = _machine()
    assert m.level is RiskLevel.NORMAL
    assert m.update(0.0, 0.0) is None

def test_single_frame_spike_is_suppressed_by_dwell():
    """One artefact frame must not raise an alert."""
    m = _machine()
    assert m.update(0.0, 0.0) is None
    assert m.update(0.1, 0.05) is None       # spike, but dwell not met
    assert m.update(0.2, 0.0) is None        # gone again
    assert m.level is RiskLevel.NORMAL

def test_sustained_rise_fires_once_dwell_is_met():
    m = _machine()
    m.update(0.0, 0.0)
    assert m.update(1.0, 0.025) is None      # candidate HIGH, dwell pending
    ev = m.update(3.0, 0.025)                # 2.0s sustained
    assert isinstance(ev, RiskEvent)
    assert ev.level is RiskLevel.HIGH
    assert ev.previous_level is RiskLevel.NORMAL
    assert ev.reason == "escalation"

def test_flapping_around_threshold_produces_one_event_not_many():
    """The core anti-spam property."""
    m = _machine()
    m.update(0.0, 0.0)
    events = []
    t = 1.0
    for i in range(40):
        p = 0.021 if i % 2 == 0 else 0.019   # straddles high_rise, stays above high_fall
        ev = m.update(t, p)
        if ev is not None:
            events.append(ev)
        t += 0.5
    assert len(events) == 1, f"expected 1 alert, got {len(events)}"

def test_pressure_and_coverage_ride_on_the_event():
    m = _machine()
    m.update(0.0, 0.0)
    m.update(1.0, 0.025, coverage=0.8)
    ev = m.update(3.0, 0.025, coverage=0.8)
    assert ev.pressure == pytest.approx(0.025)
    assert ev.coverage == pytest.approx(0.8)

def test_de_escalation_is_reported():
    m = _machine()
    m.update(0.0, 0.0)
    m.update(1.0, 0.025)
    m.update(3.0, 0.025)                      # -> HIGH
    m.update(4.0, 0.0)
    ev = m.update(6.5, 0.0)
    assert ev is not None and ev.level is RiskLevel.NORMAL
    assert ev.reason == "de-escalation"

def test_clock_going_backwards_resets_timers_without_negative_dwell():
    """Replay restart must not emit a negative-duration transition."""
    m = _machine()
    m.update(100.0, 0.0)
    m.update(101.0, 0.025)
    assert m.update(0.0, 0.025) is None        # clock reset: dwell restarts
    assert m.update(2.5, 0.025) is not None
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement**

```python
# src/sfcpi/risk/events.py
"""The alert payload. Pure data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .levels import RiskLevel


@dataclass(frozen=True)
class RiskEvent:
    timestamp: float
    level: RiskLevel
    previous_level: RiskLevel
    pressure: Optional[float]
    coverage: Optional[float]
    reason: str          # escalation | de-escalation | sustained | sensor-blind
    message: str
```

```python
# src/sfcpi/risk/machine.py
"""Hysteresis + dwell + re-alert state machine. Pure: no I/O.

Three mechanisms, all required, because a raw threshold on a noisy signal
flaps and at 15 fps that is tens of alerts a second:
  hysteresis  - separate rise/fall thresholds (in levels.classify)
  dwell       - a candidate level must persist before it is adopted
  re-alert    - a level already alerted does not re-alert until it escalates
                or min_realert_s has elapsed
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
        self._candidate: Optional[RiskLevel] = None
        self._candidate_since: Optional[float] = None
        self._last_alert_t: Optional[float] = None
        self._last_t: Optional[float] = None
        self._blind_since: Optional[float] = None

    def _reset_timers(self) -> None:
        self._candidate = None
        self._candidate_since = None
        self._blind_since = None

    def update(
        self, timestamp: float, pressure, coverage: Optional[float] = None
    ) -> Optional[RiskEvent]:
        t = float(timestamp)

        # A clock that moves backwards means the source restarted. Restart the
        # timers rather than computing a negative dwell.
        if self._last_t is not None and t < self._last_t:
            self._reset_timers()
            self._last_alert_t = None
        self._last_t = t

        candidate = classify(pressure, self.thresholds, current=self.level)

        if candidate is not self._candidate:
            self._candidate = candidate
            self._candidate_since = t

        if candidate is self.level:
            self._blind_since = None if candidate is not RiskLevel.UNKNOWN else self._blind_since
            return None

        held_for = t - (self._candidate_since if self._candidate_since is not None else t)
        if held_for < self.min_dwell_s:
            return None

        previous = self.level
        self.level = candidate
        self._candidate_since = t

        old, new = severity(previous), severity(candidate)
        if old is None or new is None:
            reason = "sensor-blind" if candidate is RiskLevel.UNKNOWN else "escalation"
        elif new > old:
            reason = "escalation"
        elif new < old:
            reason = "de-escalation"
        else:  # pragma: no cover - equal severities cannot differ in level
            reason = "sustained"

        escalating = (old is not None and new is not None and new > old) or reason == "sensor-blind"
        if not escalating and self._last_alert_t is not None:
            if (t - self._last_alert_t) < self.min_realert_s:
                return None

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
```

- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit** — `git add src/sfcpi/risk/events.py src/sfcpi/risk/machine.py tests/sfcpi/test_risk_machine.py && git commit -m "feat(sfcpi): risk state machine with hysteresis and dwell"`

---

### Task 3: Sensor-blind alerting and the re-alert interval

**Files:** Modify `src/sfcpi/risk/machine.py`; Test `tests/sfcpi/test_risk_blind.py`

**Interfaces consumed:** everything from Task 2. **Produces:** no new API — behaviour only.

- [ ] **Step 1: Write the failing test**

```python
# tests/sfcpi/test_risk_blind.py
import pytest
from sfcpi.risk.levels import RiskLevel, Thresholds
from sfcpi.risk.machine import RiskStateMachine

T = Thresholds()

def _m(**kw):
    kw.setdefault("min_dwell_s", 2.0)
    kw.setdefault("min_realert_s", 60.0)
    kw.setdefault("blind_alert_s", 10.0)
    return RiskStateMachine(thresholds=T, **kw)

def test_nan_becomes_unknown_and_alerts_sensor_blind():
    """A blind sensor at a mass gathering is itself an incident."""
    m = _m()
    m.update(0.0, 0.0)
    assert m.update(1.0, float("nan")) is None       # dwell pending
    ev = m.update(3.5, float("nan"))
    assert ev is not None
    assert ev.level is RiskLevel.UNKNOWN
    assert ev.reason == "sensor-blind"
    assert ev.pressure is None

def test_unknown_never_reports_as_normal():
    m = _m()
    m.update(0.0, 0.0)
    m.update(1.0, float("nan"))
    ev = m.update(3.5, float("nan"))
    assert ev.level is not RiskLevel.NORMAL

def test_recovery_from_blind_is_reported():
    m = _m()
    m.update(0.0, 0.0)
    m.update(1.0, float("nan"))
    m.update(3.5, float("nan"))                       # -> UNKNOWN
    m.update(4.0, 0.0)
    ev = m.update(6.5, 0.0)
    assert ev is not None and ev.level is RiskLevel.NORMAL

def test_escalation_bypasses_the_realert_interval():
    """Getting worse must always be heard immediately."""
    m = _m(min_realert_s=600.0)
    m.update(0.0, 0.0)
    m.update(1.0, 0.025); first = m.update(3.0, 0.025)      # -> HIGH
    assert first is not None
    m.update(4.0, 0.05); second = m.update(6.5, 0.05)       # -> CRITICAL, still inside 600s
    assert second is not None and second.level is RiskLevel.CRITICAL

def test_de_escalation_respects_the_realert_interval():
    m = _m(min_realert_s=600.0)
    m.update(0.0, 0.0)
    m.update(1.0, 0.025); m.update(3.0, 0.025)              # -> HIGH, alert fired
    m.update(4.0, 0.0)
    assert m.update(6.5, 0.0) is None                        # calmer, but too soon to re-alert
```

- [ ] **Step 2: Run test to verify it fails** (some will fail against Task 2's machine)

- [ ] **Step 3: Adjust `machine.py` until all pass.** The Task 2 implementation already routes UNKNOWN through `reason == "sensor-blind"` and treats it as escalating; verify `blind_alert_s` is honoured as an additional dwell for the UNKNOWN transition specifically, and that recovery from UNKNOWN uses the normal dwell. Do not duplicate the dwell logic — extend it.

- [ ] **Step 4: Run the full suite** — `python3 -m pytest tests/sfcpi/ -q`
- [ ] **Step 5: Commit**

---

### Task 4: Alert sink protocol and LogSink

**Files:** Create `src/sfcpi/alerts/__init__.py`, `src/sfcpi/alerts/base.py`, `src/sfcpi/alerts/log.py`; Test `tests/sfcpi/test_alert_log.py`

**Interfaces produced:**
- `AlertSink` Protocol with `publish(event: RiskEvent) -> None`
- `LogSink(stream=None)` writing one line per event; `.count` of published events
- `RecordingSink()` for tests, exposing `.events`

- [ ] **Step 1: Write the failing test**

```python
# tests/sfcpi/test_alert_log.py
import io
from sfcpi.alerts.log import LogSink, RecordingSink
from sfcpi.risk.events import RiskEvent
from sfcpi.risk.levels import RiskLevel

def _ev(level=RiskLevel.HIGH):
    return RiskEvent(timestamp=1.5, level=level, previous_level=RiskLevel.NORMAL,
                     pressure=0.025, coverage=0.9, reason="escalation", message="normal -> high")

def test_log_sink_writes_one_line_per_event():
    buf = io.StringIO()
    sink = LogSink(stream=buf)
    sink.publish(_ev()); sink.publish(_ev(RiskLevel.CRITICAL))
    assert len(buf.getvalue().strip().splitlines()) == 2
    assert sink.count == 2

def test_log_line_names_level_and_pressure():
    buf = io.StringIO()
    LogSink(stream=buf).publish(_ev())
    line = buf.getvalue()
    assert "high" in line.lower() and "0.025" in line

def test_recording_sink_collects_events():
    sink = RecordingSink()
    sink.publish(_ev())
    assert len(sink.events) == 1 and sink.events[0].level is RiskLevel.HIGH
```

- [ ] **Step 2: Run to verify it fails**

- [ ] **Step 3: Implement**

```python
# src/sfcpi/alerts/__init__.py
from .base import AlertSink
from .log import LogSink, RecordingSink

__all__ = ["AlertSink", "LogSink", "RecordingSink"]
```

```python
# src/sfcpi/alerts/base.py
"""Delivery is a sink LIST, not a branch -- so adding a transport touches no logic."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..risk.events import RiskEvent


@runtime_checkable
class AlertSink(Protocol):
    def publish(self, event: RiskEvent) -> None:  # pragma: no cover - protocol
        ...
```

```python
# src/sfcpi/alerts/log.py
"""Sinks that need no external service."""
from __future__ import annotations

import sys
from typing import List, Optional, TextIO

from ..risk.events import RiskEvent


class LogSink:
    """Always-available sink. The default, and the fallback when SNS is off."""

    def __init__(self, stream: Optional[TextIO] = None) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self.count = 0

    def publish(self, event: RiskEvent) -> None:
        pressure = "n/a" if event.pressure is None else f"{event.pressure:.4f}"
        self._stream.write(
            f"[ALERT {event.level.value.upper()}] t={event.timestamp:.2f} "
            f"pressure={pressure} reason={event.reason} :: {event.message}\n"
        )
        self._stream.flush()
        self.count += 1


class RecordingSink:
    """Captures events in memory. For tests and for a future dashboard feed."""

    def __init__(self) -> None:
        self.events: List[RiskEvent] = []

    def publish(self, event: RiskEvent) -> None:
        self.events.append(event)
```

- [ ] **Step 4: Run to verify it passes**
- [ ] **Step 5: Commit**

---

### Task 5: SnsSink

**Files:** Create `src/sfcpi/alerts/sns.py`; Test `tests/sfcpi/test_alert_sns.py`

**Interfaces produced:** `SnsSink(topic_arn, client, dry_run=False)`; attributes `.count`, `.failures`, `.dry_run`

- [ ] **Step 1: Write the failing test**

```python
# tests/sfcpi/test_alert_sns.py
import pytest
from sfcpi.alerts.sns import SnsSink
from sfcpi.risk.events import RiskEvent
from sfcpi.risk.levels import RiskLevel

ARN = "arn:aws:sns:ap-south-1:123456789012:crowd-alerts"

class _StubClient:
    def __init__(self, raises=None):
        self.calls = []
        self._raises = raises
    def publish(self, **kwargs):
        if self._raises:
            raise self._raises
        self.calls.append(kwargs)
        return {"MessageId": "stub-1"}

def _ev(level=RiskLevel.HIGH):
    return RiskEvent(timestamp=1.5, level=level, previous_level=RiskLevel.NORMAL,
                     pressure=0.025, coverage=0.9, reason="escalation", message="normal -> high")

def test_publishes_once_per_event():
    c = _StubClient()
    s = SnsSink(topic_arn=ARN, client=c)
    s.publish(_ev())
    assert len(c.calls) == 1
    assert c.calls[0]["TopicArn"] == ARN
    assert s.count == 1 and s.failures == 0

def test_subject_and_body_name_the_level_and_pressure():
    c = _StubClient()
    SnsSink(topic_arn=ARN, client=c).publish(_ev(RiskLevel.CRITICAL))
    call = c.calls[0]
    assert "CRITICAL" in call["Subject"]
    assert "0.025" in call["Message"]
    assert len(call["Subject"]) <= 100      # SNS hard limit

def test_publish_failure_is_counted_not_raised():
    """A dropped alert is bad; a crashed monitor is worse."""
    s = SnsSink(topic_arn=ARN, client=_StubClient(raises=RuntimeError("denied")))
    s.publish(_ev())                          # must not raise
    assert s.count == 0 and s.failures == 1

def test_dry_run_publishes_nothing_but_counts():
    c = _StubClient()
    s = SnsSink(topic_arn=ARN, client=c, dry_run=True)
    s.publish(_ev())
    assert c.calls == [] and s.count == 1

def test_missing_topic_arn_fails_at_construction():
    with pytest.raises(ValueError, match="topic_arn"):
        SnsSink(topic_arn="", client=_StubClient())

def test_unknown_level_event_still_publishes():
    c = _StubClient()
    ev = RiskEvent(timestamp=2.0, level=RiskLevel.UNKNOWN, previous_level=RiskLevel.NORMAL,
                   pressure=None, coverage=0.0, reason="sensor-blind", message="sensor blind")
    SnsSink(topic_arn=ARN, client=c).publish(ev)
    assert len(c.calls) == 1 and "n/a" in c.calls[0]["Message"]
```

- [ ] **Step 2: Run to verify it fails**

- [ ] **Step 3: Implement**

```python
# src/sfcpi/alerts/sns.py
"""AWS SNS delivery.

The boto3 client is INJECTED, never constructed here -- that is what lets every
test run without AWS credentials or network.
"""
from __future__ import annotations

import logging
from typing import Any

from ..risk.events import RiskEvent

logger = logging.getLogger(__name__)

_SUBJECT_MAX = 100  # SNS hard limit


class SnsSink:
    def __init__(self, topic_arn: str, client: Any, dry_run: bool = False) -> None:
        if not topic_arn:
            raise ValueError("topic_arn is required; refusing to start with SNS enabled and no topic")
        if client is None:
            raise ValueError("client is required; construct boto3.client('sns') at the call site")
        self.topic_arn = topic_arn
        self.client = client
        self.dry_run = dry_run
        self.count = 0
        self.failures = 0

    def _subject(self, event: RiskEvent) -> str:
        return f"[CrowdSentinel] {event.level.value.upper()}"[:_SUBJECT_MAX]

    def _body(self, event: RiskEvent) -> str:
        pressure = "n/a" if event.pressure is None else f"{event.pressure:.4f} s^-2"
        coverage = "n/a" if event.coverage is None else f"{event.coverage:.0%}"
        return (
            f"Risk level: {event.previous_level.value} -> {event.level.value}\n"
            f"Reason:     {event.reason}\n"
            f"Pressure:   {pressure}\n"
            f"Coverage:   {coverage}\n"
            f"Timestamp:  {event.timestamp:.2f}\n\n{event.message}\n"
        )

    def publish(self, event: RiskEvent) -> None:
        if self.dry_run:
            logger.info("[dry-run] would publish: %s", self._subject(event))
            self.count += 1
            return
        try:
            self.client.publish(
                TopicArn=self.topic_arn,
                Subject=self._subject(event),
                Message=self._body(event),
            )
        except Exception as exc:                      # noqa: BLE001 - never kill the monitor
            self.failures += 1
            logger.error("SNS publish failed (%s): %s", type(exc).__name__, exc)
            return
        self.count += 1
```

- [ ] **Step 4: Run to verify it passes**
- [ ] **Step 5: Commit**

---

### Task 6: `sfcpi watch` CLI and end-to-end wiring

**Files:** Modify `src/sfcpi/cli.py`; Test `tests/sfcpi/test_watch_cli.py`

**Interfaces consumed:** all of the above, plus cycle 1's JSONL metrics format.
**Produces:** `sfcpi watch <metrics.jsonl>` replaying a metrics file through the state machine into sinks.

- [ ] **Step 1: Write the failing test**

```python
# tests/sfcpi/test_watch_cli.py
import json
from sfcpi.cli import main

def _write_metrics(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for t, p in rows:
            fh.write(json.dumps({
                "index": int(t * 10), "timestamp": t, "flow_valid": True,
                "global_pressure": p, "global_max_pressure": p,
                "total_count": 5.0, "sensing_confidence": 1.0, "cell_pressure": [[p]],
            }) + "\n")

def test_watch_reports_alerts_and_exits_zero(tmp_path, capsys):
    path = tmp_path / "m.jsonl"
    _write_metrics(path, [(i * 0.5, 0.0 if i < 4 else 0.03) for i in range(20)])
    assert main(["watch", str(path)]) == 0
    err = capsys.readouterr().err
    assert "ALERT" in err and "HIGH" in err.upper()

def test_watch_is_quiet_on_calm_metrics(tmp_path, capsys):
    path = tmp_path / "m.jsonl"
    _write_metrics(path, [(i * 0.5, 0.0) for i in range(20)])
    assert main(["watch", str(path)]) == 0
    assert "ALERT" not in capsys.readouterr().err

def test_null_pressure_raises_sensor_blind_not_silence(tmp_path, capsys):
    """A null in the metrics file means the sensor failed -- it must be heard."""
    path = tmp_path / "m.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(40):
            fh.write(json.dumps({
                "index": i, "timestamp": i * 0.5, "flow_valid": True,
                "global_pressure": None, "global_max_pressure": None,
                "total_count": None, "sensing_confidence": 0.0, "cell_pressure": [[None]],
            }) + "\n")
    assert main(["watch", str(path)]) == 0
    assert "sensor-blind" in capsys.readouterr().err

def test_sns_requires_a_topic_arn(tmp_path, capsys):
    path = tmp_path / "m.jsonl"
    _write_metrics(path, [(0.0, 0.0)])
    assert main(["watch", str(path), "--sns"]) != 0
    assert "topic" in capsys.readouterr().err.lower()
```

- [ ] **Step 2: Run to verify it fails**

- [ ] **Step 3: Add `_cmd_watch` to `cli.py` and register it in `build_parser()`**, following the existing one-subparser-per-command shape. Flags: `--threshold-high` (default 0.02), `--min-dwell`, `--min-realert`, `--score-field` (default `global_max_pressure`), `--sns`, `--sns-topic-arn`, `--sns-region`, `--dry-run` (DEFAULT TRUE when `--sns` is given; require an explicit `--no-dry-run` to actually publish). Read `null` back as NaN exactly as `_cmd_eval` does. Construct `boto3.client("sns", region_name=...)` ONLY inside `_cmd_watch`, never in the sink. Exit non-zero with a message naming the missing setting if `--sns` is given without a topic ARN.

- [ ] **Step 4: Run the full suite**
- [ ] **Step 5: Update `src/sfcpi/README.md`** with the `watch` command, the dry-run default, and a one-line note that `sns:Publish` on the topic ARN is required.
- [ ] **Step 6: Commit**

---

## Deferred to cycle 2 / later

- WebRTC live ingest (`WebRTCSource`) — dependencies now installed (aiortc 1.15, av 17.1, websockets 16.1).
- Per-cell / zone alerting; alert acknowledgement; dashboard UI.
- The parked cycle-1 finding: `nanmean`/`nanmax` over a partially-NaN grid masking a partly-blind sensor — becomes reachable once per-cell sensing confidence exists.
