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
