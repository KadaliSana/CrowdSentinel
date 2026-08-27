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
    """De-escalation still respects the re-alert interval (see
    test_risk_blind.py::test_de_escalation_respects_the_realert_interval) --
    use a short interval here so the all-clear at t=6.5 (3.5s after the
    alert) genuinely falls outside it, rather than testing an exemption."""
    m = _machine(min_realert_s=1.0)
    m.update(0.0, 0.0)
    m.update(1.0, 0.025)
    m.update(3.0, 0.025)                      # -> HIGH
    m.update(4.0, 0.0)
    ev = m.update(6.5, 0.0)
    assert ev is not None and ev.level is RiskLevel.NORMAL
    assert ev.reason == "de-escalation"

def test_full_swing_oscillation_does_not_realert_at_the_same_level():
    """HIGH -> NORMAL -> HIGH -> ... must not re-alert HIGH every few seconds.
    Suppressing only de-escalation left min_realert_s dead in this direction."""
    m = _machine(min_dwell_s=2.0, min_realert_s=600.0)
    m.update(0.0, 0.0)
    events, t = [], 1.0
    for cycle in range(6):
        for _ in range(3):                      # ~3s above high_rise
            ev = m.update(t, 0.025); t += 1.0
            if ev: events.append(ev)
        for _ in range(3):                      # ~3s at zero
            ev = m.update(t, 0.0); t += 1.0
            if ev: events.append(ev)
    highs = [e for e in events if e.level is RiskLevel.HIGH]
    assert len(highs) == 1, f"expected 1 HIGH alert, got {len(highs)}"

def test_clock_going_backwards_resets_timers_without_negative_dwell():
    """Replay restart must not emit a negative-duration transition."""
    m = _machine()
    m.update(100.0, 0.0)
    m.update(101.0, 0.025)
    assert m.update(0.0, 0.025) is None        # clock reset: dwell restarts
    assert m.update(2.5, 0.025) is not None


def test_non_finite_timestamp_is_rejected():
    """A NaN t makes `held_for < required_dwell` and `(t - last) < realert`
    both False, defeating the dwell gate AND the re-alert gate at once --
    an alert on every level change, unbounded."""
    m = _machine(min_realert_s=600.0)
    m.update(0.0, 0.0)
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="timestamp"):
            m.update(bad, 0.05)
    # A rejected call must not have disturbed the state it was rejected for.
    assert m.level is RiskLevel.NORMAL
    assert m.update(1.0, 0.05) is None          # dwell still pending
    assert m.update(3.0, 0.05) is not None


def test_min_coverage_must_be_a_fraction():
    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="min_coverage"):
            RiskStateMachine(thresholds=T, min_coverage=bad)
