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
    # 4s elapsed: past min_dwell_s (2.0) but short of blind_alert_s (10.0).
    # Still None here pins that entry to UNKNOWN is gated by blind_alert_s,
    # not by the shorter min_dwell_s -- swapping them would fire here.
    assert m.update(5.0, float("nan")) is None
    ev = m.update(11.5, float("nan"))                # 10.5s elapsed: blind_alert_s met
    assert ev is not None
    assert ev.level is RiskLevel.UNKNOWN
    assert ev.reason == "sensor-blind"
    assert ev.pressure is None

def test_unknown_never_reports_as_normal():
    m = _m()
    m.update(0.0, 0.0)
    m.update(1.0, float("nan"))
    ev = m.update(11.5, float("nan"))
    assert ev.level is not RiskLevel.NORMAL

def test_recovery_from_blind_is_reported():
    """Recovery FROM UNKNOWN uses the normal dwell (min_dwell_s), not
    blind_alert_s -- the extra dwell only applies to going blind, not to
    coming back. min_realert_s is short here so the recovery-to-NORMAL
    gate (see test_recovery_to_normal_is_gated_below) does not swallow it --
    that gate is pinned separately."""
    m = _m(min_realert_s=1.0)
    m.update(0.0, 0.0)
    m.update(1.0, float("nan"))
    m.update(11.5, float("nan"))                     # -> UNKNOWN (blind_alert_s met)
    m.update(12.0, 0.0)
    # 2.5s elapsed: past min_dwell_s (2.0) but well short of blind_alert_s
    # (10.0). Firing here pins that recovery uses min_dwell_s, not
    # blind_alert_s -- swapping them would still be None at this point.
    ev = m.update(14.5, 0.0)
    assert ev is not None and ev.level is RiskLevel.NORMAL
    assert ev.reason == "recovery"

def test_recovery_to_normal_is_gated_by_the_realert_interval():
    """Recovery to NORMAL is the resolution of the prior blind alert -- it
    goes through the same re-alert gate as any other all-clear."""
    m = _m(min_realert_s=600.0)
    m.update(0.0, 0.0)
    m.update(1.0, float("nan"))
    m.update(11.5, float("nan"))                     # -> UNKNOWN, sensor-blind fires
    m.update(12.0, 0.0)
    assert m.update(14.5, 0.0) is None                # recovered, but too soon to re-alert

def test_recovery_to_danger_bypasses_the_realert_interval():
    """A sensor coming back and immediately showing danger must be heard --
    there is no pre-blind severity to compare against, so this is not gated
    the way recovery-to-NORMAL is."""
    m = _m(min_realert_s=600.0)
    m.update(0.0, 0.0)
    m.update(1.0, float("nan"))
    m.update(11.5, float("nan"))                     # -> UNKNOWN, sensor-blind fires
    m.update(12.0, 0.025)                             # candidate HIGH
    ev = m.update(14.0, 0.025)                        # min_dwell_s met
    assert ev is not None
    assert ev.level is RiskLevel.HIGH
    assert ev.reason == "recovery"

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
