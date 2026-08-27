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
