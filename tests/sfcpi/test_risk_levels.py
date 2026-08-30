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

def test_current_unknown_is_treated_as_normal_for_hysteresis():
    """Coming back from blind must not inherit a stale high level."""
    mid = (T.high_fall + T.high_rise) / 2
    assert classify(mid, T, current=RiskLevel.UNKNOWN) is RiskLevel.ELEVATED

def test_thresholds_reject_fall_above_rise():
    with pytest.raises(ValueError, match="fall"):
        Thresholds(high_rise=0.02, high_fall=0.03)

def test_default_high_rise_matches_literature_reference():
    assert T.high_rise == pytest.approx(0.02)


def test_thresholds_reject_fall_above_rise_in_every_tier():
    """Only the `high` tier was covered; the same validation must hold for
    the tiers either side of it."""
    with pytest.raises(ValueError, match="elevated_fall"):
        Thresholds(elevated_rise=0.010, elevated_fall=0.012)
    with pytest.raises(ValueError, match="critical_fall"):
        Thresholds(critical_rise=0.040, critical_fall=0.045)
    with pytest.raises(ValueError, match="elevated_fall"):
        Thresholds(elevated_rise=0.010, elevated_fall=0.010)   # equal defeats hysteresis


def test_thresholds_reject_out_of_order_rise_tiers():
    """`--threshold-high 0.05` left critical_rise at 0.040, so p=0.045
    classified CRITICAL while below the user's own HIGH threshold."""
    with pytest.raises(ValueError, match=r"high_rise.*critical_rise"):
        Thresholds(high_rise=0.05, high_fall=0.04)
    with pytest.raises(ValueError, match=r"elevated_rise.*high_rise"):
        Thresholds(elevated_rise=0.030, elevated_fall=0.025)


def test_thresholds_reject_out_of_order_fall_tiers():
    """Rises ordered but falls crossed: the hysteresis bands overlap."""
    with pytest.raises(ValueError, match=r"elevated_fall.*high_fall"):
        Thresholds(elevated_rise=0.018, elevated_fall=0.017)


def test_drops_only_below_fall_threshold():
    """Asserting `is not HIGH` would pass for UNKNOWN or CRITICAL too."""
    assert classify(T.high_fall - 1e-6, T, current=RiskLevel.HIGH) is RiskLevel.ELEVATED
    assert classify(T.high_fall, T, current=RiskLevel.HIGH) is RiskLevel.HIGH


def test_hysteresis_holds_critical_and_elevated_across_their_own_bands():
    """Each tier's band must hold its own level, not just `high`."""
    crit_mid = (T.critical_fall + T.critical_rise) / 2
    assert classify(crit_mid, T, current=RiskLevel.CRITICAL) is RiskLevel.CRITICAL
    assert classify(crit_mid, T, current=RiskLevel.NORMAL) is RiskLevel.HIGH

    elev_mid = (T.elevated_fall + T.elevated_rise) / 2
    assert classify(elev_mid, T, current=RiskLevel.ELEVATED) is RiskLevel.ELEVATED
    assert classify(elev_mid, T, current=RiskLevel.NORMAL) is RiskLevel.NORMAL


def test_multi_tier_cascade_in_a_single_call():
    """A single classify() may cross several tiers at once -- the per-tier
    hold must not pin an intermediate level on the way down."""
    assert classify(0.005, T, current=RiskLevel.CRITICAL) is RiskLevel.NORMAL
    assert classify(0.050, T, current=RiskLevel.NORMAL) is RiskLevel.CRITICAL
    assert classify(0.0, T, current=RiskLevel.HIGH) is RiskLevel.NORMAL
