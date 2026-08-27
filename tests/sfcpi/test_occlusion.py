import numpy as np
import pytest
from sfcpi.occlusion import (
    DetectionRatePoint, detection_rate_curve, is_monotonic_decreasing,
)

def _points(ratios_by_count):
    return [
        DetectionRatePoint(true_count=c, detected_count=int(round(c * r)),
                           density_proxy=float(c), ratio=r)
        for c, r in ratios_by_count
    ]

def test_ratio_computed_from_counts():
    p = DetectionRatePoint.from_counts(true_count=100, detected_count=40, density_proxy=1.0)
    assert p.ratio == pytest.approx(0.4)

def test_zero_true_count_gives_nan_ratio():
    p = DetectionRatePoint.from_counts(true_count=0, detected_count=0, density_proxy=0.0)
    assert np.isnan(p.ratio)

def test_curve_bins_and_averages():
    pts = _points([(10, 0.9), (12, 0.9), (100, 0.3), (110, 0.3)])
    df = detection_rate_curve(pts, n_bins=2)
    assert list(df.columns) == ["bin_center", "mean_ratio", "std_ratio", "n"]
    assert len(df) == 2
    assert df["mean_ratio"].iloc[0] == pytest.approx(0.9)
    assert df["mean_ratio"].iloc[-1] == pytest.approx(0.3)

def test_monotonic_decreasing_detects_clean_degradation():
    df = detection_rate_curve(_points([(10, 0.95), (50, 0.6), (100, 0.3), (150, 0.15)]), n_bins=4)
    assert is_monotonic_decreasing(df) is True

def test_noisy_curve_is_rejected():
    df = detection_rate_curve(_points([(10, 0.3), (50, 0.9), (100, 0.2), (150, 0.95)]), n_bins=4)
    assert is_monotonic_decreasing(df) is False

def test_empty_points_raise():
    with pytest.raises(ValueError, match="no points"):
        detection_rate_curve([], n_bins=4)
