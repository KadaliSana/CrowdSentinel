import math
import numpy as np
import pytest
from sfcpi.metrics.pressure import (
    crowd_pressure, density_px, velocity_variance, mean_speed,
)

def _pressure_in_metres(count, cell_area_px, speed_var_px, s, fps):
    """Reference: compute P in real units given a metres-per-pixel scale s."""
    rho_m = count / (s ** 2 * cell_area_px)          # people / m^2
    var_m = (s * fps) ** 2 * speed_var_px            # (m/s)^2
    return rho_m * var_m

@pytest.mark.parametrize("s", [0.004])
def test_pixel_formula_equals_metric_formula(s):
    """crowd_pressure(pixel-space inputs) equals the metric-space formula
    computed via an assumed metres-per-pixel scale s. This checks the two
    formulas agree; it does NOT by itself test invariance to s (see
    test_pressure_invariant_to_render_resolution for that)."""
    count, area, var, fps = 37, 4096.0, 0.83, 15.0
    assert crowd_pressure(count, area, var, fps) == pytest.approx(
        _pressure_in_metres(count, area, var, s, fps), rel=1e-9
    )

@pytest.mark.parametrize("k", [0.5, 1.0, 2.0, 7.3])
def test_pressure_invariant_to_render_resolution(k):
    """Same scene at k x resolution must give the same pressure.

    area scales k^2; px/frame speed scales k, so its variance scales k^2.
    """
    count, area, var, fps = 37, 4096.0, 0.83, 15.0
    base = crowd_pressure(count, area, var, fps)
    scaled = crowd_pressure(count, (k ** 2) * area, (k ** 2) * var, fps)
    assert scaled == pytest.approx(base, rel=1e-12)

def test_pressure_scales_with_fps_squared():
    base = crowd_pressure(10, 100.0, 2.0, 10.0)
    doubled = crowd_pressure(10, 100.0, 2.0, 20.0)
    assert doubled == pytest.approx(4 * base)

def test_uniform_motion_gives_zero_pressure():
    """Everyone moving identically is not dangerous: variance, not speed."""
    flow = np.full((8, 8, 2), 3.0)
    assert velocity_variance(flow) == pytest.approx(0.0)
    assert crowd_pressure(50, 64.0, velocity_variance(flow), 15.0) == pytest.approx(0.0)

def test_zero_motion_gives_zero_pressure():
    flow = np.zeros((8, 8, 2))
    assert mean_speed(flow) == pytest.approx(0.0)
    assert crowd_pressure(50, 64.0, velocity_variance(flow), 15.0) == pytest.approx(0.0)

def test_counterflow_gives_large_variance_not_zero():
    """Perfect counterflow (+5, -5) is the turbulent regime where crowd
    pressure PEAKS -- it must not report as zero.

    Source: Johansson, Helbing, Al-Abideen & Al-Bosta, "From Crowd Dynamics to
    Crowd Safety: A Video-Based Analysis" (Advances in Complex Systems, 2008),
    conclusions: "The 'pressure', defined as the density times the VARIANCE OF
    VELOCITIES, provides better and more specific information about critical
    areas and times."

    The variance of speed MAGNITUDES would be 0 here (|+5| == |-5|), which is
    precisely the defect this test guards. Var(vx) = 25, Var(vy) = 0 -> 25.
    """
    flow = np.zeros((2, 1, 2))
    flow[0, 0] = (5.0, 0.0)
    flow[1, 0] = (-5.0, 0.0)
    assert velocity_variance(flow) == pytest.approx(25.0)
    assert crowd_pressure(50, 64.0, velocity_variance(flow), 15.0) > 0.0

def test_differing_velocities_give_positive_variance():
    flow = np.zeros((2, 1, 2))
    flow[0, 0] = (0.0, 0.0)
    flow[1, 0] = (4.0, 0.0)
    assert velocity_variance(flow) == pytest.approx(4.0)

def test_density_px():
    assert density_px(20, 100.0) == pytest.approx(0.2)

def test_nan_count_propagates_as_nan_not_zero():
    """Fail-loud: unknown count must not read as 'safe'."""
    assert math.isnan(crowd_pressure(float("nan"), 100.0, 1.0, 15.0))

def test_zero_area_is_nan_not_inf():
    assert math.isnan(crowd_pressure(10, 0.0, 1.0, 15.0))

def test_empty_cell_variance_is_nan():
    assert math.isnan(velocity_variance(np.zeros((0, 0, 2))))
