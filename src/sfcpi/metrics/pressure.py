"""Crowd pressure and its components.

Crowd pressure P = rho * Var(v) has units s^-2 -- it carries no length
dimension, so the unknown metres-per-pixel scale cancels exactly. P computed
from pixel-space quantities equals P in real units, given only the frame rate.
See docs/superpowers/specs/2026-08-27-sf-cpi-design.md section 2.

Pure numpy. No I/O, no cv2.
"""
from __future__ import annotations

import numpy as np

NAN = float("nan")


def _speeds(flow_cell: np.ndarray) -> np.ndarray:
    """Speed magnitudes (px/frame) for a cell of shape (H, W, 2)."""
    return np.linalg.norm(flow_cell.reshape(-1, 2), axis=1)


def velocity_variance(flow_cell: np.ndarray) -> float:
    """Variance of the velocity VECTORS: Var(vx) + Var(vy). px^2/frame^2.

    Johansson, Helbing, Al-Abideen & Al-Bosta, "From Crowd Dynamics to Crowd
    Safety" (Adv. Complex Syst., 2008), conclusions: "The 'pressure', defined
    as the density times the VARIANCE OF VELOCITIES, provides better and more
    specific information about critical areas and times." The looser "variance
    of speeds" phrasing elsewhere in that paper is not what the quantity is.

    The distinction is not cosmetic: the variance of speed MAGNITUDES is 0 for
    perfect counterflow (+5, -5), which would report pressure 0 -- "safe" --
    in exactly the turbulent regime where crowd pressure peaks. The vector
    variance is 25 there.

    The sum of the per-component variances is the standard scalar reduction of
    the velocity covariance matrix (its trace); it is rotation-invariant and,
    like the magnitude variance, scales as k^2 under a k-fold resolution
    change, so scale invariance of the pressure is preserved.

    NaN for an empty cell (fail loud).
    """
    vectors = flow_cell.reshape(-1, 2)
    if vectors.shape[0] == 0:
        return NAN
    return float(np.var(vectors[:, 0]) + np.var(vectors[:, 1]))


def mean_speed(flow_cell: np.ndarray) -> float:
    """Mean speed magnitude. NaN for an empty cell."""
    speeds = _speeds(flow_cell)
    if speeds.size == 0:
        return NAN
    return float(np.mean(speeds))


def density_px(count: float, cell_area_px: float) -> float:
    """People per square pixel. NaN if the area is degenerate."""
    if cell_area_px <= 0:
        return NAN
    return float(count) / float(cell_area_px)


def crowd_pressure(
    count: float,
    cell_area_px: float,
    velocity_variance_px: float,
    fps: float,
) -> float:
    """Crowd pressure in s^-2, from uncalibrated pixel-space quantities.

    P = (count / cell_area_px) * fps^2 * Var(velocity_px_per_frame)

    Returns NaN rather than 0 when inputs are unusable: 0 would read as 'safe'.
    """
    rho = density_px(count, cell_area_px)
    if not np.isfinite(rho) or not np.isfinite(velocity_variance_px) or fps <= 0:
        return NAN
    return float(rho * (fps ** 2) * velocity_variance_px)
