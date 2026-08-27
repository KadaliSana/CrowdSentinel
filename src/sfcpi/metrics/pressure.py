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


def speed_variance(flow_cell: np.ndarray) -> float:
    """Variance of speed magnitudes. NaN for an empty cell (fail loud)."""
    speeds = _speeds(flow_cell)
    if speeds.size == 0:
        return NAN
    return float(np.var(speeds))


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
    speed_variance_px: float,
    fps: float,
) -> float:
    """Crowd pressure in s^-2, from uncalibrated pixel-space quantities.

    P = (count / cell_area_px) * fps^2 * Var(speed_px_per_frame)

    Returns NaN rather than 0 when inputs are unusable: 0 would read as 'safe'.
    """
    rho = density_px(count, cell_area_px)
    if not np.isfinite(rho) or not np.isfinite(speed_variance_px) or fps <= 0:
        return NAN
    return float(rho * (fps ** 2) * speed_variance_px)
