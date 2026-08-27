"""Occlusion characterisation.

Measures how the detector's recovered count degrades as true density rises.
Per spec section 8 this DECIDES whether an occlusion correction becomes a
contribution or is dropped in favour of a density estimator. Both outcomes
are publishable; the experiment settles it with data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DetectionRatePoint:
    true_count: int
    detected_count: int
    density_proxy: float
    ratio: float

    @classmethod
    def from_counts(
        cls, true_count: int, detected_count: int, density_proxy: float
    ) -> "DetectionRatePoint":
        ratio = float("nan") if true_count <= 0 else detected_count / true_count
        return cls(true_count, detected_count, density_proxy, ratio)


def detection_rate_curve(points: List[DetectionRatePoint], n_bins: int = 10) -> pd.DataFrame:
    """Mean detected/true ratio per density bin."""
    if not points:
        raise ValueError("no points supplied")
    density = np.array([p.density_proxy for p in points], dtype=float)
    ratio = np.array([p.ratio for p in points], dtype=float)

    lo, hi = float(density.min()), float(density.max())
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(density, edges) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        sel = ratio[idx == b]
        sel = sel[np.isfinite(sel)]
        rows.append({
            "bin_center": float((edges[b] + edges[b + 1]) / 2),
            "mean_ratio": float(sel.mean()) if sel.size else float("nan"),
            "std_ratio": float(sel.std()) if sel.size else float("nan"),
            "n": int(sel.size),
        })
    return pd.DataFrame(rows)


def is_monotonic_decreasing(df: pd.DataFrame, tolerance: float = 0.05) -> bool:
    """True if the detection ratio falls consistently with density.

    Two conditions, both necessary:
      - no bin-to-bin INCREASE larger than `tolerance` (permits small wobble), and
      - a genuine NET decrease from the first finite bin to the last.

    The net-decrease term is essential: bounding only the rise would certify a
    steadily RISING curve as "decreasing" and invert the conclusion this rule gates.
    """
    values = df["mean_ratio"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return False
    no_large_rise = np.all(np.diff(values) <= tolerance)
    net_decrease = (values[0] - values[-1]) > tolerance
    return bool(no_large_rise and net_decrease)
