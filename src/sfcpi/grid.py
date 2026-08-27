"""Fixed cell partitioning and per-cell metric aggregation.

Pressure is computed per cell because the pixel-to-metre scale cancels only
where that scale is locally constant. Cells also localise risk.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from .metrics import crowd_pressure, density_px, mean_speed, speed_variance


class CellGrid:
    def __init__(self, frame_width: int, frame_height: int, cell_size: int) -> None:
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
        if frame_width % cell_size or frame_height % cell_size:
            raise ValueError(
                f"cell_size {cell_size} must divide frame {frame_width}x{frame_height}"
            )
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.cell_size = cell_size
        self.n_cols = frame_width // cell_size
        self.n_rows = frame_height // cell_size

    @property
    def cell_area_px(self) -> float:
        return float(self.cell_size * self.cell_size)

    def iter_cells(self) -> Iterator[tuple[int, int, slice, slice]]:
        cs = self.cell_size
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                yield row, col, slice(row * cs, (row + 1) * cs), slice(col * cs, (col + 1) * cs)

    def aggregate(
        self, flow: np.ndarray, counts: np.ndarray, fps: float
    ) -> dict[str, np.ndarray]:
        expected = (self.n_rows, self.n_cols)
        if counts.shape != expected:
            raise ValueError(f"counts shape {counts.shape} != grid {expected}")

        out = {
            key: np.full(expected, np.nan, dtype=float)
            for key in ("pressure", "mean_speed", "speed_variance", "density")
        }
        for row, col, ys, xs in self.iter_cells():
            cell = flow[ys, xs]
            var = speed_variance(cell)
            out["speed_variance"][row, col] = var
            out["mean_speed"][row, col] = mean_speed(cell)
            out["density"][row, col] = density_px(counts[row, col], self.cell_area_px)
            out["pressure"][row, col] = crowd_pressure(
                counts[row, col], self.cell_area_px, var, fps
            )
        return out
