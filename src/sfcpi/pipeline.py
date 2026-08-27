"""Wires source -> flow -> grid -> metrics into a stream of MetricsFrames."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional

import numpy as np

from .detect import Detector, counts_per_cell
from .flow import FlowEstimator
from .grid import CellGrid
from .sources import Frame, FrameSource

NAN = float("nan")


@dataclass
class MetricsFrame:
    index: int
    timestamp: float
    flow_valid: bool
    cells: Dict[str, np.ndarray] = field(default_factory=dict)
    global_pressure: float = NAN
    global_max_pressure: float = NAN
    total_count: float = NAN
    sensing_confidence: float = 0.0


class Pipeline:
    def __init__(
        self,
        grid: CellGrid,
        flow_estimator: FlowEstimator,
        detector: Detector,
        fps: float,
        ewma_alpha: float = 0.3,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if not (0 < ewma_alpha <= 1):
            raise ValueError("ewma_alpha must be in (0, 1]")
        self.grid = grid
        self.flow_estimator = flow_estimator
        self.detector = detector
        self.fps = fps
        self.ewma_alpha = ewma_alpha
        self._smoothed: Optional[float] = None

    def _smooth(self, value: float) -> float:
        if not np.isfinite(value):
            return NAN
        if self._smoothed is None or not np.isfinite(self._smoothed):
            self._smoothed = value
        else:
            a = self.ewma_alpha
            self._smoothed = a * value + (1 - a) * self._smoothed
        return float(self._smoothed)

    def run(self, source: FrameSource) -> Iterator[MetricsFrame]:
        prev: Optional[Frame] = None
        for frame in source:
            if prev is None:
                prev = frame
                yield MetricsFrame(frame.index, frame.timestamp, flow_valid=False)
                continue

            flow = self.flow_estimator.estimate(prev.image, frame.image)
            prev = frame

            try:
                detections = self.detector.detect(frame.image)
                counts = counts_per_cell(detections, self.grid)
                total = float(counts.sum())
                confidence = 1.0
            except Exception:
                # Fail loud: unknown N must never render as zero pressure.
                counts = np.full((self.grid.n_rows, self.grid.n_cols), NAN)
                total = NAN
                confidence = 0.0

            cells = self.grid.aggregate(flow, counts, fps=self.fps)
            pressures = cells["pressure"]
            mean_p = float(np.nanmean(pressures)) if np.isfinite(pressures).any() else NAN
            max_p = float(np.nanmax(pressures)) if np.isfinite(pressures).any() else NAN

            yield MetricsFrame(
                index=frame.index,
                timestamp=frame.timestamp,
                flow_valid=True,
                cells=cells,
                global_pressure=self._smooth(mean_p),
                global_max_pressure=max_p,
                total_count=total,
                sensing_confidence=confidence,
            )
