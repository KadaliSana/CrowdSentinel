"""Detector interface and per-cell counting.

The detector supplies N. Its bias under occlusion is the subject of the
occlusion experiment (Task 9) -- it is measured, not assumed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, runtime_checkable

import numpy as np

from .grid import CellGrid


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@runtime_checkable
class Detector(Protocol):
    def detect(self, image: np.ndarray) -> List[Detection]:  # pragma: no cover
        ...


class FixedDetector:
    """Returns a preset list. For tests and for replaying stored detections."""

    def __init__(self, detections: List[Detection]) -> None:
        self._detections = detections

    def detect(self, image: np.ndarray) -> List[Detection]:
        return self._detections


class YoloDetector:
    """Ultralytics-backed detector. Loaded lazily so tests need no weights."""

    def __init__(self, model_path: str, conf: float = 0.35, classes=None) -> None:
        self.model_path = model_path
        self.conf = conf
        self.classes = classes
        self._model = None

    def _ensure(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
        return self._model

    def detect(self, image: np.ndarray) -> List[Detection]:
        model = self._ensure()
        results = model(image, conf=self.conf, classes=self.classes, verbose=False)
        out: List[Detection] = []
        if not results:
            raise RuntimeError(
                "YoloDetector: model returned no results (malformed output, "
                "not a legitimate zero-detection frame)"
            )
        boxes = results[0].boxes
        if boxes is None:
            raise RuntimeError(
                "YoloDetector: results[0].boxes is None (malformed output, "
                "not a legitimate zero-detection frame)"
            )
        if boxes.xyxy is None:
            raise RuntimeError(
                "YoloDetector: boxes.xyxy is None (malformed output, "
                "not a legitimate zero-detection frame)"
            )
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
        for (x1, y1, x2, y2), s in zip(xyxy, scores):
            out.append(Detection(float(x1), float(y1), float(x2), float(y2), float(s)))
        return out


def counts_per_cell(detections: List[Detection], grid: CellGrid) -> np.ndarray:
    """Count detections per cell by centroid. Out-of-frame centroids are dropped."""
    counts = np.zeros((grid.n_rows, grid.n_cols), dtype=float)
    for det in detections:
        cx, cy = det.centroid
        if not (0 <= cx < grid.frame_width and 0 <= cy < grid.frame_height):
            continue
        counts[int(cy) // grid.cell_size, int(cx) // grid.cell_size] += 1.0
    return counts
