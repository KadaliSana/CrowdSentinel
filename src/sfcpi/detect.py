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


class BoardDetector:
    """Serves detections computed on the AmebaPro2's NPU, not on this host.

    The board already runs SCRFD per frame; re-detecting here would burn CPU
    to produce a SECOND, disagreeing set of boxes over a video that already
    has the board's boxes drawn into it. This detector just holds the most
    recent data-channel message.

    Two states are deliberately errors rather than empty lists, matching
    NullDetector: nothing received yet, and a message too old to trust. Both
    mean "the count is unknown", and the pipeline's fail-loud path turns that
    into NaN with sensing_confidence 0 -- never into a calm, empty scene.
    """

    def __init__(self, max_age_s: float = 2.0) -> None:
        if max_age_s <= 0:
            raise ValueError("max_age_s must be positive")
        self.max_age_s = max_age_s
        self._latest = None
        self._received_at = None

    def update(self, detections, now: float = None) -> None:
        """Called by the data-channel handler for each board message."""
        import time as _time

        self._latest = detections
        self._received_at = _time.monotonic() if now is None else now

    @property
    def last_count(self):
        """The board's TRUE count, which may exceed len(detect(...))."""
        return None if self._latest is None else self._latest.count

    @property
    def last(self):
        return self._latest

    def detect(self, image: np.ndarray, now: float = None) -> List[Detection]:
        import time as _time

        if self._latest is None:
            raise RuntimeError(
                "BoardDetector: no detection metadata received from the board "
                "yet; the count is unknown, not zero"
            )
        current = _time.monotonic() if now is None else now
        age = current - self._received_at
        if age > self.max_age_s:
            raise RuntimeError(
                f"BoardDetector: detection metadata is stale ({age:.1f}s old, "
                f"limit {self.max_age_s}s); the count is unknown, not the last "
                f"one seen"
            )
        return list(self._latest.detections)


class NullDetector:
    """Flow-only mode: a detector that is explicitly ABSENT, not empty.

    `FixedDetector([])` is the wrong tool here. It succeeds, so the pipeline's
    degradation branch is never taken: counts come out 0, pressure comes out
    0.0 and sensing_confidence comes out 1.0 -- full trust in the one mode that
    measures no people at all. Raising routes flow-only runs through the
    existing fail-loud path, where an unknown count is NaN and confidence is 0,
    while mean_speed and velocity_variance stay finite. That is the same
    semantics the detector-failure path already has, and there is exactly one
    of it.
    """

    def detect(self, image: np.ndarray) -> List[Detection]:
        raise RuntimeError(
            "NullDetector: no detector configured (flow-only mode); "
            "the count is unknown, not zero"
        )


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
