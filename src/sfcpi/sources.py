"""Frame sources. Cycle 1 ships file replay; live WebRTC lands in cycle 2
behind the same Protocol."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, List, Optional, Protocol, runtime_checkable

import cv2
import numpy as np

DEFAULT_FPS = 25.0


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp: float
    image: np.ndarray


@runtime_checkable
class FrameSource(Protocol):
    fps: float

    def __iter__(self) -> Iterator[Frame]:  # pragma: no cover - protocol
        ...


class SyntheticSource:
    """In-memory source for tests."""

    def __init__(self, frames: List[np.ndarray], fps: float) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self._frames = frames
        self.fps = float(fps)

    def __iter__(self) -> Iterator[Frame]:
        for i, img in enumerate(self._frames):
            yield Frame(index=i, timestamp=i / self.fps, image=img)


class FileSource:
    """Replay a video file. Reports the file's fps, falling back to 25."""

    def __init__(self, path: str, max_frames: Optional[int] = None) -> None:
        self.path = path
        self.max_frames = max_frames
        self.fps = self._probe_fps()

    def _probe_fps(self) -> float:
        cap = self._open()
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            return float(fps) if fps and fps > 0 else DEFAULT_FPS
        finally:
            cap.release()

    def _open(self) -> cv2.VideoCapture:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"video not found: {self.path}")
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise OSError(f"could not open video: {self.path}")
        return cap

    def __iter__(self) -> Iterator[Frame]:
        cap = self._open()
        try:
            i = 0
            while self.max_frames is None or i < self.max_frames:
                ok, img = cap.read()
                if not ok:
                    break
                yield Frame(index=i, timestamp=i / self.fps, image=img)
                i += 1
        finally:
            cap.release()
