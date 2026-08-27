"""WebRTCSource: makes a live KVS WebRTC stream look exactly like the file
replay sources in sfcpi.sources, so the pipeline consumes both identically.

Frames arrive on aiortc's asyncio loop via feed(); the pipeline consumes them
synchronously via __iter__. FrameBridge is the bounded, drop-oldest handoff
between the two. fps is measured from the first `warmup_frames` timestamps
(median inter-frame delta) and RAISES until that warm-up completes -- frame
rate feeds crowd pressure as fps**2, so a silent default would be a quadratic
error, not a rounding error.

aiortc is imported lazily, inside the connect path only, so importing this
module (and running these tests) never requires it to be installed.
"""
from __future__ import annotations

import statistics
from typing import Any, Iterator, List, Optional

import numpy as np

from sfcpi.sources import Frame

from .bridge import FrameBridge


class WebRTCSource:
    """FrameSource-compatible live ingest from an AWS KVS WebRTC channel.

    Satisfies the FrameSource protocol structurally (exposes `fps` and
    `__iter__`); it does not import or subclass anything from sfcpi.sources
    except Frame.
    """

    def __init__(
        self,
        signaling: Any,
        bridge: Optional[FrameBridge] = None,
        warmup_frames: int = 10,
        connect_timeout_s: float = 15.0,
        maxsize: int = 8,
    ) -> None:
        if warmup_frames < 1:
            raise ValueError("warmup_frames must be >= 1")
        self.signaling = signaling
        self.warmup_frames = warmup_frames
        self.connect_timeout_s = connect_timeout_s
        self._bridge = bridge if bridge is not None else FrameBridge(maxsize=maxsize)
        self._index = 0
        self._warmup_timestamps: List[float] = []
        self._fps: Optional[float] = None

    # -- FrameSource protocol -------------------------------------------------

    @property
    def fps(self) -> float:
        if self._fps is None:
            raise RuntimeError(
                f"fps not yet measured: need {self.warmup_frames} frames to "
                f"warm up, have {len(self._warmup_timestamps)}"
            )
        return self._fps

    @property
    def dropped(self) -> int:
        return self._bridge.dropped

    def __iter__(self) -> Iterator[Frame]:
        return iter(self._bridge)

    # -- producer side ---------------------------------------------------------

    def feed(self, frame: Any, timestamp: float) -> None:
        """Called by the media task (or tests) for each incoming frame.

        Accepts a plain numpy BGR array, or an av.VideoFrame-like object
        exposing `.to_ndarray(format="bgr24")` -- detected by capability, not
        by importing `av`.
        """
        image = self._as_bgr_ndarray(frame)
        self._record_warmup_timestamp(timestamp)
        out = Frame(index=self._index, timestamp=timestamp, image=image)
        self._index += 1
        self._bridge.put(out)

    @staticmethod
    def _as_bgr_ndarray(frame: Any) -> np.ndarray:
        if isinstance(frame, np.ndarray):
            return frame
        to_ndarray = getattr(frame, "to_ndarray", None)
        if callable(to_ndarray):
            return to_ndarray(format="bgr24")
        raise TypeError(
            f"feed() needs a numpy array or an av.VideoFrame-like object "
            f"with to_ndarray(format=...); got {type(frame)!r}"
        )

    def _record_warmup_timestamp(self, timestamp: float) -> None:
        if self._fps is not None:
            return
        self._warmup_timestamps.append(float(timestamp))
        if len(self._warmup_timestamps) >= self.warmup_frames:
            deltas = [
                b - a
                for a, b in zip(self._warmup_timestamps, self._warmup_timestamps[1:])
            ]
            median_delta = statistics.median(deltas)
            if median_delta <= 0:
                raise RuntimeError(
                    "fps could not be measured: warm-up timestamps are not "
                    "monotonically increasing"
                )
            self._fps = 1.0 / median_delta

    def fail(self, exc: BaseException) -> None:
        self._bridge.fail(exc)

    def close(self) -> None:
        self._bridge.close()

    # -- connect path (lazy aiortc) --------------------------------------------

    async def connect(self) -> None:
        """Establish the WebRTC connection and start feeding frames.

        aiortc is imported here, not at module scope, so importing this
        module never requires it.
        """
        import aiortc  # noqa: F401  (lazy import; connect-path only)

        raise NotImplementedError(
            "WebRTCSource.connect: signaling/media-task wiring lands in a "
            "later task"
        )
