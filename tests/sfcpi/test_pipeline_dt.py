"""Pressure must use the ACTUAL interval between the two frames compared.

`FrameBridge` is drop-oldest: on a live stream the pipeline routinely sees
frame N then frame N+34, 0.2s apart, not 1/29.41s apart. Optical flow measures
displacement BETWEEN THE TWO FRAMES IT WAS GIVEN, so converting that
displacement to a velocity with the nominal capture fps overstates speed by
the drop factor -- and pressure, which goes as fps**2, by its square.

Observed live 2026-08-28: 181 frames processed, 1117 dropped, median gap
0.201s against a nominal 29.41 fps. Peak pressure came out 35.49 s^-2 against
a CRITICAL threshold of 0.02, firing a false alert on an empty corridor.
"""
import numpy as np
import pytest

from sfcpi.detect import FixedDetector, Detection
from sfcpi.flow import FlowEstimator
from sfcpi.grid import CellGrid
from sfcpi.pipeline import Pipeline
from sfcpi.sources import Frame


class _ConstantFlow:
    """Fixed per-frame-pair displacement, so only dt can change the answer."""

    def __init__(self, dx): self._dx = dx

    def estimate(self, a, b):
        flow = np.zeros((*a.shape[:2], 2), dtype=np.float32)
        flow[..., 0] = self._dx
        # Alternating columns, so EVERY cell carries velocity variance --
        # splitting the frame in half would leave the one counted cell
        # internally uniform, hence zero pressure, and the test would pass
        # for the wrong reason.
        flow[:, ::2, 0] = -self._dx
        return flow


def _frames(timestamps, size=128):
    for i, t in enumerate(timestamps):
        yield Frame(index=i, timestamp=t, image=np.zeros((size, size, 3), np.uint8))


def _pipeline(fps):
    return Pipeline(
        grid=CellGrid(frame_width=128, frame_height=128, cell_size=64),
        flow_estimator=_ConstantFlow(4.0),
        detector=FixedDetector([Detection(0, 0, 10, 10, 0.9)]),
        fps=fps,
    )


def _peak(timestamps, fps):
    out = [f for f in _pipeline(fps).run(_frames(timestamps)) if f.flow_valid]
    return max(f.global_max_pressure for f in out)


def test_dropped_frames_do_not_inflate_pressure():
    """Same displacement over a 6x longer gap is a 6x SLOWER crowd."""
    dense = [i / 30.0 for i in range(6)]          # 30 fps, nothing dropped
    sparse = [i / 5.0 for i in range(6)]          # same frames, 5 fps effective

    fast = _peak(dense, fps=30.0)
    slow = _peak(sparse, fps=30.0)

    assert slow < fast, (
        f"widely spaced frames must not read as MORE pressure "
        f"(sparse={slow:.4g} vs dense={fast:.4g})"
    )
    # pressure goes as (1/dt)**2, so a 6x longer gap is ~36x less pressure
    assert slow == pytest.approx(fast / 36.0, rel=0.05)


def test_nominal_fps_is_not_used_when_timestamps_disagree():
    """The declared fps must not override what the timestamps actually say."""
    stamps = [i / 5.0 for i in range(6)]
    assert _peak(stamps, fps=30.0) == pytest.approx(_peak(stamps, fps=5.0), rel=1e-6)


def test_falls_back_to_nominal_fps_when_a_gap_is_unusable():
    """Equal or backwards timestamps would divide by zero or invert velocity;
    the nominal rate is the only sane fallback, and it must not crash."""
    stamps = [0.0, 0.0, 0.0, 0.0]
    out = [f for f in _pipeline(30.0).run(_frames(stamps)) if f.flow_valid]
    assert out and all(np.isfinite(f.global_max_pressure) for f in out)
