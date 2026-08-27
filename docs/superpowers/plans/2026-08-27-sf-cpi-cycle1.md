# SF-CPI Cycle 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a replay-first pipeline that computes scale-free crowd pressure from uncalibrated video, plus the evaluation harness and the occlusion characterisation experiment.

**Architecture:** A standalone Python package `src/sfcpi/` with a pure-function `metrics/` core (numpy only, no I/O) wrapped by pluggable frame sources, an optical-flow stage, a cell grid, and a detector interface. The physics claim — that crowd pressure is invariant to the unknown pixel-to-metre scale — is enforced as an executable property test. Nothing in this cycle needs the camera board.

**Tech Stack:** Python 3.10, numpy 1.26, OpenCV 4.10 (Farneback dense flow), pytest 6.2, pandas 2.2, ultralytics 8.4 (detector).

**Spec:** `docs/superpowers/specs/2026-08-27-sf-cpi-design.md`

## Global Constraints

- Package root: `src/sfcpi/`. Tests: `tests/sfcpi/`. Run tests with `python3 -m pytest`.
- `src/sfcpi/metrics/` MUST remain pure: numpy in, numpy/float out. No cv2, no file I/O, no model calls. This is what keeps the physics testable in isolation.
- Velocities are in **pixels per frame**; `fps` converts to per-second. Never mix units.
- **Fail-loud rule:** when a value cannot be computed, emit `NaN`, never `0`. Zero reads as "safe" and this is a safety signal.
- Crowd pressure uses the **variance of speed magnitudes** (per Johansson et al.: "the variance of speeds, multiplied by the density").
- Reference threshold `P > 0.02 s^-2` is an order-of-magnitude reference, not a constant. Never hardcode it as a pass/fail in library code.
- No repo currently has tests; Task 1 establishes the harness.
- Commit after every task.

---

### Task 1: Package scaffold + the crowd pressure core

**Files:**
- Create: `src/sfcpi/__init__.py`, `src/sfcpi/metrics/__init__.py`, `src/sfcpi/metrics/pressure.py`
- Create: `pytest.ini`
- Test: `tests/sfcpi/test_pressure.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `crowd_pressure(count: float, cell_area_px: float, speed_variance_px: float, fps: float) -> float`
  - `density_px(count: float, cell_area_px: float) -> float`
  - `speed_variance(flow_cell: np.ndarray) -> float`  (flow_cell shape `(H, W, 2)`, px/frame)
  - `mean_speed(flow_cell: np.ndarray) -> float`

- [ ] **Step 1: Write the failing tests**

```python
# tests/sfcpi/test_pressure.py
import math
import numpy as np
import pytest
from sfcpi.metrics.pressure import (
    crowd_pressure, density_px, speed_variance, mean_speed,
)

def _pressure_in_metres(count, cell_area_px, speed_var_px, s, fps):
    """Reference: compute P in real units given a metres-per-pixel scale s."""
    rho_m = count / (s ** 2 * cell_area_px)          # people / m^2
    var_m = (s * fps) ** 2 * speed_var_px            # (m/s)^2
    return rho_m * var_m

@pytest.mark.parametrize("s", [0.004, 0.01, 0.05, 0.2, 1.7])
def test_pressure_is_invariant_to_pixel_scale(s):
    """The paper's central claim, as an executable test."""
    count, area, var, fps = 37, 4096.0, 0.83, 15.0
    assert crowd_pressure(count, area, var, fps) == pytest.approx(
        _pressure_in_metres(count, area, var, s, fps), rel=1e-9
    )

def test_pressure_scales_with_fps_squared():
    base = crowd_pressure(10, 100.0, 2.0, 10.0)
    doubled = crowd_pressure(10, 100.0, 2.0, 20.0)
    assert doubled == pytest.approx(4 * base)

def test_uniform_motion_gives_zero_pressure():
    """Everyone moving identically is not dangerous: variance, not speed."""
    flow = np.full((8, 8, 2), 3.0)
    assert speed_variance(flow) == pytest.approx(0.0)
    assert crowd_pressure(50, 64.0, speed_variance(flow), 15.0) == pytest.approx(0.0)

def test_zero_motion_gives_zero_pressure():
    flow = np.zeros((8, 8, 2))
    assert mean_speed(flow) == pytest.approx(0.0)
    assert crowd_pressure(50, 64.0, speed_variance(flow), 15.0) == pytest.approx(0.0)

def test_speed_variance_uses_magnitudes_not_components():
    """Opposing motion has zero mean velocity but non-zero speed variance only
    if magnitudes differ; equal-and-opposite speeds have equal magnitudes."""
    flow = np.zeros((2, 1, 2))
    flow[0, 0] = (5.0, 0.0)
    flow[1, 0] = (-5.0, 0.0)
    assert speed_variance(flow) == pytest.approx(0.0)

def test_mixed_speeds_give_positive_variance():
    flow = np.zeros((2, 1, 2))
    flow[0, 0] = (0.0, 0.0)
    flow[1, 0] = (4.0, 0.0)
    assert speed_variance(flow) == pytest.approx(4.0)

def test_density_px():
    assert density_px(20, 100.0) == pytest.approx(0.2)

def test_nan_count_propagates_as_nan_not_zero():
    """Fail-loud: unknown count must not read as 'safe'."""
    assert math.isnan(crowd_pressure(float("nan"), 100.0, 1.0, 15.0))

def test_zero_area_is_nan_not_inf():
    assert math.isnan(crowd_pressure(10, 0.0, 1.0, 15.0))

def test_empty_cell_variance_is_nan():
    assert math.isnan(speed_variance(np.zeros((0, 0, 2))))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/sfcpi/test_pressure.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sfcpi'`

- [ ] **Step 3: Create the package and pytest config**

```ini
# pytest.ini
[pytest]
testpaths = tests
pythonpath = src
```

```python
# src/sfcpi/__init__.py
"""SF-CPI: Scale-Free Crowd Pressure Index."""
__version__ = "0.1.0"
```

```python
# src/sfcpi/metrics/__init__.py
from .pressure import crowd_pressure, density_px, speed_variance, mean_speed

__all__ = ["crowd_pressure", "density_px", "speed_variance", "mean_speed"]
```

- [ ] **Step 4: Implement the metrics**

```python
# src/sfcpi/metrics/pressure.py
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/sana/Yan && python3 -m pytest tests/sfcpi/test_pressure.py -v`
Expected: all PASS (13 tests, including 5 scale-invariance parametrisations)

- [ ] **Step 6: Commit**

```bash
cd /home/sana/Yan
git add pytest.ini src/sfcpi tests/sfcpi
git commit -m "feat(sfcpi): scale-free crowd pressure core with invariance property tests"
```

---

### Task 2: Cell grid and per-cell aggregation

**Files:**
- Create: `src/sfcpi/grid.py`
- Test: `tests/sfcpi/test_grid.py`

**Interfaces:**
- Consumes: `sfcpi.metrics` (Task 1)
- Produces:
  - `CellGrid(frame_width: int, frame_height: int, cell_size: int)`
  - `CellGrid.n_rows -> int`, `CellGrid.n_cols -> int`, `CellGrid.cell_area_px -> float`
  - `CellGrid.iter_cells() -> Iterator[tuple[int, int, slice, slice]]` yielding `(row, col, y_slice, x_slice)`
  - `CellGrid.aggregate(flow: np.ndarray, counts: np.ndarray, fps: float) -> dict[str, np.ndarray]` returning arrays keyed `pressure`, `mean_speed`, `speed_variance`, `density`, each shape `(n_rows, n_cols)`

- [ ] **Step 1: Write the failing test**

```python
# tests/sfcpi/test_grid.py
import numpy as np
import pytest
from sfcpi.grid import CellGrid

def test_grid_dimensions():
    g = CellGrid(frame_width=640, frame_height=480, cell_size=160)
    assert (g.n_cols, g.n_rows) == (4, 3)
    assert g.cell_area_px == pytest.approx(160 * 160)

def test_rejects_non_dividing_cell_size():
    with pytest.raises(ValueError, match="cell_size"):
        CellGrid(frame_width=640, frame_height=480, cell_size=150)

def test_iter_cells_covers_every_pixel_once():
    g = CellGrid(frame_width=8, frame_height=4, cell_size=2)
    canvas = np.zeros((4, 8), dtype=int)
    for _r, _c, ys, xs in g.iter_cells():
        canvas[ys, xs] += 1
    assert (canvas == 1).all()

def test_aggregate_shapes_and_uniform_motion():
    g = CellGrid(frame_width=4, frame_height=4, cell_size=2)
    flow = np.full((4, 4, 2), 2.0)
    counts = np.full((2, 2), 5.0)
    out = g.aggregate(flow, counts, fps=10.0)
    for key in ("pressure", "mean_speed", "speed_variance", "density"):
        assert out[key].shape == (2, 2)
    assert out["speed_variance"] == pytest.approx(0.0)
    assert out["pressure"] == pytest.approx(0.0)
    assert out["mean_speed"] == pytest.approx(np.sqrt(8.0))

def test_aggregate_isolates_cells():
    """Motion in one cell must not leak into its neighbours."""
    g = CellGrid(frame_width=4, frame_height=2, cell_size=2)
    flow = np.zeros((2, 4, 2))
    flow[0, 0] = (6.0, 0.0)          # single moving pixel in cell (0,0)
    counts = np.ones((1, 2))
    out = g.aggregate(flow, counts, fps=10.0)
    assert out["speed_variance"][0, 0] > 0
    assert out["speed_variance"][0, 1] == pytest.approx(0.0)

def test_counts_shape_must_match_grid():
    g = CellGrid(frame_width=4, frame_height=4, cell_size=2)
    with pytest.raises(ValueError, match="counts"):
        g.aggregate(np.zeros((4, 4, 2)), np.ones((3, 3)), fps=10.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/sana/Yan && python3 -m pytest tests/sfcpi/test_grid.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sfcpi.grid'`

- [ ] **Step 3: Implement the grid**

```python
# src/sfcpi/grid.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/sana/Yan && python3 -m pytest tests/sfcpi/test_grid.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
cd /home/sana/Yan
git add src/sfcpi/grid.py tests/sfcpi/test_grid.py
git commit -m "feat(sfcpi): cell grid with per-cell pressure aggregation"
```

---

### Task 3: Frame sources (file replay)

**Files:**
- Create: `src/sfcpi/sources.py`
- Test: `tests/sfcpi/test_sources.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `Frame` dataclass: `index: int`, `timestamp: float`, `image: np.ndarray`
  - `FrameSource` Protocol with `fps: float` and `__iter__() -> Iterator[Frame]`
  - `FileSource(path: str, max_frames: int | None = None)` implementing it
  - `SyntheticSource(frames: list[np.ndarray], fps: float)` for tests

- [ ] **Step 1: Write the failing test**

```python
# tests/sfcpi/test_sources.py
import numpy as np
import pytest
from sfcpi.sources import Frame, FileSource, SyntheticSource

def test_synthetic_source_yields_indexed_frames():
    imgs = [np.zeros((4, 4, 3), np.uint8) for _ in range(3)]
    src = SyntheticSource(imgs, fps=5.0)
    frames = list(src)
    assert [f.index for f in frames] == [0, 1, 2]
    assert frames[1].timestamp == pytest.approx(0.2)
    assert isinstance(frames[0], Frame)
    assert src.fps == 5.0

def test_synthetic_source_rejects_bad_fps():
    with pytest.raises(ValueError, match="fps"):
        SyntheticSource([np.zeros((2, 2, 3), np.uint8)], fps=0.0)

def test_file_source_missing_file_names_the_path():
    with pytest.raises(FileNotFoundError, match="no_such_video.mp4"):
        list(FileSource("no_such_video.mp4"))

def test_file_source_reads_written_video(tmp_path):
    import cv2
    path = str(tmp_path / "clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32))
    for i in range(5):
        writer.write(np.full((32, 32, 3), i * 20, np.uint8))
    writer.release()

    src = FileSource(path)
    frames = list(src)
    assert len(frames) == 5
    assert src.fps == pytest.approx(10.0)
    assert frames[0].image.shape == (32, 32, 3)

def test_file_source_respects_max_frames(tmp_path):
    import cv2
    path = str(tmp_path / "clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32))
    for i in range(10):
        writer.write(np.full((32, 32, 3), i * 10, np.uint8))
    writer.release()
    assert len(list(FileSource(path, max_frames=3))) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/sana/Yan && python3 -m pytest tests/sfcpi/test_sources.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sfcpi.sources'`

- [ ] **Step 3: Implement the sources**

```python
# src/sfcpi/sources.py
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
        self.fps = DEFAULT_FPS
        self._probed = False

    def _open(self) -> cv2.VideoCapture:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"video not found: {self.path}")
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise OSError(f"could not open video: {self.path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps) if fps and fps > 0 else DEFAULT_FPS
        self._probed = True
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/sana/Yan && python3 -m pytest tests/sfcpi/test_sources.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
cd /home/sana/Yan
git add src/sfcpi/sources.py tests/sfcpi/test_sources.py
git commit -m "feat(sfcpi): frame source protocol with file replay"
```

---

### Task 4: Dense optical flow stage

**Files:**
- Create: `src/sfcpi/flow.py`
- Test: `tests/sfcpi/test_flow.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `FlowEstimator(downscale: float = 1.0)` with `estimate(prev_bgr, curr_bgr) -> np.ndarray` of shape `(H, W, 2)` in **px/frame at full frame scale**

- [ ] **Step 1: Write the failing test**

```python
# tests/sfcpi/test_flow.py
import numpy as np
import pytest
from sfcpi.flow import FlowEstimator

def _textured(h=64, w=64, shift=0):
    """Deterministic texture so Farneback has gradients to track."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, size=(h, w + 32), dtype=np.uint8)
    crop = base[:, shift:shift + w]
    return np.repeat(crop[:, :, None], 3, axis=2)

def test_flow_shape_matches_frame():
    est = FlowEstimator()
    flow = est.estimate(_textured(), _textured(shift=2))
    assert flow.shape == (64, 64, 2)
    assert flow.dtype == np.float32

def test_static_scene_gives_near_zero_flow():
    est = FlowEstimator()
    img = _textured()
    flow = est.estimate(img, img)
    assert np.abs(flow).max() < 0.1

def test_horizontal_shift_detected_with_correct_sign():
    est = FlowEstimator()
    flow = est.estimate(_textured(shift=0), _textured(shift=4))
    mean_dx = float(np.mean(flow[..., 0]))
    assert mean_dx < -1.0     # content moved left in image coords

def test_downscale_rescales_vectors_back_to_full_resolution():
    full = FlowEstimator(downscale=1.0).estimate(_textured(), _textured(shift=4))
    half = FlowEstimator(downscale=0.5).estimate(_textured(), _textured(shift=4))
    assert half.shape == (64, 64, 2)
    assert float(np.mean(half[..., 0])) == pytest.approx(float(np.mean(full[..., 0])), abs=1.5)

def test_mismatched_shapes_raise():
    est = FlowEstimator()
    with pytest.raises(ValueError, match="same shape"):
        est.estimate(_textured(64, 64), _textured(32, 32))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/sana/Yan && python3 -m pytest tests/sfcpi/test_flow.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sfcpi.flow'`

- [ ] **Step 3: Implement the estimator**

```python
# src/sfcpi/flow.py
"""Dense optical flow (Farneback).

Flow is the density-robust half of the sensing story: it keeps working when
detection collapses under occlusion. Vectors are always returned in pixels per
frame at FULL frame resolution, whatever the internal downscale.
"""
from __future__ import annotations

import cv2
import numpy as np

_FARNEBACK = dict(
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
)


class FlowEstimator:
    def __init__(self, downscale: float = 1.0) -> None:
        if not (0 < downscale <= 1.0):
            raise ValueError("downscale must be in (0, 1]")
        self.downscale = float(downscale)

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def estimate(self, prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> np.ndarray:
        if prev_bgr.shape != curr_bgr.shape:
            raise ValueError("frames must have the same shape")
        h, w = prev_bgr.shape[:2]
        prev, curr = self._gray(prev_bgr), self._gray(curr_bgr)

        if self.downscale < 1.0:
            small = (max(1, int(w * self.downscale)), max(1, int(h * self.downscale)))
            prev = cv2.resize(prev, small, interpolation=cv2.INTER_AREA)
            curr = cv2.resize(curr, small, interpolation=cv2.INTER_AREA)

        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, **_FARNEBACK)

        if self.downscale < 1.0:
            sx = w / flow.shape[1]
            sy = h / flow.shape[0]
            flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
            flow[..., 0] *= sx      # rescale vectors, not just the raster
            flow[..., 1] *= sy
        return flow.astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/sana/Yan && python3 -m pytest tests/sfcpi/test_flow.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
cd /home/sana/Yan
git add src/sfcpi/flow.py tests/sfcpi/test_flow.py
git commit -m "feat(sfcpi): Farneback dense flow with scale-correct downscaling"
```

---

### Task 5: Detector interface and per-cell counts

**Files:**
- Create: `src/sfcpi/detect.py`
- Test: `tests/sfcpi/test_detect.py`

**Interfaces:**
- Consumes: `sfcpi.grid.CellGrid` (Task 2)
- Produces:
  - `Detection` dataclass: `x1, y1, x2, y2, score: float`; property `centroid -> tuple[float, float]`
  - `Detector` Protocol: `detect(image) -> list[Detection]`
  - `FixedDetector(detections)` for tests
  - `YoloDetector(model_path: str, conf: float = 0.35)` (ultralytics)
  - `counts_per_cell(detections, grid) -> np.ndarray` shape `(n_rows, n_cols)`

- [ ] **Step 1: Write the failing test**

```python
# tests/sfcpi/test_detect.py
import numpy as np
import pytest
from sfcpi.detect import Detection, FixedDetector, counts_per_cell
from sfcpi.grid import CellGrid

def test_centroid():
    d = Detection(x1=0, y1=0, x2=10, y2=20, score=0.9)
    assert d.centroid == (5.0, 10.0)

def test_counts_land_in_the_right_cells():
    grid = CellGrid(frame_width=4, frame_height=4, cell_size=2)
    dets = [
        Detection(0, 0, 2, 2, 0.9),     # centroid (1,1)   -> cell (0,0)
        Detection(2, 0, 4, 2, 0.9),     # centroid (3,1)   -> cell (0,1)
        Detection(2, 2, 4, 4, 0.9),     # centroid (3,3)   -> cell (1,1)
    ]
    counts = counts_per_cell(dets, grid)
    assert counts.tolist() == [[1.0, 1.0], [0.0, 1.0]]

def test_counts_shape_and_empty_case():
    grid = CellGrid(frame_width=8, frame_height=4, cell_size=2)
    counts = counts_per_cell([], grid)
    assert counts.shape == (2, 4)
    assert counts.sum() == 0.0

def test_detections_outside_frame_are_ignored():
    grid = CellGrid(frame_width=4, frame_height=4, cell_size=2)
    counts = counts_per_cell([Detection(100, 100, 110, 110, 0.9)], grid)
    assert counts.sum() == 0.0

def test_fixed_detector_returns_what_it_was_given():
    dets = [Detection(0, 0, 1, 1, 0.5)]
    assert FixedDetector(dets).detect(np.zeros((4, 4, 3), np.uint8)) == dets
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/sana/Yan && python3 -m pytest tests/sfcpi/test_detect.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sfcpi.detect'`

- [ ] **Step 3: Implement detection**

```python
# src/sfcpi/detect.py
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
            return out
        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None:
            return out
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/sana/Yan && python3 -m pytest tests/sfcpi/test_detect.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
cd /home/sana/Yan
git add src/sfcpi/detect.py tests/sfcpi/test_detect.py
git commit -m "feat(sfcpi): detector protocol and per-cell counting"
```

---

### Task 6: Pipeline orchestration

**Files:**
- Create: `src/sfcpi/pipeline.py`
- Test: `tests/sfcpi/test_pipeline.py`

**Interfaces:**
- Consumes: Tasks 1-5
- Produces:
  - `MetricsFrame` dataclass: `index: int`, `timestamp: float`, `flow_valid: bool`, `cells: dict[str, np.ndarray]`, `global_pressure: float`, `global_max_pressure: float`, `total_count: float`, `sensing_confidence: float`
  - `Pipeline(grid, flow_estimator, detector, fps, ewma_alpha=0.3)` with `run(source) -> Iterator[MetricsFrame]`

- [ ] **Step 1: Write the failing test**

```python
# tests/sfcpi/test_pipeline.py
import math
import numpy as np
import pytest
from sfcpi.detect import Detection, FixedDetector
from sfcpi.flow import FlowEstimator
from sfcpi.grid import CellGrid
from sfcpi.pipeline import MetricsFrame, Pipeline
from sfcpi.sources import SyntheticSource

def _frames(n=4, h=32, w=32):
    rng = np.random.default_rng(1)
    return [rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]

def _pipeline(detector=None):
    grid = CellGrid(frame_width=32, frame_height=32, cell_size=16)
    return Pipeline(
        grid=grid,
        flow_estimator=FlowEstimator(),
        detector=detector or FixedDetector([Detection(0, 0, 8, 8, 0.9)]),
        fps=10.0,
    )

def test_first_frame_has_no_flow_but_is_still_emitted():
    out = list(_pipeline().run(SyntheticSource(_frames(3), fps=10.0)))
    assert len(out) == 3
    assert out[0].flow_valid is False
    assert math.isnan(out[0].global_pressure)
    assert all(f.flow_valid for f in out[1:])

def test_emits_metrics_frames_with_grid_shaped_cells():
    out = list(_pipeline().run(SyntheticSource(_frames(3), fps=10.0)))
    assert isinstance(out[1], MetricsFrame)
    assert out[1].cells["pressure"].shape == (2, 2)

def test_timestamps_follow_the_source():
    out = list(_pipeline().run(SyntheticSource(_frames(3), fps=10.0)))
    assert [f.timestamp for f in out] == pytest.approx([0.0, 0.1, 0.2])

def test_total_count_reflects_detections():
    det = FixedDetector([Detection(0, 0, 8, 8, 0.9), Detection(16, 16, 24, 24, 0.9)])
    out = list(_pipeline(det).run(SyntheticSource(_frames(2), fps=10.0)))
    assert out[-1].total_count == pytest.approx(2.0)

def test_detector_failure_is_nan_not_zero():
    """Fail loud: a broken detector must not report a safe-looking 0 pressure."""
    class Broken:
        def detect(self, image):
            raise RuntimeError("model exploded")

    out = list(_pipeline(Broken()).run(SyntheticSource(_frames(3), fps=10.0)))
    last = out[-1]
    assert math.isnan(last.global_pressure)
    assert last.sensing_confidence == 0.0
    assert math.isnan(last.total_count)

def test_empty_source_yields_nothing():
    assert list(_pipeline().run(SyntheticSource([], fps=10.0))) == []

def test_ewma_smooths_towards_new_values():
    """alpha=0.5 => each step moves halfway to the new observation."""
    p = _pipeline()
    p.ewma_alpha = 0.5
    assert p._smooth(10.0) == pytest.approx(10.0)   # first value seeds the filter
    assert p._smooth(20.0) == pytest.approx(15.0)
    assert p._smooth(20.0) == pytest.approx(17.5)

def test_ewma_alpha_one_is_passthrough():
    p = _pipeline()
    p.ewma_alpha = 1.0
    p._smooth(5.0)
    assert p._smooth(9.0) == pytest.approx(9.0)

def test_ewma_rejects_nan_without_corrupting_state():
    p = _pipeline()
    p.ewma_alpha = 0.5
    p._smooth(10.0)
    assert math.isnan(p._smooth(float("nan")))
    assert p._smooth(20.0) == pytest.approx(15.0)   # state survived the NaN

def test_pipeline_rejects_bad_alpha():
    from sfcpi.pipeline import Pipeline
    with pytest.raises(ValueError, match="ewma_alpha"):
        Pipeline(grid=CellGrid(32, 32, 16), flow_estimator=FlowEstimator(),
                 detector=FixedDetector([]), fps=10.0, ewma_alpha=0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/sana/Yan && python3 -m pytest tests/sfcpi/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sfcpi.pipeline'`

- [ ] **Step 3: Implement the pipeline**

```python
# src/sfcpi/pipeline.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/sana/Yan && python3 -m pytest tests/sfcpi/test_pipeline.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
cd /home/sana/Yan
git add src/sfcpi/pipeline.py tests/sfcpi/test_pipeline.py
git commit -m "feat(sfcpi): pipeline orchestration with fail-loud sensing"
```

---

### Task 7: Metrics sink and CLI `run`

**Files:**
- Create: `src/sfcpi/sinks.py`, `src/sfcpi/cli.py`
- Test: `tests/sfcpi/test_sinks.py`, `tests/sfcpi/test_cli.py`

**Interfaces:**
- Consumes: `MetricsFrame` (Task 6), `FileSource` (Task 3)
- Produces:
  - `JsonlSink(path)` with `write(frame: MetricsFrame)`, `close()`, context-manager support
  - `frame_to_row(frame) -> dict` (per-frame globals only; cell arrays summarised)
  - `main(argv: list[str] | None = None) -> int` implementing `sfcpi run <video> --out <file>`

- [ ] **Step 1: Write the failing tests**

```python
# tests/sfcpi/test_sinks.py
import json
import math
import numpy as np
from sfcpi.pipeline import MetricsFrame
from sfcpi.sinks import JsonlSink, frame_to_row

def _frame():
    return MetricsFrame(
        index=3, timestamp=0.3, flow_valid=True,
        cells={"pressure": np.array([[1.0, 2.0], [3.0, 4.0]])},
        global_pressure=2.5, global_max_pressure=4.0,
        total_count=7.0, sensing_confidence=1.0,
    )

def test_frame_to_row_has_flat_scalar_fields():
    row = frame_to_row(_frame())
    assert row["index"] == 3
    assert row["global_pressure"] == 2.5
    assert row["total_count"] == 7.0
    assert row["sensing_confidence"] == 1.0
    assert isinstance(row["cell_pressure"], list)

def test_nan_serialises_as_null_not_zero():
    f = _frame()
    f.global_pressure = float("nan")
    row = frame_to_row(f)
    assert row["global_pressure"] is None

def test_jsonl_sink_writes_one_line_per_frame(tmp_path):
    path = tmp_path / "m.jsonl"
    with JsonlSink(str(path)) as sink:
        sink.write(_frame())
        sink.write(_frame())
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["index"] == 3
```

```python
# tests/sfcpi/test_cli.py
import json
import numpy as np
from sfcpi.cli import main

def _write_clip(path, n=6, size=32):
    import cv2
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (size, size))
    rng = np.random.default_rng(2)
    for _ in range(n):
        w.write(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
    w.release()

def test_run_writes_metrics_file(tmp_path):
    clip, out = tmp_path / "c.mp4", tmp_path / "m.jsonl"
    _write_clip(clip)
    rc = main(["run", str(clip), "--out", str(out), "--cell-size", "16", "--no-detector"])
    assert rc == 0
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 6
    assert json.loads(lines[1])["flow_valid"] is True

def test_run_missing_file_returns_nonzero(tmp_path, capsys):
    rc = main(["run", str(tmp_path / "nope.mp4"), "--out", str(tmp_path / "o.jsonl")])
    assert rc != 0
    assert "not found" in capsys.readouterr().err.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/sana/Yan && python3 -m pytest tests/sfcpi/test_sinks.py tests/sfcpi/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sfcpi.sinks'`

- [ ] **Step 3: Implement sink and CLI**

```python
# src/sfcpi/sinks.py
"""Metrics output. NaN serialises as null -- never as 0."""
from __future__ import annotations

import json
import math
from typing import Any, Dict

import numpy as np

from .pipeline import MetricsFrame


def _clean(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def frame_to_row(frame: MetricsFrame) -> Dict[str, Any]:
    pressure = frame.cells.get("pressure")
    return {
        "index": frame.index,
        "timestamp": frame.timestamp,
        "flow_valid": frame.flow_valid,
        "global_pressure": _clean(frame.global_pressure),
        "global_max_pressure": _clean(frame.global_max_pressure),
        "total_count": _clean(frame.total_count),
        "sensing_confidence": frame.sensing_confidence,
        "cell_pressure": [] if pressure is None else
                         [[_clean(float(v)) for v in row] for row in np.asarray(pressure)],
    }


class JsonlSink:
    def __init__(self, path: str) -> None:
        self._fh = open(path, "w", encoding="utf-8")

    def write(self, frame: MetricsFrame) -> None:
        self._fh.write(json.dumps(frame_to_row(frame)) + "\n")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlSink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
```

```python
# src/sfcpi/cli.py
"""Command line entry points."""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .detect import FixedDetector, YoloDetector
from .flow import FlowEstimator
from .grid import CellGrid
from .pipeline import Pipeline
from .sinks import JsonlSink
from .sources import FileSource


def _cmd_run(args: argparse.Namespace) -> int:
    source = FileSource(args.video, max_frames=args.max_frames)
    try:
        first = next(iter(source))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except StopIteration:
        print("error: video contained no frames", file=sys.stderr)
        return 2

    h, w = first.image.shape[:2]
    cell = args.cell_size
    w -= w % cell
    h -= h % cell
    grid = CellGrid(frame_width=w, frame_height=h, cell_size=cell)

    detector = FixedDetector([]) if args.no_detector else YoloDetector(args.model, conf=args.conf)

    pipeline = Pipeline(
        grid=grid,
        flow_estimator=FlowEstimator(downscale=args.downscale),
        detector=detector,
        fps=source.fps,
    )

    cropped = _CroppedSource(FileSource(args.video, max_frames=args.max_frames), w, h)
    with JsonlSink(args.out) as sink:
        for frame in pipeline.run(cropped):
            sink.write(frame)
    return 0


class _CroppedSource:
    """Crops frames so the grid divides evenly."""

    def __init__(self, inner, width: int, height: int) -> None:
        self._inner = inner
        self._w, self._h = width, height
        self.fps = inner.fps

    def __iter__(self):
        for frame in self._inner:
            self.fps = self._inner.fps
            yield type(frame)(frame.index, frame.timestamp, frame.image[: self._h, : self._w])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sfcpi")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="compute metrics for a video file")
    run.add_argument("video")
    run.add_argument("--out", required=True)
    run.add_argument("--cell-size", type=int, default=64)
    run.add_argument("--downscale", type=float, default=1.0)
    run.add_argument("--max-frames", type=int, default=None)
    run.add_argument("--model", default="yolov8n.pt")
    run.add_argument("--conf", type=float, default=0.35)
    run.add_argument("--no-detector", action="store_true",
                     help="flow-only run; counts are zero and pressure is zero")
    run.set_defaults(func=_cmd_run)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/sana/Yan && python3 -m pytest tests/sfcpi/test_sinks.py tests/sfcpi/test_cli.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
cd /home/sana/Yan
git add src/sfcpi/sinks.py src/sfcpi/cli.py tests/sfcpi/test_sinks.py tests/sfcpi/test_cli.py
git commit -m "feat(sfcpi): jsonl sink and run CLI"
```

---

### Task 8: Evaluation harness

**Files:**
- Create: `src/sfcpi/eval.py`
- Modify: `src/sfcpi/cli.py` (add the `eval` subcommand)
- Test: `tests/sfcpi/test_eval.py`

**Interfaces:**
- Consumes: metrics rows from `frame_to_row` (Task 7)
- Produces:
  - `load_labels(path) -> np.ndarray` (0/1 per frame, from a one-column CSV with header `label`)
  - `detection_latency(scores, labels, threshold) -> float | None` — seconds-equivalent in frames between event onset and first alarm; negative means the alarm preceded onset
  - `roc_auc(scores, labels) -> float`
  - `false_alarms_per_hour(scores, labels, threshold, fps) -> float`
  - `evaluate(scores, labels, threshold, fps) -> dict`

- [ ] **Step 1: Write the failing test**

```python
# tests/sfcpi/test_eval.py
import numpy as np
import pytest
from sfcpi.eval import (
    detection_latency, evaluate, false_alarms_per_hour, load_labels, roc_auc,
)

def test_load_labels(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("label\n0\n0\n1\n1\n")
    assert load_labels(str(p)).tolist() == [0, 0, 1, 1]

def test_roc_auc_perfect_separation():
    scores = np.array([0.1, 0.2, 0.9, 0.95])
    labels = np.array([0, 0, 1, 1])
    assert roc_auc(scores, labels) == pytest.approx(1.0)

def test_roc_auc_is_half_for_constant_scores():
    assert roc_auc(np.array([0.5] * 4), np.array([0, 0, 1, 1])) == pytest.approx(0.5)

def test_detection_latency_counts_frames_after_onset():
    scores = np.array([0.0, 0.0, 0.0, 0.9])   # alarm at frame 3
    labels = np.array([0, 0, 1, 1])           # onset at frame 2
    assert detection_latency(scores, labels, threshold=0.5) == 1

def test_negative_latency_means_early_warning():
    scores = np.array([0.0, 0.9, 0.9, 0.9])   # alarm at frame 1
    labels = np.array([0, 0, 1, 1])           # onset at frame 2
    assert detection_latency(scores, labels, threshold=0.5) == -1

def test_detection_latency_none_when_never_triggered():
    assert detection_latency(np.zeros(4), np.array([0, 0, 1, 1]), 0.5) is None

def test_detection_latency_none_when_no_event():
    assert detection_latency(np.ones(4), np.zeros(4), 0.5) is None

def test_false_alarms_per_hour():
    scores = np.array([0.9, 0.0, 0.9, 0.0])   # 2 alarms in negative frames
    labels = np.zeros(4)
    assert false_alarms_per_hour(scores, labels, 0.5, fps=2.0) == pytest.approx(3600.0)

def test_evaluate_reports_all_fields():
    out = evaluate(np.array([0.1, 0.2, 0.9, 0.95]), np.array([0, 0, 1, 1]), 0.5, fps=10.0)
    assert set(out) >= {"auc", "detection_latency_frames", "false_alarms_per_hour", "threshold"}

def test_nan_scores_are_treated_as_no_alarm():
    scores = np.array([np.nan, np.nan, 0.9, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert detection_latency(scores, labels, 0.5) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/sana/Yan && python3 -m pytest tests/sfcpi/test_eval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sfcpi.eval'`

- [ ] **Step 3: Implement evaluation**

```python
# src/sfcpi/eval.py
"""Offline evaluation.

Detection latency is the operationally meaningful number: Johansson et al.
report crowd pressure crossing its critical value ~10 minutes before the
Jamarat crush, so how EARLY a method fires matters more than raw accuracy.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def load_labels(path: str) -> np.ndarray:
    return pd.read_csv(path)["label"].to_numpy().astype(int)


def _alarms(scores: np.ndarray, threshold: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    return np.nan_to_num(scores, nan=-np.inf) >= threshold


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U). Ties contribute 0.5."""
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf)
    labels = np.asarray(labels).astype(int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (pos.size * neg.size))


def detection_latency(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> Optional[int]:
    """Frames between event onset and first alarm. Negative = early warning."""
    labels = np.asarray(labels).astype(int)
    positives = np.flatnonzero(labels == 1)
    if positives.size == 0:
        return None
    alarms = np.flatnonzero(_alarms(scores, threshold))
    if alarms.size == 0:
        return None
    return int(alarms[0] - positives[0])


def false_alarms_per_hour(
    scores: np.ndarray, labels: np.ndarray, threshold: float, fps: float
) -> float:
    labels = np.asarray(labels).astype(int)
    negatives = labels == 0
    negative_frames = int(negatives.sum())
    if negative_frames == 0 or fps <= 0:
        return float("nan")
    false_alarms = int((_alarms(scores, threshold) & negatives).sum())
    return float(false_alarms / (negative_frames / fps) * 3600.0)


def evaluate(
    scores: np.ndarray, labels: np.ndarray, threshold: float, fps: float
) -> Dict[str, float]:
    return {
        "threshold": float(threshold),
        "auc": roc_auc(scores, labels),
        "detection_latency_frames": detection_latency(scores, labels, threshold),
        "false_alarms_per_hour": false_alarms_per_hour(scores, labels, threshold, fps),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/sana/Yan && python3 -m pytest tests/sfcpi/test_eval.py -v`
Expected: all PASS

- [ ] **Step 5: Add the `eval` subcommand**

In `src/sfcpi/cli.py`, add this function and register it in `build_parser()`:

```python
def _cmd_eval(args: argparse.Namespace) -> int:
    import json
    import numpy as np
    from .eval import evaluate, load_labels

    scores = []
    with open(args.metrics, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            value = row.get(args.score_field)
            scores.append(float("nan") if value is None else float(value))
    labels = load_labels(args.labels)
    n = min(len(scores), len(labels))
    result = evaluate(np.array(scores[:n]), labels[:n], args.threshold, args.fps)
    print(json.dumps(result, indent=2))
    return 0
```

```python
    ev = sub.add_parser("eval", help="score a metrics file against frame labels")
    ev.add_argument("metrics")
    ev.add_argument("--labels", required=True)
    ev.add_argument("--threshold", type=float, default=0.02)
    ev.add_argument("--fps", type=float, default=25.0)
    ev.add_argument("--score-field", default="global_max_pressure")
    ev.set_defaults(func=_cmd_eval)
```

- [ ] **Step 6: Run the whole suite**

Run: `cd /home/sana/Yan && python3 -m pytest -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
cd /home/sana/Yan
git add src/sfcpi/eval.py src/sfcpi/cli.py tests/sfcpi/test_eval.py
git commit -m "feat(sfcpi): evaluation harness with detection latency and AUC"
```

---

### Task 9: Occlusion characterisation experiment

**Files:**
- Create: `src/sfcpi/occlusion.py`
- Modify: `src/sfcpi/cli.py` (add the `occlusion` subcommand)
- Test: `tests/sfcpi/test_occlusion.py`

**Interfaces:**
- Consumes: `Detection` (Task 5)
- Produces:
  - `DetectionRatePoint` dataclass: `true_count: int`, `detected_count: int`, `density_proxy: float`, `ratio: float`
  - `detection_rate_curve(points, n_bins=10) -> pd.DataFrame` with columns `bin_center`, `mean_ratio`, `std_ratio`, `n`
  - `is_monotonic_decreasing(df, tolerance=0.05) -> bool` — the decision rule from spec section 8

- [ ] **Step 1: Write the failing test**

```python
# tests/sfcpi/test_occlusion.py
import numpy as np
import pytest
from sfcpi.occlusion import (
    DetectionRatePoint, detection_rate_curve, is_monotonic_decreasing,
)

def _points(ratios_by_count):
    return [
        DetectionRatePoint(true_count=c, detected_count=int(round(c * r)),
                           density_proxy=float(c), ratio=r)
        for c, r in ratios_by_count
    ]

def test_ratio_computed_from_counts():
    p = DetectionRatePoint.from_counts(true_count=100, detected_count=40, density_proxy=1.0)
    assert p.ratio == pytest.approx(0.4)

def test_zero_true_count_gives_nan_ratio():
    p = DetectionRatePoint.from_counts(true_count=0, detected_count=0, density_proxy=0.0)
    assert np.isnan(p.ratio)

def test_curve_bins_and_averages():
    pts = _points([(10, 0.9), (12, 0.9), (100, 0.3), (110, 0.3)])
    df = detection_rate_curve(pts, n_bins=2)
    assert list(df.columns) == ["bin_center", "mean_ratio", "std_ratio", "n"]
    assert len(df) == 2
    assert df["mean_ratio"].iloc[0] == pytest.approx(0.9)
    assert df["mean_ratio"].iloc[-1] == pytest.approx(0.3)

def test_monotonic_decreasing_detects_clean_degradation():
    df = detection_rate_curve(_points([(10, 0.95), (50, 0.6), (100, 0.3), (150, 0.15)]), n_bins=4)
    assert is_monotonic_decreasing(df) is True

def test_noisy_curve_is_rejected():
    df = detection_rate_curve(_points([(10, 0.3), (50, 0.9), (100, 0.2), (150, 0.95)]), n_bins=4)
    assert is_monotonic_decreasing(df) is False

def test_empty_points_raise():
    with pytest.raises(ValueError, match="no points"):
        detection_rate_curve([], n_bins=4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/sana/Yan && python3 -m pytest tests/sfcpi/test_occlusion.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sfcpi.occlusion'`

- [ ] **Step 3: Implement the experiment**

```python
# src/sfcpi/occlusion.py
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

    Tolerance permits small non-monotonic wobble; anything larger means the
    curve is not stable enough to invert into a correction.
    """
    values = df["mean_ratio"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return False
    return bool(np.all(np.diff(values) <= tolerance))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/sana/Yan && python3 -m pytest tests/sfcpi/test_occlusion.py -v`
Expected: all PASS

- [ ] **Step 5: Add the `occlusion` subcommand**

In `src/sfcpi/cli.py`, add and register:

```python
def _cmd_occlusion(args: argparse.Namespace) -> int:
    import csv
    import json
    from .occlusion import DetectionRatePoint, detection_rate_curve, is_monotonic_decreasing

    points = []
    with open(args.counts, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            points.append(DetectionRatePoint.from_counts(
                true_count=int(row["true_count"]),
                detected_count=int(row["detected_count"]),
                density_proxy=float(row.get("density_proxy", row["true_count"])),
            ))
    curve = detection_rate_curve(points, n_bins=args.bins)
    curve.to_csv(args.out, index=False)
    print(curve.to_string(index=False))
    print(json.dumps({"monotonic_decreasing": is_monotonic_decreasing(curve)}, indent=2))
    return 0
```

```python
    oc = sub.add_parser("occlusion", help="detector degradation vs density")
    oc.add_argument("counts", help="CSV with true_count,detected_count[,density_proxy]")
    oc.add_argument("--out", required=True)
    oc.add_argument("--bins", type=int, default=10)
    oc.set_defaults(func=_cmd_occlusion)
```

- [ ] **Step 6: Run the whole suite**

Run: `cd /home/sana/Yan && python3 -m pytest -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
cd /home/sana/Yan
git add src/sfcpi/occlusion.py src/sfcpi/cli.py tests/sfcpi/test_occlusion.py
git commit -m "feat(sfcpi): occlusion characterisation experiment"
```

---

### Task 10: End-to-end integration test and README

**Files:**
- Create: `tests/sfcpi/test_integration.py`, `src/sfcpi/README.md`
- Test: as above

**Interfaces:**
- Consumes: everything
- Produces: no new API

- [ ] **Step 1: Write the failing integration test**

```python
# tests/sfcpi/test_integration.py
import json
import numpy as np
import pytest
from sfcpi.cli import main

def _moving_clip(path, n=20, size=64):
    """A textured block that accelerates: pressure should rise over time."""
    import cv2
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (size, size))
    rng = np.random.default_rng(7)
    texture = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    for i in range(n):
        frame = np.zeros((size, size, 3), np.uint8)
        shift = int(i * i * 0.05)
        frame[:, :] = np.roll(texture, shift, axis=1)
        writer.write(frame)
    writer.release()

def test_end_to_end_produces_wellformed_metrics(tmp_path):
    clip, out = tmp_path / "c.mp4", tmp_path / "m.jsonl"
    _moving_clip(clip)
    assert main(["run", str(clip), "--out", str(out),
                 "--cell-size", "32", "--no-detector"]) == 0

    rows = [json.loads(l) for l in out.read_text().strip().splitlines()]
    assert len(rows) == 20
    assert rows[0]["flow_valid"] is False
    assert all(r["flow_valid"] for r in rows[1:])
    for r in rows[1:]:
        assert len(r["cell_pressure"]) == 2
        assert len(r["cell_pressure"][0]) == 2

def test_eval_runs_over_produced_metrics(tmp_path, capsys):
    clip, out, labels = tmp_path / "c.mp4", tmp_path / "m.jsonl", tmp_path / "l.csv"
    _moving_clip(clip)
    main(["run", str(clip), "--out", str(out), "--cell-size", "32", "--no-detector"])
    labels.write_text("label\n" + "\n".join(["0"] * 10 + ["1"] * 10) + "\n")

    assert main(["eval", str(out), "--labels", str(labels),
                 "--threshold", "0.0", "--fps", "10"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert "auc" in result and "detection_latency_frames" in result
```

- [ ] **Step 2: Run test to verify it fails or reveals integration bugs**

Run: `cd /home/sana/Yan && python3 -m pytest tests/sfcpi/test_integration.py -v`
Expected: FAIL initially if any wiring is wrong; fix the wiring, not the test.

- [ ] **Step 3: Write the README**

```markdown
# sfcpi — Scale-Free Crowd Pressure Index

Computes crowd pressure `P = rho * Var(v)` from **uncalibrated** video.

`P` has units of `s^-2` and carries no length dimension, so the unknown
metres-per-pixel scale cancels exactly. Pressure computed in pixel space equals
pressure in real units, given only the frame rate — see
`docs/superpowers/specs/2026-08-27-sf-cpi-design.md`.

## Usage

    python3 -m sfcpi.cli run clip.mp4 --out metrics.jsonl --cell-size 64
    python3 -m sfcpi.cli eval metrics.jsonl --labels labels.csv --fps 25
    python3 -m sfcpi.cli occlusion counts.csv --out curve.csv

## Design rules

- `metrics/` is pure numpy: no cv2, no I/O. The physics stays unit-testable.
- Velocities are px/frame; `fps` converts to per-second. Never mix units.
- **Unknown values are NaN, never 0** — zero reads as "safe" in a safety signal.

## Tests

    python3 -m pytest

The scale-invariance property test in `tests/sfcpi/test_pressure.py` is the
executable form of the paper's central claim. If it fails, the claim is wrong.
```

- [ ] **Step 4: Run the full suite**

Run: `cd /home/sana/Yan && python3 -m pytest -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
cd /home/sana/Yan
git add tests/sfcpi/test_integration.py src/sfcpi/README.md
git commit -m "test(sfcpi): end-to-end integration and package README"
```

---

## Deferred to later cycles

- **Cycle 2:** live WebRTC ingest (`WebRTCSource` behind the same Protocol), risk state machine with hysteresis, sensing-confidence gating.
- **Cycle 3:** dashboard visualisation, viewer-independent alerting, device REST/SEI metadata path.
- **Not planned:** head-orientation divergence (demoted — patent prior art, see gap analysis section 4.5).
