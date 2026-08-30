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
