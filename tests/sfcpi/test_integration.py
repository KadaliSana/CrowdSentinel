import json
import math

import numpy as np
import pytest
from sfcpi.cli import main
from sfcpi.detect import Detection, FixedDetector, NullDetector
from sfcpi.flow import FlowEstimator
from sfcpi.grid import CellGrid
from sfcpi.pipeline import Pipeline
from sfcpi.sources import SyntheticSource

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


# --- I2: end-to-end coverage of the COUNT path ---
#
# The two tests above run --no-detector, so every cell_pressure is null and
# they would pass even if crowd_pressure returned a constant. These exercise a
# detector that returns real detections, so the density factor is live and the
# asserted pressure is a real number that has to move the right way.
#
# Frames are fed in memory rather than through an mp4: VideoWriter's mp4v codec
# is lossy, and unlike the shape-only tests above these assert on magnitudes.
# The CLI/source layer is covered by the tests above and by test_cli.py.

_ACCEL_SIZE = 64


def _counterflow_accelerating_frames(n=24, size=_ACCEL_SIZE):
    """Left half slides right, right half slides left, faster every few frames.

    Counterflow, not bulk translation: a rigid translation has near-zero
    velocity variance no matter how fast it moves, so it could not show
    pressure rising. Per-frame displacement steps up every 6 frames, so the
    velocity variance -- and hence the pressure -- rises in stages.
    """
    rng = np.random.default_rng(11)
    texture = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    frames, shift = [], 0
    for i in range(n):
        frame = np.empty_like(texture)
        frame[:, : size // 2] = np.roll(texture, shift, axis=1)[:, : size // 2]
        frame[:, size // 2 :] = np.roll(texture, -shift, axis=1)[:, size // 2 :]
        frames.append(frame)
        shift += 1 + i // 6
    return frames


def _four_detections():
    return [Detection(x, y, x + 8.0, y + 8.0, 0.9)
            for x, y in [(4, 4), (20, 20), (36, 36), (52, 52)]]


def _run_accelerating_pipeline(detector, fps=10.0):
    pipeline = Pipeline(
        grid=CellGrid(frame_width=_ACCEL_SIZE, frame_height=_ACCEL_SIZE,
                      cell_size=_ACCEL_SIZE),
        flow_estimator=FlowEstimator(),
        detector=detector,
        fps=fps,
    )
    frames = _counterflow_accelerating_frames()
    return list(pipeline.run(SyntheticSource(frames, fps=fps)))


def test_pressure_is_finite_nonzero_and_rises_with_a_real_detector():
    out = _run_accelerating_pipeline(FixedDetector(_four_detections()))

    assert out[0].flow_valid is False          # no previous frame to flow from
    scored = out[1:]
    assert all(f.flow_valid for f in scored)

    # the count path is live: four detections in a single cell
    assert all(f.total_count == pytest.approx(4.0) for f in scored)
    assert all(f.sensing_confidence == pytest.approx(1.0) for f in scored)

    pressures = [f.global_max_pressure for f in scored]
    assert all(math.isfinite(p) for p in pressures)
    assert all(p > 0.0 for p in pressures)

    early = float(np.mean(pressures[:6]))
    late = float(np.mean(pressures[-6:]))
    assert late > early
    assert late > 2.0 * early                  # a real rise, not float drift


def test_flow_only_mode_yields_nan_pressure_on_the_same_clip():
    """The same clip with no detector: the motion is identical, so anything
    finite here would mean the count factor was not actually contributing.
    Counts are UNKNOWN, so pressure is NaN and confidence is 0 -- never 0.0
    pressure with confidence 1.0."""
    out = _run_accelerating_pipeline(NullDetector())
    scored = out[1:]
    assert all(math.isnan(f.total_count) for f in scored)
    assert all(math.isnan(f.global_max_pressure) for f in scored)
    assert all(f.sensing_confidence == 0.0 for f in scored)
    # motion is still measured even though the count is not
    assert all(np.isfinite(f.cells["mean_speed"]).all() for f in scored)
    assert all(np.isfinite(f.cells["velocity_variance"]).all() for f in scored)
