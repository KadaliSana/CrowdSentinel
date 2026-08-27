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

def test_nan_sensing_confidence_serialises_as_null():
    f = _frame()
    f.sensing_confidence = float("nan")
    assert frame_to_row(f)["sensing_confidence"] is None

def test_nan_inside_cell_pressure_serialises_as_null():
    f = _frame()
    f.cells = {"pressure": np.array([[1.0, np.nan], [np.nan, 4.0]])}
    row = frame_to_row(f)
    assert row["cell_pressure"] == [[1.0, None], [None, 4.0]]

def test_written_jsonl_contains_no_bare_nan_literal(tmp_path):
    f = _frame()
    f.global_pressure = float("nan")
    f.sensing_confidence = float("nan")
    f.cells = {"pressure": np.array([[np.nan]])}
    path = tmp_path / "m.jsonl"
    with JsonlSink(str(path)) as sink:
        sink.write(f)
    raw = path.read_text()
    assert "NaN" not in raw
    assert "null" in raw
