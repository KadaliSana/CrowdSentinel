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
