import json
import numpy as np
import pytest
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


def test_run_unopenable_file_returns_nonzero_naming_the_path(tmp_path, capsys):
    """An existing-but-corrupt video raises a bare OSError from FileSource.
    FileNotFoundError is a SUBCLASS, so catching only it let this escape as a
    traceback instead of the spec's explicit error naming the file."""
    bad = tmp_path / "corrupt.mp4"
    bad.write_bytes(b"\x00\xff" * 512)
    rc = main(["run", str(bad), "--out", str(tmp_path / "o.jsonl")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "corrupt.mp4" in err
    assert "traceback" not in err.lower()


def test_no_detector_reports_unknown_counts_not_zero(tmp_path):
    """--no-detector measures no people, so the count is UNKNOWN. Reporting
    count 0, pressure 0.0 and sensing_confidence 1.0 would be full trust in
    the one mode that counts nobody."""
    clip, out = tmp_path / "c.mp4", tmp_path / "m.jsonl"
    _write_clip(clip)
    assert main(["run", str(clip), "--out", str(out),
                 "--cell-size", "16", "--no-detector"]) == 0
    rows = [json.loads(l) for l in out.read_text().strip().splitlines()]
    for row in rows[1:]:
        assert row["total_count"] is None
        assert row["global_max_pressure"] is None
        assert row["sensing_confidence"] == 0.0
        assert all(v is None for r in row["cell_pressure"] for v in r)


def test_null_detector_raises_rather_than_returning_an_empty_list():
    """The degradation path must be the EXISTING fail-loud one -- a detector
    that returns [] would silently succeed and produce zeros."""
    import numpy as np
    from sfcpi.detect import NullDetector
    with pytest.raises(RuntimeError, match="not zero"):
        NullDetector().detect(np.zeros((4, 4, 3), dtype=np.uint8))
