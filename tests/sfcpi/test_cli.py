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
