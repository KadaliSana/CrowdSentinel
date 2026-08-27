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
