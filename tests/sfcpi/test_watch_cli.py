import json
from sfcpi.cli import main

def _write_metrics(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for t, p in rows:
            fh.write(json.dumps({
                "index": int(t * 10), "timestamp": t, "flow_valid": True,
                "global_pressure": p, "global_max_pressure": p,
                "total_count": 5.0, "sensing_confidence": 1.0, "cell_pressure": [[p]],
            }) + "\n")

def test_watch_reports_alerts_and_exits_zero(tmp_path, capsys):
    path = tmp_path / "m.jsonl"
    _write_metrics(path, [(i * 0.5, 0.0 if i < 4 else 0.03) for i in range(20)])
    assert main(["watch", str(path)]) == 0
    err = capsys.readouterr().err
    assert "ALERT" in err and "HIGH" in err.upper()

def test_watch_is_quiet_on_calm_metrics(tmp_path, capsys):
    path = tmp_path / "m.jsonl"
    _write_metrics(path, [(i * 0.5, 0.0) for i in range(20)])
    assert main(["watch", str(path)]) == 0
    assert "ALERT" not in capsys.readouterr().err

def test_null_pressure_raises_sensor_blind_not_silence(tmp_path, capsys):
    """A null in the metrics file means the sensor failed -- it must be heard."""
    path = tmp_path / "m.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(40):
            fh.write(json.dumps({
                "index": i, "timestamp": i * 0.5, "flow_valid": True,
                "global_pressure": None, "global_max_pressure": None,
                "total_count": None, "sensing_confidence": 0.0, "cell_pressure": [[None]],
            }) + "\n")
    assert main(["watch", str(path)]) == 0
    assert "sensor-blind" in capsys.readouterr().err

def test_sns_requires_a_topic_arn(tmp_path, capsys):
    path = tmp_path / "m.jsonl"
    _write_metrics(path, [(0.0, 0.0)])
    assert main(["watch", str(path), "--sns"]) != 0
    assert "topic" in capsys.readouterr().err.lower()
