import json
import pytest
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
    """A null in the metrics file means the sensor failed -- it must be heard.

    Uses an explicit --blind-alert (rather than the 30s default) so the run
    stays fast; the default itself is covered separately below.
    """
    path = tmp_path / "m.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(40):
            fh.write(json.dumps({
                "index": i, "timestamp": i * 0.5, "flow_valid": True,
                "global_pressure": None, "global_max_pressure": None,
                "total_count": None, "sensing_confidence": 0.0, "cell_pressure": [[None]],
            }) + "\n")
    assert main(["watch", str(path), "--blind-alert", "2.0"]) == 0
    assert "sensor-blind" in capsys.readouterr().err

def test_sns_requires_a_topic_arn(tmp_path, capsys):
    path = tmp_path / "m.jsonl"
    _write_metrics(path, [(0.0, 0.0)])
    assert main(["watch", str(path), "--sns"]) != 0
    assert "topic" in capsys.readouterr().err.lower()

def test_missing_timestamps_never_silently_produce_zero_alerts(tmp_path, capsys):
    """C1 repro: 200 rows at a clearly-CRITICAL pressure, no `timestamp` key.

    Before the fix, `ts = 0.0 if ts is None else ...` collided every row at
    t=0.0, dwell never completed, and the run exited 0 with no alerts and no
    warning -- a silent empty alert stream reading as a calm crowd.
    """
    path = tmp_path / "m.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(200):
            fh.write(json.dumps({
                "index": i, "flow_valid": True,
                "global_pressure": 0.05, "global_max_pressure": 0.05,
                "total_count": 20.0, "sensing_confidence": 0.9,
                "cell_pressure": [[0.05]],
            }) + "\n")
    rc = main(["watch", str(path)])
    err = capsys.readouterr().err
    assert rc == 0
    assert "warning" in err.lower() and "timestamp" in err.lower()
    assert "ALERT" in err and "CRITICAL" in err.upper()

def test_watch_passes_coverage_through_to_log_sink(tmp_path, capsys):
    path = tmp_path / "m.jsonl"
    _write_metrics(path, [(i * 0.5, 0.0 if i < 4 else 0.03) for i in range(20)])
    assert main(["watch", str(path)]) == 0
    err = capsys.readouterr().err
    assert "coverage=100.0%" in err

def test_blind_alert_default_is_30_independent_of_min_dwell(tmp_path, monkeypatch):
    path = tmp_path / "m.jsonl"
    _write_metrics(path, [(0.0, 0.0)])
    captured = {}

    class _FakeMachine:
        def __init__(self, thresholds, min_dwell_s, min_realert_s, blind_alert_s):
            captured["min_dwell_s"] = min_dwell_s
            captured["blind_alert_s"] = blind_alert_s

        def update(self, timestamp, pressure, coverage=None):
            return None

    monkeypatch.setattr("sfcpi.risk.machine.RiskStateMachine", _FakeMachine)
    assert main(["watch", str(path), "--min-dwell", "5.0"]) == 0
    assert captured["min_dwell_s"] == 5.0
    assert captured["blind_alert_s"] == 30.0

def test_min_realert_help_reflects_actual_bypass_rule(capsys):
    with pytest.raises(SystemExit):
        main(["watch", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "escalations always bypass" not in out
    assert "strictly above the last alerted level" in out

def test_watch_reports_missing_metrics_file(tmp_path, capsys):
    missing = tmp_path / "does-not-exist.jsonl"
    rc = main(["watch", str(missing)])
    err = capsys.readouterr().err
    assert rc != 0
    assert "error" in err.lower() and str(missing) in err

def test_watch_reports_unparseable_metrics_file(tmp_path, capsys):
    path = tmp_path / "bad.jsonl"
    path.write_text("{this is not json\n", encoding="utf-8")
    rc = main(["watch", str(path)])
    err = capsys.readouterr().err
    assert rc != 0
    assert "error" in err.lower()
