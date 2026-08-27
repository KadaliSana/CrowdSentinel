import json

import numpy as np
import pytest
from sfcpi.cli import main
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

def test_false_alarms_per_hour_nan_when_no_negative_frames():
    scores = np.array([0.9, 0.9, 0.9, 0.9])
    labels = np.ones(4)
    assert np.isnan(false_alarms_per_hour(scores, labels, 0.5, fps=2.0))


def _write_metrics_and_labels(tmp_path, score_values, label_values):
    metrics_path = tmp_path / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as fh:
        for i, value in enumerate(score_values):
            fh.write(json.dumps({"index": i, "global_max_pressure": value}) + "\n")
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text(
        "label\n" + "\n".join(str(v) for v in label_values) + "\n"
    )
    return str(metrics_path), str(labels_path)


def test_cli_eval_serialises_nan_auc_as_null_strict_json(tmp_path, capsys):
    # All-zero labels -> roc_auc returns NaN (one class absent). The printed
    # output must be strict JSON with `null`, never a bare `NaN` literal.
    metrics_path, labels_path = _write_metrics_and_labels(
        tmp_path, [0.1, 0.2, None, 0.9], [0, 0, 0, 0]
    )
    rc = main(["eval", metrics_path, "--labels", labels_path, "--fps", "10"])
    assert rc == 0
    raw = capsys.readouterr().out
    assert "NaN" not in raw
    parsed = json.loads(raw)
    assert parsed["auc"] is None


def test_cli_eval_serialises_nan_false_alarms_as_null_strict_json(tmp_path, capsys):
    # All-positive labels -> false_alarms_per_hour has zero negative frames
    # and returns NaN. Same strict-JSON requirement as the auc case.
    metrics_path, labels_path = _write_metrics_and_labels(
        tmp_path, [0.1, 0.2, 0.8, 0.9], [1, 1, 1, 1]
    )
    rc = main(["eval", metrics_path, "--labels", labels_path, "--fps", "10"])
    assert rc == 0
    raw = capsys.readouterr().out
    assert "NaN" not in raw
    parsed = json.loads(raw)
    assert parsed["false_alarms_per_hour"] is None
