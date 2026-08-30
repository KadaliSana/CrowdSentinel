import json

import numpy as np
import pytest
from sfcpi.cli import main
from sfcpi.eval import (
    coverage, detection_latency, evaluate, false_alarms_per_hour, load_labels,
    roc_auc,
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
    """A system that cannot see must not fire. This is only defensible because
    `evaluate` now reports coverage alongside -- see the coverage tests."""
    scores = np.array([np.nan, np.nan, 0.9, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert detection_latency(scores, labels, 0.5) == 0

def test_false_alarms_per_hour_nan_when_no_negative_frames():
    scores = np.array([0.9, 0.9, 0.9, 0.9])
    labels = np.ones(4)
    assert np.isnan(false_alarms_per_hour(scores, labels, 0.5, fps=2.0))


# --- C1: unscorable frames must be visible, never scored as "safe" ---

def test_coverage_reports_scorable_fraction():
    assert coverage(np.array([np.nan, 1.0, 2.0, np.nan])) == pytest.approx(0.5)
    assert coverage(np.array([1.0, 2.0])) == pytest.approx(1.0)
    assert np.isnan(coverage(np.array([])))


def test_evaluate_reports_coverage_fields():
    """A run that was blind for most frames must SAY so. Without these fields
    a broken detector reports FAR/h 0 and reads as "never false-alarms"."""
    scores = np.array([np.nan, np.nan, np.nan, 0.9])
    out = evaluate(scores, np.array([0, 0, 1, 1]), 0.5, fps=10.0)
    assert out["n_frames"] == 4
    assert out["n_unscorable"] == 3
    assert out["coverage"] == pytest.approx(0.25)
    # the counts must be plain ints so json.dumps can serialise them
    assert isinstance(out["n_frames"], int)
    assert isinstance(out["n_unscorable"], int)


def test_evaluate_coverage_is_one_when_all_frames_scored():
    out = evaluate(np.array([0.1, 0.2, 0.9, 0.95]), np.array([0, 0, 1, 1]), 0.5, fps=10.0)
    assert out["n_unscorable"] == 0
    assert out["coverage"] == pytest.approx(1.0)


def test_roc_auc_excludes_unscorable_frames_from_both_classes():
    """NaN must not be ranked as maximally-safe. Here the two finite scores
    separate perfectly; the NaNs carry no information and must not change it."""
    scores = np.array([0.1, np.nan, np.nan, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert roc_auc(scores, labels) == pytest.approx(1.0)


def test_roc_auc_nan_when_a_class_has_no_scorable_frame():
    scores = np.array([0.1, 0.2, np.nan, np.nan])
    labels = np.array([0, 0, 1, 1])
    assert np.isnan(roc_auc(scores, labels))


def test_far_denominator_excludes_unscorable_negatives():
    """Two negative frames were scored (one alarming); two were never scored.
    Counting the blind frames as quiet time would halve the reported rate."""
    scores = np.array([0.9, 0.0, np.nan, np.nan])
    labels = np.zeros(4)
    # 1 episode over 2 scorable negative frames at 2 fps = 1 s -> 3600/h
    assert false_alarms_per_hour(scores, labels, 0.5, fps=2.0) == pytest.approx(3600.0)


def test_far_nan_when_every_negative_frame_is_unscorable():
    scores = np.array([np.nan] * 4)
    assert np.isnan(false_alarms_per_hour(scores, np.zeros(4), 0.5, fps=2.0))


# --- I3: FAR/h counts alarm EPISODES, not alarming frames ---

def test_far_counts_contiguous_block_as_one_episode():
    """At 25 fps a single 4-second false alarm is 100 alarming frames. Counting
    frames reports 90,000/h for one operator interruption."""
    scores = np.concatenate([np.zeros(10), np.full(100, 0.9), np.zeros(140)])
    labels = np.zeros(250)
    # 1 episode over 250 frames at 25 fps = 10 s -> 360/h
    assert false_alarms_per_hour(scores, labels, 0.5, fps=25.0) == pytest.approx(360.0)


def test_far_counts_two_separated_blocks_as_two_episodes():
    scores = np.array([0.9, 0.9, 0.0, 0.9, 0.9, 0.0])
    labels = np.zeros(6)
    # 2 episodes over 6 frames at 6 fps = 1 s -> 7200/h
    assert false_alarms_per_hour(scores, labels, 0.5, fps=6.0) == pytest.approx(7200.0)


def test_far_alarm_at_the_very_first_frame_counts_as_an_episode():
    scores = np.array([0.9, 0.9])
    labels = np.zeros(2)
    assert false_alarms_per_hour(scores, labels, 0.5, fps=2.0) == pytest.approx(3600.0)


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


def test_cli_eval_reports_coverage_for_a_blind_run(tmp_path, capsys):
    """The end-to-end shape of C1: a run where the score was almost never
    computable must not read as a clean, quiet run."""
    metrics_path, labels_path = _write_metrics_and_labels(
        tmp_path, [None, None, None, 0.9], [0, 0, 1, 1]
    )
    rc = main(["eval", metrics_path, "--labels", labels_path, "--fps", "10"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["n_frames"] == 4
    assert parsed["n_unscorable"] == 3
    assert parsed["coverage"] == pytest.approx(0.25)


# --- I4: fps comes from the metrics file's timestamps unless given ---

def _write_metrics_with_timestamps(tmp_path, score_values, label_values, fps):
    metrics_path = tmp_path / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as fh:
        for i, value in enumerate(score_values):
            fh.write(json.dumps(
                {"index": i, "timestamp": i / fps, "global_max_pressure": value}
            ) + "\n")
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text("label\n" + "\n".join(str(v) for v in label_values) + "\n")
    return str(metrics_path), str(labels_path)


def test_cli_eval_derives_fps_from_timestamps(tmp_path, capsys):
    """A 10 fps clip evaluated with defaults must not be scored at 25 fps --
    that reports FAR/h 2.5x wrong."""
    metrics_path, labels_path = _write_metrics_with_timestamps(
        tmp_path, [0.9] + [0.0] * 9, [0] * 10, fps=10.0
    )
    assert main(["eval", metrics_path, "--labels", labels_path]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["fps"] == pytest.approx(10.0)
    # 1 episode over 10 negative frames at 10 fps = 1 s -> 3600/h
    assert parsed["false_alarms_per_hour"] == pytest.approx(3600.0)


def test_cli_eval_explicit_fps_overrides_timestamps(tmp_path, capsys):
    metrics_path, labels_path = _write_metrics_with_timestamps(
        tmp_path, [0.9] + [0.0] * 9, [0] * 10, fps=10.0
    )
    assert main(["eval", metrics_path, "--labels", labels_path, "--fps", "5"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["fps"] == pytest.approx(5.0)


def test_cli_eval_falls_back_to_default_fps_without_timestamps(tmp_path, capsys):
    # _write_metrics_and_labels writes no `timestamp` column at all.
    metrics_path, labels_path = _write_metrics_and_labels(
        tmp_path, [0.9] + [0.0] * 9, [0] * 10
    )
    assert main(["eval", metrics_path, "--labels", labels_path]) == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["fps"] == pytest.approx(25.0)
    assert "assuming --fps" in captured.err
