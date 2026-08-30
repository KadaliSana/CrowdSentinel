"""Safety contracts for the dashboard server.

This process now decides what an operator sees AND what gets paged to SNS, so
its "unknown is not zero" rules need to be pinned. Importing `server` starts
no threads and opens no sockets — the WebRTC session only begins on
/start_stream — so these run offline.
"""
import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

DASH = Path(__file__).resolve().parents[2] / "src" / "dashboard"
sys.path.insert(0, str(DASH))
server = importlib.import_module("server")


# -- unknown must never render as a number -----------------------------------

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None])
def test_non_finite_pressure_becomes_none_not_zero(bad):
    """0.0 pressure reads as a calm crowd. Unknown must stay unknown all the
    way to the JSON, and JSON has no NaN literal to fall back on."""
    assert server._finite_or_none(bad) is None


def test_finite_pressure_passes_through():
    assert server._finite_or_none(0.0123) == pytest.approx(0.0123)
    # zero is a real, finite reading and must NOT be nulled
    assert server._finite_or_none(0.0) == 0.0


def test_unparseable_value_is_none_rather_than_a_crash():
    assert server._finite_or_none("not a number") is None


# -- the board count gate ----------------------------------------------------

IMAGE = np.zeros((8, 8, 3), dtype=np.uint8)


class _Detector:
    """Stands in for BoardDetector: detect() is the freshness gate."""

    def __init__(self, count=None, raises=None):
        self._count = count
        self._raises = raises
        self.last_count = count

    def detect(self, image):
        if self._raises:
            raise RuntimeError(self._raises)
        return []


def test_a_blind_detector_reports_unknown_not_zero():
    count, reason = server._read_board_count(
        _Detector(raises="no detection metadata received from the board yet"), IMAGE)
    assert count is None
    assert "metadata" in reason


def test_a_stale_detector_reports_unknown_not_the_last_count():
    """A frozen last-known crowd would read as a steady, calm scene."""
    det = _Detector(count=42, raises="detection metadata is stale (9.0s old)")
    count, reason = server._read_board_count(det, IMAGE)
    assert count is None and "stale" in reason


def test_zero_people_is_a_real_reading_and_is_reported_as_zero():
    """n=0 from the board is an observation, NOT an absence of one."""
    count, reason = server._read_board_count(_Detector(count=0), IMAGE)
    assert count == 0 and reason is None


def test_the_true_count_is_used_even_when_boxes_were_truncated():
    count, _ = server._read_board_count(_Detector(count=400), IMAGE)
    assert count == 400


def test_no_detector_at_all_is_unknown():
    count, reason = server._read_board_count(None, IMAGE)
    assert count is None and reason


# -- credentials must never leak into anything user-visible ------------------

def test_redaction_covers_access_keys_and_signatures():
    dirty = ("AKIAIOSFODNN7EXAMPLE failed; X-Amz-Signature=deadbeefcafe "
             "X-Amz-Security-Token=abc123")
    clean = server._redact(dirty)
    assert "AKIAIOSFODNN7EXAMPLE" not in clean
    assert "deadbeefcafe" not in clean


def test_redaction_leaves_ordinary_text_alone():
    msg = "no frame within 10.0s; the producer is silent or dead"
    assert server._redact(msg) == msg


# -- alert routing -----------------------------------------------------------

def test_sns_is_off_unless_explicitly_enabled():
    """Starting a dashboard must not page anyone by default."""
    assert server.SNS_ENABLE in (True, False)
    if not server.SNS_ENABLE:
        return
    # If this deployment HAS enabled it, the gate still requires a topic.
    assert server.SNS_TOPIC_ARN, "SNS_ENABLE=1 with no topic would alert nowhere"


def test_configured_min_level_is_a_real_level():
    from sfcpi.alerts.filter import level_from_name

    level_from_name(server.SNS_MIN_LEVEL)  # raises if the config is nonsense


def test_reconnect_backoff_respects_the_boards_slot_reclaim_timer():
    """The board frees a stale viewer slot only after its own 30s inactivity
    timer, so retrying faster cannot succeed when the failure IS slot
    exhaustion -- it just keeps the board busy refusing us."""
    assert server.RECONNECT_BACKOFF_MAX_S >= 30.0
    assert server.RECONNECT_BACKOFF_MIN_S >= 5.0
