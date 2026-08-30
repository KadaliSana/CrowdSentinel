"""Detections come from the board's NPU, not from a host-side model.

The AmebaPro2 runs SCRFD on its NPU and burns the boxes into the H.264 stream
as OSD rectangles. Those pixels are unreadable as numbers, so the firmware
also emits the same inference as JSON on the WebRTC data channel; this is the
host side of that contract.
"""
import json

import numpy as np
import pytest

from sfcpi.detect import BoardDetector
from sfcpi.webrtc.metadata import BoardDetections, parse_detection_message


def _msg(**over):
    payload = {"t": 1724800000123, "w": 1280, "h": 720, "model": "scrfd",
               "d": [[10, 20, 30, 40, 87], [100, 110, 20, 25, 62]],
               "n": 2, "trunc": 0}
    payload.update(over)
    return json.dumps(payload)


def test_parses_a_board_message():
    got = parse_detection_message(_msg())
    assert got.count == 2
    assert got.frame_width == 1280 and got.frame_height == 720
    assert got.truncated is False
    assert [d.x1 for d in got.detections] == [10.0, 100.0]
    # firmware sends x,y,w,h -- the host works in x1,y1,x2,y2
    assert got.detections[0].x2 == 40.0 and got.detections[0].y2 == 60.0
    assert got.detections[0].score == pytest.approx(0.87)


def test_zero_detections_is_a_real_observation_not_a_gap():
    """The firmware emits every inference, n=0 included: an empty crowd must
    be distinguishable from the board having gone silent."""
    got = parse_detection_message(_msg(d=[], n=0))
    assert got.count == 0
    assert got.detections == []


def test_count_comes_from_n_not_from_the_array_length():
    """`d` is capped at 64 boxes; `n` is the true count. Counting the array
    would silently under-report exactly the dense crowds that matter."""
    got = parse_detection_message(_msg(d=[[1, 2, 3, 4, 50]], n=400, trunc=1))
    assert got.count == 400
    assert got.truncated is True
    assert len(got.detections) == 1


@pytest.mark.parametrize("raw", ["", "   ", "not json", "[]", '{"n":1}'])
def test_malformed_messages_raise_rather_than_report_a_crowd_of_zero(raw):
    with pytest.raises(ValueError):
        parse_detection_message(raw)


def test_negative_count_is_rejected():
    with pytest.raises(ValueError):
        parse_detection_message(_msg(n=-1))


# -- BoardDetector -----------------------------------------------------------

IMAGE = np.zeros((720, 1280, 3), dtype=np.uint8)


def test_board_detector_serves_the_latest_message():
    det = BoardDetector()
    det.update(parse_detection_message(_msg()))
    assert len(det.detect(IMAGE)) == 2


def test_board_detector_raises_before_any_message_arrives():
    """Same contract as NullDetector: an unknown count is NaN, never zero."""
    det = BoardDetector()
    with pytest.raises(RuntimeError, match="no detection"):
        det.detect(IMAGE)


def test_board_detector_goes_stale_rather_than_repeating_old_detections():
    """If the data channel stops, the count is unknown -- NOT the last known
    crowd frozen in place, which would read as a calm, steady scene."""
    det = BoardDetector(max_age_s=1.0)
    det.update(parse_detection_message(_msg()), now=100.0)
    assert len(det.detect(IMAGE, now=100.5)) == 2
    with pytest.raises(RuntimeError, match="stale"):
        det.detect(IMAGE, now=102.0)


def test_board_detector_reports_the_true_count_even_when_truncated():
    det = BoardDetector()
    det.update(parse_detection_message(_msg(d=[[1, 2, 3, 4, 50]], n=400, trunc=1)))
    assert det.last_count == 400
