import numpy as np
import pytest
from sfcpi.sources import Frame
from sfcpi.webrtc.source import WebRTCSource

class _FakeSignaling:
    channel_name = "camstream"

def _src(**kw):
    kw.setdefault("signaling", _FakeSignaling())
    kw.setdefault("warmup_frames", 3)
    return WebRTCSource(**kw)

def _img(v=0):
    return np.full((8, 8, 3), v, np.uint8)

def test_fps_before_warmup_raises_not_a_default():
    """A silent fps default is a quadratic error downstream."""
    s = _src()
    with pytest.raises(RuntimeError, match="fps"):
        _ = s.fps

def test_fps_is_measured_from_timestamps():
    s = _src()
    for i in range(3):
        s.feed(_img(i), timestamp=i * 0.1)     # 10 fps
    s.close()
    assert s.fps == pytest.approx(10.0, rel=0.05)

def test_yields_frames_matching_the_framesource_contract():
    s = _src()
    for i in range(3):
        s.feed(_img(i), timestamp=i * 0.1)
    s.close()
    frames = list(s)
    assert all(isinstance(f, Frame) for f in frames)
    assert [f.index for f in frames] == [0, 1, 2]
    assert frames[1].timestamp == pytest.approx(0.1)
    assert frames[0].image.shape == (8, 8, 3)

def test_producer_error_surfaces_to_the_consumer():
    s = _src()
    s.feed(_img(), timestamp=0.0)
    s.fail(RuntimeError("ice failed"))
    with pytest.raises(RuntimeError, match="ice failed"):
        list(s)

def test_dropped_frames_are_counted_and_visible():
    s = _src(maxsize=2)
    for i in range(6):
        s.feed(_img(i), timestamp=i * 0.1)
    s.close()
    list(s)
    assert s.dropped > 0

def test_warmup_frames_1_survives_the_first_feed():
    """warmup_frames=1 is a legal constructor value; a single timestamp gives
    zero inter-frame deltas, so the first feed() must not raise
    StatisticsError -- fps just isn't available yet."""
    s = _src(warmup_frames=1)
    s.feed(_img(), timestamp=0.0)
    with pytest.raises(RuntimeError, match="fps"):
        _ = s.fps

def test_warmup_frames_1_fps_available_and_correct_after_second_frame():
    s = _src(warmup_frames=1)
    s.feed(_img(0), timestamp=0.0)
    s.feed(_img(1), timestamp=0.1)     # 10 fps
    assert s.fps == pytest.approx(10.0, rel=0.05)
