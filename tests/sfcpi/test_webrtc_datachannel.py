"""The viewer must both OFFER a data channel and accept one from the board.

The board only allocates SCTP when the remote offer enables it
(`ucEnableDataChannelRemote`, peer_connection.c), so the offer must carry an
m=application section or the detections have nowhere to go. The board then
pushes on every open channel on the session -- including the one we opened --
so both paths have to be handled.
"""
import asyncio
import json

import pytest

from sfcpi.detect import BoardDetector
from sfcpi.webrtc.source import WebRTCSource

from test_webrtc_keepalive import (_FakeAiortc, _FakePc, _FakeWs,
                                   _answer_message)


def _detection_json(n=3):
    return json.dumps({"t": 1, "w": 1280, "h": 720, "model": "scrfd",
                       "d": [[10, 10, 20, 20, 90]] * min(n, 1), "n": n,
                       "trunc": 0})


class _FakeChannel:
    def __init__(self, label="sfcpi"):
        self.label = label
        self._handlers = {}

    def on(self, event):
        def deco(fn):
            self._handlers[event] = fn
            return fn
        return deco

    def fire(self, message):
        self._handlers["message"](message)


class _DcPc(_FakePc):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.created_channels = []

    def createDataChannel(self, label, **kw):
        ch = _FakeChannel(label)
        self.created_channels.append(ch)
        return ch


def _run(src, ws, pc):
    fake = type("A", (_FakeAiortc,), {"RTCPeerConnection": lambda *a, **k: pc})
    asyncio.run(asyncio.wait_for(
        src._negotiate_media(fake, ws, "cid", None), timeout=5))


def test_offer_includes_a_data_channel():
    """Without this the board never allocates SCTP and never sends boxes."""
    pc = _DcPc()
    _run(WebRTCSource(signaling=object(), warmup_frames=2),
         _FakeWs([_answer_message()]), pc)
    assert [c.label for c in pc.created_channels] == ["sfcpi"]


def test_detections_on_our_own_channel_reach_the_detector():
    detector = BoardDetector()
    src = WebRTCSource(signaling=object(), warmup_frames=2, detector=detector)
    pc = _DcPc()
    _run(src, _FakeWs([_answer_message()]), pc)

    pc.created_channels[0].fire(_detection_json(n=3))
    assert detector.last_count == 3


def test_detections_on_a_board_opened_channel_reach_the_detector():
    detector = BoardDetector()
    src = WebRTCSource(signaling=object(), warmup_frames=2, detector=detector)
    pc = _DcPc()
    _run(src, _FakeWs([_answer_message()]), pc)

    incoming = _FakeChannel("kvsDataChannel")
    pc._handlers["datachannel"](incoming)
    incoming.fire(_detection_json(n=7))
    assert detector.last_count == 7


def test_a_malformed_message_does_not_kill_the_stream_or_freeze_the_count():
    """One bad message must not fail() the video, and must not leave a stale
    count standing as if it were fresh."""
    detector = BoardDetector()
    src = WebRTCSource(signaling=object(), warmup_frames=2, detector=detector)
    pc = _DcPc()
    _run(src, _FakeWs([_answer_message()]), pc)

    pc.created_channels[0].fire("not json")
    assert src._bridge._error is None
    assert detector.last_count is None
