"""The viewer must not send RTCP NACK to this board.

The board cannot parse them. From its serial log:

    [WARN] HandleNonStunPackets: Failed to handle SRTCP packets, result: 36
    [INFO] SessionProcessEndlessLoop: Detect inactive connection, closing ...

result 36 is PEER_CONNECTION_RESULT_FAIL_RTCP_PARSE_NACK. That matters far
beyond a dropped retransmit request: the board refreshes its 30s inactivity
timer ONLY when HandleSrtcpPacket returns OK (peer_connection.c:1108), so
every rejected packet is a missed keepalive. The result was a session that
died every ~38s (30s timeout + connect time) and looked exactly like "the
board stopped sending video".

aiortc builds a NackGenerator for EVERY video receiver regardless of the
negotiated rtcp-fb, so removing `a=rtcp-fb:* nack` from the offer does not
help; the generator has to be turned off on the receiver.
"""
import pytest

from sfcpi.webrtc.source import (NACK_GENERATOR_ATTR, REMB_ESTIMATOR_ATTR,
                                 disable_nack)


class _Receiver:
    def __init__(self):
        setattr(self, NACK_GENERATOR_ATTR, object())
        setattr(self, REMB_ESTIMATOR_ATTR, object())


class _Transceiver:
    def __init__(self, receiver): self.receiver = receiver


class _Pc:
    def __init__(self, receivers): self._t = [_Transceiver(r) for r in receivers]
    def getTransceivers(self): return self._t


class _DummyTransport:
    """Minimal stand-in for RTCDtlsTransport: enough to construct a receiver."""
    state = "new"

    def _register_rtp_receiver(self, *a, **k):
        pass


def test_canary_aiortc_still_has_the_nack_generator_attribute():
    """If aiortc renames this, the fix silently stops working and sessions
    start dying again -- fail here instead, in CI."""
    from aiortc.rtcrtpreceiver import RTCRtpReceiver

    receiver = RTCRtpReceiver("video", _DummyTransport())
    assert hasattr(receiver, NACK_GENERATOR_ATTR), (
        f"aiortc no longer exposes {NACK_GENERATOR_ATTR}; "
        f"re-check how NACK generation is disabled"
    )
    assert hasattr(receiver, REMB_ESTIMATOR_ATTR), (
        f"aiortc no longer exposes {REMB_ESTIMATOR_ATTR}")
    assert getattr(receiver, NACK_GENERATOR_ATTR) is not None, (
        "aiortc no longer enables NACK for video by default; the workaround "
        "may be unnecessary"
    )


def test_disable_nack_clears_every_video_receiver():
    receivers = [_Receiver(), _Receiver()]
    assert disable_nack(_Pc(receivers)) == 4  # nack + remb, two receivers
    assert all(getattr(r, NACK_GENERATOR_ATTR) is None for r in receivers)


def test_disable_nack_tolerates_a_receiver_without_the_attribute():
    """A future aiortc, or an audio receiver, must not raise."""
    class _Bare: pass
    assert disable_nack(_Pc([_Bare(), _Receiver()])) == 2


def test_disable_nack_tolerates_a_missing_receiver():
    class _NoReceiver:
        receiver = None
    pc = _Pc([])
    pc._t = [_NoReceiver()]
    assert disable_nack(pc) == 0
