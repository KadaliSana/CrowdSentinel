"""Real KVS sends empty WebSocket frames as keepalives.

Observed against the live `camstream` channel (2026-08-28): the FIRST message
the signalling service sends after an SDP_OFFER is a zero-length text frame,
and more arrive during a long-lived session. `decode_message` is right to
reject it -- an empty payload IS malformed as an envelope -- but the recv
loops must recognise it as a keepalive and skip it, or negotiation dies on
message #1 and a live session dies at the first idle keepalive.

Regression test for the live smoke test's first real failure.
"""
import asyncio

import pytest

from sfcpi.webrtc.signaling import decode_message, encode_sdp_offer, is_keepalive
from sfcpi.webrtc.source import WebRTCSource


# -- the keepalive predicate itself ------------------------------------------

@pytest.mark.parametrize("raw", ["", "   ", "\n", b"", b"  "])
def test_empty_and_blank_frames_are_keepalives(raw):
    assert is_keepalive(raw) is True


@pytest.mark.parametrize("raw", ['{"messageType":"SDP_ANSWER"}', b'{"a":1}', "not json"])
def test_real_payloads_are_not_keepalives(raw):
    """A non-empty frame is never a keepalive -- even a malformed one.

    Classifying malformed-but-present traffic as a keepalive would silently
    swallow real corruption, which is the failure mode decode_message's
    ValueError exists to prevent.
    """
    assert is_keepalive(raw) is False


def test_decode_message_still_rejects_a_non_empty_malformed_frame():
    with pytest.raises(ValueError, match="payload"):
        decode_message("not json")


# -- the negotiation loop ----------------------------------------------------

class _FakeTrack:
    kind = "video"
    async def recv(self):          # never called in these tests
        await asyncio.sleep(3600)


class _FakePc:
    def __init__(self, *a, **k):
        self.iceGatheringState = "complete"
        # a real offer always carries a fingerprint; the offer path now requires
        # a sha-256 one (see test_webrtc_fingerprint.py)
        self.localDescription = type(
            "D", (), {"sdp": "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 101\r\n"
                             "a=fingerprint:sha-256 AA:BB\r\n"})()
        self.remote_set = False
        self.added_candidates = []
        self._handlers = {}

    def addTransceiver(self, *a, **k): pass

    def getTransceivers(self):
        # Real RTCPeerConnection always has this; the viewer walks it after the
        # answer to turn off NACK generation (see test_webrtc_nack.py).
        return []

    def on(self, event):
        def deco(fn):
            self._handlers[event] = fn
            return fn
        return deco

    async def createOffer(self): return object()
    async def setLocalDescription(self, offer): pass

    async def setRemoteDescription(self, desc):
        self.remote_set = True
        # aiortc fires "track" once the remote description creates the receiver
        self._handlers["track"](_FakeTrack())

    async def addIceCandidate(self, c): self.added_candidates.append(c)
    async def close(self): pass


class _FakeAiortc:
    RTCPeerConnection = _FakePc
    @staticmethod
    def RTCConfiguration(**k): return None
    @staticmethod
    def RTCIceServer(**k): return None
    @staticmethod
    def RTCSessionDescription(sdp, type): return (sdp, type)


class _FakeWs:
    """Yields a scripted message sequence, then blocks like a live socket."""
    def __init__(self, messages):
        self._messages = list(messages)
        self.sent = []

    async def send(self, raw): self.sent.append(raw)

    async def recv(self):
        if self._messages:
            return self._messages.pop(0)
        await asyncio.sleep(3600)

    def __aiter__(self): return self

    async def __anext__(self):
        if self._messages:
            return self._messages.pop(0)
        raise StopAsyncIteration


def _answer_message():
    import base64, json
    payload = base64.b64encode(
        json.dumps({"type": "answer", "sdp": "v=0\r\n"}).encode()
    ).decode()
    return json.dumps({"messageType": "SDP_ANSWER", "messagePayload": payload})


def _candidate_message(cand="candidate:1 1 udp 2130706431 10.0.0.1 5000 typ host"):
    import base64, json
    payload = base64.b64encode(
        json.dumps({"candidate": cand, "sdpMid": "0", "sdpMLineIndex": 0}).encode()
    ).decode()
    return json.dumps({"messageType": "ICE_CANDIDATE", "messagePayload": payload})


def test_negotiation_survives_a_leading_keepalive_frame():
    """The exact live failure: KVS's first frame is empty, then the answer."""
    src = WebRTCSource(signaling=object(), warmup_frames=2)
    ws = _FakeWs(["", _answer_message()])

    async def run():
        await asyncio.wait_for(
            src._negotiate_media(_FakeAiortc, ws, "cid", None), timeout=5
        )

    asyncio.run(run())          # ValueError before the fix
    assert src._pc.remote_set, "SDP_ANSWER must still be applied after the keepalive"


def test_pump_ice_candidates_survives_a_keepalive_frame():
    """Keepalives keep arriving mid-session; one must not poison the stream."""
    src = WebRTCSource(signaling=object(), warmup_frames=2)
    pc = _FakePc()
    ws = _FakeWs(["", _candidate_message()])

    asyncio.run(asyncio.wait_for(src._pump_ice_candidates(ws, pc), timeout=5))

    assert src._bridge._error is None, "a keepalive must not fail() the stream"
    assert len(pc.added_candidates) == 1
