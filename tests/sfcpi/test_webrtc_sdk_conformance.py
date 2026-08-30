"""Conformance with the AWS KVS WebRTC JS SDK (awslabs, cloned and read directly).

Three differences were found by diffing our implementation against the SDK
source rather than against documentation:

1. `recipientClientId` MUST NOT be sent by a VIEWER.
   SignalingClient.ts: "Required for 'MASTER' role. Should not be present for
   'VIEWER' role", enforced by validateRecipientClientId(), which THROWS when a
   VIEWER supplies one. examples/viewer.js calls
   `sendSdpOffer(peerConnection.localDescription)` with no recipient at all.
   We were sending our OWN client id as the recipient.

2. ICE candidates arriving BEFORE the remote SDP must be QUEUED, not dropped.
   SignalingClient.ts routes them through emitOrQueueIceCandidate() and replays
   them via emitPendingIceCandidates() once SDP_ANSWER arrives. We discarded
   them -- and a live trace showed the board's HOST candidate (the fastest
   path) arriving before the answer every time.

3. STATUS_RESPONSE is a real message type carrying service status/errors.
   We ignored it silently.
"""
import base64
import json

import pytest

from sfcpi.webrtc.signaling import decode_message, encode_ice_candidate, encode_sdp_offer


# -- 1. viewers do not address messages -------------------------------------

def test_sdp_offer_has_no_recipient_client_id():
    envelope = json.loads(encode_sdp_offer("sfcpi-viewer-abc", "v=0\r\n"))
    assert "recipientClientId" not in envelope, (
        "a VIEWER must not send recipientClientId; the SDK's "
        "validateRecipientClientId() throws when it is present"
    )
    assert envelope["action"] == "SDP_OFFER"


def test_ice_candidate_has_no_recipient_client_id():
    envelope = json.loads(encode_ice_candidate("candidate:1 1 udp 1 10.0.0.1 5 typ host", "0", 0))
    assert "recipientClientId" not in envelope


def test_sdp_offer_payload_is_still_the_session_description():
    envelope = json.loads(encode_sdp_offer("cid", "v=0\r\nm=video 9\r\n"))
    payload = json.loads(base64.b64decode(envelope["messagePayload"]))
    assert payload["type"] == "offer" and payload["sdp"].startswith("v=0")


# -- 3. status responses are surfaced ---------------------------------------

def test_status_response_is_decodable_without_a_message_payload():
    """KVS sends STATUS_RESPONSE with a `statusResponse` object and NO
    messagePayload. decode_message must not treat that as corruption."""
    raw = json.dumps({
        "messageType": "STATUS_RESPONSE",
        "statusResponse": {"code": "500", "description": "Internal error"},
    })
    message_type, _sender, payload = decode_message(raw)
    assert message_type == "STATUS_RESPONSE"
    assert payload.get("code") == "500"


def test_a_genuinely_malformed_message_still_raises():
    with pytest.raises(ValueError):
        decode_message("not json")


# -- 2. pre-answer ICE candidates must be queued, not dropped ---------------

import asyncio

from test_webrtc_keepalive import (_FakeAiortc, _FakePc, _FakeWs, _answer_message,
                                   _candidate_message)
from sfcpi.webrtc.source import WebRTCSource


class _CountingPc(_FakePc):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.added = []

    async def addIceCandidate(self, c):
        self.added.append(c)

    def createDataChannel(self, label, **kw):
        class _Ch:
            def on(self, event):
                def deco(fn):
                    return fn
                return deco
        return _Ch()


def test_candidates_arriving_before_the_answer_are_applied_not_dropped():
    """The board sends its HOST candidate -- the fastest path -- before the
    answer. Dropping it costs the best route on every session.

    The SDK queues these (emitOrQueueIceCandidate) and replays them once the
    remote SDP lands (emitPendingIceCandidates).
    """
    src = WebRTCSource(signaling=object(), warmup_frames=2)
    pc = _CountingPc()
    ws = _FakeWs([_candidate_message(), _candidate_message(), _answer_message()])
    fake = type("A", (_FakeAiortc,), {"RTCPeerConnection": lambda *a, **k: pc})

    asyncio.run(asyncio.wait_for(
        src._negotiate_media(fake, ws, "cid", None), timeout=5))

    assert pc.remote_set, "the answer must still be applied"
    assert len(pc.added) == 2, (
        f"both pre-answer candidates must survive, got {len(pc.added)}")


# -- 4. ignore an answer that arrives in the wrong signalling state ----------

class _StatePc(_CountingPc):
    """Tracks how many times a remote description was applied."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.signalingState = "have-local-offer"
        self.applied = 0

    async def setRemoteDescription(self, desc):
        self.applied += 1
        self.signalingState = "stable"
        await super().setRemoteDescription(desc)


def test_a_second_answer_is_ignored():
    """The SDK guards this explicitly:
        if (peerConnection.signalingState !== 'have-local-offer') {
            console.warn('Ignoring SDP answer in signaling state', ...); return; }
    Applying a second answer in state 'stable' raises inside aiortc and would
    kill an otherwise healthy session."""
    src = WebRTCSource(signaling=object(), warmup_frames=2)
    pc = _StatePc()
    ws = _FakeWs([_answer_message(), _answer_message()])
    fake = type("A", (_FakeAiortc,), {"RTCPeerConnection": lambda *a, **k: pc})

    asyncio.run(asyncio.wait_for(
        src._negotiate_media(fake, ws, "cid", None), timeout=5))

    assert pc.applied == 1, f"only the first answer may be applied, got {pc.applied}"


# -- 5. the data channel is optional, as it is in the SDK -------------------

def test_data_channel_can_be_disabled():
    """`if (formValues.openDataChannel)` -- the SDK makes it a choice. Ours
    always added an m=application section, and a master that answers without
    one makes aiortc raise 'Media sections in answer do not match offer',
    failing the whole connection."""
    src = WebRTCSource(signaling=object(), warmup_frames=2, data_channel=False)
    pc = _CountingPc()
    created = []
    pc.createDataChannel = lambda label, **kw: created.append(label)
    ws = _FakeWs([_answer_message()])
    fake = type("A", (_FakeAiortc,), {"RTCPeerConnection": lambda *a, **k: pc})

    asyncio.run(asyncio.wait_for(
        src._negotiate_media(fake, ws, "cid", None), timeout=5))

    assert created == [], "no data channel should be offered when disabled"


def test_data_channel_is_on_by_default():
    """The board sends detections on it, so it stays the default."""
    src = WebRTCSource(signaling=object(), warmup_frames=2)
    assert src.data_channel is True
