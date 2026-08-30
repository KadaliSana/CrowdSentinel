"""The viewer must TRICKLE its own ICE candidates to the master.

Observed against the live `camstream` channel (2026-08-28): with candidates
carried only inside the SDP offer, our side reached ICE `completed` but the
master never formed a valid pair, never sent the DTLS ClientHello (it answers
`a=setup:active`, so IT is the DTLS client), and `connectionState` sat at
`connecting` until the master tore the session down ~34s later. Sending the
same candidates as ICE_CANDIDATE signalling messages made DTLS complete
immediately.

AWS's own guidance for the JS SDK sample says to leave Trickle ICE enabled;
the KVS C SDK master this board runs does not adopt candidates from the
offer SDP alone.
"""
import asyncio
import base64
import json

import pytest

from sfcpi.webrtc.signaling import encode_ice_candidate, sdp_candidate_lines
from sfcpi.webrtc.source import WebRTCSource

from test_webrtc_keepalive import (_FakeAiortc, _FakePc, _FakeWs,
                                   _answer_message)


OFFER_SDP = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 101
a=mid:0
a=fingerprint:sha-256 AA:BB
a=candidate:1 1 udp 2130706431 172.16.144.16 43937 typ host
a=candidate:2 1 udp 1694498815 182.66.218.119 60475 typ srflx raddr 172.17.0.1 rport 60475
m=audio 9 UDP/TLS/RTP/SAVPF 111
a=mid:1
a=candidate:3 1 udp 2130706431 172.16.144.16 43938 typ host
"""


def test_sdp_candidate_lines_attributes_each_candidate_to_its_m_section():
    """A candidate belongs to the m= section it follows, not to section 0.

    Sending every candidate under sdpMLineIndex 0 would misattribute the
    audio candidate; the parser must track the section.
    """
    lines = sdp_candidate_lines(OFFER_SDP)
    assert [c.sdp_mline_index for c in lines] == [0, 0, 1]
    assert [c.sdp_mid for c in lines] == ["0", "0", "1"]
    assert lines[0].candidate.startswith("candidate:1 ")
    # the "a=" prefix is an SDP artefact and must not be sent on the wire
    assert not lines[0].candidate.startswith("a=")


def test_sdp_with_no_candidates_yields_nothing():
    assert sdp_candidate_lines("v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 101\r\n") == []


def test_encode_ice_candidate_payload_round_trips():
    raw = encode_ice_candidate("candidate:1 1 udp 2130706431 10.0.0.1 5000 typ host", "0", 0)
    envelope = json.loads(raw)
    assert envelope["action"] == "ICE_CANDIDATE"
    payload = json.loads(base64.b64decode(envelope["messagePayload"]))
    assert payload == {
        "candidate": "candidate:1 1 udp 2130706431 10.0.0.1 5000 typ host",
        "sdpMid": "0",
        "sdpMLineIndex": 0,
    }


def test_negotiation_trickles_local_candidates_after_the_offer():
    """The regression that cost the live smoke test: candidates never sent."""
    src = WebRTCSource(signaling=object(), warmup_frames=2)
    ws = _FakeWs([_answer_message()])

    pc = _FakePc()
    pc.localDescription = type("D", (), {"sdp": OFFER_SDP})()
    fake = type("A", (_FakeAiortc,), {"RTCPeerConnection": lambda *a, **k: pc})

    asyncio.run(asyncio.wait_for(
        src._negotiate_media(fake, ws, "cid", None), timeout=5))

    kinds = [json.loads(m).get("action") or json.loads(m).get("messageType")
             for m in ws.sent]
    assert kinds[0] == "SDP_OFFER", "the offer must still go first"
    assert kinds.count("ICE_CANDIDATE") == 3, (
        f"all 3 local candidates must be trickled, got {kinds}")
