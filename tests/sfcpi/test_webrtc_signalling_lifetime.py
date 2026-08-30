"""A closed SIGNALLING socket must not kill a healthy MEDIA stream.

Sessions were dying after 38-45s. The cause was ours, not the board's:
`websockets` defaults to ping_interval=20 and ping_timeout=20, so the client
unilaterally closes the KVS signalling connection ~40s in if a pong is missed.
`_pump_ice_candidates` holds `async for raw in ws` for the whole session, so
that closure raised into its handler, which called fail() and poisoned the
FrameBridge -- tearing down video that was flowing perfectly well.

WebRTC media runs over its own ICE/DTLS socket. Signalling is needed to set a
session up and to trickle candidates; once media flows, losing it is not a
media failure.
"""
import asyncio

import pytest

from sfcpi.webrtc.source import (SIGNALLING_PING_INTERVAL_S,
                                 SIGNALLING_PING_TIMEOUT_S, WebRTCSource)


class _ClosedWs:
    """A signalling socket that closes the way `websockets` closes one."""

    def __init__(self, exc):
        self._exc = exc

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise self._exc


def _closed_error():
    from websockets.exceptions import ConnectionClosedError

    return ConnectionClosedError(None, None)


def _closed_ok():
    from websockets.exceptions import ConnectionClosedOK

    return ConnectionClosedOK(None, None)


class _Pc:
    async def addIceCandidate(self, c): pass


@pytest.mark.parametrize("exc_factory", [_closed_error, _closed_ok])
def test_signalling_close_does_not_poison_the_media_stream(exc_factory):
    src = WebRTCSource(signaling=object(), warmup_frames=2)
    asyncio.run(src._pump_ice_candidates(_ClosedWs(exc_factory()), _Pc()))
    assert src._bridge._error is None, (
        "a closed signalling socket must not fail() the frame bridge; media "
        "has its own transport"
    )


def test_a_genuine_producer_error_still_reaches_the_consumer():
    """Only CONNECTION CLOSURE is benign. A real fault must not be swallowed,
    or a broken producer becomes a silently empty -- and falsely calm -- stream."""
    src = WebRTCSource(signaling=object(), warmup_frames=2)
    asyncio.run(src._pump_ice_candidates(_ClosedWs(RuntimeError("boom")), _Pc()))
    assert isinstance(src._bridge._error, RuntimeError)


def test_ping_settings_do_not_close_the_socket_under_us():
    """Either pings are off, or the timeout is generous enough that a slow
    pong is not treated as a dead peer. 20s+20s is what produced the ~40s
    sessions."""
    assert (SIGNALLING_PING_INTERVAL_S is None
            or SIGNALLING_PING_TIMEOUT_S is None
            or SIGNALLING_PING_TIMEOUT_S >= 60)
