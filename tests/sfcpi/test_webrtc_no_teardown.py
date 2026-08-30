"""A media stall must not tear the session down.

The AWS JS SDK's viewer registers `connectionstatechange` and only PRINTS -- it
never closes the peer connection, never reconnects, never restarts ICE. Our
_recv_loop did the opposite: any exception from track.recv(), including the
MediaStreamError raised when a track simply ends, called fail() and poisoned
the FrameBridge, ending the whole session.

That matters beyond video: the data channel carries the board's detection
counts on its own SCTP transport. Killing the session because the video track
hiccuped also throws away the counts, which are the thing the risk engine
actually runs on.
"""
import asyncio

import numpy as np
import pytest

from sfcpi.webrtc.source import WebRTCSource


class _EndingTrack:
    """Yields one frame, then ends the way aiortc ends a track."""

    def __init__(self, exc):
        self._sent = False
        self._exc = exc

    async def recv(self):
        if not self._sent:
            self._sent = True
            return np.zeros((4, 4, 3), dtype=np.uint8)
        raise self._exc


def _media_stream_error():
    from aiortc.mediastreams import MediaStreamError

    return MediaStreamError()


def test_a_track_ending_does_not_fail_the_session():
    src = WebRTCSource(signaling=object(), warmup_frames=2)
    asyncio.run(src._recv_loop(_EndingTrack(_media_stream_error())))
    assert src._bridge._error is None, (
        "a track ending must not poison the bridge; the connection and the "
        "data channel are still alive"
    )


def test_the_frame_before_the_end_still_arrives():
    src = WebRTCSource(signaling=object(), warmup_frames=2)
    asyncio.run(src._recv_loop(_EndingTrack(_media_stream_error())))
    assert src.frames_received == 1


def test_media_end_is_recorded_so_a_caller_can_report_it():
    """Not failing is not the same as pretending nothing happened."""
    src = WebRTCSource(signaling=object(), warmup_frames=2)
    assert src.media_ended is False
    asyncio.run(src._recv_loop(_EndingTrack(_media_stream_error())))
    assert src.media_ended is True


def test_a_genuine_error_still_reaches_the_consumer():
    """Only a track ENDING is benign. A real fault must still surface, or a
    broken producer becomes a silently empty -- and falsely calm -- stream."""
    src = WebRTCSource(signaling=object(), warmup_frames=2)
    asyncio.run(src._recv_loop(_EndingTrack(RuntimeError("decoder exploded"))))
    assert isinstance(src._bridge._error, RuntimeError)


# -- recovery: a stalled stream must END cleanly so a new session can start --

def test_media_end_closes_the_stream_so_the_consumer_is_not_stuck():
    """Not failing must not mean hanging forever.

    The bridge has no idle timeout any more, so if a track ends and we neither
    fail nor close, the consumer blocks indefinitely, the session never
    finishes, and nothing can re-establish it -- the stream simply stops and is
    never picked back up. Ending the iteration cleanly lets the caller start a
    fresh session (after the board's slot-reclaim wait) without treating a
    normal end as an error.
    """
    src = WebRTCSource(signaling=object(), warmup_frames=2)
    asyncio.run(src._recv_loop(_EndingTrack(_media_stream_error())))

    assert src._bridge.closed, "the bridge must be closed so iteration ends"
    assert src._bridge._error is None, "a clean end is not an error"

    # the consumer drains the buffered frame, then stops -- it does not hang
    frames = list(src)
    assert len(frames) == 1


def test_a_genuine_error_does_not_close_silently():
    """A real fault still raises out of iteration rather than looking like a
    tidy end of stream."""
    src = WebRTCSource(signaling=object(), warmup_frames=2)
    asyncio.run(src._recv_loop(_EndingTrack(RuntimeError("decoder exploded"))))
    with pytest.raises(RuntimeError, match="decoder exploded"):
        list(src)
