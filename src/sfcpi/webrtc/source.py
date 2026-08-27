"""WebRTCSource: makes a live KVS WebRTC stream look exactly like the file
replay sources in sfcpi.sources, so the pipeline consumes both identically.

Frames arrive on aiortc's asyncio loop via feed(); the pipeline consumes them
synchronously via __iter__. FrameBridge is the bounded, drop-oldest handoff
between the two. fps is measured from the first `warmup_frames` timestamps
(median inter-frame delta) and RAISES until that warm-up completes -- frame
rate feeds crowd pressure as fps**2, so a silent default would be a quadratic
error, not a rounding error.

aiortc is imported lazily, inside the connect path only, so importing this
module (and running these tests) never requires it to be installed.
"""
from __future__ import annotations

import asyncio
import statistics
import threading
import time
import uuid
from typing import Any, Iterator, List, Optional

import numpy as np

from sfcpi.sources import Frame

from .bridge import FrameBridge
from .signaling import decode_message, encode_sdp_offer


class WebRTCSource:
    """FrameSource-compatible live ingest from an AWS KVS WebRTC channel.

    Satisfies the FrameSource protocol structurally (exposes `fps` and
    `__iter__`); it does not import or subclass anything from sfcpi.sources
    except Frame.
    """

    def __init__(
        self,
        signaling: Any,
        bridge: Optional[FrameBridge] = None,
        warmup_frames: int = 10,
        connect_timeout_s: float = 15.0,
        maxsize: int = 8,
    ) -> None:
        if warmup_frames < 1:
            raise ValueError("warmup_frames must be >= 1")
        self.signaling = signaling
        self.warmup_frames = warmup_frames
        self.connect_timeout_s = connect_timeout_s
        self._bridge = bridge if bridge is not None else FrameBridge(maxsize=maxsize)
        self._index = 0
        self._warmup_timestamps: List[float] = []
        self._fps: Optional[float] = None

        # -- connect-path state (populated only if connect() is used) --------
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional["asyncio.AbstractEventLoop"] = None
        self._pc: Optional[Any] = None
        self._ws: Optional[Any] = None
        self._signalling_task: Optional[Any] = None
        self._phase: Optional[str] = None
        self._connect_error: Optional[BaseException] = None
        # Set only once the loop has entered its long-lived run_forever();
        # close() must not schedule work on a loop nothing is driving.
        self._loop_running = threading.Event()

    # -- FrameSource protocol -------------------------------------------------

    @property
    def fps(self) -> float:
        if self._fps is None:
            raise RuntimeError(
                f"fps not yet measured: need {self.warmup_frames} frames to "
                f"warm up, have {len(self._warmup_timestamps)}"
            )
        return self._fps

    @property
    def dropped(self) -> int:
        return self._bridge.dropped

    def __iter__(self) -> Iterator[Frame]:
        return iter(self._bridge)

    # -- producer side ---------------------------------------------------------

    def feed(self, frame: Any, timestamp: float) -> None:
        """Called by the media task (or tests) for each incoming frame.

        Accepts a plain numpy BGR array, or an av.VideoFrame-like object
        exposing `.to_ndarray(format="bgr24")` -- detected by capability, not
        by importing `av`.
        """
        image = self._as_bgr_ndarray(frame)
        self._record_warmup_timestamp(timestamp)
        out = Frame(index=self._index, timestamp=timestamp, image=image)
        self._index += 1
        self._bridge.put(out)

    @staticmethod
    def _as_bgr_ndarray(frame: Any) -> np.ndarray:
        if isinstance(frame, np.ndarray):
            return frame
        to_ndarray = getattr(frame, "to_ndarray", None)
        if callable(to_ndarray):
            return to_ndarray(format="bgr24")
        raise TypeError(
            f"feed() needs a numpy array or an av.VideoFrame-like object "
            f"with to_ndarray(format=...); got {type(frame)!r}"
        )

    def _record_warmup_timestamp(self, timestamp: float) -> None:
        if self._fps is not None:
            return
        self._warmup_timestamps.append(float(timestamp))
        if len(self._warmup_timestamps) >= self.warmup_frames:
            deltas = [
                b - a
                for a, b in zip(self._warmup_timestamps, self._warmup_timestamps[1:])
            ]
            median_delta = statistics.median(deltas)
            if median_delta <= 0:
                raise RuntimeError(
                    "fps could not be measured: warm-up timestamps are not "
                    "monotonically increasing"
                )
            self._fps = 1.0 / median_delta

    def fail(self, exc: BaseException) -> None:
        self._bridge.fail(exc)

    def close(self) -> None:
        self._bridge.close()
        loop = self._loop
        # Only schedule shutdown on the loop once it's actually in
        # run_forever(): a failed connect() leaves a loop that nothing is
        # driving anymore (it cleans up after itself and returns -- see
        # _thread_main), and run_coroutine_threadsafe against a loop no one
        # is pumping just leaks an unawaited coroutine.
        if loop is not None and not loop.is_closed() and self._loop_running.is_set():

            async def _shutdown_and_stop_loop() -> None:
                try:
                    await self._shutdown_connect()
                finally:
                    loop.stop()

            try:
                asyncio.run_coroutine_threadsafe(_shutdown_and_stop_loop(), loop)
            except RuntimeError:
                pass  # loop already stopping/stopped
        thread = self._thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=5.0)

    # -- connect path (lazy aiortc/websockets) ----------------------------------
    #
    # connect() is synchronous: it starts a background thread that owns its own
    # asyncio event loop for the lifetime of the connection (aiortc's track.recv()
    # loop and close() both need that loop to still be running after connect()
    # returns, not just during negotiation). The calling thread blocks on a
    # threading.Event until the remote video track is flowing or the timeout
    # elapses.
    #
    # Every phase that can block on real I/O (opening the signalling socket,
    # negotiating SDP/ICE) is individually bounded with asyncio.wait_for against
    # the same overall deadline, and ANY failure in that phase -- a genuine
    # timeout or an outright connection error -- is surfaced as a TimeoutError
    # naming the phase. That keeps the two required outcomes ("bad channel /
    # unreachable signalling endpoint" vs "signalling fine, ICE/media never came
    # up") distinguishable without depending on how fast the underlying failure
    # happens to be, which varies a lot between "DNS says no" (fast) and
    # "packets vanish into a firewall" (slow) -- both are the same operator
    # problem for a given phase and should read the same way.

    def connect(
        self,
        timeout_s: Optional[float] = None,
        client_id: Optional[str] = None,
        ice_servers: Optional[list] = None,
    ) -> None:
        """Open the KVS WebRTC signalling connection and start feeding frames.

        Validates that the signalling channel has a WSS endpoint BEFORE
        starting any thread or event loop, so a misconfigured channel fails
        immediately with a RuntimeError instead of hanging until the timeout.

        Blocks the calling thread until the remote video track starts
        flowing, or raises TimeoutError naming whichever phase -- "signalling
        connect" or "ICE/media negotiation" -- did not complete in time.

        aiortc and websockets are imported here, not at module scope, so
        importing this module (and running the non-connect tests) never
        requires either to be installed.
        """
        timeout_s = self.connect_timeout_s if timeout_s is None else timeout_s

        endpoints = self.signaling.endpoints()
        wss_url = endpoints.get("WSS")
        if not wss_url:
            raise RuntimeError(
                f"KVS signalling channel {self.signaling.channel_name!r} has no "
                f"WSS endpoint (got protocols {sorted(endpoints)!r}); check the "
                f"channel is ACTIVE and the caller's IAM role has "
                f"kinesisvideo:GetSignalingChannelEndpoint"
            )

        import aiortc  # noqa: F401  (lazy; connect-path only)
        import websockets  # noqa: F401  (lazy; connect-path only)

        client_id = client_id or f"sfcpi-viewer-{uuid.uuid4().hex[:12]}"
        self._phase = "signalling connect"
        self._connect_error = None
        ready = threading.Event()

        def _thread_main() -> None:
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    self._negotiate(
                        aiortc, websockets, wss_url, client_id, ice_servers,
                        timeout_s, ready,
                    )
                )
            except BaseException as exc:  # noqa: BLE001 - must reach the consumer
                self._connect_error = exc
                self.fail(exc)
                ready.set()
                # Best-effort cleanup of whatever got created before the
                # failure (a signalling socket, a peer connection). This runs
                # on the same loop/thread since nothing else will service it
                # -- close() only schedules cleanup once _loop_running is set,
                # which never happens on this path.
                try:
                    loop.run_until_complete(self._shutdown_connect())
                except BaseException:  # noqa: BLE001 - cleanup is best-effort
                    pass
                loop.close()
                return
            # Negotiation succeeded: keep the loop alive to service the
            # track's recv loop and close()'s shutdown coroutine.
            self._loop_running.set()
            try:
                loop.run_forever()
            except BaseException as exc:  # noqa: BLE001
                self.fail(exc)
            finally:
                loop.close()

        self._thread = threading.Thread(
            target=_thread_main, name="webrtc-connect", daemon=True
        )
        self._thread.start()

        if not ready.wait(timeout_s + 2.0):
            # Backstop only, with slack added so the precise, phase-scoped
            # asyncio.wait_for deadlines inside _negotiate always get to fire
            # (and set `ready`) first. Without the slack this races: it would
            # be a coin flip whether this thread's own timer or the inner
            # asyncio timer reports the timeout first, and the inner one is
            # the one whose message names the actual phase.
            raise TimeoutError(
                f"WebRTC connect timed out after {timeout_s}s during "
                f"{self._phase} (channel {self.signaling.channel_name!r}); "
                f"background thread never responded"
            )
        if self._connect_error is not None:
            raise self._connect_error

    async def _negotiate(
        self,
        aiortc: Any,
        websockets: Any,
        wss_url: str,
        client_id: str,
        ice_servers: Optional[list],
        timeout_s: float,
        ready: threading.Event,
    ) -> None:
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout_s

        self._phase = "signalling connect"
        try:
            remaining = max(0.0, deadline - loop.time())
            ws = await asyncio.wait_for(
                websockets.connect(wss_url), timeout=remaining
            )
        except Exception as exc:
            raise TimeoutError(
                f"WebRTC connect timed out after {timeout_s}s during signalling "
                f"connect to {wss_url!r} (channel "
                f"{self.signaling.channel_name!r}): {exc}"
            ) from exc
        self._ws = ws

        self._phase = "ICE/media negotiation"
        try:
            remaining = max(0.0, deadline - loop.time())
            await asyncio.wait_for(
                self._negotiate_media(aiortc, ws, client_id, ice_servers),
                timeout=remaining,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"WebRTC connect timed out after {timeout_s}s during ICE/media "
                f"negotiation (channel {self.signaling.channel_name!r})"
            ) from exc

        ready.set()

    async def _negotiate_media(
        self, aiortc: Any, ws: Any, client_id: str, ice_servers: Optional[list]
    ) -> None:
        config = aiortc.RTCConfiguration(
            iceServers=[aiortc.RTCIceServer(**s) for s in (ice_servers or [])]
        )
        pc = aiortc.RTCPeerConnection(configuration=config)
        self._pc = pc
        track_ready = asyncio.Event()

        # This class only ever receives: without a transceiver, the offer has
        # no media section, no ICE transport is created, and iceGatheringState
        # stays "new" forever -- _await_ice_gathering_complete would hang with
        # no event ever coming to wake it up.
        pc.addTransceiver("video", direction="recvonly")

        @pc.on("track")
        def on_track(track: Any) -> None:
            if track.kind == "video":
                asyncio.ensure_future(self._recv_loop(track))
                track_ready.set()

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        await self._await_ice_gathering_complete(pc)
        await ws.send(encode_sdp_offer(client_id, pc.localDescription.sdp))

        while True:
            raw = await ws.recv()
            message_type, _sender, payload = decode_message(raw)
            if message_type == "SDP_ANSWER":
                await pc.setRemoteDescription(
                    aiortc.RTCSessionDescription(
                        sdp=payload["sdp"], type=payload["type"]
                    )
                )
                break

        self._signalling_task = asyncio.ensure_future(
            self._pump_ice_candidates(ws, pc)
        )
        await track_ready.wait()

    @staticmethod
    async def _await_ice_gathering_complete(pc: Any) -> None:
        if pc.iceGatheringState == "complete":
            return
        done: "asyncio.Future[None]" = asyncio.get_event_loop().create_future()

        @pc.on("icegatheringstatechange")
        def _on_change() -> None:
            if pc.iceGatheringState == "complete" and not done.done():
                done.set_result(None)

        await done

    async def _pump_ice_candidates(self, ws: Any, pc: Any) -> None:
        """Apply trickled ICE candidates the far side sends after the answer.

        Best-effort: this class only ever sends a non-trickle (gather-then-send)
        offer of its own, but still applies whatever the master trickles in, in
        case that peer needs it. Any failure here reaches the consumer via
        fail() exactly like every other producer-side error.
        """
        try:
            async for raw in ws:
                message_type, _sender, payload = decode_message(raw)
                if message_type != "ICE_CANDIDATE":
                    continue
                candidate_str = payload.get("candidate")
                if not candidate_str:
                    continue
                from aiortc.sdp import candidate_from_sdp

                candidate = candidate_from_sdp(candidate_str)
                candidate.sdpMid = payload.get("sdpMid")
                candidate.sdpMLineIndex = payload.get("sdpMLineIndex")
                await pc.addIceCandidate(candidate)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - never swallow a producer error
            self.fail(exc)

    async def _recv_loop(self, track: Any) -> None:
        """Pull decoded frames off an aiortc MediaStreamTrack and feed() them.

        Any failure here -- the track ending, a decode error, a dead
        connection -- must reach the consumer via fail(), never end the
        stream silently: a silently-empty stream reads as "calm crowd"
        downstream, which is exactly the failure mode this project exists to
        avoid.
        """
        try:
            while True:
                frame = await track.recv()
                timestamp = getattr(frame, "time", None)
                if timestamp is None:
                    timestamp = time.monotonic()
                self.feed(frame, timestamp)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - never swallow a producer error
            self.fail(exc)

    async def _shutdown_connect(self) -> None:
        """Close whatever connect()/_negotiate managed to create.

        Used on two different paths: from close(), on the long-lived loop
        (in which case the caller also stops that loop once this returns),
        and from connect()'s own failure branch, run synchronously via
        run_until_complete on a loop nobody else is driving -- so this
        coroutine itself must never stop the loop; that's the caller's job.
        """
        if self._signalling_task is not None:
            self._signalling_task.cancel()
        if self._pc is not None:
            await self._pc.close()
        if self._ws is not None:
            await self._ws.close()
