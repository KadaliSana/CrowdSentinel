"""WebRTCSource: makes a live KVS WebRTC stream look exactly like the file
replay sources in sfcpi.sources, so the pipeline consumes both identically.

Frames arrive on aiortc's asyncio loop via feed(); the pipeline consumes them
synchronously via __iter__. FrameBridge is the bounded, drop-oldest handoff
between the two. fps is measured from the first `max(warmup_frames, 2)`
timestamps (median inter-frame delta -- at least two timestamps are needed to
produce even one delta, so warmup_frames=1 still waits for a second frame)
and RAISES until that warm-up completes -- frame rate feeds crowd pressure as
fps**2, so a silent default would be a quadratic error, not a rounding error.

aiortc is imported lazily, inside the connect path only, so importing this
module (and running these tests) never requires it to be installed.
"""
from __future__ import annotations

import asyncio
import statistics
import threading
import time
from typing import Any, Iterator, List, Optional

import numpy as np

from sfcpi.sources import Frame

from .bridge import FrameBridge
from .metadata import parse_detection_message
from .signaling import (decode_message, encode_ice_candidate, encode_sdp_offer,
                        is_keepalive, keep_single_fingerprint,
                        sdp_candidate_lines, sign_wss_url)


# Label of the data channel this viewer opens. The board pushes detections
# on every open channel, so the label is ours to choose.
DETECTION_CHANNEL_LABEL = "sfcpi"

# `websockets` defaults to ping_interval=20 and ping_timeout=20, so the CLIENT
# closes the KVS signalling socket ~40s in if one pong is missed -- which is
# exactly how long sessions were lasting. KVS drives its own keepalive (the
# board logs `wss ping ==>` / `<== wss pong`), so we do not police it here.
SIGNALLING_PING_INTERVAL_S = 20
SIGNALLING_PING_TIMEOUT_S = None  # never close on a missed pong


def local_host_addresses(sdp: str) -> list:
    """The IPv4 addresses of our own `typ host` candidates, from our offer.

    Our own gathered candidates are the cheapest, most accurate statement of
    which subnets this machine actually sits on -- no interface enumeration
    and no extra dependency.
    """
    out = []
    for line in sdp.splitlines():
        line = line.strip()
        if not line.startswith("a=candidate:") or " typ host" not in line:
            continue
        parts = line.split()
        if len(parts) > 4:
            out.append(parts[4])
    return out


def is_reachable_candidate(candidate: str, local_addresses) -> bool:
    """False only for a private HOST candidate on a subnet we are not on.

    AWS TURN answers `403 Forbidden IP` when asked to relay to an address it
    considers unroutable, which leaves an unretrieved-task traceback in the
    log and wastes connectivity checks on a pair that cannot work.

    Deliberately narrow, and biased towards KEEPING candidates:
      - srflx / relay / prflx / anything unparsed -> kept
      - public host addresses -> kept
      - private host addresses -> kept ONLY if we hold an address in the same
        /24, i.e. a genuinely same-LAN peer, whose host candidate is the
        FASTEST path available and must never be discarded.
    ICE tolerates a useless candidate far better than a missing one.
    """
    import ipaddress

    if not local_addresses:
        return True
    parts = candidate.split()
    if len(parts) < 8 or "typ" not in parts:
        return True
    try:
        typ = parts[parts.index("typ") + 1]
        address = ipaddress.ip_address(parts[4])
    except (ValueError, IndexError):
        return True
    if typ != "host" or not address.is_private:
        return True

    for local in local_addresses:
        try:
            mine = ipaddress.ip_address(local)
        except ValueError:
            continue
        if mine.version != address.version:
            continue
        net = ipaddress.ip_network(f"{local}/24", strict=False)
        if address in net:
            return True
    return False


def _is_media_ended(exc: BaseException) -> bool:
    """True when a media track simply ended, as opposed to genuinely failing."""
    try:
        from aiortc.mediastreams import MediaStreamError
    except ImportError:  # pragma: no cover - connect-path dependency
        return False
    return isinstance(exc, MediaStreamError)


def _is_connection_closed(exc: BaseException) -> bool:
    """True for a websocket closure, identified by class not message text."""
    try:
        from websockets.exceptions import ConnectionClosed
    except ImportError:  # pragma: no cover - connect-path dependency
        return False
    return isinstance(exc, ConnectionClosed)


# aiortc name-mangles this on RTCRtpReceiver. Named once, with a canary test
# (tests/sfcpi/test_webrtc_nack.py) so an aiortc upgrade that renames it fails
# loudly in CI instead of silently bringing back 38-second sessions.
NACK_GENERATOR_ATTR = "_RTCRtpReceiver__nack_generator"
REMB_ESTIMATOR_ATTR = "_RTCRtpReceiver__remote_bitrate_estimator"


def disable_nack(pc) -> int:
    """Stop this peer connection sending RTCP NACK. Returns how many were off.

    The board cannot parse NACK (`PEER_CONNECTION_RESULT_FAIL_RTCP_PARSE_NACK`,
    result 36 in its log). That would be harmless on its own -- a retransmit
    request nobody honours -- except that the board refreshes its 30s
    INACTIVITY timer only when SRTCP handling SUCCEEDS
    (`HandleNonStunPackets`, peer_connection.c:1108). A rejected packet is
    therefore a missed keepalive, and the session is closed as idle roughly
    38s after it opens: 30s of timeout plus the time to connect.

    Editing the offer's `a=rtcp-fb` lines does NOT work: aiortc builds a
    NackGenerator for every video receiver unconditionally, without consulting
    the negotiated feedback (rtcrtpreceiver.py). It has to be cleared here.

    The cost is losing packet-loss retransmission, which this board never
    honoured anyway -- it could not read the requests.
    """
    disabled = 0
    for transceiver in pc.getTransceivers():
        receiver = getattr(transceiver, "receiver", None)
        if receiver is None:
            continue
        for attr in (NACK_GENERATOR_ATTR, REMB_ESTIMATOR_ATTR):
            # REMB too: the board has a matching parser failure
            # (PEER_CONNECTION_RESULT_FAIL_RTCP_PARSE_REMB, 35) and the same
            # consequence -- a packet it cannot parse is a keepalive it does
            # not count. Leaving only plain receiver reports, which are the
            # one RTCP type it has no dedicated failure code for.
            if getattr(receiver, attr, None) is not None:
                setattr(receiver, attr, None)
                disabled += 1
    return disabled


def default_client_id(host: str | None = None) -> str:
    """A STABLE viewer identity for this machine.

    The board keys its session table on the remote client id
    (`AppCommon_GetPeerConnectionSession` in app_common.c): a matching id
    reuses that session slot, a new one consumes a free slot. Slots are
    limited (`AWS_MAX_VIEWER_NUM`, 2 on this board) and are only released by
    the board's own close timer, so minting a fresh uuid4() per connect --
    which is what this used to do -- means every reconnect burns another
    slot. After a couple of reconnects both are held by ghosts of THIS
    process: signalling still answers, the SDP exchange still completes, and
    no session is left to carry media. That failure looks exactly like "the
    board stopped sending video".

    Stable per machine, distinct between machines, so two viewers of the same
    board do not fight over one slot.
    """
    import hashlib
    import socket

    name = host if host is not None else socket.gethostname()
    digest = hashlib.sha1(name.encode("utf-8", "replace")).hexdigest()[:12]
    return f"sfcpi-viewer-{digest}"



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
        detector: Optional[Any] = None,
        data_channel: bool = True,
    ) -> None:
        if warmup_frames < 1:
            raise ValueError("warmup_frames must be >= 1")
        self.signaling = signaling
        self.warmup_frames = warmup_frames
        # BoardDetector (or anything with .update); fed from the data channel
        # so counts come from the board's NPU rather than a second, disagreeing
        # host-side model run over video that already has the board's boxes
        # drawn into it.
        self.detector = detector
        # The AWS JS SDK makes the data channel a choice
        # (`if (formValues.openDataChannel)`). Ours defaults ON because the
        # board sends its detections there -- but offering an m=application
        # section a master answers without makes aiortc raise "Media sections
        # in answer do not match offer" and fails the whole connection, so it
        # must be switchable.
        self.data_channel = data_channel
        self.metadata_errors = 0
        # Media liveness, reported rather than acted on. The reference viewer
        # never tears a session down on connection state; nor do we.
        self.frames_received = 0
        self.media_ended = False
        # Our own host addresses, learned from our offer once it exists.
        self._local_hosts: List[str] = []
        self.filtered_candidates = 0
        # STATUS_RESPONSE messages from KVS (its own error channel).
        self.status_responses: List[Any] = []
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
                f"fps not yet measured: need {self._effective_warmup_frames} "
                f"frames to warm up, have {len(self._warmup_timestamps)}"
            )
        return self._fps

    @property
    def _effective_warmup_frames(self) -> int:
        # A single timestamp produces zero inter-frame deltas, so
        # statistics.median([]) has nothing to work with. warmup_frames=1 is
        # a legal constructor value (it means "don't make me wait long for
        # fps"), so silently need a second timestamp instead of raising
        # StatisticsError on the very first feed() -- fps still never
        # becomes available a frame earlier than the math allows.
        return max(self.warmup_frames, 2)

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
        if len(self._warmup_timestamps) >= self._effective_warmup_frames:
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
        credentials: Optional[Any] = None,
    ) -> None:
        """Open the KVS WebRTC signalling connection and start feeding frames.

        Validates that the signalling channel has a WSS endpoint BEFORE
        starting any thread or event loop, so a misconfigured channel fails
        immediately with a RuntimeError instead of hanging until the timeout.

        Blocks the calling thread until the remote video track OBJECT
        exists, or raises TimeoutError naming whichever phase -- "signalling
        connect" or "ICE/media negotiation" -- did not complete in time.

        Returning is NOT proof that media is flowing: aiortc fires its
        `track` event during setRemoteDescription, before ICE nomination and
        before the DTLS handshake. Verified live against real KVS -- a
        master that negotiates fully and then sends zero RTP still gets you
        a clean return here. The media confirmation is the first frame out
        of FrameBridge; a silent producer surfaces there as
        `TimeoutError: no frame within ...`, not from this method.

        The WSS URL is SigV4-signed before use (real KVS requires this; the
        handshake can't carry a normal Authorization header). `credentials`
        is a botocore-style credentials object and is INJECTED -- if not
        given, it is looked up lazily via `boto3.Session().get_credentials()`
        (imported here, not at module scope) but ONLY when signing is
        actually needed: `self.signaling.channel_arn` is `None` for any
        signalling object that hasn't done a real `describe()` (in
        particular the fakes the network-free tests use), and in that case
        signing is skipped entirely and the raw URL is used as-is, exactly
        as before this method could sign anything -- so those tests neither
        pay for nor depend on a `boto3` credential lookup. Against a real
        `KvsSignalingClient`, `channel_arn` is already cached from the
        `describe()` call `endpoints()` just made above, so this costs no
        extra network round trip.

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

        client_id = client_id or default_client_id()

        channel_arn = getattr(self.signaling, "channel_arn", None)
        if channel_arn:
            if credentials is None:
                import boto3  # noqa: F401  (lazy; connect-path only)

                credentials = boto3.Session().get_credentials()
            if credentials is None:
                raise RuntimeError(
                    f"KVS signalling channel {self.signaling.channel_name!r} "
                    f"needs a SigV4-signed WSS connection but no AWS "
                    f"credentials were found; pass credentials= or configure "
                    f"the environment (env vars, shared config, or an "
                    f"instance/role profile)"
                )
            wss_url = sign_wss_url(
                wss_url,
                channel_arn,
                self.signaling.region,
                credentials,
                client_id=client_id,
            )

        import aiortc  # noqa: F401  (lazy; connect-path only)
        import websockets  # noqa: F401  (lazy; connect-path only)

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
                websockets.connect(
                    wss_url,
                    ping_interval=SIGNALLING_PING_INTERVAL_S,
                    ping_timeout=SIGNALLING_PING_TIMEOUT_S,
                ),
                timeout=remaining
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

        # The board only allocates SCTP when the REMOTE offer enables a data
        # channel (peer_connection.c: ucEnableDataChannelRemote), so this
        # createDataChannel is what puts the m=application section in our
        # offer. Without it the detections have nowhere to go.
        own_channel = None
        if self.data_channel:
            try:
                own_channel = pc.createDataChannel(DETECTION_CHANNEL_LABEL)
            except Exception:  # noqa: BLE001 - a peer with no SCTP still streams video
                own_channel = None
        if own_channel is not None:
            self._attach_detection_handler(own_channel)

        # The board pushes on EVERY open channel on its session, including
        # ones it opened itself.
        @pc.on("datachannel")
        def on_datachannel(channel: Any) -> None:
            self._attach_detection_handler(channel)

        @pc.on("track")
        def on_track(track: Any) -> None:
            if track.kind == "video":
                asyncio.ensure_future(self._recv_loop(track))
                track_ready.set()

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        await self._await_ice_gathering_complete(pc)
        # aiortc advertises sha-256, sha-384 AND sha-512 fingerprints; the
        # KVS C SDK master rejects the 191-character sha-512 one and then
        # fails certificate verification, tearing the session down AFTER our
        # own DTLS reports success. Send only sha-256, as a browser does.
        self._local_hosts = local_host_addresses(pc.localDescription.sdp)
        offer_sdp = keep_single_fingerprint(pc.localDescription.sdp)
        await ws.send(encode_sdp_offer(client_id, offer_sdp))

        # Trickle our own candidates. The offer already carries them in its
        # SDP (we gather-then-send), but the KVS C SDK master does not adopt
        # them from there: without these messages it forms no valid pair,
        # never sends the DTLS ClientHello, and drops the session after ~30s.
        # Verified live -- see tests/sfcpi/test_webrtc_trickle.py.
        for candidate in sdp_candidate_lines(pc.localDescription.sdp):
            await ws.send(encode_ice_candidate(
                candidate.candidate, candidate.sdp_mid, candidate.sdp_mline_index
            ))

        # Candidates that arrive BEFORE the answer must be queued, not dropped:
        # setRemoteDescription has not run yet, so addIceCandidate cannot be
        # called. The AWS JS SDK does exactly this (emitOrQueueIceCandidate,
        # replayed by emitPendingIceCandidates once the remote SDP lands).
        # Observed live: the board sends its HOST candidate -- the fastest
        # path -- before the answer, so dropping them cost the best route on
        # every single session.
        pending_candidates = []

        while True:
            raw = await ws.recv()
            # Real KVS's FIRST frame after the offer is an empty keepalive;
            # decode_message rightly calls that malformed, so skip it here
            # rather than weakening decode_message. See signaling.is_keepalive.
            if is_keepalive(raw):
                continue
            message_type, _sender, payload = decode_message(raw)
            if message_type == "ICE_CANDIDATE":
                pending_candidates.append(payload)
                continue
            if message_type == "STATUS_RESPONSE":
                # The service's own error channel. Surfacing it beats guessing
                # why a session misbehaved.
                self.status_responses.append(payload)
                continue
            if message_type == "SDP_ANSWER":
                # The SDK guards this: "Ignoring SDP answer in signaling state".
                # Applying a second answer once we are 'stable' raises inside
                # aiortc and would kill an otherwise healthy session.
                state = getattr(pc, "signalingState", "have-local-offer")
                if state != "have-local-offer":
                    continue
                await pc.setRemoteDescription(
                    aiortc.RTCSessionDescription(
                        sdp=payload["sdp"], type=payload["type"]
                    )
                )
                break

        # Must happen after setRemoteDescription: the receivers only exist
        # once the answer has been applied.
        disable_nack(pc)

        # Replay whatever arrived before the answer, now that the remote
        # description exists.
        for payload in pending_candidates:
            await self._add_remote_candidate(pc, payload)

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

    async def _add_remote_candidate(self, pc: Any, payload: dict) -> None:
        """Apply one remote ICE candidate. Shared by the pre-answer replay and
        the post-answer pump so both handle malformed entries identically."""
        candidate_str = payload.get("candidate")
        if not candidate_str:
            return
        if not is_reachable_candidate(candidate_str, self._local_hosts):
            # Unroutable private host candidate: handing it to TURN yields
            # `403 Forbidden IP`. See is_reachable_candidate.
            self.filtered_candidates += 1
            return
        from aiortc.sdp import candidate_from_sdp

        candidate = candidate_from_sdp(candidate_str)
        candidate.sdpMid = payload.get("sdpMid")
        candidate.sdpMLineIndex = payload.get("sdpMLineIndex")
        await pc.addIceCandidate(candidate)

    async def _pump_ice_candidates(self, ws: Any, pc: Any) -> None:
        """Apply trickled ICE candidates the far side sends after the answer.

        Best-effort: this class only ever sends a non-trickle (gather-then-send)
        offer of its own, but still applies whatever the master trickles in, in
        case that peer needs it. Any failure here reaches the consumer via
        fail() exactly like every other producer-side error.
        """
        try:
            async for raw in ws:
                # Keepalives keep arriving for the life of the session; one
                # reaching decode_message would fail() the whole stream.
                if is_keepalive(raw):
                    continue
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
        except Exception as exc:  # noqa: BLE001
            # A CLOSED signalling socket is not a media failure. WebRTC
            # media has its own ICE/DTLS transport; signalling exists to
            # set the session up and trickle candidates. Tearing down a
            # healthy video stream because this socket went away is the
            # bug that made every session last ~40s (websockets closes it
            # itself on ping_interval=20 + ping_timeout=20). Anything
            # else still reaches the consumer: a real producer fault must
            # never become a silently empty -- and falsely calm -- stream.
            if _is_connection_closed(exc):
                return
            self.fail(exc)

    def _attach_detection_handler(self, channel: Any) -> None:
        """Route data-channel messages into the detector."""

        @channel.on("message")
        def on_message(message: Any) -> None:
            if self.detector is None:
                return
            try:
                self.detector.update(parse_detection_message(message))
            except ValueError:
                # A malformed message must NOT fail() the stream -- the video
                # is fine and the board is still there. It must also not leave
                # the previous count standing as fresh: not calling update()
                # means the detector ages out on its own and the pipeline
                # reports an unknown count rather than a stale crowd.
                self.metadata_errors += 1

    async def _recv_loop(self, track: Any) -> None:
        """Pull decoded frames off an aiortc MediaStreamTrack and feed() them.

        Any failure here -- the track ending, a decode error, a dead
        connection -- must reach the consumer via fail(), never end the
        stream silently: a silently-empty stream reads as "calm crowd"
        downstream, which is exactly the failure mode this project exists to
        avoid.
        """
        loop = asyncio.get_event_loop()
        try:
            while True:
                frame = await track.recv()
                timestamp = getattr(frame, "time", None)
                if timestamp is None:
                    timestamp = time.monotonic()
                # Convert OFF the event loop. `to_ndarray(format="bgr24")` is a
                # full YUV->BGR conversion of a 1280x704 frame; doing it here
                # synchronously at 30fps starves the loop that also has to send
                # ICE consent checks and RTCP, and a starved connection dies.
                # PyAV releases the GIL for the conversion, so a worker thread
                # genuinely buys us the loop back.
                #
                # feed() re-runs _as_bgr_ndarray, which is a no-op for an
                # ndarray, so passing the converted image through is safe and
                # keeps one code path for both callers.
                image = await loop.run_in_executor(
                    None, self._as_bgr_ndarray, frame
                )
                self.feed(image, timestamp)
                self.frames_received += 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            # A track ENDING is not a session failure. aiortc raises
            # MediaStreamError when a track simply stops, and the AWS JS SDK's
            # viewer never tears a session down on connection state -- its
            # connectionstatechange handler only logs. Killing the session here
            # also throws away the DATA CHANNEL, which rides its own SCTP
            # transport and carries the board's detection counts: the input the
            # risk engine actually runs on. Record it and let the caller decide.
            #
            # Everything else still reaches the consumer. A real producer fault
            # must never become a silently empty -- and therefore falsely calm
            # -- stream.
            if _is_media_ended(exc):
                # A clean END of media, not a fault. Close the bridge so the
                # consumer's iteration finishes instead of blocking forever:
                # the bridge has no idle timeout, so without this the stream
                # simply stops and is never picked back up. Closing (rather
                # than failing) lets the caller start a FRESH session after the
                # board's slot-reclaim wait, without recording an error that
                # never happened.
                self.media_ended = True
                self._bridge.close()
                return
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
