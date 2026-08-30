"""CrowdSentinel dashboard — a VIEWER of the board's own detections.

There is deliberately NO detector in this process and no RTSP client.

The AmebaPro2 runs SCRFD on its NPU, draws those boxes into the outgoing
H.264 frames as OSD rectangles, and streams over AWS KVS WebRTC. It also
emits each inference as JSON on the WebRTC data channel. So this host
decodes the video and *reads* the board's counts; running a second model
here would burn CPU to produce a different, disagreeing set of boxes over
video that already has the board's boxes painted into it.

The count therefore has three states, not two:

    a number   -- fresh metadata from the board
    unknown    -- nothing received yet, or the last message is stale
    (never 0 as a stand-in for either of the above)

`sfcpi.detect.BoardDetector` raises for both unknown cases, and every path
here turns that into `count: null` / "NO DATA" on the wire and in the UI.
A stale or missing count rendered as 0 reads as "empty, calm scene" — the
exact failure this system exists to prevent.
"""
from __future__ import annotations

import os
import itertools
import math
import re
import json
import sys
import threading
import time

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# src/ holds the sfcpi package. Usually already importable; don't rely on it.
_SRC_DIR = os.path.dirname(BASE_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from sfcpi.detect import BoardDetector
from sfcpi.flow import FlowEstimator
from sfcpi.grid import CellGrid
from sfcpi.pipeline import Pipeline
from sfcpi.risk.levels import RiskLevel, Thresholds
from sfcpi.risk.machine import RiskStateMachine
from sfcpi.alerts.log import LogSink  # noqa: E402

load_dotenv(os.path.join(BASE_DIR, ".env"))

app = Flask(__name__, template_folder="templates", static_folder="static")

# --- Configuration -----------------------------------------------------------
# AWS credentials are NOT read here. boto3's own chain resolves them, so they
# never pass through this module and can never be printed by it.
KVS_CHANNEL_NAME = os.getenv("KVS_CHANNEL_NAME", "camstream")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
# 30s, matching the CLI. KVS negotiation against a busy board runs close
# enough to 15s that the tighter value turned ordinary contention into
# a reported error.
CONNECT_TIMEOUT_S = float(os.getenv("KVS_CONNECT_TIMEOUT", "30"))
# How old board metadata may get before the count becomes unknown again.
METADATA_MAX_AGE_S = float(os.getenv("METADATA_MAX_AGE", "2.0"))
PORT = int(os.getenv("PORT", "5000"))

# SF-CPI pipeline. The risk level shown here is crowd PRESSURE in s^-2
# (P = density * fps^2 * Var(velocity)), NOT a headcount threshold: 200 people
# standing still is not an emergency, 15 shoving in a doorway is.
CELL_SIZE = int(os.getenv("SFCPI_CELL_SIZE", "64"))
FLOW_DOWNSCALE = float(os.getenv("SFCPI_FLOW_DOWNSCALE", "0.5"))
SCORE_FIELD = os.getenv("SFCPI_SCORE_FIELD", "global_max_pressure")
MIN_DWELL_S = float(os.getenv("SFCPI_MIN_DWELL", "2.0"))
MIN_REALERT_S = float(os.getenv("SFCPI_MIN_REALERT", "60.0"))
BLIND_ALERT_S = float(os.getenv("SFCPI_BLIND_ALERT", "30.0"))
MIN_COVERAGE = float(os.getenv("SFCPI_MIN_COVERAGE", "0.0"))

# SNS is OFF unless a topic is configured AND explicitly enabled, and even
# then it is dry-run unless SNS_DRY_RUN=0. Paging real people is not a default.
SNS_TOPIC_ARN = os.getenv("SNS_TOPIC_ARN", "")
SNS_ENABLE = os.getenv("SNS_ENABLE", "0") == "1"
SNS_DRY_RUN = os.getenv("SNS_DRY_RUN", "1") != "0"
# Only page for this level and above. CRITICAL by default: the log sink still
# records every event, so nothing is lost -- this is about what is worth
# waking someone for. SNS_INCLUDE_UNKNOWN covers sensor-blind, which has no
# severity ordinal and is therefore NOT included by a level threshold alone.
SNS_MIN_LEVEL = os.getenv("SNS_MIN_LEVEL", "critical")
SNS_INCLUDE_UNKNOWN = os.getenv("SNS_INCLUDE_UNKNOWN", "0") == "1"

# Reconnection. A session ending is normal, not fatal.
# The board reclaims a stale viewer slot only after its own 30s inactivity
# timer (PEER_CONNECTION_INACTIVE_CONNECTION_TIMEOUT_MS). Retrying faster than
# that cannot succeed when the failure IS slot exhaustion, and a tight retry
# loop keeps the board busy refusing us instead of letting it recover.
# Reconnection is OFF by default. The board has two viewer slots and reclaims
# a stale one only after 30s, so every reconnect is a new viewer competing for
# a scarce resource. An automatic loop turns one dropped session into a storm
# that starves the board entirely -- which is far worse than showing an honest
# "session ended" and waiting for a human. Set RECONNECT=1 to re-enable.
# Reconnection is ON again, but only because it is now SAFE: the worker waits
# out the board's full 32s slot-reclaim window before reconnecting and backs
# off 10s -> 60s. The earlier storm came from retrying every 3s into a board
# that needs 30s to free a viewer slot. Without this a stream that stops
# mid-session is never picked back up. RECONNECT=0 disables it.
RECONNECT_ENABLED = os.getenv("RECONNECT", "1") == "1"
RECONNECT_BACKOFF_MIN_S = float(os.getenv("RECONNECT_BACKOFF_MIN", "10"))
RECONNECT_BACKOFF_MAX_S = float(os.getenv("RECONNECT_BACKOFF_MAX", "60"))
# A session lasting at least this long counts as healthy and resets backoff.
RECONNECT_HEALTHY_S = float(os.getenv("RECONNECT_HEALTHY", "20"))
# The board frees a viewer slot only after PEER_CONNECTION_INACTIVE_CONNECTION_
# TIMEOUT_MS (30s) of silence -- it does not learn about our close immediately.
# Restarting sooner takes the OTHER slot, and two quick restarts leave both
# held by sessions we already closed: signalling still answers ("negotiated")
# but no peer connection is left to send media.
BOARD_RECLAIM_S = float(os.getenv("BOARD_RECLAIM", "32"))
# How long without a frame before the UI stops presenting the last reading as
# current. This does NOT end the session -- the reference viewer never tears a
# connection down, and the data channel keeps delivering counts on its own
# transport. It only stops us showing a stale pressure as if it were live.
MEDIA_STALE_S = float(os.getenv("MEDIA_STALE", "10"))

# Statuses the UI understands.
IDLE = "idle"
CONNECTING = "connecting"
LIVE = "live"           # video frames arriving
NO_VIDEO = "no_video"   # connected, but the board is sending no RTP
ERROR = "error"

_STATUS_TEXT = {
    IDLE: "SYSTEM STANDBY",
    CONNECTING: "CONNECTING TO KVS...",
    LIVE: "LIVE — BOARD FEED",
    NO_VIDEO: "WAITING FOR STREAM",
    ERROR: "CONNECTION ERROR",
}

# BoardDetector.detect() ignores the image; it only checks metadata freshness.
_NO_IMAGE = np.zeros((1, 1, 3), dtype=np.uint8)

# Anything that could carry an AWS key out of a boto3/botocore error string.
_SECRET_PATTERNS = (
    re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{12,}\b"),
    re.compile(r"(?i)\b(aws_)?(secret|session|security)[_a-z]*\s*[=:]\s*\S+"),
    re.compile(r"(?i)\bX-Amz-(Signature|Credential|Security-Token)=[^&\s]+"),
)


def _redact(text: str) -> str:
    """Never let a credential reach a log line, an HTTP body or the UI."""
    out = str(text)
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub("[redacted]", out)
    return out


def log(message: str) -> None:
    print(f"[dashboard] {_redact(message)}", flush=True)


# One Thresholds instance so the UI and the state machine cannot disagree
# about where ELEVATED starts.
THRESHOLDS = Thresholds()


def _finite_or_none(value):
    """NaN/inf -> None. JSON has no NaN, and more importantly an unknown
    pressure rendered as 0.0 reads as a calm crowd."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# --- Shared state ------------------------------------------------------------
class StreamState:
    """Everything the Flask threads read and the worker thread writes.

    The lock covers only what this module owns (frame, status, counters).
    The detector is written by aiortc's loop thread via `WebRTCSource`'s
    data-channel handler; its reads here are single attribute loads.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.status = IDLE
        self.detail = ""
        self.frame = None
        self.frames_seen = 0
        self.dropped = 0
        self.detector = BoardDetector(max_age_s=METADATA_MAX_AGE_S)
        self.source = None
        self.thread = None
        # SF-CPI: pressure is the alarm quantity, not the count.
        self.pressure = None
        self.max_pressure = None
        self.level = RiskLevel.NORMAL.value
        self.coverage = None
        self.last_alert = None
        # The per-cell pressure lattice. This is the thing SF-CPI actually
        # computes -- a single global number throws away WHERE the pressure
        # is, which is the operationally useful part.
        self.cells = None
        self.grid_shape = None
        self.last_frame_at = None
        # When the last session ended, so a restart can wait for the board to
        # reclaim its slot instead of racing it.
        self.last_session_end = None

    def set_status(self, status: str, detail: str = "") -> None:
        with self.lock:
            self.status = status
            self.detail = _redact(detail)

    def snapshot(self) -> dict:
        with self.lock:
            status = self.status
            detail = self.detail
            frames_seen = self.frames_seen
            dropped = self.dropped
            image = self.frame
            detector = self.detector
            pressure = self.pressure
            max_pressure = self.max_pressure
            level = self.level
            coverage = self.coverage
            last_alert = self.last_alert
            cells = self.cells
            grid_shape = self.grid_shape
            last_frame_at = self.last_frame_at
            running = self.running

        count, reason = _read_board_count(detector, image)

        # Media stalled: keep the session (and the data channel with it), but
        # stop presenting the last pressure reading as if it were current. A
        # frozen number is indistinguishable from a live calm one, which is
        # precisely the lie this project exists to avoid. The COUNT is left
        # alone -- it arrives on the data channel and has its own freshness
        # check in BoardDetector.
        stale = (running and last_frame_at is not None
                 and (time.monotonic() - last_frame_at) > MEDIA_STALE_S)
        if stale:
            status = NO_VIDEO
            detail = (f"no video for {time.monotonic() - last_frame_at:.0f}s; "
                      f"session still open")
            pressure = max_pressure = coverage = None
            cells = grid_shape = None
            image = None
        return {
            "status": status,
            "status_text": _STATUS_TEXT.get(status, status.upper()),
            "detail": detail,
            "known": count is not None,
            "count": count,
            "count_reason": reason,
            "frames": frames_seen,
            "dropped": dropped,
            "channel": KVS_CHANNEL_NAME,
            "region": AWS_REGION,
            # None, not 0.0: an unknown pressure must not render as calm.
            "pressure": _finite_or_none(pressure),
            "max_pressure": _finite_or_none(max_pressure),
            "level": level,
            "coverage": _finite_or_none(coverage),
            "last_alert": last_alert,
            "cells": cells,
            "grid_shape": grid_shape,
            "thresholds": {
                "elevated": THRESHOLDS.elevated_rise,
                "high": THRESHOLDS.high_rise,
                "critical": THRESHOLDS.critical_rise,
            },
        }


def _read_board_count(detector, image):
    """(count, reason) — count is None whenever the board's count is UNKNOWN.

    `detect()` is what enforces freshness, so it is the gate; `last_count`
    is the authoritative number behind it (the board's `n`, which can exceed
    the number of boxes it had room to send).
    """
    if detector is None:
        return None, "no stream session"
    try:
        detector.detect(_NO_IMAGE if image is None else image)
    except Exception as exc:  # noqa: BLE001 - any failure means UNKNOWN
        return None, _redact(str(exc))
    count = detector.last_count
    if count is None:
        return None, "board sent no count"
    return int(count), None


state = StreamState()


# --- Worker ------------------------------------------------------------------
def _build_source(detector):
    """Mirror sfcpi.cli._cmd_live: signalling client, ICE servers, source."""
    import boto3

    from sfcpi.webrtc.signaling import KvsSignalingClient, ice_servers_to_rtc
    from sfcpi.webrtc.source import WebRTCSource

    kinesisvideo = boto3.client("kinesisvideo", region_name=AWS_REGION)
    signaling = KvsSignalingClient(
        KVS_CHANNEL_NAME, AWS_REGION, kinesisvideo, role="VIEWER"
    )
    endpoints = signaling.endpoints()

    # STUN always; the channel's TURN servers are best-effort. Without TURN
    # this will not traverse a symmetric NAT, but a same-LAN board still
    # connects on host candidates, so a GetIceServerConfig failure is a
    # warning rather than a fatal error.
    ice = [{"urls": [f"stun:stun.kinesisvideo.{AWS_REGION}.amazonaws.com:443"]}]
    try:
        signaling_client = boto3.client(
            "kinesis-video-signaling",
            endpoint_url=endpoints["HTTPS"],
            region_name=AWS_REGION,
        )
        ice.extend(ice_servers_to_rtc(signaling.ice_servers(signaling_client)))
    except Exception as exc:  # noqa: BLE001 - degraded, not fatal
        log(f"warning: GetIceServerConfig failed ({exc}); STUN only")

    source = WebRTCSource(
        signaling,
        warmup_frames=10,
        connect_timeout_s=CONNECT_TIMEOUT_S,
        detector=detector,
    )
    return source, ice


def _build_alerting():
    """Build the alert sinks and the risk state machine ONCE per worker.

    Deliberately outside the per-connection loop: rebuilding the state machine
    on every reconnect would reset the dwell and re-alert timers, so a board
    that flaps would page on every reconnection. The gating only means
    anything if it survives the transport.
    """
    sinks = [LogSink()]
    if SNS_ENABLE and SNS_TOPIC_ARN:
        try:
            import boto3
            from sfcpi.alerts.sns import SnsSink
            from sfcpi.alerts.filter import LevelFilterSink, level_from_name

            sns = SnsSink(SNS_TOPIC_ARN,
                          boto3.client("sns", region_name=AWS_REGION),
                          dry_run=SNS_DRY_RUN)
            min_level = level_from_name(SNS_MIN_LEVEL)
            sinks.append(LevelFilterSink(sns, min_level,
                                         include_unknown=SNS_INCLUDE_UNKNOWN))
            log(f"SNS enabled (dry_run={SNS_DRY_RUN}, "
                f"min_level={min_level.value}, "
                f"include_unknown={SNS_INCLUDE_UNKNOWN}) -- LogSink still "
                f"records every event")
        except Exception as exc:  # noqa: BLE001 - alerting is not the video path
            log(f"warning: SNS sink unavailable: {exc}")
    elif SNS_ENABLE:
        log("warning: SNS_ENABLE=1 but SNS_TOPIC_ARN is empty; not alerting")

    machine = RiskStateMachine(
        THRESHOLDS,
        min_dwell_s=MIN_DWELL_S,
        min_realert_s=MIN_REALERT_S,
        blind_alert_s=BLIND_ALERT_S,
        min_coverage=MIN_COVERAGE,
    )
    return sinks, machine


def _stream_session(detector, sinks, machine) -> None:
    """Connect, then pump frames until stopped.

    Frame timeouts do NOT tear the session down. The data channel is
    independent of the video track: a board that negotiates, streams
    detections and sends no RTP must still show a live count next to an
    honest "waiting for stream" video panel.
    """
    log(f"connecting to KVS channel {KVS_CHANNEL_NAME!r} in {AWS_REGION}")
    try:
        source, ice = _build_source(detector)
    except Exception as exc:  # noqa: BLE001
        log(f"error: could not prepare KVS session: {exc}")
        state.set_status(ERROR, f"KVS setup failed: {exc}")
        return

    with state.lock:
        state.source = source

    try:
        source.connect(ice_servers=ice)
    except Exception as exc:  # noqa: BLE001 - RuntimeError/TimeoutError and more
        log(f"error: connect failed: {exc}")
        state.set_status(ERROR, str(exc))
        try:
            source.close()
        except Exception:  # noqa: BLE001 - cleanup is best-effort
            pass
        with state.lock:
            state.source = None
        return

    # connect() returning proves the track object exists, not that media is
    # flowing. The first frame out of the bridge is the only proof of that.
    # Guarded: a /stop_stream issued DURING connect() has already put the UI
    # back to standby, and this must not resurrect it as a live session.
    with state.lock:
        if not state.running:
            state.status = IDLE
            state.detail = ""
            state.source = None
            return
        state.status = NO_VIDEO
        state.detail = "negotiated; waiting for the first video frame"
    log("negotiated; waiting for media")

    # ---- SF-CPI: frames -> pressure -> risk -> alerts, in THIS process ----
    # One WebRTC session serves both the video panel and the analytics. The
    # board only allows two viewer sessions, so a second process just to
    # compute pressure would burn half the budget and give the UI a second,
    # disagreeing notion of risk.
    def _display_and_yield(frames, width, height):
        """Publish each frame to the video panel, then hand it to the pipeline."""
        for frame in frames:
            with state.lock:
                if not state.running:
                    return
                state.frame = frame.image
                state.last_frame_at = time.monotonic()
                state.frames_seen += 1
                state.dropped = source.dropped
                if state.status != LIVE:
                    state.status = LIVE
                    state.detail = ""
            yield type(frame)(frame.index, frame.timestamp,
                              frame.image[:height, :width])

    try:
        pipeline = None
        while True:
            with state.lock:
                if not state.running:
                    break
            try:
                frames = iter(source)
                first = next(frames)

                if pipeline is None:
                    h, w = first.image.shape[:2]
                    w -= w % CELL_SIZE
                    h -= h % CELL_SIZE
                    try:
                        fps = source.fps
                    except Exception:  # noqa: BLE001 - warm-up not finished
                        fps = 30.0
                    pipeline = Pipeline(
                        grid=CellGrid(frame_width=w, frame_height=h,
                                      cell_size=CELL_SIZE),
                        flow_estimator=FlowEstimator(downscale=FLOW_DOWNSCALE),
                        detector=detector,
                        fps=fps,
                    )
                    log(f"pipeline: {w}x{h} cell={CELL_SIZE} "
                        f"flow_downscale={FLOW_DOWNSCALE}")
                    crop_w, crop_h = w, h

                stream = itertools.chain([first], frames)
                for metrics in pipeline.run(
                        _display_and_yield(stream, crop_w, crop_h)):
                    pressure = getattr(metrics, SCORE_FIELD)
                    event = None
                    try:
                        event = machine.update(
                            metrics.timestamp, pressure,
                            coverage=metrics.sensing_confidence)
                    except ValueError as exc:
                        # A bad timestamp is a data fault, not a reason to
                        # drop the video session.
                        log(f"warning: risk update skipped: {exc}")

                    lattice = metrics.cells.get("pressure")
                    if lattice is not None:
                        lattice = np.asarray(lattice)
                        # None per cell, not 0.0: an uncomputable cell is
                        # unknown, and a grid that renders it as calm is
                        # exactly the lie this project exists to avoid.
                        rows = [[(float(v) if math.isfinite(v) else None)
                                 for v in row] for row in lattice]
                    else:
                        rows = None

                    with state.lock:
                        state.pressure = metrics.global_pressure
                        state.max_pressure = metrics.global_max_pressure
                        state.coverage = metrics.sensing_confidence
                        state.level = machine.level.value
                        state.cells = rows
                        state.grid_shape = (None if rows is None
                                            else [len(rows), len(rows[0])])
                        if event is not None:
                            state.last_alert = {
                                "level": event.level.value,
                                "previous_level": event.previous_level.value,
                                "reason": event.reason,
                                "message": _redact(event.message),
                                "timestamp": event.timestamp,
                            }
                    if event is not None:
                        for sink in sinks:
                            try:
                                sink.publish(event)
                            except Exception as exc:  # noqa: BLE001
                                log(f"warning: alert sink failed: {exc}")
                break
            except StopIteration:
                break
            except TimeoutError as exc:
                # The producer is silent. Counts may still be arriving on the
                # data channel, so keep the session and re-iterate.
                with state.lock:
                    if not state.running:
                        break
                    state.frame = None
                    if state.status != NO_VIDEO:
                        log(f"no video: {exc}")
                    state.status = NO_VIDEO
                    state.detail = _redact(str(exc))
                continue
    except Exception as exc:  # noqa: BLE001 - producer-side failure
        # aiortc raises MediaStreamError with NO message when a track ends, so
        # str(exc) is "" and the log line reads "stream failed: ". Name the
        # type; a blank reason is not a diagnosis.
        reason = str(exc) or type(exc).__name__
        log(f"error: stream failed: {reason}")
        state.set_status(ERROR, reason)
    finally:
        try:
            source.close()
        except Exception:  # noqa: BLE001
            pass
        with state.lock:
            # Only clear the handle if it is still ours: a stop/start overlap
            # can leave this worker running past the next session's start.
            if state.source is source:
                state.source = None
            state.frame = None
            # Standby must not leave the last pressure/level standing as if
            # it were current.
            state.pressure = None
            state.max_pressure = None
            state.coverage = None
            state.level = RiskLevel.NORMAL.value
            state.cells = None
            state.grid_shape = None
            state.last_frame_at = None
        with state.lock:
            state.last_session_end = time.monotonic()
        log("session ended")


def stream_worker(detector) -> None:
    """Keep a live session up for as long as the operator wants one.

    A WebRTC session ends for ordinary reasons -- the board reclaims a viewer
    slot on its 30s inactivity timer, the track ends, the network blips -- and
    aiortc surfaces that as a MediaStreamError with an empty message. A
    monitoring dashboard that stops on the first of those, and needs someone
    to click Start again, is not monitoring anything.

    The alert sinks and the risk state machine are built ONCE and reused
    across reconnects, so dwell and re-alert gating survive the transport;
    rebuilding them per connection would let a flapping board page on every
    reconnect.
    """
    sinks, machine = _build_alerting()

    with state.lock:
        last_end = state.last_session_end
    if last_end is not None:
        wait = BOARD_RECLAIM_S - (time.monotonic() - last_end)
        if wait > 0:
            log(f"waiting {wait:.0f}s for the board to reclaim its viewer slot")
            with state.lock:
                state.status = CONNECTING
                state.detail = (f"waiting {wait:.0f}s for the board to release "
                                f"the previous session")
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline:
                with state.lock:
                    if not state.running:
                        break
                time.sleep(0.25)

    backoff = RECONNECT_BACKOFF_MIN_S
    while True:
        with state.lock:
            if not state.running:
                break

        started = time.monotonic()
        _stream_session(detector, sinks, machine)
        lasted = time.monotonic() - started

        with state.lock:
            if not state.running:
                # A stop was requested: standby, not an error.
                state.status = IDLE
                state.detail = ""
                break

        # A session that actually ran is evidence the path works, so the next
        # failure starts patient again rather than inheriting a long backoff
        # from an unrelated earlier problem.
        if lasted >= RECONNECT_HEALTHY_S:
            backoff = RECONNECT_BACKOFF_MIN_S

        if not RECONNECT_ENABLED:
            log(f"session ended after {lasted:.0f}s; not reconnecting "
                f"(set RECONNECT=1 to auto-reconnect)")
            with state.lock:
                state.running = False
                state.status = ERROR
                state.detail = (f"session ended after {lasted:.0f}s. Press "
                                f"Start to reconnect.")
            break

        log(f"reconnecting in {backoff:.0f}s (session lasted {lasted:.0f}s)")
        with state.lock:
            state.status = CONNECTING
            state.detail = f"reconnecting in {backoff:.0f}s"

        # Sleep in slices so a stop is honoured promptly instead of after the
        # whole backoff.
        deadline = time.monotonic() + backoff
        while time.monotonic() < deadline:
            with state.lock:
                if not state.running:
                    break
            time.sleep(0.25)

        backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX_S)

    with state.lock:
        state.running = False
        if state.status not in (ERROR,):
            state.status = IDLE
            state.detail = ""
    log("worker stopped")


# --- Placeholder video -------------------------------------------------------
_placeholder_cache: dict[str, bytes] = {}


def _placeholder_jpeg(status: str) -> bytes:
    """A clearly synthetic 'no video' card.

    Never a still of the last real frame: a frozen frame is indistinguishable
    from a live one and would imply the scene is still being observed.
    """
    cached = _placeholder_cache.get(status)
    if cached is not None:
        return cached

    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    canvas[:] = (18, 14, 10)
    for y in range(0, 360, 24):
        cv2.line(canvas, (0, y), (640, y), (28, 24, 18), 1)
    cv2.rectangle(canvas, (8, 8), (631, 351), (90, 90, 60), 1)

    headline = _STATUS_TEXT.get(status, status.upper())
    cv2.putText(canvas, headline, (40, 170), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "no video from the board", (40, 205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 140, 150), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"channel {KVS_CHANNEL_NAME} / {AWS_REGION}", (40, 232),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 110, 120), 1, cv2.LINE_AA)

    ok, buffer = cv2.imencode(".jpg", canvas)
    if not ok:  # pragma: no cover - encoder failure
        return b""
    data = buffer.tobytes()
    _placeholder_cache[status] = data
    return data


# --- Routes ------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", channel=KVS_CHANNEL_NAME,
                           region=AWS_REGION)


@app.route("/start_stream", methods=["POST"])
def start_stream():
    """Returns immediately; connecting takes seconds and happens in the worker."""
    with state.lock:
        if state.running:
            return jsonify({"status": "already_running"})
        state.running = True
        state.status = CONNECTING
        state.detail = ""
        state.frame = None
        state.frames_seen = 0
        state.dropped = 0
        # A fresh detector: the previous session's last count must not be
        # inherited as if it were current.
        state.detector = BoardDetector(max_age_s=METADATA_MAX_AGE_S)
        detector = state.detector
        state.thread = threading.Thread(
            target=stream_worker, args=(detector,),
            name="kvs-stream", daemon=True,
        )
        thread = state.thread
    thread.start()
    return jsonify({"status": "started", "channel": KVS_CHANNEL_NAME})


@app.route("/stop_stream", methods=["POST"])
def stop_stream():
    with state.lock:
        if not state.running and state.thread is None:
            return jsonify({"status": "not_running"})
        state.running = False
        source = state.source
        thread = state.thread

    if source is not None:
        try:
            source.close()
        except Exception as exc:  # noqa: BLE001
            log(f"warning: close failed: {exc}")
    if thread is not None and thread.is_alive():
        thread.join(timeout=8.0)

    with state.lock:
        state.thread = None
        state.source = None
        state.frame = None
        state.status = IDLE
        state.detail = ""
        # Standby means the count is UNKNOWN, not zero. A fresh BoardDetector
        # raises on detect(), which is exactly that.
        state.detector = BoardDetector(max_age_s=METADATA_MAX_AGE_S)
    return jsonify({"status": "stopped"})


@app.route("/status")
def status():
    return jsonify(state.snapshot())


@app.route("/video_feed")
def video_feed():
    """MJPEG of the board's decoded frames — its boxes are already in them.

    Nothing is drawn over a real frame here. When there is no frame the
    stream carries a placeholder card, so the panel is never a stale still.
    """

    def generate():
        while True:
            with state.lock:
                image = state.frame
                current_status = state.status
            if image is not None:
                ok, buffer = cv2.imencode(".jpg", image)
                payload = buffer.tobytes() if ok else None
            else:
                payload = _placeholder_jpeg(current_status)
            if payload:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + payload + b"\r\n")
            time.sleep(0.033 if image is not None else 0.5)

    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/count_feed")
def count_feed():
    """SSE of the board's count. JSON, because `null` has to be expressible."""

    def generate():
        while True:
            yield f"data: {json.dumps(state.snapshot())}\n\n"
            time.sleep(0.5)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    log(f"channel={KVS_CHANNEL_NAME} region={AWS_REGION} port={PORT}")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
