"""Command line entry points."""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .detect import NullDetector, YoloDetector
from .flow import FlowEstimator
from .grid import CellGrid
from .pipeline import Pipeline
from .sinks import JsonlSink
from .sources import DEFAULT_FPS as DEFAULT_EVAL_FPS, FileSource


# This deployment has one board on one channel, so these are defaults rather
# than required flags. Environment first, using the same names the dashboard
# reads (src/dashboard/server.py), so the two entry points cannot be pointed at
# different boards by accident.
DEFAULT_CHANNEL = os.environ.get("KVS_CHANNEL_NAME", "camstream")
DEFAULT_REGION = (
    os.environ.get("AWS_REGION")
    or os.environ.get("AWS_DEFAULT_REGION")
    or "ap-south-1"
)


def _cmd_run(args: argparse.Namespace) -> int:
    # FileSource probes fps at construction time (not first iteration), so a
    # missing file raises FileNotFoundError from the constructor itself --
    # the constructor must be inside the try, not just the first next().
    try:
        source = FileSource(args.video, max_frames=args.max_frames)
        first = next(iter(source))
    except OSError as exc:
        # OSError, not FileNotFoundError: sources.FileSource raises the bare
        # parent for an existing-but-unopenable video (corrupt file, bad
        # permissions). FileNotFoundError is a SUBCLASS, so catching only it
        # let the corrupt-file case escape as a traceback. Both carry the path.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except StopIteration:
        print("error: video contained no frames", file=sys.stderr)
        return 2

    h, w = first.image.shape[:2]
    cell = args.cell_size
    w -= w % cell
    h -= h % cell
    grid = CellGrid(frame_width=w, frame_height=h, cell_size=cell)

    detector = NullDetector() if args.no_detector else YoloDetector(args.model, conf=args.conf)

    pipeline = Pipeline(
        grid=grid,
        flow_estimator=FlowEstimator(downscale=args.downscale),
        detector=detector,
        fps=source.fps,
    )

    cropped = _CroppedSource(FileSource(args.video, max_frames=args.max_frames), w, h)
    with JsonlSink(args.out) as sink:
        for frame in pipeline.run(cropped):
            sink.write(frame)
    return 0


def _fps_from_timestamps(timestamps: List[Optional[float]]) -> Optional[float]:
    """Recover the frame rate from the JSONL `timestamp` column.

    The metrics file already encodes the true frame rate; ignoring it and
    assuming 25 fps reports FAR/h 2.5x wrong on a 10 fps clip. Uses the MEDIAN
    inter-frame delta so a single dropped or duplicated timestamp cannot skew
    it. Returns None when the timestamps cannot support an estimate, so the
    caller can fall back to the documented default.
    """
    import math

    values = [t for t in timestamps if isinstance(t, (int, float)) and math.isfinite(t)]
    if len(values) < 2:
        return None
    deltas = [b - a for a, b in zip(values, values[1:]) if b - a > 0]
    if not deltas:
        return None
    deltas.sort()
    mid = len(deltas) // 2
    median = deltas[mid] if len(deltas) % 2 else (deltas[mid - 1] + deltas[mid]) / 2.0
    if median <= 0:
        return None
    return 1.0 / median


def _cmd_eval(args: argparse.Namespace) -> int:
    import json
    import math
    import numpy as np
    from .eval import evaluate, load_labels

    scores = []
    timestamps: List[Optional[float]] = []
    with open(args.metrics, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            value = row.get(args.score_field)
            scores.append(float("nan") if value is None else float(value))
            ts = row.get("timestamp")
            timestamps.append(None if ts is None else float(ts))

    fps = args.fps
    if fps is None:
        fps = _fps_from_timestamps(timestamps)
        if fps is None:
            fps = DEFAULT_EVAL_FPS
            print(f"warning: no usable timestamps in {args.metrics}; "
                  f"assuming --fps {fps}", file=sys.stderr)

    labels = load_labels(args.labels)
    n = min(len(scores), len(labels))
    result = evaluate(np.array(scores[:n]), labels[:n], args.threshold, fps)
    result["fps"] = float(fps)

    def _jsonable(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    print(json.dumps({k: _jsonable(v) for k, v in result.items()}, indent=2))
    return 0


def _cmd_watch(args: argparse.Namespace) -> int:
    import json
    import math

    sinks = _build_alert_sinks(args)
    if sinks is None:
        return 2

    # Only --threshold-high is exposed; the rise/fall ratio is kept at the
    # Thresholds default (0.016/0.020 = 0.8) so a custom --threshold-high
    # still satisfies the hysteresis invariant. Both constructions validate
    # and raise ValueError -- e.g. --threshold-high 0.05 crosses the
    # (unexposed) critical tier. A traceback is not an error message: name
    # the offending flag and exit non-zero.
    try:
        machine = _build_machine(args)
    except ValueError as exc:
        print(f"error: invalid risk settings: {exc}", file=sys.stderr)
        return 2

    try:
        with open(args.metrics, encoding="utf-8") as fh:
            raw_lines = fh.readlines()
    except OSError as exc:
        # Mirrors _cmd_run: a missing or unopenable metrics file must name
        # the path and exit non-zero, not raise a bare traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2

    rows = []
    for lineno, line in enumerate(raw_lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(
                f"error: {args.metrics}: invalid JSON on line {lineno}: {exc}",
                file=sys.stderr,
            )
            return 2

    # A missing `timestamp` must never become 0.0 -- every row would then
    # collide at t=0, dwell would never complete, and a CRITICAL-pressure
    # file would replay as a silent, alert-free "calm crowd" (exit 0, no
    # output). Synthesise from --fps and row index instead, but only ever
    # loudly: this is a per-row time base for a safety alert, not a cosmetic
    # default.
    raw_timestamps = [row.get("timestamp") for row in rows]
    n_missing = sum(1 for ts in raw_timestamps if ts is None)
    synth_fps = None
    if n_missing:
        synth_fps = args.fps
        if synth_fps is None:
            synth_fps = _fps_from_timestamps(raw_timestamps)
        if synth_fps is None:
            synth_fps = DEFAULT_EVAL_FPS
        # Zero divides; negative or non-finite is WORSE than a crash -- it
        # synthesises decreasing (or NaN) timestamps, which the state machine
        # reads as a clock reset on every row, so the file replays as a
        # silent, alert-free "calm crowd". Refuse it.
        if not math.isfinite(synth_fps) or synth_fps <= 0:
            print(
                f"error: --fps must be a finite frame rate greater than zero, "
                f"got {synth_fps!r}; it is used to synthesise timestamps for "
                f"the {n_missing} row(s) in {args.metrics} that have none",
                file=sys.stderr,
            )
            return 2
        print(
            f"warning: {n_missing} row(s) in {args.metrics} have no timestamp; "
            f"synthesising timestamps from --fps {synth_fps} and row index -- "
            f"alert timing will be WRONG if this does not match the true frame "
            f"rate. Pass --fps to set it explicitly.",
            file=sys.stderr,
        )

    for idx, row in enumerate(rows):
        # `json.loads` accepts the bare literals NaN/Infinity, so a
        # non-finite timestamp reaches here having survived the
        # JSONDecodeError handler above; the state machine then rejects it
        # (a NaN time defeats the dwell AND re-alert gates at once). Report
        # it like any other bad input, naming the row.
        try:
            value = row.get(args.score_field)
            pressure = float("nan") if value is None else float(value)
            ts = row.get("timestamp")
            timestamp = (idx / synth_fps) if ts is None else float(ts)
            coverage_raw = row.get("sensing_confidence")
            coverage = None if coverage_raw is None else float(coverage_raw)

            event = machine.update(timestamp, pressure, coverage=coverage)
        except (ValueError, TypeError) as exc:
            print(
                f"error: {args.metrics}: row {idx + 1}: {exc}",
                file=sys.stderr,
            )
            return 2
        if event is not None:
            for sink in sinks:
                sink.publish(event)

    return 0


def drive_risk(frames, machine, sinks, metrics_sink=None,
               score_field: str = "global_max_pressure") -> int:
    """Feed MetricsFrames through the risk machine into the alert sinks.

    Shared by `live` and (in spirit) `watch`: `watch` replays a JSONL file,
    `live` runs the same frames straight off the pipeline, and both must
    obey the same dwell/re-alert gating -- a live stream that pages more
    eagerly than a replay of the same numbers would be a silent divergence
    between what is tested and what runs.

    Every frame goes to `metrics_sink` (when given), alerting or not: the
    JSONL is the run's evidence, not just its alarms. Returns the number of
    frames processed.
    """
    count = 0
    for frame in frames:
        count += 1
        if metrics_sink is not None:
            metrics_sink.write(frame)
        pressure = getattr(frame, score_field)
        coverage = frame.sensing_confidence
        event = machine.update(frame.timestamp, pressure, coverage=coverage)
        if event is not None:
            for sink in sinks:
                sink.publish(event)
    return count


def _build_alert_sinks(args) -> Optional[List]:
    """Build the alert sinks, or None if the SNS settings are unusable.

    Returns None (after printing the error) rather than raising, so both
    `watch` and `live` refuse BEFORE opening any socket -- publishing to a
    topic that was never named is the mistake worth failing loudly on.
    """
    from .alerts.log import LogSink

    sinks = [LogSink()]
    if getattr(args, "sns", False):
        if not args.sns_topic_arn:
            print(
                "error: --sns given without a topic ARN; pass --sns-topic-arn "
                "(no default -- refusing to run with SNS enabled and no destination)",
                file=sys.stderr,
            )
            return None
        import boto3
        from .alerts.sns import SnsSink

        client = boto3.client("sns", region_name=args.sns_region)
        sinks.append(SnsSink(args.sns_topic_arn, client, dry_run=args.dry_run))
    return sinks


def _build_machine(args):
    """Construct the risk state machine from the shared CLI risk flags."""
    from .risk.levels import Thresholds
    from .risk.machine import RiskStateMachine

    thresholds = Thresholds(
        high_rise=args.threshold_high, high_fall=args.threshold_high * 0.8
    )
    return RiskStateMachine(
        thresholds,
        min_dwell_s=args.min_dwell,
        min_realert_s=args.min_realert,
        blind_alert_s=args.blind_alert,
        min_coverage=args.min_coverage,
    )


def _cmd_live(args: argparse.Namespace) -> int:
    """Live KVS WebRTC ingest -> detections -> pressure -> risk -> alerts."""
    import time

    sinks = _build_alert_sinks(args)
    if sinks is None:
        return 2
    try:
        machine = _build_machine(args)
    except ValueError as exc:
        print(f"error: invalid risk settings: {exc}", file=sys.stderr)
        return 2

    import boto3

    from .webrtc.signaling import KvsSignalingClient, ice_servers_to_rtc
    from .webrtc.source import WebRTCSource

    kinesisvideo = boto3.client("kinesisvideo", region_name=args.region)
    signaling = KvsSignalingClient(
        args.channel, args.region, kinesisvideo, role="VIEWER"
    )
    try:
        endpoints = signaling.endpoints()
    except LookupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # STUN first, then the channel's TURN servers. Without TURN this will not
    # traverse a symmetric NAT; GetIceServerConfig failing is a warning, not a
    # fatal error, because a same-LAN board still connects on host candidates.
    ice = [{"urls": [f"stun:stun.kinesisvideo.{args.region}.amazonaws.com:443"]}]
    try:
        signaling_client = boto3.client(
            "kinesis-video-signaling",
            endpoint_url=endpoints["HTTPS"],
            region_name=args.region,
        )
        ice.extend(ice_servers_to_rtc(signaling.ice_servers(signaling_client)))
    except Exception as exc:  # noqa: BLE001 - degraded, not fatal
        print(f"warning: GetIceServerConfig failed ({exc}); STUN only -- this "
              f"will not traverse a symmetric NAT", file=sys.stderr)

    # Detections come from the board's NPU by default: it already runs SCRFD
    # per frame and draws those boxes into the video we are about to decode.
    # Running a host model over that same video would burn CPU to produce a
    # SECOND, disagreeing set of boxes. --host-model forces the old behaviour.
    from .detect import BoardDetector

    if args.no_detector:
        detector = NullDetector()
        board_detector = None
    elif args.host_model:
        detector = YoloDetector(args.model, conf=args.conf)
        board_detector = None
    else:
        board_detector = BoardDetector(max_age_s=args.metadata_max_age)
        detector = board_detector

    source = WebRTCSource(
        signaling, warmup_frames=args.warmup_frames,
        connect_timeout_s=args.connect_timeout,
        detector=board_detector,
    )
    try:
        source.connect(ice_servers=ice)
    except (RuntimeError, TimeoutError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        frames = iter(source)
        try:
            first = next(frames)
        except TimeoutError as exc:
            # connect() returning proves only that the track object exists;
            # a master that negotiates and then sends no RTP surfaces HERE.
            print(f"error: connected to {args.channel!r} but no video frame "
                  f"arrived ({exc}). The signalling/ICE/DTLS path is up, so "
                  f"this is the far end not sending media.", file=sys.stderr)
            return 3

        # fps feeds pressure as fps**2, so wait for a real measurement rather
        # than assuming a default.
        deadline = time.monotonic() + args.fps_timeout
        while True:
            try:
                fps = source.fps
                break
            except RuntimeError:
                if time.monotonic() > deadline:
                    print(f"error: frame rate never settled within "
                          f"{args.fps_timeout}s", file=sys.stderr)
                    return 3
                time.sleep(0.1)

        h, w = first.image.shape[:2]
        cell = args.cell_size
        w -= w % cell
        h -= h % cell
        grid = CellGrid(frame_width=w, frame_height=h, cell_size=cell)
        pipeline = Pipeline(
            grid=grid,
            flow_estimator=FlowEstimator(downscale=args.downscale),
            detector=detector,
            fps=fps,
        )

        import itertools

        bounded = _BoundedSource(
            itertools.chain([first], frames), fps, w, h,
            args.max_frames, args.max_seconds,
        )
        print(f"live: {args.channel} {w}x{h} @ {fps:.2f} fps", file=sys.stderr)

        metrics_sink = JsonlSink(args.out) if args.out else None
        try:
            n = drive_risk(pipeline.run(bounded), machine, sinks,
                           metrics_sink=metrics_sink,
                           score_field=args.score_field)
        finally:
            if metrics_sink is not None:
                metrics_sink.close()
        print(f"live: processed {n} frames, dropped {source.dropped}",
              file=sys.stderr)
        return 0
    finally:
        source.close()


class _BoundedSource:
    """Crops to the grid and stops after max_frames / max_seconds.

    A live stream has no end; without a bound the caller can never return.
    """

    def __init__(self, frames, fps, width, height, max_frames, max_seconds):
        self._frames = frames
        self.fps = fps
        self._w, self._h = width, height
        self._max_frames = max_frames
        self._max_seconds = max_seconds

    def __iter__(self):
        import time

        stop_at = (None if self._max_seconds is None
                   else time.monotonic() + self._max_seconds)
        for i, frame in enumerate(self._frames):
            if self._max_frames is not None and i >= self._max_frames:
                return
            if stop_at is not None and time.monotonic() > stop_at:
                return
            yield type(frame)(frame.index, frame.timestamp,
                              frame.image[: self._h, : self._w])


def _cmd_occlusion(args: argparse.Namespace) -> int:
    import csv
    import json
    from .occlusion import DetectionRatePoint, detection_rate_curve, is_monotonic_decreasing

    points = []
    with open(args.counts, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            points.append(DetectionRatePoint.from_counts(
                true_count=int(row["true_count"]),
                detected_count=int(row["detected_count"]),
                density_proxy=float(row.get("density_proxy", row["true_count"])),
            ))
    curve = detection_rate_curve(points, n_bins=args.bins)
    curve.to_csv(args.out, index=False)
    print(curve.to_string(index=False))
    print(json.dumps({"monotonic_decreasing": is_monotonic_decreasing(curve)}, indent=2))
    return 0


class _CroppedSource:
    """Crops frames so the grid divides evenly."""

    def __init__(self, inner, width: int, height: int) -> None:
        self._inner = inner
        self._w, self._h = width, height
        self.fps = inner.fps

    def __iter__(self):
        for frame in self._inner:
            yield type(frame)(frame.index, frame.timestamp, frame.image[: self._h, : self._w])


def _add_risk_arguments(parser: argparse.ArgumentParser) -> None:
    """Risk/alert flags shared by `watch` and `live`.

    Defined once so a live run and a replay of the same numbers cannot drift
    apart in their gating defaults.
    """
    parser.add_argument("--threshold-high", type=float, default=0.02,
                        help="high-tier rise threshold in s^-2 (fall is scaled to "
                             "keep the default 0.8 rise/fall ratio)")
    parser.add_argument("--min-dwell", type=float, default=2.0,
                        help="seconds a candidate level must persist before it is "
                             "adopted (sensor-blind uses --blind-alert instead)")
    parser.add_argument("--blind-alert", type=float, default=30.0,
                        help="seconds a sensor-blind (NaN pressure) candidate must "
                             "persist before it is adopted; kept separate from "
                             "--min-dwell because a false blind-alert pages someone")
    parser.add_argument("--min-realert", type=float, default=60.0,
                        help="seconds to suppress a repeat de-escalation alert for "
                             "the same level; only an escalation strictly above the "
                             "last alerted level bypasses this")
    parser.add_argument("--min-coverage", type=float, default=0.0,
                        help="minimum `sensing_confidence` fraction (0.0-1.0) for a "
                             "frame to be scored at all; below it the frame is "
                             "treated as unscorable (UNKNOWN) rather than calm")
    parser.add_argument("--sns", action="store_true",
                        help="also publish alerts to AWS SNS (requires "
                             "--sns-topic-arn)")
    parser.add_argument("--sns-topic-arn", default=None)
    parser.add_argument("--sns-region", default=None)
    parser.add_argument("--dry-run", dest="dry_run",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="log what would be published without calling SNS; "
                             "defaults to True whenever --sns is given -- pass "
                             "--no-dry-run to actually publish")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sfcpi")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="compute metrics for a video file")
    run.add_argument("video")
    run.add_argument("--out", required=True)
    run.add_argument("--cell-size", type=int, default=64)
    run.add_argument("--downscale", type=float, default=1.0)
    run.add_argument("--max-frames", type=int, default=None)
    run.add_argument("--model", default="yolov8n.pt")
    run.add_argument("--conf", type=float, default=0.35)
    run.add_argument("--no-detector", action="store_true",
                     help="flow-only run: the count is UNKNOWN, so counts, "
                          "density and pressure are NaN (null in the JSONL) "
                          "and sensing_confidence is 0.0. Speed metrics stay "
                          "finite. Counts are never reported as zero.")
    run.set_defaults(func=_cmd_run)

    ev = sub.add_parser("eval", help="score a metrics file against frame labels")
    ev.add_argument("metrics")
    ev.add_argument("--labels", required=True)
    ev.add_argument("--threshold", type=float, default=0.02)
    ev.add_argument("--fps", type=float, default=None,
                    help=f"frame rate for the FAR/h denominator; default is "
                         f"derived from the metrics file's timestamps, falling "
                         f"back to {DEFAULT_EVAL_FPS}")
    ev.add_argument("--score-field", default="global_max_pressure")
    ev.set_defaults(func=_cmd_eval)

    oc = sub.add_parser("occlusion", help="detector degradation vs density")
    oc.add_argument("counts", help="CSV with true_count,detected_count[,density_proxy]")
    oc.add_argument("--out", required=True)
    oc.add_argument("--bins", type=int, default=10)
    oc.set_defaults(func=_cmd_occlusion)

    watch = sub.add_parser(
        "watch", help="replay a metrics file through the risk state machine into alert sinks"
    )
    watch.add_argument("metrics")
    watch.add_argument("--score-field", default="global_max_pressure")
    watch.add_argument("--fps", type=float, default=None,
                       help="frame rate used to synthesise a timestamp "
                            "(index / fps) for rows with no `timestamp` "
                            "field; default is derived from the metrics "
                            f"file's other timestamps, falling back to "
                            f"{DEFAULT_EVAL_FPS}. A missing timestamp "
                            "always prints a warning; it is never "
                            "silently treated as t=0.")
    _add_risk_arguments(watch)
    watch.set_defaults(func=_cmd_watch)

    live = sub.add_parser(
        "live",
        help="ingest a live KVS WebRTC stream and run risk assessment on it",
    )
    live.add_argument("--channel", default=DEFAULT_CHANNEL,
                      help=f"KVS signalling channel name (default {DEFAULT_CHANNEL!r}, "
                           f"or $KVS_CHANNEL_NAME)")
    live.add_argument("--region", default=DEFAULT_REGION,
                      help=f"AWS region (default {DEFAULT_REGION!r}, or $AWS_REGION / "
                           f"$AWS_DEFAULT_REGION)")
    live.add_argument("--out", default=None,
                      help="optional JSONL metrics file; every frame is written, "
                           "not only alerting ones")
    live.add_argument("--connect-timeout", type=float, default=30.0)
    live.add_argument("--warmup-frames", type=int, default=10,
                      help="frames used to measure fps before the pipeline starts; "
                           "fps feeds pressure as fps**2, so it is measured, never "
                           "assumed")
    live.add_argument("--fps-timeout", type=float, default=20.0)
    live.add_argument("--max-frames", type=int, default=None,
                      help="stop after N frames (a live stream has no end)")
    live.add_argument("--max-seconds", type=float, default=None)
    live.add_argument("--cell-size", type=int, default=64)
    live.add_argument("--downscale", type=float, default=1.0)
    live.add_argument("--host-model", action="store_true",
                      help="run a host-side YOLO model instead of using the "
                           "board's own SCRFD detections. Off by default: the "
                           "board already detects on its NPU and draws those "
                           "boxes into the video, so a host model produces a "
                           "second, disagreeing set")
    live.add_argument("--metadata-max-age", type=float, default=2.0,
                      help="seconds before board detection metadata is treated "
                           "as stale; a stale count is UNKNOWN, never the last "
                           "one seen")
    live.add_argument("--model", default="src/dashboard/best.pt",
                      help="host-side weights, used only with --host-model")
    live.add_argument("--conf", type=float, default=0.35)
    live.add_argument("--no-detector", action="store_true",
                      help="flow-only: counts, density and pressure are NaN and "
                           "sensing_confidence is 0.0; counts are never reported "
                           "as zero")
    live.add_argument("--score-field", default="global_max_pressure")
    _add_risk_arguments(live)
    live.set_defaults(func=_cmd_live)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
