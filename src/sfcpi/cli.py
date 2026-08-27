"""Command line entry points."""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .detect import NullDetector, YoloDetector
from .flow import FlowEstimator
from .grid import CellGrid
from .pipeline import Pipeline
from .sinks import JsonlSink
from .sources import DEFAULT_FPS as DEFAULT_EVAL_FPS, FileSource


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
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
