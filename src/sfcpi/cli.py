"""Command line entry points."""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .detect import FixedDetector, YoloDetector
from .flow import FlowEstimator
from .grid import CellGrid
from .pipeline import Pipeline
from .sinks import JsonlSink
from .sources import FileSource


def _cmd_run(args: argparse.Namespace) -> int:
    # FileSource probes fps at construction time (not first iteration), so a
    # missing file raises FileNotFoundError from the constructor itself --
    # the constructor must be inside the try, not just the first next().
    try:
        source = FileSource(args.video, max_frames=args.max_frames)
        first = next(iter(source))
    except FileNotFoundError as exc:
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

    detector = FixedDetector([]) if args.no_detector else YoloDetector(args.model, conf=args.conf)

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


def _cmd_eval(args: argparse.Namespace) -> int:
    import json
    import numpy as np
    from .eval import evaluate, load_labels

    scores = []
    with open(args.metrics, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            value = row.get(args.score_field)
            scores.append(float("nan") if value is None else float(value))
    labels = load_labels(args.labels)
    n = min(len(scores), len(labels))
    result = evaluate(np.array(scores[:n]), labels[:n], args.threshold, args.fps)
    print(json.dumps(result, indent=2))
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
            self.fps = self._inner.fps
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
                     help="flow-only run; counts are zero and pressure is zero")
    run.set_defaults(func=_cmd_run)

    ev = sub.add_parser("eval", help="score a metrics file against frame labels")
    ev.add_argument("metrics")
    ev.add_argument("--labels", required=True)
    ev.add_argument("--threshold", type=float, default=0.02)
    ev.add_argument("--fps", type=float, default=25.0)
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
