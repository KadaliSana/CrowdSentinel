"""Metrics output. NaN serialises as null -- never as 0."""
from __future__ import annotations

import json
import math
from typing import Any, Dict

import numpy as np

from .pipeline import MetricsFrame


def _clean(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def frame_to_row(frame: MetricsFrame) -> Dict[str, Any]:
    pressure = frame.cells.get("pressure")
    return {
        "index": frame.index,
        "timestamp": frame.timestamp,
        "flow_valid": frame.flow_valid,
        "global_pressure": _clean(frame.global_pressure),
        "global_max_pressure": _clean(frame.global_max_pressure),
        "total_count": _clean(frame.total_count),
        "sensing_confidence": frame.sensing_confidence,
        "cell_pressure": [] if pressure is None else
                         [[_clean(float(v)) for v in row] for row in np.asarray(pressure)],
    }


class JsonlSink:
    def __init__(self, path: str) -> None:
        self._fh = open(path, "w", encoding="utf-8")

    def write(self, frame: MetricsFrame) -> None:
        self._fh.write(json.dumps(frame_to_row(frame)) + "\n")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlSink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
