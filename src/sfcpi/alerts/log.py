"""Sinks that need no external service."""
from __future__ import annotations

import sys
from typing import List, Optional, TextIO

from ..risk.events import RiskEvent


class LogSink:
    """Always-available sink. The default, and the fallback when SNS is off."""

    def __init__(self, stream: Optional[TextIO] = None) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self.count = 0

    def publish(self, event: RiskEvent) -> None:
        pressure = "n/a" if event.pressure is None else f"{event.pressure:.4f}"
        # Coverage is mandatory on every event, not optional detail: the
        # crowd sensor is a face detector whose count is biased downward
        # exactly when crowds densify, so an alert that cannot say how much
        # of the frame was actually sensed is not actionable.
        coverage = "n/a" if event.coverage is None else f"{event.coverage * 100:.1f}%"
        self._stream.write(
            f"[ALERT {event.level.value.upper()}] t={event.timestamp:.2f} "
            f"pressure={pressure} coverage={coverage} reason={event.reason} "
            f":: {event.message}\n"
        )
        self._stream.flush()
        self.count += 1


class RecordingSink:
    """Captures events in memory. For tests and for a future dashboard feed."""

    def __init__(self) -> None:
        self.events: List[RiskEvent] = []

    def publish(self, event: RiskEvent) -> None:
        self.events.append(event)
