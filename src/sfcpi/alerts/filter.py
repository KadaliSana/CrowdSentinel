"""Severity filtering in front of an alert transport.

Every event still reaches the log sink; this decides what is worth spending
someone's attention on. It wraps a sink rather than living inside one, so the
same filter works for SNS, SMS or anything added later.
"""
from __future__ import annotations

from typing import Any

from ..risk.events import RiskEvent
from ..risk.levels import RiskLevel, severity


def level_from_name(name: str) -> RiskLevel:
    """Parse a configuration string into a RiskLevel, case-insensitively."""
    try:
        return RiskLevel(str(name).strip().lower())
    except ValueError as exc:
        valid = ", ".join(level.value for level in RiskLevel)
        raise ValueError(
            f"unknown risk level {name!r}; expected one of: {valid}"
        ) from exc


class LevelFilterSink:
    """Forward to `inner` only events at or above `min_level`.

    UNKNOWN is handled separately and explicitly. `severity()` returns None
    for it by design -- it is the absence of a measurement, not a rung on the
    ladder -- so it can never be compared against a threshold. Left implicit,
    a comparison against None would either raise or silently pass everything.

    The default drops it, which is what "only page me for CRITICAL" means.
    Be aware of the cost: a sensor-blind board then pages NOBODY. "I cannot
    see the crowd" is arguably as urgent as "the crowd is critical", and
    `include_unknown=True` is how you say so.
    """

    def __init__(self, inner: Any, min_level: RiskLevel,
                 include_unknown: bool = False) -> None:
        if severity(min_level) is None:
            raise ValueError(
                f"min_level must be an ordered level; {min_level.name} has no "
                f"severity (it means 'no measurement', not a threshold)"
            )
        self.inner = inner
        self.min_level = min_level
        self.include_unknown = include_unknown
        self._min_severity = severity(min_level)

    def publish(self, event: RiskEvent) -> None:
        level_severity = severity(event.level)
        if level_severity is None:
            # UNKNOWN / sensor-blind.
            if self.include_unknown:
                self.inner.publish(event)
            return
        if level_severity >= self._min_severity:
            # Deliberately NOT wrapped in try/except: this class decides what
            # to deliver, not whether delivery succeeded. Swallowing here
            # would hide a dead transport behind a filter that looks healthy.
            self.inner.publish(event)
