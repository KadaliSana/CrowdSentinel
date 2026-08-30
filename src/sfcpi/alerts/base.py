"""Delivery is a sink LIST, not a branch -- so adding a transport touches no logic."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..risk.events import RiskEvent


@runtime_checkable
class AlertSink(Protocol):
    def publish(self, event: RiskEvent) -> None:  # pragma: no cover - protocol
        ...
