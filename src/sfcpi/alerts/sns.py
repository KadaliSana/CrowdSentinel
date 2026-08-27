"""AWS SNS delivery.

The boto3 client is INJECTED, never constructed here -- that is what lets every
test run without AWS credentials or network.
"""
from __future__ import annotations

import logging
from typing import Any

from ..risk.events import RiskEvent

logger = logging.getLogger(__name__)

_SUBJECT_MAX = 100  # SNS hard limit


class SnsSink:
    def __init__(self, topic_arn: str, client: Any, dry_run: bool = False) -> None:
        if not topic_arn:
            raise ValueError("topic_arn is required; refusing to start with SNS enabled and no topic")
        if client is None:
            raise ValueError("client is required; construct boto3.client('sns') at the call site")
        self.topic_arn = topic_arn
        self.client = client
        self.dry_run = dry_run
        self.count = 0
        self.failures = 0

    def _subject(self, event: RiskEvent) -> str:
        return f"[CrowdSentinel] {event.level.value.upper()}"[:_SUBJECT_MAX]

    def _body(self, event: RiskEvent) -> str:
        pressure = "n/a" if event.pressure is None else f"{event.pressure:.4f} s^-2"
        coverage = "n/a" if event.coverage is None else f"{event.coverage:.0%}"
        return (
            f"Risk level: {event.previous_level.value} -> {event.level.value}\n"
            f"Reason:     {event.reason}\n"
            f"Pressure:   {pressure}\n"
            f"Coverage:   {coverage}\n"
            f"Timestamp:  {event.timestamp:.2f}\n\n{event.message}\n"
        )

    def publish(self, event: RiskEvent) -> None:
        if self.dry_run:
            logger.info("[dry-run] would publish: %s", self._subject(event))
            self.count += 1
            return
        try:
            self.client.publish(
                TopicArn=self.topic_arn,
                Subject=self._subject(event),
                Message=self._body(event),
            )
        except Exception as exc:                      # noqa: BLE001 - never kill the monitor
            self.failures += 1
            logger.error("SNS publish failed (%s): %s", type(exc).__name__, exc)
            return
        self.count += 1
