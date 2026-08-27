"""The alert payload. Pure data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .levels import RiskLevel


@dataclass(frozen=True)
class RiskEvent:
    timestamp: float
    level: RiskLevel
    previous_level: RiskLevel
    pressure: Optional[float]
    coverage: Optional[float]
    reason: str          # escalation | de-escalation | sustained | sensor-blind
    message: str
