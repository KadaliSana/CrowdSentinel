"""Risk levels and hysteresis classification. Pure: no I/O."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RiskLevel(Enum):
    UNKNOWN = "unknown"
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


_SEVERITY = {
    RiskLevel.NORMAL: 0,
    RiskLevel.ELEVATED: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


def severity(level: RiskLevel) -> Optional[int]:
    """Ordinal severity, or None for UNKNOWN.

    UNKNOWN is deliberately unordered: it is the absence of a measurement,
    not a position on the scale.
    """
    return _SEVERITY.get(level)


@dataclass(frozen=True)
class Thresholds:
    """Rise/fall pairs in s^-2. Defaults reference Johansson/Helbing's
    turbulence onset at 0.02 s^-2 -- a DEFAULT, not a constant (it is
    R-dependent). Every fall must sit strictly below its rise, AND the tiers
    must be ordered across each other: a per-tier check alone lets an
    operator raise high_rise past critical_rise, after which a pressure
    below their own HIGH threshold classifies as CRITICAL."""

    elevated_rise: float = 0.010
    elevated_fall: float = 0.008
    high_rise: float = 0.020
    high_fall: float = 0.016
    critical_rise: float = 0.040
    critical_fall: float = 0.032

    def __post_init__(self) -> None:
        for name in ("elevated", "high", "critical"):
            rise = getattr(self, f"{name}_rise")
            fall = getattr(self, f"{name}_fall")
            if not fall < rise:
                raise ValueError(
                    f"{name}_fall ({fall}) must be strictly below {name}_rise ({rise}); "
                    "equal or inverted thresholds defeat hysteresis"
                )
        # Cross-tier ordering. classify() evaluates the tiers most-severe
        # first, so an out-of-order pair silently shadows a whole tier.
        for kind in ("rise", "fall"):
            for lower, upper in (("elevated", "high"), ("high", "critical")):
                a, b = f"{lower}_{kind}", f"{upper}_{kind}"
                va, vb = getattr(self, a), getattr(self, b)
                if not va < vb:
                    raise ValueError(
                        f"{a} ({va}) must be strictly below {b} ({vb}); "
                        "out-of-order tiers classify a pressure as more severe "
                        "than the threshold it has not reached"
                    )


def classify(
    pressure, thresholds: Thresholds, current: RiskLevel = RiskLevel.NORMAL
) -> RiskLevel:
    """Map a pressure to a level, holding `current` inside the hysteresis band.

    A non-finite or absent pressure is UNKNOWN -- never NORMAL.

    Each tier's rise/fall pair is evaluated independently: a tier is "at
    least" active either because the pressure crossed its rise threshold,
    or because the current level was already at or above that tier and the
    pressure has not yet dropped below its fall threshold. This must be
    checked per tier (not as a single ordered cascade of rise thresholds)
    so that e.g. a held HIGH level between high_fall and high_rise isn't
    masked by the unconditional elevated_rise check.

    UNKNOWN carries no level to hold, so it decays to NORMAL rather than
    inheriting a stale severity from before the sensor went blind.
    """
    if pressure is None or not math.isfinite(float(pressure)):
        return RiskLevel.UNKNOWN

    p = float(pressure)
    held = severity(current)
    if held is None:
        held = 0

    at_least_critical = p >= thresholds.critical_rise or (
        held >= 3 and p >= thresholds.critical_fall
    )
    at_least_high = p >= thresholds.high_rise or (
        held >= 2 and p >= thresholds.high_fall
    )
    at_least_elevated = p >= thresholds.elevated_rise or (
        held >= 1 and p >= thresholds.elevated_fall
    )

    if at_least_critical:
        return RiskLevel.CRITICAL
    if at_least_high:
        return RiskLevel.HIGH
    if at_least_elevated:
        return RiskLevel.ELEVATED
    return RiskLevel.NORMAL
