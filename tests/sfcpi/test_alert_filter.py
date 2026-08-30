"""Severity filtering for alert delivery.

Paging is expensive attention. This lets a transport carry only what is worth
waking someone for, while the log sink keeps everything.
"""
import pytest

from sfcpi.alerts.filter import LevelFilterSink
from sfcpi.risk.events import RiskEvent
from sfcpi.risk.levels import RiskLevel


class _Recorder:
    def __init__(self): self.events = []
    def publish(self, event): self.events.append(event)


def _event(level):
    return RiskEvent(timestamp=1.0, level=level, previous_level=RiskLevel.NORMAL,
                     pressure=0.05, coverage=1.0, reason="escalation",
                     message=f"-> {level.value}")


def test_critical_passes_the_critical_filter():
    inner = _Recorder()
    LevelFilterSink(inner, RiskLevel.CRITICAL).publish(_event(RiskLevel.CRITICAL))
    assert len(inner.events) == 1


@pytest.mark.parametrize("level", [RiskLevel.NORMAL, RiskLevel.ELEVATED, RiskLevel.HIGH])
def test_lower_levels_are_dropped(level):
    inner = _Recorder()
    LevelFilterSink(inner, RiskLevel.CRITICAL).publish(_event(level))
    assert inner.events == []


def test_unknown_is_dropped_by_default():
    """UNKNOWN has no severity ordinal -- `severity()` returns None -- so it
    cannot be compared against a threshold. It is dropped explicitly rather
    than by an accidental comparison, and the caller opts in to receive it.

    SAFETY NOTE: this means a sensor-blind board raises NOTHING on a filtered
    transport. "I cannot see the crowd" is arguably as urgent as "the crowd is
    critical"; include_unknown=True is how you say so.
    """
    inner = _Recorder()
    LevelFilterSink(inner, RiskLevel.CRITICAL).publish(_event(RiskLevel.UNKNOWN))
    assert inner.events == []


def test_unknown_passes_when_explicitly_included():
    inner = _Recorder()
    sink = LevelFilterSink(inner, RiskLevel.CRITICAL, include_unknown=True)
    sink.publish(_event(RiskLevel.UNKNOWN))
    assert len(inner.events) == 1


def test_a_normal_threshold_passes_every_ordered_level():
    inner = _Recorder()
    sink = LevelFilterSink(inner, RiskLevel.NORMAL)
    for level in (RiskLevel.NORMAL, RiskLevel.ELEVATED, RiskLevel.HIGH,
                  RiskLevel.CRITICAL):
        sink.publish(_event(level))
    assert len(inner.events) == 4


def test_an_unordered_threshold_is_rejected_at_construction():
    """UNKNOWN is not a severity, so it cannot be a minimum severity."""
    with pytest.raises(ValueError, match="UNKNOWN"):
        LevelFilterSink(_Recorder(), RiskLevel.UNKNOWN)


def test_a_failing_inner_sink_propagates():
    """The filter decides WHAT to deliver, never whether delivery failed --
    swallowing here would hide a broken transport behind a working filter."""
    class _Broken:
        def publish(self, event): raise RuntimeError("sns down")

    with pytest.raises(RuntimeError, match="sns down"):
        LevelFilterSink(_Broken(), RiskLevel.CRITICAL).publish(_event(RiskLevel.CRITICAL))


def test_level_from_name_accepts_config_strings():
    from sfcpi.alerts.filter import level_from_name

    assert level_from_name("critical") is RiskLevel.CRITICAL
    assert level_from_name("CRITICAL") is RiskLevel.CRITICAL
    with pytest.raises(ValueError, match="nonsense"):
        level_from_name("nonsense")
