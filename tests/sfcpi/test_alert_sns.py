import pytest
from sfcpi.alerts.sns import SnsSink
from sfcpi.risk.events import RiskEvent
from sfcpi.risk.levels import RiskLevel

ARN = "arn:aws:sns:ap-south-1:123456789012:crowd-alerts"

class _StubClient:
    def __init__(self, raises=None):
        self.calls = []
        self._raises = raises
    def publish(self, **kwargs):
        if self._raises:
            raise self._raises
        self.calls.append(kwargs)
        return {"MessageId": "stub-1"}

def _ev(level=RiskLevel.HIGH):
    return RiskEvent(timestamp=1.5, level=level, previous_level=RiskLevel.NORMAL,
                     pressure=0.025, coverage=0.9, reason="escalation", message="normal -> high")

def test_publishes_once_per_event():
    c = _StubClient()
    s = SnsSink(topic_arn=ARN, client=c)
    s.publish(_ev())
    assert len(c.calls) == 1
    assert c.calls[0]["TopicArn"] == ARN
    assert s.count == 1 and s.failures == 0

def test_subject_and_body_name_the_level_and_pressure():
    c = _StubClient()
    SnsSink(topic_arn=ARN, client=c).publish(_ev(RiskLevel.CRITICAL))
    call = c.calls[0]
    assert "CRITICAL" in call["Subject"]
    assert "0.025" in call["Message"]
    assert len(call["Subject"]) <= 100      # SNS hard limit

def test_publish_failure_is_counted_not_raised():
    """A dropped alert is bad; a crashed monitor is worse."""
    s = SnsSink(topic_arn=ARN, client=_StubClient(raises=RuntimeError("denied")))
    s.publish(_ev())                          # must not raise
    assert s.count == 0 and s.failures == 1

def test_dry_run_publishes_nothing_but_counts():
    c = _StubClient()
    s = SnsSink(topic_arn=ARN, client=c, dry_run=True)
    s.publish(_ev())
    assert c.calls == [] and s.count == 1

def test_missing_topic_arn_fails_at_construction():
    with pytest.raises(ValueError, match="topic_arn"):
        SnsSink(topic_arn="", client=_StubClient())

def test_unknown_level_event_still_publishes():
    c = _StubClient()
    ev = RiskEvent(timestamp=2.0, level=RiskLevel.UNKNOWN, previous_level=RiskLevel.NORMAL,
                   pressure=None, coverage=0.0, reason="sensor-blind", message="sensor blind")
    SnsSink(topic_arn=ARN, client=c).publish(ev)
    assert len(c.calls) == 1 and "n/a" in c.calls[0]["Message"]
