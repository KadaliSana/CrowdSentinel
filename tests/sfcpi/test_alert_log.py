import io
from sfcpi.alerts.log import LogSink, RecordingSink
from sfcpi.risk.events import RiskEvent
from sfcpi.risk.levels import RiskLevel

def _ev(level=RiskLevel.HIGH):
    return RiskEvent(timestamp=1.5, level=level, previous_level=RiskLevel.NORMAL,
                     pressure=0.025, coverage=0.9, reason="escalation", message="normal -> high")

def test_log_sink_writes_one_line_per_event():
    buf = io.StringIO()
    sink = LogSink(stream=buf)
    sink.publish(_ev()); sink.publish(_ev(RiskLevel.CRITICAL))
    assert len(buf.getvalue().strip().splitlines()) == 2
    assert sink.count == 2

def test_log_line_names_level_and_pressure():
    buf = io.StringIO()
    LogSink(stream=buf).publish(_ev())
    line = buf.getvalue()
    assert "high" in line.lower() and "0.025" in line

def test_recording_sink_collects_events():
    sink = RecordingSink()
    sink.publish(_ev())
    assert len(sink.events) == 1 and sink.events[0].level is RiskLevel.HIGH
