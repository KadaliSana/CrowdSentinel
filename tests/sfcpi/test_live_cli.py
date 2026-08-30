"""`sfcpi live`: KVS frames -> detections -> pressure -> risk -> alerts.

The live source itself needs a board streaming, but everything downstream of
it is ordinary code and is tested here without network access.
"""
import pytest

from sfcpi.cli import build_parser, drive_risk
from sfcpi.pipeline import MetricsFrame
from sfcpi.risk.levels import Thresholds
from sfcpi.risk.machine import RiskStateMachine
from sfcpi.webrtc.signaling import ice_servers_to_rtc


# -- ICE server conversion ---------------------------------------------------

def test_ice_servers_to_rtc_renames_kvs_fields_to_aiortc_names():
    """KVS says Uris/Username/Password; aiortc wants urls/username/credential."""
    out = ice_servers_to_rtc([
        {"Uris": ["turn:a:443?transport=udp", "turns:a:443"],
         "Username": "u", "Password": "p", "Ttl": 300},
    ])
    assert out == [{"urls": ["turn:a:443?transport=udp", "turns:a:443"],
                    "username": "u", "credential": "p"}]


def test_ice_servers_to_rtc_omits_absent_credentials():
    """A STUN-only entry has no username/password; passing None would make
    aiortc treat it as a TURN server with empty credentials."""
    assert ice_servers_to_rtc([{"Uris": ["stun:a:443"]}]) == [{"urls": ["stun:a:443"]}]


def test_ice_servers_to_rtc_handles_an_empty_list():
    assert ice_servers_to_rtc([]) == []


# -- the frames -> risk loop -------------------------------------------------

class _Sink:
    def __init__(self): self.events = []
    def publish(self, event): self.events.append(event)


def _frames(n, pressure, fps=25.0):
    for i in range(n):
        yield MetricsFrame(index=i, timestamp=i / fps, flow_valid=True,
                           global_pressure=pressure, global_max_pressure=pressure,
                           total_count=40.0, sensing_confidence=1.0)


def _machine():
    return RiskStateMachine(Thresholds(), min_dwell_s=2.0, min_realert_s=60.0)


def test_drive_risk_emits_one_alert_for_a_sustained_critical_run():
    """Same anti-spam invariant `watch` has: a long CRITICAL run pages once."""
    sink = _Sink()
    n = drive_risk(_frames(200, 0.05), _machine(), [sink])
    assert n == 200
    assert len(sink.events) == 1, [e.level for e in sink.events]
    assert "critical" in str(sink.events[0].level).lower()


def test_drive_risk_stays_silent_on_a_calm_stream():
    sink = _Sink()
    drive_risk(_frames(200, 0.0001), _machine(), [sink])
    assert sink.events == []


def test_drive_risk_writes_every_frame_to_the_metrics_sink(tmp_path):
    """The JSONL artefact must record all frames, not only alerting ones."""
    out = tmp_path / "m.jsonl"
    from sfcpi.sinks import JsonlSink
    with JsonlSink(str(out)) as js:
        drive_risk(_frames(10, 0.0001), _machine(), [], metrics_sink=js)
    assert len(out.read_text().strip().splitlines()) == 10


# -- CLI surface -------------------------------------------------------------

def test_live_rejects_sns_without_a_topic_arn(capsys):
    """Must fail before opening any socket -- same rule `watch` enforces."""
    from sfcpi.cli import main
    rc = main(["live", "--channel", "camstream", "--region", "ap-south-1", "--sns"])
    assert rc == 2
    assert "sns-topic-arn" in capsys.readouterr().err


def test_live_is_a_registered_subcommand():
    args = build_parser().parse_args(["live", "--channel", "c", "--region", "r"])
    assert args.channel == "c" and args.region == "r"


# -- channel/region defaults -------------------------------------------------

def test_live_defaults_to_the_deployed_channel_and_region():
    """This deployment has exactly one board on one channel, so requiring the
    flags on every invocation was pure friction."""
    args = build_parser().parse_args(["live"])
    assert args.channel == "camstream"
    assert args.region == "ap-south-1"


def test_live_flags_still_override_the_defaults():
    args = build_parser().parse_args(
        ["live", "--channel", "other", "--region", "eu-west-1"])
    assert args.channel == "other" and args.region == "eu-west-1"


def test_live_defaults_follow_the_environment_when_set(monkeypatch):
    """Same env convention the dashboard uses (KVS_CHANNEL_NAME / AWS_REGION),
    so the two entry points cannot point at different boards."""
    monkeypatch.setenv("KVS_CHANNEL_NAME", "envchan")
    monkeypatch.setenv("AWS_REGION", "us-east-2")
    import importlib

    import sfcpi.cli as cli
    importlib.reload(cli)
    try:
        args = cli.build_parser().parse_args(["live"])
        assert args.channel == "envchan" and args.region == "us-east-2"
    finally:
        monkeypatch.undo()
        importlib.reload(cli)
