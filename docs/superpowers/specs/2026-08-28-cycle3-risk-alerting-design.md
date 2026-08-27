# Cycle 3 — Risk Model and Alerting (Design)

Date: 2026-08-28
Status: design proposal
Depends on: cycle 1 (`src/sfcpi/`), spec `2026-08-27-sf-cpi-design.md`

## 1. Purpose

Turn the per-frame metric stream into a small number of trustworthy ALERTS, and
deliver them out-of-band via AWS SNS so they arrive when nobody is watching a screen.

The detection work is cycle 1. This cycle answers only: **when do we fire, and who hears it.**

## 2. The problem this actually solves

A raw threshold on a noisy signal flaps. At 15 fps a signal hovering near the line
produces tens of alerts per second. With a dashboard banner that is annoying; with SNS
it reaches real phones and can get the subscription rate-limited. So the state machine —
not the transport — is the substance of this cycle.

Three mechanisms, all required:
- **Hysteresis**: separate rise and fall thresholds. Enter ELEVATED at `rise`, leave only
  below `fall`, with `fall < rise`. A single threshold cannot avoid chatter.
- **Dwell**: the condition must hold for `min_dwell_s` before the level changes. Kills
  single-frame spikes from compression artefacts.
- **Re-alert interval**: once fired, do not re-fire for `min_realert_s` unless the level
  ESCALATES. Escalation always alerts immediately.

## 3. Risk levels

`NORMAL -> ELEVATED -> HIGH -> CRITICAL`, plus `UNKNOWN`.

`UNKNOWN` is not a severity, it is the absence of a measurement, and it is
load-bearing: cycle 1 emits NaN pressure when sensing fails. A NaN must NOT map to
NORMAL — that is the safety-inverted failure this project exists to avoid. Sustained
UNKNOWN is itself alertable (`sensor blind for N seconds`), because a blind sensor at a
mass gathering is an incident.

Thresholds are in s^-2 and configurable. The literature reference is `P > 0.02 s^-2`
(turbulence onset, ~10 min before the Jamarat crush) but it is R-dependent, so it is a
DEFAULT, never a hardcoded constant.

## 4. Architecture

    risk/levels.py      RiskLevel enum; classify(pressure, thresholds) -> RiskLevel
    risk/machine.py     RiskStateMachine: metrics in -> RiskEvent out (hysteresis+dwell+realert)
    risk/events.py      RiskEvent dataclass
    alerts/base.py      AlertSink protocol
    alerts/log.py       LogSink (default; always available)
    alerts/sns.py       SnsSink (boto3; injected client)
    cli.py              `sfcpi watch` wires metrics -> machine -> sinks

**Key boundary:** `risk/` is pure — it consumes `MetricsFrame`-shaped data and timestamps
and emits `RiskEvent`. It performs no I/O and knows nothing about SNS. That is what makes
the alerting logic testable without AWS, and what lets `--dry-run` be honest.

Delivery is a SINK LIST, not a branch. `LogSink` and `SnsSink` are interchangeable; adding
a dashboard sink later touches no logic.

## 5. RiskEvent

Fields: `timestamp`, `level`, `previous_level`, `pressure`, `coverage`,
`reason` (one of `escalation`, `de-escalation`, `sustained`, `sensor-blind`), `message`.

`coverage` rides on every event deliberately: an alert that cannot say how much of the
frame was actually sensed is not actionable.

## 6. SNS

- Client is INJECTED, never constructed inside the sink — tests use a stub, no AWS.
- Topic ARN from config/env; refuse to start if unset rather than silently no-op.
- Publish failures must not kill the pipeline: catch, log, count. A dropped alert is bad;
  a crashed monitor is worse.
- `--dry-run` renders the exact message and does not publish. This is how thresholds get
  tuned without spamming a real topic, and it must be the DEFAULT until explicitly disabled.

Known external prerequisite (human): the current IAM key is Kinesis-only; `sns:Publish`
scoped to the topic ARN must be granted, and the subscription confirmed, before live use.

## 7. Error handling

- NaN/absent pressure -> `UNKNOWN`, never NORMAL.
- Sustained UNKNOWN beyond `blind_alert_s` -> a `sensor-blind` event at HIGH.
- Missing topic ARN with SNS enabled -> fail at startup, naming the missing setting.
- SNS publish exception -> logged and counted, pipeline continues.
- Clock going backwards (replay restart) -> reset timers rather than emit negative dwell.

## 8. Testing

State-machine tests are driven by SYNTHETIC (timestamp, pressure) sequences — no video,
no AWS, deterministic.

- flapping input around `rise` produces exactly ONE event, not N
- dwell suppresses a single-frame spike
- escalation bypasses the re-alert interval; sustained same-level does not
- NaN maps to UNKNOWN and never to NORMAL
- sustained UNKNOWN raises `sensor-blind`
- `SnsSink` publishes exactly once per event against a stub client, and survives a raising client
- `--dry-run` publishes nothing

## 9. Out of scope

Dashboard UI, WebRTC ingest (cycle 2), per-cell/zone alerting, alert acknowledgement.
