# sfcpi — Scale-Free Crowd Pressure Index

Computes crowd pressure `P = rho * Var(v)` from **uncalibrated** video.

`Var(v)` is the variance of the velocity VECTORS (`Var(vx) + Var(vy)`), not of
speed magnitudes — per Johansson et al. (2008), conclusions: "the density times
the variance of velocities". Magnitude variance is 0 for perfect counterflow,
which would read "safe" in exactly the regime where pressure peaks.

`P` has units of `s^-2` and carries no length dimension, so the unknown
metres-per-pixel scale cancels exactly. Pressure computed in pixel space equals
pressure in real units, given only the frame rate — see
`docs/superpowers/specs/2026-08-27-sf-cpi-design.md`.

## Install

The package lives under `src/` (src layout), so it is not importable from a
bare checkout. Install it once, from the repository root:

    python3 -m pip install -e .

## Usage

    python3 -m sfcpi.cli run clip.mp4 --out metrics.jsonl --cell-size 64
    python3 -m sfcpi.cli eval metrics.jsonl --labels labels.csv
    python3 -m sfcpi.cli occlusion counts.csv --out curve.csv
    python3 -m sfcpi.cli watch metrics.jsonl

`eval` derives the frame rate from the `timestamp` column of the metrics file;
pass `--fps` only to override it. It falls back to 25 fps, with a warning on
stderr, when the timestamps are unusable — evaluating a 10 fps clip at an
assumed 25 fps reports false alarms per hour 2.5x wrong.

`eval` reports `n_frames`, `n_unscorable` and `coverage` alongside the scores.
Read them first: a run whose detector was dead reports no false alarms simply
because it never fired, and only `coverage` distinguishes "quiet" from "blind".
False alarms are counted as contiguous alarm EPISODES, not as alarming frames —
one 4-second alarm at 25 fps is one event, not 100.

### Flow-only mode (`--no-detector`)

`--no-detector` runs the optical-flow half alone. The count is then **unknown,
not zero**: counts, density and pressure come out NaN (`null` in the JSONL) and
`sensing_confidence` is `0.0`. Speed and velocity-variance metrics stay finite,
because motion really was measured. This is the same fail-loud path the
pipeline takes when a real detector throws — there is exactly one degradation
semantics, not two.

### `watch` — replay metrics through the risk state machine

    python3 -m sfcpi.cli watch metrics.jsonl --threshold-high 0.02 --min-dwell 2.0 \
        --blind-alert 30.0 --min-coverage 0.0

Reads a `sfcpi run` JSONL metrics file line by line and feeds `--score-field`
(default `global_max_pressure`) through the hysteresis + dwell + re-alert
state machine (`sfcpi.risk`), printing `[ALERT ...]` lines to stderr via
`LogSink` as levels change. Each `LogSink` line also names `coverage` (the
row's `sensing_confidence`, as a percentage, `n/a` when absent) — the crowd
sensor is a face detector whose count is biased downward exactly when crowds
densify, so an alert that cannot say how much of the frame was actually
sensed is not actionable. A `null` score is read as NaN, never `0.0` — a
blind sensor must reach the state machine as UNKNOWN (raising a
`sensor-blind` alert after it dwells), not as a falsely-calm `NORMAL`
reading.

`--min-dwell` (default 2.0s) gates ordinary level changes; sensor-blind uses
its own `--blind-alert` (default 30.0s, matching the library default) so a
brief camera glitch does not page someone at the same eagerness as a real
crowd-pressure escalation.

`--min-coverage` (default `0.0`, disabled) sets the minimum
`sensing_confidence` fraction for a row to be scored at all. Below it the
row is fed to the state machine as unscorable (`UNKNOWN`), not as the
`0.0` pressure the dropped-out detector actually reported — a face
detector losing the crowd is a legitimate reading of ~0 pressure, so
scoring it produces a falsely-calm `NORMAL` and the dense-crowd signature
becomes silence. With the flag set, that pattern surfaces as a
`sensor-blind` alert after `--blind-alert`. The default of `0.0` changes
nothing, so raising it is an explicit opt-in tied to how much of the frame
your detector can be expected to sense.

Bad settings fail with `error: ...` and a non-zero exit, never a
traceback: a `--threshold-high` that crosses the (unexposed) critical tier,
a negative duration or an out-of-range `--min-coverage`, a non-positive
`--fps` used for timestamp synthesis, and a row whose `timestamp` is a
non-finite JSON literal (`NaN`/`Infinity` — `json.loads` accepts these, so
they get past the JSON parse) are all reported naming the offending input.

A row missing the `timestamp` key is never silently treated as `t=0.0`
(every row would then collide at the same instant, dwell would never
complete, and a CRITICAL-pressure file could replay as a silent, alert-free
"calm crowd"). Instead `watch` prints a loud stderr warning and synthesises
`timestamp = index / fps`, where `fps` is `--fps` if given, else derived from
any other timestamps present in the file, else the eval default. Pass `--fps`
to set it explicitly whenever the true frame rate is known.

A missing or unparseable `metrics` file prints `error: ...` naming the path
and exits non-zero, the same as `run`.

Pass `--sns --sns-topic-arn arn:aws:sns:...` to also publish to AWS SNS.
**`--dry-run` defaults to `true` whenever `--sns` is given** — alerts are
logged as "would publish" but nothing is sent — because thresholds get tuned
against recorded data, not against real phones, and SNS email subscriptions
can be rate-limited by exactly that kind of iteration. Pass `--no-dry-run`
explicitly to actually publish. `--sns` without a topic ARN exits non-zero
rather than running silently without alerting. Live delivery additionally
requires the caller's AWS credentials to have `sns:Publish` on the topic ARN.

## Design rules

- `metrics/` is pure numpy: no cv2, no I/O. The physics stays unit-testable.
- Velocities are px/frame; `fps` converts to per-second. Never mix units.
- **Unknown values are NaN, never 0** — zero reads as "safe" in a safety signal.

## Tests

    python3 -m pytest tests/sfcpi/ -q

The scale-invariance property test in `tests/sfcpi/test_pressure.py` is the
executable form of the paper's central claim. If it fails, the claim is wrong.
