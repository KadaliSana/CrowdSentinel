# SF-CPI — Scale-Free Crowd Pressure Index (Cycle 1 Design)

Date: 2026-08-27
Status: approved for implementation planning
Companions: `docs/litreview/GAP_ANALYSIS_AND_MODEL.md`, `docs/litreview/REFERENCES.md`

## 1. Purpose

Compute a physically-grounded crowd-crush risk signal from **uncalibrated** video, in units
comparable with the crowd-safety literature, and validate it offline against public datasets.

Cycle 1 delivers the measurement pipeline and the evidence. It deliberately excludes live
WebRTC ingest, the dashboard and alerting (cycles 2-3), so that everything here runs with
**no hardware**.

## 2. Core result this is built on

Crowd pressure `P = rho * Var(v)` has units `s^-2` — no length dimension. Therefore the
unknown metres-per-pixel scale `s` cancels exactly:

    rho_m    = N / (s^2 * A_px)
    Var_m(v) = (s * fps)^2 * Var_px(v)
    P_m      = rho_m * Var_m(v) = (N / A_px) * fps^2 * Var_px(v) = P_px

`P` is computed **per cell**, so per-cell local scale constancy is all that is required;
perspective variation across the frame is handled by construction, and cells localise risk.

Reference threshold: turbulence onset at `P > 0.02 s^-2`, ~10 min before the Jamarat crush;
flow below `0.8 ped/m/s` >30 min before [REFERENCES A1]. Treated as an order-of-magnitude
reference, not a constant.

## 3. Scope

**In:** source abstraction (file replay), dense optical flow, per-cell velocity statistics,
detection-based counting, per-cell + global pressure, metrics output, evaluation harness,
the occlusion characterisation experiment.

**Out:** live WebRTC ingest, dashboard UI, alerting, device firmware changes, head-pose signal.

## 4. Architecture

Single Python package `src/sfcpi/`, importable and CLI-runnable, no Flask dependency.

    sources/      FrameSource protocol -> FileSource (cv2), later WebRTCSource
    flow/         dense optical flow -> per-pixel velocity field
    grid/         cell partitioning + per-cell aggregation
    detect/       Detector protocol -> face/person detector, count per cell
    metrics/      pressure, speed, density, flow-rate; pure functions, no I/O
    pipeline/     orchestration: frames -> per-frame MetricsFrame
    sinks/        JSONL / CSV writers
    eval/         dataset runner, labels, detection-latency + AUC
    cli.py        `sfcpi run <video>`, `sfcpi eval <dataset>`, `sfcpi occlusion <dataset>`

**Data flow.** `FrameSource` yields `(index, timestamp, frame)`. The pipeline maintains the
previous frame, computes the flow field, partitions into cells, aggregates `Var(v)` and mean
speed per cell, obtains `N` per cell from the detector, and emits a `MetricsFrame`
(per-cell array + global aggregates + sensing-confidence fields) to a sink.

**Key boundary:** `metrics/` contains only pure functions over numpy arrays — no video, no
model, no I/O. That is what makes the physics unit-testable in isolation.

## 5. Components

| Module | Does | Depends on |
|---|---|---|
| `sources` | yields frames from file (later: WebRTC); normalises fps | cv2 |
| `flow` | Farneback dense flow; configurable downscale for speed | cv2 |
| `grid` | fixed cell grid; per-cell masks; aggregation helpers | numpy |
| `detect` | wraps a detector behind one interface; returns boxes + centroids | ultralytics / onnx |
| `metrics` | `pressure()`, `mean_speed()`, `density()`, `flow_rate()` — pure | numpy |
| `pipeline` | wires the above; smoothing (EWMA); emits MetricsFrame | above |
| `eval` | runs over labelled datasets; computes latency-to-detect, AUC, FAR/h | pandas |

## 6. Error handling

- Unreadable/short video -> explicit error naming the file; never a silent empty run.
- Fewer than 2 frames -> no flow; emit frame with `flow_valid=False` rather than crash.
- Detector unavailable/fails -> `N=None`, `sensing_confidence=0`; pressure emitted as NaN
  rather than 0, because **0 would read as "safe"**. This is the fail-loud rule and is tested.
- Degenerate cells (no motion) -> `Var(v)=0` is legitimate, not an error.
- All config validated at startup (cell size divides frame, fps > 0).

## 7. Testing

TDD; tests before implementation.

**Property tests (the important ones):**
- **Scale invariance:** for synthetic fields scaled by arbitrary `s`, `pressure()` must be
  invariant to within float tolerance. This is the paper's central claim as an executable test.
- **fps scaling:** `P` scales as `fps^2` when velocities are expressed per frame.
- Zero motion -> zero pressure; uniform motion -> zero pressure (variance, not speed).

**Unit tests:** cell aggregation vs hand-computed values; EWMA; NaN propagation (never 0).
**Integration:** short synthetic clip end-to-end produces a well-formed metrics file.

## 8. The occlusion question (decided by experiment, not assumption)

`sfcpi occlusion <dataset>` runs the detector over a dense-crowd dataset with ground-truth
counts and emits `detected_vs_true` against true density.

- Curve **stable and monotonic** -> invert it into a correction; occlusion index becomes a
  contribution with data behind it.
- Curve **noisy or scene-dependent** -> drop the correction, adopt a density estimator, and
  report the measured error honestly.

Either outcome is publishable. Contribution 1 does not depend on which way it goes.

## 9. Success criteria

1. `pressure()` passes scale-invariance property tests.
2. Pipeline runs a dataset video end-to-end and writes per-frame metrics.
3. `eval` reports detection latency and AUC vs labels for at least UMN.
4. The occlusion experiment produces the detected-vs-true curve.
5. No hardware required for any of the above.
