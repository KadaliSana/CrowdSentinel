# sfcpi — Scale-Free Crowd Pressure Index

Computes crowd pressure `P = rho * Var(v)` from **uncalibrated** video.

`P` has units of `s^-2` and carries no length dimension, so the unknown
metres-per-pixel scale cancels exactly. Pressure computed in pixel space equals
pressure in real units, given only the frame rate — see
`docs/superpowers/specs/2026-08-27-sf-cpi-design.md`.

## Usage

    python3 -m sfcpi.cli run clip.mp4 --out metrics.jsonl --cell-size 64
    python3 -m sfcpi.cli eval metrics.jsonl --labels labels.csv --fps 25
    python3 -m sfcpi.cli occlusion counts.csv --out curve.csv

## Design rules

- `metrics/` is pure numpy: no cv2, no I/O. The physics stays unit-testable.
- Velocities are px/frame; `fps` converts to per-second. Never mix units.
- **Unknown values are NaN, never 0** — zero reads as "safe" in a safety signal.

## Tests

    python3 -m pytest

The scale-invariance property test in `tests/sfcpi/test_pressure.py` is the
executable form of the paper's central claim. If it fails, the claim is wrong.
