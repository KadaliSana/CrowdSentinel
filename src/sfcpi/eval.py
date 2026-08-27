"""Offline evaluation.

Detection latency is the operationally meaningful number: Johansson et al.
report crowd pressure crossing its critical value ~10 minutes before the
Jamarat crush, so how EARLY a method fires matters more than raw accuracy.

Non-finite scores mean "pressure could not be computed" (the detector or the
flow was dead), NOT "safe". They are never ranked as maximally-safe and never
pad a rate denominator; every metric here either excludes them or, in the one
case where treating them as a non-alarm is the defensible reading, is reported
alongside `coverage` so a blind run cannot masquerade as a quiet one.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def load_labels(path: str) -> np.ndarray:
    return pd.read_csv(path)["label"].to_numpy().astype(int)


def _scorable(scores: np.ndarray) -> np.ndarray:
    """Mask of frames where a score was actually computed."""
    return np.isfinite(np.asarray(scores, dtype=float))


def _alarms(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Alarm mask. A non-finite score is not an alarm -- but it is not a
    'safe' observation either, so callers must exclude it from any denominator
    rather than counting it as a true negative (see false_alarms_per_hour)."""
    scores = np.asarray(scores, dtype=float)
    with np.errstate(invalid="ignore"):
        return _scorable(scores) & (scores >= threshold)


def coverage(scores: np.ndarray) -> float:
    """Fraction of frames that produced a finite score. NaN for no frames."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return float("nan")
    return float(_scorable(scores).sum() / scores.size)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U). Ties contribute 0.5.

    Unscorable frames are EXCLUDED from both classes. Ranking them as
    maximally-safe (the old `nan_to_num(nan=-inf)`) manufactured a plausible
    AUC out of frames the method never saw.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    ok = _scorable(scores)
    pos, neg = scores[ok & (labels == 1)], scores[ok & (labels == 0)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (pos.size * neg.size))


def detection_latency(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> Optional[int]:
    """Frames between event onset and first alarm. Negative = early warning.

    A non-finite score does NOT fire: a system that cannot see should not
    raise an alarm. That is only defensible because `evaluate()` reports
    `coverage` alongside -- a latency of None on a blind run must be readable
    as "the method was blind", not as "the method stayed calm".
    """
    labels = np.asarray(labels).astype(int)
    positives = np.flatnonzero(labels == 1)
    if positives.size == 0:
        return None
    alarms = np.flatnonzero(_alarms(scores, threshold))
    if alarms.size == 0:
        return None
    return int(alarms[0] - positives[0])


def _count_episodes(mask: np.ndarray) -> int:
    """Number of contiguous True runs (rising edges) in a boolean mask.

    A gap of unscorable frames splits an episode: we cannot assert that the
    alarm persisted across frames we never scored.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return 0
    return int(mask[0]) + int(np.count_nonzero(mask[1:] & ~mask[:-1]))


def false_alarms_per_hour(
    scores: np.ndarray, labels: np.ndarray, threshold: float, fps: float
) -> float:
    """False-alarm EPISODES per hour of scorable negative time.

    Two corrections over the naive form:
      - EPISODES, not frames. At 25 fps one 4-second false alarm is 100
        alarming frames; reporting it as 90,000/h is meaningless. A rising
        edge is one operator interruption, which is the thing being counted.
      - Only SCORABLE negative frames enter the denominator. An unscorable
        frame contributes no alarm, so leaving it in the denominator credits
        the method with quiet time it never observed and deflates the rate.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    negatives = (labels == 0) & _scorable(scores)
    negative_frames = int(negatives.sum())
    if negative_frames == 0 or fps <= 0:
        return float("nan")
    episodes = _count_episodes(_alarms(scores, threshold) & negatives)
    return float(episodes / (negative_frames / fps) * 3600.0)


def evaluate(
    scores: np.ndarray, labels: np.ndarray, threshold: float, fps: float
) -> Dict[str, float]:
    """Score a run. `coverage` is not optional garnish: without it a run that
    was blind for 90% of frames reports FAR/h 0 and reads as flawless."""
    scores = np.asarray(scores, dtype=float)
    n_frames = int(scores.size)
    n_unscorable = int((~_scorable(scores)).sum())
    return {
        "threshold": float(threshold),
        "auc": roc_auc(scores, labels),
        "detection_latency_frames": detection_latency(scores, labels, threshold),
        "false_alarms_per_hour": false_alarms_per_hour(scores, labels, threshold, fps),
        "n_frames": n_frames,
        "n_unscorable": n_unscorable,
        "coverage": coverage(scores),
    }
