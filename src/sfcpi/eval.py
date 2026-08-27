"""Offline evaluation.

Detection latency is the operationally meaningful number: Johansson et al.
report crowd pressure crossing its critical value ~10 minutes before the
Jamarat crush, so how EARLY a method fires matters more than raw accuracy.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def load_labels(path: str) -> np.ndarray:
    return pd.read_csv(path)["label"].to_numpy().astype(int)


def _alarms(scores: np.ndarray, threshold: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    return np.nan_to_num(scores, nan=-np.inf) >= threshold


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U). Ties contribute 0.5."""
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf)
    labels = np.asarray(labels).astype(int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (pos.size * neg.size))


def detection_latency(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> Optional[int]:
    """Frames between event onset and first alarm. Negative = early warning."""
    labels = np.asarray(labels).astype(int)
    positives = np.flatnonzero(labels == 1)
    if positives.size == 0:
        return None
    alarms = np.flatnonzero(_alarms(scores, threshold))
    if alarms.size == 0:
        return None
    return int(alarms[0] - positives[0])


def false_alarms_per_hour(
    scores: np.ndarray, labels: np.ndarray, threshold: float, fps: float
) -> float:
    labels = np.asarray(labels).astype(int)
    negatives = labels == 0
    negative_frames = int(negatives.sum())
    if negative_frames == 0 or fps <= 0:
        return float("nan")
    false_alarms = int((_alarms(scores, threshold) & negatives).sum())
    return float(false_alarms / (negative_frames / fps) * 3600.0)


def evaluate(
    scores: np.ndarray, labels: np.ndarray, threshold: float, fps: float
) -> Dict[str, float]:
    return {
        "threshold": float(threshold),
        "auc": roc_auc(scores, labels),
        "detection_latency_frames": detection_latency(scores, labels, threshold),
        "false_alarms_per_hour": false_alarms_per_hour(scores, labels, threshold, fps),
    }
