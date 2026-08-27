"""Dense optical flow (Farneback).

Flow is the density-robust half of the sensing story: it keeps working when
detection collapses under occlusion. Vectors are always returned in pixels per
frame at FULL frame resolution, whatever the internal downscale.
"""
from __future__ import annotations

import cv2
import numpy as np

_FARNEBACK = dict(
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
)


class FlowEstimator:
    def __init__(self, downscale: float = 1.0) -> None:
        if not (0 < downscale <= 1.0):
            raise ValueError("downscale must be in (0, 1]")
        self.downscale = float(downscale)

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def estimate(self, prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> np.ndarray:
        if prev_bgr.shape != curr_bgr.shape:
            raise ValueError("frames must have the same shape")
        h, w = prev_bgr.shape[:2]
        prev, curr = self._gray(prev_bgr), self._gray(curr_bgr)

        if self.downscale < 1.0:
            small = (max(1, int(w * self.downscale)), max(1, int(h * self.downscale)))
            prev = cv2.resize(prev, small, interpolation=cv2.INTER_AREA)
            curr = cv2.resize(curr, small, interpolation=cv2.INTER_AREA)

        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, **_FARNEBACK)

        if self.downscale < 1.0:
            sx = w / flow.shape[1]
            sy = h / flow.shape[0]
            flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
            flow[..., 0] *= sx      # rescale vectors, not just the raster
            flow[..., 1] *= sy
        return flow.astype(np.float32)
