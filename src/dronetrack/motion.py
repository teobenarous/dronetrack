"""Moving-blob candidates for a fixed camera.

A Gaussian-mixture background model alone also fires on drifting cloud edges; requiring a
change over two frames as well keeps blobs that actually move between frames.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from dronetrack.config import MotionConfig
from dronetrack.structures import Detection
from dronetrack.video import to_gray


class MotionDetector:
    def __init__(self, config: MotionConfig):
        self.config = config
        self.background = cv2.createBackgroundSubtractorMOG2(
            history=config.history, varThreshold=config.var_threshold, detectShadows=False
        )
        self.history: deque[np.ndarray] = deque(maxlen=3)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def __call__(self, image: np.ndarray) -> list[Detection]:
        gray, scale = to_gray(image, self.config.work_width)
        foreground = self.background.apply(gray)
        self.history.append(gray)
        if len(self.history) < 3:
            return []

        change = cv2.absdiff(gray, self.history[0])
        mask = (foreground > 0) & (change > self.config.diff_threshold)
        mask = cv2.dilate(mask.astype(np.uint8), self.kernel)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        blobs = []
        for i in range(1, count):
            x, y, w, h, area = stats[i]
            if not self.config.min_area <= area <= self.config.max_area:
                continue
            contrast = float(change[y : y + h, x : x + w].max()) / 255.
            cx, cy = centroids[i]
            blobs.append(Detection(cx * scale, cy * scale, w * scale, h * scale, contrast, source="motion"))
        blobs.sort(key=lambda d: d.score, reverse=True)
        return blobs[: self.config.max_candidates]
