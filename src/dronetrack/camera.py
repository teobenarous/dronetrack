"""Pixel to direction: azimuth and elevation in the camera frame, with first-order covariance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Camera:
    K: np.ndarray  # 3x3
    dist: np.ndarray  # OpenCV order: k1, k2, p1, p2[, k3]
    size: tuple[int, int]  # width, height

    @classmethod
    def from_json(cls, path: Path, size: tuple[int, int] | None = None) -> Camera:
        spec = json.loads(Path(path).read_text())
        K = np.asarray(spec["K-matrix"], dtype=float)
        calibrated = tuple(spec["resolution"])
        camera = cls(K, np.asarray(spec["distCoeff"], dtype=float), calibrated)
        return camera.resized(size) if size and size != calibrated else camera

    @classmethod
    def pinhole(cls, size: tuple[int, int], hfov_deg: float) -> Camera:
        f = size[0] / 2 / np.tan(np.radians(hfov_deg) / 2)
        K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1]])
        return cls(K, np.zeros(5), size)

    def resized(self, size: tuple[int, int]) -> Camera:
        sx, sy = size[0] / self.size[0], size[1] / self.size[1]
        K = np.diag([sx, sy, 1.0]) @ self.K
        return Camera(K, self.dist, size)

    def normalized(self, uv: np.ndarray) -> np.ndarray:
        points = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(points, self.K, self.dist).reshape(-1, 2)

    def angles(self, uv: np.ndarray) -> np.ndarray:
        """Azimuth (right positive) and elevation (up positive), radians, for each pixel."""
        xn, yn = self.normalized(uv).T
        return np.stack([np.arctan(xn), np.arctan2(-yn, np.sqrt(1 + xn**2))], axis=1)

    def direction(self, uv: np.ndarray, cov: np.ndarray, step: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        """Angles of one pixel and their covariance J Σ J^T, J by central differences."""
        uv = np.asarray(uv, dtype=float)
        probes = uv + step * np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
        a = self.angles(probes)
        J = np.stack([(a[1] - a[2]) / (2 * step), (a[3] - a[4]) / (2 * step)], axis=1)
        return a[0], J @ cov @ J.T
