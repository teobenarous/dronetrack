"""Plain data passed between the stages of the pipeline"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class Frame:
    index: int
    t: float  # seconds since the first frame of the sequence
    image: np.ndarray  # BGR, uint8


@dataclass(slots=True)
class Detection:
    x: float  # box centre, full-frame pixels
    y: float
    w: float
    h: float
    score: float
    label: int = 0
    source: str = "detector"

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclass(slots=True)
class Estimate:
    """Per-frame output: the record a downstream 3D filter would consume."""

    frame: int
    t: float
    declared: bool
    presence: float
    x: float = np.nan
    y: float = np.nan
    w: float = np.nan
    h: float = np.nan
    cov: tuple[float, float, float] = (np.nan, np.nan, np.nan)  # sxx, sxy, syy
    azimuth: float = np.nan
    elevation: float = np.nan
    angle_cov: tuple[float, float, float] = (np.nan, np.nan, np.nan)
    latency_ms: float = np.nan
    timings: dict[str, float] = field(default_factory=dict)

    def record(self) -> dict:
        row = {
            name: getattr(self, name)
            for name in ("frame", "t", "declared", "presence", "x", "y", "w", "h", "azimuth", "elevation", "latency_ms")
        }
        row.update(
            cov_xx=self.cov[0],
            cov_xy=self.cov[1],
            cov_yy=self.cov[2],
            cov_az=self.angle_cov[0],
            cov_az_el=self.angle_cov[1],
            cov_el=self.angle_cov[2],
        )
        row.update({f"ms_{name}": value for name, value in self.timings.items()})
        return row
