"""Multi-view drone tracking datasets (Li et al. 2020): one hexacopter, fixed consumer cameras, point labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dronetrack.config import CenekConfig
from dronetrack.datasets.base import Sequence, annotation_table


def cameras(dataset_dir) -> dict[str, str]:
    """camera id -> camera model, from cameras.txt ("cam0 - gopro3")"""
    lines = (line.split(" - ") for line in (dataset_dir / "cameras.txt").read_text().splitlines() if " - " in line)
    return {cam.strip(): model.strip() for cam, model in lines}


def read_labels(path) -> pd.DataFrame:
    """Rows are "frame x y" with 1-based frame numbers; "0 0" marks frames where the drone is not discernible"""
    values = np.loadtxt(path, skiprows=1, ndmin=2)
    x, y = values[:, 1], values[:, 2]
    return annotation_table(
        pd.DataFrame(
            {
                "frame": values[:, 0].astype(int) - 1,
                "x": x,
                "y": y,
                "w": np.nan,
                "h": np.nan,
                "label": "drone",
                "visible": (x != 0) | (y != 0),
                "track": 0,
            }
        )
    )


def load(config: CenekConfig, split: str) -> list[Sequence]:
    sequences = []
    for name in config.splits[split]:
        directory = config.root / name
        for cam, model in cameras(directory).items():
            variant = config.calibration_variant.get(model, model)
            sequences.append(
                Sequence(
                    name=f"{name}/{cam}",
                    dataset="cenek",
                    split=split,
                    annotations=read_labels(directory / "detections" / f"{cam}.txt"),
                    video=directory / f"{cam}.mp4",
                    camera_file=config.root / "calibration" / model / f"{variant}.json",
                )
            )
    return sequences
