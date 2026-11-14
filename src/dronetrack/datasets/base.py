from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dronetrack.structures import Frame
from dronetrack.video import read_images, read_video

ANNOTATION_COLUMNS = ["frame", "x", "y", "w", "h", "label", "visible", "track"]


def annotation_table(rows: list[dict] | pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per labelled object per frame; box datasets leave frames without objects out"""
    frame = pd.DataFrame(rows if rows is not None else [], columns=ANNOTATION_COLUMNS)
    return frame.astype(
        {"frame": int, "x": float, "y": float, "w": float, "h": float, "label": str, "visible": bool, "track": int}
    )


@dataclass
class Sequence:
    name: str
    dataset: str
    split: str
    annotations: pd.DataFrame
    video: Path | None = None
    images: list[Path] | None = None
    times: np.ndarray | None = None
    camera_file: Path | None = None
    drone_free: bool = False
    temporal: bool = True

    def frames(self, max_frames: int | None = None, select: Callable[[int], bool] | None = None) -> Iterator[Frame]:
        if self.video is not None:
            return read_video(self.video, max_frames=max_frames, select=select)
        assert self.images is not None and self.times is not None
        images, times = self.images, self.times
        if select is not None:
            keep = [i for i in range(len(images)) if select(i)]
            frames = read_images([images[i] for i in keep], times[keep], max_frames=max_frames)
            return (Frame(keep[f.index], f.t, f.image) for f in frames)
        return read_images(images, times, max_frames=max_frames)

    def drone_boxes(self) -> pd.DataFrame:
        a = self.annotations
        return a[(a.label == "drone") & a.visible]
