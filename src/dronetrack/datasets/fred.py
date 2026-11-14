"""FRED (Magrini et al. 2025), RGB frames of the sky-facing sequences.

Frames are named by wall-clock time ([Video_<id>_]HH_MM_SS.ffffff.jpg). Boxes are keyed by
seconds since the first frame. coordinates.txt is in the coordinates of the padded 1280x720
RGB frames; coordinates_rgb.txt is not, and is ignored.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from dronetrack.config import FredConfig
from dronetrack.datasets.base import Sequence, annotation_table

STAMP = re.compile(r"(?:^|_)(\d{1,2})_(\d{2})_(\d{2}\.\d+)\.(?:jpg|png)$")


def frame_time(path: Path) -> float:
    match = STAMP.search(path.name)
    if match is None:
        raise ValueError(f"no timestamp in {path.name}")
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def match_frames(label_times: np.ndarray, frame_times: np.ndarray) -> np.ndarray:
    """Index of the nearest frame for each label time, or -1 if none lies within half a sampling period"""
    order = np.clip(np.searchsorted(frame_times, label_times), 1, len(frame_times) - 1)
    left, right = frame_times[order - 1], frame_times[order]
    nearest = np.where(label_times - left <= right - label_times, order - 1, order)
    # train sequences keep every 10th frame, so the tolerance follows the (30 Hz) label rate
    steps = [np.diff(frame_times), np.diff(np.unique(label_times))]
    tolerance = 0.5 * min(np.median(s) for s in steps if len(s))
    return np.where(np.abs(frame_times[nearest] - label_times) <= tolerance, nearest, -1)


def read_sequence(directory: Path, split: str) -> Sequence:
    images = sorted((directory / "RGB").glob("*.jpg"), key=frame_time)
    stamps = np.array([frame_time(p) for p in images])
    times = stamps - stamps[0]

    raw = []
    for line in (directory / "coordinates.txt").read_text().splitlines():
        if ":" not in line:
            continue
        t, rest = line.split(":", 1)
        x1, y1, x2, y2, track = (float(v) for v in rest.split(",")[:5])
        raw.append((float(t), x1, y1, x2, y2, int(track)))
    values = np.array(raw, dtype=float).reshape(-1, 6)
    frames = match_frames(values[:, 0], times)
    keep = frames >= 0
    rows = [
        {
            "frame": int(f),
            "x": (x1 + x2) / 2,
            "y": (y1 + y2) / 2,
            "w": x2 - x1,
            "h": y2 - y1,
            "label": "drone",
            "visible": True,
            "track": int(track),
        }
        for f, (_, x1, y1, x2, y2, track) in zip(frames[keep], values[keep], strict=True)
    ]
    return Sequence(
        name=f"fred/{directory.name}",
        dataset="fred",
        split=split,
        annotations=annotation_table(rows),
        images=images,
        times=times,
    )


def load(config: FredConfig, split: str) -> list[Sequence]:
    folder = "test" if split == "test" else "train"
    directories = sorted((config.root / folder).iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1)
    directories = [d for d in directories if (d / "coordinates.txt").exists()]
    if split == "train":
        directories = [d for d in directories if d.name not in config.val_sequences]
    elif split == "val":
        directories = [d for d in directories if d.name in config.val_sequences]
    return [read_sequence(d, split) for d in directories]
