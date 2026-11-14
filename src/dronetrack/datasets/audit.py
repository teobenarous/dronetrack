"""Counts and label overlays for every dataset split, to check readers and labels by eye."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dronetrack import datasets
from dronetrack.config import Config
from dronetrack.datasets import Sequence
from dronetrack.video import frame_times

SPLITS = {
    "cenek": ("train", "val", "test"),
    "halmstad": ("train", "val", "test"),
    "dut": ("train", "val", "test"),
    "fred": ("train", "val", "test")
}


def summary(sequence: Sequence) -> dict:
    a = sequence.annotations
    drone = a[a.label == "drone"]
    row = {
        "dataset": sequence.dataset,
        "split": sequence.split,
        "sequence": sequence.name,
        "labelled_frames": a.frame.nunique(),
        "drone_visible_frames": drone[drone.visible].frame.nunique(),
        "objects": {k: int(v) for k, v in a[a.visible].label.value_counts().items()},
        "drone_free": sequence.drone_free,
    }
    if sequence.video is not None:
        times = frame_times(sequence.video)
        row.update(
            video_frames=len(times),
            fps=round(1 / np.median(np.diff(times)), 2),
            max_gap_ms=round(1e3 * np.diff(times).max(), 1),
            labels_match_frames=bool(a.frame.max() < len(times)),
        )
    elif sequence.images is not None:
        row["video_frames"] = len(sequence.images)
    return row


def overlay(
    sequence: Sequence, rng: np.random.Generator, max_frames: int | None, n: int = 2, size: int = 320
) -> list[np.ndarray]:
    labelled = sequence.drone_boxes()
    if not len(labelled):
        labelled = sequence.annotations[sequence.annotations.visible]
    if max_frames is not None:
        labelled = labelled[labelled.frame < max_frames]
    if not len(labelled):
        return []
    frames = labelled.frame.unique()
    picks = set(rng.choice(frames, min(n, len(frames)), replace=False).tolist())
    tiles = []
    for frame in sequence.frames(select=picks.__contains__):
        image = frame.image
        rows = labelled[labelled.frame == frame.index]
        for row in rows.itertuples():
            r = 14 if np.isnan(row.w) else int(max(row.w, row.h) / 2) + 6
            cv2.rectangle(image, (int(row.x) - r, int(row.y) - r), (int(row.x) + r, int(row.y) + r), (0, 0, 255), 2)
        x, y = int(rows.x.iloc[0]), int(rows.y.iloc[0])
        x0, y0 = max(x - size, 0), max(y - size * 3 // 4, 0)
        tile = cv2.resize(image[y0 : y0 + 3 * size // 2, x0 : x0 + 2 * size], (size, 3 * size // 4))
        cv2.putText(tile, f"{sequence.name} #{frame.index}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
        tiles.append(tile)
    return tiles


def audit(config: Config, seed: int = 0) -> Path:
    out = config.cache / "audit"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for dataset, splits in SPLITS.items():
        tiles: list[np.ndarray] = []
        for split in splits:
            sequences = datasets.load(config, dataset, split)
            rows += [summary(s) for s in sequences]
            for i in rng.choice(len(sequences), min(4, len(sequences)), replace=False):
                tiles += overlay(sequences[i], rng, config.max_frames)
        if tiles:
            tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 4)
            grid = np.vstack([np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)])
            cv2.imwrite(str(out / f"{dataset}.jpg"), grid)
    table = pd.DataFrame(rows)
    table.to_csv(out / "summary.csv", index=False)
    return out
