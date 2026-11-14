"""Halmstad drone detection dataset (Svanström et al. 2021), visible clips, YOLO-format copy.

The copy keeps every third frame of 30 fps, 640x512 clips; clip names give the class
(V_DRONE_*, V_BIRD_*, V_AIRPLANE_*, V_HELICOPTER_*).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from dronetrack.config import HalmstadConfig
from dronetrack.datasets.base import Sequence, annotation_table
from dronetrack.video import imread

CLASSES = {0: "drone", 1: "bird", 2: "aircraft", 3: "aircraft"}


def load(config: HalmstadConfig, split: str) -> list[Sequence]:
    clips: dict[str, list] = defaultdict(list)
    for path in sorted((config.root / "images" / split).glob("*.jpg")):
        clip, index = path.stem.rsplit("_", 1)
        clips[clip].append((int(index), path))
    if not clips:
        return []
    height, width = imread(next(iter(clips.values()))[0][1]).shape[:2]

    sequences = []
    for clip, items in clips.items():
        items.sort()
        rows = []
        for frame, (_, path) in enumerate(items):
            label_file = config.root / "labels" / split / f"{path.stem}.txt"
            if not label_file.exists():
                continue
            for track, line in enumerate(label_file.read_text().splitlines()):
                cls, cx, cy, w, h = line.split()[:5]
                rows.append(
                    {
                        "frame": frame,
                        "x": float(cx) * width,
                        "y": float(cy) * height,
                        "w": float(w) * width,
                        "h": float(h) * height,
                        "label": CLASSES[int(cls)],
                        "visible": True,
                        "track": track,
                    }
                )
        sequences.append(
            Sequence(
                name=clip,
                dataset="halmstad",
                split=split,
                annotations=annotation_table(rows),
                images=[path for _, path in items],
                times=np.array([(index - 1) / config.fps for index, _ in items]),
                drone_free=not clip.startswith("V_DRONE"),
            )
        )
    return sequences
