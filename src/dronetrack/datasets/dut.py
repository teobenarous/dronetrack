"""DUT Anti-UAV detection set (Zhao et al. 2022): still images with Pascal VOC boxes."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from dronetrack.config import DutConfig
from dronetrack.datasets.base import Sequence, annotation_table


def load(config: DutConfig, split: str) -> list[Sequence]:
    images = sorted((config.root / split / "img").glob("*.jpg"))
    rows = []
    for frame, image in enumerate(images):
        root = ET.parse(config.root / split / "xml" / f"{image.stem}.xml").getroot()
        for track, obj in enumerate(root.iter("object")):
            box = obj.find("bndbox")
            assert box is not None
            x1, y1, x2, y2 = (float(box.findtext(k, "nan")) for k in ("xmin", "ymin", "xmax", "ymax"))
            rows.append(
                {
                    "frame": frame,
                    "x": (x1 + x2) / 2,
                    "y": (y1 + y2) / 2,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "label": "drone",
                    "visible": True,
                    "track": track,
                }
            )
    if not images:
        return []
    return [
        Sequence(
            name=f"dut/{split}",
            dataset="dut",
            split=split,
            annotations=annotation_table(rows),
            images=images,
            times=np.arange(len(images), dtype=float),
            temporal=False,
        )
    ]
