"""Learned detection on full-resolution crops chosen per frame.

A small drone shrinks below a pixel if the whole frame is resized to the network input, so the
detector only ever sees native-resolution crops: one around the track prediction, a few around
motion candidates, and one tile of a sweep that covers the frame every few frames. The cost per
frame is bounded by the number of crops, independent of the camera resolution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms
from ultralytics import YOLO

from dronetrack.config import ByteTrackConfig, DetectorConfig
from dronetrack.structures import Detection


def restore_opencv_threads() -> None:
    """Ultralytics turns OpenCV multithreading off on import (for its data loaders); inference wants it"""
    cv2.setNumThreads(os.cpu_count() or 1)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True, slots=True)
class Crop:
    x0: int
    y0: int
    size: int
    reason: str

    def contains(self, x: float, y: float, shape: tuple[int, int] | None = None, margin: float = 0) -> bool:
        """Whether (x, y) lies inside, at least `margin` from any edge that is not the image border"""
        height, width = shape if shape is not None else (np.inf, np.inf)
        left = margin if self.x0 > 0 else 0
        top = margin if self.y0 > 0 else 0
        right = margin if self.x0 + self.size < width else 0
        bottom = margin if self.y0 + self.size < height else 0
        return self.x0 + left <= x < self.x0 + self.size - right and self.y0 + top <= y < self.y0 + self.size - bottom


def sweep_positions(length: int, crop: int, overlap: int) -> list[int]:
    if length <= crop:
        return [0]
    positions = list(range(0, length - crop, crop - overlap))
    return positions + [length - crop]


class CropScheduler:
    def __init__(self, config: DetectorConfig):
        self.c = config
        self.next_tile = 0

    def centered(self, x: float, y: float, shape: tuple[int, int], reason: str) -> Crop:
        height, width = shape
        s = self.c.crop
        x0 = int(np.clip(round(x - s / 2), 0, max(width - s, 0)))
        y0 = int(np.clip(round(y - s / 2), 0, max(height - s, 0)))
        return Crop(x0, y0, s, reason)

    def plan(self, shape: tuple[int, int], prediction: np.ndarray | None, candidates: list[Detection]) -> list[Crop]:
        crops = []
        if prediction is not None:
            crops.append(self.centered(float(prediction[0]), float(prediction[1]), shape, "track"))
        for blob in candidates:
            if sum(c.reason == "motion" for c in crops) >= self.c.max_motion_crops:
                break
            if not any(c.contains(blob.x, blob.y, shape, margin=32) for c in crops):
                crops.append(self.centered(blob.x, blob.y, shape, "motion"))
        tiles = [
            (x, y)
            for y in sweep_positions(shape[0], self.c.crop, self.c.sweep_overlap)
            for x in sweep_positions(shape[1], self.c.crop, self.c.sweep_overlap)
        ]
        x0, y0 = tiles[self.next_tile % len(tiles)]
        self.next_tile += 1
        crops.append(Crop(x0, y0, self.c.crop, "sweep"))
        return crops


def cut(image: np.ndarray, crop: Crop) -> np.ndarray:
    patch = image[crop.y0 : crop.y0 + crop.size, crop.x0 : crop.x0 + crop.size]
    if patch.shape[:2] == (crop.size, crop.size):
        return patch
    padded = np.zeros((crop.size, crop.size, 3), dtype=image.dtype)
    padded[: patch.shape[0], : patch.shape[1]] = patch
    return padded


class Detector:
    def __init__(self, weights: Path, device: str, config: DetectorConfig):
        self.c = config
        self.device = resolve_device(device)
        self.model = YOLO(str(weights), task="detect")

    def __call__(self, image: np.ndarray, crops: list[Crop]) -> list[Detection]:
        if not crops:
            return []
        results = self.model.predict(
            [cut(image, c) for c in crops], imgsz=self.c.crop, conf=self.c.conf, device=self.device, verbose=False
        )
        boxes, scores, labels = [], [], []
        for crop, result in zip(crops, results, strict=True):
            if len(result.boxes) == 0:
                continue
            boxes.append(result.boxes.xyxy.cpu() + torch.tensor([crop.x0, crop.y0, crop.x0, crop.y0]))
            scores.append(result.boxes.conf.cpu())
            labels.append(result.boxes.cls.cpu())
        if not boxes:
            return []
        xyxy, score, label = torch.cat(boxes), torch.cat(scores), torch.cat(labels)
        keep = batched_nms(xyxy, score, label.long(), self.c.nms_iou)
        return [
            Detection(
                float((x1 + x2) / 2),
                float((y1 + y2) / 2),
                float(x2 - x1),
                float(y2 - y1),
                float(score[i]),
                int(label[i]),
            )
            for i in keep.tolist()
            for x1, y1, x2, y2 in [xyxy[i].tolist()]
        ]


class ByteTrackBaseline:
    """The standard tracking-by-detection setup: the same detector on the whole frame plus ByteTrack"""

    def __init__(self, weights: Path, device: str, config: ByteTrackConfig):
        self.c = config
        self.device = resolve_device(device)
        self.model = YOLO(str(weights), task="detect")
        self.current_id: int | None = None

    def __call__(self, image: np.ndarray) -> Detection | None:
        imgsz = min(self.c.imgsz, int(np.ceil(max(image.shape[:2]) / 32) * 32))
        result = self.model.track(
            image,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=imgsz,
            conf=self.c.conf,
            classes=[0],
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes.id is None or len(result.boxes) == 0:
            self.current_id = None
            return None
        ids = result.boxes.id.int().tolist()
        conf = result.boxes.conf.tolist()
        i = ids.index(self.current_id) if self.current_id in ids else int(np.argmax(conf))
        self.current_id = ids[i]
        x, y, w, h = result.boxes.xywh[i].tolist()
        return Detection(x, y, w, h, conf[i])
