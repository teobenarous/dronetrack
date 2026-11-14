"""YOLO-format training sets: native-resolution tiles from every source, plus the detection test sets.

The Cenek labels are points, so training boxes are segmented around each point against the
local background. These boxes are only ever used for training; evaluation on Cenek uses the
original points.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from dronetrack import datasets
from dronetrack.config import Config, SourceConfig
from dronetrack.datasets import Sequence
from dronetrack.video import imread

Box = tuple[str, float, float, float, float]  # label, centre x, centre y, width, height (pixels)
Sample = tuple[np.ndarray, list[Box]]

FALLBACK_SIZE = 12.0


def box_from_point(gray: np.ndarray, x: float, y: float, window: int = 48) -> tuple[tuple[float, ...], bool]:
    """Box of the blob at (x, y), segmented against the median of the window border.

    Returns (cx, cy, w, h) and whether segmentation succeeded; on failure a fixed-size box
    centred on the point is returned.
    """
    r = window // 2
    cx, cy = int(round(x)), int(round(y))
    x0, y0 = max(cx - r, 0), max(cy - r, 0)
    patch = gray[y0 : cy + r, x0 : cx + r].astype(np.float32)
    fallback = ((x, y, FALLBACK_SIZE, FALLBACK_SIZE), False)
    if patch.shape[0] < 8 or patch.shape[1] < 8:
        return fallback
    ring = np.concatenate([patch[:3].ravel(), patch[-3:].ravel(), patch[:, :3].ravel(), patch[:, -3:].ravel()])
    background = np.median(ring)
    spread = 1.4826 * np.median(np.abs(ring - background))
    mask = (np.abs(patch - background) > max(4 * spread, 8.0)).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 2:
        return fallback
    distance = np.hypot(centroids[1:, 0] + x0 - x, centroids[1:, 1] + y0 - y)
    best = int(np.argmin(distance)) + 1
    if distance[best - 1] > 6:
        return fallback
    bx, by, bw, bh, _ = stats[best]
    w, h = max(bw + 4, 4), max(bh + 4, 4)
    return (x0 + bx + bw / 2, y0 + by + bh / 2, float(w), float(h)), True


def tile_around(
    image: np.ndarray, x: float, y: float, size: int, rng: np.random.Generator, margin: int = 48
) -> tuple[np.ndarray, int, int]:
    """A size x size crop containing (x, y) at a random position, zero-padded at the image border"""
    height, width = image.shape[:2]
    lo_x, hi_x = int(x) - size + margin, int(x) - margin
    lo_y, hi_y = int(y) - size + margin, int(y) - margin
    x0 = min(max(int(rng.integers(lo_x, hi_x + 1)), 0), max(width - size, 0))
    y0 = min(max(int(rng.integers(lo_y, hi_y + 1)), 0), max(height - size, 0))
    return crop(image, x0, y0, size), x0, y0


def crop(image: np.ndarray, x0: int, y0: int, size: int) -> np.ndarray:
    patch = image[y0 : y0 + size, x0 : x0 + size]
    if patch.shape[:2] == (size, size):
        return patch
    out = np.zeros((size, size, 3), dtype=image.dtype)
    out[: patch.shape[0], : patch.shape[1]] = patch
    return out


def shift(boxes: list[Box], x0: int, y0: int, size: int) -> list[Box]:
    """Boxes in tile coordinates, clipped to the tile; boxes less than half inside are dropped"""
    out = []
    for label, x, y, w, h in boxes:
        x1, y1, x2, y2 = x - w / 2 - x0, y - h / 2 - y0, x + w / 2 - x0, y + h / 2 - y0
        cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, size), min(y2, size)
        if cx2 <= cx1 or cy2 <= cy1 or (cx2 - cx1) * (cy2 - cy1) < 0.5 * w * h:
            continue
        out.append((label, (cx1 + cx2) / 2, (cy1 + cy2) / 2, cx2 - cx1, cy2 - cy1))
    return out


def frame_boxes(sequence: Sequence) -> dict[int, list[Box]]:
    a = sequence.annotations[sequence.annotations.visible]
    boxes: dict[int, list[Box]] = {}
    for row in a.itertuples():
        boxes.setdefault(int(row.frame), []).append((row.label, row.x, row.y, row.w, row.h))
    return boxes


def cenek_samples(
    config: Config, split: str, source: SourceConfig, rng: np.random.Generator, stats: Counter
) -> Iterator[Sample]:
    tiles = config.training.tiles
    sequences = datasets.load(config, "cenek", split)
    candidates = [
        (i, f)
        for i, s in enumerate(sequences)
        for f in s.drone_boxes().frame.tolist()
        if f % tiles.cenek_frame_stride == 0
    ]
    if source.cap is not None and len(candidates) > source.cap:
        candidates = [candidates[i] for i in rng.choice(len(candidates), source.cap, replace=False)]
    for i, sequence in enumerate(sequences):
        wanted = {f for j, f in candidates if j == i}
        if not wanted:
            continue
        labels = sequence.annotations.set_index("frame")
        for frame in sequence.frames(select=wanted.__contains__, max_frames=config.max_frames):
            row = labels.loc[frame.index]
            gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
            (x, y, w, h), segmented = box_from_point(gray, row.x, row.y)
            stats["cenek_segmented" if segmented else "cenek_fallback"] += 1
            patch, x0, y0 = tile_around(frame.image, x, y, tiles.size, rng)
            yield patch, shift([("drone", x, y, w, h)], x0, y0, tiles.size)
            if rng.random() < tiles.negative_fraction:
                negative = negative_tile(frame.image, x, y, tiles.size, rng)
                if negative is not None:
                    yield negative, []


def negative_tile(image: np.ndarray, x: float, y: float, size: int, rng: np.random.Generator) -> np.ndarray | None:
    height, width = image.shape[:2]
    for _ in range(20):
        x0, y0 = (int(v) for v in rng.integers(0, [max(width - size, 0) + 1, max(height - size, 0) + 1]))
        if not (x0 - 32 <= x < x0 + size + 32 and y0 - 32 <= y < y0 + size + 32):
            return crop(image, x0, y0, size)
    return None


def boxed_samples(
    config: Config,
    dataset: str,
    split: str,
    source: SourceConfig,
    rng: np.random.Generator,
    stats: Counter,
    whole_image: bool = False,
) -> Iterator[Sample]:
    """Halmstad, DUT, FRED: per-frame boxes. Small images are used whole, large ones are tiled"""
    size = config.training.tiles.size
    for sequence in datasets.load(config, dataset, split):
        boxes = frame_boxes(sequence)
        frames = sorted(boxes) if not whole_image else list(range(len(sequence.images or [])))
        if source.cap_per_sequence is not None and len(frames) > source.cap_per_sequence:
            frames = rng.choice(frames, source.cap_per_sequence, replace=False).tolist()
        for frame in sequence.frames(select=set(frames).__contains__):
            items = boxes.get(frame.index, [])
            stats[f"{dataset}_images"] += 1
            if whole_image or max(frame.image.shape[:2]) <= size:
                yield frame.image, items
                continue
            image, items = shrink_large_targets(frame.image, items, size)
            target = items[rng.integers(len(items))]
            patch, x0, y0 = tile_around(image, target[1], target[2], size, rng)
            yield patch, shift(items, x0, y0, size)


def shrink_large_targets(image: np.ndarray, boxes: list[Box], size: int) -> tuple[np.ndarray, list[Box]]:
    """Downscale so that no box exceeds half a tile (some DUT drones fill the frame)"""
    largest = max(max(w, h) for _, _, _, w, h in boxes)
    if largest <= size / 2:
        return image, boxes
    f = size / 2 / largest
    image = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return image, [(label, x * f, y * f, w * f, h * f) for label, x, y, w, h in boxes]


def sample_stream(config: Config, name: str, split: str, rng: np.random.Generator, stats: Counter) -> Iterator[Sample]:
    recipe = config.training.models[name]
    for dataset, source in recipe.sources.items():
        if dataset == "cenek":
            yield from cenek_samples(config, split, source, rng, stats)
            continue
        stream = boxed_samples(config, dataset, split, source, rng, stats, whole_image=dataset == "halmstad")
        if source.cap is None:
            yield from stream
        else:
            yield from capped(stream, source.cap, config, dataset, split, rng)


def capped(
    stream: Iterator[Sample], cap: int, config: Config, dataset: str, split: str, rng: np.random.Generator
) -> Iterator[Sample]:
    """Uniform sample of at most `cap` frames across all sequences of a dataset, without loading them first"""
    sequences = datasets.load(config, dataset, split)
    total = sum(len(frame_boxes(s)) if dataset != "halmstad" else len(s.images or []) for s in sequences)
    keep = cap / total if total > cap else 1.0
    for sample in stream:
        if rng.random() < keep:
            yield sample


def write_split(samples: Iterator[Sample], root: Path, split: str, classes: list[str]) -> int:
    index = {name: i for i, name in enumerate(classes)}
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    count = 0
    for count, (image, boxes) in enumerate(samples, start=1):
        stem = f"{split}_{count:06d}"
        height, width = image.shape[:2]
        cv2.imwrite(str(root / "images" / split / f"{stem}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = [
            f"{index[label]} {x / width:.6f} {y / height:.6f} {w / width:.6f} {h / height:.6f}"
            for label, x, y, w, h in boxes
            if label in index
        ]
        (root / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return count


def gallery(root: Path, split: str, path: Path, n: int = 36, seed: int = 0) -> None:
    images = sorted((root / "images" / split).glob("*.jpg"))
    picks = [images[i] for i in np.random.default_rng(seed).choice(len(images), min(n, len(images)), replace=False)]
    tiles = []
    for image_path in picks:
        image = cv2.resize(imread(image_path), (256, 256))
        for line in (root / "labels" / split / f"{image_path.stem}.txt").read_text().splitlines():
            _, x, y, w, h = map(float, line.split())
            cv2.rectangle(
                image,
                (int((x - w / 2) * 256) - 3, int((y - h / 2) * 256) - 3),
                (int((x + w / 2) * 256) + 3, int((y + h / 2) * 256) + 3),
                (0, 0, 255),
                1,
            )
        tiles.append(image)
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 6)
    rows = [np.hstack(tiles[i : i + 6]) for i in range(0, len(tiles), 6)]
    cv2.imwrite(str(path), np.vstack(rows))


def build(config: Config, name: str) -> Path:
    """Write the training set for a model recipe; existing sets are reused"""
    root = config.cache / "yolo" / name
    spec = root / "dataset.yaml"
    if spec.exists():
        return spec
    shutil.rmtree(root, ignore_errors=True)  # dataset.yaml is written last; anything without it is partial
    recipe = config.training.models[name]
    rng = np.random.default_rng(config.training.tiles.seed)
    stats: Counter = Counter()
    counts = {
        split: write_split(sample_stream(config, name, split, rng, stats), root, split, recipe.classes)
        for split in ("train", "val")
    }
    for split in counts:
        if counts[split]:
            gallery(root, split, root / f"gallery_{split}.jpg")
    (root / "summary.json").write_text(json.dumps({"tiles": counts, **stats}, indent=2))
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(recipe.classes))
    spec.write_text(f"path: {root.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    return spec


def detection_set(config: Config, dataset: str, classes: list[str]) -> Path:
    """Full images of a box-labelled test split, as a YOLO set whose class list matches the model"""
    root = config.cache / "detection" / f"{dataset}-{len(classes)}"
    spec = root / "dataset.yaml"
    if spec.exists():
        return spec
    index = {name: i for i, name in enumerate(classes)}
    (root / "images" / "test").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "test").mkdir(parents=True, exist_ok=True)
    for sequence in datasets.load(config, dataset, "test"):
        boxes = frame_boxes(sequence)
        images = sequence.images or []
        sizes: dict[int, tuple[int, int]] = {}
        for frame, image in enumerate(images[: config.max_frames]):
            stem = f"{Path(sequence.name).name}_{frame:06d}"
            link = root / "images" / "test" / f"{stem}{image.suffix}"
            if not link.exists():
                link.symlink_to(image.resolve())
            if frame not in sizes:
                height, width = imread(image).shape[:2] if frame in boxes else (1, 1)
                sizes[frame] = (width, height)
            width, height = sizes[frame]
            lines = [
                f"{index[label]} {x / width:.6f} {y / height:.6f} {w / width:.6f} {h / height:.6f}"
                for label, x, y, w, h in boxes.get(frame, [])
                if label in index
            ]
            (root / "labels" / "test" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    spec.write_text(f"path: {root.resolve()}\ntrain: images/test\nval: images/test\nnames:\n{names}\n")
    return spec
