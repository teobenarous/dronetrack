"""Tracker parameters estimated on the validation flight rather than tuned

p_detect        share of labelled frames with a detection within `match_px` of the label
clutter         detections farther than that, per pixel of searched area per frame
measurement     robust standard deviation of detection-minus-label residuals
process noise   white-noise acceleration density from the labelled trajectories
"""

from __future__ import annotations

import numpy as np
import yaml

from dronetrack import datasets
from dronetrack.config import Config
from dronetrack.datasets import Sequence
from dronetrack.detector import CropScheduler, Detector, restore_opencv_threads
from dronetrack.motion import MotionDetector
from dronetrack.structures import Detection
from dronetrack.video import frame_times


def robust_std(values: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(values - np.median(values)))) if len(values) else float("nan")


def process_noise(sequence: Sequence, lag_seconds: float = 0.2) -> np.ndarray:
    """Samples of a^2 T with a = (x[k+L] - 2x[k] + x[k-L]) / T^2, T = L * delta

    Under white-noise acceleration of density q, Var(a) = 2q / (3T), so E[a^2 T] = 2q/3. A lag of a
    few frames keeps label jitter (about a pixel) from dominating the second difference.
    """
    labels = sequence.drone_boxes().set_index("frame")[["x", "y"]]
    times = frame_times(sequence.video) if sequence.video is not None else sequence.times
    assert times is not None
    lag = max(1, int(round(lag_seconds / np.median(np.diff(times)))))
    frames = labels.index.to_numpy()
    present = set(frames.tolist())
    usable = [f for f in frames if f - lag in present and f + lag in present and f + lag < len(times)]
    if not usable:
        return np.empty(0)
    f = np.array(usable)
    xy = labels.to_numpy()
    pos = {k: i for i, k in enumerate(frames)}
    prev, cur, nxt = (xy[[pos[k] for k in f + o]] for o in (-lag, 0, lag))
    dt = (times[f + lag] - times[f - lag]) / 2
    accel = (nxt - 2 * cur + prev) / dt[:, None] ** 2
    return (accel**2 * dt[:, None]).ravel()


def match(detections: list[Detection], x: float, y: float, radius: float) -> tuple[Detection | None, int]:
    """Closest detection within `radius` of (x, y), and the number of detections farther away"""
    if not detections:
        return None, 0
    distance = np.hypot(*(np.array([d.xy for d in detections]) - [x, y]).T)
    nearest = int(np.argmin(distance))
    near = int((distance <= radius).sum())
    return (detections[nearest] if distance[nearest] <= radius else None), len(detections) - near


def estimate(config: Config, key: str, match_px: float = 15.0, stride: int = 5, seed: int = 0) -> dict:
    """`key` is "motion" for the motion-only tracker, otherwise a detector model name"""
    restore_opencv_threads()
    sequences = datasets.load(config, "cenek", "val")
    rng = np.random.default_rng(seed)
    hits = frames = clutter = 0
    area = 0.0
    residuals: list[np.ndarray] = []
    q_samples = [process_noise(s) for s in sequences]

    if key == "motion":
        for sequence in sequences:
            motion = MotionDetector(config.pipeline.motion)
            labels = sequence.annotations.set_index("frame")
            for frame in sequence.frames(max_frames=config.max_frames):
                blobs = motion(frame.image)
                if frame.index not in labels.index or not labels.loc[frame.index].visible:
                    continue
                row = labels.loc[frame.index]
                best, far = match(blobs, row.x, row.y, match_px)
                frames, clutter = frames + 1, clutter + far
                area += frame.image.shape[0] * frame.image.shape[1]
                if best is not None:
                    hits += 1
                    residuals.append(best.xy - [row.x, row.y])
    else:
        p = config.pipeline
        detector = Detector(config.weights(key), p.device, p.detector)
        crop = p.detector.crop
        for sequence in sequences:
            labels = sequence.drone_boxes().set_index("frame")
            wanted = set(labels.index[::stride].tolist())
            scheduler = CropScheduler(p.detector)
            for frame in sequence.frames(select=wanted.__contains__, max_frames=config.max_frames):
                row = labels.loc[frame.index]
                jitter = rng.normal(0, 20, 2)
                track = scheduler.centered(row.x + jitter[0], row.y + jitter[1], frame.image.shape[:2], "track")
                sweep = scheduler.plan(frame.image.shape[:2], None, [])[-1]
                found = [d for d in detector(frame.image, [track, sweep]) if d.label == 0]
                best, far = match(found, row.x, row.y, match_px)
                frames, clutter = frames + 1, clutter + far
                area += 2 * crop * crop
                if best is not None and track.contains(row.x, row.y):
                    hits += 1
                    residuals.append(best.xy - [row.x, row.y])

    r = np.array(residuals).reshape(-1, 2)
    q = np.concatenate(q_samples) if q_samples else np.empty(0)
    params = {
        "p_detect": float(np.clip(hits / max(frames, 1), 0.05, 0.99)),
        "clutter_density": float(max(clutter / max(area, 1.0), 1e-9)),
        "measurement_std": float(max(np.nan_to_num(robust_std(r.ravel()), nan=3.0), 1.0)),
        # E[a^2 T] = 2q/3; the top 1 % (label slips) is clipped before averaging
        "process_noise": float(1.5 * np.minimum(q, np.percentile(q, 99)).mean())
        if len(q)
        else config.pipeline.tracker.process_noise,
    }
    path = config.tracker_params(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({**params}, sort_keys=False))
    (path.with_suffix(".counts.yaml")).write_text(
        yaml.safe_dump({"frames": frames, "hits": hits, "clutter": clutter, "searched_px": area, "residuals": len(r)})
    )
    return params
