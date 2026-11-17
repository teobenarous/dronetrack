"""The causal per-frame loop: candidates, detection, tracking, and the output record"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from dronetrack.camera import Camera
from dronetrack.config import Config, MethodConfig, TrackerConfig
from dronetrack.detector import ByteTrackBaseline, CropScheduler, Detector, restore_opencv_threads
from dronetrack.motion import MotionDetector
from dronetrack.structures import Detection, Estimate, Frame
from dronetrack.tracker import BernoulliTracker, CounterTracker
from dronetrack.video import LatestFrame


def tracker_config(config: Config, key: str) -> TrackerConfig:
    """Pipeline defaults, overridden by the parameters estimated on the validation flight"""
    path = config.tracker_params(key)
    estimated = yaml.safe_load(path.read_text()) if path.exists() else {}
    return config.pipeline.tracker.model_copy(update=estimated)


class Pipeline:
    def __init__(self, config: Config, method: MethodConfig, camera_file: Path | None = None):
        restore_opencv_threads()
        p = config.pipeline
        device = method.device or p.device
        self.method = method
        self.camera_file = camera_file
        self.camera: Camera | None = None
        self.motion = self.detector = self.scheduler = self.baseline = None

        if method.kind == "bytetrack":
            assert method.weights is not None
            self.baseline = ByteTrackBaseline(config.weights(method.weights), device, p.bytetrack)
            return

        if method.kind == "motion-kf":
            self.motion = MotionDetector(p.motion)
            params = tracker_config(config, "motion")
        else:
            assert method.weights is not None
            if method.motion_crops:
                self.motion = MotionDetector(p.motion)
            self.detector = Detector(config.weights(method.weights, method.weights_format), device, p.detector)
            self.scheduler = CropScheduler(p.detector)
            params = tracker_config(config, method.weights)
        self.tracker = (
            BernoulliTracker(params) if method.tracker == "bernoulli" else CounterTracker(params, p.counter_tracker)
        )

    def step(self, frame: Frame) -> Estimate:
        start = time.perf_counter()
        timings: dict[str, float] = {}

        def lap(name: str, since: float) -> float:
            now = time.perf_counter()
            timings[name] = (now - since) * 1e3
            return now

        if self.camera is None and self.camera_file is not None:
            height, width = frame.image.shape[:2]
            self.camera = Camera.from_json(self.camera_file, (width, height))

        if self.baseline is not None:
            detection = self.baseline(frame.image)
            lap("detector", start)
            estimate = self._baseline_estimate(frame, detection)
        else:
            self.tracker.predict(frame.t)
            mark = start
            candidates: list[Detection] = []
            if self.motion is not None:
                candidates = self.motion(frame.image)
                mark = lap("motion", mark)
            if self.detector is not None and self.scheduler is not None:
                x = self.tracker.track.x
                crops = self.scheduler.plan(frame.image.shape[:2], None if x is None else x[:2], candidates)
                detections = [d for d in self.detector(frame.image, crops) if d.label == 0]
                mark = lap("detector", mark)
            else:
                detections = candidates
            self.tracker.update(detections, frame.t)
            lap("tracker", mark)
            estimate = self._track_estimate(frame)

        estimate.timings = timings
        estimate.latency_ms = (time.perf_counter() - start) * 1e3
        return estimate

    def _track_estimate(self, frame: Frame) -> Estimate:
        track = self.tracker.track
        estimate = Estimate(frame.index, frame.t, self.tracker.declared, float(self.tracker.q))
        if track.x is not None:
            cov = track.P[:2, :2]
            estimate.x, estimate.y = float(track.x[0]), float(track.x[1])
            estimate.w, estimate.h = track.size
            estimate.cov = (cov[0, 0], cov[0, 1], cov[1, 1])
            self._add_direction(estimate, cov)
        return estimate

    def _baseline_estimate(self, frame: Frame, detection: Detection | None) -> Estimate:
        if detection is None:
            return Estimate(frame.index, frame.t, False, 0.0)
        estimate = Estimate(
            frame.index, frame.t, True, detection.score, detection.x, detection.y, detection.w, detection.h
        )
        self._add_direction(estimate, np.diag([detection.w, detection.h]) ** 2 / 12)
        return estimate

    def _add_direction(self, estimate: Estimate, cov: np.ndarray) -> None:
        if self.camera is None:
            return
        angles, angle_cov = self.camera.direction(np.array([estimate.x, estimate.y]), cov)
        estimate.azimuth, estimate.elevation = float(angles[0]), float(angles[1])
        estimate.angle_cov = (angle_cov[0, 0], angle_cov[0, 1], angle_cov[1, 1])


def track_sequence(pipeline: Pipeline, frames: Iterable[Frame], replay: bool = False) -> pd.DataFrame:
    """Run causally over a sequence; one row per frame.

    In replay mode frames arrive at their recorded times and frames missed while busy are
    dropped; they are scored with the last output, as a live system would report them.
    """
    if not replay:
        return pd.DataFrame([pipeline.step(frame).record() for frame in frames])

    source = LatestFrame(frames, paced=True)
    rows = [pipeline.step(frame).record() for frame in source]
    if not rows:
        return pd.DataFrame()
    output = pd.DataFrame(rows).set_index("frame")
    output["stale"] = False
    every = pd.RangeIndex(0, int(max(output.index.max(), *source.dropped, 0)) + 1, name="frame")
    filled = output.reindex(every).ffill()
    filled["stale"] = filled["stale"].fillna(True).astype(bool) | ~every.isin(output.index)
    filled["declared"] = filled["declared"].fillna(False).astype(bool)
    return filled.reset_index()


def draw(image: np.ndarray, estimate: Estimate) -> np.ndarray:
    if np.isnan(estimate.x):
        return image
    color = (0, 220, 0) if estimate.declared else (0, 160, 255)
    size = int(max(np.nan_to_num(max(estimate.w, estimate.h), nan=16.0), 16)) + 12
    x, y = int(estimate.x), int(estimate.y)
    cv2.rectangle(image, (x - size // 2, y - size // 2), (x + size // 2, y + size // 2), color, 2)
    cv2.putText(
        image, f"q={estimate.presence:.2f}", (x - size // 2, y - size // 2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
    )
    return image


def run(
    pipeline: Pipeline, frames: Iterable[Frame], *, live: bool, realtime: bool, show: bool, output: Path | None
) -> None:
    """Stream one JSON record per frame to `output` (or stdout), optionally with a live view.

    Live cameras and real-time replay always process the newest frame; a file without
    `realtime` is processed frame by frame, however long that takes.
    """
    sink = output.open("w") if output else sys.stdout
    source = LatestFrame(frames, paced=realtime and not live) if live or realtime else None
    try:
        for frame in source if source is not None else frames:
            estimate = pipeline.step(frame)
            record = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in estimate.record().items()}
            sink.write(json.dumps(record) + "\n")
            if show:
                cv2.imshow("dronetrack", draw(frame.image, estimate))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if output:
            sink.close()
        if show:
            cv2.destroyAllWindows()
    if source is not None:
        print(f"dropped {len(source.dropped)} frames", file=sys.stderr)
