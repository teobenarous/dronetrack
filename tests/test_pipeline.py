import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from dronetrack import metrics
from dronetrack.config import MethodConfig, load_config
from dronetrack.datasets import annotation_table
from dronetrack.pipeline import Pipeline, track_sequence
from dronetrack.video import LatestFrame, frame_times, read_video

CONFIGS = Path(__file__).parents[1] / "configs"


def synthetic_video(path: Path, n: int = 90) -> list[tuple[float, float]]:
    """A dark dot crossing a noisy sky at 30 fps; returns its true positions"""
    rng = np.random.default_rng(0)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 30, (640, 360))
    truth = []
    for k in range(n):
        image = np.full((360, 640, 3), 150, np.uint8) + rng.integers(0, 4, (360, 640, 3), dtype=np.uint8)
        x, y = 60 + 5 * k, 120 + 40 * np.sin(k / 15)
        cv2.circle(image, (int(x), int(y)), 3, (30, 30, 30), -1)
        truth.append((x, y))
        writer.write(image)
    writer.release()
    return truth


def test_video_reader_is_sequential_with_timestamps(tmp_path):
    synthetic_video(tmp_path / "v.mp4", n=30)
    frames = list(read_video(tmp_path / "v.mp4"))
    assert [f.index for f in frames] == list(range(30))
    assert np.allclose(np.diff([f.t for f in frames]), 1 / 30, atol=1e-3)
    assert np.allclose(frame_times(tmp_path / "v.mp4"), [f.t for f in frames], atol=1e-3)


def test_motion_tracker_follows_a_dot(tmp_path):
    truth = synthetic_video(tmp_path / "v.mp4")
    config = load_config(CONFIGS)
    config.pipeline.motion.work_width = 640
    config.pipeline.motion.history = 30
    pipeline = Pipeline(config, MethodConfig(kind="motion-kf"))
    predictions = track_sequence(pipeline, read_video(tmp_path / "v.mp4"))
    gt = annotation_table(
        [
            {"frame": k, "x": x, "y": y, "w": np.nan, "h": np.nan, "label": "drone", "visible": True, "track": 0}
            for k, (x, y) in enumerate(truth)
        ]
    )
    m = metrics.tracking_metrics(metrics.align(predictions, gt))
    assert m["precision_20"] > 0.8
    assert predictions.latency_ms.notna().all()


def test_latest_frame_drops_when_the_consumer_is_slow(tmp_path):
    synthetic_video(tmp_path / "v.mp4", n=30)
    source = LatestFrame(read_video(tmp_path / "v.mp4"), paced=True)
    seen = []
    for frame in source:
        seen.append(frame.index)
        time.sleep(0.1)
    assert seen[-1] == 29
    assert len(seen) + len(source.dropped) == 30
    assert len(source.dropped) > 10


def test_latest_frame_reraises_source_errors():
    def broken():
        yield from ()
        raise OSError("camera unplugged")

    with pytest.raises(OSError, match="unplugged"):
        list(LatestFrame(broken()))
