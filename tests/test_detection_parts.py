import cv2
import numpy as np

from dronetrack.config import DetectorConfig, MotionConfig
from dronetrack.detector import CropScheduler, cut, sweep_positions
from dronetrack.motion import MotionDetector
from dronetrack.structures import Detection
from dronetrack.training_data import box_from_point, shift, tile_around


def test_sweep_covers_the_frame():
    positions = sweep_positions(1920, 640, 64)
    assert positions[0] == 0 and positions[-1] == 1280
    assert all(b - a <= 640 for a, b in zip(positions, positions[1:], strict=False))
    assert sweep_positions(500, 640, 64) == [0]


def test_crop_plan_stays_inside_the_frame_and_cycles():
    scheduler = CropScheduler(DetectorConfig(crop=640, max_motion_crops=2))
    shape = (1080, 1920)
    blobs = [Detection(1900, 1070, 4, 4, 0.9, source="motion"), Detection(1890, 1060, 4, 4, 0.8, source="motion")]
    crops = scheduler.plan(shape, np.array([5.0, 5.0]), blobs)
    assert [c.reason for c in crops] == ["track", "motion", "sweep"]  # second blob already covered
    assert all(0 <= c.x0 <= 1920 - 640 and 0 <= c.y0 <= 1080 - 640 for c in crops)
    first_sweeps = {(c.x0, c.y0) for _ in range(6) for c in scheduler.plan(shape, None, []) if c.reason == "sweep"}
    assert len(first_sweeps) == 6


def test_cut_pads_small_images():
    patch = cut(
        np.ones((100, 200, 3), np.uint8), CropScheduler(DetectorConfig(crop=640)).centered(10, 10, (100, 200), "x")
    )
    assert patch.shape == (640, 640, 3) and patch[:100, :200].all() and not patch[200:].any()


def test_motion_detector_finds_a_moving_dot():
    detector = MotionDetector(MotionConfig(work_width=640, history=50))
    rng = np.random.default_rng(0)
    found = []
    for k in range(40):
        image = np.full((360, 640, 3), 140, np.uint8) + rng.integers(0, 3, (360, 640, 3), dtype=np.uint8)
        x = 100 + 6 * k
        cv2.circle(image, (x, 150), 3, (20, 20, 20), -1)
        blobs = detector(image)
        if k > 30:
            found.append(min(np.hypot(b.x - x, b.y - 150) for b in blobs) if blobs else np.inf)
    assert np.median(found) < 6


def test_box_from_point_segments_the_blob():
    gray = np.full((200, 200), 180, np.uint8)
    cv2.rectangle(gray, (95, 97), (104, 102), 40, -1)
    (cx, cy, w, h), ok = box_from_point(gray, 100, 100)
    assert ok and abs(cx - 100) < 1.5 and abs(cy - 100) < 1.5 and 10 <= w <= 16 and 6 <= h <= 12
    _, ok = box_from_point(np.full((200, 200), 180, np.uint8), 100, 100)
    assert not ok


def test_tiles_keep_the_target_and_clip_boxes():
    image = np.zeros((1080, 1920, 3), np.uint8)
    rng = np.random.default_rng(0)
    for _ in range(50):
        patch, x0, y0 = tile_around(image, 1900, 20, 640, rng)
        assert patch.shape == (640, 640, 3)
        assert x0 <= 1900 < x0 + 640 and y0 <= 20 < y0 + 640
    assert shift([("drone", -2, 5, 20, 20)], 0, 0, 640) == []  # mostly outside the tile
    assert shift([("drone", 50, 50, 10, 10)], 40, 40, 640) == [("drone", 10.0, 10.0, 10.0, 10.0)]
