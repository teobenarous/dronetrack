import numpy as np

from dronetrack.config import CenekConfig, FredConfig
from dronetrack.datasets import cenek, fred


def test_cenek_labels_are_zero_based_and_00_is_not_visible(tmp_path):
    root = tmp_path / "dataset9"
    (root / "detections").mkdir(parents=True)
    (root / "cameras.txt").write_text("cam0 - iphone6\ncam1 - sonyG\n")
    for cam in ("cam0", "cam1"):
        (root / "detections" / f"{cam}.txt").write_text(" frame no. x y\n1 0 0\n2 10.5 20.25\n")
    sequences = cenek.load(
        CenekConfig(root=tmp_path, splits={"test": ["dataset9"]}, calibration_variant={"sonyG": "sonyG_2"}), "test"
    )
    assert [s.name for s in sequences] == ["dataset9/cam0", "dataset9/cam1"]
    a = sequences[0].annotations
    assert a.frame.tolist() == [0, 1] and a.visible.tolist() == [False, True]
    assert sequences[1].camera_file.name == "sonyG_2.json"


def test_fred_time_matching_tolerates_half_a_frame():
    frames = np.arange(10) / 30
    labels = np.array([0.0, 0.034, 0.05, 1.0])
    assert fred.match_frames(labels, frames).tolist() == [0, 1, 2, -1]
    sparse_frames = np.arange(10) / 3  # every 10th frame kept, labels still at 30 Hz
    labels = np.arange(30) / 30
    assert (fred.match_frames(labels, sparse_frames) >= 0).sum() == 3


def test_fred_sequence_reader(tmp_path):
    directory = tmp_path / "train" / "52"
    (directory / "RGB").mkdir(parents=True)
    for k in range(3):
        (directory / "RGB" / f"Video_52_10_00_{k / 30:09.6f}.jpg").touch()
    (directory / "coordinates.txt").write_text("0.033333: 10, 20, 30, 60, 1, DJI Mini 2\n")
    (sequence,) = fred.load(FredConfig(root=tmp_path, val_sequences=["52"]), "val")
    row = sequence.annotations.iloc[0]
    assert (row.frame, row.x, row.y, row.w, row.h) == (1, 20, 40, 20, 40)
    assert np.allclose(sequence.times, [0, 1 / 30, 2 / 30], atol=1e-6)
    assert fred.load(FredConfig(root=tmp_path, val_sequences=["52"]), "train") == []
