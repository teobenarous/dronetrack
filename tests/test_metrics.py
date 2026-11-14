import numpy as np
import pandas as pd

from dronetrack import metrics
from dronetrack.datasets import annotation_table


def predictions(rows):
    frame = pd.DataFrame(rows, columns=["frame", "declared", "x", "y", "w", "h"])
    frame["t"] = frame.frame / 30
    return frame


def test_precision_counts_missing_outputs_as_failures():
    gt = annotation_table(
        [
            {"frame": f, "x": 100, "y": 100, "w": np.nan, "h": np.nan, "label": "drone", "visible": f < 3, "track": 0}
            for f in range(4)
        ]
    )
    pred = predictions(
        [
            (0, True, 105, 100, 5, 5),
            (1, True, 130, 100, 5, 5),
            (2, False, np.nan, np.nan, np.nan, np.nan),
            (3, False, np.nan, np.nan, np.nan, np.nan),
        ]
    )
    m = metrics.tracking_metrics(metrics.align(pred, gt))
    assert m["visible_frames"] == 3
    assert np.isclose(m["precision_20"], 1 / 3)
    # visible: hit, miss, miss; hidden frame correctly not declared
    assert np.isclose(m["state_accuracy"], (1 + 0 + 0 + 1) / 4)
    assert "success_auc" not in m


def test_box_metrics_use_iou():
    gt = annotation_table(
        [{"frame": 0, "x": 50, "y": 50, "w": 10, "h": 10, "label": "drone", "visible": True, "track": 0}]
    )
    pred = predictions([(0, True, 50, 50, 10, 10)])
    m = metrics.tracking_metrics(metrics.align(pred, gt))
    assert np.isclose(m["state_accuracy"], 1.0)
    assert np.isclose(m["success_auc"], 20 / 21)  # IoU = 1 exceeds every threshold below 1


def test_closest_of_several_drones_is_the_reference():
    gt = annotation_table(
        [
            {"frame": 0, "x": 10, "y": 10, "w": 4, "h": 4, "label": "drone", "visible": True, "track": 1},
            {"frame": 0, "x": 200, "y": 200, "w": 4, "h": 4, "label": "drone", "visible": True, "track": 2},
        ]
    )
    aligned = metrics.align(predictions([(0, True, 198, 201, 4, 4)]), gt)
    assert np.isclose(aligned.error.iloc[0], np.hypot(2, 1))


def test_false_alarm_onsets_per_minute():
    pred = predictions([(f, f in (3, 4, 10), np.nan, np.nan, np.nan, np.nan) for f in range(1801)])
    m = metrics.false_alarms(pred)
    assert m["false_alarm_onsets"] == 2
    assert np.isclose(m["minutes"], 1.0)
