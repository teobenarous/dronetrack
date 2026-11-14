"""Single-object tracking metrics as used in the anti-UAV literature + operational ones.

All tracking metrics follow one-pass evaluation: a frame where the drone is visible and the
tracker reports nothing counts as a failure. Point-labelled data (e.g. cenek) has no boxes, so
success and normalised precision are box-only and state accuracy uses a distance hit instead
of IoU there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PIXEL_THRESHOLDS = np.arange(0, 51)
NORM_THRESHOLDS = np.linspace(0, 0.5, 51)
IOU_THRESHOLDS = np.linspace(0, 1, 21)


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of centre-size boxes, row by row"""
    ax1, ay1, ax2, ay2 = a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2, a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2
    bx1, by1, bx2, by2 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    w = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    h = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = w * h
    union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def align(predictions: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    """One row per processed frame with prediction and ground truth side by side.

    With several drones in view, the ground truth for a frame is the one closest to the output.
    """
    pred = predictions.set_index("frame")[["declared", "x", "y", "w", "h"]]
    drones = annotations[(annotations.label == "drone") & annotations.visible]
    drones = drones[drones.frame.isin(pred.index)]
    joined = drones.merge(pred, left_on="frame", right_index=True, suffixes=("", "_pred"))
    joined["distance"] = np.hypot(joined.x - joined.x_pred, joined.y - joined.y_pred).fillna(np.inf)
    closest = joined.sort_values("distance").drop_duplicates("frame").set_index("frame")

    out = pred.rename(columns={c: f"p{c}" for c in ("x", "y", "w", "h")})
    out = out.join(closest[["x", "y", "w", "h"]].rename(columns=lambda c: f"g{c}"))
    out["visible"] = out.gx.notna()
    out["declared"] = out.declared.astype(bool)
    hit = out.visible & out.declared
    out["error"] = np.where(hit, np.hypot(out.px - out.gx, out.py - out.gy), np.inf)
    has_box = out.gw.notna()
    out["norm_error"] = np.where(
        hit & has_box,
        np.hypot((out.px - out.gx) / out.gw, (out.py - out.gy) / out.gh),
        np.where(has_box, np.inf, np.nan),
    )
    pred_box = out[["px", "py", "pw", "ph"]].fillna(0).to_numpy()
    gt_box = out[["gx", "gy", "gw", "gh"]].fillna(0).to_numpy()
    out["iou"] = np.where(has_box, np.where(hit, iou(pred_box, gt_box), 0.0), np.nan)
    return out.reset_index()


def tracking_metrics(aligned: pd.DataFrame, tau: float = 20.0) -> dict[str, float]:
    visible = aligned[aligned.visible]
    hidden = aligned[~aligned.visible]
    m: dict[str, float] = {"frames": len(aligned), "visible_frames": len(visible)}
    if len(visible):
        curve = (visible.error.to_numpy()[:, None] <= PIXEL_THRESHOLDS).mean(0)
        m.update(
            {"precision_10": float(curve[10]), "precision_20": float(curve[20]), "precision_auc": float(curve.mean())}
        )
        if visible.iou.notna().any():
            m["success_auc"] = float((visible.iou.to_numpy()[:, None] > IOU_THRESHOLDS).mean(0).mean())
            m["norm_precision_auc"] = float((visible.norm_error.to_numpy()[:, None] <= NORM_THRESHOLDS).mean(0).mean())

    boxes = aligned.iou.notna().any()
    visible_score = visible.iou if boxes else (visible.error <= tau).astype(float)
    hidden_score = (~hidden.declared).astype(float)
    if len(aligned):
        m["state_accuracy"] = float((visible_score.sum() + hidden_score.sum()) / len(aligned))
    m["declared_fraction"] = float(aligned.declared.mean()) if len(aligned) else float("nan")
    return m


def precision_curve(aligned: pd.DataFrame) -> np.ndarray:
    visible = aligned[aligned.visible]
    return (
        (visible.error.to_numpy()[:, None] <= PIXEL_THRESHOLDS).mean(0) if len(visible) else PIXEL_THRESHOLDS * np.nan
    )


def false_alarms(predictions: pd.DataFrame) -> dict[str, float]:
    """For sequences with no drone: declared-track onsets per minute and the declared share of time"""
    declared = predictions.sort_values("frame").declared.astype(bool).to_numpy()
    onsets = int(declared[0]) + int(np.sum(declared[1:] & ~declared[:-1])) if len(declared) else 0
    t = predictions.t.to_numpy()
    minutes = max((t.max() - t.min()) / 60, 1e-9) if len(t) > 1 else 1e-9
    return {"false_alarm_onsets": onsets, "minutes": float(minutes), "declared_fraction": float(declared.mean())}


def latency(predictions: pd.DataFrame) -> dict[str, float]:
    ms = predictions.latency_ms.dropna().to_numpy()
    if not len(ms):
        return {}
    p50, p95, p99 = np.percentile(ms, [50, 95, 99])
    return {
        "latency_p50_ms": float(p50),
        "latency_p95_ms": float(p95),
        "latency_p99_ms": float(p99),
        "throughput_fps": float(1000 / ms.mean()),
    }


def summarize(per_sequence: pd.DataFrame) -> dict[str, float]:
    """Means over sequences (the benchmark convention); false alarms pooled over time"""
    out: dict[str, float] = {"sequences": len(per_sequence)}
    for column in (
        "precision_10",
        "precision_20",
        "precision_auc",
        "success_auc",
        "norm_precision_auc",
        "state_accuracy",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "throughput_fps",
    ):
        if column in per_sequence and per_sequence[column].notna().any():
            out[column] = float(per_sequence[column].mean())
    if "false_alarm_onsets" in per_sequence and per_sequence.false_alarm_onsets.notna().any():
        out["false_alarms_per_min"] = float(per_sequence.false_alarm_onsets.sum() / per_sequence.minutes.sum())
    return out
