"""Evaluation suites: every (method, test set, perturbation) is run causally over each sequence.

Outputs per run directory: one Parquet file of per-frame estimates per sequence (reused if
present), per_sequence.csv, and metrics.json with the means over sequences and the precision
curve.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import cv2
import numpy as np
import pandas as pd
from rich.console import Console
from ultralytics import YOLO

from dronetrack import datasets, metrics, training_data
from dronetrack.config import Config, Perturbation
from dronetrack.datasets import Sequence
from dronetrack.detector import resolve_device
from dronetrack.pipeline import Pipeline, track_sequence
from dronetrack.structures import Frame

console = Console(stderr=True)


def perturb(frames: Iterable[Frame], p: Perturbation, seed: int = 0) -> Iterator[Frame]:
    """Downscaling emulates range (apparent size / k); gain, gamma and noise emulate low light."""
    rng = np.random.default_rng(seed)
    lut = np.clip(255 * p.gain * (np.arange(256) / 255) ** p.gamma, 0, 255).astype(np.uint8)
    kernel = np.ones((1, p.blur), np.float32) / p.blur if p.blur > 1 else None
    for frame in frames:
        image = frame.image
        if p.downscale > 1:
            image = cv2.resize(image, None, fx=1 / p.downscale, fy=1 / p.downscale, interpolation=cv2.INTER_AREA)
        if p.gain != 1 or p.gamma != 1:
            image = cv2.LUT(image, lut)
        if kernel is not None:
            image = cv2.filter2D(image, -1, kernel)
        if p.noise_std > 0:
            noisy = image.astype(np.float32) + rng.normal(0, p.noise_std, image.shape).astype(np.float32)
            image = np.clip(noisy, 0, 255).astype(np.uint8)
        yield Frame(frame.index, frame.t, image)


def scaled(annotations: pd.DataFrame, factor: int) -> pd.DataFrame:
    if factor == 1:
        return annotations
    out = annotations.copy()
    out[["x", "y", "w", "h"]] = out[["x", "y", "w", "h"]] / factor
    return out


def eval_sequences(config: Config, name: str) -> list[Sequence]:
    spec = config.experiments.test_sets[name]
    return datasets.load(config, spec.dataset, spec.split, spec.sequences, drone_free=spec.drone_free)


def run_one(config: Config, suite: str, method: str, test_set: str, perturbation: str | None) -> dict:
    s = config.experiments.suites[suite]
    p = config.experiments.perturbations[perturbation] if perturbation else Perturbation()
    directory = config.runs / suite / method / test_set / (perturbation or "clean")
    (directory / "sequences").mkdir(parents=True, exist_ok=True)

    rows, curves = [], []
    for sequence in eval_sequences(config, test_set):
        path = directory / "sequences" / f"{sequence.name.replace('/', '_')}.parquet"
        if not path.exists():
            console.print(f"{suite} {method} {test_set} {perturbation or 'clean'} {sequence.name}")
            camera = sequence.camera_file if p.downscale == 1 else None
            pipeline = Pipeline(config, config.experiments.methods[method], camera)
            frames = perturb(sequence.frames(max_frames=config.max_frames), p)
            partial = path.with_suffix(".partial")  # renamed when complete, so an interrupted run redoes it
            track_sequence(pipeline, frames, replay=s.replay).to_parquet(partial)
            partial.rename(path)
        predictions = pd.read_parquet(path)
        row: dict = {"sequence": sequence.name, **metrics.latency(predictions)}
        if "stale" in predictions:
            row["stale_fraction"] = float(predictions.stale.mean())
        if sequence.drone_free:
            row.update(metrics.false_alarms(predictions))
        else:
            aligned = metrics.align(predictions, scaled(sequence.annotations, p.downscale))
            row.update(metrics.tracking_metrics(aligned))
            curves.append(metrics.precision_curve(aligned))
        rows.append(row)

    per_sequence = pd.DataFrame(rows)
    per_sequence.to_csv(directory / "per_sequence.csv", index=False)
    summary = metrics.summarize(per_sequence)
    curves = [c for c in curves if not np.isnan(c).all()]
    if curves:
        summary["precision_curve"] = np.mean(np.vstack(curves), axis=0).round(4).tolist()
    if "stale_fraction" in per_sequence:
        summary["stale_fraction"] = float(per_sequence.stale_fraction.mean())
    (directory / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def evaluate(config: Config, suite: str) -> None:
    if suite == "detection":
        evaluate_detection(config)
        return
    s = config.experiments.suites[suite]
    for method in s.methods:
        for test_set in s.test_sets:
            perturbations: list[str | None] = [*s.perturbations] or [None]
            for perturbation in perturbations:
                run_one(config, suite, method, test_set, perturbation)


def evaluate_detection(config: Config) -> None:
    """AP50 and AP50-95 of the drone class on full test images (DUT at 1280 px, Halmstad at native size)"""
    spec = config.experiments.detection
    for model in spec.models:
        weights = config.weights(model)
        classes = config.training.models[model].classes
        for dataset in spec.test_sets:
            out = config.runs / "detection" / model / f"{dataset}.json"
            if out.exists():
                continue
            data = training_data.detection_set(config, dataset, classes)
            result = YOLO(str(weights)).val(
                data=str(data),
                split="val",
                imgsz=1280 if dataset == "dut" else 640,
                batch=8,
                device=resolve_device(config.pipeline.device),
                plots=False,
                verbose=False,
                project=str((config.runs / "detection" / model).resolve()),
                name=dataset,
                exist_ok=True,
            )
            # AP is undefined when the split has no drone (possible only in smoke runs): leave it out
            values = {}
            if 0 in result.box.ap_class_index:
                p, r, ap50, ap = result.box.class_result(list(result.box.ap_class_index).index(0))
                values = {"ap50": float(ap50), "ap50_95": float(ap), "precision": float(p), "recall": float(r)}
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"model": model, "dataset": dataset, **values}, indent=2))


def load_metrics(root: Path) -> pd.DataFrame:
    rows = []
    for path in root.glob("*/*/*/*/metrics.json"):
        suite, method, test_set, perturbation = path.parent.relative_to(root).parts
        rows.append(
            {
                "suite": suite,
                "method": method,
                "test_set": test_set,
                "perturbation": perturbation,
                **json.loads(path.read_text()),
            }
        )
    return pd.DataFrame(rows)
