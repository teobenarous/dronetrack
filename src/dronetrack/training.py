"""Detector training (Ultralytics) with resumable checkpoints, and ONNX export."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import torch
from ultralytics import YOLO

from dronetrack import training_data
from dronetrack.config import Config
from dronetrack.detector import resolve_device


def finished(checkpoint: Path) -> bool:
    """Ultralytics marks a completed run by stripping the optimizer state (epoch = -1)"""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return state.get("epoch", -1) == -1


def train(config: Config, name: str) -> Path:
    best = config.weights(name)
    last = best.with_name("last.pt")
    if last.exists() and finished(last) and best.exists():
        return best
    if last.exists():
        YOLO(str(last)).train(resume=True)
        return best

    recipe = config.training.models[name]
    d = config.training.defaults
    data = training_data.build(config, name)
    base = config.cache / "weights" / recipe.base
    base.parent.mkdir(parents=True, exist_ok=True)
    YOLO(str(base)).train(
        data=str(data),
        epochs=recipe.epochs,
        imgsz=d.imgsz,
        batch=d.batch,
        workers=d.workers,
        cache=d.cache,
        patience=d.patience,
        close_mosaic=d.close_mosaic,
        seed=d.seed,
        deterministic=False,  # bitwise reproducibility is unavailable on MPS anyway
        device=resolve_device(config.pipeline.device),
        project=str((config.runs / "train").resolve()),
        name=name,
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    return best


def export(config: Config, name: str) -> Path:
    onnx = config.weights(name, "onnx")
    if not onnx.exists():
        YOLO(str(config.weights(name))).export(
            format="onnx", imgsz=config.pipeline.detector.crop, dynamic=True, simplify=False
        )
    return onnx
