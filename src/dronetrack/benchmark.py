"""Latency of the full per-frame loop on fixed clips, for each deployment configuration"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator

import numpy as np
import pandas as pd

from dronetrack import datasets
from dronetrack.config import Config, MethodConfig
from dronetrack.pipeline import Pipeline
from dronetrack.structures import Frame
from dronetrack.training import export

CLIPS = {"1080p": "dataset4/cam1", "2160p": "dataset4/cam3"}


def configurations(model: str) -> dict[str, MethodConfig]:
    return {
        "dronetrack / PyTorch / auto": MethodConfig(kind="dronetrack", weights=model),
        "dronetrack / ONNX Runtime / CPU": MethodConfig(
            kind="dronetrack", weights=model, weights_format="onnx", device="cpu"
        ),
        "motion-kf / CPU": MethodConfig(kind="motion-kf"),
    }


def timed(frames: Iterable[Frame]) -> Iterator[tuple[Frame, float]]:
    """Frames with the time spent decoding each one, in ms"""
    iterator = iter(frames)
    while True:
        start = time.perf_counter()
        frame = next(iterator, None)
        if frame is None:
            return
        yield frame, (time.perf_counter() - start) * 1e3


def benchmark(config: Config, model: str = "yolo26s-mixed", frames: int = 1500, warmup: int = 50) -> pd.DataFrame:
    """Decode plus processing time per frame; the real-time factor is video time over wall time"""
    export(config, model)
    n = min(frames, config.max_frames or frames)
    warmup = min(warmup, n // 3)
    rows = []
    for clip, name in CLIPS.items():
        sequence = datasets.load(config, "cenek", "test", [name])[0]
        for label, method in configurations(model).items():
            pipeline = Pipeline(config, method)
            records = [
                (frame.t, decode, pipeline.step(frame)) for frame, decode in timed(sequence.frames(max_frames=n))
            ]
            records = records[warmup:]
            total = np.array([decode + e.latency_ms for _, decode, e in records])
            stages = pd.DataFrame([{"decode": d, **e.timings} for _, d, e in records]).mean()
            video_seconds = records[-1][0] - records[0][0]
            rows.append(
                {
                    "clip": clip,
                    "configuration": label,
                    "frames": len(records),
                    "p50_ms": np.percentile(total, 50),
                    "p95_ms": np.percentile(total, 95),
                    "p99_ms": np.percentile(total, 99),
                    **{f"{stage}_ms": value for stage, value in stages.items()},
                    "realtime_factor": video_seconds / (total[1:].sum() / 1e3),
                }
            )
    table = pd.DataFrame(rows).fillna(0.0).round(2)  # a stage a configuration lacks takes 0 ms
    out = config.runs / "benchmark" / "latency.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return table
