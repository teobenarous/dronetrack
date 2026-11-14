"""Typed configuration, loaded from the YAML files in configs/"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CenekConfig(Model):
    root: Path
    splits: dict[str, list[str]]
    calibration_variant: dict[str, str] = {}


class HalmstadConfig(Model):
    root: Path
    fps: float
    stride: int


class DutConfig(Model):
    root: Path


class FredConfig(Model):
    root: Path
    val_sequences: list[str] = []


class DataConfig(Model):
    classes: list[str]
    cenek: CenekConfig
    halmstad: HalmstadConfig
    dut: DutConfig
    fred: FredConfig


class TileConfig(Model):
    size: int = 640
    cenek_frame_stride: int = 10
    negative_fraction: float = 0.2
    seed: int = 0


class TrainDefaults(Model):
    imgsz: int = 640
    batch: int = 16
    workers: int = 8
    cache: str | bool = False
    patience: int = 10
    close_mosaic: int = 5
    seed: int = 0


class SourceConfig(Model):
    cap: int | None = None
    cap_per_sequence: int | None = None


class ModelRecipe(Model):
    base: str
    classes: list[str]
    epochs: int
    sources: dict[str, SourceConfig]


class TrainingConfig(Model):
    tiles: TileConfig
    defaults: TrainDefaults
    models: dict[str, ModelRecipe]


class MotionConfig(Model):
    work_width: int = 1920
    history: int = 300
    var_threshold: float = 25.0
    diff_threshold: int = 12
    min_area: int = 2
    max_area: int = 2500
    max_candidates: int = 4


class DetectorConfig(Model):
    crop: int = 640
    conf: float = 0.25
    nms_iou: float = 0.5
    max_motion_crops: int = 2
    sweep_overlap: int = 64


class TrackerConfig(Model):
    p_detect: float = 0.8
    clutter_density: float = 1e-6
    measurement_std: float = 3.0
    process_noise: float = 2.0e5
    p_survival: float = 0.99
    q_birth: float = 0.2
    gate_probability: float = 0.99
    declare: float = 0.5
    drop: float = 0.1
    init_velocity_std: float = 300.0
    size_smoothing: float = 0.3


class CounterTrackerConfig(Model):
    confirm_hits: int = 3
    max_misses: int = 10


class ByteTrackConfig(Model):
    imgsz: int = 1920
    conf: float = 0.1


class PipelineConfig(Model):
    device: str = "auto"
    motion: MotionConfig = MotionConfig()
    detector: DetectorConfig = DetectorConfig()
    tracker: TrackerConfig = TrackerConfig()
    counter_tracker: CounterTrackerConfig = CounterTrackerConfig()
    bytetrack: ByteTrackConfig = ByteTrackConfig()


class MethodConfig(Model):
    kind: Literal["motion-kf", "bytetrack", "dronetrack"]
    weights: str | None = None
    weights_format: Literal["pt", "onnx"] = "pt"
    device: str | None = None
    motion_crops: bool = True
    tracker: Literal["bernoulli", "counter"] = "bernoulli"


class EvalSetConfig(Model):
    dataset: Literal["cenek", "fred", "halmstad"]
    split: str
    sequences: list[str] | None = None
    drone_free: bool = False


class Perturbation(Model):
    downscale: int = 1
    gain: float = 1.0
    gamma: float = 1.0
    noise_std: float = 0.0
    blur: int = 0


class SuiteConfig(Model):
    methods: list[str]
    test_sets: list[str]
    perturbations: list[str] = []
    replay: bool = False


class DetectionEvalConfig(Model):
    models: list[str]
    test_sets: list[Literal["dut", "halmstad"]]


class ExperimentsConfig(Model):
    methods: dict[str, MethodConfig]
    test_sets: dict[str, EvalSetConfig]
    perturbations: dict[str, Perturbation]
    suites: dict[str, SuiteConfig]
    detection: DetectionEvalConfig


class Config(Model):
    data: DataConfig
    training: TrainingConfig
    pipeline: PipelineConfig
    experiments: ExperimentsConfig
    runs: Path = Path("runs")
    cache: Path = Path("data/cache")
    smoke: bool = False
    # smoke runs cap every sequence and split so the whole chain finishes in minutes on a CPU
    max_frames: int | None = None
    max_sequences: int | None = None

    def weights(self, model: str, fmt: str = "pt") -> Path:
        return self.runs / "train" / model / "weights" / f"best.{fmt}"

    def tracker_params(self, kind: str) -> Path:
        return self.runs / "estimate" / f"{kind}.yaml"


def load_config(directory: Path = Path("configs"), smoke: bool = False) -> Config:
    parts = {
        name: yaml.safe_load((directory / f"{name}.yaml").read_text())
        for name in ("data", "training", "pipeline", "experiments")
    }
    config = Config(**parts)
    return to_smoke(config) if smoke else config


def to_smoke(config: Config) -> Config:
    training = config.training.model_copy(deep=True)
    training.defaults = training.defaults.model_copy(update={"imgsz": 320, "batch": 8, "workers": 2, "cache": False})
    for recipe in training.models.values():
        recipe.base = "yolo26n.pt"
        recipe.epochs = 1
        recipe.sources = {
            name: SourceConfig(cap=min(source.cap or 64, 64), cap_per_sequence=min(source.cap_per_sequence or 8, 8))
            for name, source in recipe.sources.items()
        }
    return config.model_copy(
        update={
            "training": training,
            "runs": Path("runs/smoke"),
            "cache": Path("data/cache/smoke"),
            "smoke": True,
            "max_frames": 150,
            "max_sequences": 1,
        }
    )
