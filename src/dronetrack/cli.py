from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from dronetrack.config import Config, MethodConfig, load_config

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Real-time single-drone detection and tracking.")
console = Console(stderr=True)
state: dict[str, Config] = {}


@app.callback()
def main(
    smoke: Annotated[bool, typer.Option(help="Tiny subsets and outputs under runs/smoke, to check the chain.")] = False,
    configs: Annotated[Path, typer.Option(help="Directory with the YAML configuration.")] = Path("configs"),
) -> None:
    state["config"] = load_config(configs, smoke=smoke)


def config() -> Config:
    return state["config"]


@app.command()
def data() -> None:
    """Audit every dataset split: counts, frame/label consistency and label overlays"""
    from dronetrack.datasets.audit import audit

    console.print(f"audit written to {audit(config())}")


@app.command()
def prepare(model: str) -> None:
    """Build the YOLO training set of a model recipe (tiles, boxes from points, galleries)"""
    from dronetrack.training_data import build

    console.print(f"training set: {build(config(), model)}")


@app.command()
def train(model: str) -> None:
    """Train a detector; resumes from the last checkpoint if interrupted"""
    from dronetrack.training import train as train_model

    console.print(f"weights: {train_model(config(), model)}")


@app.command()
def export(model: str) -> None:
    """Export trained weights to ONNX."""
    from dronetrack.training import export as export_model

    console.print(f"onnx: {export_model(config(), model)}")


@app.command()
def estimate(key: Annotated[str, typer.Argument(help='"motion" or a detector model name')]) -> None:
    """Estimate tracker parameters on the validation flight"""
    from dronetrack.estimation import estimate as run_estimate

    console.print(run_estimate(config(), key))


@app.command()
def evaluate(
    suite: Annotated[str, typer.Argument(help="comparison, ablation, robustness, realtime or detection")],
) -> None:
    """Run an evaluation suite; finished sequences are reused"""
    from dronetrack.experiments import evaluate as run_suite

    run_suite(config(), suite)


@app.command()
def benchmark(model: str = "yolo26s-mixed", frames: int = 1500) -> None:
    """Latency per frame (decode included) on a 1080p and a 2160p clip"""
    from dronetrack.benchmark import benchmark as run_benchmark

    console.print(run_benchmark(config(), model, frames).to_string(index=False))


@app.command()
def run(
    source: Annotated[str, typer.Argument(help="Video file, stream URL, or camera index such as 0")],
    method: str = "dronetrack",
    weights: Annotated[str | None, typer.Option(help="Model name, overrides the method's")] = None,
    calibration: Annotated[Path | None, typer.Option(help="Camera JSON (K-matrix, distCoeff, resolution)")] = None,
    realtime: Annotated[bool, typer.Option(help="Replay a file at its recorded frame rate")] = False,
    show: bool = False,
    output: Annotated[Path | None, typer.Option(help="JSON lines file; stdout if omitted")] = None,
) -> None:
    """Track a drone in a video or a live stream; one JSON record per frame"""
    from dronetrack.pipeline import Pipeline
    from dronetrack.pipeline import run as run_pipeline
    from dronetrack.video import open_source

    spec: MethodConfig = config().experiments.methods[method]
    if weights is not None:
        spec = spec.model_copy(update={"weights": weights})
    pipeline = Pipeline(config(), spec, calibration)
    run_pipeline(pipeline, open_source(source), live=source.isdigit(), realtime=realtime, show=show, output=output)


if __name__ == "__main__":
    app()
