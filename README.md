# dronetrack

Real-time detection and tracking of a single drone from a fixed ground camera.
2D detection and tracking, with each estimate also expressed as a direction (azimuth, elevation) in the camera frame, the input a multi-camera 3D filter would use.

## Setup

- Python 3.13, [uv](https://docs.astral.sh/uv/): `uv sync`
- Data under `data/raw/` (not tracked):

```
data/raw/
├── cenek/      # multi-view drone tracking datasets (main dataset)
├── halmstad/   # drone, bird, airplane, helicopter clips (YOLO-format copy)
├── dut/        # DUT Anti-UAV detection set
└── fred/       # FRED, sky-facing subset
```

- **cenek**: https://github.com/CenekAlbl/drone-tracking-datasets. Paper: [Li et al. 2020](https://arxiv.org/abs/2003.04784)
- **halmstad**: original https://github.com/DroneDetectionThesis/Drone-detection-dataset;
  YOLO-format copy on Kaggle. Paper: [Svanström et al. 2021](https://doi.org/10.1016/j.dib.2021.107521)
- **dut**: https://github.com/wangdongdut/DUT-Anti-UAV (detection set). Paper: [Zhao et al. 2022](https://arxiv.org/abs/2205.10851)
- **fred**: https://huggingface.co/datasets/GabrieleMagrini/FRED, subset via `bash scripts/download_fred.sh`.
  Paper: [Magrini et al. 2025](https://arxiv.org/abs/2506.05163)

## Use

```bash
uv run dronetrack run video.mp4 --show              # track a file, JSON line per frame
uv run dronetrack run 0 --show                      # live camera
uv run dronetrack run video.mp4 --realtime          # replay at the recorded frame rate (drops frames if late)
```

## Experiments

`scripts/run_experiments.sh` runs the stages below in order; each skips finished work and logs to
`runs/logs/`. `--smoke` runs every stage on tiny subsets in a few minutes.

| Stage     | Command                                                                     | Output                                                           |
|-----------|-----------------------------------------------------------------------------|------------------------------------------------------------------|
| data      | `dronetrack data`                                                           | counts and label overlays per split, `data/cache/audit/`         |
| prepare   | `dronetrack prepare <model>`                                                | YOLO training set of native-resolution tiles, `data/cache/yolo/` |
| train     | `dronetrack train <model>`, `dronetrack export <model>`                     | weights and ONNX, `runs/train/`                                  |
| estimate  | `dronetrack estimate motion\|<model>`                                       | tracker parameters from the validation flight, `runs/estimate/`  |
| evaluate  | `dronetrack evaluate comparison\|ablation\|robustness\|realtime\|detection` | per-frame outputs and metrics, `runs/<suite>/`                   |
| benchmark | `dronetrack benchmark`                                                      | latency per frame, `runs/benchmark/`                             |

Configuration lives in `configs/`. Standalone analyses: `scripts/error_budget.py`, `scripts/simulate_triangulation.py`, `scripts/benchmark_training.py`.

## Tests

`uv run pytest`
