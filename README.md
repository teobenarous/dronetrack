# dronetrack

Real-time detection and tracking of a single drone from a ground camera. SPAICE technical challenge, work in progress.

## Setup

- Python 3.13, [uv](https://docs.astral.sh/uv/)
- `uv sync`

## Data

Not tracked by git. Expected layout:

```
data/raw/
├── cenek/      # Li et al. 2020, multi-view drone tracking (main dataset)
├── halmstad/   # Svanström et al. 2021, visible clips (drone, bird, airplane, helicopter)
├── dut/        # Zhao et al. 2022, DUT Anti-UAV detection set
└── fred/       # Magrini et al. 2025, FRED (sky-facing subset)
```

- **cenek**: https://github.com/CenekAlbl/drone-tracking-datasets (clone, then `unpack.sh`)
  - paper: https://doi.org/10.1109/IROS45743.2020.9341479 ([arXiv:2003.04784](https://arxiv.org/abs/2003.04784))
- **halmstad**: https://github.com/DroneDetectionThesis/Drone-detection-dataset (`Data/Video_V`)
  - paper: https://doi.org/10.1016/j.dib.2021.107521
- **dut**: https://github.com/wangdongdut/DUT-Anti-UAV (detection set: train / val / test, Google Drive links)
  - paper: [arXiv:2205.10851](https://arxiv.org/abs/2205.10851)
- **fred**: https://huggingface.co/datasets/GabrieleMagrini/FRED, fetched with `bash scripts/fetch_fred.sh`
  - paper: [arXiv:2506.05163](https://arxiv.org/abs/2506.05163), code: https://github.com/miccunifi/FRED

## Scripts

- `scripts/fetch_fred.sh`: downloads the FRED subset (55 sequences, RGB frames and boxes)
- `scripts/bench_train.py`: measures training and inference speed on Apple Silicon (MPS)
