"""Monte Carlo of the two-station design: 3D error of triangulated bearings against range

Error sources: bearing noise per axis, a fixed orientation (calibration) error per station, and
a clock offset between stations while the drone moves. Rays are intersected at the midpoint of
their closest approach and the error is expressed in the reference camera frame.

    uv run python scripts/simulate_triangulation.py    # writes runs/analysis/triangulation.{csv,pdf}
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RANGES = np.array([100, 200, 300, 500, 750, 1000, 1500])
SCENARIOS = {
    "B=100 m, 0.2 mrad": dict(baseline=100, bearing=0.2e-3, orientation=0.0, sync=0.0),
    "B=300 m, 0.2 mrad": dict(baseline=300, bearing=0.2e-3, orientation=0.0, sync=0.0),
    "B=300 m, 0.2 mrad + 0.1 mrad calib.": dict(baseline=300, bearing=0.2e-3, orientation=0.1e-3, sync=0.0),
    "B=300 m, 0.2 mrad + 5 ms sync": dict(baseline=300, bearing=0.2e-3, orientation=0.0, sync=5e-3),
}


def perturb(direction: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0, sigma, direction.shape)
    noise -= (noise * direction).sum(1, keepdims=True) * direction
    out = direction + noise
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def closest_point(c1: np.ndarray, d1: np.ndarray, c2: np.ndarray, d2: np.ndarray) -> np.ndarray:
    w = c1 - c2
    a, b, c = (d1 * d1).sum(1), (d1 * d2).sum(1), (d2 * d2).sum(1)
    d, e = (d1 * w).sum(1), (d2 * w).sum(1)
    z = a * c - b**2
    s, t = (b * e - c * d) / z, (a * e - b * d) / z
    return (c1 + s[:, None] * d1 + c2 + t[:, None] * d2) / 2


def simulate(
    z: float,
    baseline: float,
    bearing: float,
    orientation: float,
    sync: float,
    n: int = 20_000,
    speed: float = 15.0,
    seed: int = 0,
) -> float:
    """RMS 3D error at range z from station A, drone anywhere in a 120 x 40 degree sector"""
    rng = np.random.default_rng(seed)
    azimuth = np.radians(rng.uniform(-60, 60, n))
    elevation = np.radians(rng.uniform(5, 45, n))
    target = z * np.stack(
        [np.sin(azimuth) * np.cos(elevation), np.cos(azimuth) * np.cos(elevation), np.sin(elevation)], axis=1
    )
    a, b = np.zeros(3), np.array([baseline, 0.0, 0.0])
    velocity = speed * perturb(np.tile([1.0, 0.0, 0.0], (n, 1)), 1.0, rng)
    seen_by_b = target + velocity * sync
    da = perturb((target - a) / np.linalg.norm(target - a, axis=1, keepdims=True), bearing, rng)
    db = perturb((seen_by_b - b) / np.linalg.norm(seen_by_b - b, axis=1, keepdims=True), bearing, rng)
    if orientation:
        bias_rng = np.random.default_rng(seed + 1)
        da = perturb(da, orientation, bias_rng)
        db = perturb(db, orientation, bias_rng)
    estimate = closest_point(np.tile(a, (n, 1)), da, np.tile(b, (n, 1)), db)
    return float(np.sqrt(((estimate - target) ** 2).sum(1).mean()))


def main() -> None:
    out = Path("runs/analysis")
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(
        {name: [simulate(z, **p) for z in RANGES] for name, p in SCENARIOS.items()},
        index=pd.Index(RANGES, name="range_m"),
    ).round(2)
    table["closed form B=300 (Z^2 sigma/B)"] = (RANGES**2 * 0.2e-3 / 300).round(2)
    table.to_csv(out / "triangulation.csv")
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for name in table.columns:
        ax.plot(table.index, table[name], "--" if "closed" in name else "-o", ms=3, label=name)
    ax.set(
        xlabel="Range from reference station (m)",
        ylabel="RMS 3D error (m)",
        yscale="log",
        title="Two-station triangulation",
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "triangulation.pdf")
    print(table.to_string())


if __name__ == "__main__":
    main()
