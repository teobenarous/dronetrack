"""Closed-form sizing numbers for the report: pixels on target, and range error by method

uv run python scripts/error_budget.py          # writes runs/analysis/error_budget.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd

WIDTH = 1920  # sensor width in pixels
DRONE = 0.35  # characteristic size of a small quadcopter, m
RANGES = [50, 100, 200, 500, 1000, 2000]


def focal(hfov_deg: float, width: int = WIDTH) -> float:
    return width / 2 / np.tan(np.radians(hfov_deg) / 2)


def pixels_on_target() -> pd.DataFrame:
    rows = {f"HFOV {h} deg": [focal(h) * DRONE / z for z in RANGES] for h in (90, 60, 20, 5, 2)}
    return pd.DataFrame(rows, index=[f"{z} m" for z in RANGES]).T.round(1)


def range_errors(sigma_theta: float = 0.2e-3, sigma_d: float = 0.25, sigma_s: float = 1.0) -> pd.DataFrame:
    """1-sigma range error: two-station triangulation, a 1 m stereo rig, and a size prior.

    Triangulation: Z^2 * sigma * theta / B (perpendicular geometry). Stereo: Z^2 * sigma * d / (f B).
    Size prior: Z = f * D / s with D unknown in [0.25, 1] m; the 5-95 % spread of Z is shown.
    """
    rng = np.random.default_rng(0)
    f_zoom = focal(5)
    rows = []
    for z in RANGES:
        s_true = f_zoom * DRONE / z
        s = np.clip(s_true + rng.normal(0, sigma_s, 50_000), 0.3, None)
        size_prior = f_zoom * rng.uniform(0.25, 1.0, 50_000) / s
        lo, hi = np.percentile(size_prior, [5, 95])
        rows.append(
            {
                "range_m": z,
                "triangulation_B100_m": z**2 * sigma_theta / 100,
                "triangulation_B300_m": z**2 * sigma_theta / 300,
                "stereo_B1_HFOV5_m": z**2 * sigma_d / (f_zoom * 1.0),
                "size_prior_5_95_m": f"{lo:.0f}-{hi:.0f}",
                "sync_10ms_at_30mps_m": 30 * 0.010,
            }
        )
    return pd.DataFrame(rows).round(2)


def main() -> None:
    out = Path("runs/analysis")
    out.mkdir(parents=True, exist_ok=True)
    pixels, errors = pixels_on_target(), range_errors()
    text = [
        "Apparent size of a 0.35 m drone (pixels, 1920 px wide sensor)",
        "",
        pixels.to_string(),
        "",
        r"Range error ($\sigma$) by method; size prior as a 5-95 % interval",
        "",
        errors.to_string(index=False),
        "",
    ]
    (out / "error_budget.txt").write_text("\n".join(text))
    pixels.to_csv(out / "pixels_on_target.csv")
    errors.to_csv(out / "range_errors.csv", index=False)
    print("\n".join(text))


if __name__ == "__main__":
    main()
