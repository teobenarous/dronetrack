"""Measure YOLO training and inference speed on this machine and project the planned training runs.

    uv run python scripts/bench_train.py [--quick]

Writes runs/bench/results.json and runs/bench/summary.md.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import argparse
import json
import math
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

# Planned runs: model, training images per epoch, validation images, epochs.
PLANNED_RUNS = {
    "M-in (YOLO26-S, suggested data)": ("yolo26s", 12_000, 2_000, 40),
    "M-mix (YOLO26-S, suggested + external)": ("yolo26s", 25_000, 4_000, 30),
    "nano (YOLO26-N, optional CPU model)": ("yolo26n", 25_000, 4_000, 30),
}
LOCAL_BUDGET_H = 10.0


def build_subset(src: Path, out: Path, n_train: int, n_val: int, seed: int = 0) -> Path:
    """Symlink a clip-balanced sample of a YOLO-format dataset into `out`."""
    rng = random.Random(seed)
    shutil.rmtree(out, ignore_errors=True)
    for split, n in (("train", n_train), ("val", n_val)):
        by_clip: dict[str, list[Path]] = {}
        for f in sorted((src / "images" / split).glob("*.jpg")):
            by_clip.setdefault(f.stem.rsplit("_", 1)[0], []).append(f)
        if not by_clip:
            sys.exit(f"no images in {src / 'images' / split}")
        for files in by_clip.values():
            rng.shuffle(files)
        queues = list(by_clip.values())
        picked: list[Path] = []
        while len(picked) < n and any(queues):
            for q in queues:
                if q and len(picked) < n:
                    picked.append(q.pop())
        for kind in ("images", "labels"):
            (out / kind / split).mkdir(parents=True)
        for f in picked:
            (out / "images" / split / f.name).symlink_to(f.resolve())
            label = src / "labels" / split / f"{f.stem}.txt"
            if label.exists():
                (out / "labels" / split / label.name).symlink_to(label.resolve())
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(["uav", "bird", "airplane", "helicopter"]))
    yaml = out / "dataset.yaml"
    yaml.write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    return yaml


@dataclass
class TrainResult:
    model: str
    device: str
    batch: int
    batch_used: int | None = None
    ok: bool = False
    error: str | None = None
    train_images: int | None = None
    val_images: int | None = None
    epochs: list[dict] = field(default_factory=list)
    img_per_s: float | None = None
    val_s_per_img: float | None = None
    save_s: float | None = None
    checkpoint_mb: float | None = None
    peak_mps_gb: float | None = None
    amp: bool | None = None
    loss_finite: bool | None = None
    mps_fallbacks: list[str] = field(default_factory=list)


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def train_once(
    model_name: str,
    batch: int,
    device: str,
    data: Path,
    args,
    warmup: int,
    timed: int,
    fraction: float = 1.0,
    val: bool = True,
) -> TrainResult:
    from ultralytics import YOLO

    res = TrainResult(model_name, device, batch)
    ep: dict = {}
    batch_t: list[float] = []
    peak = 0.0

    def now() -> float:
        sync(device)
        return time.perf_counter()

    def on_train_start(tr):
        res.train_images = len(tr.train_loader.dataset)
        res.amp = bool(tr.amp)

    def on_train_epoch_start(tr):
        ep.clear()
        ep.update(epoch=tr.epoch + 1, t0=now())
        batch_t.clear()

    def on_train_batch_end(tr):
        nonlocal peak
        batch_t.append(now())
        if device == "mps":
            peak = max(peak, torch.mps.driver_allocated_memory() / 1e9)

    def on_train_epoch_end(tr):
        ep["t_train"] = now()
        res.batch_used = tr.batch_size
        k = max(2, len(batch_t) // 10)  # skip start-up batches
        if len(batch_t) > k + 2:
            ep["img_per_s"] = tr.batch_size * (len(batch_t) - 1 - k) / (batch_t[-1] - batch_t[k])
        losses = tr.tloss.values() if isinstance(tr.tloss, dict) else torch.as_tensor(tr.tloss).flatten()
        ep["loss"] = float(sum(float(v) for v in losses))

    def on_val_start(v):
        ep["t_val0"] = now()

    def on_val_end(v):
        ep["t_val1"] = now()
        res.val_images = len(v.dataloader.dataset)

    def on_model_save(tr):
        ep["save_s"] = now() - ep.get("t_val1", ep["t_train"])
        ep["checkpoint_mb"] = Path(tr.last).stat().st_size / 1e6

    def on_fit_epoch_end(tr):
        if any(e["epoch"] == ep["epoch"] for e in res.epochs):
            return  # fired again by the final validation of best.pt
        row = {"epoch": ep["epoch"], "train_s": ep["t_train"] - ep["t0"]}
        if "t_val1" in ep:
            row["val_s"] = ep["t_val1"] - ep["t_val0"]
        row.update({k: ep[k] for k in ("img_per_s", "loss", "save_s", "checkpoint_mb") if k in ep})
        res.epochs.append(row)

    model = YOLO(str(args.out / "weights" / f"{model_name}.pt"))
    for fn in (
        on_train_start,
        on_train_epoch_start,
        on_train_batch_end,
        on_train_epoch_end,
        on_val_start,
        on_val_end,
        on_model_save,
        on_fit_epoch_end,
    ):
        model.add_callback(fn.__name__, fn)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model.train(
                data=str(data),
                epochs=warmup + timed,
                imgsz=640,
                batch=batch,
                device=device,
                workers=args.workers,
                cache="ram",
                fraction=fraction,
                val=val,
                save_period=1,
                close_mosaic=0,
                plots=False,
                project=str(args.out / "train"),
                name=f"{model_name}_{device}_b{batch}",
                exist_ok=True,
                seed=0,
                verbose=False,
            )
            res.ok = True
        except Exception as exc:
            res.error = f"{type(exc).__name__}: {exc}"[:300]
    res.mps_fallbacks = sorted({str(w.message)[:160] for w in caught if "MPS" in str(w.message)})

    timed_eps = [e for e in res.epochs if e["epoch"] > warmup] or res.epochs
    median = lambda key: statistics.median(v) if (v := [e[key] for e in timed_eps if key in e]) else None  # noqa: E731
    res.img_per_s = median("img_per_s")
    res.save_s = median("save_s")
    res.checkpoint_mb = median("checkpoint_mb")
    if (val_s := median("val_s")) and res.val_images:
        res.val_s_per_img = val_s / res.val_images
    losses = [e["loss"] for e in res.epochs if "loss" in e]
    res.loss_finite = bool(losses) and all(math.isfinite(x) for x in losses)
    res.peak_mps_gb = round(peak, 1) if device == "mps" else None
    if device == "mps":
        torch.mps.empty_cache()
    return res


def bench_inference(model_name: str, args, sample: Path) -> list[dict]:
    """End-to-end predict() latency for one crop, per backend and crop size."""
    import cv2
    from ultralytics import YOLO

    pt = args.out / "weights" / f"{model_name}.pt"
    img = cv2.imread(str(sample))
    rows = []
    for size in (640, 320):
        crop = cv2.resize(img, (size, size))
        backends = [("torch-mps", "mps", pt), ("torch-cpu", "cpu", pt)]
        try:
            onnx = Path(YOLO(str(pt)).export(format="onnx", imgsz=size, simplify=False, verbose=False))
            backends.append(("onnxruntime-cpu", "cpu", onnx.rename(onnx.with_name(f"{model_name}_{size}.onnx"))))
        except Exception as exc:
            rows.append({"model": model_name, "size": size, "backend": "onnxruntime-cpu", "error": str(exc)[:200]})
        for name, device, path in backends:
            try:
                m = YOLO(str(path), task="detect")
                for _ in range(10):
                    m.predict(crop, imgsz=size, device=device, verbose=False)
                times = []
                for _ in range(args.infer_iters):
                    t = time.perf_counter()
                    m.predict(crop, imgsz=size, device=device, verbose=False)
                    sync(device)
                    times.append((time.perf_counter() - t) * 1e3)
                rows.append(
                    {
                        "model": model_name,
                        "size": size,
                        "backend": name,
                        "p50_ms": round(float(np.percentile(times, 50)), 2),
                        "p95_ms": round(float(np.percentile(times, 95)), 2),
                    }
                )
            except Exception as exc:
                rows.append({"model": model_name, "size": size, "backend": name, "error": str(exc)[:200]})
    return rows


def project(results: list[TrainResult]) -> list[dict]:
    best: dict[str, TrainResult] = {}
    for r in results:
        if (
            r.ok
            and r.device == "mps"
            and r.img_per_s
            and (r.model not in best or r.img_per_s > best[r.model].img_per_s)
        ):
            best[r.model] = r
    rows = []
    for run, (model, n_train, n_val, epochs) in PLANNED_RUNS.items():
        r = best.get(model)
        if r is None:
            rows.append({"run": run, "error": "no MPS result"})
            continue
        epoch_s = 1.05 * n_train / r.img_per_s + (r.val_s_per_img or 0) * n_val + (r.save_s or 0)
        hours = epochs * epoch_s / 3600
        rows.append(
            {
                "run": run,
                "batch": r.batch_used,
                "img_per_s": round(r.img_per_s, 1),
                "epochs": epochs,
                "min_per_epoch": round(epoch_s / 60, 1),
                "hours": round(hours, 1),
                "local_ok": hours <= LOCAL_BUDGET_H,
            }
        )
    return rows


def system_info() -> dict:
    import ultralytics

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "mps": torch.backends.mps.is_available(),
        "platform": platform.platform(),
    }
    if sys.platform == "darwin":
        sysctl = lambda key: subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()  # noqa: E731
        info["chip"] = sysctl("machdep.cpu.brand_string")
        info["memory_gb"] = round(int(sysctl("hw.memsize")) / 2**30)
    return info


def summary(info: dict, train: list[TrainResult], infer: list[dict], proj: list[dict]) -> str:
    fmt = lambda x, d=1: "–" if x is None else round(x, d)  # noqa: E731
    lines = [
        "# Training-speed benchmark",
        "",
        f"`{json.dumps(info)}`",
        "",
        "## Training",
        "",
        "| model | device | batch | img/s | val s/img | save s | ckpt MB | peak MPS GB | AMP | finite loss "
        "| MPS fallbacks | error |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in train:
        lines.append(
            f"| {r.model} | {r.device} | {r.batch} ({r.batch_used}) | {fmt(r.img_per_s)} | {fmt(r.val_s_per_img, 4)} | "
            f"{fmt(r.save_s, 2)} | {fmt(r.checkpoint_mb)} | {fmt(r.peak_mps_gb)} | {r.amp} | {r.loss_finite} | "
            f"{len(r.mps_fallbacks)} | {r.error or ''} |"
        )
    lines += [
        "",
        "## Projected training time",
        "",
        "| run | batch | img/s | epochs | min/epoch | hours | ≤ 10 h |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in proj:
        lines.append(
            f"| {p['run']} | – | – | – | – | – | {p['error']} |"
            if "error" in p
            else f"| {p['run']} | {p['batch']} | {p['img_per_s']} | {p['epochs']} | {p['min_per_epoch']} | "
            f"{p['hours']} | {'yes' if p['local_ok'] else 'no'} |"
        )
    lines += [
        "",
        "## Inference latency per crop (predict())",
        "",
        "| model | size | backend | p50 ms | p95 ms |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {i['model']} | {i['size']} | {i['backend']} | {i.get('p50_ms', '–')} "
        f"| {i.get('p95_ms', i.get('error', '–'))} |"
        for i in infer
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/raw/halmstad"), help="YOLO-format dataset")
    ap.add_argument("--out", type=Path, default=Path("runs/bench"))
    ap.add_argument("--models", nargs="+", default=["yolo26s", "yolo26n"])
    ap.add_argument("--batches", nargs="+", type=int, default=[16, 32])
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--timed-epochs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--infer-iters", type=int, default=100)
    ap.add_argument("--no-cpu-ref", action="store_true")
    ap.add_argument("--no-inference", action="store_true")
    ap.add_argument("--quick", action="store_true", help="300 images, batch 16, 1 timed epoch")
    args = ap.parse_args()
    if args.quick:
        args.n_train, args.n_val, args.batches, args.timed_epochs, args.infer_iters = 300, 60, [16], 1, 20
        args.no_cpu_ref = True
    if not torch.backends.mps.is_available():
        sys.exit("MPS not available (requires Apple Silicon and macOS >= 14)")

    args.out = args.out.resolve()
    (args.out / "weights").mkdir(parents=True, exist_ok=True)
    info = system_info()
    data = build_subset(args.data, args.out / "data", args.n_train, args.n_val)
    train: list[TrainResult] = []
    infer: list[dict] = []
    t0 = time.time()

    def save() -> None:
        proj = project(train)
        (args.out / "results.json").write_text(
            json.dumps(
                {
                    "system": info,
                    "planned_runs": PLANNED_RUNS,
                    "training": [asdict(r) for r in train],
                    "inference": infer,
                    "projection": proj,
                    "minutes": round((time.time() - t0) / 60, 1),
                },
                indent=2,
            )
        )
        (args.out / "summary.md").write_text(summary(info, train, infer, proj))

    for model in args.models:
        for batch in args.batches:
            print(f"\n== {model}, batch {batch}, mps ==")
            train.append(train_once(model, batch, "mps", data, args, warmup=1, timed=args.timed_epochs))
            save()
    if not args.no_cpu_ref:
        print("\n== CPU reference: yolo26s, batch 16, 200 images ==")
        train.append(
            train_once(
                "yolo26s", 16, "cpu", data, args, warmup=0, timed=1, fraction=min(1.0, 200 / args.n_train), val=False
            )
        )
        save()
    if not args.no_inference:
        sample = next((args.out / "data" / "images" / "val").iterdir())
        for model in args.models:
            infer += bench_inference(model, args, sample)
            save()
    print("\n" + (args.out / "summary.md").read_text())


if __name__ == "__main__":
    main()
