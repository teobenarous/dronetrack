from __future__ import annotations

from dronetrack.config import Config
from dronetrack.datasets import cenek, dut, fred, halmstad
from dronetrack.datasets.base import Sequence, annotation_table

__all__ = ["Sequence", "annotation_table", "load"]


def load(
    config: Config, dataset: str, split: str, names: list[str] | None = None, drone_free: bool = False
) -> list[Sequence]:
    readers = {
        "cenek": lambda: cenek.load(config.data.cenek, split),
        "halmstad": lambda: halmstad.load(config.data.halmstad, split),
        "dut": lambda: dut.load(config.data.dut, split),
        "fred": lambda: fred.load(config.data.fred, split),
    }
    sequences = readers[dataset]()
    if names is not None:
        sequences = [s for s in sequences if any(s.name.endswith(n) for n in names)]
    if drone_free:
        sequences = [s for s in sequences if s.drone_free]
    if config.max_sequences is not None:
        if config.max_frames is not None:  # smoke runs: prefer sequences where the drone shows up early
            sequences.sort(key=lambda s: -(s.drone_boxes().frame < config.max_frames).sum())
        sequences = sequences[: config.max_sequences]
    return sequences
