"""Frame sources: recorded video, image sequences, live cameras, and a real-time replay clock."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np

from dronetrack.structures import Frame


def open_video(path: Path | str) -> cv2.VideoCapture:
    """FFmpeg-backed capture that applies the rotation stored in the file and decodes on all cores.

    Some phone videos (the P20 Pro cameras of the Cenek datasets) are stored upside down with a
    180-degree display rotation; the labels are in the displayed orientation.
    """
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, os.cpu_count() or 1])
    if not capture.isOpened():
        raise OSError(f"cannot open {path}")
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    return capture


def read_video(
    path: Path | str, max_frames: int | None = None, select: Callable[[int], bool] | None = None
) -> Iterator[Frame]:
    """Decode strictly in order, with presentation timestamps.

    Labels index decoded frames; seeking on variable-frame-rate phone video lands on the wrong
    frame, so there is no random access here. Frames rejected by `select` are decoded (H.264
    requires it) but not converted to BGR, which saves most of the cost.
    """
    capture = open_video(path)
    try:
        t0 = None
        index = 0
        while (max_frames is None or index < max_frames) and capture.grab():
            t = capture.get(cv2.CAP_PROP_POS_MSEC) / 1e3
            t0 = t if t0 is None else t0
            if select is None or select(index):
                ok, image = capture.retrieve()
                if not ok:
                    raise OSError(f"cannot decode frame {index} of {path}")
                yield Frame(index, t - t0, image)
            index += 1
    finally:
        capture.release()


def frame_times(path: Path | str) -> np.ndarray:
    """Presentation time of every frame, in seconds from the first."""
    capture = open_video(path)
    times = []
    while capture.grab():
        times.append(capture.get(cv2.CAP_PROP_POS_MSEC) / 1e3)
    capture.release()
    return np.asarray(times) - times[0]


def imread(path: Path | str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise OSError(f"cannot read {path}")
    return image


def read_images(
    paths: Sequence[Path], times: Sequence[float] | np.ndarray, max_frames: int | None = None
) -> Iterator[Frame]:
    for index, (path, t) in enumerate(zip(paths, times, strict=True)):
        if max_frames is not None and index >= max_frames:
            return
        yield Frame(index, float(t), imread(path))


def read_camera(device: int) -> Iterator[Frame]:
    capture = cv2.VideoCapture(device)
    if not capture.isOpened():
        raise OSError(f"cannot open camera {device}")
    start = time.monotonic()
    try:
        index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                return
            yield Frame(index, time.monotonic() - start, image)
            index += 1
    finally:
        capture.release()


def open_source(source: str, max_frames: int | None = None) -> Iterator[Frame]:
    """A camera index ("0"), a video file, or a stream URL (anything FFmpeg opens)."""
    if source.isdigit():
        return read_camera(int(source))
    return read_video(source, max_frames=max_frames)


class LatestFrame:
    """Hands the consumer the most recent frame only; anything older is dropped and counted.

    With `paced=True` frames are released at their recorded timestamps, so a file behaves like
    a live camera: if processing falls behind, frames are lost exactly as they would be live.
    """

    def __init__(self, frames: Iterable[Frame], paced: bool = False):
        self.dropped: list[int] = []
        self._frames = frames
        self._paced = paced
        self._slot: Frame | None = None
        self._done = False
        self._error: BaseException | None = None
        self._ready = threading.Condition()
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()

    def _produce(self) -> None:
        start = time.monotonic()
        try:
            for frame in self._frames:
                if self._paced:
                    delay = frame.t - (time.monotonic() - start)
                    if delay > 0:
                        time.sleep(delay)
                with self._ready:
                    if self._slot is not None:
                        self.dropped.append(self._slot.index)
                    self._slot = frame
                    self._ready.notify()
        except BaseException as error:  # handed to the consumer, which re-raises it
            self._error = error
        finally:
            with self._ready:
                self._done = True
                self._ready.notify()

    def __iter__(self) -> Iterator[Frame]:
        while True:
            with self._ready:
                while self._slot is None and not self._done:
                    self._ready.wait()
                if self._slot is None:
                    if self._error is not None:
                        raise self._error
                    return
                frame, self._slot = self._slot, None
            yield frame


def to_gray(image: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    """Grayscale copy no wider than `width`, and the factor back to full-frame pixels."""
    scale = min(1.0, width / image.shape[1])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray, 1.0 / scale
