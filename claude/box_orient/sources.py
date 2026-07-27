"""Unified RGB-D frame sources: offline recordings and a live RealSense camera.

Both yield ``Frame(bgr, depth_m, intr, index, meta)`` so the estimator is
identical offline and on-robot. The offline format is the
``box-perception-recording-v2`` layout produced by the KETI recording scripts
(rgb/*.npy BGR, depth/*.depth.npz key ``depth_m`` in metres, manifest.json).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .geometry import CameraIntrinsics


@dataclass
class Frame:
    bgr: np.ndarray
    depth_m: np.ndarray
    intr: CameraIntrinsics
    index: int
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Offline recording
# --------------------------------------------------------------------------- #
class RecordingSource:
    """Replay a box-perception-recording-v2 session directory."""

    def __init__(self, session_dir: str | Path) -> None:
        self.dir = Path(session_dir)
        manifest_path = self.dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in {self.dir}")
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        intr_map = self.manifest.get("intrinsics") or self.manifest.get("color_intrinsics")
        if not intr_map:
            raise ValueError("manifest has no color intrinsics")
        self.intr = CameraIntrinsics.from_mapping(intr_map)
        self.depth_scale_m = self.manifest.get("depth_scale_m_per_unit")
        index_path = self.dir / "index.jsonl"
        self.records: list[dict[str, Any]] = [
            json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]

    def __len__(self) -> int:
        return len(self.records)

    @property
    def camera_name(self) -> str:
        return str(self.manifest.get("camera", {}).get("name", ""))

    def frame(self, index: int) -> Frame:
        rec = self.records[index]
        bgr = _load_rgb(self.dir / rec["rgb_path"])
        depth = _load_depth(self.dir / rec["depth_path"], self.depth_scale_m)
        return Frame(bgr, depth, self.intr, index, rec)

    def frames(self, *, start: int = 0, stop: int | None = None, step: int = 1) -> Iterator[Frame]:
        stop = len(self.records) if stop is None else min(stop, len(self.records))
        for i in range(start, stop, step):
            yield self.frame(i)


def _load_rgb(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        img = np.load(path)
    else:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise OSError(f"failed to read image {path}")
    return np.ascontiguousarray(img)


def _load_depth(path: Path, depth_scale_m: float | None) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path) as data:
            key = "depth_m" if "depth_m" in data else list(data.keys())[0]
            depth = np.asarray(data[key], dtype=np.float32)
    else:
        depth = np.asarray(np.load(path), dtype=np.float32)
    # v2 stores depth already in metres; guard against a raw-unit .npy just in
    # case (large integer-like values) by applying the scale.
    if depth_scale_m and np.nanmax(depth) > 100.0:
        depth = depth * float(depth_scale_m)
    return depth


# --------------------------------------------------------------------------- #
# Live RealSense (D435 / D405)
# --------------------------------------------------------------------------- #
class RealsenseSource:
    """Live pyrealsense2 stream with depth aligned to colour. Import is lazy so
    the offline path never needs the SDK. Use as a context manager."""

    def __init__(
        self,
        *,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        serial_number: str | None = None,
        align_to_color: bool = True,
        warmup_frames: int = 30,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.serial_number = serial_number
        self.align_to_color = align_to_color
        self.warmup_frames = warmup_frames
        self._rs = None
        self._pipeline = None
        self._align = None
        self._depth_scale = 1.0
        self.intr: CameraIntrinsics | None = None
        self._index = 0

    def __enter__(self) -> "RealsenseSource":
        import pyrealsense2 as rs

        self._rs = rs
        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial_number:
            config.enable_device(self.serial_number)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        profile = pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        i = color_profile.get_intrinsics()
        self.intr = CameraIntrinsics(float(i.fx), float(i.fy), float(i.ppx), float(i.ppy), int(i.width), int(i.height))
        self._align = rs.align(rs.stream.color) if self.align_to_color else None
        self._pipeline = pipeline
        for _ in range(self.warmup_frames):
            pipeline.wait_for_frames()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

    def read(self, timeout_ms: int = 5000) -> Frame | None:
        assert self._pipeline is not None and self.intr is not None
        frames = self._pipeline.wait_for_frames(timeout_ms)
        if self._align is not None:
            frames = self._align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None
        bgr = np.ascontiguousarray(np.asanyarray(color_frame.get_data()))
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_m = depth_raw.astype(np.float32) * self._depth_scale
        frame = Frame(bgr, depth_m, self.intr, self._index, {})
        self._index += 1
        return frame

    def frames(self) -> Iterator[Frame]:
        while True:
            frame = self.read()
            if frame is not None:
                yield frame
