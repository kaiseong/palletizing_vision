"""Lazy Intel RealSense D435 live/record boundary.

Importing this module does not import :mod:`pyrealsense2`.  The SDK is loaded
only when a live adapter is started, keeping replay and tests hardware-free.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import time
from typing import Any, Mapping

import numpy as np

from .models import CameraIntrinsics
from .session import (
    FactoryExtrinsics,
    RecordedFrame,
    SessionMetadata,
    StreamProfile,
)


class RealSenseUnavailableError(RuntimeError):
    pass


def load_realsense_sdk() -> Any:
    try:
        return importlib.import_module("pyrealsense2")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RealSenseUnavailableError(
            "pyrealsense2 is unavailable; install the RealSense SDK Python binding "
            "for Python 3.12 and connect the D435 over USB 3 before using live/record"
        ) from exc


@dataclass(frozen=True, slots=True)
class D435StreamConfig:
    depth_width: int = 640
    depth_height: int = 480
    color_width: int = 640
    color_height: int = 480
    fps: int = 30
    align_color_to_depth: bool = True
    warmup_frames: int = 30

    def __post_init__(self) -> None:
        if min(
            self.depth_width,
            self.depth_height,
            self.color_width,
            self.color_height,
            self.fps,
        ) <= 0:
            raise ValueError("stream dimensions and FPS must be positive")
        if self.warmup_frames < 0:
            raise ValueError("warmup_frames cannot be negative")


def _video_profile(profile: Any) -> Any:
    method = getattr(profile, "as_video_stream_profile", None)
    return method() if callable(method) else profile


def _intrinsics_from_profile(profile: Any, fps: int) -> CameraIntrinsics:
    video = _video_profile(profile)
    intrinsics = video.get_intrinsics()
    return CameraIntrinsics(
        width=int(intrinsics.width),
        height=int(intrinsics.height),
        fps=int(fps),
        fx=float(intrinsics.fx),
        fy=float(intrinsics.fy),
        cx=float(getattr(intrinsics, "ppx", getattr(intrinsics, "cx", 0.0))),
        cy=float(getattr(intrinsics, "ppy", getattr(intrinsics, "cy", 0.0))),
        distortion_model=str(getattr(intrinsics, "model", "none")),
        coeffs=tuple(float(v) for v in getattr(intrinsics, "coeffs", ())),
    )


def _extrinsics_from_profiles(
    source_profile: Any,
    target_profile: Any,
    *,
    source_name: str,
    target_name: str,
) -> FactoryExtrinsics:
    extrinsics = source_profile.get_extrinsics_to(target_profile)
    return FactoryExtrinsics(
        target_stream=target_name,
        source_stream=source_name,
        rotation=tuple(float(v) for v in extrinsics.rotation),
        translation_m=tuple(float(v) for v in extrinsics.translation),
    )


def _device_info(device: Any, sdk: Any, name: str, default: str = "unknown") -> str:
    info = getattr(getattr(sdk, "camera_info", object()), name, None)
    if info is None:
        return default
    try:
        supports = getattr(device, "supports", None)
        if callable(supports) and not supports(info):
            return default
        return str(device.get_info(info))
    except (AttributeError, RuntimeError, TypeError):
        return default


def _sensor_option(sensor: Any, sdk: Any, name: str) -> Any:
    if sensor is None:
        return None
    option = getattr(getattr(sdk, "option", object()), name, None)
    if option is None:
        return None
    try:
        supports = getattr(sensor, "supports", None)
        if callable(supports) and not supports(option):
            return None
        return sensor.get_option(option)
    except (AttributeError, RuntimeError, TypeError):
        return None


def _first_color_sensor(device: Any, sdk: Any) -> Any | None:
    direct = getattr(device, "first_color_sensor", None)
    if callable(direct):
        try:
            return direct()
        except (AttributeError, RuntimeError, TypeError):
            pass
    query = getattr(device, "query_sensors", None)
    name_info = getattr(getattr(sdk, "camera_info", object()), "name", None)
    if not callable(query) or name_info is None:
        return None
    try:
        for sensor in query():
            supports = getattr(sensor, "supports", None)
            if callable(supports) and not supports(name_info):
                continue
            if "rgb" in str(sensor.get_info(name_info)).lower():
                return sensor
    except (AttributeError, RuntimeError, TypeError):
        return None
    return None


class RealSenseAdapter:
    def __init__(
        self,
        stream_config: D435StreamConfig | None = None,
        *,
        sdk: Any | None = None,
    ) -> None:
        self.stream_config = stream_config or D435StreamConfig()
        self._sdk = sdk
        self._pipeline: Any | None = None
        self._pipeline_profile: Any | None = None
        self._depth_profile: Any | None = None
        self._color_profile: Any | None = None
        self._depth_sensor: Any | None = None
        self._color_sensor: Any | None = None
        self._align: Any | None = None
        self._profile_metadata: dict[str, Any] | None = None

    @property
    def started(self) -> bool:
        return self._pipeline is not None

    def start(self) -> "RealSenseAdapter":
        if self.started:
            return self
        sdk = self._sdk if self._sdk is not None else load_realsense_sdk()
        self._sdk = sdk
        pipeline = sdk.pipeline()
        config = sdk.config()
        settings = self.stream_config
        config.enable_stream(
            sdk.stream.depth,
            settings.depth_width,
            settings.depth_height,
            sdk.format.z16,
            settings.fps,
        )
        config.enable_stream(
            sdk.stream.color,
            settings.color_width,
            settings.color_height,
            sdk.format.bgr8,
            settings.fps,
        )
        try:
            pipeline_profile = pipeline.start(config)
        except Exception as exc:
            raise RealSenseUnavailableError(
                f"cannot start D435 640x480@{settings.fps}: {exc}"
            ) from exc
        try:
            device = pipeline_profile.get_device()
            depth_sensor = device.first_depth_sensor()
            color_sensor = _first_color_sensor(device, sdk)
            depth_profile = _video_profile(pipeline_profile.get_stream(sdk.stream.depth))
            color_profile = _video_profile(pipeline_profile.get_stream(sdk.stream.color))
            self._pipeline = pipeline
            self._pipeline_profile = pipeline_profile
            self._depth_sensor = depth_sensor
            self._color_sensor = color_sensor
            self._depth_profile = depth_profile
            self._color_profile = color_profile
            self._align = sdk.align(sdk.stream.depth) if settings.align_color_to_depth else None
            depth_intrinsics = _intrinsics_from_profile(depth_profile, settings.fps)
            color_intrinsics = _intrinsics_from_profile(color_profile, settings.fps)
            for _ in range(settings.warmup_frames):
                pipeline.wait_for_frames()
            self._profile_metadata = {
                "camera_serial": _device_info(device, sdk, "serial_number"),
                "camera_firmware": _device_info(device, sdk, "firmware_version"),
                "usb_type": _device_info(device, sdk, "usb_type_descriptor"),
                "depth_scale_m": float(depth_sensor.get_depth_scale()),
                "depth_profile": StreamProfile("depth", "z16", depth_intrinsics),
                "color_profile": StreamProfile("color", "bgr8", color_intrinsics),
                "depth_to_color": _extrinsics_from_profiles(
                    depth_profile,
                    color_profile,
                    source_name="depth",
                    target_name="color",
                ),
                "color_to_depth": _extrinsics_from_profiles(
                    color_profile,
                    depth_profile,
                    source_name="color",
                    target_name="depth",
                ),
                "capture_options": {
                    "exposure": _sensor_option(depth_sensor, sdk, "exposure"),
                    "gain": _sensor_option(depth_sensor, sdk, "gain"),
                    "depth_auto_exposure": _sensor_option(
                        depth_sensor, sdk, "enable_auto_exposure"
                    ),
                    "emitter_enabled": _sensor_option(depth_sensor, sdk, "emitter_enabled"),
                    "laser_power": _sensor_option(depth_sensor, sdk, "laser_power"),
                    "visual_preset": _sensor_option(depth_sensor, sdk, "visual_preset"),
                    "color_exposure": _sensor_option(color_sensor, sdk, "exposure"),
                    "color_gain": _sensor_option(color_sensor, sdk, "gain"),
                    "color_auto_exposure": _sensor_option(
                        color_sensor, sdk, "enable_auto_exposure"
                    ),
                },
            }
        except Exception:
            pipeline.stop()
            self._pipeline = None
            raise
        return self

    def session_metadata(
        self,
        *,
        robot_state: Mapping[str, Any],
        nominal_transform: Mapping[str, Any],
        table: Mapping[str, Any],
        annotation: Mapping[str, Any] | None = None,
    ) -> SessionMetadata:
        if self._profile_metadata is None:
            raise RuntimeError("start the RealSense adapter before requesting metadata")
        return SessionMetadata(
            **self._profile_metadata,
            robot_state=dict(robot_state),
            nominal_transform=dict(nominal_transform),
            table=dict(table),
            annotation={} if annotation is None else dict(annotation),
        )

    def capture(self) -> RecordedFrame:
        if self._pipeline is None:
            raise RuntimeError("start the RealSense adapter before capture")
        frames = self._pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("D435 returned an incomplete RGB-D frameset")
        depth = np.array(depth_frame.get_data(), dtype=np.uint16, copy=True)
        color = np.array(color_frame.get_data(), dtype=np.uint8, copy=True)
        aligned_color: np.ndarray | None = None
        if self._align is not None:
            aligned_frames = self._align.process(frames)
            frame = aligned_frames.get_color_frame()
            if frame:
                aligned_color = np.array(frame.get_data(), dtype=np.uint8, copy=True)
        return RecordedFrame(
            raw_depth_z16=depth,
            raw_color_bgr=color,
            color_on_depth_bgr=aligned_color,
            depth_timestamp_ms=float(depth_frame.get_timestamp()),
            color_timestamp_ms=float(color_frame.get_timestamp()),
            depth_frame_number=int(depth_frame.get_frame_number()),
            color_frame_number=int(color_frame.get_frame_number()),
            hardware_timestamp_ms=float(depth_frame.get_timestamp()),
            system_timestamp_ns=time.time_ns(),
            frame_metadata={
                "depth_timestamp_domain": str(
                    getattr(depth_frame, "get_frame_timestamp_domain", lambda: "unknown")()
                ),
                "color_timestamp_domain": str(
                    getattr(color_frame, "get_frame_timestamp_domain", lambda: "unknown")()
                ),
            },
        )

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
        self._pipeline = None
        self._pipeline_profile = None
        self._color_sensor = None

    def __enter__(self) -> "RealSenseAdapter":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()


D435Adapter = RealSenseAdapter


__all__ = [
    "D435Adapter",
    "D435StreamConfig",
    "RealSenseAdapter",
    "RealSenseUnavailableError",
    "load_realsense_sdk",
]
