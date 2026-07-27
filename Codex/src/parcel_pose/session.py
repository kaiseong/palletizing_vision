"""Versioned recording-session contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .models import BoxModel, CameraIntrinsics


SCHEMA_VERSION = 1


class SessionValidationError(ValueError):
    """An actionable recording/session schema error."""


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise SessionValidationError(f"{context} is missing required field '{key}'")
    return mapping[key]


@dataclass(frozen=True, slots=True)
class StreamProfile:
    stream: str
    format: str
    intrinsics: CameraIntrinsics

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream": self.stream,
            "format": self.format,
            "width": self.intrinsics.width,
            "height": self.intrinsics.height,
            "fps": self.intrinsics.fps,
            "intrinsics": self.intrinsics.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StreamProfile":
        try:
            stream = str(_required(value, "stream", "stream profile"))
            pixel_format = str(_required(value, "format", "stream profile"))
            intrinsics_value = _required(value, "intrinsics", "stream profile")
            if not isinstance(intrinsics_value, Mapping):
                raise SessionValidationError("stream profile intrinsics must be an object")
            augmented = dict(intrinsics_value)
            for key in ("width", "height", "fps"):
                if key in value:
                    augmented[key] = value[key]
            intrinsics = CameraIntrinsics.from_dict(augmented)
            return cls(stream=stream, format=pixel_format, intrinsics=intrinsics)
        except (TypeError, ValueError) as exc:
            if isinstance(exc, SessionValidationError):
                raise
            raise SessionValidationError(f"invalid stream profile: {exc}") from exc


@dataclass(frozen=True, slots=True)
class FactoryExtrinsics:
    target_stream: str
    source_stream: str
    rotation: tuple[float, ...]
    translation_m: tuple[float, float, float]
    convention: str = "active_column_vector_target_from_source"
    rotation_storage: str = "column_major"

    def __post_init__(self) -> None:
        rotation = tuple(float(v) for v in self.rotation)
        translation = tuple(float(v) for v in self.translation_m)
        if len(rotation) != 9 or len(translation) != 3:
            raise SessionValidationError(
                "factory extrinsics require 9 rotation and 3 translation values"
            )
        if not np.all(np.isfinite(rotation + translation)):
            raise SessionValidationError("factory extrinsics must be finite")
        if self.rotation_storage not in {"column_major", "row_major"}:
            raise SessionValidationError(
                "factory extrinsics rotation_storage must be column_major or row_major"
            )
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation_m", translation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_stream": self.target_stream,
            "source_stream": self.source_stream,
            "rotation": list(self.rotation),
            "translation_m": list(self.translation_m),
            "convention": self.convention,
            "rotation_storage": self.rotation_storage,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FactoryExtrinsics":
        return cls(
            target_stream=str(_required(value, "target_stream", "factory extrinsics")),
            source_stream=str(_required(value, "source_stream", "factory extrinsics")),
            rotation=tuple(_required(value, "rotation", "factory extrinsics")),
            translation_m=tuple(_required(value, "translation_m", "factory extrinsics")),
            convention=str(
                value.get("convention", "active_column_vector_target_from_source")
            ),
            rotation_storage=str(value.get("rotation_storage", "column_major")),
        )


@dataclass(frozen=True, slots=True)
class SessionMetadata:
    camera_serial: str
    camera_firmware: str
    usb_type: str
    depth_scale_m: float
    depth_profile: StreamProfile
    color_profile: StreamProfile
    depth_to_color: FactoryExtrinsics
    color_to_depth: FactoryExtrinsics
    capture_options: Mapping[str, Any]
    robot_state: Mapping[str, Any]
    nominal_transform: Mapping[str, Any]
    table: Mapping[str, Any]
    box_model: BoxModel = field(default_factory=BoxModel)
    annotation: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SessionValidationError(
                f"unsupported schema_version {self.schema_version}; expected {SCHEMA_VERSION}"
            )
        scale = float(self.depth_scale_m)
        if not np.isfinite(scale) or scale <= 0.0:
            raise SessionValidationError("depth_scale_m must be finite and positive")
        object.__setattr__(self, "depth_scale_m", scale)
        if self.depth_profile.stream != "depth" or self.color_profile.stream != "color":
            raise SessionValidationError("active stream profiles must be depth and color")
        if self.depth_to_color.source_stream != "depth" or self.depth_to_color.target_stream != "color":
            raise SessionValidationError("depth_to_color direction must be color-from-depth")
        if self.color_to_depth.source_stream != "color" or self.color_to_depth.target_stream != "depth":
            raise SessionValidationError("color_to_depth direction must be depth-from-color")
        for name in (
            "capture_options",
            "robot_state",
            "nominal_transform",
            "table",
            "annotation",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise SessionValidationError(f"{name} must be an object")
            object.__setattr__(self, name, dict(value))
        for key in ("head_joints", "torso_joints", "base_state", "T_base_from_head"):
            _required(self.robot_state, key, "robot_state")
        for key in (
            "target_frame",
            "source_frame",
            "translation_m",
            "euler_zyx_deg",
            "euler_input_order",
            "rotation_formula",
        ):
            _required(self.nominal_transform, key, "nominal_transform")
        for key in ("plane", "config_schema_version"):
            _required(self.table, key, "table")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "camera": {
                "serial": self.camera_serial,
                "firmware": self.camera_firmware,
                "usb_type": self.usb_type,
            },
            "depth_scale_m": self.depth_scale_m,
            "streams": {
                "depth": self.depth_profile.to_dict(),
                "color": self.color_profile.to_dict(),
            },
            "factory_extrinsics": {
                "depth_to_color": self.depth_to_color.to_dict(),
                "color_to_depth": self.color_to_depth.to_dict(),
            },
            "capture_options": dict(self.capture_options),
            "robot_state": dict(self.robot_state),
            "nominal_transform": dict(self.nominal_transform),
            "table": dict(self.table),
            "box_model_m": self.box_model.to_dict(),
            "annotation": dict(self.annotation),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SessionMetadata":
        try:
            camera = _required(value, "camera", "session metadata")
            streams = _required(value, "streams", "session metadata")
            extrinsics = _required(value, "factory_extrinsics", "session metadata")
            if not all(isinstance(v, Mapping) for v in (camera, streams, extrinsics)):
                raise SessionValidationError(
                    "camera, streams, and factory_extrinsics must be objects"
                )
            return cls(
                schema_version=int(_required(value, "schema_version", "session metadata")),
                camera_serial=str(_required(camera, "serial", "camera")),
                camera_firmware=str(_required(camera, "firmware", "camera")),
                usb_type=str(_required(camera, "usb_type", "camera")),
                depth_scale_m=float(_required(value, "depth_scale_m", "session metadata")),
                depth_profile=StreamProfile.from_dict(_required(streams, "depth", "streams")),
                color_profile=StreamProfile.from_dict(_required(streams, "color", "streams")),
                depth_to_color=FactoryExtrinsics.from_dict(
                    _required(extrinsics, "depth_to_color", "factory_extrinsics")
                ),
                color_to_depth=FactoryExtrinsics.from_dict(
                    _required(extrinsics, "color_to_depth", "factory_extrinsics")
                ),
                capture_options=dict(_required(value, "capture_options", "session metadata")),
                robot_state=dict(_required(value, "robot_state", "session metadata")),
                nominal_transform=dict(
                    _required(value, "nominal_transform", "session metadata")
                ),
                table=dict(_required(value, "table", "session metadata")),
                box_model=BoxModel.from_dict(_required(value, "box_model_m", "session metadata")),
                annotation=dict(_required(value, "annotation", "session metadata")),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, SessionValidationError):
                raise
            raise SessionValidationError(f"invalid session metadata: {exc}") from exc


@dataclass(slots=True)
class RecordedFrame:
    raw_depth_z16: NDArray[np.uint16]
    raw_color_bgr: NDArray[np.uint8]
    depth_timestamp_ms: float
    color_timestamp_ms: float
    depth_frame_number: int
    color_frame_number: int
    hardware_timestamp_ms: float | None = None
    system_timestamp_ns: int | None = None
    frame_metadata: Mapping[str, Any] = field(default_factory=dict)
    color_on_depth_bgr: NDArray[np.uint8] | None = None

    def __post_init__(self) -> None:
        depth = np.asarray(self.raw_depth_z16)
        color = np.asarray(self.raw_color_bgr)
        if depth.dtype != np.uint16 or depth.ndim != 2:
            raise SessionValidationError("raw_depth_z16 must be a 2-D uint16 array")
        if color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3:
            raise SessionValidationError("raw_color_bgr must be an HxWx3 uint8 array")
        self.raw_depth_z16 = np.ascontiguousarray(depth)
        self.raw_color_bgr = np.ascontiguousarray(color)
        if self.color_on_depth_bgr is not None:
            aligned = np.asarray(self.color_on_depth_bgr)
            if aligned.dtype != np.uint8 or aligned.shape != (*depth.shape, 3):
                raise SessionValidationError(
                    "color_on_depth_bgr must be uint8 and match raw depth dimensions"
                )
            self.color_on_depth_bgr = np.ascontiguousarray(aligned)
        for name in ("depth_timestamp_ms", "color_timestamp_ms"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise SessionValidationError(f"{name} must be finite")
            setattr(self, name, value)
        self.depth_frame_number = int(self.depth_frame_number)
        self.color_frame_number = int(self.color_frame_number)
        self.frame_metadata = dict(self.frame_metadata)

    def depth_m(self, depth_scale_m: float) -> NDArray[np.float32]:
        return self.raw_depth_z16.astype(np.float32) * float(depth_scale_m)


__all__ = [
    "FactoryExtrinsics",
    "RecordedFrame",
    "SCHEMA_VERSION",
    "SessionMetadata",
    "SessionValidationError",
    "StreamProfile",
]
