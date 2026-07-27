"""Strict, perception-only JSON output.

This module is intentionally independent of robot SDKs.  Its recursive key
guard is a safety boundary: perception results cannot be repurposed into a
robot command payload by adding a nested field.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .models import PoseEstimate


class UnsafeOutputError(ValueError):
    """Raised when a payload crosses the perception-only safety boundary."""


_PROHIBITED_FRAGMENTS = (
    "address",
    "command",
    "servo",
    "power",
    "contact",
    "trajectory",
    "grasp",
    "end_effector",
    "endeffector",
    "tcp_target",
    "target_pose",
)


def _normalized_key(key: Any) -> str:
    text = str(key)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def validate_perception_only_keys(payload: Any, *, path: str = "$") -> None:
    """Reject prohibited keys recursively through mappings and sequences."""

    if isinstance(payload, Mapping):
        for raw_key, value in payload.items():
            key = _normalized_key(raw_key)
            if any(fragment in key for fragment in _PROHIBITED_FRAGMENTS):
                raise UnsafeOutputError(f"prohibited output key at {path}.{raw_key}")
            validate_perception_only_keys(value, path=f"{path}.{raw_key}")
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for index, value in enumerate(payload):
            validate_perception_only_keys(value, path=f"{path}[{index}]")


def to_jsonable(value: Any, *, path: str = "$") -> Any:
    """Convert supported values to strict-JSON data while rejecting NaN/Inf."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"non-finite number at {path}")
        return result
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist(), path=path)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return to_jsonable(asdict(value), path=path)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)):
                raise TypeError(f"unsupported JSON key type at {path}: {type(key).__name__}")
            result[str(key)] = to_jsonable(item, path=f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_jsonable(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"unsupported JSON value at {path}: {type(value).__name__}")


def pose_estimate_to_dict(estimate: PoseEstimate) -> dict[str, Any]:
    """Serialize a pose with explicit base-transform and validity gating."""

    result: dict[str, Any] = {
        "timestamp_ms": estimate.timestamp_ms,
        "frame_id": estimate.frame_id,
        "frame": estimate.frame,
        "box_model_m": estimate.box_model.to_dict(),
        "center_plane_xy_m": estimate.center_plane_xy_m,
        "center_depth_m": estimate.center_depth_m,
        "yaw_rad": estimate.yaw_rad,
        "yaw_mod_180_deg": estimate.yaw_mod_180_deg,
        "canonical_reference_deg": estimate.canonical_reference_deg,
        "canonical_residual_deg": estimate.canonical_residual_deg,
        "classification_margin_deg": estimate.classification_margin_deg,
        "long_axis_plane_xy": estimate.long_axis_plane_xy,
        "short_axis_plane_xy": estimate.short_axis_plane_xy,
        "observability": estimate.observability,
        "center_feasible_set": estimate.feasible_set,
        "calibration": {
            "state": estimate.calibration_state.value,
            "base_registration": estimate.base_registration,
            "base_registration_valid": estimate.base_registration_valid,
            "absolute_base_validated": (
                estimate.base_registration_valid
                and estimate.calibration_state.value == "base_validated"
            ),
        },
        "confidence": {
            "geometry_valid": estimate.geometry_valid,
            "full_pose_valid": estimate.full_pose_valid,
            "absolute_base_pose_valid": estimate.absolute_valid,
            "per_field": estimate.per_field_confidence,
            "reasons": list(estimate.reasons),
        },
        "diagnostics": estimate.diagnostics,
    }
    if estimate.base_registration_valid:
        result.update(
            {
                "center_base_xy_m": estimate.center_base_xy_m,
                "long_axis_base_xy": estimate.long_axis_base_xy,
                "short_axis_base_xy": estimate.short_axis_base_xy,
                "long_axis_yaw_base_deg": estimate.yaw_mod_180_deg,
            }
        )
    jsonable = to_jsonable(result)
    validate_perception_only_keys(jsonable)
    return jsonable


def dumps_strict(payload: Any, *, indent: int | None = None, sort_keys: bool = True) -> str:
    converted = to_jsonable(payload)
    validate_perception_only_keys(converted)
    return json.dumps(
        converted,
        allow_nan=False,
        ensure_ascii=False,
        indent=indent,
        sort_keys=sort_keys,
        separators=None if indent is not None else (",", ":"),
    )


def pose_estimate_to_json(estimate: PoseEstimate, *, indent: int | None = None) -> str:
    return dumps_strict(pose_estimate_to_dict(estimate), indent=indent)


__all__ = [
    "UnsafeOutputError",
    "dumps_strict",
    "pose_estimate_to_dict",
    "pose_estimate_to_json",
    "to_jsonable",
    "validate_perception_only_keys",
]
