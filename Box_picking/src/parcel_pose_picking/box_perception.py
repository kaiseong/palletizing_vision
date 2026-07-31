"""Dependency-neutral box-pose facade.

This module is deliberately limited to adapting the existing picking
estimator output into :class:`parcel_pose_common.models.PoseResult`.  Camera
ownership and robot lifecycle concerns belong to their respective adapters
and orchestration layers.

Box yaw is a line orientation: ``yaw_rad`` is normalized modulo pi to the
half-open interval ``[0, pi)``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from parcel_pose_common.models import (
    Calibration,
    CameraIntrinsics,
    EstimatorConfig,
    PoseEstimate,
    PoseResult,
)

from .estimator import ParcelPoseEstimator
from .evaluation import BasePoseDiagnostic, base_pose_from_estimate


_BASE_POSE_UNAVAILABLE = "box_base_pose_unavailable"
_BASE_POSE_INVALID = "box_base_pose_invalid"
_BASE_POSE_CONVERSION_FAILED = "box_base_pose_conversion_failed"


def _source_diagnostics(
    estimate: PoseEstimate,
    extra: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Retain estimator quality and capture provenance without key collisions."""

    diagnostics: dict[str, Any] = {
        "source": "parcel_pose_picking.ParcelPoseEstimator",
        "source_frame": estimate.frame,
        "source_frame_id": estimate.frame_id,
        "source_timestamp_ms": estimate.timestamp_ms,
        "source_reasons": list(estimate.reasons),
        "source_validity": {
            "geometry_valid": estimate.geometry_valid,
            "full_pose_valid": estimate.full_pose_valid,
            "base_registration_valid": estimate.base_registration_valid,
            "absolute_valid": estimate.absolute_valid,
        },
        "base_registration": estimate.base_registration,
        "estimator": dict(estimate.diagnostics),
    }
    if extra:
        diagnostics["capture"] = dict(extra)
    return diagnostics


def pose_result_from_base_diagnostic(
    base_pose: BasePoseDiagnostic | None,
    *,
    timestamp_s: float,
    invalid_reason: str = _BASE_POSE_UNAVAILABLE,
    diagnostics: Mapping[str, Any] | None = None,
) -> PoseResult:
    """Adapt one existing base-frame diagnostic without any side effects.

    ``timestamp_s`` is intentionally explicit.  The live controller uses a
    monotonic clock, while recorded sensor timestamps can have a different
    domain; silently converting between those clocks would make freshness
    checks ambiguous.
    """

    output_diagnostics = dict(diagnostics or {})
    if base_pose is None:
        reason = str(invalid_reason).strip() or _BASE_POSE_UNAVAILABLE
        return PoseResult(
            x_m=None,
            y_m=None,
            yaw_rad=None,
            valid=False,
            reason=reason,
            timestamp_s=timestamp_s,
            diagnostics=output_diagnostics,
            frame="base",
        )

    output_diagnostics.setdefault("base_pose", base_pose.to_dict())
    try:
        x_m = float(base_pose.box_center_xyz_m[0])
        y_m = float(base_pose.box_center_xyz_m[1])
        yaw_rad = math.radians(float(base_pose.yaw_mod_180_deg) % 180.0)
        if not all(math.isfinite(value) for value in (x_m, y_m, yaw_rad)):
            raise ValueError("base pose contains a non-finite x/y/yaw value")
    except (IndexError, TypeError, ValueError) as exc:
        output_diagnostics["adapter_error"] = str(exc)
        return PoseResult(
            x_m=None,
            y_m=None,
            yaw_rad=None,
            valid=False,
            reason=_BASE_POSE_INVALID,
            timestamp_s=timestamp_s,
            diagnostics=output_diagnostics,
            frame="base",
        )

    return PoseResult(
        x_m=x_m,
        y_m=y_m,
        yaw_rad=yaw_rad,
        valid=True,
        reason="",
        timestamp_s=timestamp_s,
        diagnostics=output_diagnostics,
        frame="base",
    )


def pose_result_from_estimate(
    estimate: PoseEstimate,
    calibration: Calibration,
    *,
    timestamp_s: float,
    capture_diagnostics: Mapping[str, Any] | None = None,
) -> PoseResult:
    """Convert the current estimator result into the common facade contract.

    Geometry conversion failures become inspectable invalid results.  The
    richer estimator record remains nested under ``diagnostics['estimator']``;
    sensor timestamp and frame provenance remain available alongside it.
    """

    diagnostics = _source_diagnostics(estimate, capture_diagnostics)
    try:
        base_pose = base_pose_from_estimate(estimate, calibration)
    except (ArithmeticError, TypeError, ValueError) as exc:
        diagnostics["adapter_error"] = str(exc)
        return pose_result_from_base_diagnostic(
            None,
            timestamp_s=timestamp_s,
            invalid_reason=_BASE_POSE_CONVERSION_FAILED,
            diagnostics=diagnostics,
        )

    invalid_reason = next(
        (str(reason).strip() for reason in estimate.reasons if str(reason).strip()),
        _BASE_POSE_UNAVAILABLE,
    )
    return pose_result_from_base_diagnostic(
        base_pose,
        timestamp_s=timestamp_s,
        invalid_reason=invalid_reason,
        diagnostics=diagnostics,
    )


def _array_provenance(value: Any) -> dict[str, Any]:
    """Describe a frame input without retaining or logging its pixel payload."""

    provenance: dict[str, Any] = {"provided": value is not None}
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            provenance["shape"] = [int(item) for item in shape]
        except (TypeError, ValueError):
            provenance["shape"] = str(shape)
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        provenance["dtype"] = str(dtype)
    return provenance


def perceive_box_pose(
    rgb: Any,
    depth: Any,
    intrinsics: CameraIntrinsics | None = None,
    calibration: Calibration | None = None,
    *,
    estimator: Any | None = None,
    estimator_config: EstimatorConfig | None = None,
    depth_scale: float | None,
    sensor_timestamp_ms: float,
    frame_id: int,
    timestamp_s: float,
    rgb_support_mask: Any | None = None,
) -> PoseResult:
    """Estimate one already-acquired RGB-D frame exactly once.

    A long-lived ``estimator`` should be injected by live orchestration.  For
    offline/single-frame callers, ``intrinsics`` and ``calibration`` are enough
    to construct the existing deterministic estimator.  Camera acquisition,
    robot construction, streams, and commands remain outside this function.

    ``sensor_timestamp_ms`` is forwarded unchanged to the estimator and kept
    as provenance.  The independent ``timestamp_s`` remains the controller's
    explicit freshness-clock timestamp.
    """

    reused_estimator = estimator is not None
    if estimator is None:
        if intrinsics is None or calibration is None:
            raise ValueError(
                "intrinsics and calibration are required when estimator is not injected"
            )
        estimator = ParcelPoseEstimator(intrinsics, calibration, estimator_config)
    elif estimator_config is not None:
        raise ValueError("estimator_config cannot be combined with an injected estimator")

    resolved_calibration = calibration
    if resolved_calibration is None:
        resolved_calibration = getattr(estimator, "calibration", None)
    if not isinstance(resolved_calibration, Calibration):
        raise ValueError(
            "calibration is required explicitly or on the injected estimator"
        )

    estimate = estimator.estimate(
        depth,
        depth_scale=depth_scale,
        timestamp_ms=sensor_timestamp_ms,
        frame_id=frame_id,
        rgb_support_mask=rgb_support_mask,
    )
    capture_diagnostics = {
        "rgb": _array_provenance(rgb),
        "depth": _array_provenance(depth),
        "rgb_support_mask": _array_provenance(rgb_support_mask),
        "depth_scale_m": depth_scale,
        "sensor_timestamp_ms": sensor_timestamp_ms,
        "frame_id": frame_id,
        "estimator_reused": reused_estimator,
        "intrinsics_source": (
            "argument" if intrinsics is not None else "injected_estimator"
        ),
        "calibration_source": (
            "argument" if calibration is not None else "injected_estimator"
        ),
    }
    return pose_result_from_estimate(
        estimate,
        resolved_calibration,
        timestamp_s=timestamp_s,
        capture_diagnostics=capture_diagnostics,
    )


__all__ = [
    "perceive_box_pose",
    "pose_result_from_base_diagnostic",
    "pose_result_from_estimate",
]
