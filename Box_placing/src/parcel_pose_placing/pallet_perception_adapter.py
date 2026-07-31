"""Dependency-neutral pallet-pose facade for entrypoint orchestration.

The legacy :mod:`parcel_pose_placing.pallet_perception` module still owns the
current runtime-coupled frame path.  Phase 4 can consume this narrow seam while
the estimator remains responsible for geometry and this module remains limited
to one frame-to-result conversion.  It deliberately has no robot, controller,
stream, or SDK dependency.
"""

from __future__ import annotations

import math
from typing import Any

from parcel_pose_common.models import PoseResult

from .pallet_models import PalletSceneObservation


_INCOMPLETE_SLOT_REASONS = {
    2: "slot_2_pose_unavailable",
    5: "slot_5_pose_unavailable",
    6: "slot_6_pose_unavailable",
}


def _observation_diagnostics(
    scene: PalletSceneObservation,
    *,
    slot: int,
    frame_id: int | None,
) -> dict[str, Any]:
    stack = scene.stack
    return {
        "source": "pallet_scene_observation",
        "slot": slot,
        "frame_id": frame_id,
        "position_source": "stack.center_base" if slot == 1 else None,
        "yaw_source": "stack.yaw_base_rad",
        "stack_se2_source": stack.stack_se2_source,
        "axis_branch": stack.axis_branch,
        "calibration_status": stack.calibration_status,
        "quality": dict(stack.quality),
        "rejection_reasons": list(stack.rejection_reasons),
        "observation": scene.to_dict(),
    }


def _invalid_result(
    *,
    timestamp_s: float,
    reason: str,
    diagnostics: dict[str, Any],
) -> PoseResult:
    return PoseResult(
        x_m=None,
        y_m=None,
        yaw_rad=None,
        valid=False,
        reason=reason,
        timestamp_s=timestamp_s,
        diagnostics=diagnostics,
        frame="base",
    )


def pallet_pose_result(
    scene: PalletSceneObservation,
    *,
    slot: int,
    frame_id: int | None = None,
) -> PoseResult:
    """Adapt one estimator scene to the shared base-frame pose contract.

    Slot 1 is the only currently demonstrated live placing branch. Its x/y
    position is the current observed hole centre used by the existing fine
    servo, and its yaw is the fitted pallet line yaw. Slots 2, 5, and 6 fail
    closed until their independent
    visual references are added; a valid slot-1 scene never grants them pose
    authority.
    """

    if isinstance(slot, bool) or not isinstance(slot, int):
        raise TypeError("slot must be an integer")

    stack = scene.stack
    timestamp_s = float(stack.timestamp_s)
    diagnostics = _observation_diagnostics(scene, slot=slot, frame_id=frame_id)

    if slot in _INCOMPLETE_SLOT_REASONS:
        return _invalid_result(
            timestamp_s=timestamp_s,
            reason=_INCOMPLETE_SLOT_REASONS[slot],
            diagnostics=diagnostics,
        )
    if slot != 1:
        return _invalid_result(
            timestamp_s=timestamp_s,
            reason=f"unsupported_slot_{slot}",
            diagnostics=diagnostics,
        )
    if not stack.valid:
        reason = (
            stack.rejection_reasons[0]
            if stack.rejection_reasons
            else "invalid_pallet_observation"
        )
        return _invalid_result(
            timestamp_s=timestamp_s,
            reason=reason,
            diagnostics=diagnostics,
        )

    missing_fields: list[str] = []
    center = stack.center_base
    if center is None:
        missing_fields.append("center_base")
    yaw = stack.yaw_base_rad
    if yaw is None:
        missing_fields.append("yaw_base_rad")

    x_m: float | None = None
    y_m: float | None = None
    if center is not None:
        try:
            x_m = float(center[0])
            y_m = float(center[1])
        except (IndexError, TypeError, ValueError):
            missing_fields.append("center_base")
        else:
            if not math.isfinite(x_m) or not math.isfinite(y_m):
                missing_fields.append("center_base")
    if yaw is not None and not math.isfinite(float(yaw)):
        missing_fields.append("yaw_base_rad")

    missing_fields = list(dict.fromkeys(missing_fields))
    if missing_fields:
        diagnostics["missing_fields"] = missing_fields
        return _invalid_result(
            timestamp_s=timestamp_s,
            reason="slot_1_pose_missing_fields",
            diagnostics=diagnostics,
        )

    return PoseResult(
        x_m=x_m,
        y_m=y_m,
        yaw_rad=float(yaw),
        valid=True,
        reason="",
        timestamp_s=timestamp_s,
        diagnostics=diagnostics,
        frame="base",
    )


def perceive_pallet_pose(
    rgb: Any,
    depth_m: Any,
    intrinsics: Any,
    T_base_depth: Any,
    *,
    slot: int,
    estimator: Any,
    timestamp_s: float,
    frame_id: int = 0,
    held_box_hint: Any | None = None,
    calibration_status: str = "nominal_ready_assumed",
) -> PoseResult:
    """Run the injected estimator once and return one shared pose result.

    Inputs are already-acquired RGB/depth/intrinsic/transform values.  Camera
    acquisition and all robot lifecycle work remain outside this facade.
    """

    scene = estimator.estimate(
        depth_m,
        intrinsics,
        T_base_depth,
        timestamp_s=timestamp_s,
        frame_id=frame_id,
        color_on_depth_bgr=rgb,
        held_box_hint=held_box_hint,
        calibration_status=calibration_status,
    )
    return pallet_pose_result(scene, slot=slot, frame_id=frame_id)


__all__ = ["pallet_pose_result", "perceive_pallet_pose"]
