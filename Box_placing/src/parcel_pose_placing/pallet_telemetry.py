"""Per-frame telemetry records for the pallet placing loop.

These builders turn one frame of runtime state into the JSONL record and the
recovery-contract audit.  They live apart from the control loop so that the loop
reads as control flow, and so a telemetry change cannot alter motion.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from dataclasses import asdict, is_dataclass

from parcel_pose_common.mobile_servo import VelocityCommand
from .pallet_acquisition import (
    AcquisitionOutput,
    HoleGateStatus,
    LCornerGateStatus,
    OdometrySample,
)
from .pallet_models import (
    PalletGeometry,
    PalletSceneObservation,
    Slot1HoleReference,
)
from .pallet_place import PlacementOutput
from .pallet_servo import PalletServoOutput
from parcel_pose_common.output import to_jsonable


def _servo_measurement_source(
    scene: PalletSceneObservation,
    geometry: PalletGeometry,
) -> str:
    if scene.stack.valid:
        return scene.stack.stack_se2_source or "metric_stack_se2"
    coarse = scene.coarse
    if coarse is not None and coarse.fixed_outer_center_base(
        geometry.outer_size_m
    ) is not None:
        return "fixed_outer_l_corner_proxy"
    return "unavailable"
def _placement_telemetry_payload(placement_telemetry: Any | None) -> Any:
    if placement_telemetry is None:
        return None
    if is_dataclass(placement_telemetry):
        return to_jsonable(asdict(placement_telemetry))
    payload = {
        name: getattr(placement_telemetry, name, None)
        for name in (
            "arm_mode",
            "placement_started",
            "source_state_sequence",
            "target_created_monotonic_s",
            "right_T_base_eef_target",
            "left_T_base_eef_target",
            "zero_latched",
            "wheel_stopped",
            "stream_running",
            "target_acknowledged",
            "acknowledged_command_sequence",
            "last_reason",
            "descent_plan_id",
            "lowering_distance_m",
            "release_spread_m",
            "release_axis_base",
            "release_axis_deviation_rad",
        )
        if hasattr(placement_telemetry, name)
    }
    if "arm_mode" in payload:
        mode = payload["arm_mode"]
        payload["arm_mode"] = str(getattr(mode, "value", mode))
    return to_jsonable(payload)
def _recovery_contract_record(
    scene: PalletSceneObservation,
    held: Any,  # HeldPoseProxy; runtime-owned, kept lazy to avoid an import cycle
    *,
    geometry: PalletGeometry,
    estimator_config: Any | None,
    bridge_diagnostics: Mapping[str, Any] | None,
    odometry: OdometrySample | None,
    grip_result: Any | None,
    placement: PlacementOutput | None,
    placement_runtime_diagnostics: Mapping[str, Any] | None,
    measured_state: Any | None,
) -> dict[str, Any]:
    """Build the explicit audit schema used before each commissioning stage."""

    stack = scene.stack
    coarse = scene.coarse
    proxy_center = (
        None
        if coarse is None
        else coarse.fixed_outer_center_base(geometry.outer_size_m)
    )
    stack_center = stack.center_base if stack.valid else proxy_center
    stack_yaw = (
        stack.yaw_base_rad
        if stack.valid
        else None if coarse is None else coarse.yaw_base_rad
    )
    u_right = (
        stack.u_right_base
        if stack.valid
        else None if coarse is None else coarse.u_right_base
    )
    v_far = (
        stack.v_far_base
        if stack.valid
        else None if coarse is None else coarse.v_far_base
    )
    quality: dict[str, Any] = {}
    quality.update(dict(getattr(coarse, "quality", {}) or {}))
    quality.update(dict(getattr(stack, "quality", {}) or {}))
    front_line = None if coarse is None else coarse.front_line
    side_line = None if coarse is None else coarse.side_line
    line_support = {
        "front_m": None
        if front_line is None
        else float(front_line.support_length_m),
        "side_m": None
        if side_line is None
        else float(side_line.support_length_m),
    }
    line_residual = {
        "front_m": None
        if front_line is None
        else float(front_line.p95_residual_m),
        "side_m": None
        if side_line is None
        else float(side_line.p95_residual_m),
    }
    fixed_axis = getattr(
        estimator_config,
        "fixed_approach_v_far_axis_base_xy",
        None,
    )
    fixed_axis_source = getattr(
        estimator_config,
        "fixed_approach_axis_source",
        None,
    )
    placement_plan = None
    if placement is not None:
        placement_plan = getattr(placement, "descent_plan", None)
    if placement_plan is None and placement_runtime_diagnostics is not None:
        placement_plan = placement_runtime_diagnostics.get("placement_descent_plan")
    plan_payload = (
        None
        if placement_plan is None
        else to_jsonable(
            asdict(placement_plan)
            if is_dataclass(placement_plan)
            else placement_plan
        )
    )
    bridge = dict(bridge_diagnostics or {})
    return {
        "stack_se2_source": _servo_measurement_source(scene, geometry),
        "stack_branch": (
            stack.axis_branch
            if stack.axis_branch is not None
            else None if coarse is None else coarse.topology_branch
        ),
        "stack_center_base_m": stack_center,
        "stack_yaw_base_rad": stack_yaw,
        "stack_covariance": quality.get("stack_covariance"),
        "line_support_m": line_support,
        "line_residual_m": line_residual,
        "connection_gap_m": None if coarse is None else coarse.connection_gap_m,
        "opening_crosscheck": {
            "available": stack.opening_size_m is not None,
            "opening_size_m": stack.opening_size_m,
        },
        "fixed_approach_v_far_axis_base_xy": fixed_axis,
        "fixed_approach_axis_source": fixed_axis_source,
        "resolved_u_right_base": u_right,
        "resolved_v_far_base": v_far,
        "fixed_approach_signed_alignment": quality.get(
            "fixed_approach_signed_alignment"
        ),
        "fixed_approach_axis_residual_rad": quality.get(
            "fixed_approach_axis_residual_rad",
            quality.get("front_axis_residual_rad"),
        ),
        "fixed_approach_role_result": quality.get(
            "fixed_approach_role_result",
            "accepted" if stack_center is not None and u_right is not None else "rejected",
        ),
        "carried_box_pose_source": held.source,
        "carried_box_center_base_m": held.center_base_xyz_m,
        "carried_box_yaw_base_rad": held.yaw_base_rad,
        "eef_state_sequence": None
        if measured_state is None
        else getattr(measured_state, "sequence", None),
        "eef_timestamp_s": None
        if measured_state is None
        else getattr(measured_state, "received_monotonic_s", None),
        "fk_disagreement_m": None,
        "fk_disagreement_reason": "per_eef_box_offsets_unconfigured",
        "clearance_source": None
        if grip_result is None
        else getattr(grip_result, "clearance_source", None),
        "box_bottom_z_lower_bound_m": None
        if grip_result is None
        else getattr(grip_result, "box_bottom_z_lower_bound_m", None),
        "stack_top_z_upper_bound_m": None
        if grip_result is None
        else getattr(grip_result, "stack_top_z_upper_bound_m", None),
        "clearance_lower_bound_m": None
        if grip_result is None
        else getattr(grip_result, "clearance_lower_bound_m", None),
        "clearance_rejection_reason": None
        if grip_result is None
        else ";".join(getattr(grip_result, "reasons", ())),
        "dropout_age_s": bridge.get("dropout_age_s"),
        "odometry_prediction_used": bool(
            bridge.get("odometry_prediction_used", False)
        ),
        "odometry_timestamp_s": None if odometry is None else odometry.timestamp_s,
        "odometry_sequence": None
        if odometry is None
        else getattr(odometry, "sequence", None),
        "dropout_ttl_reason": bridge.get("prediction_reason"),
        "placement_descent_plan_id": None
        if not isinstance(plan_payload, Mapping)
        else plan_payload.get("plan_id"),
        "planned_delta_z_m": None
        if not isinstance(plan_payload, Mapping)
        else plan_payload.get("planned_delta_z_m"),
        "placement_gap_m": None
        if not isinstance(plan_payload, Mapping)
        else plan_payload.get("gap_m"),
        "placement_gap_uncertainty_m": None
        if not isinstance(plan_payload, Mapping)
        else plan_payload.get("gap_uncertainty_m"),
        "placement_descent_plan": plan_payload,
    }
def _telemetry_record(
    frame_id: int,
    hardware_timestamp_ms: float,
    scene: PalletSceneObservation,
    held: Any,  # HeldPoseProxy; runtime-owned, kept lazy to avoid an import cycle
    output: PalletServoOutput,
    *,
    execute: bool,
    controller: Any | None,
    acquisition: AcquisitionOutput | None = None,
    l_gate: LCornerGateStatus | None = None,
    hole_gate: HoleGateStatus | None = None,
    stationary_source: str = "unknown",
    odometry: OdometrySample | None = None,
    odometry_error: str | None = None,
    motion_interlocks_ok: bool = False,
    motion_interlock_reason: str = "",
    grip_result: Any | None = None,
    dispatch_result: str = "dry_run_no_actuation",
    T_base_depth: np.ndarray | None = None,
    slot1_hole_reference: Slot1HoleReference | None = None,
    placement: PlacementOutput | None = None,
    placement_runtime_diagnostics: Mapping[str, Any] | None = None,
    loop_timing: Mapping[str, Any] | None = None,
    geometry: PalletGeometry | None = None,
    estimator_config: Any | None = None,
    bridge_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    controller_telemetry: Any | None = None
    measured_state: Any | None = None
    controller_state_error: str | None = None
    if controller is not None:
        controller_telemetry = controller.telemetry()
        try:
            measured_state = controller.get_measured_state()
        except Exception as exc:
            controller_state_error = f"{type(exc).__name__}:{exc}"
    proposal_accepted = dispatch_result == "nonzero_proposal_accepted"
    selected_zero = dispatch_result in {
        "state_requires_persistent_zero",
        "exact_zero_decision",
        "frame_result_stale_selected_zero",
        "motion_interlock_selected_zero",
    }
    selected = (
        output.command
        if proposal_accepted
        else VelocityCommand()
        if selected_zero
        else None
    )
    transmitted = (
        None
        if not execute or controller_telemetry is None
        else getattr(controller_telemetry, "last_sent_mobility", None)
    )
    record: dict[str, Any] = {
        "schema_version": 1,
        "frame_id": frame_id,
        "hardware_timestamp_ms": hardware_timestamp_ms,
        "mode": "actuation_enabled" if execute else "dry_run_no_robot_commands",
        "scene": scene.to_dict(),
        "held_proxy": asdict(held),
        "slot1_hole_reference": (
            None
            if slot1_hole_reference is None
            else slot1_hole_reference.to_dict()
        ),
        "geometry_provenance": {
            "T_base_depth": None
            if T_base_depth is None
            else np.asarray(T_base_depth, dtype=np.float64).tolist(),
            "calibration_status": scene.stack.calibration_status,
        },
        "control_authority": {
            "owner": output.diagnostics.get("controller_owner", "fine_slot1_servo"),
            "stationary_source": stationary_source,
            "parent_motion_interlocks_passed": bool(execute and motion_interlocks_ok),
            "proposal_accepted": proposal_accepted,
            # ``send_cycle`` accepts a proposal for the asynchronous stream
            # owner; it does not wait for the SDK packet acknowledgement.
            "packet_acknowledged": False,
            "motion_authorized": bool(execute and proposal_accepted),
            "dispatch_result": dispatch_result,
            "motion_interlock_reason": motion_interlock_reason,
        },
        "acquisition": None if acquisition is None else acquisition.to_dict(),
        "l_corner_gate": None if l_gate is None else asdict(l_gate),
        "hole_gate": None if hole_gate is None else asdict(hole_gate),
        "grip_clearance_interlock": (
            None if grip_result is None else to_jsonable(grip_result)
        ),
        "odometry": None if odometry is None else asdict(odometry),
        "odometry_error": odometry_error,
        "timing": dict(loop_timing or {}),
        "alignment": {
            "state": output.state.value,
            "arrived": output.arrived,
            "measurement_accepted": output.measurement_accepted,
            "reason": output.reason,
            "proposed_twist": {
                "vx_mps": output.vx_mps,
                "vy_mps": output.vy_mps,
                "wz_radps": output.wz_radps,
            },
            "selected_twist": (
                None
                if selected is None
                else {
                    "vx_mps": selected.vx_mps,
                    "vy_mps": selected.vy_mps,
                    "wz_radps": selected.wz_radps,
                }
            ),
            "transmitted_twist": (
                None
                if transmitted is None
                else {
                    "vx_mps": transmitted.vx_mps,
                    "vy_mps": transmitted.vy_mps,
                    "wz_radps": transmitted.wz_radps,
                }
            ),
            "diagnostics": dict(output.diagnostics),
        },
        "placement": (
            None
            if placement is None
            else {
                "state": placement.state.value,
                "request": placement.request.value,
                "reason": placement.reason,
                "done": placement.done,
                "faulted": placement.faulted,
                "release_authorized": placement.release_authorized,
                "diagnostics": dict(placement.diagnostics),
                "runtime": dict(placement_runtime_diagnostics or {}),
            }
        ),
    }
    if geometry is not None:
        record["recovery_contract"] = _recovery_contract_record(
            scene,
            held,
            geometry=geometry,
            estimator_config=estimator_config,
            bridge_diagnostics=bridge_diagnostics,
            odometry=odometry,
            grip_result=grip_result,
            placement=placement,
            placement_runtime_diagnostics=placement_runtime_diagnostics,
            measured_state=measured_state,
        )
    if controller_telemetry is not None:
        record["whole_body_owner"] = to_jsonable(controller_telemetry)
    if measured_state is not None:
        record["robot_state"] = to_jsonable(measured_state)
    elif controller_state_error is not None:
        record["robot_state_error"] = controller_state_error
    return to_jsonable(record)
