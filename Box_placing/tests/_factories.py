"""Deterministic builders for placement/controller tests.

Every helper is hardware free: no ``rby1_sdk``, no camera, no threads.  The
nominal scene keeps the held carton bottom 90 mm above the stack top so a
descent plan is admissible under the 50 mm clearance floor.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from parcel_pose_placing.pallet_control import (
    ArmStreamMode,
    MeasuredRobotState,
    PalletControlConfig,
    RBY1PalletController,
)
from parcel_pose_placing.pallet_place import (
    READY_HOLD_MODE,
    PlacementConfig,
    PlacementDescentPlan,
    PlacementInput,
)


# Nominal held-carton scene ---------------------------------------------------
RIGHT_EEF_XYZ = (0.450, -0.130, 0.300)
LEFT_EEF_XYZ = (0.450, 0.130, 0.300)
EEF_SEPARATION_M = 0.260

BOX_BOTTOM_Z_M = 0.200
BOX_BOTTOM_SIGMA_M = 0.004
STACK_TOP_Z_M = 0.110
STACK_TOP_SIGMA_M = 0.004

GAP_M = BOX_BOTTOM_Z_M - STACK_TOP_Z_M
GAP_SIGMA_M = BOX_BOTTOM_SIGMA_M + STACK_TOP_SIGMA_M
MIN_DELTA_M = (BOX_BOTTOM_Z_M - BOX_BOTTOM_SIGMA_M) - (
    STACK_TOP_Z_M + STACK_TOP_SIGMA_M
)
MAX_DELTA_M = (BOX_BOTTOM_Z_M + BOX_BOTTOM_SIGMA_M) - (
    STACK_TOP_Z_M - STACK_TOP_SIGMA_M
)

# Joint layout used by the fake measured state.  Index blocks mirror the
# RB-Y1 Model M ordering the controller expects from ``model.*_idx``.
MOBILITY_INDICES = np.arange(0, 4, dtype=np.int64)
TORSO_INDICES = np.arange(4, 10, dtype=np.int64)
RIGHT_ARM_INDICES = np.arange(10, 17, dtype=np.int64)
LEFT_ARM_INDICES = np.arange(17, 24, dtype=np.int64)
HEAD_INDICES = np.arange(24, 26, dtype=np.int64)
JOINT_COUNT = 26


def transform(xyz: tuple[float, float, float]) -> np.ndarray:
    """Return an identity-rotation transform at ``xyz``."""

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return matrix


def measured_state(
    *,
    sequence: int = 7,
    received_monotonic_s: float = 100.0,
    right_xyz: tuple[float, float, float] = RIGHT_EEF_XYZ,
    left_xyz: tuple[float, float, float] = LEFT_EEF_XYZ,
) -> MeasuredRobotState:
    position = np.zeros(JOINT_COUNT, dtype=np.float64)
    config = PalletControlConfig()
    position[TORSO_INDICES] = np.asarray(config.ready_pose.torso_rad)
    position[RIGHT_ARM_INDICES] = np.asarray(config.ready_pose.right_arm_rad)
    position[LEFT_ARM_INDICES] = np.asarray(config.ready_pose.left_arm_rad)
    position[HEAD_INDICES] = np.asarray(config.ready_pose.head_rad)
    return MeasuredRobotState(
        sequence=int(sequence),
        received_monotonic_s=float(received_monotonic_s),
        robot_timestamp_s=float(received_monotonic_s),
        position_rad=position,
        velocity_radps=np.zeros(JOINT_COUNT, dtype=np.float64),
        is_ready=np.ones(JOINT_COUNT, dtype=np.bool_),
        T_base_head=transform((0.10, 0.0, 1.20)),
        T_base_right_eef=transform(right_xyz),
        T_base_left_eef=transform(left_xyz),
        T_odom_base=np.eye(3, dtype=np.float64),
        base_twist_w_vx_vy=(0.0, 0.0, 0.0),
        wheel_max_abs_radps=0.0,
        T_base_torso=transform((0.0, 0.0, 0.90)),
    )


def offline_controller(
    config: PalletControlConfig | None = None,
    fk_provider: Any = None,
) -> RBY1PalletController:
    """Build a controller that never touches the SDK, stream, or threads.

    ``fk_provider`` is needed only by paths that compute forward kinematics at a
    demonstrated posture.
    """

    controller = RBY1PalletController(
        execute=False, config=config, fk_provider=fk_provider
    )
    controller._indices = {
        "mobility": MOBILITY_INDICES,
        "torso": TORSO_INDICES,
        "right_arm": RIGHT_ARM_INDICES,
        "left_arm": LEFT_ARM_INDICES,
        "head": HEAD_INDICES,
    }
    return controller


def descent_plan(
    *,
    planned_delta_z_m: float | None = None,
    right_xyz: tuple[float, float, float] = RIGHT_EEF_XYZ,
    left_xyz: tuple[float, float, float] = LEFT_EEF_XYZ,
    freeze_monotonic_s: float = 100.0,
    plan_id: str = "test-plan",
    **overrides: Any,
) -> PlacementDescentPlan:
    delta = (
        GAP_M * PlacementConfig().descent_fraction
        if planned_delta_z_m is None
        else float(planned_delta_z_m)
    )
    right_eef = transform(right_xyz)
    left_eef = transform(left_xyz)
    right_target = np.array(right_eef, copy=True)
    left_target = np.array(left_eef, copy=True)
    right_target[2, 3] -= delta
    left_target[2, 3] -= delta
    fields: dict[str, Any] = {
        "plan_id": plan_id,
        "freeze_monotonic_s": freeze_monotonic_s,
        "planned_delta_z_m": delta,
        "min_delta_z_m": MIN_DELTA_M,
        "max_delta_z_m": MAX_DELTA_M,
        "gap_m": GAP_M,
        "gap_uncertainty_m": GAP_SIGMA_M,
        "box_bottom_z_lower_bound_m": BOX_BOTTOM_Z_M - BOX_BOTTOM_SIGMA_M,
        "stack_top_z_upper_bound_m": STACK_TOP_Z_M + STACK_TOP_SIGMA_M,
        "stack_plane_z_base_m": STACK_TOP_Z_M,
        "stack_plane_uncertainty_m": STACK_TOP_SIGMA_M,
        "stack_plane_timestamp_s": freeze_monotonic_s - 0.10,
        "stack_plane_sequence": 3,
        "bilateral_eef_timestamp_s": freeze_monotonic_s - 0.10,
        "bilateral_eef_state_sequence": 7,
        "right_eef_base": right_eef,
        "left_eef_base": left_eef,
        "right_target_base": right_target,
        "left_target_base": left_target,
        "valid": True,
        "rejection_reason": None,
        "source": "test",
    }
    fields.update(overrides)
    return PlacementDescentPlan(**fields)


def placement_input(
    *,
    now_s: float = 100.0,
    sequence: int = 1,
    controller_arm_mode: str = READY_HOLD_MODE,
    ready_posture_verified: bool = True,
    controller_target_ack: bool = True,
    right_xyz: tuple[float, float, float] = RIGHT_EEF_XYZ,
    left_xyz: tuple[float, float, float] = LEFT_EEF_XYZ,
    box_bottom_z_base_m: float = BOX_BOTTOM_Z_M,
    stack_top_z_base_m: float = STACK_TOP_Z_M,
    right_target_base: Any | None = None,
    left_target_base: Any | None = None,
    **overrides: Any,
) -> PlacementInput:
    gap = box_bottom_z_base_m - stack_top_z_base_m
    fields: dict[str, Any] = {
        "now_s": now_s,
        "feedback_timestamp_s": now_s - 0.10,
        "right_eef_base": transform(right_xyz),
        "left_eef_base": transform(left_xyz),
        "arrived_hold": True,
        "post_zero_wheel_stop": True,
        "zero_command_ack": True,
        "measured_state_fresh": True,
        "controller_stream_healthy": True,
        "controller_arm_mode": controller_arm_mode,
        "ready_posture_verified": ready_posture_verified,
        "controller_target_ack": controller_target_ack,
        "right_target_base": right_target_base,
        "left_target_base": left_target_base,
        "allow_vision_geometry_release": True,
        "predicted_box_bottom_gap_m": gap,
        "predicted_box_bottom_gap_uncertainty_m": GAP_SIGMA_M,
        "gap_observation_timestamp_s": now_s - 0.10,
        "gap_observation_sequence": int(sequence),
        "box_bottom_z_base_m": box_bottom_z_base_m,
        "box_bottom_z_uncertainty_m": BOX_BOTTOM_SIGMA_M,
        "stack_top_z_base_m": stack_top_z_base_m,
        "stack_top_uncertainty_m": STACK_TOP_SIGMA_M,
        "stack_plane_z_base_m": stack_top_z_base_m,
        "stack_plane_uncertainty_m": STACK_TOP_SIGMA_M,
        "stack_plane_timestamp_s": now_s - 0.10,
        "stack_plane_sequence": int(sequence),
        "bilateral_eef_timestamp_s": now_s - 0.10,
        "bilateral_eef_state_sequence": int(sequence),
        "descent_plan_source": "test",
    }
    fields.update(overrides)
    return PlacementInput(**fields)


def loaded_hold_target(
    controller: RBY1PalletController,
    state: MeasuredRobotState,
    *,
    squeeze_offset_m: float | None = None,
) -> Any:
    squeeze = (
        controller.config.placement_squeeze_offset_m
        if squeeze_offset_m is None
        else float(squeeze_offset_m)
    )
    return controller._make_cartesian_arm_target(
        state,
        mode=ArmStreamMode.CARTESIAN_LOADED_HOLD,
        base_z_offset_m=0.0,
        squeeze_offset_m=squeeze,
        release_spread_m=0.0,
    )


def rotate_axis_about_z(
    axis: tuple[float, float, float],
    degrees: float,
) -> np.ndarray:
    angle = math.radians(float(degrees))
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    return rotation @ np.asarray(axis, dtype=np.float64)
