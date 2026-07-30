"""Fail-closed RB-Y1 whole-body stream adapter for pallet slot-1 placement.

The module intentionally imports ``rby1_sdk`` only after explicit execution
authorization.  One stream owner sends every robot command: a single ready
transition followed by one fixed-rate combined stream containing torso/head
Position, bilateral Cartesian impedance, and SE(2) mobility commands.

This controller owns only the persistent combined stream.  Higher-level runtime
code sequences hover/alignment, placement descent, and release gates.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import importlib
import json
import math
import threading
import time
from typing import Any, Callable, Mapping, Sequence
import uuid
import warnings

import numpy as np

from .pallet_control_feedback import (
    ComponentFeedbackAck,
    parse_component_feedback,
    raise_for_terminal_feedback,
)
from .pallet_place import PlacementConfig, PlacementDescentPlan
from .pallet_servo import PalletServoConfig


EXPECTED_ROBOT_MODEL = "M"
EXPECTED_ROBOT_VERSION = "1.2"

READY_TORSO_RAD = (0.0, 1.410, -1.514, 0.6295, 0.0, 0.0)
READY_RIGHT_ARM_RAD = (-1.120, -0.383, -0.048, -1.066, -0.824, 1.109, 0.157)
READY_LEFT_ARM_RAD = (-1.120, 0.383, 0.048, -1.066, 0.824, 1.109, -0.157)
READY_HEAD_RAD = (0.0, 0.870)
ARM_STIFFNESS = (150.0,) * 7
TORQUE_POLICY = "sdk_default"

# Slowest admissible slot-1 ready one-shot.  A loaded full-body Position move
# below this is a jerk risk for the carried carton.
READY_TRANSITION_MINIMUM_TIME_FLOOR_S = 3.0

HARD_MAX_LINEAR_SPEED_MPS = 0.08
HARD_MAX_ANGULAR_SPEED_RADPS = 0.10

# Below this |axis_y| the left/right hand order along base Y is not decidable,
# so no release direction may be inferred.
_RELEASE_AXIS_SIGN_EPSILON = 1e-6

TORSO_REFERENCE_LINK = "link_torso_5"
RIGHT_EEF_LINK = "ee_right"
LEFT_EEF_LINK = "ee_left"

_DYN_LINK_NAMES = ("base", TORSO_REFERENCE_LINK, "link_head_2", RIGHT_EEF_LINK, LEFT_EEF_LINK)
_BASE_LINK_INDEX = 0
_TORSO_LINK_INDEX = 1
_HEAD_LINK_INDEX = 2
_RIGHT_EEF_LINK_INDEX = 3
_LEFT_EEF_LINK_INDEX = 4


class PalletControlError(RuntimeError):
    """Base error for the pallet whole-body controller."""


class RobotMotionDisabledError(PalletControlError):
    """Raised when a caller attempts robot I/O without explicit authorization."""


class RobotIdentityError(PalletControlError):
    """Raised when the connected controller is not RB-Y1 Model M v1.2."""


class CommandOwnershipError(PalletControlError):
    """Raised when a non-owner attempts to publish a command."""


class CombinedStreamError(PalletControlError):
    """Raised when the combined command stream cannot be proven healthy."""


class ReadyTransitionError(PalletControlError):
    """Raised when the exactly-once ready transition is invalid or incomplete."""


class MeasuredStateError(PalletControlError):
    """Raised when fresh measured joint/FK/base state is unavailable."""


class HandoffPendingError(PalletControlError):
    """Raised when cleanup is requested before a successor acknowledgement."""


class ControllerPhase(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTED = "CONNECTED"
    PALLET_READY_TRANSITION = "PALLET_READY_TRANSITION"
    STEADY_HOLD = "STEADY_HOLD"
    SHUTDOWN_PENDING = "SHUTDOWN_PENDING"
    HANDOFF_ACKNOWLEDGED = "HANDOFF_ACKNOWLEDGED"
    FAULT_HOLD = "FAULT_HOLD"
    CLOSED = "CLOSED"


def _finite_vector(value: Sequence[float], length: int, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain exactly {length} finite values")
    return tuple(float(item) for item in array)


def _finite_float_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _readonly_matrix(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 transform")
    result = np.array(matrix, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _readonly_se2_matrix(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 3x3 SE(2) transform")
    if not np.allclose(matrix[2], (0.0, 0.0, 1.0), rtol=0.0, atol=1e-6):
        raise ValueError(f"{name} must have homogeneous bottom row [0, 0, 1]")
    rotation = matrix[:2, :2]
    if not np.allclose(rotation.T @ rotation, np.eye(2), rtol=0.0, atol=1e-4):
        raise ValueError(f"{name} rotation must be orthonormal")
    if float(np.linalg.det(rotation)) <= 0.0:
        raise ValueError(f"{name} rotation must be proper")
    result = np.array(matrix, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _rotation_error_rad(left: np.ndarray, right: np.ndarray) -> float:
    relative = left[:3, :3].T @ right[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(cosine)


def _read_field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def resolve_release_axis(
    inter_eef_axis_base: Sequence[float],
    reference_axis_base: Sequence[float],
    *,
    max_deviation_rad: float,
) -> tuple[np.ndarray, float]:
    """Snap a measured inter-EEF axis onto a commanded opening axis.

    Slot-1 release opens the hands along one fixed direction, so the commanded
    target moves on that axis only.  ``reference_axis_base`` is that direction
    expressed in the base frame; the slot-1 path passes the torso-tip frame's
    ``Y`` axis so the opening matches the box-pick grip convention regardless of
    torso yaw.  The returned axis is the unit reference axis signed to agree
    with the measured hand order, and the returned angle is the deviation
    between the measured axis and it.

    This function only reports the deviation; the caller owns the accept/reject
    decision so a single place in the controller can fail closed.
    """

    axis = np.asarray(inter_eef_axis_base, dtype=np.float64).reshape(3)
    reference = np.asarray(reference_axis_base, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(axis)) or not np.all(np.isfinite(reference)):
        raise ValueError("release axes must be finite")
    axis_norm = float(np.linalg.norm(axis))
    reference_norm = float(np.linalg.norm(reference))
    if not math.isfinite(axis_norm) or axis_norm <= 1e-9:
        raise ValueError("inter_eef_axis_base must be a non-zero direction")
    if not math.isfinite(reference_norm) or reference_norm <= 1e-9:
        raise ValueError("reference_axis_base must be a non-zero direction")
    limit = float(max_deviation_rad)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("max_deviation_rad must be finite and positive")
    unit = axis / axis_norm
    reference_unit = reference / reference_norm
    projection = float(unit @ reference_unit)
    if abs(projection) <= _RELEASE_AXIS_SIGN_EPSILON:
        raise ValueError(
            "inter_eef_axis_base is perpendicular to the reference axis, so the "
            "opening direction has no decidable sign"
        )
    sign = 1.0 if projection > 0.0 else -1.0
    release_axis = sign * reference_unit
    deviation_rad = math.acos(float(np.clip(abs(projection), 0.0, 1.0)))
    return release_axis, deviation_rad


@dataclass(frozen=True, slots=True)
class ReadyPose:
    torso_rad: tuple[float, ...] = READY_TORSO_RAD
    right_arm_rad: tuple[float, ...] = READY_RIGHT_ARM_RAD
    left_arm_rad: tuple[float, ...] = READY_LEFT_ARM_RAD
    head_rad: tuple[float, ...] = READY_HEAD_RAD

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "torso_rad", _finite_vector(self.torso_rad, 6, "torso_rad")
        )
        object.__setattr__(
            self,
            "right_arm_rad",
            _finite_vector(self.right_arm_rad, 7, "right_arm_rad"),
        )
        object.__setattr__(
            self,
            "left_arm_rad",
            _finite_vector(self.left_arm_rad, 7, "left_arm_rad"),
        )
        object.__setattr__(
            self, "head_rad", _finite_vector(self.head_rad, 2, "head_rad")
        )


@dataclass(frozen=True, slots=True)
class PlacePose:
    """Operator-demonstrated joint posture that lowers the carried carton.

    Slot-1 placement no longer commands a base-Z descent.  The operator jogs the
    robot to the posture that seats the carton and hands the joint values over;
    the controller turns them into Cartesian EEF targets with forward kinematics,
    because the combined stream is bound to Cartesian arm controllers from its
    first packet and would silently ignore joint targets for the arms.

    The head is deliberately absent: it holds the ready pose throughout.
    """

    torso_rad: tuple[float, ...]
    right_arm_rad: tuple[float, ...]
    left_arm_rad: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "torso_rad", _finite_vector(self.torso_rad, 6, "torso_rad")
        )
        object.__setattr__(
            self,
            "right_arm_rad",
            _finite_vector(self.right_arm_rad, 7, "right_arm_rad"),
        )
        object.__setattr__(
            self, "left_arm_rad", _finite_vector(self.left_arm_rad, 7, "left_arm_rad")
        )

    @classmethod
    def from_degrees(cls, value: Mapping[str, Any]) -> "PlacePose":
        def radians(name: str, length: int) -> tuple[float, ...]:
            raw = value.get(name)
            if raw is None:
                raise ValueError(f"place pose requires {name!r}")
            degrees = _finite_vector(raw, length, f"place_pose_deg.{name}")
            return tuple(math.radians(item) for item in degrees)

        return cls(
            torso_rad=radians("torso", 6),
            right_arm_rad=radians("right_arm", 7),
            left_arm_rad=radians("left_arm", 7),
        )


@dataclass(frozen=True, slots=True)
class PalletControlConfig:
    address: str = "192.168.30.1:50051"
    priority: int = 10
    ready_pose: ReadyPose = field(default_factory=ReadyPose)
    # Operator-requested 2026-07-30: the slot-1 ready one-shot runs in 3 s.
    ready_transition_minimum_time_s: float = 3.0
    ready_tolerance_rad: float = math.radians(1.0)
    transition_timeout_s: float = 15.0
    initial_state_timeout_s: float = 2.0
    state_update_rate_hz: float = 50.0
    state_stale_after_s: float = 0.15
    send_rate_hz: float = 20.0
    control_hold_time_s: float = 1.0
    steady_minimum_time_s: float = 0.05
    command_stale_after_s: float = 0.15
    send_timeout_ms: int = 250
    startup_timeout_s: float = 2.0
    zero_ack_timeout_s: float = 2.0
    shutdown_timeout_ms: int = 2000
    join_timeout_s: float = 2.0
    linear_acceleration_limit_mps2: float = 0.15
    angular_acceleration_limit_radps2: float = 0.20
    maximum_linear_speed_mps: float = 0.05
    maximum_angular_speed_radps: float = 0.08
    wheel_stop_linear_mps: float = 0.01
    wheel_stop_angular_radps: float = 0.02
    wheel_stop_dwell_s: float = 0.35
    grip_dwell_s: float = 0.50
    grip_min_samples: int = 10
    arm_tracking_tolerance_rad: float = math.radians(1.0)
    eef_separation_peak_to_peak_m: float = 0.004
    eef_separation_axis_std_m: float = 0.002
    held_top_peak_to_peak_m: float = 0.010
    held_top_std_m: float = 0.005
    held_top_downward_drift_m: float = 0.005
    held_top_direct_plane_dwell_frames: int = 5
    held_top_sample_fresh_after_s: float = 0.30
    clearance_evidence_fresh_after_s: float = 0.30
    clearance_scene_max_span_s: float = 0.50
    maximum_box_height_m: float = 0.164
    minimum_clearance_m: float = 0.050
    fixed_ready_geometry_only_commissioning_enabled: bool = False
    # A streamed Cartesian packet must not restart a multi-second trajectory on
    # every 20 Hz update.  Smoothness comes from the explicit velocity and
    # acceleration limits; minimum_time only spans one stream period.
    placement_minimum_time_s: float = 0.05
    placement_linear_velocity_limit_mps: float = 0.03
    placement_angular_velocity_limit_radps: float = 0.20
    placement_linear_acceleration_limit_mps2: float = 0.08
    placement_angular_acceleration_limit_radps2: float = 0.30
    # Operator-requested 2026-07-30: hold the slot-1 ready posture unchanged
    # while the base drives, so the loaded hold commands the measured wrists with
    # no inward offset.  Raising this restores the box-pick style compliant
    # squeeze (a target error, not a requested penetration into the carton).
    placement_squeeze_offset_m: float = 0.0
    # Demonstrated placement posture and how long the arms may take to reach it.
    # The trajectory is paced by per-target velocity limits, not by minimum_time,
    # because a streamed packet must not restart a multi-second trajectory at
    # 20 Hz.  ``None`` keeps the legacy base-Z descent behaviour.
    place_pose: "PlacePose | None" = None
    placement_place_pose_duration_s: float = 1.0
    # Slot-1 release opens along base Y only.  The commanded travel per hand is
    # exactly this spread because the loaded-hold squeeze cancels out, so a
    # small value keeps both arms far from the reach singularity.
    placement_release_spread_m: float = 0.030
    placement_max_release_spread_m: float = 0.040
    # A measured inter-EEF axis further than this from base +/-Y would shear the
    # carton sideways instead of opening, so release fails closed instead.
    placement_release_axis_max_deviation_rad: float = math.radians(10.0)
    placement_joint_stiffness: tuple[float, ...] = (150.0,) * 7
    placement_joint_damping_ratio: float = 1.0
    placement_nullspace_weight: tuple[float, ...] = (1.0,) * 7
    placement_nullspace_kp: float = 0.5
    placement_nullspace_kd: float = 0.5
    placement_nullspace_cost_weight: float = 0.01

    @classmethod
    def from_root_config(
        cls,
        root: Mapping[str, Any],
        address_override: str | None = None,
    ) -> "PalletControlConfig":
        """Build from the repository pallet JSON while preserving hard gates."""

        if not isinstance(root, Mapping):
            raise TypeError("root pallet config must be a mapping")
        defaults = cls()

        def section(name: str) -> Mapping[str, Any]:
            value = root.get(name, {})
            if not isinstance(value, Mapping):
                raise ValueError(f"root config section {name!r} must be a mapping")
            return value

        robot = section("robot")
        servo_config = PalletServoConfig.from_root_config(root)
        safety = section("safety")
        held_box = section("held_box")
        camera = section("camera")
        stream = section("control_stream")
        grip_interlock = section("grip_interlock")
        placement = section("placement")
        PlacementConfig.from_root_config(root)

        safety_keys = {"minimum_clearance_m", "state_fresh_after_s"}
        unknown_safety_keys = sorted(set(safety) - safety_keys)
        if unknown_safety_keys:
            raise ValueError(
                "unknown safety configuration key(s): "
                + ", ".join(unknown_safety_keys)
            )

        grip_interlock_keys = {
            "minimum_dwell_s",
            "minimum_samples",
            "maximum_arm_tracking_error_deg",
            "maximum_eef_separation_peak_to_peak_m",
            "maximum_eef_separation_axis_std_m",
            "maximum_scene_evidence_span_s",
            "clearance_evidence_fresh_after_s",
            "fixed_ready_geometry_only_commissioning_enabled",
        }
        unknown_grip_keys = sorted(set(grip_interlock) - grip_interlock_keys)
        if unknown_grip_keys:
            raise ValueError(
                "unknown grip_interlock configuration key(s): "
                + ", ".join(unknown_grip_keys)
            )

        model_name = str(robot.get("model", "")).strip().upper()
        version = str(robot.get("version", "")).strip().lower().removeprefix("v")
        if model_name != EXPECTED_ROBOT_MODEL:
            raise ValueError(
                f"root config robot.model must be {EXPECTED_ROBOT_MODEL!r}"
            )
        if version != EXPECTED_ROBOT_VERSION and not version.startswith(
            f"{EXPECTED_ROBOT_VERSION}."
        ):
            raise ValueError(
                f"root config robot.version must be {EXPECTED_ROBOT_VERSION!r}"
            )

        ready_raw = robot.get("ready_pose_rad")
        if not isinstance(ready_raw, Mapping):
            raise ValueError("root config robot.ready_pose_rad must be a mapping")
        ready_pose = ReadyPose(
            torso_rad=ready_raw.get("torso", ()),
            right_arm_rad=ready_raw.get("right_arm", ()),
            left_arm_rad=ready_raw.get("left_arm", ()),
            head_rad=ready_raw.get("head", ()),
        )
        if ready_pose != ReadyPose():
            raise ValueError(
                "root config ready pose differs from the approved slot-1 pose"
            )

        stiffness = _finite_vector(
            robot.get("arm_joint_impedance_stiffness_nm_per_rad", ()),
            7,
            "robot.arm_joint_impedance_stiffness_nm_per_rad",
        )
        if stiffness != ARM_STIFFNESS:
            raise ValueError("root config arm stiffness must be exactly [150.0] * 7")
        if str(robot.get("arm_torque_limit", "")).strip() != TORQUE_POLICY:
            raise ValueError("root config arm torque policy must be 'sdk_default'")

        address = (
            str(address_override).strip()
            if address_override is not None
            else str(robot.get("default_address", "")).strip()
        )
        if not address:
            raise ValueError("root config robot.default_address must not be empty")

        geometry_only_enabled = grip_interlock.get(
            "fixed_ready_geometry_only_commissioning_enabled",
            defaults.fixed_ready_geometry_only_commissioning_enabled,
        )
        if not isinstance(geometry_only_enabled, bool):
            raise ValueError(
                "fixed_ready_geometry_only_commissioning_enabled must be a boolean"
            )

        return cls(
            address=address,
            priority=int(stream.get("priority", defaults.priority)),
            ready_pose=ready_pose,
            ready_transition_minimum_time_s=float(
                robot.get(
                    "ready_transition_minimum_time_s",
                    defaults.ready_transition_minimum_time_s,
                )
            ),
            ready_tolerance_rad=math.radians(
                float(
                    robot.get(
                        "ready_tolerance_deg",
                        math.degrees(defaults.ready_tolerance_rad),
                    )
                )
            ),
            state_stale_after_s=float(
                safety.get("state_fresh_after_s", defaults.state_stale_after_s)
            ),
            send_rate_hz=float(stream.get("send_rate_hz", defaults.send_rate_hz)),
            control_hold_time_s=float(
                stream.get("control_hold_time_s", defaults.control_hold_time_s)
            ),
            steady_minimum_time_s=float(
                stream.get("minimum_time_s", defaults.steady_minimum_time_s)
            ),
            command_stale_after_s=float(
                stream.get("command_stale_after_s", defaults.command_stale_after_s)
            ),
            send_timeout_ms=int(
                stream.get("send_timeout_ms", defaults.send_timeout_ms)
            ),
            shutdown_timeout_ms=int(
                stream.get("shutdown_timeout_ms", defaults.shutdown_timeout_ms)
            ),
            maximum_linear_speed_mps=servo_config.max_linear_speed_mps,
            maximum_angular_speed_radps=servo_config.max_angular_speed_radps,
            wheel_stop_linear_mps=servo_config.wheel_linear_stop_mps,
            wheel_stop_angular_radps=servo_config.wheel_angular_stop_radps,
            wheel_stop_dwell_s=servo_config.wheel_stop_duration_s,
            grip_dwell_s=float(
                grip_interlock.get("minimum_dwell_s", defaults.grip_dwell_s)
            ),
            grip_min_samples=int(
                grip_interlock.get("minimum_samples", defaults.grip_min_samples)
            ),
            arm_tracking_tolerance_rad=math.radians(
                float(
                    grip_interlock.get(
                        "maximum_arm_tracking_error_deg",
                        math.degrees(defaults.arm_tracking_tolerance_rad),
                    )
                )
            ),
            eef_separation_peak_to_peak_m=float(
                grip_interlock.get(
                    "maximum_eef_separation_peak_to_peak_m",
                    defaults.eef_separation_peak_to_peak_m,
                )
            ),
            eef_separation_axis_std_m=float(
                grip_interlock.get(
                    "maximum_eef_separation_axis_std_m",
                    defaults.eef_separation_axis_std_m,
                )
            ),
            held_top_std_m=float(
                held_box.get(
                    "maximum_top_plane_std_m",
                    defaults.held_top_std_m,
                )
            ),
            held_top_downward_drift_m=float(
                held_box.get(
                    "maximum_downward_drift_m",
                    defaults.held_top_downward_drift_m,
                )
            ),
            held_top_direct_plane_dwell_frames=int(
                held_box.get(
                    "direct_top_plane_dwell_frames",
                    defaults.held_top_direct_plane_dwell_frames,
                )
            ),
            held_top_sample_fresh_after_s=float(
                camera.get(
                    "frame_fresh_after_s",
                    defaults.held_top_sample_fresh_after_s,
                )
            ),
            clearance_evidence_fresh_after_s=float(
                grip_interlock.get(
                    "clearance_evidence_fresh_after_s",
                    defaults.clearance_evidence_fresh_after_s,
                )
            ),
            clearance_scene_max_span_s=float(
                grip_interlock.get(
                    "maximum_scene_evidence_span_s",
                    defaults.clearance_scene_max_span_s,
                )
            ),
            maximum_box_height_m=float(
                held_box.get("maximum_height_m", defaults.maximum_box_height_m)
            ),
            minimum_clearance_m=float(
                safety.get("minimum_clearance_m", defaults.minimum_clearance_m)
            ),
            fixed_ready_geometry_only_commissioning_enabled=(
                geometry_only_enabled
            ),
            placement_minimum_time_s=float(
                placement.get("minimum_time_s", defaults.placement_minimum_time_s)
            ),
            placement_linear_velocity_limit_mps=float(
                placement.get(
                    "linear_velocity_limit_mps",
                    defaults.placement_linear_velocity_limit_mps,
                )
            ),
            placement_angular_velocity_limit_radps=float(
                placement.get(
                    "angular_velocity_limit_radps",
                    defaults.placement_angular_velocity_limit_radps,
                )
            ),
            placement_linear_acceleration_limit_mps2=float(
                placement.get(
                    "linear_acceleration_limit_mps2",
                    defaults.placement_linear_acceleration_limit_mps2,
                )
            ),
            placement_angular_acceleration_limit_radps2=float(
                placement.get(
                    "angular_acceleration_limit_radps2",
                    defaults.placement_angular_acceleration_limit_radps2,
                )
            ),
            placement_squeeze_offset_m=float(
                placement.get(
                    "squeeze_offset_m",
                    defaults.placement_squeeze_offset_m,
                )
            ),
            place_pose=(
                None
                if placement.get("place_pose_deg") is None
                else PlacePose.from_degrees(placement["place_pose_deg"])
            ),
            placement_place_pose_duration_s=float(
                placement.get(
                    "place_pose_duration_s",
                    defaults.placement_place_pose_duration_s,
                )
            ),
            placement_release_spread_m=float(
                placement.get(
                    "release_spread_m",
                    defaults.placement_release_spread_m,
                )
            ),
            placement_max_release_spread_m=float(
                placement.get(
                    "maximum_release_spread_m",
                    defaults.placement_max_release_spread_m,
                )
            ),
            placement_release_axis_max_deviation_rad=math.radians(
                float(
                    placement.get(
                        "release_axis_max_deviation_deg",
                        math.degrees(
                            defaults.placement_release_axis_max_deviation_rad
                        ),
                    )
                )
            ),
            placement_joint_stiffness=tuple(
                placement.get(
                    "joint_stiffness_nm_per_rad",
                    defaults.placement_joint_stiffness,
                )
            ),
            placement_joint_damping_ratio=float(
                placement.get(
                    "joint_damping_ratio",
                    defaults.placement_joint_damping_ratio,
                )
            ),
            placement_nullspace_weight=tuple(
                placement.get(
                    "nullspace_weight",
                    defaults.placement_nullspace_weight,
                )
            ),
            placement_nullspace_kp=float(
                placement.get("nullspace_kp", defaults.placement_nullspace_kp)
            ),
            placement_nullspace_kd=float(
                placement.get("nullspace_kd", defaults.placement_nullspace_kd)
            ),
            placement_nullspace_cost_weight=float(
                placement.get(
                    "nullspace_cost_weight",
                    defaults.placement_nullspace_cost_weight,
                )
            ),
        )

    def __post_init__(self) -> None:
        if not str(self.address).strip():
            raise ValueError("address must not be empty")
        if self.priority < 1:
            raise ValueError("priority must be positive")
        for name in (
            "ready_transition_minimum_time_s",
            "ready_tolerance_rad",
            "transition_timeout_s",
            "initial_state_timeout_s",
            "state_update_rate_hz",
            "state_stale_after_s",
            "send_rate_hz",
            "control_hold_time_s",
            "steady_minimum_time_s",
            "command_stale_after_s",
            "startup_timeout_s",
            "zero_ack_timeout_s",
            "join_timeout_s",
            "linear_acceleration_limit_mps2",
            "angular_acceleration_limit_radps2",
            "maximum_linear_speed_mps",
            "maximum_angular_speed_radps",
            "wheel_stop_linear_mps",
            "wheel_stop_angular_radps",
            "wheel_stop_dwell_s",
            "grip_dwell_s",
            "arm_tracking_tolerance_rad",
            "eef_separation_peak_to_peak_m",
            "eef_separation_axis_std_m",
            "held_top_peak_to_peak_m",
            "held_top_std_m",
            "held_top_downward_drift_m",
            "held_top_sample_fresh_after_s",
            "clearance_evidence_fresh_after_s",
            "clearance_scene_max_span_s",
            "maximum_box_height_m",
            "minimum_clearance_m",
            "placement_minimum_time_s",
            "placement_linear_velocity_limit_mps",
            "placement_angular_velocity_limit_radps",
            "placement_linear_acceleration_limit_mps2",
            "placement_angular_acceleration_limit_radps2",
            "placement_release_spread_m",
            "placement_max_release_spread_m",
            "placement_release_axis_max_deviation_rad",
            "placement_joint_damping_ratio",
            "placement_nullspace_kp",
            "placement_nullspace_kd",
            "placement_nullspace_cost_weight",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.ready_transition_minimum_time_s < READY_TRANSITION_MINIMUM_TIME_FLOOR_S:
            raise ValueError(
                "ready transition must take at least "
                f"{READY_TRANSITION_MINIMUM_TIME_FLOOR_S:.1f} seconds"
            )
        if self.ready_pose != ReadyPose():
            raise ValueError("ready pose must remain the approved slot-1 pose")
        if self.ready_tolerance_rad > math.radians(1.0) + 1e-12:
            raise ValueError("ready joint tolerance cannot exceed 1 degree")
        if self.arm_tracking_tolerance_rad > math.radians(1.0) + 1e-12:
            raise ValueError("arm tracking tolerance cannot exceed 1 degree")
        if self.control_hold_time_s > 1.0:
            raise ValueError("control_hold_time_s cannot exceed 1 second")
        if self.maximum_linear_speed_mps > HARD_MAX_LINEAR_SPEED_MPS:
            raise ValueError("maximum_linear_speed_mps exceeds the 0.08 m/s hard limit")
        if self.maximum_angular_speed_radps > HARD_MAX_ANGULAR_SPEED_RADPS:
            raise ValueError(
                "maximum_angular_speed_radps exceeds the 0.10 rad/s hard limit"
            )
        if self.command_stale_after_s < 2.0 / self.send_rate_hz:
            raise ValueError("command staleness must cover at least two send periods")
        if self.command_stale_after_s > self.state_stale_after_s + 1e-12:
            raise ValueError("command staleness cannot exceed measured-state freshness")
        if self.state_stale_after_s > 0.15 + 1e-12:
            raise ValueError("measured-state freshness cannot exceed 0.15 seconds")
        if self.send_timeout_ms <= 0 or self.shutdown_timeout_ms <= 0:
            raise ValueError("SDK timeouts must be positive")
        if self.grip_dwell_s < 0.50:
            raise ValueError("grip continuity dwell must be at least 0.50 seconds")
        if self.grip_min_samples < 10:
            raise ValueError("grip_min_samples must be at least 10")
        if self.eef_separation_peak_to_peak_m > 0.004 + 1e-12:
            raise ValueError("EEF separation peak-to-peak limit cannot exceed 4 mm")
        if self.eef_separation_axis_std_m > 0.002 + 1e-12:
            raise ValueError("EEF separation axis std limit cannot exceed 2 mm")
        if self.held_top_peak_to_peak_m > 0.010 + 1e-12:
            raise ValueError("held-top peak-to-peak limit cannot exceed 10 mm")
        if self.held_top_std_m > 0.005 + 1e-12:
            raise ValueError("held-top std limit cannot exceed 5 mm")
        if self.held_top_downward_drift_m > 0.005 + 1e-12:
            raise ValueError("held-top downward drift limit cannot exceed 5 mm")
        if self.held_top_direct_plane_dwell_frames < 5:
            raise ValueError(
                "held-top direct-plane dwell must contain at least 5 frames"
            )
        if self.held_top_sample_fresh_after_s > 0.30 + 1e-12:
            raise ValueError(
                "fixed-ready box-bottom capture freshness cannot exceed 0.30 seconds"
            )
        if abs(self.clearance_evidence_fresh_after_s - 0.30) > 1e-12:
            raise ValueError("clearance evidence freshness must be exactly 0.30 seconds")
        if self.clearance_scene_max_span_s > 0.50 + 1e-12:
            raise ValueError("clearance scene evidence span cannot exceed 0.50 seconds")
        if self.maximum_box_height_m < 0.164 - 1e-12:
            raise ValueError("maximum box height cannot be below measured 164 mm")
        if self.minimum_clearance_m < 0.050 - 1e-12:
            raise ValueError("minimum vertical clearance cannot be below 50 mm")
        stream_period_s = 1.0 / self.send_rate_hz
        if self.placement_minimum_time_s + 1e-12 < stream_period_s:
            raise ValueError(
                "placement minimum time must span at least one stream period"
            )
        if self.placement_minimum_time_s > 0.10 + 1e-12:
            raise ValueError(
                "placement minimum time cannot exceed 0.10 s in the repeated stream; "
                "velocity/acceleration limits own trajectory smoothing"
            )
        if self.placement_linear_velocity_limit_mps > 0.05 + 1e-12:
            raise ValueError("placement linear velocity limit cannot exceed 0.05 m/s")
        if self.placement_angular_velocity_limit_radps > 0.30 + 1e-12:
            raise ValueError("placement angular velocity limit cannot exceed 0.30 rad/s")
        if self.placement_linear_acceleration_limit_mps2 > 0.15 + 1e-12:
            raise ValueError("placement linear acceleration cannot exceed 0.15 m/s^2")
        if self.placement_angular_acceleration_limit_radps2 > 0.50 + 1e-12:
            raise ValueError(
                "placement angular acceleration cannot exceed 0.50 rad/s^2"
            )
        if self.place_pose is not None and not isinstance(self.place_pose, PlacePose):
            raise ValueError("place_pose must be a PlacePose")
        object.__setattr__(
            self,
            "placement_place_pose_duration_s",
            _positive(
                self.placement_place_pose_duration_s,
                "placement_place_pose_duration_s",
            ),
        )
        if self.placement_place_pose_duration_s > 5.0 + 1e-12:
            raise ValueError("place pose duration cannot exceed 5 seconds")
        squeeze = _finite_float_nonnegative(
            self.placement_squeeze_offset_m,
            "placement_squeeze_offset_m",
        )
        object.__setattr__(self, "placement_squeeze_offset_m", squeeze)
        if squeeze > 0.150 + 1e-12:
            raise ValueError("placement squeeze offset cannot exceed 150 mm")
        if self.placement_release_spread_m > self.placement_max_release_spread_m:
            raise ValueError("placement release spread cannot exceed its max bound")
        if self.placement_max_release_spread_m > 0.040 + 1e-12:
            raise ValueError("placement max release spread cannot exceed 40 mm")
        if self.placement_release_axis_max_deviation_rad > math.radians(30.0) + 1e-12:
            raise ValueError(
                "placement release axis deviation limit cannot exceed 30 degrees"
            )
        object.__setattr__(
            self,
            "placement_joint_stiffness",
            _finite_vector(
                self.placement_joint_stiffness,
                7,
                "placement_joint_stiffness",
            ),
        )
        if min(self.placement_joint_stiffness) <= 0.0:
            raise ValueError("placement joint stiffness values must be positive")
        object.__setattr__(
            self,
            "placement_nullspace_weight",
            _finite_vector(
                self.placement_nullspace_weight,
                7,
                "placement_nullspace_weight",
            ),
        )
        if min(self.placement_nullspace_weight) < 0.0:
            raise ValueError("placement nullspace weights must be non-negative")
        if max(self.placement_nullspace_weight) <= 0.0:
            raise ValueError("at least one placement nullspace weight must be positive")
        if self.wheel_stop_linear_mps > 0.01 + 1e-12:
            raise ValueError("wheel-stop linear threshold cannot exceed 0.01 m/s")
        if self.wheel_stop_angular_radps > 0.02 + 1e-12:
            raise ValueError("wheel-stop angular threshold cannot exceed 0.02 rad/s")
        if self.wheel_stop_dwell_s < 0.35:
            raise ValueError("wheel-stop dwell must be at least 0.35 seconds")
@dataclass(frozen=True, slots=True)
class MobilityCommand:
    vx_mps: float
    vy_mps: float
    wz_radps: float
    source_timestamp_s: float | None = None

    def __post_init__(self) -> None:
        values = (self.vx_mps, self.vy_mps, self.wz_radps)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("mobility command must be finite")
        if self.source_timestamp_s is not None and not math.isfinite(
            float(self.source_timestamp_s)
        ):
            raise ValueError("source_timestamp_s must be finite when provided")

    @property
    def linear_norm_mps(self) -> float:
        return math.hypot(float(self.vx_mps), float(self.vy_mps))

    @property
    def is_zero(self) -> bool:
        return self.linear_norm_mps <= 1e-12 and abs(float(self.wz_radps)) <= 1e-12


ZERO_MOBILITY = MobilityCommand(0.0, 0.0, 0.0)


class ArmStreamMode(str, Enum):
    JOINT_READY_HOLD = "JOINT_READY_HOLD"
    CARTESIAN_LOADED_HOLD = "CARTESIAN_LOADED_HOLD"
    CARTESIAN_PLACEMENT_LOWERING = "CARTESIAN_PLACEMENT_LOWERING"
    CARTESIAN_PLACEMENT_RELEASE = "CARTESIAN_PLACEMENT_RELEASE"
    CARTESIAN_PLACEMENT_FAIL_CLOSED_HOLD = "CARTESIAN_PLACEMENT_FAIL_CLOSED_HOLD"


@dataclass(frozen=True, slots=True)
class CartesianArmTarget:
    mode: ArmStreamMode
    right_T_torso_eef: np.ndarray
    left_T_torso_eef: np.ndarray
    right_T_base_eef: np.ndarray
    left_T_base_eef: np.ndarray
    right_nullspace_joint_rad: tuple[float, ...]
    left_nullspace_joint_rad: tuple[float, ...]
    inter_eef_axis_base: tuple[float, float, float]
    created_monotonic_s: float
    source_state_sequence: int
    lowering_distance_m: float = 0.0
    squeeze_offset_m: float = 0.0
    release_spread_m: float = 0.0
    release_axis_deviation_rad: float = 0.0
    descent_plan_id: str | None = None
    target_source: str = "measured_eef"
    linear_velocity_limit_mps: float | None = None
    angular_velocity_limit_radps: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ArmStreamMode):
            object.__setattr__(self, "mode", ArmStreamMode(str(self.mode)))
        object.__setattr__(
            self,
            "right_T_torso_eef",
            _readonly_matrix(self.right_T_torso_eef, "right_T_torso_eef"),
        )
        object.__setattr__(
            self,
            "left_T_torso_eef",
            _readonly_matrix(self.left_T_torso_eef, "left_T_torso_eef"),
        )
        object.__setattr__(
            self,
            "right_T_base_eef",
            _readonly_matrix(self.right_T_base_eef, "right_T_base_eef"),
        )
        object.__setattr__(
            self,
            "left_T_base_eef",
            _readonly_matrix(self.left_T_base_eef, "left_T_base_eef"),
        )
        object.__setattr__(
            self,
            "right_nullspace_joint_rad",
            _finite_vector(
                self.right_nullspace_joint_rad,
                7,
                "right_nullspace_joint_rad",
            ),
        )
        object.__setattr__(
            self,
            "left_nullspace_joint_rad",
            _finite_vector(
                self.left_nullspace_joint_rad,
                7,
                "left_nullspace_joint_rad",
            ),
        )
        axis = _finite_vector(self.inter_eef_axis_base, 3, "inter_eef_axis_base")
        if abs(float(np.linalg.norm(axis)) - 1.0) > 1e-6:
            raise ValueError("inter_eef_axis_base must be unit length")
        object.__setattr__(self, "inter_eef_axis_base", axis)
        if not math.isfinite(float(self.created_monotonic_s)):
            raise ValueError("created_monotonic_s must be finite")
        if self.source_state_sequence < 1:
            raise ValueError("source_state_sequence must be positive")
        for name in ("lowering_distance_m", "squeeze_offset_m", "release_spread_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        deviation = float(self.release_axis_deviation_rad)
        if not math.isfinite(deviation) or deviation < 0.0:
            raise ValueError(
                "release_axis_deviation_rad must be finite and non-negative"
            )
        object.__setattr__(self, "release_axis_deviation_rad", deviation)
        object.__setattr__(self, "target_source", str(self.target_source).strip() or "measured_eef")
        for name in ("linear_velocity_limit_mps", "angular_velocity_limit_radps"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _positive(value, name))
        if self.descent_plan_id is not None:
            plan_id = str(self.descent_plan_id).strip()
            if not plan_id:
                raise ValueError("descent_plan_id must not be empty when provided")
            object.__setattr__(self, "descent_plan_id", plan_id)


@dataclass(frozen=True, slots=True)
class PlacementTelemetry:
    arm_mode: ArmStreamMode
    placement_started: bool
    source_state_sequence: int | None
    target_created_monotonic_s: float | None
    right_T_base_eef_target: np.ndarray | None
    left_T_base_eef_target: np.ndarray | None
    measured_T_base_torso_available: bool
    measured_T_base_right_eef_available: bool
    measured_T_base_left_eef_available: bool
    zero_latched: bool
    wheel_stopped: bool
    stream_running: bool
    target_acknowledged: bool
    acknowledged_command_sequence: int
    last_reason: str | None = None
    descent_plan_id: str | None = None
    lowering_distance_m: float | None = None
    release_spread_m: float | None = None
    release_axis_base: tuple[float, float, float] | None = None
    release_axis_deviation_rad: float | None = None


@dataclass(frozen=True, slots=True)
class CommandId:
    owner_epoch: str
    sequence: int

    def __post_init__(self) -> None:
        if not str(self.owner_epoch).strip():
            raise ValueError("command owner_epoch must not be empty")
        if self.sequence < 1:
            raise ValueError("command sequence must be positive")


@dataclass(frozen=True, slots=True)
class LoadedSlot1ReadyBootstrap:
    """Same-session proof for an already-held box at the slot-1 ready pose.

    This is deliberately not an upstream ownership transfer.  It only records
    that this controller session measured the configured slot-1 ready posture
    before opening its own combined body/mobility stream.
    """

    owner_epoch: str
    source_phase: str
    measured_state_sequence: int
    measured_robot_state_timestamp_s: float | None
    measured_monotonic_s: float
    acknowledged_loaded_box: bool
    max_joint_error_rad: float
    torso_target_rad: tuple[float, ...]
    right_arm_target_rad: tuple[float, ...]
    left_arm_target_rad: tuple[float, ...]
    head_target_rad: tuple[float, ...]
    right_stiffness: tuple[float, ...]
    left_stiffness: tuple[float, ...]
    torque_policy: str

    def __post_init__(self) -> None:
        if not str(self.owner_epoch).strip() or not str(self.source_phase).strip():
            raise ValueError("loaded-ready owner_epoch/source_phase must not be empty")
        if self.measured_state_sequence < 1:
            raise ValueError("loaded-ready measured state sequence must be positive")
        if not math.isfinite(float(self.measured_monotonic_s)):
            raise ValueError("loaded-ready measured monotonic timestamp must be finite")
        if self.measured_robot_state_timestamp_s is not None and not math.isfinite(
            float(self.measured_robot_state_timestamp_s)
        ):
            raise ValueError(
                "loaded-ready robot-state timestamp must be finite when present"
            )
        if self.acknowledged_loaded_box is not True:
            raise ValueError("acknowledged_loaded_box must be explicitly true")
        if not math.isfinite(float(self.max_joint_error_rad)):
            raise ValueError("loaded-ready max joint error must be finite")
        if self.max_joint_error_rad < 0.0:
            raise ValueError("loaded-ready max joint error must be non-negative")
        if not str(self.torque_policy).strip():
            raise ValueError("loaded-ready torque policy must not be empty")
        object.__setattr__(
            self,
            "torso_target_rad",
            _finite_vector(self.torso_target_rad, 6, "torso_target_rad"),
        )
        object.__setattr__(
            self,
            "right_arm_target_rad",
            _finite_vector(self.right_arm_target_rad, 7, "right_arm_target_rad"),
        )
        object.__setattr__(
            self,
            "left_arm_target_rad",
            _finite_vector(self.left_arm_target_rad, 7, "left_arm_target_rad"),
        )
        object.__setattr__(
            self,
            "head_target_rad",
            _finite_vector(self.head_target_rad, 2, "head_target_rad"),
        )
        object.__setattr__(
            self,
            "right_stiffness",
            _finite_vector(self.right_stiffness, 7, "right_stiffness"),
        )
        object.__setattr__(
            self,
            "left_stiffness",
            _finite_vector(self.left_stiffness, 7, "left_stiffness"),
        )


@dataclass(frozen=True, slots=True)
class ReadyTransitionAck:
    command_id: CommandId
    feedback: ComponentFeedbackAck
    measured_state_sequence: int | None
    measured_state_age_s: float | None
    joint_errors_rad: tuple[float, ...]
    max_joint_error_rad: float
    all_target_joints_ready: bool
    received_monotonic_s: float

    @property
    def mobility(self) -> bool:
        return self.feedback.mobility

    @property
    def torso(self) -> bool:
        return self.feedback.torso

    @property
    def head(self) -> bool:
        return self.feedback.head

    @property
    def right_arm(self) -> bool:
        return self.feedback.right_arm

    @property
    def left_arm(self) -> bool:
        return self.feedback.left_arm

    def ready_within(self, tolerance_rad: float) -> bool:
        return (
            self.feedback.running
            and self.all_target_joints_ready
            and math.isfinite(self.max_joint_error_rad)
            and self.max_joint_error_rad <= float(tolerance_rad)
        )


@dataclass(frozen=True, slots=True)
class MeasuredRobotState:
    sequence: int
    received_monotonic_s: float
    robot_timestamp_s: float | None
    position_rad: np.ndarray
    velocity_radps: np.ndarray
    is_ready: np.ndarray
    T_base_head: np.ndarray | None
    T_base_right_eef: np.ndarray | None
    T_base_left_eef: np.ndarray | None
    T_odom_base: np.ndarray | None
    base_twist_w_vx_vy: tuple[float, float, float] | None
    wheel_max_abs_radps: float | None
    T_base_torso: np.ndarray | None = None
    kinematics_error: str | None = None
    odometry_error: str | None = None

    def age_s(self, now_s: float) -> float:
        return max(0.0, float(now_s) - self.received_monotonic_s)


@dataclass(frozen=True, slots=True)
class WheelStopStatus:
    feedback_fresh: bool
    stopped: bool
    linear_speed_mps: float | None
    angular_speed_radps: float | None
    max_wheel_speed_radps: float | None
    dwell_s: float
    measured_state_sequence: int | None


@dataclass(frozen=True, slots=True)
class GripContinuityResult:
    passed: bool
    reasons: tuple[str, ...]
    evaluated_monotonic_s: float
    state_sample_count: int
    scene_sample_count: int
    dwell_s: float
    arm_tracking_error_max_rad: float | None
    eef_separation_peak_to_peak_m: float | None
    eef_separation_axis_std_max_m: float | None
    held_top_std_m: float | None
    held_top_downward_drift_m: float | None
    clearance_lower_bound_m: float | None
    box_bottom_z_lower_bound_m: float | None = None
    stack_top_z_upper_bound_m: float | None = None
    scene_evidence_span_s: float | None = None
    scene_latest_age_s: float | None = None
    scene_max_capture_age_at_acceptance_s: float | None = None
    clearance_source: str = "unavailable"
    fixed_ready_geometry_only_authorized: bool = False


@dataclass(frozen=True, slots=True)
class _ClearanceSceneSample:
    held_z_m: float
    held_uncertainty_m: float
    stack_z_m: float
    stack_uncertainty_m: float
    capture_timestamp_s: float
    accepted_monotonic_s: float


@dataclass(frozen=True, slots=True)
class HandoffAck:
    source_owner_epoch: str
    next_owner: str
    acknowledged: bool
    accepted_command_sequence: int
    body_target_token: str
    zero_mobility: bool
    body_hold_included: bool
    wheel_stopped: bool
    timestamp_s: float
    message: str = ""

    def __post_init__(self) -> None:
        if not str(self.source_owner_epoch).strip():
            raise ValueError("handoff source_owner_epoch must not be empty")
        if not str(self.next_owner).strip():
            raise ValueError("handoff next_owner must not be empty")
        if self.accepted_command_sequence < 0:
            raise ValueError("handoff command sequence must be nonnegative")
        if not str(self.body_target_token).strip():
            raise ValueError("handoff body_target_token must not be empty")
        if not math.isfinite(float(self.timestamp_s)):
            raise ValueError("handoff timestamp must be finite")


@dataclass(frozen=True, slots=True)
class StreamTelemetry:
    phase: ControllerPhase
    owner_epoch: str
    command_sequence: int
    ready_transition_command_count: int
    steady_send_count: int
    last_sent_mobility: MobilityCommand
    last_send_monotonic_s: float | None
    maximum_send_gap_s: float
    zero_latched: bool
    shutdown_pending: bool
    body_hold_included: bool
    mobility_included: bool
    right_arm_stiffness: tuple[float, ...]
    left_arm_stiffness: tuple[float, ...]
    torque_policy: str
    last_feedback_error: str | None
    last_error: str | None
    arm_mode: ArmStreamMode = ArmStreamMode.JOINT_READY_HOLD
    placement_started: bool = False
    placement_target_sequence: int | None = None
def evaluate_grip_continuity(
    config: PalletControlConfig,
    scene_window: Sequence[Any],
    *,
    now_s: float,
    cartesian_arm_motion: bool,
    states: Sequence[MeasuredRobotState],
    ready_joint_errors: Callable[[MeasuredRobotState], tuple[np.ndarray, bool]],
) -> GripContinuityResult:
    """Decide loaded-hold continuity and conservative vertical clearance.

    Pure decision function: it holds no lock, reads no controller state, and
    sends nothing.  The controller snapshots ``now_s``, ``cartesian_arm_motion``
    and ``states`` under its condition variable and passes them in, so this
    interlock is testable without an SDK, a stream, or a thread.

    ``ready_joint_errors`` returns ``(errors, all_ready)`` for one measured
    state, matching :meth:`RBY1PalletController._ready_joint_errors`.  Its error
    vector is the torso, right-arm, left-arm, head concatenation, which is why
    the arm slice below indexes into that layout.
    """

    reasons: list[str] = []
    dwell_s = (
        0.0
        if len(states) < 2
        else states[-1].received_monotonic_s - states[0].received_monotonic_s
    )
    if len(states) < config.grip_min_samples:
        reasons.append("insufficient_fresh_robot_state_samples")
    if dwell_s < config.grip_dwell_s * 0.95:
        reasons.append("insufficient_grip_dwell")

    arm_error_max: float | None = None
    separations: list[np.ndarray] = []
    for state in states:
        if not cartesian_arm_motion:
            errors, all_ready = ready_joint_errors(state)
            arm_indices = np.r_[
                np.arange(6, 13, dtype=np.int64),
                np.arange(13, 20, dtype=np.int64),
            ]
            arm_errors = np.abs(errors[arm_indices])
            current_max = float(np.max(arm_errors))
            arm_error_max = (
                current_max
                if arm_error_max is None
                else max(arm_error_max, current_max)
            )
            if not all_ready:
                reasons.append("target_joint_not_ready")
        if state.T_base_right_eef is None or state.T_base_left_eef is None:
            reasons.append("fresh_eef_fk_unavailable")
        else:
            separations.append(
                state.T_base_right_eef[:3, 3] - state.T_base_left_eef[:3, 3]
            )

    if not cartesian_arm_motion and (
        arm_error_max is None
        or arm_error_max > config.arm_tracking_tolerance_rad
    ):
        reasons.append("arm_tracking_error")

    separation_p2p: float | None = None
    separation_std: float | None = None
    if len(separations) >= 2:
        separation_array = np.asarray(separations, dtype=np.float64)
        separation_norm = np.linalg.norm(separation_array, axis=1)
        separation_p2p = float(np.ptp(separation_norm))
        separation_std = float(np.max(np.std(separation_array, axis=0)))
        if separation_p2p > config.eef_separation_peak_to_peak_m:
            reasons.append("eef_separation_peak_to_peak")
        if separation_std > config.eef_separation_axis_std_m:
            reasons.append("eef_separation_axis_std")
    else:
        reasons.append("insufficient_eef_separation_samples")

    direct_plane_run: list[_ClearanceSceneSample] = []
    previous_frame_id: int | None = None
    previous_observation_sequence: int | None = None
    previous_capture_timestamp_s: float | None = None
    previous_accepted_monotonic_s: float | None = None
    previous_stack_source: str | None = None
    previous_stack_z_m: float | None = None
    previous_stack_uncertainty_m: float | None = None
    continuity_rejection: str | None = None
    for accepted_window_index, scene in enumerate(scene_window, start=1):
        pose_source = str(_read_field(scene, "held_box_pose_source", ""))
        distinct = pose_source == "fresh_dual_eef_fixed_ready_nominal_box_offset"
        held = _read_field(scene, "held_box_bottom_z_base_m")
        held_uncertainty_field = "held_box_bottom_uncertainty_m"
        invalid_reason = "fixed_ready_box_bottom_geometry_invalid"
        stack_source = str(_read_field(scene, "stack_top_source", ""))
        stack_source_valid = stack_source in {
            "complete_stack_plane",
            "metric_stack_plane_candidate",
            "metric_fixed_outer_l_corner_plane",
            "metric_inner_opening_plane",
            "metric_coarse_l_corner_plane",
            "metric_forward_edge_pair_plane",
        }
        stack = _read_field(scene, "stack_top_z_base_m")
        frame_id_raw = _read_field(scene, "frame_id")
        observation_sequence_raw = _read_field(
            scene,
            "accepted_observation_sequence",
            accepted_window_index,
        )
        timestamp_raw = _read_field(scene, "capture_timestamp_s")
        accepted_raw = _read_field(scene, "accepted_monotonic_s")
        try:
            frame_id = int(frame_id_raw)
            observation_sequence = int(observation_sequence_raw)
            capture_timestamp_s = float(timestamp_raw)
            accepted_monotonic_s = float(accepted_raw)
        except (TypeError, ValueError):
            direct_plane_run.clear()
            previous_frame_id = None
            previous_observation_sequence = None
            previous_capture_timestamp_s = None
            previous_accepted_monotonic_s = None
            previous_stack_source = None
            previous_stack_z_m = None
            previous_stack_uncertainty_m = None
            continuity_rejection = "clearance_eef_box_bottom_frame_identity_unavailable"
            continue
        identity_valid = (
            not isinstance(frame_id_raw, bool)
            and frame_id >= 0
            and frame_id_raw == frame_id
            and not isinstance(observation_sequence_raw, bool)
            and observation_sequence > 0
            and observation_sequence_raw == observation_sequence
            and math.isfinite(capture_timestamp_s)
            and not isinstance(accepted_raw, bool)
            and math.isfinite(accepted_monotonic_s)
        )
        capture_age_at_acceptance_s = (
            accepted_monotonic_s - capture_timestamp_s
        )
        accepted_fresh = (
            identity_valid
            and 0.0 <= capture_age_at_acceptance_s
            <= config.held_top_sample_fresh_after_s
        )
        if (
            not identity_valid
            or not distinct
            or held is None
            or stack is None
            or not accepted_fresh
        ):
            direct_plane_run.clear()
            previous_frame_id = None
            previous_observation_sequence = None
            previous_capture_timestamp_s = None
            previous_accepted_monotonic_s = None
            previous_stack_source = None
            previous_stack_z_m = None
            previous_stack_uncertainty_m = None
            continuity_rejection = (
                "clearance_evidence_stale_at_acceptance"
                if identity_valid and not accepted_fresh
                else invalid_reason
            )
            continue

        held_value = float(held)
        stack_value = float(stack)
        held_sigma = float(_read_field(scene, held_uncertainty_field, math.nan))
        stack_sigma = float(_read_field(scene, "stack_top_uncertainty_m", math.nan))
        values_valid = (
            all(
                math.isfinite(value)
                for value in (held_value, stack_value, held_sigma, stack_sigma)
            )
            and held_sigma >= 0.0
            and stack_sigma >= 0.0
            and stack_source_valid
        )
        if not values_valid:
            direct_plane_run.clear()
            previous_frame_id = None
            previous_observation_sequence = None
            previous_capture_timestamp_s = None
            previous_accepted_monotonic_s = None
            previous_stack_source = None
            previous_stack_z_m = None
            previous_stack_uncertainty_m = None
            continuity_rejection = (
                "fixed_ready_stack_plane_source_invalid"
                if not stack_source_valid
                else invalid_reason
            )
            continue

        if previous_frame_id is not None:
            if frame_id <= previous_frame_id:
                direct_plane_run.clear()
                continuity_rejection = "clearance_frame_not_monotonic"
            elif (
                previous_observation_sequence is None
                or observation_sequence <= previous_observation_sequence
                or previous_capture_timestamp_s is None
                or capture_timestamp_s <= previous_capture_timestamp_s
                or previous_accepted_monotonic_s is None
                or accepted_monotonic_s <= previous_accepted_monotonic_s
            ):
                direct_plane_run.clear()
                continuity_rejection = "clearance_evidence_not_monotonic"
            elif stack_source != previous_stack_source:
                source_dt_s = accepted_monotonic_s - previous_accepted_monotonic_s
                previous_stack_upper = (
                    float(previous_stack_z_m)
                    + float(previous_stack_uncertainty_m)
                    + config.held_top_std_m
                )
                previous_stack_lower = (
                    float(previous_stack_z_m)
                    - float(previous_stack_uncertainty_m)
                    - config.held_top_std_m
                )
                current_stack_upper = stack_value + stack_sigma + config.held_top_std_m
                current_stack_lower = stack_value - stack_sigma - config.held_top_std_m
                height_consistent = (
                    current_stack_lower <= previous_stack_upper
                    and previous_stack_lower <= current_stack_upper
                )
                if (
                    source_dt_s < -1e-12
                    or source_dt_s > config.clearance_evidence_fresh_after_s
                    or not height_consistent
                ):
                    direct_plane_run.clear()
                    continuity_rejection = "clearance_stack_height_source_change_rejected"

        direct_plane_run.append(
            _ClearanceSceneSample(
                held_z_m=held_value,
                held_uncertainty_m=held_sigma,
                stack_z_m=stack_value,
                stack_uncertainty_m=stack_sigma,
                capture_timestamp_s=capture_timestamp_s,
                accepted_monotonic_s=accepted_monotonic_s,
            )
        )
        previous_frame_id = frame_id
        previous_observation_sequence = observation_sequence
        previous_capture_timestamp_s = capture_timestamp_s
        previous_accepted_monotonic_s = accepted_monotonic_s
        previous_stack_source = stack_source
        previous_stack_z_m = stack_value
        previous_stack_uncertainty_m = stack_sigma

    required_direct_frames = config.held_top_direct_plane_dwell_frames
    if len(direct_plane_run) > required_direct_frames:
        direct_plane_run = direct_plane_run[-required_direct_frames:]
    held_top_z = [sample.held_z_m for sample in direct_plane_run]
    held_top_uncertainty = [
        sample.held_uncertainty_m for sample in direct_plane_run
    ]
    stack_top_z = [sample.stack_z_m for sample in direct_plane_run]
    stack_top_uncertainty = [
        sample.stack_uncertainty_m for sample in direct_plane_run
    ]
    accepted_timestamps = [
        sample.accepted_monotonic_s for sample in direct_plane_run
    ]
    capture_ages_at_acceptance = [
        sample.accepted_monotonic_s - sample.capture_timestamp_s
        for sample in direct_plane_run
    ]

    held_std: float | None = None
    held_drift: float | None = None
    clearance: float | None = None
    box_bottom_lower_bound: float | None = None
    stack_top_upper_bound: float | None = None
    evidence_span_s: float | None = None
    latest_age_s: float | None = None
    max_capture_age_at_acceptance_s: float | None = None
    clearance_source = "fixed_ready_dual_eef_box_bottom_to_stack_plane"
    if len(held_top_z) < required_direct_frames:
        reasons.append(
            continuity_rejection or "insufficient_clearance_box_bottom_samples"
        )
        reasons.append("insufficient_clearance_box_bottom_samples")
    else:
        evidence_span_s = accepted_timestamps[-1] - accepted_timestamps[0]
        latest_age_s = now_s - accepted_timestamps[-1]
        max_capture_age_at_acceptance_s = max(capture_ages_at_acceptance)
        if (
            not math.isfinite(latest_age_s)
            or latest_age_s < 0.0
            or latest_age_s > config.clearance_evidence_fresh_after_s
        ):
            reasons.append("clearance_eef_box_bottom_evidence_stale")
        if (
            not math.isfinite(evidence_span_s)
            or evidence_span_s < 0.0
            or evidence_span_s > config.clearance_scene_max_span_s
        ):
            reasons.append("clearance_evidence_span_too_long")
        held_array = np.asarray(held_top_z, dtype=np.float64)
        held_std = float(np.std(held_array))
        held_drift = max(0.0, float(held_array[0] - held_array[-1]))
        if float(np.ptp(held_array)) > config.held_top_peak_to_peak_m:
            reasons.append("fixed_ready_box_bottom_peak_to_peak")
        if held_std > config.held_top_std_m:
            reasons.append("fixed_ready_box_bottom_std")
        if held_drift > config.held_top_downward_drift_m:
            reasons.append("fixed_ready_box_bottom_downward_drift")
        box_bottom_lower_bound = min(
            z - uncertainty
            for z, uncertainty in zip(held_top_z, held_top_uncertainty, strict=True)
        )
        stack_top_upper_bound = max(
            z + uncertainty
            for z, uncertainty in zip(
                stack_top_z, stack_top_uncertainty, strict=True
            )
        )
        clearance = box_bottom_lower_bound - stack_top_upper_bound
        if clearance < config.minimum_clearance_m:
            reasons.append("insufficient_vertical_clearance")

    result = GripContinuityResult(
        passed=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        evaluated_monotonic_s=now_s,
        state_sample_count=len(states),
        scene_sample_count=len(direct_plane_run),
        dwell_s=dwell_s,
        arm_tracking_error_max_rad=arm_error_max,
        eef_separation_peak_to_peak_m=separation_p2p,
        eef_separation_axis_std_max_m=separation_std,
        held_top_std_m=held_std,
        held_top_downward_drift_m=held_drift,
        clearance_lower_bound_m=clearance,
        box_bottom_z_lower_bound_m=box_bottom_lower_bound,
        stack_top_z_upper_bound_m=stack_top_upper_bound,
        scene_evidence_span_s=evidence_span_s,
        scene_latest_age_s=latest_age_s,
        scene_max_capture_age_at_acceptance_s=(
            max_capture_age_at_acceptance_s
        ),
        clearance_source=clearance_source,
        fixed_ready_geometry_only_authorized=True,
    )
    return result




class RBY1PalletController:
    """One-owner RB-Y1 Model M v1.2 combined body/mobility controller."""

    def __init__(
        self,
        robot: Any | None = None,
        *,
        execute: bool = False,
        config: PalletControlConfig | None = None,
        sdk_module: Any | None = None,
        owner_epoch: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        fk_provider: Callable[[np.ndarray, np.ndarray], Any] | None = None,
    ) -> None:
        self.config = config or PalletControlConfig()
        self._robot = robot
        self._execute = bool(execute)
        self._sdk = sdk_module
        self._clock = clock
        self._fk_provider = fk_provider
        self._owner_epoch = owner_epoch or f"pallet-{uuid.uuid4()}"
        self._source_handoff: LoadedSlot1ReadyBootstrap | None = None
        self._active_right_arm_target_rad = self.config.ready_pose.right_arm_rad
        self._active_left_arm_target_rad = self.config.ready_pose.left_arm_rad
        self._arm_stream_mode = ArmStreamMode.JOINT_READY_HOLD
        self._cartesian_arm_target: CartesianArmTarget | None = None
        self._placement_started = False
        self._placement_last_reason: str | None = None
        self._descent_plan: PlacementDescentPlan | None = None
        self._last_acknowledged_body_target_token: str | None = None
        self._grip_result: GripContinuityResult | None = None

        self._condition = threading.Condition(threading.RLock())
        self._sdk_send_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pump_thread: threading.Thread | None = None
        self._stream: Any | None = None
        self._phase = ControllerPhase.DISCONNECTED
        self._state_update_started = False

        self._model: Any | None = None
        self._indices: dict[str, np.ndarray] = {}
        self._dyn_model: Any | None = None
        self._dyn_state: Any | None = None
        self._latest_state: MeasuredRobotState | None = None
        self._state_history: deque[MeasuredRobotState] = deque(maxlen=512)
        self._state_sequence = 0
        self._wheel_stopped_since_s: float | None = None

        self._command_sequence = 0
        self._last_acknowledged_command_sequence = 0
        self._ready_transition_command_count = 0
        self._transition_id: CommandId | None = None
        self._transition_feedback: Any | None = None
        self._ready_ack: ReadyTransitionAck | None = None
        self._steady_send_count = 0
        self._steady_running_count = 0
        self._latest_proposal = ZERO_MOBILITY
        self._latest_proposal_s = self._clock()
        self._proposal_generation = 0
        self._sent_generation = -1
        self._last_sent_mobility = ZERO_MOBILITY
        self._last_send_s: float | None = None
        self._maximum_send_gap_s = 0.0
        self._zero_latched = False
        self._zero_latch_generation: int | None = None
        self._zero_latch_command_sequence: int | None = None
        self._shutdown_next_owner: str | None = None
        self._pending_handoff: HandoffAck | None = None
        self._accepted_handoff: HandoffAck | None = None
        self._last_error: Exception | None = None
        self._last_feedback_error: str | None = None

        self._body_target_token = self._make_body_target_token()

    @property
    def execute_enabled(self) -> bool:
        return self._execute

    @property
    def owner_epoch(self) -> str:
        return self._owner_epoch

    @property
    def phase(self) -> ControllerPhase:
        with self._condition:
            return self._phase

    @property
    def is_connected(self) -> bool:
        with self._condition:
            return self._phase not in (
                ControllerPhase.DISCONNECTED,
                ControllerPhase.CLOSED,
            )

    @property
    def stream_is_open(self) -> bool:
        with self._condition:
            return self._stream is not None

    @property
    def body_target_token(self) -> str:
        return self._body_target_token

    def connect(self) -> None:
        """Connect only after explicit execution authorization and validate identity."""

        if not self._execute:
            raise RobotMotionDisabledError(
                "robot execution is disabled; construct with execute=True explicitly"
            )
        with self._condition:
            if self._phase == ControllerPhase.CLOSED:
                raise PalletControlError("controller is already closed")
            if self._phase != ControllerPhase.DISCONNECTED:
                return

        if self._sdk is None:
            try:
                self._sdk = importlib.import_module("rby1_sdk")
            except ImportError as exc:
                raise PalletControlError(
                    "rby1_sdk is required only for explicitly enabled robot execution"
                ) from exc
        if self._robot is None:
            self._robot = self._sdk.create_robot(self.config.address, "m")

        try:
            connected = False
            is_connected = getattr(self._robot, "is_connected", None)
            if callable(is_connected):
                connected = bool(is_connected())
            if not connected:
                connect = getattr(self._robot, "connect", None)
                if not callable(connect):
                    raise PalletControlError("robot object does not provide connect()")
                result = connect()
                if result is False:
                    raise PalletControlError(
                        f"failed to connect to RB-Y1 at {self.config.address}"
                    )
                if callable(is_connected) and not bool(is_connected()):
                    raise PalletControlError(
                        f"RB-Y1 connection is not active at {self.config.address}"
                    )
            self._validate_robot_identity()
            self._initialize_model_state()
            self._start_measured_state_updates()
            self._wait_for_initial_state()
        except Exception:
            self._stop_measured_state_updates_best_effort()
            raise

        with self._condition:
            self._phase = ControllerPhase.CONNECTED
            self._condition.notify_all()

    def bootstrap_loaded_slot1_ready(
        self,
        *,
        loaded_box_acknowledged: bool,
    ) -> LoadedSlot1ReadyBootstrap:
        """Adopt an already-held box only from this measured ready session.

        The caller must explicitly acknowledge that the box is still held at
        the configured slot-1 ready posture.  This method opens no stream and
        sends no command; it installs local provenance and the immutable
        loaded Cartesian hold used by the first combined-stream packet.
        """

        self._require_execute_and_connected()
        if not isinstance(loaded_box_acknowledged, bool) or not loaded_box_acknowledged:
            raise CommandOwnershipError(
                "loaded slot-1 ready bootstrap requires explicit loaded-box "
                "acknowledgement"
            )

        with self._condition:
            if self._phase != ControllerPhase.CONNECTED:
                raise CommandOwnershipError(
                    "loaded slot-1 ready bootstrap is allowed only before the "
                    f"combined stream starts; phase={self._phase.value}"
                )
            if self._stream is not None:
                raise CommandOwnershipError(
                    "loaded slot-1 ready bootstrap refuses an already-open stream"
                )
            if self._pump_thread is not None and self._pump_thread.is_alive():
                raise CommandOwnershipError(
                    "loaded slot-1 ready bootstrap refuses a running pump"
                )
            if self._ready_transition_command_count != 0:
                raise CommandOwnershipError(
                    "loaded slot-1 ready bootstrap must precede the ready transition"
                )
            if self._source_handoff is not None:
                raise CommandOwnershipError(
                    "loaded slot-1 ready bootstrap cannot replace existing provenance"
                )

        state = self.get_measured_state()
        missing: list[str] = []
        if state.T_base_head is None:
            missing.append("T_base_head")
        if state.T_base_right_eef is None:
            missing.append("T_base_right_eef")
        if state.T_base_left_eef is None:
            missing.append("T_base_left_eef")
        if state.T_odom_base is None:
            missing.append("T_odom_base")
        if state.base_twist_w_vx_vy is None:
            missing.append("base_twist_w_vx_vy")
        if missing:
            detail = state.kinematics_error or state.odometry_error
            raise MeasuredStateError(
                "loaded slot-1 ready bootstrap requires fresh FK/odometry: "
                f"missing={tuple(missing)} detail={detail}"
            )

        errors, all_ready = self._ready_joint_errors(state)
        max_error = float(np.max(np.abs(errors)))
        if not all_ready or max_error > self.config.ready_tolerance_rad:
            raise ReadyTransitionError(
                "loaded slot-1 ready bootstrap requires measured configured "
                "ready posture within 1.0 deg; "
                f"max_joint_error_deg={math.degrees(max_error):.3f}, "
                f"all_ready={all_ready}"
            )

        cartesian_hold = self._make_cartesian_arm_target(
            state,
            mode=ArmStreamMode.CARTESIAN_LOADED_HOLD,
            base_z_offset_m=0.0,
            squeeze_offset_m=self.config.placement_squeeze_offset_m,
            release_spread_m=0.0,
        )
        pose = self.config.ready_pose
        provenance = LoadedSlot1ReadyBootstrap(
            owner_epoch=self._owner_epoch,
            source_phase=ControllerPhase.CONNECTED.value,
            measured_state_sequence=state.sequence,
            measured_robot_state_timestamp_s=state.robot_timestamp_s,
            measured_monotonic_s=state.received_monotonic_s,
            acknowledged_loaded_box=True,
            max_joint_error_rad=max_error,
            torso_target_rad=pose.torso_rad,
            right_arm_target_rad=pose.right_arm_rad,
            left_arm_target_rad=pose.left_arm_rad,
            head_target_rad=pose.head_rad,
            right_stiffness=ARM_STIFFNESS,
            left_stiffness=ARM_STIFFNESS,
            torque_policy=TORQUE_POLICY,
        )
        with self._condition:
            if self._phase != ControllerPhase.CONNECTED or self._stream is not None:
                raise CommandOwnershipError(
                    "loaded slot-1 ready bootstrap lost exclusive pre-stream state"
                )
            self._active_right_arm_target_rad = pose.right_arm_rad
            self._active_left_arm_target_rad = pose.left_arm_rad
            self._arm_stream_mode = ArmStreamMode.CARTESIAN_LOADED_HOLD
            self._cartesian_arm_target = cartesian_hold
            self._source_handoff = provenance
            self._body_target_token = self._make_body_target_token()
            self._condition.notify_all()
        return provenance

    def send_ready_transition_once(
        self,
        minimum_time_s: float = 5.0,
        *,
        on_send_attempt: Callable[[], None] | None = None,
    ) -> CommandId:
        """Send exactly one zero-base, five-component ready transition.

        RB-Y1 binds a command stream to the controller types in its first
        packet.  The loaded arms therefore start in Cartesian impedance here;
        later lowering and release targets must reuse that same stream type.

        ``on_send_attempt`` runs after stream creation and command construction,
        immediately before transport I/O.  The containment layer uses that
        boundary to distinguish harmless pre-send failures from an ambiguous
        transport failure that may already have reached the robot.
        """

        self._require_execute_and_connected()
        if on_send_attempt is not None and not callable(on_send_attempt):
            raise TypeError("on_send_attempt must be callable when provided")
        minimum_time_s = _positive(minimum_time_s, "minimum_time_s")
        minimum_time_s = max(
            minimum_time_s,
            self.config.ready_transition_minimum_time_s,
        )
        with self._condition:
            if self._ready_transition_command_count != 0:
                raise ReadyTransitionError(
                    "PALLET_READY_TRANSITION has already been sent; it cannot be replayed"
                )
            if self._phase != ControllerPhase.CONNECTED:
                raise ReadyTransitionError(
                    f"cannot start ready transition from phase {self._phase.value}"
                )
            if self._source_handoff is None:
                raise ReadyTransitionError(
                    "ready transition requires same-process loaded-ready provenance"
                )
            self._command_sequence += 1
            command_id = CommandId(self._owner_epoch, self._command_sequence)
            self._ready_transition_command_count = 1
            self._transition_id = command_id
            self._phase = ControllerPhase.PALLET_READY_TRANSITION

        command = self._build_combined_command(
            ZERO_MOBILITY,
            minimum_time_s=minimum_time_s,
        )
        with self._condition:
            self._ensure_stream_locked()
            stream = self._stream
        if stream is None:  # defensive: _ensure_stream_locked must assign it
            raise ReadyTransitionError("combined command stream was not created")
        if on_send_attempt is not None:
            on_send_attempt()
        try:
            with self._sdk_send_lock:
                feedback = stream.send_command(
                    command,
                    timeout_ms=self.config.send_timeout_ms,
                )
        except Exception as exc:
            self._record_fatal_stream_error(exc)
            raise ReadyTransitionError(
                f"failed to send exactly-once ready transition: {exc}"
            ) from exc

        with self._condition:
            self._transition_feedback = feedback
            self._condition.notify_all()
        return command_id

    def wait_ready_transition_ack(
        self,
        command_id: CommandId,
    ) -> ReadyTransitionAck:
        """Require all five feedback branches and measured joints within 1 degree."""

        with self._condition:
            if command_id != self._transition_id:
                raise ReadyTransitionError("ready transition command id does not match")
            stream = self._stream
            first_feedback = self._transition_feedback
        if stream is None:
            raise ReadyTransitionError("combined command stream is not open")

        deadline_s = self._clock() + self.config.transition_timeout_s
        latest_ack: ReadyTransitionAck | None = None
        feedback = first_feedback
        while self._clock() < deadline_s:
            if feedback is None:
                try:
                    with self._sdk_send_lock:
                        feedback = stream.request_feedback(
                            timeout_ms=self.config.send_timeout_ms
                        )
                except Exception as exc:
                    self._record_fatal_stream_error(exc)
                    raise ReadyTransitionError(
                        f"ready transition feedback failed: {exc}"
                    ) from exc

            latest_ack = self._make_ready_ack(command_id, feedback)
            raise_for_terminal_feedback(
                latest_ack.feedback,
                "ready transition",
                error_type=CombinedStreamError,
            )
            if latest_ack.ready_within(self.config.ready_tolerance_rad):
                with self._condition:
                    self._ready_ack = latest_ack
                    self._condition.notify_all()
                return latest_ack

            feedback = None
            with self._condition:
                self._condition.wait(
                    timeout=min(0.05, max(0.0, deadline_s - self._clock()))
                )

        detail = "no feedback"
        if latest_ack is not None:
            detail = (
                f"components={latest_ack.feedback}, "
                f"max_joint_error_deg={math.degrees(latest_ack.max_joint_error_rad):.3f}, "
                f"all_ready={latest_ack.all_target_joints_ready}"
            )
        raise ReadyTransitionError(f"ready transition timed out: {detail}")

    def ready_within(self, tolerance_rad: float | None = None) -> bool:
        tolerance = (
            self.config.ready_tolerance_rad
            if tolerance_rad is None
            else _positive(tolerance_rad, "tolerance_rad")
        )
        try:
            state = self.get_measured_state()
        except MeasuredStateError:
            return False
        errors, all_ready = self._ready_joint_errors(state)
        return all_ready and bool(np.max(np.abs(errors)) <= tolerance)

    def start_combined_stream(self) -> None:
        """Start the fixed-rate steady hold pump after transition verification."""

        self._require_execute_and_connected()
        with self._condition:
            if self._phase is ControllerPhase.FAULT_HOLD:
                raise CombinedStreamError(
                    self._last_feedback_error
                    or "steady whole-body stream is permanently fault-latched"
                )
            if self._ready_ack is None or not self._ready_ack.ready_within(
                self.config.ready_tolerance_rad
            ):
                raise ReadyTransitionError(
                    "steady pump requires component ack and measured ready joints"
                )
            if self._pump_thread is not None and self._pump_thread.is_alive():
                return
            if self._stream is None:
                raise CombinedStreamError("combined command stream is not open")
            if (
                self._arm_stream_mode is not ArmStreamMode.CARTESIAN_LOADED_HOLD
                or self._cartesian_arm_target is None
                or self._cartesian_arm_target.mode
                is not ArmStreamMode.CARTESIAN_LOADED_HOLD
            ):
                raise ReadyTransitionError(
                    "steady pump requires the first-packet Cartesian loaded hold"
                )
            self._stop_event.clear()
            self._latest_proposal = ZERO_MOBILITY
            self._latest_proposal_s = self._clock()
            self._placement_started = False
            self._placement_last_reason = None
            self._proposal_generation += 1
            startup_generation = self._proposal_generation
            self._phase = ControllerPhase.STEADY_HOLD
            self._last_error = None
            self._pump_thread = threading.Thread(
                target=self._run_steady_pump,
                name="rby1-pallet-combined-command-owner",
                daemon=False,
            )
            self._pump_thread.start()

        deadline_s = self._clock() + self.config.startup_timeout_s
        with self._condition:
            while (
                self._steady_running_count < 1
                or self._sent_generation < startup_generation
            ):
                self._raise_background_error_locked()
                if self._phase is ControllerPhase.FAULT_HOLD:
                    raise CombinedStreamError(
                        self._last_feedback_error
                        or "steady whole-body stream entered permanent FAULT_HOLD"
                    )
                remaining_s = deadline_s - self._clock()
                if remaining_s <= 0.0:
                    raise CombinedStreamError(
                        "steady whole-body pump did not receive all-component Running feedback"
                    )
                self._condition.wait(timeout=remaining_s)

    def ensure_persistent_zero_body_hold(self) -> None:
        """Recover an opened destination stream into a non-resumable safe hold.

        This is the containment path for faults between the first transition
        command and normal steady-stream startup.  It never authorizes motion:
        mobility is latched to zero and the handed-off arm targets remain
        unchanged.  Missing source provenance or an unopened stream fails
        closed.
        """

        self._require_execute_and_connected()
        with self._condition:
            if self._source_handoff is None:
                raise CombinedStreamError(
                    "cannot establish a carried-load hold without same-process "
                    "loaded-ready provenance"
                )
            if self._stream is None:
                raise CombinedStreamError(
                    "cannot establish a carried-load hold before the stream is open"
                )
            if self._pump_thread is not None and self._pump_thread.is_alive():
                pump_running = True
            else:
                pump_running = False
                previous_error = self._last_error
                if previous_error is not None:
                    self._last_feedback_error = (
                        "fault recovery retained prior stream error: "
                        f"{previous_error}"
                    )
                self._last_error = None
                self._stop_event.clear()
                self._latest_proposal = ZERO_MOBILITY
                self._latest_proposal_s = self._clock()
                self._proposal_generation += 1
                self._zero_latched = True
                self._zero_latch_generation = self._proposal_generation
                self._phase = ControllerPhase.FAULT_HOLD
                startup_generation = self._proposal_generation
                baseline_running_count = self._steady_running_count
                self._pump_thread = threading.Thread(
                    target=self._run_steady_pump,
                    name="rby1-pallet-fault-zero-body-hold-owner",
                    daemon=False,
                )
                self._pump_thread.start()

        if pump_running:
            self.send_zero_mobility_hold(latch=True)
            with self._condition:
                self._phase = ControllerPhase.FAULT_HOLD
                self._condition.notify_all()
            return

        deadline_s = self._clock() + self.config.startup_timeout_s
        with self._condition:
            while (
                self._steady_running_count <= baseline_running_count
                or self._sent_generation < startup_generation
                or not self._last_sent_mobility.is_zero
            ):
                if self._last_error is not None:
                    raise CombinedStreamError(
                        "fault zero/body-hold pump failed: " f"{self._last_error}"
                    ) from self._last_error
                remaining_s = deadline_s - self._clock()
                if remaining_s <= 0.0:
                    raise CombinedStreamError(
                        "fault recovery could not confirm persistent zero/body hold"
                    )
                self._condition.wait(timeout=remaining_s)
            self._zero_latch_command_sequence = self._last_acknowledged_command_sequence
            self._condition.notify_all()

    def send_cycle(
        self,
        mobility: MobilityCommand | Any,
        *,
        owner_epoch: str | None = None,
    ) -> int:
        """Publish a bounded proposal; only the fixed-rate owner touches the SDK."""

        command = self._normalize_mobility(mobility)
        self._validate_mobility(command)
        requested_owner = self._owner_epoch if owner_epoch is None else owner_epoch
        if requested_owner != self._owner_epoch:
            raise CommandOwnershipError(
                f"command owner {requested_owner!r} does not match {self._owner_epoch!r}"
            )

        now_s = self._clock()
        source_s = command.source_timestamp_s
        if (
            source_s is not None
            and now_s - source_s + 1e-12 >= self.config.command_stale_after_s
        ):
            command = ZERO_MOBILITY
        with self._condition:
            placement_started = self._placement_started
        if placement_started and not command.is_zero:
            raise CombinedStreamError("placement mode permits zero mobility only")
        if not command.is_zero:
            self._require_nonzero_motion_evidence(now_s)

        with self._condition:
            self._raise_background_error_locked()
            if self._pump_thread is None or not self._pump_thread.is_alive():
                raise CombinedStreamError("steady whole-body pump is not running")
            if self._zero_latched and not command.is_zero:
                raise CombinedStreamError("mobility is zero-latched")
            if (
                self._phase
                in (
                    ControllerPhase.SHUTDOWN_PENDING,
                    ControllerPhase.HANDOFF_ACKNOWLEDGED,
                    ControllerPhase.FAULT_HOLD,
                )
                and not command.is_zero
            ):
                raise CombinedStreamError(
                    f"phase {self._phase.value} permits zero mobility only"
                )
            self._latest_proposal = command
            self._latest_proposal_s = now_s
            self._proposal_generation += 1
            generation = self._proposal_generation
            self._condition.notify_all()
            return generation

    def send_zero_mobility_hold(self, *, latch: bool = True) -> None:
        """Select persistent zero mobility while retaining every body component."""

        with self._condition:
            self._raise_background_error_locked()
            if self._pump_thread is None or not self._pump_thread.is_alive():
                raise CombinedStreamError("steady whole-body pump is not running")
            if latch:
                self._zero_latched = True
            self._latest_proposal = ZERO_MOBILITY
            self._latest_proposal_s = self._clock()
            self._proposal_generation += 1
            generation = self._proposal_generation
            if latch:
                self._zero_latch_generation = generation
            self._condition.notify_all()

        deadline_s = self._clock() + self.config.zero_ack_timeout_s
        with self._condition:
            while (
                self._sent_generation < generation
                or not self._last_sent_mobility.is_zero
            ):
                self._raise_background_error_locked()
                remaining_s = deadline_s - self._clock()
                if remaining_s <= 0.0:
                    raise CombinedStreamError(
                        "timed out waiting for acknowledged zero-mobility body hold"
                    )
                self._condition.wait(timeout=remaining_s)
            if latch:
                self._zero_latch_command_sequence = (
                    self._last_acknowledged_command_sequence
                )

    def resume_mobility(self, *, owner_epoch: str) -> None:
        """Clear an operational zero latch; shutdown and fault latches stay permanent."""

        if owner_epoch != self._owner_epoch:
            raise CommandOwnershipError("only the active owner may clear a zero latch")
        with self._condition:
            if self._phase != ControllerPhase.STEADY_HOLD:
                raise CombinedStreamError(
                    f"zero latch cannot be cleared in phase {self._phase.value}"
                )
            self._zero_latched = False
            self._zero_latch_generation = None
            self._zero_latch_command_sequence = None
            self._latest_proposal = ZERO_MOBILITY
            self._latest_proposal_s = self._clock()
            self._proposal_generation += 1
            self._condition.notify_all()

    def get_measured_state(
        self,
        *,
        max_age_s: float | None = None,
    ) -> MeasuredRobotState:
        age_limit = (
            self.config.state_stale_after_s
            if max_age_s is None
            else _positive(max_age_s, "max_age_s")
        )
        with self._condition:
            state = self._latest_state
        if state is None:
            raise MeasuredStateError("no measured robot state has arrived")
        age_s = state.age_s(self._clock())
        if age_s > age_limit:
            raise MeasuredStateError(
                f"measured robot state is stale: age={age_s:.3f}s limit={age_limit:.3f}s"
            )
        return state

    def get_measured_T_base_head(self) -> np.ndarray:
        state = self.get_measured_state()
        if state.T_base_head is None:
            raise MeasuredStateError(
                f"fresh measured head FK is unavailable: {state.kinematics_error}"
            )
        return np.array(state.T_base_head, copy=True)

    def get_measured_T_base_torso(self) -> np.ndarray:
        state = self.get_measured_state()
        T_base_torso = self._T_base_torso_for_cartesian(state)
        if T_base_torso is None:
            raise MeasuredStateError(
                f"fresh measured torso FK is unavailable: {state.kinematics_error}"
            )
        return np.array(T_base_torso, copy=True)

    def get_measured_eef_transforms(self) -> tuple[np.ndarray, np.ndarray]:
        state = self.get_measured_state()
        if state.T_base_right_eef is None or state.T_base_left_eef is None:
            raise MeasuredStateError(
                f"fresh measured EEF FK is unavailable: {state.kinematics_error}"
            )
        return (
            np.array(state.T_base_right_eef, copy=True),
            np.array(state.T_base_left_eef, copy=True),
        )

    def start_cartesian_lowering_hold(
        self,
        *,
        descent_plan: PlacementDescentPlan,
        squeeze_offset_m: float | None = None,
    ) -> CartesianArmTarget:
        """Switch the persistent stream to bilateral Cartesian lowering targets.

        The target comes from a frozen ``PlacementDescentPlan`` built after
        arrival and wheel-stop.  The 50 mm value remains a pre-motion clearance
        floor; it is not the commanded descent distance.  Entry is allowed only
        after an acknowledged zero latch, a fresh wheel-stop dwell, and healthy
        Running stream feedback.
        """

        self._validate_descent_plan(descent_plan)
        requested_squeeze = (
            None
            if squeeze_offset_m is None
            else _positive(squeeze_offset_m, "squeeze_offset_m")
        )
        with self._condition:
            self._require_cartesian_entry_locked()
            loaded_target = self._cartesian_arm_target
            if (
                loaded_target is None
                or loaded_target.mode is not ArmStreamMode.CARTESIAN_LOADED_HOLD
            ):
                self._placement_fail_closed_locked("loaded_hold_target_unavailable")
                raise CombinedStreamError(
                    "cartesian lowering requires the acknowledged loaded-hold target"
                )
        state = self.get_measured_state()
        self._validate_cartesian_state(state)
        if self.config.place_pose is not None:
            target = self._make_place_pose_target(
                state,
                loaded_target=loaded_target,
                descent_plan=descent_plan,
                place_pose=self.config.place_pose,
            )
        else:
            target = self._make_lowering_target_from_loaded_hold(
                state,
                loaded_target=loaded_target,
                descent_plan=descent_plan,
                requested_squeeze_offset_m=requested_squeeze,
            )
        with self._condition:
            self._require_cartesian_entry_locked()
            if self._cartesian_arm_target is not loaded_target:
                self._placement_fail_closed_locked("loaded_hold_target_changed")
                raise CombinedStreamError(
                    "loaded-hold target changed while constructing lowering target"
                )
            self._arm_stream_mode = target.mode
            self._cartesian_arm_target = target
            self._descent_plan = descent_plan
            self._placement_started = True
            self._placement_last_reason = None
            self._zero_latched = True
            self._latest_proposal = ZERO_MOBILITY
            self._latest_proposal_s = self._clock()
            self._proposal_generation += 1
            self._body_target_token = self._make_body_target_token()
            self._condition.notify_all()
        return target

    def _make_lowering_target_from_loaded_hold(
        self,
        state: MeasuredRobotState,
        *,
        loaded_target: CartesianArmTarget,
        descent_plan: PlacementDescentPlan,
        requested_squeeze_offset_m: float | None,
    ) -> CartesianArmTarget:
        """Lower from the immutable placement plan, not a fixed config delta."""

        self._validate_cartesian_state(state)
        self._validate_descent_plan(descent_plan)
        if loaded_target.mode is not ArmStreamMode.CARTESIAN_LOADED_HOLD:
            raise ValueError("loaded_target must be a loaded-hold target")
        adopted_squeeze = float(loaded_target.squeeze_offset_m)
        if (
            requested_squeeze_offset_m is not None
            and abs(float(requested_squeeze_offset_m) - adopted_squeeze) > 1e-9
        ):
            raise ValueError(
                "squeeze_offset_m must match the acknowledged loaded-hold target"
            )
        T_base_torso = self._T_base_torso_for_cartesian(state)
        if T_base_torso is None:
            raise MeasuredStateError("T_base_torso unexpectedly missing")
        if state.T_base_right_eef is None or state.T_base_left_eef is None:
            raise MeasuredStateError("EEF FK unexpectedly missing")
        self._validate_descent_plan_matches_state(descent_plan, state)
        # Continue from the acknowledged loaded-hold command, not from the
        # compliant measured wrists.  Re-basing on measurement would ratchet
        # or discard the virtual squeeze that keeps the carton supported.
        right_base = np.array(
            loaded_target.right_T_base_eef,
            dtype=np.float64,
            copy=True,
        )
        left_base = np.array(
            loaded_target.left_T_base_eef,
            dtype=np.float64,
            copy=True,
        )
        right_base[2, 3] -= descent_plan.planned_delta_z_m
        left_base[2, 3] -= descent_plan.planned_delta_z_m
        T_torso_base = np.linalg.inv(T_base_torso)
        axis = tuple(float(value) for value in loaded_target.inter_eef_axis_base)
        return CartesianArmTarget(
            mode=ArmStreamMode.CARTESIAN_PLACEMENT_LOWERING,
            right_T_torso_eef=T_torso_base @ right_base,
            left_T_torso_eef=T_torso_base @ left_base,
            right_T_base_eef=right_base,
            left_T_base_eef=left_base,
            right_nullspace_joint_rad=loaded_target.right_nullspace_joint_rad,
            left_nullspace_joint_rad=loaded_target.left_nullspace_joint_rad,
            inter_eef_axis_base=axis,
            created_monotonic_s=self._clock(),
            source_state_sequence=state.sequence,
            lowering_distance_m=descent_plan.planned_delta_z_m,
            squeeze_offset_m=adopted_squeeze,
            release_spread_m=loaded_target.release_spread_m,
            descent_plan_id=descent_plan.plan_id,
        )

    def _kinematics_at_place_pose(
        self,
        state: MeasuredRobotState,
        place_pose: PlacePose,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward kinematics of the demonstrated posture.

        The measured joint vector supplies every joint the posture does not name
        (the wheels and the head), so only the torso and both arms are replaced.
        Returns ``(T_base_torso, T_base_right_eef, T_base_left_eef)`` at that
        posture, which is what the Cartesian arm targets must be expressed
        against.
        """

        position = np.array(state.position_rad, dtype=np.float64, copy=True)
        for name, target in (
            ("torso", place_pose.torso_rad),
            ("right_arm", place_pose.right_arm_rad),
            ("left_arm", place_pose.left_arm_rad),
        ):
            indices = self._indices.get(name)
            if indices is None:
                raise MeasuredStateError(
                    f"place pose requires measured {name} joint indices"
                )
            position[indices] = np.asarray(target, dtype=np.float64)
        velocity = np.zeros_like(position)
        try:
            T_torso, _T_head, T_right, T_left, _twist = (
                self._compute_measured_kinematics(position, velocity)
            )
        except Exception as exc:
            with self._condition:
                self._placement_fail_closed_locked("place_pose_kinematics_unavailable")
            raise MeasuredStateError(
                f"cannot compute forward kinematics at the place pose: {exc}"
            ) from exc
        return T_torso, T_right, T_left

    def _make_place_pose_target(
        self,
        state: MeasuredRobotState,
        *,
        loaded_target: CartesianArmTarget,
        descent_plan: PlacementDescentPlan,
        place_pose: PlacePose,
    ) -> CartesianArmTarget:
        """Command the demonstrated placement posture as Cartesian EEF targets.

        The arms are bound to Cartesian impedance by the first stream packet, so
        a joint target for them would be accepted and ignored.  The posture is
        therefore converted with forward kinematics and paced by per-target
        velocity limits sized to cover the required travel in
        ``placement_place_pose_duration_s``.
        """

        self._validate_cartesian_state(state)
        if state.T_base_right_eef is None or state.T_base_left_eef is None:
            raise MeasuredStateError("EEF FK unexpectedly missing")
        if descent_plan.target_source != "demonstrated_place_pose":
            with self._condition:
                self._placement_fail_closed_locked("place_pose_plan_source_mismatch")
            raise CombinedStreamError(
                "a demonstrated place pose requires a plan frozen for it; the plan "
                f"declares {descent_plan.target_source!r}"
            )
        self._validate_descent_plan_matches_state(descent_plan, state)
        T_base_torso_place, right_base, left_base = self._kinematics_at_place_pose(
            state, place_pose
        )
        right_travel = float(
            np.linalg.norm(right_base[:3, 3] - state.T_base_right_eef[:3, 3])
        )
        left_travel = float(
            np.linalg.norm(left_base[:3, 3] - state.T_base_left_eef[:3, 3])
        )
        travel = max(right_travel, left_travel)
        duration = self.config.placement_place_pose_duration_s
        # Pace the move so it spans the requested duration, but never exceed the
        # reviewed placement ceiling.
        linear_limit = min(
            self.config.placement_linear_velocity_limit_mps,
            max(travel / duration, 1e-4),
        )
        rotation_error = max(
            _rotation_error_rad(state.T_base_right_eef, right_base),
            _rotation_error_rad(state.T_base_left_eef, left_base),
        )
        angular_limit = min(
            self.config.placement_angular_velocity_limit_radps,
            max(rotation_error / duration, 1e-4),
        )
        axis = right_base[:3, 3] - left_base[:3, 3]
        axis_norm = float(np.linalg.norm(axis))
        if not math.isfinite(axis_norm) or axis_norm < 0.10:
            with self._condition:
                self._placement_fail_closed_locked("place_pose_inter_eef_axis_invalid")
            raise MeasuredStateError("place pose inter-EEF axis is invalid")
        T_torso_base = np.linalg.inv(T_base_torso_place)
        right_nullspace = _finite_vector(
            place_pose.right_arm_rad, 7, "place_pose_right_arm_rad"
        )
        left_nullspace = _finite_vector(
            place_pose.left_arm_rad, 7, "place_pose_left_arm_rad"
        )
        return CartesianArmTarget(
            mode=ArmStreamMode.CARTESIAN_PLACEMENT_LOWERING,
            right_T_torso_eef=T_torso_base @ right_base,
            left_T_torso_eef=T_torso_base @ left_base,
            right_T_base_eef=right_base,
            left_T_base_eef=left_base,
            right_nullspace_joint_rad=right_nullspace,
            left_nullspace_joint_rad=left_nullspace,
            inter_eef_axis_base=tuple(float(value) for value in axis / axis_norm),
            created_monotonic_s=self._clock(),
            source_state_sequence=state.sequence,
            lowering_distance_m=travel,
            squeeze_offset_m=loaded_target.squeeze_offset_m,
            release_spread_m=loaded_target.release_spread_m,
            descent_plan_id=descent_plan.plan_id,
            target_source="demonstrated_place_pose",
            linear_velocity_limit_mps=linear_limit,
            angular_velocity_limit_radps=angular_limit,
        )

    def _validate_descent_plan(self, descent_plan: PlacementDescentPlan) -> None:
        if not isinstance(descent_plan, PlacementDescentPlan):
            raise TypeError("descent_plan must be a PlacementDescentPlan")
        if not descent_plan.valid:
            with self._condition:
                self._placement_fail_closed_locked("descent_plan_invalid")
            raise CombinedStreamError(
                "cartesian lowering requires a valid descent plan: "
                f"{descent_plan.rejection_reason}"
            )
        if descent_plan.min_delta_z_m < self.config.minimum_clearance_m - 1e-12:
            with self._condition:
                self._placement_fail_closed_locked("descent_plan_clearance_below_floor")
            raise CombinedStreamError(
                "descent plan violates the 50 mm pre-motion clearance floor"
            )
        if (
            descent_plan.box_bottom_z_lower_bound_m
            - descent_plan.stack_top_z_upper_bound_m
            < self.config.minimum_clearance_m - 1e-12
        ):
            with self._condition:
                self._placement_fail_closed_locked("descent_plan_bounds_below_floor")
            raise CombinedStreamError(
                "descent plan box-bottom/stack-top bounds violate the clearance floor"
            )

    def _validate_descent_plan_matches_state(
        self,
        descent_plan: PlacementDescentPlan,
        state: MeasuredRobotState,
    ) -> None:
        if state.T_base_right_eef is None or state.T_base_left_eef is None:
            raise MeasuredStateError("EEF FK unexpectedly missing")
        translation_tolerance_m = max(0.015, self.config.minimum_clearance_m * 0.25)
        rotation_tolerance_rad = math.radians(3.0)
        right_translation_error = float(
            np.linalg.norm(
                state.T_base_right_eef[:3, 3] - descent_plan.right_eef_base[:3, 3]
            )
        )
        left_translation_error = float(
            np.linalg.norm(
                state.T_base_left_eef[:3, 3] - descent_plan.left_eef_base[:3, 3]
            )
        )
        right_rotation_error = _rotation_error_rad(
            state.T_base_right_eef,
            descent_plan.right_eef_base,
        )
        left_rotation_error = _rotation_error_rad(
            state.T_base_left_eef,
            descent_plan.left_eef_base,
        )
        if (
            right_translation_error > translation_tolerance_m
            or left_translation_error > translation_tolerance_m
            or right_rotation_error > rotation_tolerance_rad
            or left_rotation_error > rotation_tolerance_rad
        ):
            with self._condition:
                self._placement_fail_closed_locked("descent_plan_eef_mismatch")
            raise MeasuredStateError(
                "current EEF FK does not match frozen descent plan provenance"
            )

    def start_cartesian_release_hold(
        self,
        *,
        release_spread_m: float | None = None,
    ) -> CartesianArmTarget:
        """Open both EEF targets outward along the base ``+/-Y`` axis only."""

        spread = (
            self.config.placement_release_spread_m
            if release_spread_m is None
            else _positive(release_spread_m, "release_spread_m")
        )
        if spread > self.config.placement_max_release_spread_m + 1e-12:
            raise ValueError("release_spread_m exceeds the configured release bound")
        state = self._require_cartesian_placement_continuation_state()
        with self._condition:
            lowering_target = self._cartesian_arm_target
            descent_plan = self._descent_plan
        if (
            lowering_target is None
            or lowering_target.mode is not ArmStreamMode.CARTESIAN_PLACEMENT_LOWERING
        ):
            raise CombinedStreamError(
                "cartesian release requires the acknowledged lowering target"
            )
        if descent_plan is None:
            with self._condition:
                self._placement_fail_closed_locked("release_descent_plan_unavailable")
            raise CombinedStreamError(
                "cartesian release requires the frozen descent plan that started "
                "the placement"
            )
        target = self._make_release_target_from_plan(
            state,
            lowering_target=lowering_target,
            descent_plan=descent_plan,
            release_spread_m=spread,
        )
        with self._condition:
            self._require_cartesian_continuation_locked()
            if self._cartesian_arm_target is not lowering_target:
                raise CombinedStreamError(
                    "lowering target changed while constructing release target"
                )
            self._arm_stream_mode = target.mode
            self._cartesian_arm_target = target
            self._placement_last_reason = None
            self._zero_latched = True
            self._latest_proposal = ZERO_MOBILITY
            self._latest_proposal_s = self._clock()
            self._proposal_generation += 1
            self._body_target_token = self._make_body_target_token()
            self._condition.notify_all()
        return target

    def fail_closed_cartesian_placement_hold(
        self,
        *,
        reason: str,
    ) -> CartesianArmTarget | None:
        """Abort active placement into exact-zero mobility and measured arm hold."""

        reason_text = str(reason).strip() or "placement_fault"
        try:
            state = self.get_measured_state()
            target = self._make_measured_cartesian_hold_target(
                state,
                mode=ArmStreamMode.CARTESIAN_PLACEMENT_FAIL_CLOSED_HOLD,
            )
        except MeasuredStateError:
            with self._condition:
                self._placement_fail_closed_locked(
                    f"{reason_text}:measured_cartesian_hold_unavailable"
                )
                if self._placement_started:
                    self._phase = ControllerPhase.FAULT_HOLD
                    self._condition.notify_all()
            return None

        with self._condition:
            if not self._placement_started:
                self._placement_fail_closed_locked(reason_text)
                return None
            self._arm_stream_mode = target.mode
            self._cartesian_arm_target = target
            self._placement_last_reason = reason_text
            self._zero_latched = True
            self._latest_proposal = ZERO_MOBILITY
            self._latest_proposal_s = self._clock()
            self._body_target_token = self._make_body_target_token()
            self._proposal_generation += 1
            self._condition.notify_all()
        return target

    def placement_telemetry(self) -> PlacementTelemetry:
        with self._condition:
            state = self._latest_state
            target = self._cartesian_arm_target
            wheel = self.wheel_stop_status()
            stream_running = self._stream_feedback_healthy_locked(self._clock())
            return PlacementTelemetry(
                arm_mode=self._arm_stream_mode,
                placement_started=self._placement_started,
                source_state_sequence=(
                    None if target is None else target.source_state_sequence
                ),
                target_created_monotonic_s=(
                    None if target is None else target.created_monotonic_s
                ),
                right_T_base_eef_target=(
                    None
                    if target is None
                    else np.array(target.right_T_base_eef, copy=True)
                ),
                left_T_base_eef_target=(
                    None if target is None else np.array(target.left_T_base_eef, copy=True)
                ),
                measured_T_base_torso_available=(
                    state is not None
                    and self._T_base_torso_for_cartesian(state) is not None
                ),
                measured_T_base_right_eef_available=(
                    state is not None and state.T_base_right_eef is not None
                ),
                measured_T_base_left_eef_available=(
                    state is not None and state.T_base_left_eef is not None
                ),
                zero_latched=self._zero_latched,
                wheel_stopped=wheel.stopped,
                stream_running=stream_running,
                target_acknowledged=bool(
                    self._last_acknowledged_body_target_token
                    == self._body_target_token
                ),
                acknowledged_command_sequence=(
                    self._last_acknowledged_command_sequence
                ),
                last_reason=self._placement_last_reason,
                descent_plan_id=None if target is None else target.descent_plan_id,
                lowering_distance_m=(
                    None if target is None else target.lowering_distance_m
                ),
                release_spread_m=(
                    None if target is None else target.release_spread_m
                ),
                release_axis_base=(
                    None if target is None else tuple(target.inter_eef_axis_base)
                ),
                release_axis_deviation_rad=(
                    None if target is None else target.release_axis_deviation_rad
                ),
            )

    def get_measured_odometry(self) -> tuple[np.ndarray, int, float]:
        """Return fresh ``T_odom_base``, source state sequence, and sample age.

        Odometry measures a previously authorized acquisition increment. It is
        never sufficient to authorize a new increment without fresh vision.
        """

        state = self.get_measured_state()
        if state.T_odom_base is None:
            raise MeasuredStateError(
                f"fresh measured odometry is unavailable: {state.odometry_error}"
            )
        return (
            np.array(state.T_odom_base, copy=True),
            state.sequence,
            state.age_s(self._clock()),
        )

    def wheel_stop_status(self) -> WheelStopStatus:
        now_s = self._clock()
        with self._condition:
            state = self._latest_state
            stopped_since_s = self._wheel_stopped_since_s
        if state is None:
            return WheelStopStatus(False, False, None, None, None, 0.0, None)
        fresh = state.age_s(now_s) <= self.config.state_stale_after_s
        twist = state.base_twist_w_vx_vy
        linear = None if twist is None else math.hypot(twist[1], twist[2])
        angular = None if twist is None else abs(twist[0])
        dwell_s = 0.0 if stopped_since_s is None else max(0.0, now_s - stopped_since_s)
        stopped = bool(
            fresh
            and linear is not None
            and angular is not None
            and linear < self.config.wheel_stop_linear_mps
            and angular < self.config.wheel_stop_angular_radps
            and dwell_s >= self.config.wheel_stop_dwell_s
        )
        return WheelStopStatus(
            fresh,
            stopped,
            linear,
            angular,
            state.wheel_max_abs_radps,
            dwell_s,
            state.sequence,
        )

    def wait_for_wheel_stop(self, timeout_s: float) -> WheelStopStatus:
        deadline_s = self._clock() + _positive(timeout_s, "timeout_s")
        with self._condition:
            while True:
                status = self.wheel_stop_status()
                if status.stopped:
                    return status
                remaining_s = deadline_s - self._clock()
                if remaining_s <= 0.0:
                    raise MeasuredStateError(
                        "fresh measured base velocity did not satisfy the wheel-stop dwell"
                    )
                self._condition.wait(
                    timeout=min(remaining_s, self.config.state_stale_after_s)
                )

    def reverify_wheel_stop_after_stream_start(
        self,
        timeout_s: float,
    ) -> WheelStopStatus:
        """Require a new full stop dwell after the steady pump is Running."""

        deadline_s = self._clock() + _positive(timeout_s, "timeout_s")
        with self._condition:
            initial_sequence = (
                0 if self._latest_state is None else self._latest_state.sequence
            )
            self._wheel_stopped_since_s = None
            self._condition.notify_all()
            while True:
                status = self.wheel_stop_status()
                if (
                    status.stopped
                    and status.measured_state_sequence is not None
                    and status.measured_state_sequence > initial_sequence
                ):
                    return status
                remaining_s = deadline_s - self._clock()
                if remaining_s <= 0.0:
                    raise MeasuredStateError(
                        "no fresh post-pump robot states completed the wheel-stop dwell"
                    )
                self._condition.wait(
                    timeout=min(remaining_s, self.config.state_stale_after_s)
                )

    def evaluate_grip_and_clearance_dwell(
        self,
        scene_window: Sequence[Any],
        *,
        allow_fixed_ready_geometry_only: bool = False,
    ) -> GripContinuityResult:
        """Evaluate loaded hold continuity and conservative vertical clearance.

        The loaded demo uses the freshly measured fixed-ready EEF box-bottom
        model.  Joint tracking, EEF stability, frame freshness, stack-plane,
        clearance, and command-ownership gates remain active.  F/T feedback is
        not used as a grip or motion gate.  Joint tracking is intentionally
        skipped while a Cartesian arm stream is active because loaded hold,
        lowering, and release deliberately command offsets from the unsqueezed
        ready joints.

        This is intentionally a rolling fresh motion interlock, not a
        stationary perception gate.  Step authorization, post-stop
        reacquisition, complete-hole handoff, and arrival verification impose
        their own measured-stationary windows.  Requiring these clearance
        samples to be stationary would revoke an already authorized coarse
        step or continuous fine alignment on its first moving frame.
        """

        if not isinstance(allow_fixed_ready_geometry_only, bool):
            raise TypeError("allow_fixed_ready_geometry_only must be a boolean")
        if not allow_fixed_ready_geometry_only:
            raise CommandOwnershipError(
                "loaded demo requires explicit fixed-ready geometry-only grip "
                "checking"
            )
        if not self.config.fixed_ready_geometry_only_commissioning_enabled:
            raise CommandOwnershipError(
                "fixed-ready geometry-only grip checking is not enabled by the "
                "reviewed grip-interlock configuration"
            )

        now_s = self._clock()
        sample_margin_s = 2.0 / self.config.state_update_rate_hz
        with self._condition:
            cartesian_arm_motion = self._arm_stream_mode in (
                ArmStreamMode.CARTESIAN_LOADED_HOLD,
                ArmStreamMode.CARTESIAN_PLACEMENT_LOWERING,
                ArmStreamMode.CARTESIAN_PLACEMENT_RELEASE,
            )
            states = [
                state
                for state in self._state_history
                if now_s - state.received_monotonic_s
                <= self.config.grip_dwell_s + sample_margin_s
            ]
        result = evaluate_grip_continuity(
            self.config,
            scene_window,
            now_s=now_s,
            cartesian_arm_motion=cartesian_arm_motion,
            states=states,
            ready_joint_errors=self._ready_joint_errors,
        )
        with self._condition:
            self._grip_result = result
            self._condition.notify_all()
        return result

    def transfer_owner(self, next_owner: str) -> HandoffAck:
        """Enter persistent shutdown hold and return a non-self-acknowledged offer."""

        next_owner = str(next_owner).strip()
        if not next_owner:
            raise ValueError("next_owner must not be empty")
        self.send_zero_mobility_hold(latch=True)
        wheel = self.wheel_stop_status()
        with self._condition:
            self._phase = ControllerPhase.SHUTDOWN_PENDING
            self._shutdown_next_owner = next_owner
            accepted_sequence = (
                self._zero_latch_command_sequence
                or self._last_acknowledged_command_sequence
            )
            pending = HandoffAck(
                source_owner_epoch=self._owner_epoch,
                next_owner=next_owner,
                acknowledged=False,
                accepted_command_sequence=accepted_sequence,
                body_target_token=self._body_target_token,
                zero_mobility=True,
                body_hold_included=True,
                wheel_stopped=wheel.stopped,
                timestamp_s=self._clock(),
                message="successor acknowledgement required; current hold remains active",
            )
            self._pending_handoff = pending
            self._condition.notify_all()
            return pending

    def acknowledge_handoff(self, acknowledgement: HandoffAck) -> HandoffAck:
        """Accept only a successor-generated acknowledgement of the pending offer."""

        with self._condition:
            pending = self._pending_handoff
        if pending is None:
            raise HandoffPendingError("no owner transfer is pending")
        if not acknowledgement.acknowledged:
            raise HandoffPendingError("successor acknowledgement flag is false")
        if acknowledgement.source_owner_epoch != pending.source_owner_epoch:
            raise HandoffPendingError("successor acknowledgement owner epoch mismatch")
        if acknowledgement.next_owner != pending.next_owner:
            raise HandoffPendingError("successor acknowledgement target mismatch")
        if acknowledgement.body_target_token != pending.body_target_token:
            raise HandoffPendingError(
                "successor did not acknowledge the same body target"
            )
        if (
            acknowledgement.accepted_command_sequence
            < pending.accepted_command_sequence
        ):
            raise HandoffPendingError(
                "successor acknowledgement predates the zero hold"
            )
        if not acknowledgement.zero_mobility or not acknowledgement.body_hold_included:
            raise HandoffPendingError(
                "successor acknowledgement must include body hold and zero mobility"
            )
        wheel = self.wheel_stop_status()
        if not wheel.stopped:
            raise HandoffPendingError(
                "fresh measured wheel-stop dwell is required before handoff acceptance"
            )
        accepted = replace(
            acknowledgement,
            wheel_stopped=True,
            timestamp_s=self._clock(),
        )
        with self._condition:
            self._accepted_handoff = accepted
            self._phase = ControllerPhase.HANDOFF_ACKNOWLEDGED
            self._condition.notify_all()
        return accepted

    def close(
        self,
        *,
        handoff_ack: HandoffAck | None = None,
        force: bool = False,
    ) -> bool:
        """Release resources only after successor ack, or via conspicuous force path.

        A normal call with an active stream returns ``False`` and continues the
        zero/body hold until a valid successor acknowledgement is supplied.
        """

        with self._condition:
            if self._phase == ControllerPhase.CLOSED:
                return True
            stream_open = self._stream is not None
        if not stream_open:
            self._stop_measured_state_updates_best_effort()
            with self._condition:
                self._phase = ControllerPhase.CLOSED
                self._condition.notify_all()
            return True

        if self._pending_handoff is None:
            next_owner = (
                handoff_ack.next_owner
                if handoff_ack is not None
                else "unspecified-successor"
            )
            if force:
                try:
                    self.transfer_owner(next_owner)
                except Exception as exc:
                    warnings.warn(
                        "could not establish persistent shutdown hold before forced "
                        f"cancellation: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    with self._condition:
                        self._phase = ControllerPhase.SHUTDOWN_PENDING
                        self._shutdown_next_owner = next_owner
                        self._condition.notify_all()
            else:
                self.transfer_owner(next_owner)
        if handoff_ack is not None:
            self.acknowledge_handoff(handoff_ack)

        with self._condition:
            acknowledged = self._accepted_handoff is not None
        if not acknowledged and not force:
            return False
        if force and not acknowledged:
            warnings.warn(
                "FORCED RB-Y1 stream cancellation: carried-load support continuity "
                "is not acknowledged by a successor",
                RuntimeWarning,
                stacklevel=2,
            )
            self._best_effort_zero_before_force()

        self._stop_pump_and_cancel_stream()
        self._stop_measured_state_updates_best_effort()
        with self._condition:
            self._phase = ControllerPhase.CLOSED
            self._condition.notify_all()
        return True

    def telemetry(self) -> StreamTelemetry:
        with self._condition:
            return StreamTelemetry(
                phase=self._phase,
                owner_epoch=self._owner_epoch,
                command_sequence=self._command_sequence,
                ready_transition_command_count=self._ready_transition_command_count,
                steady_send_count=self._steady_send_count,
                last_sent_mobility=self._last_sent_mobility,
                last_send_monotonic_s=self._last_send_s,
                maximum_send_gap_s=self._maximum_send_gap_s,
                zero_latched=self._zero_latched,
                shutdown_pending=self._phase == ControllerPhase.SHUTDOWN_PENDING,
                body_hold_included=self._stream is not None,
                mobility_included=self._stream is not None,
                right_arm_stiffness=ARM_STIFFNESS,
                left_arm_stiffness=ARM_STIFFNESS,
                torque_policy=TORQUE_POLICY,
                last_feedback_error=self._last_feedback_error,
                last_error=None if self._last_error is None else str(self._last_error),
                arm_mode=self._arm_stream_mode,
                placement_started=self._placement_started,
                placement_target_sequence=(
                    None
                    if self._cartesian_arm_target is None
                    else self._cartesian_arm_target.source_state_sequence
                ),
            )

    def ingest_robot_state(self, robot_state: Any) -> None:
        """State-update callback; public so an in-process owner can multiplex it."""

        received_s = self._clock()
        try:
            position = np.asarray(robot_state.position, dtype=np.float64)
            velocity = np.asarray(robot_state.velocity, dtype=np.float64)
            is_ready = np.asarray(robot_state.is_ready, dtype=np.bool_)
            expected_shape = (len(self._model.robot_joint_names),)
            if (
                position.shape != expected_shape
                or velocity.shape != expected_shape
                or is_ready.shape != expected_shape
                or not np.all(np.isfinite(position))
                or not np.all(np.isfinite(velocity))
            ):
                raise ValueError(
                    f"joint state shape must be {expected_shape} with finite values"
                )
        except Exception as exc:
            with self._condition:
                self._last_feedback_error = f"invalid robot state: {exc}"
                self._wheel_stopped_since_s = None
                self._condition.notify_all()
            return

        T_torso: np.ndarray | None = None
        T_head: np.ndarray | None = None
        T_right: np.ndarray | None = None
        T_left: np.ndarray | None = None
        base_twist: tuple[float, float, float] | None = None
        kinematics_error: str | None = None
        try:
            (
                T_torso,
                T_head,
                T_right,
                T_left,
                base_twist,
            ) = self._compute_measured_kinematics(position, velocity)
        except Exception as exc:
            kinematics_error = str(exc)

        mobility_indices = self._indices.get("mobility", np.empty(0, dtype=np.int64))
        wheel_max = (
            None
            if mobility_indices.size == 0
            else float(np.max(np.abs(velocity[mobility_indices])))
        )
        robot_timestamp_s = self._robot_timestamp_s(robot_state)
        T_odom_base: np.ndarray | None = None
        odometry_error: str | None = None
        try:
            T_odom_base = _readonly_se2_matrix(
                getattr(robot_state, "odometry"),
                "robot_state.odometry",
            )
        except Exception as exc:
            odometry_error = str(exc)

        position_copy = np.array(position, copy=True)
        velocity_copy = np.array(velocity, copy=True)
        ready_copy = np.array(is_ready, copy=True)
        position_copy.setflags(write=False)
        velocity_copy.setflags(write=False)
        ready_copy.setflags(write=False)

        with self._condition:
            self._state_sequence += 1
            measured = MeasuredRobotState(
                sequence=self._state_sequence,
                received_monotonic_s=received_s,
                robot_timestamp_s=robot_timestamp_s,
                position_rad=position_copy,
                velocity_radps=velocity_copy,
                is_ready=ready_copy,
                T_base_torso=T_torso,
                T_base_head=T_head,
                T_base_right_eef=T_right,
                T_base_left_eef=T_left,
                T_odom_base=T_odom_base,
                base_twist_w_vx_vy=base_twist,
                wheel_max_abs_radps=wheel_max,
                kinematics_error=kinematics_error,
                odometry_error=odometry_error,
            )
            self._latest_state = measured
            self._state_history.append(measured)
            self._update_wheel_stop_dwell_locked(measured)
            self._condition.notify_all()

    def _validate_robot_identity(self) -> None:
        get_info = getattr(self._robot, "get_robot_info", None)
        if not callable(get_info):
            raise RobotIdentityError("connected SDK cannot report robot model/version")
        info = get_info()
        model_name = str(getattr(info, "robot_model_name", "")).strip().upper()
        version = (
            str(getattr(info, "robot_model_version", ""))
            .strip()
            .lower()
            .removeprefix("v")
        )
        if model_name != EXPECTED_ROBOT_MODEL:
            raise RobotIdentityError(
                f"pallet controller requires Model M; controller reports {model_name!r}"
            )
        if version != EXPECTED_ROBOT_VERSION and not version.startswith(
            f"{EXPECTED_ROBOT_VERSION}."
        ):
            raise RobotIdentityError(
                "pallet controller requires Model M v1.2; controller reports "
                f"version {version!r}"
            )

    def _initialize_model_state(self) -> None:
        try:
            model = self._robot.model()
        except Exception as exc:
            raise PalletControlError(f"cannot read RB-Y1 model layout: {exc}") from exc
        model_name = str(getattr(model, "model_name", "")).strip().upper()
        if model_name and model_name != EXPECTED_ROBOT_MODEL:
            raise RobotIdentityError(
                f"SDK model layout is {model_name!r}, expected Model M"
            )
        self._model = model
        expected_lengths = {
            "mobility": 4,
            "torso": 6,
            "right_arm": 7,
            "left_arm": 7,
            "head": 2,
        }
        for name, expected in expected_lengths.items():
            raw = getattr(model, f"{name}_idx", None)
            if raw is None:
                raise PalletControlError(f"RB-Y1 model does not expose {name}_idx")
            indices = np.asarray(raw, dtype=np.int64)
            if indices.shape != (expected,):
                raise PalletControlError(
                    f"RB-Y1 {name}_idx has shape {indices.shape}, expected {(expected,)}"
                )
            self._indices[name] = indices

        if self._fk_provider is not None:
            return
        try:
            self._dyn_model = self._robot.get_dynamics()
            self._dyn_state = self._dyn_model.make_state(
                list(_DYN_LINK_NAMES),
                model.robot_joint_names,
            )
        except Exception as exc:
            self._dyn_model = None
            self._dyn_state = None
            self._last_feedback_error = f"measured FK initialization unavailable: {exc}"

    def _start_measured_state_updates(self) -> None:
        start = getattr(self._robot, "start_state_update", None)
        if not callable(start):
            raise MeasuredStateError("robot does not provide measured-state updates")
        start(self.ingest_robot_state, self.config.state_update_rate_hz)
        self._state_update_started = True

    def _wait_for_initial_state(self) -> None:
        deadline_s = self._clock() + self.config.initial_state_timeout_s
        with self._condition:
            while self._latest_state is None:
                remaining_s = deadline_s - self._clock()
                if remaining_s <= 0.0:
                    raise MeasuredStateError(
                        "timed out waiting for the first measured robot state"
                    )
                self._condition.wait(timeout=remaining_s)

    def _compute_measured_kinematics(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float]]:
        if self._fk_provider is not None:
            result = self._fk_provider(position, velocity)
            if isinstance(result, Mapping):
                T_torso = result["T_base_torso"]
                T_head = result["T_base_head"]
                T_right = result["T_base_right_eef"]
                T_left = result["T_base_left_eef"]
                twist = result["base_twist_w_vx_vy"]
            else:
                if len(result) == 5:
                    T_torso, T_head, T_right, T_left, twist = result
                else:
                    T_head, T_right, T_left, twist = result
                    T_torso = np.eye(4, dtype=np.float64)
        elif self._dyn_model is not None and self._dyn_state is not None:
            self._dyn_state.set_q(position)
            self._dyn_state.set_qdot(velocity)
            self._dyn_model.compute_forward_kinematics(self._dyn_state)
            T_torso = self._dyn_model.compute_transformation(
                self._dyn_state,
                _BASE_LINK_INDEX,
                _TORSO_LINK_INDEX,
            )
            T_head = self._dyn_model.compute_transformation(
                self._dyn_state,
                _BASE_LINK_INDEX,
                _HEAD_LINK_INDEX,
            )
            T_right = self._dyn_model.compute_transformation(
                self._dyn_state,
                _BASE_LINK_INDEX,
                _RIGHT_EEF_LINK_INDEX,
            )
            T_left = self._dyn_model.compute_transformation(
                self._dyn_state,
                _BASE_LINK_INDEX,
                _LEFT_EEF_LINK_INDEX,
            )
            twist = self._dyn_model.compute_mobility_diff_kinematics(self._dyn_state)
        else:
            raise MeasuredStateError("dynamics/FK provider is unavailable")

        twist_array = np.asarray(twist, dtype=np.float64)
        if twist_array.shape != (3,) or not np.all(np.isfinite(twist_array)):
            raise MeasuredStateError("measured base twist must be finite [w, vx, vy]")
        return (
            _readonly_matrix(T_torso, "T_base_torso"),
            _readonly_matrix(T_head, "T_base_head"),
            _readonly_matrix(T_right, "T_base_right_eef"),
            _readonly_matrix(T_left, "T_base_left_eef"),
            tuple(float(value) for value in twist_array),
        )

    def _ready_joint_errors(
        self,
        state: MeasuredRobotState,
    ) -> tuple[np.ndarray, bool]:
        pose = self.config.ready_pose
        parts = (
            ("torso", pose.torso_rad),
            ("right_arm", self._active_right_arm_target_rad),
            ("left_arm", self._active_left_arm_target_rad),
            ("head", pose.head_rad),
        )
        errors: list[np.ndarray] = []
        all_ready = True
        for name, target in parts:
            indices = self._indices[name]
            current = state.position_rad[indices]
            errors.append(current - np.asarray(target, dtype=np.float64))
            all_ready = all_ready and bool(np.all(state.is_ready[indices]))
        return np.concatenate(errors), all_ready

    def _make_ready_ack(
        self,
        command_id: CommandId,
        feedback: Any,
    ) -> ReadyTransitionAck:
        component_ack = parse_component_feedback(feedback)
        now_s = self._clock()
        try:
            state = self.get_measured_state()
        except MeasuredStateError:
            state = None
        if state is None:
            errors: tuple[float, ...] = ()
            maximum = float("inf")
            all_ready = False
            sequence = None
            age_s = None
        else:
            error_array, all_ready = self._ready_joint_errors(state)
            errors = tuple(float(value) for value in error_array)
            maximum = float(np.max(np.abs(error_array)))
            sequence = state.sequence
            age_s = state.age_s(now_s)
        return ReadyTransitionAck(
            command_id=command_id,
            feedback=component_ack,
            measured_state_sequence=sequence,
            measured_state_age_s=age_s,
            joint_errors_rad=errors,
            max_joint_error_rad=maximum,
            all_target_joints_ready=all_ready,
            received_monotonic_s=now_s,
        )

    def _run_steady_pump(self) -> None:
        period_s = 1.0 / self.config.send_rate_hz
        next_tick_s = self._clock()
        try:
            while not self._stop_event.is_set():
                now_s = self._clock()
                with self._condition:
                    proposal = self._latest_proposal
                    generation = self._proposal_generation
                    stale = (
                        now_s - self._latest_proposal_s + 1e-12
                        >= self.config.command_stale_after_s
                    )
                    force_zero = self._zero_latched or self._phase in (
                        ControllerPhase.SHUTDOWN_PENDING,
                        ControllerPhase.HANDOFF_ACKNOWLEDGED,
                        ControllerPhase.FAULT_HOLD,
                    )
                selected = ZERO_MOBILITY if stale or force_zero else proposal
                command, built_body_target_token = self._build_combined_command(
                    selected,
                    minimum_time_s=self.config.steady_minimum_time_s,
                    return_body_target_token=True,
                )
                with self._condition:
                    self._command_sequence += 1
                    command_sequence = self._command_sequence
                with self._sdk_send_lock:
                    feedback = self._stream.send_command(
                        command,
                        timeout_ms=self.config.send_timeout_ms,
                    )
                ack = parse_component_feedback(feedback)
                completed_s = self._clock()
                with self._condition:
                    if self._last_send_s is not None:
                        self._maximum_send_gap_s = max(
                            self._maximum_send_gap_s,
                            completed_s - self._last_send_s,
                        )
                    self._last_send_s = completed_s
                    self._steady_send_count += 1
                    if ack.running:
                        self._steady_running_count += 1
                        self._last_acknowledged_command_sequence = command_sequence
                        self._last_acknowledged_body_target_token = (
                            built_body_target_token
                        )
                        self._last_sent_mobility = selected
                        if not stale or proposal.is_zero:
                            self._sent_generation = max(
                                self._sent_generation, generation
                            )
                        if self._phase is ControllerPhase.FAULT_HOLD:
                            if selected.is_zero:
                                self._zero_latch_command_sequence = command_sequence
                        else:
                            self._last_feedback_error = None
                    else:
                        self._last_feedback_error = (
                            "steady feedback incomplete or non-Running; permanent "
                            "FAULT_HOLD zero latch engaged: "
                            f"{ack}"
                        )
                        self._phase = ControllerPhase.FAULT_HOLD
                        self._zero_latched = True
                        self._latest_proposal = ZERO_MOBILITY
                        self._latest_proposal_s = completed_s
                        self._proposal_generation += 1
                        self._zero_latch_generation = self._proposal_generation
                    self._condition.notify_all()

                next_tick_s += period_s
                delay_s = next_tick_s - self._clock()
                if delay_s <= 0.0:
                    next_tick_s = self._clock()
                else:
                    self._stop_event.wait(delay_s)
        except Exception as exc:
            self._record_fatal_stream_error(exc)

    def _stream_feedback_healthy_locked(self, now_s: float) -> bool:
        return bool(
            self._stream is not None
            and self._pump_thread is not None
            and self._pump_thread.is_alive()
            and self._phase is ControllerPhase.STEADY_HOLD
            and self._steady_running_count > 0
            and self._last_feedback_error is None
            and self._last_error is None
            and self._last_send_s is not None
            and now_s - self._last_send_s <= self.config.state_stale_after_s
        )

    def _exact_zero_latch_acked_locked(self) -> bool:
        return bool(
            self._zero_latched
            and self._zero_latch_command_sequence is not None
            and self._last_acknowledged_command_sequence
            >= self._zero_latch_command_sequence
            and self._last_sent_mobility.is_zero
        )

    def _placement_fail_closed_locked(self, reason: str) -> None:
        self._placement_last_reason = reason
        if self._placement_started:
            self._zero_latched = True
            self._latest_proposal = ZERO_MOBILITY
            self._latest_proposal_s = self._clock()
            self._proposal_generation += 1
            self._condition.notify_all()

    def _require_cartesian_entry_locked(self) -> None:
        now_s = self._clock()
        if self._placement_started:
            self._placement_fail_closed_locked("placement_already_started")
            raise CombinedStreamError("cartesian placement has already started")
        if not self._stream_feedback_healthy_locked(now_s):
            self._placement_fail_closed_locked("stream_not_healthy")
            raise CombinedStreamError("cartesian placement requires a healthy stream")
        if not self._exact_zero_latch_acked_locked():
            self._placement_fail_closed_locked("zero_latch_not_acknowledged")
            raise CombinedStreamError(
                "cartesian placement requires an acknowledged zero-mobility latch"
            )
        wheel = self.wheel_stop_status()
        if not wheel.stopped:
            self._placement_fail_closed_locked("wheel_stop_not_fresh")
            raise MeasuredStateError(
                "cartesian placement requires fresh stopped-wheel dwell"
            )

    def _require_cartesian_continuation_locked(self) -> None:
        now_s = self._clock()
        if not self._placement_started:
            self._placement_fail_closed_locked("placement_not_started")
            raise CombinedStreamError("cartesian release requires placement start")
        if not self._stream_feedback_healthy_locked(now_s):
            self._placement_fail_closed_locked("stream_not_healthy")
            raise CombinedStreamError("cartesian placement stream is not healthy")
        if not self._exact_zero_latch_acked_locked():
            self._placement_fail_closed_locked("zero_latch_not_acknowledged")
            raise CombinedStreamError(
                "cartesian placement continuation requires acknowledged zero latch"
            )
        wheel = self.wheel_stop_status()
        if not wheel.stopped:
            self._placement_fail_closed_locked("wheel_stop_not_fresh")
            raise MeasuredStateError(
                "cartesian placement continuation requires fresh wheel stop"
            )

    def _require_cartesian_placement_entry_state(self) -> MeasuredRobotState:
        with self._condition:
            self._require_cartesian_entry_locked()
        state = self.get_measured_state()
        self._validate_cartesian_state(state)
        return state

    def _require_cartesian_placement_continuation_state(self) -> MeasuredRobotState:
        with self._condition:
            self._require_cartesian_continuation_locked()
        state = self.get_measured_state()
        self._validate_cartesian_state(state)
        return state

    def _validate_cartesian_state(self, state: MeasuredRobotState) -> None:
        missing: list[str] = []
        if self._T_base_torso_for_cartesian(state) is None:
            missing.append("T_base_torso")
        if state.T_base_right_eef is None:
            missing.append("T_base_right_eef")
        if state.T_base_left_eef is None:
            missing.append("T_base_left_eef")
        if missing:
            with self._condition:
                self._placement_fail_closed_locked("cartesian_fk_unavailable")
            raise MeasuredStateError(
                "cartesian arm target requires fresh torso and EEF FK: "
                f"missing={tuple(missing)} detail={state.kinematics_error}"
            )

    def _make_cartesian_arm_target(
        self,
        state: MeasuredRobotState,
        *,
        mode: ArmStreamMode,
        base_z_offset_m: float,
        squeeze_offset_m: float,
        release_spread_m: float,
    ) -> CartesianArmTarget:
        self._validate_cartesian_state(state)
        T_base_torso = self._T_base_torso_for_cartesian(state)
        if T_base_torso is None:
            raise MeasuredStateError("T_base_torso unexpectedly missing")
        if state.T_base_right_eef is None or state.T_base_left_eef is None:
            raise MeasuredStateError("EEF FK unexpectedly missing")
        right_base = np.array(state.T_base_right_eef, dtype=np.float64, copy=True)
        left_base = np.array(state.T_base_left_eef, dtype=np.float64, copy=True)
        right_nullspace, left_nullspace = self._arm_joint_vectors_for_target(state)
        axis = right_base[:3, 3] - left_base[:3, 3]
        axis_norm = float(np.linalg.norm(axis))
        if not math.isfinite(axis_norm) or axis_norm < 0.10:
            with self._condition:
                self._placement_fail_closed_locked("inter_eef_axis_invalid")
            raise MeasuredStateError("measured inter-EEF axis is invalid")
        axis /= axis_norm
        base_z = np.asarray((0.0, 0.0, float(base_z_offset_m)), dtype=np.float64)
        right_base[:3, 3] += base_z - axis * squeeze_offset_m + axis * release_spread_m
        left_base[:3, 3] += base_z + axis * squeeze_offset_m - axis * release_spread_m
        T_torso_base = np.linalg.inv(T_base_torso)
        right_torso = T_torso_base @ right_base
        left_torso = T_torso_base @ left_base
        return CartesianArmTarget(
            mode=mode,
            right_T_torso_eef=right_torso,
            left_T_torso_eef=left_torso,
            right_T_base_eef=right_base,
            left_T_base_eef=left_base,
            right_nullspace_joint_rad=right_nullspace,
            left_nullspace_joint_rad=left_nullspace,
            inter_eef_axis_base=tuple(float(value) for value in axis),
            created_monotonic_s=self._clock(),
            source_state_sequence=state.sequence,
            lowering_distance_m=abs(float(base_z_offset_m)),
            squeeze_offset_m=squeeze_offset_m,
            release_spread_m=release_spread_m,
        )

    def _arm_joint_vectors_for_target(
        self,
        state: MeasuredRobotState,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Freeze redundant Cartesian DOFs near the measured arm posture."""

        right_indices = self._indices.get("right_arm")
        left_indices = self._indices.get("left_arm")
        if right_indices is not None and left_indices is not None:
            right = state.position_rad[right_indices]
            left = state.position_rad[left_indices]
            return (
                _finite_vector(right, 7, "measured_right_arm_rad"),
                _finite_vector(left, 7, "measured_left_arm_rad"),
            )
        raise MeasuredStateError(
            "Cartesian nullspace hold requires measured right/left arm indices"
        )

    def _make_release_target_from_plan(
        self,
        state: MeasuredRobotState,
        *,
        lowering_target: CartesianArmTarget,
        descent_plan: PlacementDescentPlan,
        release_spread_m: float,
    ) -> CartesianArmTarget:
        """Open from the frozen plan geometry along base ``+/-Y`` only.

        ``descent_plan.right_target_base`` already is "measured hand at plan
        freeze, lowered by the planned descent, no squeeze", so adding the
        spread here makes the commanded travel exactly one spread per hand and
        leaves base X and Z untouched.  Deriving the target from the frozen plan
        rather than the live wrists keeps it invariant to compliance, creep, or
        a stalled measured pose at the instant release is requested.
        """

        self._validate_cartesian_state(state)
        if lowering_target.mode is not ArmStreamMode.CARTESIAN_PLACEMENT_LOWERING:
            raise ValueError("lowering_target must be a placement-lowering target")
        if lowering_target.descent_plan_id != descent_plan.plan_id:
            with self._condition:
                self._placement_fail_closed_locked("release_plan_id_mismatch")
            raise CombinedStreamError(
                "release descent plan does not match the acknowledged lowering "
                f"target: {lowering_target.descent_plan_id!r} != "
                f"{descent_plan.plan_id!r}"
            )
        T_base_torso = self._T_base_torso_for_cartesian(state)
        if T_base_torso is None:
            raise MeasuredStateError("T_base_torso unexpectedly missing")
        max_deviation_rad = self.config.placement_release_axis_max_deviation_rad
        # The hands open along the torso-tip Y axis, matching the box-pick grip
        # convention.  Expressing it in base keeps the frozen plan arithmetic in
        # one frame while the commanded target is still emitted in torso coords.
        torso_y_axis_base = np.asarray(T_base_torso[:3, 1], dtype=np.float64)
        try:
            release_axis, deviation_rad = resolve_release_axis(
                lowering_target.inter_eef_axis_base,
                torso_y_axis_base,
                max_deviation_rad=max_deviation_rad,
            )
        except ValueError as exc:
            with self._condition:
                self._placement_fail_closed_locked("release_axis_unresolved")
            raise MeasuredStateError(
                f"base-Y release axis is unresolvable: {exc}"
            ) from exc
        if deviation_rad > max_deviation_rad:
            with self._condition:
                self._placement_fail_closed_locked("release_axis_deviation")
            raise MeasuredStateError(
                "measured inter-EEF axis deviates "
                f"{math.degrees(deviation_rad):.2f} deg from base Y; the limit is "
                f"{math.degrees(max_deviation_rad):.2f} deg"
            )
        outward = release_axis * float(release_spread_m)
        # Open from whatever the acknowledged lowering command asked for.  With a
        # demonstrated place posture that is the posture itself, so the frozen
        # plan's base-Z targets no longer describe where the hands are going.
        right_base = np.array(
            lowering_target.right_T_base_eef,
            dtype=np.float64,
            copy=True,
        )
        left_base = np.array(
            lowering_target.left_T_base_eef,
            dtype=np.float64,
            copy=True,
        )
        right_base[:3, 3] += outward
        left_base[:3, 3] -= outward
        T_torso_base = np.linalg.inv(T_base_torso)
        right_nullspace, left_nullspace = self._arm_joint_vectors_for_target(state)
        return CartesianArmTarget(
            mode=ArmStreamMode.CARTESIAN_PLACEMENT_RELEASE,
            right_T_torso_eef=T_torso_base @ right_base,
            left_T_torso_eef=T_torso_base @ left_base,
            right_T_base_eef=right_base,
            left_T_base_eef=left_base,
            right_nullspace_joint_rad=right_nullspace,
            left_nullspace_joint_rad=left_nullspace,
            inter_eef_axis_base=tuple(float(value) for value in release_axis),
            created_monotonic_s=self._clock(),
            source_state_sequence=state.sequence,
            lowering_distance_m=lowering_target.lowering_distance_m,
            squeeze_offset_m=0.0,
            release_spread_m=float(release_spread_m),
            descent_plan_id=descent_plan.plan_id,
            release_axis_deviation_rad=deviation_rad,
            target_source=lowering_target.target_source,
        )

    def _make_measured_cartesian_hold_target(
        self,
        state: MeasuredRobotState,
        *,
        mode: ArmStreamMode,
    ) -> CartesianArmTarget:
        self._validate_cartesian_state(state)
        T_base_torso = self._T_base_torso_for_cartesian(state)
        if T_base_torso is None:
            raise MeasuredStateError("T_base_torso unexpectedly missing")
        if state.T_base_right_eef is None or state.T_base_left_eef is None:
            raise MeasuredStateError("EEF FK unexpectedly missing")
        right_base = np.array(state.T_base_right_eef, dtype=np.float64, copy=True)
        left_base = np.array(state.T_base_left_eef, dtype=np.float64, copy=True)
        axis = right_base[:3, 3] - left_base[:3, 3]
        axis_norm = float(np.linalg.norm(axis))
        if not math.isfinite(axis_norm) or axis_norm < 0.10:
            raise MeasuredStateError("measured inter-EEF axis is invalid")
        axis /= axis_norm
        T_torso_base = np.linalg.inv(T_base_torso)
        right_nullspace, left_nullspace = self._arm_joint_vectors_for_target(state)
        return CartesianArmTarget(
            mode=mode,
            right_T_torso_eef=T_torso_base @ right_base,
            left_T_torso_eef=T_torso_base @ left_base,
            right_T_base_eef=right_base,
            left_T_base_eef=left_base,
            right_nullspace_joint_rad=right_nullspace,
            left_nullspace_joint_rad=left_nullspace,
            inter_eef_axis_base=tuple(float(value) for value in axis),
            created_monotonic_s=self._clock(),
            source_state_sequence=state.sequence,
        )

    def _T_base_torso_for_cartesian(
        self,
        state: MeasuredRobotState,
    ) -> np.ndarray | None:
        return state.T_base_torso

    def _build_combined_command(
        self,
        mobility: MobilityCommand,
        *,
        minimum_time_s: float,
        return_body_target_token: bool = False,
    ) -> Any:
        if self._sdk is None:
            raise CombinedStreamError("RB-Y1 SDK is not loaded")
        pose = self.config.ready_pose
        with self._condition:
            arm_mode = self._arm_stream_mode
            cartesian_target = self._cartesian_arm_target
            right_joint_target = self._active_right_arm_target_rad
            left_joint_target = self._active_left_arm_target_rad
            body_target_token = self._body_target_token

        def header() -> Any:
            return self._sdk.CommandHeaderBuilder().set_control_hold_time(
                self.config.control_hold_time_s
            )

        torso = (
            self._sdk.JointPositionCommandBuilder()
            .set_command_header(header())
            .set_minimum_time(minimum_time_s)
            .set_position(np.asarray(pose.torso_rad, dtype=np.float64))
        )
        head = (
            self._sdk.JointPositionCommandBuilder()
            .set_command_header(header())
            .set_minimum_time(minimum_time_s)
            .set_position(np.asarray(pose.head_rad, dtype=np.float64))
        )

        def arm(target: tuple[float, ...]) -> Any:
            # Match box-pick: omit the override so controller defaults remain active.
            return (
                self._sdk.JointImpedanceControlCommandBuilder()
                .set_command_header(header())
                .set_minimum_time(minimum_time_s)
                .set_position(np.asarray(target, dtype=np.float64))
                .set_stiffness(np.asarray(ARM_STIFFNESS, dtype=np.float64))
            )

        def cartesian_arm(
            link_name: str,
            target: np.ndarray,
            nullspace_joint_rad: tuple[float, ...],
            linear_velocity_limit: float,
            angular_velocity_limit: float,
        ) -> Any:
            # Match box-pick lift: controller defaults own per-joint saturation.
            cartesian_builder = getattr(
                self._sdk,
                "CartesianImpedanceControlCommandBuilder",
                None,
            )
            if not callable(cartesian_builder):
                raise CombinedStreamError(
                    "RB-Y1 SDK does not provide Cartesian impedance commands"
                )
            builder = (
                cartesian_builder()
                .set_command_header(header())
                .add_target(
                    TORSO_REFERENCE_LINK,
                    link_name,
                    np.asarray(target, dtype=np.float64),
                    linear_velocity_limit,
                    angular_velocity_limit,
                    self.config.placement_linear_acceleration_limit_mps2,
                    self.config.placement_angular_acceleration_limit_radps2,
                )
                .set_joint_stiffness(
                    np.asarray(self.config.placement_joint_stiffness, dtype=np.float64)
                )
                .set_joint_damping_ratio(self.config.placement_joint_damping_ratio)
                .set_minimum_time(minimum_time_s)
            )
            set_nullspace = getattr(builder, "set_nullspace_joint_target", None)
            if not callable(set_nullspace):
                raise CombinedStreamError(
                    "RB-Y1 Cartesian impedance builder lacks nullspace hold"
                )
            builder = set_nullspace(
                np.asarray(nullspace_joint_rad, dtype=np.float64),
                np.asarray(
                    self.config.placement_nullspace_weight,
                    dtype=np.float64,
                ),
                self.config.placement_nullspace_kp,
                self.config.placement_nullspace_kd,
                self.config.placement_nullspace_cost_weight,
            )
            set_reset_reference = getattr(builder, "set_reset_reference", None)
            if callable(set_reset_reference):
                builder = set_reset_reference(False)
            return builder

        if arm_mode is ArmStreamMode.JOINT_READY_HOLD or cartesian_target is None:
            right_arm = arm(right_joint_target)
            left_arm = arm(left_joint_target)
        else:
            linear_limit = (
                self.config.placement_linear_velocity_limit_mps
                if cartesian_target.linear_velocity_limit_mps is None
                else cartesian_target.linear_velocity_limit_mps
            )
            angular_limit = (
                self.config.placement_angular_velocity_limit_radps
                if cartesian_target.angular_velocity_limit_radps is None
                else cartesian_target.angular_velocity_limit_radps
            )
            right_arm = cartesian_arm(
                RIGHT_EEF_LINK,
                cartesian_target.right_T_torso_eef,
                cartesian_target.right_nullspace_joint_rad,
                linear_limit,
                angular_limit,
            )
            left_arm = cartesian_arm(
                LEFT_EEF_LINK,
                cartesian_target.left_T_torso_eef,
                cartesian_target.left_nullspace_joint_rad,
                linear_limit,
                angular_limit,
            )

        body = (
            self._sdk.BodyComponentBasedCommandBuilder()
            .set_torso_command(torso)
            .set_right_arm_command(right_arm)
            .set_left_arm_command(left_arm)
        )
        se2 = (
            self._sdk.SE2VelocityCommandBuilder()
            .set_command_header(header())
            .set_minimum_time(minimum_time_s)
            .set_velocity(
                np.asarray((mobility.vx_mps, mobility.vy_mps), dtype=np.float64),
                float(mobility.wz_radps),
            )
            .set_acceleration_limit(
                np.full(
                    2,
                    self.config.linear_acceleration_limit_mps2,
                    dtype=np.float64,
                ),
                self.config.angular_acceleration_limit_radps2,
            )
        )
        component = (
            self._sdk.ComponentBasedCommandBuilder()
            .set_mobility_command(se2)
            .set_body_command(body)
            .set_head_command(head)
        )
        command = self._sdk.RobotCommandBuilder().set_command(component)
        if return_body_target_token:
            return command, body_target_token
        return command

    def _normalize_mobility(self, mobility: MobilityCommand | Any) -> MobilityCommand:
        if isinstance(mobility, MobilityCommand):
            return mobility
        try:
            return MobilityCommand(
                vx_mps=float(mobility.vx_mps),
                vy_mps=float(mobility.vy_mps),
                wz_radps=float(mobility.wz_radps),
                source_timestamp_s=getattr(
                    mobility,
                    "source_timestamp_s",
                    getattr(mobility, "timestamp_s", None),
                ),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "mobility must expose finite vx_mps, vy_mps, and wz_radps"
            ) from exc

    def _validate_mobility(self, command: MobilityCommand) -> None:
        if command.linear_norm_mps > self.config.maximum_linear_speed_mps + 1e-12:
            raise ValueError(
                "linear velocity exceeds the configured pallet limit "
                f"{self.config.maximum_linear_speed_mps:.3f} m/s"
            )
        if abs(command.wz_radps) > self.config.maximum_angular_speed_radps + 1e-12:
            raise ValueError(
                "angular velocity exceeds the configured pallet limit "
                f"{self.config.maximum_angular_speed_radps:.3f} rad/s"
            )

    def _require_nonzero_motion_evidence(self, now_s: float) -> None:
        with self._condition:
            handoff = self._source_handoff
            grip_result = self._grip_result
        if handoff is None:
            raise CommandOwnershipError(
                "nonzero mobility requires same-process loaded-ready provenance"
            )
        if grip_result is None or not grip_result.passed:
            reasons = () if grip_result is None else grip_result.reasons
            raise CombinedStreamError(
                "nonzero mobility requires fresh measured grip/clearance evidence; "
                f"reasons={reasons}"
            )
        if (
            grip_result.fixed_ready_geometry_only_authorized
            and not isinstance(handoff, LoadedSlot1ReadyBootstrap)
        ):
            raise CommandOwnershipError(
                "fixed-ready geometry-only evidence is valid only for the local "
                "loaded slot-1 ready bootstrap"
            )
        if (
            now_s - grip_result.evaluated_monotonic_s
            > self.config.command_stale_after_s
        ):
            raise CombinedStreamError("grip/clearance evidence is stale")
        state = self.get_measured_state()
        if (
            state.T_base_head is None
            or state.T_base_right_eef is None
            or state.T_base_left_eef is None
            or state.base_twist_w_vx_vy is None
        ):
            raise MeasuredStateError(
                f"fresh measured FK/base velocity is unavailable: {state.kinematics_error}"
            )

    def _ensure_stream_locked(self) -> None:
        if self._stream is None:
            self._stream = self._robot.create_command_stream(
                priority=self.config.priority
            )

    def _require_execute_and_connected(self) -> None:
        if not self._execute:
            raise RobotMotionDisabledError("robot execution is disabled")
        with self._condition:
            if self._phase in (ControllerPhase.DISCONNECTED, ControllerPhase.CLOSED):
                raise PalletControlError("controller is not connected")

    def _update_wheel_stop_dwell_locked(self, state: MeasuredRobotState) -> None:
        twist = state.base_twist_w_vx_vy
        mobility_indices = self._indices.get("mobility", np.empty(0, dtype=np.int64))
        mobility_ready = bool(
            mobility_indices.size and np.all(state.is_ready[mobility_indices])
        )
        if twist is None:
            self._wheel_stopped_since_s = None
            return
        linear = math.hypot(twist[1], twist[2])
        angular = abs(twist[0])
        if (
            mobility_ready
            and linear < self.config.wheel_stop_linear_mps
            and angular < self.config.wheel_stop_angular_radps
        ):
            if self._wheel_stopped_since_s is None:
                self._wheel_stopped_since_s = state.received_monotonic_s
        else:
            self._wheel_stopped_since_s = None

    @staticmethod
    def _robot_timestamp_s(robot_state: Any) -> float | None:
        value = getattr(robot_state, "timestamp", None)
        if value is None:
            return None
        timestamp = getattr(value, "timestamp", None)
        try:
            result = float(timestamp() if callable(timestamp) else value)
        except (TypeError, ValueError, OverflowError):
            return None
        return result if math.isfinite(result) else None

    def _make_body_target_token(self) -> str:
        pose = self.config.ready_pose
        cartesian_payload: dict[str, Any] | None = None
        if self._cartesian_arm_target is not None:
            cartesian_payload = {
                "mode": self._cartesian_arm_target.mode.value,
                "right_T_torso_eef": self._cartesian_arm_target.right_T_torso_eef.tolist(),
                "left_T_torso_eef": self._cartesian_arm_target.left_T_torso_eef.tolist(),
            }
        payload = {
            "torso": pose.torso_rad,
            "right_arm": self._active_right_arm_target_rad,
            "left_arm": self._active_left_arm_target_rad,
            "head": pose.head_rad,
            "right_stiffness": ARM_STIFFNESS,
            "left_stiffness": ARM_STIFFNESS,
            "torque_policy": TORQUE_POLICY,
            "arm_stream_mode": self._arm_stream_mode.value,
            "cartesian": cartesian_payload,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _record_fatal_stream_error(self, exc: Exception) -> None:
        with self._condition:
            self._last_error = exc
            if self._phase not in (
                ControllerPhase.CLOSED,
                ControllerPhase.HANDOFF_ACKNOWLEDGED,
            ):
                self._phase = ControllerPhase.FAULT_HOLD
            self._zero_latched = True
            self._condition.notify_all()

    def _raise_background_error_locked(self) -> None:
        if self._last_error is not None:
            raise CombinedStreamError(
                f"combined command owner failed: {self._last_error}"
            ) from self._last_error

    def _best_effort_zero_before_force(self) -> None:
        try:
            if self._pump_thread is not None and self._pump_thread.is_alive():
                self.send_zero_mobility_hold(latch=True)
                return
            if self._stream is not None:
                command = self._build_combined_command(
                    ZERO_MOBILITY,
                    minimum_time_s=self.config.steady_minimum_time_s,
                )
                with self._sdk_send_lock:
                    self._stream.send_command(
                        command,
                        timeout_ms=self.config.send_timeout_ms,
                    )
        except Exception as exc:
            warnings.warn(
                f"best-effort zero/body hold before forced cancellation failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    def _stop_pump_and_cancel_stream(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
            thread = self._pump_thread
            stream = self._stream
        if thread is not None:
            thread.join(timeout=self.config.join_timeout_s)
        thread_alive = thread is not None and thread.is_alive()
        if stream is not None:
            try:
                stream.cancel()
            finally:
                completed = bool(stream.wait_for(self.config.shutdown_timeout_ms))
                if not completed:
                    raise CombinedStreamError(
                        "RB-Y1 combined stream did not finish after cancellation"
                    )
        if thread_alive and thread is not None:
            thread.join(timeout=self.config.join_timeout_s)
            if thread.is_alive():
                raise CombinedStreamError(
                    "combined command owner thread remained alive after stream cancellation"
                )
        with self._condition:
            self._stream = None
            self._pump_thread = None
            self._condition.notify_all()

    def _stop_measured_state_updates_best_effort(self) -> None:
        if not self._state_update_started or self._robot is None:
            return
        stop = getattr(self._robot, "stop_state_update", None)
        try:
            if callable(stop):
                stop()
        finally:
            self._state_update_started = False


__all__ = [
    "ARM_STIFFNESS",
    "ArmStreamMode",
    "CartesianArmTarget",
    "CommandId",
    "CommandOwnershipError",
    "CombinedStreamError",
    "ComponentFeedbackAck",
    "ControllerPhase",
    "EXPECTED_ROBOT_MODEL",
    "EXPECTED_ROBOT_VERSION",
    "GripContinuityResult",
    "HandoffAck",
    "HandoffPendingError",
    "HARD_MAX_ANGULAR_SPEED_RADPS",
    "HARD_MAX_LINEAR_SPEED_MPS",
    "READY_TRANSITION_MINIMUM_TIME_FLOOR_S",
    "LoadedSlot1ReadyBootstrap",
    "MeasuredRobotState",
    "MeasuredStateError",
    "MobilityCommand",
    "PalletControlConfig",
    "PalletControlError",
    "PlacePose",
    "PlacementTelemetry",
    "RBY1PalletController",
    "ReadyPose",
    "ReadyTransitionAck",
    "ReadyTransitionError",
    "RobotIdentityError",
    "RobotMotionDisabledError",
    "StreamTelemetry",
    "TORQUE_POLICY",
    "TORSO_REFERENCE_LINK",
    "WheelStopStatus",
    "ZERO_MOBILITY",
    "evaluate_grip_continuity",
    "resolve_release_axis",
]
