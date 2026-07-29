"""Pure slot-1 pallet alignment servo with no robot-SDK side effects.

The controller works entirely in metric RB-Y1 base coordinates.  It supports
the legacy held-carton-to-world-target contract and the explicit current-world-
feature-to-demonstrated-body-reference contract used by slot 1.  In both cases
a world point ``p`` is expressed in the moving base frame.  For
``xi_B = [vx, vy, wz]``,

``dot(p) = -v - wz * J * p``.

The controller analytically inverts that moving-frame Jacobian, then converts
the resulting base twist to the SDK mobility frame with an explicit SE(2)
adjoint.  This module deliberately contains no SDK import or I/O.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .angles import line_angle_difference_rad, normalize_line_angle_rad
from .mobile_servo import (
    MAX_ALLOWED_ANGULAR_SPEED_RADPS,
    MAX_ALLOWED_LINEAR_SPEED_MPS,
    VelocityCommand,
    ZERO_VELOCITY,
)


FloatArray = NDArray[np.float64]


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _xy(value: object, name: str) -> tuple[float, float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size < 2 or not np.all(np.isfinite(array[:2])):
        raise ValueError(f"{name} must contain at least two finite values")
    return (float(array[0]), float(array[1]))


def _signed_line_angle(angle_rad: float) -> float:
    return line_angle_difference_rad(float(angle_rad), 0.0)


class PalletServoState(str, Enum):
    """Logical states; every ``*_HOLD`` state commands exact zero mobility."""

    IDLE = "IDLE"
    ACQUIRING = "ACQUIRING"
    TRACKING = "TRACKING"
    ARRIVAL_EVIDENCE = "ARRIVAL_EVIDENCE"
    ARRIVAL_WHEEL_STOP = "ARRIVAL_WHEEL_STOP"
    PERCEPTION_HOLD = "PERCEPTION_HOLD"
    ARRIVED_HOLD = "ARRIVED_HOLD"
    FAULT_HOLD = "FAULT_HOLD"
    SHUTDOWN_PENDING_HOLD = "SHUTDOWN_PENDING_HOLD"

    @property
    def is_persistent_hold(self) -> bool:
        return self in {
            PalletServoState.PERCEPTION_HOLD,
            PalletServoState.ARRIVAL_WHEEL_STOP,
            PalletServoState.ARRIVED_HOLD,
            PalletServoState.FAULT_HOLD,
            PalletServoState.SHUTDOWN_PENDING_HOLD,
        }


@dataclass(frozen=True, slots=True)
class SE2FrameTransform:
    """Coordinate transform ``T_target_source`` for planar command frames.

    ``translation_xy_m`` is the source-frame origin expressed in the target
    frame.  For twist coordinates ``[vx, vy, wz]`` at each frame origin,

    ``v_target = R_target_source v_source - wz J t_target_source``.
    """

    translation_xy_m: tuple[float, float] = (0.0, 0.0)
    yaw_target_source_rad: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "translation_xy_m",
            _xy(self.translation_xy_m, "translation_xy_m"),
        )
        object.__setattr__(
            self,
            "yaw_target_source_rad",
            _finite(self.yaw_target_source_rad, "yaw_target_source_rad"),
        )

    @classmethod
    def from_matrix(cls, matrix: object) -> "SE2FrameTransform":
        transform = np.asarray(matrix, dtype=np.float64)
        if transform.shape == (4, 4):
            if not np.all(np.isfinite(transform)):
                raise ValueError("SE(2) transform must be finite")
            if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
                raise ValueError("invalid homogeneous transform bottom row")
            rotation_3d = transform[:3, :3]
            if not np.allclose(rotation_3d.T @ rotation_3d, np.eye(3), atol=1e-8):
                raise ValueError("transform rotation must be orthonormal")
            if not math.isclose(float(np.linalg.det(rotation_3d)), 1.0, abs_tol=1e-8):
                raise ValueError("transform rotation determinant must be +1")
            if not np.allclose(rotation_3d[2], (0.0, 0.0, 1.0), atol=1e-8):
                raise ValueError("mobility/base transform must be planar")
            transform = np.asarray(
                (
                    (transform[0, 0], transform[0, 1], transform[0, 3]),
                    (transform[1, 0], transform[1, 1], transform[1, 3]),
                    (0.0, 0.0, 1.0),
                ),
                dtype=np.float64,
            )
        if transform.shape != (3, 3) or not np.all(np.isfinite(transform)):
            raise ValueError("SE(2) transform must be a finite 3x3 or 4x4 matrix")
        if not np.allclose(transform[2], (0.0, 0.0, 1.0), atol=1e-9):
            raise ValueError("invalid SE(2) homogeneous bottom row")
        rotation = transform[:2, :2]
        if not np.allclose(rotation.T @ rotation, np.eye(2), atol=1e-8):
            raise ValueError("SE(2) rotation must be orthonormal")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-8):
            raise ValueError("SE(2) rotation determinant must be +1")
        return cls(
            translation_xy_m=(float(transform[0, 2]), float(transform[1, 2])),
            yaw_target_source_rad=math.atan2(
                float(rotation[1, 0]),
                float(rotation[0, 0]),
            ),
        )

    def transform_twist(self, command: VelocityCommand) -> VelocityCommand:
        """Apply ``Ad(T_target_source)`` to a source-frame twist."""

        yaw = self.yaw_target_source_rad
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        tx, ty = self.translation_xy_m
        vx = cosine * command.vx_mps - sine * command.vy_mps
        vy = sine * command.vx_mps + cosine * command.vy_mps
        # -J t * wz = [ty, -tx] * wz.
        vx += ty * command.wz_radps
        vy -= tx * command.wz_radps
        return VelocityCommand(vx, vy, command.wz_radps)

    def as_matrix(self) -> FloatArray:
        yaw = self.yaw_target_source_rad
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        tx, ty = self.translation_xy_m
        return np.asarray(
            ((cosine, -sine, tx), (sine, cosine, ty), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class PalletServoConfig:
    """Conservative slot-1 SE(2) gains, filters, and hard safety gates."""

    position_gain_per_s: float = 0.8
    yaw_gain_per_s: float = 0.8
    max_linear_speed_mps: float = 0.04
    max_angular_speed_radps: float = 0.06
    max_linear_acceleration_mps2: float = 0.12
    max_angular_acceleration_radps2: float = 0.18
    filter_window: int = 3
    jump_threshold_m: float = 0.030
    yaw_jump_threshold_rad: float = math.radians(15.0)
    jump_reseed_frames: int = 3
    stale_after_s: float = 0.30
    timeout_s: float = 30.0
    max_correction_m: float = 0.25
    start_yaw_limit_rad: float = math.radians(15.0)
    arrival_inner_m: float = 0.010
    arrival_outer_m: float = 0.015
    arrival_yaw_inner_rad: float = math.radians(3.0)
    arrival_yaw_outer_rad: float = math.radians(5.0)
    arrival_min_frames: int = 5
    arrival_min_duration_s: float = 0.35
    wheel_linear_stop_mps: float = 0.010
    wheel_angular_stop_radps: float = 0.020
    wheel_stop_duration_s: float = 0.35
    wheel_feedback_stale_after_s: float = 0.20
    expected_axis_branch: str = "image_right"
    mobility_from_base: SE2FrameTransform = field(default_factory=SE2FrameTransform)

    def __post_init__(self) -> None:
        positive_names = (
            "position_gain_per_s",
            "yaw_gain_per_s",
            "max_linear_speed_mps",
            "max_angular_speed_radps",
            "max_linear_acceleration_mps2",
            "max_angular_acceleration_radps2",
            "jump_threshold_m",
            "yaw_jump_threshold_rad",
            "stale_after_s",
            "timeout_s",
            "max_correction_m",
            "start_yaw_limit_rad",
            "arrival_inner_m",
            "arrival_outer_m",
            "arrival_yaw_inner_rad",
            "arrival_yaw_outer_rad",
            "arrival_min_duration_s",
            "wheel_linear_stop_mps",
            "wheel_angular_stop_radps",
            "wheel_stop_duration_s",
            "wheel_feedback_stale_after_s",
        )
        for name in positive_names:
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if (
            isinstance(self.filter_window, bool)
            or int(self.filter_window) != self.filter_window
        ):
            raise ValueError("filter_window must be an integer")
        if int(self.filter_window) != 3:
            raise ValueError("filter_window must be 3 for the pallet median filter")
        object.__setattr__(self, "filter_window", int(self.filter_window))
        if (
            isinstance(self.jump_reseed_frames, bool)
            or int(self.jump_reseed_frames) != self.jump_reseed_frames
        ):
            raise ValueError("jump_reseed_frames must be an integer")
        object.__setattr__(self, "jump_reseed_frames", int(self.jump_reseed_frames))
        if self.jump_reseed_frames < 2:
            raise ValueError("jump_reseed_frames must be at least 2")
        if (
            isinstance(self.arrival_min_frames, bool)
            or int(self.arrival_min_frames) != self.arrival_min_frames
        ):
            raise ValueError("arrival_min_frames must be an integer")
        object.__setattr__(self, "arrival_min_frames", int(self.arrival_min_frames))
        if self.arrival_min_frames < 5:
            raise ValueError("arrival_min_frames cannot be less than 5")
        if self.arrival_outer_m <= self.arrival_inner_m:
            raise ValueError("arrival_outer_m must exceed arrival_inner_m")
        if self.arrival_yaw_outer_rad <= self.arrival_yaw_inner_rad:
            raise ValueError("arrival_yaw_outer_rad must exceed arrival_yaw_inner_rad")
        hard_upper_bounds = {
            "max_linear_speed_mps": MAX_ALLOWED_LINEAR_SPEED_MPS,
            "max_angular_speed_radps": MAX_ALLOWED_ANGULAR_SPEED_RADPS,
            "jump_threshold_m": 0.030,
            "yaw_jump_threshold_rad": math.radians(15.0),
            "stale_after_s": 0.30,
            "max_correction_m": 0.25,
            "start_yaw_limit_rad": math.radians(15.0),
            "arrival_inner_m": 0.010,
            "arrival_outer_m": 0.015,
            "arrival_yaw_inner_rad": math.radians(3.0),
            "arrival_yaw_outer_rad": math.radians(5.0),
            "wheel_linear_stop_mps": 0.010,
            "wheel_angular_stop_radps": 0.020,
            "wheel_feedback_stale_after_s": 0.20,
        }
        for name, limit in hard_upper_bounds.items():
            if float(getattr(self, name)) > limit + 1e-12:
                raise ValueError(f"{name} exceeds its pallet safety limit {limit}")
        if self.arrival_min_duration_s < 0.35:
            raise ValueError("arrival_min_duration_s cannot be less than 0.35")
        if self.wheel_stop_duration_s < 0.35:
            raise ValueError("wheel_stop_duration_s cannot be less than 0.35")
        expected_branch = str(self.expected_axis_branch).strip()
        if not expected_branch:
            raise ValueError("expected_axis_branch must be nonempty")
        object.__setattr__(self, "expected_axis_branch", expected_branch)

    @classmethod
    def from_root_config(cls, value: Mapping[str, object]) -> "PalletServoConfig":
        """Build from either the root pallet config or a servo-only mapping.

        The canonical root layout is ``pallet.servo`` with a sibling
        ``pallet.wheel_stop`` and ``pallet.T_mobility_base``.  Direct
        ``pallet_servo``/``servo`` sections are accepted for replay tools.
        Unknown fields stay owned by their respective subsystems and are not
        forwarded into this safety-critical dataclass.
        """

        if not isinstance(value, Mapping):
            raise TypeError("root config must be a mapping")
        pallet_value = value.get("pallet", {})
        pallet = pallet_value if isinstance(pallet_value, Mapping) else {}
        servo_value = pallet.get(
            "servo",
            value.get("pallet_servo", value.get("servo", {})),
        )
        servo = servo_value if isinstance(servo_value, Mapping) else {}
        wheel_value = pallet.get("wheel_stop", servo.get("wheel_stop", {}))
        wheel = wheel_value if isinstance(wheel_value, Mapping) else {}
        if bool(servo.get("strict_unknown_keys", False)):
            allowed_servo_keys = {
                "strict_unknown_keys",
                "position_gain_per_s",
                "xy_gain_per_s",
                "yaw_gain_per_s",
                "max_linear_speed_mps",
                "max_angular_speed_radps",
                "max_linear_acceleration_mps2",
                "max_angular_acceleration_radps2",
                "filter_window",
                "jump_threshold_m",
                "yaw_jump_threshold_rad",
                "jump_reseed_frames",
                "stale_after_s",
                "observation_stale_after_s",
                "timeout_s",
                "safety_timeout_s",
                "max_correction_m",
                "start_yaw_limit_rad",
                "arrival_inner_m",
                "arrival_outer_m",
                "arrival_yaw_inner_rad",
                "arrival_yaw_outer_rad",
                "arrival_min_frames",
                "arrival_min_duration_s",
                "expected_axis_branch",
                "maximum_measurement_jump_m",
                "maximum_initial_correction_m",
                "arrival_inner_xy_m",
                "arrival_outer_xy_m",
                "maximum_linear_speed_mps",
                "maximum_angular_speed_radps",
                "kp_xy_per_s",
                "kp_yaw_per_s",
                "maximum_initial_yaw_deg",
                "arrival_inner_yaw_deg",
                "arrival_outer_yaw_deg",
                "wheel_stop",
                "linear_speed_mps",
                "angular_speed_radps",
                "wheel_linear_stop_mps",
                "wheel_angular_stop_radps",
                "duration_s",
                "wheel_stop_duration_s",
                "feedback_stale_after_s",
                "wheel_feedback_stale_after_s",
                "wheel_stop_linear_mps",
                "wheel_stop_angular_radps",
                "wheel_stop_dwell_s",
                "control_rate_hz",
                "absolute_linear_speed_limit_mps",
                "absolute_angular_speed_limit_radps",
            }
            unknown = sorted(set(servo) - allowed_servo_keys)
            if unknown:
                raise ValueError(
                    "unknown servo configuration key(s): " + ", ".join(unknown)
                )

        aliases: dict[str, tuple[str, ...]] = {
            "position_gain_per_s": ("position_gain_per_s", "xy_gain_per_s"),
            "yaw_gain_per_s": ("yaw_gain_per_s",),
            "max_linear_speed_mps": ("max_linear_speed_mps",),
            "max_angular_speed_radps": ("max_angular_speed_radps",),
            "max_linear_acceleration_mps2": ("max_linear_acceleration_mps2",),
            "max_angular_acceleration_radps2": ("max_angular_acceleration_radps2",),
            "filter_window": ("filter_window",),
            "jump_threshold_m": ("jump_threshold_m",),
            "yaw_jump_threshold_rad": ("yaw_jump_threshold_rad",),
            "jump_reseed_frames": ("jump_reseed_frames",),
            "stale_after_s": ("stale_after_s", "observation_stale_after_s"),
            "timeout_s": ("timeout_s", "safety_timeout_s"),
            "max_correction_m": ("max_correction_m",),
            "start_yaw_limit_rad": ("start_yaw_limit_rad",),
            "arrival_inner_m": ("arrival_inner_m",),
            "arrival_outer_m": ("arrival_outer_m",),
            "arrival_yaw_inner_rad": ("arrival_yaw_inner_rad",),
            "arrival_yaw_outer_rad": ("arrival_yaw_outer_rad",),
            "arrival_min_frames": ("arrival_min_frames",),
            "arrival_min_duration_s": ("arrival_min_duration_s",),
        }
        fields: dict[str, object] = {}
        for field_name, candidate_names in aliases.items():
            for candidate_name in candidate_names:
                if candidate_name in servo:
                    fields[field_name] = servo[candidate_name]
                    break
        axis_branch = pallet.get("axis_branch", servo.get("expected_axis_branch"))
        if axis_branch is not None:
            fields["expected_axis_branch"] = str(axis_branch)

        # Canonical shipped JSON uses descriptive ``maximum_*`` names and
        # degrees at its human-facing boundary.
        direct_aliases = {
            "jump_threshold_m": "maximum_measurement_jump_m",
            "max_correction_m": "maximum_initial_correction_m",
            "arrival_inner_m": "arrival_inner_xy_m",
            "arrival_outer_m": "arrival_outer_xy_m",
            "max_linear_speed_mps": "maximum_linear_speed_mps",
            "max_angular_speed_radps": "maximum_angular_speed_radps",
            "yaw_gain_per_s": "kp_yaw_per_s",
        }
        for field_name, config_name in direct_aliases.items():
            if field_name not in fields and config_name in servo:
                fields[field_name] = servo[config_name]
        if "position_gain_per_s" not in fields and "kp_xy_per_s" in servo:
            gains = np.asarray(servo["kp_xy_per_s"], dtype=np.float64).reshape(-1)
            if gains.size == 1:
                fields["position_gain_per_s"] = float(gains[0])
            elif gains.size == 2 and math.isclose(
                float(gains[0]),
                float(gains[1]),
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                fields["position_gain_per_s"] = float(gains[0])
            else:
                raise ValueError(
                    "kp_xy_per_s must be scalar or equal XY gains for this controller"
                )
        degree_aliases = {
            "start_yaw_limit_rad": "maximum_initial_yaw_deg",
            "arrival_yaw_inner_rad": "arrival_inner_yaw_deg",
            "arrival_yaw_outer_rad": "arrival_outer_yaw_deg",
        }
        for field_name, config_name in degree_aliases.items():
            if field_name not in fields and config_name in servo:
                fields[field_name] = math.radians(float(servo[config_name]))

        wheel_aliases = {
            "wheel_linear_stop_mps": (
                "linear_speed_mps",
                "wheel_linear_stop_mps",
            ),
            "wheel_angular_stop_radps": (
                "angular_speed_radps",
                "wheel_angular_stop_radps",
            ),
            "wheel_stop_duration_s": ("duration_s", "wheel_stop_duration_s"),
            "wheel_feedback_stale_after_s": (
                "feedback_stale_after_s",
                "wheel_feedback_stale_after_s",
            ),
        }
        for field_name, candidate_names in wheel_aliases.items():
            for candidate_name in candidate_names:
                if candidate_name in wheel:
                    fields[field_name] = wheel[candidate_name]
                    break
                if candidate_name in servo:
                    fields[field_name] = servo[candidate_name]
                    break
        shipped_wheel_aliases = {
            "wheel_linear_stop_mps": "wheel_stop_linear_mps",
            "wheel_angular_stop_radps": "wheel_stop_angular_radps",
            "wheel_stop_duration_s": "wheel_stop_dwell_s",
        }
        for field_name, config_name in shipped_wheel_aliases.items():
            if field_name not in fields and config_name in servo:
                fields[field_name] = servo[config_name]

        mobility_frame_value = value.get("mobility_frame", {})
        mobility_frame = (
            mobility_frame_value if isinstance(mobility_frame_value, Mapping) else {}
        )
        transform_value = pallet.get(
            "T_mobility_base",
            value.get(
                "T_mobility_base",
                mobility_frame.get(
                    "T_mobility_from_base",
                    servo.get("T_mobility_base"),
                ),
            ),
        )
        if transform_value is not None:
            if isinstance(transform_value, Mapping):
                translation = transform_value.get(
                    "translation_xy_m",
                    transform_value.get("translation_m", (0.0, 0.0)),
                )
                yaw = transform_value.get(
                    "yaw_target_source_rad",
                    transform_value.get("yaw_rad", 0.0),
                )
                fields["mobility_from_base"] = SE2FrameTransform(
                    translation_xy_m=_xy(translation, "T_mobility_base.translation"),
                    yaw_target_source_rad=float(yaw),
                )
            else:
                fields["mobility_from_base"] = SE2FrameTransform.from_matrix(
                    transform_value
                )
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class PalletServoMeasurement:
    """Minimal metric alignment observation consumed by the pure controller."""

    timestamp_s: float
    held_center_base: tuple[float, float] | None
    held_yaw_base_rad: float | None
    target_center_base: tuple[float, float] | None
    target_yaw_base_rad: float | None
    axis_branch: str | None = None
    valid: bool = True
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "timestamp_s", _finite(self.timestamp_s, "timestamp_s")
        )
        object.__setattr__(
            self,
            "rejection_reasons",
            tuple(str(reason) for reason in self.rejection_reasons),
        )
        if not self.valid:
            return
        branch = None if self.axis_branch is None else str(self.axis_branch).strip()
        object.__setattr__(self, "axis_branch", branch or None)
        if not branch:
            raise ValueError("valid measurement requires axis_branch")
        if self.held_center_base is None or self.target_center_base is None:
            raise ValueError("valid measurement requires held and target centres")
        if self.held_yaw_base_rad is None or self.target_yaw_base_rad is None:
            raise ValueError("valid measurement requires held and target yaw")
        object.__setattr__(
            self,
            "held_center_base",
            _xy(self.held_center_base, "held_center_base"),
        )
        object.__setattr__(
            self,
            "target_center_base",
            _xy(self.target_center_base, "target_center_base"),
        )
        object.__setattr__(
            self,
            "held_yaw_base_rad",
            normalize_line_angle_rad(
                _finite(self.held_yaw_base_rad, "held_yaw_base_rad")
            ),
        )
        object.__setattr__(
            self,
            "target_yaw_base_rad",
            normalize_line_angle_rad(
                _finite(self.target_yaw_base_rad, "target_yaw_base_rad")
            ),
        )

    @classmethod
    def invalid(
        cls,
        timestamp_s: float,
        *rejection_reasons: str,
    ) -> "PalletServoMeasurement":
        return cls(
            timestamp_s=timestamp_s,
            held_center_base=None,
            held_yaw_base_rad=None,
            target_center_base=None,
            target_yaw_base_rad=None,
            axis_branch=None,
            valid=False,
            rejection_reasons=tuple(rejection_reasons),
        )


@dataclass(frozen=True, slots=True)
class WorldFeatureToBodyReferenceMeasurement:
    """Fine-servo contract for a fixed world feature and body-frame reference.

    ``current_observed_feature_center_base`` is the current world feature
    expressed in the moving base/body frame.
    ``demonstrated_body_reference_center_base`` is the demonstrated feature
    reference in that same body frame.  The servo error is always current
    observed feature minus demonstrated body reference.
    """

    timestamp_s: float
    current_observed_feature_center_base: tuple[float, float] | None
    current_observed_feature_yaw_base_rad: float | None
    demonstrated_body_reference_center_base: tuple[float, float] | None
    demonstrated_body_reference_yaw_base_rad: float | None
    axis_branch: str | None
    reference_source: str
    valid: bool = True
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "timestamp_s", _finite(self.timestamp_s, "timestamp_s")
        )
        object.__setattr__(
            self,
            "rejection_reasons",
            tuple(str(reason) for reason in self.rejection_reasons),
        )
        source = str(self.reference_source).strip()
        object.__setattr__(self, "reference_source", source)
        if not self.valid:
            return
        if not source:
            raise ValueError("valid measurement requires reference_source")
        branch = None if self.axis_branch is None else str(self.axis_branch).strip()
        object.__setattr__(self, "axis_branch", branch or None)
        if not branch:
            raise ValueError("valid measurement requires axis_branch")
        if (
            self.current_observed_feature_center_base is None
            or self.demonstrated_body_reference_center_base is None
        ):
            raise ValueError(
                "valid measurement requires observed feature and body reference centres"
            )
        if (
            self.current_observed_feature_yaw_base_rad is None
            or self.demonstrated_body_reference_yaw_base_rad is None
        ):
            raise ValueError(
                "valid measurement requires observed feature and body reference yaw"
            )
        object.__setattr__(
            self,
            "current_observed_feature_center_base",
            _xy(
                self.current_observed_feature_center_base,
                "current_observed_feature_center_base",
            ),
        )
        object.__setattr__(
            self,
            "demonstrated_body_reference_center_base",
            _xy(
                self.demonstrated_body_reference_center_base,
                "demonstrated_body_reference_center_base",
            ),
        )
        object.__setattr__(
            self,
            "current_observed_feature_yaw_base_rad",
            normalize_line_angle_rad(
                _finite(
                    self.current_observed_feature_yaw_base_rad,
                    "current_observed_feature_yaw_base_rad",
                )
            ),
        )
        object.__setattr__(
            self,
            "demonstrated_body_reference_yaw_base_rad",
            normalize_line_angle_rad(
                _finite(
                    self.demonstrated_body_reference_yaw_base_rad,
                    "demonstrated_body_reference_yaw_base_rad",
                )
            ),
        )

    @classmethod
    def invalid(
        cls,
        timestamp_s: float,
        *rejection_reasons: str,
        reference_source: str = "invalid",
    ) -> "WorldFeatureToBodyReferenceMeasurement":
        return cls(
            timestamp_s=timestamp_s,
            current_observed_feature_center_base=None,
            current_observed_feature_yaw_base_rad=None,
            demonstrated_body_reference_center_base=None,
            demonstrated_body_reference_yaw_base_rad=None,
            axis_branch=None,
            reference_source=reference_source,
            valid=False,
            rejection_reasons=tuple(rejection_reasons),
        )


@runtime_checkable
class SlotAlignmentGeometryLike(Protocol):
    """Geometry fields shared with ``pallet_models.SlotAlignmentObservation``."""

    held_center_base: object
    held_yaw_base_rad: float
    target_center_base: object
    target_yaw_base_rad: float


@runtime_checkable
class SlotAlignmentLike(SlotAlignmentGeometryLike, Protocol):
    """Timestamped alignment contract used directly by the servo."""

    timestamp_s: float
    axis_branch: str


@runtime_checkable
class WorldFeatureToBodyReferenceLike(Protocol):
    """Explicit fine-servo feature/reference alignment contract."""

    timestamp_s: float
    current_observed_feature_center_base: object
    current_observed_feature_yaw_base_rad: float
    demonstrated_body_reference_center_base: object
    demonstrated_body_reference_yaw_base_rad: float
    axis_branch: str
    reference_source: str


@dataclass(frozen=True, slots=True)
class WheelMotionMeasurement:
    """Fresh measured planar wheel/base motion, not a commanded velocity."""

    timestamp_s: float
    linear_speed_mps: float
    angular_speed_radps: float
    valid: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "timestamp_s", _finite(self.timestamp_s, "timestamp_s")
        )
        object.__setattr__(
            self,
            "linear_speed_mps",
            abs(_finite(self.linear_speed_mps, "linear_speed_mps")),
        )
        object.__setattr__(
            self,
            "angular_speed_radps",
            abs(_finite(self.angular_speed_radps, "angular_speed_radps")),
        )


@dataclass(frozen=True, slots=True)
class PalletServoOutput:
    """One deterministic control decision for the combined stream owner."""

    command: VelocityCommand
    state: PalletServoState
    arrived: bool
    hold_body: bool
    measurement_accepted: bool
    reason: str
    diagnostics: Mapping[str, object]

    @property
    def vx_mps(self) -> float:
        return self.command.vx_mps

    @property
    def vy_mps(self) -> float:
        return self.command.vy_mps

    @property
    def wz_radps(self) -> float:
        return self.command.wz_radps


@dataclass(frozen=True, slots=True)
class _Sample:
    timestamp_s: float
    current_observed_xy: FloatArray
    body_reference_xy: FloatArray
    compensation_point_xy: FloatArray
    yaw_error_rad: float
    axis_branch: str
    measurement_mode: str
    reference_source: str
    target_xy: FloatArray | None = None
    held_xy: FloatArray | None = None

    @property
    def error_xy(self) -> FloatArray:
        return self.current_observed_xy - self.body_reference_xy


class PalletSlot1Servo:
    """Deterministic slot-1 controller with explicit zero-mobility holds."""

    def __init__(self, config: PalletServoConfig | None = None) -> None:
        self.config = config or PalletServoConfig()
        self.state = PalletServoState.IDLE
        self._samples: deque[_Sample] = deque(maxlen=self.config.filter_window)
        self._jump_candidates: deque[_Sample] = deque(
            maxlen=self.config.jump_reseed_frames
        )
        self._last_raw: _Sample | None = None
        self._last_command = ZERO_VELOCITY
        self._last_update_s: float | None = None
        self._started_at_s: float | None = None
        self._perception_lost_at_s: float | None = None
        self._arrival_started_at_s: float | None = None
        self._arrival_frames = 0
        self._zero_latched_at_s: float | None = None
        self._wheel_stop_started_at_s: float | None = None
        self._last_wheel_timestamp_s: float | None = None
        self._motion_started = False
        self._hold_reason = ""
        self._locked_axis_branch = self.config.expected_axis_branch

    def start(self, now_s: float) -> PalletServoOutput:
        """Explicitly arm the pure controller; this method cannot move a robot."""

        now = _finite(now_s, "now_s")
        self._samples.clear()
        self._jump_candidates.clear()
        self._last_raw = None
        self._last_command = ZERO_VELOCITY
        self._last_update_s = now
        self._started_at_s = now
        self._perception_lost_at_s = None
        self._clear_arrival()
        self._clear_wheel_stop()
        self._motion_started = False
        self._hold_reason = ""
        self._locked_axis_branch = self.config.expected_axis_branch
        self.state = PalletServoState.ACQUIRING
        return self._output(False, "started", now)

    def fault(self, reason: str, now_s: float) -> PalletServoOutput:
        """Enter a persistent fault hold; only a new ``start`` clears it."""

        now = self._check_time(now_s)
        self.state = PalletServoState.FAULT_HOLD
        self._hold_reason = str(reason).strip() or "fault"
        self._force_zero(now)
        self._clear_arrival()
        self._clear_wheel_stop()
        return self._output(False, self._hold_reason, now)

    def request_shutdown_hold(self, now_s: float) -> PalletServoOutput:
        """Latch zero mobility while a successor handoff is pending."""

        now = self._check_time(now_s)
        self.state = PalletServoState.SHUTDOWN_PENDING_HOLD
        self._hold_reason = "shutdown_handoff_pending"
        self._force_zero(now)
        self._clear_arrival()
        self._clear_wheel_stop()
        return self._output(False, self._hold_reason, now)

    def update(
        self,
        observation: (
            SlotAlignmentLike
            | SlotAlignmentGeometryLike
            | PalletServoMeasurement
            | WorldFeatureToBodyReferenceLike
            | WorldFeatureToBodyReferenceMeasurement
            | None
        ),
        now_s: float,
        wheel_motion: WheelMotionMeasurement | None = None,
        *,
        observation_timestamp_s: float | None = None,
    ) -> PalletServoOutput:
        """Advance perception servo and measured wheel-stop confirmation.

        ``wheel_motion`` is ignored before the zero-latched
        ``ARRIVAL_WHEEL_STOP`` state.  Its timestamp must be newer than the
        zero-latch cycle, preventing a pre-command sample from proving stop.
        ``observation_timestamp_s`` lets the timestamp-free geometry contract
        in :mod:`pallet_models` inherit its enclosing stack-frame timestamp.
        """

        rollback_output = self._fault_on_time_rollback(now_s)
        if rollback_output is not None:
            return rollback_output
        now = self._check_time(now_s)
        if self.state is PalletServoState.IDLE:
            self._force_zero(now)
            return self._output(False, "inactive", now)
        if self.state in {
            PalletServoState.ARRIVED_HOLD,
            PalletServoState.FAULT_HOLD,
            PalletServoState.SHUTDOWN_PENDING_HOLD,
        }:
            self._force_zero(now)
            return self._output(
                False, self._hold_reason or self.state.value.lower(), now
            )

        assert self._started_at_s is not None
        branch_fault = self._axis_branch_fault(observation)
        if branch_fault is not None:
            return self.fault(branch_fault, now)
        sample, invalid_reason = self._coerce_observation(
            observation,
            now,
            timestamp_override_s=observation_timestamp_s,
        )
        accepted = False
        accept_reason = invalid_reason
        if sample is not None:
            accepted, accept_reason = self._accept_or_reseed(sample)

        if self.state is PalletServoState.ARRIVAL_WHEEL_STOP:
            return self._update_wheel_stop(
                sample if accepted else None,
                wheel_motion,
                now,
                observation_reason=accept_reason,
            )

        if not accepted or sample is None:
            return self._perception_hold(accept_reason or "observation_invalid", now)

        # A perception hold may persist indefinitely with zero mobility.  It
        # cannot resume motion after the active-servo deadline; fresh vision at
        # that point transitions to the persistent fault hold instead.
        if now - self._started_at_s >= self.config.timeout_s:
            return self.fault("servo_timeout", now)

        self._perception_lost_at_s = None
        filtered = self._filtered_sample()
        if filtered is None or len(self._samples) < self.config.filter_window:
            self.state = PalletServoState.ACQUIRING
            self._force_zero(now)
            return self._output(
                True,
                "acquiring_stable_alignment",
                now,
                raw=sample,
                filtered=filtered,
            )

        raw_distance = float(np.linalg.norm(sample.error_xy))
        filtered_distance = float(np.linalg.norm(filtered.error_xy))
        max_distance = max(raw_distance, filtered_distance)
        if max_distance > self.config.max_correction_m:
            return self.fault("correction_limit_exceeded", now)
        if (
            not self._motion_started
            and max(abs(sample.yaw_error_rad), abs(filtered.yaw_error_rad))
            > self.config.start_yaw_limit_rad
        ):
            return self.fault("start_yaw_limit_exceeded", now)

        inside_inner = self._inside_arrival(sample, filtered, inner=True)
        inside_outer = self._inside_arrival(sample, filtered, inner=False)
        evidence_active = self._arrival_started_at_s is not None
        if evidence_active:
            if inside_outer:
                self._arrival_frames += 1
            else:
                self._clear_arrival()
                evidence_active = False
        elif inside_inner:
            self._arrival_started_at_s = now
            self._arrival_frames = 1
            evidence_active = True

        if evidence_active:
            assert self._arrival_started_at_s is not None
            elapsed = now - self._arrival_started_at_s
            if (
                self._arrival_frames >= self.config.arrival_min_frames
                and elapsed >= self.config.arrival_min_duration_s
            ):
                self.state = PalletServoState.ARRIVAL_WHEEL_STOP
                self._zero_latched_at_s = now
                self._clear_wheel_stop(keep_zero_latch=True)
                self._force_zero(now)
                return self._output(
                    True,
                    "arrival_candidate_zero_latched",
                    now,
                    raw=sample,
                    filtered=filtered,
                )
            self.state = PalletServoState.ARRIVAL_EVIDENCE
        else:
            self.state = PalletServoState.TRACKING

        command = self._tracking_command(filtered, now)
        self._last_update_s = now
        if not command.is_zero:
            self._motion_started = True
        return self._output(
            True,
            "arrival_evidence" if evidence_active else "tracking",
            now,
            raw=sample,
            filtered=filtered,
            command=command,
        )

    def update_wheel_state(
        self,
        wheel_motion: WheelMotionMeasurement,
        now_s: float,
    ) -> PalletServoOutput:
        """Advance wheel-stop dwell between camera updates.

        The most recent accepted camera observation is reused only while it is
        still fresh.  Outside ``ARRIVAL_WHEEL_STOP`` this method is a no-op
        decision and never changes the commanded twist.
        """

        rollback_output = self._fault_on_time_rollback(now_s)
        if rollback_output is not None:
            return rollback_output
        now = self._check_time(now_s)
        if self.state is not PalletServoState.ARRIVAL_WHEEL_STOP:
            return self._output(
                False,
                "wheel_feedback_not_required",
                now,
            )
        sample = self._last_raw
        if sample is None or now - sample.timestamp_s > self.config.stale_after_s:
            sample = None
            reason = "observation_stale_during_wheel_stop"
        else:
            reason = None
        return self._update_wheel_stop(
            sample,
            wheel_motion,
            now,
            observation_reason=reason,
        )

    def _coerce_observation(
        self,
        observation: (
            SlotAlignmentLike
            | SlotAlignmentGeometryLike
            | PalletServoMeasurement
            | WorldFeatureToBodyReferenceLike
            | WorldFeatureToBodyReferenceMeasurement
            | None
        ),
        now_s: float,
        *,
        timestamp_override_s: float | None = None,
    ) -> tuple[_Sample | None, str | None]:
        if observation is None:
            return None, "observation_missing"
        if not bool(getattr(observation, "valid", True)):
            reasons = tuple(getattr(observation, "rejection_reasons", ()))
            return None, str(reasons[0]) if reasons else "observation_invalid"
        try:
            timestamp_source = (
                timestamp_override_s
                if timestamp_override_s is not None
                else getattr(observation, "timestamp_s")
            )
            timestamp = _finite(timestamp_source, "timestamp_s")
            age = now_s - timestamp
            if age < -1e-6:
                return None, "observation_timestamp_in_future"
            if age > self.config.stale_after_s:
                return None, "observation_stale"
            axis_branch = str(getattr(observation, "axis_branch")).strip()
            if not axis_branch:
                return None, "axis_branch_missing"
            if (
                self._last_raw is not None
                and timestamp < self._last_raw.timestamp_s - 1e-9
            ):
                return None, "observation_timestamp_regressed"
            if hasattr(observation, "current_observed_feature_center_base") or hasattr(
                observation, "observed_feature_center_base"
            ):
                current_center = getattr(
                    observation,
                    "current_observed_feature_center_base",
                    getattr(observation, "observed_feature_center_base", None),
                )
                reference_center = getattr(
                    observation,
                    "demonstrated_body_reference_center_base",
                    getattr(observation, "body_reference_center_base", None),
                )
                current_yaw_value = getattr(
                    observation,
                    "current_observed_feature_yaw_base_rad",
                    getattr(observation, "observed_feature_yaw_base_rad", None),
                )
                reference_yaw_value = getattr(
                    observation,
                    "demonstrated_body_reference_yaw_base_rad",
                    getattr(observation, "body_reference_yaw_base_rad", None),
                )
                current_xy = np.asarray(
                    _xy(
                        current_center,
                        "current_observed_feature_center_base",
                    ),
                    dtype=np.float64,
                )
                reference_xy = np.asarray(
                    _xy(
                        reference_center,
                        "demonstrated_body_reference_center_base",
                    ),
                    dtype=np.float64,
                )
                current_yaw = _finite(
                    current_yaw_value,
                    "current_observed_feature_yaw_base_rad",
                )
                reference_yaw = _finite(
                    reference_yaw_value,
                    "demonstrated_body_reference_yaw_base_rad",
                )
                reference_source = str(
                    getattr(observation, "reference_source")
                ).strip()
                if not reference_source:
                    return None, "reference_source_missing"
                yaw_error = line_angle_difference_rad(current_yaw, reference_yaw)
                return (
                    _Sample(
                        timestamp,
                        current_xy,
                        reference_xy,
                        current_xy,
                        yaw_error,
                        axis_branch,
                        "feature_to_body_reference",
                        reference_source,
                    ),
                    None,
                )
            target_xy = np.asarray(
                _xy(getattr(observation, "target_center_base"), "target_center_base"),
                dtype=np.float64,
            )
            held_xy = np.asarray(
                _xy(getattr(observation, "held_center_base"), "held_center_base"),
                dtype=np.float64,
            )
            target_yaw = _finite(
                getattr(observation, "target_yaw_base_rad"),
                "target_yaw_base_rad",
            )
            held_yaw = _finite(
                getattr(observation, "held_yaw_base_rad"),
                "held_yaw_base_rad",
            )
        except (AttributeError, TypeError, ValueError) as exc:
            return None, f"observation_corrupt:{exc}"
        yaw_error = line_angle_difference_rad(target_yaw, held_yaw)
        return (
            _Sample(
                timestamp,
                target_xy,
                held_xy,
                target_xy,
                yaw_error,
                axis_branch,
                "held_box_alignment",
                "held_box_measurement",
                target_xy=target_xy,
                held_xy=held_xy,
            ),
            None,
        )

    def _axis_branch_fault(
        self,
        observation: SlotAlignmentLike
        | SlotAlignmentGeometryLike
        | PalletServoMeasurement
        | WorldFeatureToBodyReferenceLike
        | WorldFeatureToBodyReferenceMeasurement
        | None,
    ) -> str | None:
        if observation is None or not bool(getattr(observation, "valid", True)):
            return None
        branch = str(getattr(observation, "axis_branch", "")).strip()
        if not branch:
            return "axis_branch_missing"
        if branch != self._locked_axis_branch:
            return "axis_branch_mismatch"
        return None

    def _accept_or_reseed(self, sample: _Sample) -> tuple[bool, str]:
        if self._last_raw is None:
            self._samples.append(sample)
            self._last_raw = sample
            self._jump_candidates.clear()
            return True, "accepted"
        if sample.measurement_mode != self._last_raw.measurement_mode:
            self._jump_candidates.clear()
            return False, "measurement_mode_changed_requires_restart"
        if sample.reference_source != self._last_raw.reference_source:
            self._jump_candidates.clear()
            return False, "reference_source_changed_requires_restart"
        if self._within_jump(sample, self._last_raw):
            self._samples.append(sample)
            self._last_raw = sample
            self._jump_candidates.clear()
            return True, "accepted"

        if self._jump_candidates:
            candidate = self._filtered_sample(self._jump_candidates)
            assert candidate is not None
            if not self._within_jump(sample, candidate):
                self._jump_candidates.clear()
        self._jump_candidates.append(sample)
        if len(self._jump_candidates) < self.config.jump_reseed_frames:
            return False, "alignment_jump_rejected"
        self._samples.clear()
        self._samples.extend(self._jump_candidates)
        self._last_raw = sample
        self._jump_candidates.clear()
        self._clear_arrival()
        return True, "alignment_jump_reseeded"

    def _within_jump(self, first: _Sample, second: _Sample) -> bool:
        position_jump = float(np.linalg.norm(first.error_xy - second.error_xy))
        yaw_jump = abs(
            line_angle_difference_rad(first.yaw_error_rad, second.yaw_error_rad)
        )
        return (
            position_jump <= self.config.jump_threshold_m
            and yaw_jump <= self.config.yaw_jump_threshold_rad
        )

    def _filtered_sample(
        self,
        samples: deque[_Sample] | None = None,
    ) -> _Sample | None:
        selected = self._samples if samples is None else samples
        if not selected:
            return None
        values = tuple(selected)
        current_observed = np.stack(
            tuple(sample.current_observed_xy for sample in values),
            axis=0,
        )
        body_reference = np.stack(
            tuple(sample.body_reference_xy for sample in values),
            axis=0,
        )
        compensation_points = np.stack(
            tuple(sample.compensation_point_xy for sample in values),
            axis=0,
        )
        yaw = self._line_medoid(
            np.asarray(tuple(sample.yaw_error_rad for sample in values))
        )
        target_xy = None
        held_xy = None
        if all(sample.target_xy is not None for sample in values):
            target_values = tuple(
                sample.target_xy
                for sample in values
                if sample.target_xy is not None
            )
            target_xy = np.median(
                np.stack(target_values, axis=0),
                axis=0,
            )
        if all(sample.held_xy is not None for sample in values):
            held_values = tuple(
                sample.held_xy for sample in values if sample.held_xy is not None
            )
            held_xy = np.median(
                np.stack(held_values, axis=0),
                axis=0,
            )
        return _Sample(
            timestamp_s=max(sample.timestamp_s for sample in values),
            current_observed_xy=np.median(current_observed, axis=0),
            body_reference_xy=np.median(body_reference, axis=0),
            compensation_point_xy=np.median(compensation_points, axis=0),
            yaw_error_rad=_signed_line_angle(yaw),
            axis_branch=values[-1].axis_branch,
            measurement_mode=values[-1].measurement_mode,
            reference_source=values[-1].reference_source,
            target_xy=target_xy,
            held_xy=held_xy,
        )

    @staticmethod
    def _line_medoid(angles_rad: FloatArray) -> float:
        normalized = np.mod(np.asarray(angles_rad, dtype=np.float64), math.pi)
        pairwise = np.abs(
            (normalized[:, None] - normalized[None, :] + math.pi / 2.0) % math.pi
            - math.pi / 2.0
        )
        return float(normalized[int(np.argmin(np.sum(pairwise, axis=1)))])

    def _inside_arrival(
        self,
        raw: _Sample,
        filtered: _Sample,
        *,
        inner: bool,
    ) -> bool:
        distance_limit = (
            self.config.arrival_inner_m if inner else self.config.arrival_outer_m
        )
        yaw_limit = (
            self.config.arrival_yaw_inner_rad
            if inner
            else self.config.arrival_yaw_outer_rad
        )
        return (
            float(np.linalg.norm(raw.error_xy)) <= distance_limit
            and float(np.linalg.norm(filtered.error_xy)) <= distance_limit
            and abs(raw.yaw_error_rad) <= yaw_limit
            and abs(filtered.yaw_error_rad) <= yaw_limit
        )

    def _tracking_command(self, filtered: _Sample, now_s: float) -> VelocityCommand:
        error_x, error_y = (float(value) for value in filtered.error_xy)
        point_x, point_y = (float(value) for value in filtered.compensation_point_xy)
        wz_base = self.config.yaw_gain_per_s * filtered.yaw_error_rad
        # J_b(p_current)^-1 e: vx = k e_x + y_current*wz,
        #                            vy = k e_y - x_current*wz.
        base = VelocityCommand(
            self.config.position_gain_per_s * error_x + point_y * wz_base,
            self.config.position_gain_per_s * error_y - point_x * wz_base,
            wz_base,
        )
        mobility = self.config.mobility_from_base.transform_twist(base)
        desired = self._coupled_speed_limit(mobility)
        return self._slew_towards(desired, now_s)

    def _coupled_speed_limit(self, command: VelocityCommand) -> VelocityCommand:
        scale = 1.0
        if command.linear_norm_mps > self.config.max_linear_speed_mps:
            scale = min(
                scale,
                self.config.max_linear_speed_mps / command.linear_norm_mps,
            )
        if abs(command.wz_radps) > self.config.max_angular_speed_radps:
            scale = min(
                scale,
                self.config.max_angular_speed_radps / abs(command.wz_radps),
            )
        return VelocityCommand(
            command.vx_mps * scale,
            command.vy_mps * scale,
            command.wz_radps * scale,
        )

    def _slew_towards(
        self,
        desired: VelocityCommand,
        now_s: float,
    ) -> VelocityCommand:
        assert self._last_update_s is not None
        dt = max(0.0, now_s - self._last_update_s)
        previous_linear = np.asarray(
            (self._last_command.vx_mps, self._last_command.vy_mps),
            dtype=np.float64,
        )
        desired_linear = np.asarray(
            (desired.vx_mps, desired.vy_mps),
            dtype=np.float64,
        )
        linear_delta = desired_linear - previous_linear
        angular_delta = desired.wz_radps - self._last_command.wz_radps
        linear_allowance = self.config.max_linear_acceleration_mps2 * dt
        angular_allowance = self.config.max_angular_acceleration_radps2 * dt
        scale = 1.0
        linear_delta_norm = float(np.linalg.norm(linear_delta))
        if linear_delta_norm > linear_allowance:
            scale = min(
                scale,
                0.0
                if linear_allowance <= 0.0
                else linear_allowance / linear_delta_norm,
            )
        if abs(angular_delta) > angular_allowance:
            scale = min(
                scale,
                0.0
                if angular_allowance <= 0.0
                else angular_allowance / abs(angular_delta),
            )
        linear = previous_linear + scale * linear_delta
        angular = self._last_command.wz_radps + scale * angular_delta
        command = self._coupled_speed_limit(
            VelocityCommand(float(linear[0]), float(linear[1]), float(angular))
        )
        self._last_command = command
        return command

    def _perception_hold(self, reason: str, now_s: float) -> PalletServoOutput:
        if self._perception_lost_at_s is None:
            self._perception_lost_at_s = now_s
        self.state = PalletServoState.PERCEPTION_HOLD
        self._force_zero(now_s)
        self._clear_arrival()
        return self._output(False, reason, now_s)

    def _update_wheel_stop(
        self,
        sample: _Sample | None,
        wheel_motion: WheelMotionMeasurement | None,
        now_s: float,
        *,
        observation_reason: str | None,
    ) -> PalletServoOutput:
        self._force_zero(now_s)
        filtered = self._filtered_sample()
        pose_in_hysteresis = (
            sample is not None
            and filtered is not None
            and self._inside_arrival(sample, filtered, inner=False)
        )
        wheel_reason, stopped = self._wheel_stop_status(wheel_motion, now_s)
        if not stopped:
            return self._output(
                sample is not None,
                wheel_reason
                if pose_in_hysteresis
                else observation_reason or "arrival_pose_outside_hysteresis",
                now_s,
                raw=sample,
                filtered=filtered,
            )
        if not pose_in_hysteresis:
            self.state = PalletServoState.PERCEPTION_HOLD
            self._samples.clear()
            self._jump_candidates.clear()
            self._last_raw = None
            self._clear_arrival()
            self._clear_wheel_stop()
            self._perception_lost_at_s = now_s
            return self._output(
                False,
                "arrival_cancelled_after_wheel_stop",
                now_s,
            )
        self.state = PalletServoState.ARRIVED_HOLD
        self._hold_reason = "arrived_wheels_stopped"
        self._clear_arrival()
        return self._output(
            True,
            self._hold_reason,
            now_s,
            raw=sample,
            filtered=filtered,
        )

    def _wheel_stop_status(
        self,
        measurement: WheelMotionMeasurement | None,
        now_s: float,
    ) -> tuple[str, bool]:
        if measurement is None:
            self._wheel_stop_started_at_s = None
            return "wheel_feedback_missing", False
        if not measurement.valid:
            self._wheel_stop_started_at_s = None
            return "wheel_feedback_invalid", False
        age = now_s - measurement.timestamp_s
        if age < -1e-6:
            self._wheel_stop_started_at_s = None
            return "wheel_feedback_in_future", False
        if age > self.config.wheel_feedback_stale_after_s:
            self._wheel_stop_started_at_s = None
            return "wheel_feedback_stale", False
        assert self._zero_latched_at_s is not None
        if measurement.timestamp_s <= self._zero_latched_at_s + 1e-9:
            return "wheel_feedback_predates_zero", False
        if (
            self._last_wheel_timestamp_s is not None
            and measurement.timestamp_s <= self._last_wheel_timestamp_s + 1e-9
        ):
            return "wheel_feedback_not_new", False
        gap = (
            None
            if self._last_wheel_timestamp_s is None
            else measurement.timestamp_s - self._last_wheel_timestamp_s
        )
        self._last_wheel_timestamp_s = measurement.timestamp_s
        below_threshold = (
            measurement.linear_speed_mps < self.config.wheel_linear_stop_mps
            and measurement.angular_speed_radps < self.config.wheel_angular_stop_radps
        )
        if not below_threshold:
            self._wheel_stop_started_at_s = None
            return "wheels_moving", False
        if gap is not None and gap > self.config.wheel_feedback_stale_after_s:
            self._wheel_stop_started_at_s = measurement.timestamp_s
            return "wheel_stop_dwell_restarted_after_gap", False
        if self._wheel_stop_started_at_s is None:
            self._wheel_stop_started_at_s = measurement.timestamp_s
            return "wheel_stop_dwell_started", False
        dwell = measurement.timestamp_s - self._wheel_stop_started_at_s
        if dwell < self.config.wheel_stop_duration_s:
            return "wheel_stop_dwell", False
        return "wheel_stop_confirmed", True

    def _clear_arrival(self) -> None:
        self._arrival_started_at_s = None
        self._arrival_frames = 0
        self._zero_latched_at_s = None

    def _clear_wheel_stop(self, *, keep_zero_latch: bool = False) -> None:
        self._wheel_stop_started_at_s = None
        self._last_wheel_timestamp_s = None
        if not keep_zero_latch:
            self._zero_latched_at_s = None

    def _force_zero(self, now_s: float) -> None:
        self._last_command = ZERO_VELOCITY
        self._last_update_s = now_s

    def _fault_on_time_rollback(self, now_s: float) -> PalletServoOutput | None:
        now = _finite(now_s, "now_s")
        if self._last_update_s is None or now >= self._last_update_s - 1e-12:
            return None
        self.state = PalletServoState.FAULT_HOLD
        self._hold_reason = "clock_rollback_detected"
        self._last_command = ZERO_VELOCITY
        self._clear_arrival()
        self._clear_wheel_stop()
        return self._output(False, self._hold_reason, now)

    def _check_time(self, now_s: float) -> float:
        now = _finite(now_s, "now_s")
        if self._last_update_s is not None and now < self._last_update_s - 1e-12:
            previous = float(self._last_update_s)
            self.state = PalletServoState.FAULT_HOLD
            self._hold_reason = "clock_rollback_detected"
            self._last_command = ZERO_VELOCITY
            self._clear_arrival()
            self._clear_wheel_stop()
            return previous
        return now

    def _output(
        self,
        accepted: bool,
        reason: str,
        now_s: float,
        *,
        raw: _Sample | None = None,
        filtered: _Sample | None = None,
        command: VelocityCommand | None = None,
    ) -> PalletServoOutput:
        selected = self._last_command if command is None else command
        diagnostics: dict[str, object] = {
            "timestamp_s": now_s,
            "reason": reason,
            "raw_error_xy_m": None,
            "filtered_error_xy_m": None,
            "raw_yaw_error_rad": None,
            "filtered_yaw_error_rad": None,
            "raw_planar_error_m": None,
            "filtered_planar_error_m": None,
            "measurement_mode": None,
            "reference_source": None,
            "raw_current_observed_feature_center_base_xy_m": None,
            "raw_demonstrated_body_reference_center_base_xy_m": None,
            "raw_compensation_point_base_xy_m": None,
            "filtered_current_observed_feature_center_base_xy_m": None,
            "filtered_demonstrated_body_reference_center_base_xy_m": None,
            "filtered_compensation_point_base_xy_m": None,
            "arrival_frames": self._arrival_frames,
            "arrival_duration_s": (
                0.0
                if self._arrival_started_at_s is None
                else max(0.0, now_s - self._arrival_started_at_s)
            ),
            "wheel_stop_duration_s": (
                0.0
                if self._wheel_stop_started_at_s is None
                else max(
                    0.0,
                    (self._last_wheel_timestamp_s or now_s)
                    - self._wheel_stop_started_at_s,
                )
            ),
            "zero_latched": self._zero_latched_at_s is not None,
            "motion_started": self._motion_started,
            "mobility_from_base": tuple(
                tuple(float(value) for value in row)
                for row in self.config.mobility_from_base.as_matrix()
            ),
        }
        if raw is not None:
            diagnostics.update(
                raw_error_xy_m=tuple(float(value) for value in raw.error_xy),
                raw_yaw_error_rad=float(raw.yaw_error_rad),
                raw_planar_error_m=float(np.linalg.norm(raw.error_xy)),
                measurement_mode=raw.measurement_mode,
                reference_source=raw.reference_source,
                raw_current_observed_feature_center_base_xy_m=tuple(
                    float(value) for value in raw.current_observed_xy
                ),
                raw_demonstrated_body_reference_center_base_xy_m=tuple(
                    float(value) for value in raw.body_reference_xy
                ),
                raw_compensation_point_base_xy_m=tuple(
                    float(value) for value in raw.compensation_point_xy
                ),
            )
            if raw.target_xy is not None:
                diagnostics["target_center_base_xy_m"] = tuple(
                    float(value) for value in raw.target_xy
                )
            if raw.held_xy is not None:
                diagnostics["held_center_base_xy_m"] = tuple(
                    float(value) for value in raw.held_xy
                )
        if filtered is not None:
            diagnostics.update(
                filtered_error_xy_m=tuple(float(value) for value in filtered.error_xy),
                filtered_yaw_error_rad=float(filtered.yaw_error_rad),
                filtered_planar_error_m=float(np.linalg.norm(filtered.error_xy)),
                filtered_current_observed_feature_center_base_xy_m=tuple(
                    float(value) for value in filtered.current_observed_xy
                ),
                filtered_demonstrated_body_reference_center_base_xy_m=tuple(
                    float(value) for value in filtered.body_reference_xy
                ),
                filtered_compensation_point_base_xy_m=tuple(
                    float(value) for value in filtered.compensation_point_xy
                ),
            )
            if diagnostics["measurement_mode"] is None:
                diagnostics["measurement_mode"] = filtered.measurement_mode
            if diagnostics["reference_source"] is None:
                diagnostics["reference_source"] = filtered.reference_source
        return PalletServoOutput(
            command=selected,
            state=self.state,
            arrived=self.state is PalletServoState.ARRIVED_HOLD,
            hold_body=self.state is not PalletServoState.IDLE,
            measurement_accepted=accepted,
            reason=reason,
            diagnostics=diagnostics,
        )


# Explicit alias for callers that prefer the slot-scoped configuration name.
PalletSlot1ServoConfig = PalletServoConfig
SlotAlignmentMeasurement = PalletServoMeasurement
SlotAlignmentObservation = PalletServoMeasurement
FeatureReferenceMeasurement = WorldFeatureToBodyReferenceMeasurement


__all__ = [
    "FeatureReferenceMeasurement",
    "PalletServoConfig",
    "PalletServoMeasurement",
    "PalletServoOutput",
    "PalletServoState",
    "PalletSlot1Servo",
    "PalletSlot1ServoConfig",
    "SE2FrameTransform",
    "SlotAlignmentMeasurement",
    "SlotAlignmentObservation",
    "SlotAlignmentGeometryLike",
    "SlotAlignmentLike",
    "WheelMotionMeasurement",
    "WorldFeatureToBodyReferenceLike",
    "WorldFeatureToBodyReferenceMeasurement",
]
