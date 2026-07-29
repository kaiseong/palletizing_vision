"""Pure fail-closed slot-1 placement sequencer.

The module has no robot-SDK side effects.  It begins only after fine mobile
alignment has latched ``ARRIVED_HOLD`` and every output requires exact-zero
mobility.  Spreading the hands requires a Running stream acknowledgement,
measured vertical descent, and fresh bounded vision geometry. Placement does
not read force/torque feedback.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping
import uuid

import numpy as np


ZERO_MOBILITY_COMMAND: tuple[float, float, float] = (0.0, 0.0, 0.0)
LOADED_HOLD_MODE = "CARTESIAN_LOADED_HOLD"
LOWERING_MODE = "CARTESIAN_PLACEMENT_LOWERING"
RELEASE_MODE = "CARTESIAN_PLACEMENT_RELEASE"


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


def _nonnegative(value: float, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _readonly_transform(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-6):
        raise ValueError(f"{name} must have homogeneous bottom row [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=2e-4):
        raise ValueError(f"{name} rotation must be orthonormal")
    result = np.array(matrix, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _optional_transform(value: Any | None, name: str) -> np.ndarray | None:
    return None if value is None else _readonly_transform(value, name)


def _rotation_error_rad(left: np.ndarray, right: np.ndarray) -> float:
    relative = left[:3, :3].T @ right[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(cosine)


class PlacementState(str, Enum):
    PRE_PLACE_VERIFY = "PRE_PLACE_VERIFY"
    LOWERING = "LOWERING"
    SEATED = "SEATED"
    RELEASING = "RELEASING"
    RELEASED = "RELEASED"
    FAULT_HOLD = "FAULT_HOLD"


class PlacementRequest(str, Enum):
    HOLD_CURRENT = "HOLD_CURRENT"
    LOWER_CARTESIAN_PLANNED = "LOWER_CARTESIAN_PLANNED"
    SPREAD_RELEASE = "SPREAD_RELEASE"


@dataclass(frozen=True, slots=True)
class PlacementConfig:
    pre_motion_clearance_floor_m: float = 0.050
    maximum_descent_m: float = 0.250
    pre_place_verify_dwell_s: float = 0.15
    lowering_timeout_s: float = 4.0
    seated_dwell_s: float = 0.35
    release_timeout_s: float = 4.5
    release_target_dwell_s: float = 0.35
    feedback_stale_s: float = 0.25
    lower_z_tolerance_m: float = 0.008
    lower_midpoint_xy_drift_m: float = 0.015
    lower_rotation_tolerance_rad: float = math.radians(3.0)
    release_target_translation_tolerance_m: float = 0.012
    release_target_rotation_tolerance_rad: float = math.radians(4.0)
    release_spread_m: float = 0.080
    vision_seating_max_uncertainty_m: float = 0.015
    vision_evidence_fresh_after_s: float = 0.30
    vision_plan_valid_for_s: float = 5.0
    vision_gap_stability_tolerance_m: float = 0.008
    vision_evidence_min_samples: int = 3

    def __post_init__(self) -> None:
        for name in (
            "pre_motion_clearance_floor_m",
            "maximum_descent_m",
            "lowering_timeout_s",
            "release_timeout_s",
            "feedback_stale_s",
            "lower_z_tolerance_m",
            "lower_midpoint_xy_drift_m",
            "lower_rotation_tolerance_rad",
            "release_target_translation_tolerance_m",
            "release_target_rotation_tolerance_rad",
            "release_spread_m",
            "vision_seating_max_uncertainty_m",
            "vision_evidence_fresh_after_s",
            "vision_plan_valid_for_s",
            "vision_gap_stability_tolerance_m",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        for name in (
            "pre_place_verify_dwell_s",
            "seated_dwell_s",
            "release_target_dwell_s",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        if self.pre_motion_clearance_floor_m < 0.050 - 1e-12:
            raise ValueError("pre-motion clearance floor cannot be below 50 mm")
        if self.maximum_descent_m > 0.300 + 1e-12:
            raise ValueError("maximum planned descent cannot exceed 300 mm")
        if self.vision_seating_max_uncertainty_m > 0.030 + 1e-12:
            raise ValueError("descent uncertainty limit cannot exceed 30 mm")
        if self.vision_plan_valid_for_s < self.lowering_timeout_s:
            raise ValueError(
                "vision plan validity must cover the full lowering timeout"
            )
        if int(self.vision_evidence_min_samples) < 2:
            raise ValueError("vision_evidence_min_samples must be at least 2")
        object.__setattr__(
            self,
            "vision_evidence_min_samples",
            int(self.vision_evidence_min_samples),
        )

    @classmethod
    def from_root_config(cls, root: Mapping[str, Any]) -> "PlacementConfig":
        if not isinstance(root, Mapping):
            raise TypeError("root config must be a mapping")
        raw_value = root.get("placement", {})
        if not isinstance(raw_value, Mapping):
            raise ValueError("placement configuration block must be an object")
        raw = raw_value
        allowed = {
            "enabled",
            "vision_geometry_release_enabled",
            "minimum_time_s",
            "linear_velocity_limit_mps",
            "angular_velocity_limit_radps",
            "linear_acceleration_limit_mps2",
            "angular_acceleration_limit_radps2",
            "pre_motion_clearance_floor_m",
            "maximum_descent_m",
            "squeeze_offset_m",
            "release_spread_m",
            "maximum_release_spread_m",
            "joint_stiffness_nm_per_rad",
            "joint_damping_ratio",
            "nullspace_weight",
            "nullspace_kp",
            "nullspace_kd",
            "nullspace_cost_weight",
            "pre_place_verify_dwell_s",
            "lowering_timeout_s",
            "seated_dwell_s",
            "release_timeout_s",
            "release_target_dwell_s",
            "feedback_stale_s",
            "lower_z_tolerance_m",
            "lower_midpoint_xy_drift_m",
            "lower_rotation_tolerance_deg",
            "release_target_translation_tolerance_m",
            "release_target_rotation_tolerance_deg",
            "vision_seating_max_uncertainty_m",
            "vision_evidence_fresh_after_s",
            "vision_plan_valid_for_s",
            "vision_gap_stability_tolerance_m",
            "vision_evidence_min_samples",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(
                "unknown placement configuration key(s): " + ", ".join(unknown)
            )

        defaults = cls()
        return cls(
            pre_motion_clearance_floor_m=float(
                raw.get(
                    "pre_motion_clearance_floor_m",
                    defaults.pre_motion_clearance_floor_m,
                )
            ),
            maximum_descent_m=float(
                raw.get("maximum_descent_m", defaults.maximum_descent_m)
            ),
            pre_place_verify_dwell_s=float(
                raw.get(
                    "pre_place_verify_dwell_s",
                    defaults.pre_place_verify_dwell_s,
                )
            ),
            lowering_timeout_s=float(
                raw.get("lowering_timeout_s", defaults.lowering_timeout_s)
            ),
            seated_dwell_s=float(
                raw.get("seated_dwell_s", defaults.seated_dwell_s)
            ),
            release_timeout_s=float(
                raw.get("release_timeout_s", defaults.release_timeout_s)
            ),
            release_target_dwell_s=float(
                raw.get(
                    "release_target_dwell_s",
                    defaults.release_target_dwell_s,
                )
            ),
            feedback_stale_s=float(
                raw.get("feedback_stale_s", defaults.feedback_stale_s)
            ),
            lower_z_tolerance_m=float(
                raw.get("lower_z_tolerance_m", defaults.lower_z_tolerance_m)
            ),
            lower_midpoint_xy_drift_m=float(
                raw.get(
                    "lower_midpoint_xy_drift_m",
                    defaults.lower_midpoint_xy_drift_m,
                )
            ),
            lower_rotation_tolerance_rad=math.radians(
                float(
                    raw.get(
                        "lower_rotation_tolerance_deg",
                        math.degrees(defaults.lower_rotation_tolerance_rad),
                    )
                )
            ),
            release_target_translation_tolerance_m=float(
                raw.get(
                    "release_target_translation_tolerance_m",
                    defaults.release_target_translation_tolerance_m,
                )
            ),
            release_target_rotation_tolerance_rad=math.radians(
                float(
                    raw.get(
                        "release_target_rotation_tolerance_deg",
                        math.degrees(
                            defaults.release_target_rotation_tolerance_rad
                        ),
                    )
                )
            ),
            release_spread_m=float(
                raw.get("release_spread_m", defaults.release_spread_m)
            ),
            vision_seating_max_uncertainty_m=float(
                raw.get(
                    "vision_seating_max_uncertainty_m",
                    defaults.vision_seating_max_uncertainty_m,
                )
            ),
            vision_evidence_fresh_after_s=float(
                raw.get(
                    "vision_evidence_fresh_after_s",
                    defaults.vision_evidence_fresh_after_s,
                )
            ),
            vision_plan_valid_for_s=float(
                raw.get(
                    "vision_plan_valid_for_s",
                    defaults.vision_plan_valid_for_s,
                )
            ),
            vision_gap_stability_tolerance_m=float(
                raw.get(
                    "vision_gap_stability_tolerance_m",
                    defaults.vision_gap_stability_tolerance_m,
                )
            ),
            vision_evidence_min_samples=int(
                raw.get(
                    "vision_evidence_min_samples",
                    defaults.vision_evidence_min_samples,
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class PlacementDescentPlan:
    plan_id: str
    freeze_monotonic_s: float
    planned_delta_z_m: float
    min_delta_z_m: float
    max_delta_z_m: float
    gap_m: float
    gap_uncertainty_m: float
    box_bottom_z_lower_bound_m: float
    stack_top_z_upper_bound_m: float
    stack_plane_z_base_m: float
    stack_plane_uncertainty_m: float
    stack_plane_timestamp_s: float
    stack_plane_sequence: int
    bilateral_eef_timestamp_s: float
    bilateral_eef_state_sequence: int
    right_eef_base: Any
    left_eef_base: Any
    right_target_base: Any
    left_target_base: Any
    valid: bool
    rejection_reason: str | None
    source: str

    def __post_init__(self) -> None:
        plan_id = str(self.plan_id).strip()
        if not plan_id:
            raise ValueError("plan_id must not be empty")
        object.__setattr__(self, "plan_id", plan_id)
        source = str(self.source).strip()
        if not source:
            raise ValueError("source must not be empty")
        object.__setattr__(self, "source", source)
        for name in (
            "freeze_monotonic_s",
            "planned_delta_z_m",
            "min_delta_z_m",
            "max_delta_z_m",
            "gap_m",
            "box_bottom_z_lower_bound_m",
            "stack_top_z_upper_bound_m",
            "stack_plane_z_base_m",
            "stack_plane_timestamp_s",
            "bilateral_eef_timestamp_s",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        for name in (
            "gap_uncertainty_m",
            "stack_plane_uncertainty_m",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        if self.planned_delta_z_m <= 0.0:
            raise ValueError("planned_delta_z_m must be positive")
        if self.min_delta_z_m <= 0.0 or self.max_delta_z_m <= 0.0:
            raise ValueError("delta bounds must be positive")
        if self.min_delta_z_m > self.planned_delta_z_m + 1e-12:
            raise ValueError("min_delta_z_m cannot exceed planned_delta_z_m")
        if self.planned_delta_z_m > self.max_delta_z_m + 1e-12:
            raise ValueError("planned_delta_z_m cannot exceed max_delta_z_m")
        for name in ("stack_plane_sequence", "bilateral_eef_state_sequence"):
            sequence = int(getattr(self, name))
            if sequence < 1:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, sequence)
        object.__setattr__(
            self,
            "right_eef_base",
            _readonly_transform(self.right_eef_base, "right_eef_base"),
        )
        object.__setattr__(
            self,
            "left_eef_base",
            _readonly_transform(self.left_eef_base, "left_eef_base"),
        )
        object.__setattr__(
            self,
            "right_target_base",
            _readonly_transform(self.right_target_base, "right_target_base"),
        )
        object.__setattr__(
            self,
            "left_target_base",
            _readonly_transform(self.left_target_base, "left_target_base"),
        )
        expected_right_target = np.array(self.right_eef_base, dtype=np.float64, copy=True)
        expected_left_target = np.array(self.left_eef_base, dtype=np.float64, copy=True)
        expected_right_target[2, 3] -= self.planned_delta_z_m
        expected_left_target[2, 3] -= self.planned_delta_z_m
        if not np.allclose(
            self.right_target_base,
            expected_right_target,
            rtol=0.0,
            atol=1e-9,
        ) or not np.allclose(
            self.left_target_base,
            expected_left_target,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError("target transforms must match planned_delta_z_m")
        reason = None if self.rejection_reason is None else str(self.rejection_reason)
        object.__setattr__(self, "valid", bool(self.valid))
        if not self.valid:
            raise ValueError("PlacementDescentPlan must be valid")
        if reason:
            raise ValueError("valid descent plan cannot carry a rejection reason")
        object.__setattr__(self, "rejection_reason", reason)


@dataclass(frozen=True, slots=True)
class PlacementInput:
    now_s: float
    feedback_timestamp_s: float
    right_eef_base: Any
    left_eef_base: Any
    arrived_hold: bool
    post_zero_wheel_stop: bool
    zero_command_ack: bool
    measured_state_fresh: bool
    controller_stream_healthy: bool
    controller_arm_mode: str
    controller_target_ack: bool
    right_target_base: Any | None = None
    left_target_base: Any | None = None
    allow_vision_geometry_release: bool = False
    predicted_box_bottom_gap_m: float | None = None
    predicted_box_bottom_gap_uncertainty_m: float | None = None
    gap_observation_timestamp_s: float | None = None
    gap_observation_sequence: int | None = None
    box_bottom_z_base_m: float | None = None
    box_bottom_z_uncertainty_m: float | None = None
    stack_top_z_base_m: float | None = None
    stack_top_uncertainty_m: float | None = None
    stack_plane_z_base_m: float | None = None
    stack_plane_uncertainty_m: float | None = None
    stack_plane_timestamp_s: float | None = None
    stack_plane_sequence: int | None = None
    bilateral_eef_timestamp_s: float | None = None
    bilateral_eef_state_sequence: int | None = None
    descent_plan_source: str = "vision_eef_fk"

    def __post_init__(self) -> None:
        object.__setattr__(self, "now_s", _finite(self.now_s, "now_s"))
        object.__setattr__(
            self,
            "feedback_timestamp_s",
            _finite(self.feedback_timestamp_s, "feedback_timestamp_s"),
        )
        object.__setattr__(
            self,
            "right_eef_base",
            _readonly_transform(self.right_eef_base, "right_eef_base"),
        )
        object.__setattr__(
            self,
            "left_eef_base",
            _readonly_transform(self.left_eef_base, "left_eef_base"),
        )
        object.__setattr__(
            self,
            "right_target_base",
            _optional_transform(self.right_target_base, "right_target_base"),
        )
        object.__setattr__(
            self,
            "left_target_base",
            _optional_transform(self.left_target_base, "left_target_base"),
        )
        object.__setattr__(
            self,
            "controller_arm_mode",
            str(self.controller_arm_mode).strip(),
        )
        if self.predicted_box_bottom_gap_m is not None:
            object.__setattr__(
                self,
                "predicted_box_bottom_gap_m",
                _finite(
                    self.predicted_box_bottom_gap_m,
                    "predicted_box_bottom_gap_m",
                ),
            )
        if self.predicted_box_bottom_gap_uncertainty_m is not None:
            object.__setattr__(
                self,
                "predicted_box_bottom_gap_uncertainty_m",
                _nonnegative(
                    self.predicted_box_bottom_gap_uncertainty_m,
                    "predicted_box_bottom_gap_uncertainty_m",
                ),
            )
        if self.gap_observation_timestamp_s is not None:
            object.__setattr__(
                self,
                "gap_observation_timestamp_s",
                _finite(
                    self.gap_observation_timestamp_s,
                    "gap_observation_timestamp_s",
                ),
            )
        if self.gap_observation_sequence is not None:
            sequence = int(self.gap_observation_sequence)
            if sequence < 0:
                raise ValueError("gap_observation_sequence must be non-negative")
            object.__setattr__(self, "gap_observation_sequence", sequence)
        for name in (
            "box_bottom_z_base_m",
            "stack_top_z_base_m",
            "stack_plane_z_base_m",
            "stack_plane_timestamp_s",
            "bilateral_eef_timestamp_s",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(value, name))
        for name in (
            "box_bottom_z_uncertainty_m",
            "stack_top_uncertainty_m",
            "stack_plane_uncertainty_m",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonnegative(value, name))
        for name in ("stack_plane_sequence", "bilateral_eef_state_sequence"):
            value = getattr(self, name)
            if value is not None:
                sequence = int(value)
                if sequence < 1:
                    raise ValueError(f"{name} must be positive")
                object.__setattr__(self, name, sequence)
        object.__setattr__(
            self,
            "descent_plan_source",
            str(self.descent_plan_source).strip() or "vision_eef_fk",
        )


@dataclass(frozen=True, slots=True)
class PlacementOutput:
    state: PlacementState
    request: PlacementRequest
    reason: str
    mobility_command: tuple[float, float, float]
    done: bool
    faulted: bool
    release_authorized: bool
    diagnostics: dict[str, object]
    descent_plan: PlacementDescentPlan | None = None

    def __post_init__(self) -> None:
        if tuple(self.mobility_command) != ZERO_MOBILITY_COMMAND:
            raise ValueError("placement output must command exact-zero mobility")


class Slot1PlacementSequencer:
    """Deterministic placement state machine; faults are permanently latched."""

    def __init__(self, config: PlacementConfig | None = None) -> None:
        self.config = config or PlacementConfig()
        self.state = PlacementState.PRE_PLACE_VERIFY
        self._last_update_s: float | None = None
        self._state_started_s: float | None = None
        self._verify_started_s: float | None = None
        self._contact_started_s: float | None = None
        self._release_target_started_s: float | None = None
        self._fault_reason: str | None = None
        self._baseline_gap_m: float | None = None
        self._baseline_gap_uncertainty_m: float | None = None
        self._baseline_gap_timestamp_s: float | None = None
        self._baseline_gap_sequence: int | None = None
        self._vision_gap_anchor_m: float | None = None
        self._vision_gap_sample_count = 0
        self._lower_start_right: np.ndarray | None = None
        self._lower_start_left: np.ndarray | None = None
        self._descent_plan: PlacementDescentPlan | None = None
        self._descent_plan_rejection_reason: str | None = None

    def update(self, sample: PlacementInput) -> PlacementOutput:
        if not isinstance(sample, PlacementInput):
            raise TypeError("sample must be PlacementInput")
        if self._last_update_s is not None and sample.now_s < self._last_update_s:
            return self._fault("nonmonotonic_controller_time", sample)
        self._last_update_s = sample.now_s

        if self.state is PlacementState.FAULT_HOLD:
            return self._output(
                sample,
                PlacementRequest.HOLD_CURRENT,
                self._fault_reason or "fault_latched",
            )

        safety_reason = self._safety_reason(sample)
        if safety_reason is not None:
            return self._fault(safety_reason, sample)

        if self.state is PlacementState.PRE_PLACE_VERIFY:
            return self._update_pre_place(sample)
        if self.state is PlacementState.LOWERING:
            return self._update_lowering(sample)
        if self.state is PlacementState.SEATED:
            return self._update_seated(sample)
        if self.state is PlacementState.RELEASING:
            return self._update_releasing(sample)
        if self.state is PlacementState.RELEASED:
            return self._output(
                sample,
                PlacementRequest.HOLD_CURRENT,
                "released_hold",
                done=True,
            )
        return self._fault("unknown_placement_state", sample)

    def _update_pre_place(self, sample: PlacementInput) -> PlacementOutput:
        reason = self._start_gate_reason(sample)
        if reason is not None:
            self._reset_pre_place_evidence()
            return self._output(sample, PlacementRequest.HOLD_CURRENT, reason)
        if self._verify_started_s is None:
            self._begin_pre_place_evidence(sample)
            return self._output(
                sample,
                PlacementRequest.HOLD_CURRENT,
                "pre_place_verify_dwell",
            )
        if sample.allow_vision_geometry_release:
            if not self._record_stable_vision_gap(sample):
                self._begin_pre_place_evidence(sample)
                return self._output(
                    sample,
                    PlacementRequest.HOLD_CURRENT,
                    "vision_gap_stability_reset",
                )
            if self._vision_gap_sample_count < self.config.vision_evidence_min_samples:
                return self._output(
                    sample,
                    PlacementRequest.HOLD_CURRENT,
                    "insufficient_contiguous_vision_gap_samples",
                )
        if sample.now_s - self._verify_started_s < self.config.pre_place_verify_dwell_s:
            return self._output(
                sample,
                PlacementRequest.HOLD_CURRENT,
                "pre_place_verify_dwell",
            )
        plan, rejection_reason = self._make_descent_plan(sample)
        if rejection_reason is not None:
            self._descent_plan_rejection_reason = rejection_reason
            return self._fault(rejection_reason, sample)
        assert plan is not None
        self._descent_plan_rejection_reason = None
        self._descent_plan = plan
        self._lower_start_right = plan.right_eef_base
        self._lower_start_left = plan.left_eef_base
        self._transition(PlacementState.LOWERING, sample.now_s)
        return self._output(
            sample,
            PlacementRequest.LOWER_CARTESIAN_PLANNED,
            "lowering_started",
            descent_plan=plan,
        )

    def _reset_pre_place_evidence(self) -> None:
        self._verify_started_s = None
        self._baseline_gap_m = None
        self._baseline_gap_uncertainty_m = None
        self._baseline_gap_timestamp_s = None
        self._baseline_gap_sequence = None
        self._vision_gap_anchor_m = None
        self._vision_gap_sample_count = 0
        self._descent_plan = None
        self._descent_plan_rejection_reason = None

    def _begin_pre_place_evidence(self, sample: PlacementInput) -> None:
        self._verify_started_s = sample.now_s
        self._baseline_gap_m = sample.predicted_box_bottom_gap_m
        self._baseline_gap_uncertainty_m = (
            sample.predicted_box_bottom_gap_uncertainty_m
        )
        self._baseline_gap_timestamp_s = sample.gap_observation_timestamp_s
        self._baseline_gap_sequence = sample.gap_observation_sequence
        self._vision_gap_anchor_m = sample.predicted_box_bottom_gap_m
        self._vision_gap_sample_count = int(
            sample.allow_vision_geometry_release
            and sample.predicted_box_bottom_gap_m is not None
        )

    def _record_stable_vision_gap(self, sample: PlacementInput) -> bool:
        gap = sample.predicted_box_bottom_gap_m
        if gap is None or self._vision_gap_anchor_m is None:
            return False
        if (
            abs(gap - self._vision_gap_anchor_m)
            > self.config.vision_gap_stability_tolerance_m
        ):
            return False
        sequence = sample.gap_observation_sequence
        if (
            sequence is not None
            and self._baseline_gap_sequence is not None
            and sequence <= self._baseline_gap_sequence
        ):
            return True
        self._baseline_gap_m = gap
        self._baseline_gap_uncertainty_m = (
            sample.predicted_box_bottom_gap_uncertainty_m
        )
        self._baseline_gap_timestamp_s = sample.gap_observation_timestamp_s
        self._baseline_gap_sequence = sequence
        self._vision_gap_sample_count += 1
        return True

    def _update_lowering(self, sample: PlacementInput) -> PlacementOutput:
        if self._state_elapsed(sample) > self.config.lowering_timeout_s:
            return self._fault("lowering_or_seating_timeout", sample)
        if sample.controller_arm_mode != LOWERING_MODE:
            return self._output(
                sample,
                PlacementRequest.LOWER_CARTESIAN_PLANNED,
                "waiting_for_lowering_mode",
                descent_plan=self._descent_plan,
            )
        if not sample.controller_target_ack:
            return self._output(
                sample,
                PlacementRequest.HOLD_CURRENT,
                "waiting_for_lowering_command_ack",
            )
        if not self._lower_geometry_reached(sample):
            return self._output(
                sample,
                PlacementRequest.HOLD_CURRENT,
                "waiting_for_measured_planned_descent",
            )
        if not self._seating_evidence(sample):
            return self._fault("seating_evidence_unavailable", sample)
        self._contact_started_s = sample.now_s
        self._transition(PlacementState.SEATED, sample.now_s)
        return self._output(
            sample,
            PlacementRequest.HOLD_CURRENT,
            "seating_evidence_started",
        )

    def _update_seated(self, sample: PlacementInput) -> PlacementOutput:
        if sample.controller_arm_mode != LOWERING_MODE or not sample.controller_target_ack:
            return self._fault("lowering_hold_ack_lost_before_release", sample)
        if not self._lower_geometry_reached(sample):
            return self._fault("lowered_geometry_lost_before_release", sample)
        if not self._seating_evidence(sample):
            return self._fault("seating_evidence_lost", sample)
        release_evidence_reason = self._release_evidence_fault_reason(sample)
        if release_evidence_reason is not None:
            return self._fault(release_evidence_reason, sample)
        if self._contact_started_s is None:
            return self._fault("seating_dwell_timestamp_missing", sample)
        if sample.now_s - self._contact_started_s < self.config.seated_dwell_s:
            return self._output(
                sample,
                PlacementRequest.HOLD_CURRENT,
                "seating_evidence_dwell",
            )
        self._transition(PlacementState.RELEASING, sample.now_s)
        return self._output(
            sample,
            PlacementRequest.SPREAD_RELEASE,
            "release_started",
            release_authorized=True,
        )

    def _update_releasing(self, sample: PlacementInput) -> PlacementOutput:
        release_evidence_reason = self._release_evidence_fault_reason(sample)
        if release_evidence_reason is not None:
            return self._fault(release_evidence_reason, sample)
        if self._state_elapsed(sample) > self.config.release_timeout_s:
            return self._fault("release_target_timeout", sample)
        if sample.controller_arm_mode != RELEASE_MODE:
            return self._output(
                sample,
                PlacementRequest.SPREAD_RELEASE,
                "waiting_for_release_mode",
                release_authorized=True,
            )
        if not sample.controller_target_ack:
            return self._output(
                sample,
                PlacementRequest.HOLD_CURRENT,
                "waiting_for_release_command_ack",
                release_authorized=True,
            )
        if not self._release_target_reached(sample):
            self._release_target_started_s = None
            return self._output(
                sample,
                PlacementRequest.HOLD_CURRENT,
                "waiting_for_release_target",
                release_authorized=True,
            )
        if self._release_target_started_s is None:
            self._release_target_started_s = sample.now_s
            return self._output(
                sample,
                PlacementRequest.HOLD_CURRENT,
                "release_target_dwell",
                release_authorized=True,
            )
        if (
            sample.now_s - self._release_target_started_s
            < self.config.release_target_dwell_s
        ):
            return self._output(
                sample,
                PlacementRequest.HOLD_CURRENT,
                "release_target_dwell",
                release_authorized=True,
            )
        self._transition(PlacementState.RELEASED, sample.now_s)
        return self._output(
            sample,
            PlacementRequest.HOLD_CURRENT,
            "released",
            done=True,
            release_authorized=True,
        )

    def _start_gate_reason(self, sample: PlacementInput) -> str | None:
        if not sample.arrived_hold:
            return "arrival_state_not_arrived_hold"
        if not sample.post_zero_wheel_stop:
            return "post_zero_wheel_stop_missing"
        if not sample.zero_command_ack:
            return "zero_command_ack_missing"
        if sample.controller_arm_mode != LOADED_HOLD_MODE:
            return "loaded_cartesian_hold_mode_missing"
        if not sample.controller_target_ack:
            return "loaded_cartesian_hold_ack_missing"
        release_path_available = bool(
            sample.allow_vision_geometry_release
            and self._vision_input_is_fresh(sample)
            and self._vision_seating_evidence_for(
                sample.predicted_box_bottom_gap_m,
                sample.predicted_box_bottom_gap_uncertainty_m,
            )
        )
        if not release_path_available:
            return "seating_evidence_unavailable"
        return None

    def _vision_input_is_fresh(self, sample: PlacementInput) -> bool:
        timestamp_s = sample.gap_observation_timestamp_s
        if timestamp_s is None:
            return False
        age_s = sample.now_s - timestamp_s
        return bool(
            -1e-12 <= age_s <= self.config.vision_evidence_fresh_after_s
        )

    def _safety_reason(self, sample: PlacementInput) -> str | None:
        if not sample.measured_state_fresh:
            return "measured_state_stale"
        if not sample.controller_stream_healthy:
            return "controller_stream_unhealthy"
        if (
            self.state is not PlacementState.PRE_PLACE_VERIFY
            and not sample.post_zero_wheel_stop
        ):
            return "post_zero_wheel_stop_lost"
        feedback_age_s = sample.now_s - sample.feedback_timestamp_s
        if feedback_age_s < -1e-12:
            return "feedback_timestamp_from_future"
        if feedback_age_s > self.config.feedback_stale_s:
            return "feedback_timestamp_stale"
        return None

    def _lower_geometry_reached(self, sample: PlacementInput) -> bool:
        plan = self._descent_plan
        if plan is None or not plan.valid:
            return False
        expected_right_z = float(plan.right_target_base[2, 3])
        expected_left_z = float(plan.left_target_base[2, 3])
        z_ok = bool(
            abs(float(sample.right_eef_base[2, 3]) - expected_right_z)
            <= self.config.lower_z_tolerance_m
            and abs(float(sample.left_eef_base[2, 3]) - expected_left_z)
            <= self.config.lower_z_tolerance_m
        )
        start_midpoint = 0.5 * (
            plan.right_eef_base[:2, 3] + plan.left_eef_base[:2, 3]
        )
        current_midpoint = 0.5 * (
            sample.right_eef_base[:2, 3] + sample.left_eef_base[:2, 3]
        )
        midpoint_ok = bool(
            np.linalg.norm(current_midpoint - start_midpoint)
            <= self.config.lower_midpoint_xy_drift_m
        )
        rotation_ok = bool(
            _rotation_error_rad(plan.right_eef_base, sample.right_eef_base)
            <= self.config.lower_rotation_tolerance_rad
            and _rotation_error_rad(plan.left_eef_base, sample.left_eef_base)
            <= self.config.lower_rotation_tolerance_rad
        )
        target_z_ok = True
        if sample.right_target_base is not None and sample.left_target_base is not None:
            target_z_ok = bool(
                abs(
                    float(sample.right_eef_base[2, 3])
                    - float(sample.right_target_base[2, 3])
                )
                <= self.config.lower_z_tolerance_m
                and abs(
                    float(sample.left_eef_base[2, 3])
                    - float(sample.left_target_base[2, 3])
                )
                <= self.config.lower_z_tolerance_m
            )
        return z_ok and midpoint_ok and rotation_ok and target_z_ok

    def _release_target_reached(self, sample: PlacementInput) -> bool:
        if sample.right_target_base is None or sample.left_target_base is None:
            return False
        target_reached = bool(
            np.linalg.norm(
                sample.right_eef_base[:3, 3] - sample.right_target_base[:3, 3]
            )
            <= self.config.release_target_translation_tolerance_m
            and np.linalg.norm(
                sample.left_eef_base[:3, 3] - sample.left_target_base[:3, 3]
            )
            <= self.config.release_target_translation_tolerance_m
            and _rotation_error_rad(sample.right_eef_base, sample.right_target_base)
            <= self.config.release_target_rotation_tolerance_rad
            and _rotation_error_rad(sample.left_eef_base, sample.left_target_base)
            <= self.config.release_target_rotation_tolerance_rad
        )
        if not target_reached:
            return False
        plan = self._descent_plan
        if plan is None:
            return False
        initial_separation_m = float(
            np.linalg.norm(
                plan.right_eef_base[:3, 3] - plan.left_eef_base[:3, 3]
            )
        )
        current_separation_m = float(
            np.linalg.norm(
                sample.right_eef_base[:3, 3] - sample.left_eef_base[:3, 3]
            )
        )
        minimum_increase_m = max(
            0.0,
            2.0
            * (
                self.config.release_spread_m
                - self.config.release_target_translation_tolerance_m
            ),
        )
        return bool(
            current_separation_m - initial_separation_m >= minimum_increase_m
        )

    def _vision_seating_evidence(self, now_s: float) -> bool:
        if self._baseline_gap_timestamp_s is None:
            return False
        plan_age_s = now_s - self._baseline_gap_timestamp_s
        if plan_age_s < -1e-12 or plan_age_s > self.config.vision_plan_valid_for_s:
            return False
        return self._vision_seating_evidence_for(
            self._baseline_gap_m,
            self._baseline_gap_uncertainty_m,
        )

    def _vision_seating_evidence_for(
        self,
        gap: float | None,
        uncertainty: float | None,
    ) -> bool:
        if gap is None or uncertainty is None:
            return False
        if uncertainty > self.config.vision_seating_max_uncertainty_m:
            return False
        return bool(
            gap - uncertainty >= self.config.pre_motion_clearance_floor_m
            and gap > 0.0
        )

    def _seating_evidence(self, sample: PlacementInput) -> bool:
        plan = self._descent_plan
        if plan is None or not plan.valid:
            return False
        if (
            sample.stack_plane_z_base_m is not None
            and abs(sample.stack_plane_z_base_m - plan.stack_plane_z_base_m)
            > self.config.vision_gap_stability_tolerance_m
        ):
            return False
        if (
            sample.stack_plane_uncertainty_m is not None
            and sample.stack_plane_uncertainty_m
            > self.config.vision_seating_max_uncertainty_m
        ):
            return False
        return bool(
            sample.allow_vision_geometry_release
            and self._vision_seating_evidence(sample.now_s)
        )

    def _release_evidence_fault_reason(self, sample: PlacementInput) -> str | None:
        plan = self._descent_plan
        if plan is None or not plan.valid:
            return "release_descent_plan_missing"
        required = {
            "stack_plane_z_base_m": sample.stack_plane_z_base_m,
            "stack_plane_uncertainty_m": sample.stack_plane_uncertainty_m,
            "stack_plane_timestamp_s": sample.stack_plane_timestamp_s,
            "stack_plane_sequence": sample.stack_plane_sequence,
            "bilateral_eef_timestamp_s": sample.bilateral_eef_timestamp_s,
            "bilateral_eef_state_sequence": sample.bilateral_eef_state_sequence,
        }
        missing = tuple(name for name, value in required.items() if value is None)
        if missing:
            return "release_evidence_missing:" + ",".join(missing)

        stack_plane_timestamp = float(sample.stack_plane_timestamp_s)
        bilateral_eef_timestamp = float(sample.bilateral_eef_timestamp_s)
        stack_plane_sequence = int(sample.stack_plane_sequence)
        bilateral_eef_sequence = int(sample.bilateral_eef_state_sequence)
        if (
            stack_plane_timestamp < plan.stack_plane_timestamp_s - 1e-12
            or stack_plane_sequence < plan.stack_plane_sequence
        ):
            return "release_stack_plane_regressed"
        if (
            bilateral_eef_timestamp < plan.bilateral_eef_timestamp_s - 1e-12
            or bilateral_eef_sequence < plan.bilateral_eef_state_sequence
        ):
            return "release_bilateral_eef_regressed"
        if not (
            -1e-12
            <= sample.now_s - stack_plane_timestamp
            <= self.config.vision_evidence_fresh_after_s
        ):
            return "release_stack_plane_stale"
        if not (
            -1e-12
            <= sample.now_s - bilateral_eef_timestamp
            <= self.config.feedback_stale_s
        ):
            return "release_bilateral_eef_stale"
        if (
            abs(float(sample.stack_plane_z_base_m) - plan.stack_plane_z_base_m)
            > self.config.vision_gap_stability_tolerance_m
            or float(sample.stack_plane_uncertainty_m)
            > self.config.vision_seating_max_uncertainty_m
        ):
            return "release_stack_plane_inconsistent"
        expected_midpoint = 0.5 * (
            plan.right_eef_base[:2, 3] + plan.left_eef_base[:2, 3]
        )
        current_midpoint = 0.5 * (
            sample.right_eef_base[:2, 3] + sample.left_eef_base[:2, 3]
        )
        initial_separation_m = float(
            np.linalg.norm(
                plan.right_eef_base[:3, 3] - plan.left_eef_base[:3, 3]
            )
        )
        current_separation_m = float(
            np.linalg.norm(
                sample.right_eef_base[:3, 3] - sample.left_eef_base[:3, 3]
            )
        )
        max_release_separation_m = initial_separation_m + 2.0 * (
            self.config.release_spread_m
            + self.config.release_target_translation_tolerance_m
        )
        right_z_error_m = abs(
            float(sample.right_eef_base[2, 3])
            - float(plan.right_target_base[2, 3])
        )
        left_z_error_m = abs(
            float(sample.left_eef_base[2, 3])
            - float(plan.left_target_base[2, 3])
        )
        if (
            right_z_error_m > self.config.lower_z_tolerance_m
            or left_z_error_m > self.config.lower_z_tolerance_m
            or np.linalg.norm(current_midpoint - expected_midpoint)
            > self.config.lower_midpoint_xy_drift_m
            or _rotation_error_rad(plan.right_eef_base, sample.right_eef_base)
            > self.config.release_target_rotation_tolerance_rad
            or _rotation_error_rad(plan.left_eef_base, sample.left_eef_base)
            > self.config.release_target_rotation_tolerance_rad
            or current_separation_m + self.config.release_target_translation_tolerance_m
            < initial_separation_m
            or current_separation_m > max_release_separation_m
        ):
            return "release_bilateral_eef_inconsistent"
        if not sample.allow_vision_geometry_release:
            return "release_vision_geometry_disabled"
        return None

    def _make_descent_plan(
        self,
        sample: PlacementInput,
    ) -> tuple[PlacementDescentPlan | None, str | None]:
        required = {
            "predicted_box_bottom_gap_m": sample.predicted_box_bottom_gap_m,
            "predicted_box_bottom_gap_uncertainty_m": (
                sample.predicted_box_bottom_gap_uncertainty_m
            ),
            "gap_observation_timestamp_s": sample.gap_observation_timestamp_s,
            "gap_observation_sequence": sample.gap_observation_sequence,
            "box_bottom_z_base_m": sample.box_bottom_z_base_m,
            "box_bottom_z_uncertainty_m": sample.box_bottom_z_uncertainty_m,
            "stack_top_z_base_m": sample.stack_top_z_base_m,
            "stack_top_uncertainty_m": sample.stack_top_uncertainty_m,
            "stack_plane_z_base_m": sample.stack_plane_z_base_m,
            "stack_plane_uncertainty_m": sample.stack_plane_uncertainty_m,
            "stack_plane_timestamp_s": sample.stack_plane_timestamp_s,
            "stack_plane_sequence": sample.stack_plane_sequence,
            "bilateral_eef_timestamp_s": sample.bilateral_eef_timestamp_s,
            "bilateral_eef_state_sequence": sample.bilateral_eef_state_sequence,
        }
        missing = tuple(name for name, value in required.items() if value is None)
        if missing:
            return None, "descent_authority_missing:" + ",".join(missing)

        reported_gap = float(sample.predicted_box_bottom_gap_m)
        reported_uncertainty = float(
            sample.predicted_box_bottom_gap_uncertainty_m
        )
        box_bottom_z = float(sample.box_bottom_z_base_m)
        box_uncertainty = float(sample.box_bottom_z_uncertainty_m)
        stack_top_z = float(sample.stack_top_z_base_m)
        stack_uncertainty = float(sample.stack_top_uncertainty_m)
        stack_plane_z = float(sample.stack_plane_z_base_m)
        stack_plane_uncertainty = float(sample.stack_plane_uncertainty_m)
        stack_plane_timestamp = float(sample.stack_plane_timestamp_s)
        bilateral_eef_timestamp = float(sample.bilateral_eef_timestamp_s)
        stack_plane_sequence = int(sample.stack_plane_sequence)
        bilateral_eef_sequence = int(sample.bilateral_eef_state_sequence)

        gap = box_bottom_z - stack_top_z
        uncertainty = box_uncertainty + stack_uncertainty
        box_lower_bound = box_bottom_z - box_uncertainty
        stack_upper_bound = stack_top_z + stack_uncertainty
        min_delta = box_lower_bound - stack_upper_bound
        max_delta = (
            box_bottom_z + box_uncertainty
            - (stack_top_z - stack_uncertainty)
        )
        rejection: str | None = None
        if (
            abs(reported_gap - gap) > 1e-6
            or abs(reported_uncertainty - uncertainty) > 1e-6
        ):
            rejection = "descent_gap_bounds_inconsistent"
        elif (
            abs(stack_plane_z - stack_top_z)
            > self.config.vision_gap_stability_tolerance_m
            or abs(stack_plane_uncertainty - stack_uncertainty) > 1e-6
        ):
            rejection = "descent_stack_plane_inconsistent"
        elif not (
            -1e-12
            <= sample.now_s - stack_plane_timestamp
            <= self.config.vision_evidence_fresh_after_s
        ):
            rejection = "descent_stack_plane_stale"
        elif not (
            -1e-12
            <= sample.now_s - bilateral_eef_timestamp
            <= self.config.feedback_stale_s
        ):
            rejection = "descent_bilateral_eef_stale"
        elif uncertainty > self.config.vision_seating_max_uncertainty_m:
            rejection = "descent_gap_uncertainty_too_large"
        elif min_delta < self.config.pre_motion_clearance_floor_m:
            rejection = "descent_clearance_below_50mm_floor"
        elif gap <= 0.0 or min_delta <= 0.0 or max_delta <= 0.0:
            rejection = "descent_gap_nonpositive"
        elif gap > self.config.maximum_descent_m:
            rejection = "descent_distance_too_large"
        if rejection is not None:
            return None, rejection

        right_target = np.array(sample.right_eef_base, dtype=np.float64, copy=True)
        left_target = np.array(sample.left_eef_base, dtype=np.float64, copy=True)
        right_target[2, 3] -= gap
        left_target[2, 3] -= gap
        return (
            PlacementDescentPlan(
                plan_id=uuid.uuid4().hex,
                freeze_monotonic_s=sample.now_s,
                planned_delta_z_m=gap,
                min_delta_z_m=min_delta,
                max_delta_z_m=max_delta,
                gap_m=gap,
                gap_uncertainty_m=uncertainty,
                box_bottom_z_lower_bound_m=box_lower_bound,
                stack_top_z_upper_bound_m=stack_upper_bound,
                stack_plane_z_base_m=stack_plane_z,
                stack_plane_uncertainty_m=stack_plane_uncertainty,
                stack_plane_timestamp_s=stack_plane_timestamp,
                stack_plane_sequence=int(stack_plane_sequence),
                bilateral_eef_timestamp_s=bilateral_eef_timestamp,
                bilateral_eef_state_sequence=int(bilateral_eef_sequence),
                right_eef_base=sample.right_eef_base,
                left_eef_base=sample.left_eef_base,
                right_target_base=right_target,
                left_target_base=left_target,
                valid=True,
                rejection_reason=None,
                source=sample.descent_plan_source,
            ),
            None,
        )

    def _state_elapsed(self, sample: PlacementInput) -> float:
        return (
            0.0
            if self._state_started_s is None
            else sample.now_s - self._state_started_s
        )

    def _transition(self, state: PlacementState, now_s: float) -> None:
        self.state = state
        self._state_started_s = now_s

    def _fault(self, reason: str, sample: PlacementInput) -> PlacementOutput:
        self.state = PlacementState.FAULT_HOLD
        self._fault_reason = reason
        return self._output(
            sample,
            PlacementRequest.HOLD_CURRENT,
            reason,
            faulted=True,
        )

    def _output(
        self,
        sample: PlacementInput,
        request: PlacementRequest,
        reason: str,
        *,
        done: bool = False,
        faulted: bool | None = None,
        release_authorized: bool = False,
        descent_plan: PlacementDescentPlan | None = None,
    ) -> PlacementOutput:
        active_plan = descent_plan or self._descent_plan
        rejected_plan_reason = (
            self._descent_plan_rejection_reason if active_plan is None else None
        )
        diagnostics: dict[str, object] = {
            "controller_arm_mode": sample.controller_arm_mode,
            "controller_target_ack": sample.controller_target_ack,
            "vision_seating_evidence": self._vision_seating_evidence(sample.now_s),
            "predicted_box_bottom_gap_m": self._baseline_gap_m,
            "predicted_box_bottom_gap_uncertainty_m": (
                self._baseline_gap_uncertainty_m
            ),
            "release_authority": "vision_geometry_only",
            "vision_geometry_release": sample.allow_vision_geometry_release,
            "vision_gap_timestamp_s": self._baseline_gap_timestamp_s,
            "vision_gap_sequence": self._baseline_gap_sequence,
            "vision_gap_sample_count": self._vision_gap_sample_count,
            "vision_plan_age_s": (
                None
                if self._baseline_gap_timestamp_s is None
                else sample.now_s - self._baseline_gap_timestamp_s
            ),
            "descent_plan_id": None if active_plan is None else active_plan.plan_id,
            "descent_plan_valid": (
                False if rejected_plan_reason is not None else None
            )
            if active_plan is None
            else active_plan.valid,
            "descent_plan_rejection_reason": (
                rejected_plan_reason
                if active_plan is None
                else active_plan.rejection_reason
            ),
            "planned_delta_z_m": (
                None if active_plan is None else active_plan.planned_delta_z_m
            ),
            "state_elapsed_s": self._state_elapsed(sample),
        }
        return PlacementOutput(
            state=self.state,
            request=request,
            reason=reason,
            mobility_command=ZERO_MOBILITY_COMMAND,
            done=done,
            faulted=(self.state is PlacementState.FAULT_HOLD)
            if faulted is None
            else faulted,
            release_authorized=release_authorized,
            diagnostics=diagnostics,
            descent_plan=active_plan,
        )


__all__ = [
    "LOADED_HOLD_MODE",
    "LOWERING_MODE",
    "PlacementConfig",
    "PlacementDescentPlan",
    "PlacementInput",
    "PlacementOutput",
    "PlacementRequest",
    "PlacementState",
    "RELEASE_MODE",
    "Slot1PlacementSequencer",
    "ZERO_MOBILITY_COMMAND",
]
