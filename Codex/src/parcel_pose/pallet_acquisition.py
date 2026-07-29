"""Pure forward-only pallet-hole acquisition control.

This module deliberately has no robot-SDK, camera, or perception-model import.
The runtime reduces those systems to immutable metric odometry plus explicit
boolean evidence/gate decisions.  Consequently the controller can be exercised
with a fake monotonic clock and synthetic SE(2) samples before it is connected
to the RB-Y1 command arbiter.

Partial L-corner geometry has one narrow authority: it may authorize one new
forward step.  It never creates a pallet target, lateral command, yaw command,
or fine-servo measurement.  A complete, dwell-qualified hole requests a
zero-speed handoff to the existing fine controller.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping


RELEASE_ACQUISITION_BUDGET_CAP_M = 0.150
"""Largest budget accepted by this software release."""

ABSOLUTE_ACQUISITION_BUDGET_CAP_M = 0.200
"""MVP design ceiling; intentionally unreachable without a reviewed change."""

MAX_ACQUISITION_STEP_M = 0.010
MAX_ACQUISITION_SPEED_MPS = 0.030
MIN_STATIONARY_L_CORNER_FRAMES = 5
ACQUISITION_MODE_STOP_STEP = "stop_step"
ACQUISITION_MODE_CONTINUOUS_FORWARD = "continuous_forward"
ACQUISITION_MODES = frozenset(
    (ACQUISITION_MODE_STOP_STEP, ACQUISITION_MODE_CONTINUOUS_FORWARD)
)


def _finite(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: object, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative(value: object, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _integer_at_least(value: object, minimum: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    numeric = _finite(value, name)
    result = int(numeric)
    if float(result) != numeric or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _optional_timestamp(value: object | None, name: str) -> float | None:
    if value is None:
        return None
    return _finite(value, name)


def _angle_difference(angle_rad: float, reference_rad: float) -> float:
    return (float(angle_rad) - float(reference_rad) + math.pi) % (
        2.0 * math.pi
    ) - math.pi


def _line_angle_difference(angle_rad: float, reference_rad: float) -> float:
    return (float(angle_rad) - float(reference_rad) + 0.5 * math.pi) % math.pi - (
        0.5 * math.pi
    )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _acquisition_mode(value: object, name: str = "mode") -> str:
    mode = str(value).strip().lower().replace("-", "_")
    if mode in {"stop", "step", "stop_and_observe", "stop_observe"}:
        mode = ACQUISITION_MODE_STOP_STEP
    elif mode in {"continuous", "continuous_cruise", "continuous_acquire"}:
        mode = ACQUISITION_MODE_CONTINUOUS_FORWARD
    if mode not in ACQUISITION_MODES:
        raise ValueError(
            f"{name} must be one of {sorted(ACQUISITION_MODES)}, got {value!r}"
        )
    return mode


def _first(
    mappings: tuple[Mapping[str, object], ...],
    names: tuple[str, ...],
    default: object,
) -> object:
    for mapping in mappings:
        for name in names:
            if name in mapping:
                return mapping[name]
    return default


@dataclass(frozen=True, slots=True)
class AcquisitionConfig:
    """Release-bounded stop-and-observe acquisition parameters.

    ``budget_m == 0`` is the production default and mechanically disables every
    nonzero coarse command.  This release intentionally rejects a configured
    budget above 0.15 m even though the documented design ceiling is 0.20 m.
    """

    mode: str = ACQUISITION_MODE_STOP_STEP
    budget_m: float = 0.0
    step_m: float = MAX_ACQUISITION_STEP_M
    speed_mps: float = MAX_ACQUISITION_SPEED_MPS
    stationary_frames: int = MIN_STATIONARY_L_CORNER_FRAMES
    camera_freshness_s: float = 0.30
    odometry_freshness_s: float = 0.20
    wheel_freshness_s: float = 0.20
    step_timeout_s: float = 1.50
    acquisition_timeout_s: float = 30.0
    no_progress_timeout_s: float = 0.60
    no_progress_min_m: float = 0.001
    lateral_drift_limit_m: float = 0.015
    yaw_drift_limit_rad: float = math.radians(3.0)
    target_tolerance_m: float = 0.001
    # Reverse odometry tolerance; this remains tighter than forward braking.
    overshoot_tolerance_m: float = 0.003
    # Maximum physical travel beyond one forward step target, including
    # zero-command braking. It is also reserved from the session budget.
    braking_allowance_m: float = 0.003
    settle_duration_s: float = 0.35
    brake_timeout_s: float = 1.50

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", _acquisition_mode(self.mode))
        object.__setattr__(self, "budget_m", _nonnegative(self.budget_m, "budget_m"))
        if self.budget_m > RELEASE_ACQUISITION_BUDGET_CAP_M + 1e-12:
            raise ValueError(
                "budget_m exceeds this release cap "
                f"{RELEASE_ACQUISITION_BUDGET_CAP_M:.3f} m"
            )
        if RELEASE_ACQUISITION_BUDGET_CAP_M >= ABSOLUTE_ACQUISITION_BUDGET_CAP_M:
            raise RuntimeError("release acquisition cap must remain below absolute cap")

        object.__setattr__(self, "step_m", _positive(self.step_m, "step_m"))
        if self.step_m > MAX_ACQUISITION_STEP_M + 1e-12:
            raise ValueError(f"step_m cannot exceed {MAX_ACQUISITION_STEP_M:.3f} m")

        object.__setattr__(self, "speed_mps", _positive(self.speed_mps, "speed_mps"))
        if self.speed_mps > MAX_ACQUISITION_SPEED_MPS + 1e-12:
            raise ValueError(
                f"speed_mps cannot exceed {MAX_ACQUISITION_SPEED_MPS:.3f} m/s"
            )

        object.__setattr__(
            self,
            "stationary_frames",
            _integer_at_least(
                self.stationary_frames,
                MIN_STATIONARY_L_CORNER_FRAMES,
                "stationary_frames",
            ),
        )

        positive_fields = (
            "camera_freshness_s",
            "odometry_freshness_s",
            "wheel_freshness_s",
            "step_timeout_s",
            "acquisition_timeout_s",
            "no_progress_timeout_s",
            "no_progress_min_m",
            "lateral_drift_limit_m",
            "yaw_drift_limit_rad",
            "target_tolerance_m",
            "overshoot_tolerance_m",
            "braking_allowance_m",
            "settle_duration_s",
            "brake_timeout_s",
        )
        for name in positive_fields:
            object.__setattr__(self, name, _positive(getattr(self, name), name))

        hard_upper_bounds = {
            "camera_freshness_s": 0.30,
            "odometry_freshness_s": 0.20,
            "wheel_freshness_s": 0.20,
            "step_timeout_s": 5.0,
            "acquisition_timeout_s": 60.0,
            "no_progress_timeout_s": 2.0,
            "lateral_drift_limit_m": 0.030,
            "yaw_drift_limit_rad": math.radians(5.0),
            "target_tolerance_m": 0.003,
            "overshoot_tolerance_m": 0.010,
            "braking_allowance_m": 0.010,
            "brake_timeout_s": 5.0,
        }
        for name, limit in hard_upper_bounds.items():
            if float(getattr(self, name)) > limit + 1e-12:
                raise ValueError(f"{name} exceeds its safety limit {limit}")
        if self.acquisition_timeout_s < self.step_timeout_s:
            raise ValueError("acquisition_timeout_s must cover step_timeout_s")
        if self.no_progress_timeout_s > self.step_timeout_s:
            raise ValueError("no_progress_timeout_s cannot exceed step_timeout_s")
        if self.no_progress_min_m > self.step_m:
            raise ValueError("no_progress_min_m cannot exceed step_m")
        if self.target_tolerance_m >= self.step_m:
            raise ValueError("target_tolerance_m must be smaller than step_m")
        if self.overshoot_tolerance_m < self.target_tolerance_m:
            raise ValueError(
                "overshoot_tolerance_m must be at least target_tolerance_m"
            )
        if self.braking_allowance_m < self.target_tolerance_m:
            raise ValueError(
                "braking_allowance_m must be at least target_tolerance_m"
            )
        if self.settle_duration_s < 0.35:
            raise ValueError("settle_duration_s cannot be less than 0.35 s")

    @classmethod
    def from_root_config(cls, value: Mapping[str, object]) -> "AcquisitionConfig":
        """Load ``pallet.acquisition`` (or a direct acquisition mapping).

        The parser accepts a small set of explicit aliases so replay tools can
        pass a section directly.  Unknown keys remain owned by other
        subsystems and cannot silently enter this safety-critical dataclass.
        Yaw may be configured in radians or degrees; radians take precedence.
        """

        if not isinstance(value, Mapping):
            raise TypeError("root config must be a mapping")
        pallet = _mapping(value.get("pallet", {}))
        section_value = pallet.get(
            "acquisition",
            value.get("pallet_acquisition", value.get("acquisition", value)),
        )
        section = _mapping(section_value)
        freshness = _mapping(section.get("freshness", {}))
        timeouts = _mapping(section.get("timeouts", {}))
        no_progress = _mapping(section.get("no_progress", {}))
        drift = _mapping(section.get("drift", {}))
        defaults = cls()

        yaw_rad_value = _first(
            (drift, section),
            ("yaw_limit_rad", "yaw_drift_limit_rad"),
            None,
        )
        if yaw_rad_value is None:
            yaw_deg_value = _first(
                (drift, section),
                ("yaw_limit_deg", "yaw_drift_limit_deg"),
                math.degrees(defaults.yaw_drift_limit_rad),
            )
            yaw_drift_limit_rad = math.radians(
                _finite(yaw_deg_value, "yaw_drift_limit_deg")
            )
        else:
            yaw_drift_limit_rad = _finite(yaw_rad_value, "yaw_drift_limit_rad")

        fields: dict[str, object] = {
            "budget_m": _first(
                (section,),
                ("budget_m", "forward_budget_m", "live_forward_budget_m"),
                defaults.budget_m,
            ),
            "mode": _first(
                (section,),
                ("mode", "acquisition_mode", "forward_mode"),
                defaults.mode,
            ),
            "step_m": _first(
                (section,),
                ("step_m", "forward_step_m"),
                defaults.step_m,
            ),
            "speed_mps": _first(
                (section,),
                ("speed_mps", "forward_speed_mps", "max_forward_speed_mps"),
                defaults.speed_mps,
            ),
            "stationary_frames": _first(
                (section,),
                ("stationary_frames", "min_stationary_frames", "stable_frames"),
                defaults.stationary_frames,
            ),
            "camera_freshness_s": _first(
                (freshness, section),
                ("camera_s", "camera_freshness_s", "observation_stale_after_s"),
                defaults.camera_freshness_s,
            ),
            "odometry_freshness_s": _first(
                (freshness, section),
                ("odometry_s", "odometry_freshness_s", "odometry_stale_after_s"),
                defaults.odometry_freshness_s,
            ),
            "wheel_freshness_s": _first(
                (freshness, section),
                ("wheel_s", "wheel_freshness_s", "wheel_stale_after_s"),
                defaults.wheel_freshness_s,
            ),
            "step_timeout_s": _first(
                (timeouts, section),
                ("step_s", "step_timeout_s"),
                defaults.step_timeout_s,
            ),
            "acquisition_timeout_s": _first(
                (timeouts, section),
                ("acquisition_s", "acquisition_timeout_s", "timeout_s"),
                defaults.acquisition_timeout_s,
            ),
            "brake_timeout_s": _first(
                (timeouts, section),
                ("brake_s", "brake_timeout_s", "wheel_stop_timeout_s"),
                defaults.brake_timeout_s,
            ),
            "no_progress_timeout_s": _first(
                (no_progress, section),
                ("timeout_s", "no_progress_timeout_s"),
                defaults.no_progress_timeout_s,
            ),
            "no_progress_min_m": _first(
                (no_progress, section),
                ("minimum_m", "min_m", "no_progress_min_m"),
                defaults.no_progress_min_m,
            ),
            "lateral_drift_limit_m": _first(
                (drift, section),
                ("lateral_limit_m", "lateral_drift_limit_m"),
                defaults.lateral_drift_limit_m,
            ),
            "yaw_drift_limit_rad": yaw_drift_limit_rad,
            "target_tolerance_m": _first(
                (section,),
                ("target_tolerance_m", "step_target_tolerance_m"),
                defaults.target_tolerance_m,
            ),
            "overshoot_tolerance_m": _first(
                (section,),
                ("overshoot_tolerance_m",),
                defaults.overshoot_tolerance_m,
            ),
            "braking_allowance_m": _first(
                (section,),
                (
                    "braking_allowance_m",
                    "stopping_allowance_m",
                    "coast_allowance_m",
                ),
                defaults.braking_allowance_m,
            ),
            "settle_duration_s": _first(
                (section,),
                ("settle_duration_s", "stationary_settle_s"),
                defaults.settle_duration_s,
            ),
        }
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class OdometrySample:
    """One planar base pose in a monotonic timestamp domain.

    Values are converted to floats but intentionally not rejected here: a
    transport can deliver a NaN/Inf sample, and the controller must turn that
    runtime fault into an exact-zero, named hold instead of throwing before an
    output can be selected.
    """

    timestamp_s: float
    x_m: float
    y_m: float
    yaw_rad: float

    def __post_init__(self) -> None:
        for name in ("timestamp_s", "x_m", "y_m", "yaw_rad"):
            object.__setattr__(self, name, float(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class AcquisitionDecision:
    """SDK-independent evidence presented to one controller update.

    All timestamps use the same monotonic domain as ``now_s``.  A stable
    L-corner must include the first and last capture times of its stationary
    window.  ``hole_dwell_complete`` means the complete-hole estimator has
    already enforced its three-rim/two-direction geometry and dwell gates.
    """

    now_s: float
    odometry: OdometrySample | None = None
    l_corner_visible: bool = False
    l_corner_stable: bool = False
    l_corner_stationary_frames: int = 0
    l_corner_window_started_at_s: float | None = None
    l_corner_timestamp_s: float | None = None
    l_corner_topology_branch: str | None = None
    hole_visible: bool = False
    hole_visible_timestamp_s: float | None = None
    hole_dwell_complete: bool = False
    hole_window_started_at_s: float | None = None
    hole_timestamp_s: float | None = None
    motion_interlocks_ok: bool = False
    interlock_reason: str = ""
    wheel_stopped: bool = False
    wheel_timestamp_s: float | None = None
    zero_command_acknowledged: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "now_s", _finite(self.now_s, "now_s"))
        object.__setattr__(
            self,
            "l_corner_stationary_frames",
            _integer_at_least(
                self.l_corner_stationary_frames,
                0,
                "l_corner_stationary_frames",
            ),
        )
        for name in (
            "l_corner_window_started_at_s",
            "l_corner_timestamp_s",
            "hole_visible_timestamp_s",
            "hole_window_started_at_s",
            "hole_timestamp_s",
            "wheel_timestamp_s",
        ):
            object.__setattr__(
                self,
                name,
                _optional_timestamp(getattr(self, name), name),
            )
        if self.l_corner_stable and not self.l_corner_visible:
            raise ValueError("stable L-corner must also be visible")
        if self.l_corner_stable and (
            self.l_corner_window_started_at_s is None
            or self.l_corner_timestamp_s is None
        ):
            raise ValueError("stable L-corner requires a timestamped window")
        if (
            self.l_corner_window_started_at_s is not None
            and self.l_corner_timestamp_s is not None
            and self.l_corner_window_started_at_s > self.l_corner_timestamp_s
        ):
            raise ValueError("L-corner window cannot end before it starts")
        if self.l_corner_visible and self.l_corner_timestamp_s is None:
            raise ValueError("visible L-corner requires l_corner_timestamp_s")
        branch = (
            None
            if self.l_corner_topology_branch is None
            else str(self.l_corner_topology_branch).strip()
        )
        object.__setattr__(self, "l_corner_topology_branch", branch or None)
        if self.l_corner_visible and not branch:
            raise ValueError("visible L-corner requires l_corner_topology_branch")
        if self.hole_visible and self.hole_visible_timestamp_s is None:
            raise ValueError("visible complete hole requires hole_visible_timestamp_s")
        if self.hole_dwell_complete and (
            self.hole_window_started_at_s is None or self.hole_timestamp_s is None
        ):
            raise ValueError("complete hole dwell requires a timestamped window")
        if (
            self.hole_window_started_at_s is not None
            and self.hole_timestamp_s is not None
            and self.hole_window_started_at_s > self.hole_timestamp_s
        ):
            raise ValueError("hole dwell window cannot end before it starts")
        if self.wheel_stopped and self.wheel_timestamp_s is None:
            raise ValueError("wheel_stopped requires wheel_timestamp_s")
        object.__setattr__(self, "interlock_reason", str(self.interlock_reason))


@dataclass(frozen=True, slots=True)
class LCornerGateStatus:
    """Temporal gate over raw L-corner evidence from stationary frames only."""

    visible: bool
    stable: bool
    stationary_frames: int
    window_started_at_s: float | None
    window_ended_at_s: float | None
    topology_branch: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class _LCornerGateSample:
    timestamp_s: float
    yaw_rad: float
    plane_height_m: float
    topology_branch: str


class StationaryLCornerGate:
    """Require five fresh, same-branch L observations captured while stopped.

    Corner translation is intentionally not treated as a slot pose or a control
    error.  The runtime supplies ``stationary=False`` for every moving or
    pre-settle frame, which clears the window.  Stability is therefore limited
    to the observable line branch, plane height, and line orientation.
    """

    def __init__(
        self,
        required_frames: int = MIN_STATIONARY_L_CORNER_FRAMES,
        *,
        max_yaw_spread_rad: float = math.radians(2.0),
        max_plane_height_spread_m: float = 0.008,
    ) -> None:
        self.required_frames = _integer_at_least(
            required_frames,
            MIN_STATIONARY_L_CORNER_FRAMES,
            "required_frames",
        )
        self.max_yaw_spread_rad = _positive(
            max_yaw_spread_rad,
            "max_yaw_spread_rad",
        )
        self.max_plane_height_spread_m = _positive(
            max_plane_height_spread_m,
            "max_plane_height_spread_m",
        )
        if self.max_yaw_spread_rad > math.radians(2.0) + 1e-12:
            raise ValueError("max_yaw_spread_rad cannot exceed 2 degrees")
        if self.max_plane_height_spread_m > 0.008 + 1e-12:
            raise ValueError("max_plane_height_spread_m cannot exceed 8 mm")
        self._samples: deque[_LCornerGateSample] = deque(maxlen=self.required_frames)

    def clear(self) -> None:
        self._samples.clear()

    def update(self, observation: Any, *, stationary: bool) -> LCornerGateStatus:
        if not stationary:
            self.clear()
            return self._status(False, "frame_not_stationary")
        strict_valid = bool(getattr(observation, "valid", False))
        acquisition_valid = bool(
            getattr(observation, "forward_acquisition_valid", False)
        )
        if observation is None or not (strict_valid or acquisition_valid):
            self.clear()
            reasons = tuple(
                getattr(observation, "forward_acquisition_rejection_reasons", ())
            ) or tuple(getattr(observation, "rejection_reasons", ()))
            reason = str(reasons[0]) if reasons else "l_corner_invalid"
            return self._status(False, reason)
        try:
            timestamp_s = _finite(getattr(observation, "timestamp_s"), "timestamp_s")
            yaw_value = getattr(observation, "yaw_base_rad", None)
            if yaw_value is None:
                yaw_value = getattr(observation, "forward_acquisition_yaw_base_rad")
            yaw_rad = _finite(yaw_value, "yaw_base_rad")
            plane_height_m = _finite(
                getattr(observation, "plane_height_base_m"),
                "plane_height_base_m",
            )
            branch = str(getattr(observation, "topology_branch")).strip()
        except (AttributeError, TypeError, ValueError):
            self.clear()
            return self._status(False, "l_corner_temporal_fields_invalid")
        if not branch:
            self.clear()
            return self._status(False, "l_corner_branch_missing")
        if self._samples and timestamp_s <= self._samples[-1].timestamp_s:
            self.clear()
            return self._status(False, "l_corner_timestamp_not_increasing")
        if self._samples and branch != self._samples[-1].topology_branch:
            self.clear()
            return self._status(False, "l_corner_branch_flip")
        self._samples.append(
            _LCornerGateSample(timestamp_s, yaw_rad, plane_height_m, branch)
        )
        if len(self._samples) < self.required_frames:
            return self._status(True, "l_corner_window_warmup")
        yaws = tuple(sample.yaw_rad for sample in self._samples)
        yaw_reference = yaws[0]
        yaw_spread = max(
            _angle_difference(value, yaw_reference) for value in yaws
        ) - min(_angle_difference(value, yaw_reference) for value in yaws)
        heights = tuple(sample.plane_height_m for sample in self._samples)
        if yaw_spread > self.max_yaw_spread_rad:
            return self._status(True, "l_corner_yaw_unstable")
        if max(heights) - min(heights) > self.max_plane_height_spread_m:
            return self._status(True, "l_corner_plane_height_unstable")
        return self._status(True, "stationary_l_corner_gate_complete", stable=True)

    def _status(
        self,
        visible: bool,
        reason: str,
        *,
        stable: bool = False,
    ) -> LCornerGateStatus:
        first = self._samples[0] if self._samples else None
        last = self._samples[-1] if self._samples else None
        return LCornerGateStatus(
            visible=bool(visible),
            stable=bool(stable),
            stationary_frames=len(self._samples),
            window_started_at_s=None if first is None else first.timestamp_s,
            window_ended_at_s=None if last is None else last.timestamp_s,
            topology_branch=None if last is None else last.topology_branch,
            reason=str(reason),
        )


@dataclass(frozen=True, slots=True)
class HoleGateStatus:
    valid: bool
    dwell_complete: bool
    stationary_frames: int
    window_started_at_s: float | None
    window_ended_at_s: float | None
    topology_branch: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class _HoleGateSample:
    timestamp_s: float
    center_x_m: float
    center_y_m: float
    yaw_rad: float
    topology_branch: str


class StationaryHoleGate:
    """Dwell gate for the complete metric opening before fine-servo handoff."""

    def __init__(
        self,
        required_frames: int = 5,
        *,
        minimum_duration_s: float = 0.35,
        max_center_spread_m: float = 0.008,
        max_yaw_spread_rad: float = math.radians(2.0),
    ) -> None:
        self.required_frames = _integer_at_least(
            required_frames,
            5,
            "required_frames",
        )
        self.minimum_duration_s = _positive(
            minimum_duration_s,
            "minimum_duration_s",
        )
        self.max_center_spread_m = _positive(
            max_center_spread_m,
            "max_center_spread_m",
        )
        self.max_yaw_spread_rad = _positive(
            max_yaw_spread_rad,
            "max_yaw_spread_rad",
        )
        if self.minimum_duration_s < 0.35:
            raise ValueError("minimum_duration_s cannot be less than 0.35 s")
        if self.max_center_spread_m > 0.008 + 1e-12:
            raise ValueError("max_center_spread_m cannot exceed 8 mm")
        if self.max_yaw_spread_rad > math.radians(2.0) + 1e-12:
            raise ValueError("max_yaw_spread_rad cannot exceed 2 degrees")
        self._samples: deque[_HoleGateSample] = deque()

    def clear(self) -> None:
        self._samples.clear()

    def update(self, observation: Any, *, stationary: bool) -> HoleGateStatus:
        if not stationary:
            self.clear()
            return self._status(False, "frame_not_stationary")
        stack = getattr(observation, "stack", observation)
        if stack is None or not bool(getattr(stack, "valid", False)):
            self.clear()
            reasons = tuple(getattr(stack, "rejection_reasons", ()))
            reason = str(reasons[0]) if reasons else "complete_hole_invalid"
            return self._status(False, reason)
        try:
            timestamp_s = _finite(getattr(stack, "timestamp_s"), "timestamp_s")
            center = tuple(float(value) for value in getattr(stack, "center_base"))
            if len(center) < 2 or not all(math.isfinite(value) for value in center[:2]):
                raise ValueError("center_base must contain finite XY")
            yaw_rad = _finite(getattr(stack, "yaw_base_rad"), "yaw_base_rad")
            branch = str(getattr(stack, "axis_branch")).strip()
        except (AttributeError, TypeError, ValueError):
            self.clear()
            return self._status(False, "complete_hole_temporal_fields_invalid")
        if not branch:
            self.clear()
            return self._status(False, "complete_hole_branch_missing")
        if self._samples and timestamp_s <= self._samples[-1].timestamp_s:
            self.clear()
            return self._status(False, "complete_hole_timestamp_not_increasing")
        if self._samples and branch != self._samples[-1].topology_branch:
            self.clear()
            return self._status(False, "complete_hole_branch_flip")
        self._samples.append(
            _HoleGateSample(timestamp_s, center[0], center[1], yaw_rad, branch)
        )
        while (
            len(self._samples) > self.required_frames
            and self._samples[-1].timestamp_s - self._samples[1].timestamp_s
            >= self.minimum_duration_s
        ):
            self._samples.popleft()
        if len(self._samples) < self.required_frames:
            return self._status(True, "complete_hole_window_warmup")
        xs = tuple(sample.center_x_m for sample in self._samples)
        ys = tuple(sample.center_y_m for sample in self._samples)
        if max(max(xs) - min(xs), max(ys) - min(ys)) > self.max_center_spread_m:
            return self._status(True, "complete_hole_center_unstable")
        reference = self._samples[0].yaw_rad
        yaw_errors = tuple(
            _line_angle_difference(sample.yaw_rad, reference)
            for sample in self._samples
        )
        if max(yaw_errors) - min(yaw_errors) > self.max_yaw_spread_rad:
            return self._status(True, "complete_hole_yaw_unstable")
        duration = self._samples[-1].timestamp_s - self._samples[0].timestamp_s
        if duration < self.minimum_duration_s:
            return self._status(True, "complete_hole_dwell_too_short")
        return self._status(True, "complete_hole_dwell_complete", complete=True)

    def _status(
        self,
        valid: bool,
        reason: str,
        *,
        complete: bool = False,
    ) -> HoleGateStatus:
        first = self._samples[0] if self._samples else None
        last = self._samples[-1] if self._samples else None
        return HoleGateStatus(
            valid=bool(valid),
            dwell_complete=bool(complete),
            stationary_frames=len(self._samples),
            window_started_at_s=None if first is None else first.timestamp_s,
            window_ended_at_s=None if last is None else last.timestamp_s,
            topology_branch=None if last is None else last.topology_branch,
            reason=str(reason),
        )


class AcquisitionState(str, Enum):
    """Forward acquisition states; only ``STEP`` may emit nonzero velocity."""

    OBSERVE = "OBSERVE"
    STEP = "STEP"
    BRAKE = "BRAKE"
    WAIT_STOP = "WAIT_STOP"
    SETTLE = "SETTLE"
    HANDOFF_ZERO = "HANDOFF_ZERO"
    DISABLED_HOLD = "DISABLED_HOLD"
    PERCEPTION_HOLD = "PERCEPTION_HOLD"
    FAULT_HOLD = "FAULT_HOLD"

    @property
    def commands_zero(self) -> bool:
        return self is not AcquisitionState.STEP


@dataclass(frozen=True, slots=True)
class AcquisitionOutput:
    """One pure controller proposal and its accounting/observability fields."""

    state: AcquisitionState
    reason: str
    vx_mps: float
    vy_mps: float
    wz_radps: float
    cumulative_distance_m: float
    observed_forward_travel_m: float
    remaining_budget_m: float
    step_target_m: float
    step_actual_m: float
    lateral_drift_m: float | None
    yaw_drift_rad: float | None
    last_consumed_l_corner_timestamp_s: float | None
    requires_fresh_l_corner: bool
    fine_handoff_requested: bool
    fault_latched: bool

    def __post_init__(self) -> None:
        for name in (
            "vx_mps",
            "vy_mps",
            "wz_radps",
            "cumulative_distance_m",
            "observed_forward_travel_m",
            "remaining_budget_m",
            "step_target_m",
            "step_actual_m",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        for name in ("lateral_drift_m", "yaw_drift_rad"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(value, name))
        object.__setattr__(
            self,
            "last_consumed_l_corner_timestamp_s",
            _optional_timestamp(
                self.last_consumed_l_corner_timestamp_s,
                "last_consumed_l_corner_timestamp_s",
            ),
        )
        if self.vx_mps < -1e-12:
            raise ValueError("acquisition output can never command reverse")
        if self.vx_mps > MAX_ACQUISITION_SPEED_MPS + 1e-12:
            raise ValueError("acquisition output exceeds release speed cap")
        if self.vy_mps != 0.0 or self.wz_radps != 0.0:
            raise ValueError("acquisition output must be strict forward-only")
        if self.state is not AcquisitionState.STEP and self.vx_mps != 0.0:
            raise ValueError("only STEP may emit nonzero velocity")
        if self.cumulative_distance_m < -1e-12:
            raise ValueError("cumulative_distance_m cannot be negative")
        if self.remaining_budget_m < -1e-12:
            raise ValueError("remaining_budget_m cannot be negative")
        object.__setattr__(self, "reason", str(self.reason))

    @property
    def motion_requested(self) -> bool:
        return self.vx_mps > 0.0

    @property
    def is_exact_zero(self) -> bool:
        return self.vx_mps == self.vy_mps == self.wz_radps == 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "reason": self.reason,
            "command": {
                "vx_mps": self.vx_mps,
                "vy_mps": self.vy_mps,
                "wz_radps": self.wz_radps,
            },
            "cumulative_distance_m": self.cumulative_distance_m,
            "observed_forward_travel_m": self.observed_forward_travel_m,
            "remaining_budget_m": self.remaining_budget_m,
            "step_target_m": self.step_target_m,
            "step_actual_m": self.step_actual_m,
            "lateral_drift_m": self.lateral_drift_m,
            "yaw_drift_rad": self.yaw_drift_rad,
            "last_consumed_l_corner_timestamp_s": (
                self.last_consumed_l_corner_timestamp_s
            ),
            "requires_fresh_l_corner": self.requires_fresh_l_corner,
            "fine_handoff_requested": self.fine_handoff_requested,
            "fault_latched": self.fault_latched,
        }


class PalletControlOwner(str, Enum):
    """The sole producer whose mobility proposal may reach the stream."""

    FORWARD_ACQUISITION = "forward_acquisition"
    FINE_SLOT1_SERVO = "fine_slot1_servo"
    SHUTDOWN_HOLD = "shutdown_hold"


class CoarseFineAuthority:
    """One-way authority transfer from coarse acquisition to fine PBVS.

    The coarse owner cannot be restored within a session.  This prevents a
    late partial-L observation from publishing after the complete-hole
    controller has taken over.
    """

    def __init__(self) -> None:
        self._owner = PalletControlOwner.FORWARD_ACQUISITION
        self._coarse_revoked = False

    @property
    def owner(self) -> PalletControlOwner:
        return self._owner

    @property
    def coarse_revoked(self) -> bool:
        return self._coarse_revoked

    def handoff_to_fine(
        self,
        output: AcquisitionOutput,
        *,
        zero_command_acknowledged: bool,
        wheel_stopped: bool,
    ) -> None:
        if self._owner is not PalletControlOwner.FORWARD_ACQUISITION:
            raise RuntimeError(
                f"fine handoff is invalid from controller owner {self._owner.value}"
            )
        if not output.fine_handoff_requested:
            raise RuntimeError("acquisition output did not request fine handoff")
        if not output.is_exact_zero:
            raise RuntimeError("fine handoff requires an exact-zero coarse output")
        if not zero_command_acknowledged or not wheel_stopped:
            raise RuntimeError(
                "fine handoff requires zero acknowledgement and wheel stop"
            )
        self._owner = PalletControlOwner.FINE_SLOT1_SERVO
        self._coarse_revoked = True

    def request_shutdown_hold(self) -> None:
        self._owner = PalletControlOwner.SHUTDOWN_HOLD
        self._coarse_revoked = True

    def assert_publish(self, owner: PalletControlOwner, command: Any) -> None:
        if owner is not self._owner:
            raise RuntimeError(
                f"controller {owner.value} cannot publish while {self._owner.value} owns"
            )
        try:
            vx = float(getattr(command, "vx_mps"))
            vy = float(getattr(command, "vy_mps"))
            wz = float(getattr(command, "wz_radps"))
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError("published mobility command is malformed") from exc
        if not all(math.isfinite(value) for value in (vx, vy, wz)):
            raise RuntimeError("published mobility command is nonfinite")
        if owner is PalletControlOwner.FORWARD_ACQUISITION and (
            vx < 0.0 or vx > MAX_ACQUISITION_SPEED_MPS + 1e-12 or vy != 0.0 or wz != 0.0
        ):
            raise RuntimeError("forward acquisition attempted a non-forward command")
        if owner is PalletControlOwner.SHUTDOWN_HOLD and (
            vx != 0.0 or vy != 0.0 or wz != 0.0
        ):
            raise RuntimeError("shutdown owner permits exact-zero mobility only")


class _AfterStop(str, Enum):
    SETTLE = "SETTLE"
    PERCEPTION_HOLD = "PERCEPTION_HOLD"
    HANDOFF_ZERO = "HANDOFF_ZERO"
    FAULT_HOLD = "FAULT_HOLD"


class ForwardAcquireServo:
    """Stateful, pure stop-and-observe controller for hole acquisition.

    Constructing a new instance starts a new acquisition session.  There is no
    public reset that could accidentally refund consumed distance.  Perception
    holds and re-entry retain both the entry corridor and cumulative committed
    budget.  A target increment is conservatively charged in full when a step
    starts, so an interrupted/retried step cannot reclaim distance.
    """

    def __init__(self, config: AcquisitionConfig) -> None:
        if not isinstance(config, AcquisitionConfig):
            raise TypeError("config must be AcquisitionConfig")
        self.config = config
        self._state = AcquisitionState.OBSERVE
        self._reason = "waiting_for_stationary_l_corner"
        self._last_update_s: float | None = None
        self._acquisition_started_at_s: float | None = None
        self._entry_odometry: OdometrySample | None = None
        self._step_start_odometry: OdometrySample | None = None
        self._step_started_at_s: float | None = None
        self._step_target_m = 0.0
        self._step_actual_m = 0.0
        self._step_progress_mark_m = 0.0
        self._last_progress_at_s: float | None = None
        self._cumulative_distance_m = 0.0
        self._observed_forward_travel_m = 0.0
        self._last_consumed_l_corner_timestamp_s: float | None = None
        self._active_l_corner_branch: str | None = None
        self._settle_started_at_s: float | None = None
        self._post_stop_not_before_s: float | None = None
        self._brake_started_at_s: float | None = None
        self._after_stop = _AfterStop.SETTLE
        self._after_stop_reason = ""
        self._fault_reason: str | None = None

    @property
    def state(self) -> AcquisitionState:
        return self._state

    @property
    def cumulative_distance_m(self) -> float:
        """Conservatively committed forward budget; it never decreases."""

        return self._cumulative_distance_m

    @property
    def remaining_budget_m(self) -> float:
        return max(0.0, self.config.budget_m - self._cumulative_distance_m)

    def update(self, decision: AcquisitionDecision) -> AcquisitionOutput:
        """Advance the controller by one current, immutable evidence sample."""

        if not isinstance(decision, AcquisitionDecision):
            raise TypeError("decision must be AcquisitionDecision")
        now_s = decision.now_s
        if self._last_update_s is not None and now_s < self._last_update_s - 1e-12:
            self._latch_fault("nonmonotonic_controller_time")
        self._last_update_s = now_s

        if self._state is AcquisitionState.FAULT_HOLD:
            return self._output(decision, self._fault_reason or "fault_latched")
        if self.config.budget_m == 0.0:
            self._state = AcquisitionState.DISABLED_HOLD
            self._reason = "acquisition_disabled_zero_budget"
            return self._output(decision, self._reason)
        if self._state is AcquisitionState.HANDOFF_ZERO:
            ready = self._stop_verified(decision) and self._fresh_hole(decision)
            reason = "fine_handoff_ready_at_zero" if ready else "fine_handoff_zero_wait"
            return self._output(decision, reason, fine_handoff_requested=ready)

        hole_ready = self._fresh_hole(decision)
        raw_hole_ready = self._fresh_raw_hole(decision)
        if hole_ready:
            if self._state is AcquisitionState.STEP:
                self._begin_brake(
                    now_s,
                    _AfterStop.SETTLE,
                    "hole_seen_braking_for_post_stop_revalidation",
                )
                return self._output(decision, self._reason)
            if self._state not in {
                AcquisitionState.BRAKE,
                AcquisitionState.WAIT_STOP,
            }:
                return self._request_handoff_or_wait(decision)
        if (
            self.config.mode == ACQUISITION_MODE_CONTINUOUS_FORWARD
            and raw_hole_ready
            and self._state is AcquisitionState.STEP
        ):
            self._begin_brake(
                now_s,
                _AfterStop.SETTLE,
                "raw_hole_seen_braking_for_stationary_dwell",
            )
            return self._output(decision, self._reason)

        if self._state is AcquisitionState.BRAKE:
            self._state = AcquisitionState.WAIT_STOP
            self._reason = "waiting_for_zero_ack_and_wheel_stop"
            return self._update_wait_stop(decision)
        if self._state is AcquisitionState.WAIT_STOP:
            return self._update_wait_stop(decision)

        if self._active_timeout_expired(now_s):
            if self._state is AcquisitionState.STEP:
                self._begin_brake(
                    now_s,
                    _AfterStop.FAULT_HOLD,
                    "acquisition_timeout",
                )
                return self._output(decision, self._reason)
            self._latch_fault("hole_not_acquired_within_time_budget")
            return self._output(decision, self._fault_reason or "acquisition_timeout")

        if self._state is AcquisitionState.STEP:
            return self._update_step(decision)

        if self._state is AcquisitionState.DISABLED_HOLD:
            # A positive budget cannot appear without constructing a new,
            # validated config/controller session.
            return self._output(decision, "acquisition_disabled_zero_budget")

        if self._state is AcquisitionState.SETTLE:
            return self._update_settle(decision)
        if self._state is AcquisitionState.PERCEPTION_HOLD:
            return self._update_perception_hold(decision)
        return self._update_observe(decision)

    def _update_observe(self, decision: AcquisitionDecision) -> AcquisitionOutput:
        if not decision.motion_interlocks_ok:
            self._state = AcquisitionState.PERCEPTION_HOLD
            self._reason = decision.interlock_reason or "motion_interlock_failed"
            return self._output(decision, self._reason)
        odometry_reason = self._odometry_reason(decision)
        if odometry_reason is not None:
            self._state = AcquisitionState.PERCEPTION_HOLD
            self._reason = odometry_reason
            return self._output(decision, self._reason)
        if not self._stop_verified(decision):
            self._after_stop = _AfterStop.PERCEPTION_HOLD
            self._after_stop_reason = "initial_zero_or_wheel_stop_unverified"
            self._brake_started_at_s = decision.now_s
            self._state = AcquisitionState.WAIT_STOP
            return self._output(decision, self._after_stop_reason)

        gate_reason = self._stable_l_corner_reason(decision)
        if gate_reason is not None:
            self._reason = gate_reason
            return self._output(decision, self._reason)

        odometry = decision.odometry
        assert odometry is not None
        if self._entry_odometry is None:
            self._entry_odometry = odometry
        drift_reason = self._drift_reason(odometry)
        if drift_reason is not None:
            self._latch_fault(drift_reason)
            return self._output(decision, self._fault_reason or drift_reason)

        target = self._next_target_m()
        if target is None:
            self._latch_fault("insufficient_stopping_allowance_in_budget")
            return self._output(decision, self._fault_reason or "budget_exhausted")

        if self.config.mode == ACQUISITION_MODE_STOP_STEP:
            # Charge the full target at start.  A perception hold or interrupted
            # step cannot refund it and therefore cannot exceed the session budget.
            self._cumulative_distance_m = min(
                self.config.budget_m,
                self._cumulative_distance_m + target,
            )
        self._step_start_odometry = odometry
        self._step_started_at_s = decision.now_s
        self._step_target_m = target
        self._step_actual_m = 0.0
        self._step_progress_mark_m = 0.0
        self._last_progress_at_s = decision.now_s
        self._last_consumed_l_corner_timestamp_s = decision.l_corner_timestamp_s
        self._active_l_corner_branch = decision.l_corner_topology_branch
        if self._acquisition_started_at_s is None:
            self._acquisition_started_at_s = decision.now_s
        self._state = AcquisitionState.STEP
        self._reason = (
            "fresh_stationary_l_corner_authorized_continuous_forward"
            if self.config.mode == ACQUISITION_MODE_CONTINUOUS_FORWARD
            else "fresh_stationary_l_corner_authorized_step"
        )
        return self._output(decision, self._reason, vx_mps=self.config.speed_mps)

    def _update_step(self, decision: AcquisitionDecision) -> AcquisitionOutput:
        now_s = decision.now_s
        if not decision.motion_interlocks_ok:
            self._begin_brake(
                now_s,
                _AfterStop.PERCEPTION_HOLD,
                decision.interlock_reason or "motion_interlock_failed_during_step",
            )
            return self._output(decision, self._reason)

        # The preceding stationary five-frame window authorizes exactly this
        # bounded step.  Moving frames cannot accumulate another authorization
        # window, but their raw L-corner is still a fail-closed visibility
        # predicate.  Fresh stationary geometry is required again after stop.
        if not self._fresh_visible_l_corner(decision):
            self._begin_brake(
                now_s,
                _AfterStop.PERCEPTION_HOLD,
                "l_corner_lost_during_step",
            )
            return self._output(decision, self._reason)
        if (
            self._active_l_corner_branch is None
            or decision.l_corner_topology_branch != self._active_l_corner_branch
        ):
            self._begin_brake(
                now_s,
                _AfterStop.FAULT_HOLD,
                "l_corner_branch_changed_during_step",
            )
            return self._output(decision, self._reason)

        odometry_reason = self._odometry_reason(decision)
        if odometry_reason is not None:
            self._begin_brake(now_s, _AfterStop.FAULT_HOLD, odometry_reason)
            return self._output(decision, self._reason)
        odometry = decision.odometry
        assert odometry is not None
        drift_reason = self._drift_reason(odometry)
        if drift_reason is not None:
            self._begin_brake(now_s, _AfterStop.FAULT_HOLD, drift_reason)
            return self._output(decision, self._reason)

        progress = self._step_forward_progress(odometry)
        if progress < -self.config.overshoot_tolerance_m:
            self._begin_brake(
                now_s,
                _AfterStop.FAULT_HOLD,
                "unexpected_reverse_odometry_during_step",
            )
            return self._output(decision, self._reason)
        self._record_step_progress(max(0.0, progress))
        if self._observed_forward_travel_m > self.config.budget_m + 1e-12:
            self._begin_brake(
                now_s,
                _AfterStop.FAULT_HOLD,
                "acquisition_budget_overrun_during_step",
            )
            return self._output(decision, self._reason)
        if progress > self._step_target_m + self.config.braking_allowance_m:
            self._begin_brake(
                now_s,
                _AfterStop.FAULT_HOLD,
                "step_braking_allowance_exceeded",
            )
            return self._output(decision, self._reason)
        if self._step_actual_m >= self._step_target_m - self.config.target_tolerance_m:
            after_stop = (
                _AfterStop.FAULT_HOLD
                if self.config.mode == ACQUISITION_MODE_CONTINUOUS_FORWARD
                else _AfterStop.SETTLE
            )
            reason = (
                "continuous_forward_budget_exhausted_before_hole"
                if self.config.mode == ACQUISITION_MODE_CONTINUOUS_FORWARD
                else "step_target_reached"
            )
            self._begin_brake(now_s, after_stop, reason)
            return self._output(decision, self._reason)

        assert self._step_started_at_s is not None
        if (
            self.config.mode == ACQUISITION_MODE_STOP_STEP
            and now_s - self._step_started_at_s > self.config.step_timeout_s
        ):
            self._begin_brake(now_s, _AfterStop.FAULT_HOLD, "step_timeout")
            return self._output(decision, self._reason)
        assert self._last_progress_at_s is not None
        if now_s - self._last_progress_at_s > self.config.no_progress_timeout_s:
            self._begin_brake(now_s, _AfterStop.FAULT_HOLD, "step_no_progress")
            return self._output(decision, self._reason)
        active_reason = (
            "continuous_forward_cruise_active"
            if self.config.mode == ACQUISITION_MODE_CONTINUOUS_FORWARD
            else "forward_step_active"
        )
        return self._output(decision, active_reason, vx_mps=self.config.speed_mps)

    def _update_wait_stop(self, decision: AcquisitionDecision) -> AcquisitionOutput:
        if self._brake_started_at_s is None:
            self._brake_started_at_s = decision.now_s
        monitoring_reason = self._zeroing_odometry_reason(decision)
        if monitoring_reason is not None:
            self._after_stop = _AfterStop.FAULT_HOLD
            self._after_stop_reason = monitoring_reason
            self._reason = monitoring_reason
        if self._stop_verified(decision):
            if self._after_stop is _AfterStop.HANDOFF_ZERO:
                if self._fresh_hole(decision):
                    self._state = AcquisitionState.HANDOFF_ZERO
                    return self._output(
                        decision,
                        "fine_handoff_ready_at_zero",
                        fine_handoff_requested=True,
                    )
                self._state = AcquisitionState.PERCEPTION_HOLD
                self._reason = "hole_lost_before_zero_handoff"
                return self._output(decision, self._reason)
            if self._after_stop is _AfterStop.FAULT_HOLD:
                self._latch_fault(self._after_stop_reason or "fault_after_stop")
                return self._output(decision, self._fault_reason or "fault_after_stop")
            if self._after_stop is _AfterStop.PERCEPTION_HOLD:
                self._state = AcquisitionState.PERCEPTION_HOLD
                self._reason = self._after_stop_reason or "perception_hold_after_stop"
                return self._output(decision, self._reason)
            self._state = AcquisitionState.SETTLE
            self._settle_started_at_s = decision.now_s
            self._post_stop_not_before_s = decision.now_s
            self._reason = "wheel_stop_confirmed_collecting_new_frames"
            return self._output(decision, self._reason)

        if decision.now_s - self._brake_started_at_s > self.config.brake_timeout_s:
            self._latch_fault("wheel_stop_or_zero_ack_timeout")
            return self._output(decision, self._fault_reason or "wheel_stop_timeout")
        return self._output(
            decision,
            self._after_stop_reason
            if self._after_stop is _AfterStop.FAULT_HOLD
            else "waiting_for_zero_ack_and_wheel_stop",
        )

    def _update_settle(self, decision: AcquisitionDecision) -> AcquisitionOutput:
        if not decision.motion_interlocks_ok:
            self._state = AcquisitionState.PERCEPTION_HOLD
            self._reason = (
                decision.interlock_reason or "motion_interlock_failed_during_settle"
            )
            return self._output(decision, self._reason)
        if not self._stop_verified(decision):
            self._after_stop = _AfterStop.PERCEPTION_HOLD
            self._after_stop_reason = "wheel_motion_detected_during_settle"
            self._brake_started_at_s = decision.now_s
            self._state = AcquisitionState.WAIT_STOP
            return self._output(decision, self._after_stop_reason)
        odometry_reason = self._odometry_reason(decision)
        if odometry_reason is not None:
            self._state = AcquisitionState.PERCEPTION_HOLD
            self._reason = odometry_reason
            return self._output(decision, self._reason)
        odometry = decision.odometry
        assert odometry is not None
        drift_reason = self._drift_reason(odometry)
        if drift_reason is not None:
            self._latch_fault(drift_reason)
            return self._output(decision, self._fault_reason or drift_reason)

        assert self._settle_started_at_s is not None
        if decision.now_s - self._settle_started_at_s < self.config.settle_duration_s:
            return self._output(decision, "stationary_settle_dwell")
        gate_reason = self._stable_l_corner_reason(
            decision,
            not_before_s=self._post_stop_not_before_s,
        )
        if gate_reason is not None:
            return self._output(decision, gate_reason)
        if self.remaining_budget_m <= 1e-12:
            self._latch_fault("hole_not_acquired_within_budget")
            return self._output(decision, self._fault_reason or "budget_exhausted")
        self._state = AcquisitionState.OBSERVE
        self._reason = "fresh_post_stop_l_corner_ready"
        return self._output(decision, self._reason)

    def _update_perception_hold(
        self,
        decision: AcquisitionDecision,
    ) -> AcquisitionOutput:
        if not decision.motion_interlocks_ok:
            return self._output(
                decision,
                decision.interlock_reason or "motion_interlock_failed",
            )
        if not self._stop_verified(decision):
            self._after_stop = _AfterStop.PERCEPTION_HOLD
            self._after_stop_reason = "hold_waiting_for_zero_ack_and_wheel_stop"
            self._brake_started_at_s = decision.now_s
            self._state = AcquisitionState.WAIT_STOP
            return self._output(decision, self._after_stop_reason)
        odometry_reason = self._odometry_reason(decision)
        if odometry_reason is not None:
            return self._output(decision, odometry_reason)
        odometry = decision.odometry
        assert odometry is not None
        drift_reason = self._drift_reason(odometry)
        if drift_reason is not None:
            self._latch_fault(drift_reason)
            return self._output(decision, self._fault_reason or drift_reason)
        gate_reason = self._stable_l_corner_reason(decision)
        if gate_reason is not None:
            return self._output(decision, gate_reason)
        if self.remaining_budget_m <= 1e-12:
            self._latch_fault("hole_not_acquired_within_budget")
            return self._output(decision, self._fault_reason or "budget_exhausted")
        self._state = AcquisitionState.OBSERVE
        self._reason = "fresh_l_corner_reacquired"
        return self._output(decision, self._reason)

    def _request_handoff_or_wait(
        self,
        decision: AcquisitionDecision,
    ) -> AcquisitionOutput:
        if self._stop_verified(decision):
            self._state = AcquisitionState.HANDOFF_ZERO
            self._reason = "fine_handoff_ready_at_zero"
            return self._output(decision, self._reason, fine_handoff_requested=True)
        self._state = AcquisitionState.WAIT_STOP
        self._after_stop = _AfterStop.HANDOFF_ZERO
        self._after_stop_reason = "hole_dwell_complete_waiting_for_zero"
        self._brake_started_at_s = decision.now_s
        return self._output(decision, self._after_stop_reason)

    def _begin_brake(
        self,
        now_s: float,
        after_stop: _AfterStop,
        reason: str,
    ) -> None:
        self._state = AcquisitionState.BRAKE
        self._reason = reason
        self._after_stop = after_stop
        self._after_stop_reason = reason
        self._brake_started_at_s = now_s

    def _latch_fault(self, reason: str) -> None:
        self._state = AcquisitionState.FAULT_HOLD
        self._fault_reason = str(reason)
        self._reason = self._fault_reason

    def _next_target_m(self) -> float | None:
        if self.config.mode == ACQUISITION_MODE_CONTINUOUS_FORWARD:
            target = self.remaining_budget_m - self.config.braking_allowance_m
            return target if target > self.config.target_tolerance_m else None

        remaining = self.remaining_budget_m
        required_budget_m = self.config.step_m + self.config.braking_allowance_m
        # A shortened tail step is below the useful resolution of this 10 Hz
        # stop-and-observe loop, while its physical stopping distance is not.
        if remaining + 1e-12 < required_budget_m:
            return None
        return self.config.step_m

    def _fresh_hole(self, decision: AcquisitionDecision) -> bool:
        return bool(
            decision.hole_dwell_complete
            and self._timestamp_fresh(
                decision.hole_timestamp_s,
                decision.now_s,
                self.config.camera_freshness_s,
            )
            and (
                self._post_stop_not_before_s is None
                or (
                    decision.hole_window_started_at_s is not None
                    and decision.hole_window_started_at_s
                    >= self._post_stop_not_before_s - 1e-12
                )
            )
        )

    def _fresh_raw_hole(self, decision: AcquisitionDecision) -> bool:
        return bool(
            decision.hole_visible
            and self._timestamp_fresh(
                decision.hole_visible_timestamp_s,
                decision.now_s,
                self.config.camera_freshness_s,
            )
        )

    def _fresh_visible_l_corner(self, decision: AcquisitionDecision) -> bool:
        return decision.l_corner_visible and self._timestamp_fresh(
            decision.l_corner_timestamp_s,
            decision.now_s,
            self.config.camera_freshness_s,
        )

    def _stable_l_corner_reason(
        self,
        decision: AcquisitionDecision,
        *,
        not_before_s: float | None = None,
    ) -> str | None:
        if not decision.l_corner_visible:
            return "l_corner_not_visible"
        if not decision.l_corner_stable:
            return "l_corner_not_stable"
        if not decision.l_corner_topology_branch:
            return "l_corner_branch_missing"
        if decision.l_corner_stationary_frames < self.config.stationary_frames:
            return "l_corner_stationary_window_too_short"
        if not self._timestamp_fresh(
            decision.l_corner_timestamp_s,
            decision.now_s,
            self.config.camera_freshness_s,
        ):
            return "l_corner_stale"
        if decision.l_corner_window_started_at_s is None:
            return "l_corner_window_provenance_missing"
        if (
            not_before_s is not None
            and decision.l_corner_window_started_at_s < not_before_s - 1e-12
        ):
            return "l_corner_window_contains_pre_stop_frames"
        if (
            self._last_consumed_l_corner_timestamp_s is not None
            and decision.l_corner_timestamp_s is not None
            and decision.l_corner_timestamp_s
            <= self._last_consumed_l_corner_timestamp_s + 1e-12
        ):
            return "l_corner_gate_already_consumed"
        return None

    def _odometry_reason(self, decision: AcquisitionDecision) -> str | None:
        sample = decision.odometry
        if sample is None:
            return "odometry_unavailable"
        if not all(
            math.isfinite(value)
            for value in (sample.timestamp_s, sample.x_m, sample.y_m, sample.yaw_rad)
        ):
            return "odometry_nonfinite"
        if not self._timestamp_fresh(
            sample.timestamp_s,
            decision.now_s,
            self.config.odometry_freshness_s,
        ):
            return "odometry_stale_or_future"
        return None

    def _stop_verified(self, decision: AcquisitionDecision) -> bool:
        return (
            decision.zero_command_acknowledged
            and decision.wheel_stopped
            and self._timestamp_fresh(
                decision.wheel_timestamp_s,
                decision.now_s,
                self.config.wheel_freshness_s,
            )
        )

    @staticmethod
    def _timestamp_fresh(
        timestamp_s: float | None,
        now_s: float,
        limit_s: float,
    ) -> bool:
        if timestamp_s is None:
            return False
        age_s = now_s - timestamp_s
        return math.isfinite(age_s) and -1e-9 <= age_s <= limit_s

    def _entry_displacement(self, sample: OdometrySample) -> tuple[float, float, float]:
        if self._entry_odometry is None:
            return (0.0, 0.0, 0.0)
        reference = self._entry_odometry
        dx = sample.x_m - reference.x_m
        dy = sample.y_m - reference.y_m
        cosine = math.cos(reference.yaw_rad)
        sine = math.sin(reference.yaw_rad)
        forward = cosine * dx + sine * dy
        lateral = -sine * dx + cosine * dy
        yaw = _angle_difference(sample.yaw_rad, reference.yaw_rad)
        return (forward, lateral, yaw)

    def _drift_reason(self, sample: OdometrySample) -> str | None:
        _, lateral, yaw = self._entry_displacement(sample)
        if abs(lateral) > self.config.lateral_drift_limit_m:
            return "lateral_odometry_drift_limit"
        if abs(yaw) > self.config.yaw_drift_limit_rad:
            return "yaw_odometry_drift_limit"
        return None

    def _step_forward_progress(self, sample: OdometrySample) -> float:
        if self._step_start_odometry is None or self._entry_odometry is None:
            return 0.0
        dx = sample.x_m - self._step_start_odometry.x_m
        dy = sample.y_m - self._step_start_odometry.y_m
        cosine = math.cos(self._entry_odometry.yaw_rad)
        sine = math.sin(self._entry_odometry.yaw_rad)
        return cosine * dx + sine * dy

    def _record_step_progress(self, progress_m: float) -> None:
        previous = self._step_actual_m
        self._step_actual_m = max(previous, float(progress_m))
        self._observed_forward_travel_m += self._step_actual_m - previous
        # A step target is charged up front, while larger measured travel
        # (including coasting under a zero command) replaces that estimate.
        self._cumulative_distance_m = max(
            self._cumulative_distance_m,
            self._observed_forward_travel_m,
        )
        if (
            self._step_actual_m - self._step_progress_mark_m
            >= self.config.no_progress_min_m
        ):
            self._step_progress_mark_m = self._step_actual_m
            self._last_progress_at_s = self._last_update_s

    def _zeroing_odometry_reason(
        self,
        decision: AcquisitionDecision,
    ) -> str | None:
        """Track physical step travel until zero acknowledgement and wheel stop."""

        if self._step_start_odometry is None:
            return None
        odometry_reason = self._odometry_reason(decision)
        if odometry_reason is not None:
            return odometry_reason
        odometry = decision.odometry
        assert odometry is not None
        drift_reason = self._drift_reason(odometry)
        if drift_reason is not None:
            return drift_reason
        progress = self._step_forward_progress(odometry)
        if progress < -self.config.overshoot_tolerance_m:
            return "unexpected_reverse_odometry_while_stopping"
        self._record_step_progress(max(0.0, progress))
        if self._observed_forward_travel_m > self.config.budget_m + 1e-12:
            return "acquisition_budget_overrun_while_stopping"
        if progress > self._step_target_m + self.config.braking_allowance_m:
            return "step_braking_allowance_exceeded_while_stopping"
        return None

    def _active_timeout_expired(self, now_s: float) -> bool:
        return (
            self._acquisition_started_at_s is not None
            and now_s - self._acquisition_started_at_s
            > self.config.acquisition_timeout_s
        )

    def _drift_values(
        self,
        decision: AcquisitionDecision,
    ) -> tuple[float | None, float | None]:
        if decision.odometry is None or self._entry_odometry is None:
            return (None, None)
        if not all(
            math.isfinite(value)
            for value in (
                decision.odometry.x_m,
                decision.odometry.y_m,
                decision.odometry.yaw_rad,
            )
        ):
            return (None, None)
        _, lateral, yaw = self._entry_displacement(decision.odometry)
        return (lateral, yaw)

    def _output(
        self,
        decision: AcquisitionDecision,
        reason: str,
        *,
        vx_mps: float = 0.0,
        fine_handoff_requested: bool = False,
    ) -> AcquisitionOutput:
        # Config validation plus this final clamp make a negative/reverse or
        # above-cap proposal structurally unreachable.
        command_vx = min(MAX_ACQUISITION_SPEED_MPS, max(0.0, float(vx_mps)))
        if self.config.budget_m == 0.0:
            command_vx = 0.0
        if self._state is not AcquisitionState.STEP:
            command_vx = 0.0
        lateral, yaw = self._drift_values(decision)
        return AcquisitionOutput(
            state=self._state,
            reason=str(reason),
            vx_mps=command_vx,
            vy_mps=0.0,
            wz_radps=0.0,
            cumulative_distance_m=self._cumulative_distance_m,
            observed_forward_travel_m=self._observed_forward_travel_m,
            remaining_budget_m=self.remaining_budget_m,
            step_target_m=self._step_target_m,
            step_actual_m=self._step_actual_m,
            lateral_drift_m=lateral,
            yaw_drift_rad=yaw,
            last_consumed_l_corner_timestamp_s=(
                self._last_consumed_l_corner_timestamp_s
            ),
            requires_fresh_l_corner=self._state
            in {
                AcquisitionState.OBSERVE,
                AcquisitionState.SETTLE,
                AcquisitionState.PERCEPTION_HOLD,
            },
            fine_handoff_requested=bool(
                fine_handoff_requested and self.config.budget_m > 0.0
            ),
            fault_latched=self._state is AcquisitionState.FAULT_HOLD,
        )


__all__ = [
    "ABSOLUTE_ACQUISITION_BUDGET_CAP_M",
    "ACQUISITION_MODE_CONTINUOUS_FORWARD",
    "ACQUISITION_MODE_STOP_STEP",
    "AcquisitionConfig",
    "AcquisitionDecision",
    "AcquisitionOutput",
    "AcquisitionState",
    "CoarseFineAuthority",
    "ForwardAcquireServo",
    "HoleGateStatus",
    "LCornerGateStatus",
    "MAX_ACQUISITION_SPEED_MPS",
    "MAX_ACQUISITION_STEP_M",
    "MIN_STATIONARY_L_CORNER_FRAMES",
    "OdometrySample",
    "PalletControlOwner",
    "RELEASE_ACQUISITION_BUDGET_CAP_M",
    "StationaryLCornerGate",
    "StationaryHoleGate",
]
