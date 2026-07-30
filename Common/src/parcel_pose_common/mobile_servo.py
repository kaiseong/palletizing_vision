"""Pure mobile-base visual servo logic and an opt-in RB-Y1 stream adapter.

The controller consumes parcel centres expressed in the corrected RB-Y1 base
frame.  A positive centre error therefore commands positive base velocity:
moving the base in that direction reduces the next relative parcel error.
Yaw uses the same current-minus-target convention modulo 180 degrees: a
positive relative line-yaw error commands positive RB-Y1 ``wz``, which reduces
the next fixed-object relative yaw.

No robot motion is possible merely by importing this module or constructing an
adapter.  :class:`RBY1MobilityStream` requires both ``execute=True`` and an
explicit :meth:`~RBY1MobilityStream.open` call before it will create a command
stream.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import importlib
import math
import threading
import time
from typing import Any

import numpy as np

from .angles import line_angle_difference_rad, normalize_line_angle_rad


MAX_ALLOWED_LINEAR_SPEED_MPS = 0.08
MAX_ALLOWED_ANGULAR_SPEED_RADPS = 0.10


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _xy(values: tuple[float, float], name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return (_finite(values[0], name), _finite(values[1], name))


class ServoState(str, Enum):
    """Observable states of the mobile visual-servo controller."""

    IDLE = "idle"
    ACQUIRING = "acquiring"
    TRACKING = "tracking"
    HOLDING = "holding"
    POSE_LOST = "pose_lost"
    ARRIVED = "arrived"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class ServoConfig:
    """Safety and convergence parameters for parcel SE(2) servoing.

    The current packaged grasp posture is valid only when the parcel long axis is
    parallel to base +Y.  Since a rectangle axis is an unoriented line,
    ``+90`` and ``-90`` degrees represent the same target.
    """

    target_xy_m: tuple[float, float] = (0.740, 0.0)
    proportional_gain_per_s: float = 1.0
    max_linear_speed_mps: float = MAX_ALLOWED_LINEAR_SPEED_MPS
    max_linear_acceleration_mps2: float = 0.15
    filter_window: int = 3
    jump_threshold_m: float = 0.030
    jump_reseed_frames: int = 3
    stale_after_s: float = 0.30
    arrival_inner_m: float = 0.010
    arrival_outer_m: float = 0.015
    # Operator-requested 2026-07-30: box-pick arrival needs three contiguous
    # frames.  The 0.35 s dwell and the zero-command settle still apply, so a
    # fast camera cannot shortcut arrival on frame count alone.
    arrival_min_frames: int = 3
    arrival_min_duration_s: float = 0.35
    lost_abort_after_s: float = 2.0
    timeout_s: float = 30.0
    # Appended after the original XY configuration fields so legacy positional
    # construction cannot silently reinterpret an XY gain as a yaw target.
    target_long_axis_yaw_rad: float = math.pi / 2.0
    yaw_proportional_gain_per_s: float = 1.0
    max_angular_speed_radps: float = MAX_ALLOWED_ANGULAR_SPEED_RADPS
    max_angular_acceleration_radps2: float = 0.20
    yaw_jump_threshold_rad: float = math.radians(15.0)
    arrival_yaw_inner_rad: float = math.radians(3.0)
    arrival_yaw_outer_rad: float = math.radians(5.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_xy_m", _xy(self.target_xy_m, "target_xy_m"))
        target_yaw = _finite(
            self.target_long_axis_yaw_rad,
            "target_long_axis_yaw_rad",
        )
        object.__setattr__(
            self,
            "target_long_axis_yaw_rad",
            normalize_line_angle_rad(target_yaw),
        )
        for name in (
            "proportional_gain_per_s",
            "yaw_proportional_gain_per_s",
            "max_linear_speed_mps",
            "max_angular_speed_radps",
            "max_linear_acceleration_mps2",
            "max_angular_acceleration_radps2",
            "jump_threshold_m",
            "yaw_jump_threshold_rad",
            "stale_after_s",
            "arrival_inner_m",
            "arrival_outer_m",
            "arrival_yaw_inner_rad",
            "arrival_yaw_outer_rad",
            "arrival_min_duration_s",
            "lost_abort_after_s",
            "timeout_s",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.max_linear_speed_mps > MAX_ALLOWED_LINEAR_SPEED_MPS:
            raise ValueError(
                "max_linear_speed_mps cannot exceed the 0.08 m/s safety limit"
            )
        if self.max_angular_speed_radps > MAX_ALLOWED_ANGULAR_SPEED_RADPS:
            raise ValueError(
                "max_angular_speed_radps cannot exceed the 0.10 rad/s safety limit"
            )
        if self.filter_window != 3:
            raise ValueError("filter_window must be 3 for the robust median filter")
        if self.jump_reseed_frames < 2:
            raise ValueError("jump_reseed_frames must be at least 2")
        if self.arrival_min_frames < 1:
            raise ValueError("arrival_min_frames must be positive")
        if self.arrival_outer_m <= self.arrival_inner_m:
            raise ValueError("arrival_outer_m must be greater than arrival_inner_m")
        if self.arrival_yaw_outer_rad <= self.arrival_yaw_inner_rad:
            raise ValueError(
                "arrival_yaw_outer_rad must be greater than arrival_yaw_inner_rad"
            )
        if self.arrival_yaw_outer_rad >= math.pi / 2.0:
            raise ValueError("arrival_yaw_outer_rad must be less than pi/2")
        if self.yaw_jump_threshold_rad >= math.pi / 2.0:
            raise ValueError("yaw_jump_threshold_rad must be less than pi/2")


@dataclass(frozen=True, slots=True)
class PoseMeasurement:
    """One corrected base-frame parcel centre and long-axis observation."""

    xy_m: tuple[float, float] | None
    timestamp_s: float
    valid: bool = True
    # Kept after the original ``valid`` field for positional API compatibility.
    long_axis_yaw_rad: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", _finite(self.timestamp_s, "timestamp_s"))
        if self.valid:
            if self.xy_m is None:
                raise ValueError("a valid pose measurement requires xy_m")
            if self.long_axis_yaw_rad is None:
                raise ValueError(
                    "a valid pose measurement requires long_axis_yaw_rad"
                )
            object.__setattr__(self, "xy_m", _xy(self.xy_m, "xy_m"))
            yaw = _finite(self.long_axis_yaw_rad, "long_axis_yaw_rad")
            object.__setattr__(
                self,
                "long_axis_yaw_rad",
                normalize_line_angle_rad(yaw),
            )
        elif self.xy_m is not None:
            object.__setattr__(self, "xy_m", _xy(self.xy_m, "xy_m"))
        if not self.valid and self.long_axis_yaw_rad is not None:
            yaw = _finite(self.long_axis_yaw_rad, "long_axis_yaw_rad")
            object.__setattr__(
                self,
                "long_axis_yaw_rad",
                normalize_line_angle_rad(yaw),
            )

    @classmethod
    def invalid(cls, timestamp_s: float) -> "PoseMeasurement":
        return cls(
            xy_m=None,
            timestamp_s=timestamp_s,
            valid=False,
            long_axis_yaw_rad=None,
        )


@dataclass(frozen=True, slots=True)
class VelocityCommand:
    """Bounded planar RB-Y1 translation and yaw velocity command."""

    vx_mps: float = 0.0
    vy_mps: float = 0.0
    wz_radps: float = 0.0

    def __post_init__(self) -> None:
        for name in ("vx_mps", "vy_mps", "wz_radps"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))

    @property
    def linear_norm_mps(self) -> float:
        return float(np.hypot(self.vx_mps, self.vy_mps))

    @property
    def is_zero(self) -> bool:
        return self.linear_norm_mps <= 1e-12 and abs(self.wz_radps) <= 1e-12


ZERO_VELOCITY = VelocityCommand()


@dataclass(frozen=True, slots=True)
class ServoDecision:
    """One controller result suitable for display, logging, and streaming."""

    state: ServoState
    command: VelocityCommand
    filtered_xy_m: tuple[float, float] | None
    error_xy_m: tuple[float, float] | None
    measurement_accepted: bool
    handoff_ready: bool
    reason: str
    filtered_long_axis_yaw_rad: float | None = None
    yaw_error_rad: float | None = None


class MobileVisualServo:
    """Deterministic parcel-centre and 180-symmetric-yaw servo state machine.

    Calling :meth:`step` while idle cannot produce motion.  Call :meth:`start`
    explicitly, then provide monotonically increasing ``now_s`` values.
    """

    def __init__(self, config: ServoConfig | None = None) -> None:
        self.config = config or ServoConfig()
        self.state = ServoState.IDLE
        self._samples: deque[np.ndarray] = deque(maxlen=self.config.filter_window)
        self._jump_candidates: deque[np.ndarray] = deque(
            maxlen=self.config.jump_reseed_frames
        )
        self._last_raw: np.ndarray | None = None
        self._last_command = ZERO_VELOCITY
        self._started_at_s: float | None = None
        self._last_update_s: float | None = None
        self._hold_started_at_s: float | None = None
        self._hold_frames = 0
        self._lost_started_at_s: float | None = None
        self._handoff_emitted = False
        self._abort_reason = ""

    def start(self, now_s: float) -> ServoDecision:
        """Arm the pure controller; this method itself sends no robot command."""

        now = _finite(now_s, "now_s")
        self._samples.clear()
        self._jump_candidates.clear()
        self._last_raw = None
        self._last_command = ZERO_VELOCITY
        self._started_at_s = now
        self._last_update_s = now
        self._clear_hold()
        self._lost_started_at_s = None
        self._handoff_emitted = False
        self._abort_reason = ""
        self.state = ServoState.ACQUIRING
        return self._decision(reason="started")

    def abort(self, reason: str, now_s: float) -> ServoDecision:
        """Enter the terminal aborted state and command an immediate stop."""

        now = self._check_time(now_s)
        message = str(reason).strip() or "aborted"
        self.state = ServoState.ABORTED
        self._abort_reason = message
        self._last_command = ZERO_VELOCITY
        self._last_update_s = now
        self._clear_hold()
        return self._decision(reason=message)

    def step(
        self,
        measurement: PoseMeasurement | None,
        *,
        now_s: float,
    ) -> ServoDecision:
        """Advance the controller by one vision update.

        Missing, invalid, or stale poses bypass the slew limiter and request an
        immediate zero command.  The arrival handoff flag is a rising-edge
        event: it is true exactly once per :meth:`start` call.
        """

        now = self._check_time(now_s)
        if self.state is ServoState.IDLE:
            self._last_update_s = now
            return self._decision(reason="inactive")
        if self.state is ServoState.ARRIVED:
            self._last_update_s = now
            return self._decision(reason="arrived")
        if self.state is ServoState.ABORTED:
            self._last_update_s = now
            return self._decision(reason=self._abort_reason or "aborted")

        assert self._started_at_s is not None
        if now - self._started_at_s >= self.config.timeout_s:
            return self.abort("timeout", now)

        invalid_reason = self._invalid_reason(measurement, now)
        if invalid_reason is not None:
            return self._pose_lost(invalid_reason, now)

        assert (
            measurement is not None
            and measurement.xy_m is not None
            and measurement.long_axis_yaw_rad is not None
        )
        raw = np.asarray(
            (*measurement.xy_m, measurement.long_axis_yaw_rad),
            dtype=np.float64,
        )
        accepted, reason = self._accept_or_reseed(raw)
        if not accepted:
            if self._lost_started_at_s is None:
                self._lost_started_at_s = now
            elif now - self._lost_started_at_s >= self.config.lost_abort_after_s:
                return self.abort("pose_lost_timeout", now)
            self.state = ServoState.POSE_LOST
            self._last_command = ZERO_VELOCITY
            self._last_update_s = now
            self._clear_hold()
            filtered_pose = self._filtered_pose()
            return self._decision(
                reason=reason,
                filtered_pose=filtered_pose,
                accepted=False,
            )
        self._lost_started_at_s = None

        filtered_pose = self._filtered_pose()
        if filtered_pose is None or len(self._samples) < self.config.filter_window:
            self.state = ServoState.ACQUIRING
            self._last_command = ZERO_VELOCITY
            self._last_update_s = now
            return self._decision(
                reason="acquiring_stable_pose",
                filtered_pose=filtered_pose,
                accepted=True,
            )

        target = np.asarray(self.config.target_xy_m, dtype=np.float64)
        filtered_xy = filtered_pose[:2]
        filtered_yaw = float(filtered_pose[2])
        error_xy = filtered_xy - target
        raw_error_xy = raw[:2] - target
        yaw_error = line_angle_difference_rad(
            filtered_yaw,
            self.config.target_long_axis_yaw_rad,
        )
        raw_yaw_error = line_angle_difference_rad(
            float(raw[2]),
            self.config.target_long_axis_yaw_rad,
        )
        filtered_distance = float(np.linalg.norm(error_xy))
        raw_distance = float(np.linalg.norm(raw_error_xy))

        holding = self.state is ServoState.HOLDING
        if holding:
            if (
                filtered_distance <= self.config.arrival_outer_m
                and raw_distance <= self.config.arrival_outer_m
                and abs(yaw_error) <= self.config.arrival_yaw_outer_rad
                and abs(raw_yaw_error) <= self.config.arrival_yaw_outer_rad
            ):
                self._hold_frames += 1
            else:
                self._clear_hold()
                holding = False
        elif (
            filtered_distance <= self.config.arrival_inner_m
            and raw_distance <= self.config.arrival_inner_m
            and abs(yaw_error) <= self.config.arrival_yaw_inner_rad
            and abs(raw_yaw_error) <= self.config.arrival_yaw_inner_rad
        ):
            self._hold_started_at_s = now
            self._hold_frames = 1
            holding = True

        if holding:
            self.state = ServoState.HOLDING
            command = self._slew_towards(
                np.zeros(2, dtype=np.float64),
                0.0,
                now,
            )
            hold_elapsed = now - float(self._hold_started_at_s)
            if (
                self._hold_frames >= self.config.arrival_min_frames
                and hold_elapsed >= self.config.arrival_min_duration_s
                and command.is_zero
            ):
                self.state = ServoState.ARRIVED
                handoff = not self._handoff_emitted
                self._handoff_emitted = True
                self._last_update_s = now
                return self._decision(
                    reason="arrival_stable",
                    filtered_pose=filtered_pose,
                    error_xy=error_xy,
                    yaw_error=yaw_error,
                    accepted=True,
                    handoff=handoff,
                )
            self._last_update_s = now
            return self._decision(
                reason="arrival_holding",
                filtered_pose=filtered_pose,
                error_xy=error_xy,
                yaw_error=yaw_error,
                accepted=True,
            )

        self.state = ServoState.TRACKING
        desired_wz = self.config.yaw_proportional_gain_per_s * yaw_error
        # A rotating base changes the coordinates of a stationary parcel even
        # with zero translation.  This orbit feed-forward keeps the parcel
        # centre approximately fixed while yaw and XY converge together:
        # v_ff = wz * [p_y, -p_x].
        orbit_feedforward = desired_wz * np.asarray(
            (filtered_xy[1], -filtered_xy[0]),
            dtype=np.float64,
        )
        desired_xy = (
            self.config.proportional_gain_per_s * error_xy
            + orbit_feedforward
        )
        desired_xy, desired_wz = self._limit_se2(desired_xy, desired_wz)
        command = self._slew_towards(desired_xy, desired_wz, now)
        self._last_update_s = now
        return self._decision(
            reason="tracking",
            filtered_pose=filtered_pose,
            error_xy=error_xy,
            yaw_error=yaw_error,
            accepted=True,
            command=command,
        )

    def _check_time(self, now_s: float) -> float:
        now = _finite(now_s, "now_s")
        if self._last_update_s is not None and now < self._last_update_s:
            raise ValueError("now_s must be monotonically non-decreasing")
        return now

    def _invalid_reason(
        self,
        measurement: PoseMeasurement | None,
        now_s: float,
    ) -> str | None:
        if measurement is None:
            return "pose_missing"
        if not measurement.valid:
            return "pose_invalid"
        if measurement.xy_m is None or measurement.long_axis_yaw_rad is None:
            return "pose_incomplete"
        age_s = now_s - measurement.timestamp_s
        if age_s < -1e-6:
            return "pose_timestamp_in_future"
        if age_s > self.config.stale_after_s:
            return "pose_stale"
        return None

    def _pose_lost(self, reason: str, now_s: float) -> ServoDecision:
        if self._lost_started_at_s is None:
            self._lost_started_at_s = now_s
        elif now_s - self._lost_started_at_s >= self.config.lost_abort_after_s:
            return self.abort("pose_lost_timeout", now_s)
        self.state = ServoState.POSE_LOST
        self._samples.clear()
        self._jump_candidates.clear()
        self._last_raw = None
        self._last_command = ZERO_VELOCITY
        self._last_update_s = now_s
        self._clear_hold()
        return self._decision(reason=reason)

    def _accept_or_reseed(self, raw: np.ndarray) -> tuple[bool, str]:
        if self._last_raw is None:
            self._samples.append(raw.copy())
            self._last_raw = raw.copy()
            self._jump_candidates.clear()
            return True, "accepted"

        if self._pose_jump_is_within_limit(raw, self._last_raw):
            self._samples.append(raw.copy())
            self._last_raw = raw.copy()
            self._jump_candidates.clear()
            return True, "accepted"

        if self._jump_candidates:
            candidate_centre = self._filtered_pose(self._jump_candidates)
            assert candidate_centre is not None
            if not self._pose_jump_is_within_limit(raw, candidate_centre):
                self._jump_candidates.clear()
        self._jump_candidates.append(raw.copy())

        if len(self._jump_candidates) < self.config.jump_reseed_frames:
            return False, "jump_rejected"

        self._samples.clear()
        self._samples.extend(point.copy() for point in self._jump_candidates)
        self._last_raw = raw.copy()
        self._jump_candidates.clear()
        return True, "jump_reseeded"

    def _pose_jump_is_within_limit(
        self,
        first: np.ndarray,
        second: np.ndarray,
    ) -> bool:
        xy_jump = float(np.linalg.norm(first[:2] - second[:2]))
        yaw_jump = abs(line_angle_difference_rad(float(first[2]), float(second[2])))
        return (
            xy_jump <= self.config.jump_threshold_m
            and yaw_jump <= self.config.yaw_jump_threshold_rad
        )

    @staticmethod
    def _line_medoid_rad(angles_rad: np.ndarray) -> float:
        """Return a robust sample angle without a 0/pi scalar-median seam."""

        normalized = np.mod(np.asarray(angles_rad, dtype=np.float64), math.pi)
        pairwise = np.abs(
            (normalized[:, None] - normalized[None, :] + math.pi / 2.0)
            % math.pi
            - math.pi / 2.0
        )
        return float(normalized[int(np.argmin(np.sum(pairwise, axis=1)))])

    def _filtered_pose(
        self,
        samples: deque[np.ndarray] | None = None,
    ) -> np.ndarray | None:
        selected = self._samples if samples is None else samples
        if not selected:
            return None
        stacked = np.stack(tuple(selected), axis=0)
        filtered_xy = np.median(stacked[:, :2], axis=0)
        filtered_yaw = self._line_medoid_rad(stacked[:, 2])
        return np.asarray((*filtered_xy, filtered_yaw), dtype=np.float64)

    @staticmethod
    def _limit_norm(vector: np.ndarray, limit: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= limit or norm <= 1e-12:
            return vector
        return vector * (limit / norm)

    def _limit_se2(
        self,
        linear: np.ndarray,
        angular: float,
    ) -> tuple[np.ndarray, float]:
        """Scale linear/angular together so orbit geometry survives speed caps."""

        scale = 1.0
        linear_norm = float(np.linalg.norm(linear))
        if linear_norm > self.config.max_linear_speed_mps:
            scale = min(scale, self.config.max_linear_speed_mps / linear_norm)
        if abs(angular) > self.config.max_angular_speed_radps:
            scale = min(scale, self.config.max_angular_speed_radps / abs(angular))
        return linear * scale, float(angular) * scale

    def _slew_towards(
        self,
        desired_linear: np.ndarray,
        desired_angular: float,
        now_s: float,
    ) -> VelocityCommand:
        assert self._last_update_s is not None
        previous_linear = np.asarray(
            (self._last_command.vx_mps, self._last_command.vy_mps),
            dtype=np.float64,
        )
        previous_angular = self._last_command.wz_radps
        dt_s = max(0.0, now_s - self._last_update_s)
        max_linear_delta = self.config.max_linear_acceleration_mps2 * dt_s
        max_angular_delta = self.config.max_angular_acceleration_radps2 * dt_s
        linear_delta = desired_linear - previous_linear
        angular_delta = float(desired_angular - previous_angular)

        # Use one dimensionless scale for the complete twist delta.  Applying
        # independent clamps would distort the orbit feed-forward during
        # acceleration even though the steady-state speed cap is coupled.
        slew_scale = 1.0
        linear_delta_norm = float(np.linalg.norm(linear_delta))
        if linear_delta_norm > max_linear_delta:
            slew_scale = min(
                slew_scale,
                0.0
                if max_linear_delta <= 0.0
                else max_linear_delta / linear_delta_norm,
            )
        if abs(angular_delta) > max_angular_delta:
            slew_scale = min(
                slew_scale,
                0.0
                if max_angular_delta <= 0.0
                else max_angular_delta / abs(angular_delta),
            )
        linear_delta = linear_delta * slew_scale
        angular_delta = angular_delta * slew_scale
        linear = self._limit_norm(
            previous_linear + linear_delta,
            self.config.max_linear_speed_mps,
        )
        angular = float(
            np.clip(
                previous_angular + angular_delta,
                -self.config.max_angular_speed_radps,
                self.config.max_angular_speed_radps,
            )
        )
        command = VelocityCommand(float(linear[0]), float(linear[1]), angular)
        self._last_command = command
        return command

    def _clear_hold(self) -> None:
        self._hold_started_at_s = None
        self._hold_frames = 0

    def _decision(
        self,
        *,
        reason: str,
        filtered_pose: np.ndarray | None = None,
        error_xy: np.ndarray | None = None,
        yaw_error: float | None = None,
        accepted: bool = False,
        handoff: bool = False,
        command: VelocityCommand | None = None,
    ) -> ServoDecision:
        selected_command = command if command is not None else self._last_command
        return ServoDecision(
            state=self.state,
            command=selected_command,
            filtered_xy_m=(
                None
                if filtered_pose is None
                else (float(filtered_pose[0]), float(filtered_pose[1]))
            ),
            error_xy_m=(
                None
                if error_xy is None
                else (float(error_xy[0]), float(error_xy[1]))
            ),
            filtered_long_axis_yaw_rad=(
                None if filtered_pose is None else float(filtered_pose[2])
            ),
            yaw_error_rad=None if yaw_error is None else float(yaw_error),
            measurement_accepted=accepted,
            handoff_ready=handoff,
            reason=reason,
        )


class RobotMotionDisabledError(RuntimeError):
    """Raised when an RB-Y1 stream is used without explicit execution opt-in."""


class StreamShutdownError(RuntimeError):
    """Raised when an RB-Y1 command stream does not finish after cancellation."""


class StreamFeedbackError(RuntimeError):
    """Raised when RB-Y1 feedback does not confirm a live SE(2) controller."""


class MobilityCommandPumpError(RuntimeError):
    """Raised when the fixed-rate mobility sender cannot stay healthy."""


@dataclass(frozen=True, slots=True)
class RBY1StreamConfig:
    """RB-Y1 SE(2) stream parameters with conservative hold and limits."""

    priority: int = 10
    # RB-Y1 ends the whole stream when this command hold expires.  The Jetson
    # vision loop can hold the GIL for several hundred milliseconds, so the
    # hold must bridge those stalls while a separate watchdog sender refreshes
    # the command at a fixed rate.
    control_hold_time_s: float = 1.0
    minimum_time_s: float = 0.05
    linear_acceleration_limit_mps2: float = 0.15
    angular_acceleration_limit_radps2: float = 0.20
    send_timeout_ms: int = 250
    zero_repetitions: int = 3
    shutdown_timeout_ms: int = 2000

    def __post_init__(self) -> None:
        if self.priority < 1:
            raise ValueError("priority must be positive")
        for name in (
            "control_hold_time_s",
            "minimum_time_s",
            "linear_acceleration_limit_mps2",
            "angular_acceleration_limit_radps2",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.control_hold_time_s > 1.0:
            raise ValueError("control_hold_time_s cannot exceed 1.0 second")
        if self.send_timeout_ms <= 0 or self.shutdown_timeout_ms <= 0:
            raise ValueError("stream timeouts must be positive")
        if self.zero_repetitions < 2:
            raise ValueError("zero_repetitions must be at least 2")


class RBY1MobilityStream:
    """Explicitly enabled, lazy-import RB-Y1 SE(2) command-stream adapter."""

    def __init__(
        self,
        robot: Any,
        *,
        execute: bool = False,
        config: RBY1StreamConfig | None = None,
        sdk_module: Any | None = None,
    ) -> None:
        self._robot = robot
        self._execute = bool(execute)
        self.config = config or RBY1StreamConfig()
        self._sdk = sdk_module
        self._stream: Any | None = None

    @property
    def is_open(self) -> bool:
        return self._stream is not None

    def open(self) -> "RBY1MobilityStream":
        if not self._execute:
            raise RobotMotionDisabledError(
                "robot motion is disabled; construct with execute=True explicitly"
            )
        if self._stream is not None:
            return self
        if self._sdk is None:
            self._sdk = importlib.import_module("rby1_sdk")
        self._stream = self._robot.create_command_stream(priority=self.config.priority)
        return self

    def send(self, command: VelocityCommand) -> Any:
        """Send one bounded SE(2) command."""

        if not self._execute:
            raise RobotMotionDisabledError("robot motion is disabled")
        stream = self._require_stream()
        if command.linear_norm_mps > MAX_ALLOWED_LINEAR_SPEED_MPS + 1e-12:
            raise ValueError("linear velocity exceeds the 0.08 m/s safety limit")
        if abs(command.wz_radps) > MAX_ALLOWED_ANGULAR_SPEED_RADPS + 1e-12:
            raise ValueError("angular velocity exceeds the 0.10 rad/s safety limit")
        try:
            feedback = stream.send_command(
                self._command_builder(command),
                timeout_ms=self.config.send_timeout_ms,
            )
            self._feedback_state(feedback)
            return feedback
        except Exception as exc:
            cleanup_error = self._invalidate_stream(stream)
            if cleanup_error is not None:
                raise StreamShutdownError(
                    "RB-Y1 mobility command failed and stream release was not "
                    f"confirmed: command={exc}; cleanup={cleanup_error}"
                ) from exc
            raise

    def feedback_is_running(self, feedback: Any) -> bool:
        """Return whether validated feedback confirms active base control."""

        return self._feedback_state(feedback) == "running"

    def stop_and_release(self) -> None:
        """Send repeated zeros, cancel, and wait before releasing the stream."""

        for _ in range(self.config.zero_repetitions):
            self.send(ZERO_VELOCITY)
        self.cancel_and_wait()

    def cancel_and_wait(self) -> None:
        """Cancel without sending; used only after the sole sender has stopped."""

        stream = self._require_stream()
        completed = False
        try:
            stream.cancel()
        finally:
            try:
                completed = bool(
                    stream.wait_for(self.config.shutdown_timeout_ms)
                )
            finally:
                self._stream = None
        if not completed:
            raise StreamShutdownError(
                "RB-Y1 mobility stream did not finish after cancellation"
            )

    def stop_for_handoff(self) -> None:
        """Compatibility alias for :meth:`stop_and_release`."""

        self.stop_and_release()

    def close(self) -> None:
        if self._stream is not None:
            self.stop_and_release()

    def __enter__(self) -> "RBY1MobilityStream":
        return self.open()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _require_stream(self) -> Any:
        if self._stream is None:
            raise RuntimeError("RB-Y1 mobility stream is not open")
        return self._stream

    def _invalidate_stream(self, stream: Any) -> Exception | None:
        """Release after send failure and report whether shutdown was confirmed."""

        self._stream = None
        failures: list[str] = []
        try:
            stream.cancel()
        except Exception as exc:
            failures.append(f"cancel failed: {exc}")
        try:
            completed = bool(stream.wait_for(self.config.shutdown_timeout_ms))
        except Exception as exc:
            failures.append(f"wait failed: {exc}")
        else:
            if not completed:
                failures.append("wait timed out")
        if failures:
            return StreamShutdownError("; ".join(failures))
        return None

    def _feedback_state(self, feedback: Any) -> str:
        """Validate top-level and SE(2) feedback using wire numeric codes.

        The locally available robot-core source and SDK disagree on some
        finish-code names, so control decisions intentionally use only the
        stable wire values.  Initializing is accepted temporarily, but the
        command pump does not acknowledge a generation until Running is seen.
        """

        if feedback is None or not bool(getattr(feedback, "valid", False)):
            raise StreamFeedbackError("missing or invalid RB-Y1 mobility feedback")

        try:
            component = feedback.component_based_command
            mobility = component.mobility_command
            se2 = mobility.se2_velocity_command
        except AttributeError as exc:
            raise StreamFeedbackError(
                "RB-Y1 feedback does not contain a component-based SE(2) command"
            ) from exc
        if not all(bool(getattr(node, "valid", False)) for node in (component, mobility, se2)):
            raise StreamFeedbackError(
                "RB-Y1 feedback does not validate the requested SE(2) command"
            )

        status = self._wire_enum_code(feedback.status, "status")
        finish = self._wire_enum_code(feedback.finish_code, "finish_code")
        detail = (
            f"status={status} ({feedback.status!r}), "
            f"finish={finish} ({feedback.finish_code!r})"
        )
        if status == 2 and finish == 0:
            return "running"
        if status == 1 and finish == 0:
            return "initializing"
        if status == 3:
            raise StreamFeedbackError(f"RB-Y1 mobility stream terminated: {detail}")
        if status == 0:
            raise StreamFeedbackError(
                f"RB-Y1 mobility command was not activated: {detail}"
            )
        raise StreamFeedbackError(f"unexpected RB-Y1 mobility feedback: {detail}")

    @staticmethod
    def _wire_enum_code(value: Any, field: str) -> int:
        raw = getattr(value, "value", value)
        try:
            return int(raw)
        except (TypeError, ValueError) as exc:
            raise StreamFeedbackError(
                f"invalid RB-Y1 mobility feedback {field}: {value!r}"
            ) from exc

    def _command_builder(self, command: VelocityCommand) -> Any:
        assert self._sdk is not None
        linear = np.asarray((command.vx_mps, command.vy_mps), dtype=np.float64)
        linear_acceleration = np.full(
            2,
            self.config.linear_acceleration_limit_mps2,
            dtype=np.float64,
        )
        se2 = (
            self._sdk.SE2VelocityCommandBuilder()
            .set_command_header(
                self._sdk.CommandHeaderBuilder().set_control_hold_time(
                    self.config.control_hold_time_s
                )
            )
            .set_minimum_time(self.config.minimum_time_s)
            .set_velocity(linear, command.wz_radps)
            .set_acceleration_limit(
                linear_acceleration,
                self.config.angular_acceleration_limit_radps2,
            )
        )
        component = self._sdk.ComponentBasedCommandBuilder().set_mobility_command(se2)
        return self._sdk.RobotCommandBuilder().set_command(component)


@dataclass(frozen=True, slots=True)
class RBY1CommandPumpConfig:
    """Fixed-rate sender settings independent of D435 estimation latency."""

    send_rate_hz: float = 20.0
    command_stale_after_s: float = 0.30
    startup_timeout_s: float = 2.0
    zero_ack_timeout_s: float = 2.0
    zero_ack_repetitions: int = 3
    join_timeout_s: float = 2.0

    def __post_init__(self) -> None:
        for name in (
            "send_rate_hz",
            "command_stale_after_s",
            "startup_timeout_s",
            "zero_ack_timeout_s",
            "join_timeout_s",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        period_s = 1.0 / self.send_rate_hz
        if self.command_stale_after_s < period_s * 2.0:
            raise ValueError(
                "command_stale_after_s must cover at least two pump periods"
            )
        if self.zero_ack_repetitions < 2:
            raise ValueError("zero_ack_repetitions must be at least 2")


class RBY1MobilityCommandPump:
    """Sole fixed-rate owner of an open RB-Y1 mobility command stream.

    Vision publishes the newest bounded command, but never calls the SDK
    stream directly.  The pump refreshes that command at a fixed rate and
    substitutes zero when the producer stops updating.  Shutdown latches zero,
    waits until that generation was sent, joins the sender, then cancels the
    underlying stream; there is no concurrent ``send``/``cancel`` race.
    """

    def __init__(
        self,
        stream: RBY1MobilityStream,
        *,
        config: RBY1CommandPumpConfig | None = None,
    ) -> None:
        self._stream = stream
        self.config = config or RBY1CommandPumpConfig()
        self._condition = threading.Condition()
        self._shutdown_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_command = ZERO_VELOCITY
        self._latest_update_s = time.monotonic()
        self._generation = 0
        self._sent_generation = -1
        self._last_sent_command = ZERO_VELOCITY
        self._zero_sends_after_latch = 0
        self._send_count = 0
        self._last_send_completed_s: float | None = None
        self._max_send_gap_s = 0.0
        self._running = False
        self._zero_latched = False
        self._closed = False
        self._last_error: Exception | None = None

    @property
    def is_running(self) -> bool:
        with self._condition:
            return self._running

    @property
    def is_closed(self) -> bool:
        with self._condition:
            return self._closed

    @property
    def send_count(self) -> int:
        with self._condition:
            return self._send_count

    @property
    def last_error(self) -> Exception | None:
        with self._condition:
            return self._last_error

    @property
    def max_send_gap_s(self) -> float:
        with self._condition:
            return self._max_send_gap_s

    def start(self) -> "RBY1MobilityCommandPump":
        """Start with zero velocity and verify active ``Running`` feedback."""

        with self._condition:
            if self._closed:
                raise MobilityCommandPumpError("mobility command pump is closed")
            if self._running:
                return self
            if not self._stream.is_open:
                raise MobilityCommandPumpError("mobility stream is not open")
            self._stop_event.clear()
            self._latest_command = ZERO_VELOCITY
            self._latest_update_s = time.monotonic()
            self._generation += 1
            startup_generation = self._generation
            self._zero_latched = False
            self._zero_sends_after_latch = 0
            self._last_error = None
            self._running = True
            self._thread = threading.Thread(
                target=self._run,
                name="rby1-mobility-command-pump",
                daemon=True,
            )
            self._thread.start()
        try:
            self._wait_until_sent(
                startup_generation,
                timeout_s=self.config.startup_timeout_s,
                operation="start mobility command pump",
            )
        except Exception as startup_error:
            cleanup_error = self._abort_startup()
            if cleanup_error is not None:
                raise MobilityCommandPumpError(
                    f"{startup_error}; startup cleanup failed: {cleanup_error}"
                ) from startup_error
            raise
        return self

    def publish(self, command: VelocityCommand) -> int:
        """Atomically replace the command consumed by the sender thread."""

        self._validate_command(command)
        with self._condition:
            self._raise_if_failed_locked()
            if not self._running or self._closed:
                raise MobilityCommandPumpError("mobility command pump is not running")
            if self._zero_latched and not command.is_zero:
                raise MobilityCommandPumpError(
                    "mobility command pump is zero-latched for body handoff"
                )
            self._latest_command = command
            self._latest_update_s = time.monotonic()
            self._generation += 1
            generation = self._generation
            self._condition.notify_all()
            return generation

    def raise_if_failed(self) -> None:
        """Surface a background SDK failure on the owning thread."""

        with self._condition:
            self._raise_if_failed_locked()

    def latch_zero_and_wait(self) -> None:
        """Permanently select zero and wait for repeated acknowledged sends."""

        with self._condition:
            self._raise_if_failed_locked()
            if not self._running or self._closed:
                raise MobilityCommandPumpError("mobility command pump is not running")
            self._zero_latched = True
            self._zero_sends_after_latch = 0
            self._latest_command = ZERO_VELOCITY
            self._latest_update_s = time.monotonic()
            self._generation += 1
            zero_generation = self._generation
            self._condition.notify_all()
        deadline_s = time.monotonic() + self.config.zero_ack_timeout_s
        with self._condition:
            while (
                self._sent_generation < zero_generation
                or self._zero_sends_after_latch
                < self.config.zero_ack_repetitions
            ):
                self._raise_if_failed_locked()
                if not self._running:
                    raise MobilityCommandPumpError(
                        "cannot send zero mobility command: sender thread stopped"
                    )
                remaining_s = deadline_s - time.monotonic()
                if remaining_s <= 0.0:
                    raise MobilityCommandPumpError(
                        "timed out while trying to send zero mobility command"
                    )
                self._condition.wait(timeout=remaining_s)
            if not self._last_sent_command.is_zero:
                raise MobilityCommandPumpError(
                    "mobility pump acknowledged a non-zero handoff command"
                )

    def stop_and_release(self) -> None:
        """Stop the sole sender, then zero/cancel/wait the SDK stream."""

        with self._shutdown_lock:
            with self._condition:
                if self._closed:
                    return
                primary_error = self._background_error_locked()

            if self.is_running:
                try:
                    self.latch_zero_and_wait()
                except Exception as exc:
                    primary_error = primary_error or exc

            thread_alive, emergency_error = self._stop_sender()
            if thread_alive and primary_error is None:
                primary_error = MobilityCommandPumpError(
                    "mobility command pump thread remained alive after stream cancel"
                )
            if emergency_error is not None and primary_error is None:
                primary_error = emergency_error

            # A send can fail after the zero latch completed but before the
            # stop event reaches the sender.  Re-read the background result
            # after join so body handoff can never follow that late failure.
            with self._condition:
                primary_error = primary_error or self._background_error_locked()

            stream_error: Exception | None = None
            if self._stream.is_open and not thread_alive:
                try:
                    self._stream.stop_and_release()
                except Exception as exc:
                    stream_error = exc

            with self._condition:
                primary_error = primary_error or self._background_error_locked()
                if not thread_alive:
                    self._running = False
                self._closed = not thread_alive and not self._stream.is_open
                cleanup_complete = self._closed
                self._condition.notify_all()

            if primary_error is not None:
                raise primary_error
            if stream_error is not None:
                raise stream_error
            if not cleanup_complete:
                raise MobilityCommandPumpError(
                    "mobility command pump shutdown was not confirmed"
                )

    def close(self) -> None:
        self.stop_and_release()

    def _abort_startup(self) -> Exception | None:
        """Transactionally unwind a startup that never confirmed Running."""

        with self._shutdown_lock:
            thread_alive, cleanup_error = self._stop_sender()
            if self._stream.is_open and not thread_alive:
                try:
                    # No Running acknowledgement exists, so do not issue more
                    # owner-thread commands; cancel the unconfirmed stream.
                    self._stream.cancel_and_wait()
                except Exception as exc:
                    cleanup_error = cleanup_error or exc
            with self._condition:
                if not thread_alive:
                    self._running = False
                self._closed = not thread_alive and not self._stream.is_open
                self._condition.notify_all()
            if thread_alive:
                return MobilityCommandPumpError(
                    "startup sender remained alive after emergency stream cancel"
                )
            return cleanup_error

    def _stop_sender(self) -> tuple[bool, Exception | None]:
        """Join the sole sender, canceling only to unblock a timed-out send."""

        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
            thread = self._thread
        if thread is None:
            return False, None

        thread.join(timeout=self.config.join_timeout_s)
        thread_alive = thread.is_alive()
        cleanup_error: Exception | None = None
        if thread_alive:
            if self._stream.is_open:
                try:
                    self._stream.cancel_and_wait()
                except Exception as exc:
                    cleanup_error = exc
            thread.join(timeout=self.config.join_timeout_s)
            thread_alive = thread.is_alive()
        return thread_alive, cleanup_error

    def _run(self) -> None:
        period_s = 1.0 / self.config.send_rate_hz
        next_tick_s = time.monotonic()
        try:
            while not self._stop_event.is_set():
                now_s = time.monotonic()
                with self._condition:
                    command = self._latest_command
                    generation = self._generation
                    stale = (
                        now_s - self._latest_update_s
                        > self.config.command_stale_after_s
                    )
                selected = ZERO_VELOCITY if stale else command
                feedback = self._stream.send(selected)
                feedback_is_running = getattr(
                    self._stream,
                    "feedback_is_running",
                    lambda _: True,
                )(feedback)
                send_completed_s = time.monotonic()
                with self._condition:
                    if self._last_send_completed_s is not None:
                        self._max_send_gap_s = max(
                            self._max_send_gap_s,
                            send_completed_s - self._last_send_completed_s,
                        )
                    self._last_send_completed_s = send_completed_s
                    self._send_count += 1
                    if feedback_is_running:
                        self._last_sent_command = selected
                        if self._zero_latched and selected.is_zero:
                            self._zero_sends_after_latch += 1
                        if not stale or command.is_zero:
                            self._sent_generation = max(
                                self._sent_generation,
                                generation,
                            )
                    self._condition.notify_all()

                next_tick_s += period_s
                delay_s = next_tick_s - time.monotonic()
                if delay_s <= 0.0:
                    next_tick_s = time.monotonic()
                    continue
                self._stop_event.wait(delay_s)
        except Exception as exc:
            with self._condition:
                self._last_error = exc
                self._condition.notify_all()
        finally:
            with self._condition:
                self._running = False
                self._condition.notify_all()

    def _wait_until_sent(
        self,
        generation: int,
        *,
        timeout_s: float,
        operation: str,
    ) -> None:
        deadline_s = time.monotonic() + timeout_s
        with self._condition:
            while self._sent_generation < generation:
                self._raise_if_failed_locked()
                if not self._running:
                    raise MobilityCommandPumpError(
                        f"cannot {operation}: sender thread stopped"
                    )
                remaining_s = deadline_s - time.monotonic()
                if remaining_s <= 0.0:
                    raise MobilityCommandPumpError(
                        f"timed out while trying to {operation}"
                    )
                self._condition.wait(timeout=remaining_s)

    @staticmethod
    def _validate_command(command: VelocityCommand) -> None:
        if command.linear_norm_mps > MAX_ALLOWED_LINEAR_SPEED_MPS + 1e-12:
            raise ValueError("linear velocity exceeds the 0.08 m/s safety limit")
        if abs(command.wz_radps) > MAX_ALLOWED_ANGULAR_SPEED_RADPS + 1e-12:
            raise ValueError("angular velocity exceeds the 0.10 rad/s safety limit")

    def _raise_if_failed_locked(self) -> None:
        if self._last_error is not None:
            raise MobilityCommandPumpError(
                f"mobility command pump failed: {self._last_error}"
            ) from self._last_error

    def _background_error_locked(self) -> Exception | None:
        if self._last_error is None:
            return None
        return MobilityCommandPumpError(
            f"mobility command pump failed: {self._last_error}"
        )
