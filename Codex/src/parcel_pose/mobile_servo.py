"""Pure mobile-base visual servo logic and an opt-in RB-Y1 stream adapter.

The controller consumes parcel centres expressed in the corrected RB-Y1 base
frame.  A positive centre error therefore commands positive base velocity:
moving the base in that direction reduces the next relative parcel error.

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
from typing import Any

import numpy as np


MAX_ALLOWED_LINEAR_SPEED_MPS = 0.08


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
    """Safety and convergence parameters for parcel-centre servoing."""

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
    arrival_min_frames: int = 5
    arrival_min_duration_s: float = 0.35
    lost_abort_after_s: float = 2.0
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_xy_m", _xy(self.target_xy_m, "target_xy_m"))
        for name in (
            "proportional_gain_per_s",
            "max_linear_speed_mps",
            "max_linear_acceleration_mps2",
            "jump_threshold_m",
            "stale_after_s",
            "arrival_inner_m",
            "arrival_outer_m",
            "arrival_min_duration_s",
            "lost_abort_after_s",
            "timeout_s",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.max_linear_speed_mps > MAX_ALLOWED_LINEAR_SPEED_MPS:
            raise ValueError(
                "max_linear_speed_mps cannot exceed the 0.08 m/s safety limit"
            )
        if self.filter_window != 3:
            raise ValueError("filter_window must be 3 for the robust median filter")
        if self.jump_reseed_frames < 2:
            raise ValueError("jump_reseed_frames must be at least 2")
        if self.arrival_min_frames < 1:
            raise ValueError("arrival_min_frames must be positive")
        if self.arrival_outer_m <= self.arrival_inner_m:
            raise ValueError("arrival_outer_m must be greater than arrival_inner_m")


@dataclass(frozen=True, slots=True)
class PoseMeasurement:
    """One corrected base-frame parcel-centre observation."""

    xy_m: tuple[float, float] | None
    timestamp_s: float
    valid: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", _finite(self.timestamp_s, "timestamp_s"))
        if self.valid:
            if self.xy_m is None:
                raise ValueError("a valid pose measurement requires xy_m")
            object.__setattr__(self, "xy_m", _xy(self.xy_m, "xy_m"))
        elif self.xy_m is not None:
            object.__setattr__(self, "xy_m", _xy(self.xy_m, "xy_m"))

    @classmethod
    def invalid(cls, timestamp_s: float) -> "PoseMeasurement":
        return cls(xy_m=None, timestamp_s=timestamp_s, valid=False)


@dataclass(frozen=True, slots=True)
class VelocityCommand:
    """Planar RB-Y1 velocity command; yaw is deliberately fixed at zero."""

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


class MobileVisualServo:
    """Deterministic XY-only visual-servo state machine.

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

        assert measurement is not None and measurement.xy_m is not None
        raw = np.asarray(measurement.xy_m, dtype=np.float64)
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
            return self._decision(
                reason=reason,
                filtered=self._filtered_xy(),
                accepted=False,
            )
        self._lost_started_at_s = None

        filtered = self._filtered_xy()
        if filtered is None or len(self._samples) < self.config.filter_window:
            self.state = ServoState.ACQUIRING
            self._last_command = ZERO_VELOCITY
            self._last_update_s = now
            return self._decision(
                reason="acquiring_stable_pose",
                filtered=filtered,
                accepted=True,
            )

        target = np.asarray(self.config.target_xy_m, dtype=np.float64)
        error = filtered - target
        raw_error = raw - target
        filtered_distance = float(np.linalg.norm(error))
        raw_distance = float(np.linalg.norm(raw_error))

        holding = self.state is ServoState.HOLDING
        if holding:
            if (
                filtered_distance <= self.config.arrival_outer_m
                and raw_distance <= self.config.arrival_outer_m
            ):
                self._hold_frames += 1
            else:
                self._clear_hold()
                holding = False
        elif (
            filtered_distance <= self.config.arrival_inner_m
            and raw_distance <= self.config.arrival_inner_m
        ):
            self._hold_started_at_s = now
            self._hold_frames = 1
            holding = True

        if holding:
            self.state = ServoState.HOLDING
            command = self._slew_towards(np.zeros(2, dtype=np.float64), now)
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
                    filtered=filtered,
                    error=error,
                    accepted=True,
                    handoff=handoff,
                )
            self._last_update_s = now
            return self._decision(
                reason="arrival_holding",
                filtered=filtered,
                error=error,
                accepted=True,
            )

        self.state = ServoState.TRACKING
        desired = self.config.proportional_gain_per_s * error
        desired = self._limit_norm(desired, self.config.max_linear_speed_mps)
        command = self._slew_towards(desired, now)
        self._last_update_s = now
        return self._decision(
            reason="tracking",
            filtered=filtered,
            error=error,
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

        if float(np.linalg.norm(raw - self._last_raw)) <= self.config.jump_threshold_m:
            self._samples.append(raw.copy())
            self._last_raw = raw.copy()
            self._jump_candidates.clear()
            return True, "accepted"

        if self._jump_candidates:
            candidate_centre = np.median(
                np.stack(tuple(self._jump_candidates), axis=0), axis=0
            )
            if (
                float(np.linalg.norm(raw - candidate_centre))
                > self.config.jump_threshold_m
            ):
                self._jump_candidates.clear()
        self._jump_candidates.append(raw.copy())

        if len(self._jump_candidates) < self.config.jump_reseed_frames:
            return False, "jump_rejected"

        self._samples.clear()
        self._samples.extend(point.copy() for point in self._jump_candidates)
        self._last_raw = raw.copy()
        self._jump_candidates.clear()
        return True, "jump_reseeded"

    def _filtered_xy(self) -> np.ndarray | None:
        if not self._samples:
            return None
        return np.median(np.stack(tuple(self._samples), axis=0), axis=0)

    @staticmethod
    def _limit_norm(vector: np.ndarray, limit: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= limit or norm <= 1e-12:
            return vector
        return vector * (limit / norm)

    def _slew_towards(self, desired: np.ndarray, now_s: float) -> VelocityCommand:
        assert self._last_update_s is not None
        previous = np.asarray(
            (self._last_command.vx_mps, self._last_command.vy_mps),
            dtype=np.float64,
        )
        dt_s = max(0.0, now_s - self._last_update_s)
        max_delta = self.config.max_linear_acceleration_mps2 * dt_s
        delta = self._limit_norm(desired - previous, max_delta)
        velocity = self._limit_norm(
            previous + delta,
            self.config.max_linear_speed_mps,
        )
        command = VelocityCommand(float(velocity[0]), float(velocity[1]), 0.0)
        self._last_command = command
        return command

    def _clear_hold(self) -> None:
        self._hold_started_at_s = None
        self._hold_frames = 0

    def _decision(
        self,
        *,
        reason: str,
        filtered: np.ndarray | None = None,
        error: np.ndarray | None = None,
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
                if filtered is None
                else (float(filtered[0]), float(filtered[1]))
            ),
            error_xy_m=(
                None if error is None else (float(error[0]), float(error[1]))
            ),
            measurement_accepted=accepted,
            handoff_ready=handoff,
            reason=reason,
        )


class RobotMotionDisabledError(RuntimeError):
    """Raised when an RB-Y1 stream is used without explicit execution opt-in."""


class StreamShutdownError(RuntimeError):
    """Raised when an RB-Y1 command stream does not finish after cancellation."""


@dataclass(frozen=True, slots=True)
class RBY1StreamConfig:
    """RB-Y1 SE(2) stream parameters with conservative hold and limits."""

    priority: int = 10
    control_hold_time_s: float = 0.25
    minimum_time_s: float = 0.05
    linear_acceleration_limit_mps2: float = 0.15
    angular_acceleration_limit_radps2: float = 0.5
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
        if self.control_hold_time_s > 0.5:
            raise ValueError("control_hold_time_s cannot exceed 0.5 seconds")
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
        """Send one bounded XY command; non-zero yaw is rejected."""

        if not self._execute:
            raise RobotMotionDisabledError("robot motion is disabled")
        stream = self._require_stream()
        if abs(command.wz_radps) > 1e-12:
            raise ValueError("mobile visual servo is XY-only; wz must be zero")
        if command.linear_norm_mps > MAX_ALLOWED_LINEAR_SPEED_MPS + 1e-12:
            raise ValueError("linear velocity exceeds the 0.08 m/s safety limit")
        try:
            return stream.send_command(
                self._command_builder(command),
                timeout_ms=self.config.send_timeout_ms,
            )
        except Exception:
            self._invalidate_stream(stream)
            raise

    def stop_and_release(self) -> None:
        """Send repeated zeros, cancel, and wait before releasing the stream."""

        stream = self._require_stream()
        completed = False
        try:
            for _ in range(self.config.zero_repetitions):
                self.send(ZERO_VELOCITY)
        finally:
            if self._stream is not None:
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

    def _invalidate_stream(self, stream: Any) -> None:
        """Best-effort release after a send/build failure, preserving its error."""

        self._stream = None
        try:
            stream.cancel()
        except Exception:
            pass
        try:
            stream.wait_for(self.config.shutdown_timeout_ms)
        except Exception:
            pass

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
            .set_velocity(linear, 0.0)
            .set_acceleration_limit(
                linear_acceleration,
                self.config.angular_acceleration_limit_radps2,
            )
        )
        component = self._sdk.ComponentBasedCommandBuilder().set_mobility_command(se2)
        return self._sdk.RobotCommandBuilder().set_command(component)
