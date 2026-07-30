"""Automatic RB-Y1 box-picking lifecycle orchestration.

The default operator entrypoint prepares the fixed RB-Y1 M setup, runs bounded
mobile alignment from parcel pose measurements, waits for a measured wheel
stop, shuts down the mobility stream, and hands off to the same robot session
for the packaged grasp sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import math
import threading
import time
from types import ModuleType
from typing import Any, Callable

import numpy as np

from parcel_pose_common.angles import line_angle_difference_rad
from .evaluation import BasePoseDiagnostic
from parcel_pose_common.mobile_servo import (
    MobileVisualServo,
    PoseMeasurement,
    RBY1CommandPumpConfig,
    RBY1MobilityCommandPump,
    RBY1MobilityStream,
    RBY1StreamConfig,
    ServoConfig,
    ServoDecision,
    ServoState,
)


DEFAULT_ROBOT_ADDRESS = "192.168.30.1:50051"
EXPECTED_ROBOT_MODEL = "M"
EXPECTED_ROBOT_VERSION = "1.2"
EXPECTED_TORSO_POSITION_DEG = (0.0, 55.0, -59.988, 6.532, 0.0, 0.0)
EXPECTED_HEAD_POSITION_DEG = (0.0, 49.846)


class AutoGrabError(RuntimeError):
    """Raised when mobile alignment cannot safely reach the grasp handoff."""


@dataclass(frozen=True, slots=True)
class AutoGrabConfig:
    """Robot and controller settings for the fixed RB-Y1 M v1.2 setup."""

    address: str = DEFAULT_ROBOT_ADDRESS
    power: str = ".*"
    servo: ServoConfig = field(default_factory=ServoConfig)
    stream: RBY1StreamConfig = field(default_factory=RBY1StreamConfig)
    pump: RBY1CommandPumpConfig = field(default_factory=RBY1CommandPumpConfig)
    fixed_posture_tolerance_deg: float = 1.0
    max_grasp_yaw_residual_deg: float = 8.0
    mobile_stop_velocity_threshold_radps: float = 0.05
    mobile_stop_min_duration_s: float = 0.35
    mobile_state_update_rate_hz: float = 20.0
    mobile_state_stale_after_s: float = 0.15
    mobile_stop_timeout_s: float = 2.0

    def __post_init__(self) -> None:
        if not str(self.address).strip():
            raise ValueError("robot address cannot be empty")
        if not str(self.power).strip():
            raise ValueError("power device pattern cannot be empty")
        if abs(
            line_angle_difference_rad(
                self.servo.target_long_axis_yaw_rad,
                math.pi / 2.0,
            )
        ) > 1e-9:
            raise ValueError(
                "the current packaged grasp posture requires horizontal "
                "long-axis yaw=90 deg mod 180"
            )
        if (
            not np.isfinite(self.fixed_posture_tolerance_deg)
            or self.fixed_posture_tolerance_deg <= 0.0
        ):
            raise ValueError("fixed_posture_tolerance_deg must be positive and finite")
        if (
            not np.isfinite(self.max_grasp_yaw_residual_deg)
            or self.max_grasp_yaw_residual_deg <= 0.0
            or self.max_grasp_yaw_residual_deg >= 45.0
        ):
            raise ValueError(
                "max_grasp_yaw_residual_deg must be finite and between 0 and 45"
            )
        for name in (
            "mobile_stop_velocity_threshold_radps",
            "mobile_stop_min_duration_s",
            "mobile_state_update_rate_hz",
            "mobile_state_stale_after_s",
            "mobile_stop_timeout_s",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.mobile_state_stale_after_s >= self.mobile_stop_timeout_s:
            raise ValueError(
                "mobile_state_stale_after_s must be less than mobile_stop_timeout_s"
            )
        if self.mobile_stop_min_duration_s >= self.mobile_stop_timeout_s:
            raise ValueError(
                "mobile_stop_min_duration_s must be less than mobile_stop_timeout_s"
            )
        if (
            1.0 / self.mobile_state_update_rate_hz
            >= self.mobile_state_stale_after_s
        ):
            raise ValueError(
                "mobile_state_update_rate_hz is too slow for the state freshness limit"
            )


def _load_grabbing_sequence() -> ModuleType:
    """Lazy-load the package-local grasp sequence only for robot execution."""

    try:
        return importlib.import_module(".grabbing", package=__package__)
    except ImportError as exc:
        raise AutoGrabError(
            f"cannot load the packaged RB-Y1 grasp sequence: {exc}"
        ) from exc


def _validate_robot_identity(robot: Any) -> None:
    """Fail closed unless the connected controller reports Model M v1.2."""

    get_info = getattr(robot, "get_robot_info", None)
    if not callable(get_info):
        raise AutoGrabError("connected SDK cannot report RB-Y1 model/version")
    info = get_info()
    model_name = str(getattr(info, "robot_model_name", "")).strip().upper()
    version = (
        str(getattr(info, "robot_model_version", ""))
        .strip()
        .lower()
        .removeprefix("v")
    )
    if model_name != EXPECTED_ROBOT_MODEL:
        raise AutoGrabError(
            f"auto-grab requires RB-Y1 Model M; controller reports {model_name!r}"
        )
    if version != EXPECTED_ROBOT_VERSION and not version.startswith(
        f"{EXPECTED_ROBOT_VERSION}."
    ):
        raise AutoGrabError(
            "auto-grab requires RB-Y1 Model M v1.2; controller reports "
            f"version {version!r}"
        )


def _fixed_camera_posture_mismatch(
    robot: Any,
    tolerance_deg: float,
) -> str | None:
    """Return a calibrated-posture mismatch, raising on invalid robot state."""

    try:
        model = robot.model()
        position = np.asarray(robot.get_state().position, dtype=np.float64)
    except Exception as exc:
        raise AutoGrabError(
            f"cannot read the torso/head state required by camera calibration: {exc}"
        ) from exc
    if position.ndim != 1 or not np.all(np.isfinite(position)):
        raise AutoGrabError("robot joint state is invalid for camera-posture validation")

    expected = {
        "torso": (
            getattr(model, "torso_idx", None),
            np.deg2rad(np.asarray(EXPECTED_TORSO_POSITION_DEG, dtype=np.float64)),
        ),
        "head": (
            getattr(model, "head_idx", None),
            np.deg2rad(np.asarray(EXPECTED_HEAD_POSITION_DEG, dtype=np.float64)),
        ),
    }
    first_mismatch: str | None = None
    for component, (raw_indices, target) in expected.items():
        if raw_indices is None:
            raise AutoGrabError(
                f"robot model does not expose {component}_idx for posture validation"
            )
        indices = np.asarray(raw_indices, dtype=np.int64)
        try:
            current = position[indices]
        except (IndexError, TypeError) as exc:
            raise AutoGrabError(
                f"robot {component} indices do not match the current joint state"
            ) from exc
        if current.shape != target.shape or not np.all(np.isfinite(current)):
            raise AutoGrabError(
                f"robot {component} state does not match the calibrated joint layout"
            )
        delta_deg = np.abs(np.rad2deg(current - target))
        maximum = float(np.max(delta_deg))
        if maximum > tolerance_deg and first_mismatch is None:
            joint = int(np.argmax(delta_deg))
            first_mismatch = (
                f"fixed camera calibration requires {component} posture within "
                f"{tolerance_deg:.1f} deg; {component}[{joint}] differs by "
                f"{maximum:.3f} deg"
            )
    return first_mismatch


def _validate_fixed_camera_posture(robot: Any, tolerance_deg: float) -> None:
    """Confirm that FK still matches the posture used by camera calibration."""

    mismatch = _fixed_camera_posture_mismatch(robot, tolerance_deg)
    if mismatch is not None:
        raise AutoGrabError(mismatch)


def _ensure_fixed_camera_posture(
    robot: Any,
    grabbing: ModuleType,
    tolerance_deg: float,
) -> bool:
    """Restore the calibrated torso/head pose only when it is out of tolerance."""

    mismatch = _fixed_camera_posture_mismatch(robot, tolerance_deg)
    if mismatch is None:
        print(
            "[auto-grab] fixed camera posture is already within "
            f"{tolerance_deg:.1f} deg; skipping calibration position command"
        )
        return False

    move_to_calibration = getattr(
        grabbing,
        "move_to_camera_calibration_posture",
        None,
    )
    if not callable(move_to_calibration):
        raise AutoGrabError(
            "packaged grasp module does not provide the required camera "
            "calibration posture command"
        )

    print(f"[auto-grab] {mismatch}; restoring the calibrated torso/head pose")
    torso_target = np.deg2rad(
        np.asarray(EXPECTED_TORSO_POSITION_DEG, dtype=np.float64)
    )
    head_target = np.deg2rad(
        np.asarray(EXPECTED_HEAD_POSITION_DEG, dtype=np.float64)
    )
    if not bool(move_to_calibration(robot, torso_target, head_target)):
        raise AutoGrabError(
            "camera calibration Joint Position command did not finish with OK "
            "feedback; mobility stream was not started"
        )

    _validate_fixed_camera_posture(robot, tolerance_deg)
    print("[auto-grab] fixed camera posture verified after position command")
    return True


def _horizontal_grasp_family_gate_reason(
    base_pose: BasePoseDiagnostic | None,
) -> str | None:
    """Reject poses outside the 90-degree family used by this arm posture."""

    if base_pose is None or base_pose.yaw_mod_180_deg is None:
        return None
    reference = base_pose.canonical_reference_deg
    residual = base_pose.canonical_residual_deg
    if reference != 90 or residual is None or not np.isfinite(residual):
        return (
            "unsupported_grasp_orientation: current packaged grasp posture "
            "requires horizontal long-axis reference=90 deg"
        )
    return None


def _grasp_yaw_gate_reason(
    base_pose: BasePoseDiagnostic | None,
    target_yaw_rad: float,
    maximum_residual_deg: float,
) -> str | None:
    """Fail closed unless yaw matches the current horizontal-grasp posture."""

    if base_pose is None or base_pose.yaw_mod_180_deg is None:
        return "grasp_yaw_unavailable"
    family_failure = _horizontal_grasp_family_gate_reason(base_pose)
    if family_failure is not None:
        return family_failure
    measured_yaw_rad = math.radians(float(base_pose.yaw_mod_180_deg))
    if not np.isfinite(measured_yaw_rad):
        return "grasp_yaw_unavailable"
    residual_deg = math.degrees(
        line_angle_difference_rad(measured_yaw_rad, target_yaw_rad)
    )
    if abs(residual_deg) > maximum_residual_deg:
        return (
            "grasp_yaw_out_of_tolerance: "
            f"target={math.degrees(target_yaw_rad):.1f} deg "
            f"residual={residual_deg:+.2f} deg "
            f"limit={maximum_residual_deg:.2f} deg"
        )
    return None


def _wait_for_mobile_stop(
    robot: Any,
    config: AutoGrabConfig,
    *,
    clock: Callable[[], float],
) -> None:
    """Require fresh measured wheel velocities to settle before arm motion."""

    try:
        model = robot.model()
        indices = np.asarray(model.mobility_idx, dtype=np.int64)
    except Exception as exc:
        raise AutoGrabError(
            f"cannot resolve mobility joints for stop verification: {exc}"
        ) from exc
    if indices.ndim != 1 or indices.size == 0:
        raise AutoGrabError("robot model has no mobility joints for stop verification")

    condition = threading.Condition()
    settled_since_s: float | None = None
    settled = False
    latest_arrival_s: float | None = None
    latest_speed = float("inf")
    latest_ready = False
    callback_error: str | None = None

    def state_callback(state: Any) -> None:
        nonlocal callback_error
        nonlocal latest_arrival_s
        nonlocal latest_ready
        nonlocal latest_speed
        nonlocal settled
        nonlocal settled_since_s
        arrival_s = clock()
        try:
            velocity = np.asarray(state.velocity, dtype=np.float64)
            is_ready = np.asarray(state.is_ready, dtype=np.bool_)
            mobility_velocity = velocity[indices]
            mobility_ready = is_ready[indices]
        except Exception as exc:
            with condition:
                callback_error = f"cannot read mobility state: {exc}"
                latest_arrival_s = arrival_s
                condition.notify_all()
            return
        with condition:
            latest_arrival_s = arrival_s
            if (
                mobility_velocity.shape != indices.shape
                or mobility_ready.shape != indices.shape
                or not np.all(np.isfinite(mobility_velocity))
            ):
                callback_error = "mobility state is invalid during stop verification"
                condition.notify_all()
                return
            latest_speed = float(np.max(np.abs(mobility_velocity)))
            latest_ready = bool(np.all(mobility_ready))
            if (
                latest_ready
                and latest_speed <= config.mobile_stop_velocity_threshold_radps
            ):
                if settled_since_s is None:
                    settled_since_s = arrival_s
                settled = (
                    arrival_s - settled_since_s
                    >= config.mobile_stop_min_duration_s
                )
            else:
                settled_since_s = None
                settled = False
            condition.notify_all()

    deadline = clock() + config.mobile_stop_timeout_s
    update_started = False
    try:
        try:
            robot.start_state_update(
                state_callback,
                config.mobile_state_update_rate_hz,
            )
            update_started = True
        except Exception as exc:
            raise AutoGrabError(
                f"cannot start mobility-state stop verification: {exc}"
            ) from exc

        with condition:
            while not settled:
                now_s = clock()
                if callback_error is not None:
                    raise AutoGrabError(callback_error)
                if (
                    latest_arrival_s is not None
                    and now_s - latest_arrival_s
                    > config.mobile_state_stale_after_s
                ):
                    raise AutoGrabError(
                        "mobility state became stale before grasp handoff"
                    )
                remaining_s = deadline - now_s
                if remaining_s <= 0.0:
                    readiness = "ready" if latest_ready else "not-ready"
                    raise AutoGrabError(
                        "mobile base did not settle before grasp: "
                        f"max wheel speed={latest_speed:.3f} rad/s, "
                        f"limit={config.mobile_stop_velocity_threshold_radps:.3f} "
                        f"rad/s, state={readiness}"
                    )
                condition.wait(
                    timeout=min(
                        remaining_s,
                        config.mobile_state_stale_after_s,
                    )
                )
    finally:
        if update_started:
            try:
                robot.stop_state_update()
            except Exception as exc:
                raise AutoGrabError(
                    f"cannot stop mobility-state verification stream: {exc}"
                ) from exc


class AutoGrabRuntime:
    """Bridge current base-frame poses to one pumped mobile stream and one grasp.

    The same connected robot object is retained from preparation through
    mobility-stream cancellation and the existing start/grab/lift sequence.
    RB-Y1 controller arbitration does not permit the mobility stream and the
    separate body one-shots to execute concurrently, so the stream is kept
    alive through measured base settling and released immediately before the
    body sequence.
    """

    def __init__(
        self,
        config: AutoGrabConfig | None = None,
        *,
        execute: bool = False,
        sdk_module: Any | None = None,
        grabbing_module: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or AutoGrabConfig()
        self._execute = bool(execute)
        self._sdk = sdk_module
        self._grabbing = grabbing_module
        self._clock = clock
        self._servo = MobileVisualServo(self.config.servo)
        self._robot: Any | None = None
        self._stream: RBY1MobilityStream | None = None
        self._pump: RBY1MobilityCommandPump | None = None
        self._started = False
        self._closed = False
        self._handoff_ready = False
        self._grasp_invoked = False
        self._completed = False
        self._last_state: ServoState | None = None

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def grasp_invoked(self) -> bool:
        return self._grasp_invoked

    def start(self) -> None:
        """Connect, restore the camera pose, and start one zeroed mobility pump."""

        if not self._execute:
            raise AutoGrabError(
                "robot execution is disabled; automatic box-picking execution is required"
            )
        if self._started:
            raise AutoGrabError("auto-grab runtime has already started")
        if self._closed:
            raise AutoGrabError("auto-grab runtime is already closed")
        try:
            if self._sdk is None:
                try:
                    self._sdk = importlib.import_module("rby1_sdk")
                except ImportError as exc:
                    raise AutoGrabError(
                        "rby1_sdk is required for automatic box-picking on the robot PC"
                    ) from exc
            if self._grabbing is None:
                self._grabbing = _load_grabbing_sequence()

            self._robot = self._sdk.create_robot(self.config.address, "m")
            connected = self._robot.connect()
            if connected is False or not self._robot.is_connected():
                raise AutoGrabError(
                    f"failed to connect to RB-Y1 at {self.config.address}"
                )
            _validate_robot_identity(self._robot)
            self._grabbing.prepare_robot(self._robot, power=self.config.power)
            _ensure_fixed_camera_posture(
                self._robot,
                self._grabbing,
                self.config.fixed_posture_tolerance_deg,
            )
            move_to_mobile_ready = getattr(
                self._grabbing,
                "move_arms_to_mobile_ready_pose",
                None,
            )
            if not callable(move_to_mobile_ready):
                raise AutoGrabError(
                    "packaged grasp module does not provide the required "
                    "mobile-ready arm command"
                )
            if not bool(move_to_mobile_ready(self._robot)):
                raise AutoGrabError(
                    "mobile-ready Joint Position command did not finish with OK feedback; "
                    "mobility stream was not started"
                )
            _validate_fixed_camera_posture(
                self._robot,
                self.config.fixed_posture_tolerance_deg,
            )
            self._stream = RBY1MobilityStream(
                self._robot,
                execute=True,
                config=self.config.stream,
                sdk_module=self._sdk,
            ).open()
            self._pump = RBY1MobilityCommandPump(
                self._stream,
                config=self.config.pump,
            )
            self._pump.start()
            print(
                "[auto-grab] mobility pump active: "
                f"rate={self.config.pump.send_rate_hz:.1f} Hz, "
                f"hold={self.config.stream.control_hold_time_s:.2f} s, "
                f"stale-zero={self.config.pump.command_stale_after_s:.2f} s"
            )
            started = self._servo.start(self._clock())
            self._started = True
            self._report_transition(started)
        except AutoGrabError:
            self._cleanup_after_start_failure()
            raise
        except Exception as exc:
            self._cleanup_after_start_failure()
            raise AutoGrabError(f"failed to initialize RB-Y1 auto-grab: {exc}") from exc

    def update(
        self,
        base_pose: BasePoseDiagnostic | None,
        *,
        pose_timestamp_s: float,
        now_s: float,
    ) -> bool:
        """Publish one bounded XY+yaw command for the mobility pump."""

        if (
            not self._started
            or self._closed
            or self._stream is None
            or self._pump is None
        ):
            raise AutoGrabError("auto-grab runtime is not active")
        orientation_failure = _horizontal_grasp_family_gate_reason(base_pose)
        measurement = None
        if (
            orientation_failure is None
            and base_pose is not None
            and base_pose.yaw_mod_180_deg is not None
        ):
            x_m, y_m, _ = base_pose.box_center_xyz_m
            measurement = PoseMeasurement(
                (x_m, y_m),
                timestamp_s=pose_timestamp_s,
                long_axis_yaw_rad=math.radians(base_pose.yaw_mod_180_deg),
            )
        if orientation_failure is None:
            decision = self._servo.step(measurement, now_s=now_s)
        else:
            decision = self._servo.abort(orientation_failure, now_s)
        if decision.handoff_ready:
            yaw_failure = _grasp_yaw_gate_reason(
                base_pose,
                self.config.servo.target_long_axis_yaw_rad,
                self.config.max_grasp_yaw_residual_deg,
            )
            if yaw_failure is not None:
                decision = self._servo.abort(yaw_failure, now_s)
        try:
            self._pump.publish(decision.command)
        except Exception as exc:
            self._servo.abort("mobility_stream_error", now_s)
            raise AutoGrabError(f"RB-Y1 mobility stream failed: {exc}") from exc
        self._report_transition(decision)
        if decision.state is ServoState.ABORTED:
            raise AutoGrabError(
                f"mobile visual servo aborted before grasp: {decision.reason}"
            )
        if decision.handoff_ready:
            self._handoff_ready = True
        return decision.handoff_ready

    def handoff(self) -> None:
        """Latch/settle/release mobility, then run the existing grasp once."""

        if not self._started or self._closed:
            raise AutoGrabError("auto-grab runtime is not active")
        if not self._handoff_ready:
            raise AutoGrabError("grasp handoff requested before stable arrival")
        if self._grasp_invoked:
            raise AutoGrabError("grasp sequence has already been invoked")
        if (
            self._stream is None
            or self._pump is None
            or self._robot is None
            or self._grabbing is None
        ):
            raise AutoGrabError("auto-grab handoff resources are unavailable")

        try:
            self._pump.latch_zero_and_wait()
        except Exception as exc:
            raise AutoGrabError(
                f"cannot zero the active mobility stream for handoff: {exc}"
            ) from exc
        _wait_for_mobile_stop(
            self._robot,
            self.config,
            clock=self._clock,
        )
        try:
            self._pump.raise_if_failed()
            pump_send_count = self._pump.send_count
            pump_max_gap_s = self._pump.max_send_gap_s
            self._pump.stop_and_release()
        except Exception as exc:
            raise AutoGrabError(
                f"cannot release the stopped mobility stream for body handoff: {exc}"
            ) from exc
        self._pump = None
        self._stream = None
        print(
            "[auto-grab] mobility stream released for body handoff: "
            f"sends={pump_send_count}, max-gap={pump_max_gap_s * 1000.0:.1f} ms"
        )
        _validate_fixed_camera_posture(
            self._robot,
            self.config.fixed_posture_tolerance_deg,
        )
        self._grasp_invoked = True
        print("[auto-grab] base stopped; starting packaged grasp sequence")
        try:
            grasp_succeeded = bool(
                self._grabbing.run_grabbing_sequence(self._robot)
            )
        except Exception as exc:
            raise AutoGrabError(f"packaged grasp sequence failed: {exc}") from exc
        if not grasp_succeeded:
            raise AutoGrabError("packaged grasp sequence reported failure")
        self._completed = True
        print("[auto-grab] box grasp and lift completed")

    def close(self) -> None:
        """Stop any active mobility stream and disconnect exactly once."""

        if self._closed:
            return
        stream_error: Exception | None = None
        disconnect_error: Exception | None = None
        try:
            if self._pump is not None:
                pump = self._pump
                try:
                    pump.close()
                except Exception as exc:  # preserve disconnect on shutdown failure
                    stream_error = exc
                finally:
                    if pump.is_closed:
                        self._pump = None
                        self._stream = None
            elif self._stream is not None:
                try:
                    self._stream.close()
                except Exception as exc:
                    stream_error = exc
                finally:
                    self._stream = None
        finally:
            if self._robot is not None:
                disconnect = getattr(self._robot, "disconnect", None)
                if callable(disconnect):
                    try:
                        disconnect()
                    except Exception as exc:
                        disconnect_error = exc
                    else:
                        self._robot = None
                else:
                    self._robot = None
        self._closed = (
            self._pump is None
            and self._stream is None
            and self._robot is None
        )
        if disconnect_error is not None:
            raise AutoGrabError(
                f"failed to disconnect RB-Y1 cleanly: {disconnect_error}"
            ) from disconnect_error
        if stream_error is not None:
            raise AutoGrabError(
                f"failed to stop RB-Y1 mobility stream cleanly: {stream_error}"
            ) from stream_error

    def _cleanup_after_start_failure(self) -> None:
        if self._pump is not None:
            pump = self._pump
            try:
                pump.close()
            except Exception:
                pass
            if pump.is_closed:
                self._pump = None
                self._stream = None
        elif self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._robot is not None:
            disconnect = getattr(self._robot, "disconnect", None)
            if callable(disconnect):
                try:
                    disconnect()
                except Exception:
                    pass
            self._robot = None

    def _report_transition(self, decision: ServoDecision) -> None:
        if decision.state is self._last_state:
            return
        self._last_state = decision.state
        target_x, target_y = self.config.servo.target_xy_m
        target_yaw_deg = math.degrees(
            self.config.servo.target_long_axis_yaw_rad
        )
        print(
            f"[auto-grab] state={decision.state.value} reason={decision.reason} "
            f"target=({target_x:.3f}, {target_y:.3f}) m, "
            f"long-axis yaw={target_yaw_deg:.1f} deg"
        )


__all__ = [
    "AutoGrabConfig",
    "AutoGrabError",
    "AutoGrabRuntime",
    "DEFAULT_ROBOT_ADDRESS",
    "EXPECTED_ROBOT_MODEL",
    "EXPECTED_ROBOT_VERSION",
    "EXPECTED_HEAD_POSITION_DEG",
    "EXPECTED_TORSO_POSITION_DEG",
]
