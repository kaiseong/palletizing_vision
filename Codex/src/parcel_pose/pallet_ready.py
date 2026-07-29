"""One-shot RB-Y1 slot-1 ready-pose transition.

This module is deliberately independent of the pallet live runtime.  Importing
it has no robot-side effect and does not import ``rby1_sdk``.  The public helper
requires an exclusive, initially disconnected robot and owns that connection
for the duration of one conditional ready-pose transition.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import sys
from typing import Any, Mapping

import numpy as np

from .pallet_control import (
    EXPECTED_ROBOT_MODEL,
    EXPECTED_ROBOT_VERSION,
    PalletControlConfig,
    ReadyPose,
)


_SERVO_DEVICE_PATTERN = ".*"
_CANCEL_TIMEOUT_MS = 2_000


class Slot1ReadyError(RuntimeError):
    """Raised when the standalone slot-1 ready transition cannot be verified."""


@dataclass(frozen=True, slots=True)
class _ReadyMeasurement:
    maximum_error_rad: float
    target_joints_ready: bool

    def within(self, tolerance_rad: float) -> bool:
        return (
            self.target_joints_ready
            and math.isfinite(self.maximum_error_rad)
            and self.maximum_error_rad <= float(tolerance_rad)
        )


def _load_sdk(sdk_module: Any | None) -> Any:
    if sdk_module is not None:
        return sdk_module
    try:
        return importlib.import_module("rby1_sdk")
    except ImportError as exc:
        raise Slot1ReadyError(
            "rby1_sdk is required for the slot-1 ready transition"
        ) from exc


def _connect(robot: Any) -> None:
    is_connected = getattr(robot, "is_connected", None)
    connect = getattr(robot, "connect", None)
    if not callable(is_connected) or not callable(connect):
        raise Slot1ReadyError("robot must provide connect() and is_connected()")
    if bool(is_connected()):
        return
    if connect() is False or not bool(is_connected()):
        raise Slot1ReadyError("failed to connect to RB-Y1")


def _validate_identity(robot: Any) -> None:
    get_info = getattr(robot, "get_robot_info", None)
    if not callable(get_info):
        raise Slot1ReadyError("connected robot cannot report its model and version")
    info = get_info()
    model_name = str(getattr(info, "robot_model_name", "")).strip().upper()
    version = (
        str(getattr(info, "robot_model_version", "")).strip().lower().removeprefix("v")
    )
    if model_name != EXPECTED_ROBOT_MODEL:
        raise Slot1ReadyError(
            "slot-1 ready requires RB-Y1 Model M; controller reports " f"{model_name!r}"
        )
    if version != EXPECTED_ROBOT_VERSION and not version.startswith(
        f"{EXPECTED_ROBOT_VERSION}."
    ):
        raise Slot1ReadyError(
            "slot-1 ready requires RB-Y1 Model M v1.2; controller reports "
            f"version {version!r}"
        )


def _prepare_robot(robot: Any, power: str) -> None:
    required_methods = (
        "is_power_on",
        "power_on",
        "is_servo_on",
        "servo_on",
        "reset_fault_control_manager",
        "enable_control_manager",
    )
    missing = [
        name for name in required_methods if not callable(getattr(robot, name, None))
    ]
    if missing:
        raise Slot1ReadyError(
            "robot preparation API is incomplete: " + ", ".join(sorted(missing))
        )

    if not bool(robot.is_power_on(power)) and not bool(robot.power_on(power)):
        raise Slot1ReadyError(f"failed to power devices matching {power!r}")
    if not bool(robot.is_servo_on(_SERVO_DEVICE_PATTERN)) and not bool(
        robot.servo_on(_SERVO_DEVICE_PATTERN)
    ):
        raise Slot1ReadyError("failed to enable RB-Y1 servos")
    robot.reset_fault_control_manager()
    if not bool(robot.enable_control_manager()):
        raise Slot1ReadyError("failed to enable the RB-Y1 control manager")


def _component_targets(pose: ReadyPose) -> tuple[tuple[str, np.ndarray], ...]:
    return (
        ("torso", np.asarray(pose.torso_rad, dtype=np.float64)),
        ("right_arm", np.asarray(pose.right_arm_rad, dtype=np.float64)),
        ("left_arm", np.asarray(pose.left_arm_rad, dtype=np.float64)),
        ("head", np.asarray(pose.head_rad, dtype=np.float64)),
    )


def _model_indices(model: Any, component: str, length: int) -> np.ndarray:
    raw_indices = getattr(model, f"{component}_idx", None)
    if raw_indices is None:
        raise Slot1ReadyError(f"robot model does not expose {component}_idx")
    try:
        indices = np.asarray(raw_indices, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise Slot1ReadyError(f"robot {component}_idx is invalid") from exc
    if indices.shape != (length,) or np.any(indices < 0):
        raise Slot1ReadyError(
            f"robot {component}_idx must contain exactly {length} nonnegative indices"
        )
    return indices


def _read_ready_measurement(
    robot: Any,
    model: Any,
    pose: ReadyPose,
) -> _ReadyMeasurement:
    get_state = getattr(robot, "get_state", None)
    if not callable(get_state):
        raise Slot1ReadyError("robot does not provide synchronous get_state()")
    try:
        state = get_state()
        position = np.asarray(state.position, dtype=np.float64)
    except Exception as exc:
        raise Slot1ReadyError(
            f"cannot read the current robot joint state: {exc}"
        ) from exc
    if position.ndim != 1 or not np.all(np.isfinite(position)):
        raise Slot1ReadyError(
            "robot joint position must be a finite one-dimensional array"
        )

    raw_ready = getattr(state, "is_ready", None)
    if raw_ready is None:
        raise Slot1ReadyError(
            "robot joint is_ready state is required before any slot-1 ready motion"
        )
    ready_flags: np.ndarray | None = None
    try:
        ready_flags = np.asarray(raw_ready, dtype=np.bool_)
    except (TypeError, ValueError) as exc:
        raise Slot1ReadyError("robot joint is_ready state is invalid") from exc
    if ready_flags.shape != position.shape:
        raise Slot1ReadyError(
            "robot joint is_ready state must match the position vector shape"
        )

    maximum_error = 0.0
    target_joints_ready = True
    for component, target in _component_targets(pose):
        indices = _model_indices(model, component, target.size)
        if indices.size and int(np.max(indices)) >= position.size:
            raise Slot1ReadyError(
                f"robot {component}_idx does not fit the current joint state"
            )
        current = position[indices]
        maximum_error = max(maximum_error, float(np.max(np.abs(current - target))))
        if ready_flags is not None:
            target_joints_ready = target_joints_ready and bool(
                np.all(ready_flags[indices])
            )

    return _ReadyMeasurement(
        maximum_error_rad=maximum_error,
        target_joints_ready=target_joints_ready,
    )


def _build_ready_command(
    sdk: Any,
    pose: ReadyPose,
    minimum_time_s: float,
) -> Any:
    minimum_time_s = max(5.0, float(minimum_time_s))

    def position_command(target: tuple[float, ...]) -> Any:
        return (
            sdk.JointPositionCommandBuilder()
            .set_minimum_time(minimum_time_s)
            .set_position(np.asarray(target, dtype=np.float64))
        )

    body = (
        sdk.BodyComponentBasedCommandBuilder()
        .set_torso_command(position_command(pose.torso_rad))
        .set_right_arm_command(position_command(pose.right_arm_rad))
        .set_left_arm_command(position_command(pose.left_arm_rad))
    )
    component = (
        sdk.ComponentBasedCommandBuilder()
        .set_body_command(body)
        .set_head_command(sdk.HeadCommandBuilder(position_command(pose.head_rad)))
    )
    return sdk.RobotCommandBuilder().set_command(component)


def _cancel_handler(handler: Any) -> tuple[str, ...]:
    errors: list[str] = []
    cancel = getattr(handler, "cancel", None)
    if callable(cancel):
        try:
            cancel()
        except Exception as exc:
            errors.append(f"cancel failed: {exc}")
    else:
        errors.append("handler does not provide cancel()")

    wait_for = getattr(handler, "wait_for", None)
    if callable(wait_for):
        try:
            if wait_for(_CANCEL_TIMEOUT_MS) is False:
                errors.append("cancel completion timed out")
        except Exception as exc:
            errors.append(f"cancel wait failed: {exc}")
    else:
        errors.append("handler does not provide wait_for()")
    return tuple(errors)


def _add_cancel_notes(exc: BaseException, errors: tuple[str, ...]) -> None:
    for message in errors:
        add_note = getattr(exc, "add_note", None)
        if callable(add_note):
            add_note(message)


def _send_ready_once(
    robot: Any,
    sdk: Any,
    command: Any,
    timeout_ms: int,
) -> None:
    send_command = getattr(robot, "send_command", None)
    if not callable(send_command):
        raise Slot1ReadyError("robot does not provide send_command()")
    try:
        handler = send_command(command)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        raise Slot1ReadyError(
            f"failed to send the slot-1 ready command: {exc}"
        ) from exc

    try:
        wait_for = getattr(handler, "wait_for", None)
        get_feedback = getattr(handler, "get", None)
        if not callable(wait_for) or not callable(get_feedback):
            raise Slot1ReadyError(
                "slot-1 ready command handler lacks wait_for() or get()"
            )
        if wait_for(timeout_ms) is False:
            raise Slot1ReadyError(
                f"slot-1 ready command timed out after {timeout_ms} ms"
            )
        feedback = get_feedback()
        try:
            expected_finish = sdk.RobotCommandFeedback.FinishCode.Ok
            actual_finish = feedback.finish_code
        except AttributeError as exc:
            raise Slot1ReadyError(
                "slot-1 ready command returned malformed finish feedback"
            ) from exc
        if actual_finish != expected_finish:
            raise Slot1ReadyError(
                "slot-1 ready command did not finish with Ok feedback: "
                f"{actual_finish!r}"
            )
    except KeyboardInterrupt as exc:
        _add_cancel_notes(exc, _cancel_handler(handler))
        raise
    except Slot1ReadyError as exc:
        _add_cancel_notes(exc, _cancel_handler(handler))
        raise
    except Exception as exc:
        wrapped = Slot1ReadyError(f"slot-1 ready command feedback failed: {exc}")
        _add_cancel_notes(wrapped, _cancel_handler(handler))
        raise wrapped from exc


def _disconnect(robot: Any) -> None:
    disconnect = getattr(robot, "disconnect", None)
    if not callable(disconnect):
        raise Slot1ReadyError("robot does not provide disconnect()")
    result = disconnect()
    if result is False:
        raise Slot1ReadyError("RB-Y1 disconnect() reported failure")
    is_connected = getattr(robot, "is_connected", None)
    if callable(is_connected) and bool(is_connected()):
        raise Slot1ReadyError("RB-Y1 remains connected after disconnect()")


def ensure_slot1_ready_from_config(
    root_config: Mapping[str, Any],
    *,
    address: str,
    power: str,
    sdk_module: Any | None = None,
    robot: Any | None = None,
) -> bool:
    """Conditionally send one full-body slot-1 Joint Position command.

    Returns ``True`` only when a robot command was sent.  ``False`` means a
    fresh synchronous measurement already placed every configured target joint
    within tolerance and every target joint reported ready, so zero ready-pose
    commands were sent.

    An injected ``robot`` is accepted for verification only when it is initially
    disconnected.  Rejecting an active connection prevents this standalone
    helper from resetting another publisher's control manager or disconnecting
    a caller-owned session.
    """

    if not isinstance(root_config, Mapping):
        raise TypeError("root_config must be a mapping")
    if not isinstance(address, str) or not address.strip():
        raise ValueError("address must be a non-empty string")
    if not isinstance(power, str) or not power.strip():
        raise ValueError("power must be a non-empty device pattern")

    config = PalletControlConfig.from_root_config(
        root_config,
        address_override=address,
    )
    owned_robot = robot
    primary_exception: BaseException | None = None
    connection_owned = False
    try:
        sdk = _load_sdk(sdk_module)
        if owned_robot is None:
            create_robot = getattr(sdk, "create_robot", None)
            if not callable(create_robot):
                raise Slot1ReadyError("rby1_sdk does not provide create_robot()")
            owned_robot = create_robot(config.address, "m")
            if owned_robot is None:
                raise Slot1ReadyError("rby1_sdk.create_robot() returned no robot")

        is_connected = getattr(owned_robot, "is_connected", None)
        if not callable(is_connected):
            raise Slot1ReadyError("robot must provide is_connected()")
        if bool(is_connected()):
            raise Slot1ReadyError(
                "slot-1 ready requires an exclusive disconnected robot; an "
                "existing command owner must stop and disconnect first"
            )
        _connect(owned_robot)
        connection_owned = True
        _validate_identity(owned_robot)
        model_method = getattr(owned_robot, "model", None)
        if not callable(model_method):
            raise Slot1ReadyError("robot does not provide model()")
        model = model_method()
        reported_model = str(getattr(model, "model_name", "")).strip().upper()
        if reported_model and reported_model != EXPECTED_ROBOT_MODEL:
            raise Slot1ReadyError(
                f"robot joint model is {reported_model!r}, expected Model M"
            )

        before = _read_ready_measurement(owned_robot, model, config.ready_pose)
        if before.within(config.ready_tolerance_rad):
            print(
                "[pallet] slot-1 ready posture is already within "
                f"{math.degrees(config.ready_tolerance_rad):.1f} deg; "
                "skipping the Position command",
                file=sys.stderr,
                flush=True,
            )
            return False

        _prepare_robot(owned_robot, power)
        minimum_time_s = max(5.0, config.ready_transition_minimum_time_s)
        print(
            "[pallet] slot-1 ready transition required "
            f"(max_error={math.degrees(before.maximum_error_rad):.3f} deg, "
            f"all_target_joints_ready={before.target_joints_ready}); sending "
            f"one all-joint Position command with minimum_time={minimum_time_s:.1f}s",
            file=sys.stderr,
            flush=True,
        )
        command = _build_ready_command(sdk, config.ready_pose, minimum_time_s)
        timeout_ms = max(1, int(math.ceil(config.transition_timeout_s * 1_000.0)))
        _send_ready_once(owned_robot, sdk, command, timeout_ms)

        after = _read_ready_measurement(owned_robot, model, config.ready_pose)
        if not after.within(config.ready_tolerance_rad):
            raise Slot1ReadyError(
                "slot-1 ready command completed but measured verification failed: "
                f"max_error_deg={math.degrees(after.maximum_error_rad):.3f}, "
                f"tolerance_deg={math.degrees(config.ready_tolerance_rad):.3f}, "
                f"all_target_joints_ready={after.target_joints_ready}"
            )
        print(
            "[pallet] slot-1 ready Position command finished OK and the measured "
            "posture was verified",
            file=sys.stderr,
            flush=True,
        )
        return True
    except BaseException as exc:
        primary_exception = exc
        raise
    finally:
        if owned_robot is not None and connection_owned:
            try:
                _disconnect(owned_robot)
            except Exception as exc:
                if primary_exception is None:
                    raise
                add_note = getattr(primary_exception, "add_note", None)
                if callable(add_note):
                    add_note(f"RB-Y1 disconnect failed during cleanup: {exc}")


__all__ = ["Slot1ReadyError", "ensure_slot1_ready_from_config"]
