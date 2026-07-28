from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import types

import numpy as np
import pytest

from parcel_pose.auto_grab import (
    AutoGrabConfig,
    AutoGrabError,
    AutoGrabRuntime,
    _load_grabbing_box,
)
from parcel_pose.evaluation import BasePoseDiagnostic
from parcel_pose.mobile_servo import RBY1CommandPumpConfig, ServoConfig


CALIBRATED_TORSO_DEG = (0.0, 55.0, -59.988, 6.532, 0.0, 0.0)
CALIBRATED_HEAD_DEG = (0.0, 49.846)


def test_importing_auto_grab_keeps_robot_sdk_lazy_in_isolated_python() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    program = """
import sys
sys.path.insert(0, sys.argv[1])
import parcel_pose.auto_grab
assert 'rby1_sdk' not in sys.modules
assert 'parcel_pose.grabbing' not in sys.modules
"""

    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, str(source_root)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_default_grasp_sequence_is_packaged_inside_codex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_sdk = types.ModuleType("rby1_sdk")
    fake_sdk.RobotCommandFeedback = SimpleNamespace(
        FinishCode=SimpleNamespace(Ok=object())
    )
    monkeypatch.setitem(sys.modules, "rby1_sdk", fake_sdk)
    monkeypatch.delitem(sys.modules, "parcel_pose.grabbing", raising=False)

    try:
        module = _load_grabbing_box()

        module_path = Path(module.__file__).resolve()
        assert module_path.name == "grabbing.py"
        assert module_path.parent.name == "parcel_pose"
        assert module_path.parents[2].name == "Codex"
    finally:
        sys.modules.pop("parcel_pose.grabbing", None)
        import parcel_pose

        parcel_pose.__dict__.pop("grabbing", None)


def _mobility_feedback(*, status: int = 2, finish_code: int = 0):
    valid = SimpleNamespace(valid=True)
    mobility = SimpleNamespace(valid=True, se2_velocity_command=valid)
    component = SimpleNamespace(valid=True, mobility_command=mobility)
    return SimpleNamespace(
        valid=True,
        status=status,
        finish_code=finish_code,
        component_based_command=component,
    )


def _pose(
    x: float = 0.740,
    y: float = 0.0,
    *,
    canonical_reference_deg: int | None = 90,
    canonical_residual_deg: float | None = 0.0,
    yaw_deg: float | None = None,
) -> BasePoseDiagnostic:
    yaw = (
        float(yaw_deg)
        if yaw_deg is not None
        else float(canonical_reference_deg or 0)
        + float(canonical_residual_deg or 0.0)
    )
    return BasePoseDiagnostic(
        box_center_xyz_m=(x, y, 0.82),
        top_center_xyz_m=(x, y, 0.90),
        yaw_mod_180_deg=yaw % 180.0,
        yaw_signed_deg=((yaw + 90.0) % 180.0) - 90.0,
        canonical_reference_deg=canonical_reference_deg,
        canonical_residual_deg=canonical_residual_deg,
        registration="nominal_unverified",
    )


class _HeaderBuilder:
    def set_control_hold_time(self, value):
        return self


class _SE2Builder:
    def __init__(self) -> None:
        self.velocity = (0.0, 0.0, 0.0)

    def set_command_header(self, value):
        return self

    def set_minimum_time(self, value):
        return self

    def set_velocity(self, linear, angular):
        self.velocity = (
            *(float(value) for value in np.asarray(linear)),
            float(angular),
        )
        return self

    def set_acceleration_limit(self, linear, angular):
        return self


class _ComponentBuilder:
    def __init__(self) -> None:
        self.mobility = None

    def set_mobility_command(self, value):
        self.mobility = value
        return self


class _RobotCommandBuilder:
    def __init__(self) -> None:
        self.component = None

    def set_command(self, value):
        self.component = value
        return self


class _CommandStream:
    def __init__(
        self,
        events: list[tuple],
        *,
        wait_result: bool = True,
        fail_after_sends: int | None = None,
    ) -> None:
        self.events = events
        self.wait_result = wait_result
        self.fail_after_sends = fail_after_sends
        self.send_count = 0

    def send_command(self, command, timeout_ms):
        if (
            self.fail_after_sends is not None
            and self.send_count >= self.fail_after_sends
        ):
            self.events.append(("stream_send_failed", timeout_ms))
            raise RuntimeError("simulated command stream expiry")
        self.send_count += 1
        velocity = command.component.mobility.velocity
        self.events.append(("velocity", velocity))
        return _mobility_feedback()

    def cancel(self):
        self.events.append(("stream_cancel",))

    def wait_for(self, timeout_ms):
        self.events.append(("stream_wait", timeout_ms))
        return self.wait_result


class _StepClock:
    def __init__(self, step_s: float = 0.05) -> None:
        self.value = -step_s
        self.step_s = step_s

    def __call__(self) -> float:
        self.value += self.step_s
        return self.value


class _Robot:
    def __init__(
        self,
        events: list[tuple],
        *,
        model: str = "M",
        version: str = "1.2",
        stream_wait_result: bool = True,
        torso_deg: tuple[float, ...] = CALIBRATED_TORSO_DEG,
        head_deg: tuple[float, ...] = CALIBRATED_HEAD_DEG,
        disconnect_failures: int = 0,
        mobility_velocity_radps: tuple[float, float] = (0.0, 0.0),
        state_update_samples: int = 9,
        mobility_ready: bool = True,
        stream_fail_after_sends: int | None = None,
    ) -> None:
        self.events = events
        self.connected = False
        self.info = SimpleNamespace(
            robot_model_name=model,
            robot_model_version=version,
        )
        self.stream = _CommandStream(
            events,
            wait_result=stream_wait_result,
            fail_after_sends=stream_fail_after_sends,
        )
        self.robot_model = SimpleNamespace(
            model_name="m",
            torso_idx=np.arange(6, dtype=np.int64),
            head_idx=np.arange(6, 8, dtype=np.int64),
            mobility_idx=np.arange(8, 10, dtype=np.int64),
        )
        self.position = np.deg2rad(
            np.asarray((*torso_deg, *head_deg, 0.0, 0.0), dtype=np.float64)
        )
        self.velocity = np.asarray(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, *mobility_velocity_radps),
            dtype=np.float64,
        )
        self.disconnect_failures = disconnect_failures
        self.state_update_samples = state_update_samples
        self.mobility_ready = mobility_ready

    def connect(self):
        self.events.append(("connect", self))
        self.connected = True
        return True

    def is_connected(self):
        return self.connected

    def get_robot_info(self):
        self.events.append(("identity", self))
        return self.info

    def model(self):
        self.events.append(("model", self))
        return self.robot_model

    def get_state(self):
        self.events.append(("get_state", self))
        return SimpleNamespace(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            is_ready=np.ones_like(self.velocity, dtype=np.bool_),
        )

    def start_state_update(self, callback, rate):
        self.events.append(("state_update_start", rate))
        for _ in range(self.state_update_samples):
            is_ready = np.ones_like(self.velocity, dtype=np.bool_)
            is_ready[self.robot_model.mobility_idx] = self.mobility_ready
            callback(
                SimpleNamespace(
                    velocity=self.velocity.copy(),
                    is_ready=is_ready,
                )
            )

    def stop_state_update(self):
        self.events.append(("state_update_stop",))

    def create_command_stream(self, priority):
        self.events.append(("create_stream", priority, self))
        return self.stream

    def disconnect(self):
        self.events.append(("disconnect", self))
        if self.disconnect_failures:
            self.disconnect_failures -= 1
            raise RuntimeError("simulated disconnect failure")
        self.connected = False


def _sdk(events: list[tuple], robot: _Robot):
    def create_robot(address, model):
        events.append(("create_robot", address, model, robot))
        return robot

    return SimpleNamespace(
        create_robot=create_robot,
        CommandHeaderBuilder=_HeaderBuilder,
        SE2VelocityCommandBuilder=_SE2Builder,
        ComponentBasedCommandBuilder=_ComponentBuilder,
        RobotCommandBuilder=_RobotCommandBuilder,
    )


def _grabbing(
    events: list[tuple],
    *,
    mobile_ready_result: bool = True,
):
    def prepare(robot, *, power):
        events.append(("prepare", power, robot))

    def move_to_mobile_ready(robot):
        events.append(("mobile_ready", robot))
        return mobile_ready_result

    def run(robot):
        events.append(("grasp", robot))
        return True

    return SimpleNamespace(
        prepare_robot=prepare,
        move_arms_to_mobile_ready_pose=move_to_mobile_ready,
        run_grabbing_sequence=run,
    )


def test_execution_must_be_explicit_before_any_robot_is_created() -> None:
    events: list[tuple] = []
    robot = _Robot(events)
    runtime = AutoGrabRuntime(
        sdk_module=_sdk(events, robot),
        grabbing_module=_grabbing(events),
        clock=lambda: 0.0,
    )

    with pytest.raises(AutoGrabError, match="execution is disabled"):
        runtime.start()

    assert events == []


def test_current_grasp_posture_rejects_a_vertical_yaw_target() -> None:
    with pytest.raises(ValueError, match="horizontal long-axis yaw=90"):
        AutoGrabConfig(
            servo=ServoConfig(target_long_axis_yaw_rad=0.0)
        )


@pytest.mark.parametrize(
    ("model", "version", "message"),
    [
        ("A", "1.2", "Model M"),
        ("M", "1.1", "v1.2"),
    ],
)
def test_wrong_robot_identity_disconnects_before_prepare(
    model: str,
    version: str,
    message: str,
) -> None:
    events: list[tuple] = []
    robot = _Robot(events, model=model, version=version)
    runtime = AutoGrabRuntime(
        execute=True,
        sdk_module=_sdk(events, robot),
        grabbing_module=_grabbing(events),
        clock=lambda: 0.0,
    )

    with pytest.raises(AutoGrabError, match=message):
        runtime.start()

    names = [event[0] for event in events]
    assert names[-1] == "disconnect"
    assert "prepare" not in names
    assert "create_stream" not in names


@pytest.mark.parametrize(
    ("torso_deg", "head_deg", "component"),
    [
        (
            (0.0, 57.0, -59.988, 6.532, 0.0, 0.0),
            CALIBRATED_HEAD_DEG,
            "torso",
        ),
        (
            CALIBRATED_TORSO_DEG,
            (0.0, 52.0),
            "head",
        ),
    ],
)
def test_wrong_fixed_camera_posture_disconnects_before_prepare_or_stream(
    torso_deg: tuple[float, ...],
    head_deg: tuple[float, ...],
    component: str,
) -> None:
    events: list[tuple] = []
    robot = _Robot(events, torso_deg=torso_deg, head_deg=head_deg)
    runtime = AutoGrabRuntime(
        execute=True,
        sdk_module=_sdk(events, robot),
        grabbing_module=_grabbing(events),
        clock=lambda: 0.0,
    )

    with pytest.raises(AutoGrabError, match=component):
        runtime.start()

    names = [event[0] for event in events]
    assert names[-1] == "disconnect"
    assert "prepare" not in names
    assert "create_stream" not in names
    assert "grasp" not in names


def test_mobile_ready_failure_disconnects_before_mobility_stream_creation() -> None:
    events: list[tuple] = []
    robot = _Robot(events)
    runtime = AutoGrabRuntime(
        execute=True,
        sdk_module=_sdk(events, robot),
        grabbing_module=_grabbing(events, mobile_ready_result=False),
        clock=lambda: 0.0,
    )

    with pytest.raises(AutoGrabError, match="Joint Position"):
        runtime.start()

    names = [event[0] for event in events]
    assert names.index("prepare") < names.index("mobile_ready")
    assert names[-1] == "disconnect"
    assert "create_stream" not in names
    assert "velocity" not in names
    assert "grasp" not in names


def test_mobile_ready_cannot_disturb_fixed_camera_posture_before_stream() -> None:
    events: list[tuple] = []
    robot = _Robot(events)
    grabbing = _grabbing(events)

    def disturb_torso(candidate):
        events.append(("mobile_ready", candidate))
        candidate.position[1] += np.deg2rad(2.0)
        return True

    grabbing.move_arms_to_mobile_ready_pose = disturb_torso
    runtime = AutoGrabRuntime(
        execute=True,
        sdk_module=_sdk(events, robot),
        grabbing_module=grabbing,
        clock=lambda: 0.0,
    )

    with pytest.raises(AutoGrabError, match="torso"):
        runtime.start()

    names = [event[0] for event in events]
    assert names.index("mobile_ready") < names.index("disconnect")
    assert "create_stream" not in names
    assert "velocity" not in names
    assert "grasp" not in names


def test_stable_target_keeps_zero_stream_through_stop_check_then_grasps() -> None:
    events: list[tuple] = []
    robot = _Robot(events)
    runtime = AutoGrabRuntime(
        execute=True,
        sdk_module=_sdk(events, robot),
        grabbing_module=_grabbing(events),
        clock=_StepClock(),
    )
    runtime.start()

    ready_count = 0
    for index in range(1, 9):
        now = 0.1 * index
        ready_count += int(
            runtime.update(_pose(), pose_timestamp_s=now, now_s=now)
        )
        if ready_count:
            break

    assert ready_count == 1
    runtime.handoff()
    assert runtime.completed
    assert runtime.grasp_invoked
    with pytest.raises(AutoGrabError, match="already been invoked"):
        runtime.handoff()
    runtime.close()

    names = [event[0] for event in events]
    prepare_index = names.index("prepare")
    mobile_ready_index = names.index("mobile_ready")
    create_stream_index = names.index("create_stream")
    first_velocity_index = names.index("velocity")
    state_start_index = names.index("state_update_start")
    state_stop_index = names.index("state_update_stop")
    cancel_index = names.index("stream_cancel")
    wait_index = names.index("stream_wait")
    grasp_index = names.index("grasp")
    disconnect_index = names.index("disconnect")
    assert (
        prepare_index
        < mobile_ready_index
        < create_stream_index
        < first_velocity_index
        < state_start_index
        < state_stop_index
        < cancel_index
        < wait_index
        < grasp_index
        < disconnect_index
    )
    assert names.count("grasp") == 1
    assert all(
        event[-1] is robot
        for event in events
        if event[0]
        in {
            "create_robot",
            "connect",
            "prepare",
            "mobile_ready",
            "create_stream",
            "grasp",
        }
    )
    assert [event for event in events if event[0] == "velocity"][-3:] == [
        ("velocity", (0.0, 0.0, 0.0)),
        ("velocity", (0.0, 0.0, 0.0)),
        ("velocity", (0.0, 0.0, 0.0)),
    ]


def test_invalid_pose_stops_without_triggering_grasp() -> None:
    events: list[tuple] = []
    robot = _Robot(events)
    runtime = AutoGrabRuntime(
        execute=True,
        sdk_module=_sdk(events, robot),
        grabbing_module=_grabbing(events),
        clock=lambda: 0.0,
    )
    runtime.start()

    assert not runtime.update(None, pose_timestamp_s=0.1, now_s=0.1)
    runtime.close()

    assert ("velocity", (0.0, 0.0, 0.0)) in events
    assert not runtime.grasp_invoked
    assert all(event[0] != "grasp" for event in events)
    names = [event[0] for event in events]
    assert names.index("stream_cancel") < names.index("stream_wait") < names.index(
        "disconnect"
    )


def test_stale_estimator_result_sends_zero_and_cannot_reach_handoff() -> None:
    events: list[tuple] = []
    robot = _Robot(events)
    runtime = AutoGrabRuntime(
        execute=True,
        sdk_module=_sdk(events, robot),
        grabbing_module=_grabbing(events),
        clock=lambda: 0.0,
    )
    runtime.start()

    assert not runtime.update(_pose(), pose_timestamp_s=0.0, now_s=0.301)
    assert [event for event in events if event[0] == "velocity"][-1] == (
        "velocity",
        (0.0, 0.0, 0.0),
    )
    assert not runtime.grasp_invoked
    runtime.close()


@pytest.mark.parametrize(
    "pose",
    [
        _pose(canonical_reference_deg=0, canonical_residual_deg=0.0),
        _pose(
            canonical_reference_deg=None,
            canonical_residual_deg=None,
            yaw_deg=45.0,
        ),
    ],
)
def test_non_horizontal_yaw_is_zeroed_and_rejected_by_current_grasp_mode(
    pose: BasePoseDiagnostic,
) -> None:
    events: list[tuple] = []
    robot = _Robot(events)
    runtime = AutoGrabRuntime(
        execute=True,
        sdk_module=_sdk(events, robot),
        grabbing_module=_grabbing(events),
        clock=lambda: 0.0,
    )
    runtime.start()

    with pytest.raises(AutoGrabError, match="requires horizontal"):
        runtime.update(
            pose,
            pose_timestamp_s=0.1,
            now_s=0.1,
        )

    assert not runtime.grasp_invoked
    assert all(event[0] != "grasp" for event in events)
    assert runtime._pump is not None
    assert runtime._pump._latest_command.is_zero
    runtime.close()


@pytest.mark.parametrize("residual_deg", [-2.0, 1.0])
def test_horizontal_yaw_on_either_signed_display_side_can_handoff(
    residual_deg: float,
) -> None:
    events: list[tuple] = []
    robot = _Robot(events)
    runtime = AutoGrabRuntime(
        execute=True,
        sdk_module=_sdk(events, robot),
        grabbing_module=_grabbing(events),
        clock=lambda: 0.0,
    )
    runtime.start()

    ready = False
    for index in range(1, 9):
        now = 0.1 * index
        ready = runtime.update(
            _pose(
                canonical_reference_deg=90,
                canonical_residual_deg=residual_deg,
            ),
            pose_timestamp_s=now,
            now_s=now,
        )
        if ready:
            break

    assert ready
    runtime.close()


def test_stream_shutdown_failure_prevents_grasp() -> None:
    events: list[tuple] = []
    robot = _Robot(events, stream_wait_result=False)
    runtime = AutoGrabRuntime(
        execute=True,
        sdk_module=_sdk(events, robot),
        grabbing_module=_grabbing(events),
        clock=_StepClock(),
    )
    runtime.start()
    for index in range(1, 9):
        now = 0.1 * index
        if runtime.update(_pose(), pose_timestamp_s=now, now_s=now):
            break

    with pytest.raises(Exception, match="did not finish"):
        runtime.handoff()
    runtime.close()

    assert all(event[0] != "grasp" for event in events)


def test_background_pump_failure_is_surfaced_and_prevents_grasp() -> None:
    events: list[tuple] = []
    robot = _Robot(events, stream_fail_after_sends=1)
    runtime = AutoGrabRuntime(
        AutoGrabConfig(
            pump=RBY1CommandPumpConfig(
                send_rate_hz=100.0,
                command_stale_after_s=0.05,
            )
        ),
        execute=True,
        sdk_module=_sdk(events, robot),
        grabbing_module=_grabbing(events),
        clock=lambda: 0.0,
    )
    runtime.start()
    deadline = time.monotonic() + 1.0
    while not any(event[0] == "stream_send_failed" for event in events):
        if time.monotonic() >= deadline:
            raise AssertionError("pump failure did not occur before timeout")
        time.sleep(0.002)

    with pytest.raises(AutoGrabError, match="command stream expiry"):
        runtime.update(None, pose_timestamp_s=0.1, now_s=0.1)
    with pytest.raises(AutoGrabError, match="command stream expiry"):
        runtime.close()

    assert not runtime.grasp_invoked
    assert all(event[0] != "grasp" for event in events)


def test_late_pump_failure_during_release_prevents_body_handoff() -> None:
    events: list[tuple] = []
    robot = _Robot(events)
    runtime = AutoGrabRuntime(
        AutoGrabConfig(
            pump=RBY1CommandPumpConfig(
                send_rate_hz=100.0,
                command_stale_after_s=0.05,
            )
        ),
        execute=True,
        sdk_module=_sdk(events, robot),
        grabbing_module=_grabbing(events),
        clock=_StepClock(),
    )
    runtime.start()
    for index in range(1, 9):
        now = 0.1 * index
        if runtime.update(_pose(), pose_timestamp_s=now, now_s=now):
            break

    pump = runtime._pump
    assert pump is not None
    original_latch = pump.latch_zero_and_wait
    latch_count = 0

    def fail_after_second_zero_latch() -> None:
        nonlocal latch_count
        original_latch()
        latch_count += 1
        if latch_count == 2:
            robot.stream.fail_after_sends = robot.stream.send_count
            deadline = time.monotonic() + 1.0
            while not any(event[0] == "stream_send_failed" for event in events):
                if time.monotonic() >= deadline:
                    raise AssertionError("late pump failure did not occur")
                time.sleep(0.002)

    pump.latch_zero_and_wait = fail_after_second_zero_latch

    with pytest.raises(AutoGrabError, match="command stream expiry"):
        runtime.handoff()
    runtime.close()

    assert latch_count == 2
    assert not runtime.grasp_invoked
    assert all(event[0] != "grasp" for event in events)


def test_measured_mobile_velocity_must_settle_before_grasp() -> None:
    events: list[tuple] = []
    robot = _Robot(events, mobility_velocity_radps=(0.2, -0.2))
    runtime = AutoGrabRuntime(
        AutoGrabConfig(
            mobile_stop_min_duration_s=0.10,
            mobile_state_stale_after_s=0.10,
            mobile_stop_timeout_s=0.20,
        ),
        execute=True,
        sdk_module=_sdk(events, robot),
        grabbing_module=_grabbing(events),
        clock=_StepClock(),
    )
    runtime.start()
    for index in range(1, 9):
        now = 0.1 * index
        if runtime.update(_pose(), pose_timestamp_s=now, now_s=now):
            break

    with pytest.raises(AutoGrabError, match="did not settle"):
        runtime.handoff()
    runtime.close()

    assert all(event[0] != "grasp" for event in events)


def test_disconnect_failure_can_be_retried_after_stream_is_stopped() -> None:
    events: list[tuple] = []
    robot = _Robot(events, disconnect_failures=1)
    runtime = AutoGrabRuntime(
        execute=True,
        sdk_module=_sdk(events, robot),
        grabbing_module=_grabbing(events),
        clock=lambda: 0.0,
    )
    runtime.start()

    with pytest.raises(AutoGrabError, match="disconnect"):
        runtime.close()
    runtime.close()

    assert [event[0] for event in events].count("disconnect") == 2
    assert not robot.connected
