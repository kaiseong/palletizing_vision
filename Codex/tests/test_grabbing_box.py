from __future__ import annotations

import importlib.util
import inspect
import sys
import types
from pathlib import Path

import pytest


GRABBING_BOX_PATH = Path(__file__).resolve().parents[2] / "grabbing_box.py"


@pytest.fixture
def grabbing_box(monkeypatch: pytest.MonkeyPatch):
    """Load the standalone script without importing a real robot SDK."""
    fake_sdk = types.ModuleType("rby1_sdk")
    fake_sdk.RobotCommandFeedback = types.SimpleNamespace(
        FinishCode=types.SimpleNamespace(Ok=object())
    )
    monkeypatch.setitem(sys.modules, "rby1_sdk", fake_sdk)

    module_name = "_test_grabbing_box"
    spec = importlib.util.spec_from_file_location(module_name, GRABBING_BOX_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


class PrepareRobotFake:
    def __init__(
        self,
        *,
        connected: bool = True,
        powered: bool = False,
        power_on_result: bool = True,
        servoed: bool = False,
        servo_on_result: bool = True,
        enable_result: bool = True,
    ) -> None:
        self.connected = connected
        self.powered = powered
        self.power_on_result = power_on_result
        self.servoed = servoed
        self.servo_on_result = servo_on_result
        self.enable_result = enable_result
        self.calls: list[tuple[object, ...]] = []

    def is_connected(self):
        self.calls.append(("is_connected",))
        return self.connected

    def is_power_on(self, pattern):
        self.calls.append(("is_power_on", pattern))
        return self.powered

    def power_on(self, pattern):
        self.calls.append(("power_on", pattern))
        return self.power_on_result

    def is_servo_on(self, pattern):
        self.calls.append(("is_servo_on", pattern))
        return self.servoed

    def servo_on(self, pattern):
        self.calls.append(("servo_on", pattern))
        return self.servo_on_result

    def reset_fault_control_manager(self):
        self.calls.append(("reset_fault_control_manager",))

    def enable_control_manager(self):
        self.calls.append(("enable_control_manager",))
        return self.enable_result


def test_prepare_robot_brings_up_an_already_connected_robot(grabbing_box):
    robot = PrepareRobotFake()

    grabbing_box.prepare_robot(robot, power="48v.*")

    assert robot.calls == [
        ("is_connected",),
        ("is_power_on", "48v.*"),
        ("power_on", "48v.*"),
        ("is_servo_on", ".*"),
        ("servo_on", ".*"),
        ("reset_fault_control_manager",),
        ("enable_control_manager",),
    ]


def test_prepare_robot_does_not_cycle_power_or_servos_that_are_already_on(
    grabbing_box,
):
    robot = PrepareRobotFake(powered=True, servoed=True)

    grabbing_box.prepare_robot(robot)

    assert robot.calls == [
        ("is_connected",),
        ("is_power_on", ".*"),
        ("is_servo_on", ".*"),
        ("reset_fault_control_manager",),
        ("enable_control_manager",),
    ]


@pytest.mark.parametrize(
    ("robot", "error_type", "error_text", "last_call"),
    [
        (
            PrepareRobotFake(connected=False),
            ConnectionError,
            "not connected",
            ("is_connected",),
        ),
        (
            PrepareRobotFake(power_on_result=False),
            RuntimeError,
            "power on",
            ("power_on", ".*"),
        ),
        (
            PrepareRobotFake(servo_on_result=False),
            RuntimeError,
            "servo on",
            ("servo_on", ".*"),
        ),
        (
            PrepareRobotFake(enable_result=False),
            RuntimeError,
            "control manager",
            ("enable_control_manager",),
        ),
    ],
)
def test_prepare_robot_stops_at_each_failed_prerequisite(
    grabbing_box,
    robot,
    error_type,
    error_text,
    last_call,
):
    with pytest.raises(error_type, match=error_text):
        grabbing_box.prepare_robot(robot)

    assert robot.calls[-1] == last_call


class DynamicsFake:
    def __init__(self, events):
        self.events = events
        self.state = object()

    def make_state(self, link_names, joint_names):
        self.events.append(("make_state", tuple(link_names), tuple(joint_names)))
        return self.state


class SequenceRobotFake:
    def __init__(self, events):
        self.events = events
        self.dynamics = DynamicsFake(events)
        self.state_reads = 0

    def is_connected(self):
        self.events.append(("is_connected",))
        return True

    def model(self):
        self.events.append(("model",))
        return types.SimpleNamespace(model_name="m", robot_joint_names=("j0", "j1"))

    def get_dynamics(self):
        self.events.append(("get_dynamics",))
        return self.dynamics

    def get_state(self):
        self.state_reads += 1
        position = f"q{self.state_reads}"
        self.events.append(("get_state", position))
        return types.SimpleNamespace(position=position)

    def start_state_update(self, callback, rate):
        self.events.append(("start_state_update", callback, rate))

    def stop_state_update(self):
        self.events.append(("stop_state_update",))

    def connect(self):  # pragma: no cover - a failure is the assertion
        raise AssertionError("run_grabbing_sequence must not connect")

    def disconnect(self):  # pragma: no cover - a failure is the assertion
        raise AssertionError("run_grabbing_sequence must not disconnect")


def install_sequence_spies(grabbing_box, monkeypatch, events, outcomes=(True, True, True)):
    commands = iter(("start-command", "grab-command", "lift-command"))
    results = iter(outcomes)

    def build_pose(pose, minimum_time):
        events.append(("build_pose", pose, minimum_time))
        return next(commands)

    def build_grab(dyn_model, dyn_state, q):
        events.append(("build_grab", dyn_model, dyn_state, q))
        return next(commands)

    def build_lift(dyn_model, dyn_state, q):
        events.append(("build_lift", dyn_model, dyn_state, q))
        return next(commands)

    def send_once(robot, command):
        events.append(("send_once", robot, command))
        return next(results)

    monkeypatch.setattr(grabbing_box, "build_pose_command", build_pose)
    monkeypatch.setattr(grabbing_box, "build_impedance_grab_command", build_grab)
    monkeypatch.setattr(grabbing_box, "build_impedance_lift_command", build_lift)
    monkeypatch.setattr(grabbing_box, "send_once", send_once)


@pytest.mark.parametrize("monitor_ft", [False, True])
def test_run_grabbing_sequence_sends_each_existing_command_once_on_same_robot(
    grabbing_box,
    monkeypatch,
    monitor_ft,
):
    events: list[tuple[object, ...]] = []
    robot = SequenceRobotFake(events)
    install_sequence_spies(grabbing_box, monkeypatch, events)

    assert grabbing_box.run_grabbing_sequence(robot, monitor_ft=monitor_ft) is True

    sends = [event for event in events if event[0] == "send_once"]
    assert [event[2] for event in sends] == [
        "start-command",
        "grab-command",
        "lift-command",
    ]
    assert all(event[1] is robot for event in sends)
    assert sum(event[0] == "build_pose" for event in events) == 1
    assert sum(event[0] == "build_grab" for event in events) == 1
    assert sum(event[0] == "build_lift" for event in events) == 1
    assert robot.state_reads == 2

    start_calls = [event for event in events if event[0] == "start_state_update"]
    stop_calls = [event for event in events if event[0] == "stop_state_update"]
    if monitor_ft:
        assert len(start_calls) == 1
        assert callable(start_calls[0][1])
        assert start_calls[0][2] == grabbing_box.FT_MONITOR_RATE
        assert stop_calls == [("stop_state_update",)]
    else:
        assert start_calls == []
        assert stop_calls == []


def test_run_grabbing_sequence_stops_ft_monitor_when_a_command_fails(
    grabbing_box,
    monkeypatch,
):
    events: list[tuple[object, ...]] = []
    robot = SequenceRobotFake(events)
    install_sequence_spies(grabbing_box, monkeypatch, events, outcomes=(True, False))

    assert grabbing_box.run_grabbing_sequence(robot, monitor_ft=True) is False

    assert [event[2] for event in events if event[0] == "send_once"] == [
        "start-command",
        "grab-command",
    ]
    assert events[-1] == ("stop_state_update",)


class StandaloneRobotFake:
    def __init__(self, events):
        self.events = events

    def connect(self):
        self.events.append(("connect",))
        return True

    def is_connected(self):
        self.events.append(("is_connected",))
        return True

    def disconnect(self):
        self.events.append(("disconnect",))


def test_standalone_main_defaults_to_model_m_and_owns_connection_lifecycle(
    grabbing_box,
    monkeypatch,
):
    events: list[tuple[object, ...]] = []
    robot = StandaloneRobotFake(events)

    def create_robot(address, model):
        events.append(("create_robot", address, model))
        return robot

    def prepare_robot(candidate, *, power):
        assert candidate is robot
        events.append(("prepare_robot", power))

    def run_grabbing_sequence(candidate):
        assert candidate is robot
        events.append(("run_grabbing_sequence",))
        return True

    monkeypatch.setattr(grabbing_box.rby, "create_robot", create_robot, raising=False)
    monkeypatch.setattr(grabbing_box, "prepare_robot", prepare_robot)
    monkeypatch.setattr(grabbing_box, "run_grabbing_sequence", run_grabbing_sequence)

    assert inspect.signature(grabbing_box.main).parameters["model"].default == "m"
    assert grabbing_box.main("192.0.2.10:50051", power="48v.*") is True
    assert events == [
        ("create_robot", "192.0.2.10:50051", "m"),
        ("connect",),
        ("is_connected",),
        ("prepare_robot", "48v.*"),
        ("run_grabbing_sequence",),
        ("disconnect",),
    ]


def test_standalone_main_rejects_non_model_m_before_creating_robot(
    grabbing_box,
    monkeypatch,
):
    def unexpected_create(*_args, **_kwargs):  # pragma: no cover
        raise AssertionError("non-M input must be rejected before SDK robot creation")

    monkeypatch.setattr(
        grabbing_box.rby,
        "create_robot",
        unexpected_create,
        raising=False,
    )

    assert grabbing_box.main("192.0.2.10:50051", model="a") is False


def test_send_once_cancels_and_waits_when_ctrl_c_interrupts_handler(
    grabbing_box,
) -> None:
    events: list[tuple[object, ...]] = []

    class InterruptingHandler:
        def get(self):
            events.append(("get",))
            raise KeyboardInterrupt

        def cancel(self):
            events.append(("cancel",))

        def wait_for(self, timeout_ms):
            events.append(("wait_for", timeout_ms))
            return True

    class Robot:
        def send_command(self, command):
            events.append(("send_command", command))
            return InterruptingHandler()

    with pytest.raises(KeyboardInterrupt):
        grabbing_box.send_once(Robot(), "command")

    assert events == [
        ("send_command", "command"),
        ("get",),
        ("cancel",),
        ("wait_for", 2_000),
    ]


def test_send_once_cancels_and_waits_when_sdk_feedback_fails(
    grabbing_box,
) -> None:
    events: list[tuple[object, ...]] = []

    class FailingHandler:
        def get(self):
            events.append(("get",))
            raise RuntimeError("feedback channel failed")

        def cancel(self):
            events.append(("cancel",))

        def wait_for(self, timeout_ms):
            events.append(("wait_for", timeout_ms))
            return True

    class Robot:
        def send_command(self, command):
            events.append(("send_command", command))
            return FailingHandler()

    with pytest.raises(RuntimeError, match="feedback channel failed"):
        grabbing_box.send_once(Robot(), "command")

    assert events == [
        ("send_command", "command"),
        ("get",),
        ("cancel",),
        ("wait_for", 2_000),
    ]
