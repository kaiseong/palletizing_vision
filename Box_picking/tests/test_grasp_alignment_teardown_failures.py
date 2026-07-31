"""Software-only teardown retry contract for :class:`AutoGrabRuntime`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from parcel_pose_picking import auto_grab
from parcel_pose_picking.auto_grab import AutoGrabError, AutoGrabRuntime


@dataclass
class _Robot:
    disconnect_failures_remaining: int = 0
    events: list[str] = field(default_factory=list)

    def disconnect(self) -> None:
        self.events.append("teardown:disconnect")
        if self.disconnect_failures_remaining > 0:
            self.disconnect_failures_remaining -= 1
            raise RuntimeError("forced teardown:disconnect failure")


@dataclass
class _Stream:
    robot: _Robot
    close_failures_remaining: int = 0
    closed: bool = False

    def close(self) -> None:
        self.robot.events.append("teardown:stream_close")
        if self.close_failures_remaining > 0:
            self.close_failures_remaining -= 1
            raise RuntimeError("forced teardown:stream_close failure")
        self.closed = True


class _Pump:
    def __init__(self, robot: _Robot, stream: _Stream) -> None:
        self.robot = robot
        self.stream = stream
        self.close_failures_remaining = 0
        self.is_closed = False

    def close(self) -> None:
        self.robot.events.append("teardown:pump_close")
        if self.close_failures_remaining > 0:
            self.close_failures_remaining -= 1
            raise RuntimeError("forced teardown:pump_close failure")
        self.is_closed = True
        self.stream.closed = True


class _Grabbing:
    @staticmethod
    def run_grabbing_sequence(robot: _Robot) -> bool:
        robot.events.append("grasp_command")
        return True


def _active_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[AutoGrabRuntime, _Robot, _Stream, _Pump]:
    def measured_wheel_stop(
        robot: _Robot,
        config: Any,
        *,
        clock: Any,
    ) -> None:
        del robot, config
        assert callable(clock)

    def validate_posture(robot: _Robot, tolerance_deg: float) -> None:
        del tolerance_deg
        robot.events.append("posture_validation")

    monkeypatch.setattr(auto_grab, "_wait_for_mobile_stop", measured_wheel_stop)
    monkeypatch.setattr(auto_grab, "_validate_fixed_camera_posture", validate_posture)

    robot = _Robot()
    stream = _Stream(robot)
    pump = _Pump(robot, stream)
    runtime = AutoGrabRuntime(
        execute=True,
        grabbing_module=_Grabbing,
        clock=lambda: 1.0,
    )
    # Inject lower lifecycle resources directly.  No SDK is imported or used.
    runtime._robot = robot
    runtime._stream = stream
    runtime._pump = pump
    runtime._started = True
    runtime._handoff_ready = True
    return runtime, robot, stream, pump


def _assert_no_dependent_motion(robot: _Robot) -> None:
    assert "posture_validation" not in robot.events
    assert "grasp_command" not in robot.events


def test_pump_close_failure_retries_only_unresolved_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, robot, stream, pump = _active_runtime(monkeypatch)
    pump.close_failures_remaining = 1

    with pytest.raises(AutoGrabError) as failure:
        runtime.close()

    assert str(failure.value) == (
        "failed to stop RB-Y1 mobility stream cleanly: "
        "forced teardown:pump_close failure"
    )
    assert robot.events == ["teardown:pump_close", "teardown:disconnect"]
    assert runtime._pump is pump
    assert runtime._stream is stream
    assert runtime._robot is None
    assert runtime._closed is False
    _assert_no_dependent_motion(robot)

    runtime.close()
    assert robot.events == [
        "teardown:pump_close",
        "teardown:disconnect",
        "teardown:pump_close",
    ]
    assert runtime._pump is None
    assert runtime._stream is None
    assert runtime._robot is None
    assert runtime._closed is True
    _assert_no_dependent_motion(robot)

    after_cleanup = list(robot.events)
    runtime.close()
    assert robot.events == after_cleanup


def test_bare_stream_close_failure_retries_only_unresolved_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, robot, stream, _pump = _active_runtime(monkeypatch)
    runtime._pump = None
    stream.close_failures_remaining = 1

    with pytest.raises(AutoGrabError) as failure:
        runtime.close()

    assert str(failure.value) == (
        "failed to stop RB-Y1 mobility stream cleanly: "
        "forced teardown:stream_close failure"
    )
    assert robot.events == ["teardown:stream_close", "teardown:disconnect"]
    assert runtime._pump is None
    assert runtime._stream is stream
    assert runtime._robot is None
    assert runtime._closed is False
    _assert_no_dependent_motion(robot)

    runtime.close()
    assert robot.events == [
        "teardown:stream_close",
        "teardown:disconnect",
        "teardown:stream_close",
    ]
    assert runtime._stream is None
    assert runtime._robot is None
    assert runtime._closed is True
    _assert_no_dependent_motion(robot)

    after_cleanup = list(robot.events)
    runtime.close()
    assert robot.events == after_cleanup


def test_disconnect_failure_retries_only_unresolved_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, robot, _stream, _pump = _active_runtime(monkeypatch)
    robot.disconnect_failures_remaining = 1

    with pytest.raises(AutoGrabError) as failure:
        runtime.close()

    assert str(failure.value) == (
        "failed to disconnect RB-Y1 cleanly: "
        "forced teardown:disconnect failure"
    )
    assert robot.events == ["teardown:pump_close", "teardown:disconnect"]
    assert runtime._pump is None
    assert runtime._stream is None
    assert runtime._robot is robot
    assert runtime._closed is False
    _assert_no_dependent_motion(robot)

    runtime.close()
    assert robot.events == [
        "teardown:pump_close",
        "teardown:disconnect",
        "teardown:disconnect",
    ]
    assert runtime._robot is None
    assert runtime._closed is True
    _assert_no_dependent_motion(robot)

    after_cleanup = list(robot.events)
    runtime.close()
    assert robot.events == after_cleanup
