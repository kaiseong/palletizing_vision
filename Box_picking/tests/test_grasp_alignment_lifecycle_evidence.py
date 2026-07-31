"""Software-only proof of the picking stop/release evidence contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from parcel_pose_picking import auto_grab
from parcel_pose_picking.auto_grab import (
    AutoGrabError,
    AutoGrabRuntime,
    GraspAlignmentStoppedAndReleased,
)


@dataclass
class _Robot:
    fail_at: str | None = None
    grasp_result: bool | Exception = True
    events: list[str] = field(default_factory=list)

    def disconnect(self) -> None:
        self.events.append("teardown:disconnect")


@dataclass
class _Stream:
    robot: _Robot
    closed: bool = False

    def close(self) -> None:
        self.robot.events.append("teardown:stream_close")
        self.closed = True


class _Pump:
    def __init__(self, robot: _Robot, stream: _Stream) -> None:
        self.robot = robot
        self.stream = stream
        self.is_closed = False
        self.send_count = 11
        self.max_send_gap_s = 0.013

    def _step(self, name: str) -> None:
        self.robot.events.append(name)
        if self.robot.fail_at == name:
            raise RuntimeError(f"forced {name} failure")

    def latch_zero_and_wait(self) -> None:
        self._step("exact_zero_latch")

    def raise_if_failed(self) -> None:
        self._step("pump_health")

    def stop_and_release(self) -> None:
        self._step("stream_release")
        self.is_closed = True
        self.stream.closed = True

    def close(self) -> None:
        self.robot.events.append("teardown:pump_close")
        self.is_closed = True
        self.stream.closed = True


class _Grabbing:
    @staticmethod
    def run_grabbing_sequence(robot: _Robot) -> bool:
        robot.events.append("grasp_command")
        if isinstance(robot.grasp_result, Exception):
            raise robot.grasp_result
        return robot.grasp_result


def _active_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_at: str | None = None,
) -> tuple[AutoGrabRuntime, _Robot]:
    def measured_wheel_stop(
        robot: _Robot,
        config: Any,
        *,
        clock: Any,
    ) -> None:
        del config
        assert callable(clock)
        robot.events.append("measured_wheel_stop")
        if robot.fail_at == "measured_wheel_stop":
            raise AutoGrabError("forced measured_wheel_stop failure")

    def validate_posture(robot: _Robot, tolerance_deg: float) -> None:
        assert tolerance_deg > 0.0
        robot.events.append("posture_validation")

    monkeypatch.setattr(auto_grab, "_wait_for_mobile_stop", measured_wheel_stop)
    monkeypatch.setattr(auto_grab, "_validate_fixed_camera_posture", validate_posture)

    robot = _Robot(fail_at=fail_at)
    stream = _Stream(robot)
    runtime = AutoGrabRuntime(
        execute=True,
        grabbing_module=_Grabbing,
        clock=lambda: 1.0,
    )
    # Exercise only the lower lifecycle seam.  No SDK is imported or connected.
    runtime._robot = robot
    runtime._stream = stream
    runtime._pump = _Pump(robot, stream)
    runtime._started = True
    runtime._handoff_ready = True
    return runtime, robot


def test_success_fact_is_emitted_once_and_is_required_for_grasp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, robot = _active_runtime(monkeypatch)

    with pytest.raises(
        AutoGrabError,
        match="requires GraspAlignmentStoppedAndReleased evidence",
    ):
        runtime.execute_grasp(None)  # type: ignore[arg-type]
    assert "grasp_command" not in robot.events

    evidence = runtime.stop_alignment_for_grasp()

    assert isinstance(evidence, GraspAlignmentStoppedAndReleased)
    assert evidence.grasp_alignment_stopped_and_released is True
    assert evidence.exact_zero_latched is True
    assert evidence.measured_wheel_stop is True
    assert evidence.pump_healthy is True
    assert evidence.stream_released is True
    assert evidence.pump_send_count == 11
    assert evidence.pump_max_send_gap_s == pytest.approx(0.013)
    assert robot.events == [
        "exact_zero_latch",
        "measured_wheel_stop",
        "pump_health",
        "stream_release",
    ]

    # Re-reading the fact is side-effect free; it cannot send a second stop.
    assert runtime.stop_alignment_for_grasp() is evidence
    assert robot.events == [
        "exact_zero_latch",
        "measured_wheel_stop",
        "pump_health",
        "stream_release",
    ]

    forged = GraspAlignmentStoppedAndReleased(
        pump_send_count=evidence.pump_send_count,
        pump_max_send_gap_s=evidence.pump_max_send_gap_s,
    )
    assert forged == evidence and forged is not evidence
    with pytest.raises(AutoGrabError, match="not emitted by this runtime"):
        runtime.execute_grasp(forged)
    assert "posture_validation" not in robot.events
    assert "grasp_command" not in robot.events

    runtime.execute_grasp(evidence)
    assert robot.events[-2:] == ["posture_validation", "grasp_command"]
    assert runtime.grasp_invoked is True
    assert runtime.completed is True

    runtime.close()
    after_first_close = list(robot.events)
    runtime.close()
    assert robot.events == after_first_close
    assert robot.events.count("teardown:disconnect") == 1


def test_cross_runtime_success_fact_cannot_authorize_grasp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, source_robot = _active_runtime(monkeypatch)
    target, target_robot = _active_runtime(monkeypatch)
    source_evidence = source.stop_alignment_for_grasp()

    with pytest.raises(AutoGrabError, match="not emitted by this runtime"):
        target.execute_grasp(source_evidence)

    assert "posture_validation" not in target_robot.events
    assert "grasp_command" not in target_robot.events
    source.close()
    target.close()
    assert "grasp_command" not in source_robot.events
    assert "grasp_command" not in target_robot.events


def test_posture_failure_after_valid_stop_evidence_blocks_grasp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, robot = _active_runtime(monkeypatch)
    evidence = runtime.stop_alignment_for_grasp()

    def reject_posture(selected_robot: _Robot, tolerance_deg: float) -> None:
        assert selected_robot is robot
        assert tolerance_deg > 0.0
        robot.events.append("posture_validation")
        raise AutoGrabError("forced posture validation failure")

    monkeypatch.setattr(auto_grab, "_validate_fixed_camera_posture", reject_posture)

    with pytest.raises(AutoGrabError, match="forced posture validation failure"):
        runtime.execute_grasp(evidence)

    assert robot.events.count("posture_validation") == 1
    assert robot.events.count("grasp_command") == 0
    assert runtime.grasp_invoked is False
    assert runtime.completed is False
    runtime.close()


@pytest.mark.parametrize(
    ("grasp_result", "message"),
    [
        (False, "packaged grasp sequence reported failure"),
        (RuntimeError("sequence boom"), "packaged grasp sequence failed: sequence boom"),
    ],
    ids=("false_feedback", "sequence_exception"),
)
def test_packaged_sequence_failure_is_not_retried_or_marked_complete(
    monkeypatch: pytest.MonkeyPatch,
    grasp_result: bool | Exception,
    message: str,
) -> None:
    runtime, robot = _active_runtime(monkeypatch)
    robot.grasp_result = grasp_result
    evidence = runtime.stop_alignment_for_grasp()

    with pytest.raises(AutoGrabError, match=message):
        runtime.execute_grasp(evidence)

    assert robot.events.count("posture_validation") == 1
    assert robot.events.count("grasp_command") == 1
    assert runtime.grasp_invoked is True
    assert runtime.completed is False
    after_failure = list(robot.events)

    with pytest.raises(AutoGrabError, match="already been invoked"):
        runtime.execute_grasp(evidence)

    assert robot.events == after_failure
    assert robot.events.count("grasp_command") == 1
    assert runtime.completed is False
    runtime.close()


@pytest.mark.parametrize(
    ("failed_step", "expected_before_teardown", "message"),
    [
        (
            "exact_zero_latch",
            ["exact_zero_latch"],
            "cannot zero the active mobility stream",
        ),
        (
            "measured_wheel_stop",
            ["exact_zero_latch", "measured_wheel_stop"],
            "forced measured_wheel_stop failure",
        ),
        (
            "pump_health",
            ["exact_zero_latch", "measured_wheel_stop", "pump_health"],
            "forced pump_health failure",
        ),
        (
            "stream_release",
            [
                "exact_zero_latch",
                "measured_wheel_stop",
                "pump_health",
                "stream_release",
            ],
            "forced stream_release failure",
        ),
    ],
)
def test_each_stop_substep_failure_blocks_grasp_and_teardown_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    failed_step: str,
    expected_before_teardown: list[str],
    message: str,
) -> None:
    runtime, robot = _active_runtime(monkeypatch, fail_at=failed_step)

    with pytest.raises(AutoGrabError, match=message):
        runtime.stop_alignment_for_grasp()

    assert runtime._grasp_alignment_evidence is None
    assert robot.events == expected_before_teardown
    with pytest.raises(
        AutoGrabError,
        match="requires GraspAlignmentStoppedAndReleased evidence",
    ):
        runtime.execute_grasp(None)  # type: ignore[arg-type]
    assert "posture_validation" not in robot.events
    assert "grasp_command" not in robot.events

    runtime.close()
    after_first_close = list(robot.events)
    runtime.close()
    assert robot.events == after_first_close
    assert robot.events.count("teardown:pump_close") == 1
    assert robot.events.count("teardown:disconnect") == 1
    assert "grasp_command" not in robot.events
