"""Phase-0 characterization of the demonstrated horizontal-picking path.

The assertions deliberately record semantic stages instead of implementation
method names. Robot, stream, and camera collaborators are fakes; the
``AutoGrabRuntime`` orchestration and mobile-servo decisions remain real.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from parcel_pose_common.mobile_servo import ServoConfig, ServoDecision, ServoState
from parcel_pose_common.models import PoseResult
from parcel_pose_picking import auto_grab, realtime
from parcel_pose_picking.auto_grab import AutoGrabConfig, AutoGrabError, AutoGrabRuntime
from parcel_pose_picking.evaluation import BasePoseDiagnostic


SemanticEvent = tuple[str, str]


@dataclass
class _SemanticTrace:
    events: list[SemanticEvent] = field(default_factory=list)

    def add(self, category: str, outcome: str) -> None:
        event = (category, outcome)
        # Frame rate and filter-window details are not part of the baseline.
        # Keep only semantic decision transitions when consecutive frames agree.
        if category == "decision" and self.events[-1:] == [event]:
            return
        self.events.append(event)


def _horizontal_pose() -> BasePoseDiagnostic:
    return BasePoseDiagnostic(
        box_center_xyz_m=(0.740, 0.0, 0.200),
        top_center_xyz_m=(0.740, 0.0, 0.320),
        yaw_mod_180_deg=90.0,
        yaw_signed_deg=-90.0,
        canonical_reference_deg=90,
        canonical_residual_deg=0.0,
        registration="validated",
    )


def _vertical_pose() -> BasePoseDiagnostic:
    return BasePoseDiagnostic(
        box_center_xyz_m=(0.740, 0.0, 0.200),
        top_center_xyz_m=(0.740, 0.0, 0.320),
        yaw_mod_180_deg=0.0,
        yaw_signed_deg=0.0,
        canonical_reference_deg=0,
        canonical_residual_deg=0.0,
        registration="validated",
    )


def _decision_outcome(decision: ServoDecision) -> str:
    if decision.state is ServoState.ACQUIRING:
        return "pose_acquisition"
    if decision.state is ServoState.TRACKING:
        return "base_alignment"
    if decision.state is ServoState.HOLDING:
        return "arrival_verification"
    if decision.state is ServoState.ARRIVED and decision.handoff_ready:
        return "handoff_ready"
    if decision.state is ServoState.POSE_LOST:
        return "pose_unavailable_stop"
    if decision.state is ServoState.ABORTED:
        if decision.reason.startswith("unsupported_grasp_orientation"):
            return "refused_non_horizontal_grasp"
        return "aborted"
    return decision.state.value


@dataclass
class _RuntimeHarness:
    runtime: AutoGrabRuntime
    trace: _SemanticTrace
    pump: Any
    grasp_calls: list[str]


def _runtime_harness(monkeypatch: pytest.MonkeyPatch) -> _RuntimeHarness:
    trace = _SemanticTrace()
    grasp_calls: list[str] = []
    real_servo_type = auto_grab.MobileVisualServo

    class TracedServo:
        """Observe public decisions while retaining the real state machine."""

        def __init__(self, config: ServoConfig) -> None:
            self._inner = real_servo_type(config)

        def _record(self, decision: ServoDecision) -> ServoDecision:
            trace.add("decision", _decision_outcome(decision))
            return decision

        def start(self, now_s: float) -> ServoDecision:
            return self._record(self._inner.start(now_s))

        def step(self, measurement: Any, *, now_s: float) -> ServoDecision:
            return self._record(self._inner.step(measurement, now_s=now_s))

        def abort(self, reason: str, now_s: float) -> ServoDecision:
            return self._record(self._inner.abort(reason, now_s))

    class FakeRobot:
        def __init__(self) -> None:
            torso = np.deg2rad(
                np.asarray(auto_grab.EXPECTED_TORSO_POSITION_DEG, dtype=np.float64)
            )
            head = np.deg2rad(
                np.asarray(auto_grab.EXPECTED_HEAD_POSITION_DEG, dtype=np.float64)
            )
            self._positions = np.concatenate((torso, head, np.zeros(2)))
            self._connected = False

        def connect(self) -> bool:
            self._connected = True
            return True

        def is_connected(self) -> bool:
            return self._connected

        def get_robot_info(self) -> Any:
            return SimpleNamespace(
                robot_model_name=auto_grab.EXPECTED_ROBOT_MODEL,
                robot_model_version=auto_grab.EXPECTED_ROBOT_VERSION,
            )

        def model(self) -> Any:
            return SimpleNamespace(
                torso_idx=np.arange(0, 6),
                head_idx=np.arange(6, 8),
                mobility_idx=np.arange(8, 10),
            )

        def get_state(self) -> Any:
            return SimpleNamespace(position=self._positions.copy())

        def disconnect(self) -> None:
            self._connected = False
            trace.add("stage", "robot_session_closed")

    robot = FakeRobot()

    class FakeSdk:
        @staticmethod
        def create_robot(address: str, model: str) -> FakeRobot:
            assert address == auto_grab.DEFAULT_ROBOT_ADDRESS
            assert model == "m"
            return robot

    class FakeGrabbing:
        @staticmethod
        def prepare_robot(selected_robot: FakeRobot, *, power: str) -> None:
            assert selected_robot is robot
            assert power == ".*"
            trace.add("stage", "robot_prepared")

        @staticmethod
        def move_arms_to_mobile_ready_pose(selected_robot: FakeRobot) -> bool:
            assert selected_robot is robot
            trace.add("stage", "grasp_posture_ready")
            return True

        @staticmethod
        def run_grabbing_sequence(selected_robot: FakeRobot) -> bool:
            assert selected_robot is robot
            grasp_calls.append("grasp_and_lift")
            trace.add("stage", "grasp_and_lift_completed")
            return True

    class FakeStream:
        def __init__(
            self,
            selected_robot: FakeRobot,
            *,
            execute: bool,
            config: Any,
            sdk_module: Any,
        ) -> None:
            assert selected_robot is robot
            assert execute is True
            assert sdk_module is FakeSdk
            self.closed = False

        def open(self) -> FakeStream:
            trace.add("stage", "alignment_stream_open")
            return self

        def close(self) -> None:
            self.closed = True
            trace.add("stop", "alignment_stream_closed")

    class FakePump:
        instance: FakePump | None = None

        def __init__(self, stream: FakeStream, *, config: Any) -> None:
            assert not stream.closed
            self.stream = stream
            self.commands: list[Any] = []
            self.is_closed = False
            self.send_count = 7
            self.max_send_gap_s = 0.012
            FakePump.instance = self

        def start(self) -> None:
            trace.add("stage", "alignment_active")

        def publish(self, command: Any) -> None:
            self.commands.append(command)

        def latch_zero_and_wait(self) -> None:
            trace.add("stop", "zero_command_latched")

        def raise_if_failed(self) -> None:
            return None

        def stop_and_release(self) -> None:
            self.is_closed = True
            self.stream.closed = True
            trace.add("stop", "measured_stop_stream_released")

        def close(self) -> None:
            self.is_closed = True
            self.stream.closed = True
            trace.add("stop", "alignment_stream_closed")

    def fake_wait_for_mobile_stop(
        selected_robot: FakeRobot,
        config: AutoGrabConfig,
        *,
        clock: Any,
    ) -> None:
        assert selected_robot is robot
        assert callable(clock)
        trace.add("stop", "wheel_state_settled")

    monkeypatch.setattr(auto_grab, "MobileVisualServo", TracedServo)
    monkeypatch.setattr(auto_grab, "RBY1MobilityStream", FakeStream)
    monkeypatch.setattr(auto_grab, "RBY1MobilityCommandPump", FakePump)
    monkeypatch.setattr(auto_grab, "_wait_for_mobile_stop", fake_wait_for_mobile_stop)

    config = AutoGrabConfig(
        servo=ServoConfig(
            arrival_min_frames=2,
            arrival_min_duration_s=0.10,
        )
    )
    runtime = AutoGrabRuntime(
        config,
        execute=True,
        sdk_module=FakeSdk,
        grabbing_module=FakeGrabbing,
        clock=lambda: 0.0,
    )
    assert FakePump.instance is None
    runtime.start()
    assert FakePump.instance is not None
    return _RuntimeHarness(runtime, trace, FakePump.instance, grasp_calls)


def test_horizontal_pick_success_has_stable_stage_decision_and_stop_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _runtime_harness(monkeypatch)
    pose = _horizontal_pose()

    handoff_results = [
        harness.runtime.update(pose, pose_timestamp_s=now_s, now_s=now_s)
        for now_s in (0.10, 0.20, 0.30, 0.45)
    ]
    assert handoff_results == [False, False, False, True]

    harness.runtime.handoff()
    harness.runtime.close()

    assert harness.runtime.grasp_invoked is True
    assert harness.runtime.completed is True
    assert harness.grasp_calls == ["grasp_and_lift"]
    assert harness.pump.commands
    assert all(command.is_zero for command in harness.pump.commands)
    assert harness.trace.events == [
        ("stage", "robot_prepared"),
        ("stage", "grasp_posture_ready"),
        ("stage", "alignment_stream_open"),
        ("stage", "alignment_active"),
        ("decision", "pose_acquisition"),
        ("decision", "arrival_verification"),
        ("decision", "handoff_ready"),
        ("stop", "zero_command_latched"),
        ("stop", "wheel_state_settled"),
        ("stop", "measured_stop_stream_released"),
        ("stage", "grasp_and_lift_completed"),
        ("stage", "robot_session_closed"),
    ]


def test_non_horizontal_pick_is_refused_with_zero_command_and_no_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _runtime_harness(monkeypatch)

    with pytest.raises(
        AutoGrabError,
        match="unsupported_grasp_orientation",
    ):
        harness.runtime.update(
            _vertical_pose(),
            pose_timestamp_s=0.10,
            now_s=0.10,
        )
    harness.runtime.close()

    assert harness.runtime.grasp_invoked is False
    assert harness.runtime.completed is False
    assert harness.grasp_calls == []
    assert len(harness.pump.commands) == 1
    assert harness.pump.commands[0].is_zero
    assert harness.trace.events == [
        ("stage", "robot_prepared"),
        ("stage", "grasp_posture_ready"),
        ("stage", "alignment_stream_open"),
        ("stage", "alignment_active"),
        ("decision", "pose_acquisition"),
        ("decision", "refused_non_horizontal_grasp"),
        ("stop", "alignment_stream_closed"),
        ("stage", "robot_session_closed"),
    ]


def test_live_path_preserves_plan_frame_counter_through_first_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bounded run consumes one frame without resetting the validated counter."""

    trace = _SemanticTrace()
    calibration = SimpleNamespace(
        T_base_from_depth=np.eye(4, dtype=np.float64),
        diagnostics={},
    )
    plan = realtime.resolve_live_view_plan(
        calibration=calibration,
        fullscreen=False,
        headless=True,
        log_jsonl=None,
        max_frames=1,
        output_mp4=None,
        warmup_frames=0,
        window_name="characterization",
    )
    assert plan.processed_frames == 0

    metadata = SimpleNamespace(
        depth_profile=SimpleNamespace(intrinsics=object()),
        depth_scale_m=0.001,
    )

    class FakeCamera:
        def __init__(self, stream_config: Any) -> None:
            assert stream_config is plan.stream_config

        def __enter__(self) -> FakeCamera:
            trace.add("stage", "camera_open")
            return self

        def __exit__(self, *exc_info: Any) -> None:
            trace.add("stop", "camera_closed")

        def session_metadata(self, **context: Any) -> Any:
            assert context == {"source": "characterization"}
            return metadata

        def capture(self) -> Any:
            return SimpleNamespace(
                raw_depth_z16=np.zeros((1, 1), dtype=np.uint16),
                depth_timestamp_ms=10.0,
                depth_frame_number=1,
            )

    class FakeEstimator:
        last_evidence = None

        def __init__(self, *args: Any) -> None:
            return None

        def estimate(self, *args: Any, **kwargs: Any) -> object:
            pytest.fail("realtime must delegate estimation to perceive_box_pose")

    def fake_perceive_box_pose(
        rgb: Any,
        depth: Any,
        intrinsics: Any,
        selected_calibration: Any,
        **kwargs: Any,
    ) -> PoseResult:
        assert rgb is None
        assert depth.shape == (1, 1)
        assert intrinsics is metadata.depth_profile.intrinsics
        assert selected_calibration is calibration
        assert isinstance(kwargs["estimator"], FakeEstimator)
        assert kwargs["depth_scale"] == 0.001
        assert kwargs["sensor_timestamp_ms"] == 10.0
        assert kwargs["frame_id"] == 1
        pose = _horizontal_pose()
        return PoseResult(
            x_m=pose.box_center_xyz_m[0],
            y_m=pose.box_center_xyz_m[1],
            yaw_rad=np.pi / 2.0,
            valid=True,
            reason="",
            timestamp_s=kwargs["timestamp_s"],
            diagnostics={"base_pose": pose.to_dict()},
        )

    class FakeAutomation:
        def start(self) -> None:
            trace.add("stage", "automation_started")

        def update(
            self,
            base_pose: BasePoseDiagnostic | None,
            *,
            pose_timestamp_s: float,
            now_s: float,
        ) -> bool:
            assert base_pose == _horizontal_pose()
            assert pose_timestamp_s <= now_s
            trace.add("decision", "frame_consumed_without_handoff")
            return False

        def handoff(self) -> None:
            pytest.fail("one non-arrival frame must not trigger a grasp handoff")

        def close(self) -> None:
            trace.add("stop", "automation_closed")

    monkeypatch.setattr(realtime, "RealSenseAdapter", FakeCamera)
    monkeypatch.setattr(realtime, "ParcelPoseEstimator", FakeEstimator)
    monkeypatch.setattr(realtime, "perceive_box_pose", fake_perceive_box_pose)

    outcome = realtime.watch_and_grab(
        handoff_ready=plan.handoff_ready,
        handoff_started=plan.handoff_started,
        log_stream=plan.log_stream,
        processed_frames=plan.processed_frames,
        user_cancelled=plan.user_cancelled,
        video_writer=plan.video_writer,
        window_created=plan.window_created,
        plan=plan,
        automation=FakeAutomation(),
        calibration=calibration,
        estimator_config=object(),
        fullscreen=False,
        headless=True,
        max_frames=1,
        metadata_context={"source": "characterization"},
        window_name="characterization",
    )

    assert outcome.processed_frames == 1
    assert trace.events == [
        ("stage", "camera_open"),
        ("stage", "automation_started"),
        ("decision", "frame_consumed_without_handoff"),
        ("stop", "camera_closed"),
        ("stop", "automation_closed"),
    ]
