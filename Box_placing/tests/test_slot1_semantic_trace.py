"""Phase-0 semantic characterization of the observable slot-1 lifecycle.

The trace deliberately records state-machine and command-category semantics,
not timestamps, transforms, packet counts, or SDK object serialization.
"""

from __future__ import annotations

import json

import pytest

from parcel_pose_placing.pallet_control import (
    CombinedStreamError,
    ControllerPhase,
)
from parcel_pose_placing.pallet_place import (
    LOWERING_MODE,
    PlacementConfig,
    PlacementRequest,
    PlacementState,
    Slot1PlacementSequencer,
)
from parcel_pose_placing.pallet_servo import (
    PalletServoConfig,
    PalletServoObservation,
    PalletServoState,
    PalletSlot1Servo,
    WheelMotionMeasurement,
)

from _factories import placement_input
from test_slot1_motion_sequence import (
    CONFIG_PATH,
    build,
    placement_plan,
    teardown,
)


SemanticEvent = tuple[str, str, str, str]
FORCED_CANCEL_WARNING = (
    "FORCED RB-Y1 stream cancellation: carried-load support continuity "
    "is not acknowledged by a successor"
)


@pytest.fixture(scope="module")
def root_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _aligned_servo_stop_trace(root_config: dict) -> tuple[SemanticEvent, ...]:
    config = PalletServoConfig.from_root_config(root_config)
    servo = PalletSlot1Servo(config)
    servo.start(100.0)

    def observation(now_s: float) -> PalletServoObservation:
        return PalletServoObservation(
            timestamp_s=now_s,
            current_observed_feature_center_base=(0.70, 0.0),
            current_observed_feature_yaw_base_rad=0.0,
            demonstrated_body_reference_center_base=(0.70, 0.0),
            demonstrated_body_reference_yaw_base_rad=0.0,
            axis_branch=config.expected_axis_branch,
            reference_source="demonstrated_slot1",
        )

    zero_latched = None
    for now_s in (100.0, 100.1, 100.2, 100.4, 100.6, 100.8):
        zero_latched = servo.update(observation(now_s), now_s)
    assert zero_latched is not None
    assert zero_latched.state is PalletServoState.ARRIVAL_WHEEL_STOP

    stopped = None
    for now_s in (100.9, 101.05, 101.2, 101.35):
        stopped = servo.update(
            observation(now_s),
            now_s,
            WheelMotionMeasurement(now_s, 0.0, 0.0),
        )
    assert stopped is not None
    assert stopped.state is PalletServoState.ARRIVED_HOLD

    def command_kind(output) -> str:
        command = (output.vx_mps, output.vy_mps, output.wz_radps)
        return "zero_mobility" if command == (0.0, 0.0, 0.0) else "nonzero_mobility"

    return (
        (
            "alignment",
            zero_latched.state.value,
            zero_latched.reason,
            command_kind(zero_latched),
        ),
        (
            "alignment_stop",
            stopped.state.value,
            stopped.reason,
            command_kind(stopped),
        ),
    )


def _release_authorization_event(root_config: dict) -> SemanticEvent:
    sequencer = Slot1PlacementSequencer(
        PlacementConfig.from_root_config(root_config)
    )
    started = None
    for sequence, now_s in enumerate((100.0, 100.1, 100.2), start=1):
        started = sequencer.update(
            placement_input(
                now_s=now_s,
                sequence=sequence,
                demonstrated_place_pose=True,
            )
        )
    assert started is not None and started.descent_plan is not None
    plan = started.descent_plan
    at_place = {
        "controller_arm_mode": LOWERING_MODE,
        "demonstrated_place_pose": True,
        "right_target_base": plan.right_target_base,
        "left_target_base": plan.left_target_base,
    }
    seated = sequencer.update(
        placement_input(now_s=100.3, sequence=4, **at_place)
    )
    assert seated.state is PlacementState.SEATED
    authorized = sequencer.update(
        placement_input(now_s=100.7, sequence=5, **at_place)
    )
    assert authorized.request is PlacementRequest.SPREAD_RELEASE
    assert authorized.release_authorized
    return (
        "release",
        authorized.state.value,
        authorized.reason,
        authorized.request.value,
    )


def _forced_close_event(controller, stream) -> SemanticEvent:
    with pytest.warns(RuntimeWarning) as caught:
        assert controller.close(force=True)
    assert [(record.category, str(record.message)) for record in caught] == [
        (RuntimeWarning, FORCED_CANCEL_WARNING)
    ]
    assert stream.cancelled
    return (
        "teardown",
        controller.phase.value,
        "forced_stream_cancellation_warning",
        "stream_cancelled",
    )


def test_slot1_success_has_a_stable_semantic_trace(root_config) -> None:
    controller, robot, _config = build(root_config)
    try:
        telemetry = controller.telemetry()
        placement = controller.placement_telemetry()
        first_packet = robot.packets[0]
        ready_action = (
            "joint_ready_zero_base"
            if first_packet.arm_mode == "JOINT"
            and first_packet.mobility_velocity == (0.0, 0.0, 0.0)
            else "other_ready_command"
        )
        ready_reason = (
            "ready_transition_acknowledged"
            if telemetry.ready_transition_command_count == 1
            and placement.ready_posture_verified
            else "ready_transition_unverified"
        )
        trace: list[SemanticEvent] = [
            ("ready", telemetry.phase.value, ready_reason, ready_action),
            *_aligned_servo_stop_trace(root_config),
        ]

        assert controller.reverify_wheel_stop_after_stream_start(2.0).stopped
        controller.send_zero_mobility_hold(latch=True)
        plan, _delta = placement_plan(root_config)
        lowering = controller.start_cartesian_lowering_hold(descent_plan=plan)
        place_packet = robot.one_shot_packets[-1]
        trace.append(
            (
                "place",
                lowering.mode.value,
                lowering.target_source,
                "one_shot_cartesian"
                if place_packet.arm_mode == "CARTESIAN"
                and place_packet.mobility_velocity is None
                else "other_arm_command",
            )
        )
        trace.append(_release_authorization_event(root_config))

        retreat = controller.start_cartesian_release_hold()
        retreat_packet = robot.one_shot_packets[-1]
        trace.append(
            (
                "retreat",
                retreat.mode.value,
                retreat.target_source,
                "one_shot_cartesian"
                if retreat_packet.arm_mode == "CARTESIAN"
                and retreat_packet.mobility_velocity is None
                else "other_arm_command",
            )
        )
        trace.append(_forced_close_event(controller, robot.streams[-1]))

        assert tuple(trace) == (
            (
                "ready",
                "STEADY_HOLD",
                "ready_transition_acknowledged",
                "joint_ready_zero_base",
            ),
            (
                "alignment",
                "ARRIVAL_WHEEL_STOP",
                "arrival_candidate_zero_latched",
                "zero_mobility",
            ),
            (
                "alignment_stop",
                "ARRIVED_HOLD",
                "arrived_wheels_stopped",
                "zero_mobility",
            ),
            (
                "place",
                "CARTESIAN_PLACEMENT_LOWERING",
                "demonstrated_place_pose",
                "one_shot_cartesian",
            ),
            ("release", "RELEASING", "release_started", "SPREAD_RELEASE"),
            (
                "retreat",
                "CARTESIAN_PLACEMENT_RELEASE",
                "demonstrated_place_pose",
                "one_shot_cartesian",
            ),
            (
                "teardown",
                "CLOSED",
                "forced_stream_cancellation_warning",
                "stream_cancelled",
            ),
        )
    finally:
        if controller.phase is not ControllerPhase.CLOSED:
            teardown(controller)


def test_refused_place_one_shot_has_a_zero_latched_stop_trace(root_config) -> None:
    controller, robot, _config = build(root_config)
    try:
        assert controller.reverify_wheel_stop_after_stream_start(2.0).stopped
        controller.send_zero_mobility_hold(latch=True)
        plan, _delta = placement_plan(root_config)
        robot.one_shot_raises = True
        with pytest.raises(CombinedStreamError, match="mobility stream was open"):
            controller.start_cartesian_lowering_hold(descent_plan=plan)
        placement = controller.placement_telemetry()
        trace = (
            (
                "place_refusal_stop",
                placement.arm_mode.value,
                placement.last_reason or "missing_reason",
                "zero_latched" if placement.zero_latched else "zero_not_latched",
            ),
            _forced_close_event(controller, robot.streams[-1]),
        )
        assert trace == (
            (
                "place_refusal_stop",
                "CARTESIAN_PLACEMENT_LOWERING",
                "arm_send_once_rejected_while_stream_open",
                "zero_latched",
            ),
            (
                "teardown",
                "CLOSED",
                "forced_stream_cancellation_warning",
                "stream_cancelled",
            ),
        )
    finally:
        if controller.phase is not ControllerPhase.CLOSED:
            teardown(controller)
