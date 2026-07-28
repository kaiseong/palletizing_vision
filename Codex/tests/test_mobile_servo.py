from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from parcel_pose.angles import line_angle_difference_rad
from parcel_pose.mobile_servo import (
    MAX_ALLOWED_ANGULAR_SPEED_RADPS,
    MAX_ALLOWED_LINEAR_SPEED_MPS,
    MobilityCommandPumpError,
    MobileVisualServo,
    PoseMeasurement,
    RBY1CommandPumpConfig,
    RBY1MobilityCommandPump,
    RBY1MobilityStream,
    RobotMotionDisabledError,
    ServoConfig,
    ServoState,
    StreamFeedbackError,
    VelocityCommand,
)


def _pose(
    x: float,
    y: float,
    timestamp: float,
    yaw_deg: float = 90.0,
) -> PoseMeasurement:
    return PoseMeasurement(
        (x, y),
        timestamp,
        long_axis_yaw_rad=np.deg2rad(yaw_deg),
    )


def _prime(
    servo: MobileVisualServo,
    xy: tuple[float, float],
    *,
    start_s: float = 0.0,
    yaw_deg: float = 90.0,
) -> float:
    servo.start(start_s)
    for index in range(1, 4):
        now = start_s + 0.1 * index
        servo.step(_pose(*xy, now, yaw_deg=yaw_deg), now_s=now)
    return start_s + 0.3


def test_defaults_encode_requested_target_and_hard_speed_limit() -> None:
    config = ServoConfig()

    assert config.target_xy_m == (0.740, 0.0)
    assert line_angle_difference_rad(
        config.target_long_axis_yaw_rad,
        np.deg2rad(90.0),
    ) == pytest.approx(0.0)
    assert line_angle_difference_rad(
        config.target_long_axis_yaw_rad,
        np.deg2rad(-90.0),
    ) == pytest.approx(0.0)
    assert config.filter_window == 3
    assert config.jump_threshold_m == pytest.approx(0.030)
    assert config.max_linear_speed_mps == MAX_ALLOWED_LINEAR_SPEED_MPS
    with pytest.raises(ValueError, match="0.08 m/s"):
        ServoConfig(max_linear_speed_mps=0.081)
    assert config.max_angular_speed_radps == MAX_ALLOWED_ANGULAR_SPEED_RADPS
    with pytest.raises(ValueError, match="0.10 rad/s"):
        ServoConfig(max_angular_speed_radps=0.101)


def test_legacy_positional_config_and_invalid_pose_meanings_are_preserved() -> None:
    config = ServoConfig((0.740, 0.0), 0.5)
    assert config.proportional_gain_per_s == pytest.approx(0.5)
    assert config.target_long_axis_yaw_rad == pytest.approx(np.pi / 2.0)

    legacy_invalid = PoseMeasurement((0.900, 0.0), 0.1, False)
    assert not legacy_invalid.valid
    servo = MobileVisualServo()
    servo.start(0.0)
    decision = servo.step(legacy_invalid, now_s=0.1)
    assert decision.reason == "pose_invalid"
    assert decision.command.is_zero


def test_idle_controller_never_commands_motion() -> None:
    servo = MobileVisualServo()

    decision = servo.step(_pose(1.0, 0.2, 1.0), now_s=1.0)

    assert decision.state is ServoState.IDLE
    assert decision.command.is_zero
    assert decision.reason == "inactive"


def test_three_sample_median_sign_and_speed_cap_at_aligned_yaw() -> None:
    servo = MobileVisualServo(
        ServoConfig(max_linear_acceleration_mps2=10.0)
    )
    servo.start(0.0)

    assert servo.step(_pose(0.800, 0.020, 0.1), now_s=0.1).command.is_zero
    assert servo.step(_pose(0.820, 0.000, 0.2), now_s=0.2).command.is_zero
    decision = servo.step(_pose(0.800, 0.020, 0.3), now_s=0.3)

    assert decision.filtered_xy_m == pytest.approx((0.800, 0.020))
    assert decision.error_xy_m == pytest.approx((0.060, 0.020))
    assert decision.command.vx_mps > 0.0
    assert decision.command.vy_mps > 0.0
    assert decision.command.wz_radps == 0.0
    assert decision.command.linear_norm_mps <= MAX_ALLOWED_LINEAR_SPEED_MPS

    capped = MobileVisualServo(ServoConfig(max_linear_acceleration_mps2=10.0))
    _prime(capped, (1.0, 1.0))
    assert capped.step(_pose(1.0, 1.0, 0.4), now_s=0.4).command.linear_norm_mps \
        == pytest.approx(MAX_ALLOWED_LINEAR_SPEED_MPS)


@pytest.mark.parametrize(
    ("yaw_deg", "expected_sign"),
    [(89.0, -1.0), (-89.0, 1.0), (80.0, -1.0), (100.0, 1.0)],
)
def test_shortest_mod180_yaw_error_drives_expected_wz_sign(
    yaw_deg: float,
    expected_sign: float,
) -> None:
    servo = MobileVisualServo(
        ServoConfig(
            max_linear_acceleration_mps2=10.0,
            max_angular_acceleration_radps2=10.0,
            arrival_yaw_inner_rad=np.deg2rad(0.1),
            arrival_yaw_outer_rad=np.deg2rad(0.2),
        )
    )
    _prime(servo, servo.config.target_xy_m, yaw_deg=yaw_deg)

    decision = servo.step(
        _pose(*servo.config.target_xy_m, 0.4, yaw_deg=yaw_deg),
        now_s=0.4,
    )

    assert decision.yaw_error_rad is not None
    assert np.sign(decision.yaw_error_rad) == expected_sign
    assert np.sign(decision.command.wz_radps) == expected_sign
    assert abs(decision.command.wz_radps) <= MAX_ALLOWED_ANGULAR_SPEED_RADPS


def test_xy_and_yaw_commands_are_emitted_together_with_orbit_feedforward() -> None:
    config = ServoConfig(
        max_linear_acceleration_mps2=10.0,
        max_angular_acceleration_radps2=10.0,
    )
    servo = MobileVisualServo(config)
    _prime(servo, (0.800, 0.020), yaw_deg=80.0)

    decision = servo.step(_pose(0.800, 0.020, 0.4, 80.0), now_s=0.4)

    assert decision.command.linear_norm_mps > 0.0
    assert decision.command.wz_radps < 0.0
    assert decision.command.linear_norm_mps <= config.max_linear_speed_mps
    assert abs(decision.command.wz_radps) <= config.max_angular_speed_radps

    # At zero XY error, the linear component is the orbit velocity needed to
    # keep the stationary parcel at the same relative centre while yawing.
    orbit = MobileVisualServo(config)
    _prime(orbit, config.target_xy_m, yaw_deg=80.0)
    orbit_decision = orbit.step(
        _pose(*config.target_xy_m, 0.4, yaw_deg=80.0),
        now_s=0.4,
    )
    expected_ratio = -config.target_xy_m[0]
    assert orbit_decision.command.vx_mps == pytest.approx(0.0, abs=1e-12)
    assert (
        orbit_decision.command.vy_mps / orbit_decision.command.wz_radps
        == pytest.approx(expected_ratio)
    )


def test_yaw_slew_limit_bounds_angular_velocity_change() -> None:
    servo = MobileVisualServo(
        ServoConfig(max_angular_acceleration_radps2=0.20)
    )
    _prime(servo, (0.740, 0.0), yaw_deg=70.0)

    first = servo.step(_pose(0.740, 0.0, 0.4, 70.0), now_s=0.4).command
    second = servo.step(_pose(0.740, 0.0, 0.5, 70.0), now_s=0.5).command

    assert first.wz_radps == pytest.approx(-0.04)
    assert second.wz_radps == pytest.approx(-0.06)
    assert second.wz_radps - first.wz_radps == pytest.approx(-0.02)
    assert first.vy_mps / first.wz_radps == pytest.approx(-0.740)


def test_signed_yaw_display_wrap_stays_a_small_continuous_line_error() -> None:
    servo = MobileVisualServo(
        ServoConfig(
            arrival_yaw_inner_rad=np.deg2rad(0.05),
            arrival_yaw_outer_rad=np.deg2rad(0.10),
        )
    )
    servo.start(0.0)
    decisions = []
    for index, yaw_deg in enumerate((89.8, -89.8, 89.9, -89.9), start=1):
        now = 0.1 * index
        decisions.append(
            servo.step(_pose(0.740, 0.0, now, yaw_deg), now_s=now)
        )

    assert all(decision.measurement_accepted for decision in decisions)
    assert all(decision.state is not ServoState.POSE_LOST for decision in decisions)
    assert decisions[-1].yaw_error_rad is not None
    assert abs(np.rad2deg(decisions[-1].yaw_error_rad)) < 0.3
    assert abs(decisions[-1].command.wz_radps) < 0.01


def test_per_update_slew_limit_bounds_velocity_change() -> None:
    servo = MobileVisualServo(
        ServoConfig(max_linear_acceleration_mps2=0.15)
    )
    _prime(servo, (0.900, 0.0))

    first = servo.step(_pose(0.900, 0.0, 0.4), now_s=0.4).command
    second = servo.step(_pose(0.900, 0.0, 0.5), now_s=0.5).command

    assert first.linear_norm_mps == pytest.approx(0.030)
    assert second.linear_norm_mps == pytest.approx(0.045)
    assert second.linear_norm_mps - first.linear_norm_mps == pytest.approx(0.015)


def test_30mm_jump_is_rejected_then_three_consistent_samples_reseed() -> None:
    servo = MobileVisualServo(ServoConfig(max_linear_acceleration_mps2=10.0))
    _prime(servo, (0.800, 0.0))

    first = servo.step(_pose(0.850, 0.0, 0.4), now_s=0.4)
    second = servo.step(_pose(0.851, 0.0, 0.5), now_s=0.5)
    reseeded = servo.step(_pose(0.849, 0.0, 0.6), now_s=0.6)

    assert first.state is ServoState.POSE_LOST
    assert first.reason == "jump_rejected"
    assert first.command.is_zero
    assert second.command.is_zero
    assert reseeded.measurement_accepted
    assert reseeded.filtered_xy_m == pytest.approx((0.850, 0.0))
    assert reseeded.state is ServoState.TRACKING


def test_yaw_jump_is_rejected_then_three_consistent_line_angles_reseed() -> None:
    servo = MobileVisualServo(
        ServoConfig(
            max_linear_acceleration_mps2=10.0,
            max_angular_acceleration_radps2=10.0,
        )
    )
    _prime(servo, (0.740, 0.0), yaw_deg=90.0)

    first = servo.step(_pose(0.740, 0.0, 0.4, 60.0), now_s=0.4)
    second = servo.step(_pose(0.740, 0.0, 0.5, 59.0), now_s=0.5)
    reseeded = servo.step(_pose(0.740, 0.0, 0.6, 61.0), now_s=0.6)

    assert first.state is ServoState.POSE_LOST
    assert first.reason == "jump_rejected"
    assert first.command.is_zero
    assert second.command.is_zero
    assert reseeded.measurement_accepted
    assert reseeded.filtered_long_axis_yaw_rad == pytest.approx(np.deg2rad(60.0))
    assert reseeded.state is ServoState.TRACKING
    assert reseeded.command.wz_radps < 0.0


@pytest.mark.parametrize(
    ("measurement", "now_s", "reason"),
    [
        (None, 0.4, "pose_missing"),
        (PoseMeasurement.invalid(0.4), 0.4, "pose_invalid"),
        (_pose(0.900, 0.0, 0.0), 0.4, "pose_stale"),
    ],
)
def test_missing_invalid_or_stale_pose_commands_immediate_zero(
    measurement: PoseMeasurement | None,
    now_s: float,
    reason: str,
) -> None:
    servo = MobileVisualServo(ServoConfig(max_linear_acceleration_mps2=10.0))
    _prime(servo, (0.900, 0.0))
    moving = servo.step(_pose(0.900, 0.0, 0.35), now_s=0.35)
    assert not moving.command.is_zero

    stopped = servo.step(measurement, now_s=now_s)

    assert stopped.state is ServoState.POSE_LOST
    assert stopped.command.is_zero
    assert stopped.reason == reason


def test_missing_pose_stops_an_active_angular_command_without_slew() -> None:
    servo = MobileVisualServo(
        ServoConfig(max_angular_acceleration_radps2=10.0)
    )
    _prime(servo, (0.740, 0.0), yaw_deg=70.0)
    moving = servo.step(_pose(0.740, 0.0, 0.4, 70.0), now_s=0.4)
    assert moving.command.wz_radps != 0.0

    stopped = servo.step(None, now_s=0.5)

    assert stopped.state is ServoState.POSE_LOST
    assert stopped.command.is_zero
    assert stopped.command.wz_radps == 0.0


def test_arrival_requires_inner_entry_outer_hysteresis_frames_and_time_once() -> None:
    config = ServoConfig(max_linear_acceleration_mps2=10.0)
    servo = MobileVisualServo(config)
    servo.start(0.0)

    times = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.64)
    decisions = [
        servo.step(_pose(0.740, 0.0, now), now_s=now)
        for now in times
    ]
    assert all(not decision.handoff_ready for decision in decisions)

    arrived = servo.step(_pose(0.740, 0.0, 0.66), now_s=0.66)
    repeated = servo.step(_pose(0.740, 0.0, 0.70), now_s=0.70)

    assert arrived.state is ServoState.ARRIVED
    assert arrived.command.is_zero
    assert arrived.handoff_ready
    assert not repeated.handoff_ready

    reset = MobileVisualServo(config)
    reset.start(0.0)
    reset.step(_pose(0.740, 0.0, 0.1), now_s=0.1)
    reset.step(_pose(0.740, 0.0, 0.2), now_s=0.2)
    assert reset.step(_pose(0.740, 0.0, 0.3), now_s=0.3).state \
        is ServoState.HOLDING
    outside = reset.step(_pose(0.760, 0.0, 0.4), now_s=0.4)
    assert outside.state is ServoState.TRACKING
    assert not outside.handoff_ready


def test_arrival_requires_yaw_and_resets_when_yaw_leaves_outer_band() -> None:
    config = ServoConfig(
        max_linear_acceleration_mps2=10.0,
        max_angular_acceleration_radps2=10.0,
    )
    servo = MobileVisualServo(config)
    servo.start(0.0)

    for index in range(1, 8):
        now = index * 0.1
        decision = servo.step(_pose(0.740, 0.0, now, 75.0), now_s=now)
        assert not decision.handoff_ready
    assert decision.state is ServoState.TRACKING
    assert decision.command.wz_radps != 0.0

    # Re-seed after the physically plausible gradual rotation into tolerance.
    for index, yaw_deg in enumerate((80.0, 84.0, 87.5), start=8):
        now = index * 0.1
        servo.step(_pose(0.740, 0.0, now, yaw_deg), now_s=now)
    assert servo.step(_pose(0.740, 0.0, 1.1, 90.0), now_s=1.1).state \
        in {ServoState.TRACKING, ServoState.HOLDING}
    assert servo.step(_pose(0.740, 0.0, 1.2, 90.0), now_s=1.2).state \
        is ServoState.HOLDING

    left_outer = servo.step(_pose(0.740, 0.0, 1.3, 83.0), now_s=1.3)
    assert left_outer.state is ServoState.TRACKING
    assert not left_outer.handoff_ready


def test_closed_loop_relative_pose_converges_through_outlier_and_dropout() -> None:
    servo = MobileVisualServo()
    servo.start(0.0)
    relative_xy = np.asarray((0.950, -0.080), dtype=np.float64)
    target = np.asarray(servo.config.target_xy_m, dtype=np.float64)
    handoffs = 0

    for index in range(1, 500):
        now = index * 0.05
        if index == 20:
            measurement = _pose(*(relative_xy + (0.05, 0.0)), now)
        elif index == 40:
            measurement = None
        else:
            noise = 0.0005 * np.asarray((np.sin(index), np.cos(index)))
            measurement = _pose(*(relative_xy + noise), now)
        decision = servo.step(measurement, now_s=now)
        relative_xy -= np.asarray(
            (decision.command.vx_mps, decision.command.vy_mps)
        ) * 0.05
        handoffs += int(decision.handoff_ready)
        if decision.state is ServoState.ARRIVED:
            break

    assert servo.state is ServoState.ARRIVED
    assert np.linalg.norm(relative_xy - target) <= servo.config.arrival_outer_m
    assert handoffs == 1
    assert not servo.step(_pose(*relative_xy, now + 0.05), now_s=now + 0.05).handoff_ready


def test_world_fixed_se2_closed_loop_converges_xy_and_horizontal_yaw() -> None:
    servo = MobileVisualServo()
    servo.start(0.0)
    relative_xy = np.asarray((0.950, -0.080), dtype=np.float64)
    relative_yaw = np.deg2rad(60.0)
    target_xy = np.asarray(servo.config.target_xy_m, dtype=np.float64)

    for index in range(1, 600):
        now = index * 0.05
        decision = servo.step(
            _pose(
                *relative_xy,
                now,
                yaw_deg=float(np.rad2deg(relative_yaw)),
            ),
            now_s=now,
        )
        command_xy = np.asarray(
            (decision.command.vx_mps, decision.command.vy_mps),
            dtype=np.float64,
        )
        rotation_induced = decision.command.wz_radps * np.asarray(
            (relative_xy[1], -relative_xy[0]),
            dtype=np.float64,
        )
        relative_xy += (rotation_induced - command_xy) * 0.05
        relative_yaw -= decision.command.wz_radps * 0.05
        if decision.state is ServoState.ARRIVED:
            break

    assert servo.state is ServoState.ARRIVED
    assert np.linalg.norm(relative_xy - target_xy) <= servo.config.arrival_outer_m
    assert abs(
        line_angle_difference_rad(
            relative_yaw,
            servo.config.target_long_axis_yaw_rad,
        )
    ) <= servo.config.arrival_yaw_outer_rad


def test_timeout_and_explicit_abort_are_terminal_zero_states() -> None:
    servo = MobileVisualServo(ServoConfig(timeout_s=1.0))
    servo.start(0.0)

    timeout = servo.step(_pose(0.900, 0.0, 1.0), now_s=1.0)
    after_timeout = servo.step(_pose(0.900, 0.0, 1.1), now_s=1.1)

    assert timeout.state is ServoState.ABORTED
    assert timeout.reason == "timeout"
    assert timeout.command.is_zero
    assert after_timeout.state is ServoState.ABORTED

    servo.start(2.0)
    aborted = servo.abort("operator_stop", 2.1)
    assert aborted.state is ServoState.ABORTED
    assert aborted.reason == "operator_stop"
    assert aborted.command.is_zero


def test_continuous_pose_loss_has_its_own_abort_timeout() -> None:
    servo = MobileVisualServo(ServoConfig(lost_abort_after_s=2.0, timeout_s=30.0))
    _prime(servo, (0.900, 0.0))

    lost = servo.step(None, now_s=0.4)
    timed_out = servo.step(None, now_s=2.4)

    assert lost.state is ServoState.POSE_LOST
    assert lost.command.is_zero
    assert timed_out.state is ServoState.ABORTED
    assert timed_out.reason == "pose_lost_timeout"
    assert timed_out.command.is_zero


class _FakeBuilder:
    def __init__(self, kind: str, events: list[tuple]) -> None:
        self.kind = kind
        self.events = events

    def set_control_hold_time(self, value):
        self.events.append((self.kind, "hold", value))
        return self

    def set_command_header(self, value):
        self.events.append((self.kind, "header", value.kind))
        return self

    def set_minimum_time(self, value):
        self.events.append((self.kind, "minimum_time", value))
        return self

    def set_velocity(self, linear, angular):
        self.events.append(
            (self.kind, "velocity", tuple(np.asarray(linear)), angular)
        )
        return self

    def set_acceleration_limit(self, linear, angular):
        self.events.append(
            (self.kind, "acceleration", tuple(np.asarray(linear)), angular)
        )
        return self

    def set_mobility_command(self, value):
        self.events.append((self.kind, "mobility", value.kind))
        return self

    def set_command(self, value):
        self.events.append((self.kind, "command", value.kind))
        return self


class _FakeStream:
    def __init__(self, events: list[tuple]) -> None:
        self.events = events

    def send_command(self, command, timeout_ms):
        self.events.append(("stream", "send", timeout_ms))
        return _mobility_feedback()

    def cancel(self):
        self.events.append(("stream", "cancel"))

    def wait_for(self, timeout_ms):
        self.events.append(("stream", "wait", timeout_ms))
        return True


class _FakeRobot:
    def __init__(self, events: list[tuple]) -> None:
        self.events = events
        self.stream = _FakeStream(events)

    def create_command_stream(self, priority):
        self.events.append(("robot", "create_stream", priority))
        return self.stream


def _fake_sdk(events: list[tuple]):
    return SimpleNamespace(
        CommandHeaderBuilder=lambda: _FakeBuilder("header", events),
        SE2VelocityCommandBuilder=lambda: _FakeBuilder("se2", events),
        ComponentBasedCommandBuilder=lambda: _FakeBuilder("component", events),
        RobotCommandBuilder=lambda: _FakeBuilder("robot_command", events),
    )


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


def test_rby1_stream_is_opt_in_and_shutdown_orders_zeros_cancel_then_wait() -> None:
    events: list[tuple] = []
    robot = _FakeRobot(events)
    disabled = RBY1MobilityStream(robot, sdk_module=_fake_sdk(events))

    with pytest.raises(RobotMotionDisabledError):
        disabled.open()
    assert events == []

    adapter = RBY1MobilityStream(
        robot,
        execute=True,
        sdk_module=_fake_sdk(events),
    ).open()
    adapter.send(VelocityCommand(0.02, -0.01, 0.04))
    adapter.stop_and_release()

    stream_events = [event for event in events if event[0] == "stream"]
    assert stream_events == [
        ("stream", "send", 250),
        ("stream", "send", 250),
        ("stream", "send", 250),
        ("stream", "send", 250),
        ("stream", "cancel"),
        ("stream", "wait", 2000),
    ]
    velocities = [
        event[2]
        for event in events
        if event[:2] == ("se2", "velocity")
    ]
    assert velocities[0] == pytest.approx((0.02, -0.01))
    assert velocities[1:] == [pytest.approx((0.0, 0.0))] * 3
    angular_velocities = [
        event[3]
        for event in events
        if event[:2] == ("se2", "velocity")
    ]
    assert angular_velocities == pytest.approx((0.04, 0.0, 0.0, 0.0))
    assert ("se2", "acceleration", (0.15, 0.15), 0.20) in events
    assert ("se2", "minimum_time", 0.05) in events
    assert ("header", "hold", 1.0) in events
    assert not adapter.is_open


def test_rby1_stream_accepts_bounded_yaw_and_rejects_excess_speed() -> None:
    events: list[tuple] = []
    adapter = RBY1MobilityStream(
        _FakeRobot(events),
        execute=True,
        sdk_module=_fake_sdk(events),
    ).open()

    for angular in (
        -MAX_ALLOWED_ANGULAR_SPEED_RADPS,
        MAX_ALLOWED_ANGULAR_SPEED_RADPS,
    ):
        adapter.send(VelocityCommand(0.0, 0.0, angular))
    assert [
        event[3]
        for event in events
        if event[:2] == ("se2", "velocity")
    ] == pytest.approx(
        (-MAX_ALLOWED_ANGULAR_SPEED_RADPS, MAX_ALLOWED_ANGULAR_SPEED_RADPS)
    )
    with pytest.raises(ValueError, match="angular velocity"):
        adapter.send(
            VelocityCommand(0.0, 0.0, MAX_ALLOWED_ANGULAR_SPEED_RADPS + 0.001)
        )
    with pytest.raises(ValueError, match="0.08 m/s"):
        adapter.send(VelocityCommand(0.081, 0.0, 0.0))
    adapter.close()


def test_terminal_stream_feedback_is_rejected_and_released() -> None:
    events: list[tuple] = []

    class TerminalFeedbackStream(_FakeStream):
        def send_command(self, command, timeout_ms):
            self.events.append(("stream", "send_terminal", timeout_ms))
            return _mobility_feedback(status=3, finish_code=2)

    robot = _FakeRobot(events)
    robot.stream = TerminalFeedbackStream(events)
    adapter = RBY1MobilityStream(
        robot,
        execute=True,
        sdk_module=_fake_sdk(events),
    ).open()

    with pytest.raises(StreamFeedbackError, match="terminated"):
        adapter.send(VelocityCommand(0.01, 0.0, 0.0))

    assert not adapter.is_open
    assert events[-2:] == [
        ("stream", "cancel"),
        ("stream", "wait", 2000),
    ]


def test_pump_start_waits_for_running_feedback_not_just_initializing() -> None:
    events: list[tuple] = []

    class InitializingThenRunningStream(_FakeStream):
        def __init__(self, stream_events: list[tuple]) -> None:
            super().__init__(stream_events)
            self.feedback_statuses = [1, 1, 2]

        def send_command(self, command, timeout_ms):
            status = self.feedback_statuses.pop(0) if self.feedback_statuses else 2
            self.events.append(("stream", "send_status", status))
            return _mobility_feedback(status=status)

    robot = _FakeRobot(events)
    robot.stream = InitializingThenRunningStream(events)
    adapter = RBY1MobilityStream(
        robot,
        execute=True,
        sdk_module=_fake_sdk(events),
    ).open()
    pump = RBY1MobilityCommandPump(
        adapter,
        config=RBY1CommandPumpConfig(
            send_rate_hz=100.0,
            command_stale_after_s=0.05,
        ),
    ).start()

    assert [event[2] for event in events if event[1] == "send_status"][:3] == [
        1,
        1,
        2,
    ]
    pump.close()

    assert pump.is_closed
    assert not adapter.is_open


def test_stream_is_not_reused_when_shutdown_wait_raises() -> None:
    events: list[tuple] = []

    class RaisingWaitStream(_FakeStream):
        def wait_for(self, timeout_ms):
            self.events.append(("stream", "wait", timeout_ms))
            raise RuntimeError("transport closed")

    robot = _FakeRobot(events)
    robot.stream = RaisingWaitStream(events)
    adapter = RBY1MobilityStream(
        robot,
        execute=True,
        sdk_module=_fake_sdk(events),
    ).open()

    with pytest.raises(RuntimeError, match="transport closed"):
        adapter.stop_and_release()

    assert not adapter.is_open


def test_send_failure_invalidates_and_best_effort_releases_stream() -> None:
    events: list[tuple] = []

    class FailingSendStream(_FakeStream):
        def send_command(self, command, timeout_ms):
            self.events.append(("stream", "send_failed", timeout_ms))
            raise RuntimeError("feedback timeout")

    robot = _FakeRobot(events)
    robot.stream = FailingSendStream(events)
    adapter = RBY1MobilityStream(
        robot,
        execute=True,
        sdk_module=_fake_sdk(events),
    ).open()

    with pytest.raises(RuntimeError, match="feedback timeout"):
        adapter.send(VelocityCommand(0.02, 0.0, 0.0))

    assert not adapter.is_open
    assert events[-2:] == [
        ("stream", "cancel"),
        ("stream", "wait", 2000),
    ]


class _PumpStreamFake:
    def __init__(self, *, fail_after: int | None = None) -> None:
        self.is_open = True
        self.fail_after = fail_after
        self.sends: list[tuple[int, float, VelocityCommand]] = []
        self.released = False

    def send(self, command: VelocityCommand):
        if self.fail_after is not None and len(self.sends) >= self.fail_after:
            raise RuntimeError("simulated stream expiry")
        self.sends.append((threading.get_ident(), time.monotonic(), command))
        return _mobility_feedback()

    def stop_and_release(self) -> None:
        self.released = True
        self.is_open = False

    def cancel_and_wait(self) -> None:
        self.released = True
        self.is_open = False


def _wait_for(predicate, *, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition did not become true before timeout")
        time.sleep(0.002)


def test_fixed_rate_pump_repeats_latest_command_then_watchdogs_to_zero() -> None:
    stream = _PumpStreamFake()
    pump = RBY1MobilityCommandPump(
        stream,
        config=RBY1CommandPumpConfig(
            send_rate_hz=100.0,
            command_stale_after_s=0.05,
            zero_ack_repetitions=3,
        ),
    ).start()

    pump.publish(VelocityCommand(0.02, -0.01, 0.04))
    _wait_for(lambda: sum(not item[2].is_zero for item in stream.sends) >= 2)
    _wait_for(
        lambda: any(item[2].is_zero for item in stream.sends[3:]),
        timeout_s=1.0,
    )
    nonzero_commands = [item[2] for item in stream.sends if not item[2].is_zero]
    assert nonzero_commands
    assert all(command.wz_radps == pytest.approx(0.04) for command in nonzero_commands)

    pump.latch_zero_and_wait()
    with pytest.raises(MobilityCommandPumpError, match="zero-latched"):
        pump.publish(VelocityCommand(0.01, 0.0, 0.0))
    pump.stop_and_release()

    sender_threads = {item[0] for item in stream.sends}
    assert len(sender_threads) == 1
    assert threading.get_ident() not in sender_threads
    assert stream.released
    assert not pump.is_running


def test_pump_surfaces_background_stream_failure_and_releases() -> None:
    stream = _PumpStreamFake(fail_after=2)
    pump = RBY1MobilityCommandPump(
        stream,
        config=RBY1CommandPumpConfig(
            send_rate_hz=100.0,
            command_stale_after_s=0.05,
        ),
    ).start()
    pump.publish(VelocityCommand(0.01, 0.0, 0.0))
    _wait_for(lambda: pump.last_error is not None)

    with pytest.raises(MobilityCommandPumpError, match="simulated stream expiry"):
        pump.raise_if_failed()
    with pytest.raises(MobilityCommandPumpError, match="simulated stream expiry"):
        pump.stop_and_release()

    assert stream.released


def test_startup_timeout_cancels_blocked_send_and_joins_sender() -> None:
    class BlockingStartupStream(_PumpStreamFake):
        def __init__(self) -> None:
            super().__init__()
            self.send_entered = threading.Event()
            self.send_unblocked = threading.Event()
            self.cancel_count = 0

        def send(self, command: VelocityCommand):
            self.send_entered.set()
            self.send_unblocked.wait(timeout=1.0)
            if not self.is_open:
                raise RuntimeError("startup stream canceled")
            return super().send(command)

        def cancel_and_wait(self) -> None:
            self.cancel_count += 1
            self.is_open = False
            self.released = True
            self.send_unblocked.set()

    stream = BlockingStartupStream()
    pump = RBY1MobilityCommandPump(
        stream,
        config=RBY1CommandPumpConfig(
            send_rate_hz=100.0,
            command_stale_after_s=0.05,
            startup_timeout_s=0.03,
            join_timeout_s=0.05,
        ),
    )

    try:
        with pytest.raises(MobilityCommandPumpError, match="timed out"):
            pump.start()
    finally:
        stream.send_unblocked.set()

    assert stream.send_entered.is_set()
    assert stream.cancel_count == 1
    assert stream.released
    assert pump.is_closed
    assert not pump.is_running
    assert pump._thread is not None
    assert not pump._thread.is_alive()


def test_late_send_failure_after_zero_ack_blocks_successful_release() -> None:
    stream = _PumpStreamFake()
    pump = RBY1MobilityCommandPump(
        stream,
        config=RBY1CommandPumpConfig(
            send_rate_hz=100.0,
            command_stale_after_s=0.05,
            zero_ack_repetitions=3,
        ),
    ).start()
    original_latch = pump.latch_zero_and_wait
    failure_seen = threading.Event()

    def latch_then_fail() -> None:
        original_latch()
        stream.fail_after = len(stream.sends)
        _wait_for(lambda: pump.last_error is not None)
        failure_seen.set()

    pump.latch_zero_and_wait = latch_then_fail

    with pytest.raises(MobilityCommandPumpError, match="simulated stream expiry"):
        pump.stop_and_release()

    assert failure_seen.is_set()
    assert stream.released
    assert pump.is_closed
    assert not pump.is_running
