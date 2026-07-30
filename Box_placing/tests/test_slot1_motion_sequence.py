"""End-to-end slot-1 motion, driven offline through a recording fake SDK.

Every packet the controller hands to the SDK is captured, so this file is both a
regression test and a simulator: run it with ``-s`` and the timeline of commanded
motion prints out.  Nothing here touches a robot, a camera, or rby1_sdk.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
        tests/test_slot1_motion_sequence.py -s -k timeline
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from parcel_pose_placing.pallet_control import (
    CombinedStreamError,
    ArmStreamMode,
    ControllerPhase,
    PalletControlConfig,
    RBY1PalletController,
)
from parcel_pose_placing.pallet_place import PlacementConfig

from _fake_rby1 import FakeRobot, FakeSdk
from _factories import descent_plan

CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "rby1m_v1_2_pallet_slot1_nominal.json"
)
# The measured slot-1 ready pose from the physical logs.
RIGHT_EEF = (1.00046, -0.16948, 0.73157)
LEFT_EEF = (1.00046, 0.16948, 0.73157)
MEASURED_SEPARATION_M = LEFT_EEF[1] - RIGHT_EEF[1]


@pytest.fixture(scope="module")
def root_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def build(root_config: dict):
    """Connect, bootstrap, open the stream: the state just before base motion."""

    config = PalletControlConfig.from_root_config(root_config)
    robot = FakeRobot(
        right_eef_xyz=RIGHT_EEF, left_eef_xyz=LEFT_EEF, ready_pose=config.ready_pose
    )
    controller = RBY1PalletController(
        execute=True,
        config=config,
        sdk_module=FakeSdk(robot),
        fk_provider=robot.fk_provider,
    )
    controller.connect()
    controller.bootstrap_loaded_slot1_ready(loaded_box_acknowledged=True)
    command_id = controller.send_ready_transition_once(
        config.ready_transition_minimum_time_s
    )
    controller.wait_ready_transition_ack(command_id)
    controller.start_combined_stream()
    return controller, robot, config


def teardown(controller: RBY1PalletController) -> None:
    try:
        controller.close(force=True)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# stage 1: ready transition, the first packet that binds controller types
# --------------------------------------------------------------------------- #
def test_the_ready_transition_carries_the_body_and_zero_mobility(root_config) -> None:
    """The one all-joint Position move is the only body command in the stream."""

    controller, robot, config = build(root_config)
    try:
        first = robot.packets[0]
        assert first.arm_mode == "JOINT", (
            "the ready transition is a joint Position move; Cartesian binding is "
            "no longer needed because the stream never carries the arms again"
        )
        assert first.mobility_velocity == (0.0, 0.0, 0.0)
        assert first.minimum_time_s == pytest.approx(
            config.ready_transition_minimum_time_s
        )
        assert first.control_hold_time_s == pytest.approx(config.control_hold_time_s)
        np.testing.assert_allclose(first.torso_position, config.ready_pose.torso_rad)
        np.testing.assert_allclose(first.head_position, config.ready_pose.head_rad)
        np.testing.assert_allclose(
            first.right_arm_joint_position, config.ready_pose.right_arm_rad
        )
        np.testing.assert_allclose(
            first.left_arm_joint_position, config.ready_pose.left_arm_rad
        )
    finally:
        teardown(controller)


def test_every_streamed_packet_after_the_ready_move_is_mobility_only(
    root_config,
) -> None:
    """Omitting the body is what leaves the arms free for one-shot commands."""

    controller, robot, config = build(root_config)
    try:
        _wait_for_packets(robot, 3)
        streamed = robot.packets[1:]
        assert streamed, "the steady pump never sent a packet"
        for index, packet in enumerate(streamed, start=1):
            assert packet.right_arm is None, f"packet {index} carried a right arm"
            assert packet.left_arm is None, f"packet {index} carried a left arm"
            assert packet.torso_position is None, f"packet {index} carried a torso"
            assert packet.head_position is None, f"packet {index} carried a head"
            assert packet.mobility_velocity is not None
    finally:
        teardown(controller)


def test_ready_one_shot_minimum_time_is_three_seconds(root_config) -> None:
    config = PalletControlConfig.from_root_config(root_config)
    assert config.ready_transition_minimum_time_s == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# stage 2: the loaded hold squeeze
# --------------------------------------------------------------------------- #
def test_the_ready_move_asks_for_the_approved_posture_only(root_config) -> None:
    """No inward squeeze: the hands are commanded exactly where the posture says."""

    controller, robot, config = build(root_config)
    try:
        packet = robot.packets[0]
        np.testing.assert_allclose(
            packet.right_arm_joint_position, config.ready_pose.right_arm_rad, atol=1e-12
        )
        np.testing.assert_allclose(
            packet.left_arm_joint_position, config.ready_pose.left_arm_rad, atol=1e-12
        )
    finally:
        teardown(controller)


def config_torso_origin() -> tuple[float, float, float]:
    """The fake torso sits at base (0, 0, 0.90) with identity rotation."""

    return (0.0, 0.0, 0.90)


def test_the_arms_are_commanded_exactly_twice_and_only_by_one_shot(
    root_config,
) -> None:
    """Ready move, place posture, retreat posture: three arm commands, in order."""

    (controller, robot, config, _plan, _delta, lowering,
     _lowering_packet, release, _release_packet) = drive_to_release(root_config)
    try:
        # The ready move goes through the stream; the two postures do not.
        assert len(robot.one_shot_packets) == 2, (
            f"expected place and retreat, saw {len(robot.one_shot_packets)}"
        )
        place_packet, retreat_packet = robot.one_shot_packets
        assert place_packet.arm_mode == "CARTESIAN"
        assert retreat_packet.arm_mode == "CARTESIAN"
        # Neither one-shot may command the base.
        assert place_packet.mobility_velocity is None
        assert retreat_packet.mobility_velocity is None
        # Torso and head are re-commanded at the ready posture, so the camera
        # cannot drift while placement is judged from depth.
        for packet in (place_packet, retreat_packet):
            np.testing.assert_allclose(
                packet.torso_position, config.ready_pose.torso_rad, atol=1e-12
            )
            np.testing.assert_allclose(
                packet.head_position, config.ready_pose.head_rad, atol=1e-12
            )
        # The two postures differ from each other.
        assert not np.allclose(
            place_packet.right_arm.transform, retreat_packet.right_arm.transform
        )
        assert lowering.mode is ArmStreamMode.CARTESIAN_PLACEMENT_LOWERING
        assert release.mode is ArmStreamMode.CARTESIAN_PLACEMENT_RELEASE
    finally:
        teardown(controller)


def test_base_velocity_reaches_the_stream_without_disturbing_the_arms(
    root_config,
) -> None:
    controller, robot, config = build(root_config)
    try:
        controller.reverify_wheel_stop_after_stream_start(timeout_s=2.0)
        before = len(robot.packets)

        # Grip evidence is required before any nonzero mobility, so assert the
        # gate first and then bypass it the way the runtime does.
        from parcel_pose_placing.pallet_control import CombinedStreamError, MobilityCommand

        with pytest.raises(CombinedStreamError, match="grip/clearance evidence"):
            controller.send_cycle(MobilityCommand(0.02, 0.0, 0.0))

        controller._grip_result = _passing_grip_result(controller)
        controller.send_cycle(
            MobilityCommand(0.02, 0.0, 0.0, source_timestamp_s=controller._clock())
        )
        _wait_for_packets(robot, before + 3)

        moving = [p for p in robot.packets[before:] if p.mobility_velocity != (0.0, 0.0, 0.0)]
        assert moving, "a nonzero proposal never reached the stream"
        assert moving[-1].mobility_velocity == pytest.approx((0.02, 0.0, 0.0))
        # There is no arm command to disturb: the stream carries mobility only.
        assert all(packet.right_arm is None for packet in robot.packets[before:])
        assert moving[-1].minimum_time_s == pytest.approx(config.steady_minimum_time_s)
    finally:
        teardown(controller)


def _passing_grip_result(controller: RBY1PalletController):
    from parcel_pose_placing.pallet_control import GripContinuityResult

    return GripContinuityResult(
        passed=True,
        reasons=(),
        evaluated_monotonic_s=controller._clock(),
        state_sample_count=12,
        scene_sample_count=6,
        dwell_s=0.55,
        arm_tracking_error_max_rad=None,
        eef_separation_peak_to_peak_m=0.0,
        eef_separation_axis_std_max_m=0.0,
        held_top_std_m=0.0,
        held_top_downward_drift_m=0.0,
        clearance_lower_bound_m=0.08,
        fixed_ready_geometry_only_authorized=True,
    )


def _wait_for_packets(robot: FakeRobot, target: int, timeout_s: float = 2.0) -> None:
    import time

    deadline = time.monotonic() + timeout_s
    while len(robot.packets) < target and time.monotonic() < deadline:
        time.sleep(0.01)


# --------------------------------------------------------------------------- #
# stage 4: placement — no descent, then the opening
# --------------------------------------------------------------------------- #
def placement_plan(root_config: dict, *, gap_m: float = 0.159):
    """A frozen plan matching the measured slot-1 geometry."""

    placement = PlacementConfig.from_root_config(root_config)
    delta = min(gap_m * placement.descent_fraction, placement.maximum_planned_descent_m)
    sigma = 0.004
    box_bottom = 0.6056
    stack_top = box_bottom - gap_m
    return descent_plan(
        planned_delta_z_m=delta,
        right_xyz=RIGHT_EEF,
        left_xyz=LEFT_EEF,
        gap_m=gap_m,
        gap_uncertainty_m=2.0 * sigma,
        min_delta_z_m=gap_m - 2.0 * sigma,
        max_delta_z_m=gap_m + 2.0 * sigma,
        box_bottom_z_lower_bound_m=box_bottom - sigma,
        stack_top_z_upper_bound_m=stack_top + sigma,
        stack_plane_z_base_m=stack_top,
        stack_plane_uncertainty_m=sigma,
        plan_id="sim-slot1-plan",
        target_source=(
            "demonstrated_place_pose"
            if PalletControlConfig.from_root_config(root_config).place_pose is not None
            else "base_z_descent"
        ),
    ), delta


def drive_to_release(root_config: dict):
    controller, robot, config = build(root_config)
    controller.reverify_wheel_stop_after_stream_start(timeout_s=2.0)
    controller.send_zero_mobility_hold(latch=True)
    plan, delta = placement_plan(root_config)
    lowering = controller.start_cartesian_lowering_hold(descent_plan=plan)
    lowering_packet = robot.one_shot_packets[-1]
    release = controller.start_cartesian_release_hold()
    release_packet = robot.one_shot_packets[-1]
    return controller, robot, config, plan, delta, lowering, lowering_packet, release, release_packet


def test_lowering_moves_to_the_place_pose_not_by_a_base_z_delta(root_config) -> None:
    """The descent plan asks for no base-Z delta; the posture owns the motion."""

    (controller, robot, config, plan, delta, lowering,
     lowering_packet, _release, _release_packet) = drive_to_release(root_config)
    try:
        assert delta == 0.0, "no base-Z descent is planned"
        assert plan.target_source == "demonstrated_place_pose"
        np.testing.assert_allclose(
            plan.right_target_base[:3, 3], plan.right_eef_base[:3, 3], atol=1e-12
        )
        # The commanded arm target is the posture, which does lower the wrists.
        drop = float(
            plan.right_eef_base[2, 3] - lowering.right_T_base_eef[2, 3]
        )
        assert drop > 0.0, "the demonstrated posture must lower the carton"
        # lowering_distance_m is the full 3D travel, so it can only exceed the
        # vertical component of the move.
        assert lowering.lowering_distance_m >= abs(drop) - 1e-12
        assert lowering_packet.mobility_velocity is None, (
            "a one-shot arm posture must never command the base"
        )
    finally:
        teardown(controller)


def test_timeline(root_config, capsys) -> None:
    """Print the streamed packets and the one-shot arm commands side by side."""

    (controller, robot, config, _plan, delta, lowering,
     lowering_packet, release, release_packet) = drive_to_release(root_config)
    try:
        def describe(packet) -> tuple[str, ...]:
            sep = packet.eef_separation_m()
            return (
                packet.arm_mode,
                f"{packet.minimum_time_s:.2f}" if packet.minimum_time_s else "-",
                "-" if sep is None else f"{1000 * sep:7.1f}",
                "-" if packet.right_arm is None
                else f"{packet.right_arm.transform[1, 3]:+.4f}",
                "-" if packet.right_arm is None
                else f"{packet.right_arm.transform[2, 3]:+.4f}",
                str(packet.mobility_velocity),
            )

        with capsys.disabled():
            print("\n=== 슬롯-1 명령 타임라인 (가짜 SDK) ===")
            print(f"측정 손 간격 {1000 * MEASURED_SEPARATION_M:.1f} mm, "
                  f"하강 {1000 * delta:.0f} mm, "
                  f"place {config.placement_place_pose_duration_s:.1f} s, "
                  f"retreat {config.placement_retreat_pose_duration_s:.1f} s")
            print(f"{'#':>4} {'arm':<10}{'min_t':>6}{'간격mm':>9}"
                  f"{'R_y':>9}{'R_z':>9}  mobility")
            print("-- 스트림 (mobility 전용, 첫 패킷만 ready Position) --")
            rows = [describe(p) for p in robot.packets]
            shown = rows[:2] + [None] + rows[-2:] if len(rows) > 5 else rows
            for index, row in enumerate(shown):
                if row is None:
                    print(f"{'...':>4}")
                    continue
                print(f"{index:>4} {row[0]:<10}{row[1]:>6}{row[2]:>9}"
                      f"{row[3]:>9}{row[4]:>9}  {row[5]}")
            print("-- 원샷 (팔 자세) --")
            for index, packet in enumerate(robot.one_shot_packets):
                row = describe(packet)
                label = ("place", "retreat")[index] if index < 2 else f"#{index}"
                print(f"{label:>4} {row[0]:<10}{row[1]:>6}{row[2]:>9}"
                      f"{row[3]:>9}{row[4]:>9}  {row[5]}")
            print(f"스트림 패킷 {len(robot.packets)}, 원샷 "
                  f"{len(robot.one_shot_packets)}, phase={controller.phase.value}")
        assert len(robot.packets) >= 3
        assert len(robot.one_shot_packets) == 2
        assert controller.phase is ControllerPhase.STEADY_HOLD
    finally:
        teardown(controller)


# --------------------------------------------------------------------------- #
# perception hot path: the ray grid must survive a per-frame transform change
# --------------------------------------------------------------------------- #
def test_ray_grid_is_cached_across_measured_transform_changes() -> None:
    """The live path feeds a new measured T_base_depth every frame.

    Keying the finished coefficients on that transform missed the cache on every
    frame and rebuilt a full-resolution array, which cost about 7.6 ms per frame
    on the development host.  Only the pixel-ray grid may be cached.
    """

    from parcel_pose_common.models import CameraIntrinsics
    from parcel_pose_placing.pallet_geometry import PalletStackEstimator

    estimator = PalletStackEstimator()
    intrinsics = CameraIntrinsics(
        width=640, height=480, fx=380.0, fy=380.0, cx=320.0, cy=240.0
    )

    def transform(x: float) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = (x, 0.0, 1.2)
        return matrix

    first = estimator._ray_coefficients(intrinsics, transform(0.10), 2)
    grid = estimator._ray_x
    second = estimator._ray_coefficients(intrinsics, transform(0.11), 2)

    assert estimator._ray_x is grid, "the ray grid was rebuilt for a new transform"
    assert first.shape == (240, 320, 3), "the grid must be built at stride resolution"
    # A different rotation must still change the coefficients.
    rotated = transform(0.10)
    rotated[:3, :3] = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64
    )
    third = estimator._ray_coefficients(intrinsics, rotated, 2)
    assert not np.allclose(third, second)


def test_strided_ray_grid_matches_a_full_resolution_grid_bit_for_bit() -> None:
    """Striding after the combine and combining after the stride must agree."""

    from parcel_pose_common.models import CameraIntrinsics
    from parcel_pose_placing.pallet_geometry import PalletStackEstimator

    intrinsics = CameraIntrinsics(
        width=640, height=480, fx=381.5, fy=381.5, cx=319.5, cy=239.5
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(
        ((0.86, -0.30, -0.43), (-0.03, 0.85, 0.51), (0.52, -0.42, 0.73)),
        dtype=np.float64,
    )
    rows, cols = np.indices((480, 640), dtype=np.float32)
    ray_x = (cols - np.float32(intrinsics.cx)) / np.float32(intrinsics.fx)
    ray_y = (rows - np.float32(intrinsics.cy)) / np.float32(intrinsics.fy)
    rotation = np.asarray(matrix[:3, :3], dtype=np.float32)
    reference = (
        ray_x[..., None] * rotation[:, 0]
        + ray_y[..., None] * rotation[:, 1]
        + rotation[:, 2]
    )[::2, ::2]

    actual = PalletStackEstimator()._ray_coefficients(intrinsics, matrix, 2)
    np.testing.assert_array_equal(actual, reference)


# --------------------------------------------------------------------------- #
# demonstrated placement posture
# --------------------------------------------------------------------------- #
def test_place_pose_is_commanded_as_cartesian_targets(root_config) -> None:
    """The arms are bound to Cartesian impedance, so a joint posture must be FK'd."""

    from parcel_pose_placing.pallet_control import PalletControlConfig

    config = PalletControlConfig.from_root_config(root_config)
    assert config.place_pose is not None, "the shipped config demonstrates a posture"
    # The demonstrated torso equals the ready torso, so only the arms move.
    np.testing.assert_allclose(
        config.place_pose.torso_rad, config.ready_pose.torso_rad, atol=1e-3
    )
    assert config.place_pose.right_arm_rad != config.ready_pose.right_arm_rad

    (controller, robot, config, _plan, _delta, lowering,
     lowering_packet, _release, _release_packet) = drive_to_release(root_config)
    try:
        assert lowering.target_source == "demonstrated_place_pose"
        assert lowering.mode is ArmStreamMode.CARTESIAN_PLACEMENT_LOWERING
        # The commanded nullspace posture is the demonstrated arm joints.
        np.testing.assert_allclose(
            lowering.right_nullspace_joint_rad, config.place_pose.right_arm_rad
        )
        # The posture is issued as a Cartesian pose, not the joint values, and
        # its minimum_time is the whole posture duration -- something a streamed
        # packet could not ask for.
        assert lowering_packet.arm_mode == "CARTESIAN"
        assert lowering_packet.right_arm.reference_link == "link_torso_5"
        assert lowering_packet.right_arm.link == "ee_right"
        assert lowering_packet.minimum_time_s == pytest.approx(
            config.placement_place_pose_duration_s
        )
        np.testing.assert_allclose(
            lowering_packet.right_arm.transform, lowering.right_T_torso_eef, atol=1e-12
        )
    finally:
        teardown(controller)


def test_place_pose_travel_is_paced_over_the_configured_duration(root_config) -> None:
    (controller, robot, config, _plan, _delta, lowering,
     _lowering_packet, _release, _release_packet) = drive_to_release(root_config)
    try:
        duration = config.placement_place_pose_duration_s
        assert duration == pytest.approx(1.0)
        expected = min(
            config.placement_linear_velocity_limit_mps,
            max(lowering.lowering_distance_m / duration, 1e-4),
        )
        assert lowering.linear_velocity_limit_mps == pytest.approx(expected)
        assert (
            lowering.linear_velocity_limit_mps
            <= config.placement_linear_velocity_limit_mps
        ), "the reviewed placement ceiling still bounds the pacing"
    finally:
        teardown(controller)


def test_place_pose_plan_rejects_a_base_z_descent(root_config) -> None:
    from parcel_pose_placing.pallet_place import PlacementDescentPlan

    plan, _delta = placement_plan(root_config)
    fields = {
        name: getattr(plan, name)
        for name in plan.__dataclass_fields__
        if name != "target_source"
    }
    fields["target_source"] = "demonstrated_place_pose"
    fields["planned_delta_z_m"] = 0.020
    with pytest.raises(ValueError, match="cannot also request a base-Z descent"):
        PlacementDescentPlan(**fields)


# --------------------------------------------------------------------------- #
# demonstrated retreat posture after seating
# --------------------------------------------------------------------------- #
def test_retreat_posture_is_a_separate_move_after_the_carton_is_seated(
    root_config,
) -> None:
    """Seat first, then withdraw: two distinct Cartesian arm targets, in order."""

    (controller, robot, config, _plan, _delta, lowering,
     lowering_packet, release, release_packet) = drive_to_release(root_config)
    try:
        assert config.retreat_pose is not None
        assert release.mode is ArmStreamMode.CARTESIAN_PLACEMENT_RELEASE
        assert release.target_source == "demonstrated_place_pose"
        # The retreat commands the demonstrated retreat joints as its nullspace.
        np.testing.assert_allclose(
            release.right_nullspace_joint_rad, config.retreat_pose.right_arm_rad
        )
        # It is a different pose from the seated one.
        assert not np.allclose(
            release.right_T_base_eef[:3, 3], lowering.right_T_base_eef[:3, 3]
        )
        # Ordering: the seated posture is commanded before the retreat.
        order = [id(packet) for packet in robot.one_shot_packets]
        assert order.index(id(lowering_packet)) < order.index(id(release_packet))
    finally:
        teardown(controller)


def test_retreat_is_paced_over_its_configured_duration(root_config) -> None:
    (controller, robot, config, _plan, _delta, _lowering,
     _lowering_packet, release, _release_packet) = drive_to_release(root_config)
    try:
        duration = config.placement_retreat_pose_duration_s
        assert duration == pytest.approx(1.0)
        expected = min(
            config.placement_linear_velocity_limit_mps,
            max(release.lowering_distance_m / duration, 1e-4),
        )
        assert release.linear_velocity_limit_mps == pytest.approx(expected)
    finally:
        teardown(controller)


def test_a_retreat_posture_requires_a_place_posture() -> None:
    from parcel_pose_placing.pallet_control import PalletControlConfig, PlacePose

    pose = PlacePose(
        torso_rad=(0.0,) * 6, right_arm_rad=(0.0,) * 7, left_arm_rad=(0.0,) * 7
    )
    with pytest.raises(ValueError, match="retreat posture requires a place posture"):
        PalletControlConfig(retreat_pose=pose)


# --------------------------------------------------------------------------- #
# one-shot arm commands: failures must name the untested assumption
# --------------------------------------------------------------------------- #
def _drive_to_lowering(root_config: dict):
    controller, robot, config = build(root_config)
    controller.reverify_wheel_stop_after_stream_start(timeout_s=2.0)
    controller.send_zero_mobility_hold(latch=True)
    plan, _delta = placement_plan(root_config)
    return controller, robot, config, plan


def test_a_refused_one_shot_names_the_open_mobility_stream(root_config) -> None:
    """Whether a one-shot coexists with an open stream is untested on hardware.

    If the RB-Y1 refuses it, the operator must be told that is the suspicion
    rather than being left with an unexplained hold.
    """

    controller, robot, _config, plan = _drive_to_lowering(root_config)
    try:
        robot.one_shot_raises = True
        with pytest.raises(CombinedStreamError, match="mobility stream was open"):
            controller.start_cartesian_lowering_hold(descent_plan=plan)
        assert controller.placement_telemetry().last_reason == (
            "arm_send_once_rejected_while_stream_open"
        )
    finally:
        teardown(controller)


def test_a_non_ok_finish_code_fails_closed(root_config) -> None:
    controller, robot, _config, plan = _drive_to_lowering(root_config)
    try:
        robot.one_shot_finish_code = 2  # Rejected
        with pytest.raises(CombinedStreamError, match="instead of Ok"):
            controller.start_cartesian_lowering_hold(descent_plan=plan)
        assert controller.placement_telemetry().last_reason == (
            "arm_send_once_not_ok"
        )
    finally:
        teardown(controller)


def test_a_one_shot_that_never_reports_fails_closed(root_config) -> None:
    controller, robot, _config, plan = _drive_to_lowering(root_config)
    try:
        robot.one_shot_completes = False
        with pytest.raises(CombinedStreamError, match="no feedback within"):
            controller.start_cartesian_lowering_hold(descent_plan=plan)
        assert controller.placement_telemetry().last_reason == (
            "arm_send_once_timeout"
        )
    finally:
        teardown(controller)


def test_the_one_shot_timeout_covers_the_longest_posture(root_config) -> None:
    from parcel_pose_placing.pallet_control import PalletControlConfig

    config = PalletControlConfig.from_root_config(root_config)
    assert config.arm_send_once_timeout_s >= max(
        config.placement_place_pose_duration_s,
        config.placement_retreat_pose_duration_s,
    )
    with pytest.raises(ValueError, match="must cover the longest commanded posture"):
        PalletControlConfig(
            placement_place_pose_duration_s=2.0, arm_send_once_timeout_s=1.0
        )
