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

from parcel_pose.pallet_control import (
    ArmStreamMode,
    ControllerPhase,
    PalletControlConfig,
    RBY1PalletController,
)
from parcel_pose.pallet_place import PlacementConfig

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
def test_first_packet_binds_cartesian_arms_and_zero_mobility(root_config) -> None:
    controller, robot, config = build(root_config)
    try:
        first = robot.packets[0]
        assert first.arm_mode == "CARTESIAN", (
            "RB-Y1 binds controller types on the first packet; a joint-impedance "
            "first packet would accept later Cartesian targets without moving"
        )
        assert first.right_arm.reference_link == "link_torso_5"
        assert first.left_arm.reference_link == "link_torso_5"
        assert first.right_arm.link == "ee_right"
        assert first.left_arm.link == "ee_left"
        assert first.mobility_velocity == (0.0, 0.0, 0.0)
        assert first.minimum_time_s == pytest.approx(
            config.ready_transition_minimum_time_s
        )
        assert first.control_hold_time_s == pytest.approx(config.control_hold_time_s)
        np.testing.assert_allclose(first.torso_position, config.ready_pose.torso_rad)
        np.testing.assert_allclose(first.head_position, config.ready_pose.head_rad)
    finally:
        teardown(controller)


def test_ready_one_shot_minimum_time_is_three_seconds(root_config) -> None:
    config = PalletControlConfig.from_root_config(root_config)
    assert config.ready_transition_minimum_time_s == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# stage 2: the loaded hold squeeze
# --------------------------------------------------------------------------- #
def test_loaded_hold_commands_the_ready_posture_unchanged(root_config) -> None:
    """No inward offset: the first packet asks for the measured wrists."""

    controller, robot, config = build(root_config)
    try:
        assert config.placement_squeeze_offset_m == 0.0
        packet = robot.packets[0]
        assert packet.eef_separation_m() == pytest.approx(
            MEASURED_SEPARATION_M, abs=1e-6
        )
        np.testing.assert_allclose(
            packet.right_arm.transform[:3, 3],
            np.asarray(RIGHT_EEF) - np.asarray(config_torso_origin()),
            atol=1e-9,
        )
    finally:
        teardown(controller)


def config_torso_origin() -> tuple[float, float, float]:
    """The fake torso sits at base (0, 0, 0.90) with identity rotation."""

    return (0.0, 0.0, 0.90)


def test_arm_target_is_constant_from_ready_until_release(root_config) -> None:
    """Nothing moves the arms between the ready posture and the opening."""

    (controller, robot, config, _plan, _delta, _lowering,
     lowering_packet, release, release_packet) = drive_to_release(root_config)
    try:
        first = robot.packets[0]
        # Every packet up to and including the lowering hold repeats packet 0.
        release_index = next(
            i for i, p in enumerate(robot.packets)
            if p.eef_separation_m() is not None
            and abs(p.eef_separation_m() - first.eef_separation_m()) > 1e-9
        )
        for index, packet in enumerate(robot.packets[:release_index]):
            np.testing.assert_allclose(
                packet.right_arm.transform, first.right_arm.transform, atol=1e-12,
                err_msg=f"packet {index} moved the right arm before release",
            )
            np.testing.assert_allclose(
                packet.left_arm.transform, first.left_arm.transform, atol=1e-12,
                err_msg=f"packet {index} moved the left arm before release",
            )
        # The only arm motion in the whole sequence is the opening.
        spread = config.placement_release_spread_m
        assert release_packet.eef_separation_m() == pytest.approx(
            MEASURED_SEPARATION_M + 2.0 * spread, abs=1e-6
        )
    finally:
        teardown(controller)


# --------------------------------------------------------------------------- #
# stage 3: base motion while the hold is streamed
# --------------------------------------------------------------------------- #
def test_base_velocity_reaches_the_stream_without_disturbing_the_arms(
    root_config,
) -> None:
    controller, robot, config = build(root_config)
    try:
        controller.reverify_wheel_stop_after_stream_start(timeout_s=2.0)
        before = len(robot.packets)
        arm_before = robot.packets[-1].right_arm.transform.copy()

        # Grip evidence is required before any nonzero mobility, so assert the
        # gate first and then bypass it the way the runtime does.
        from parcel_pose.pallet_control import CombinedStreamError, MobilityCommand

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
        # The arm target must not drift while the base drives.
        np.testing.assert_allclose(
            moving[-1].right_arm.transform, arm_before, atol=1e-12
        )
        assert moving[-1].minimum_time_s == pytest.approx(config.steady_minimum_time_s)
    finally:
        teardown(controller)


def _passing_grip_result(controller: RBY1PalletController):
    from parcel_pose.pallet_control import GripContinuityResult

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
    ), delta


def drive_to_release(root_config: dict):
    controller, robot, config = build(root_config)
    controller.reverify_wheel_stop_after_stream_start(timeout_s=2.0)
    controller.send_zero_mobility_hold(latch=True)
    plan, delta = placement_plan(root_config)
    lowering = controller.start_cartesian_lowering_hold(descent_plan=plan)
    _wait_for_packets(robot, len(robot.packets) + 2)
    lowering_packet = robot.packets[-1]
    release = controller.start_cartesian_release_hold()
    _wait_for_packets(robot, len(robot.packets) + 2)
    release_packet = robot.packets[-1]
    return controller, robot, config, plan, delta, lowering, lowering_packet, release, release_packet


def test_lowering_commands_no_vertical_motion(root_config) -> None:
    (controller, robot, config, plan, delta, lowering,
     lowering_packet, _release, _release_packet) = drive_to_release(root_config)
    try:
        assert delta == 0.0, "the commissioned descent cap is zero"
        assert lowering.mode is ArmStreamMode.CARTESIAN_PLACEMENT_LOWERING
        # The lowering packet must equal the loaded hold: same squeeze, same z.
        first = robot.packets[0]
        np.testing.assert_allclose(
            lowering_packet.right_arm.transform,
            first.right_arm.transform,
            atol=1e-12,
        )
        assert lowering_packet.mobility_velocity == (0.0, 0.0, 0.0)
    finally:
        teardown(controller)


def test_release_opens_along_torso_y_only(root_config) -> None:
    (controller, robot, config, plan, _delta, _lowering,
     lowering_packet, release, release_packet) = drive_to_release(root_config)
    try:
        spread = config.placement_release_spread_m
        assert release.mode is ArmStreamMode.CARTESIAN_PLACEMENT_RELEASE
        assert release.release_spread_m == pytest.approx(spread)
        assert release.release_axis_deviation_rad == pytest.approx(0.0, abs=1e-9)

        # Commanded targets are in the torso frame; compare against the plan.
        right_base = release.right_T_base_eef[:3, 3]
        left_base = release.left_T_base_eef[:3, 3]
        np.testing.assert_allclose(right_base[[0, 2]], np.asarray(RIGHT_EEF)[[0, 2]])
        np.testing.assert_allclose(left_base[[0, 2]], np.asarray(LEFT_EEF)[[0, 2]])
        assert right_base[1] == pytest.approx(RIGHT_EEF[1] - spread)
        assert left_base[1] == pytest.approx(LEFT_EEF[1] + spread)

        # The squeeze is gone, so the hands travel exactly one spread each.
        assert release.squeeze_offset_m == 0.0
        opened = release_packet.eef_separation_m()
        held = lowering_packet.eef_separation_m()
        assert opened > held
        assert opened == pytest.approx(MEASURED_SEPARATION_M + 2.0 * spread, abs=1e-6)
    finally:
        teardown(controller)


def test_gap_above_the_release_limit_is_refused(root_config) -> None:
    """The 190.6 mm gap recorded in place_10 must not produce a plan."""

    from parcel_pose.pallet_place import PlacementState, Slot1PlacementSequencer

    from _factories import placement_input

    placement = PlacementConfig.from_root_config(root_config)
    assert 0.1906 > placement.maximum_release_gap_m, "the shipped cap must reject it"

    sequencer = Slot1PlacementSequencer(placement)
    box_bottom = 0.6056
    scene = {
        "box_bottom_z_base_m": box_bottom,
        "stack_top_z_base_m": box_bottom - 0.1906,
    }
    for index in range(3):
        output = sequencer.update(
            placement_input(now_s=100.0 + index * 0.10, sequence=index + 1, **scene)
        )
    assert output.faulted
    assert output.reason == "descent_gap_above_release_limit"
    assert sequencer.state is PlacementState.FAULT_HOLD


def test_measured_gap_from_place_07_is_admissible(root_config) -> None:
    """159 mm, the gap the physical run reported, must pass and give zero descent."""

    from parcel_pose.pallet_place import PlacementState, Slot1PlacementSequencer

    from _factories import placement_input

    placement = PlacementConfig.from_root_config(root_config)
    sequencer = Slot1PlacementSequencer(placement)
    box_bottom = 0.6056
    scene = {
        "box_bottom_z_base_m": box_bottom,
        "stack_top_z_base_m": box_bottom - 0.159,
    }
    for index in range(3):
        output = sequencer.update(
            placement_input(now_s=100.0 + index * 0.10, sequence=index + 1, **scene)
        )
    assert not output.faulted, output.reason
    assert sequencer.state is PlacementState.LOWERING
    assert output.descent_plan is not None
    assert output.descent_plan.planned_delta_z_m == 0.0
    assert output.descent_plan.gap_m == pytest.approx(0.159)


# --------------------------------------------------------------------------- #
# stage 5: printable timeline
# --------------------------------------------------------------------------- #
def test_timeline(root_config, capsys) -> None:
    (controller, robot, config, plan, delta, lowering,
     lowering_packet, release, release_packet) = drive_to_release(root_config)
    try:
        rows = []
        for index, packet in enumerate(robot.packets):
            sep = packet.eef_separation_m()
            rows.append(
                (
                    index,
                    packet.arm_mode,
                    f"{packet.minimum_time_s:.2f}" if packet.minimum_time_s else "-",
                    "-" if sep is None else f"{1000 * sep:7.1f}",
                    "-" if packet.right_arm is None
                    else f"{packet.right_arm.transform[1, 3]:+.4f}",
                    "-" if packet.right_arm is None
                    else f"{packet.right_arm.transform[2, 3]:+.4f}",
                    str(packet.mobility_velocity),
                )
            )
        with capsys.disabled():
            print("\n=== 슬롯-1 명령 패킷 타임라인 (가짜 SDK) ===")
            print(f"측정 손 간격 {1000 * MEASURED_SEPARATION_M:.1f} mm, "
                  f"압착 {1000 * config.placement_squeeze_offset_m:.0f} mm/손, "
                  f"하강 {1000 * delta:.0f} mm, "
                  f"개방 {1000 * config.placement_release_spread_m:.0f} mm/손")
            print(f"{'#':>4} {'arm':<10}{'min_t':>6}{'간격mm':>9}"
                  f"{'R_y':>9}{'R_z':>9}  mobility")
            shown = rows[:3] + [(None,)] + rows[-4:] if len(rows) > 8 else rows
            for row in shown:
                if row[0] is None:
                    print(f"{'...':>4}")
                    continue
                print(f"{row[0]:>4} {row[1]:<10}{row[2]:>6}{row[3]:>9}"
                      f"{row[4]:>9}{row[5]:>9}  {row[6]}")
            print(f"총 패킷 {len(robot.packets)}, phase={controller.phase.value}")
        assert len(robot.packets) >= 3
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

    from parcel_pose.models import CameraIntrinsics
    from parcel_pose.pallet_geometry import PalletStackEstimator

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

    from parcel_pose.models import CameraIntrinsics
    from parcel_pose.pallet_geometry import PalletStackEstimator

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
