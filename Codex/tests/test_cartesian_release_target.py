"""Cartesian arm-target geometry for the loaded hold, descent, and release."""

from __future__ import annotations

import math

import numpy as np
import pytest

from parcel_pose.pallet_control import (
    ArmStreamMode,
    CombinedStreamError,
    MeasuredStateError,
)

from _factories import (
    EEF_SEPARATION_M,
    LEFT_EEF_XYZ,
    RIGHT_EEF_XYZ,
    descent_plan,
    loaded_hold_target,
    measured_state,
    offline_controller,
)


def separation(target) -> float:
    return float(
        np.linalg.norm(target.right_T_base_eef[:3, 3] - target.left_T_base_eef[:3, 3])
    )


def release_from(controller, state, plan, *, spread=None):
    loaded = loaded_hold_target(controller, state)
    lowering = controller._make_lowering_target_from_loaded_hold(
        state,
        loaded_target=loaded,
        descent_plan=plan,
        requested_squeeze_offset_m=None,
    )
    return lowering, controller._make_release_target_from_plan(
        state,
        lowering_target=lowering,
        descent_plan=plan,
        release_spread_m=(
            controller.config.placement_release_spread_m if spread is None else spread
        ),
    )


def test_loaded_hold_axis_points_from_left_to_right_hand() -> None:
    controller = offline_controller()
    target = loaded_hold_target(controller, measured_state())
    # The nominal scene puts the right hand on base -Y.
    assert target.inter_eef_axis_base == pytest.approx((0.0, -1.0, 0.0))


def test_loaded_hold_commands_an_inward_squeeze_on_both_hands() -> None:
    controller = offline_controller()
    squeeze = controller.config.placement_squeeze_offset_m
    target = loaded_hold_target(controller, measured_state())
    # Each commanded hand target sits `squeeze` inside the measured hand.
    assert target.right_T_base_eef[1, 3] == pytest.approx(RIGHT_EEF_XYZ[1] + squeeze)
    assert target.left_T_base_eef[1, 3] == pytest.approx(LEFT_EEF_XYZ[1] - squeeze)
    assert separation(target) == pytest.approx(abs(EEF_SEPARATION_M - 2.0 * squeeze))


def test_lowering_target_only_changes_base_z() -> None:
    controller = offline_controller()
    state = measured_state()
    loaded = loaded_hold_target(controller, state)
    plan = descent_plan()
    lowering = controller._make_lowering_target_from_loaded_hold(
        state,
        loaded_target=loaded,
        descent_plan=plan,
        requested_squeeze_offset_m=None,
    )
    assert lowering.mode is ArmStreamMode.CARTESIAN_PLACEMENT_LOWERING
    assert lowering.descent_plan_id == plan.plan_id
    for axis in (0, 1):
        assert lowering.right_T_base_eef[axis, 3] == pytest.approx(
            loaded.right_T_base_eef[axis, 3]
        )
        assert lowering.left_T_base_eef[axis, 3] == pytest.approx(
            loaded.left_T_base_eef[axis, 3]
        )
    assert lowering.right_T_base_eef[2, 3] == pytest.approx(
        loaded.right_T_base_eef[2, 3] - plan.planned_delta_z_m
    )
    assert lowering.left_T_base_eef[2, 3] == pytest.approx(
        loaded.left_T_base_eef[2, 3] - plan.planned_delta_z_m
    )


def test_release_opens_along_base_y_only() -> None:
    controller = offline_controller()
    state = measured_state()
    plan = descent_plan(planned_delta_z_m=0.0)
    _, release = release_from(controller, state, plan)
    spread = controller.config.placement_release_spread_m

    assert release.mode is ArmStreamMode.CARTESIAN_PLACEMENT_RELEASE
    # Base X and Z are bit-identical to the frozen plan: no vertical command.
    for axis in (0, 2):
        assert release.right_T_base_eef[axis, 3] == plan.right_target_base[axis, 3]
        assert release.left_T_base_eef[axis, 3] == plan.left_target_base[axis, 3]
    # Right hand sits on base -Y here, so it opens further negative.
    assert release.right_T_base_eef[1, 3] == pytest.approx(RIGHT_EEF_XYZ[1] - spread)
    assert release.left_T_base_eef[1, 3] == pytest.approx(LEFT_EEF_XYZ[1] + spread)
    assert release.inter_eef_axis_base == pytest.approx((0.0, -1.0, 0.0))


def test_release_travel_is_exactly_one_spread_per_hand() -> None:
    controller = offline_controller()
    state = measured_state()
    plan = descent_plan(planned_delta_z_m=0.0)
    _, release = release_from(controller, state, plan)
    spread = controller.config.placement_release_spread_m
    assert release.squeeze_offset_m == 0.0
    assert release.release_spread_m == pytest.approx(spread)
    assert separation(release) == pytest.approx(EEF_SEPARATION_M + 2.0 * spread)


def test_release_keeps_a_reenabled_descent_in_the_target() -> None:
    controller = offline_controller()
    state = measured_state()
    plan = descent_plan(planned_delta_z_m=0.025)
    lowering, release = release_from(controller, state, plan)
    assert release.right_T_base_eef[2, 3] == pytest.approx(RIGHT_EEF_XYZ[2] - 0.025)
    assert release.lowering_distance_m == pytest.approx(lowering.lowering_distance_m)


def test_release_axis_deviation_within_the_limit_is_recorded() -> None:
    controller = offline_controller()
    # Rotate the hand pair about base Z; the opening axis must stay on base Y.
    right = (0.450 - 0.0113, -0.1295, 0.300)
    left = (0.450 + 0.0113, 0.1295, 0.300)
    state = measured_state(right_xyz=right, left_xyz=left)
    plan = descent_plan(planned_delta_z_m=0.0, right_xyz=right, left_xyz=left)
    _, release = release_from(controller, state, plan)
    assert release.inter_eef_axis_base == pytest.approx((0.0, -1.0, 0.0))
    assert 0.0 < release.release_axis_deviation_rad < math.radians(10.0)


def test_release_axis_beyond_the_limit_fails_closed() -> None:
    controller = offline_controller()
    # A 20 degree hand-pair rotation exceeds the 10 degree base-Y limit.
    right = (0.450 - 0.0445, -0.1222, 0.300)
    left = (0.450 + 0.0445, 0.1222, 0.300)
    state = measured_state(right_xyz=right, left_xyz=left)
    plan = descent_plan(planned_delta_z_m=0.0, right_xyz=right, left_xyz=left)
    controller._placement_started = True
    with pytest.raises(MeasuredStateError, match="deviates"):
        release_from(controller, state, plan)
    assert controller.placement_telemetry().last_reason == "release_axis_deviation"


def test_release_rejects_a_plan_from_another_placement() -> None:
    controller = offline_controller()
    state = measured_state()
    plan = descent_plan(planned_delta_z_m=0.0, plan_id="frozen-plan")
    loaded = loaded_hold_target(controller, state)
    lowering = controller._make_lowering_target_from_loaded_hold(
        state,
        loaded_target=loaded,
        descent_plan=plan,
        requested_squeeze_offset_m=None,
    )
    other = descent_plan(planned_delta_z_m=0.0, plan_id="different-plan")
    controller._placement_started = True
    with pytest.raises(CombinedStreamError, match="does not match"):
        controller._make_release_target_from_plan(
            state,
            lowering_target=lowering,
            descent_plan=other,
            release_spread_m=0.030,
        )
    assert controller.placement_telemetry().last_reason == "release_plan_id_mismatch"


def test_spread_above_the_configured_bound_is_rejected() -> None:
    controller = offline_controller()
    bound = controller.config.placement_max_release_spread_m
    assert bound == pytest.approx(0.040)
    with pytest.raises(ValueError, match="exceeds the configured release bound"):
        controller.start_cartesian_release_hold(release_spread_m=bound + 0.001)


def test_inter_eef_axis_below_the_floor_is_rejected() -> None:
    controller = offline_controller()
    state = measured_state(right_xyz=(0.45, -0.02, 0.30), left_xyz=(0.45, 0.02, 0.30))
    with pytest.raises(MeasuredStateError, match="inter-EEF axis is invalid"):
        loaded_hold_target(controller, state)


def test_descent_plan_state_mismatch_is_rejected() -> None:
    controller = offline_controller()
    state = measured_state()
    loaded = loaded_hold_target(controller, state)
    drifted = descent_plan(right_xyz=(0.45, -0.130, 0.50))
    with pytest.raises(MeasuredStateError, match="frozen descent plan provenance"):
        controller._make_lowering_target_from_loaded_hold(
            state,
            loaded_target=loaded,
            descent_plan=drifted,
            requested_squeeze_offset_m=None,
        )


def test_release_follows_the_torso_frame_not_the_base_frame() -> None:
    """A yawed torso opens along its own Y axis."""

    import dataclasses
    import math as _math

    controller = offline_controller()
    yaw = _math.radians(25.0)
    rotation = np.asarray(
        ((_math.cos(yaw), -_math.sin(yaw), 0.0),
         (_math.sin(yaw), _math.cos(yaw), 0.0),
         (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    torso = np.eye(4, dtype=np.float64)
    torso[:3, :3] = rotation
    torso[:3, 3] = (0.0, 0.0, 0.90)

    # Hands separated along the rotated torso Y so the grip axis matches.
    torso_y = rotation @ np.asarray((0.0, 1.0, 0.0))
    right = tuple(np.asarray((0.450, 0.0, 0.300)) - 0.130 * torso_y)
    left = tuple(np.asarray((0.450, 0.0, 0.300)) + 0.130 * torso_y)
    state = dataclasses.replace(
        measured_state(right_xyz=right, left_xyz=left), T_base_torso=torso
    )
    plan = descent_plan(planned_delta_z_m=0.0, right_xyz=right, left_xyz=left)
    _, release = release_from(controller, state, plan)
    spread = controller.config.placement_release_spread_m

    # Deviation is zero because the grip axis is exactly the torso Y axis.
    assert release.release_axis_deviation_rad == pytest.approx(0.0, abs=1e-9)
    assert release.inter_eef_axis_base == pytest.approx(tuple(-torso_y))
    # Each hand travels one spread along torso Y, which now has a base X part.
    delta_right = release.right_T_base_eef[:3, 3] - plan.right_target_base[:3, 3]
    assert delta_right == pytest.approx(-spread * torso_y)
    assert abs(delta_right[0]) > 1e-3  # would be exactly 0 for a base-Y opening
    # Vertical motion stays exactly zero.
    assert release.right_T_base_eef[2, 3] == plan.right_target_base[2, 3]
    assert release.left_T_base_eef[2, 3] == plan.left_target_base[2, 3]
