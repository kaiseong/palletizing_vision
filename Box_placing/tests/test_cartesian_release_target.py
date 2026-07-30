"""Cartesian arm-target geometry for the loaded hold, descent, and release."""

from __future__ import annotations


import numpy as np
import pytest

from parcel_pose_placing.pallet_control import (
    ArmStreamMode,
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


def test_loaded_hold_holds_the_measured_wrists_unchanged() -> None:
    """The commissioned slot-1 hold commands no inward offset at all."""

    controller = offline_controller()
    assert controller.config.placement_squeeze_offset_m == 0.0
    target = loaded_hold_target(controller, measured_state())
    assert target.right_T_base_eef[1, 3] == pytest.approx(RIGHT_EEF_XYZ[1])
    assert target.left_T_base_eef[1, 3] == pytest.approx(LEFT_EEF_XYZ[1])
    assert separation(target) == pytest.approx(EEF_SEPARATION_M)
    assert target.squeeze_offset_m == 0.0


def test_loaded_hold_still_supports_a_configured_squeeze() -> None:
    """Restoring the box-pick style squeeze stays a configuration change."""

    controller = offline_controller()
    squeeze = 0.150
    target = loaded_hold_target(controller, measured_state(), squeeze_offset_m=squeeze)
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
