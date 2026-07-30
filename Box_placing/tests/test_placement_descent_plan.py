"""Invariants of the frozen slot-1 descent plan."""

from __future__ import annotations

import numpy as np
import pytest

from _factories import GAP_M, MIN_DELTA_M, descent_plan


def test_targets_must_match_the_planned_delta() -> None:
    plan = descent_plan()
    assert plan.planned_delta_z_m == pytest.approx(GAP_M * (2.0 / 3.0))
    assert plan.right_target_base[2, 3] == pytest.approx(
        plan.right_eef_base[2, 3] - plan.planned_delta_z_m
    )
    assert plan.left_target_base[2, 3] == pytest.approx(
        plan.left_eef_base[2, 3] - plan.planned_delta_z_m
    )


def test_target_transform_mismatch_is_rejected() -> None:
    plan = descent_plan()
    tampered = np.array(plan.right_target_base, copy=True)
    tampered[2, 3] -= 0.010
    with pytest.raises(ValueError, match="must match planned_delta_z_m"):
        descent_plan(right_target_base=tampered)


def test_zero_descent_plans_are_admissible() -> None:
    plan = descent_plan(planned_delta_z_m=0.0)
    assert plan.planned_delta_z_m == 0.0
    assert plan.right_target_base[2, 3] == pytest.approx(plan.right_eef_base[2, 3])
    assert plan.left_target_base[2, 3] == pytest.approx(plan.left_eef_base[2, 3])


def test_negative_descent_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        descent_plan(planned_delta_z_m=-0.001)


def test_planned_delta_cannot_exceed_the_measured_gap() -> None:
    with pytest.raises(ValueError, match="cannot exceed gap_m"):
        descent_plan(planned_delta_z_m=GAP_M + 0.010)


def test_clearance_bounds_stay_positive() -> None:
    plan = descent_plan()
    assert plan.min_delta_z_m == pytest.approx(MIN_DELTA_M)
    assert plan.min_delta_z_m > 0.0
    assert plan.max_delta_z_m > plan.min_delta_z_m
    assert (
        plan.box_bottom_z_lower_bound_m - plan.stack_top_z_upper_bound_m
    ) == pytest.approx(plan.min_delta_z_m)


def test_invalid_plans_cannot_be_constructed() -> None:
    with pytest.raises(ValueError, match="must be valid"):
        descent_plan(valid=False)
    with pytest.raises(ValueError, match="cannot carry a rejection reason"):
        descent_plan(rejection_reason="anything")


def test_sequences_and_identifiers_are_required() -> None:
    with pytest.raises(ValueError, match="plan_id must not be empty"):
        descent_plan(plan_id="   ")
    with pytest.raises(ValueError, match="stack_plane_sequence must be positive"):
        descent_plan(stack_plane_sequence=0)
