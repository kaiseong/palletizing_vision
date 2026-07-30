"""State-transition contract of the slot-1 placement sequencer."""

from __future__ import annotations

import pytest

from parcel_pose.pallet_place import (
    LOADED_HOLD_MODE,
    LOWERING_MODE,
    RELEASE_MODE,
    PlacementConfig,
    PlacementRequest,
    PlacementState,
    Slot1PlacementSequencer,
)

from _factories import (
    BOX_BOTTOM_Z_M,
    GAP_M,
    LEFT_EEF_XYZ,
    RIGHT_EEF_XYZ,
    placement_input,
)


def lowered(
    xyz: tuple[float, float, float],
    delta: float,
) -> tuple[float, float, float]:
    return (xyz[0], xyz[1], xyz[2] - delta)


def drive_to_lowering(
    sequencer: Slot1PlacementSequencer,
    *,
    start_s: float = 100.0,
    step_s: float = 0.10,
):
    """Feed the three contiguous stable vision samples the gate requires."""

    output = None
    for index in range(3):
        output = sequencer.update(
            placement_input(now_s=start_s + index * step_s, sequence=index + 1)
        )
    assert output is not None
    return output


def at_target_kwargs(plan) -> dict:
    delta = plan.planned_delta_z_m
    return {
        "right_xyz": lowered(RIGHT_EEF_XYZ, delta),
        "left_xyz": lowered(LEFT_EEF_XYZ, delta),
        "right_target_base": plan.right_target_base,
        "left_target_base": plan.left_target_base,
        "controller_arm_mode": LOWERING_MODE,
        "box_bottom_z_base_m": BOX_BOTTOM_Z_M - delta,
    }


def test_pre_place_requires_contiguous_vision_samples() -> None:
    sequencer = Slot1PlacementSequencer()
    first = sequencer.update(placement_input(now_s=100.0, sequence=1))
    assert first.state is PlacementState.PRE_PLACE_VERIFY
    assert first.reason == "pre_place_verify_dwell"

    second = sequencer.update(placement_input(now_s=100.10, sequence=2))
    assert second.reason == "insufficient_contiguous_vision_gap_samples"
    assert second.request is PlacementRequest.HOLD_CURRENT


def test_missing_start_gate_holds_without_faulting() -> None:
    sequencer = Slot1PlacementSequencer()
    output = sequencer.update(placement_input(arrived_hold=False))
    assert output.state is PlacementState.PRE_PLACE_VERIFY
    assert output.reason == "arrival_state_not_arrived_hold"
    assert not output.faulted

    output = sequencer.update(placement_input(controller_arm_mode="SOMETHING_ELSE"))
    assert output.reason == "loaded_cartesian_hold_mode_missing"
    assert not output.faulted


def test_lowering_freezes_a_zero_descent_plan() -> None:
    """The commissioned plan holds the aligned pose instead of descending."""

    sequencer = Slot1PlacementSequencer()
    output = drive_to_lowering(sequencer)
    assert sequencer.state is PlacementState.LOWERING
    assert output.request is PlacementRequest.LOWER_CARTESIAN_PLANNED
    plan = output.descent_plan
    assert plan is not None
    assert plan.planned_delta_z_m == 0.0
    assert plan.gap_m == pytest.approx(GAP_M)
    # No vertical command at all: the frozen targets equal the measured hands.
    assert plan.right_target_base[2, 3] == pytest.approx(plan.right_eef_base[2, 3])
    assert plan.left_target_base[2, 3] == pytest.approx(plan.left_eef_base[2, 3])


def test_planned_descent_returns_when_the_cap_is_raised() -> None:
    """Re-enabling the descent is a configuration change, not a code change."""

    config = PlacementConfig(maximum_planned_descent_m=0.025)
    sequencer = Slot1PlacementSequencer(config)
    plan = drive_to_lowering(sequencer).descent_plan
    assert plan is not None
    assert plan.planned_delta_z_m == pytest.approx(0.025)
    assert plan.right_target_base[2, 3] == pytest.approx(
        plan.right_eef_base[2, 3] - 0.025
    )


def test_descent_cap_never_exceeds_the_fraction_of_a_small_gap() -> None:
    config = PlacementConfig(maximum_planned_descent_m=0.250)
    sequencer = Slot1PlacementSequencer(config)
    plan = drive_to_lowering(sequencer).descent_plan
    assert plan is not None
    assert plan.planned_delta_z_m == pytest.approx(GAP_M * config.descent_fraction)
    assert plan.planned_delta_z_m <= plan.min_delta_z_m


def test_gap_above_the_release_limit_faults() -> None:
    config = PlacementConfig(maximum_release_gap_m=0.070)
    sequencer = Slot1PlacementSequencer(config)
    output = drive_to_lowering(sequencer)
    assert output.faulted
    assert output.reason.startswith("descent_gap_above_release_limit")
    # The reason must carry the measurement so the operator can act on it.
    assert "mm >" in output.reason, output.reason
    assert sequencer.state is PlacementState.FAULT_HOLD
    assert output.descent_plan is None


def test_clearance_below_floor_never_freezes_a_plan() -> None:
    sequencer = Slot1PlacementSequencer()
    tight = {"box_bottom_z_base_m": 0.140}  # gap 30 mm, below the 50 mm floor
    for index in range(3):
        output = sequencer.update(
            placement_input(now_s=100.0 + index * 0.10, sequence=index + 1, **tight)
        )
    # The start gate withholds authority instead of faulting, so the run can
    # recover if the operator repositions the pallet.
    assert not output.faulted
    assert output.reason == "seating_evidence_unavailable"
    assert sequencer.state is PlacementState.PRE_PLACE_VERIFY
    assert output.descent_plan is None


def test_full_sequence_reaches_release_authorization() -> None:
    sequencer = Slot1PlacementSequencer()
    plan = drive_to_lowering(sequencer).descent_plan
    assert plan is not None
    at_target = at_target_kwargs(plan)

    seated = sequencer.update(placement_input(now_s=100.30, sequence=4, **at_target))
    assert sequencer.state is PlacementState.SEATED
    assert seated.reason == "seating_evidence_started"

    dwelling = sequencer.update(placement_input(now_s=100.50, sequence=5, **at_target))
    assert dwelling.reason == "seating_evidence_dwell"
    assert not dwelling.release_authorized

    released = sequencer.update(placement_input(now_s=100.70, sequence=6, **at_target))
    assert sequencer.state is PlacementState.RELEASING
    assert released.request is PlacementRequest.SPREAD_RELEASE
    assert released.release_authorized


def test_lost_lowering_ack_faults_before_release() -> None:
    sequencer = Slot1PlacementSequencer()
    plan = drive_to_lowering(sequencer).descent_plan
    assert plan is not None
    at_target = at_target_kwargs(plan)
    sequencer.update(placement_input(now_s=100.30, sequence=4, **at_target))
    lost = sequencer.update(
        placement_input(
            now_s=100.40,
            sequence=5,
            **{**at_target, "controller_target_ack": False},
        )
    )
    assert lost.faulted
    assert lost.reason == "lowering_hold_ack_lost_before_release"


def test_every_output_commands_exact_zero_mobility() -> None:
    sequencer = Slot1PlacementSequencer()
    for index in range(3):
        output = sequencer.update(
            placement_input(now_s=100.0 + index * 0.10, sequence=index + 1)
        )
        assert output.mobility_command == (0.0, 0.0, 0.0)


def test_nonmonotonic_controller_time_faults() -> None:
    sequencer = Slot1PlacementSequencer()
    sequencer.update(placement_input(now_s=100.0, sequence=1))
    output = sequencer.update(placement_input(now_s=99.0, sequence=2))
    assert output.faulted
    assert output.reason == "nonmonotonic_controller_time"


def test_stale_measured_state_faults_immediately() -> None:
    sequencer = Slot1PlacementSequencer()
    output = sequencer.update(placement_input(measured_state_fresh=False))
    assert output.faulted
    assert output.reason == "measured_state_stale"


def test_release_mode_strings_match_the_controller_enum() -> None:
    from parcel_pose.pallet_control import ArmStreamMode

    assert LOADED_HOLD_MODE == ArmStreamMode.CARTESIAN_LOADED_HOLD.value
    assert LOWERING_MODE == ArmStreamMode.CARTESIAN_PLACEMENT_LOWERING.value
    assert RELEASE_MODE == ArmStreamMode.CARTESIAN_PLACEMENT_RELEASE.value


# --------------------------------------------------------------------------- #
# the release ceiling applies after the demonstrated posture has lowered the box
# --------------------------------------------------------------------------- #
def _drive(sequencer, **extra):
    output = None
    for index in range(3):
        output = sequencer.update(
            placement_input(now_s=100.0 + index * 0.10, sequence=index + 1, **extra)
        )
    assert output is not None
    return output


def test_place_pose_drop_is_subtracted_before_the_release_ceiling() -> None:
    """A gap the posture will close must not be rejected as if it stayed open."""

    # A ceiling just under the fixture gap: refused without a posture.
    config = PlacementConfig(maximum_release_gap_m=GAP_M - 0.010)
    refused = _drive(Slot1PlacementSequencer(config))
    assert refused.faulted
    assert refused.reason.startswith("descent_gap_above_release_limit"), refused.reason

    # The same gap, with a posture that lowers the carton past the excess.
    accepted = _drive(
        Slot1PlacementSequencer(config),
        demonstrated_place_pose=True,
        place_pose_vertical_drop_m=0.020,
    )
    assert not accepted.faulted, accepted.reason
    assert accepted.reason == "lowering_started", accepted.reason
    assert accepted.descent_plan is not None
    assert accepted.descent_plan.target_source == "demonstrated_place_pose"
    assert accepted.descent_plan.planned_delta_z_m == 0.0


def test_the_reason_shows_the_gap_the_posture_leaves_behind() -> None:
    config = PlacementConfig(maximum_release_gap_m=GAP_M - 0.030)
    output = _drive(
        Slot1PlacementSequencer(config),
        demonstrated_place_pose=True,
        place_pose_vertical_drop_m=0.005,
    )
    assert output.faulted
    assert "place pose 5mm" in output.reason, output.reason
    assert f"{(GAP_M - 0.005) * 1000.0:.0f}mm >" in output.reason, output.reason


def test_a_missing_place_pose_drop_is_missing_evidence_not_a_zero_drop() -> None:
    output = _drive(
        Slot1PlacementSequencer(PlacementConfig()),
        demonstrated_place_pose=True,
        place_pose_vertical_drop_m=None,
    )
    assert output.faulted
    assert output.reason == "descent_place_pose_drop_unavailable", output.reason
