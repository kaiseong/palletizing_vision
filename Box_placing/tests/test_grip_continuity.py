"""Loaded-hold grip and vertical-clearance interlock.

This gate decides whether the base may move at all while a carton is held, so
it is the most safety-relevant pure function in the controller.  It had no test
coverage before it was extracted from ``RBY1PalletController``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from parcel_pose_placing.pallet_control import (
    ArmStreamMode,
    CommandOwnershipError,
    PalletControlConfig,
    evaluate_grip_continuity,
)

from _factories import (
    JOINT_COUNT,
    measured_state,
    offline_controller,
)


CONFIG = PalletControlConfig(fixed_ready_geometry_only_commissioning_enabled=True)
NOW_S = 100.0


@dataclass
class Scene:
    """One accepted clearance observation, matching the runtime scene contract."""

    frame_id: int
    accepted_observation_sequence: int
    capture_timestamp_s: float
    accepted_monotonic_s: float
    held_box_bottom_z_base_m: float = 0.200
    held_box_bottom_uncertainty_m: float = 0.004
    stack_top_z_base_m: float = 0.110
    stack_top_uncertainty_m: float = 0.004
    held_box_pose_source: str = "fresh_dual_eef_fixed_ready_nominal_box_offset"
    stack_top_source: str = "complete_stack_plane"


def scene_window(count: int = 6, *, now_s: float = NOW_S, **overrides) -> list[Scene]:
    """Contiguous, monotonic, fresh observations ending just before ``now_s``."""

    window = []
    for index in range(count):
        capture = now_s - 0.30 + 0.05 * index
        window.append(
            Scene(
                frame_id=10 + index,
                accepted_observation_sequence=1 + index,
                capture_timestamp_s=capture,
                accepted_monotonic_s=capture + 0.01,
                **overrides,
            )
        )
    return window


def states(count: int = 12, *, now_s: float = NOW_S, span_s: float = 0.55):
    """Enough fresh measured states to satisfy the dwell and sample gates."""

    step = span_s / (count - 1)
    return [
        measured_state(sequence=i + 1, received_monotonic_s=now_s - span_s + i * step)
        for i in range(count)
    ]


def ready_errors_factory(arm_error_rad: float = 0.0, all_ready: bool = True):
    def ready_joint_errors(state):
        errors = np.zeros(JOINT_COUNT - 4, dtype=np.float64)  # torso+arms+head
        errors[6] = float(arm_error_rad)
        return errors, all_ready

    return ready_joint_errors


def evaluate(**overrides):
    kwargs = dict(
        config=CONFIG,
        scene_window=scene_window(),
        now_s=NOW_S,
        cartesian_arm_motion=True,
        states=states(),
        ready_joint_errors=ready_errors_factory(),
    )
    kwargs.update(overrides)
    scene = kwargs.pop("scene_window")
    config = kwargs.pop("config")
    return evaluate_grip_continuity(config, scene, **kwargs)


def test_nominal_loaded_hold_passes() -> None:
    result = evaluate()
    assert result.passed, result.reasons
    assert result.reasons == ()
    assert result.clearance_lower_bound_m == pytest.approx(0.082)
    assert result.fixed_ready_geometry_only_authorized


def test_too_few_state_samples_is_refused() -> None:
    result = evaluate(states=states(count=4, span_s=0.55))
    assert not result.passed
    assert "insufficient_fresh_robot_state_samples" in result.reasons


def test_short_dwell_is_refused() -> None:
    result = evaluate(states=states(count=12, span_s=0.20))
    assert not result.passed
    assert "insufficient_grip_dwell" in result.reasons


def test_stale_scene_evidence_is_refused() -> None:
    stale = scene_window(now_s=NOW_S - 1.0)
    result = evaluate(scene_window=stale)
    assert not result.passed
    assert "clearance_eef_box_bottom_evidence_stale" in result.reasons


def test_non_monotonic_frames_break_the_run() -> None:
    window = scene_window()
    window[3].frame_id = window[2].frame_id  # repeat, so not strictly increasing
    result = evaluate(scene_window=window)
    assert not result.passed
    assert any("monotonic" in reason for reason in result.reasons)


def test_unknown_stack_source_is_refused() -> None:
    result = evaluate(scene_window=scene_window(stack_top_source="rgb_guess"))
    assert not result.passed
    assert "fixed_ready_stack_plane_source_invalid" in result.reasons


def test_wrong_held_pose_source_is_refused() -> None:
    result = evaluate(scene_window=scene_window(held_box_pose_source="nominal_guess"))
    assert not result.passed
    assert "fixed_ready_box_bottom_geometry_invalid" in result.reasons


def test_too_few_scene_samples_is_refused() -> None:
    result = evaluate(scene_window=scene_window(count=3))
    assert not result.passed
    assert "insufficient_clearance_box_bottom_samples" in result.reasons


def test_cartesian_arm_motion_skips_ready_joint_tracking() -> None:
    """The loaded hold deliberately offsets the arms from the ready joints."""

    big = math.radians(30.0)
    holding = evaluate(cartesian_arm_motion=True, ready_joint_errors=ready_errors_factory(big))
    assert holding.passed, holding.reasons
    assert holding.arm_tracking_error_max_rad is None

    joint_hold = evaluate(
        cartesian_arm_motion=False, ready_joint_errors=ready_errors_factory(big)
    )
    assert not joint_hold.passed
    assert "arm_tracking_error" in joint_hold.reasons
    assert joint_hold.arm_tracking_error_max_rad == pytest.approx(big)


def test_joint_hold_within_tolerance_still_passes() -> None:
    small = math.radians(0.5)
    result = evaluate(
        cartesian_arm_motion=False, ready_joint_errors=ready_errors_factory(small)
    )
    assert result.passed, result.reasons
    assert result.arm_tracking_error_max_rad == pytest.approx(small)


def test_joint_not_ready_is_refused_in_joint_hold() -> None:
    result = evaluate(
        cartesian_arm_motion=False,
        ready_joint_errors=ready_errors_factory(all_ready=False),
    )
    assert not result.passed
    assert "target_joint_not_ready" in result.reasons


def test_moving_hands_break_the_eef_separation_gate() -> None:
    drifting = [
        measured_state(
            sequence=i + 1,
            received_monotonic_s=NOW_S - 0.55 + i * 0.05,
            right_xyz=(0.450, -0.130 - 0.004 * i, 0.300),
        )
        for i in range(12)
    ]
    result = evaluate(states=drifting)
    assert not result.passed
    assert "eef_separation_peak_to_peak" in result.reasons


def test_missing_eef_fk_is_refused() -> None:
    import dataclasses

    broken = [dataclasses.replace(s, T_base_right_eef=None) for s in states()]
    result = evaluate(states=broken)
    assert not result.passed
    assert "fresh_eef_fk_unavailable" in result.reasons


def test_reasons_are_deduplicated() -> None:
    result = evaluate(scene_window=scene_window(count=3))
    assert len(result.reasons) == len(set(result.reasons))


# --- the controller entry point keeps its authorization contract -------------


def test_controller_requires_explicit_geometry_only_acknowledgement() -> None:
    controller = offline_controller(CONFIG)
    with pytest.raises(CommandOwnershipError, match="explicit fixed-ready"):
        controller.evaluate_grip_and_clearance_dwell(scene_window())


def test_controller_requires_the_reviewed_config_flag() -> None:
    controller = offline_controller(PalletControlConfig())
    with pytest.raises(CommandOwnershipError, match="not enabled by the"):
        controller.evaluate_grip_and_clearance_dwell(
            scene_window(), allow_fixed_ready_geometry_only=True
        )


def test_controller_rejects_a_non_boolean_acknowledgement() -> None:
    controller = offline_controller(CONFIG)
    with pytest.raises(TypeError, match="must be a boolean"):
        controller.evaluate_grip_and_clearance_dwell(
            scene_window(), allow_fixed_ready_geometry_only=1  # type: ignore[arg-type]
        )


def test_controller_publishes_the_result_for_the_motion_gate() -> None:
    controller = offline_controller(CONFIG)
    controller._arm_stream_mode = ArmStreamMode.CARTESIAN_LOADED_HOLD
    for state in states():
        controller._state_history.append(state)
    controller._clock = lambda: NOW_S
    result = controller.evaluate_grip_and_clearance_dwell(
        scene_window(), allow_fixed_ready_geometry_only=True
    )
    assert result is controller._grip_result
    assert result.passed, result.reasons


# --------------------------------------------------------------------------- #
# intermittent clearance evidence must still latch
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# gates that a demonstrated placement posture made inapplicable
# --------------------------------------------------------------------------- #
def test_a_low_clearance_no_longer_blocks_motion() -> None:
    """The 50 mm floor bounded a computed descent that no longer exists.

    Placement lowers the carton by a posture whose travel the operator fixed, so
    the floor has nothing left to bound.  The measurement is still reported.
    """

    result = evaluate(scene_window=scene_window(held_box_bottom_z_base_m=0.140))
    assert result.passed, result.reasons
    assert "insufficient_vertical_clearance" not in result.reasons
    assert result.clearance_lower_bound_m is not None
    assert result.clearance_lower_bound_m < 0.050, "the low clearance was measured"


def test_intermittently_collected_evidence_no_longer_waits_on_a_span() -> None:
    """place_26 held for a whole run because five frames spread past 0.50 s."""

    window = []
    for index in range(6):
        capture = NOW_S - 0.20 - 0.25 * (5 - index)
        window.append(
            Scene(
                frame_id=10 + index,
                accepted_observation_sequence=1 + index,
                capture_timestamp_s=capture,
                accepted_monotonic_s=capture + 0.01,
            )
        )
    result = evaluate(scene_window=window)
    assert "clearance_evidence_span_too_long" not in result.reasons
    assert result.scene_evidence_span_s is not None
    assert result.scene_evidence_span_s > 0.50, "the span really is long now"


def test_stale_evidence_is_still_refused() -> None:
    """Freshness stays: acting on an old stack reading is a different failure."""

    window = scene_window()
    result = evaluate(
        scene_window=window,
        now_s=window[-1].accepted_monotonic_s + 0.50,
    )
    assert "clearance_eef_box_bottom_evidence_stale" in result.reasons


def test_a_sinking_carton_is_still_refused() -> None:
    """Removing the floor must not stop the interlock noticing a slipping box."""

    window = scene_window()
    for index, scene in enumerate(window):
        object.__setattr__(scene, "held_box_bottom_z_base_m", 0.200 - 0.004 * index)
    result = evaluate(scene_window=window)
    assert any("box_bottom" in reason for reason in result.reasons), result.reasons


def test_two_contiguous_frames_are_enough() -> None:
    from parcel_pose_placing.pallet_control import PalletControlConfig

    config = PalletControlConfig(
        fixed_ready_geometry_only_commissioning_enabled=True,
        held_top_direct_plane_dwell_frames=2,
    )
    assert config.held_top_direct_plane_dwell_frames == 2
    with pytest.raises(ValueError, match="at least 2 frames"):
        PalletControlConfig(
            fixed_ready_geometry_only_commissioning_enabled=True,
            held_top_direct_plane_dwell_frames=1,
        )
