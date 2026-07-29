"""Runtime telemetry must expose the release geometry it acted on."""

from __future__ import annotations

import json
import math

import pytest

from parcel_pose.pallet_place import PlacementRequest, PlacementState
from parcel_pose.pallet_runtime import (
    _dispatch_placement_fault_hold_if_needed,
    _placement_telemetry_payload,
)

from _factories import (
    descent_plan,
    loaded_hold_target,
    measured_state,
    offline_controller,
)


def release_target(controller, spread: float = 0.030):
    state = measured_state()
    plan = descent_plan(planned_delta_z_m=0.0)
    loaded = loaded_hold_target(controller, state)
    lowering = controller._make_lowering_target_from_loaded_hold(
        state,
        loaded_target=loaded,
        descent_plan=plan,
        requested_squeeze_offset_m=None,
    )
    target = controller._make_release_target_from_plan(
        state,
        lowering_target=lowering,
        descent_plan=plan,
        release_spread_m=spread,
    )
    controller._arm_stream_mode = target.mode
    controller._cartesian_arm_target = target
    return target


def test_placement_telemetry_reports_the_release_geometry() -> None:
    controller = offline_controller()
    release_target(controller)
    telemetry = controller.placement_telemetry()
    assert telemetry.release_spread_m == pytest.approx(0.030)
    assert telemetry.release_axis_base == pytest.approx((0.0, -1.0, 0.0))
    assert telemetry.release_axis_deviation_rad == pytest.approx(0.0)
    assert telemetry.lowering_distance_m == pytest.approx(0.0)


def test_placement_telemetry_payload_is_json_serializable() -> None:
    controller = offline_controller()
    release_target(controller)
    payload = _placement_telemetry_payload(controller.placement_telemetry())
    # The live JSONL writer uses the same strict json.dumps settings.
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["release_spread_m"] == pytest.approx(0.030)
    assert decoded["release_axis_base"] == pytest.approx([0.0, -1.0, 0.0])
    assert decoded["release_axis_deviation_rad"] == pytest.approx(0.0)
    assert decoded["arm_mode"] == "CARTESIAN_PLACEMENT_RELEASE"


def test_placement_telemetry_payload_tolerates_no_target() -> None:
    controller = offline_controller()
    payload = _placement_telemetry_payload(controller.placement_telemetry())
    assert payload["release_spread_m"] is None
    assert payload["release_axis_base"] is None


def test_fault_hold_is_dispatched_only_after_a_placement_command() -> None:
    calls: list[str] = []

    class FakeController:
        def fail_closed_cartesian_placement_hold(self, *, reason: str) -> None:
            calls.append(reason)

    class FakeOutput:
        state = PlacementState.FAULT_HOLD
        request = PlacementRequest.HOLD_CURRENT
        faulted = True
        reason = "release_axis_deviation"

    controller = FakeController()
    assert (
        _dispatch_placement_fault_hold_if_needed(
            controller,  # type: ignore[arg-type]
            FakeOutput(),  # type: ignore[arg-type]
            lowering_started=False,
            release_started=False,
        )
        is None
    )
    assert calls == []

    assert (
        _dispatch_placement_fault_hold_if_needed(
            controller,  # type: ignore[arg-type]
            FakeOutput(),  # type: ignore[arg-type]
            lowering_started=True,
            release_started=False,
        )
        == "fail_closed_cartesian_placement_hold"
    )
    assert calls == ["release_axis_deviation"]


def test_axis_deviation_is_reported_in_degrees_for_operators() -> None:
    controller = offline_controller()
    right = (0.450 - 0.0113, -0.1295, 0.300)
    left = (0.450 + 0.0113, 0.1295, 0.300)
    state = measured_state(right_xyz=right, left_xyz=left)
    plan = descent_plan(planned_delta_z_m=0.0, right_xyz=right, left_xyz=left)
    loaded = loaded_hold_target(controller, state)
    lowering = controller._make_lowering_target_from_loaded_hold(
        state,
        loaded_target=loaded,
        descent_plan=plan,
        requested_squeeze_offset_m=None,
    )
    controller._cartesian_arm_target = controller._make_release_target_from_plan(
        state,
        lowering_target=lowering,
        descent_plan=plan,
        release_spread_m=0.030,
    )
    telemetry = controller.placement_telemetry()
    assert math.degrees(telemetry.release_axis_deviation_rad) == pytest.approx(
        5.0, abs=0.05
    )
