"""Runtime telemetry must expose the release geometry it acted on."""

from __future__ import annotations

import json

import pytest

from parcel_pose_placing.pallet_control import ArmStreamMode

from parcel_pose_placing.pallet_place import PlacementRequest, PlacementState
from parcel_pose_placing.pallet_runtime import (
    _dispatch_placement_fault_hold_if_needed,
    _placement_telemetry_payload,
)

from _factories import (
    RIGHT_ARM_INDICES,
    descent_plan,
    loaded_hold_target,
    measured_state,
    offline_controller,
)


def _flat_fk(position, velocity):
    """Torso at base z=0.90, wrists offset by joint 0 so a posture actually moves."""

    import numpy as np

    def transform(xyz):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = np.asarray(xyz, dtype=np.float64)
        return matrix

    shift = float(position[RIGHT_ARM_INDICES][0])
    return {
        "T_base_torso": transform((0.0, 0.0, 0.90)),
        "T_base_head": transform((0.10, 0.0, 1.20)),
        "T_base_right_eef": transform((0.45 + shift, -0.17, 0.73)),
        "T_base_left_eef": transform((0.45 + shift, 0.17, 0.73)),
        "base_twist_w_vx_vy": (0.0, 0.0, 0.0),
    }


def retreat_config():
    """A controller config that demonstrates both placement postures."""

    from parcel_pose_placing.pallet_control import PalletControlConfig, PlacePose, ReadyPose

    ready = ReadyPose()
    place = PlacePose(
        torso_rad=ready.torso_rad,
        right_arm_rad=ready.right_arm_rad,
        left_arm_rad=ready.left_arm_rad,
    )
    retreat = PlacePose(
        torso_rad=ready.torso_rad,
        right_arm_rad=tuple(v + 0.05 for v in ready.right_arm_rad),
        left_arm_rad=tuple(v + 0.05 for v in ready.left_arm_rad),
    )
    return PalletControlConfig(place_pose=place, retreat_pose=retreat)


def release_target(controller, spread: float = 0.030):
    state = measured_state()
    plan = descent_plan(
        planned_delta_z_m=0.0, target_source="demonstrated_place_pose"
    )
    loaded = loaded_hold_target(controller, state)
    lowering = controller._make_lowering_target_from_loaded_hold(
        state,
        loaded_target=loaded,
        descent_plan=plan,
        requested_squeeze_offset_m=None,
    )
    from parcel_pose_placing.pallet_control import ArmStreamMode

    target = controller._make_posture_target(
        state,
        loaded_target=lowering,
        descent_plan=plan,
        place_pose=controller.config.retreat_pose,
        mode=ArmStreamMode.CARTESIAN_PLACEMENT_RELEASE,
        duration_s=controller.config.placement_retreat_pose_duration_s,
    )
    controller._arm_stream_mode = target.mode
    controller._cartesian_arm_target = target
    return target


def test_placement_telemetry_reports_the_release_geometry() -> None:
    controller = offline_controller(retreat_config(), fk_provider=_flat_fk)
    release_target(controller)
    telemetry = controller.placement_telemetry()
    assert telemetry.arm_mode is ArmStreamMode.CARTESIAN_PLACEMENT_RELEASE
    assert telemetry.release_axis_base == pytest.approx((0.0, -1.0, 0.0))
    # The retreat posture moves the wrists, so the travel is reported.
    assert telemetry.lowering_distance_m is not None
    assert telemetry.lowering_distance_m > 0.0


def test_placement_telemetry_payload_is_json_serializable() -> None:
    controller = offline_controller(retreat_config(), fk_provider=_flat_fk)
    release_target(controller)
    payload = _placement_telemetry_payload(controller.placement_telemetry())
    # The live JSONL writer uses the same strict json.dumps settings.
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["release_axis_base"] == pytest.approx([0.0, -1.0, 0.0])
    assert decoded["arm_mode"] == "CARTESIAN_PLACEMENT_RELEASE"


def test_placement_telemetry_payload_tolerates_no_target() -> None:
    controller = offline_controller()
    payload = _placement_telemetry_payload(controller.placement_telemetry())
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
