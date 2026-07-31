from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

from parcel_pose_placing import pallet_perception_adapter
from parcel_pose_placing.pallet_models import PalletSceneObservation, StackObservation
from parcel_pose_placing.pallet_perception_adapter import (
    pallet_pose_result,
    perceive_pallet_pose,
)


def _scene(*, valid: bool = True) -> PalletSceneObservation:
    return PalletSceneObservation(
        stack=StackObservation(
            timestamp_s=12.5,
            center_base=np.array([0.80, 0.10, 0.20]) if valid else None,
            u_right_base=np.array([1.0, 0.0, 0.0]) if valid else None,
            v_far_base=np.array([0.0, 1.0, 0.0]) if valid else None,
            yaw_base_rad=-1.25 if valid else None,
            plane_height_base_m=0.20 if valid else None,
            slot1_target_base=np.array([0.865, 0.139523, 0.20]) if valid else None,
            opening_size_m=(0.40, 0.253) if valid else None,
            quality={"line_support_ratio": 0.91},
            valid=valid,
            rejection_reasons=() if valid else ("opening_not_found", "low_support"),
            calibration_status="base_validated",
            axis_branch="image_right" if valid else None,
            stack_se2_source="fixed_outer_l_corner" if valid else None,
        ),
        held_top=None,
    )


def test_slot1_adapter_returns_base_xy_line_yaw_and_preserves_provenance() -> None:
    result = pallet_pose_result(_scene(), slot=1, frame_id=37)

    assert result.valid is True
    assert result.frame == "base"
    assert result.x_m == pytest.approx(0.80)
    assert result.y_m == pytest.approx(0.10)
    assert result.yaw_rad == pytest.approx(-1.25)
    assert result.timestamp_s == pytest.approx(12.5)
    assert result.reason == ""
    assert result.diagnostics["frame_id"] == 37
    assert result.diagnostics["position_source"] == "stack.center_base"
    assert result.diagnostics["yaw_source"] == "stack.yaw_base_rad"
    assert result.diagnostics["stack_se2_source"] == "fixed_outer_l_corner"
    assert result.diagnostics["quality"] == {"line_support_ratio": 0.91}
    assert result.diagnostics["observation"]["stack"]["timestamp_s"] == 12.5


def test_invalid_scene_keeps_timestamp_and_rejection_provenance() -> None:
    result = pallet_pose_result(_scene(valid=False), slot=1)

    assert result.valid is False
    assert result.x_m is None
    assert result.y_m is None
    assert result.yaw_rad is None
    assert result.reason == "opening_not_found"
    assert result.timestamp_s == pytest.approx(12.5)
    assert result.diagnostics["rejection_reasons"] == [
        "opening_not_found",
        "low_support",
    ]


@pytest.mark.parametrize("slot", [2, 5, 6])
def test_incomplete_slot_pose_is_a_stable_fail_closed_result(slot: int) -> None:
    result = pallet_pose_result(_scene(), slot=slot, frame_id=4)

    assert result.valid is False
    assert result.reason == f"slot_{slot}_pose_unavailable"
    assert result.timestamp_s == pytest.approx(12.5)
    assert result.diagnostics["slot"] == slot
    assert result.diagnostics["stack_se2_source"] == "fixed_outer_l_corner"


def test_frame_facade_calls_the_injected_estimator_exactly_once() -> None:
    expected_scene = _scene()

    class EstimatorSpy:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def estimate(self, *args: object, **kwargs: object) -> PalletSceneObservation:
            self.calls.append((args, kwargs))
            return expected_scene

    estimator = EstimatorSpy()
    rgb = object()
    depth = object()
    intrinsics = object()
    transform = object()
    held_hint = object()

    result = perceive_pallet_pose(
        rgb,
        depth,
        intrinsics,
        transform,
        slot=1,
        estimator=estimator,
        timestamp_s=12.5,
        frame_id=37,
        held_box_hint=held_hint,
        calibration_status="base_validated",
    )

    assert result.valid is True
    assert len(estimator.calls) == 1
    args, kwargs = estimator.calls[0]
    assert args == (depth, intrinsics, transform)
    assert kwargs == {
        "timestamp_s": 12.5,
        "frame_id": 37,
        "color_on_depth_bgr": rgb,
        "held_box_hint": held_hint,
        "calibration_status": "base_validated",
    }


def test_facade_source_has_no_robot_sdk_or_command_side_effect_calls() -> None:
    tree = ast.parse(inspect.getsource(pallet_perception_adapter))
    forbidden_imports: list[str] = []
    forbidden_calls: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            forbidden_imports.extend(
                alias.name for alias in node.names if alias.name == "rby1_sdk"
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "rby1_sdk":
            forbidden_imports.append(node.module)
        elif isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else None
            if name in {
                "create_robot",
                "create_command_stream",
                "send_command",
                "send_command_builder",
            }:
                forbidden_calls.append(name)

    assert forbidden_imports == []
    assert forbidden_calls == []
