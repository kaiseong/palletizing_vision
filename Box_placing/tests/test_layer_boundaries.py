"""Layer boundaries: drawing and telemetry must not live in the control loop."""

from __future__ import annotations

import ast
import pathlib

SOURCE_DIR = (
    pathlib.Path(__file__).resolve().parents[1] / "src" / "parcel_pose_placing"
)


def functions(module: str) -> dict[str, int]:
    tree = ast.parse((SOURCE_DIR / module).read_text(encoding="utf-8"))
    return {
        node.name: node.end_lineno - node.lineno + 1
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }


def test_drawing_lives_in_the_visualization_module() -> None:
    assert "draw_live_overlay" in functions("pallet_visualization.py")
    assert "project_base_points" in functions("pallet_visualization.py")
    assert "_draw_live_overlay" not in functions("pallet_runtime.py")


def test_telemetry_builders_live_in_their_own_module() -> None:
    telemetry = functions("pallet_telemetry.py")
    for name in ("_telemetry_record", "_recovery_contract_record"):
        assert name in telemetry, name
        assert name not in functions("pallet_runtime.py")


def test_the_telemetry_module_cannot_reach_the_control_loop() -> None:
    """A telemetry change must not be able to alter motion."""

    source = (SOURCE_DIR / "pallet_telemetry.py").read_text(encoding="utf-8")
    assert "pallet_runtime" not in source
    assert "pallet_control" not in source


def test_the_drawing_module_cannot_reach_the_control_loop() -> None:
    source = (SOURCE_DIR / "pallet_visualization.py").read_text(encoding="utf-8")
    assert "pallet_runtime" not in source


def test_the_control_loop_keeps_shrinking() -> None:
    """A ratchet, not a target: the loop had 3,457 lines before the split."""

    lines = len(
        (SOURCE_DIR / "pallet_runtime.py").read_text(encoding="utf-8").splitlines()
    )
    assert lines < 2900, f"pallet_runtime.py grew back to {lines} lines"


def test_perception_is_one_call_in_the_control_loop() -> None:
    """The loop must read as control flow, not as depth scaling."""

    source = (SOURCE_DIR / "pallet_runtime.py").read_text(encoding="utf-8")
    assert "observe_pallet_frame(" in source
    # The estimator is reached through that call, not inline.
    assert "estimator.estimate(" not in source
    assert "observe_pallet_frame" in functions("pallet_perception.py")


def test_the_camera_pose_is_an_explicit_input() -> None:
    """A wrong camera pose looks exactly like bad perception; keep it visible."""

    import inspect

    from parcel_pose_placing.pallet_perception import observe_pallet_frame

    parameters = inspect.signature(observe_pallet_frame).parameters
    assert "configured_T_base_depth" in parameters
    assert "controller" in parameters


def test_perception_only_runs_keep_the_configured_camera_pose() -> None:
    import numpy as np

    from parcel_pose_placing.pallet_perception import observe_pallet_frame

    configured = np.eye(4, dtype=np.float64)
    configured[:3, 3] = (0.1, 0.2, 1.2)

    class Frame:
        raw_depth_z16 = np.zeros((4, 4), dtype=np.uint16)
        raw_color_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        color_on_depth_bgr = None
        depth_frame_number = 7

    class Contract:
        depth_scale_m = 0.001
        depth_intrinsics = None

    seen: dict[str, object] = {}

    class Estimator:
        def estimate(self, depth, intrinsics, transform, **kwargs):
            seen["transform"] = transform
            return None

    class Held:
        center_base_xyz_m = (0.9, 0.0, 0.7)
        yaw_base_rad = 0.0
        source = "test"

    observation = observe_pallet_frame(
        Frame(),
        root_config={},
        contract=Contract(),
        estimator=Estimator(),
        controller=None,
        calibration_status="nominal",
        configured_T_base_depth=configured,
        configured_held_proxy=Held(),
        capture_monotonic_s=1.0,
        accepted_scene_sequence=0,
        maximum_box_height_m=0.164,
        box_bottom_uncertainty_m=0.015,
    )
    np.testing.assert_array_equal(seen["transform"], configured)
    np.testing.assert_array_equal(observation.T_base_depth, configured)
