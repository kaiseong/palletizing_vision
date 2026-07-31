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


def test_the_entry_point_owns_the_sequence() -> None:
    """box_pallet.py must visibly own alignment and release loop policy."""

    entry = pathlib.Path(__file__).resolve().parents[1] / "box_pallet.py"
    tree = ast.parse(entry.read_text(encoding="utf-8"))
    flows = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_placing_flow"
    ]
    assert len(flows) == 1
    flow = flows[0]
    assert len([node for node in ast.walk(flow) if isinstance(node, ast.While)]) == 2
    called = {
        node.func.attr
        for node in ast.walk(flow)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert {
        "acquire_frame",
        "perceive_frame",
        "decide_base_motion",
        "advance_placement",
        "record_frame",
        "execute_place",
        "execute_retreat",
    } <= called
    assert {"align", "await_release_authorization", "align_and_place"}.isdisjoint(
        called
    )


def test_the_stages_are_public_so_the_entry_point_can_call_them() -> None:
    """Planning remains public, while frame policy uses loop-free primitives."""

    from parcel_pose_placing import pallet_runtime
    from parcel_pose_placing.placing_session import PlacingSession

    for stage in (
        "resolve_live_plan",
        "assemble_live_stack",
        "initial_run_state",
        "open_placing_session",
    ):
        assert stage in pallet_runtime.__all__, f"{stage} must be public"
        assert callable(getattr(pallet_runtime, stage))
    for primitive in (
        "has_frame_budget",
        "open_acquisition",
        "acquire_frame",
        "perceive_frame",
        "decide_base_motion",
        "advance_placement",
        "record_frame",
        "finish_frame",
        "accept_descent_plan",
        "begin_release_observation",
    ):
        assert callable(getattr(PlacingSession, primitive))
    assert not hasattr(pallet_runtime, "run_pallet_live"), (
        "the old monolith must not come back alongside place_box"
    )

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


def test_perception_only_runs_keep_the_configured_camera_pose(
    monkeypatch,
) -> None:
    import numpy as np

    from parcel_pose_common.models import PoseResult
    from parcel_pose_placing import pallet_perception
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

    monkeypatch.setattr(
        pallet_perception,
        "pallet_pose_result",
        lambda _scene, *, slot, frame_id: PoseResult(
            x_m=None,
            y_m=None,
            yaw_rad=None,
            valid=False,
            reason="test_scene_unavailable",
            timestamp_s=1.0,
            diagnostics={"slot": slot, "frame_id": frame_id},
        ),
    )

    observation = observe_pallet_frame(
        Frame(),
        slot=1,
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


def test_the_picking_entry_point_owns_its_sequence_too() -> None:
    """box_picking.py visibly owns frame continuation and handoff decisions."""

    entry = (
        pathlib.Path(__file__).resolve().parents[2] / "Box_picking" / "box_picking.py"
    )
    tree = ast.parse(entry.read_text(encoding="utf-8"))
    flows = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_authorized_horizontal_pick"
    ]
    assert len(flows) == 1
    flow = flows[0]
    assert len([node for node in ast.walk(flow) if isinstance(node, ast.While)]) == 1
    called = {
        node.func.attr
        for node in ast.walk(flow)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert {"acquire_frame", "perceive_frame", "update", "record_frame"} <= called
    assert {"watch", "watch_for_alignment", "watch_and_grab"}.isdisjoint(called)

    from parcel_pose_picking import realtime
    from parcel_pose_picking.realtime import AlignmentSession

    for stage in ("resolve_live_view_plan", "open_alignment_session"):
        assert stage in realtime.__all__, f"{stage} must be public"
    for primitive in (
        "has_frame_budget",
        "acquire_frame",
        "perceive_frame",
        "record_frame",
        "cancel",
        "outcome",
    ):
        assert callable(getattr(AlignmentSession, primitive))
    assert not hasattr(realtime, "run_live_view"), (
        "the old monolith must not come back alongside pick_box"
    )
