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
