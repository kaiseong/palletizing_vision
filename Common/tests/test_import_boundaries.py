from __future__ import annotations

import ast
from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON_ROOT = REPO_ROOT / "Common" / "src" / "parcel_pose_common"
PICKING_ROOT = REPO_ROOT / "Box_picking" / "src" / "parcel_pose_picking"
PLACING_ROOT = REPO_ROOT / "Box_placing" / "src" / "parcel_pose_placing"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_common_does_not_import_feature_packages() -> None:
    offenders: dict[str, list[str]] = {}
    for path in sorted((REPO_ROOT / "Common").rglob("*.py")):
        bad = sorted(
            name
            for name in _imports(path)
            if name in {"parcel_pose_picking", "parcel_pose_placing"}
            or name.startswith("parcel_pose_picking.")
            or name.startswith("parcel_pose_placing.")
        )
        if bad:
            offenders[str(path.relative_to(REPO_ROOT))] = bad

    assert offenders == {}


def test_feature_packages_do_not_cross_import() -> None:
    offenders: dict[str, list[str]] = {}
    feature_roots = {
        PICKING_ROOT: ("parcel_pose_placing", "parcel_pose_placing."),
        PLACING_ROOT: ("parcel_pose_picking", "parcel_pose_picking."),
    }
    for root, forbidden in feature_roots.items():
        for path in sorted(root.glob("*.py")):
            bad = sorted(
                name
                for name in _imports(path)
                if name == forbidden[0] or name.startswith(forbidden[1])
            )
            if bad:
                offenders[str(path.relative_to(REPO_ROOT))] = bad

    assert offenders == {}


def test_feature_packages_import_shared_modules_from_common() -> None:
    moved_modules = {
        "angles",
        "calibration",
        "mobile_servo",
        "models",
        "output",
        "plane",
        "realsense_adapter",
        "recording",
        "session",
        "transforms",
        "visualization",
    }
    offenders: dict[str, list[str]] = {}
    for path in sorted(PLACING_ROOT.glob("*.py")):
        bad: list[str] = []
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module in moved_modules:
                bad.append(f"from .{node.module}")
        if bad:
            offenders[str(path.relative_to(REPO_ROOT))] = sorted(bad)

    assert offenders == {}


def test_pyproject_declares_only_split_source_roots() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["description"] == (
        "RB-Y1 parcel picking, pallet placing, and shared recording utilities"
    )
    assert pyproject["tool"]["setuptools"]["packages"] == [
        "parcel_pose_common",
        "parcel_pose_picking",
        "parcel_pose_placing",
    ]
    package_dir = pyproject["tool"]["setuptools"]["package-dir"]
    assert package_dir == {
        "parcel_pose_common": "Common/src/parcel_pose_common",
        "parcel_pose_picking": "Box_picking/src/parcel_pose_picking",
        "parcel_pose_placing": "Box_placing/src/parcel_pose_placing",
    }
    pytest_options = pyproject["tool"]["pytest"]["ini_options"]
    assert "Codex" not in repr(pytest_options)
    assert "parcel_pose\"" not in repr(pyproject["tool"]["setuptools"])
