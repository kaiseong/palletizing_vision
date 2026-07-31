from __future__ import annotations

from pathlib import Path
import runpy


REPO_ROOT = Path(__file__).resolve().parents[2]
_POLICY = runpy.run_path(
    str(Path(__file__).with_name("test_perception_side_effect_boundaries.py"))
)
_format_violations = _POLICY["_format_violations"]
_scan_perception_sources = _POLICY["_scan_perception_sources"]


def test_feature_package_initializers_cannot_hide_perception_sdk_side_effects() -> None:
    sources = (
        REPO_ROOT / "Box_picking/src/parcel_pose_picking/__init__.py",
        REPO_ROOT / "Box_placing/src/parcel_pose_placing/__init__.py",
    )

    assert all(path.is_file() for path in sources)
    violations = _scan_perception_sources(sources)
    assert violations == (), _format_violations(violations)
