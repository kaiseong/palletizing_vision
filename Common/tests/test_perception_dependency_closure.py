from __future__ import annotations

import ast
from pathlib import Path
import runpy
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
PICKING_ROOT = REPO_ROOT / "Box_picking" / "src" / "parcel_pose_picking"
PLACING_ROOT = REPO_ROOT / "Box_placing" / "src" / "parcel_pose_placing"

_POLICY = runpy.run_path(
    str(Path(__file__).with_name("test_perception_side_effect_boundaries.py"))
)
_collect_named_perception_sources = _POLICY["_collect_perception_sources"]
_format_violations = _POLICY["_format_violations"]
_scan_perception_sources = _POLICY["_scan_perception_sources"]


def _existing_module_paths(base: Path) -> tuple[Path, ...]:
    candidates = (base.with_suffix(".py"), base / "__init__.py")
    return tuple(candidate for candidate in candidates if candidate.is_file())


def _absolute_local_parts(module: str, feature_root: Path) -> tuple[str, ...] | None:
    parts = tuple(part for part in module.split(".") if part)
    if not parts or parts[0] != feature_root.name:
        return None
    return parts[1:]


def _local_import_targets(path: Path, feature_root: Path) -> tuple[Path, ...]:
    """Resolve local modules imported by ``path`` without importing them."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package_parts = list(path.relative_to(feature_root).parent.parts)
    targets: set[Path] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_parts = _absolute_local_parts(alias.name, feature_root)
                if module_parts is not None:
                    targets.update(
                        _existing_module_paths(feature_root.joinpath(*module_parts))
                    )
            continue

        if not isinstance(node, ast.ImportFrom):
            continue

        if node.level:
            parents_up = node.level - 1
            if parents_up > len(package_parts):
                continue
            base_parts = package_parts[: len(package_parts) - parents_up]
            module_parts = (
                () if node.module is None else tuple(node.module.split("."))
            )
            local_parts = (*base_parts, *module_parts)
        elif node.module is not None:
            absolute_parts = _absolute_local_parts(node.module, feature_root)
            if absolute_parts is None:
                continue
            local_parts = absolute_parts
        else:
            continue

        base = feature_root.joinpath(*local_parts)
        targets.update(_existing_module_paths(base))
        for alias in node.names:
            if alias.name != "*":
                targets.update(_existing_module_paths(base / alias.name))

    return tuple(sorted(targets))


def _dependency_closure(
    seeds: Iterable[Path],
    *,
    feature_root: Path,
) -> tuple[Path, ...]:
    """Collect every same-feature Python dependency reachable from ``seeds``."""

    pending = list(seeds)
    discovered: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in discovered:
            continue
        try:
            path.relative_to(feature_root)
        except ValueError:
            continue
        if not path.is_file() or path.suffix != ".py":
            continue
        discovered.add(path)
        pending.extend(
            target
            for target in _local_import_targets(path, feature_root)
            if target not in discovered
        )
    return tuple(sorted(discovered))


def _production_perception_sources() -> tuple[Path, ...]:
    sources = set(
        _collect_named_perception_sources((PICKING_ROOT, PLACING_ROOT))
    )
    seeds_by_root = {
        PICKING_ROOT: (PICKING_ROOT / "box_perception.py",),
        PLACING_ROOT: (
            PLACING_ROOT / "pallet_perception_adapter.py",
            PLACING_ROOT / "pallet_geometry.py",
        ),
    }
    for root, seeds in seeds_by_root.items():
        sources.update(_dependency_closure(seeds, feature_root=root))
    return tuple(sorted(sources))


def test_production_facade_dependency_closure_has_no_sdk_side_effects() -> None:
    sources = _production_perception_sources()
    expected_helpers = {
        PICKING_ROOT / "estimator.py",
        PICKING_ROOT / "evaluation.py",
        PICKING_ROOT / "projection.py",
        PICKING_ROOT / "rectangle_fit.py",
        PLACING_ROOT / "pallet_geometry.py",
        PLACING_ROOT / "pallet_models.py",
    }

    assert expected_helpers <= set(sources)
    violations = _scan_perception_sources(sources)
    assert violations == (), _format_violations(violations)


def test_name_independent_nested_helper_is_discovered_and_rejected(
    tmp_path: Path,
) -> None:
    feature_root = tmp_path / "parcel_pose_feature"
    facade = feature_root / "box_perception.py"
    helper_package = feature_root / "helpers"
    helper = helper_package / "motion.py"
    helper_package.mkdir(parents=True)
    (feature_root / "__init__.py").write_text("", encoding="utf-8")
    facade.write_text("from .helpers import motion\n", encoding="utf-8")
    (helper_package / "__init__.py").write_text("", encoding="utf-8")
    helper.write_text(
        "def move(stream, command):\n    return stream.send_command(command)\n",
        encoding="utf-8",
    )

    named_sources = _collect_named_perception_sources((feature_root,))
    sources = _dependency_closure(named_sources, feature_root=feature_root)
    violations = _scan_perception_sources(sources)

    assert facade in sources
    assert helper in sources
    assert "perception" not in str(helper.relative_to(feature_root)).casefold()
    assert "command_send" in {violation.code for violation in violations}
