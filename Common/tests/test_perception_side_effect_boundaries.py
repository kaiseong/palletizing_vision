from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_ROOTS = (
    REPO_ROOT / "Box_picking" / "src" / "parcel_pose_picking",
    REPO_ROOT / "Box_placing" / "src" / "parcel_pose_placing",
)

SDK_MODULE = "rby1_sdk"
ROBOT_CONSTRUCTORS = frozenset({"Robot", "create_robot"})
STREAM_CONSTRUCTORS = frozenset(
    {
        "CommandStream",
        "RBY1MobilityStream",
        "RobotCommandStream",
        "create_command_stream",
    }
)
COMMAND_SEND_METHODS = frozenset(
    {"send_command", "send_command_async", "send_command_builder"}
)
SDK_REFERENCE_NAMES = frozenset(
    {
        "CommandHeader",
        "RobotCommand",
        "RobotCommandFeedback",
    }
)


@dataclass(frozen=True, slots=True)
class AstViolation:
    path: Path
    line: int
    column: int
    code: str
    detail: str


def _is_perception_source(path: Path, feature_root: Path) -> bool:
    relative = path.relative_to(feature_root)
    return any("perception" in part.casefold() for part in relative.parts)


def _collect_perception_sources(feature_roots: Iterable[Path]) -> tuple[Path, ...]:
    """Return every Python file in a perception module or package recursively."""

    sources = {
        path
        for root in feature_roots
        if root.is_dir()
        for path in root.rglob("*.py")
        if _is_perception_source(path, root)
    }
    return tuple(sorted(sources))


def _is_sdk_module(name: str) -> bool:
    return name == SDK_MODULE or name.startswith(f"{SDK_MODULE}.")


def _literal_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_string(node.left)
        right = _literal_string(node.right)
        if left is not None and right is not None:
            return left + right
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            part = _literal_string(value)
            if part is None:
                return None
            parts.append(part)
        return "".join(parts)
    return None


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_sdk_builder_or_reference(name: str) -> bool:
    return name.endswith("CommandBuilder") or name in SDK_REFERENCE_NAMES


class _PerceptionSideEffectVisitor(ast.NodeVisitor):
    def __init__(self, path: Path, tree: ast.AST) -> None:
        self.path = path
        self.violations: list[AstViolation] = []
        self.importlib_aliases = {"importlib"}
        self.import_module_aliases: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "importlib":
                        self.importlib_aliases.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
                for alias in node.names:
                    if alias.name == "import_module":
                        self.import_module_aliases.add(alias.asname or alias.name)

    def _add(self, node: ast.AST, code: str, detail: str) -> None:
        self.violations.append(
            AstViolation(
                path=self.path,
                line=int(getattr(node, "lineno", 0)),
                column=int(getattr(node, "col_offset", 0)),
                code=code,
                detail=detail,
            )
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if _is_sdk_module(alias.name):
                self._add(node, "sdk_import", f"import {alias.name}")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is not None and _is_sdk_module(node.module):
            self._add(node, "sdk_import", f"from {node.module} import ...")

    def visit_Call(self, node: ast.Call) -> None:
        call_name = _call_name(node.func)
        dynamic_import = call_name == "__import__"
        if isinstance(node.func, ast.Name):
            dynamic_import = dynamic_import or node.func.id in self.import_module_aliases
        elif isinstance(node.func, ast.Attribute):
            dynamic_import = dynamic_import or (
                node.func.attr == "import_module"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in self.importlib_aliases
            )

        if dynamic_import:
            target_node = node.args[0] if node.args else next(
                (item.value for item in node.keywords if item.arg == "name"),
                None,
            )
            target = None if target_node is None else _literal_string(target_node)
            if target is not None and _is_sdk_module(target):
                self._add(node, "dynamic_sdk_import", f"dynamic import {target}")

        if call_name in ROBOT_CONSTRUCTORS:
            self._add(node, "robot_construction", f"call {call_name}()")
        if call_name in STREAM_CONSTRUCTORS:
            self._add(node, "stream_construction", f"call {call_name}()")
        if call_name in COMMAND_SEND_METHODS:
            self._add(node, "command_send", f"call {call_name}()")

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if _is_sdk_builder_or_reference(node.attr):
            self._add(node, "sdk_builder_or_reference", f"reference {node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if _is_sdk_builder_or_reference(node.id):
            self._add(node, "sdk_builder_or_reference", f"reference {node.id}")


def _scan_perception_sources(paths: Iterable[Path]) -> tuple[AstViolation, ...]:
    violations: list[AstViolation] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor = _PerceptionSideEffectVisitor(path, tree)
        visitor.visit(tree)
        violations.extend(visitor.violations)
    return tuple(
        sorted(
            violations,
            key=lambda item: (
                str(item.path),
                item.line,
                item.column,
                item.code,
                item.detail,
            ),
        )
    )


def _format_violations(violations: Iterable[AstViolation]) -> str:
    return "\n".join(
        f"{item.path}:{item.line}:{item.column}: {item.code}: {item.detail}"
        for item in violations
    )


def _write_nested_probe(tmp_path: Path, source: str) -> tuple[Path, Path]:
    feature_root = tmp_path / "parcel_pose_feature"
    probe = feature_root / "box_perception" / "nested" / "probe.py"
    probe.parent.mkdir(parents=True)
    probe.write_text(source, encoding="utf-8")
    return feature_root, probe


def test_production_perception_facades_have_no_sdk_side_effects() -> None:
    sources = _collect_perception_sources(FEATURE_ROOTS)

    assert REPO_ROOT / "Box_placing/src/parcel_pose_placing/pallet_perception.py" in sources
    violations = _scan_perception_sources(sources)
    assert violations == (), _format_violations(violations)


def test_collector_discovers_new_nested_perception_python_files(tmp_path: Path) -> None:
    feature_root, probe = _write_nested_probe(
        tmp_path,
        "def open_stream(robot):\n    return robot.create_command_stream()\n",
    )
    unrelated = feature_root / "runtime" / "robot_adapter.py"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("import rby1_sdk\n", encoding="utf-8")

    sources = _collect_perception_sources((feature_root,))

    assert probe in sources
    assert unrelated not in sources
    assert {item.code for item in _scan_perception_sources(sources)} == {
        "stream_construction"
    }


@pytest.mark.parametrize(
    ("source", "expected_code"),
    (
        ("import rby1_sdk\n", "sdk_import"),
        ("import rby1_sdk as rby\n", "sdk_import"),
        (
            "from rby1_sdk import RobotCommandBuilder as Builder\n",
            "sdk_import",
        ),
        (
            "import importlib as loader\n"
            "sdk = loader.import_module('rby1_sdk')\n",
            "dynamic_sdk_import",
        ),
        (
            "from importlib import import_module as load\n"
            "sdk = load('rby1' + '_sdk.control')\n",
            "dynamic_sdk_import",
        ),
        ("sdk = __import__(name='rby1_sdk')\n", "dynamic_sdk_import"),
        ("builder_type = sdk.RobotCommandBuilder\n", "sdk_builder_or_reference"),
        ("robot = create_robot('address', 'm')\n", "robot_construction"),
        ("stream = robot.create_command_stream()\n", "stream_construction"),
        ("feedback = stream.send_command(command)\n", "command_send"),
        ("builder = robot.send_command_builder(command)\n", "command_send"),
    ),
    ids=(
        "direct-sdk-import",
        "aliased-sdk-import",
        "from-sdk-import",
        "aliased-importlib-dynamic-import",
        "concatenated-string-dynamic-import",
        "dunder-dynamic-import",
        "sdk-builder-reference",
        "robot-construction",
        "stream-construction",
        "command-send",
        "command-send-builder",
    ),
)
def test_adversarial_nested_perception_source_is_rejected(
    tmp_path: Path,
    source: str,
    expected_code: str,
) -> None:
    feature_root, probe = _write_nested_probe(tmp_path, source)

    sources = _collect_perception_sources((feature_root,))
    violations = _scan_perception_sources(sources)

    assert sources == (probe,)
    assert expected_code in {item.code for item in violations}
