"""One-command box-picking facade for the RB-Y1 parcel workflow."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent


def _reexec_active_conda_python() -> None:
    """Honor the activated conda environment even if PATH shadows python3.12."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    conda_python = Path(conda_prefix) / "bin" / "python"
    if not conda_python.is_file():
        return
    if conda_python.resolve() == Path(sys.executable).resolve():
        return

    os.execv(
        str(conda_python),
        [str(conda_python), str(Path(__file__).resolve()), *sys.argv[1:]],
    )


def _ensure_source_tree_imports() -> None:
    repo_root = PROJECT_ROOT.parent
    for source_root in (repo_root / "Common" / "src", PROJECT_ROOT / "src"):
        source_entry = str(source_root)
        if source_entry not in sys.path:
            sys.path.insert(0, source_entry)


def _run_box_picking(argv: Sequence[str]) -> int:
    _ensure_source_tree_imports()
    from parcel_pose_picking.cli import main as box_picking_main

    return box_picking_main(argv)


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        _reexec_active_conda_python()
    user_args = sys.argv[1:] if argv is None else list(argv)
    return _run_box_picking(user_args)


if __name__ == "__main__":
    raise SystemExit(main())
