"""One-command D435 recorder for the RB-Y1 parcel-pose dataset."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_OUTPUT = REPO_ROOT / "recordings" / "codex_640x480"
DEFAULT_CONFIG = REPO_ROOT / "Box_picking" / "configs" / "d435_rby1_nominal.json"


def _has_option(argv: Sequence[str], name: str) -> bool:
    return any(argument == name or argument.startswith(f"{name}=") for argument in argv)


def build_record_args(argv: Sequence[str]) -> list[str]:
    arguments = ["record"]
    if not _has_option(argv, "--output"):
        arguments.extend(("--output", str(DEFAULT_OUTPUT)))
    arguments.extend(argv)
    return arguments


def _reexec_active_conda_python() -> None:
    """Honor the activated conda interpreter if PATH resolves another Python."""
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


def _run_parcel_pose(argv: Sequence[str]) -> int:
    for source_root in (REPO_ROOT / "Common" / "src",):
        source_entry = str(source_root)
        if source_entry not in sys.path:
            sys.path.insert(0, source_entry)
    from parcel_pose_common.record_cli import main as record_main

    return int(record_main(argv[1:], default_config=DEFAULT_CONFIG))


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        _reexec_active_conda_python()
    user_args = sys.argv[1:] if argv is None else list(argv)
    return _run_parcel_pose(build_record_args(user_args))


if __name__ == "__main__":
    raise SystemExit(main())
