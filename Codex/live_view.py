"""One-command D435 live viewer for the RB-Y1 parcel-pose estimator."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CALIBRATION = PROJECT_ROOT / "configs" / "rby1m_v1_2_fixed_table_nominal.json"


def _has_option(argv: Sequence[str], name: str) -> bool:
    return any(argument == name or argument.startswith(f"{name}=") for argument in argv)


def build_live_view_args(argv: Sequence[str]) -> list[str]:
    arguments = ["live-view"]
    if not _has_option(argv, "--calibration"):
        arguments.extend(("--calibration", str(DEFAULT_CALIBRATION)))
    arguments.extend(argv)
    return arguments


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


def _run_parcel_pose(argv: Sequence[str]) -> int:
    source_root = PROJECT_ROOT / "src"
    sys.path.insert(0, str(source_root))
    from parcel_pose.cli import main as parcel_pose_main

    return parcel_pose_main(argv)


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        _reexec_active_conda_python()
    user_args = sys.argv[1:] if argv is None else list(argv)
    return _run_parcel_pose(build_live_view_args(user_args))


if __name__ == "__main__":
    raise SystemExit(main())
