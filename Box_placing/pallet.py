"""Short entry point for the supervised RB-Y1 pallet slot-1 hover MVP."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent


def _reexec_active_conda_python() -> None:
    """Honor the active conda interpreter when PATH resolves another Python."""

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


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        _reexec_active_conda_python()
    for source_root in (PROJECT_ROOT.parent / "Common" / "src", PROJECT_ROOT / "src"):
        source_entry = str(source_root)
        if source_entry not in sys.path:
            sys.path.insert(0, source_entry)
    from parcel_pose_placing.pallet_cli import main as pallet_main

    return int(pallet_main(sys.argv[1:] if argv is None else list(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
