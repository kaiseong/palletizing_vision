"""One-command D435 recorder for the RB-Y1 parcel-pose dataset."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT.parent / "recordings" / "codex_640x480"


def build_record_args(argv: Sequence[str]) -> list[str]:
    return ["record", "--output", str(DEFAULT_OUTPUT), *argv]


def _run_parcel_pose(argv: Sequence[str]) -> int:
    source_root = PROJECT_ROOT / "src"
    sys.path.insert(0, str(source_root))
    from parcel_pose.cli import main as parcel_pose_main

    return parcel_pose_main(argv)


def main(argv: Sequence[str] | None = None) -> int:
    user_args = sys.argv[1:] if argv is None else list(argv)
    return _run_parcel_pose(build_record_args(user_args))


if __name__ == "__main__":
    raise SystemExit(main())
