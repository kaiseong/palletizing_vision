"""Make ``src/parcel_pose`` and the local test helpers importable."""

from __future__ import annotations

from pathlib import Path
import sys


TESTS_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = TESTS_ROOT.parents[0] / "src"
for entry in (str(SOURCE_ROOT), str(TESTS_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)
