"""Make source packages and local test helpers importable."""

from __future__ import annotations

from pathlib import Path
import sys


TESTS_ROOT = Path(__file__).resolve().parent
PLACING_SOURCE_ROOT = TESTS_ROOT.parents[0] / "src"
COMMON_SOURCE_ROOT = TESTS_ROOT.parents[1] / "Common" / "src"
PICKING_SOURCE_ROOT = TESTS_ROOT.parents[1] / "Box_picking" / "src"
for entry in (
    str(COMMON_SOURCE_ROOT),
    str(PICKING_SOURCE_ROOT),
    str(PLACING_SOURCE_ROOT),
    str(TESTS_ROOT),
):
    if entry not in sys.path:
        sys.path.insert(0, entry)
