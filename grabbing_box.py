"""Compatibility entrypoint for the package-local RB-Y1 grasp sequence.

The authoritative implementation lives in ``Codex/src/parcel_pose/grabbing.py``
so copying the Codex directory is sufficient for live auto-grab operation.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys


def _implementation():
    codex_src = Path(__file__).resolve().parent / "Codex" / "src"
    source_path = str(codex_src)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    return importlib.import_module("parcel_pose.grabbing")


_grabbing = _implementation()

if __name__ == "__main__":
    arguments = _grabbing._parse_args()
    try:
        succeeded = _grabbing.main(
            address=arguments.address,
            model=arguments.model,
            power=arguments.power,
        )
    except KeyboardInterrupt:
        print("\n[grabbing] interrupted by user")
        raise SystemExit(130) from None
    raise SystemExit(0 if succeeded else 1)
else:
    # Preserve ``import grabbing_box`` compatibility without duplicating the
    # implementation or giving wrapper functions a second globals namespace.
    sys.modules[__name__] = _grabbing
