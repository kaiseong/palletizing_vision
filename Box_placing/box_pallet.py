"""One-command box-placing facade for the RB-Y1 pallet interlock workflow.

``python box_pallet.py`` runs perception only.  ``--execute`` runs the sequence on
the robot: verify the ready posture, align the base on the pallet hole, seat the
carton with the demonstrated placement posture, then withdraw the hands.
``--slot N`` selects the pallet slot; an undemonstrated slot is refused by name.

TODO(flatten-flow): this file is still only a launcher.  The intent is that main()
reads as the whole sequence, top to bottom, so a wrong motion is one visible line:

    config = load_placing_config()
    target_xy, ready, place, retreat = slot_plan(config, slot)
    robot = connect(config.address)
    send_once_joint_position(robot, ready, minimum_time_s=3.0)
    with open_camera(config) as camera, mobility_stream(robot) as base:
        while True:
            rgb, depth, intrinsics = camera.read()
            T_base_depth = camera_pose(robot.measured_head_fk(), config)
            x, y, yaw = find_pallet_hole(rgb, depth, intrinsics, T_base_depth, slot)
            ...
    send_once_cartesian(robot, place, duration_s=1.0)
    send_once_cartesian(robot, retreat, duration_s=1.0)

Containment, stream-expiry handling, telemetry and the overlay stay in the library
behind ``mobility_stream``; they are guarantees, not flow.  Blocked on extracting
two phases still inside pallet_runtime.run_pallet_live: the servo/dispatch block
(245 lines, no cross-frame state, 19 outputs) and the placement block (163 lines,
two persistent flags, 9 outputs).  run_pallet_live is deleted once main() owns the
loop, so there is never a second implementation of the same sequence.
"""

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
