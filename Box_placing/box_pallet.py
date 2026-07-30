"""One-command box-placing facade for the RB-Y1 pallet interlock workflow.

``python box_pallet.py`` runs perception only.  ``--execute`` runs the sequence on
the robot: verify the ready posture, align the base on the pallet hole, seat the
carton with the demonstrated placement posture, then withdraw the hands.
``--slot N`` selects the pallet slot; an undemonstrated slot is refused by name.

The sequence lives in place_box below, four stages deep:

    plan  = resolve_live_plan(...)    # refuse a bad request before anything moves
    stack = assemble_live_stack(...)  # estimator, gates, servo, placement sequencer
    state = initial_run_state(...)    # what the frame loop starts with
            align_and_place(...)      # open camera, drive onto the slot, place, tear down

Inside align_and_place every frame is observe, decide, advance, record, draw.  To
change a motion, change the posture in the slot config; to change how the base
decides to move, change decide_base_motion; to change the seating or the retreat,
change advance_placement.  Containment, stream expiry, telemetry and the overlay
stay in the library because they are guarantees, not flow.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

    from parcel_pose_placing.pallet_runtime import _ControllerLike


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


def place_box(
    root_config: Mapping[str, Any],
    *,
    execute: bool = False,
    auto_place_slot1: bool = False,
    ensure_slot1_ready: bool = False,
    slot: int | None = None,
    robot_address: str = "192.168.30.1:50051",
    robot_power: str = ".*",
    warmup_frames: int = 30,
    max_frames: int | None = None,
    headless: bool = False,
    window_name: str = "RB-Y1 Pallet Slot-1",
    output_mp4: str | Path | None = None,
    log_jsonl: str | Path | None = None,
    controller: _ControllerLike | None = None,
) -> int:
    """Place one carton in one pallet slot: this is the whole sequence.

    Execution is a standalone post-pick boundary: the previous process must be
    stopped, the configured loaded ready posture is verified, and this process
    becomes the sole combined body/mobility stream owner.
    """

    from parcel_pose_placing.pallet_runtime import (
        align_and_place,
        assemble_live_stack,
        initial_run_state,
        resolve_live_plan,
    )

    plan = resolve_live_plan(
        auto_place_slot1=auto_place_slot1,
        controller=controller,
        ensure_slot1_ready=ensure_slot1_ready,
        execute=execute,
        headless=headless,
        log_jsonl=log_jsonl,
        max_frames=max_frames,
        output_mp4=output_mp4,
        root_config=root_config,
        slot=slot,
        warmup_frames=warmup_frames,
    )

    # Imports stay below the standalone execute interlock.  In dry-run this is
    # still pure camera/perception code and cannot import rby1_sdk.
    stack = assemble_live_stack(
        auto_place_slot1=auto_place_slot1,
        root_config=root_config,
        selected_slot=plan.selected_slot,
    )
    state = initial_run_state(
        plan=plan,
        root_config=root_config,
    )
    align_and_place(
        controller=controller,
        auto_place_slot1=auto_place_slot1,
        ensure_slot1_ready=ensure_slot1_ready,
        execute=execute,
        plan=plan,
        state=state,
        stack=stack,
        headless=headless,
        log_jsonl=log_jsonl,
        max_frames=max_frames,
        output_mp4=output_mp4,
        robot_address=robot_address,
        robot_power=robot_power,
        root_config=root_config,
        window_name=window_name,
    )

    # Returning zero never implies the robot was disarmed.  The execute path stays
    # open unless forced cancellation or an acknowledged owner handoff closes it.
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        _reexec_active_conda_python()
    for source_root in (PROJECT_ROOT.parent / "Common" / "src", PROJECT_ROOT / "src"):
        source_entry = str(source_root)
        if source_entry not in sys.path:
            sys.path.insert(0, source_entry)
    from parcel_pose_placing.pallet_cli import build_parser

    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    # The live sequence lives in this file, so hand it to the live handler.
    args.place_box = place_box
    from parcel_pose_placing.pallet_cli import run_handler

    return int(run_handler(parser, args))


if __name__ == "__main__":
    raise SystemExit(main())
