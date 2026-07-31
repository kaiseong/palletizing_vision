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

from collections.abc import Mapping
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

    from parcel_pose_placing.pallet_runtime import _ControllerLike


PROJECT_ROOT = Path(__file__).resolve().parent

PLACING_STAGE_ORDER = (
    "preflight",
    "authorize",
    "initialize",
    "ready",
    "acquire_perceive_error_align",
    "place_alignment_stop",
    "place",
    "ack_release",
    "retreat",
    "teardown",
)


def authorize_slot_operation(slot: int, mode: str):
    """Return the pure authority verdict for one independently owned slot.

    The import stays lazy so invoking this file directly can install the local
    ``Common/src`` path in :func:`main` first. Callers must obtain this verdict
    before importing or constructing any live placing runtime object.
    """

    from parcel_pose_common.operation_authority import (
        OperationRequest,
        authorize_operation,
    )

    return authorize_operation(OperationRequest.place(slot, mode=mode))


def _selected_slot(root_config: Mapping[str, Any], slot: int | None) -> int:
    """Resolve only the high-level slot selection without runtime imports."""

    if slot is not None:
        return int(slot)
    if not isinstance(root_config, Mapping):
        raise TypeError("root_config must be a mapping")
    pallet = root_config.get("pallet")
    if not isinstance(pallet, Mapping):
        raise ValueError("root config section pallet must be a mapping")
    return int(pallet.get("default_slot", 1))


def _require_matching_controller_slot(controller: Any, selected_slot: int) -> None:
    """Refuse an injected controller bound to another slot before initialization."""

    if controller is None:
        return
    controller_config = getattr(controller, "config", None)
    controller_slot = getattr(controller_config, "selected_slot", None)
    if controller_slot != selected_slot:
        raise ValueError(
            "injected controller selected_slot "
            f"{controller_slot} does not match requested slot {selected_slot}"
        )


def _live_stage_kwargs(
    *,
    auto_place_slot1: bool,
    controller: Any,
    ensure_slot1_ready: bool,
    execute: bool,
    headless: bool,
    log_jsonl: str | Path | None,
    max_frames: int | None,
    output_mp4: str | Path | None,
    plan: Any,
    robot_address: str,
    robot_power: str,
    root_config: Mapping[str, Any],
    stack: Any,
    state: Any,
    window_name: str,
) -> dict[str, Any]:
    """Arguments shared by the staged session and the compatibility dry run."""

    return {
        "controller": controller,
        "auto_place_slot1": auto_place_slot1,
        "ensure_slot1_ready": ensure_slot1_ready,
        "execute": execute,
        "plan": plan,
        "state": state,
        "stack": stack,
        "headless": headless,
        "log_jsonl": log_jsonl,
        "max_frames": max_frames,
        "output_mp4": output_mp4,
        "robot_address": robot_address,
        "robot_power": robot_power,
        "root_config": root_config,
        "window_name": window_name,
    }


def _run_authorized_slot1_place(
    *,
    stage_kwargs: Mapping[str, Any],
    open_placing_session: Any,
    lifecycle_type: Any,
) -> None:
    """Run ready -> align -> evidence-gated place/retreat -> teardown."""

    lifecycle = None
    try:
        with open_placing_session(**stage_kwargs) as session:
            lifecycle = lifecycle_type(
                controller=session.controller,
                release_alignment=session.release_alignment,
                prepare=session.prepare,
            )
            lifecycle.start()
            alignment = session.align(lifecycle)
            if (
                bool(stage_kwargs["auto_place_slot1"])
                and alignment.handoff_ready
                and alignment.place_ready
                and not alignment.user_cancelled
            ):
                stopped = lifecycle.stop_alignment_for_place()
                placed = lifecycle.execute_place(
                    stopped,
                    descent_plan=alignment.descent_plan,
                    await_release_authorization=session.await_release_authorization,
                )
                lifecycle.execute_retreat(placed)
    finally:
        if lifecycle is not None:
            lifecycle.close()



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
    """Run the standalone post-pick placement sequence for one pallet slot."""
    selected_slot = _selected_slot(root_config, slot)  # preflight
    if execute:
        authorize_slot_operation(selected_slot, "live").require_authorized()
        _require_matching_controller_slot(controller, selected_slot)

    import parcel_pose_placing.pallet_runtime as pallet_runtime

    plan = pallet_runtime.resolve_live_plan(
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
    stack = pallet_runtime.assemble_live_stack(
        auto_place_slot1=auto_place_slot1,
        root_config=root_config,
        selected_slot=plan.selected_slot,
    )
    state = pallet_runtime.initial_run_state(
        plan=plan,
        root_config=root_config,
    )
    stage_kwargs = _live_stage_kwargs(
        auto_place_slot1=auto_place_slot1,
        controller=controller,
        ensure_slot1_ready=ensure_slot1_ready,
        execute=execute,
        headless=headless,
        log_jsonl=log_jsonl,
        max_frames=max_frames,
        output_mp4=output_mp4,
        plan=plan,
        robot_address=robot_address,
        robot_power=robot_power,
        root_config=root_config,
        stack=stack,
        state=state,
        window_name=window_name,
    )

    if execute:
        from parcel_pose_placing.placement_lifecycle import PlacementLifecycleRuntime

        _run_authorized_slot1_place(
            stage_kwargs=stage_kwargs,
            open_placing_session=pallet_runtime.open_placing_session,
            lifecycle_type=PlacementLifecycleRuntime,
        )
    else:
        # Replay/perception-only compatibility remains non-actuating while the
        # staged session is intentionally limited to the sole live slot-1 branch.
        pallet_runtime.align_and_place(**stage_kwargs)

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
