"""Explicit RB-Y1 pallet placing orchestration.

``place_box`` owns slot preflight, authority, initialization, readiness, both
frame loops, every continue/exit decision, evidence-gated place, release
observation, retreat, and teardown order.  Read ``_run_placing_flow`` from top
to bottom to see the complete run:

    acquire_frame → perceive_frame → decide_base_motion
    → advance_placement → record_frame → choose continue/exit

Lower services still own the mechanisms that must not be reimplemented here:
D435/SDK resources, estimator internals, servo and safety gates, exact-zero and
wheel-stop proof, command construction/acknowledgement, containment, telemetry,
and display rendering.
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
    "alignment_acquire",
    "alignment_perceive",
    "alignment_decide_x_y_yaw",
    "alignment_advance_placement",
    "alignment_record",
    "alignment_loop_exit",
    "stop_alignment",
    "place",
    "release_acquire",
    "release_perceive",
    "release_decide_x_y_yaw",
    "release_advance_placement",
    "release_record",
    "release_authorized",
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


def _run_placing_flow(
    *,
    stage_kwargs: Mapping[str, Any],
    open_placing_session: Any,
    lifecycle_type: Any | None,
) -> None:
    """Run every frame/stage transition visibly in this entrypoint."""

    from parcel_pose_placing.pallet_place import PlacementRequest

    lifecycle = None
    try:
        with open_placing_session(**stage_kwargs) as session:
            if lifecycle_type is not None:
                lifecycle = lifecycle_type(
                    controller=session.controller,
                    release_alignment=session.release_alignment,
                    prepare=session.prepare,
                )
                lifecycle.start()  # ready + lower-owned stream preparation
            else:
                session.prepare()  # perception-only no-op preparation

            session.open_acquisition()
            descent_plan = None

            # Alignment loop: this file owns frame budget and every exit branch.
            while session.has_frame_budget():
                try:
                    frame = session.acquire_frame()  # acquire exactly one
                except KeyboardInterrupt:
                    session.handle_interrupt()
                    break

                perceived = session.perceive_frame(frame)  # one facade call
                base_motion = session.decide_base_motion(perceived)  # x/y/yaw once
                placement_step = session.advance_placement(  # one sequencer step
                    perceived,
                    base_motion,
                )
                session.record_frame(  # telemetry/overlay only
                    perceived,
                    base_motion,
                    placement_step,
                )
                session.finish_frame()

                placement = placement_step.placement_output
                if session.user_cancelled:
                    break
                if placement is not None and placement.faulted:
                    raise RuntimeError(
                        f"slot-1 placement sequencer faulted: {placement.reason}"
                    )
                if (
                    bool(stage_kwargs["auto_place_slot1"])
                    and placement is not None
                    and placement.request
                    is PlacementRequest.LOWER_CARTESIAN_PLANNED
                ):
                    descent_plan = placement.descent_plan
                    if descent_plan is None or not descent_plan.valid:
                        raise RuntimeError(
                            "placement sequencer returned no valid descent plan"
                        )
                    session.accept_descent_plan(descent_plan)
                    break

            if (
                lifecycle is not None
                and descent_plan is not None
                and not session.user_cancelled
            ):
                stopped = lifecycle.stop_alignment_for_place()

                def await_release_authorization() -> bool:
                    """Post-place loop remains visible next to place/retreat."""

                    session.begin_release_observation()
                    while session.has_frame_budget():
                        try:
                            frame = session.acquire_frame()  # acquire exactly one
                        except KeyboardInterrupt:
                            session.handle_interrupt()
                            return False

                        perceived = session.perceive_frame(frame)  # perceive once
                        base_motion = session.decide_base_motion(perceived)
                        placement_step = session.advance_placement(
                            perceived,
                            base_motion,
                        )
                        session.record_frame(
                            perceived,
                            base_motion,
                            placement_step,
                        )
                        session.finish_frame()

                        placement = placement_step.placement_output
                        if session.user_cancelled:
                            return False
                        if placement is None:
                            continue
                        if placement.faulted:
                            raise RuntimeError(
                                "slot-1 release authorization faulted: "
                                f"{placement.reason}"
                            )
                        if (
                            placement.request is PlacementRequest.SPREAD_RELEASE
                            and placement.release_authorized
                        ):
                            return True
                    return False

                placed = lifecycle.execute_place(
                    stopped,
                    descent_plan=descent_plan,
                    await_release_authorization=await_release_authorization,
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

    lifecycle_type = None
    if execute:
        from parcel_pose_placing.placement_lifecycle import PlacementLifecycleRuntime

        lifecycle_type = PlacementLifecycleRuntime

    _run_placing_flow(
        stage_kwargs=stage_kwargs,
        open_placing_session=pallet_runtime.open_placing_session,
        lifecycle_type=lifecycle_type,
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
