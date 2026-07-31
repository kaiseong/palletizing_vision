"""Explicit RB-Y1 parcel-picking orchestration.

``_run_authorized_horizontal_pick`` visibly owns the complete authorized run:
ready, acquire one frame, perceive once, make one x/y/yaw decision, record,
choose loop continuation or exit, close acquisition, stop/release alignment,
grasp/lift, and teardown.

Lower services retain D435/SDK resources, estimator internals, servo and safety
algorithms, mobility-stream evidence, command construction, telemetry, overlay,
and idempotent robot cleanup.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent

# Migrated unchanged from ServoConfig. XY arrival is one Euclidean radius;
# independent x/y tolerances do not exist in the demonstrated horizontal path.
HORIZONTAL_PICK_TARGET_XY_M = (0.740, 0.0)
HORIZONTAL_PICK_TARGET_LINE_YAW_RAD = math.pi / 2.0
HORIZONTAL_PICK_ARRIVAL_RADIUS_M = 0.010
HORIZONTAL_PICK_ARRIVAL_YAW_RAD = math.radians(3.0)

PICKING_STAGE_ORDER = (
    "preflight",
    "authorize",
    "initialize",
    "ready",
    "acquire",
    "perceive",
    "decide_x_y_yaw",
    "record",
    "loop_exit",
    "stop_release_alignment",
    "grasp_lift",
    "teardown",
)


class _PickExecutionError(RuntimeError):
    """Expected live-service failure reported by the operator entrypoint."""


class _PreparedHorizontalPick:
    """Prepared collaborators kept dependency-neutral at entrypoint scope."""

    __slots__ = (
        "automation",
        "calibration",
        "estimator_config",
        "metadata_context",
        "open_alignment_session",
        "plan",
        "service_errors",
    )

    def __init__(
        self,
        *,
        automation: Any,
        calibration: Any,
        estimator_config: Any,
        metadata_context: Any,
        open_alignment_session: Any,
        plan: Any,
        service_errors: tuple[type[BaseException], ...],
    ) -> None:
        self.automation = automation
        self.calibration = calibration
        self.estimator_config = estimator_config
        self.metadata_context = metadata_context
        self.open_alignment_session = open_alignment_session
        self.plan = plan
        self.service_errors = service_errors


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


def _ensure_source_tree_imports() -> None:
    repo_root = PROJECT_ROOT.parent
    for source_root in (repo_root / "Common" / "src", PROJECT_ROOT / "src"):
        source_entry = str(source_root)
        if source_entry not in sys.path:
            sys.path.insert(0, source_entry)


def _horizontal_auto_grab_config(
    args: Any,
    *,
    auto_grab_config_type: Any,
    servo_config_type: Any,
) -> Any:
    """Resolve entrypoint values while retaining lower safety/tuning defaults."""
    servo = servo_config_type(
        target_xy_m=HORIZONTAL_PICK_TARGET_XY_M,
        target_long_axis_yaw_rad=HORIZONTAL_PICK_TARGET_LINE_YAW_RAD,
        arrival_inner_m=HORIZONTAL_PICK_ARRIVAL_RADIUS_M,
        arrival_yaw_inner_rad=HORIZONTAL_PICK_ARRIVAL_YAW_RAD,
    )
    return auto_grab_config_type(
        address=args.robot_address,
        power=args.robot_power,
        servo=servo,
    )


def _prepare_horizontal_pick(args: Any) -> _PreparedHorizontalPick:
    """Initialize pure plan/config first, then the authorized servo runtime."""
    from parcel_pose_common.calibration import load_calibration, load_json
    from parcel_pose_common.mobile_servo import ServoConfig
    from parcel_pose_common.realsense_adapter import RealSenseUnavailableError
    from parcel_pose_picking.auto_grab import AutoGrabConfig, AutoGrabError, AutoGrabRuntime
    from parcel_pose_picking.cli import _estimator_config, _recording_context
    from parcel_pose_picking.realtime import (
        LiveViewUnavailableError,
        open_alignment_session,
        resolve_live_view_plan,
    )

    try:
        config = load_json(args.config)
        calibration = load_calibration(args.calibration)
        plan = resolve_live_view_plan(
            calibration=calibration,
            fullscreen=args.fullscreen,
            headless=args.headless,
            log_jsonl=args.log_jsonl,
            max_frames=args.max_frames,
            output_mp4=args.output_mp4,
            warmup_frames=args.warmup_frames,
            window_name=args.window_name,
        )
        estimator_config = _estimator_config(config)
        metadata_context = _recording_context(config, {})
        automation = AutoGrabRuntime(
            _horizontal_auto_grab_config(
                args,
                auto_grab_config_type=AutoGrabConfig,
                servo_config_type=ServoConfig,
            ),
            execute=True,
        )
    except (LiveViewUnavailableError, RealSenseUnavailableError, AutoGrabError) as exc:
        raise _PickExecutionError(str(exc)) from exc

    if not calibration.absolute_base_validated:
        print(
            "warning: base coordinates use nominal_unverified camera registration "
            "with an empirical +0.050 m y correction; automatic RB-Y1 motion "
            "is enabled by the box_picking entrypoint",
            file=sys.stderr,
        )
    return _PreparedHorizontalPick(
        automation=automation,
        calibration=calibration,
        estimator_config=estimator_config,
        metadata_context=metadata_context,
        open_alignment_session=open_alignment_session,
        plan=plan,
        service_errors=(
            LiveViewUnavailableError,
            RealSenseUnavailableError,
            AutoGrabError,
        ),
    )


def _alignment_session_kwargs(
    prepared: _PreparedHorizontalPick,
    args: Any,
) -> dict[str, Any]:
    plan = prepared.plan
    return {
        "plan": plan,
        "calibration": prepared.calibration,
        "estimator_config": prepared.estimator_config,
        "metadata_context": prepared.metadata_context,
        "fullscreen": args.fullscreen,
        "headless": args.headless,
        "max_frames": args.max_frames,
        "window_name": args.window_name,
        "handoff_ready": plan.handoff_ready,
        "processed_frames": plan.processed_frames,
        "user_cancelled": plan.user_cancelled,
        "log_stream": plan.log_stream,
        "video_writer": plan.video_writer,
        "window_created": plan.window_created,
    }


def _run_authorized_horizontal_pick(args: Any) -> int:
    """Run the complete frame and manipulation orchestration in this entrypoint."""

    prepared = _prepare_horizontal_pick(args)
    automation = prepared.automation
    try:
        try:
            # Camera/profile resources are lower-owned, but this entrypoint owns
            # every frame iteration and every continue/exit decision.
            with prepared.open_alignment_session(
                **_alignment_session_kwargs(prepared, args)
            ) as session:
                automation.start()  # ready
                try:
                    while session.has_frame_budget():
                        frame = session.acquire_frame()  # acquire exactly one
                        observation = session.perceive_frame(frame)  # perceive once
                        handoff_ready = automation.update(  # decide x/y/yaw once
                            observation.base_pose,
                            pose_timestamp_s=observation.pose_result.timestamp_s,
                            now_s=time.monotonic(),
                        )
                        session.record_frame(  # telemetry/display, never loop policy
                            observation,
                            handoff_ready=handoff_ready,
                        )

                        if session.user_cancelled:
                            break
                        if handoff_ready:
                            break
                except KeyboardInterrupt:
                    session.cancel()
                outcome = session.outcome()

            # Camera acquisition is closed before ownership transfers to grasp.
            if outcome.handoff_ready and not outcome.user_cancelled:
                evidence = automation.stop_alignment_for_grasp()
                automation.execute_grasp(evidence)  # grasp/lift
        finally:
            automation.close()  # teardown is lower-service-owned and idempotent
    except prepared.service_errors as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def pick_box(args: Any) -> int:
    """Preflight and authorize one branch before constructing live services."""
    _ensure_source_tree_imports()
    from parcel_pose_common.operation_authority import (
        OperationMode,
        OperationRequest,
        authorize_operation,
    )

    request = OperationRequest.pick(args.orientation, mode=OperationMode.LIVE)
    verdict = authorize_operation(request)
    if not verdict.allowed:
        print(verdict.reason, file=sys.stderr)
        return 2
    verdict.require_authorized()
    try:
        return _run_authorized_horizontal_pick(args)
    except _PickExecutionError as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _run_box_picking(argv: Sequence[str]) -> int:
    _ensure_source_tree_imports()
    from parcel_pose_picking.cli import build_parser, run_handler

    parser = build_parser()
    args = parser.parse_args(argv)
    args.pick_box = pick_box
    return int(run_handler(parser, args))


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        _reexec_active_conda_python()
    user_args = sys.argv[1:] if argv is None else list(argv)
    return _run_box_picking(user_args)


if __name__ == "__main__":
    raise SystemExit(main())
