"""One-command box-picking facade for the RB-Y1 parcel workflow.

The sequence lives in pick_box below:

    config      = load_json(...)              # picking_config.json
    calibration = load_calibration(...)       # picking_calibration.json
    automation  = AutoGrabRuntime(...)        # owns the grab motion
    plan        = resolve_live_view_plan(...) # refuse a bad request first
                  watch_and_grab(...)         # camera, estimate, grab

The grab fires when the box centre is inside the tolerance around the target x,y in
base coordinates.  To change the grasp, change the posture constants in auto_grab;
to change when it fires, change AutoGrabRuntime.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent


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


def pick_box(args: Any) -> int:
    """Pick one box: this is the whole sequence."""

    _ensure_source_tree_imports()
    from parcel_pose_common.calibration import load_calibration, load_json
    from parcel_pose_common.realsense_adapter import RealSenseUnavailableError

    from parcel_pose_picking.auto_grab import AutoGrabConfig, AutoGrabError, AutoGrabRuntime
    from parcel_pose_picking.cli import _estimator_config, _recording_context
    from parcel_pose_picking.realtime import (
        LiveViewUnavailableError,
        resolve_live_view_plan,
        watch_and_grab,
    )

    config = load_json(args.config)
    calibration = load_calibration(args.calibration)
    automation = AutoGrabRuntime(
        AutoGrabConfig(
            address=args.robot_address,
            power=args.robot_power,
        ),
        execute=True,
    )
    if not calibration.absolute_base_validated:
        print(
            "warning: base coordinates use nominal_unverified camera registration "
            "with an empirical +0.050 m y correction; automatic RB-Y1 motion "
            "is enabled by the box_picking entrypoint",
            file=sys.stderr,
        )
    try:
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
        watch_and_grab(
            plan=plan,
            automation=automation,
            calibration=calibration,
            estimator_config=_estimator_config(config),
            metadata_context=_recording_context(config, {}),
            fullscreen=args.fullscreen,
            handoff_ready=plan.handoff_ready,
            handoff_started=plan.handoff_started,
            headless=args.headless,
            log_stream=plan.log_stream,
            max_frames=args.max_frames,
            processed_frames=plan.processed_frames,
            user_cancelled=plan.user_cancelled,
            video_writer=plan.video_writer,
            window_created=plan.window_created,
            window_name=args.window_name,
        )
    except (LiveViewUnavailableError, RealSenseUnavailableError, AutoGrabError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def _run_box_picking(argv: Sequence[str]) -> int:
    _ensure_source_tree_imports()
    from parcel_pose_picking.cli import build_parser, run_handler

    parser = build_parser()
    args = parser.parse_args(argv)
    # The picking sequence lives in this file, so hand it to the handler.
    args.pick_box = pick_box
    return int(run_handler(parser, args))


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        _reexec_active_conda_python()
    user_args = sys.argv[1:] if argv is None else list(argv)
    return _run_box_picking(user_args)


if __name__ == "__main__":
    raise SystemExit(main())
