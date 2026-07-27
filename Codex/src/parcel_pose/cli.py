"""Command-line entry points for calibration, recording, replay, and live input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

from .output import dumps_strict


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "d435_rby1_nominal.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parcel-pose",
        description="Perception-only D435 parcel center/yaw tools (no robot commands)",
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    calibrate = subparsers.add_parser(
        "calibrate-plane", help="fit a table plane from an empty-table recording"
    )
    calibrate.add_argument("session_pos", type=Path, nargs="?")
    calibrate.add_argument("--session", dest="session_opt", type=Path)
    calibrate.add_argument("--config", type=Path, default=_default_config_path())
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--stride", type=int, default=4)
    calibrate.add_argument(
        "--robot-state-json",
        type=Path,
        help="optional fixed robot-state/FK override for recordings that omitted it",
    )
    calibrate.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("U0", "V0", "U1", "V1"),
        help="optional half-open raw-depth table ROI; overrides config",
    )
    calibrate.set_defaults(handler=_run_calibrate_plane)

    replay = subparsers.add_parser("replay", help="validate and deterministically replay a session")
    replay.add_argument("session_pos", type=Path, nargs="?")
    replay.add_argument("--session", dest="session_opt", type=Path)
    replay.add_argument("--config", type=Path, default=_default_config_path())
    replay.add_argument("--calibration", type=Path)
    replay.add_argument("--output-jsonl", type=Path)
    replay.add_argument("--jsonl", action="store_true", help="emit per-frame replay records")
    replay.add_argument(
        "--burst-size",
        type=int,
        default=0,
        help="also emit one stationary aggregate after each N frames (0 disables)",
    )
    replay.add_argument("--burst-min-valid", type=int, default=3)
    replay.set_defaults(handler=_run_replay)

    evaluate = subparsers.add_parser(
        "evaluate-video",
        help="measure a recorded session and render a base-pose diagnostic MP4",
    )
    evaluate.add_argument("session_pos", type=Path, nargs="?")
    evaluate.add_argument("--session", dest="session_opt", type=Path)
    evaluate.add_argument("--config", type=Path, default=_default_config_path())
    evaluate.add_argument("--calibration", type=Path, required=True)
    evaluate.add_argument("--output-mp4", type=Path, required=True)
    evaluate.add_argument("--output-summary", type=Path)
    evaluate.add_argument("--output-jsonl", type=Path)
    evaluate.add_argument("--overwrite", action="store_true")
    evaluate.set_defaults(handler=_run_evaluate_video)

    record = subparsers.add_parser("record", help="record authoritative raw D435 RGB-D evidence")
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--session-name")
    record.add_argument("--config", type=Path, default=_default_config_path())
    record.add_argument("--duration-sec", type=float, default=3.0)
    record.add_argument("--max-frames", type=int, default=None)
    record.add_argument("--warmup-frames", type=int, default=30)
    record.add_argument("--overwrite", action="store_true")
    record.add_argument("--annotation", default="{}", help="JSON object with qualitative labels")
    record.add_argument(
        "--robot-state-json",
        type=Path,
        help="optional JSON object with fixed head/torso/base state and T_base_from_head",
    )
    record.set_defaults(handler=_run_record)

    live = subparsers.add_parser("live", help="stream D435 perception input without robot control")
    live.add_argument("--config", type=Path, default=_default_config_path())
    live.add_argument("--calibration", type=Path, required=True)
    live.add_argument("--frames", type=int, default=5)
    live.add_argument("--warmup-frames", type=int, default=30)
    live.add_argument("--burst-size", type=int, default=5)
    live.add_argument("--burst-min-valid", type=int, default=3)
    live.set_defaults(handler=_run_live)

    live_view = subparsers.add_parser(
        "live-view",
        help="show live D435 top edges and base x/y/z/yaw/latency",
    )
    live_view.add_argument("--config", type=Path, default=_default_config_path())
    live_view.add_argument("--calibration", type=Path, required=True)
    live_view.add_argument("--warmup-frames", type=int, default=30)
    live_view.add_argument("--max-frames", type=int)
    live_view.add_argument("--fullscreen", action="store_true")
    live_view.add_argument("--window-name", default="RB-Y1 Parcel Pose")
    live_view.set_defaults(handler=_run_live_view)
    return parser


def _resolve_session_arg(args: argparse.Namespace) -> Path:
    positional = getattr(args, "session_pos", None)
    optional = getattr(args, "session_opt", None)
    if positional is not None and optional is not None:
        raise ValueError("provide the session either positionally or with --session, not both")
    if positional is None and optional is None:
        raise ValueError("a recording session is required (SESSION or --session PATH)")
    return positional if positional is not None else optional


def _estimator_config(config: dict[str, Any]):
    from .models import BoxModel, EstimatorConfig

    values = dict(config.get("estimator", {}))
    values["box_model"] = BoxModel.from_dict(config.get("box_model_m", {}))
    return EstimatorConfig.from_dict(values)


def _validate_burst_args(burst_size: int, min_valid: int) -> None:
    if burst_size < 0:
        raise ValueError("--burst-size cannot be negative")
    if min_valid < 1:
        raise ValueError("--burst-min-valid must be positive")
    if burst_size and min_valid > burst_size:
        raise ValueError("--burst-min-valid cannot exceed --burst-size")


def _pose_records_with_bursts(
    estimates: Sequence[Any],
    *,
    burst_size: int,
    min_valid: int,
) -> list[dict[str, Any]]:
    from .burst import aggregate_pose_burst
    from .output import pose_estimate_to_dict

    _validate_burst_args(burst_size, min_valid)
    if burst_size == 0:
        return [pose_estimate_to_dict(estimate) for estimate in estimates]
    records: list[dict[str, Any]] = []
    window: list[Any] = []
    for estimate in estimates:
        single = pose_estimate_to_dict(estimate)
        single["result_kind"] = "single_frame"
        records.append(single)
        window.append(estimate)
        if len(window) == burst_size:
            aggregate = pose_estimate_to_dict(
                aggregate_pose_burst(window, min_valid_frames=min_valid)
            )
            aggregate["result_kind"] = "stationary_burst"
            records.append(aggregate)
            window.clear()
    return records


def _run_calibrate_plane(args: argparse.Namespace) -> int:
    from .calibration import calibrate_table_plane_from_session, load_json, save_calibration

    config = load_json(args.config)
    robot_state = (
        None if args.robot_state_json is None else load_json(args.robot_state_json)
    )
    session = _resolve_session_arg(args)
    calibration = calibrate_table_plane_from_session(
        session,
        config,
        stride=args.stride,
        roi_uv=None if args.roi is None else tuple(args.roi),
        robot_state_override=robot_state,
    )
    save_calibration(args.output, calibration)
    print(dumps_strict({"calibration": calibration.to_dict(), "output": args.output}, indent=2))
    return 0


def _run_replay(args: argparse.Namespace) -> int:
    from .calibration import load_calibration, load_json
    from .estimator import ParcelPoseEstimator
    from .recording import SessionReader, recording_summary, replay_session

    session = _resolve_session_arg(args)
    if args.calibration is None:
        records = replay_session(session) if args.jsonl or args.output_jsonl else None
        if records is None:
            print(dumps_strict(recording_summary(session), indent=2))
            return 0
    else:
        config = load_json(args.config)
        calibration = load_calibration(args.calibration)
        reader = SessionReader(session)
        estimator = ParcelPoseEstimator(
            reader.metadata.depth_profile.intrinsics,
            calibration,
            _estimator_config(config),
        )
        estimates = [
            estimator.estimate(
                    frame.raw_depth_z16,
                    depth_scale=reader.metadata.depth_scale_m,
                    timestamp_ms=frame.depth_timestamp_ms,
                    frame_id=frame.depth_frame_number,
            )
            for frame in reader
        ]
        records = _pose_records_with_bursts(
            estimates,
            burst_size=args.burst_size,
            min_valid=args.burst_min_valid,
        )
    lines = [dumps_strict(record) for record in records]
    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.output_jsonl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(
            dumps_strict(
                {"output": args.output_jsonl, "frame_count": len(lines), "status": "replayed"},
                indent=2,
            )
        )
    else:
        for line in lines:
            print(line)
    return 0


def _run_evaluate_video(args: argparse.Namespace) -> int:
    from .calibration import load_calibration, load_json
    from .evaluation import evaluate_session_video

    session = _resolve_session_arg(args)
    summary = evaluate_session_video(
        session,
        load_calibration(args.calibration),
        _estimator_config(load_json(args.config)),
        args.output_mp4,
        output_summary=args.output_summary,
        output_jsonl=args.output_jsonl,
        overwrite=args.overwrite,
    )
    print(dumps_strict(summary, indent=2))
    return 0


def _recording_context(
    config: dict[str, Any],
    annotation: dict[str, Any],
    robot_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frames = config["frames"]
    nominal = config["calibration"]["nominal_T_head_from_color"]
    default_robot_state = {
        "head_joints": None,
        "torso_joints": None,
        "base_state": None,
        "T_base_from_head": None,
    }
    if robot_state is not None:
        default_robot_state.update(robot_state)
    return {
        "robot_state": default_robot_state,
        "nominal_transform": {
            "target_frame": frames["head_candidate"],
            "source_frame": frames["color"],
            "translation_m": nominal["translation_m"],
            "euler_zyx_deg": nominal["euler_zyx_deg"],
            "euler_input_order": nominal["euler_input_order"],
            "rotation_formula": nominal["rotation_formula"],
            "origin_status": nominal["origin_status"],
        },
        "table": {"plane": None, "config_schema_version": 1},
        "annotation": annotation,
    }


def _run_record(args: argparse.Namespace) -> int:
    from .calibration import load_json
    from .realsense_adapter import (
        D435StreamConfig,
        RealSenseAdapter,
        RealSenseUnavailableError,
    )
    from .recording import SessionWriter

    try:
        annotation = json.loads(args.annotation)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--annotation must be a JSON object: {exc}") from exc
    if not isinstance(annotation, dict):
        raise ValueError("--annotation must be a JSON object")
    config = load_json(args.config)
    robot_state: dict[str, Any] | None = None
    if args.robot_state_json is not None:
        robot_state = load_json(args.robot_state_json)
    context = _recording_context(config, annotation, robot_state)
    output = args.output
    if args.session_name is not None:
        session_name = args.session_name.strip()
        if not session_name or Path(session_name).name != session_name or session_name in {".", ".."}:
            raise ValueError("--session-name must be one safe path component")
        output = output / session_name
    stream_config = D435StreamConfig(warmup_frames=args.warmup_frames)
    try:
        with RealSenseAdapter(stream_config) as camera:
            metadata = camera.session_metadata(**context)
            with SessionWriter(output, metadata, overwrite=args.overwrite) as writer:
                deadline = time.monotonic() + args.duration_sec
                while time.monotonic() < deadline:
                    writer.add_frame(camera.capture())
                    if args.max_frames is not None and writer.frame_count >= args.max_frames:
                        break
    except RealSenseUnavailableError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(dumps_strict({"output": output, "status": "recorded"}, indent=2))
    return 0


def _run_live(args: argparse.Namespace) -> int:
    from .calibration import load_calibration, load_json
    from .estimator import ParcelPoseEstimator
    from .output import dumps_strict, pose_estimate_to_dict
    from .realsense_adapter import (
        D435StreamConfig,
        RealSenseAdapter,
        RealSenseUnavailableError,
    )

    config = load_json(args.config)
    calibration = load_calibration(args.calibration)
    context = _recording_context(config, {})
    _validate_burst_args(args.burst_size, args.burst_min_valid)
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    try:
        with RealSenseAdapter(D435StreamConfig(warmup_frames=args.warmup_frames)) as camera:
            metadata = camera.session_metadata(**context)
            estimator = ParcelPoseEstimator(
                metadata.depth_profile.intrinsics,
                calibration,
                _estimator_config(config),
            )
            window: list[Any] = []
            for _ in range(args.frames):
                frame = camera.capture()
                estimate = estimator.estimate(
                    frame.raw_depth_z16,
                    depth_scale=metadata.depth_scale_m,
                    timestamp_ms=frame.depth_timestamp_ms,
                    frame_id=frame.depth_frame_number,
                )
                if args.burst_size:
                    single = pose_estimate_to_dict(estimate)
                    single["result_kind"] = "single_frame"
                    print(dumps_strict(single))
                    window.append(estimate)
                    if len(window) == args.burst_size:
                        aggregate = _pose_records_with_bursts(
                            window,
                            burst_size=args.burst_size,
                            min_valid=args.burst_min_valid,
                        )[-1]
                        print(dumps_strict(aggregate))
                        window.clear()
                else:
                    print(dumps_strict(pose_estimate_to_dict(estimate)))
    except RealSenseUnavailableError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def _run_live_view(args: argparse.Namespace) -> int:
    from .calibration import load_calibration, load_json
    from .realsense_adapter import RealSenseUnavailableError
    from .realtime import LiveViewUnavailableError, run_live_view

    config = load_json(args.config)
    calibration = load_calibration(args.calibration)
    if not calibration.absolute_base_validated:
        print(
            "warning: displayed base coordinates use nominal_unverified camera "
            "registration; validate the camera-to-base transform before robot use",
            file=sys.stderr,
        )
    try:
        run_live_view(
            calibration,
            _estimator_config(config),
            _recording_context(config, {}),
            warmup_frames=args.warmup_frames,
            max_frames=args.max_frames,
            fullscreen=args.fullscreen,
            window_name=args.window_name,
        )
    except (LiveViewUnavailableError, RealSenseUnavailableError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    try:
        return int(args.handler(args))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
