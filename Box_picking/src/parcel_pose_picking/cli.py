"""Command-line entry point for the automatic RB-Y1 box-picking flow."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Sequence

from parcel_pose_common.output import dumps_strict


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_config_path() -> Path:
    return _repo_root() / "Box_picking" / "configs" / "picking_config.json"


def _default_calibration_path() -> Path:
    return _repo_root() / "Box_picking" / "configs" / "picking_calibration.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="box_picking",
        description="Run the D435-guided RB-Y1 box-picking sequence",
    )
    parser.add_argument("--config", type=Path, default=_default_config_path())
    parser.add_argument("--calibration", type=Path, default=_default_calibration_path())
    parser.add_argument(
        "--orientation",
        choices=("horizontal", "vertical"),
        default="horizontal",
        help=(
            "box orientation branch to request; vertical currently fails closed "
            "until its perception and demonstrated poses are complete"
        ),
    )
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run capture, visual servo, and auto-grab without an OpenCV window",
    )
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--window-name", default="RB-Y1 Parcel Pose")
    parser.add_argument(
        "--output-mp4",
        type=Path,
        help="optional annotated live video (the path must not already exist)",
    )
    parser.add_argument(
        "--log-jsonl",
        type=Path,
        help="optional per-frame pose telemetry (the path must not already exist)",
    )
    parser.add_argument(
        "--robot-address",
        default="192.168.30.1:50051",
        help="RB-Y1 controller address (default: 192.168.30.1:50051)",
    )
    parser.add_argument(
        "--robot-power",
        default=".*",
        help="power-device regex used while preparing the robot (default: .*)",
    )
    parser.set_defaults(handler=_run_box_picking)

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

    replay = subparsers.add_parser(
        "replay", help="validate and deterministically replay a session"
    )
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
    from parcel_pose_common.models import EstimatorConfig

    return EstimatorConfig.from_root_config(config)


def _recording_context(
    config: dict[str, Any],
    annotation: dict[str, Any],
    robot_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from parcel_pose_common.record_cli import recording_context

    return recording_context(config, annotation, robot_state)


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
    from parcel_pose_common.output import pose_estimate_to_dict

    from .burst import aggregate_pose_burst

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
    from parcel_pose_common.calibration import (
        calibrate_table_plane_from_session,
        load_json,
        save_calibration,
    )

    config = load_json(args.config)
    robot_state = None if args.robot_state_json is None else load_json(args.robot_state_json)
    session = _resolve_session_arg(args)
    calibration = calibrate_table_plane_from_session(
        session,
        config,
        stride=args.stride,
        roi_uv=None if args.roi is None else tuple(args.roi),
        robot_state_override=robot_state,
    )
    save_calibration(args.output, calibration)
    print(
        dumps_strict(
            {"calibration": calibration.to_dict(), "output": args.output},
            indent=2,
        )
    )
    return 0


def _run_replay(args: argparse.Namespace) -> int:
    from parcel_pose_common.calibration import load_calibration, load_json
    from parcel_pose_common.recording import SessionReader, recording_summary, replay_session

    from .estimator import ParcelPoseEstimator

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
        args.output_jsonl.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        print(
            dumps_strict(
                {
                    "output": args.output_jsonl,
                    "frame_count": len(lines),
                    "status": "replayed",
                },
                indent=2,
            )
        )
    else:
        for line in lines:
            print(line)
    return 0


def _run_evaluate_video(args: argparse.Namespace) -> int:
    from parcel_pose_common.calibration import load_calibration, load_json

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


def _run_box_picking(args: argparse.Namespace) -> int:
    """Run the picking sequence box_picking.py owns.

    The sequence is not here: it is in box_picking.py, which attaches it so the
    dependency points from the entry point into the library and not back.
    """
    pick_box = getattr(args, "pick_box", None)
    if pick_box is None:  # pragma: no cover - box_picking.py always supplies it
        raise RuntimeError(
            "picking runs are owned by box_picking.py; run it rather than this module"
        )
    return int(pick_box(args))


def run_handler(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Run one parsed subcommand and turn faults into the right exit code."""
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("interrupted by user", file=sys.stderr)
        return 130
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_handler(parser, args)


if __name__ == "__main__":
    raise SystemExit(main())
