"""Shared command-line implementation for authoritative D435 recordings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

from .calibration import load_json
from .models import BoxDimensionPrior, BoxModel
from .output import dumps_strict
from .realsense_adapter import (
    D435StreamConfig,
    RealSenseAdapter,
    RealSenseUnavailableError,
)
from .recording import SessionWriter


def add_record_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_config: Path | None = None,
) -> None:
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-name")
    parser.add_argument("--config", type=Path, default=default_config, required=default_config is None)
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--annotation", default="{}", help="JSON object with qualitative labels")
    parser.add_argument(
        "--robot-state-json",
        type=Path,
        help="optional JSON object with fixed head/torso/base state and T_base_from_head",
    )


def build_record_parser(*, default_config: Path | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="record",
        description="record authoritative raw D435 RGB-D evidence",
    )
    add_record_arguments(parser, default_config=default_config)
    parser.set_defaults(handler=run_record)
    return parser


def recording_context(
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
        "box_model": BoxModel.from_dict(config.get("box_model_m", {})),
        "box_dimension_prior": (
            None
            if config.get("box_dimension_prior_m") is None
            else BoxDimensionPrior.from_dict(config["box_dimension_prior_m"])
        ),
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


def run_record(args: argparse.Namespace) -> int:
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
    context = recording_context(config, annotation, robot_state)
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


def main(argv: Sequence[str] | None = None, *, default_config: Path | None = None) -> int:
    parser = build_record_parser(default_config=default_config)
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    try:
        return int(args.handler(args))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 2
