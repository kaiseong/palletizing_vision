"""CLI for metric pallet-stack replay and supervised slot-1 control."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Sequence

from .output import dumps_strict


def _default_config_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "rby1m_v1_2_pallet_slot1_nominal.json"
    )


def _default_artifact_root() -> Path:
    return Path(__file__).resolve().parents[3] / "out"


def _artifact_session_name(session: Path) -> str:
    value = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in session.name
    ).strip("-")
    return value or "session"


def _new_replay_run_dir(session: Path) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    stem = f"{timestamp}-slot1-acquire-{_artifact_session_name(session)}"
    root = _default_artifact_root()
    candidate = root / stem
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stem}-{suffix:02d}"
        suffix += 1
    return candidate


def _resolve_session(args: argparse.Namespace) -> Path:
    positional = getattr(args, "session_pos", None)
    optional = getattr(args, "session_opt", None)
    if positional is not None and optional is not None:
        raise ValueError("provide the session positionally or with --session, not both")
    session = positional if positional is not None else optional
    if session is None:
        raise ValueError("a recording session is required")
    return Path(session)


def _control_error_types() -> tuple[type[BaseException], ...]:
    """Import the controller error base lazily; the SDK stays untouched."""

    from .pallet_control import PalletControlError

    return (PalletControlError,)


def _load_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load pallet config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("pallet config root must be a JSON object")
    return value


def _add_single_session(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("session_pos", type=Path, nargs="?")
    parser.add_argument("--session", dest="session_opt", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pallet",
        description=(
            "D435 metric pallet-stack tools and supervised RB-Y1 slot-1 "
            "alignment/placement"
        ),
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    replay = subparsers.add_parser(
        "replay",
        help="run metric stack/opening estimation on one recorded session",
    )
    _add_single_session(replay)
    replay.add_argument("--config", type=Path, default=_default_config_path())
    replay.add_argument("--output-mp4", type=Path)
    replay.add_argument("--output-summary", type=Path)
    replay.add_argument("--output-jsonl", type=Path)
    replay.add_argument("--output-config-snapshot", type=Path)
    replay.add_argument(
        "--output-run-dir",
        type=Path,
        help=(
            "write replay.mp4, summary.json, telemetry.jsonl, and "
            "config_snapshot.json here; explicit per-file paths take precedence"
        ),
    )
    replay.add_argument(
        "--no-default-artifacts",
        action="store_true",
        help="print the summary only when no explicit output path is provided",
    )
    replay.add_argument("--max-frames", type=int)
    replay.add_argument("--overwrite", action="store_true")
    replay.set_defaults(handler=_run_replay)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="evaluate one or more sessions and report temporal acceptance metrics",
    )
    evaluate.add_argument("--session", type=Path, action="append", required=True)
    evaluate.add_argument("--config", type=Path, default=_default_config_path())
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--output-dir", type=Path)
    evaluate.add_argument("--render-mp4", action="store_true")
    evaluate.add_argument("--overwrite", action="store_true")
    evaluate.set_defaults(handler=_run_evaluate)

    live = subparsers.add_parser(
        "live",
        help=(
            "run D435 perception and optionally align a held box to the slot-1 "
            "ready position"
        ),
    )
    live.add_argument("--config", type=Path, default=_default_config_path())
    live.add_argument("--warmup-frames", type=int, default=30)
    live.add_argument(
        "--max-frames",
        type=int,
        help="bound perception-only runs; rejected for loaded robot execution",
    )
    live.add_argument("--headless", action="store_true")
    live.add_argument("--window-name", default="RB-Y1 Pallet Slot-1")
    live.add_argument("--output-mp4", type=Path)
    live.add_argument("--log-jsonl", type=Path)
    live.add_argument(
        "--ensure-slot1-ready",
        action="store_true",
        help=(
            "connect to RB-Y1 and, only when needed, send one 5-second "
            "all-joint Position command to the configured slot-1 ready pose "
            "before opening the camera"
        ),
    )
    live.add_argument(
        "--auto-palletize-slot1",
        action="store_true",
        help=(
            "start the supervised loaded-box slot-1 base alignment; requires "
            "--ensure-slot1-ready and the explicit calibration/safety acknowledgements"
        ),
    )
    live.add_argument(
        "--auto-place-slot1",
        action="store_true",
        help=(
            "after slot-1 ARRIVED_HOLD, freeze a metric descent plan, lower both "
            "Cartesian EEFs, and run the optional release sequence; requires "
            "--auto-palletize-slot1 and placement.enabled=true in the config"
        ),
    )
    live.add_argument(
        "--allow-nominal-registration",
        action="store_true",
        help="explicitly accept the nominal_unverified camera/base registration",
    )
    live.add_argument(
        "--allow-geometry-only-grip-check",
        action="store_true",
        help=(
            "commissioning-only: acknowledge fixed-ready FK/EEF geometry, measured "
            "joint continuity, and depth clearance as the loaded-demo interlock; "
            "F/T is not a gate; "
            "the loaded config must enable the same reviewed commissioning policy"
        ),
    )
    live.add_argument(
        "--allow-vision-geometry-release",
        action="store_true",
        help=(
            "commissioning-only: allow release when bilateral EEF/FK box-bottom "
            "and metric stack-plane bounds produce a valid frozen descent plan; "
            "placement does not read F/T"
        ),
    )
    live.add_argument(
        "--robot-address",
        default="192.168.30.1:50051",
        help="RB-Y1 controller address",
    )
    live.add_argument(
        "--robot-power",
        default=".*",
        help="power-device regex used during controller preparation",
    )
    live.set_defaults(handler=_run_live)
    return parser


def _run_replay(args: argparse.Namespace) -> int:
    from .pallet_evaluation import evaluate_pallet_session

    session = _resolve_session(args)
    explicit_paths = any(
        value is not None
        for value in (
            args.output_mp4,
            args.output_summary,
            args.output_jsonl,
            args.output_config_snapshot,
        )
    )
    run_dir = args.output_run_dir
    if run_dir is None and not explicit_paths and not args.no_default_artifacts:
        run_dir = _new_replay_run_dir(session)
    output_mp4 = args.output_mp4
    output_summary = args.output_summary
    output_jsonl = args.output_jsonl
    output_config_snapshot = args.output_config_snapshot
    if run_dir is not None:
        output_mp4 = output_mp4 or run_dir / "replay.mp4"
        output_summary = output_summary or run_dir / "summary.json"
        output_jsonl = output_jsonl or run_dir / "telemetry.jsonl"
        output_config_snapshot = (
            output_config_snapshot or run_dir / "config_snapshot.json"
        )
    summary = evaluate_pallet_session(
        session,
        _load_config(args.config),
        output_mp4=output_mp4,
        output_summary=output_summary,
        output_jsonl=output_jsonl,
        output_config_snapshot=output_config_snapshot,
        overwrite=bool(args.overwrite),
        max_frames=args.max_frames,
    )
    print(dumps_strict(summary, indent=2))
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    from .pallet_evaluation import evaluate_pallet_session

    config = _load_config(args.config)
    if args.output is not None and len(args.session) != 1:
        raise ValueError("--output may be used only with one --session")
    summaries: list[dict[str, Any]] = []
    for session in args.session:
        output_mp4 = None
        output_summary = None
        if args.output_dir is not None:
            output_summary = args.output_dir / f"{session.name}.json"
            if args.render_mp4:
                output_mp4 = args.output_dir / f"{session.name}.mp4"
        elif args.output is not None:
            output_summary = args.output
        summaries.append(
            evaluate_pallet_session(
                session,
                config,
                output_mp4=output_mp4,
                output_summary=output_summary,
                overwrite=bool(args.overwrite),
            )
        )
    payload: dict[str, Any] = {"sessions": summaries, "session_count": len(summaries)}
    print(dumps_strict(payload, indent=2))
    return 0


def _run_live(args: argparse.Namespace) -> int:
    if args.allow_nominal_registration and not args.auto_palletize_slot1:
        raise ValueError("--allow-nominal-registration requires --auto-palletize-slot1")
    if args.allow_geometry_only_grip_check and not args.auto_palletize_slot1:
        raise ValueError(
            "--allow-geometry-only-grip-check requires --auto-palletize-slot1"
        )
    if args.auto_palletize_slot1 and not args.ensure_slot1_ready:
        raise ValueError(
            "loaded slot-1 alignment requires --ensure-slot1-ready so the configured "
            "held-box posture is verified before the combined command stream starts"
        )
    if args.auto_place_slot1 and not args.auto_palletize_slot1:
        raise ValueError("--auto-place-slot1 requires --auto-palletize-slot1")
    if args.allow_vision_geometry_release and not args.auto_place_slot1:
        raise ValueError(
            "--allow-vision-geometry-release requires --auto-place-slot1"
        )
    if args.auto_place_slot1 and not args.allow_vision_geometry_release:
        raise ValueError(
            "--auto-place-slot1 requires --allow-vision-geometry-release; "
            "omit --auto-place-slot1 for alignment-only commissioning"
        )
    if args.ensure_slot1_ready and not args.auto_palletize_slot1:
        print(
            "[pallet] mobile base disabled: --ensure-slot1-ready only verifies or "
            "restores the fixed posture; add --auto-palletize-slot1 and the required "
            "calibration/grip acknowledgements to enable alignment",
            file=sys.stderr,
            flush=True,
        )
    config = _load_config(args.config)
    from .pallet_runtime import run_pallet_live

    return int(
        run_pallet_live(
            config,
            execute=bool(args.auto_palletize_slot1),
            allow_nominal_registration=bool(args.allow_nominal_registration),
            allow_geometry_only_grip_check=bool(
                args.allow_geometry_only_grip_check
            ),
            auto_place_slot1=bool(args.auto_place_slot1),
            allow_vision_geometry_release=bool(
                args.allow_vision_geometry_release
            ),
            ensure_slot1_ready=bool(args.ensure_slot1_ready),
            robot_address=str(args.robot_address),
            robot_power=str(args.robot_power),
            warmup_frames=int(args.warmup_frames),
            max_frames=args.max_frames,
            headless=bool(args.headless),
            window_name=str(args.window_name),
            output_mp4=args.output_mp4,
            log_jsonl=args.log_jsonl,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("interrupted by user", file=sys.stderr)
        return 130
    except _control_error_types() as exc:
        # A control/stream fault is an operational failure, not CLI misuse, so it
        # must not be reported as an argparse usage error.
        traceback.print_exc()
        print(f"pallet control fault: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
