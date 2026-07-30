"""CLI contract for the box-picking facade after the package split."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest

from parcel_pose_picking import cli


REPO_ROOT = Path(__file__).resolve().parents[2]
FACADE = REPO_ROOT / "Box_picking" / "box_picking.py"


def test_parser_defaults_to_auto_pick_without_public_ack_flags() -> None:
    parser = cli.build_parser()
    help_text = parser.format_help()

    assert "--headless" in help_text
    assert "--config" in help_text
    assert "--calibration" in help_text
    assert "--robot-address" in help_text
    assert "--robot-power" in help_text
    assert "--output-mp4" in help_text
    assert "--log-jsonl" in help_text
    assert "--auto-grab" not in help_text
    assert "--allow-nominal-registration" not in help_text

    args = parser.parse_args([])
    assert args.headless is False
    assert args.output_mp4 is None
    assert args.log_jsonl is None
    assert args.robot_address == "192.168.30.1:50051"
    assert args.robot_power == ".*"
    assert args.config == REPO_ROOT / "Box_picking" / "configs" / "picking_config.json"
    assert (
        args.calibration
        == REPO_ROOT / "Box_picking" / "configs" / "picking_calibration.json"
    )
    assert not hasattr(args, "auto_grab")
    assert not hasattr(args, "allow_nominal_registration")


def test_non_live_diagnostic_subcommands_are_preserved() -> None:
    parser = cli.build_parser()
    subcommands = {
        action.dest: action.choices
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    }["subcommand"]

    assert {"calibrate-plane", "replay", "evaluate-video"} <= set(subcommands)
    assert "live" not in subcommands
    assert "live-view" not in subcommands
    assert "record" not in subcommands

    replay = parser.parse_args(["replay", "--session", "session", "--jsonl"])
    assert replay.subcommand == "replay"
    assert replay.config == REPO_ROOT / "Box_picking" / "configs" / "picking_config.json"
    assert replay.calibration is None
    assert replay.jsonl is True

    calibration_output = REPO_ROOT / "out" / "calibration.json"
    calibrate = parser.parse_args(
        ["calibrate-plane", "--session", "empty", "--output", str(calibration_output)]
    )
    assert calibrate.subcommand == "calibrate-plane"
    assert calibrate.output == calibration_output

    evaluate = parser.parse_args(
        [
            "evaluate-video",
            "--session",
            "box",
            "--calibration",
            "calibration.json",
            "--output-mp4",
            "overlay.mp4",
        ]
    )
    assert evaluate.subcommand == "evaluate-video"
    assert evaluate.output_mp4 == Path("overlay.mp4")


@pytest.mark.parametrize("flag", ["--auto-grab", "--allow-nominal-registration"])
def test_removed_ack_flags_are_rejected(flag: str) -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args([flag])

    assert excinfo.value.code == 2


def test_headless_only_changes_display_mode() -> None:
    parser = cli.build_parser()
    baseline = vars(parser.parse_args([]))
    headless = vars(parser.parse_args(["--headless"]))

    changed = {
        key: (baseline[key], headless[key])
        for key in sorted(baseline)
        if key != "handler" and baseline[key] != headless[key]
    }
    assert changed == {"headless": (False, True)}


def test_explicit_output_paths_propagate_without_defaults(tmp_path: Path) -> None:
    parser = cli.build_parser()
    video = tmp_path / "overlay.mp4"
    telemetry = tmp_path / "pose.jsonl"

    args = parser.parse_args(["--output-mp4", str(video), "--log-jsonl", str(telemetry)])

    assert args.output_mp4 == video
    assert args.log_jsonl == telemetry


@pytest.mark.parametrize("argv, expected_headless", [([], False), (["--headless"], True)])
def test_no_arg_and_headless_build_auto_pick_without_camera_or_robot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    expected_headless: bool,
) -> None:
    config = tmp_path / "config.json"
    calibration = tmp_path / "calibration.json"
    video = tmp_path / "overlay.mp4"
    telemetry = tmp_path / "pose.jsonl"
    config.write_text("{}", encoding="utf-8")
    calibration.write_text("{}", encoding="utf-8")

    run_calls: list[dict[str, object]] = []
    automation_configs: list[tuple[str, str, bool]] = []

    class FakeAutoGrabConfig:
        def __init__(self, *, address: str, power: str) -> None:
            self.address = address
            self.power = power

    class FakeAutoGrabRuntime:
        def __init__(self, config: FakeAutoGrabConfig, *, execute: bool) -> None:
            automation_configs.append((config.address, config.power, execute))

    auto_grab_module = ModuleType("parcel_pose_picking.auto_grab")
    auto_grab_module.AutoGrabConfig = FakeAutoGrabConfig
    auto_grab_module.AutoGrabError = RuntimeError
    auto_grab_module.AutoGrabRuntime = FakeAutoGrabRuntime
    monkeypatch.setitem(sys.modules, "parcel_pose_picking.auto_grab", auto_grab_module)

    realtime_module = ModuleType("parcel_pose_picking.realtime")
    realtime_module.LiveViewUnavailableError = RuntimeError

    plan_calls: list[dict[str, object]] = []

    def fake_resolve_live_view_plan(**kwargs: object) -> SimpleNamespace:
        plan_calls.append(kwargs)
        return SimpleNamespace(
            handoff_ready=False,
            handoff_started=False,
            log_stream=None,
            processed_frames=0,
            user_cancelled=False,
            video_writer=None,
            window_created=False,
        )

    def fake_watch_and_grab(**kwargs: object) -> None:
        run_calls.append(kwargs)

    realtime_module.resolve_live_view_plan = fake_resolve_live_view_plan
    realtime_module.watch_and_grab = fake_watch_and_grab
    monkeypatch.setitem(sys.modules, "parcel_pose_picking.realtime", realtime_module)

    def fake_load_json(path: Path) -> dict[str, object]:
        assert path == config
        return {}

    def fake_load_calibration(path: Path) -> SimpleNamespace:
        assert path == calibration
        return SimpleNamespace(absolute_base_validated=False)

    monkeypatch.setattr("parcel_pose_common.calibration.load_json", fake_load_json)
    monkeypatch.setattr("parcel_pose_common.calibration.load_calibration", fake_load_calibration)
    monkeypatch.setattr("parcel_pose_picking.cli._estimator_config", lambda _config: object())
    monkeypatch.setattr("parcel_pose_picking.cli._recording_context", lambda *_args: {})

    args = [
        "--config",
        str(config),
        "--calibration",
        str(calibration),
        "--output-mp4",
        str(video),
        "--log-jsonl",
        str(telemetry),
        "--robot-address",
        "robot.test:50051",
        "--robot-power",
        "main",
        *argv,
    ]

    # The sequence lives in box_picking.py, which attaches it to the handler.
    import box_picking

    assert box_picking.main(args) == 0
    assert automation_configs == [("robot.test:50051", "main", True)]
    assert len(plan_calls) == 1
    assert len(run_calls) == 1
    assert plan_calls[0]["headless"] is expected_headless
    assert plan_calls[0]["output_mp4"] == video
    assert plan_calls[0]["log_jsonl"] == telemetry
    assert run_calls[0]["headless"] is expected_headless
    assert run_calls[0]["automation"] is not None
    captured = capsys.readouterr()
    assert "nominal_unverified camera registration" in captured.err
    assert "automatic RB-Y1 motion is enabled by the box_picking entrypoint" in captured.err


def test_foreign_cwd_facade_help_resolves_source_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = importlib.util.spec_from_file_location("box_picking_facade_under_test", FACADE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.chdir("/")

    with pytest.raises(SystemExit) as excinfo:
        module.main(["--help"])

    assert excinfo.value.code == 0
    assert str(REPO_ROOT / "Common" / "src") in sys.path
    assert str(REPO_ROOT / "Box_picking" / "src") in sys.path
