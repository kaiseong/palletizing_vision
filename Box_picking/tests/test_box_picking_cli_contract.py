"""CLI contract for the staged box-picking facade."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
from typing import Any

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
    assert "--orientation {horizontal,vertical}" in help_text
    assert "--robot-address" in help_text
    assert "--robot-power" in help_text
    assert "--output-mp4" in help_text
    assert "--log-jsonl" in help_text
    assert "--auto-grab" not in help_text
    assert "--allow-nominal-registration" not in help_text

    args = parser.parse_args([])
    assert args.orientation == "horizontal"
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
def test_no_arg_and_headless_run_authorized_horizontal_staged_flow_without_hardware(
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

    lifecycle: list[str] = []
    plan_calls: list[dict[str, object]] = []
    session_calls: list[dict[str, object]] = []
    automation_configs: list[tuple[str, str, bool, object]] = []
    stopped_and_released = object()
    fake_frame = object()
    fake_base_pose = object()

    class FakeLiveError(RuntimeError):
        pass

    class FakeAutoGrabConfig:
        def __init__(self, *, address: str, power: str, servo: object) -> None:
            self.address = address
            self.power = power
            self.servo = servo

    class FakeAutoGrabRuntime:
        def __init__(self, selected: FakeAutoGrabConfig, *, execute: bool) -> None:
            automation_configs.append(
                (selected.address, selected.power, execute, selected.servo)
            )

        def start(self) -> None:
            lifecycle.append("ready")

        def update(
            self,
            base_pose: object,
            *,
            pose_timestamp_s: float,
            now_s: float,
        ) -> bool:
            assert base_pose is fake_base_pose
            assert pose_timestamp_s <= now_s
            lifecycle.append("decide_xy_yaw")
            return True

        def stop_alignment_for_grasp(self) -> object:
            lifecycle.append("grasp_alignment_stopped_and_released")
            return stopped_and_released

        def execute_grasp(self, evidence: object) -> None:
            assert evidence is stopped_and_released
            lifecycle.append("grasp_and_lift")

        def close(self) -> None:
            lifecycle.append("robot_teardown")

    auto_grab_module = ModuleType("parcel_pose_picking.auto_grab")
    auto_grab_module.AutoGrabConfig = FakeAutoGrabConfig
    auto_grab_module.AutoGrabError = FakeLiveError
    auto_grab_module.AutoGrabRuntime = FakeAutoGrabRuntime
    monkeypatch.setitem(sys.modules, "parcel_pose_picking.auto_grab", auto_grab_module)

    class FakeAlignmentSession:
        def __init__(self) -> None:
            self.processed_frames = 0
            self.handoff_ready = False
            self.user_cancelled = False

        def __enter__(self) -> "FakeAlignmentSession":
            lifecycle.append("acquisition_prepared")
            return self

        def has_frame_budget(self) -> bool:
            return self.processed_frames < 1

        def acquire_frame(self) -> object:
            lifecycle.append("acquire_frame")
            return fake_frame

        def perceive_frame(self, frame: object) -> SimpleNamespace:
            assert frame is fake_frame
            lifecycle.append("perceive_frame")
            return SimpleNamespace(
                base_pose=fake_base_pose,
                pose_result=SimpleNamespace(timestamp_s=1.0),
            )

        def record_frame(
            self,
            observation: SimpleNamespace,
            *,
            handoff_ready: bool,
        ) -> None:
            assert observation.base_pose is fake_base_pose
            lifecycle.append("record_frame")
            self.processed_frames += 1
            self.handoff_ready = handoff_ready

        def cancel(self) -> None:
            self.user_cancelled = True
            self.handoff_ready = False

        def outcome(self) -> SimpleNamespace:
            return SimpleNamespace(
                processed_frames=self.processed_frames,
                handoff_ready=self.handoff_ready,
                user_cancelled=self.user_cancelled,
            )

        def watch(self, *_args: Any, **_kwargs: Any) -> None:
            pytest.fail("box_picking CLI must not delegate to session.watch")

        def __exit__(self, *_exc_info: Any) -> None:
            lifecycle.append("acquisition_teardown")

    realtime_module = ModuleType("parcel_pose_picking.realtime")
    realtime_module.LiveViewUnavailableError = FakeLiveError

    def fake_resolve_live_view_plan(**kwargs: object) -> SimpleNamespace:
        plan_calls.append(kwargs)
        return SimpleNamespace(
            handoff_ready=False,
            log_stream=None,
            processed_frames=0,
            user_cancelled=False,
            video_writer=None,
            window_created=False,
        )

    def fake_open_alignment_session(**kwargs: object) -> FakeAlignmentSession:
        session_calls.append(kwargs)
        return FakeAlignmentSession()

    realtime_module.resolve_live_view_plan = fake_resolve_live_view_plan
    realtime_module.open_alignment_session = fake_open_alignment_session
    monkeypatch.setitem(sys.modules, "parcel_pose_picking.realtime", realtime_module)

    def fake_load_json(path: Path) -> dict[str, object]:
        assert path == config
        return {}

    def fake_load_calibration(path: Path) -> SimpleNamespace:
        assert path == calibration
        return SimpleNamespace(absolute_base_validated=False)

    monkeypatch.setattr("parcel_pose_common.calibration.load_json", fake_load_json)
    monkeypatch.setattr(
        "parcel_pose_common.calibration.load_calibration", fake_load_calibration
    )
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

    import box_picking

    assert box_picking.main(args) == 0
    assert len(automation_configs) == 1
    assert automation_configs[0][:3] == ("robot.test:50051", "main", True)
    assert len(plan_calls) == 1
    assert len(session_calls) == 1
    assert plan_calls[0]["headless"] is expected_headless
    assert plan_calls[0]["output_mp4"] == video
    assert plan_calls[0]["log_jsonl"] == telemetry
    assert session_calls[0]["headless"] is expected_headless
    assert session_calls[0]["processed_frames"] == 0
    assert lifecycle == [
        "acquisition_prepared",
        "ready",
        "acquire_frame",
        "perceive_frame",
        "decide_xy_yaw",
        "record_frame",
        "acquisition_teardown",
        "grasp_alignment_stopped_and_released",
        "grasp_and_lift",
        "robot_teardown",
    ]
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
