import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from parcel_pose.cli import build_parser, main


@pytest.mark.parametrize(
    "subcommand",
    ["calibrate-plane", "replay", "evaluate-video", "record", "live", "live-view"],
)
def test_all_subcommand_help_runs_without_realsense(subcommand, capsys):
    parser = build_parser()
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args([subcommand, "--help"])
    assert exit_info.value.code == 0
    assert "usage:" in capsys.readouterr().out
    assert "pyrealsense2" not in sys.modules


def test_root_help_runs_without_realsense(capsys):
    assert main([]) == 0
    assert "D435 parcel pose tools" in capsys.readouterr().out


def test_keyboard_interrupt_returns_shell_interrupt_status(monkeypatch, capsys) -> None:
    import parcel_pose.cli as cli

    class FakeParser:
        def parse_args(self, argv):
            def interrupt(_args):
                raise KeyboardInterrupt

            return SimpleNamespace(handler=interrupt)

        def print_help(self):  # pragma: no cover - handler is present
            raise AssertionError("help must not be printed")

        def error(self, message):  # pragma: no cover - interrupt has its own path
            raise AssertionError(message)

    monkeypatch.setattr(cli, "build_parser", FakeParser)

    assert cli.main([]) == 130
    assert "interrupted by user" in capsys.readouterr().err


def test_live_view_defaults_remain_perception_only() -> None:
    args = build_parser().parse_args(
        ["live-view", "--calibration", "calibration.json"]
    )

    assert not args.auto_grab
    assert not args.allow_nominal_registration
    assert args.robot_address == "192.168.30.1:50051"


def test_auto_grab_requires_explicit_nominal_registration_acceptance(
    capsys,
) -> None:
    calibration = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "rby1m_v1_2_fixed_table_nominal.json"
    )

    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "live-view",
                "--calibration",
                str(calibration),
                "--auto-grab",
            ]
        )

    assert exit_info.value.code == 2
    assert "--allow-nominal-registration" in capsys.readouterr().err


def test_auto_grab_builds_opt_in_runtime_and_forwards_it_to_live_loop(
    monkeypatch,
) -> None:
    import parcel_pose.auto_grab as auto_grab
    import parcel_pose.realtime as realtime

    calibration = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "rby1m_v1_2_fixed_table_nominal.json"
    )
    created = []
    forwarded = []

    class FakeRuntime:
        def __init__(self, config, *, execute=False) -> None:
            created.append((config, execute, self))

    def fake_live_view(calibration, estimator_config, metadata_context, **kwargs):
        forwarded.append(kwargs)
        return 1

    monkeypatch.setattr(auto_grab, "AutoGrabRuntime", FakeRuntime)
    monkeypatch.setattr(realtime, "run_live_view", fake_live_view)

    assert main(
        [
            "live-view",
            "--calibration",
            str(calibration),
            "--auto-grab",
            "--allow-nominal-registration",
            "--robot-address",
            "10.0.0.7:50051",
        ]
    ) == 0
    config, execute, runtime = created[0]
    assert execute is True
    assert config.address == "10.0.0.7:50051"
    assert forwarded[0]["automation"] is runtime


def test_live_without_sdk_returns_actionable_error_no_import_traceback(monkeypatch, capsys):
    import parcel_pose.realsense_adapter as adapter_module

    def missing():
        raise adapter_module.RealSenseUnavailableError("install SDK and connect D435")

    monkeypatch.setattr(adapter_module, "load_realsense_sdk", missing)
    config = Path(__file__).resolve().parents[1] / "configs" / "d435_rby1_nominal.json"
    assert main(
        [
            "live",
            "--calibration",
            str(config),
            "--frames",
            "1",
            "--warmup-frames",
            "0",
        ]
    ) == 2
    captured = capsys.readouterr()
    assert "install SDK and connect D435" in captured.err
    assert "Traceback" not in captured.err


def test_replay_with_calibration_writes_deterministic_pose_jsonl(tmp_path, capsys):
    from parcel_pose.models import Calibration, CalibrationState, Plane
    from parcel_pose.calibration import save_calibration
    from parcel_pose.recording import write_session
    from tests.test_recording import make_frame, make_metadata

    session = tmp_path / "recording"
    calibration_path = tmp_path / "calibration.json"
    output = tmp_path / "results" / "poses.jsonl"
    write_session(session, make_metadata(), [make_frame(0), make_frame(1)])
    save_calibration(
        calibration_path,
        Calibration(
            state=CalibrationState.PLANE_CALIBRATED_PARTIAL,
            table_plane=Plane(normal=[0.0, 0.0, -1.0], d=-0.85, frame="depth"),
        ),
    )
    config = Path(__file__).resolve().parents[1] / "configs" / "d435_rby1_nominal.json"
    arguments = [
        "replay",
        "--session",
        str(session),
        "--calibration",
        str(calibration_path),
        "--config",
        str(config),
        "--output-jsonl",
        str(output),
    ]
    assert main(arguments) == 0
    first = output.read_text(encoding="utf-8")
    assert len(first.splitlines()) == 2
    assert "insufficient_top_plane_points" in first
    assert "center_base_xy_m" not in first
    assert main(arguments) == 0
    assert output.read_text(encoding="utf-8") == first
    assert "replayed" in capsys.readouterr().out


def test_replay_can_emit_single_frames_and_stationary_burst(tmp_path):
    from parcel_pose.models import Calibration, CalibrationState, Plane
    from parcel_pose.calibration import save_calibration
    from parcel_pose.recording import write_session
    from tests.test_recording import make_frame, make_metadata

    session = tmp_path / "recording"
    calibration_path = tmp_path / "calibration.json"
    output = tmp_path / "poses.jsonl"
    write_session(session, make_metadata(), [make_frame(0), make_frame(1)])
    save_calibration(
        calibration_path,
        Calibration(
            state=CalibrationState.PLANE_CALIBRATED_PARTIAL,
            table_plane=Plane(normal=[0.0, 0.0, -1.0], d=-0.85, frame="depth"),
        ),
    )

    assert main(
        [
            "replay",
            "--session",
            str(session),
            "--calibration",
            str(calibration_path),
            "--output-jsonl",
            str(output),
            "--burst-size",
            "2",
            "--burst-min-valid",
            "2",
        ]
    ) == 0
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [record["result_kind"] for record in records] == [
        "single_frame",
        "single_frame",
        "stationary_burst",
    ]
    assert records[-1]["diagnostics"]["burst"]["input_frames"] == 2
