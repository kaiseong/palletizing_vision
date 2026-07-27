import json
from pathlib import Path
import sys

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
    assert "Perception-only" in capsys.readouterr().out


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
