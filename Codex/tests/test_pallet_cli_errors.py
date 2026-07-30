"""Control faults must not be reported as CLI usage errors."""

from __future__ import annotations

import math
import pathlib

import pytest

from parcel_pose import pallet_cli
from parcel_pose.pallet_control import CombinedStreamError


ARGV = ["replay", "--session", "nonexistent-session", "--no-default-artifacts"]


def test_control_fault_exits_with_its_own_code(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def explode(*args, **kwargs):
        raise CombinedStreamError("This command stream is expired")

    monkeypatch.setattr(
        "parcel_pose.pallet_evaluation.evaluate_pallet_session", explode
    )
    assert pallet_cli.main(ARGV) == 3
    captured = capsys.readouterr()
    assert "pallet control fault: CombinedStreamError" in captured.err
    # A usage error would print the argparse banner instead of a traceback.
    assert "usage: pallet" not in captured.err
    assert "Traceback" in captured.err


def test_plain_value_errors_remain_usage_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*args, **kwargs):
        raise ValueError("bad session layout")

    monkeypatch.setattr(
        "parcel_pose.pallet_evaluation.evaluate_pallet_session", explode
    )
    with pytest.raises(SystemExit) as excinfo:
        pallet_cli.main(ARGV)
    assert excinfo.value.code == 2


def test_keyboard_interrupt_keeps_its_conventional_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "parcel_pose.pallet_evaluation.evaluate_pallet_session", explode
    )
    assert pallet_cli.main(ARGV) == 130


# --------------------------------------------------------------------------- #
# artifact paths must be rejected before the robot is touched
# --------------------------------------------------------------------------- #
def test_existing_telemetry_is_refused_before_any_robot_contact(tmp_path) -> None:
    """``_open_log`` runs after the arms are held; a refusal there strands a load.

    A live run lost a carton this way: the telemetry file already existed, the
    refusal fired after the combined stream owned the arms, containment printed
    DANGER until the operator forced a cancellation.
    """

    import json

    from parcel_pose import pallet_runtime

    config = json.loads(
        (
            pathlib.Path(__file__).resolve().parents[1]
            / "configs"
            / "rby1m_v1_2_pallet_slot1_nominal.json"
        ).read_text(encoding="utf-8")
    )
    existing = tmp_path / "live.jsonl"
    existing.write_text("{}\n", encoding="utf-8")

    connected: list[str] = []

    def forbidden(*args, **kwargs):  # pragma: no cover - must never run
        connected.append("connect")
        raise AssertionError("the robot was contacted despite an unusable log path")

    original = pallet_runtime._open_log
    pallet_runtime._open_log = forbidden  # type: ignore[assignment]
    try:
        with pytest.raises(FileExistsError, match="refusing to overwrite live"):
            pallet_runtime.run_pallet_live(
                config,
                execute=True,
                ensure_slot1_ready=True,
                allow_nominal_registration=True,
                allow_geometry_only_grip_check=True,
                auto_place_slot1=True,
                allow_vision_geometry_release=True,
                log_jsonl=existing,
            )
    finally:
        pallet_runtime._open_log = original  # type: ignore[assignment]
    assert connected == [], "preflight must run before the robot is touched"


def test_existing_overlay_video_is_refused_too(tmp_path) -> None:
    from parcel_pose.pallet_runtime import _preflight_output_paths

    video = tmp_path / "overlay.mp4"
    video.write_bytes(b"")
    with pytest.raises(FileExistsError, match="refusing to overwrite overlay video"):
        _preflight_output_paths(None, video)


def test_preflight_creates_the_artifact_directory_and_leaves_no_probe(tmp_path) -> None:
    from parcel_pose.pallet_runtime import _preflight_output_paths

    target = tmp_path / "nested" / "out" / "live.jsonl"
    _preflight_output_paths(target, None)
    assert target.parent.is_dir()
    assert not target.exists(), "preflight must not create the artifact itself"
    assert list(target.parent.iterdir()) == [], "the write probe must be removed"


# --------------------------------------------------------------------------- #
# alignment faults must state the measurement that tripped them
# --------------------------------------------------------------------------- #
def test_correction_limit_fault_reports_the_measured_error() -> None:
    """A bare reason cannot tell a far-parked base from a misidentified feature."""

    from parcel_pose.pallet_servo import (
        PalletServoConfig,
        PalletServoObservation,
        PalletServoState,
        PalletSlot1Servo,
    )

    config = PalletServoConfig()
    servo = PalletSlot1Servo(config)
    now = 100.0
    servo.start(now)
    far = float(config.max_correction_m) + 0.16
    reason = ""
    for index in range(config.filter_window + 2):
        moment = now + 0.05 * (index + 1)
        reason = servo.update(
            PalletServoObservation(
                timestamp_s=moment,
                current_observed_feature_center_base=(0.865 + far, 0.139523),
                current_observed_feature_yaw_base_rad=math.radians(-90.0),
                demonstrated_body_reference_center_base=(0.865, 0.139523),
                demonstrated_body_reference_yaw_base_rad=math.radians(-90.0),
                axis_branch=config.expected_axis_branch,
                reference_source="fixed_outer_l_corner",
            ),
            moment,
        ).reason
        if reason.startswith("correction_limit_exceeded"):
            break
    assert servo.state is PalletServoState.FAULT_HOLD
    assert reason.startswith("correction_limit_exceeded"), reason
    assert f"{far * 1000.0:.0f}mm" in reason, reason
    assert f"{config.max_correction_m * 1000.0:.0f}mm" in reason, reason
    assert "dx=" in reason and "dy=" in reason, reason


def test_correction_ceiling_can_be_disabled_and_the_servo_drives_from_far() -> None:
    """With the ceiling disabled the base tracks a distant hole estimate."""

    from parcel_pose.pallet_servo import (
        PalletServoConfig,
        PalletServoObservation,
        PalletServoState,
        PalletSlot1Servo,
    )

    config = PalletServoConfig(max_correction_m=None)
    servo = PalletSlot1Servo(config)
    now = 100.0
    servo.start(now)
    far = 0.60  # far beyond the former 250 mm ceiling
    output = None
    for index in range(config.filter_window + 2):
        moment = now + 0.05 * (index + 1)
        output = servo.update(
            PalletServoObservation(
                timestamp_s=moment,
                current_observed_feature_center_base=(0.865 + far, 0.139523),
                current_observed_feature_yaw_base_rad=math.radians(-90.0),
                demonstrated_body_reference_center_base=(0.865, 0.139523),
                demonstrated_body_reference_yaw_base_rad=math.radians(-90.0),
                axis_branch=config.expected_axis_branch,
                reference_source="fixed_outer_l_corner",
            ),
            moment,
        )
    assert output is not None
    assert servo.state is not PalletServoState.FAULT_HOLD, output.reason
    # Motion is still bounded by the speed ceiling, so distance only costs time.
    speed = math.hypot(output.command.vx_mps, output.command.vy_mps)
    assert speed > 0.0, output.reason
    assert speed <= config.max_linear_speed_mps + 1e-9


def test_the_shipped_slot1_config_disables_the_correction_ceiling() -> None:
    import json

    from parcel_pose.pallet_servo import PalletServoConfig

    root = json.loads(
        (
            pathlib.Path(__file__).resolve().parents[1]
            / "configs"
            / "rby1m_v1_2_pallet_slot1_nominal.json"
        ).read_text(encoding="utf-8")
    )
    assert PalletServoConfig.from_root_config(root).max_correction_m is None


def test_the_correction_ceiling_cannot_be_widened_instead_of_disabled() -> None:
    """Silently raising the ceiling would hide a misread feature; null is explicit."""

    from parcel_pose.pallet_servo import PalletServoConfig

    with pytest.raises(ValueError, match="use null to disable"):
        PalletServoConfig(max_correction_m=0.60)
