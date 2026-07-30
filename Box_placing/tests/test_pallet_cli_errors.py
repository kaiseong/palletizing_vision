"""Control faults must not be reported as CLI usage errors."""

from __future__ import annotations

import pathlib

import pytest

from parcel_pose_placing import pallet_cli
from parcel_pose_placing.pallet_control import CombinedStreamError


ARGV = ["replay", "--session", "nonexistent-session", "--no-default-artifacts"]


def test_control_fault_exits_with_its_own_code(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def explode(*args, **kwargs):
        raise CombinedStreamError("This command stream is expired")

    monkeypatch.setattr(
        "parcel_pose_placing.pallet_evaluation.evaluate_pallet_session", explode
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
        "parcel_pose_placing.pallet_evaluation.evaluate_pallet_session", explode
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
        "parcel_pose_placing.pallet_evaluation.evaluate_pallet_session", explode
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

    from parcel_pose_placing import pallet_runtime

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
                auto_place_slot1=True,
                log_jsonl=existing,
            )
    finally:
        pallet_runtime._open_log = original  # type: ignore[assignment]
    assert connected == [], "preflight must run before the robot is touched"


def test_existing_overlay_video_is_refused_too(tmp_path) -> None:
    from parcel_pose_placing.pallet_runtime import _preflight_output_paths

    video = tmp_path / "overlay.mp4"
    video.write_bytes(b"")
    with pytest.raises(FileExistsError, match="refusing to overwrite overlay video"):
        _preflight_output_paths(None, video)


def test_preflight_creates_the_artifact_directory_and_leaves_no_probe(tmp_path) -> None:
    from parcel_pose_placing.pallet_runtime import _preflight_output_paths

    target = tmp_path / "nested" / "out" / "live.jsonl"
    _preflight_output_paths(target, None)
    assert target.parent.is_dir()
    assert not target.exists(), "preflight must not create the artifact itself"
    assert list(target.parent.iterdir()) == [], "the write probe must be removed"


# --------------------------------------------------------------------------- #
# alignment faults must state the measurement that tripped them
# --------------------------------------------------------------------------- #
