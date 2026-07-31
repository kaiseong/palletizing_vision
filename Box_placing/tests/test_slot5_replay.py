"""Software-only Phase-5 contracts for slot-5 replay diagnostics."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import shutil
from typing import Any

import pytest

from parcel_pose_common.operation_authority import OperationMode
from parcel_pose_placing import slot5_replay
from parcel_pose_placing.slot5_replay import (
    Slot5ManifestValidationError,
    approved_slot5_frame_count,
    replay_slot5_session,
    validate_slot5_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "Box_placing" / "configs" / "placing_config.json"
RECORDING_ROOT = REPO_ROOT / "recordings" / "codex_640x480"
STATIC_SESSION = RECORDING_ROOT / "pallet_slot5"
MOVING_SESSION = RECORDING_ROOT / "pallet_slot5_moving"


@pytest.mark.parametrize(
    ("session", "expected_count"),
    [(STATIC_SESSION, 96), (MOVING_SESSION, 938)],
)
def test_approved_slot5_manifests_load_every_frame_with_valid_calibration(
    session: Path,
    expected_count: int,
) -> None:
    assert approved_slot5_frame_count(session) == expected_count

    audit = validate_slot5_manifest(
        session,
        expected_frame_count=expected_count,
    )

    assert audit["validation_status"] == "pass"
    assert audit["complete"] is True
    assert audit["manifest_frame_count"] == expected_count
    assert audit["loaded_frame_count"] == expected_count
    assert audit["all_manifest_frames_loaded"] is True
    assert audit["intrinsics_validated"] is True
    assert audit["extrinsics_validated"] is True
    assert audit["timestamps_strictly_increasing"] is True
    assert audit["frame_numbers_strictly_increasing"] is True
    assert audit["maximum_rgb_depth_timestamp_skew_ms"] <= 50.0
    assert audit["extrinsics_inverse_max_abs_error"] <= 1e-6


def test_missing_manifest_frame_has_a_named_frame_local_failure(
    tmp_path: Path,
) -> None:
    payload = json.loads(
        (STATIC_SESSION / "manifest.json").read_text(encoding="utf-8")
    )
    payload["frames"] = [
        {
            **payload["frames"][0],
            "file": "frames/intentionally-missing.npz",
        }
    ]
    payload["frame_count"] = 1
    session = tmp_path / "missing-frame"
    session.mkdir()
    (session / "manifest.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        Slot5ManifestValidationError,
        match=r"slot-5 frame 0 failed validation:.*file is missing",
    ):
        validate_slot5_manifest(session, expected_frame_count=1)


def test_non_increasing_timestamp_has_a_named_frame_local_failure(
    tmp_path: Path,
) -> None:
    payload = json.loads(
        (STATIC_SESSION / "manifest.json").read_text(encoding="utf-8")
    )
    entries = [dict(payload["frames"][0]), dict(payload["frames"][1])]
    entries[1]["depth_timestamp_ms"] = entries[0]["depth_timestamp_ms"]
    payload["frames"] = entries
    payload["frame_count"] = 2
    session = tmp_path / "bad-timestamps"
    frames = session / "frames"
    frames.mkdir(parents=True)
    for entry in entries:
        source = STATIC_SESSION / entry["file"]
        target = session / entry["file"]
        shutil.copyfile(source, target)
    (session / "manifest.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        Slot5ManifestValidationError,
        match=r"slot-5 frame 1 failed validation: depth timestamps",
    ):
        validate_slot5_manifest(session, expected_frame_count=2)


def test_bounded_replay_is_byte_semantically_deterministic_and_fail_closed() -> None:
    kwargs = {
        "session_path": STATIC_SESSION,
        "root_config": CONFIG_PATH,
        "expected_frame_count": 96,
        "max_frames": 12,
    }
    first = replay_slot5_session(**kwargs)
    second = replay_slot5_session(**kwargs)

    assert first == second
    assert first["artifact_sha256"] == second["artifact_sha256"]
    assert first["manifest"]["loaded_frame_count"] == 12
    assert first["manifest"]["all_manifest_frames_loaded"] is False
    assert first["stability"]["processed_frame_count"] == 12
    assert first["stability"]["state_transition_count"] == max(
        len(first["state_runs"]) - 1,
        0,
    )
    assert all(
        row["pose_result"]
        == {
            "valid": False,
            "reason": "slot_5_pose_unavailable",
            "frame": "base",
            "x_m": None,
            "y_m": None,
            "yaw_rad": None,
        }
        for row in first["frames"]
    )
    assert first["candidate_policy"] == {
        "live_reference_validated": False,
        "place_pose_available": False,
        "retreat_pose_available": False,
        "facade_result_required_reason": "slot_5_pose_unavailable",
        "diagnostic_candidates_grant_motion_authority": False,
    }


@pytest.mark.parametrize("mode", [OperationMode.REPLAY, OperationMode.DRY_RUN])
def test_slot5_offline_modes_construct_no_live_stack_and_issue_no_commands(
    monkeypatch: pytest.MonkeyPatch,
    mode: OperationMode,
) -> None:
    from parcel_pose_placing import (
        pallet_control,
        pallet_place,
        pallet_ready,
        placement_lifecycle,
        placing_session,
    )

    effects: Counter[str] = Counter()

    def forbidden(name: str):
        def fail(*_args: Any, **_kwargs: Any) -> None:
            effects[name] += 1
            raise AssertionError(f"slot-5 offline replay reached {name}")

        return fail

    monkeypatch.setattr(
        pallet_control,
        "RBY1PalletController",
        forbidden("controller"),
    )
    monkeypatch.setattr(
        pallet_ready,
        "ensure_slot1_ready_from_config",
        forbidden("ready_posture"),
    )
    monkeypatch.setattr(
        placing_session,
        "RealSenseAdapter",
        forbidden("stream"),
    )
    monkeypatch.setattr(
        pallet_place,
        "Slot1PlacementSequencer",
        forbidden("sequencer"),
    )
    monkeypatch.setattr(
        placement_lifecycle,
        "PlacementLifecycleRuntime",
        forbidden("lifecycle"),
    )

    artifact = replay_slot5_session(
        STATIC_SESSION,
        CONFIG_PATH,
        mode=mode,
        expected_frame_count=96,
        max_frames=1,
    )

    assert effects == Counter()
    assert artifact["actuation"] == {
        "authority_reason": (
            f"slot 5 place {mode.value} authorized for offline_perception_only"
        ),
        "offline_perception_permitted": True,
        "robot_connection_count": 0,
        "controller_construction_count": 0,
        "ready_posture_construction_count": 0,
        "stream_construction_count": 0,
        "sequencer_construction_count": 0,
        "place_actuation_count": 0,
        "retreat_actuation_count": 0,
        "motion_authorized": False,
    }


def test_live_mode_refuses_before_manifest_or_perception_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    effects: list[str] = []

    def forbidden_reader(*_args: Any, **_kwargs: Any) -> None:
        effects.append("session_reader")
        raise AssertionError("live slot 5 reached recording construction")

    monkeypatch.setattr(slot5_replay, "SessionReader", forbidden_reader)
    with pytest.raises(ValueError, match="require replay or dry_run"):
        replay_slot5_session(
            STATIC_SESSION,
            CONFIG_PATH,
            mode=OperationMode.LIVE,
        )
    assert effects == []


def test_artifact_write_is_exact_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "slot5.json"
    artifact = replay_slot5_session(
        STATIC_SESSION,
        CONFIG_PATH,
        expected_frame_count=96,
        max_frames=2,
        output_artifact=output,
    )
    assert json.loads(output.read_text(encoding="utf-8")) == artifact

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        replay_slot5_session(
            STATIC_SESSION,
            CONFIG_PATH,
            expected_frame_count=96,
            max_frames=1,
            output_artifact=output,
        )


def test_slot5_cli_is_explicit_offline_and_infers_reviewed_frame_count(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from parcel_pose_placing import pallet_cli

    calls: list[dict[str, Any]] = []

    def fake_replay(session: Path, config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        calls.append({"session": session, "config": config, **kwargs})
        return {"artifact_type": "slot5_deterministic_replay_diagnostic"}

    monkeypatch.setattr(slot5_replay, "replay_slot5_session", fake_replay)
    assert (
        pallet_cli.main(
            [
                "slot5-replay",
                "--session",
                str(STATIC_SESSION),
                "--config",
                str(CONFIG_PATH),
                "--dry-run",
                "--max-frames",
                "3",
            ]
        )
        == 0
    )

    assert len(calls) == 1
    assert calls[0]["session"] == STATIC_SESSION
    assert calls[0]["mode"] is OperationMode.DRY_RUN
    assert calls[0]["expected_frame_count"] == 96
    assert calls[0]["max_frames"] == 3
    assert calls[0]["output_artifact"] is None
    assert calls[0]["overwrite"] is False
    assert json.loads(capsys.readouterr().out) == {
        "artifact_type": "slot5_deterministic_replay_diagnostic"
    }
