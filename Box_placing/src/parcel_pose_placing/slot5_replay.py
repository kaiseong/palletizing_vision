"""Deterministic, non-actuating replay diagnostics for pallet slot 5.

The supplied slot-5 recordings contain RGB-D evidence and a demonstrated ready
pose, but no validated live reference, place pose, or retreat pose.  This
module therefore has no robot/controller/stream/sequencer surface.  It obtains
only offline-perception authority, runs one dependency-neutral perception
facade per recorded frame, and records candidates as diagnostics that can never
be consumed as live placement authority.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from parcel_pose_common.calibration import factory_extrinsics_to_transform
from parcel_pose_common.operation_authority import (
    AuthorityCapability,
    OperationMode,
    OperationRequest,
    authorize_operation,
)
from parcel_pose_common.recording import SessionReader
from parcel_pose_common.session import RecordedFrame, SessionValidationError

from .pallet_evaluation import _replay_base_from_depth, _replay_held_hint
from .pallet_geometry import PalletStackEstimator
from .pallet_models import load_pallet_estimator_config
from .pallet_perception_adapter import perceive_pallet_pose


SLOT = 5
EXPECTED_POSE_REASON = "slot_5_pose_unavailable"
APPROVED_SLOT5_FRAME_COUNTS: Mapping[str, int] = MappingProxyType(
    {
        "pallet_slot5": 96,
        "pallet_slot5_moving": 938,
    }
)


class Slot5ManifestValidationError(ValueError):
    """A named, frame-local recording validation failure."""


class Slot5ReplayError(RuntimeError):
    """The offline replay could not preserve its diagnostic-only contract."""


@dataclass(slots=True)
class _SequenceAudit:
    max_timestamp_skew_s: float
    loaded_count: int = 0
    first_depth_timestamp_ms: float | None = None
    last_depth_timestamp_ms: float | None = None
    first_color_timestamp_ms: float | None = None
    last_color_timestamp_ms: float | None = None
    maximum_rgb_depth_timestamp_skew_ms: float = 0.0
    previous_depth_timestamp_ms: float | None = None
    previous_color_timestamp_ms: float | None = None
    previous_hardware_timestamp_ms: float | None = None
    previous_system_timestamp_ns: int | None = None
    previous_depth_frame_number: int | None = None
    previous_color_frame_number: int | None = None

    def consume(self, frame: RecordedFrame, reader: SessionReader, index: int) -> None:
        depth_intrinsics = reader.metadata.depth_profile.intrinsics
        color_intrinsics = reader.metadata.color_profile.intrinsics
        if frame.raw_depth_z16.shape != (
            depth_intrinsics.height,
            depth_intrinsics.width,
        ):
            raise SessionValidationError(
                "raw depth shape does not match recorded intrinsics"
            )
        if frame.raw_color_bgr.shape[:2] != (
            color_intrinsics.height,
            color_intrinsics.width,
        ):
            raise SessionValidationError(
                "raw color shape does not match recorded intrinsics"
            )

        depth_ms = float(frame.depth_timestamp_ms)
        color_ms = float(frame.color_timestamp_ms)
        if self.previous_depth_timestamp_ms is not None and not (
            depth_ms > self.previous_depth_timestamp_ms
        ):
            raise SessionValidationError(
                "depth timestamps must be finite and strictly increasing"
            )
        if self.previous_color_timestamp_ms is not None and not (
            color_ms > self.previous_color_timestamp_ms
        ):
            raise SessionValidationError(
                "color timestamps must be finite and strictly increasing"
            )
        skew_ms = abs(depth_ms - color_ms)
        if skew_ms > self.max_timestamp_skew_s * 1_000.0:
            raise SessionValidationError(
                "RGB/depth timestamp skew exceeds the configured replay limit: "
                f"frame={index}, skew_ms={skew_ms:.6f}, "
                f"limit_ms={self.max_timestamp_skew_s * 1_000.0:.6f}"
            )

        if self.previous_depth_frame_number is not None and not (
            frame.depth_frame_number > self.previous_depth_frame_number
        ):
            raise SessionValidationError(
                "depth frame numbers must be strictly increasing"
            )
        if self.previous_color_frame_number is not None and not (
            frame.color_frame_number > self.previous_color_frame_number
        ):
            raise SessionValidationError(
                "color frame numbers must be strictly increasing"
            )

        hardware_ms = frame.hardware_timestamp_ms
        if hardware_ms is not None:
            hardware_ms = float(hardware_ms)
            if not math.isfinite(hardware_ms):
                raise SessionValidationError("hardware timestamp must be finite")
            if self.previous_hardware_timestamp_ms is not None and not (
                hardware_ms > self.previous_hardware_timestamp_ms
            ):
                raise SessionValidationError(
                    "hardware timestamps must be strictly increasing"
                )
            self.previous_hardware_timestamp_ms = hardware_ms

        system_ns = frame.system_timestamp_ns
        if system_ns is not None:
            system_ns = int(system_ns)
            if system_ns <= 0:
                raise SessionValidationError("system timestamp must be positive")
            if self.previous_system_timestamp_ns is not None and not (
                system_ns > self.previous_system_timestamp_ns
            ):
                raise SessionValidationError(
                    "system timestamps must be strictly increasing"
                )
            self.previous_system_timestamp_ns = system_ns

        if self.loaded_count == 0:
            self.first_depth_timestamp_ms = depth_ms
            self.first_color_timestamp_ms = color_ms
        self.last_depth_timestamp_ms = depth_ms
        self.last_color_timestamp_ms = color_ms
        self.maximum_rgb_depth_timestamp_skew_ms = max(
            self.maximum_rgb_depth_timestamp_skew_ms,
            skew_ms,
        )
        self.previous_depth_timestamp_ms = depth_ms
        self.previous_color_timestamp_ms = color_ms
        self.previous_depth_frame_number = int(frame.depth_frame_number)
        self.previous_color_frame_number = int(frame.color_frame_number)
        self.loaded_count += 1

    def to_dict(self) -> dict[str, Any]:
        duration_s = None
        if (
            self.first_depth_timestamp_ms is not None
            and self.last_depth_timestamp_ms is not None
        ):
            duration_s = _rounded(
                (self.last_depth_timestamp_ms - self.first_depth_timestamp_ms)
                / 1_000.0
            )
        return {
            "loaded_frame_count": self.loaded_count,
            "first_depth_timestamp_ms": _optional_rounded(
                self.first_depth_timestamp_ms
            ),
            "last_depth_timestamp_ms": _optional_rounded(
                self.last_depth_timestamp_ms
            ),
            "first_color_timestamp_ms": _optional_rounded(
                self.first_color_timestamp_ms
            ),
            "last_color_timestamp_ms": _optional_rounded(
                self.last_color_timestamp_ms
            ),
            "duration_s": duration_s,
            "maximum_rgb_depth_timestamp_skew_ms": _rounded(
                self.maximum_rgb_depth_timestamp_skew_ms
            ),
            "timestamp_limit_ms": _rounded(
                self.max_timestamp_skew_s * 1_000.0
            ),
            "timestamps_strictly_increasing": True,
            "frame_numbers_strictly_increasing": True,
        }


def approved_slot5_frame_count(session_path: str | Path) -> int | None:
    """Return the reviewed frame count for either supplied slot-5 recording."""

    return APPROVED_SLOT5_FRAME_COUNTS.get(Path(session_path).name)


def _rounded(value: float, digits: int = 9) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise Slot5ReplayError("slot-5 diagnostic contains a non-finite value")
    return round(result, digits)


def _optional_rounded(value: float | None) -> float | None:
    return None if value is None else _rounded(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _root_payload(root_config: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(root_config, Mapping):
        return dict(root_config)
    path = Path(root_config)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read slot-5 replay config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("slot-5 replay config root must be an object")
    return payload


def _open_validated_reader(
    session_path: str | Path,
    *,
    expected_frame_count: int | None,
) -> tuple[SessionReader, dict[str, Any]]:
    session = Path(session_path)
    try:
        reader = SessionReader(session, verify_hashes=True)
    except (OSError, ValueError) as exc:
        raise Slot5ManifestValidationError(
            f"slot-5 manifest is invalid: {exc}"
        ) from exc

    if not reader.complete:
        raise Slot5ManifestValidationError(
            "slot-5 manifest is incomplete; complete=true is required"
        )
    if len(reader) <= 0:
        raise Slot5ManifestValidationError(
            "slot-5 manifest must contain at least one frame"
        )
    if expected_frame_count is not None:
        expected = int(expected_frame_count)
        if expected <= 0:
            raise ValueError("expected_frame_count must be positive")
        if len(reader) != expected:
            raise Slot5ManifestValidationError(
                "slot-5 manifest frame count mismatch: "
                f"expected {expected}, found {len(reader)}"
            )

    try:
        depth_to_color = factory_extrinsics_to_transform(
            reader.metadata.depth_to_color
        )
        color_to_depth = factory_extrinsics_to_transform(
            reader.metadata.color_to_depth
        )
    except (TypeError, ValueError) as exc:
        raise Slot5ManifestValidationError(
            f"slot-5 camera extrinsics are invalid: {exc}"
        ) from exc
    inverse_error = float(
        np.max(np.abs(depth_to_color @ color_to_depth - np.eye(4)))
    )
    if inverse_error > 1e-6:
        raise Slot5ManifestValidationError(
            "slot-5 depth/color extrinsics are not mutual inverses: "
            f"max_abs_error={inverse_error:.9g}"
        )

    manifest_path = session / "manifest.json"
    metadata = reader.metadata
    header = {
        "manifest_sha256": _sha256_file(manifest_path),
        "complete": reader.complete,
        "manifest_frame_count": len(reader),
        "expected_frame_count": expected_frame_count,
        "camera_serial": metadata.camera_serial,
        "depth_scale_m": _rounded(metadata.depth_scale_m, 12),
        "depth_profile": metadata.depth_profile.to_dict(),
        "color_profile": metadata.color_profile.to_dict(),
        "depth_to_color": metadata.depth_to_color.to_dict(),
        "color_to_depth": metadata.color_to_depth.to_dict(),
        "extrinsics_inverse_max_abs_error": _rounded(inverse_error, 12),
        "intrinsics_validated": True,
        "extrinsics_validated": True,
    }
    return reader, header


def _load_validated_frames(
    reader: SessionReader,
    *,
    limit: int,
    max_timestamp_skew_s: float,
) -> tuple[Iterator[tuple[int, RecordedFrame]], _SequenceAudit]:
    if not math.isfinite(max_timestamp_skew_s) or max_timestamp_skew_s <= 0.0:
        raise ValueError("max_timestamp_skew_s must be finite and positive")
    audit = _SequenceAudit(max_timestamp_skew_s=max_timestamp_skew_s)
    iterator = iter(reader)

    def generate() -> Iterator[tuple[int, RecordedFrame]]:
        for index in range(limit):
            try:
                frame = next(iterator)
            except StopIteration as exc:
                raise Slot5ManifestValidationError(
                    f"slot-5 frame {index} failed validation: manifest ended early"
                ) from exc
            except (OSError, ValueError) as exc:
                raise Slot5ManifestValidationError(
                    f"slot-5 frame {index} failed validation: {exc}"
                ) from exc
            try:
                audit.consume(frame, reader, index)
            except (TypeError, ValueError) as exc:
                raise Slot5ManifestValidationError(
                    f"slot-5 frame {index} failed validation: {exc}"
                ) from exc
            yield index, frame

    return generate(), audit


def validate_slot5_manifest(
    session_path: str | Path,
    *,
    expected_frame_count: int | None = None,
    max_timestamp_skew_s: float = 0.05,
) -> dict[str, Any]:
    """Load and validate every manifest frame, hash, profile, and timestamp."""

    reader, header = _open_validated_reader(
        session_path,
        expected_frame_count=expected_frame_count,
    )
    frames, audit = _load_validated_frames(
        reader,
        limit=len(reader),
        max_timestamp_skew_s=max_timestamp_skew_s,
    )
    for _index, _frame in frames:
        pass
    return {
        "schema_version": 1,
        "slot": SLOT,
        "session_name": Path(session_path).name,
        **header,
        **audit.to_dict(),
        "all_manifest_frames_loaded": audit.loaded_count == len(reader),
        "validation_status": "pass",
    }


def _finite_vector(value: Any, length: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    try:
        result = [_rounded(float(item)) for item in value]
    except (TypeError, ValueError, Slot5ReplayError):
        return None
    return result


def _quality_scalars(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for key in sorted(value):
        raw = value[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        number = float(raw)
        if math.isfinite(number):
            result[str(key)] = _rounded(number)
    return result


def _frame_diagnostic(
    *,
    index: int,
    frame: RecordedFrame,
    pose_result: Any,
) -> dict[str, Any]:
    if bool(pose_result.valid) or str(pose_result.reason) != EXPECTED_POSE_REASON:
        raise Slot5ReplayError(
            "slot-5 perception facade granted or changed pose authority: "
            f"valid={pose_result.valid!r}, reason={pose_result.reason!r}"
        )
    diagnostics = pose_result.diagnostics
    if not isinstance(diagnostics, Mapping):
        raise Slot5ReplayError("slot-5 pose diagnostics must be a mapping")
    observation = diagnostics.get("observation")
    stack = observation.get("stack") if isinstance(observation, Mapping) else None
    if not isinstance(stack, Mapping):
        raise Slot5ReplayError("slot-5 diagnostics are missing observation.stack")

    stack_valid = bool(stack.get("valid", False))
    center = _finite_vector(stack.get("center_base_xyz_m"), 3)
    yaw_raw = stack.get("yaw_base_rad")
    yaw: float | None
    try:
        yaw = None if yaw_raw is None else _rounded(float(yaw_raw))
    except (TypeError, ValueError, Slot5ReplayError):
        yaw = None
    axis_branch = stack.get("axis_branch")
    axis = str(axis_branch) if axis_branch is not None else None
    candidate_available = bool(
        stack_valid and center is not None and yaw is not None and axis
    )
    rejection_reasons = [str(item) for item in stack.get("rejection_reasons", ())]
    if candidate_available:
        state = f"diagnostic_candidate:{axis}"
        diagnostic_reason = "stack_geometry_candidate_only"
    else:
        diagnostic_reason = (
            rejection_reasons[0] if rejection_reasons else "candidate_fields_unavailable"
        )
        state = f"rejected:{diagnostic_reason}"

    return {
        "frame_index": index,
        "depth_frame_number": int(frame.depth_frame_number),
        "color_frame_number": int(frame.color_frame_number),
        "timestamp_s": _rounded(float(pose_result.timestamp_s), 6),
        "pose_result": {
            "valid": False,
            "reason": EXPECTED_POSE_REASON,
            "frame": "base",
            "x_m": None,
            "y_m": None,
            "yaw_rad": None,
        },
        "diagnostic_candidate": {
            "available": candidate_available,
            "status": "diagnostic_only_not_a_live_reference",
            "reason": diagnostic_reason,
            "center_base_xyz_m": center,
            "yaw_base_rad": yaw,
            "axis_branch": axis,
            "stack_se2_source": stack.get("stack_se2_source"),
            "calibration_status": stack.get("calibration_status"),
            "rejection_reasons": rejection_reasons,
            "quality": _quality_scalars(stack.get("quality")),
        },
        "state": state,
    }


def _state_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for row in rows:
        state = str(row["state"])
        index = int(row["frame_index"])
        if not runs or runs[-1]["state"] != state:
            runs.append(
                {
                    "state": state,
                    "start_frame": index,
                    "end_frame": index,
                    "frame_count": 1,
                }
            )
        else:
            runs[-1]["end_frame"] = index
            runs[-1]["frame_count"] += 1
    return runs


def _stability_summary(
    rows: list[dict[str, Any]],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if bool(row["diagnostic_candidate"]["available"])
    ]
    centers = [row["diagnostic_candidate"]["center_base_xyz_m"] for row in candidates]
    yaws = [row["diagnostic_candidate"]["yaw_base_rad"] for row in candidates]
    center_std = None
    yaw_std = None
    maximum_consecutive_center_jump_m = None
    if centers:
        center_array = np.asarray(centers, dtype=np.float64)
        center_std = [_rounded(value) for value in np.std(center_array, axis=0)]
        yaw_std = _rounded(float(np.std(np.asarray(yaws, dtype=np.float64))))
        jumps = [
            float(np.linalg.norm(right - left))
            for left, right in zip(center_array, center_array[1:])
        ]
        if jumps:
            maximum_consecutive_center_jump_m = _rounded(max(jumps))
    state_histogram = Counter(str(row["state"]) for row in rows)
    rejection_histogram = Counter(
        reason
        for row in rows
        for reason in row["diagnostic_candidate"]["rejection_reasons"]
    )
    return {
        "processed_frame_count": len(rows),
        "diagnostic_candidate_count": len(candidates),
        "diagnostic_candidate_ratio": _rounded(
            len(candidates) / max(len(rows), 1)
        ),
        "center_std_xyz_m": center_std,
        "yaw_std_rad": yaw_std,
        "maximum_consecutive_candidate_center_jump_m": (
            maximum_consecutive_center_jump_m
        ),
        "state_run_count": len(runs),
        "state_transition_count": max(len(runs) - 1, 0),
        "longest_diagnostic_candidate_run": max(
            (
                int(run["frame_count"])
                for run in runs
                if str(run["state"]).startswith("diagnostic_candidate:")
            ),
            default=0,
        ),
        "state_histogram": dict(sorted(state_histogram.items())),
        "rejection_histogram": dict(sorted(rejection_histogram.items())),
    }


def _prepare_output(path: str | Path | None, *, overwrite: bool) -> Path | None:
    if path is None:
        return None
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite slot-5 artifact: {target}")
    if target.exists() and not target.is_file():
        raise FileExistsError(f"slot-5 artifact path is not a file: {target}")
    return target


def replay_slot5_session(
    session_path: str | Path,
    root_config: Mapping[str, Any] | str | Path,
    *,
    mode: OperationMode | str = OperationMode.REPLAY,
    expected_frame_count: int | None = None,
    max_frames: int | None = None,
    output_artifact: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run an authorized slot-5 replay without any actuation capability."""

    normalized_mode = OperationMode(mode)
    if normalized_mode not in (OperationMode.REPLAY, OperationMode.DRY_RUN):
        raise ValueError("slot-5 diagnostics require replay or dry_run mode")
    verdict = authorize_operation(
        OperationRequest.place(SLOT, mode=normalized_mode)
    ).require_authorized()
    verdict.require_capability(AuthorityCapability.OFFLINE_PERCEPTION)
    if verdict.permissions.any_live_side_effect:
        raise Slot5ReplayError("offline slot-5 authority unexpectedly permits actuation")

    output_path = _prepare_output(output_artifact, overwrite=overwrite)
    root = _root_payload(root_config)
    camera = root.get("camera", {})
    if not isinstance(camera, Mapping):
        raise ValueError("slot-5 replay camera config must be an object")
    max_timestamp_skew_s = float(
        camera.get("maximum_rgb_depth_timestamp_skew_s", 0.05)
    )

    reader, manifest_header = _open_validated_reader(
        session_path,
        expected_frame_count=expected_frame_count,
    )
    if max_frames is not None and int(max_frames) <= 0:
        raise ValueError("max_frames must be positive")
    requested_count = (
        len(reader)
        if max_frames is None
        else min(len(reader), int(max_frames))
    )
    frames, sequence_audit = _load_validated_frames(
        reader,
        limit=requested_count,
        max_timestamp_skew_s=max_timestamp_skew_s,
    )

    try:
        T_base_depth, registration = _replay_base_from_depth(reader.metadata, root)
        estimator = PalletStackEstimator(load_pallet_estimator_config(root))
        held_hint = _replay_held_hint(root)
    except (TypeError, ValueError) as exc:
        raise Slot5ReplayError(f"cannot initialize slot-5 perception: {exc}") from exc

    rows: list[dict[str, Any]] = []
    for index, frame in frames:
        try:
            pose_result = perceive_pallet_pose(
                frame.color_on_depth_bgr,
                frame.depth_m(reader.metadata.depth_scale_m),
                reader.metadata.depth_profile.intrinsics,
                T_base_depth,
                slot=SLOT,
                estimator=estimator,
                timestamp_s=float(frame.depth_timestamp_ms) / 1_000.0,
                frame_id=index,
                held_box_hint=held_hint,
                calibration_status="nominal_ready_assumed",
            )
            rows.append(
                _frame_diagnostic(
                    index=index,
                    frame=frame,
                    pose_result=pose_result,
                )
            )
        except Slot5ReplayError:
            raise
        except Exception as exc:
            raise Slot5ReplayError(
                f"slot-5 replay frame {index} perception failed: {exc}"
            ) from exc

    runs = _state_runs(rows)
    config_sha256 = _sha256_bytes(_canonical_json(root).encode("utf-8"))
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "slot5_deterministic_replay_diagnostic",
        "purpose": "offline_perception_candidate_not_motion_replay",
        "slot": SLOT,
        "operation_mode": normalized_mode.value,
        "session_name": Path(session_path).name,
        "config_sha256": config_sha256,
        "manifest": {
            **manifest_header,
            **sequence_audit.to_dict(),
            "requested_frame_count": requested_count,
            "all_manifest_frames_loaded": (
                sequence_audit.loaded_count == len(reader)
            ),
        },
        "candidate_policy": {
            "live_reference_validated": False,
            "place_pose_available": False,
            "retreat_pose_available": False,
            "facade_result_required_reason": EXPECTED_POSE_REASON,
            "diagnostic_candidates_grant_motion_authority": False,
        },
        "registration": registration,
        "T_base_from_depth_diagnostic": [
            [_rounded(value) for value in row] for row in T_base_depth
        ],
        "frames": rows,
        "state_runs": runs,
        "stability": _stability_summary(rows, runs),
        "actuation": {
            "authority_reason": verdict.reason,
            "offline_perception_permitted": True,
            "robot_connection_count": 0,
            "controller_construction_count": 0,
            "ready_posture_construction_count": 0,
            "stream_construction_count": 0,
            "sequencer_construction_count": 0,
            "place_actuation_count": 0,
            "retreat_actuation_count": 0,
            "motion_authorized": False,
        },
    }
    artifact["artifact_sha256"] = _sha256_bytes(
        _canonical_json(artifact).encode("utf-8")
    )

    # The artifact is diagnostics-only even when written through the direct API;
    # enforce the same recursive key boundary used by the public CLI before any
    # file is created.
    from parcel_pose_common.output import validate_perception_only_keys

    validate_perception_only_keys(artifact)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                artifact,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_path)
    return artifact


__all__ = [
    "APPROVED_SLOT5_FRAME_COUNTS",
    "Slot5ManifestValidationError",
    "Slot5ReplayError",
    "approved_slot5_frame_count",
    "replay_slot5_session",
    "validate_slot5_manifest",
]
